#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Stage 5 capacity report — implements R7 from `references/config-derivation-rules.md`.

Reads the microbench scaling CSV, optionally a sampled `nvidia-smi dmon` trace, plus a
fresh `nvidia-smi --query-gpu=...` snapshot, then prints:

  1. Bound type. This script classifies into one of:
       DECODE_BOUND, COMPUTE_BOUND, MEMORY_BW_BOUND, INCONCLUSIVE, UNKNOWN_BOTTLENECK
     The other types in `references/boundedness-rules.md` (ENGINE_BUILD_CONTAMINATED,
     MEMCPY_BOUND, TRACKER_BOUND, SYNC_BOUND) require inputs this script does not take —
     `nsys stats --report cuda_api_sum` for engine-rebuild contamination, the
     `cuda_gpu_mem_time_sum` for memcpy share, and per-element FrameCounter probe FPS
     for tracker / sync — and are surfaced by the skill at the prompt layer reading the
     same trace, not by this script.
  2. Closed-form max-streams capacity at the user's target FPS.
  3. Bottleneck → remediation (compute vs decode vs BW) in plain language.

Usage:
    python3 capacity_report.py \
        --microbench-csv /tmp/microbench_results.csv \
        --target-fps 30 \
        [--dmon-csv /tmp/dmon.txt] \
        [--codec h264] [--source-res 1080p]

The script does NOT need to run on the host that captured the data — it is pure logic.
But if `--dmon-csv` is omitted, it will run `nvidia-smi --query-gpu=...` itself to read
the host GPU model and use the codec/res defaults to estimate ceilings.
"""
import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path


# Per-NVDEC throughput estimates (frames/sec) — idealized; real workloads often hit ~70%.
# Lookup is by (architecture-bucket, codec, resolution).
# Per-NVDEC throughput (frames/sec) — empirical estimates, NOT from NVIDIA's matrix.
# NVIDIA's video-encode-decode-support-matrix at
#   https://developer.nvidia.com/video-encode-decode-support-matrix
# publishes NVDEC counts and codec support (YES/NO) but does NOT publish per-codec fps.
# Numbers below are conservative midpoints from informal benchmarks; real workloads
# typically realize 50-70% of these. Stage 3 microbench overrides when they conflict.
_NVDEC_FPS = {
    # (architecture-bucket, codec, res): fps_per_unit
    ("turing_or_older", "h264", "1080p"): 1000,
    ("turing_or_older", "h265", "1080p"): 900,
    ("turing_or_older", "h264", "4k"): 250,
    ("turing_or_older", "h265", "4k"): 220,
    ("ada_hopper_blackwell", "h264", "1080p"): 1400,
    ("ada_hopper_blackwell", "h265", "1080p"): 1300,
    ("ada_hopper_blackwell", "h264", "4k"): 350,
    ("ada_hopper_blackwell", "h265", "4k"): 340,
}

# NVDEC count by GPU name (substring match — ORDER MATTERS, longer/more-specific first
# so that e.g. "L40S" matches before "L4", "A100" before "A10", "GB200" before "B200").
#
# Source: NVIDIA Video Encode/Decode Support Matrix
#   https://developer.nvidia.com/video-encode-decode-support-matrix
# (NVDEC engine counts column; matrix updated periodically — re-verify when DS major
# version changes. Counts below reflect the matrix as of skill review date in
# `SKILL.md`'s frontmatter.)
_NVDEC_COUNT = [
    # Workstation / Quadro
    ("RTX A6000", 2),
    ("RTX 5090", 2),    # Blackwell consumer
    ("RTX 4090", 1),    # Ada consumer
    # Datacenter — Blackwell
    ("GB200", 7),
    ("B200", 7),
    # Datacenter — Hopper
    ("H100", 7),
    # Datacenter — Ada
    ("L40S", 3),
    ("L40", 3),
    ("L4", 4),
    # Datacenter — Ampere
    ("A100", 5),
    ("A6000", 2),       # plain "A6000" without RTX prefix
    ("A40", 2),
    ("A10", 2),
    ("A2", 1),
    # Datacenter — Turing
    ("T4", 2),
    # Datacenter — Volta
    ("V100", 1),
    # Jetson
    ("Thor", 2),        # Jetson T5000 / Thor — 2 NVDECs per matrix
    ("Orin", 2),        # AGX Orin
]


def _arch_bucket_from_compute_cap(cc: str) -> str:
    """Map compute capability to NVDEC throughput bucket."""
    try:
        major = int(str(cc).split(".")[0])
    except (ValueError, TypeError):
        return "ada_hopper_blackwell"
    return "ada_hopper_blackwell" if major >= 8 else "turing_or_older"


def _nvdec_count_for_gpu(name: str) -> int:
    for needle, n in _NVDEC_COUNT:
        if needle.lower() in name.lower():
            return n
    return 2  # safe default


def query_gpu(gpu_id: int = 0) -> dict:
    """Read identity + clocks for ONE GPU via ``nvidia-smi -i <gpu_id>``.

    The skill profiles a single inference + decode pipeline at a time. In
    DeepStream each plugin can take its own ``gpu-id`` (``nvinfer``,
    ``nvstreammux``, ``nvtracker``, ``nvurisrcbin`` / ``nvv4l2decoder``,
    sinks, etc.), but the inference and source-side decode for a given
    pipeline almost always run on the same GPU — the one named in the
    nvinfer config (default ``gpu-id: 0``). Pass that index as ``gpu_id``
    so the report describes the actual GPU the pipeline is bound to,
    not just GPU 0.

    Returns ``{}`` if ``nvidia-smi`` is unavailable or the GPU index
    doesn't exist on the host.
    """
    if not shutil.which("nvidia-smi"):
        return {}
    out = subprocess.run(
        [
            "nvidia-smi",
            "-i", str(gpu_id),
            "--query-gpu=name,compute_cap,memory.total,memory.free,"
            "clocks.max.sm,clocks.max.memory,pcie.link.gen.current,"
            "pcie.link.width.current,driver_version",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True, text=True, check=False,
    )
    if out.returncode != 0:
        return {}
    line = out.stdout.strip().splitlines()
    if not line:
        return {}
    keys = ("name", "compute_cap", "mem_total_mib", "mem_free_mib",
            "sm_clock_mhz", "mem_clock_mhz", "pcie_gen", "pcie_width",
            "driver")
    vals = [v.strip() for v in line[0].split(",")]
    return dict(zip(keys, vals))


def parse_microbench(path: Path) -> list[dict]:
    """Parse microbench CSV.

    Returns a list of dicts, each with keys:
      - B (int): the batch size / parallel-source count for that row
      - fps_aggregate (float): total frames/sec across all sources

    Accepts both schemas:
      - current: ``B,fps_aggregate,fps_per_stream`` (fps_per_stream is read but
        not retained — derive it as ``fps_aggregate / B`` if needed downstream)
      - legacy:  ``B,fps_per_batch,fps_aggregate`` (where ``fps_per_batch`` was
        already aggregate fps — see the bug fix in microbench.sh)
    """
    rows: list[dict] = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            try:
                B = int(r.get("B", 0))
                # Accept either current schema (fps_aggregate, fps_per_stream)
                # or the legacy (buggy) one (fps_per_batch, fps_aggregate)
                if "fps_aggregate" in r and "fps_per_stream" in r:
                    fps_agg = float(r["fps_aggregate"])
                else:  # legacy — `fps_per_batch` was already aggregate fps
                    fps_agg = float(r.get("fps_per_batch", r.get("fps_aggregate", 0)))
                rows.append({"B": B, "fps_aggregate": fps_agg})
            except (ValueError, TypeError):
                continue
    return rows


def parse_dmon(path: Path, gpu_id: int | None = None) -> dict:
    """Parse ``nvidia-smi dmon`` output.

    Robust to whichever ``-s <flags>`` selector the user captured with:

    * ``-s u``  → ``# gpu  sm  mem  enc  dec  jpg  ofa``
    * ``-s m``  → ``# gpu  fb  bar1  ccpm``
    * ``-s mu`` → ``# gpu  fb  bar1  ccpm  sm  mem  enc  dec  jpg  ofa``

    The first ``# gpu …`` header line is parsed to build a column-name → index
    map; data rows then use named lookups so positions don't matter. If the
    capture lacks any of ``sm``/``mem``/``dec``, the corresponding max stays
    at 0 and the missing-signal lands as INCONCLUSIVE in ``classify_bound``
    rather than getting a wrong reading from a fixed column index.

    Returns ``{"sm_max", "mem_max", "dec_max"}``. When ``gpu_id`` is given,
    only samples for that GPU index are considered (first column of each
    data row is the GPU index). When ``gpu_id`` is ``None``, samples from
    every GPU in the log are aggregated.
    """
    sm_max = mem_max = dec_max = 0
    col_idx: dict[str, int] = {}
    with open(path) as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            # The FIRST header row of nvidia-smi dmon is `# gpu <cols...>`.
            # The second header row is units (`# Idx % % ...`) — ignore.
            if stripped.startswith("#"):
                tokens = stripped.lstrip("#").split()
                if tokens and tokens[0].lower() == "gpu" and not col_idx:
                    # Map column name → index (case-insensitive).
                    col_idx = {tok.lower(): i for i, tok in enumerate(tokens)}
                continue
            if not col_idx:
                # No header seen yet — can't tell which column is which.
                continue
            parts = stripped.split()
            try:
                row_gpu = int(parts[col_idx["gpu"]])
            except (ValueError, KeyError, IndexError):
                continue
            if gpu_id is not None and row_gpu != gpu_id:
                continue
            for name, var in (("sm", "sm_max"), ("mem", "mem_max"), ("dec", "dec_max")):
                if name not in col_idx:
                    continue
                try:
                    val = int(parts[col_idx[name]])
                except (ValueError, IndexError):
                    continue
                if var == "sm_max":
                    sm_max = max(sm_max, val)
                elif var == "mem_max":
                    mem_max = max(mem_max, val)
                else:
                    dec_max = max(dec_max, val)
    return {"sm_max": sm_max, "mem_max": mem_max, "dec_max": dec_max}


def classify_bound(microbench: list[dict], dmon: dict, target_fps: int) -> dict:
    """Apply the boundedness-rules.md decision tree.

    Returns dict with `bound`, `confidence`, `evidence` (list of strings).
    `bound` is one of: DECODE_BOUND, COMPUTE_BOUND, MEMORY_BW_BOUND, INCONCLUSIVE,
    UNKNOWN_BOTTLENECK. The remaining types listed in `boundedness-rules.md`
    (ENGINE_BUILD_CONTAMINATED, MEMCPY_BOUND, TRACKER_BOUND, SYNC_BOUND) need
    inputs this function doesn't take and are surfaced by the skill at the prompt
    layer rather than here.
    """
    by_b = {r["B"]: r["fps_aggregate"] for r in microbench}
    evidence: list[str] = []
    bound = "UNKNOWN_BOTTLENECK"
    conf = "low"

    # --- Decode-bound checks ---
    # Guard every division: if a microbench iteration produced fps==0
    # (pipeline failed to start, probe never fired, etc.) the row's
    # value is 0 in the CSV, which would otherwise crash with
    # ZeroDivisionError instead of being reported as inconclusive data.
    decode_signals = 0
    if 1 in by_b and 2 in by_b and by_b[1] > 0 and by_b[2] > 0:
        ratio = by_b[1] / by_b[2]
        evidence.append(
            f"microbench: fps[B=1]/fps[B=2] = {ratio:.2f}  "
            f"(< 0.55 ⇒ decode-limited at low B)"
        )
        if ratio < 0.55:
            decode_signals += 1
    elif 1 in by_b and 4 in by_b and by_b[1] > 0 and by_b[4] > 0:
        ratio = by_b[1] / by_b[4]
        evidence.append(
            f"microbench: fps[B=1]/fps[B=4] = {ratio:.2f}  "
            f"(< 0.30 ⇒ decode-limited at low B)"
        )
        if ratio < 0.30:
            decode_signals += 1

    # Plateau test: if fps stops growing past some B, that B's ceiling is the wall.
    Bs = sorted(by_b.keys())
    if len(Bs) >= 2:
        last_B = Bs[-1]
        max_fps = max(by_b.values())
        ceiling_fps = by_b[last_B]
        # Same zero-guard reasoning — if the highest-B run produced 0
        # fps, the ceiling is undefined and we just skip the plateau
        # check rather than crash on the relative-difference div.
        if ceiling_fps > 0:
            flat = all(
                abs(by_b[b] - ceiling_fps) / ceiling_fps < 0.05
                for b in Bs if b >= max(2, Bs[0])
            )
            if flat:
                evidence.append(
                    f"microbench: fps_aggregate is flat at ~{ceiling_fps:.0f} for B≥2  "
                    f"(ceiling reached)"
                )

    if dmon.get("dec_max", 0) >= 90:
        decode_signals += 1
        evidence.append(f"nvidia-smi dmon: NVDEC peak {dmon['dec_max']}% (≥90% ⇒ NVDEC saturated)")

    # --- Memory-BW-bound check ---
    mem_bw_bound = (dmon.get("mem_max", 0) >= 85
                    and dmon.get("sm_max", 0) < 70)
    if mem_bw_bound:
        evidence.append(
            f"nvidia-smi dmon: DRAM peak {dmon['mem_max']}%  "
            f"(>85% with SM<70% ⇒ memory-BW bound)"
        )

    # --- Compute-bound check ---
    compute_bound = (dmon.get("sm_max", 0) >= 70
                     and not mem_bw_bound
                     and decode_signals == 0)
    if compute_bound:
        evidence.append(
            f"nvidia-smi dmon: SM peak {dmon['sm_max']}% (≥70% with no decode/BW signals)"
        )

    # --- Verdict ---
    # Order MUST match the decision tree in references/boundedness-rules.md:
    # decode → memory-BW → compute → inconclusive.
    if decode_signals >= 2:
        bound, conf = "DECODE_BOUND", "high"
    elif decode_signals == 1:
        bound, conf = "DECODE_BOUND", "medium"
    elif mem_bw_bound:
        bound, conf = "MEMORY_BW_BOUND", "high"
    elif compute_bound:
        bound, conf = "COMPUTE_BOUND", "medium"
    elif by_b:
        bound, conf = "INCONCLUSIVE", "low"

    return {"bound": bound, "confidence": conf, "evidence": evidence}


def compute_capacity(microbench: list[dict], target_fps: int,
                     codec: str, source_res: str, gpu: dict,
                     bound: str = "UNKNOWN_BOTTLENECK") -> dict:
    """R7 closed form. The MEASURED peak FPS is authoritative — it already reflects
    whichever bottleneck dominated. The theoretical NVDEC ceiling is reported as a
    sanity-check comparison only.
    """
    by_b = {r["B"]: r["fps_aggregate"] for r in microbench}
    peak_fps = max(by_b.values()) if by_b else 0

    nvdec_count = _nvdec_count_for_gpu(gpu.get("name", "")) if gpu else 2
    arch = _arch_bucket_from_compute_cap(gpu.get("compute_cap", "")) if gpu else "ada_hopper_blackwell"
    per_unit = _NVDEC_FPS.get((arch, codec, source_res), 1000)
    decode_ceiling_theoretical = nvdec_count * per_unit

    # Authoritative: the measurement IS the practical ceiling.
    n_overall = int(peak_fps // target_fps) if target_fps > 0 else 0

    # Sanity: compare measured to theoretical decode (which is itself idealized).
    n_decode_theoretical = int(decode_ceiling_theoretical // target_fps) if target_fps > 0 else 0
    decode_realization = (peak_fps / decode_ceiling_theoretical
                          if decode_ceiling_theoretical else 0)

    dominant_map = {
        "DECODE_BOUND": "decode",
        "COMPUTE_BOUND": "compute",
        "MEMORY_BW_BOUND": "memory bandwidth",
    }
    dominant = dominant_map.get(bound, "unknown")

    return {
        "peak_fps_measured": peak_fps,
        "decode_ceiling_theoretical": decode_ceiling_theoretical,
        "decode_realization_pct": round(decode_realization * 100, 1),
        "n_overall": n_overall,
        "n_decode_theoretical": n_decode_theoretical,
        "dominant": dominant,
        "nvdec_count": nvdec_count,
        "per_nvdec_theoretical": per_unit,
        "arch_bucket": arch,
    }


_REMEDIATION = {
    "DECODE_BOUND": (
        "More decode capacity is needed to scale beyond this stream count. Options\n"
        "(ordered roughly cheapest-to-deepest):\n"
        "  • Lower input resolution — universal, works regardless of source codec.\n"
        "    e.g. 1080p → 720p roughly doubles per-NVDEC fps.\n"
        "  • Lower target FPS — also universal; halves the throughput needed.\n"
        "  • Codec-conditional (per NVDEC throughput table for Ada/Hopper/Blackwell,\n"
        "    1080p ≈ 1400 fps H264 vs 1300 fps H265 per engine):\n"
        "      - If source is H.265: switching to H.264 buys ~7% per-engine fps.\n"
        "      - If source is already H.264: no codec-level win possible — go to\n"
        "        the resolution / fps / hardware levers below.\n"
        "      - If source is AV1 / VP9 / MJPEG / other: throughput differs by\n"
        "        card. Consult NVIDIA's video-encode-decode-support-matrix\n"
        "        (https://developer.nvidia.com/video-encode-decode-support-matrix).\n"
        "        Note MJPEG uses NVJPEG, not NVDEC — different ceiling entirely.\n"
        "  • A GPU with more NVDEC engines (datacenter cards usually have more —\n"
        "    H100 / B200 ship with ~7 NVDECs vs typical workstation cards with 1-2).\n"
        "  • For offline / non-real-time, pre-decode to disk.\n"
        "  Adding a faster compute GPU will NOT help — compute is already underutilized."
    ),
    "COMPUTE_BOUND": (
        "More compute is needed to scale beyond this stream count. Options:\n"
        "  • A larger / newer-architecture GPU (more SMs, higher clocks, newer tensor\n"
        "    cores). The Stage 2 ceilings table compares architectures by ops/clock/SM.\n"
        "  • Lower precision: FP16 → INT8 (2× tensor-core throughput on Ampere/Ada/Hopper)\n"
        "    if not already INT8.\n"
        "  • Smaller model, or `nvinfer.interval=1` to skip every other frame (halves\n"
        "    compute load at half the temporal resolution).\n"
        "  Adding more NVDECs will NOT help — decoder is already underutilized."
    ),
    "MEMORY_BW_BOUND": (
        "Memory bandwidth is the wall. Options:\n"
        "  • A GPU with HBM memory (e.g. A100, H100, B200) has 3-5× the bandwidth of\n"
        "    GDDR-based cards.\n"
        "  • INT8 weights (halve weight bytes, smaller working set).\n"
        "  • Smaller / lower-resolution model."
    ),
    "INCONCLUSIVE": (
        "Could not classify the bottleneck with high confidence. Capture a longer trace,\n"
        "ensure --gpu-metrics-devices=all is enabled (needs CAP_SYS_ADMIN), and rerun the\n"
        "microbench with parallel sources to clearly separate decode vs compute."
    ),
    "UNKNOWN_BOTTLENECK": (
        "No bound type matched the decision rules. The pipeline is below the HW ceiling for\n"
        "no measured reason. Check: model accuracy at the configured precision, TRT engine\n"
        "build success, NVMM zero-copy across all elements (memcpy share <5%)."
    ),
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--microbench-csv", required=True, type=Path,
                    help="Stage 3 output CSV (B,fps_aggregate,fps_per_stream)")
    ap.add_argument("--target-fps", type=int, default=30,
                    help="Per-stream FPS the user wants (default 30)")
    ap.add_argument("--dmon-csv", type=Path, default=None,
                    help="`nvidia-smi dmon -s mu` log captured during the run")
    ap.add_argument("--codec", default="h264", choices=["h264", "h265"])
    ap.add_argument("--source-res", default="1080p", choices=["1080p", "4k"])
    ap.add_argument("--gpu-id", type=int, default=0,
                    help="GPU index the pipeline is bound to (matches the "
                         "`gpu-id` set in your nvinfer config). Default 0. "
                         "In DS each plugin can have its own gpu-id; this "
                         "skill assumes the inference and source-decode plugins "
                         "are on the same GPU and reports for that one only.")
    ap.add_argument("--json", action="store_true",
                    help="Emit machine-readable JSON only")
    args = ap.parse_args()

    if not args.microbench_csv.is_file():
        print(f"error: microbench CSV not found: {args.microbench_csv}",
              file=sys.stderr)
        return 2

    microbench = parse_microbench(args.microbench_csv)
    if not microbench:
        print("error: no usable rows in microbench CSV", file=sys.stderr)
        return 2

    dmon = (parse_dmon(args.dmon_csv, gpu_id=args.gpu_id)
            if args.dmon_csv and args.dmon_csv.is_file() else {})
    gpu = query_gpu(gpu_id=args.gpu_id)
    classification = classify_bound(microbench, dmon, args.target_fps)
    capacity = compute_capacity(microbench, args.target_fps, args.codec,
                                args.source_res, gpu,
                                bound=classification["bound"])
    remediation = _REMEDIATION.get(classification["bound"],
                                   _REMEDIATION["UNKNOWN_BOTTLENECK"])

    payload = {
        "gpu": gpu,
        "microbench": microbench,
        "dmon": dmon,
        "classification": classification,
        "capacity": capacity,
        "target_fps": args.target_fps,
        "codec": args.codec,
        "source_res": args.source_res,
        "remediation": remediation,
    }

    if args.json:
        print(json.dumps(payload, indent=2))
        return 0

    # Plain-English report
    print("=" * 72)
    print("DeepStream Profiling — Capacity Report (R7)")
    print("=" * 72)
    if gpu:
        print(f"GPU       : id={args.gpu_id}, {gpu.get('name')}, "
              f"compute_cap {gpu.get('compute_cap')}, "
              f"{gpu.get('mem_total_mib')} MiB, driver {gpu.get('driver')}")
    else:
        print(f"GPU       : id={args.gpu_id} (nvidia-smi unavailable or index "
              f"not found; HW ceilings estimated from defaults)")
    print(f"Codec/res : {args.codec.upper()} {args.source_res}")
    print(f"Target    : {args.target_fps} fps per stream")
    print()
    print(f"Bound type: {classification['bound']}  (confidence: {classification['confidence']})")
    for line in classification["evidence"]:
        print(f"  - {line}")
    print()
    print("Capacity (R7 — measurement is authoritative):")
    print(f"  Peak measured FPS aggregate : {capacity['peak_fps_measured']:.0f}  "
          f"(this is the practical ceiling — already reflects the active bottleneck)")
    print(f"  Theoretical decode ceiling  : {capacity['decode_ceiling_theoretical']:.0f}  "
          f"({capacity['nvdec_count']} × ~{capacity['per_nvdec_theoretical']} fps "
          f"per NVDEC, {capacity['arch_bucket']})")
    print(f"  Decode realization          : {capacity['decode_realization_pct']:.0f}% of theoretical")
    print(f"  → MAX OVERALL               : {capacity['n_overall']} streams at "
          f"{args.target_fps} fps  ({capacity['dominant']}-limited per classification above)")
    if capacity['n_decode_theoretical'] > capacity['n_overall']:
        print(f"  Note: theoretical NVDEC table predicted up to "
              f"{capacity['n_decode_theoretical']} streams; measurement realized "
              f"{capacity['decode_realization_pct']:.0f}% of that — typical for real workloads.")
    print()
    print("Remediation:")
    for line in remediation.splitlines():
        print(f"  {line}")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
