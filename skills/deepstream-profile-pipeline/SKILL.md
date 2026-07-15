---
name: "deepstream-profile-pipeline"
description: "Profile a DeepStream pipeline with Nsight Systems and derive its configs from the measurement. Use when the user asks for an efficient, performant, or profiled pipeline — or to benchmark, tune, or measure FPS."
metadata:
  author: "NVIDIA CORPORATION"
  tags:
    - deepstream
    - profiling
    - nsight-systems
    - nvtx
    - nvidia-smi
    - benchmarking
  languages:
    - bash
    - python
    - yaml
  domain: video-analytics
  team: deepstream-sdk
owner: "NVIDIA CORPORATION"
service: "deepstream"
version: "0.1.0"
reviewed: "2026-04-24"
license: CC-BY-4.0 AND Apache-2.0
compatibility: >
  DeepStream SDK 9.0 on Ubuntu 22.04 or 24.04, run from the
  `nvcr.io/nvidia/deepstream:9.0-triton-multiarch` container (the dev image; the slimmer
  `samples-multiarch` variant strips the nsys NVTX injector and produces empty per-plugin
  NVTX traces — do not use it for profiling). Requires `nsys` (Nsight Systems 2024+) and
  `nvidia-smi` on PATH. No GUI dependency — the skill runs fully headless and uses only
  `nsys profile` + `nsys stats`.
data_classification: "internal"
---

# DeepStream Profiling Skill

Profile-driven pipeline creation. When the user indicates they want an efficient DeepStream
pipeline, this skill replaces guesswork with two measured numbers — **inference plateau
batch** and **HW ceiling** — and derives every other config from them. Then it profiles the
E2E pipeline with Nsight Systems and reports per-plugin NVTX timings.

**Model- and pipeline-agnostic.** The skill assumes only that the inference element is
`nvinfer` or `nvinferserver` (so model dims, precision, and batch knobs are settable through
the standard config). It works for detection (with or without tracker), classification,
segmentation, VLM, and embedding pipelines. Source can be file, RTSP, USB camera, or any
mix. The skill reads the user's actual config to discover model dims / target FPS / source
properties — it does NOT assume any particular model, codec, or resolution.

> **Constraint.** Terminal only. Use `nsys profile` to capture and `nsys stats` to extract.
> Do not depend on Nsight Lens or any GUI.

## When to trigger

Activate this skill **at pipeline creation time** when the user's ask carries efficiency
intent. Concrete triggers:

- "build an **efficient** / **fast** / **performant** / **optimized** pipeline"
- "give me a pipeline that runs well on this GPU"
- "benchmark / profile / measure / tune / optimize this pipeline"
- "I want to run N streams at M FPS"
- "how many streams can this GPU handle"
- user explicitly asks for `nsys` or Nsight

For plain "build a pipeline" / "display this video" / "save this stream" with no perf intent,
hand off to the `deepstream-generate-pipeline` skill instead.

## The 6-stage flow

Run the stages in order. Stage 0 fires *before* the pipeline is generated, so the user
starts from a perf-tuned skeleton. Stages 1–5 measure and verify.

## Stage 0 — Preset-apply (at pipeline-creation time)

Trigger: any time the coding agent is about to generate a new DS pipeline AND the user's
prompt carries efficiency intent (see "When to trigger" above).

Action: pre-apply these defaults *without prompting*. The user does not need to know any of
them; they just get a pipeline that's already in the right shape.

| Knob | Default value | Skip when |
|---|---|---|
| `nvinfer.network-mode` | `1` (INT8) if a calibration file is present at `int8-calib-file=<path>`, else `2` (FP16). Never FP32. | Model has no INT8 calibration AND the user explicitly says "FP32". |
| `nvinfer.model-engine-file` | Pre-built `.engine` path | Always set. Force a one-shot prebuild before measurement. |
| `nvinfer.infer-dims` | `3;<H>;<W>` matching the model's native input | Always set, even for static-shape ONNX (harmless). |
| `nvstreammux.batch-size` | `min(N_streams, 16)` until microbench refines it | — |
| `nvstreammux.width / height` | model's native input dims (read from the nvinfer config's `infer-dims=3;H;W`) | User explicitly asks for native source resolution at the muxer. |
| `nvstreammux.batched-push-timeout` | `1e6 / source_fps` µs (33333 for 30 fps) | — |
| `nvstreammux.nvbuf-memory-type` | `0` (NVMM) | — |
| Decoder `num-extra-surfaces` | `min(batch_size, 5)` | — |
| Decoder `cudadec-memtype` | `0` (NVMM) | — |
| Sink | `fakesink sync=False` for the benchmark variant | User asked for on-screen display or on-disk recording (then keep OSD/tiler/encoder/sink and produce TWO variants). |
| OSD + tiler | omit | User asked for visible output. |
| Tracker `ll-config-file` | `config_tracker_NvDCF_max_perf.yml` (perf-tuned NvDCF preset shipped with DS 9.0) | Tracker not present. |
| Tracker `tracker-width / height` | 480 / 288 | — |
| Tracker `enable-batch-process` (in linked YAML) | `1` | — |
| Queue between source and pgie | `max-size-buffers = batch_size × 4` | No queue requested (rare). |
| Kafka/message queue | `max-size-buffers=2, leaky=2` | No Kafka. |
| Decode-side `PerfMonitor` | attach (in addition to pgie-side) | Pipeline is `nvurisrcbin → pgie` direct without intermediate queue. |

**Why Stage 0 exists:** without it, every newly generated pipeline starts from
display-first defaults and Stages 1–5 spend cycles fixing avoidable issues. Stage 0 is the
"don't write a bad pipeline in the first place" gate.

The student / API user never sees these knobs. The skill's response back to the user is in
plain English (FPS, stream count, observed bottleneck), not knob names.

## The verification flow (Stages 1–5)

Run the stages in order. Do not skip a stage — later stages depend on earlier ones' outputs.

### Stage 1 — NVTX coverage check

DeepStream plugins emit NVTX ranges natively; custom plugins and plain GStreamer-core
elements (`queue`, `tee`, `h264parse`, etc.) do not. Before profiling, list the elements the
pipeline uses and classify each.

- Read the pipeline definition (gst-launch string or `pipeline.py`).
- For each element, look it up in [references/nvtx-coverage.md](references/nvtx-coverage.md).
- Classify **COVERED** (emits NVTX in this DS / image / nsys combo) or **UNINSTRUMENTED**.
- **MVP rule:** the skill *prefers* per-plugin NVTX as confirmation but does not require it.
  Decode-bound diagnosis works from microbench shape + `nvidia-smi dmon`; compute-bound from
  CUDA kernel mix; memcpy from `cuda_gpu_mem_time_sum`. NVTX is a bonus.
- For UNINSTRUMENTED elements, the skill reports "not directly measurable in this build" and
  still applies the closed-form R1–R6 knobs (which are derived from inputs, not from
  per-plugin profile data).
- Auto-injecting NVTX for uninstrumented elements is **out of scope** for this version —
  flag it as follow-up in the final report.

Output of Stage 1: a short coverage table, e.g.

```text
nvurisrcbin       COVERED
nvstreammux       COVERED
nvinfer           COVERED
nvtracker         COVERED
queue_src         UNINSTRUMENTED — not re-tuned
fakesink          UNINSTRUMENTED — not re-tuned
```

### Stage 2 — HW discovery

Run `nvidia-smi` and derive theoretical ceilings for the host GPU. Minimum queries:

```bash
# Identity + memory + compute
nvidia-smi --query-gpu=name,compute_cap,memory.total,memory.free,\
clocks.max.sm,clocks.max.memory,utilization.gpu \
--format=csv,noheader,nounits

# NVDEC / NVENC utilization (per-engine)
nvidia-smi --query-gpu=utilization.decoder,utilization.encoder \
--format=csv,noheader,nounits

# PCIe link width/gen (for H2D memcpy ceiling)
nvidia-smi --query-gpu=pcie.link.gen.current,pcie.link.width.current \
--format=csv,noheader,nounits
```

Derive from those numbers:

- **Decode ceiling (fps)**: NVDEC_count × per-unit H265/H264 fps for the source resolution
  (table in [references/hw-ceiling-formulas.md](references/hw-ceiling-formulas.md)).
- **Compute ceiling (TOPS)**: SM count × clock × ops-per-clock at the target precision. Gives
  an upper bound — real models hit 30–60% of this.
- **Memory-bandwidth ceiling (GB/s)**: memory clock × bus width. Model weight reads +
  activations should fit well under this.
- **Memcpy ceiling (GB/s)**: PCIe gen × width × 0.8 practical. Only relevant if NVMM is
  broken and H2D/D2H transfers appear in Stage 5.

Store the derived ceilings — they drive the Stage 5 "actual vs. theoretical" section.

Full formulas and the per-codec NVDEC throughput table:
[references/hw-ceiling-formulas.md](references/hw-ceiling-formulas.md).

### Stage 3 — Inference-only micro-benchmark

Run **only the inference stage** (source → streammux → nvinfer → fakesink), sweeping
`batch-size` to find the plateau. This isolates the model's true peak FPS from everything
else, and answers "how many streams fit into a single batch without FPS dropping?".

Sweep: `batch-size ∈ {1, 2, 4, 8, 16, 32}` (cap at `N_streams` and at GPU memory).

For each batch size:

- Set `nvstreammux.batch-size = nvinfer.batch-size = B`.
- Set `nvstreammux.width/height` = the model's native `infer-dims` (read from the nvinfer config).
- `fakesink sync=False` as the only branch.
- Run 30 s; measure FPS from `measure_fps_probe` (console) or DS `PerfMonitor`.
- Record `(B, fps)`.

**Plateau batch** = the smallest B where increasing to 2×B yields < 5% FPS gain. That is the
target batch for the full pipeline.

If the user's N_streams ≤ plateau batch, set final batch = N_streams. Otherwise set final
batch = plateau batch and note that the pipeline will process streams in multiple batches
per tick.

### Stage 4 — Derive configs

From `(plateau_batch, HW_ceilings, N_streams, source_res, source_fps)`, set every tunable
knob at once. Do not tune one knob at a time — the derivation rules are closed-form.

Knobs to set, in order:

1. **Streammux**: `batch-size = final_batch`, `width/height = min(source_res, infer_dims)`,
   `batched-push-timeout = 1e6 / source_fps` µs, `nvbuf-memory-type = 0`.
2. **Inference**: `batch-size = final_batch`, `network-mode = 1 (INT8) if calib file exists
   else 2 (FP16)`, `interval = 0`, `infer-dims = model's native dims`, `model-engine-file =
   pre-built .engine path`.
3. **Decoder** (on `nvurisrcbin` / `nvmultiurisrcbin` / `nvv4l2decoder`):
   `num-extra-surfaces = min(final_batch, 5)`, `cudadec-memtype = 0`, `nvbuf-memory-type = 0`.
4. **Tracker** (if present): `enable-batch-process = 1`, tracker res 480×288, point
   `ll-config-file` at `config_tracker_NvDCF_max_perf.yml`.
5. **Queues** (if present between decoder and streammux, or streammux and nvinfer):
   `max-size-buffers = final_batch × 2`. Kafka/message branches: `leaky=2,
   max-size-buffers=2`.

Full derivation table with each formula and a one-line "why":
[references/config-derivation-rules.md](references/config-derivation-rules.md).

Write the derived values into the user's config files (`pgie_config.yml`,
`tracker_config.yml`, `pipeline.py` source properties, any `deepstream-app` `.txt`). Always
`Read` before `Edit`. Keep edits surgical — do not reformat unrelated lines.

### Stage 5 — E2E profile + report

Run the E2E pipeline under `nsys profile` and extract per-plugin timings via `nsys stats`.

Capture:

```bash
TS=$(date +%Y%m%d_%H%M%S)
OUT=/tmp/ds_profile_${TS}
nsys profile \
  --trace=cuda,nvtx,osrt \
  --gpu-metrics-devices=all \
  --cuda-memory-usage=true \
  --force-overwrite=true \
  --duration=30 \
  --output=${OUT} \
  <your-pipeline-launch-command>
```

Extract:

```bash
# Per-kernel GPU time (top 10)
nsys stats --report cuda_gpu_kern_sum --format csv ${OUT}.nsys-rep | head -20

# Per-NVTX-range time (top 10) — this is the DS per-plugin breakdown
nsys stats --report nvtx_sum --format csv ${OUT}.nsys-rep | head -20

# Memcpy totals
nsys stats --report cuda_gpu_mem_time_sum --format csv ${OUT}.nsys-rep

# GPU metrics (SM activity, DRAM throughput) — requires --gpu-metrics-devices
nsys stats --report gpu_metric_gpu_util_sum --format csv ${OUT}.nsys-rep
```

Full command reference: [references/nsys-cli-recipes.md](references/nsys-cli-recipes.md).

Report (Markdown, to stdout — no external UI):

```markdown
## Profile summary

**Hardware**: <name>, <mem_total> GB, SM x<sm>, NVDEC x<nvdec>, PCIe Gen<g> x<w>
**Ceilings**: decode <X> fps, compute ~<Y> TOPS @ INT8, memory <Z> GB/s

**Inference plateau**: batch=<B>, peak=<F> fps per batch → <F × B> fps aggregate

**E2E measured**: <actual> fps  (=<pct>% of inference plateau)

### Per-plugin time (from NVTX) — only for plugins emitting NVTX in this build

| Plugin          | Share of wall time | GPU / CPU | Notes |
|-----------------|--------------------|-----------|-------|
| nvinfer         | <pct>%             | GPU       | (always emitted; if absent, NVTX injection is broken) |
| nvdsosd         | <pct>%             | GPU       | (when in pipeline) |
| ...             | ...                | ...       | (other plugins as the verification probe shows) |

(Numbers above are illustrative — fill in from `nsys stats --report nvtx_sum`. Plugins
that don't emit NVTX in your DS / image combo simply don't appear; that's not a bug, it's
the limit of what NVTX captures here. See `references/nvtx-coverage.md`.)

### Applied configs (sample shape; values come from R1–R6 + Stage 3 measurements)

- `nvstreammux.batch-size = <plateau_batch>`
- `nvinfer.network-mode = 1 (INT8)` if calibration available, else `2 (FP16)`
- decoder `num-extra-surfaces = min(plateau_batch, 5)`
- queue between source and pgie, `max-size-buffers = plateau_batch × 4`
- ... (full list per the user's pipeline shape)

### Uninstrumented (skipped re-tune)

List the elements that didn't emit NVTX in this build (typically the closed-source binary
plugins — see `references/nvtx-coverage.md`) plus plain GStreamer-core helpers. Report
them so the user knows what wasn't directly measurable.
```

Keep the summary terse. Raw `nsys stats` CSV goes into the temp file, not the response.

## Reference documents

| Document | Use when |
|----------|----------|
| [references/nvtx-coverage.md](references/nvtx-coverage.md) | Stage 1 — classifying each pipeline element as COVERED or UNINSTRUMENTED. |
| [references/hw-ceiling-formulas.md](references/hw-ceiling-formulas.md) | Stage 2 — turning `nvidia-smi` output into decode / compute / memory ceilings. |
| [references/config-derivation-rules.md](references/config-derivation-rules.md) | Stage 4 — per-knob formula keyed to `(plateau_batch, HW, N_streams, source_res, source_fps)`. |
| [references/nsys-cli-recipes.md](references/nsys-cli-recipes.md) | Stages 3 & 5 — exact `nsys profile` / `nsys stats` invocations. |

## Non-goals (this version)

- **No Nsight Lens / no GUI.** Terminal only.
- **No NVTX auto-injection** for uninstrumented plugins. MVP skips their knobs. Future work.
- **No iterative tune-measure-tune loop.** Stage 4 derives configs once from closed-form
  rules; Stage 5 measures and reports. If the user wants to keep tuning, they can re-invoke
  the skill with updated inputs.

## Related skills

- `deepstream-generate-pipeline` — upstream pipeline generation. This skill
  assumes a pipeline already exists or is about to be generated.
- `deepstream-byovm` — HF → TensorRT engine building. Run first if the user
  brought a new model; come here after.

## Notes

- Lives in `skills/deepstream-profile-pipeline/` alongside the other DS skills, per
  the repo convention in `CLAUDE.md`.
- For ground-truth on **any** plugin's properties (types, defaults, ranges) and pad caps,
  query the loaded binary inside the DS container:
  ```bash
  gst-inspect-1.0 nvinfer
  gst-inspect-1.0 nvstreammux
  gst-inspect-1.0 nvurisrcbin   # works on closed-source binary plugins too
  gst-inspect-1.0 | grep ^nv    # list every NVIDIA-specific element this build ships
  ```
  Plugin naming convention: any element prefixed `nv*` is NVIDIA DeepStream-specific
  (NVMM-capable, may emit NVTX); everything else is upstream GStreamer-core (no NVMM,
  never emits DS NVTX). Use this prefix as the first-pass classifier when triaging an
  unfamiliar pipeline.
- The open-source subset of plugin code lives under
  `/opt/nvidia/deepstream/deepstream/sources/gst-plugins/` if you need to read the
  implementation (only some plugins are open — closed ones must be inspected via
  `gst-inspect-1.0` and behaviour observed at runtime).

<!-- Signing refresh marker. -->
