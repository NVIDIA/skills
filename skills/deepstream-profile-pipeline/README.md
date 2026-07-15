# DeepStream Profiling

User-facing pointer for the `deepstream-profile-pipeline` skill. The agent-facing
specification — 6-stage measurement flow, NVTX coverage rules, and config-derivation
logic — lives in [SKILL.md](SKILL.md). The sections below collect quick-reference
material for running the skill manually.

## What It Does

Six stages — Stage 0 fires *before* the pipeline is generated; Stages 1–5 measure the
result.

| Stage | Action |
|-------|--------|
| 0. **Preset-apply** | At pipeline-creation time, pre-apply perf-correct defaults (INT8/FP16, NVMM, model-dim streammux, decoder surfaces, fakesink, no OSD/tiler unless asked). The user starts from a tuned skeleton, not a display-first one. |
| 1. **NVTX coverage check** | Run a short verification probe; classify each plugin in the pipeline as COVERED or UNINSTRUMENTED in the current DS / image / nsys combo. NVTX is treated as a bonus, never a hard requirement. |
| 2. **HW discovery** | `nvidia-smi` → theoretical decode / compute / memory-BW / PCIe ceilings via lookup tables. |
| 3. **Inference micro-benchmark** | Sweep parallel sources B = 1, 2, 4, 8, 16; the plateau batch is where doubling B yields < 5% gain. |
| 4. **Derive configs** | Closed-form rules R1–R6 set every knob from `(plateau_batch, HW_ceilings, N_streams, source_res, source_fps)`. |
| 5. **E2E profile + capacity report** | Capture under `nsys profile`, extract via `nsys stats`, classify the bound type, run R7 capacity report (`max_streams = peak_measured_fps / target_fps`) with bottleneck-specific remediation. |

Output: a single block stating bound type + evidence + max-streams capacity + which
hardware upgrade path actually helps.

## Quick Start

Trigger when the user's request carries efficiency intent:

```
# Profile an existing pipeline
profile this pipeline

# Generate a new pipeline with perf intent (Stage 0 fires)
build me an efficient pipeline that runs ResNet18 detection on N RTSP streams

# Capacity question
how many streams can this GPU handle for my model
```

Run the standalone capacity report from the command line:

```bash
python3 scripts/capacity_report.py \
  --microbench-csv /tmp/microbench_results.csv \
  --target-fps 30 \
  --dmon-csv /tmp/dmon.txt \
  --codec h264 --source-res 1080p
```

## Prerequisites

| Requirement | Required? | Notes |
|---|---|---|
| `nvcr.io/nvidia/deepstream:9.0-triton-multiarch` container | **Yes** | The slimmer `9.0-samples-multiarch` strips the nsys NVTX injector and produces empty per-plugin traces. Do not use it for profiling. |
| `nsys` (Nsight Systems 2024+) on PATH | **Yes** | Bundled in the recommended image. |
| `nvidia-smi` on PATH | **Yes** | For HW discovery (Stage 2) and live `dmon` capture during the run. |
| pyservicemaker (for the microbench/E2E test apps) | Optional | Bundled in DS containers; install with `pip install /opt/nvidia/deepstream/deepstream/service-maker/python/pyservicemaker*.whl`. |

## Repository Layout

```
deepstream-profile-pipeline/
├── SKILL.md                    Full skill description, Stage 0 preset, 5-stage flow
├── README.md                   This file
├── references/
│   ├── nvtx-coverage.md        Verified per-plugin NVTX status; verification probe; non-NVTX fallbacks
│   ├── hw-ceiling-formulas.md  nvidia-smi queries + per-GPU SM/NVDEC/bus tables + closed-form ceilings
│   ├── config-derivation-rules.md   R1–R7 closed-form rules; capacity report formula
│   ├── boundedness-rules.md    Decision tree mapping signals → bound type (decode/compute/BW/...)
│   └── nsys-cli-recipes.md     nsys profile + nsys stats invocations; container-attach pattern
├── scripts/
│   └── capacity_report.py      Executable R7 implementation: classifies bound, prints capacity + remediation
├── evals/
│   └── evals.json              Trigger / output assertions for the skill
└── tests/
    ├── README.md
    └── test_capacity_report.py unittest unit tests for the script's parser + classifier
```

## Bottleneck → Remediation Map (the headline output)

The skill always tells the user *which* upgrade actually helps:

| Dominant ceiling | What "more streams" requires |
|---|---|
| **DECODE_BOUND** | A GPU with **more NVDEC engines** (datacenter cards have more). Lower input resolution, switch codec, or pre-decode for offline. **A faster compute GPU will NOT help — compute is already underutilized.** |
| **COMPUTE_BOUND** | A **larger / newer-architecture GPU** (more SMs, higher tensor-core throughput). Lower precision (FP16→INT8) where possible, or smaller model. **More NVDECs will NOT help — decoder is already underutilized.** |
| **MEMORY_BW_BOUND** | A GPU with **HBM memory** (A100/H100/B200). INT8 weights, smaller model, lower resolution. |
| **TRACKER_BOUND** | Switch to the perf-tuned tracker preset (R4); drop tracker resolution; consider IOU. |
| **SYNC_BOUND** | Investigate threading / blocking-sync; rarely fixed by hardware change. |

## Testing

```bash
cd skills/deepstream-profile-pipeline
python3 -m unittest discover -s tests -v
```

The unit tests cover the parser and bound-classifier logic in `capacity_report.py` against
canned microbench / dmon inputs. They do not require a GPU or DeepStream — they run
anywhere with Python 3.10+.

## Related skills

See the `Related skills` section in [SKILL.md](SKILL.md) for the canonical list of
adjacent DeepStream skills and when to defer to them.
