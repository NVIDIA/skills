# Sizing and GPU Placement

Use this guide after selecting the Foundation and effective service set. Treat
the resolved Compose model—not a hardware-profile label—as the deployment
contract.

## Sizing flow

1. Inventory every GPU's model, total memory, free memory, and current users.
2. List every GPU service in the effective build and its model, precision,
   device ID, concurrency or stream count, and whether its footprint is fixed
   or elastic.
3. Place fixed-footprint services first according to the placement and sizing
   contracts for [RT-CV](services/rt-cv.md) and
   [RT-Embed](services/rt-embed.md).
4. Place singleton RT-VLM last, into the capacity step 3 leaves behind; never
   displace or co-pack a fixed service to free a GPU for it. When composition
   must converge different integrated Cosmos3 Nano variants, use BF16 only if a
   GPU remains free after every fixed-footprint service has its preferred
   dedicated device; otherwise use FP8 co-resident on the RT-CV device
   (`RT_VLM_DEVICE_ID = RT_CV_DEVICE_ID`), which retains the most headroom among
   the fixed-footprint services placed in step 3. Share only when the combined
   budget fits. Resolve the atomic variant/placement set without inheriting
   consumer wiring, as specified by the [RT-VLM owner](services/rt-vlm.md). Stock
   mode retains its Foundation's reviewed variant and placement, except that a
   co-located placement on a host with a free GPU is the user's call — ask, per
   "Ask before co-locating" below. For continuous
   VLM inference on a shared GPU, reduce `NUM_STREAMS` and verify utilization
   headroom under load.
5. Use a remote endpoint only when the user requested one or approved it after
   local sizing failed.
6. Put every device ID and utilization value the placement **changes** from the
   Foundation default into the build `override.env` (with its derived closure; do
   not repeat unchanged defaults, per `composition.md`), resolve Compose, and
   verify the full effective placement in `resolved.yml`.
7. Under load, watch `nvidia-smi` and model-service logs. Tune one variable at
   a time and regenerate `resolved.yml`.

Never silently substitute a smaller model, lower precision, or remote endpoint.
The placement-driven RT-VLM BF16/FP8 convergence above is part of singleton
resolution, not a fallback substitution.
If the requested shape does not fit, show the measured capacity and required
budget, then ask the user to choose a different placement.

## General memory budget

```text
weights_GB = parameters_billions * bits_per_parameter / 8
model_GB   = weights_GB * 1.3

dedicated fits when model_GB <= 0.85 * GPU_VRAM_GB
shared fits when sum(service budgets) <= 0.85 * GPU_VRAM_GB
```

Use 16 bits for FP16/BF16, 8 for FP8/INT8, and 4 for INT4/NVFP4.
The 30% model overhead covers KV cache and activations; the remaining 15% GPU
reserve covers CUDA graphs, framework allocations, and runtime variance. Do not
tune a discrete-GPU allocation above `0.85`.

| GPU | Memory | 85% budget |
|---|---:|---:|
| H100 / A100 80 GB | 80 GB | 68 GB |
| H200 | 141 GB | 119.85 GB |
| B200 / GB200 | 192 GB | 163.2 GB |
| DGX Station GB300 | 251 GB | 213.35 GB |
| RTX PRO 6000 Blackwell | 96 GB | 81.6 GB |
| L40S / L40 / RTX 6000 Ada / A40 | 48 GB | 40.8 GB |
| RTX PRO 4500 Blackwell | 32 GB | 27.2 GB |
| DGX Spark / Thor | 128 GB unified | Size from actual free memory; see `edge.md` |

Representative model estimates:

| Model | Precision | Estimated model budget |
|---|---|---:|
| Nemotron 3.5 Lightning 30B-A3B (default) | BF16 | about 78 GB; budget total parameters (30 B), not active parameters (3 B). The NIM's own BF16 profile asks for at least 66 GB per GPU |
| Nemotron 3.5 Lightning 30B-A3B (default) | INT4 | about 45 GB observed; `vllm-int4-tp1-pp1-32.0` is pinned in every `hw-*.env` |
| Nemotron Nano 9B v2 FP8 (alternate for GPUs too small for the default) | FP8 | 11.7 GB |
| Cosmos Reason 1 7B | FP16 | 18.2 GB |
| Cosmos Reason 2 8B | FP16 | 20.8 GB |
| Qwen3-VL 8B | FP16 | 20.8 GB |

The default Cosmos Reason3 Nano checkpoint is now BF16. Older utilization
values were tuned against FP8 variants, so verify its actual startup footprint
and leave additional headroom until the BF16 values are revalidated.

## Effective utilization knobs

| Serving path | Knob |
|---|---|
| LLM NIM | Set both `NIM_KVCACHE_PERCENT` and `NIM_GPU_MEM_FRACTION`; behavior is version-dependent. |
| Current Cosmos Reason3 NIM | `NIM_GPU_MEMORY_UTILIZATION` |
| Legacy Cosmos Reason 1 NIM | `NIM_KVCACHE_PERCENT` |
| RT-VLM | `RTVI_VLLM_GPU_MEMORY_UTILIZATION` |
| Generic vLLM/DLFW | `--gpu-memory-utilization` or its supported env passthrough |
| Default Cosmos-Embed1 RT-Embed | No vLLM utilization knob; size with stream count, workers, and batch size. |
| RT-CV | No model-memory fraction; size with stream count and model family. |

For shared services, the sum of their fractions must stay within the GPU
budget. A common H100 or RTX PRO 6000 starting point for a Nano 9B LLM plus
RT-VLM is `0.40 + 0.40`, leaving 20% unallocated.

## Developer-profile layouts

| Foundation | Starting layout | Important constraints |
|---|---|---|
| Base | One GPU: LLM + integrated RT-VLM shared. Two GPUs: dedicate GPU 0 to LLM and GPU 1 to RT-VLM. | Use `0.40 + 0.40` as the H100/RTX PRO 6000 shared starting point. A 48 GB L40S cannot fit the default FP16/BF16 pair inside its 40.8 GB budget. |
| Alerts `2d_cv` | GPU 0: RT-CV. GPU 1: LLM + RT-VLM; `rtvi-vlm` performs per-clip verification through Alert Bridge. | Set `RESERVED_DEVICE_IDS=0`. Size the shared LLM and RT-VLM against the combined budget. |
| Alerts `2d_vlm` | No RT-CV; default device values co-locate LLM + RT-VLM on GPU 1. Move RT-VLM to the free GPU 0 when possible. | Continuous VLM inference needs more headroom; prefer separate GPUs or a user-approved remote model endpoint. |
| LVS | One GPU: LLM + RT-VLM shared. Two GPUs: LLM on GPU 0 and RT-VLM on GPU 1. | When shared on H100/RTX PRO 6000, set `RTVI_VLLM_GPU_MEMORY_UTILIZATION=0.40` and cap the LLM at about `0.40`. |
| Search | GPU 0: RT-CV + RT-VLM FP8 at `0.40`. GPU 1: RT-Embed + LLM. | The stock local profile uses two shared GPUs (`FIXED_SHARED_DEVICE_IDS=0,1`). |

**DGX Station GB300 is one GPU.** Do not inherit the two-GPU device IDs above.
Every GPU consumer in Base, LVS, Alerts (`2d_cv` and `2d_vlm`), and Search
lands on one selected GB300. Resolve its index from the prerequisite GPU
inventory: use the sole GB300 when exactly one is present; when multiple are
present, ask which one to use. Do not inherit a Foundation device ID that does
not identify the selected GB300.

Write this complete placement closure directly to the build's
`_builds/<name>/override.env`:

```text
HARDWARE_PROFILE=GB300
LLM_MODE=local_shared
VLM_MODE=local_shared
LLM_DEVICE_ID=<gb300-id>
VLM_DEVICE_ID=<gb300-id>
SHARED_LLM_VLM_DEVICE_ID=<gb300-id>
RT_CV_DEVICE_ID=<gb300-id>
RT_EMBED_DEVICE_ID=<gb300-id>
RT_VLM_DEVICE_ID=<gb300-id>
RESERVED_DEVICE_IDS=
FIXED_SHARED_DEVICE_IDS=<gb300-id>
RTVI_VLLM_GPU_MEMORY_UTILIZATION=0.2
RTVI_VLLM_ATTENTION_BACKEND=TRITON_ATTN
```

`HARDWARE_PROFILE=GB300` plus `LLM_MODE=local_shared` selects
`hw-GB300-shared.env`, whose LLM fraction is `NIM_GPU_MEM_FRACTION=0.3`
(~75 GiB). RT-VLM `0.2` reserves ~50 GiB of ~251 GiB. Do not apply the H100
`0.40 + 0.40` pair or the Search `(VRAM-10)/VRAM - 0.15` LLM formula; vLLM
reserves `fraction × total` without subtracting co-residents.

SBSA tags are also part of the build override. Because `containers.env` is
expanded before `override.env`, setting only `VSS_CONTAINER_TAG_SUFFIX=-sbsa`
in the build override is too late to recompute per-image tags. Read the
effective `VSS_CONTAINER_TAG` — the tag the build selected, or `develop-latest`
when it selected none — and derive `<sbsa-tag>` idempotently: if the tag
already ends in `-sbsa`, use it unchanged; otherwise append `-sbsa` exactly
once. Write that concrete `<sbsa-tag>` for the services the Foundation uses:

| Foundation | Additional `override.env` values |
|---|---|
| Base | `VSS_RT_VLM_TAG=<sbsa-tag>` |
| LVS | `VSS_RT_VLM_TAG=<sbsa-tag>`, `VSS_VIDEO_SUMMARIZATION_TAG=<sbsa-tag>` |
| Alerts `2d_cv` | `VSS_RT_CV_TAG=<sbsa-tag>`, `VSS_RT_VLM_TAG=<sbsa-tag>` |
| Alerts `2d_vlm` | `VSS_RT_VLM_TAG=<sbsa-tag>` |
| Search | `VSS_RT_CV_TAG=<sbsa-tag>`, `VSS_RT_EMBED_TAG=<sbsa-tag>`, `VSS_RT_VLM_TAG=<sbsa-tag>` |

Verify these device IDs, modes, utilization values, and concrete `-sbsa` image
tags in `resolved.yml` before deployment.

RT-VLM placement and utilization starting values:

| Placement | Example profile and hardware | `RTVI_VLLM_GPU_MEMORY_UTILIZATION` |
|---|---|---:|
| Shared with another GPU service | Search FP8 on H100 or RTX PRO 6000; Alerts/LVS BF16 on H100, RTX PRO 6000, or DGX Spark | 0.40 |
| Shared on DGX Station GB300 | Base, LVS, Alerts, and Search on the single GB300 | 0.2 |
| Dedicated | Alerts/LVS BF16 on H100, RTX PRO 6000, or supported discrete GPUs not listed below | 0.70 |
| Dedicated | Alerts/LVS BF16 on L40S | 0.80 |
| Dedicated | Alerts BF16 on RTX PRO 4500 (remote LLM) | 0.80 |

**Ask before co-locating.** When the Foundation puts RT-VLM on the same GPU as
another model and the host has a free GPU, ask the user which layout they want
before writing `override.env`. Ask every time — never infer the answer from
"stock", from the Foundation's device IDs, or from the fact that both models fit.
Offer two choices and write the utilization the chosen one names:

- **One GPU** — keep the Foundation's device IDs and set the shared value above.
- **Two GPUs** — point `RT_VLM_DEVICE_ID` at the free GPU and set the dedicated
  value above.

Set `RTVI_VLLM_GPU_MEMORY_UTILIZATION` explicitly either way. Every Foundation
ships it blank, and RT-VLM reads blank as the **dedicated** `0.7` no matter what
shares the device, so leaving it blank on a shared GPU overcommits the card and
the second model to start dies at init.

These values apply when `rtvi-vlm` is in the effective service set, including
stock Alerts `2d_cv` and `2d_vlm`. The BF16 co-resident row is a stock-Foundation
layout (Alerts/LVS share BF16 with the LLM); a generated build that must converge
variants co-resides on FP8, per step 4 of the sizing flow. Do not share the
Alerts LLM and RT-VLM on L40S. RTX PRO 4500 Blackwell is alerts-only: use a
remote LLM, start RT-VLM with `RTVI_VLLM_GPU_MEMORY_UTILIZATION=0.80` and
`RTVI_VLM_MAX_MODEL_LEN=18000`.

## Warehouse industry-profile layout

Warehouse sizes differently from every developer profile: stream capacity, not
model memory, is the binding constraint. `NUM_STREAMS` is fixed by the sample
dataset (`profiles/warehouse.md`), so the question is not "how many streams fit"
but "does this hardware support the count this dataset requires".

`blueprint_config.yml` is authoritative for that ceiling. `HARDWARE_PROFILE`
selects the section; `MODE` selects the row. For the supported deployment
targets covered here:

| `HARDWARE_PROFILE` | `2d` | `3d` |
|---|---:|---:|
| H100 | 61 | 19 |
| L40S | 28 | 9 |
| GB300 | 161 | 71 |
| RTXPRO6000BW | 52 | 21 |
| RTXPRO6000BW-SE | 47 | 20 |
| RTXPRO4500BW | 20 | 9 |
| IGX-THOR | 9 | 8 |
| DGX-SPARK | 7 | 7 |

Check the dataset's stream count against the cell before deploying:

- `nv-warehouse-4cams` (2D `bp_wh`) — 4 streams. Fits every profile listed
  above.
- `warehouse-loading-dock-3cams-synthetic` (2D kafka/redis) — 3 streams. Fits
  every profile listed above.
- `warehouse-4cams-20mx20m-synthetic` (3D) — 4 streams. Fits every profile
  listed above.

`NUM_STREAMS` is an input and is never rewritten — the file-count prerequisite
that would recompute it from the video directory is `enabled: false` in every
mode. The configurator derives `final_stream_count = min(NUM_STREAMS,
max_streams_supported)` from it.

The ceiling is **primarily a DeepStream batch bound**, but not only that. Where
each value lands in `3d`:

| Rendered key | Target | From |
|---|---|---|
| `num_sensors` | `${DS_CONFIG_DIR}/config.yaml` | `final_stream_count` |
| `batch-size`, `max-batch-size` | `${DS_CONFIG_DIR}/ds-main-config.txt` | `final_stream_count` |
| `network-input-shape` | `${DS_CONFIG_DIR}/ds-mtmc-preprocess-config.txt` | `final_stream_count` |
| `data.nv_streamer_sync_file_count` | nvstreamer `vst-config.json` | `final_stream_count` |
| `onvif.max_devices_supported` | nvstreamer `vst-config.json` **and** VST `vst_config.json` | `effective_max_streams_supported` |
| video `keep_count` trim | `${VSS_DATA_DIR}/videos/<dataset>` | `enabled: false` — never runs |

**What the ceiling never reaches is the sdrc/WDM path.** Raw `NUM_STREAMS` is
rendered into the sdrc templates as `WDM_WL_THRESHOLD` — the SDR controller's
workload-dispatch threshold, for both the streamprocessing and rtvi-cv
workloads. So exceeding the ceiling leaves an internally inconsistent build —
ingest and inference sized to the ceiling, workload admission to the requested
count — rather than a clean downshift.

**Nothing enforces this automatically.** `HARDWARE_PROFILE` reaches no service
`environment:` block, so Compose cannot see it, and no rule in
`blueprint_config.yml` compares `NUM_STREAMS` to the ceiling. Check the cell
against the dataset yourself before deploying, then verify after bring-up the
same way you verify liveness: `Active sources` must equal `NUM_STREAMS`
(`profiles/warehouse.md`).

**An unrecognized `HARDWARE_PROFILE` silently matches no section** and skips
that profile's stream limit *and* its `file_operations` tuning, so the build
runs on the common configuration — a quiet capability loss, not an error. The
configurator only uppercases the value; it does not normalize spacing or
hyphenation, so `IGX THOR` does **not** match the canonical `IGX-THOR`. Use a
name from the table above verbatim, and confirm what resolved in the
`bp-configurator-<mode>` log:

```bash
docker logs vss-configurator 2>&1 | grep -aE "HARDWARE_PROFILE|hardware profile"
```

`Using common configurations only` there means the name did not match and the
tuning above was skipped.

GPU placement by variant — warehouse never runs a standalone VLM NIM
(`VLM_MODE` is `none`, or `remote` for a 2D `bp_wh` build pointing the
integrated RT-VLM at an external endpoint):

| Variant | GPU consumers | Layout |
|---|---|---|
| `bp_wh_kafka` / `bp_wh_redis`, `2d` or `3d` | RT-CV only | Everything else is CPU-bound (ELK, VIOS, analytics). One GPU at `RT_CV_DEVICE_ID=0` is the whole budget. |
| `bp_wh`, `2d` | RT-CV, integrated RT-VLM, LLM NIM | Stock layout is `RT_CV_DEVICE_ID=0`, `RT_VLM_DEVICE_ID=1`, `LLM_DEVICE_ID=2`. With fewer GPUs, place RT-CV first (fixed footprint) and apply the RT-VLM placement rules above; a remote LLM removes the third consumer entirely. |

Headless warehouse perception is far lighter than a VLM profile — a measured
4-stream 3D Sparse4D build (FP16, cached TensorRT engine) used **under 2 GB**
of a 48 GB card. Do not budget a headless warehouse build as if it hosted a
model server; the constraint is the stream ceiling above, not the memory
formula.

## Search stream sizing

The default Cosmos-Embed1 path uses Triton/ONNX. Reserve about 10 GB for
RT-Embed and give the shared LLM the remaining budget:

```text
LLM fraction = (GPU_VRAM_GB - 10) / GPU_VRAM_GB - 0.15
```

| GPU hosting RT-Embed + LLM | LLM starting fraction |
|---|---:|
| H100 / A100 80 GB | 0.72 |
| H200 | 0.78 |
| RTX PRO 6000 Blackwell | 0.75 |
| DGX Station GB300 | 0.3 (`hw-GB300-shared.env`); do not use the formula above |
| L40S | 0.65; verify under load |

Dedicated RT-Embed stream ceilings from its benchmark data:

| GPU | Maximum dedicated streams |
|---|---:|
| H100 80 GB SXM | 140 |
| H100 80 GB PCIe / H100 NVL | 100 |
| RTX PRO 6000 Blackwell | 120 |
| L40S | 60 |
| A40 | 30 |
| Thor / DGX Spark | 30 |

These are dedicated-GPU ceilings, not shared-layout targets. Start Search at
`NUM_STREAMS=16` on H100 or RTX PRO 6000. Use `NUM_STREAMS=8` on L40S, Thor,
or DGX Spark when memory is shared. Reduce `RTVI_EMBED_NUM_VLM_PROCS` from 10
to 4 if RT-Embed crowds out the LLM.

RT-CV memory is driven primarily by `NUM_STREAMS`, `DS_MODEL_FAMILY`, and
`DS_TRACKER_REID`. Start at 16 streams on H100, RTX PRO 6000, or L40S and
reduce the count on smaller or co-located GPUs.

## Edge and unified memory

Read `edge.md` for DGX Spark and AGX/IGX Thor. CPU, GPU, page cache, and
containers share one pool on those systems. Size against actual free memory,
keep the sum of co-resident fractions at or below `0.80`, and preserve at least
20% for the OS and runtime.

Use the platform-specific model and image path from `edge.md`; do not apply
x86 discrete-GPU assumptions.

Search runs on **DGX Spark and AGX Thor only**, never IGX Thor. On both boards
the single GPU is taken by the perception pipeline, so the LLM **and** the VLM
must be remote: set `LLM_MODE=remote` and `VLM_MODE=remote` with `LLM_BASE_URL`
and `VLM_BASE_URL`, and do not place `rtvi-vlm` or an LLM NIM locally. Device ID
overrides do not apply; every local service lands on the board's GPU.

## Validate and tune

Before deployment, verify in `resolved.yml`:

- every GPU service has the intended device ID;
- every shared GPU's utilization fractions stay within its budget;
- `RESERVED_DEVICE_IDS` and `FIXED_SHARED_DEVICE_IDS` match the layout;
- remote services have no unintended local model service;
- RT-VLM does not coexist with an unintended standalone VLM;
- stream counts and worker counts match the budget.

After startup:

1. Confirm model weights load and readiness endpoints pass.
2. Exercise the requested workload while sampling `nvidia-smi`.
3. For startup OOM, reduce the relevant utilization fraction by `0.05`.
4. For inference OOM, also reduce model length or sequence concurrency.
5. For RT-Embed or RT-CV pressure, reduce workers, batch size, or streams.
6. Regenerate and revalidate `resolved.yml` after every adjustment.

Never report a sizing value as validated solely because the container started;
verify it under the requested workload.
