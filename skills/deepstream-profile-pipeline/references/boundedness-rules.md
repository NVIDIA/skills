# Boundedness Rules

Map measured signals to a single bound type. The skill **must** classify the bound type
explicitly; raw `nsys stats` CSVs alone are not a diagnosis.

> **Lead with non-NVTX signals.** DS plugin NVTX coverage is inconsistent across DS versions
> and container variants — some images strip the CUPTI NVTX shim, named-domain ranges may
> be filtered, and dlopen'd plugin libs aren't always intercepted. Use NVTX share as
> confirmation when present, never as the primary signal. The skill must always be able to
> classify the bound type without it.

## Inputs the rules consume (in order of reliability)

1. **Microbench scaling shape** (Stage 3 output) — *always works*, doesn't need nsys.
2. **`nvidia-smi dmon -s u`** captured during the run — *always works*, gives live decoder /
   encoder / SM / memory utilization.
3. **`nsys stats --report cuda_gpu_kern_sum`** — *always works* with `--trace=cuda`; tells
   which kernels dominate.
4. **`nsys stats --report cuda_gpu_mem_time_sum`** — *always works*; H2D/D2H/D2D totals.
5. **`nsys stats --report cuda_api_sum`** — *always works*; CPU sync hotspots.
6. **`nsys stats --report gpu_metric_gpu_util_sum`** — needs `--gpu-metrics-devices=all` and
   `CAP_SYS_ADMIN`. Often unavailable in containers.
7. **`nsys stats --report nvtx_sum`** — *unreliable across DS versions*; bonus only.

## Decision tree (run in order; first match wins)

### 1. Engine-rebuild contamination (always check first)

Trigger: `cuda_api_sum` shows `cuModuleLoadData` > 10% of duration, or `cuda_gpu_kern_sum`
top kernels include `trtBuilder*` or `cask_engine_build*`.

Bound: `ENGINE_BUILD_CONTAMINATED` (transient — first run only).

Action: pre-build the engine (`gst-launch ... ! nvinfer config-file=... ! fakesink num-buffers=10`)
and re-measure.

### 2. NVMM broken / memcpy bound

Trigger: H2D + D2H + D2D total > **15% of wall time** (from `cuda_gpu_mem_time_sum`).

Bound: `MEMCPY_BOUND`.

Fix: search the pipeline for a CPU `videoconvert`, missing `memory:NVMM` caps, or
`nvbuf-memory-type != 0`.

### 3. Decode / source bound

Trigger (any one — high reliability):

- **Microbench scaling shape:** `fps_aggregate[B=1] / fps_aggregate[B=2]` < 0.55 — i.e.
  doubling parallel sources nearly doubled aggregate FPS, meaning decoder was the limit at
  B=1 (NVDEC engine count was the bottleneck). **This is the most reliable signal** because
  it doesn't require NVTX.
- `nvidia-smi dmon -s u` shows `dec` column ≥ 90% during steady state.
- (NVTX bonus): `nvstreammux:m_collectingBuffers` > 40% of NVTX time, *if* NVTX is captured
  in your DS+nsys combo.

Bound: `DECODE_BOUND`.

Fix: more parallel sources (each gets its own NVDEC slice), lower input resolution, drop to
H264 from H265 (slightly cheaper to decode), pre-decode to disk if the source is small.

### 4. Compute bound (the desired state at peak FPS)

Trigger (all of):
- TRT GEMM kernels (`*xmma_*`, `*cask_*`, `*sm{70,80,90}_*_int8_*` or `*_fp16_*`) > **50% of
  CUDA kernel time**, **and**
- microbench `fps_per_batch[B]` plateaus by B=4 (≤ 5% gain doubling B), **and**
- if available: `gpu_metric_gpu_util_sum` SM Active > 70%.

Bound: `COMPUTE_BOUND`.

Action: this is the goal at peak FPS. Lower precision (FP16 → INT8) doubles ceiling on
Ampere/Ada/Hopper. If already INT8, the answer is "more / bigger GPU" or "smaller model."

### 5. Memory-bandwidth bound

Trigger (need `--gpu-metrics-devices`, so often unavailable):
- DRAM throughput > **85% of peak** AND SM Active < 70%.

Bound: `MEMORY_BW_BOUND`.

Fix: smaller model, lower resolution, INT8 (halves weight bytes), reduce batch if cache
thrashing.

### 6. Tracker bound

Trigger: per-element frame-counter probe shows tracker-output FPS << pgie-output FPS
(attach two probes — one at pgie src pad, one at tracker src pad — and compare); **or**
(NVTX bonus) `NvDsTracker*` total > nvinfer total.

Bound: `TRACKER_BOUND`.

Fix:
  - If R4's defaults aren't already applied: apply them
    (NvDCF max-perf preset, 480×288, `enable-batch-process: 1`).
  - If R4 is already in place and you're still tracker-bound: drop tracker
    resolution further (e.g. 384×216), or switch tracker algorithm
    (NvDCF → IOU is much cheaper at the cost of weaker re-ID quality).

### 7. Sync / CPU bound

Trigger: `cuda_api_sum` shows `cudaStreamSynchronize` or `cudaEventSynchronize` > **30% of
total CPU time**.

Bound: `SYNC_BOUND`.

Fix: only flip `NVDS_DISABLE_CUDADEV_BLOCKINGSYNC=1` if also confirmed compute-bound;
otherwise look upstream for starvation.

### 8. None of the above

Report `UNKNOWN_BOTTLENECK` with the top 10 CUDA kernels, top memcpy categories, and
microbench scaling table. Do not invent a cause.

## Microbench scaling shape — the single most useful signal

| Pattern | What it means |
|---|---|
| `fps_per_batch` doubles from B=1→B=2, plateaus B≥4 | Compute-bound at B≥4. Decoder was the limit at B=1 (single NVDEC). |
| `fps_per_batch` plateaus immediately B=1→B=2 | Compute-bound from B=1. Decoder has plenty of headroom. |
| `fps_aggregate` scales linearly through B=16 | GPU has compute headroom; you can serve more streams. |
| `fps_aggregate` plateaus at some B*N | At the real ceiling for THIS combo of HW + codec + model. |
| All B fail (FPS=0) | Source/streammux can't fill the batch — pipeline error or single-source-only with batch>1. |

Worked example (any small / cheap inference model on a file source): single-source per-batch
FPS caps at the single-NVDEC ceiling for the codec+resolution. With two parallel sources of
the same file, per-batch FPS jumps ~2× as the second NVDEC engages. Aggregate FPS plateaus
once all NVDEC engines are saturated. Per-batch *plateau* alongside *idle compute* (low SM%)
is the signature: decode-bound, not compute-bound. Heavier models (transformer detectors,
VLMs) flip the picture — same shape but compute saturates first.

## How the skill should report

After running `nsys stats` plus the microbench, produce a single block like:

```
Bound type: DECODE_BOUND (high confidence)

Evidence:
  - Microbench: fps_aggregate[B=1]/fps_aggregate[B=2] = <ratio>  (threshold < 0.55)
  - nvidia-smi dmon: decoder utilization peaks observed during steady state
  - (NVTX bonus, if present): nvstreammux:m_collectingBuffers > 40%

Action: add parallel sources, or lower input resolution / switch codec.
HW ceiling check (Stage 2):
  Detected GPU has <N_NVDEC> NVDECs × ~<per_unit_fps> fps for the source codec/res =
  ~<decode_ceiling> fps decode ceiling.
  Measured aggregate at the relevant B compared against that ceiling.

Max output for this pipeline on this hardware (R7):
  decode_ceiling / target_fps = <max_streams> streams at <target_fps> fps before NVDEC
  saturates. Adding more input streams beyond that hits the decode wall, not compute.
```

That single block answers all four user questions: bound type, fix, max output, and
already-applied configs.

## Cross-checking

Each bound type should be confirmed by **at least two** independent signals before reporting
high confidence. The decision tree above already encodes this — DECODE_BOUND, for example,
has three independent triggers. If only one signal fires and others contradict, report
`INCONCLUSIVE` with the contradicting evidence.
