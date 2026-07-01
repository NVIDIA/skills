# Config Derivation Rules

Stage 4 closed-form rules. Inputs come from Stages 2 (HW) and 3 (inference plateau). Every
knob below is derived, not tuned.

## Inputs

- `plateau_batch` — from Stage 3 micro-benchmark (smallest batch where doubling B yields
  < 5% FPS gain).
- `N_streams` — user-requested stream count.
- `source_res = (W, H)` — e.g. (1920, 1080).
- `source_fps` — per stream, e.g. 30.
- `model_dims = (Wm, Hm)` — read from the model's nvinfer config (`infer-dims=3;H;W`).
  Works for detection / classification / segmentation models alike.
- `has_int8_calib` — bool; true if an `int8-calib-file` is present on disk.
- `hw_nvdec_count`, `hw_sm_count`, `hw_tops_int8`, `hw_bw_gb_s` — from Stage 2.

**Derived:**

```
final_batch = min(plateau_batch, N_streams)
```

## Rules

### R1 — nvstreammux

| Key | Value | Why |
|---|---|---|
| `batch-size` | `final_batch` | Matches what inference can consume in one tick. |
| `width` | `min(source_res.W, model_dims.Wm)` | No point upscaling before inference. |
| `height` | `min(source_res.H, model_dims.Hm)` | Same. |
| `batched-push-timeout` | `round(1e6 / source_fps)` µs | One full frame interval; waits long enough to fill a batch without starving at live sources. |
| `nvbuf-memory-type` | `0` | NVMM zero-copy. Anything else kills throughput. |
| `live-source` | `1` if RTSP/camera, else `0` | RTSP needs live-source behavior. |
| `enable-padding` | `0` | Padding wastes GPU time when input already matches model dims. |

### R2 — nvinfer (primary)

In `pgie_config.yml` / `config_infer_primary.txt`:

| Key | Value | Why |
|---|---|---|
| `batch-size` | `final_batch` | Must match streammux; mismatch silently caps FPS. |
| `network-mode` | `1` if `has_int8_calib` else `2` | INT8 is ~2× FP16 TOPS on Ampere/Ada/Hopper. FP16 as fallback; never FP32. |
| `interval` | `0` | Every frame. Only use `1` if user explicitly asked to skip frames. |
| `infer-dims` | `3;Hm;Wm` | Required for dynamic-shape ONNX; harmless for static. |
| `model-engine-file` | path to pre-built `.engine` | Do not rebuild during measurement — `trtBuilder*` would dominate the profile. |
| `int8-calib-file` | path | Only if `network-mode=1`. |

In the outer nvinfer element properties (when set via `pipeline.add(...)` or `gst-launch`):
just `config-file-path=<path-to-pgie_config.yml>`. Do not duplicate keys in both places.

### R3 — Decoder (`nvurisrcbin` / `nvmultiurisrcbin` / `nvv4l2decoder`)

| Key | Value | Why |
|---|---|---|
| `num-extra-surfaces` | `min(final_batch, 5)` | Buffer pool sized for the batch; 5 is a safe cap that doesn't waste GPU memory. |
| `cudadec-memtype` | `0` | NVMM device memory; matches `nvstreammux.nvbuf-memory-type=0`. |
| `file-loop` | `1` when using file sources for benchmarking | Keeps the pipeline saturated past the first file playthrough. |
| `max-batch-size` | `final_batch` | On `nvmultiurisrcbin` only; must equal streammux batch. |

### R4 — Tracker (if present)

In the outer `nvtracker` element properties:

| Key | Value | Why |
|---|---|---|
| `enable-batch-process` | `1` (in the linked YAML) | Batches track updates across streams. |
| `tracker-width` | `480` | Matches NVIDIA's max-perf reference config. |
| `tracker-height` | `288` | Same. |
| `ll-config-file` | `/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_tracker_NvDCF_max_perf.yml` | NVIDIA-tuned perf preset (DS 9.0). |
| `ll-lib-file` | `/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so` | Standard tracker lib. |

#### R4a — `nvinfer.interval` when a tracker is present

A tracker can carry object IDs across frames where inference did not run, so
when a tracker is in the pipeline you may safely raise `nvinfer.interval` from
`0` (every frame) to `1`, `2`, `…` (skip 1, 2, … frames between inferences),
multiplying inference capacity proportionally. Whether this is safe depends on
**whether the tracker actually keeps IDs alive across the gap** — that's a
property of the workload (object motion, occlusion frequency, frame rate),
not something a closed-form rule can pick.

The right signal is **tracker ID retention rate** measured at the tracker's
output:

```
retention(N→N+1) = |IDs in frame N+1 ∩ IDs in frame N| / |IDs in frame N|
```

Capture it by attaching a `BatchMetadataOperator` probe at the tracker's src
pad and tallying object IDs per frame across a steady-state window. Practical
thresholds:

| Retention | Interpretation | Recommendation |
|---|---|---|
| **≥ 99 %** | Tracker keeps virtually every object across consecutive frames; gaps are safe to fill in. | Try `interval=1` (halves inference compute), re-measure retention at the new interval. Step up to 2/3/… while retention holds ≥ 99 %. |
| 90 – 99 % | Some objects flicker or are reacquired with new IDs, but most persist. | `interval=1` is borderline; acceptable for many use cases but expect a few duplicate / re-issued IDs. Don't go higher. |
| **< 90 %** | A noticeable fraction of objects come out with new IDs each frame — tracker can't bridge the gaps reliably. | **Force `interval=0`.** Skipping frames will compound the ID churn; per-frame inference is required. |

Quick visual check without writing a probe: turn on `display-tracking-id=1`
on the tracker, render the OSD-overlaid output for ~5 seconds, and watch
whether the IDs labelled on objects stay stable or constantly change. Lots of
new IDs ⇒ retention is poor ⇒ keep `interval=0`.

This rule applies *only when a tracker is present*. Without a tracker,
`interval > 0` produces literal detection holes (no metadata between
inferences) and should never be used.

### R5 — Queues (only when present)

A queue between `source` and `pgie`:

| Key | Value | Why |
|---|---|---|
| `max-size-buffers` | `final_batch × 4` | Decode-side depth; keeps GPU fed when decode is bursty. |
| `max-size-bytes` | `0` (default) | Let buffer count rule. |
| `max-size-time` | `0` | Let buffer count rule. |

A queue feeding a Kafka/message branch:

| Key | Value | Why |
|---|---|---|
| `max-size-buffers` | `2` | Tiny. |
| `leaky` | `2` (downstream) | Drop oldest when broker is slow; never back-pressure the main chain. |

A queue feeding a file-sink / encoder branch:

| Key | Value | Why |
|---|---|---|
| `max-size-buffers` | `final_batch` | One-batch slack, no more (disk back-pressure is a real signal). |
| `leaky` | `0` | Don't drop; if the disk is slow we want to see it as a stall. |

### R6 — OSD / tiler / visible sinks

If the user asked for on-screen display or on-disk MP4: leave OSD and tiler enabled; they are
a known perf cost the user accepted. Do not re-tune their knobs in the MVP.

If the user did NOT ask for visible output: omit OSD + tiler entirely; use `fakesink
sync=False` as the only branch. This is the largest single perf delta available.

### R7 — Capacity report (max output)

After Stage 5 (E2E profile), compute the closed-form max-stream capacity for the user's
target FPS. Report this in plain English.

**Inputs:**
- `peak_per_frame_fps` — from the microbench plateau (Stage 3)
- `nvdec_decode_ceiling` — from the Stage 2 HW ceiling lookup
  (`NVDEC_count × per_unit_fps_for_codec_res`, table in `hw-ceiling-formulas.md`)
- `target_fps` — what the user wants per stream (default 30)
- `aggregate_bw_ceiling_gbps`, `bw_per_stream_gbps` — only relevant if Stage 5 flagged
  MEMORY_BW_BOUND

**Formula:**

```
max_streams_compute  = peak_per_frame_fps / target_fps
max_streams_decode   = nvdec_decode_ceiling / target_fps
max_streams_bw       = aggregate_bw_ceiling_gbps / bw_per_stream_gbps   # only if BW-bound
max_streams_overall  = min(max_streams_compute, max_streams_decode, max_streams_bw)
```

**The reported number is `max_streams_overall`** — the lowest of the three ceilings, since
that is the actual bottleneck. The skill should also state *which* ceiling is dominating, so
the user knows what to upgrade if they need more.

**Output template (skill should produce a block in this shape):**

```
Capacity (this hardware, this model, <precision>):
  Compute ceiling : <peak_per_frame_fps> fps / <target_fps> fps = <N_compute> streams
  Decode  ceiling : <nvdec_decode_ceiling> fps / <target_fps> fps = <N_decode> streams
  Memory  ceiling : (not bound)  OR  <N_bw> streams
  → Max output:    min(...) = <N_overall> streams  (<dominant>-limited)
```

**Bottleneck → remediation mapping** (the skill must include this section in the report,
keyed off which ceiling dominated):

| Dominant ceiling | What "more streams" requires |
|---|---|
| **COMPUTE_BOUND** | A larger / faster GPU (more SMs, higher clock, or newer architecture with higher TC throughput). Lowering precision (FP16→INT8) is the cheapest first move IF not already INT8. Shrinking the model or using `nvinfer.interval` to skip frames also unlocks capacity at the cost of accuracy / temporal resolution. |
| **DECODE_BOUND** | A GPU with more NVDEC engines (counts vary widely — datacenter cards usually have more). Lowering input resolution or switching codec (H265 is slightly cheaper to decode than its H264 equivalent on most hardware) also helps. Pre-decoding to disk is an option for offline workloads. |
| **MEMORY_BW_BOUND** | A GPU with higher memory bandwidth (HBM-class cards). INT8 weights help (halve weight bytes). Smaller model or lower input resolution shrinks the working set. |
| **TRACKER_BOUND** | Switch to the perf-tuned tracker preset (R4). If already on max-perf preset and still bound, drop tracker resolution further or swap tracker algorithm (NvDCF→IOU). |
| **SYNC_BOUND** | Investigate threading / blocking-sync; rarely fixed by hardware change. |

**The skill must explicitly say which ceiling is dominating** so the user knows whether
upgrading to a larger compute GPU or one with more NVDECs is the right purchase. The skill
cannot run on a hypothetical larger GPU itself — its job is to identify the bottleneck and
project the remediation.

## Sanity checks the skill must run after applying R1–R5

Before launching Stage 5:

1. `nvstreammux.batch-size == nvinfer.batch-size` — else BATCH_MISMATCH; fix.
2. `nvinfer.model-engine-file` exists on disk — else the first run will rebuild and pollute
   the profile. Pre-build:

   ```bash
   # Trigger engine build once outside profiling
   gst-launch-1.0 fakesrc num-buffers=10 ! nvinfer config-file-path=<path> ! fakesink
   ```
3. `nvbuf-memory-type == 0` everywhere it applies — else memcpy dominates.
4. If `network-mode=1`, the calibration file exists and matches the model architecture.
5. All queue `leaky` values match the rules above.

If any check fails, fix and re-run before profiling.
