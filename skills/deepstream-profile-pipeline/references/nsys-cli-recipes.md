# `nsys` CLI Recipes

Everything in this skill is captured and extracted with `nsys` on the terminal. No GUI. No
Nsight Lens. If `nsys` is not on PATH, the skill cannot run — install it first
(`apt install nsight-systems-<version>` or use a DS container that ships with it).

## Capture — Stage 3 (inference-only micro-benchmark)

Same flags as Stage 5, short duration because the micro-bench only needs steady-state FPS.

```bash
TS=$(date +%Y%m%d_%H%M%S)
OUT=/tmp/ds_microbench_B${BATCH}_${TS}

nsys profile \
  --trace=cuda,nvtx \
  --force-overwrite=true \
  --duration=20 \
  --output=${OUT} \
  <inference-only-pipeline-launch>
```

Then read FPS either from the pipeline's own `measure_fps_probe` stdout, or from:

```bash
nsys stats --report nvtx_sum --format csv ${OUT}.nsys-rep \
  | awk -F, '$3 ~ /nvinfer/ {print $0}'
```

The `Instances / duration` column of the `nvinfer` row gives per-batch inference rate →
multiply by `BATCH` for aggregate FPS.

## Capture — Stage 5 (E2E)

```bash
TS=$(date +%Y%m%d_%H%M%S)
OUT=/tmp/ds_e2e_${TS}

nsys profile \
  --trace=cuda,nvtx,osrt \
  --gpu-metrics-devices=all \
  --cuda-memory-usage=true \
  --force-overwrite=true \
  --duration=30 \
  --output=${OUT} \
  <full-pipeline-launch>
```

Flag-by-flag reason:

| Flag | Purpose |
|---|---|
| `--trace=cuda,nvtx,osrt` | CUDA kernels + NVTX ranges from DS plugins + OS-runtime (for CPU stall diagnosis). |
| `--gpu-metrics-devices=all` | SM occupancy, DRAM throughput, tensor-core utilization — essential for "compute- vs memory-bound" diagnosis. Requires `nsys` 2023.3+. |
| `--cuda-memory-usage=true` | Populates memcpy-byte totals so the memcpy report below has data. |
| `--force-overwrite=true` | Idempotent reruns. |
| `--duration=30` | 30 s of steady state; drop if the launch command terminates on its own within that window. |

If running inside Docker and the DS app is already running as a PID in another container,
`--attach=<pid>` works instead of a launch command (`docker exec <ctr> nsys profile ...
--attach=<pid>`).

## Extract — per-plugin NVTX time

This is the headline table in the Stage 5 report. Each DS plugin emits an NVTX range; sum
them.

```bash
nsys stats --report nvtx_sum --format csv ${OUT}.nsys-rep > /tmp/nvtx.csv

# Top 10 by total GPU time
awk -F, 'NR>1 {print $0}' /tmp/nvtx.csv | sort -t, -k5 -nr | head -10
```

Columns (approx, order may shift by `nsys` version):

1. `Range`
2. `Instances`
3. `Total Time (ns)`
4. `Avg (ns)`
5. `Med (ns)`
6. `Min`, `Max`, `StdDev`

Use `Total Time (ns)` as the share-of-wall-time measure. Normalize by `--duration` × 1e9.

## Extract — top CUDA kernels

Useful for confirming inference is the hot path (not a preproc or resize kernel).

```bash
nsys stats --report cuda_gpu_kern_sum --format csv ${OUT}.nsys-rep | head -15
```

What to look for: TRT kernels dominate (names like `trt_volta_int8_i8816cudnn_*`,
`sm80_xmma_*`, `cask_*`). If a `nppiResize` or `cudaMemcpy*` kernel is in the top 3, NVMM or
preproc is broken — but re-tuning non-NVTX helpers is out of scope this MVP; just report.

## Extract — memcpy totals

```bash
nsys stats --report cuda_gpu_mem_time_sum --format csv ${OUT}.nsys-rep
```

Expected: H2D + D2H total < 5% of wall time for a correct DS pipeline. If > 15%, something
has broken NVMM (usually a CPU `videoconvert` crept in, or `nvbuf-memory-type` isn't 0).

## Extract — GPU utilization / DRAM throughput

Requires the `--gpu-metrics-devices=all` capture flag.

```bash
nsys stats --report gpu_metric_gpu_util_sum --format csv ${OUT}.nsys-rep
```

Report keys:

- **SM Active (%)** — average SM activity. > 80% during steady state = GPU-bound (good at
  peak FPS).
- **DRAM Throughput (%)** — relative to the card's peak. > 85% = memory-bound.
- **Tensor Active (%)** — tensor core engagement. Low on an FP16/INT8 model means TRT didn't
  pick TC kernels; re-export the engine.

## Extract — OS-runtime (CPU stalls)

Optional; useful when the profile shows GPU-idle bubbles.

```bash
nsys stats --report osrt_sum --format csv ${OUT}.nsys-rep | head -15
```

Big `pthread_cond_wait` or `poll` times on pgie threads = upstream starvation (increase
queue depth or check decoder). Big `cudaStreamSynchronize` on CPU = GPU done but pipeline
waiting — usually fine on dGPU.

## Exporting CSV for downstream tooling (optional)

If a parent workflow wants to ingest the data:

```bash
# Full sqlite export (all tables — same as what Nsight Lens would read)
nsys export --type sqlite --output ${OUT}.sqlite ${OUT}.nsys-rep

# SQL query for top NVTX
sqlite3 ${OUT}.sqlite <<'SQL'
SELECT s.value AS plugin,
       COUNT(*) AS batches,
       SUM("end" - "start") AS total_ns
FROM   NVTX_EVENTS n
JOIN   StringIds s ON s.id = n.text
GROUP  BY n.text
ORDER  BY total_ns DESC
LIMIT  10;
SQL
```

The skill's Stage 5 report uses `nsys stats` directly; the sqlite path is a fallback for
environments where `nsys stats` is unavailable.

## Nothing-works fallback

If `nsys profile` fails (e.g. locked-down container without `perf_event_paranoid < 2`):

1. `sudo sysctl -w kernel.perf_event_paranoid=1` — or add `--privileged` to the container.
2. If still failing, capture with `--trace=cuda,nvtx` only (drop `osrt` and
   `--gpu-metrics-devices=all`). Less info, but at least the per-plugin NVTX table survives.
3. If `nsys` itself is unavailable: skip to a DS-native-only path using
   `measure_fps_probe` + Prometheus counters; **but report to the user that there is no
   Nsight trace and the plugin-level breakdown will be missing**. Do not pretend a DS-only
   measurement is a profile.
