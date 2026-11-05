# Base measurements extracted from an nsys SQLite export (rank 0)

Workload: Llama-3.1-70B-Instruct BF16, TensorRT-LLM PyTorch backend,
TP=8 on 8x H100 SXM, ISL 2048 / OSL 512, concurrency 64, CUDA graphs on.
Capture: `nsys profile -t cuda,nvtx,osrt`, exported to SQLite. The numbers below
were pulled with SQL from the `CUPTI_ACTIVITY_KIND_KERNEL`,
`CUPTI_ACTIVITY_KIND_RUNTIME` and `NVTX_EVENTS` tables (StringIds joined). The
`.sqlite` itself is not attached.

All values in microseconds (us) unless noted.

| Quantity | Value | How it was computed |
|---|---|---|
| Analysis window (`total_time_us`) | 4,820,000 | MAX(kernel.end) - MIN(kernel.start) |
| GPU active time, merged (`gpu_active_time_us`) | 3,085,000 | union of overlapping kernel intervals |
| `cudaLaunchKernel` total time | 106,000 | SUM(end-start) over CUPTI runtime rows named cudaLaunchKernel |
| Host-prep NVTX total (`_prepare_tp_inputs`) | 870,000 | sum of NVTX range durations, TensorRT-LLM domain |
| Host-prep exposed (`_prepare_tp_inputs` overlapping GPU-idle gaps) | 352,000 | NVTX ranges intersected with merged-kernel idle gaps |
| NCCL kernel time (`shortName LIKE '%nccl%'`) | 1,210,000 | SUM(end-start) of NCCL kernels |
| Forward steps in window | 1,412 | `[Executor] _forward_step` NVTX ranges, TP-deduplicated |

Notes from the person who ran it:

- "GPU idle is 36%, so the host is obviously the bottleneck - we want to start
  host-side optimization of `_prepare_tp_inputs` right away."
- Per-phase (context/generation) splits were not extracted.
