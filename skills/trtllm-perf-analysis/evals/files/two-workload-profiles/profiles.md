# Measurements returned by the profiling agent (nsys + CUPTI runtime trace)

Both workloads ran on a single H100 SXM 80GB. Values are per iteration, median of
200 steady-state iterations. No other metrics were collected.

## Workload A - Llama-3.1-8B decode, batch 4, CUDA graphs disabled (eager)

| Metric | Value |
|---|---|
| Iteration wall time | 12.6 ms |
| GPU busy time (union of kernel intervals) | 4.1 ms (33% of wall time) |
| Kernels launched per iteration | 1,184 |
| Mean kernel duration | 3.5 us |
| `cudaLaunchKernel` calls / host time | 1,184 calls / 7.9 ms |
| `cudaStreamSynchronize` calls / host time | 1 call / 0.2 ms |
| DRAM bandwidth (time-weighted over kernels) | 41% of peak |
| Tensor-core active (time-weighted over kernels) | 6% |
| NCCL time | 0 (single GPU) |

## Workload B - custom MoE layer microbenchmark, 32 layers, batch 64

| Metric | Value |
|---|---|
| Iteration wall time | 9.8 ms |
| GPU busy time (union of kernel intervals) | 5.0 ms (51% of wall time) |
| Kernels launched per iteration | 224 |
| Mean kernel duration | 22 us |
| `cudaLaunchKernel` calls / host time | 224 calls / 0.6 ms |
| `cudaStreamSynchronize` calls / host time | 64 calls / 4.1 ms (2 per layer, issued by `.item()` on per-expert token counts) |
| GPU idle gaps | one gap after each `cudaStreamSynchronize` returns, before the next launch |
| DRAM bandwidth (time-weighted over kernels) | 47% of peak |
| Tensor-core active (time-weighted over kernels) | 28% |
| NCCL time | 0 (single GPU) |
