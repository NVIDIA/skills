# Llama-3.1-8B TP=2 throughput regression: v1.2 -> main

Benchmark: `trtllm-bench throughput`, Llama-3.1-8B-Instruct BF16, TP=2 on 2x H200,
ISL 128 / OSL 128, 500 requests. Same container, same flags, same GPUs.

| Build | Request throughput |
|---|---|
| v1.2 | 452.1 req/s |
| main | 371.3 req/s (-17.9%) |

Both runs were captured with `nsys profile -t cuda,nvtx,osrt -w true` and exported
to SQLite. Raw traces are not attached; the extracts below were produced from them.

## 1. First attempt (teammate) - kernel-density window

The teammate picked the 0.87 s window with the highest kernel density in each trace
(main: t = 5.84-6.71 s). In that window main launched 41,200 kernels; 97% of them
were `triton_poi_fused_*` / `reduce_kernel` elementwise/reduce kernels and the window
contained 0 GEMM and 0 attention kernels. Conclusion written in the ticket:
"main launches way more elementwise kernels per step -> kernel launch regression".

## 2. Allreduce-based iteration detection (both traces)

- Allreduce kernels grouped by < 1 ms gaps; modal group size = 66 allreduce kernels
  per forward step in both builds.
- Phase boundaries (> 100 ms gaps) found; the benchmark phase is the last phase with
  the modal group size: 201 steps (v1.2) and 203 steps (main).
- NVTX `[Executor] _forward_step` ranges de-duplicated across the 2 TP ranks
  (entries within 100 us grouped); NVTX step count matches the allreduce step count.

## 3. Per-step timing (benchmark phase, steady state: 0 ctx reqs, max gen reqs)

| Metric | v1.2 | main | Delta |
|---|---|---|---|
| Avg wall time per forward step | 3,412 us | 4,071 us | +19.3% |
| Inter-step gap P50 | 2,104 us | 2,642 us | +25.6% |
| GPU kernel time per step (sum of kernel durations) | 2,214 us | 2,219 us | +0.2% |
| Kernels per step (individually launched) | 6.4 | 20.8 | +225% |
| cudaLaunchKernel host time per step | 31 us | 96 us | +65 us |

## 4. NVTX per-step breakdown (host operations, us/step, TP-deduplicated)

NVTX durations are host-side and partly overlap GPU execution, so they are not
additive to the wall-time delta.

| Operation | v1.2 | main | Delta |
|---|---|---|---|
| _schedule | 119 | 122 | +3 |
| _fetch_new_requests | 38 | 281 | +243 |
| broadcast_requests | - | 247 | +247 (new) |
| _prepare_inputs | 702 | 731 | +29 |
| _sample_async | 1,151 | 726 | -425 |
| _process_requests | 1,047 | 402 | -645 |
| _update_requests | 405 | 739 | +334 |
| _handle_responses | 331 | 384 | +53 |
| _write_finish_reasons | - | 128 | +128 (new) |
| prepare_resources | 190 | 352 | +162 |
