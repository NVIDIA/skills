# DeepSeek-R1 NVFP4 decode on 8x B200 - measurements

Setup: DeepSeek-R1 NVFP4 checkpoint (`quant_algo: NVFP4`), TensorRT-LLM PyTorch
backend, 8x B200 (sm_100), TP8 / EP8, concurrency 512, ISL/OSL 1k/2k, CUDA graphs on,
overlap scheduler on (default). Current `moe_config.backend: CUTLASS`.

## Host-overhead detection (perf-host-analysis)

| Metric | Value | Threshold | Crossed? |
|---|---|---|---|
| GPU idle ratio | 3.8% | > 30% | NO |
| cudaLaunchKernel overhead ratio | 1.1% | > 10% | NO |
| GPU utilization (time-based) | 96.2% | < 60% | NO |
| NCCL ratio (caveat) | 11% | > 20% | NO |

Verdict: NO - host overhead is not the bottleneck.

## Kernel time per decode step (nsys, rank 0)

| Kernel group | Share of step |
|---|---|
| MoE grouped GEMM (`cutlass_moe_fp4_grouped_gemm_*`, fc1 + fc2) | 57% |
| MLA decode attention | 16% |
| All-to-all dispatch/combine + allreduce | 11% |
| Router, quantize, RMSNorm and other small kernels | 9% |
| Everything else | 7% |

## Ideas currently proposed by the team

1. Tune the overlap scheduler.
2. Move per-step bookkeeping from Python into a C++ op.
3. Enable piecewise CUDA graphs for prefill.
