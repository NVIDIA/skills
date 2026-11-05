# Kernel-level drill-down extract (rank 0)

Workload: DeepSeek-V3 FP8 with MTP speculative decoding (max_draft_len 3),
TensorRT-LLM PyTorch backend, 8x B200, TP8/EP8, concurrency 32, piecewise CUDA
graphs enabled via torch.compile. Window: 200 steady-state decode steps
(0 ctx reqs, 32 gen reqs), isolated with allreduce grouping + NVTX
`_forward_step` ranges. Median step time: 31.4 ms.

Detection already ran on this trace: overall verdict YES (GPU idle ratio 0.43,
generation phase GPU idle 0.43 > 0.15), NCCL ratio 0.09 (no caveat).

## 1. Inter-kernel gaps (same device + stream, positive gaps only)

| Gap bucket | Count | Total (ms) | Share of gap time |
|---|---|---|---|
| < 1 us | 852,000 | 383 | 14.3% |
| 1-5 us | 54,300 | 130 | 4.9% |
| 5-10 us | 8,900 | 62 | 2.3% |
| 10-50 us | 22,600 | 429 | 16.0% |
| 50-100 us | 13,180 | 975 | 36.5% |
| 100-500 us | 620 | 130 | 4.9% |
| 500 us-1 ms | 70 | 49 | 1.8% |
| 1-5 ms | 211 | 329 | 12.3% |
| > 5 ms | 21 | 187 | 7.0% |
| **Total** | 951,902 | 2,674 | 100% |

Location notes:
- 50-100 us gaps sit between consecutive `cudaGraphLaunch` replays of piecewise
  segments inside each of the 61 decoder layers (graph segment -> eager attention
  custom op -> next graph segment).
- 1-5 ms gaps cluster in the MTP tail of each step, between eager kernels of the
  3 MTP draft iterations; the host thread is in Python with no CUDA API call in flight.

### Top-5 largest gaps with host-side context (CUDA API + NVTX at the same time)

| # | Gap | Host thread during the gap | Enclosing NVTX range |
|---|---|---|---|
| 1 | 9.8 ms | `cudaStreamSynchronize`, preceded by a 4-byte D2H `cudaMemcpyAsync` | `_sample_async` |
| 2 | 9.1 ms | 3x `cudaEventCreateWithFlags`, then `cudaStreamWaitEvent` | `MTPWorker.forward` |
| 3 | 8.7 ms | `cudaStreamSynchronize`, preceded by an 8-byte D2H copy | `_prepare_tp_inputs` |
| 4 | 8.2 ms | `cudaEventCreateWithFlags` | `MTPWorker.forward` |
| 5 | 7.9 ms | `cudaStreamSynchronize` | `_sample_async` |

## 2. Eager vs graph kernels (per steady-state step, averaged)

| Launch path | Kernels/step | What they are |
|---|---|---|
| `cudaGraphLaunch` (piecewise replay) | 2,904 | GEMMs, RMSNorm, allreduce-fusion, MoE kernels of the 61 main decoder layers |
| `cudaLaunchKernel` (eager) | 1,858 | see breakdown below |

Eager breakdown:
- 1,624/step: MLA decode attention + KV-cache update kernels (the attention custom op
  is the piecewise split point in every layer).
- 234/step: the MTP draft path. The first eager MTP kernel is
  `triton_red_fused_cumsum_sub_0` (from a `torch.cumsum` on draft-token metadata,
  `mtp.py:987`); every kernel after it in the 3 MTP draft iterations runs eagerly,
  including `nvjet_*` GEMMs, `fusedAddRMSNormKernel`,
  `allreduce_fusion_kernel_oneshot_lamport` and `fp8_blockscale_moe_*` kernels -
  the same kernel types that are graph-captured in the main decoder layers.

torch.compile log for the MTP graph:

```
[piecewise] stop_partition=True at aten.cumsum.default (mtp.py:987); remaining nodes in this graph are not captured
```
