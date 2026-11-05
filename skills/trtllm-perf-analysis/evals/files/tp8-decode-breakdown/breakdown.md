# Per-step breakdown returned by the profiling agent (nsys, rank 0)

Workload: Llama-3.1-70B-Instruct BF16, TensorRT-LLM PyTorch backend, TP=8 on
8x H100 SXM (NVLink/NVSwitch), steady-state decode, batch 16, CUDA graphs enabled.
Median over 200 decode steps.

| Category | Time per step | Share of step |
|---|---|---|
| NCCL AllReduce kernels (160 per step: 2 per layer x 80 layers) | 8.1 ms | 44% |
| GEMM kernels (QKV, O, gate/up, down, LM head) | 5.6 ms | 30% |
| Attention (decode, GQA) | 1.9 ms | 10% |
| RMSNorm / residual add / elementwise | 1.7 ms | 9% |
| GPU idle | 1.1 ms | 6% |
| **Step time** | **18.4 ms** | 100% (shares rounded) |

Other measurements:

- GPU busy time: 17.3 ms per step (94%).
- `cudaStreamSynchronize`: 1 call per step (sampler), 0.1 ms.
- Output throughput: 870 tokens/s (16 tokens per step).
- AllReduce message size: 16 x 8192 x 2 bytes = 256 KiB per call.

Not collected in this run: FLOP counts, achieved TFLOP/s, DRAM bandwidth, SM/tensor-core
utilization, per-kernel SOL%.
