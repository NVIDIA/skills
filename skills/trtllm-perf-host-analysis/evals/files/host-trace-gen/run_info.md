# Run info for trace.sql

- Model: Llama-3.1-8B-Instruct (BF16), TensorRT-LLM PyTorch backend, `trtllm-serve`
- Hardware: 1x H100 SXM 80GB, TP=1 (no NCCL traffic)
- Load: ISL 512 / OSL 256, concurrency 16, 160 requests
- Capture: `nsys profile -t cuda,nvtx,osrt` over the serving window, then
  `nsys export --type=sqlite`; the SQLite file was dumped to SQL text so it
  could be attached (restore command is in the first lines of `trace.sql`).
- The trace was down-sampled before export: each kernel row stands for a
  coalesced group of per-layer kernels, so kernel *counts* are not meaningful,
  but start/end timestamps, gaps and durations are preserved.
- Serving YAML excerpt:

```yaml
max_batch_size: 16
max_num_tokens: 8192
cuda_graph_config:
  enable_padding: false
  batch_sizes: [1, 2, 4, 8]
kv_cache_config:
  free_gpu_memory_fraction: 0.85
```

- Observation from the benchmark client: TPOT is ~3 ms and the team believed
  the run was "GPU-bound because overall GPU idle is only ~20%".
