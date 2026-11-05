# Serving regression after upgrading 1.3.0rc14 -> 1.3.0rc15

Workload: 120B-parameter MoE model, `trtllm-serve` PyTorch backend, 4x B200 with one
TP1 server per GPU, 896 concurrent streams per server, 24,000 requests per arm,
steady-window aggregates. No perf CI bar covers this workload.

| Build | QPS | TPOT (mean) | TTFT (p50) |
|---|---|---|---|
| 1.3.0rc14 | 51.2 | 55.6 ms | 1.9 s |
| 1.3.0rc15 | 44.6 (-12.9%) | 64.8 ms | 4.3 s |

## Serving YAML (unchanged between builds; excerpt)

```yaml
torch_compile_config:
  enable_piecewise_cuda_graph: true
  capture_num_tokens: [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 13914]
```

## Startup log lines

1.3.0rc14:
```
[TRT-LLM] [I] Piecewise CUDA graph capture_num_tokens: [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 13914]
```

1.3.0rc15:
```
[TRT-LLM] [W] capture_num_tokens tops out at 13914, below the reachable ceiling max_batch_size*(max_seq_len-1-num_extra_decoding_steps)=65536. Capturing the ceiling itself.
[TRT-LLM] [I] Piecewise CUDA graph capture_num_tokens: [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 13914, 65536]
```

## Per-iteration device time for context-bearing iterations (from the iteration log)

| True num_tokens in the iteration | rc14 device time | rc15 device time |
|---|---|---|
| 9,000-13,914 | 101 ms | 101 ms |
| 16,000-20,000 | 118 ms | 362 ms |
| 24,000-28,000 | 179 ms | 366 ms |
| 32,000-40,000 | 241 ms | 359 ms |
| 48,000-56,000 | 352 ms | 371 ms |

## Notable commit between rc14 and rc15

`[None][fix] Broader capture of piecewise cudagraph` - drops capture candidates above
the reachable num_tokens ceiling and adds coverage for token counts in the gap below it.
