# trtllm-bench throughput regression: Qwen3.5-9B, 1.3.0rc19 -> 1.3.0rc21

Command (identical for both builds, text-only synthetic dataset, no `--modality`):

```
trtllm-bench --model Qwen/Qwen3.5-9B throughput --backend pytorch \
  --dataset text_500_2000.jsonl --concurrency 544 \
  --max_batch_size 512 --max_num_tokens 2048 --max_seq_len 2500
```

Hardware: 1x H100 SXM 80GB, BF16. The model's `config.json` contains
`"architectures": ["Qwen3_5ForConditionalGeneration"]`.

| Build | Output throughput | Mean per-iteration device step time (gen) |
|---|---|---|
| 1.3.0rc19 | 9,620 tok/s | 41.2 ms |
| 1.3.0rc21 | 8,870 tok/s (-7.8%) | 41.3 ms |

Kernel names and per-kernel durations match between the builds in nsys.

## Startup log excerpts

1.3.0rc19:
```
[TRT-LLM] [I] Memory used after loading model weights (inside torch) in memory usage profiling: 16.94 GiB
[TRT-LLM] [I] Memory used after loading model weights (outside torch) in memory usage profiling: 0.71 GiB
[TRT-LLM] [I] Estimated max memory in KV cache : 54.04 GiB
[TRT-LLM] [I] [MemUsageChange] Allocated 54.04 GiB for max tokens in paged KV cache (1374848).
```

1.3.0rc21:
```
[TRT-LLM] [I] Memory used after loading model weights (inside torch) in memory usage profiling: 17.87 GiB
[TRT-LLM] [I] Memory used after loading model weights (outside torch) in memory usage profiling: 1.26 GiB
[TRT-LLM] [I] Estimated max memory in KV cache : 52.56 GiB
[TRT-LLM] [I] [MemUsageChange] Allocated 52.56 GiB for max tokens in paged KV cache (1337184).
```

With max_seq_len 2500, the KV pool holds 549 full-length requests on rc19 and 534 on
rc21; the benchmark keeps 544 requests in flight. The rc21 run ends with a ~700-iteration
tail at low batch size that rc19 does not have.

## Extra probe

`python -c "import tensorrt_llm"` in an otherwise idle container, then `nvidia-smi`
while the interpreter is still alive:
- rc19: the process does not appear on any GPU.
- rc21: the process holds 548 MiB on GPU 0.

## Commit subjects in rc19..rc21 (selected)

- `[None][feat] Add support for Qwen3.5 VL Dense`
- `[None][feat] DSv4 prep: MoE routing and backend support`
- `[None][perf] Tune FMHA tile selection for head_dim 256`
- `[None][fix] Fix stop-word handling for empty sequences`
- `[None][chore] Bump flashinfer to the next minor release`
- `[None][test] Add Qwen3.5 accuracy tests`
