# "Unstable" P99 TTFT on DeepSeek-V3 with chunked prefill

Setup: DeepSeek-V3 FP8, TensorRT-LLM PyTorch backend, 8x H200, `enable_chunked_prefill: true`,
ISL 32k / OSL 1k, concurrency 16, 400 requests per rep. The benchmark script starts a
fresh `trtllm-serve` process for every rep and stops it afterwards.

| Rep | TTFT p50 | TTFT p99 | Output tok/s |
|---|---|---|---|
| 1 | 0.42 s | 4.21 s | 1,612 |
| 2 | 0.41 s | 4.18 s | 1,618 |
| 3 | 0.43 s | 4.26 s | 1,605 |
| 4 | 0.42 s | 4.19 s | 1,615 |
| 5 | 0.42 s | 4.23 s | 1,609 |

Rep-to-rep spread of TTFT p99 is under 2%; throughput spread is under 1%.

## Server log, every rep

In every rep, exactly once per server process, on the first request that takes the
chunked-prefill path (iteration 3 in each rep), the log shows:

```
[torch.compile] compiling maybe_compiled_cat (MLA chunked prefill) ... done in 3.7 s
```

It never appears again later in the same process. With chunked prefill disabled, TTFT
p99 drops to ~0.9 s (but long prompts then need more memory).

Question from the team: "Our P99 TTFT is spiky/unstable - which instability pattern is
this, and how do we stabilize it?"
