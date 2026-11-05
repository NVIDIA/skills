# Throughput regression: Qwen3-8B serving, nightly build N vs N+1

Setup (identical for both builds): Qwen3-8B BF16, `trtllm-serve` PyTorch backend,
1x H100 SXM, ISL/OSL 1000/1000, concurrency 256, default config (overlap scheduler
enabled), greedy sampling, the model's `generation_config.json` lists 2 EOS ids.

| Build | Output throughput | Mean TPOT |
|---|---|---|
| N (good) | 9,912 tok/s | 25.4 ms |
| N+1 (bad) | 3,987 tok/s | 63.1 ms |

Same comparison with `disable_overlap_scheduler: true` on both builds:

| Build | Output throughput |
|---|---|
| N | 6,410 tok/s |
| N+1 | 5,980 tok/s (-6.7%) |

## Bisect

Bisected (pure Python changes, same wheel binaries) to a single commit:
`[None][feat] add eos_token_id in generation_config to sampling params` - it folds
the generation-config EOS ids into each request's stop-word ids, so the stop-word
check in the sampler now runs for every ordinary request.

## nsys on build N+1 (decode steady state)

- Inside the NVTX range `sample_async` there are now ~512 `cudaStreamSynchronize`
  calls per step (256 requests x 2 stop words); build N shows 0 there.
- GPU idle between decode steps: 61% (build N: 7%).
- Kernel names and per-kernel durations are unchanged between builds.
- The GPU idle per step grows linearly with batch size x number of stop words.
