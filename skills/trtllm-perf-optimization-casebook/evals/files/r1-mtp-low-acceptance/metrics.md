# DeepSeek-R1-0528 serving metrics (config.yaml in this directory)

Workload: reasoning-heavy prompts (math and code), ISL ~1k, OSL up to 8k, concurrency 128.
Responses spend ~85% of their output tokens inside `<think> ... </think>`.

| Metric | Value |
|---|---|
| Output throughput | 11,300 tok/s |
| Mean acceptance length (tokens per step incl. the target token, max 4) | 1.71 |
| Acceptance length inside `<think>` | 1.64 |
| Acceptance length in the final-answer segment | 2.42 |
| GPU idle ratio (nsys, decode steady state) | 5% |
| GSM8K / GPQA accuracy of this config (strict MTP acceptance) | recorded in `accuracy_baseline.json` (not attached) |

Bottleneck classification from the analysis step: compute-bound decode; MTP
acceptance is low and caps the speculative-decoding speedup.

Question from the team: "How do we raise acceptance? Someone suggested turning on
relaxed acceptance for the thinking phase - can we just flip it on in this config?"
