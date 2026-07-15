---
name: jetson-speculative-decoding
description: Add EAGLE-3 or draft-model speculative decoding to a Jetson vLLM server when TPOT is the bottleneck.
version: 0.0.1
license: "Apache-2.0"
metadata:
  author: "Jetson Team"
  tags: [jetson, llm, speculative-decoding]
  languages: [markdown]
  data-classification: public
---

# Jetson Speculative Decoding (vLLM)

Speculative decoding lets a small "draft" model propose tokens that the target model verifies in a single forward pass, reducing per-token latency. On Jetson, the win/loss is **dominated by VRAM headroom**, not by the draft quality. This skill encodes the parts an LLM won't already know.

## Purpose

Tune an existing Jetson vLLM deployment for faster token generation by appending the right `--speculative-config` and validating whether it improves single-stream decode speed.

## When to use

- TPOT/ITL is the bottleneck (TTFT is fine, output is just slow).
- Workload is single-stream or low-concurrency (≤2). Speculation usually loses at high concurrency.
- Jetson family is **Thor or AGX Orin**. Do **not** suggest EAGLE-3 on Orin Nano/NX — there is rarely enough VRAM headroom to host both target and draft, and you'll OOM at startup.

## When NOT to use

- High-concurrency serving (≥8): batched decode usually beats speculation; the draft model just steals VRAM.
- Models without a published EAGLE-3 head — do not train one ad-hoc as a "fix".
- After applying `jetson-inference-mem-tune` flags that already pushed `--gpu-memory-utilization` near the ceiling. Free at least ~2 GB first.

## Prerequisites

- A working vLLM server recipe from `jetson-llm-serve`.
- Enough memory headroom for the draft model or EAGLE-3 head in addition to the target model.
- A benchmark baseline from `jetson-llm-benchmark` before enabling speculation.
- A target model with a compatible EAGLE-3 head, or a small same-family draft model for the fallback path.

## Instructions

Append `--speculative-config` to the `vllm serve` command shown in `jetson-llm-serve`.

EAGLE-3 (preferred when a head is published for the target model):

```bash
--speculative-config '{
  "method": "eagle3",
  "model": "<eagle3-head-repo-id>",
  "num_speculative_tokens": 5,
  "draft_tensor_parallel_size": 1
}'
```

Draft-model (fallback — pair a small same-family model):

```bash
--speculative-config '{
  "method": "draft_model",
  "model": "<small-draft-model-repo-id>",
  "num_speculative_tokens": 4,
  "draft_tensor_parallel_size": 1
}'
```

### Jetson-specific tuning rules

- `num_speculative_tokens`: start at **5** on Thor, **3** on AGX Orin. Higher values pay off only if the draft acceptance rate is >0.6.
- Always pair with the same vLLM runtime path used by `jetson-llm-serve`: upstream vLLM 0.20+ (`vllm/vllm-openai:latest`) or validated native vLLM 0.20+ on Thor, upstream vLLM 0.20+ on Orin JetPack 7.2 / L4T r39+, or the NVIDIA-AI-IOT vLLM image on older Orin. Do not use an Orin NVIDIA-AI-IOT vLLM image on Thor. Older runtimes may lack EAGLE-3 or the current `--speculative-config` shape.
- Drop `--gpu-memory-utilization` by ~0.05 vs the non-speculative baseline to give the draft model headroom.

## How to verify it actually helped

1. Run `jetson-llm-benchmark` (vLLM path) at `--concurrency 1` **before and after** enabling speculation.
2. Acceptance: target ≥30% improvement in `throughput_tok_s` and ≥20% drop in `tpot_ms_p50` at concurrency 1.
3. If improvement is <10%, or `throughput_tok_s` regresses at concurrency 8, **disable** speculation. The draft model is costing more than it returns.

## Limitations

- Speculative decoding improves decode-heavy workloads; it does not reduce TTFT-dominated latency.
- High concurrency can erase the benefit because continuous batching already keeps the GPU busy.
- Orin Nano/NX usually lack enough memory headroom for both target and draft models.
- Acceptance rate and draft overhead are model-specific, so benchmark before and after instead of assuming a speedup.

## Error handling

- If vLLM rejects `--speculative-config`, verify that Thor and Orin JetPack 7.2 / L4T r39+ are using vLLM 0.20+ and that older Orin is using a JetPack-matched NVIDIA-AI-IOT vLLM image; then switch back to the non-speculative serving command if the runtime still rejects it.
- If startup OOMs, lower `--gpu-memory-utilization`, use a smaller draft, or disable speculation and hand off to `jetson-inference-mem-tune`.
- If benchmark throughput regresses, remove `--speculative-config`; a bad draft path is worse than no speculation.

## Hand off to

- `jetson-llm-benchmark` to quantify the change.
- `jetson-inference-mem-tune` if startup OOMs after enabling speculation.

## Source

vLLM speculative decoding docs and the [Jetson AI Lab GenAI tutorial](https://www.jetson-ai-lab.com/tutorials/genai-on-jetson-llms-vlms/).
