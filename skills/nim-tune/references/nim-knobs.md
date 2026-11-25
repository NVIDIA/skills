# NIM runtime knobs

<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: CC-BY-4.0 AND Apache-2.0
-->

The knob reference for `nim-tune`. It applies to a plain NVIDIA NIM container on the vLLM
backend, deployed with Docker or the NIM Operator; `nim-tune` does not tune SGLang or TensorRT-LLM
NIMs yet (SGLang names below are for reference only). A NIM maps a small set of NIM-level settings
onto its engine and passes anything else through. This is an evidence index, not a default recipe:
bind an observed symptom to one experiment.

Entries marked **[run]** were measured; the rest come from the NIM documentation and are unverified
on a live image. Runs: Docker on one H100 NVL (2026-10-21), and the NIM Operator v3.0.2 on a
Kubernetes node with 8 x H200 NVL linked by PCIe (2026-10-23 and 2026-10-25).

## Contents
- Version and sources
- How configuration resolves
- Prove the effective configuration
- Profiles
- Runtime candidates
- Speculative decoding
- Profile selection (not tuning)
- Check the default first
- Objective and latency target
- Pruning before GPU time
- Knob families
- Measured layout results
- Measured before and after
- Workload shape
- Capturing a real workload
- Settings not to tune
- Not yet verified

## Version and sources

This page targets **NIM for LLM and VLM 2.1.x** (docs release 2.1.4, image
`nvcr.io/nim/nvidia/vllm-model-free-nim:2.1.4`, vLLM 0.30.0). Sources, read on 2026-10-21:

- https://docs.nvidia.com/nim/large-language-models/latest/reference/environment-variables.html
- https://docs.nvidia.com/nim/large-language-models/latest/reference/advanced-configuration.html
- https://docs.nvidia.com/nim/large-language-models/latest/about-nim-llm/release-notes.html

Two generations coexist. Model-specific images such as
`nvcr.io/nim/meta/llama-3.2-1b-instruct:latest` were still NIM **1.12.0** (built Aug 2025) when
checked; they follow the 1.x configuration reference, which has different variables
(`NIM_MAX_BATCH_SIZE`, `NIM_SCHEDULER_POLICY`, `NIM_ENABLE_KV_CACHE_HOST_OFFLOAD`,
`NIM_NUM_KV_CACHE_SEQ_LENS`, ...). Read the reference that matches the image's
`com.nvidia.nim.version` label before proposing a knob, and never reuse a 1.x variable on a 2.x
image without engagement proof.

## How configuration resolves

Highest priority first: CLI arguments after the image, `NIM_PASSTHROUGH_ARGS`, `NIM_*` environment
variables, `runtime_config.json`, profile tags, internal values (memory guardrails), NIM defaults.
JSON-object values compose across sources. Canonical arguments translate to the active backend
(for example `--tensor-parallel-size`, `--enable-prefix-caching`); backend-native flags go through
`NIM_PASSTHROUGH_ARGS` or the CLI and must match the backend version. Some arguments are denied
(`--host`, `--port`, `--ssl-*`, `--api-key`, ...). `NIM_STRICT_ARG_PROCESSING=true` makes conflicts
fatal.

## Prove the effective configuration

Run the image with `--dry-run` and the same arguments. It prints a provenance table (value and
source per setting, for example `tensor_parallel_size = 4 (source: CLI)`), the resolved JSON, the
selected profile and the exact `vllm serve ...` command, then exits without serving **[run]**. The
final "NIM-LLM process died (exit code: 0)" line is the normal end of a dry run. The live server's
startup log line `Launching vLLM: vllm serve ...` shows what was actually launched **[run]**.

## Profiles

The NIM selects a profile for the GPUs it sees and the parallel sizes it is given. On the 2.1.4
model-free image the profile is `vllm-tp<N>-pp1`, chosen by `--tensor-parallel-size` or, with no
argument, by the number of GPUs visible to the container: a NIMService with 4 GPUs and no arguments
selected `vllm-tp4-pp1` and launched `--tensor-parallel-size 4` **[run]**. Pin the profile with
`NIM_MODEL_PROFILE=<id>` when comparing other settings, and change a profile only as its own
candidate.

## Runtime candidates

Tuning keeps the NIM's own profile (TP, PP, replicas) and changes runtime settings. Read each
setting's effective value from the untuned NIM's startup log first. On `vllm-model-free-nim:2.1.4`
with Qwen2.5-7B on 4 x H200 (TP4) the untuned values were **[run]**: `max_num_batched_tokens` 8192
with chunked prefill on, prefix caching on, `kv_cache_dtype` auto (BF16),
`gpu_memory_utilization` 0.92 (122 GiB of KV cache), `max_model_len` 32768.

Phase 1, one setting per candidate, values around the effective default (ranges from vLLM
tuning practice):

| setting (vLLM flag) | what it trades | candidate values |
|---|---|---|
| `--max-num-batched-tokens` | per-step token budget: larger lowers TTFT under load, smaller protects ITL | ISL+OSL about 512: 1024, 2048, 4096; about 2048: 4096, 8192, 16384; long context: 16384 to 65536 |
| `--max-num-seqs` | sequences per step: the main throughput and latency knob | latency (low load): 4 to 32; general: 32 to 512; throughput: 128 to 1024. Start near the target concurrency |
| `--max-cudagraph-capture-size` or the compilation config's capture sizes | CUDA graphs cover decode batch sizes; capturing far above the load wastes memory | min(max_num_seqs, highest concurrency), rounded up to a power of 2 |
| `--gpu-memory-utilization` (`NIM_KVCACHE_PERCENT`) | KV cache size | only when memory is tight: dense GQA 0.80 to 0.90; never above 0.95 |
| `--max-model-len` (`NIM_MAX_MODEL_LEN`) | KV reserved per request | fix to the workload maximum plus template headroom; lower only when memory is tight |
| `--kv-cache-dtype` | KV memory and bandwidth | `fp8` paired with the weight precision; accuracy check required |
| `--async-scheduling`, other scheduler flags, `VLLM_*` environment variables | CPU scheduling overlap and backend choice | only when listed in `engine_flags.txt`; read the engine source at that version first |

Leave chunked prefill and prefix caching on (both default on). Keep `dtype` at `auto`.

## Speculative decoding

Phase 2, only after phase 1, starting from its winner, and only on a real text dataset
(`workload.dataset`); random tokens make draft acceptance meaningless.

- Dataset: `scripts/prep_dataset.py ultrachat --tokenizer HF_ID --out datasets/ultrachat.jsonl`
  writes 2,000 UltraChat 200k conversations (up to 4 user turns each) as AIPerf multi-turn JSONL,
  each turn sized by its real reply in the model's tokens. It reads the `test_sft` split: drafts
  are often trained on `train_sft`, which would flatter their acceptance. Runs are greedy
  (`temperature:0.0`). Re-measure the phase-1 winner on this dataset as the phase-2 parent:
  synthetic and dataset results never compare.
- Method: one the model and image support, in this order:
  - `ngram` (prompt lookup): no draft model; try first. `{"method": "ngram",
    "num_speculative_tokens": N, "prompt_lookup_max": 4}`. Helps when replies repeat the prompt.
  - EAGLE3 or EAGLE: a draft checkpoint trained for that exact model, e.g. from Hugging Face.
    `nim-select` and the NIM Metadata API do not list drafts, so find one by model name, then
    record its repository, revision, license and training data; prefer one not trained on the
    evaluation dataset. The NIM pod must be able to download it (its `HF_TOKEN`, network).
  - MTP: only when the checkpoint has MTP heads.
  Check that the NIM passes `--speculative-config` (or `NIM_SPECDEC_*`) with a dry run.
- Draft length N: start from the method's recommended value, then 1, 3, 5 and 7, one candidate each,
  each against the phase-1 winner re-measured on the dataset (the dataset baseline). Then promote
  the best speculative candidate as the parent and re-tune `--max-num-seqs` and
  `--max-num-batched-tokens` from it, one setting at a time (a draft changes how much each step
  costs); each later winner becomes the next parent.
- Read `engine_fallbacks` for each speculative candidate: a method can switch the engine to another
  model runner or turn off async scheduling, and the result then measures both.
- Measure coarse at the winner's best load first, then the full campaign for the best N.
- Record `vllm:spec_decode_num_draft_tokens_total` and `vllm:spec_decode_num_accepted_tokens_total`
  (the campaign does). Acceptance explains a result; it is not the objective. Missing counters on a
  speculative candidate mean the result cannot be explained: report it.
- Keep speculative decoding off unless it wins on the target metric, and run an accuracy check.

## Profile selection (not tuning)

Changing TP, PP or the replica count changes the profile, not the runtime. Do it only when the user
asks, as a separate study, and compare at the same total GPUs. Results are under "Measured layout
results".

## Check the default first

A feature may already be on. On 2.1.4 with vLLM 0.30.0, **prefix caching is on by default [run]**:
with no flag, vLLM reported 455,360 prefix-cache hits on 513,900 queried tokens (88.6%) on a
shared-prefix workload, `--enable-prefix-caching` matched the default within noise (TTFT 34.5 ms vs
34.6 ms), and `--no-enable-prefix-caching` was 17% worse on TTFT. So "enable prefix caching" is not
a tuning candidate here. Before proposing an "enable X" knob, read the engine's counters or log for
X under the default configuration (AIPerf's `server_metrics_export.csv` carries vLLM counters such
as `vllm:prefix_cache_hits` and `vllm:prefix_cache_queries`).

## Objective and latency target

Tune for the most output tokens/s per GPU at loads where the user's latency target holds, for
example TTFT p99 at most 1000 ms. A load passes only when the worst of its repeats meets the target
and no request failed. A case that misses the target is reported separately and never recommended,
however fast it is on another metric. Per-GPU throughput keeps layouts with different GPU counts
comparable. With `sla.search`, `run_campaign.py` doubles the load from a start value while the
target holds, then bisects to within 10% of the highest passing load.

## Pruning before GPU time

- Probe the engine in the image: `vllm serve --help=all` from a running baseline pod or container.
  Plan only flags it lists, at that vLLM version; cite the engine source at that version when a
  flag's effect matters. Report a dropped setting to the user; never drop it silently.
- The campaign prunes before deploying (`run_campaign.py`): a `true`/`false` boolean value, TP x PP
  not equal to the GPUs per replica, a memory fraction above 0.95, 2048 or fewer batched tokens with
  a memory fraction of 0.90 or more, a max model length below ISL + OSL + 64, and speculative
  decoding without a real dataset. It rejects any flag missing from `engine_flags.txt`.
- Do not pair a small batched-token budget (2048 or less) with a memory fraction of 0.90 or more,
  and do not plan a memory fraction above 0.95.
- Tensor-parallel size times pipeline-parallel size (and expert or data-parallel sizes where used)
  must equal the GPUs given to one container, and the tensor-parallel size must divide the model's
  attention heads.
- Leave headroom above `ISL + OSL` for chat-template tokens in `NIM_MAX_MODEL_LEN`.
- `NIM_KVCACHE_PERCENT` maps to `--gpu-memory-utilization` on vLLM (default 0.92 observed **[run]**)
  and `--mem-fraction-static` on SGLang. NIM clamps it on shared or small GPUs (vGPU 0.75, UMA 0.50,
  busy GPU free ratio minus 0.05, floor 0.10), so confirm the effective value in the dry run.
- `NIM_DISABLE_CUDA_GRAPH=1` is a diagnostic control, not a throughput candidate.
- `NIM_PASSTHROUGH_ARGS` flags must exist in the backend version in the image (vLLM 0.30.0); test
  them with `--dry-run`. Write a disabled boolean as its `--no-` flag, not as `false`.
- Precision: use the checkpoint's own quantization (FP8 on FP8-capable GPUs; NVFP4 only with a
  pre-quantized checkpoint), pair the KV-cache data type with it, and gate promotion on an accuracy
  check against the baseline.
- Speculative decoding comes after the other levels, on real text (not random tokens), with an
  accuracy check. The draft acceptance rate explains a result; it is not the objective.
- Keep logging, TLS, CORS, capture and LoRA variables constant across candidates.

## Knob families

| Symptom | Candidate knobs | Expected effect | Validation | Caveat |
|---|---|---|---|---|
| KV cache exhausted before compute saturates | `NIM_KVCACHE_PERCENT`, `NIM_MAX_MODEL_LEN` | more KV blocks, higher sustainable concurrency | dry-run value, startup KV size log, concurrency sweep | NIM may clamp the value |
| Long shared prefixes but low hit rate | engine prefix-cache counters first; then capacity (`NIM_KVCACHE_PERCENT`) | higher hit rate, lower TTFT | `vllm:prefix_cache_hits` / `_queries` | on by default; do not "enable" it **[run]** |
| Decode batch limited by scheduler | `NIM_PASSTHROUGH_ARGS=--max-num-seqs N` (docs example), other native vLLM scheduler flags | larger decode batches | iteration logs, throughput | flag names are backend-version specific; unverified |
| Prefill dominates TTFT | native vLLM flags via passthrough (for example the batched-token budget) | lower TTFT | prefill token counts, TTFT | unverified on this image |
| Decode latency from launch overhead | `NIM_DISABLE_CUDA_GRAPH` (control only) | confirms whether graphs help | A/B on identical workload | diagnostic |
| Latency-bound decode | `NIM_SPECDEC_ENABLE`, `NIM_SPECDEC_ARGS`, `NIM_DRAFT_MODEL_PATH` | lower inter-token latency | accepted-token metrics | profile decides default; model-specific |
| Replicas receive uneven or cache-cold traffic | Service or ingress affinity (session or prefix hash) | higher cache reuse, lower TTFT | per-replica cache counters, TTFT | unverified for NIM; keep the replica count fixed |

## Measured layout results

Qwen2.5-7B-Instruct on `vllm-model-free-nim:2.1.4`, 4 x H200 NVL linked by PCIe, NIM Operator
v3.0.2, AIPerf 0.13.0, random prompts of 1024 input and 256 output tokens, 2 repeats **[run]**:

| Concurrent | 1 x TP4 | 2 x TP2 | 4 x TP1 |
|---|---|---|---|
| 8: output tokens/s per GPU | 227 | 469 | 355 |
| 32: output tokens/s per GPU | 384 | 1,429 | 1,206 |
| 128: output tokens/s per GPU | 357 | 2,837 | 3,330 |

- One instance across all 4 GPUs, which is what an untuned NIM picks, was the worst layout at every
  load: 8 to 9 times less throughput per GPU than replicas at 128 concurrent requests.
- 4 x TP1 always gave the fastest first token; 2 x TP2 gave the fastest time between tokens. The
  right layout depends on the load level and the binding latency target.
- The Kubernetes Service spread requests evenly enough (1,940 / 1,997 for 2 replicas; 828 to 1,072
  for 4 replicas).
- PCIe-only GPUs penalise tensor parallelism; expect a smaller gap on NVLink.

## Measured before and after

Same node, image, model and workload, run back to back with 3 repeats (2026-10-25) **[run]**. The
untuned NIM got 4 GPUs and no arguments, so it launched one TP4 instance. Every change below is
outside the min-max range of the repeats.

| Concurrent | Case | TTFT avg ms | TTFT p99 ms | ITL avg ms | ITL p99 ms | Tokens/s per GPU |
|---|---|---|---|---|---|---|
| 8 | untuned 1 x TP4 | 120 | 217 | 8.32 | 8.70 | 227 |
| 8 | 2 x TP2 | 67 | 151 | 4.01 | 4.29 | 460 |
| 8 | 4 x TP1 | 46 | 89 | 5.40 | 5.58 | 354 |
| 32 | untuned 1 x TP4 | 332 | 819 | 24.5 | 25.7 | 310 |
| 32 | 2 x TP2 | 222 | 455 | 4.72 | 5.93 | 1,370 |
| 32 | 4 x TP1 | 139 | 278 | 5.78 | 6.47 | 1,221 |
| 128 | untuned 1 x TP4 | 563 | 2,958 | 87.0 | 91.3 | 358 |
| 128 | 2 x TP2 | 526 | 1,492 | 8.85 | 11.8 | 2,880 |
| 128 | 4 x TP1 | 418 | 1,093 | 7.87 | 11.3 | 3,247 |

Projecting replicas from one instance overstates them. Four 1-GPU replicas delivered 0.95 of 4 x
one 1-GPU instance at 32 concurrent (8 per replica) and 0.88 at 128, with higher TTFT (418 ms
against 336 ms at 32 per replica). The Service gave the replicas 1,339 to 1,725 requests over the
series. Measure the replicas; never promote a projection.

## Workload shape

- Results hold only for the shape measured. Prefer a captured trace; replay it at its own schedule.
- Synthetic: pin output length (`--osl-stddev 0` and `--extra-inputs ignore_eos:true`), use a new
  seed per repeat, and label the result a proxy.
- Shared-prefix: rotate several prefixes rather than one, or the cache hit rate is unrealistically
  high. Read the hit rate from the engine counters over the measured window only, and record it as
  unavailable, not zero, when the counters are missing.
- Do not disable prefix caching for benchmarks; match how production runs.

## Capturing a real workload

`NIM_CAPTURE_ENABLE=1` with `NIM_CAPTURE_ARGS` captures request payloads for AIPerf replay (docs).
Use it to derive a workload trace from real traffic instead of a synthetic shape. Captured payloads
contain prompts: keep them out of the repository. Unverified on a live image.

## Settings not to tune

`NGC_API_KEY`, `HF_TOKEN`, `NIM_CACHE_PATH`, `NIM_SERVER_PORT`, `NIM_MODEL_PATH`,
`NIM_SERVED_MODEL_NAME`, `NIM_SSL_*`, `NIM_CORS_*`, logging and OpenTelemetry variables, and LoRA
variables are identity or feature switches. Record them and hold them constant.

## Not yet verified

- Whether `NIM_ENABLE_KV_CACHE_REUSE` is honored on the vLLM backend of 2.1.x (the release notes
  mention it for SGLang).
- Which native vLLM flags beyond the dry-run-tested ones pass through unchanged.
- Behavior on models with a tuned recipe under `/opt/nim/etc/tuned/`; the tested models had none.
- The Helm path.
