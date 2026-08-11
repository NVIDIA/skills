---
name: nemo-mbridge-perf-moe-comm-overlap
description: MoE expert-parallel communication overlap in Megatron Bridge. Covers dispatch/combine overlap, flex dispatcher backends, and expert wgrad scheduling.
license: Apache-2.0
when_to_use: Tuning MoE communication overlap, or tracing a MoE throughput regression to a comm-overlap config change; 'overlap_moe_expert_parallel_comm', 'MoE dispatch overlap', 'flex dispatcher', 'DeepEP overlap', 'expert wgrad scheduling'.
---

# MoE Communication Overlap

For the higher-level overview, see:

- @docs/training/communication-overlap.md
- @skills/nemo-mbridge-perf-moe-comm-overlap/card.yaml

## Quick Decision

Use MoE communication overlap when:

- `EP > 1`
- token dispatch or combine time is visible in the profile
- the run is already correct and you are now tuning throughput

Avoid turning it on as an early bring-up step. It is easier to validate after
the dispatcher, routing mode, and recompute plan are already stable.

## Enablement

```python
cfg.comm_overlap.overlap_moe_expert_parallel_comm = True

# Optional: delayed wgrad for additional overlap
cfg.comm_overlap.delay_wgrad_compute = True

# IMPORTANT: disable shared expert overlap when using dispatch overlap
cfg.model.moe_shared_expert_overlap = False
```

### Prerequisites

- `expert_model_parallel_size > 1`
- `num_moe_experts > 1`
- `moe_token_dispatcher_type` must be `"alltoall"` or `"flex"`
- Precision: BF16 or FP16
- If PP is used, VPP (`virtual_pipeline_model_parallel_size`) must be set (non-`None`)

### Flex dispatcher activation

Setting `moe_flex_dispatcher_backend` alone does **not** activate flex dispatch.
You must also set `moe_token_dispatcher_type = "flex"`.

## Recompute And CUDA Graph Interaction

- Full recompute is not a good companion for the overlap path.
- `delay_wgrad_compute` adds further constraints if CUDA-graph scopes include
  attention or MoE-router work.
- In practice, selective recompute is the safer pairing when overlap is enabled.

## Measured Evidence

### HybridEP production-shape validation

A 2026-07-25 controlled Qwen3 30B-A3B pretraining comparison used 16 H100
GPUs, BF16, sequence length 4096, `TP=1`, `PP=1`, `CP=1`, `EP=16`,
`MBS=1`, `GBS=1024`, forced-balanced routing, HybridEP, and Transformer
Engine CUDA-graph scopes `moe_router` and `moe_preprocess`. The only
performance change was plain EP overlap; delayed wgrad stayed disabled.

| Case | Steady window | Step time | Model TFLOPS/GPU |
|---|---:|---:|---:|
| EP overlap off | iterations 5-20 | 24.7138s | 244.039 |
| EP overlap on, search run | iterations 5-20 | 21.0725s | 286.208 |
| EP overlap on, independent validation | iterations 41-50 | 20.9920s | 287.305 |

The independent result reduced step time by 15.059% and increased throughput
by 17.729% over the reproduced baseline. Loss remained finite, no iterations
were skipped or NaN, and rank-0 peak allocated memory was 62.166 GiB.

A same-method rank-0 Nsight Systems comparison captured 463,348 kernels in
each case:

| Profile metric | Overlap off | Overlap on |
|---|---:|---:|
| Communication concurrent with GEMM/attention | 9.079ms | 3,958.997ms |
| Communication time hidden by compute | 0.11% | 36.55% |
| GPU-active interval union | 22.821s | 21.221s |
| HybridEP dispatch-with-permute NVTX | 4.253s | 1.767s |
| HybridEP metadata-preprocess NVTX | 3.109s | 0.670s |

This is direct evidence that the gain came from hiding exposed HybridEP
dispatch/combine work, not from changing the dispatcher, routing, graph
scopes, batch shape, or parallel layout.

### Correctness-first alltoall smoke

A 2026-05-18 current-main H100 x16 smoke on Qwen3 30B-A3B mock pretraining
used `EP=16`, `alltoall`, global batch size 1024, CUDA graphs disabled, and
`moe_permute_fusion=false` because the PyTorch 25.11 / TE / Triton stack failed
in Transformer Engine fused permutation in prior bring-up.

Results were directional rather than release-grade:

- no EP overlap: 41.25s steady-state mean over iterations 3-8
- EP overlap: 31.31s steady-state mean over iterations 3-8
- EP overlap plus `delay_wgrad_compute`: 31.20s steady-state mean over
  iterations 3-8

Treat this as evidence that EP overlap can help an inter-node `alltoall` MoE
shape when communication is exposed. It is not proof that delayed wgrad is a
separate win, and it does not validate the fused permutation path. An earlier
2026-05-16 short smoke on the same shape showed the same pattern.

## Code Anchors

- Overlap validation: `src/megatron/bridge/training/comm_overlap.py`
- Flex dispatcher backend: `src/megatron/bridge/training/flex_dispatcher_backend.py`
- Config: `src/megatron/bridge/training/config.py`
- Unit tests: `tests/unit_tests/training/test_comm_overlap.py`
- DeepEP tests: `tests/unit_tests/training/test_deepep.py`

## Pitfalls

1. **Shared expert overlap conflict**: `moe_shared_expert_overlap` and
   `overlap_moe_expert_parallel_comm` can conflict. Disable shared expert
   overlap when using the dispatch overlap path.

2. **PP without VPP**: MoE overlap requires VPP when pipeline parallelism is
   active. Without it, the overlap scheduling cannot interleave correctly.

3. **Flex != backend flag**: `moe_flex_dispatcher_backend="deepep"` alone
   does nothing if `moe_token_dispatcher_type` is still `"alltoall"`.

4. **Conservative recipe defaults**: Most public recipes leave MoE overlap
   disabled. You need to explicitly enable it via overrides.

5. **Performance gains are workload-dependent**: overlap helps most when dispatch
   communication is already a visible slice of step time. It is not guaranteed
   to help every small or lightly loaded EP run.

6. **Summed kernel time is not wall time**: concurrent kernels can run longer
   because they contend for SMs or bandwidth, so overlap may increase summed
   per-stream kernel duration while reducing the exposed interval union and
   end-to-end step time.

## Verification

Look for overlap-related log messages during initialization. The comm overlap
validation in `comm_overlap.py` will raise if prerequisites are not met, so a
clean startup confirms the feature is active.

For a short performance-harness smoke, keep the command shape explicit and vary
only one overlap knob at a time:

```bash
uv run python scripts/performance/run_script.py \
  -m qwen \
  -mr qwen3_30b_a3b \
  --task pretrain \
  -g h100 \
  -c bf16 \
  -ng 16 \
  -gn 8 \
  --max_steps 8 \
  --cuda_graph_impl none \
  --moe_flex_dispatcher_backend None \
  --moe_a2a_overlap false \
  --tokenizer_type NullTokenizer \
  comm_overlap.overlap_moe_expert_parallel_comm=true \
  comm_overlap.delay_wgrad_compute=false \
  model.moe_shared_expert_overlap=false
```

If fused MoE permutation fails during bring-up, add
`model.moe_permute_fusion=false` to separate overlap timing from runtime-stack
validation, then retest with the matched production container.

For performance validation, use an unprofiled steady window as the acceptance
metric. Use a matched Nsight A/B to establish causality:

1. Keep dispatcher, routing, CUDA graphs, batch shape, parallelism, and runtime
   fixed.
2. Toggle only `overlap_moe_expert_parallel_comm`; keep
   `delay_wgrad_compute=false` for the first isolation.
3. Compare communication and compute interval unions and their intersection,
   not only summed kernel durations.
4. Report steady step time, model TFLOPS/GPU, loss finiteness, skipped/NaN
   iterations, and peak allocated memory.

_Last signature refresh: 2026-08-03._
