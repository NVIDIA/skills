---
name: perf-moe-comm-overlap
description: MoE expert-parallel communication overlap in Megatron Bridge. Covers dispatch/combine overlap, flex dispatcher backends, and expert wgrad scheduling.
when_to_use: Tuning MoE communication overlap, or tracing a MoE throughput regression to a comm-overlap config change; 'overlap_moe_expert_parallel_comm', 'MoE dispatch overlap', 'flex dispatcher', 'DeepEP overlap', 'expert wgrad scheduling'.
---

# MoE Communication Overlap

For the higher-level overview, see:

- @docs/training/communication-overlap.md
- @skills/perf-moe-comm-overlap/card.yaml

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

## Verification

Look for overlap-related log messages during initialization. The comm overlap
validation in `comm_overlap.py` will raise if prerequisites are not met, so a
clean startup confirms the feature is active.
