---
name: hybrid-context-parallel
description: Operational guide for enabling hierarchical context parallelism in Megatron-Bridge, including config knobs, code anchors, pitfalls, and verification. Use when the user asks about hierarchical_context_parallel_sizes, a2a+p2p, CP scaling beyond KV heads, or multi-level context parallelism.
---

# Hybrid / Hierarchical Context Parallel Skill

For what HCP is, when to use it, and the decision tree (a2a+p2p vs pure a2a vs p2p), see:

- @docs/training/hybrid-context-parallel.md
- `card.yaml` (co-located)

## Enablement

Minimal Bridge override:

```python
cfg.model.context_parallel_size = 4
cfg.model.cp_comm_type = "a2a+p2p"
cfg.model.hierarchical_context_parallel_sizes = [2, 2]
cfg.dist.use_decentralized_pg = False
```

Required constraints:

- `prod(hierarchical_context_parallel_sizes) == context_parallel_size`
- `seq_length % (2 * context_parallel_size) == 0`
- Transformer Engine `>= 1.12.0`

## Code Anchors

Upstream config and validation:

```45:54:3rdparty/Megatron-LM/megatron/core/model_parallel_config.py
context_parallel_size: int = 1
"""Splits network input along sequence dimension across GPU ranks."""

hierarchical_context_parallel_sizes: Optional[list[int]] = None
"""Degrees of the hierarchical context parallelism. Users should provide a list to specify 
   the sizes for different levels. Taking the a2a+p2p cp comm type as example, it contains
   groups of two levels, so the first value of the list indicates the group size of the a2a
   communication type, and the second value indicates the group size of the p2p communication
   type.
"""
```

```428:433:3rdparty/Megatron-LM/megatron/training/arguments.py
if args.hierarchical_context_parallel_sizes:
    from numpy import prod
    assert args.context_parallel_size == prod(args.hierarchical_context_parallel_sizes)
if "a2a+p2p" in args.cp_comm_type:
    assert args.hierarchical_context_parallel_sizes is not None, \
    "--hierarchical-context-parallel-sizes must be set when a2a+p2p is used in cp comm"
```

Bridge MPU path:

```613:648:src/megatron/bridge/training/initialize.py
parallel_state.initialize_model_parallel(
    ...
    context_parallel_size=model_config.context_parallel_size,
    hierarchical_context_parallel_sizes=model_config.hierarchical_context_parallel_sizes,
    ...
)
...
return ProcessGroupCollection.use_mpu_process_groups()
```

Bridge decentralized-PG path:

```503:524:src/megatron/bridge/training/initialize.py
pg_collection = ProcessGroupCollection(
    ...
    cp=cp_pg,
    tp_cp=tp_cp_pg,
    hcp=None,
    ep=ep_pg,
    ...
)
```

## Implementation Map

### Config definition

`hierarchical_context_parallel_sizes` is declared in `ModelParallelConfig`:

```
# 3rdparty/Megatron-LM/megatron/core/model_parallel_config.py
hierarchical_context_parallel_sizes: Optional[list[int]] = None
# First value = a2a group size, second value = p2p group size.
# Product must equal context_parallel_size.
```

`cp_comm_type` is declared in `TransformerConfig`:

```
# 3rdparty/Megatron-LM/megatron/core/transformer/transformer_config.py
cp_comm_type: Optional[Union[str, List[str]]] = None
# Can be per-layer (List[str]) or uniform (str).
# Values: "p2p", "all_gather", "a2a", "a2a+p2p"
```

### Validation (MCore)

`TransformerConfig.__post_init__` enforces that `a2a+p2p` requires HCP sizes and the product matches CP.

### Process group creation

`parallel_state.initialize_model_parallel` creates hierarchical CP sub-groups when HCP sizes are provided via `create_hierarchical_groups`.

### TE integration

`TEDotProductAttention` passes the hierarchical groups to Transformer Engine when `a2a+p2p` is used. Requires **Transformer Engine >= 1.12.0**.

## Pitfalls

1. **Different features**: `a2a+p2p` and upstream `hybrid_context_parallel=True` are different features. The latter is for balancing packed/variable-length workloads.
2. **Bridge HCP is MPU-only today**: If `use_decentralized_pg=True`, Bridge initializes flat CP groups and leaves HCP unset.
3. **No checked-in Bridge recipe** currently exercises HCP directly.
4. **Single-GPU load helpers** clear `hierarchical_context_parallel_sizes`.
5. **Silent broken training**: If you use `a2a+p2p` without setting `hierarchical_context_parallel_sizes`, MCore now asserts. Older versions would silently disable CP communication — each rank attended only to its local chunk, producing artificially high throughput but completely broken gradients.
6. **Product must match**: `prod(hierarchical_context_parallel_sizes)` must exactly equal `context_parallel_size`. A mismatch triggers an assertion.
7. **Verify in logs**: Look for the process group initialization output. You should see `HIERARCHICAL_CONTEXT_PARALLEL_GROUPS` being created. If you only see `CONTEXT_PARALLEL_GROUP`, HCP is not active.

## Verification

No dedicated Bridge end-to-end test exists yet for HCP (see `card.yaml`
`follow_up_validation`). Use the existing unit tests and log inspection instead.

Run the decentralized-PG unit test to confirm the flat-CP behavior is preserved:

```bash
uv run python -m pytest tests/unit_tests/training/test_decentralized_pg.py -q
```

For a manual smoke check, launch a 4-GPU run with a small recipe and
`cp_comm_type=a2a+p2p` plus `hierarchical_context_parallel_sizes=[2,2]`:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 uv run python -m torch.distributed.run --nproc_per_node=4 \
  scripts/training/run_recipe.py \
  --recipe llama32_1b_pretrain_config \
  model.context_parallel_size=4 \
  model.cp_comm_type=a2a+p2p \
  "model.hierarchical_context_parallel_sizes=[2,2]" \
  train.train_iters=2
```

Success criteria:

- Logs show `HIERARCHICAL_CONTEXT_PARALLEL_GROUPS` being created
- Training completes at least one step without error
- If you only see `CONTEXT_PARALLEL_GROUP`, HCP is not active
