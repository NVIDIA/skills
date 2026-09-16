---
name: nemo-mbridge-perf-activation-recompute
description: >-
  Validate and use selective and full activation recompute in Megatron Bridge
  to reduce GPU memory usage at the cost of extra compute. Use for activation
  memory OOMs or regressions involving recompute_granularity,
  recompute_num_layers, recompute_modules, recompute_method, selective
  recompute, full recompute, or activation checkpointing.
license: Apache-2.0
---

# Activation Recompute

Stable docs: @docs/training/activation-recomputation.md
Card: @skills/nemo-mbridge-perf-activation-recompute/card.yaml

<!-- Guidance refreshed: 2026-08-12. -->

Activation recompute (activation checkpointing) trades additional forward work during backward for lower retained-activation memory. The useful checkpoint boundary depends on the model architecture, attention backend, parallelism, and the tensor that actually drives the per-rank peak.

## Quick Decision Guide

1. Confirm the pressure is real allocation, not allocator fragmentation. Compare `max_memory_allocated()` with `max_memory_reserved()` on every rank.
2. Keep an explicit no-recompute control when the workload fits. Under selective granularity, `recompute_modules=[]` is valid and useful for this comparison.
3. Select the first boundary from the architecture and observed peak:
   - **Standard attention:** `core_attn` is the common first candidate. It is strongest when unfused attention materializes score/probability tensors. With Transformer Engine fused or Flash Attention, compare it against `[]` because those backends already rematerialize attention internals.
   - **Multi-Latent Attention (MLA):** start with `mla_up_proj` when expanded Q/K/V projections dominate. Add `core_attn` only when the attention-core state still matters.
   - **Grouped MoE:** start with `moe_act` when the expert intermediate activation dominates; add `layernorm` when norm outputs are material. Use whole `moe` recompute only after accounting for the extra expert compute and communication it replays.
   - **Dense FFN:** `mlp` can save the whole dense-MLP activation region, but it usually costs more compute than a narrow output-discard boundary.
4. Change one label at a time. Record per-rank allocated/reserved peaks plus steady-state step time or throughput; do not infer a global module ranking from one recipe.
5. Use full-layer recompute only when targeted selective boundaries do not make the workload fit. Full recompute has the broadest memory effect and the largest replay cost.
6. Treat CUDA graphs, FP8, context-parallel communication, and overlap features as compatibility constraints, not afterthoughts.

Megatron Core's `cpu_offloading=True` is an alternative when PCIe/NVLink transfer overhead is preferable to replayed compute. It cannot be combined with activation recompute and is not compatible with pipeline parallelism greater than one.

## Enablement

### Selective recompute

```python
cfg.model.recompute_granularity = "selective"
cfg.model.recompute_modules = ["core_attn"]  # Common standard-attention candidate, not a universal default.
```

Use the decision table below to replace or extend that list for MLA, MoE, dense-MLP, or GDN workloads.

### Full-layer recompute

```python
cfg.model.recompute_granularity = "full"
cfg.model.recompute_method = "uniform"
cfg.model.recompute_num_layers = 1
```

- `uniform`: checkpoint fixed groups of `recompute_num_layers` transformer layers.
- `block`: checkpoint the first `recompute_num_layers` layers on each pipeline stage, with virtual-pipeline-aware distribution.

## Selective Module Decision Table

The currently pinned Megatron Core accepts these labels. A development branch can add model-specific labels, so validate against the exact target revision rather than copying a list across branches.

| Module | Checkpoint boundary | When to test it | Main cost or caveat |
|---|---|---|---|
| `core_attn` | Core attention | Standard attention, especially an unfused backend retaining attention intermediates | Replays attention. Incremental savings can be small with TE fused/Flash Attention; context parallelism can replay attention communication. |
| `mla_up_proj` | MLA Q/KV up-projection plus RoPE region | MLA models retaining expanded Q/K/V tensors | Replays the MLA expansion path. It is a distinct, potentially additive boundary from `core_attn`. |
| `layernorm` | Input and pre-MLP normalization outputs | Norm outputs contribute materially to the peak, often alongside MoE or MLA boundaries | Usually narrow, but savings depend on hidden size, sequence length, and which graph paths are active. |
| `moe_act` | Activation output between grouped expert FC1 and FC2 | Grouped MoE expert-intermediate activations dominate | Narrow output-discard checkpoint. It does not replay dispatch, FC1, or FC2, but has FP8 delayed-scaling restrictions. |
| `mlp` | Whole dense MLP | Dense layers dominate after narrower boundaries are exhausted | Replays the complete dense MLP. It has no effect on layers whose MLP is MoE. |
| `moe` | Whole MoE forward | A broad MoE region must be discarded to make the workload fit | Replays routing, dispatch/combine communication, experts, and shared-expert work. It is incompatible with expert-parallel overlap. |
| `shared_experts` | Non-overlapped shared-expert MLP | Shared experts are a distinct material peak | Replays the shared-expert MLP and is invalid with shared-expert overlap. Outer `moe` already removes its original-forward saves, but nesting can still change the transient backward-replay peak. |
| `gdn_norm_out` | GDN gated-normalization output | GDN/hybrid models retain this output | Replays the normalization and its HP-to-CP all-to-all path. |

For example, DeepSeek V4 configurations can use the model-specific `mhc`
label only with their required Megatron Core development branch. It is not a
portable label for the pinned revision and therefore is not included in the
table above.

Common performance configurations consequently fall into several patterns rather than one universal list:

- standard transformer recipes often use `core_attn`;
- MLA recipes often use `mla_up_proj`, sometimes with `mlp`;
- grouped-MoE recipes often use `moe_act` or `layernorm` plus `moe_act`;
- higher-pressure MoE recipes sometimes use broader combinations such as `moe` plus `layernorm`.

These are candidate patterns, not an ordering guarantee. Peak attribution and matched measurements decide the final list.

## Measurement Contract

For every candidate, capture:

- exact Bridge and Megatron Core revisions;
- model, sequence length, micro/global batch sizes, precision, attention backend, and parallelism;
- the exact `recompute_granularity`, module list, method, and layer count;
- per-rank `max_memory_allocated()` and `max_memory_reserved()`;
- steady-state step time or throughput after warmup;
- a short convergence or numerical-sanity check appropriate to the task.

Use a matched no-recompute control and change one recompute choice at a time. Peak memory from different jobs, backends, or parallel layouts is not a module-ranking benchmark.

Do not call a candidate successful merely because it advances farther than the
control. Run through optimizer-state initialization and multiple steady-state
steps: selective recompute can move the memory wall from forward into gradient
synchronization or the optimizer without making the workload viable.

## Matched H100 Evidence: Moonlight 16B

A 2026-08-12 short-run study used the exact Bridge revision
`600d069b824dd5ce50367a311a5a3244478faf22` and Megatron Core revision
`24bad8e677d22625d86ef2a54c9506b6e4992c93`. The Moonlight 16B BF16
pretraining recipe ran on 8 H100 80GB GPUs with sequence length 4096, MBS=1,
GBS=4, TP=2, PP=1, CP=1, EP=8, mock data, and 20 steps. This model mixes one
dense layer with 26 MLA+MoE layers. Each row changed only
`recompute_modules`; all 20 losses were finite with zero skipped or NaN
iterations.

Peak allocated memory is the maximum post-optimizer value reported after
iteration 2. Time and throughput are means over iterations 11--20.

| Selective modules | Peak allocated (GB) | Step time (ms) | TFLOP/s/GPU | Allocated vs `[]` | Time vs `[]` |
|---|---:|---:|---:|---:|---:|
| `[]` | 36.618 | 457.18 | 77.50 | control | control |
| `core_attn` | 36.614 | 474.80 | 74.44 | -0.01% | +3.85% |
| `mla_up_proj` | 35.902 | 480.89 | 73.72 | -1.96% | +5.19% |
| `mla_up_proj`, `mlp` | 35.917 | 496.73 | 71.50 | -1.91% | +8.65% |
| `moe_act` | 35.941 | 466.26 | 75.50 | -1.85% | +1.99% |
| `layernorm`, `moe_act` | 35.949 | 506.53 | 70.27 | -1.83% | +10.79% |

For this exact workload, `moe_act` is the best first boundary: it recovered
nearly as much allocated memory as `mla_up_proj` for less replay cost.
`mla_up_proj` is the next candidate if its roughly 39 MB additional reduction
matters. Adding `mlp` to `mla_up_proj` or `layernorm` to `moe_act` did not
improve the observed peak and made steps slower. Explicit `core_attn` added
cost without material memory benefit under fused attention.

Maximum reserved memory stayed near 40 GB and did not fall monotonically.
That is allocator caching, not contrary evidence: boundary selection in this
study is based on allocated memory and successful end-to-end steps.

## Matched H100 Evidence: Nemotron 3 Nano

The same 2026-08-12 study used the native 16-H100 BF16 performance recipe for
the 52-layer hybrid Mamba/fused-attention MoE model. The matched short-run
configuration used sequence length 8192, MBS=1, GBS=16, TP=1, PP=1, CP=1,
EP=8, DP=16, expert-DP=2, HybridEP, grouped GEMM, TE CUDA graphs for attention and Mamba,
mock data, and 12 steps. Each row changed only `recompute_modules`.

| Selective modules | Outcome | Rank-0 measured peak | Failure or steady-state evidence |
|---|---|---:|---|
| `[]` | OOM after iteration 1 | 66.297 GB after iteration 1 | Iteration-2 MoE router allocation failed; hot ranks had about 72.9 GiB allocated. |
| `core_attn` | OOM in iteration 1 | not comparable | Grouped-expert linear allocation failed; explicit attention recompute did not make the fused-attention workload fit. |
| `moe_act` | OOM after iteration 1 | 62.103 GB after iteration 1 | 4.194 GB (6.33%) below the control at the matched checkpoint, but the iteration-2 output projection still needed 2 GiB. |
| `layernorm`, `moe_act` | OOM in iteration 1 | not comparable | Output projection still needed 2 GiB; CUDA-graph private pools were material. |
| `moe` | completed 12 steps | 64.653 GB after iteration 2 | 657.42 ms and 277.72 TFLOP/s/GPU over iterations 7--12. |
| `moe`, `layernorm` | completed 12 steps | 63.639 GB after iteration 2 | 677.62 ms and 270.62 TFLOP/s/GPU over iterations 7--12. |

Both successful rows had finite losses and zero skipped or NaN iterations.
For this exact capacity-limited recipe, whole-`moe` recompute is the smallest
tested passing boundary. Adding `layernorm` recovered another 1.014 GB (1.57%)
of rank-0 peak at 3.07% higher step time, so the recipe's broader combination
is justified when that headroom is required. Narrow `moe_act` produced real
activation relief but did not make the whole training step viable.

An exploratory native 8-H100 layout failed during FP32 optimizer-state
initialization even at sequence length 4096. That is optimizer capacity, not a
selective-boundary throughput baseline; no timing comparison from those runs
is used here.

### Cross-model conclusion

These measurements do not define one ranking. Moonlight fit with an empty
control and favored narrow `moe_act`; Nemotron required broad whole-`moe`
recompute; historical dense Llama evidence found whole-`mlp` replay costly and
lacked an empty control. The correct first candidate is therefore the narrowest
boundary implicated by the architecture and peak, followed by broader replay
only when the narrow choice does not pass the complete step.

## Compatibility and Validation

### Configuration semantics

- `recompute_granularity="selective"` uses `recompute_modules`; an empty list is accepted as an explicit control.
- `recompute_granularity="full"` uses `recompute_method` and `recompute_num_layers`; selective labels do not apply.
- Full granularity supersedes selective module choices rather than composing with them.
- Unknown labels fail Megatron Core validation. Labels may differ on development branches, so use the exact revision's `TransformerConfig` validator as the source of truth.

### Attention backend and context parallelism

- TE fused and Flash Attention already use internal rematerialization. Explicit `core_attn` may still change retained inputs/outputs, but it must earn its place in a matched `[]` comparison.
- Under context parallelism, an attention checkpoint can replay communication as well as compute. Include CP size and topology in the measurement record.

### MoE restrictions

- Whole-`moe` recompute is incompatible with expert-parallel overlap because backward replay would repeat the overlapped routing/communication region.
- `shared_experts` recompute is incompatible with shared-expert overlap.
- `moe_act` applies to grouped-GEMM experts and is the narrower choice when only the expert activation needs to be discarded.
- `mlp` targets dense MLPs and is a no-op on MoE layers; mixed dense/MoE models can still benefit on their dense layers.

### FP8 restrictions

- `moe_act` and `layernorm` recompute are not supported with FP8 delayed scaling and require a compatible Transformer Engine version.
- Absorbed MLA paths have additional FP8/FP4 restrictions. Validate the exact model/provider path before selecting `mla_up_proj`.

### CUDA graphs

- Selective recompute is valid only when a checkpointed module lies wholly inside or wholly outside the selected graph scope. A checkpoint boundary that straddles a graph boundary is invalid.
- Capture/warmup can bypass checkpoint wrappers, so verify the final graph scope and replay path rather than assuming eager behavior carries over.
- Full recompute with CUDA graphs requires `cuda_graph_impl="full_iteration"` in the pinned Megatron Core. Otherwise disable CUDA graphs; scoped/local graph capture is not a substitute for full-iteration capture here.

## Historical Measurement: Context, Not a Module Ranking

Historical H100 measurements from Bridge PR #3107 used Llama 3 70B SFT on 32 H100 80GB GPUs with FP8 current scaling, sequence length 4096, micro-batch size 1, global batch size 32, TP=4, PP=4, VPP=5, and DP=2:

| Configuration | TFLOP/s/GPU | Peak memory |
|---|---:|---:|
| `core_attn` baseline in that run | ~704 | 58.8 GB (OOM on rank 0) |
| `mlp` | 593.6 | 55.6 GB |
| `mlp` + `core_attn` | 586.8 | 55.6 GB |
| `core_attn` + `layernorm` | ~702 | 59.6 GB (OOM on rank 0) |
| Golden throughput recorded in the PR context | 709.93 | Not a paired memory measurement |

Limitations of this evidence:

- it did not include a matched no-recompute row;
- the golden row was not a paired module-only comparison;
- the measurements cover one dense Llama workload, not MLA or MoE;
- the table supports the local memory/throughput tradeoff only and must not be used to rank all recompute labels.

## Code Anchors

- Selective-label validation and cross-feature checks: `3rdparty/Megatron-LM/megatron/core/transformer/transformer_config.py`
- Checkpoint implementations: `3rdparty/Megatron-LM/megatron/core/tensor_parallel/random.py`
- Standard-attention checkpoint boundary: `3rdparty/Megatron-LM/megatron/core/transformer/attention.py`
- MLA up-projection boundary: `3rdparty/Megatron-LM/megatron/core/transformer/multi_latent_attention.py`
- Layernorm, dense-MLP, and outer-MoE placement: `3rdparty/Megatron-LM/megatron/core/transformer/transformer_layer.py`
- Grouped expert activation boundary: `3rdparty/Megatron-LM/megatron/core/transformer/moe/experts.py`
- Shared-expert and whole-MoE paths: `3rdparty/Megatron-LM/megatron/core/transformer/moe/moe_layer.py`
- GDN normalization boundary: `3rdparty/Megatron-LM/megatron/core/ssm/gated_delta_net/gdn.py`

## Failure Diagnosis

| Symptom | Likely cause | Next action |
|---|---|---|
| `core_attn` gives little or no peak reduction | Fused/Flash attention already rematerializes the expensive internals, or the peak is elsewhere | Compare with `[]`, attribute the peak, then test the architecture-specific boundary such as `mla_up_proj` or `moe_act`. |
| MLA still OOMs after `core_attn` | Expanded Q/K/V projection tensors, not attention-core tensors, dominate | Test `mla_up_proj`; add `core_attn` only if matched evidence supports it. |
| MoE peak remains high | Expert intermediate or norm outputs dominate | Test `moe_act`, then `layernorm`; reserve whole `moe` for broader pressure. |
| Expert-overlap validation fails | Whole-`moe` or `shared_experts` recompute conflicts with overlap | Keep overlap and use a compatible inner boundary, or disable overlap and remeasure the entire configuration. |
| A selected label has no measurable effect | That module is absent or inactive on the measured layers, or graph capture bypassed the wrapper | Inspect the provider/layer mix and final graph scope; for example, `mlp` is ineffective on pure-MoE layers. |
| Full recompute plus CUDA graphs asserts | Graph implementation is not full-iteration | Set `cuda_graph_impl="full_iteration"` or disable CUDA graphs. |
| Reserved memory is high but allocated memory is stable | Allocator fragmentation or caching | Try `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` before adding recompute. |
| OOM moves to a different rank after enabling recompute | Pipeline/virtual-pipeline layer distribution changed the bottleneck | Compare per-rank peaks and tune full block/uniform placement or selective boundaries for the actual hot stage. |
| A candidate gets farther but still OOMs | Recompute moved the peak into gradient synchronization or optimizer-state initialization | Record the changed failure stage as diagnostic evidence, but require optimizer initialization and multiple steady steps before calling it a pass. |

## Known Limitations

- A module list is not portable across model families, attention backends, parallel layouts, or Megatron Core revisions.
- Memory savings are nonlinear when boundaries overlap or nest; additive arithmetic is unreliable.
- Full recompute changes RNG execution paths; dropout workloads need a numerical/convergence check.
- Activation recompute does not address parameter, optimizer-state, or allocator-fragmentation pressure.
- The correct result is the smallest measured replay cost that satisfies the per-rank memory target, not the longest module list.

## Further Reading

- `docs/performance-guide.md`
- `skills/nemo-mbridge-perf-memory-tuning/SKILL.md`
- `skills/nemo-mbridge-perf-cuda-graphs/SKILL.md`
- `skills/nemo-mbridge-perf-cpu-offloading/SKILL.md`
- Megatron Core activation recomputation guide: <https://docs.nvidia.com/megatron-core/developer-guide/latest/api-guide/index.html>
