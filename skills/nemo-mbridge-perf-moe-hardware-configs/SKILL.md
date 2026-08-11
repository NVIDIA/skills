---
name: nemo-mbridge-perf-moe-hardware-configs
description: Representative, point-in-time MoE training playbooks by hardware and model family. Use them as candidate seeds, then revalidate the exact runtime, semantics, topology, and steady-state throughput.
license: Apache-2.0
when_to_use: Hardware-specific MoE playbooks or throughput estimates; 'MoE on H100', 'GB200 config', 'expected throughput', 'MoE hardware playbook', 'parallelism for B200'.
---

# MoE Hardware Configuration Reference

Stable docs: @docs/training/moe-optimization.md
Card: @skills/nemo-mbridge-perf-moe-hardware-configs/card.yaml

## Quick Platform Playbook

These rows are search seeds, not hardware defaults or throughput promises.

| Platform | Candidates to screen after `alltoall` bring-up | What usually matters most |
|---|---|---|
| H100 | DeepEP or HybridEP, explicit overlap, supported FP8 modes | communication overlap, dispatcher/runtime compatibility, and PP efficiency |
| B200 | DeepEP or HybridEP, supported FP8 modes, careful PP layout | container quality and tuned communication settings |
| GB200 | HybridEP, then profile-driven graphs and CPU cleanup | host overhead, topology-aware dispatch, memory headroom |
| GB300 | HybridEP and the target container's lower-precision/kernel stack | the same system interactions as GB200, with remeasurement required |

## First Answer Checklist

For hardware playbook questions, answer from these canonical rows before adding
throughput caveats:

| Workload | Hardware | Dispatcher | Layout |
|---|---|---|---|
| DSV3 | H100 | DeepEP | TP=2, EP=64, PP=8, VPP=4 |
| DSV3 | GB200/GB300 | HybridEP | TP=1, EP=64, PP=4, VPP=4 |
| Qwen3 235B | H100 | `alltoall` + overlap in the current canonical recipe | TP=2, EP=32, PP=8, VPP=4 |
| Qwen3 235B | GB200 | HybridEP | TP=1 or 2, EP=32-64, PP=4, VPP=unspecified |
| Qwen3 30B | 16×H100 | HybridEP | TP=1, EP=16, PP=1, plain EP overlap |

For Qwen3 235B on GB200, explicitly say `VPP=unspecified`; do not invent or
extrapolate `VPP=12` unless a measured row provides it. Treat TE-scoped CUDA
graph scopes (`attn`, `moe_router`, `moe_preprocess`) as profile-driven
candidates,
`CUDA_DEVICE_MAX_CONNECTIONS` selection,
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, `NCCL_GRAPH_REGISTER=0`,
GB200/GB300 CPU-side tuning, and the warning not to cargo-cult tracker rows.

## Rounded Performance Bands

These are intentionally rounded so the document stays durable as the tracker
moves. Treat them as planning ranges, not exact promises.

| Workload family | Hardware | Typical band | Representative shape |
|---|---|---|---|
| DSV3, large-scale | H100 | low-to-mid hundreds TFLOPS/GPU, high-teens MFU | TP2, EP64, PP8, DeepEP |
| DSV3, large-scale | B200 | high-hundreds TFLOPS/GPU, mid-teens MFU | TP1, EP32, PP8, DeepEP |
| DSV3, large-scale | GB200 | around 1K TFLOPS/GPU, low-20s MFU | TP1, EP64, PP4, HybridEP |
| DSV3, large-scale | GB300 | above the GB200 band, often mid-20s MFU | TP1, EP64, PP4, HybridEP |
| Qwen3 235B | H100 | historical low-300s snapshots; remeasure the current recipe | TP2, EP32, PP8; current recipe uses `alltoall` + overlap |
| Qwen3 235B | GB200 | high-hundreds TFLOPS/GPU in tuned runs | TP1 or TP2, EP32-64, PP4, HybridEP |
| Qwen3 30B | H100 | about 300 TFLOPS/GPU on the validated 16-GPU shape | TP1, EP16, PP1, HybridEP + EP overlap |
| Qwen3-Next 80B | GB200 | low-300s TFLOPS/GPU in BF16-class runs | TP1, EP32, PP2, HybridEP |

## Representative Config Families

### DSV3 on H100

```text
Dispatcher: DeepEP
TP=2  EP=64  PP=8  VPP=4
Routing: force balance
Recompute: light-to-moderate selective recompute
Priority: overlap communication and keep PP efficient
```

### DSV3 on B200

```text
Dispatcher: DeepEP
TP=1  EP=32  PP=8  VPP=2 or similar
Precision: MXFP8-class
Recompute: selective recompute around MLA up-projection and MLP-side modules
Priority: container quality, PP layout, and DeepEP SMS tuning
```

### DSV3 on GB200 or GB300

```text
Dispatcher: HybridEP
TP=1  EP=64  PP=4  VPP=4
Precision: MXFP8-class
CUDA Graph: attn + moe_router + moe_preprocess
Priority: HybridEP, CPU optimization, and graph-friendly static shapes
```

### Qwen3 235B on H100

```text
Dispatcher: alltoall in the current canonical recipe; re-screen flex backends on the target stack
TP=2  EP=32  PP=8  VPP=4
Recompute: none in the current canonical recipe
Priority: communication overlap and router-path cleanup
```

### Qwen3 235B on GB200

```text
Dispatcher: HybridEP
TP=1 or 2  EP=32 to 64  PP=4  VPP=unspecified unless measured
CUDA Graph: attn + moe_router + moe_preprocess
Recompute: moe_act, mlp, or norm depending on memory pressure
Priority: balance throughput against memory headroom
```

### Qwen3 30B-A3B on 16 H100

```text
Dispatcher: HybridEP
TP=1  EP=16  PP=1  CP=1
Precision: BF16
Sequence: 4096
Batch: MBS1 GBS1024
Routing: force balance
EP overlap: enabled
Delayed wgrad: disabled
CUDA Graph: moe_router + moe_preprocess
HybridEP: permute fusion, 32 SMs, 64-token combine chunks
Measured: 20.14729s/step, 299.352 model TFLOPS/GPU over iterations 41-50
Rank-0 peak allocated memory: 62.166 GiB
```

The current number is the final multi-knob canonical recipe result. An earlier
matched A/B isolated plain EP overlap: 244.039 to 287.305 TFLOPS/GPU, with
communication hidden by GEMM/attention increasing from 0.11% to 36.55%. Do not
attribute the later 299.352 result entirely to overlap.

### Qwen3-Next 80B on GB200

```text
Dispatcher: HybridEP
TP=1  EP=32  PP=2  VPP around 4
CUDA Graph: attn + moe_router + moe_preprocess
Priority: pipeline layout and grouped GEMM quality
```

## Cross-Cutting Patterns

### PP layout

- `E` = embedding
- `t` = transformer
- `m` = MTP
- `L` = loss
- `|` = stage boundary

The biggest platform difference is usually not just the dispatcher. It is the
combination of dispatcher, PP shape, and whether VPP keeps each stage balanced.

### Recompute strategy

| Memory pressure | Starting point |
|---|---|
| low | none or a very narrow selective set |
| moderate | `moe_act`, `mlp`, `norm`, or similar selective modules |
| high | model-specific up-projection plus selective MoE and MLP modules |
| extreme or long-context | full recompute only if the selective path still does not fit |

### Environment variables

```bash
CUDA_DEVICE_MAX_CONNECTIONS=1
CUDA_DEVICE_MAX_CONNECTIONS=32   # common when EP overlap and CUDA graphs are combined
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
NCCL_GRAPH_REGISTER=0
```

### CPU-side tuning

On GB200 and GB300, CPU affinity and general host-overhead cleanup can move the
needle almost as much as a dispatcher swap. Treat them as first-class tuning
work, not as afterthoughts.

## Pitfalls

1. **Do not cargo-cult a tracker row**: the winning config usually depends on
   routing mode, container, and PP layout as much as on hardware name.

2. **Container quality matters**: large regressions can come from the software
   stack rather than the model recipe.

3. **VPP must be intentional**: a bad VPP split can erase the gain from a better
   dispatcher.

4. **Compare absolute throughput, not only MFU**: MFU can mislead when switching
   between BF16, FP8, and other precision modes.

5. **Force-balance routing is benchmark-only**: it can control routing variance,
   but it changes semantics. Keep routing fixed within an A/B and validate
   natural routing separately for training acceptance.

6. **Do not treat the dispatcher table as a hard platform rule**: HybridEP is
   the validated winner for the canonical 16×H100 Qwen3 30B shape, while the
   current 256×H100 Qwen3 235B recipe uses `alltoall`. Benchmark backend
   compatibility and throughput in the production container.

7. **Separate screening, causality, and acceptance**: short runs reject weak
   candidates, matched one-variable A/Bs explain a mechanism, and a 50-step
   final run validates the complete winner.

_Last signature refresh: 2026-08-03._
