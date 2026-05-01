---
name: moe-dispatcher-selection
description: Choose the right MoE token dispatcher (`alltoall`, DeepEP, or HybridEP) for the hardware, EP degree, and optimization stage. Summarizes patterns from DSV3, Qwen3, Qwen3-Next, and VLM bring-up work.
---

# MoE Dispatcher Selection Guide

Stable docs: @docs/training/moe-optimization.md
Card: `card.yaml` (co-located)

## Quick Decision

### By hardware

| Hardware | First choice | Why |
|---|---|---|
| H100 | DeepEP | Strong default for cross-node EP on Hopper |
| B200 | DeepEP | Good first choice unless a platform-specific HybridEP path is available |
| GB200 / GB300 NVL72 | HybridEP | Best fit for NVLink-domain-aware dispatch and lower memory pressure |
| Unknown or first bring-up | `alltoall` | Easiest path for correctness and debugging |

### By EP degree

| EP size | Guidance |
|---|---|
| Small EP | Dispatcher choice is usually second-order; start with `alltoall` or DeepEP |
| Medium EP | DeepEP often becomes worthwhile |
| Large EP | HybridEP is usually the best target on NVL72 systems |

## Model-Family Patterns

| Workload | Common best path | Notes |
|---|---|---|
| DSV3 at large scale | HybridEP on GB200 or GB300, DeepEP on H100 | Dispatcher choice matters more as EP and PP both grow |
| Qwen3 235B | DeepEP on H100, HybridEP on GB200 | HybridEP usually wins on GB200 and often uses less memory |
| Qwen3 30B | DeepEP | Smaller models still benefit, but the absolute gap is smaller |
| Qwen3-Next | Close race in BF16, HybridEP stronger in FP8 or memory-tight runs | Good reminder to test, not assume |
| MoE VLMs | Start simple, then test HybridEP on GB200-class systems | Vision workloads are sensitive to both memory and host overhead |

## Rounded Evidence Summary

### DSV3 on GB200 or GB300

The broad trend is more important than any single row in the tracker:

- plain `alltoall` is usually the conservative baseline
- DeepEP improves that baseline once EP communication becomes visible
- HybridEP adds another step up on NVL72 systems, especially after CUDA graphs,
  routing improvements, and CPU-side cleanup are already in place

In practice, the stack often moves from roughly "low-teens MFU" territory with
an untuned baseline into "high-teens to low-20s MFU" territory after the full
dispatcher and kernel stack is tuned.

### Qwen3 235B on GB200

For Qwen3 235B, the practical ordering is usually:

1. `alltoall` for initial bring-up
2. DeepEP if you want a familiar tuned path
3. HybridEP for the strongest steady-state result on GB200

HybridEP is usually modestly faster than `alltoall` on this workload and often
has noticeably better memory headroom.

### Qwen3-Next on GB200

This family is a good reminder that dispatcher wins are workload-dependent:

- in BF16, `alltoall` and HybridEP can be close
- in FP8 or memory-constrained settings, HybridEP tends to look better
- pipeline layout and grouped-GEMM changes can matter almost as much as the
  dispatcher itself

## Tuning Parameters

### DeepEP

DeepEP is selected by setting
`moe_token_dispatcher_type="flex"` and `moe_flex_dispatcher_backend="deepep"`.

```bash
--moe-deepep-num-sms 20
```

Tune the SM count allocated to DeepEP communication kernels (default 20).
The optimal value depends on the workload and EP degree.

### HybridEP

HybridEP is selected by setting
`moe_token_dispatcher_type="flex"` and `moe_flex_dispatcher_backend="hybridep"`.

```bash
--moe-hybridep-num-sms 16
```

Tune the SM count allocated to HybridEP communication (default 16). The
performance harness uses 32 for HybridEP workloads. Sweep between 16 and 32
for the target hardware. Set
`NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN` to match the NVLink domain size of
the deployment. If it does not match the actual topology, performance and
sometimes correctness will suffer.

### Routing mode

```bash
--moe-router-force-load-balancing
```

For performance benchmarking, force-balance routing is the safer default. It
usually outperforms dropless routing in large-scale benchmarks and makes results
more comparable across dispatcher backends.

## Key Interactions

| Feature | Interaction |
|---|---|
| CUDA graphs | Best paired with `attn moe_router moe_preprocess` on dropless MoE |
| EP overlap | Helps when dispatcher time is still visible after backend tuning |
| FP8 | Often increases the relative importance of communication and host overhead |
| CPU affinity | Can matter as much as dispatcher choice on GB200 or GB300 |
| Pipeline layout | Poor PP or VPP layout can erase dispatcher gains |

## When To Use Each

### `alltoall`

- first correctness bring-up
- small EP configurations
- debugging communication regressions

### DeepEP

- Hopper or B200 deployments
- cross-node EP is clearly visible in profiles
- you want a mature intermediate step before testing HybridEP

### HybridEP

- GB200 or GB300 NVL72 systems
- large EP degrees
- memory headroom matters in addition to throughput

## Pitfalls

1. **Do not compare dispatchers on different stacks**: container, routing mode,
   PP layout, and CUDA-graph scope can move the result as much as the dispatcher.

2. **HybridEP is topology-sensitive**: it is not a universal win outside the
   hardware it was designed for.

3. **Both dispatchers need SM tuning**: default `moe_deepep_num_sms` (20) and
   `moe_hybridep_num_sms` (16) are reasonable starting points but rarely optimal.

4. **Force-balance and dropless are not interchangeable baselines**: keep the
   routing mode fixed when comparing dispatcher backends.

5. **Memory and throughput can trade off differently by model**: Qwen3-style
   runs may show a smaller speed delta than DSV3, but still justify HybridEP for
   memory headroom.
