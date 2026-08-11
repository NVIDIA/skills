---
name: nemo-mbridge-perf-expert-parallel-overlap
description: Validate and use MoE expert-parallel communication overlap in Megatron-Bridge, including overlap_moe_expert_parallel_comm, delay_wgrad_compute, and flex dispatcher backends such as DeepEP and HybridEP.
license: Apache-2.0
when_to_use: Enabling EP overlap to hide dispatch/combine latency, or tracing a throughput regression to an EP overlap config change; 'overlap_moe_expert_parallel_comm', 'delay_wgrad_compute', 'flex dispatcher', 'DeepEP overlap', 'HybridEP overlap'.
---

# MoE Expert-Parallel Overlap Skill

## References

- Stable docs: @docs/training/communication-overlap.md
- Structured metadata: @skills/nemo-mbridge-perf-expert-parallel-overlap/card.yaml

## What It Is

Expert-parallel (EP) overlap hides the cost of token dispatch/combine all-to-all
communication by running it concurrently with expert FFN compute. Optionally,
delayed expert weight-gradient computation (`delay_wgrad_compute`) provides
additional overlap by deferring wgrad to overlap with the next layer's forward.

Bridge supports two dispatcher paths:

| Dispatcher | Backend | When to use |
|---|---|---|
| `alltoall` | Standard MoE all-to-all | Default, broadest compatibility |
| `flex` | DeepEP or HybridEP | Higher overlap on Ampere/Hopper/Blackwell |

## Quick Decision

Use EP overlap when:

- the model is MoE with `EP > 1`
- expert dispatch/combine communication is a meaningful part of step time
- you have memory headroom and are tuning for throughput

Prefer:

- `alltoall` dispatcher for the first rollout (broader compatibility)
- `flex` + DeepEP/HybridEP when running on supported GPUs and seeking
  additional gains

Avoid EP overlap when:

- full activation recompute is enabled
- `moe_shared_expert_overlap` is enabled
- the run is still being brought up for correctness
- PyTorch < 2.6.0

Expected outcome:

- if all-to-all dispatch is a clear profile bottleneck, overlap can produce a
  modest to meaningful speedup
- if the run is tiny, communication-light, or dominated by another wall, the
  gain may be negligible

## Correctness-First alltoall Benchmark

For the plain EP-overlap isolation benchmark, keep flex dispatch and delayed
wgrad disabled. The measured shape was Qwen3 MoE 30B-A3B SFT on 16 H100 GPUs:
`EP=16`, `alltoall`, BF16, global batch size 1024, CUDA graphs disabled,
`moe_permute_fusion=false`, measured over iterations 3-8.

Use these overrides for the plain-overlap case:

```bash
--cuda_graph_impl none \
--moe_flex_dispatcher_backend None \
--moe_a2a_overlap false \
comm_overlap.overlap_moe_expert_parallel_comm=true \
comm_overlap.delay_wgrad_compute=false \
model.moe_shared_expert_overlap=false
```

Do not use `--moe_a2a_overlap true` for this isolation test: the performance
harness helper enables both `overlap_moe_expert_parallel_comm` and
`delay_wgrad_compute`, so it does not isolate plain EP overlap.

Steady-window timing from that benchmark:

| Case | Steady mean | Relative |
|---|---:|---:|
| no EP overlap | 41.25s | 1.000x |
| EP overlap | 31.31s | 1.317x |
| EP overlap plus `delay_wgrad_compute` | 31.20s | 1.322x |

This is evidence for enabling plain EP overlap on this inter-node all-to-all
shape. It does not show a meaningful independent win from delayed wgrad, and it
does not validate fused MoE permutation because that path was disabled for the
runtime stack.

## HybridEP Production-Shape Benchmark

A 2026-07-25 controlled Qwen3 30B-A3B pretraining comparison validated plain
EP overlap with the production HybridEP path:

```text
Hardware: 16×H100
Precision: BF16
Sequence: 4096
Parallelism: TP1 / PP1 / CP1 / EP16
Batch: MBS1 / GBS1024
Routing: force balance
Dispatcher: flex + HybridEP
CUDA graph: Transformer Engine scopes moe_router + moe_preprocess
Delayed wgrad: disabled
```

| Case | Steady window | Step time | Model TFLOPS/GPU |
|---|---:|---:|---:|
| overlap off | iterations 5-20 | 24.7138s | 244.039 |
| overlap on, search run | iterations 5-20 | 21.0725s | 286.208 |
| overlap on, independent validation | iterations 41-50 | 20.9920s | 287.305 |

The independent run reduced step time by 15.059% and raised throughput by
17.729% over the reproduced baseline. Loss was finite, skipped and NaN
iterations remained zero, and rank-0 peak allocated memory was 62.166 GiB.

A matched Nsight Systems comparison captured the same 463,348 rank-0 kernels
per case. Enabling overlap increased communication concurrent with GEMM and
attention from 9.079ms (0.11% of communication time) to 3,958.997ms (36.55%).
GPU-active interval union fell from 22.821s to 21.221s.

Use this as evidence for the mechanism, not as a universal speedup promise.
The dispatcher, graph scopes, routing, parallelism, batch shape, and runtime
were held fixed while only plain EP overlap changed.

## Enablement

### alltoall dispatcher

```python
cfg.comm_overlap.overlap_moe_expert_parallel_comm = True
cfg.comm_overlap.delay_wgrad_compute = False
cfg.model.moe_shared_expert_overlap = False

cfg.model.expert_model_parallel_size = 8
cfg.model.num_moe_experts = 64
cfg.model.moe_token_dispatcher_type = "alltoall"
cfg.model.bf16 = True
cfg.model.fp16 = False
```

Enable `delay_wgrad_compute=True` only after the plain overlap path is known to
work and its extra compatibility constraints have been checked.

### flex dispatcher (DeepEP or HybridEP)

```python
from megatron.bridge.training.flex_dispatcher_backend import apply_flex_dispatcher_backend

cfg.comm_overlap.overlap_moe_expert_parallel_comm = True
cfg.comm_overlap.delay_wgrad_compute = False
cfg.model.moe_shared_expert_overlap = False

apply_flex_dispatcher_backend(cfg.model, moe_flex_dispatcher_backend="deepep")
# or: apply_flex_dispatcher_backend(cfg.model, moe_flex_dispatcher_backend="hybridep")
```

Benchmark plain EP overlap first. Enable `delay_wgrad_compute=True` only as a
separate follow-up A/B after its CUDA-graph and TE compatibility constraints
are satisfied.

## Compatibility And Constraints

- `expert_model_parallel_size > 1`
- `num_moe_experts > 1`
- `moe_token_dispatcher_type` must be `"alltoall"` or `"flex"`
- `moe_shared_expert_overlap = False`
- Base precision is BF16 or FP16
- PyTorch `>= 2.6.0`
- If `PP > 1`, `virtual_pipeline_model_parallel_size` must be set
- `recompute_granularity != "full"`, `recompute_method = None`,
  `recompute_num_layers = None`
- `mtp_num_layers` must be `None` or `1`
- `delay_wgrad_compute` requires `overlap_moe_expert_parallel_comm` as a
  prerequisite
- `delay_wgrad_compute` with `overlap_grad_reduce` requires TE >= 2.7.0
- `delay_wgrad_compute` with `gradient_accumulation_fusion` requires TE >= 2.7.0
- CUDA graph `attn` scope + `delay_wgrad_compute` requires TE >= 2.12.0,
  `gradient_accumulation_fusion = True`, and no attention bias
- DeepEP: Ampere, Hopper, B200, B300 GPUs only
- HybridEP: Ampere, Hopper, B200, B300, GB200/GB300 with NVL72

## Minimal Working Config

```python
cfg.comm_overlap.overlap_moe_expert_parallel_comm = True
cfg.comm_overlap.delay_wgrad_compute = False
cfg.model.expert_model_parallel_size = 4
cfg.model.num_moe_experts = 64
cfg.model.moe_token_dispatcher_type = "alltoall"
cfg.model.moe_shared_expert_overlap = False
cfg.model.bf16 = True
```

Use this as the correctness-first starting point. Add delayed wgrad, flex
dispatch, and CUDA-graph interactions only after the plain overlap path is
known to work.

## Minimal Runnable Command

Performance harness example inside a Slurm allocation. Keep the model,
parallelism, dispatcher, and runtime fixed, and vary only the two overlap
overrides:

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

Do not use `--moe_a2a_overlap true` when separating plain EP overlap from
delayed wgrad: the performance harness helper enables both
`overlap_moe_expert_parallel_comm` and `delay_wgrad_compute`.

Unit test verification:

```bash
uv run python -m pytest \
  tests/unit_tests/training/test_comm_overlap.py -k "moe" \
  tests/unit_tests/training/test_deepep.py -q
```

## Verification

### Unit tests

```bash
uv run python -m pytest \
  tests/unit_tests/training/test_comm_overlap.py \
  tests/unit_tests/training/test_deepep.py -q
```

### Log checks

After a successful run with EP overlap:

1. Confirm no assertion errors during `CommOverlapConfig` finalization
2. Confirm `overlap_moe_expert_parallel_comm` appears as `True` in the logged
   config
3. If using flex dispatcher, confirm `moe_token_dispatcher_type = "flex"` and
   the correct backend in logs

### Success criteria

- Config validation passes for the selected dispatcher and overlap settings
- Training runs complete without hangs or assertion failures
- Throughput improves or at least does not regress for the target workload
- Loss trajectory matches baseline (overlap should not affect convergence)

### Profile interpretation

Use an unprofiled steady window for the throughput acceptance result. Use a
matched profile to explain the mechanism:

1. Keep the dispatcher, routing, graph scopes, batch shape, parallel layout,
   and runtime fixed.
2. Capture the same rank and steady iteration while toggling only plain EP
   overlap.
3. Build interval unions for communication and compute kernels, then measure
   their intersection.
4. Do not use summed kernel duration as wall time. Concurrent kernels can run
   longer under SM or bandwidth contention even when exposed time decreases.
5. Corroborate interval results with dispatch/combine NVTX ranges, final step
   time, loss finiteness, skipped/NaN counts, and peak memory.

## Code Anchors

### Bridge overlap validation

```470:505:src/megatron/bridge/training/comm_overlap.py
if self.user_comm_overlap_cfg.overlap_moe_expert_parallel_comm is True:
    assert model_cfg.expert_model_parallel_size > 1, ...
    assert model_cfg.num_moe_experts > 1, ...
    assert model_cfg.moe_token_dispatcher_type in ["alltoall", "flex"], ...
    assert model_cfg.bf16 or model_cfg.fp16, ...
    assert is_torch_min_version("2.6.0"), ...
    # ... PP + VPP check, recompute checks, shared_expert_overlap check ...
```

### Delayed wgrad validation

```507:557:src/megatron/bridge/training/comm_overlap.py
if self.user_comm_overlap_cfg.delay_wgrad_compute is True:
    # TE version checks for overlap_grad_reduce and gradient_accumulation_fusion
    # CUDA graph scope validations for delayed wgrad
    assert overlap_moe_expert_parallel_comm, ...
```

### Flex-dispatcher activation

```27:72:src/megatron/bridge/training/flex_dispatcher_backend.py
def apply_flex_dispatcher_backend(...):
    # GPU architecture check for DeepEP / HybridEP
    model_config.moe_token_dispatcher_type = "flex"
    model_config.moe_flex_dispatcher_backend = moe_flex_dispatcher_backend
    model_config.moe_shared_expert_overlap = False
```

### Perf harness override

```149:156:scripts/performance/utils/overrides.py
def _set_moe_a2a_overlap_overrides(recipe, moe_a2a_overlap=False):
    if moe_a2a_overlap:
        recipe.comm_overlap.overlap_moe_expert_parallel_comm = True
        recipe.comm_overlap.delay_wgrad_compute = True
        recipe.model.moe_shared_expert_overlap = False
```

### Tests

| File | Coverage |
|---|---|
| `tests/unit_tests/training/test_comm_overlap.py` | EP overlap validation, delayed wgrad, CUDA graph + wgrad interaction |
| `tests/unit_tests/training/test_deepep.py` | DeepEP/HybridEP helper activation and GPU gating |

## Failure Diagnosis

| Symptom | Likely Cause | How To Confirm | Fix |
|---|---|---|---|
| assert `expert_model_parallel_size > 1` | EP not configured | Check `expert_model_parallel_size` | Set EP > 1 |
| assert `moe_token_dispatcher_type` | Wrong dispatcher | Check dispatcher type | Use `"alltoall"` or `"flex"` |
| assert on BF16/FP16 | Wrong precision | Check `bf16` and `fp16` | Set `bf16 = True` |
| hang during training | PyTorch < 2.6 | Check PyTorch version | Upgrade to >= 2.6.0 |
| assert `virtual_pipeline_model_parallel_size` | PP > 1 without VPP | Check PP and VPP config | Set VPP when PP > 1 |
| assert `recompute_granularity` | Full recompute enabled | Check recompute settings | Disable full recompute |
| assert `overlap_moe_expert_parallel_comm required` | delayed wgrad without EP overlap | Check `delay_wgrad_compute` without overlap | Enable EP overlap first |
| assert `gradient_accumulation_fusion` | CUDA graph + delayed wgrad | Check graph scope + wgrad settings | Enable `gradient_accumulation_fusion` |
| assert on attention bias | CUDA graph attn + delayed wgrad + bias | Check `add_bias_linear` / `add_qkv_bias` | Disable attention bias |
| no throughput gain from flex dispatcher | `apply_flex_dispatcher_backend` not called | Check `moe_token_dispatcher_type` in logs | Call `apply_flex_dispatcher_backend(...)` |
| DeepEP/HybridEP silently skipped | Unsupported GPU | Check warning logs | Run on Ampere/Hopper/Blackwell |
| summed kernel time increases after overlap | Expected concurrency contention or a regression | Compare interval unions, comm/compute intersection, and unprofiled step time | Judge overlap from exposed wall time, not summed per-stream duration |

## Known Limitations

- Setting `moe_flex_dispatcher_backend` alone does not activate flex dispatch —
  you must call `apply_flex_dispatcher_backend(...)`.
- Public recipes are often conservative and leave MoE overlap disabled by
  default.
- Controlled end-to-end and profile evidence exists for one Qwen3 30B-A3B
  HybridEP H100 shape; repeat the matched A/B before generalizing it to another
  model, dispatcher, topology, precision, or batch shape.
- MoE overlap and shared-expert overlap are mutually exclusive.
- CUDA graph plus delayed wgrad is a multi-constraint path that requires
  careful TE version and scope validation.

_Last signature refresh: 2026-08-03._
