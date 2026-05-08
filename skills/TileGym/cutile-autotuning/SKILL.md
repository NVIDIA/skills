---
name: cutile-autotuning
description: "Use when adding, modifying, optimizing, or debugging CuTile autotuning code. Trigger signals: `exhaustive_search` / `replace_hints` / `hints_fn` / `cuda.tile.tune` in code, `autotune` in filenames, or correctness/performance issues in autotuned CuTile kernels. Covers: tune-once/cache/launch pattern, per-architecture configs (sm80–sm120), parameter space design (tile sizes, occupancy, num_ctas), and 7 common pitfalls with solutions."
license: CC-BY-4.0 AND Apache-2.0
---

# CuTile Autotuning

Add autotuning to CuTile kernels using the `exhaustive_search` API with tune-once/cache/direct-launch pattern.

## Instructions

Follow the decision tree to classify the kernel, design a search space, implement the tune-once/cache/launch pattern, and validate performance.

1. **Classify** — use the Decision Tree to determine search dimensions (occupancy-only vs full tile search)
2. **Design search space** — select the matching template from `references/kernel-type-templates.md`; prune to ≤ 30 configs via arch filters
3. **Implement** — add `exhaustive_search` + cache + `ct.launch` following the Step-by-Step Workflow; handle in-place writes with split-buffer if needed
4. **Test** — run correctness with autotune enabled and with `DISABLE_AUTOTUNE=1`
5. **Validate** — A/B benchmark against fixed best-known config; see `references/search-strategies.md`

## Task Router — Jump to What You Need

| What are you trying to do? | Go to |
|---|---|
| Add autotune to a new kernel (most common) | Quick Reference below → Workflow: Adding Autotune → `references/kernel-type-templates.md` (pick by kernel type: T1=elementwise, T2=in-place, T3=matmul, T4=persistent, T5=FMHA, T6=FP8, T7=grouped GEMM, T8=varlen attention, T9=dual-GEMM fusion) |
| Debug: data corruption / wrong results after first run | Pitfall #1 (In-Place Kernel) |
| Debug: autotune taking 5+ minutes | Pitfall #2 (Compilation Timeout) |
| Debug: search space generator returning zero configs | Pitfall #5 first; also check arch filters, size guards, and `num_ctas` constraints |
| Optimize an existing autotune config | Workflow: Optimizing an Existing Config |

## Quick Reference — Occupancy-Only Autotune (Tune-Once/Cache/Launch)

Most CuTile kernels (elementwise, reduction, LayerNorm) need only occupancy tuning. Copy this pattern:

```python
from types import SimpleNamespace
from cuda.tile.tune import exhaustive_search
import cuda.tile as ct
import torch

def _my_autotune_configs():
    for occ in [1, 2, 4, 8]:
        yield SimpleNamespace(occupancy=occ)

# Module-level cache: tune once, launch fast forever after
_autotune_cache = {}

def my_op(x, output):
    stream = torch.cuda.current_stream()
    NUM_SM = torch.cuda.get_device_properties(x.device).multi_processor_count

    # Cache key: anything that affects optimal config (use str() for device)
    cache_key = (x.shape, x.dtype, str(x.device))

    if cache_key not in _autotune_cache:
        configs = list(_my_autotune_configs())
        result = exhaustive_search(
            configs,
            stream,
            grid_fn=lambda cfg: (min(NUM_SM * cfg.occupancy, M), 1, 1),
            kernel=my_kernel,
            args_fn=lambda cfg: (x, output, ...),
            hints_fn=lambda cfg: {"occupancy": cfg.occupancy},
        )
        best_cfg = result.best.config
        tuned_kernel = my_kernel.replace_hints(occupancy=best_cfg.occupancy)
        _autotune_cache[cache_key] = (best_cfg, tuned_kernel)  # cache BOTH

    cfg, tuned_kernel = _autotune_cache[cache_key]
    grid = (min(NUM_SM * cfg.occupancy, M), 1, 1)
    ct.launch(stream, grid, tuned_kernel, (x, output, ...))
```

Key rules:
- **Tune once, cache, launch directly** — `exhaustive_search` runs only on first call per shape; subsequent calls use cached config + `ct.launch` with zero overhead
- For in-place kernels use split-buffer during search (separate input/output tensors)
- Keep ≤ 30 configs total
- `exhaustive_search` requires a `Sequence` (list/tuple) — convert generators with `list()`
- **Search space must include the original fixed config** — this guarantees autotuning never makes performance worse

**When to use this pattern**: Kernel has fixed block size (not tile-size tunable). Includes: elementwise (SwiGLU, GeGLU), reduction (RMSNorm, LayerNorm), RoPE, and persistent kernels with heuristic block sizes (grouped GEMM).

For complex kernels (matmul with tile sizes, FMHA, FP8 with num_ctas), read the full guide below + [`kernel-type-templates.md`](references/kernel-type-templates.md).

> **⚠️ Two pitfalls catch almost everyone — check before submitting:**
> - **In-place kernel** (writes back to input tensor)? → MUST use split-buffer pattern during search → Pitfall #1
> - **Search space empty?** → Check arch filters and `num_ctas` constraints → Pitfall #5

## Reading Guide

- **Occupancy-only kernels** (elementwise, reduction, persistent with fixed block sizes): Quick Reference + Pitfall Checklist is sufficient — skip `references/` docs. For in-place kernels, also read Pitfall #1.
- **Complex kernels** (matmul with tunable tile sizes, FMHA, FP8 with num_ctas): Quick Reference → Decision Tree → API Reference → Step-by-Step Workflow → relevant `references/` docs.

**5-step summary**: Classify kernel → Design search space ([`parameter-space-design.md`](references/parameter-space-design.md)) → Implement using template ([`kernel-type-templates.md`](references/kernel-type-templates.md)) → Validate with A/B test → Check Pitfall Checklist.

## Design Philosophy

**Build a small, precise search space bottom-up — not a large space trimmed down.** CuTile compilation is much heavier than Triton (~0.5-1s per config), so 30 configs is the hard upper limit. The approach is: classify the kernel type first, then construct only the relevant configs for that type and architecture. Never start with a large cartesian product and prune — start with the minimum viable space and expand only if data shows it's needed.

## Decision Tree: What Search Dimensions Does This Kernel Need?

All kernels should have autotuning added. The question is not *whether* to autotune, but *what dimensions* to search:

```
What type of kernel is this?
├── Compute-bound (matmul, GEMM, FMHA) → Does it have multiple tunable dimensions (tile sizes)?
│   ├── YES → Is it a fused multi-GEMM kernel (dual-GEMM, e.g. Linear+GLUAct)?
│   │   ├── YES → Template 9: low occupancy (1–2), conservative tiles (2× SHMEM/register pressure)
│   │   └── NO  → Full search: TILE_M × TILE_N × (TILE_K) × occupancy × num_ctas
│   │             (see matmul/FMHA templates in kernel-type-templates.md)
│   └── NO  → Occupancy-only search: [1, 2, 4, 8]
│             (see Quick Reference above)
├── Balanced (LayerNorm, reduction + compute) →
│   Occupancy-only search: [1, 2, 4, 8]
│   Expected benefit: 2-15%
└── Memory-bound (CE Loss, pure elementwise) →
    Occupancy-only search: [1, 2, 4, 8]
    Expected benefit: 0-15% (varies by kernel; zero-cost after tuning)
```

**Why memory-bound kernels only search occupancy (not num_ctas or tile sizes)**:
- **`num_ctas` has zero benefit**: `num_ctas > 1` enables TMA multicast, where multiple CTAs share tile data in shared memory (e.g., matmul A/B tiles reused across CTAs). Memory-bound kernels use per-element `ct.gather`/`ct.scatter` with no tile reuse — multi-CTA cooperation adds overhead with no data sharing benefit.
- **Tile sizes are pre-determined**: BLOCK_SIZE for memory-bound kernels is determined by offline sweep (e.g., 1024 is globally optimal on B200 across [256, 512, 1024, 2048, 4096, 8192]). This is a constant, not a runtime tunable.
- **Occupancy is the only effective knob**: Higher occupancy lets the GPU hide memory latency by switching to another CTA while one is stalled on a memory request.

> **Evidence — CE Loss experiment**: A 12-config search (occupancy × num_ctas) on Cross-Entropy Loss yielded only 2.5% gain (0.79x → 0.81x vs Triton). The `num_ctas` dimension contributed nothing; the result was reverted because compilation cost outweighed the marginal benefit. Occupancy-only (4 configs) achieves the same result at 3x less compilation time.

**Note on memory-bound kernels**: Adding occupancy-only autotune is always worthwhile because:
- The tune-once/cache/launch pattern has zero runtime overhead after the first call
- The search space is tiny (4 configs, ~2-4s compilation)
- Even small improvements have value at scale

## Occupancy Selection Guide

Occupancy controls how many CTAs run concurrently per SM. Use this as a starting point when designing the occupancy search space:

| Occupancy Range | Best For | Example Kernels |
|-----------------|----------|-----------------|
| 1–4 | Compute-bound (heavy math) | Complex transforms, matmul |
| 4–8 | Balanced (GEMM, TMA) | Matrix multiply, FMHA |
| 8–16 | Memory-bound (reductions) | Softmax, LayerNorm |
| 16–32 | Very light (copies, casts) | Type conversions, elementwise |

Use these ranges to seed your initial search space. For occupancy-only kernels, `[1, 2, 4, 8]` covers most cases — see Quick Reference above.

## exhaustive_search API Reference

> **⚠️ Deprecated API**: `cuda.tile_experimental.autotune_launch()` (aka `ct_experimental.autotune_launch`) is deprecated and should NOT be used. It combines search + launch in one call with random sampling, which produces less reproducible results and worse config selection compared to `exhaustive_search`. Always use `cuda.tile.tune.exhaustive_search` (the current API below) with explicit caching and `ct.launch`.

### Current API (`cuda.tile.tune`)

```python
from cuda.tile.tune import exhaustive_search, TuningResult

result: TuningResult = exhaustive_search(
    search_space,   # Sequence[T] — list or tuple of configs (NOT a generator)
    stream,         # torch.cuda.current_stream()
    grid_fn,        # callable(cfg) → tuple[int, ...]
    kernel,         # @ct.kernel decorated function
    args_fn,        # callable(cfg) → tuple of kernel args
    hints_fn=None,  # callable(cfg) → {"occupancy": int, "num_ctas": int}
    *,
    quiet=False     # suppress output
)
```

### TuningResult

```python
@dataclass
class TuningResult[T]:
    best: Measurement       # best config + timing (mean_us, error_margin_us, num_samples)
    successes: Sequence[Measurement]   # all successful configs (sorted by performance)
    failures: Sequence[tuple[T, str, str]]  # (config, exception_type, message)
```

Key properties:
- **Exhaustive**: evaluates ALL configs in order — no random sampling, no skipped configs
- **Search only**: does not perform the final production launch — it executes trial runs internally for benchmarking, but you call `ct.launch` separately for the actual production invocation
- **No built-in cache**: you manage caching explicitly (see tune-once/cache/launch pattern)
- **Deterministic**: same search space always produces the same evaluation order

### Tune-Once / Cache / Launch Pattern

This is the **recommended pattern** for all autotuned kernels. It ensures:
- First call: runs `exhaustive_search` to find the best config (~2-30s depending on space size)
- Subsequent calls: uses cached config with `ct.launch` — zero overhead (identical to a fixed `ct.launch`)

```python
_cache = {}

def run_kernel_autotuned(x, ...):
    stream = torch.cuda.current_stream()
    cache_key = (x.shape, x.dtype, str(x.device))

    if cache_key not in _cache:
        configs = list(_my_autotune_configs())
        result = exhaustive_search(
            configs, stream,
            grid_fn=lambda cfg: ...,
            kernel=my_kernel,
            args_fn=lambda cfg: ...,
            hints_fn=lambda cfg: {"occupancy": cfg.occupancy},
        )
        best_cfg = result.best.config
        tuned_kernel = my_kernel.replace_hints(occupancy=best_cfg.occupancy)
        _cache[cache_key] = (best_cfg, tuned_kernel)  # cache BOTH config and compiled kernel

    cfg, tuned_kernel = _cache[cache_key]
    grid = compute_grid(cfg)
    ct.launch(stream, grid, tuned_kernel, (x, ...))
```

**Why this pattern matters**: The `ct.launch` call in the fast path is identical to what you'd write for a fixed-config kernel. There is zero per-call overhead — no lock, no hash lookup, no lambda invocation. The only cost is the Python dict lookup for `_cache[cache_key]`.

> **⚠️ Critical: always cache the tuned kernel object, not just the config.** `replace_hints()` returns a **new** kernel object with its own independent JIT cache. Calling it on every invocation triggers recompilation each time, degrading performance by 100–500×. Call `replace_hints()` once after `exhaustive_search`, store the returned kernel in the cache alongside the config, and reuse it directly on the fast path. See Pitfall #7.

### replace_hints

After finding the best config, use `kernel.replace_hints()` to create a kernel variant with the optimal hints:

```python
# For occupancy-only:
tuned_kernel = my_kernel.replace_hints(occupancy=cfg.occupancy)

# For occupancy + num_ctas:
tuned_kernel = my_kernel.replace_hints(occupancy=cfg.occupancy, num_ctas=cfg.num_ctas)
```

`replace_hints` accepts only `occupancy` and `num_ctas` — these are the only compiler hints controllable via the autotune API.

**`ByTarget` wrapping for cross-architecture portability**: When creating tuned kernel variants via `ct.kernel()`, prefer wrapping hint values in `ct.ByTarget` for portability across GPU architectures:

```python
# Preferred: explicit architecture targeting (portable)
tuned_kernel = ct.kernel(
    my_kernel._pyfunc,
    occupancy=ct.ByTarget(sm_100=best_cfg.occupancy),
    num_ctas=ct.ByTarget(sm_100=best_cfg.num_ctas, default=1),
)

# Also acceptable: plain integers (when targeting a single architecture)
tuned_kernel = ct.kernel(my_kernel._pyfunc, occupancy=best_cfg.occupancy)
```

When targeting only the current GPU (the common case in autotuning), plain integers work fine. Use `ByTarget` when the code may run on multiple architectures or when following production conventions (TileGym production code consistently uses `ByTarget`).

### Kernel Hints

CuTile kernel performance is controlled by two compile-time hints:

- **`occupancy`**: Number of CTAs per SM. Higher occupancy = more parallelism but less shared memory per CTA.
- **`num_ctas`**: Number of CTAs in a CGA (Cooperative Group Array). Used for multi-CTA cooperation (e.g., TMA multicast). Only supported on sm90+.

Three ways to set hints:

```python
# 1. Fixed value in decorator (no autotune needed)
@ct.kernel(occupancy=2, num_ctas=1)
def my_kernel(...): ...

# 2. Architecture-specific fixed value (no autotune needed)
@ct.kernel(num_ctas=ct.ByTarget(sm_100=2, sm_120=1, default=1))
def my_kernel(...): ...

# 3. Runtime autotune via exhaustive_search + replace_hints
# IMPORTANT: Remove fixed hints from decorator first!
@ct.kernel
def my_kernel(...): ...

# Then in the host wrapper:
tuned_kernel = my_kernel.replace_hints(occupancy=best_occ, num_ctas=best_ctas)
ct.launch(stream, grid, tuned_kernel, args)
```

**Important**: `replace_hints` correctly overrides decorator hints (it uses `dataclasses.replace()` internally). However, if you forget to call `replace_hints`, the decorator's fixed values are used instead of the autotuned values. To avoid this confusion, always remove fixed hints from the `@ct.kernel(...)` decorator before adding autotuning — this makes it explicit that hints come only from the autotune path.

### search_space Design

The search space is a list of `SimpleNamespace` objects. Each namespace holds config fields that `grid_fn`, `args_fn`, and `hints_fn` can read.

```python
from types import SimpleNamespace

# Occupancy-only (elementwise kernels)
def autotune_configs():
    for occ in [1, 2, 4, 8]:
        yield SimpleNamespace(occupancy=occ)

# Full matmul search space — see parameter-space-design.md for complete per-architecture configs
# Pattern: yield SimpleNamespace(TILE_SIZE_M=..., TILE_SIZE_N=..., TILE_SIZE_K=..., num_ctas=..., occupancy=...)
```

**Note**: `exhaustive_search` requires a `Sequence` (list/tuple), not a generator. Always convert with `list()`:
```python
configs = list(autotune_configs())
result = exhaustive_search(configs, ...)
```

### grid_fn Patterns

```python
from math import ceil

# Pattern A: Simple tile coverage (matmul, elementwise)
grid_fn=lambda cfg: (ceil(M / cfg.TILE_SIZE_M) * ceil(N / cfg.TILE_SIZE_N), 1, 1)

# Pattern B: Persistent matmul (static_persistent_matmul_kernel)
NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
grid_fn=lambda cfg: (
    min(NUM_SMS // cfg.num_ctas, ceil(M / cfg.TILE_M) * ceil(N / cfg.TILE_N)) * cfg.occupancy,
    1, 1,
)

# Pattern C: 2D grid (FMHA — one dim for seq tiles, one for batch*heads)
grid_fn=lambda cfg: (ceil(q_len / cfg.TILE_M), batch_size * num_heads, 1)

# Pattern D: 1D elementwise (cdiv = math.ceil(a/b), from ct_ops.py)
grid_fn=lambda cfg: (cdiv(n_elements, BLOCK_SIZE),)

# Pattern E: Grouped GEMM persistent (grid fixed at NUM_SMS, occupancy via hints_fn only)
grid_fn=lambda cfg: (NUM_SMS, 1, 1)
```

## Step-by-Step Workflow

### Adding Autotune to a New Kernel

1. **Classify the kernel** using the decision tree above.
   - *VERIFY*: You know whether this is occupancy-only or requires tile-size tuning.

2. **Remove hardcoded hints from decorator** (strongly recommended): If the kernel currently has hardcoded hints in its decorator (e.g. `@ct.kernel(occupancy=2, num_ctas=1)`), **remove those fixed hints** and change to bare `@ct.kernel` before adding autotuning. While `replace_hints` does correctly override decorator values at runtime, leaving them creates a silent fallback trap: if any code path (e.g., `DISABLE_AUTOTUNE`, error handling, or a future refactor) skips `replace_hints`, the decorator's fixed hints are used instead of the autotuned values — and this produces no error, just silently worse performance. Removing them makes the failure mode explicit (missing hints → compiler defaults) rather than silent (wrong fixed hints used).
   - *VERIFY*: The `@ct.kernel` decorator has no `occupancy=` or `num_ctas=` arguments before proceeding. Use bare `@ct.kernel` instead.

3. **Check for in-place writes**: If the kernel modifies input tensors in-place, you MUST use the split-buffer pattern during `exhaustive_search` — see Pitfall #1.
   - *VERIFY*: Either the kernel is not in-place, or you have added a split-buffer scratch tensor for the search phase.

4. **Select the template** from [`kernel-type-templates.md`](references/kernel-type-templates.md) based on kernel type.

5. **Design the search space** following [`parameter-space-design.md`](references/parameter-space-design.md):
   - **Start from reference configs**, not from scratch. Clone configs from existing production kernels of the same type (e.g., `ops/cutile/matmul.py` for GEMM) and adapt. For GEMM-class kernels, `nvMatmulHeuristics` can suggest 8-16 high-quality candidates that reach 96-99% peak performance — see [`parameter-space-design.md`](references/parameter-space-design.md) for details.
   - Detect the current GPU architecture with `torch.cuda.get_device_capability()`.
   - **Target one architecture at a time.** Generate configs only for the detected arch. Do NOT add branches for other architectures — they cannot be tested on this machine and untested code paths are unreliable. If multi-arch support is needed later, add it in a separate pass on the appropriate hardware.
   - Identify tunable parameters (tile sizes, occupancy, num_ctas)
   - **Ensure the search space includes the original fixed config** (or an equivalent). This guarantees that the autotuned result is at least as good as the original — no performance regression is possible.
   - If the generated set exceeds 30, apply tile size filters and pruning rules to reduce it to ≤ 30
   - *VERIFY*: Total configs ≤ 30 (hard limit: CuTile compilation is heavy, >30 configs will timeout).

6. **Implement** the tune-once/cache/launch pattern:
   - Define a `_cache` dict at module level
   - Define a cache key that captures all parameters affecting optimal config (shapes, dtypes, device, any flags like `is_causal`). **⚠️ Use `str(x.device)` not `x.device`** in the cache key — `torch.device` objects are not reliably hashable and can cause `TypeError: unhashable type` at runtime. Always convert to string: `cache_key = (..., x.dtype, str(x.device))`. **Tip**: For GEMM-class kernels, round dimensions to the next power of 2 in the cache key (e.g., `cache_key = (next_pow2(M), next_pow2(N), next_pow2(K), dtype, str(device))`) to reduce unique key count and avoid re-tuning for similar shapes.
   - Call `exhaustive_search(list(configs), ...)` only when cache misses
   - Store `result.best.config` in cache
   - Use `kernel.replace_hints(...)` to create the tuned kernel variant
   - Use `ct.launch()` for the actual kernel invocation
   - `grid_fn` correctly computes grid from config
   - `args_fn` passes all kernel arguments including tile sizes as `ct.Constant[int]`
   - `hints_fn` passes `occupancy` and/or `num_ctas` from config
   - *VERIFY*: `exhaustive_search` receives a `list()` of configs, not a raw generator.

7. **(MANDATORY) Add DISABLE_AUTOTUNE support** for CI and profiling: check `os.environ.get("DISABLE_AUTOTUNE", "0") == "1"` — when set, skip `exhaustive_search` entirely and fall back to `ct.launch` with the first valid config. This is required for:
   - CI determinism (autotune adds variable wall time)
   - NCU profiling (prevents autotune trial runs from cluttering the trace — see Pitfall #4)
   - Debugging (isolates kernel correctness from autotune behavior)
   Place the check *before* the cache lookup so that `DISABLE_AUTOTUNE=1` bypasses all autotune logic. Provide a hardcoded fallback config in case the generator yields zero configs.
   - *VERIFY*: Running with `DISABLE_AUTOTUNE=1` produces correct results and does not call `exhaustive_search`.

8. **Test**: Run correctness tests first (`pytest -k "test_op and cutile"`), then benchmark.
   - *VERIFY*: Correctness passes with autotune enabled AND with `DISABLE_AUTOTUNE=1`.

9. **Validate with A/B test**: Compare autotune version vs fixed best-known config. See [`search-strategies.md`](references/search-strategies.md) for methodology.
   - *VERIFY*: Autotune version ≥ baseline (or within noise). If worse, check that the search space includes the original fixed config, and that `replace_hints` is being used correctly.

10. **(MANDATORY) Run the test and verify performance before submitting.**

    Execute the provided test script (e.g. `ENABLE_TILE=1 python3 test.py`) and check:
    - `correctness: PASS`
    - `speedup_over_fixed >= 1.0` (autotuned must not be slower than fixed baseline)

    If `speedup_over_fixed < 1.0`:
    - Check that the search space includes the original fixed config (this guarantees no regression)
    - Check if `replace_hints` is being called on every code path — revisit Step 2 (if any path skips `replace_hints`, the decorator's fixed hints are used instead of autotuned values)
    - Expand search space if all configs perform similarly (see `references/parameter-space-design.md` → "Adapting Search Space")

    *⚠️ DO NOT submit without running the test at least once. Writing correct-looking code is not sufficient — autotuning bugs (silent hint override, split-buffer omission) are only caught at runtime.*

### Integration with torch.autograd.Function

When the kernel is used inside a `torch.autograd.Function`:
- Place the tune-once/cache/launch logic in `forward()` only. The cached config is reused across calls.
- In `backward()`, using `ct.launch` with a fixed or cached config is often sufficient. However, if backward has its own independent search space (e.g. grouped GEMM dX and dW have separate optimal configs), autotuning is appropriate there too.
- Example: `rope_embedding.py` — forward uses `exhaustive_search` + cache with split-buffer, backward uses `ct.launch` with same-buffer (Q_in=Q_out).

### Cross-Backend Config Transfer (Triton → CuTile)

Use `src/tilegym/autotune.py`: maps `BLOCK_SIZE_M/N/K` → `TILE_SIZE_M/N/K`; `num_warps`/`num_stages` have no CuTile equivalent.

### Optimizing an Existing Autotune Config

1. **Profile first**: Use NCU (set `DISABLE_AUTOTUNE=1`).
2. **Expand** (too narrow): add tile sizes, `num_ctas` (sm90+), `swap_ab`.
3. **Prune** (too slow): remove suboptimal configs, use arch-conditional yield, add size filters.
4. **Re-validate**: A/B test to confirm improvement.

## Pitfall Checklist

Before submitting code with autotune, verify these:

### Pitfall #1: In-Place Kernel Data Corruption

**Problem**: `exhaustive_search` runs the kernel multiple times to benchmark. If the kernel modifies input tensors in-place, the data is corrupted after the first trial run.

**Solution**: Split-buffer pattern — use separate read-only input and write-only output during search:

```python
# During exhaustive_search: use separate output buffer
Q_scratch = torch.empty_like(Q)
configs = list(_rope_autotune_configs())
result = exhaustive_search(
    configs, stream,
    grid_fn=...,
    kernel=rope_kernel,
    args_fn=lambda cfg: (Q, Q_scratch, ...),  # Q_in != Q_out
    hints_fn=...,
)

# After search: launch with in-place args using tuned config
cfg = result.best.config
tuned_kernel = rope_kernel.replace_hints(occupancy=cfg.occupancy)
ct.launch(stream, grid, tuned_kernel, (Q, Q, ...))  # Q_in == Q_out (in-place)
```

**Real example**: `rope_embedding.py` — Search uses split-buffer, final launch uses same-buffer.

**Also wrong**: Using `Q.clone()` in `args_fn` — this adds ~4us per clone, which is fatal for small kernels (~5us). The clone+copy pattern caused 0.48x performance in RoPE.

**Tip — isolating output buffers in `args_fn`**: For kernels that write to a dedicated output tensor (not in-place), use `c.clone()` inside `args_fn` to prevent trial runs from overwriting the final output buffer:

```python
# Output tensor c will be overwritten by each trial — clone it so trials don't
# corrupt the buffer the caller expects to use after exhaustive_search returns.
result = exhaustive_search(
    configs, stream,
    grid_fn=...,
    kernel=my_kernel,
    args_fn=lambda cfg: (a, b, c.clone()),  # each trial gets a fresh output
    hints_fn=...,
)
```

This is safe because the clone cost (~4us) is negligible relative to compute-bound kernel execution time (~50us+). Only avoid `clone()` for very small, memory-bound kernels where 4us is a significant fraction of runtime — in that case, pre-allocate a single scratch buffer outside `args_fn` (as in the split-buffer pattern above).

### Pitfall #2: Compilation Timeout

**Problem**: >30 configs causes compilation to exceed 5 minutes. CuTile compilation is heavier than Triton.

**Solution**:
- Keep total search space ≤ 30 configs — apply arch filters, tile size filters, and pruning rules until you're under the limit
- Use architecture-conditional yield to only generate relevant configs
- Prune the search space using architecture-conditional yield and size filters until total configs ≤ 30

**Real example**: Grouped GEMM expanded from 4 to 32 configs → all backward tests timed out. Reverted to occupancy-only (4 configs) with no performance loss.

### Pitfall #3: Cold-Cache Performance Skew

**Problem**: First process run is slower due to driver/JIT caches. Can cause wrong config selection.

**Solution**: Always warm up before measuring. `exhaustive_search` has built-in warmup, but first-process cold start is unavoidable. Re-run if you suspect the initial result was affected.

### Pitfall #4: NCU Profiling Interference

**Problem**: NCU profiles autotune trial runs, cluttering the trace.

**Solution**: Set `DISABLE_AUTOTUNE=1` before profiling, or use `ncu --launch-skip N`.

### Pitfall #5: search_space as Generator (Exhaustion)

**Problem**: `exhaustive_search` requires a `Sequence` (list/tuple), not a generator. Passing a generator directly will fail or produce unexpected results.

**Solution**: Always convert to list:
```python
# CORRECT: convert generator to list
configs = list(_matmul_autotune_configs())
result = exhaustive_search(configs, ...)

# WRONG: passing generator directly
result = exhaustive_search(_matmul_autotune_configs(), ...)
```

### Pitfall #6: FP8 Precision Loss

**Problem**: Hardware `/` breaks FP8 quantization bucket boundaries.

**Solution**: Use `ct.truediv(x, y, rounding_mode=RoundingMode.FULL)` for IEEE-compliant division in FP8 kernels. Never use `/` operator for FP8 scale computation.

### Pitfall #7: `replace_hints` on Hot Path (Recompilation)

**Problem**: `replace_hints()` returns a **new kernel object** with its own JIT cache (internally uses `dataclasses.replace()` which creates a fresh instance). Calling it on every kernel invocation — even with the same arguments — triggers recompilation every time. This is the most common autotune performance bug: `cutile_ms` jumps from ~0.04ms to 16–39ms (100–500× slower).

**Incorrect** (recompiles on every call):
```python
_cache[key] = result.best.config  # only stores config

cfg = _cache[key]
tuned = my_kernel.replace_hints(occupancy=cfg.occupancy)  # NEW kernel each time!
ct.launch(stream, grid, tuned, ...)
```

**Correct** (compile once, reuse forever):
```python
best_cfg = result.best.config
tuned = my_kernel.replace_hints(occupancy=best_cfg.occupancy)  # compile ONCE
_cache[key] = (best_cfg, tuned)  # cache both

cfg, tuned = _cache[key]
ct.launch(stream, grid, tuned, ...)  # reuse compiled kernel
```

**Rule**: Call `replace_hints` exactly once per config (immediately after `exhaustive_search`), cache the returned kernel object, and never call `replace_hints` again on the fast path.

## Scope and Boundaries

This skill covers *only* autotune configuration: search space design, `exhaustive_search` invocation, caching, and `ct.launch` with tuned hints. It does **not** modify kernel code.

**In scope** (autotune config):
- Search space generator functions
- `exhaustive_search()` calls and result handling
- `kernel.replace_hints()` for applying tuned hints
- Cache logic (key design, dict management)
- `ct.launch()` with tuned kernel
- `DISABLE_AUTOTUNE` fallback path

**Out of scope** (kernel code modifications — do NOT make these changes):
- Math flags (flush_to_zero, rounding_mode)
- Performance Hints (slice_hint, buffer_depth, copy_config)
- Memory access patterns (2D→1D gather/scatter conversion)
- Codegen optimizations (safe_offs → padding_value)
- Algorithm changes (K-loop split, load balancing)

## Further Optimization Suggestions

After adding autotuning, the following kernel-level optimizations may yield additional gains. These are *outside the scope of this skill* — mention them to the user as potential next steps, but do not implement them as part of autotuning:

- **Math flags**: `flush_to_zero=True` + `rounding_mode=APPROX` can provide 34-72% improvement for FMHA-class kernels (set via environment variables `TILEIR_ENABLE_FTZ=1 TILEIR_ENABLE_APPROX=1` or in kernel code). *Causal chain*: larger tiles initially *decrease* performance by 18-43% due to subnormal handling overhead; enabling FTZ+APPROX rescues this and flips the result to +34-72%. Math flags are therefore a *prerequisite* for large-tile configs to be effective on FMHA-class kernels.
- **Performance Hints**: `slice_hint`, `buffer_depth`, `copy_config` — requires modifying kernel IR code
- **Memory access patterns**: Using TMA loads (`ct.load`) instead of `ct.gather`; removing unnecessary bounds checks (`check_bounds=False` when safe)
- **Codegen quality**: Using `padding_value` parameter instead of manual `ct.where` masking; removing `safe_offs`
- **Algorithm restructuring**: K-loop split, load balancing, algebraic simplification

## Differences from Triton Autotune

Key differences: Triton uses `@triton.autotune` decorator with `Config(...)` objects; CuTile uses `exhaustive_search()` with `SimpleNamespace` configs + separate cache + `ct.launch`. CuTile has no `num_warps`/`num_stages` (compiler decides) — only tile sizes + `occupancy` + `num_ctas`. CuTile compilation is heavier (keep ≤30 configs). CuTile cache is user-managed in-memory (no automatic persistence). CuTile separates `args_fn` (kernel args) from `hints_fn` (compiler hints).

## Reference Documents

| Category | Document | Content |
|----------|----------|---------|
| **Parameter Design** | [`parameter-space-design.md`](references/parameter-space-design.md) | Per-kernel-type parameter spaces, cross-arch patterns, grid_fn patterns, pruning rules |
| **Search Strategies** | [`search-strategies.md`](references/search-strategies.md) | Exhaustive search, A/B test methodology, DISABLE_AUTOTUNE pattern |
| **Templates** | [`kernel-type-templates.md`](references/kernel-type-templates.md) | Copy-paste autotune templates for 8 kernel types |
| **Hardware** | [`hardware-constraints.md`](references/hardware-constraints.md) | Per-architecture constraints, tile size ranges, num_ctas rules, TMA requirements |

## Source Code References

Key files: `ops/cutile/matmul.py` (matmul autotune), `ops/cutile/attention.py` (FMHA autotune), `suites/unsloth/cutile/ct_ops.py` (shared `autotune_configs()` occupancy=[1,2,4,8]), `suites/unsloth/cutile/swiglu.py` (elementwise example), `suites/unsloth/cutile/rope_embedding.py` (split-buffer pattern), `suites/unsloth/cutile/grouped_gemm.py` (persistent GEMM, occupancy-only).

## Worked Examples

Each example shows the **before → after** pattern: `fixed_launch.py` (hardcoded `ct.launch`) and `autotuned_launch.py` (refactored to tune-once/cache/launch).

| Directory | Kernel | Autotune Pattern | Complexity | Key Teaching Point |
|-----------|--------|-----------------|------------|-------------------|
| [`assets/examples/01_rmsnorm_occupancy_only/`](assets/examples/01_rmsnorm_occupancy_only/) | RMSNorm (reduction) | Occupancy-only `[1,2,4,8]` | Low | Most common pattern — no tile tuning, just find best occupancy. Grid = `NUM_SM * cfg.occupancy`. Not in-place. |
| [`assets/examples/02_matmul_full_search/`](assets/examples/02_matmul_full_search/) | GEMM C=A@B | Full: `TILE_M/N/K` + `occupancy` + `num_ctas` (sm90+) | High | Compute-bound kernel with multiple tunable dimensions. `args_fn` passes tile sizes as `ct.Constant[int]`. `grid_fn` depends on `cfg`. ≤30 configs. |
| [`assets/examples/03_rope_inplace_splitbuffer/`](assets/examples/03_rope_inplace_splitbuffer/) | RoPE embedding (in-place) | Occupancy-only, with split-buffer | Medium | In-place kernel MUST use split-buffer during search to avoid corruption. Search writes to scratch; final `ct.launch` uses real in-place args. |
