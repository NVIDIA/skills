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
2. **Design search space** — select the matching template from `references/kernel-type-templates.md`; prune to ≤ 30 configs in the final code via arch filters (directed exploration probes may temporarily exceed this — see Design Philosophy)
3. **Implement** — add `exhaustive_search` + cache + `ct.launch` following the Step-by-Step Workflow; handle in-place writes with split-buffer if needed
4. **Test** — run correctness with autotune enabled and with `DISABLE_AUTOTUNE=1`
5. **Validate** — A/B benchmark against fixed best-known config; see `references/search-strategies.md`
6. **Shrink** — prune dead-weight configs that never win, targeting ≤ 8 configs per architecture to minimize compilation cost (Step 10)

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
- Keep ≤ 30 configs in final code (see Design Philosophy for temporary directed probes)
- `exhaustive_search` requires a `Sequence` (list/tuple) — convert generators with `list()`
- **Search space must include the original fixed config** — this guarantees autotuning never makes performance worse

**When to use this pattern**: Kernel has fixed block size (not tile-size tunable). Includes: elementwise (SwiGLU, GeGLU), reduction (RMSNorm, LayerNorm), RoPE, and persistent kernels with heuristic block sizes (grouped GEMM).

For complex kernels (matmul with tile sizes, FMHA, FP8 with num_ctas), read the full guide below + [`kernel-type-templates.md`](references/kernel-type-templates.md).

> **⚠️ Three pitfalls catch almost everyone — check before submitting:**
> - **`replace_hints` on hot path?** → Cache BOTH config AND kernel object from `exhaustive_search`. Calling `replace_hints()` every invocation recompiles (100–500× slower) → Pitfall #7
> - **In-place kernel** (writes back to input tensor)? → MUST use split-buffer pattern during search → Pitfall #1
> - **Search space empty?** → Check arch filters and `num_ctas` constraints → Pitfall #5

> **Minimum coverage**: On sm100+, FMHA/matmul/varlen search spaces must include both `num_ctas=1` and `num_ctas=2`. For core dimensions (tile sizes, occupancy), keep at least 2 distinct values even if unsure which is better — let `exhaustive_search` decide.

> **When to stop tuning**: A mean speedup in [0.98, 1.02] means your *current* search space isn't helping — but doesn't mean no config will help. Before stopping, check whether you've covered the key dimensions for this kernel type (consult `references/kernel-type-templates.md`). If the search space already covers the template's recommended dimensions and the best result is still noise-floor, then stop — further micro-adjustments won't help. If key dimensions are missing (e.g., never tried `num_ctas=2` for a dual-GEMM kernel), expand the search space rather than giving up.
>
> Once correctness tests pass and the autotuned kernel shows speedup over the fixed-config baseline, **stop — do not re-run to "confirm".** GPU kernel timing fluctuates ±5–10 % between invocations due to clock scaling and OS scheduling; a subsequent timing dip does not mean your code is wrong.
>
> To improve speedup, only modify the autotune search space (configs, tile sizes, occupancy, num_ctas). Do not modify other code (Python wrapper, stream management, etc.) to chase speedup — kernel performance is determined by the config selection, not by host-side code.

## Reading Guide

- **Occupancy-only kernels** (elementwise, reduction, persistent with fixed block sizes): Quick Reference + Pitfall Checklist is sufficient — skip `references/` docs. For in-place kernels, also read Pitfall #1.
- **Complex kernels** (matmul with tunable tile sizes, FMHA, FP8 with num_ctas): Quick Reference → Decision Tree → API Reference → Step-by-Step Workflow → relevant `references/` docs.

**5-step summary**: Classify kernel → Design search space ([`parameter-space-design.md`](references/parameter-space-design.md)) → Implement using template ([`kernel-type-templates.md`](references/kernel-type-templates.md)) → Validate with A/B test → Check Pitfall Checklist.

**Reading references**: Read only the reference relevant to your kernel type — e.g., for FMHA, read the Template 5 section in `references/kernel-type-templates.md`; for hardware constraints, read only the target architecture's section. Avoid reading all references end-to-end when a targeted lookup suffices.

## Design Philosophy

**Build a small, precise search space bottom-up — not a large space trimmed down.** CuTile compilation is much heavier than Triton (~0.5-1s per config), so the **final code** should contain ≤ 30 configs. The approach is: classify the kernel type first, then construct only the relevant configs for that type and architecture.

**Directed exploration during development**: If the initial template configs yield speedup < 1.0, you may run a *temporary* larger probe (30–100 configs) via `bash + python3 -c` to identify which dimensions matter — but this probe must be **directional**, not a blind cartesian product. Use the kernel type classification to decide *which* dimensions to vary (e.g. for dual-GEMM, probe `num_ctas × occupancy` while fixing tile sizes; for FMHA, probe `TILE_M × num_ctas` while fixing TILE_N). Once the probe identifies the winning region, lock the final code's search space to ≤ 8 top candidates. Do NOT write the large probe into the source file — it is a one-shot diagnostic tool.

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
   - **When modifying code that already has autotune configs**: see "Handling Existing Autotune Configs (Multi-Architecture)" below. The "do NOT add branches" rule means do not *invent new configs* for untested architectures — it does NOT mean remove existing configs that were previously validated.
   - Identify tunable parameters (tile sizes, occupancy, num_ctas)
   - **Ensure the search space includes the original fixed config** (or an equivalent). This guarantees that the autotuned result is at least as good as the original — no performance regression is possible.
   - If the generated set exceeds 30, apply tile size filters and pruning rules to reduce it to ≤ 30 in the final code
   - *VERIFY*: Total configs in final code ≤ 30 (CuTile compilation is heavy, >30 configs will timeout). Temporary directed probes during development (30–100 configs, run via `bash + python3 -c`) are allowed — see Design Philosophy.

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

7. **(Optional) Add DISABLE_AUTOTUNE support** for CI and profiling: check `os.environ.get("DISABLE_AUTOTUNE", "0") == "1"` — when set, skip `exhaustive_search` entirely and fall back to `ct.launch` with the first valid config. Useful for:
   - CI determinism (autotune adds variable wall time)
   - NCU profiling (prevents autotune trial runs from cluttering the trace — see Pitfall #4)
   - Debugging (isolates kernel correctness from autotune behavior)
   Skip this step if your task only requires adding autotuning and the project's tests don't check for `DISABLE_AUTOTUNE`.

8. **Test**: Run correctness tests first (`pytest -k "test_op and cutile"`), then benchmark.
   - *VERIFY*: Correctness passes with autotune enabled AND with `DISABLE_AUTOTUNE=1`.

9. **Validate with A/B test**: Compare autotune version vs fixed best-known config. See [`search-strategies.md`](references/search-strategies.md) for methodology.
   - *VERIFY*: Autotune version ≥ baseline (or within noise). If worse, check that the search space includes the original fixed config, and that `replace_hints` is being used correctly.

10. **Shrink the search space** — reduce compilation cost without losing performance.

    Templates provide broad search spaces as a starting point (e.g., 9 configs for varlen attention). Not all configs contribute to finding the optimal one — on a given architecture and kernel shape, many large-tile or multi-CTA configs compile for seconds each but are never selected. The goal of this step is to *prune the dead weight* so the final committed code has 5–8 configs per architecture instead of 10–15.

    **Why this matters**: Each config in `exhaustive_search` requires a full JIT compilation + warmup + benchmark of the kernel. For complex kernels (FMHA, varlen attention), this costs 2–4 seconds *per config*. Cutting from 9 to 5 configs saves 8–16 seconds of one-time autotuning cost per unique shape, with zero performance loss.

    **Procedure**:

    1. After Step 9 passes, you already have a working autotuned kernel with the full template search space. Now run the test on 2–3 representative shapes and observe which config wins for each shape. You can inspect this by temporarily adding a print inside the cache-miss block:
       ```python
       print(f"[autotune] shape={cache_key[:5]} best={result.best.config} "
             f"time={result.best.time_ms:.3f}ms  "
             f"configs_tried={len(result.successes)}")
       ```

    2. Identify which configs are *competitive* — within 5% of the best for at least one shape. Configs that are never within 5% of the best across any test shape are *dead weight*.

    3. Remove dead-weight configs from the generator. Always keep:
       - The original fixed config (safety net — guarantees no regression)
       - The config(s) that won on each test shape
       - Any config within 5% of a winner (may win on untested shapes)

    4. Re-run the test to confirm speedup is unchanged after pruning.

    **Common dead-weight patterns** (prune these first):
    - `TILE_M=256` configs for attention/varlen kernels where `S_qo` in the test shapes is ≤ 4096 and batch×heads is large — the grid is already saturated at TILE_M=128.
    - `num_ctas=2` configs for kernels with irregular or small grids — multi-CTA parallelism requires enough CTAs to benefit from cooperative launch, which doesn't hold when `grid[0]` is small.
    - `occupancy=4` or `occupancy=8` configs on sm100+ for compute-bound kernels — Blackwell typically prefers lower occupancy (1–2) with larger tiles.

    **Target**: ≤ 8 configs per architecture branch in the final code. This keeps the one-time tuning cost under 25 seconds even for the most complex kernels (FMHA, varlen attention).

    - *VERIFY*: Config count ≤ 8 per architecture. `speedup_over_fixed` unchanged after pruning.

11. **(MANDATORY) Verify correctness and performance before finalizing.**

    The verification requirements depend on the task type. In ALL cases, start with the code-level sanity check, then apply the task-specific verification.

    ---

    **A. Code-level sanity check (ALL tasks — do this first)**

    Review your implementation for known performance anti-patterns. These checks catch *implementation bugs*, not algorithmic issues — they apply regardless of whether you are adding, modifying, or fixing autotune code.

    - `replace_hints` must be called *exactly once* per config and the returned kernel object cached (Pitfall #7). If `replace_hints` appears on the hot path (outside the `if cache_key not in` block), you have a recompilation bug that causes 100-500× slowdown.
    - `exhaustive_search` must be inside the cache-miss block, not called on every kernel invocation.
    - The fast path should only do: cache lookup → `ct.launch` with the cached tuned kernel. No JIT-triggering calls in between.
    - The cache must store `(best_cfg, tuned_kernel)` together — not just `best_cfg` alone.

    ---

    **B. Task-specific verification**

    **B1. Adding or modifying autotune configs** (the original code is correct):

    - *Correctness*: autotuned kernel output matches the reference (e.g. `torch` or fixed-config kernel) within tolerance.
    - *Performance*: autotuned kernel must be *at least as fast* as the original fixed-config kernel. If it is slower:
      - Check that the search space includes the original fixed config (this guarantees no regression).
      - Check if `replace_hints` is being called on every code path — revisit Step 2 (if any path skips `replace_hints`, the decorator's fixed hints are used instead of autotuned values).
      - Expand search space if all configs perform similarly (see `references/parameter-space-design.md` → "Adapting Search Space").

    **B2. Fixing a correctness bug** (the original code produces wrong results):

    - *Correctness is the primary goal*: the fixed kernel must produce correct results. Do NOT compare speedup against the broken original — a correct-but-slower kernel is always better than a fast-but-wrong one.
    - *Perf sanity check*: after fixing, verify that the implementation is not catastrophically slow due to an implementation bug (e.g. Pitfall #7). Two ways to check:
      1. *Code review*: confirm the code-level sanity check (Section A above) passes — this catches the most common perf bugs.
      2. *Runtime check*: if possible, compare your fixed+autotuned kernel against a simple correct baseline (e.g. the equivalent `torch` operation, or the kernel launched with a single hardcoded config and no autotuning). Your autotuned version should not be slower than this naive baseline. Minor overhead from the fix itself (e.g. split-buffer allocation) is acceptable.

    ---

    *⚠️ Autotuning bugs (silent hint override, split-buffer omission, hot-path recompilation) are only caught at runtime — always verify by running the kernel, not just by reading the code.*

### Handling Existing Autotune Configs (Multi-Architecture)

When adding autotune to a kernel, the source code may already contain autotune configs from a previous pass on different hardware. There are three scenarios:

**Scenario 1: No existing autotune code.** The source has no autotune at all — follow the standard "Adding Autotune to a New Kernel" workflow above. Generate configs for the current GPU architecture only.

**Scenario 2: Existing autotune, but no config for the current architecture.** The source already has autotune with configs for other architecture(s) (e.g., sm103) but NOT for the current GPU (e.g., sm100). Steps:

1. Detect the current architecture with `torch.cuda.get_device_capability()`.
2. Check whether the existing config generator already uses architecture-conditional branching (i.e., `if/elif` on device capability).
   - **If yes** (conditional yield structure exists): Add a new `elif` branch for the current architecture. Preserve all existing branches **unchanged** — do not modify their config values.
   - **If no** (flat configs, no architecture branching): Add an `if` branch for the current architecture with new configs, and keep the existing flat configs in the `else` block as the default fallback. This ensures that all other architectures continue to use the original configs unchanged — the code modification must not alter kernel behavior on any architecture other than the current one.
3. Design configs for the current architecture following the standard workflow (Steps 4–10 above).
4. Validate only the current architecture's configs (Step 11). Other branches are assumed correct since they were previously validated on their respective hardware.

Example — adding sm100 to a generator that already has sm103 configs (conditional structure exists):

```python
def _my_autotune_configs():
    gpu_capability = torch.cuda.get_device_capability()

    if gpu_capability == (10, 0):                   # sm100 (B200)
        # NEW: configs for sm100 (added in this pass)
        for occ in [1, 2, 4]:
            yield SimpleNamespace(occupancy=occ, TILE_M=128, TILE_N=128)
    elif gpu_capability == (10, 3):                  # sm103 (GB300)
        # EXISTING: configs for sm103 (do NOT modify)
        for occ in [2, 4, 8]:
            yield SimpleNamespace(occupancy=occ, TILE_M=256, TILE_N=128)
    else:
        # Fallback for unknown architectures
        yield SimpleNamespace(occupancy=2, TILE_M=128, TILE_N=128)
```

Example — adding current-arch configs to flat (non-branching) code:

```python
# BEFORE: flat configs (no architecture branching)
def _my_autotune_configs():
    for occ in [2, 4, 8]:
        yield SimpleNamespace(occupancy=occ, TILE_M=256, TILE_N=128)

# AFTER: if-branch for current arch, original configs become the else-default
def _my_autotune_configs():
    gpu_capability = torch.cuda.get_device_capability()

    if gpu_capability == (10, 0):                    # sm100 (B200) — current arch
        # NEW: configs designed and tested for sm100
        for occ in [1, 2, 4]:
            yield SimpleNamespace(occupancy=occ, TILE_M=128, TILE_N=128)
    else:
        # UNCHANGED: original flat configs as default for all other architectures
        for occ in [2, 4, 8]:
            yield SimpleNamespace(occupancy=occ, TILE_M=256, TILE_N=128)
```

**Scenario 3: Existing autotune with config for the current architecture.** The source already has a conditional branch for the current GPU architecture. Only modify the current architecture's branch (e.g., adjust tile sizes, add/remove occupancy values). Do **NOT** modify or remove configs for other architectures.

**Key principles:**

- **"Target one architecture at a time" means only *add or modify* configs for the detected arch** — it does NOT mean delete existing configs for other architectures. Existing configs were validated on their respective hardware and must be preserved.
- **When adding architecture branching to flat configs**: add an `if` for the current architecture and keep existing configs in the `else` as the default. This guarantees that the code change does not alter kernel behavior on any non-current architecture — the `else` path is identical to the original flat code.
- **Test/validation (Step 11) only applies to the current architecture's branch.** Other branches are assumed correct since they were previously validated on their respective hardware. You cannot test them here because you don't have access to that hardware.

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

**Tip — isolating output buffers in `args_fn`**: For kernels that write to a dedicated output tensor (not in-place), you *may* use `c.clone()` inside `args_fn` to prevent trial runs from overwriting the final output buffer. This is only needed when the caller reads the output tensor after `exhaustive_search` returns — if you immediately overwrite it with `ct.launch`, clone is unnecessary:

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

**Problem**: >30 configs in the **final code** causes compilation to exceed 5 minutes. CuTile compilation is heavier than Triton.

**Solution**:
- Keep the final code's search space ≤ 30 configs — apply arch filters, tile size filters, and pruning rules until you're under the limit
- Use architecture-conditional yield to only generate relevant configs
- If the initial template configs don't beat baseline, use a temporary directed probe (30–100 configs, via bash, not written to file) to identify winning dimensions, then lock the final code to ≤ 8 top candidates (see Design Philosophy)

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

Key differences: Triton uses `@triton.autotune` decorator with `Config(...)` objects; CuTile uses `exhaustive_search()` with `SimpleNamespace` configs + separate cache + `ct.launch`. CuTile has no `num_warps`/`num_stages` (compiler decides) — only tile sizes + `occupancy` + `num_ctas`. CuTile compilation is heavier (keep ≤30 configs in final code). CuTile cache is user-managed in-memory (no automatic persistence). CuTile separates `args_fn` (kernel args) from `hints_fn` (compiler hints).

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
