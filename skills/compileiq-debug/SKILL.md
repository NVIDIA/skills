---
name: compileiq-debug
description: >
  Use when something is wrong: Search() hangs, all evaluations return
  INVALID_SCORE, scores aren't improving, every config returns the same
  number, ptxas errors fill the log, CV% is too high, or a winning ACF
  candidate needs NCU profiling to explain. Symptom-indexed table on top.
  Triggers on "compileiq hang", "socket timeout", "INVALID_SCORE",
  "not converging", "every score is the same", "TypeError fromhex",
  "ncu profile", "register spill", "ptxas error",
  "not in expected format", "high cv".
when_to_use: |
  - User reports any unexpected behavior from Search() or its results.
  - User wants to NCU-profile a winning ACF to understand WHY it helps.
  - High CV% on validation runs that won't drop with longer warmup.
  Don't use when:
  - It's actually an environment problem (use compileiq-bootstrap first).
  - The search ran fine and the user just wants to validate (use
    compileiq-validate-result).
license: CC-BY-4.0 AND Apache-2.0
metadata:
  version: "1.2.0"
  author: CompileIQ Team <compileiq@nvidia.com>
  domain: compiler-optimization
  tags: [compileiq, cuda, ptxas, compiler-optimization, autotuning]
allowed-tools: Bash Read
paths: ["**/*.py", "**/*.csv", "**/*.ptx", "**/*.acf"]
---

# compileiq-debug

A symptom-indexed cheat sheet. Find the row that matches what the user is
seeing, follow the first action, then dig into the matching detail section.

## Symptom table

| Symptom | Most-likely cause | First action |
|---|---|---|
| Search hangs on first eval | Release fetch slow/blocked; OR `forkserver` issue | `CIQ_SEARCH_SPACES_DIR=<local mirror>`; if still hangs, `CIQ_PROCESS_MODE=spawn`. |
| `Engine 'amber' is not registered` | Frontend-only install (source checkout without the native runtime) | `pip install compileiq` for a complete wheel; from a source checkout, `git lfs pull && make install` |
| `FutureWarning: SearchConfiguration settings that have no effect since CompileIQ 1.1 are ignored` | `SearchConfiguration(cull_size=..., mutate_rate=..., objective_weights=..., enable_large_fail_pool=False)` | The setting has no effect. Remove it, or pass the named amber option through `EngineConfig(name="amber", options={...})` |
| `SearchSpaceExhaustedWarning` (search ends early) or `Unable to generate N unique candidates` from `ask()` | `amber` never repeats a candidate; the space is small or the search converged | Lower `pool_size` or `generations`, or widen the search space |
| Every eval returns `INVALID_SCORE` | Return-type mismatch (tuple vs scalar) / correctness gate / `task_timeout` too tight | `Search.sample(1)` + call objective by hand; check `num_objectives` vs return shape; raise `task_timeout` |
| Every eval returns the **same** score | ACF not reaching the compiler — framework cache is hiding it | Apply Debug-pack `O0` ACF by hand; if the score doesn't regress, fix cache-bust (`TRITON_ALWAYS_COMPILE=1`, `HELION_SKIP_CACHE=1`, fresh `TRITON_CACHE_DIR`, drop FlashInfer cubin packages) |
| `TypeError: fromhex() … not dict` | Objective has no baseline branch: with `normalize=True` the first call passes `BASELINE_CONFIG` (an empty dict) | Add the baseline-knockout branch (run without an ACF when `config` is an empty dict) and write ACFs with `save_compiler_config(acf_path, config)`; see `compileiq-author-objective` |
| `"not in expected format"` | Objective returned wrong shape | `num_objectives` must equal `len(return_tuple)`; scalar return only when `num_objectives=1` |
| Convergence stalled (best score flat) | Pool too small for space; mutation too low; or kernel near-optimal | Raise `pool_size`; raise amber's `mutation_probability` option; sample diversity with `Search.sample(20)` |
| Increasing invalid rate over generations | Mutation arm spreading; compiler version drift mid-run | `CIQ_KEEP_CACHE=1`, re-run, inspect failing configs offline |
| CV% > 10% on validation | Unlocked clocks / thermal throttling / GPU contention | Lock GPU + memory clocks; pin `CUDA_VISIBLE_DEVICES`; watch `nvidia-smi dmon` for thermals |
| Need to know *why* a winning ACF helps | Profile with NCU | See **NCU section** below |

## Details

### Hang on first eval

> Not BLAS. Do not send users to symlink `libblas.so`.

Current shipped binaries link only `libm`/`libc`/`libstdc++`/`libgcc_s` and
do not require BLAS/LAPACK.

Real causes today, in order of frequency:

1. **Release-backed search-space fetch is slow or blocked.** First call to
   `PtxasSearchSpace().retrieve()` downloads from `github.com`. On a corporate
   firewall this can stall. Pre-stage the mirror from a host that can reach
   GitHub (tags are dated, `search-spaces-YYYY.MM.DD`; there is no literal
   `latest` tag), then point CompileIQ at it:
   ```bash
   SS_TAG="$(gh release list -R NVIDIA/CompileIQ --limit 100 --json tagName,isDraft,isPrerelease \
     --jq '.[] | select(.isDraft == false and .isPrerelease == false) | select(.tagName | startswith("search-spaces-")) | .tagName' \
     | head -n 1)"
   gh release download "$SS_TAG" -R NVIDIA/CompileIQ -D /shared/mirror
   export CIQ_SEARCH_SPACES_DIR=/shared/mirror
   ```
   See `compileiq-search-space` for the full air-gapped setup.
2. **`forkserver` is unsupported on the host.** Set `CIQ_PROCESS_MODE=spawn`.
   `IsoMultiProcessWorker` already uses `fork` by default.

### Every eval returns INVALID_SCORE

Sanity-check the shape before assuming the worst:

```python
sample = tuner.sample(1)[0]
score = objective(sample)
print(type(score), score)
```

Common shape mismatches:

- `num_objectives=1` but `objective` returns a tuple `(latency,)`. Drop the
  trailing comma.
- `num_objectives=2` but `objective` returns a scalar.
- Correctness gate is rejecting everything because the reference call itself
  is wrong.
- `task_timeout` is shorter than a clean compile takes; raise it.

### Every eval returns the SAME score

Almost always a framework cache serving a stale binary. **Run the O0/O3 canary
from the Debug pack** to confirm — see `compileiq-booster-pack` for the exact
test. If O0 doesn't regress vs baseline, the ACF is not reaching PTXAS. Fix:

| Framework | Cache-bust |
|---|---|
| Triton | `TRITON_ALWAYS_COMPILE=1` + unique `TRITON_CACHE_DIR` per eval |
| Helion | `HELION_SKIP_CACHE=1` |
| FlashInfer | Confirm `flashinfer_cubin` and `flashinfer_jit_cache` packages are absent ([`docs/flashinfer_booster.md`](https://github.com/NVIDIA/CompileIQ/blob/main/docs/flashinfer_booster.md)) |
| Raw nvcc | Clean the build dir between candidates |

### TypeError around fromhex

`TypeError: fromhex() argument must be str, not dict` means the objective
was called with `BASELINE_CONFIG`, an empty dict. With `normalize=True`,
CompileIQ's first task is a baseline run that passes `BASELINE_CONFIG`
instead of a hex string (`compileiq/worker.py`), and anything that calls
`bytes.fromhex` on it fails, including `save_compiler_config`. Swapping in
`save_compiler_config` alone does not fix it.

Often the objective also uses the legacy pattern from the pre-2026 skill set:

```python
# OLD — DO NOT USE
def objective(config_blob):
    with open(tmp_path, "wb") as f:
        f.write(bytes.fromhex(config_blob))
    ...
```

Fix both at once: handle the baseline first, then write the ACF with
`save_compiler_config` (it does the `bytes.fromhex` internally):

```python
from compileiq.types import BASELINE_CONFIG
from compileiq.utils.helpers import save_compiler_config

def objective(config):
    if config == BASELINE_CONFIG:        # empty dict: baseline run
        return measure_without_acf()
    save_compiler_config(tmp_path, config)
    ...
```

See `compileiq-author-objective` for the full pattern.

### "Not in expected format"

The objective returned a shape CompileIQ doesn't expect. Rules:

- `num_objectives=1`: objective must return a single scalar (`int` | `float`).
  Not a 1-tuple, not a list.
- `num_objectives>=2`: objective must return a tuple or list of that length.

```python
result = objective(sample)
assert (
    (search_config.num_objectives == 1 and isinstance(result, (int, float)))
    or (search_config.num_objectives  > 1 and len(result) == search_config.num_objectives)
), f"shape mismatch: {result!r} vs num_objectives={search_config.num_objectives}"
```

### Convergence stalled

Three causes, in order:

1. **Pool too small for the space.** `pool_size = max(2 * num_objectives + 1, 32)` is the auto-derived floor — for spaces with >1k design points, raise to 64-128.
2. **Mutation too low.** Pass a higher amber `mutation_probability`, for example `engine=EngineConfig(name="amber", options={"mutation_probability": 0.3})`, if the search is converging on the first generation. (`mutate_rate` has no effect since 1.1.)
3. **The kernel is genuinely near-optimal.** Verify by running `Search.sample(20)` and timing each sample by hand — if the spread is <5%, the search space is shallow.

### Increasing invalid rate

Probably a mutation arm spreading a structurally-bad config across the
population. Re-run with `CIQ_KEEP_CACHE=1` so the failing configs are
preserved at `~/.cache/compileiq/`, then replay them by hand to identify the
common factor.

### High CV% on validation

> If `cv = std/mean > 10%`, validation can't tell the signal from the noise.

Fixes, in order:

1. Lock GPU and memory clocks (see `compileiq-run-search` for the
   `nvidia-smi --lock-*-clocks` snippet).
2. Pin `CUDA_VISIBLE_DEVICES=<gpu>` so the validation has the GPU to itself.
3. Watch `nvidia-smi dmon -i <gpu> -s pucvm` for thermal throttling events.
4. Raise warmup count (50 → 100) and trial count (100 → 200).
5. Switch from `cudaEvent` to NVBench (entropy-based stopping criterion, cold-cache between samples).

## CIQ_KEEP_CACHE for post-mortem

When a search misbehaves, re-run with:

```bash
export CIQ_KEEP_CACHE=1
```

The cache at `~/.cache/compileiq/` is preserved after the run. You can:

- Inspect each generation's serialized configs.
- Replay a specific config without paying for a whole new search.
- Compare two runs by diffing their cache directories.

## Diagnose from the dump_results CSV

Quick pandas snippet:

```python
import pandas as pd
df = pd.read_csv("results.csv")
df["score_numeric"] = pd.to_numeric(df["score_1"], errors="coerce")

gen_summary = df.groupby("generation").agg(
    n=("score_numeric", "size"),
    invalid=("score_numeric", lambda s: s.isna().sum() + (s > 1e10).sum()),
    best=("score_numeric", "min"),
)
print(gen_summary)
```

If `invalid` doesn't decrease across generations, your search is structurally
broken — try the O0/O3 canary in `compileiq-author-objective`.

For an automated version: `python scripts/diagnose_csv.py results.csv`.

## NCU section (replaces old /compileiq-profile)

Profile **only after a validated ACF candidate exists**. Don't profile every
config — it's slow.

### Two-shot pattern

```bash
# Baseline (no ACF)
ncu --set full -o baseline -f --kernel-name my_kernel python bench.py

# ACF-applied — match the injection your objective uses
# Triton 3.6+ (reads PTXAS_OPTIONS at startup; otherwise pass ptx_options= in bench.py):
ncu --set full -o opt -f --kernel-name my_kernel \
    env PTXAS_OPTIONS="--apply-controls=best.acf" TRITON_ALWAYS_COMPILE=1 python bench.py
# NVCC build for sm_100+ (for sm_90 and older, nvcc rejects --apply-controls;
# use -Xptxas --apply-controls=best.acf instead):
nvcc --apply-controls best.acf -arch=sm_100 bench.cu -o bench && \
    ncu --set full -o opt -f --kernel-name my_kernel ./bench

# Diff
ncu --import baseline.ncu-rep --import opt.ncu-rep --csv --page raw > diff.csv
```

### Five metrics that explain most CompileIQ wins

| Metric | What it means |
|---|---|
| `sm__throughput.avg.pct_of_peak_sustained_elapsed` | Compute throughput |
| `gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed` | Memory throughput |
| `sm__warps_active.avg.pct_of_peak_sustained_active` | Achieved occupancy |
| `launch__registers_per_thread` | Register pressure |
| `l2__throughput.avg.pct_of_peak_sustained_elapsed` | L2 pressure |

If the ACF moved any of these meaningfully, that's the mechanism. If none of
them moved but the win is real, look at lower-level metrics (warp stalls,
issue slot utilization) — those are harder to interpret but often the answer.

### Register-spill check (no NCU needed)

```bash
ptxas -v -arch=sm_100 --apply-controls best.acf kernel.ptx 2>&1 \
    | grep -E "registers|spill|stack"
```

Reports `Used N registers, X bytes stack frame, Y bytes spill stores,
Z bytes spill loads`. If `Y + Z` goes **up** vs baseline, the ACF traded
register pressure for memory traffic — sometimes a real win, sometimes not.
Investigate before shipping.

## Self-test

```bash
python scripts/diagnose_csv.py --self-test
```

Synthesizes a small `results.csv` covering each pathology (clean convergence,
rising invalid rate, stalled best-score) and asserts the heuristic
classifications match.

## Explicitly dropped from this skill (vs old /compileiq-debug + /compileiq-profile)

- **BLAS symlink section** — obsolete; current binaries don't link BLAS
  (verified by `ldd`). Carrying it forward sends users on a wild goose chase.
- **Verbose `COMMON_PTXAS_ERRORS` dict** mapping individual ptxas error
  strings to fixes — users get `INVALID_SCORE` instead, no need to recognize
  specific messages.
- **Duplicate CUDA 13.3+ check** — lives in `compileiq-bootstrap`.
- **`validate_objective_function` introspection helper** — replaced by
  `Search.sample(1)` + the Debug-pack O0/O3 canary.

## Next

- For the canary itself: `compileiq-booster-pack` or
  `compileiq-author-objective`.
- For environment issues: `compileiq-bootstrap`.
- After fixing: re-run `compileiq-run-search`.
