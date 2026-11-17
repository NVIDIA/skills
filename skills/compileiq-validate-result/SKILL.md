---
name: compileiq-validate-result
description: >
  Use AFTER a Search has completed and BEFORE claiming any speedup or
  shipping an ACF. Loads the dump_results CSV, extracts top-K candidates
  (single-objective) or the Pareto front (multi-objective), re-measures each
  against the no-ACF baseline with 100+ trials on fresh caches, runs Welch's
  t-test plus Cohen's d, rejects three classic false-positive patterns
  (lucky-min / higher-variance / multiple-comparisons-of-N), and saves the
  validated winner as best.acf. Triggers on "validate result", "extract best
  config", "Welch's t-test", "is my speedup real", "save best ACF", "pareto
  front", "claim speedup", "ship config".
when_to_use: |
  - tuner.start() returned and there's a results CSV.
  - User wants to ship an ACF to production.
  - Reported speedup feels too good to be true.
  Don't use when:
  - Search hasn't completed (and user hasn't decided to stop early).
  - User just wants the raw best row — that's one line of pandas.
license: CC-BY-4.0 AND Apache-2.0
metadata:
  version: "1.2.0"
  author: CompileIQ Team <compileiq@nvidia.com>
  domain: compiler-optimization
  tags: [compileiq, cuda, ptxas, compiler-optimization, autotuning]
allowed-tools: Bash Read Write
permissions: [shell, file_write, env]
paths: ["**/*.csv", "**/*.acf", "**/*.py"]
---

# compileiq-validate-result

The score CompileIQ reports during a search uses N=5-15 trials per evaluation
and a shared cache. That's appropriate for the search loop but **wildly
insufficient for shipping**. This skill is the gate before any ACF goes to
production.

## When

- `tuner.start()` has returned and there's a `dump_results=` CSV on disk.
- User wants to claim a speedup or ship an ACF.
- A reported speedup feels too clean — validate it.

## Steps

### 1. Load the CSV

```python
from compileiq.results import SearchResult

results = SearchResult.from_csv("results.csv", problem_type="min", clear_duplicates=True)
df = results.get_results()
print(f"{len(df)} evaluations across {df['generation'].max()+1} generations")
```

### 2. Extract candidates

**Single-objective:**
```python
best = results.get_best_result()
# dict: {metadata, generation, score_1, params, [norm_score_1]}
score = best.get("score_1", best.get("score"))   # legacy defensiveness
acf_hex = best["params"]                          # hex string
```

`score_1` (with underscore-one) is the canonical key — it matches the
multi-objective convention `score_N`. Older code sometimes uses plain `score`;
the fallback above handles both shapes.

For top-K:
```python
import pandas as pd
# Invalid evaluations are stored as "*", so convert before ranking.
scores = pd.to_numeric(df["score_1"], errors="coerce")
df_valid = df.assign(score_1=scores)[scores < 1e10]
top_k = df_valid.nsmallest(5, "score_1")          # nlargest for MAX problems
```

**Multi-objective:**
```python
front = results.pareto_front()   # raises if num_objectives == 1
for candidate in front:
    print(candidate["score_1"], candidate["score_2"], candidate["params"])
```

**Mixed user+compiler search space:** results carry separate keys —
`best["user_space"]` for the user-side knobs, `best["params"]` for the ACF
hex. Save both.

### 3. Re-measure on fresh cache (the actual validation)

| Stage | Warmup | Trials | Cache | GPU clocks |
|---|---|---|---|---|
| Optimization (during `tuner.start()`) | 5-25 | 5-15 | per-eval | recommended locked |
| **Validation** | **≥50** | **≥100** | per-measurement | **must be locked** |

Both the baseline (no ACF) and each top-K candidate are re-measured at
validation N. The optimization-time measurement is too noisy to ship from.

### 4. Statistical gate — the ship rule

```python
import numpy as np
from scipy import stats

def validate_speedup(baseline_ms: np.ndarray, optimized_ms: np.ndarray) -> dict:
    t, p = stats.ttest_ind(baseline_ms, optimized_ms, equal_var=False)   # Welch's
    b_mean, b_std = baseline_ms.mean(),  baseline_ms.std(ddof=1)
    o_mean, o_std = optimized_ms.mean(), optimized_ms.std(ddof=1)
    pooled = np.sqrt((b_std**2 + o_std**2) / 2)
    d = (b_mean - o_mean) / pooled if pooled > 0 else 0.0
    return {
        "speedup_mean":  b_mean / o_mean,
        "speedup_median": np.median(baseline_ms) / np.median(optimized_ms),
        "p_value":       float(p),
        "cohens_d":      float(d),
        "significant":   bool(p < 0.05 and o_mean < b_mean and d > 0.2),
        "baseline":  {"mean": b_mean, "std": b_std,
                      "p5": np.percentile(baseline_ms,  5),
                      "p95": np.percentile(baseline_ms, 95)},
        "optimized": {"mean": o_mean, "std": o_std,
                      "p5": np.percentile(optimized_ms,  5),
                      "p95": np.percentile(optimized_ms, 95)},
    }
```

**Ship rule:** `p_value < 0.05` AND `cohens_d > 0.2` (preferably `> 0.5`)
AND `optimized.mean < baseline.mean`. Anything weaker, **do not claim a
speedup**.

### 5. Three false-positive patterns to actively check

| # | Pattern | Symptom | Cause | Check | Disposition |
|---|---|---|---|---|---|
| 1 | **Lucky-min** | Optimized `min` is lower but `mean` is equal or worse | Optimizer picked a config that occasionally runs fast | Compare **means**, not minimums; reject if `optimized.mean ≥ baseline.mean` | Reject. |
| 2 | **Higher-variance** | Optimized `p5-p95` range is wider than baseline with same mean | ACF didn't speed anything up; just spread the distribution | Compute `(p95 - p5)` for both; reject if optimized range is materially wider (>25%) | Reject. |
| 3 | **Multiple-comparisons** | Best of 500 evaluations looks 2-5% faster but doesn't reproduce | With 500 evals some will look good by chance | Re-measure top-K on a *fresh* cache and *fresh* trials; reject candidates that don't survive | Reject. |

### 6. Save the validated winner

```python
from compileiq.utils.helpers import save_compiler_config
save_compiler_config("best.acf", best["params"])

# Mixed search spaces: persist the user_space knobs separately
if "user_space" in best:
    import json
    Path("best.user_space.json").write_text(json.dumps(best["user_space"], indent=2))
```

### 7. Reproducibility log

Append one row per candidate decision to `validation-log.csv`. Fields, per
[`docs/flashinfer_booster.md`](https://github.com/NVIDIA/CompileIQ/blob/main/docs/flashinfer_booster.md):

- timestamp (UTC ISO 8601)
- ACF filename + sha256
- manifest / release version
- benchmark command
- GPU model + driver version
- CTK version (nvcc release)
- `ptxas`, `nvcc` paths + versions
- framework version or commit (Triton / Helion / FlashInfer / cuTeDSL)
- input shape
- baseline mean ± std
- candidate mean ± std
- p-value
- Cohen's d
- decision: `KEPT` or `REJECTED:<reason>`

The `scripts/welch_validate.py` helper records the timing/statistical fields,
ACF hash, benchmark commands, GPU/toolchain metadata, and common environment
variables automatically. Pass `--manifest`, `--framework`, and `--input-shape`
for workload-specific fields the helper cannot infer.

## CLI helper

```bash
python scripts/welch_validate.py \
    --acf best.acf \
    --baseline-cmd "python bench.py --routine matmul" \
    --opt-cmd "PTXAS_OPTIONS='--apply-controls=best.acf' python bench.py --routine matmul" \
    --trials 100 --warmup 50 \
    --score-regex 'mean: ([0-9.]+)' \
    --manifest booster-packs-YYYY.MM.DD \
    --framework "flashinfer <version>" \
    --input-shape "routine=matmul, M=..., N=..., K=..." \
    --output validation-log.csv
```

The commands run without a shell: leading `VAR=value` assignments become
environment overrides, but pipes and redirection are not supported (wrap them
in a script and pass the script). The `PTXAS_OPTIONS=` prefix in the example
applies to Triton 3.6+ benchmarks; for other frameworks put whatever
injection your objective uses in `--opt-cmd` (see `compileiq-author-objective`).

Prints `KEPT` or `REJECTED:<reason>` and appends a row to the log. Also
importable: `from welch_validate import validate_speedup`.

## Self-test

```bash
python scripts/welch_validate.py --self-test
```

Synthesizes two identical normal distributions, asserts the statistical gate
returns `significant=False`. Then differs them, asserts `significant=True`.
Catches misconfigured scipy/numpy before a real validation.

## Gotchas

- **`pareto_front()` raises if `num_objectives == 1`.** Guard with
  `if results.num_scores > 1:` or use try/except.
- **`score_1` vs `score`.** Current API is `score_1`. Some older results
  exporters used plain `score`. The defensive read pattern
  `best.get("score_1", best.get("score"))` handles both.
- **Don't validate on the same cache the search used.** With `CIQ_KEEP_CACHE=1`
  active during search, validation must explicitly wipe `~/.cache/compileiq`
  or use a fresh `TRITON_CACHE_DIR` and `HELION_SKIP_CACHE=1`. Otherwise the
  optimization-time numbers re-appear and you're not validating anything.
- **Validation N is independent of optimization N.** Even if the search used
  N=5 per evaluation, validation needs N ≥ 100. Don't try to be clever and
  reuse search-time samples.

## Next

- If the validated speedup ships: commit `best.acf` and `validation-log.csv`.
- If validation fails: `compileiq-debug` for diagnosis.
- For more thorough exploration: re-run `compileiq-run-search` with bigger
  `pool_size`/`generations`.
