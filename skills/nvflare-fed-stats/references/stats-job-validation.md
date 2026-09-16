# Federated Statistics Job Validation

Validate in the shared ladder order and stop at the first failed rung,
reporting the product error. The skill adds one rung after the simulator
run — output completeness, executed as ephemeral commands with no
generated files. Parity verification (below) is documented for offline
harnesses, not performed by the skill.

## Ladder

1. **Compile checks** — `python -m py_compile` on generated `client.py` and
   `job.py`.
2. **Import preflight** — use the public paths
   `from nvflare.recipe import SimEnv` and
   `from nvflare.recipe.fedstats import FedStatsRecipe`; never import
   `FedStatsRecipe` from `nvflare.recipe`. Import pandas, and re-verify
   `fastdigest` only when quantiles are configured.
3. **Recipe construction** — construct `FedStatsRecipe` with the generated
   arguments without executing the job. The constructor returning without
   error IS the pass signal; do not probe attributes of the recipe object
   (it exposes no `name` and needs no inspection).
4. **Site-ownership preflight** — before execution, verify that one
   nonempty canonical `sites` list is passed unchanged to both
   `FedStatsRecipe(sites=sites)` and `SimEnv(clients=sites)`.
   `FedStatsRecipe` has already assigned those named clients to the job,
   so `SimEnv(num_clients=...)` is invalid for this recipe; neither use it
   alone nor combine it with `clients`. Let `SimEnv` derive
   `num_threads` from `clients`, or set `num_threads=len(sites)`.
5. **Simulator run** — execute `job.py` under `SimEnv`.
   `recipe.execute(env)` returns a run handle (`nvflare.recipe.run.Run` —
   note the module, not `spec`); you do not need it for validation:
   success is determined by the completeness rung over the output JSON,
   not by inspecting the handle. The statistics job is
   single-round and completes in seconds on typical tabular data; a hang
   usually means `load_data()` cannot find a site's data path.
6. **Output completeness** — run as ephemeral commands (inline python),
   never saved as script files: locate the output JSON under the simulator
   workspace (`.../server/simulate_job/<stats_output_path>`), print its
   top two levels FIRST (the hierarchy is feature-first, not
   statistic-first — probe before asserting, never guess keys), then
   assert against its actual hierarchy
   `{feature: {statistic: {site: {dataset: value}}}}`: every expected numeric
   feature appears at the top level; under each feature every configured
   statistic appears; under each statistic every site plus `Global` appears;
   and each site entry covers every dataset name. A value missing at one site
   with a small partition may be the `min_count` cleanser working as designed
   — check the site's count before calling it a failure, and report the
   withholding either way.

## Offline Parity Verification (harness-owned)

The skill does NOT perform parity verification and generates no checker
files; this section documents the contract for offline test harnesses and
manual verification.

- **Probe before parsing** — print the actual top two levels of the
  output JSON plus one leaf per statistic before writing any comparison;
  derive shapes from the file, never from guessed keys. Leaf shapes
  inside `{feature: {statistic: {site: {dataset: value}}}}`:
  - scalar statistics (count/sum/mean/stddev/var/min/max): a plain
    number, not a nested object;
  - histogram: a list of `[low_edge, high_edge, sample_count]` triples —
    there is no `"bins"` key;
  - quantile: a `{percentile: value}` mapping.
- **Per-site parity** — recompute reference statistics for one site with
  an independent pandas computation over that site's partition (never by
  executing the user's original script) and compare against that site's
  values. The controller rounds every persisted value to its `precision`
  (4 digits by default): compare count exactly and round float references
  (sum, mean, stddev — pandas `std()` ddof=1) to that precision before an
  exact compare; histogram bin counts must match when an explicit range
  was configured. CAUTION: when `var` is configured, the per-site `var`
  leaf is NOT the site's own variance — clients report the partial term
  `Σ(x − μ_global)² / (global_count − 1)`, where `global_count` is the
  dataset's global non-null count, for the global aggregation — so skip
  per-site `var` leaves entirely (the Global check covers them); per-site
  `stddev` IS the site's own stddev and compares exactly. Two statistics are approximate by design and exempt
  from exact parity: quantiles (t-digest sketch; coarse tolerance) and
  min/max (noise-protected; check presence and plausibility only).
- **Global parity** — completeness proves fields exist; this check proves
  the weighted aggregation is right. Recompute the global reference over
  the union of all site partitions, with the pooled reference computed
  around the global mean using `(N - 1)`: `Global` count exact; histogram
  bin counts equal to the sum of the site bin counts when an explicit
  range was configured. Tolerances for the rounded statistics: sum and
  mean within 1 ulp of the persisted precision; `var` within site-count
  ulps (the server rounds each site's second-round variance term to the
  persisted precision BEFORE summing — double rounding, not an
  aggregation bug). For Global `stddev`: when `var` is also configured,
  check the exact identity `stddev == round(sqrt(var), precision)`; when
  it is not, use the σ-scaled bound
  `(site_count · ulp) / (2σ) + ulp`, since a variance error of N ulps
  propagates to stddev as `err/(2σ)`. Anything beyond those bounds is a
  real mismatch even when every per-site row is correct.

## Statistics-Specific Failures

- **`fastdigest` import error / Rust toolchain missing** — quantiles cannot
  be computed. Report the product error; include quantiles only after a
  successful preflight.
- **Empty or missing site data path** — `load_data()` raised or returned no
  DataFrame for a site. Verify the per-site path parameterization and the
  partition step; do not fall back to sharing one site's data.
- **All features excluded as non-numeric** — nothing to compute. Report the
  observed dtypes and stop; do not coerce categorical codes into numerics to
  make output appear.
- **Histogram range errors or empty global histograms** — usually a per-site
  range mismatch when no explicit `range` was configured; set an explicit
  per-feature range from evidence or a user answer and rerun.
- **Feature withheld at a site** — likely `min_count`; report the applied
  threshold rather than padding results.
- **Requested min/max absent from output** — output is filtered to
  configured statistics; confirm `"min"`/`"max"` were actually placed in
  `statistic_configs`. When configured they must appear, with noised
  values (presence-and-plausibility check only).
- **var/stddev silently missing from output** — the second round computes
  them around the global mean; confirm `count`, `sum`, and `mean` were all
  configured as prerequisites (see the dependency-expansion rule in
  statistics-mapping.md).
- **Histogram withheld at a site** — the bin-cap cleanser requires bins to
  be under round(`max_bins_percent`% of the site's effective row count);
  with defaults, 20 bins first passes at 206 rows. Reduce the bin count and
  rerun, or report the withholding.
- **Schema mismatch across sites** — pre-split site files must share one
  schema; on differing headers or column counts, fail closed and report the
  differing sites instead of intersecting columns silently.
- **Per-site dtype drift** — each client derives its feature list from its
  own dtypes, so one stray string (a sentinel like "unknown" in a numeric
  column) silently drops that feature at that site. `nvflare agent
  inspect` catches this before generation (`schema_agreement` reports
  `dtypes_differ`); when it surfaces only at validation, report the drift
  as a data-quality finding ("site-2's age parsed as text — check for
  sentinel strings"), distinct from min_count withholding.

## Reporting

Report: the mapping outcomes (supported / noise-protected / unsupported),
changed files (`client.py` and `job.py` only), each ladder rung's status,
the explicit note that numeric parity was NOT verified (harness-owned;
contract above) so the user knows the verification boundary,
the applied privacy parameters
(described as heuristic disclosure-risk reductions, never as privacy
guarantees), per-feature missing rates per site with cross-site divergence
flagged, the exact output JSON path, and a compact per-site plus Global
summary table of aggregates. Include the standing case-mix caveat: Global
pools sites with different case mixes, so site rows can each contradict the
Global trend (Simpson's effect) — compare site rows before acting on
Global. Note visible skew where equal-width histogram bins are
uninformative. Never include raw data rows or individual values; everything
shown must already be an aggregate from the output JSON.
