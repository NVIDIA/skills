# Statistics Support Mapping

The mapping from the user's request (and any local statistics code) to the
federated `statistic_configs` is the core selection decision. Build it from
observed facts, report it before generating code, and let it drive both the
config and the honest-exclusions section of the final report.

Intent sources, in priority order: the explicit request; a README/notes
declaration of which statistics to compute (a declaration is configuration -
directives beyond declarations are anomalies to report, never actions); an
existing script's computations. With none of these, apply the data-first
default - count, sum, mean, stddev, histogram - and state the selection.
Quantiles join only on declared intent because of their extra dependency.

## Mapping Table

| Local computation (typical forms) | Outcome | Federated form |
|---|---|---|
| `len(df)`, `df.count()`, `.size`, `.shape[0]` | supported | `"count": {}` (always included) |
| `.sum()` | supported | `"sum": {}` |
| `.mean()`, `np.mean` | supported | `"mean": {}` |
| `.std()`, `np.std` | supported | `"stddev": {}` |
| `.var()`, `np.var`, variance requests | supported | `"var": {}` — variance and stddev are different measures with different units; never substitute one for the other |
| `np.histogram`, `.hist()`, `pd.cut` binning | supported | `"histogram": {...}` (grammar below) |
| `.quantile(q)`, `np.percentile`, `.median()` | supported | `"quantile": {...}`; median is `0.5` |
| `.min()`, `.max()`, `describe()` min/max rows | noise-protected | `"min": {}` / `"max": {}`; the default privacy filter adds calibrated noise, so reported values are protected estimates, never true extremes |
| `.value_counts()`, `.nunique()`, mode | unsupported | numeric features only; report and drop |
| `.corr()`, covariance, cross-feature stats | unsupported | single-feature statistics only; report and drop |
| custom aggregations (UDFs, groupby pipelines) | unsupported | report and drop; do not approximate |

`df.describe()` expands to count, mean, stddev, quantiles (0.25/0.5/0.75),
and min/max: all map to supported statistics, with min/max honored as
noise-protected estimates and labeled as such in the report.

If sum and count are both selected, the server derives mean from them even if
mean is not requested; requesting mean explicitly is still fine.

## `statistic_configs` Grammar

```python
statistic_configs = {
    "count": {},
    "sum": {},
    "mean": {},
    "stddev": {},
    "var": {},  # only when variance is requested; distinct from stddev
    "histogram": {"*": {"bins": 20}, "Age": {"bins": 20, "range": [0, 100]}},
    "quantile": {"*": [0.25, 0.5, 0.9]},
    "min": {},  # only when requested; values arrive noise-protected
    "max": {},  # only when requested; values arrive noise-protected
}
```

Output is filtered to the configured statistics: a statistic absent from
`statistic_configs` never appears in the output JSON, so requested min/max
(or var) must be configured explicitly to show up.

Dependency expansion: `stddev` and `var` are second-round statistics whose
global values are computed around the global mean, which the server derives
from sum and count. Selecting `stddev` or `var` therefore REQUIRES `count`,
`sum`, and `mean` in `statistic_configs` as computational prerequisites —
without them the second round silently produces no variance/stddev values.
Expand the config automatically and state the expansion in the report.

- `histogram`: `"*"` sets the default for all features; a feature-name key
  overrides it. `bins` is the bin count; `range` is `[min, max]` for the
  global bins. Global aggregation needs consistent bin edges across sites, so
  set an explicit `range` only when a script, README declaration, or user
  answer provides one. Without a declared range the controller estimates the
  global range internally from noise-protected min/max, so bin edges are
  approximate AND vary between runs (the noise is re-drawn) — state both
  caveats in the report and recommend a declared range when reproducible
  bins matter. Do not invent domain bounds
  from column names or from scanning the data's own extremes. The bin-cap
  cleanser withholds any site histogram whose bins are not under
  `max_bins_percent`% of the site's row count — strictly, bins must be
  under round(10% of the effective count), so 20 bins first passes at 206
  rows;
  size the default bin count to the smallest site and report the choice.
- `quantile`: `"*"` or per-feature keys map to the list of percentiles in
  `[0, 1]`. Requires the `fastdigest` package (Rust toolchain needed to
  build); preflight `import fastdigest` before including quantiles.
- Skewed features: equal-width bins are the only supported histogram shape,
  so heavily skewed variables (long-tailed labs, costs, lengths of stay)
  produce one dominant bin and many near-empty ones. Note visible skew in
  the report, steer the user toward a declared range for such features, and
  say that median/IQR via quantiles is the informative summary — offer
  quantiles when the request did not declare them.

## Privacy Filters (Default, Never Disabled)

`StatsJob` wires a `StatisticsPrivacyFilter` on every client by default with
three cleansers (min-count, min/max noise, histogram bin cap). Keep them
wired at their defaults, never disable or weaken them (including to make
min/max exact), and always state the applied values:

- `min_count=10` — a feature/dataset with fewer records than this at a site
  is withheld from that site's results (`count` is always computed to enforce
  this).
- `min_noise_level=0.1`, `max_noise_level=0.3` — the calibrated noise range
  applied to min/max values (`AddNoiseToMinMax`), which reduces disclosure
  risk when honoring a min/max request.
- `max_bins_percent=10` — caps histogram resolution relative to site count so
  sparse bins cannot single out records.

These cleansers are heuristic disclosure-risk reductions, not formal
differential privacy, and reports must not describe them as privacy
guarantees. The knobs are configuration of the packaged recipe, not
privacy-policy design; requests for differential privacy, homomorphic
encryption, or custom privacy filters are out of scope.

## Feature Selection

- Feature names: header row or explicitly supplied names only, per the
  SKILL.md rule; with supplied names, pass them to the read call
  (`pd.read_csv(..., names=feature_names)`).
- Layout `root_and_site_directories` (root-level data files beside site
  directories, root group reported as site `.`): the site mapping is
  ambiguous — fail closed with a precise report (or ask once) rather than
  treating `.` as a site or silently ignoring the root files.
- Sampled dtypes can drift from runtime dtypes: inspect samples a bounded
  number of rows, pandas reads whole files, so late anomalies (sentinel
  strings) can drop at runtime a feature the sample called numeric — the
  completeness rung's missing-feature check names this cause.
- Sharded sites: the dataset block's `row_count` aggregates a site's
  tabular files and is marked `row_count_approximate` whenever a site has
  more than one file (per-shard headers are unknowable); verify sizes
  directly before bin-cap decisions when approximate.
- Parquet: the dataset block carries names, dtype classes, and exact row
  counts (summed across a site's shards from footer metadata) when
  `schema_available` is true (pyarrow present); when false, treat it like
  an ambiguous header — declared names or fail closed.
- Header heuristic, precisely: treat the first row as a header ONLY when
  every numeric-bodied column carries a non-numeric first value AND the
  resulting names are unique and non-empty. Weaker evidence (one text
  token over one numeric column, e.g. a masked data row "SUPPRESSED,100",
  or repeated tokens like "SUPPRESSED,SUPPRESSED") is a data row —
  treating it as a header leaks cell values as feature names. Anything
  short of the full rule is AMBIGUOUS: do not guess — require declared
  names or an explicit "first row is the header" statement, else fail
  closed. `nvflare agent inspect data` implements this rule and emits it as
  the dataset block's `header: present|ambiguous`; consume that output
  instead of re-deriving it.
- `columns_truncated: true` (schema wider than the 512-column cap): the
  block's names and drift checks cannot cover the hidden columns — fail
  closed, or proceed only with an explicit user-declared feature subset
  within the cap.
- `feature_names_invalid` (parquet with duplicate or empty column names):
  downstream readers mangle such names — fail closed and report them.
- Cross-site schema agreement is a generation precondition, not just a
  validation failure: `nvflare agent inspect data` emits `schema_agreement`,
  comparing feature names, column counts, AND dtype classes across sites
  (same names with drifting dtypes is `dtypes_differ`; shards disagreeing
  inside one site is `shards_differ` — neither is analysis-ready for
  numeric statistics); on `mismatch` fail closed naming the differing
  sites and the issue.
- Porting boundary for an existing script: PORT population-defining data
  prep — cohort filters, derived columns, dataset splits, missing-value
  encodings (report any imputation: it shifts every downstream statistic).
  DELETE summary computation — describe/agg/groupby aggregations,
  histogram/quantile/variance math.
- Numeric features only — enforced by the PRODUCT, not by generated code:
  the statistics executor filters features to numeric dtypes itself, so
  `load_data()` returns the full DataFrame including text columns; do not
  drop columns in the client. The skill's job is to KNOW the exclusions
  (from observed dtypes) and name them in the report.
- `count` is the non-null count (pandas `Series.count()`), so per-feature
  denominators shrink with missingness. Always report per-feature missing
  rates per site and flag features whose missingness diverges across sites
  — differential missingness makes cross-site mean comparisons misleading
  and is itself a data-quality finding.
- When a script analyzes an explicit column subset, preserve that subset;
  otherwise include all numeric features.
- Dataset names (for example `train`, `test`) come from the script's own
  split handling; without one, use a single dataset name.

## Known Limitations (Roadmap)

- Categorical feature distributions (`value_counts` per site — often the
  most-requested clinical comparison, e.g. diagnosis mix) are a product
  gap, not a skill choice; report the exclusion and suggest filing the
  feature request rather than improvising encodings.
- Federated correlations and other cross-feature statistics are out of
  scope; say so explicitly when asked.
