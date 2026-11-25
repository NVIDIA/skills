# DEFT Loop Report Rendering Protocol

Template: `references/DEFT_Loop_Report.html`. Output: `results/DEFT_Loop_Report.html`.
`scripts/render_report.py` is the only renderer. `init_deft_state.py` invokes it
once at loop start, and `commit_stage.py` invokes it as a post-commit hook after
every stage, including `loop_stop`. Embed all images as base64 data URIs so the
file opens offline.

## When to update which data

| Stage trigger | New data available |
|---|---|
| Loop start (config loaded) | `{{ RUN_SUMMARY_ROWS_HTML }}`, baseline `{{ GROWTH_ROWS_HTML }}`, `{{ PROBLEM_STATEMENT_HTML }}`, and `{{ APPROACH_HTML }}`. The summary records the approved KPI, Visual ChangeNet identity, GPU count/model, routing, mining top-K/cosine floor/history-aware filepath dedup policy, and available outcome evidence. |
| Baseline evaluate done | baseline primary `metric_result`, optional threshold, and evaluator diagnostics; `{{ KPI_DATASET_HTML }}` populated from the evaluation manifest scanned during baseline. |
| Baseline RCA done | RCA insight, score distribution, evaluator diagnostics, and defect type rows; also refines `{{ KPI_DATASET_HTML }}`'s per-defect-type breakdown using `${results_dir}/baseline/rca_results/defect_type_rows.csv`. |
| Iter N evaluate done | iter N primary metric, constraints, optional threshold, diagnostics, checkpoint |
| Iter N RCA done | updated RCA insight and tables |
| Iter N AnomalyGen done | an explicit `AnomalyGen: enabled` status line, the `num_SDG` generated count, and sample images (base64 thumbnails) for iter N |
| Iter N k-NN filtering / data merge done | `knn_summary.csv` supplies `KNN Raw Mined`; the committed combined training CSV supplies cumulative rows and the monotonic difference from the preceding total supplies `New Unique Images (After Dedup)` and `Δ`. |
| Loop stop (KPI met or max_iterations) | final status, `best_iter`, recommendations |

The **Run Configuration & Outcome** card always contains the two-column DEFT
experiment summary followed by **Training Set Growth** with the exact columns
`Iteration`, `KNN Raw Mined`, `SDG Generated`,
`New Unique Images (After Dedup)`, `Training Set Total`, and `Δ`. `SDG
Generated` is the row count of the committed `SDG_result.csv`; the two final
growth columns come from the cumulative combined training CSV, so `New Unique
Images (After Dedup)` always equals `Δ`. Runtime totals are sums of positive measured
`duration_sec` values from `deft_state.json.events`; identify missing historical
durations as unavailable data rather than treating zero as elapsed time.

Stub values for data not yet available:
- future iter metric/rows → render `—`
- missing image columns → `.sample-img-placeholder`

## In-progress rendering rules

While the loop has not stopped:

- `{{ FINAL_KPI_STATUS }}` → `"IN PROGRESS"`, class → `""` (no green).
- `{{ ITERATIONS_RUN }}` → count of iterations with `status == "complete"` at render time.
- Iteration table rows → only completed iterations; omit rows for unstarted iterations.
- `{{ ITER_CARDS_HTML }}` → only emit cards for completed iterations.
- KPI banner → empty string while running or when a terminal run misses the
  target; inject it only for KPI MET or a committed hard stop.
- `{{ METRIC_DATA_JSON }}` → primary-metric points from completed iterations.

## KPI status phrasing — be neutral, never say "NOT MET"

We are the product team. When the target is not yet reached, describe the **gap**
in the final status panel instead of stamping a failure label. Phrasing rules
for `{{ FINAL_KPI_STATUS }}`:

| Condition | `FINAL_KPI_STATUS` | `FINAL_KPI_STATUS_CLASS` |
|---|---|---|
| Contract comparison and every constraint pass | `"MET"` | `"green"` |
| Contract misses | `"{absolute direction-aware gap}{unit} from target"` | `""` |
| Loop still running | `"IN PROGRESS"` | `""` |

For `<`/`<=`, gap is `best_value - target`; for `>`/`>=`, gap is
`target - best_value`. Show `pp` only when the metric unit is `%`; otherwise
use the configured unit. Recompute rather than trusting a stored `passed` flag.
When substituting `{{ PRIMARY_METRIC_UNIT }}`, use `%` directly and prepend one
space for non-percent units (for example ` cost/board`) so value labels remain
readable without encoding presentation whitespace in `metric_contract.unit`.

Do **not** emit `"NOT MET"`, `"FAILED"`, the `red` CSS class, or a KPI banner
when the target is missed. In particular, never emit the yellow informational
`i` / "Best result" banner. The final status panel and metric tables already
report the gap factually.

## Renderer entry point

```bash
PYTHON=$(bash <skill_root>/scripts/deft_python.sh)
"$PYTHON" <skill_root>/scripts/render_report.py \
  --results-dir "${RESULTS_DIR}"
```

Do not copy this protocol into inline Python. The bundled script owns
placeholder construction, escaping, validation, and atomic replacement. Use
`--require-terminal` for a final manual verification/rebuild.

### CRITICAL: Always render in a single pass from the source template

**Never read the output file and apply a second round of `.replace()` calls on it.**
Each render must start fresh from `references/DEFT_Loop_Report.html`, apply all
substitutions in one chained block, then write the output. Reading the output file
for a second pass causes two silent bugs:

1. **Section duplication.** If any placeholder was not filled in pass 1 (e.g.
   `{{ ITERATION_TABLE_ROWS_HTML }}`), pass 2 may split the partially-rendered HTML
   on that token and inject the second half of the file as the replacement value,
   duplicating every subsequent section and producing two `<script>` blocks.
2. **Stale data.** A second pass may overwrite already-correct values with stale data
   from a different state snapshot.

Pattern to follow — collect all values before writing:

```python
html = (
    template                                              # from source, not output
    .replace('{{ GENERATED_DATE }}',           generated_date)
    .replace('{{ KPI_TARGET }}',               kpi_target)
    .replace('{{ PRIMARY_METRIC_LABEL }}',     primary_metric_label)
    .replace('{{ PRIMARY_METRIC_UNIT }}',      primary_metric_unit)
    .replace('{{ PRIMARY_METRIC_UNIT_JSON }}', json.dumps(primary_metric_unit))
    .replace('{{ METRIC_DIRECTION_LABEL }}',  metric_direction_label)
    .replace('{{ METRIC_DATA_JSON }}',         metric_data_json)
    .replace('{{ METRIC_TARGET_VALUE }}',      metric_target_value)
    .replace('{{ METRIC_Y_MIN }}',             metric_y_min)
    .replace('{{ METRIC_Y_MAX }}',             metric_y_max)
    .replace('{{ METRIC_Y_STEPS_JSON }}',      metric_y_steps_json)
    .replace('{{ METRIC_MINIMIZES_JSON }}',    metric_minimizes_json)
    # ... ALL remaining tokens in one chain ...
    .replace('{{ RECOMMENDATIONS_HTML }}',     recommendations_html)
)
out_path.write_text(html)
assert html.count('{{ ') == 0, "Unfilled placeholders remain"
```

## Template prep gotchas

### Strip the doc-comment header before any placeholder replacement

The template starts with a `<!-- ... -->` author-documentation block that must be
removed before substitution. **Do not** use a greedy or non-greedy `<!--.*?-->`
regex — it will stop at the first `-->` inside the block and leave the remainder
as raw visible text. Use exact boundary detection:

```python
outer_close = template.index('-->\n<html')
doc_start   = template.index('<!--\n====')
template    = template[:doc_start] + template[outer_close + 3:]
```

### Image embedding

Embed sample images as base64 JPEG data URIs (`data:image/jpeg;base64,...`)
resized to **256×256** with `PIL.Image.thumbnail` (each image now occupies twice
the screen area as before, so the previous 128px thumbnails look soft). The
sample strip is **2 columns only** — left column is **AnomalyGen Input**, right
column is **AnomalyGen Output** — matching
`.sample-strip { grid-template-columns: repeat(2, 1fr); max-width: 640px }`
in the template.

Column direction follows AnomalyGen-model semantics: the **OK / normal**
reference crop is the *input* the generator was conditioned on; the
**synthetic NG** crop is its *output* with the defect added. Do not flip the
columns — a previous revision had them reversed and made it look like the
loop reconstructed a clean image from a defect, which is the opposite of
what AnomalyGen does.

| Column (template label)                  | Source path |
|---|---|
| Left — AnomalyGen Input (OK / normal)    | `${RESULTS_DIR}/iter${N}/dataset/images/synthetic_iter${N}_ok/` |
| Right — AnomalyGen Output (synthetic NG) | `${RESULTS_DIR}/iter${N}/dataset/images/synthetic_iter${N}_ng/` |

EA variant: these dirs are populated by the pre-gen ingest stage
(`scripts/changenet_data_pair_prepare.py` staging output), not by an SDG
container. Sample selection still works on the same iter-scoped staging tree.

Emit **exactly one** `.sample-iter-block` containing **one** pair — not one per
iteration. Selection rule: pick the first existing pair (sorted by filename)
from the best iteration. If the best iteration has no AnomalyGen output, fall
back to the most recent iteration that does; if none, emit two
`<div class="sample-img-placeholder">No image</div>` cells. The earlier `Normal`,
`OV SDG Defect`, and `Mask` columns were removed and the per-iteration loop was
collapsed — do not emit any of them. Rationale: every extra sample is one more
crop the reader can complain about; one clean pair is the deliverable.

### Chart data field names (must match the template's JavaScript)

The template's JavaScript accesses specific field names. Using wrong names renders
blank charts with no error. Confirmed correct schemas from the template source:

| Placeholder | Required JSON schema | JS field accessed |
|---|---|---|
| `{{ METRIC_DATA_JSON }}` | `[{"label": "Baseline", "value": 0.018, "color": "#c2262d"}, ...]` for the configured primary metric | `d.value`, `d.color`, `d.label` |

Common mistake: using an evaluator-specific field instead of the generic `value`.

Build the chart extent from the primary-metric values and target, add a small
padding when the extent is non-zero, and emit numeric `METRIC_Y_MIN`,
`METRIC_Y_MAX`, and tick values. `METRIC_MINIMIZES_JSON` is lowercase JSON
`true` for `<` / `<=` and `false` for `>` / `>=`; it controls segment colors.
Do not infer direction or a percent unit from the metric name.

The training-data stacked bar chart (`DATA_DATA_JSON`, `DATA_Y_MAX`,
`DATA_Y_STEPS_JSON`) was removed from the Progress Overview. The Augmentation
Pool table below the primary-metric chart now carries that information instead — do not
attempt to render the old chart.

### Global context cards (Problem Statement / KPI Dataset / Approach)

These three pre-rendered HTML blocks sit between the hero and Progress Overview
and provide the run's global context — mirroring the gap-analysis report
sections "Problem Statement", "KPI Dataset Statistics", and "DEFT". The
renderer builds each as a Python string and substitutes it once per render.
Schemas:

| Placeholder | Required pieces | Source on disk |
|---|---|---|
| `{{ PROBLEM_STATEMENT_HTML }}` | Task + model, configured primary metric predicate, evaluator and constraints, plus failure modes targeted by mining/synthetic/fine-tuning. Keep evaluator diagnostics separate from the primary goal. Bake values directly into the block. | `deft_state.json` → `metric_contract`, `kpi_target`, `max_iterations` |
| `{{ KPI_DATASET_HTML }}` | Totals (component crops + PASS/NO_PASS split), component categories, per-defect-type breakdown within NO_PASS, lighting variants, notable imbalances. Render as a one-paragraph summary + `.data-table` with one row per component category. | KPI eval manifest under `${results_dir}/baseline/eval/` + `${results_dir}/baseline/rca_results/defect_type_rows.csv`. If RCA has not yet run, omit the per-defect-type column and keep the totals paragraph only. |
| `{{ APPROACH_HTML }}` | The five-stage iterative recipe (evaluate → RCA → route → augment via k-NN + AnomalyGen → fine-tune); top-K, cosine floor, and cross-iteration filepath dedup; the run's headline lever (which augmentation source dominates). Render as a paragraph + `.insight` box. | `deft_state.json` → `config.mining_filter`, `max_iterations`; latest iter's `mining_pool.csv` plus `mining_history_summary.json` (sdg vs novel-mined ratio and already-mined rejects drive the headline lever). |

Keep these blocks **populated from the first render onward** — the user sees
them when they open the live HTML even before the first iteration completes.
Use `.context-list` for bullet lists inside the Problem Statement block (green
arrow bullets matching the iter-card style) and `.info-text` for paragraphs.

### Table row schemas (must match template `<thead>` column counts)

Each `*_ROWS_HTML` placeholder is injected inside a `<tbody>` whose `<thead>` is
fixed in the template. Column counts must match exactly or cells overflow/underflow
silently. Confirmed column counts from the template:

| Placeholder | Columns (count) | Column names |
|---|---|---|
| `{{ ITERATION_TABLE_ROWS_HTML }}` | 8 | Phase, Primary Metric, Δ vs Baseline, Threshold, Training Rows, Synthetic, Syn Ratio, Note |
| `{{ SCORE_DIST_ROWS_HTML }}` | 4 | Metric (Score Range), PASS, NO_PASS, Notes |
| `{{ EVALUATOR_DIAGNOSTIC_ROWS_HTML }}` | 4 | Measure, Value, Threshold, Status |
| `{{ DEFECT_TYPE_ROWS_HTML }}` | 4 | Defect Type, Count, Score Range, Detectable at KPI threshold? |

### Verifying placeholder count

When counting rendered placeholder divs for verification, search for
`<div class="sample-img-placeholder">` — not the bare class string, which also
appears in CSS and comment text.
