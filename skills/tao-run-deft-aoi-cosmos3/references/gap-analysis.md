# Proxy RCCA and Benchmark KPI

Run `analyze_gaps.py` only after the matching evaluator job reaches
`COMPLETE`.

## Proxy

```bash
"$PYTHON" "$SKILL_ROOT/scripts/analyze_gaps.py" \
  --results-json "$RESULTS_DIR/$LABEL/evaluate_proxy/results.json" \
  --output-dir "$RESULTS_DIR/$LABEL/proxy_rcca" \
  --evaluation-role proxy \
  --kpi-metric "$KPI_METRIC" \
  --kpi-threshold "$KPI_THRESHOLD"
```

Proxy writes:

- `gaps_summary.json`;
- `false_accepts.json` / `.csv`;
- `false_rejects.json` / `.csv`;
- `unknown_predictions.json`.

Before the `proxy_rcca` commit, write `RCCA_Report.md` in this same output
directory from `gaps_summary.json`, `false_accepts.json`, and
`false_rejects.json`, following `references/RCCA_REPORT_TEMPLATE.md`. The
commit requires `--rcca-report <absolute path to RCCA_Report.md>` and validates
all six section headings. Artifact classes, state fields, and required headings
come from `references/rcca-artifact-manifest.json`.

Only these Proxy artifacts may produce routing/mining targets. Proxy
`kpi.met` is intentionally null.

### Recovering image paths for routing

The evaluator's `results.json` carries only `video_id`, `response`, `question`,
and `gt`, so every gap row's `images` field is `null`. The gap artifacts alone
are therefore **not** enough to build `mining_targets.json` — routing must join
each row back to its source record by `id`.

Join against `annotations/proxy_kpi.json` **only**. Pulling a row from
Benchmark, or from a merged view of both, feeds Benchmark error signal into
mining and is a hard stop. Fail the join loudly: an `id` that is absent from
Proxy means the gap row came from somewhere it should not have.

## Frozen Benchmark

```bash
"$PYTHON" "$SKILL_ROOT/scripts/analyze_gaps.py" \
  --results-json "$RESULTS_DIR/$LABEL/evaluate_benchmark/results.json" \
  --output-dir "$RESULTS_DIR/$LABEL/benchmark_metrics" \
  --evaluation-role benchmark \
  --kpi-metric "$KPI_METRIC" \
  --kpi-threshold "$KPI_THRESHOLD"
```

Benchmark writes aggregate `metrics_summary.json` and
`metric_result.json`. The metric result contains the configured primary value
and `unknown_predictions`; `record_metric_result.py` compares both with the
approved contract. Benchmark does not write routing artifacts.

## Normalization and confusion matrix

`NG` is positive:

- TP: NG -> NG
- FN / false accept: NG -> OK
- FP / false reject: OK -> NG
- TN: OK -> OK

The evaluator and analyzer use the last standalone `OK`/`NG` token. A response
without either token is `UNKNOWN` and blocks the Benchmark gate.
