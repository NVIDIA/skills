# Gap Analysis

Run gap analysis only after an evaluation fails its KPI gate (or has no target)
and another iteration remains.

The stage is committed under the label whose evaluation it analyzes, but its
artifact belongs to the iteration it feeds:

| Evaluated label | `--iter-label` at commit | Feed iteration / output |
|---|---|---|
| baseline | `baseline` | `iter_1/gaps/kpi_gaps.parquet` |
| iterN | `iterN` | `iter_<N+1>/gaps/kpi_gaps.parquet` |

Do not run a gap stage after a passing KPI or after the final allowed
iteration. The transition validator rejects both paths.

## Command

Let `FEED_N=1` for baseline, otherwise the evaluated iteration plus one:

```bash
"$SKILL_ROOT/scripts/deft_python.sh" --workspace "$WORKSPACE" --runtime \
  "$SKILL_ROOT/scripts/run_pas_stage.py" gap-analysis \
    --results-dir "$RESULTS_DIR" \
    --deft-config "$RESULTS_DIR/config/deft_config.yaml" \
    --iter-num "$FEED_N"
```

The adapter resolves the exact preceding eval directory through the bundled PAS runtime,
uses the configured PAS eval split and gap metric, applies caption-diversity
limits, and writes the iteration-scoped parquet. It also maintains
`caption_selection_history.json` using the runtime's idempotent iteration
semantics.

The output must be a non-empty parquet with `filepath`, `text`, and
`weak_attribute`. If the metric gate is unmet but no gap row can be produced,
hard-stop: the query type, metric, or slicing contract is inconsistent. Do not
mine the full pool as a fallback.

Commit:

```bash
"$SKILL_ROOT/scripts/deft_python.sh" --workspace "$WORKSPACE" --runtime \
  "$SKILL_ROOT/scripts/commit_stage.py" \
    --results-dir "$RESULTS_DIR" --iter-label "$EVALUATED_LABEL" \
    --stage gap_analysis \
    --gaps-parquet "$RESULTS_DIR/iter_$FEED_N/gaps/kpi_gaps.parquet" \
    --caption-history "$RESULTS_DIR/caption_selection_history.json" \
    --gap-analysis-status \
      "$RESULTS_DIR/iter_$FEED_N/gaps/gap-analysis.host.status.json" \
    --summary "$EVALUATED_LABEL gap analysis prepared iteration $FEED_N"
```

Run the audit. The only legal next action is `iter$FEED_N/data_mining`.
If a process ended after the caption ledger write but before this commit, the
audit reports the one in-flight feed as a warning. Rerun the same idempotent
adapter once and commit; never edit the ledger manually.
