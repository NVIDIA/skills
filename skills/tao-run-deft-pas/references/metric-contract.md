# PAS Metric Contract

The immutable metric contract controls evaluation parsing, KPI stopping, and
best-iteration reporting:

```json
{
  "metric_name": "Rank-1",
  "query_type": "medium",
  "op": ">=",
  "target": 0.25
}
```

## Valid values

- metric: `mAP`, `Rank-1`, `Rank-5`, `Separability`, `Match@5`, or `Zero@5`;
- query type: `easy`, `medium`, or `hard` (the rows emitted by the PAS
  aggregate evaluator; broader caption categories are gap filters, not KPI
  rows);
- operator: `<`, `<=`, `>`, or `>=`;
- target: finite number or null.

Defaults are `Rank-1`, `medium`, `>=`, and null. Null means no gate: every
allowed iteration runs and completion is `max_iterations`. All listed metrics
are naturally higher-is-better except `Zero@5`, but the user's approved
operator is authoritative. An unusual direction is warned about during
initialization and must be visible in pre-flight; it is not silently corrected.
Best-result selection follows the operator.

Gap analysis uses the same approved metric by default. Do not change either
metric after state initialization.

## Bind evaluation evidence

After the TAO evaluation has a zero native backend exit and its canonical
`status.json` contains `Evaluate finished successfully`, parse the exact
label's CSV:

```bash
TARGET_ARGS=()
if [ -n "${METRIC_TARGET:-}" ]; then
  TARGET_ARGS=(--target "$METRIC_TARGET")
fi

"$SKILL_ROOT/scripts/deft_python.sh" --workspace "$WORKSPACE" \
  "$SKILL_ROOT/scripts/parse_pas_metrics.py" \
    --metrics-csv "$PHASE_DIR/evaluate/nvidia_pas_metrics_aggregate.csv" \
    --metric-name "$METRIC_NAME" --query-type "$QUERY_TYPE" \
    --op "$METRIC_OP" "${TARGET_ARGS[@]}" \
    --iter-label "$LABEL" \
    --output "$PHASE_DIR/evaluate/metric_result.json"
```

`LABEL` is exactly `baseline` or `iterN`; `PHASE_DIR` is exactly `zs` or
`iter_N`. The parser rejects missing files, absent/duplicate query rows,
unknown columns, and non-finite values. It writes source path, full selected
row, normalized contract, numeric value, and computed `passed`. Its exit is
zero for a valid non-passing result; gate failure is workflow data, not a
parser error.

Pass the aggregate CSV, TAO status, parser result, platform-action status, and (for
iterN) iteration summary together to `commit_stage.py`, as shown in
`clip-train-eval.md`. Commit reopens the CSV, verifies label/path provenance,
re-derives the value and comparison, and records the canonical result. Never
call `record_metric_result.py` directly.

## Decision after commit

Run the audit and follow only its state-aware next action:

- target present and comparison passes: commit `loop_stop --reason kpi_met`;
- comparison fails and another iteration remains: baseline proceeds to gap
  analysis; iterN proceeds to gap analysis for N+1;
- comparison fails or target is absent at the final iteration: commit
  `loop_stop --reason max_iterations`.

Once a passing result is committed, mining/training transitions are illegal.
A numeric metric printed in a log, CSV, or report that was not bound through
this transaction cannot stop the loop.
