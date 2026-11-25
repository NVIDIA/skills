# Customer Metric Contract

Read this during Pre-Flight and every `evaluate` stage. DEFT optimizes one
primary customer metric and may enforce zero or more secondary constraints.
`state.metric_contract` selects the evaluator and comparison. Metrics are not
restricted to a fixed name or unit: use a supported bundled evaluator or the
command / artifact adapter below.

## State schema

Store the approved contract once at `state["metric_contract"]`:
The examples use `/home/user/workspace`, the resolved absolute form of the
default `~/workspace` root.

```json
{
  "name": "weighted_escape_cost",
  "display_name": "Weighted escape cost",
  "operator": "<=",
  "target": 0.02,
  "unit": "cost/board",
  "evaluator": {
    "type": "command",
    "path": "/home/user/workspace/metrics/evaluate_cost.py",
    "args": ["--policy", "/home/user/workspace/metrics/policy.json"]
  },
  "constraints": [
    {
      "name": "recall_pct",
      "display_name": "Recall",
      "operator": ">=",
      "target": 99.5,
      "unit": "%"
    }
  ]
}
```

Supported operators are `<`, `<=`, `>`, and `>=`. They determine both success
and best-iteration selection: first keep candidates that satisfy every
secondary constraint, then `<`/`<=` select the lowest primary result or
`>`/`>=` select the highest. If no candidate satisfies all constraints, retain
the direction-aware primary best as a clearly non-passing result. Never infer
direction from the metric's name.

`FAR < 10% at Recall=100%` is canonicalized as primary `far_pct < 10` with
unit `%`, bundled evaluator `far_at_recall`, and secondary constraint
`recall_pct >= 100%`. The evaluator result must place the achieved recall in
`constraints.recall_pct`; a diagnostic-only recall cannot satisfy the gate.
Never compare a fractional FAR value to a percentage target.

## Evaluator types

- `builtin`: an evaluator shipped with this skill. Use only an ID accepted by
  `scripts/metric_contract.py`; each bundled evaluator owns its parameters and
  diagnostics.
- `command`: an absolute customer-owned executable. Invoke it as an argv list,
  never through `eval` or an interpolated shell string:

  ```text
  <path> <configured args...> --inference-csv <absolute CSV> --output-json <absolute JSON>
  ```

- `artifact`: a declared customer system supplies the result JSON at an exact
  absolute path template containing `{iter_label}`. Store both its producer
  and template, for example:

  ```json
  {
    "type": "artifact",
    "producer": "factory-quality-service",
    "path_template": "/home/user/workspace/metrics/{iter_label}/metric_result.json"
  }
  ```

  Wait for that exact artifact; the recorder rejects a different path and the
  workflow must not substitute another metric.

A custom metric must have a command or artifact evaluator before the
Pre-Flight Summary. The evaluator is part of the approved run configuration;
changing it requires a revised Summary.

## Evaluator output

Every evaluator writes one JSON object:

```json
{
  "name": "weighted_escape_cost",
  "value": 0.018,
  "unit": "cost/board",
  "threshold": 0.31,
  "constraints": {"recall_pct": 99.8},
  "diagnostics": {"precision_pct": 94.0, "sample_count": 12480}
}
```

Required fields are `name`, finite numeric `value`, an exact unit match with
the contract, and a value for
every configured constraint. `threshold` and `diagnostics` are optional.
`passed` is not trusted from the evaluator; the bundled recorder recomputes it
from the approved contract.

## Evaluate commit

After inference and evaluator completion, commit the result and ordered event
as one atomic state update:

```bash
PYTHON=$(bash <skill_root>/scripts/deft_python.sh)
"$PYTHON" <skill_root>/scripts/commit_stage.py \
  --results-dir "${RESULTS_DIR}" \
  --iter-label "${ITER_LABEL}" \
  --stage evaluate \
  --metric-result "${METRIC_RESULT_JSON}" \
  --best-ckpt "${BEST_CHECKPOINT}" \
  --inference-csv "${INFERENCE_CSV}" \
  --training-spec "${ITER_TRAINING_SPEC}" \
  --threshold "${THRESHOLD}" \
  --duration-sec "${STAGE_DURATION_SEC}" \
  --summary "Evaluate: ${METRIC_NAME}=${METRIC_VALUE}"
```

`commit_stage.py` invokes the bundled metric recorder internally and appends
an `evaluate` event to `deft_state.json`; if validation fails it restores the
original state. Evaluator-specific compatibility fields, if any, are
secondary; new code reads `metric_result`.

## Completion and reporting

The metric recorder recomputes the primary comparison and all constraints. A
result may be structurally valid while missing the target; the loop continues
until the metric passes or `max_iterations` is reached. Reports use `display_name`,
operator, value, target, and unit for the headline. Evaluator `diagnostics`
may populate secondary charts; the primary contract controls the completion
decision.
