# Frozen Benchmark Metric Contract

`init_deft_state.py` writes one approved metric contract. Default:

```json
{
  "name": "recall_ng",
  "display_name": "NG recall",
  "operator": ">=",
  "target": 1.0,
  "unit": "",
  "evaluator": {
    "type": "artifact",
    "producer": "scripts/analyze_gaps.py",
    "path_template": "<results>/{iter_label}/benchmark_metrics/metric_result.json"
  },
  "constraints": [
    {
      "name": "unknown_predictions",
      "operator": "<=",
      "target": 0,
      "unit": ""
    }
  ]
}
```

The contract is the source of truth for stop decisions and best-iteration
selection. Proxy metrics are diagnostic and never committed as the iteration
metric result.

Supported primary metrics in this migration are `recall_ng`,
`precision_ng`, `f1_ng`, and `accuracy`, all compared with `>=`. Values and
targets are fractions in `[0, 1]`.

`benchmark_metrics/metric_result.json` must match the configured name and unit
exactly, and must provide the value and constraints. `record_metric_result.py`
adds its absolute path as
evidence; `commit_stage.py` validates it before recording the stage.
