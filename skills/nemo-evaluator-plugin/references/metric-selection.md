# Metric Selection

Read this file when converting a rubric into evaluator metrics.

## Prefer the simplest metric

Import metrics from the package root: `from nemo_evaluator_sdk import <METRIC_NAME>`.
CLI metric names are lowercase (`exact-match`, `bleu`); class names are not
derivable from them by string substitution. Run
`nemo evaluator metric-types <name>` for the schema.

The supported set is exactly: `bleu`, `exact-match`, `f1`, `llm-judge`,
`nemo-agent-toolkit-remote`, `number-check`, `remote`, `rouge`,
`string-check`, and `tool-calling`.

| Goal | Prefer |
| --- | --- |
| Exact label, enum, or regression | `ExactMatchMetric` |
| Contains, equals, or starts/ends with | `StringCheckMetric` |
| Numeric value or threshold | `NumberCheckMetric` |
| Text overlap | `F1Metric`, `BLEUMetric`, or `ROUGEMetric` |
| Semantic quality or a written rubric | `LLMJudgeMetric` |
| Retrieval smoke test | A deterministic context assertion or `LLMJudgeMetric` |
| Tool-call correctness | `ToolCallingMetric` |
| Existing scoring service | `RemoteMetric` or `NemoAgentToolkitRemoteMetric` |
| Agent answer or goal completion | A task-specific custom metric that implements `nemo_evaluator_sdk.metrics.protocol.Metric` or `LLMJudgeMetric` |

Use deterministic metrics before an LLM judge. Use an LLM only when the
criterion requires semantic judgment.

## Template context differs by evaluation shape

Metric templates are Jinja over a context that depends on the evaluation shape:

| Shape | Available roots |
| --- | --- |
| Dataset-driven | `item.*` (the dataset row), `sample.*` (generated output) |
| Task-driven | `inputs.*`, `reference.*` (grader-only), `task.*`, `trial.*`, `sample.output_text` |

A dataset-driven template (`{{item.expected}}`) fails on every trial in an
agent evaluation. The error names the available keys — read it before changing
the metric.

### Explore the metrics provided by the SDK

List current metric names and inspect one schema:

```bash
uv run nemo evaluator metric-types
uv run nemo evaluator metric-types exact-match
```

## Validate the mapping

Create one row that must pass and one that must fail:

```python
from nemo_evaluator_sdk import ExactMatchMetric

metric = ExactMatchMetric(
    reference="{{item.expected}}",
    candidate="{{item.output}}",
)
rows = [
    {"expected": "Paris", "output": "Paris"},
    {"expected": "Paris", "output": "London"},
]
```

Check:

- Dataset keys match every Jinja template.
- Normalization is intentional; do not hide case or whitespace differences
  unless the rubric says they are irrelevant.
- Judge prompts specify the rubric and parser-compatible output.
- Tool metrics receive their required canonical fields or a `FieldMapping`.

## Use multiple metrics only for distinct dimensions

SDK — pass a metric sequence in one call:

```python
from nemo_evaluator_sdk import Evaluator
result = Evaluator().run_sync(metrics=[accuracy, style], dataset=rows)
```

Platform job — put multiple stored metrics on the job spec:

```bash
uv run nemo evaluator evaluate submit --spec \
  '{"metrics":["default/accuracy","default/style"],"dataset":"default/eval-data"}'
```

Each `metrics` entry may be an inline metric bundle, a stored `MetricRef`, or
a mix of both. The high-level `client.evaluator.submit` helper still accepts
only one runtime metric per call.
