# SDK Execution

Read this file before choosing a dataset representation, execution mode,
target, configuration, field mapping, or job-result operation.

CLI snippets use the installed `nemo` command. In a repository checkout,
prefix them with `uv run`.

## Validate standalone, then submit to the platform

### Standalone SDK

Use the standalone SDK for the fastest in-process metric loop:

```python
from nemo_evaluator_sdk import Evaluator, ExactMatchMetric

result = Evaluator().run_sync(
    metrics=ExactMatchMetric(
        reference="{{item.expected}}",
        candidate="{{item.output}}",
    ),
    dataset=[
        {"expected": "Paris", "output": "Paris"},
        {"expected": "Paris", "output": "London"},
    ],
)
print(result.row_scores)
print(result.aggregate_scores)
```

**Platform CLI**

Platform CLI equivalent for the same checked metric and rows:

```bash
uv run nemo evaluator evaluate submit \
  --spec-file skills/nemo-evaluator-plugin/assets/specs/exact_match_metric.json
```

**Platform Python SDK**

Use `client.evaluator.submit` for execution through the installed nemo-evaluator-plugin:

```python
from nemo_evaluator_sdk import ExactMatchMetric, RunConfig
from nemo_platform import NeMoPlatform

client = NeMoPlatform(base_url="http://localhost:8080", workspace="default")
job = client.evaluator.submit(
    metric=ExactMatchMetric(
        reference="{{item.expected}}",
        candidate="{{item.output}}",
    ),
    dataset=[
        {"expected": "Paris", "output": "Paris"},
        {"expected": "Paris", "output": "London"},
    ],
    config=RunConfig(parallelism=2),
)
job.wait_until_done()
result = job.get_result()
```

### Use a Fileset for the dataset

The job will run in the container environment.
The plugin submission dataset accepts inline rows, a `str` or `Path`, and
`FilesetRef`:

**Platform SDK**

```python
from nemo_evaluator.sdk import FilesetRef

dataset = FilesetRef("default/eval-data")
```

**Platform CLI**

CLI equivalent, using a stored metric and fileset:

```bash
nemo evaluator evaluate submit \
  --spec '{"metrics":["default/exact-answer"],"dataset":"default/eval-data"}'
```

## Configure online generation

Use `RunConfigOnlineModel` with `Model`, or `RunConfigOnline` with `Agent` when
the evaluator should generate output before scoring. Provide a prompt template
alongside the target.

**Platform SDK**

```python
from nemo_evaluator_sdk import (
    ExactMatchMetric,
    Model,
    RunConfigOnlineModel,
    SecretRef,
)

target = Model(
    url="https://provider.example/v1/chat/completions",
    name="<model-id>",
    format="openai",
    api_key_secret=SecretRef(root="nvidia-api-key"),
)

job = client.evaluator.submit(
    metric=ExactMatchMetric(reference="{{item.expected}}"),
    dataset=[{"question": "Capital of France?", "expected": "Paris"}],
    target=target,
    prompt_template={
        "messages": [{"role": "user", "content": "{{item.question}}"}],
    },
    config=RunConfigOnlineModel(parallelism=2),
)
job.wait_until_done()
result = job.get_result()
```

`nvidia-api-key` names a NeMo Platform workspace secret; the example does not
embed the credential value.

**Platform CLI**

The checked LLM-judge spec is local-first and reads `NVIDIA_API_KEY`. Before
platform submission, follow the API Auth guidance to remap the target and
metric-bundle secret references, creating `llm_as_judge.platform.json`, then
submit that copy:

```bash
nemo evaluator evaluate submit \
  --spec-file llm_as_judge.platform.json
```

Platform submission requires provider access and the referenced platform workspace
secret.

## Map noncanonical fields

Use `FieldMapping` when the metric expects canonical evaluator fields but the
dataset uses different column names:

**Platform SDK**

```python
from nemo_evaluator_sdk import FieldMapping

mapping = FieldMapping(
    output="assistant_answer",
    reference="gold_answer",
)
```

Pass it as `field_mapping=mapping` when submitting the job.

**Platform CLI**

```bash
nemo evaluator evaluate submit --spec \
  '{
    "metrics": ["default/exact-answer"],
    "dataset": [{"gold_answer": "Paris", "assistant_answer": "Paris"}],
    "field_mapping": {
      "output": "assistant_answer",
      "reference": "gold_answer"
    }
  }'
```

Use field_mapping when a metric or online prompt uses canonical evaluator fields but the dataset uses different column names. One job-level mapping applies to every metric and the generation prompt.

## Getting job results

**Platform SDK**

```python
job = client.evaluator.submit(
    metric=metric,
    dataset=dataset,
    config=config,
    target=target,
    prompt_template=prompt_template,
)

job.wait_until_done()
result = job.get_result()
artifacts = job.download_artifacts("./artifacts")  # local output dir
```

`EvaluatorJobResource` also exposes methods for job lifecycle management:

- `name` and `job`
- `get_job_status()`
- `check_if_complete(raise_if_not_complete=False)`
- `get_result(aggregate_fields=...)`
- `as_async()`

**Platform CLI**

Poll until the job is completed before downloading results:

```bash
nemo evaluator evaluate submit --spec-file evaluation.json
nemo jobs get-status <job-name>
nemo jobs results list <job-name>
nemo jobs results download aggregate-scores \
  --job <job-name> --output-file aggregate-scores.json
nemo jobs results download row-scores \
  --job <job-name> --output-file row-scores.jsonl
```

The CLI `submit` command returns the created job record immediately. It does
not wait or expose follow-up result/download commands under the `evaluate`
group. Use the SDK submission handle when the workflow needs those lifecycle
operations.

### Notes

- Always wait for terminal completion. A metric can report 100 percent progress
before the platform finishes publishing result artifacts.
- `submit` accepts a concrete `Model` or `ModelRef`; the platform resolves model
references in the target workspace.

## Multiple metrics

The high-level plugin helper takes one runtime `metric`. To combine metrics,
build an `EvaluateInputSpec` and submit it with the CLI. Stored metric references
are resolved by the platform submission path:

```json
{
  "metrics": [
    "default/accuracy",
    "default/style"
  ],
  "dataset": "default/eval-data",
  "params": {"parallelism": 4}
}
```

Save the spec as `multi-metric.json`, then submit it:

```bash
nemo evaluator evaluate submit --spec-file multi-metric.json
```

Inspect the authoritative wire schema before authoring a spec:

```bash
nemo evaluator evaluate explain
```

## Package metrics safely

Built-in metrics default to declarative inline bundles. For a custom Python
metric submitted to a service, opt in explicitly:

```python
from nemo_evaluator.shared.metric_bundles.hybrid import HybridMetricBundlePackager

job = client.evaluator.submit(
    metric=custom_metric,
    dataset=rows,
    metric_bundle_packager=HybridMetricBundlePackager(),
)
```

The CLI cannot package a Python metric object or select
`metric_bundle_packager`. After Python serializes the bundled metric into a
complete spec, submit that spec with:

```bash
nemo evaluator evaluate submit --spec-file custom-metric.json
```
