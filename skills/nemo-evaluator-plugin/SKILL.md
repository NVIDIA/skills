---
name: nemo-evaluator-plugin
description: Use when working on the Evaluator plugin CLI, jobs, SDK-backed specs, or plugin-owned Evaluator skills.
metadata:
  owner: nemo-platform
  maturity: active
license: Apache-2.0
---

# Evaluator Plugin

Use this skill when the task is about Evaluator functionality on the plugin architecture. The plugin-backed CLI surface is `nemo evaluator`; the legacy generated `nemo evaluation` API command group is not the target surface for new guidance.

## Current Surfaces

- a minimal `nemo.services` health surface
- an SDK-backed `nemo.jobs` entry, `evaluator.evaluate`, for inline metric execution
- a minimal CLI and SDK namespace
- plugin-owned docs and skills directories

## CLI Commands

> **Prerequisite:** activate the Python virtual environment before invoking the `nemo` CLI: `source .venv/bin/activate`.

Check plugin status from the CLI:

```bash
nemo evaluator info
```

Inspect the registered job contract:

```bash
nemo evaluator evaluate explain
```

Run an inline `exact-match` metric:

```bash
nemo evaluator evaluate run --spec '{"metric":{"type":"exact-match","reference":"{{item.expected}}","candidate":"{{item.model_output}}"},"dataset":[{"expected":"blue","model_output":"Blue"},{"expected":"Jupiter","model_output":"Saturn"}],"params":{"parallelism":2}}'
```

Run an inline `string-check` metric:

```bash
nemo evaluator evaluate run --spec '{"metric":{"type":"string-check","operation":"contains","left_template":"{{item.answer}}","right_template":"NeMo"},"dataset":[{"answer":"NeMo Platform supports evaluator plugins."}]}'
```

For non-trivial specs, prefer `--spec-file` over inline shell JSON:

```bash
nemo evaluator evaluate run --spec-file evaluation-spec.json
```

Submit the same spec to a cluster:

```bash
nemo evaluator evaluate submit \
  --spec-file evaluation-spec.json \
  --workspace default \
  --profile default
```

Use `nemo evaluator evaluate explain` as the source of truth for the current plugin job schema.

## Evaluation Specs

The current job accepts inline SDK-backed evaluation specs. At a high level, specs describe:

- `metric`: inline Evaluator SDK metric configuration or benchmark metrics
- `dataset`: inline rows to evaluate
- `params`: optional Evaluator SDK execution parameters
- `target`: optional model or agent target for online evaluation

For LLM-judge setup notes, see [LLM Judge Notes](references/llm-judge.md).

For evaluator API key auth, see [Evaluator API Auth](references/api-auth.md).

For local and cluster troubleshooting, see [Evaluation Troubleshooting](references/troubleshooting.md).

Call the SDK-backed status route through the platform SDK:

```python
from nemo_platform import NeMoPlatform

client = NeMoPlatform(base_url="http://localhost:8000")
status = client.evaluator.plugin_status()
```

## Next Decisions

Before replacing stubs, verify the target surface:

1. service route adaptation
2. job submission or compilation strategy
3. packaging split between service and task dependencies
