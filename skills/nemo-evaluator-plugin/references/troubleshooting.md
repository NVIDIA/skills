# Evaluation Troubleshooting

The plugin CLI surface is `nemo evaluator`. In a repository checkout, prefix
the commands below with `uv run`.

## Inspect the installed contracts

```bash
nemo evaluator info
nemo evaluator metric-types
nemo evaluator evaluate explain
nemo evaluator agent-evaluate explain
```

## Common failures

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| `No such command 'evaluation'` | The legacy generated CLI group is not the plugin surface | Use `nemo evaluator ...` |
| Guidance or `--help` references a local plugin `run` verb | That execution path is being retired | Use `submit`, or the standalone SDK for local iteration |
| Agent-eval metric fails every trial with a missing template key | The metric uses the dataset-driven `item.*` context in a task-driven run | Use `inputs.*`, `reference.*`, `task.*`, `trial.*`, or `sample.output_text` |
| Spec validation error | Fields do not match the current job schema | Run the matching `explain` command and validate against the spec class before submission |
| Dataset row has missing fields | Jinja templates or `field_mapping` do not match row keys | Inspect one row and every referenced template before rerunning |
| Standalone model/agent authentication fails | `api_key_secret` names a platform secret instead of an environment variable, or the variable is unset | Use the name of a populated local environment variable |
| Remote submission returns 409 | The response may describe a missing platform secret, not a duplicate job | Read the response body and verify the workspace secret |
| Built-in metric bundle contains cloudpickle | A legacy or explicit packager was used | Regenerate with `InlineMetricBundlePackager` or the current default |
| `cloudpickle metric payload was created with Python ...` (HTTP 422) | The bundle was created with a different Python major/minor runtime | For a built-in metric, regenerate the checked inline JSON spec; for an intentional custom metric, recreate the bundle with the worker's Python major/minor version |
| Custom metric submission rejects the default packager | Shipping custom code requires explicit opt-in | Pass `HybridMetricBundlePackager()` (preferred) or `CloudpickleMetricBundlePackager()` |
| `ModelRef` fails with the standalone SDK | Model references are resolved by the platform submission path | Use a concrete `Model` with the standalone SDK or use `submit` with `ModelRef` |
| Fileset evaluation cannot load data | The reference, fragment, or workspace is wrong | Verify the `FilesetRef` and access it through the same workspace |
| Result download fails while progress shows 100% | Metric progress finished before the platform job finalized artifacts | Call `job.wait_until_done()` before `get_result()` or `download_artifacts()` |
| Agent-eval rejects the spec | Both or neither of `target` and `trials` were provided | Provide exactly one |
| Taskset evaluation lacks held-out reference data | Stored tasks do not carry grader-only `reference` | Use inline `AgentEvalTaskInput` when the metric needs held-out per-task data |
| Runner target fails to start | The runtime dependency, CLI, config, credentials, or Docker access is missing | Check the selected Codex, Fabric, or Harbor runner prerequisites |

## Debug in the smallest scope

1. Validate one expected pass and one expected failure.
2. Inspect row scores or task trials before aggregates.
3. Reproduce metric behavior with the standalone SDK before diagnosing platform infrastructure.
4. For submitted jobs, inspect terminal status and error details.
5. Retry only the failed row, task, or runner configuration when possible.
