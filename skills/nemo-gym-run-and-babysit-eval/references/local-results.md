<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Check a direct evaluation result

A direct `gym eval run` needs a score reconciled with its saved population and grading evidence. An exit code of
zero, a complete output file or healthy accounting alone does not establish a trustworthy benchmark result.

## Read the actual artifacts

For `--output results/ifeval/full/artifacts/rollouts.jsonl`, read:

| File beside the rollouts | Check |
| --- | --- |
| `rollouts_resolved_config.yaml` | Exact model, harness, data, prompts, repeats, limits and grading configuration. |
| `rollouts_materialized_inputs.jsonl` | Expected task/repeat identities. |
| `rollouts_coverage.json` | Expected, measured, masked, unscored, failed, missing and omitted counts; `complete`, `reconciled` and `identity_verified`. |
| `rollouts.jsonl`, `rollouts_failures.jsonl` | Selected answers, rewards and failed attempts; history is not additional independent samples. |
| `rollouts_aggregate_metrics.json` | Actual metric and correct `agent_ref.name` and environment entry. |
| `quality_summary.json`, `rollout_verdicts.jsonl` | Inventory accounting, evidence quality, observation gaps and failure causes. |
| `rollouts_run_summary.json`, for task targets | This run's files, stale/missing outputs, per-task rewards and verifier logs. |

Sidecar stems follow the chosen rollout filename. The quality files retain their fixed names in its directory.
Use separate directories for independent runs and preserve artifacts before any diagnostic operation that rewrites
reports. Do not edit saved rows, masks, coverage or rewards to make a result pass.

## Reconcile population and health

1. Confirm the resolved configuration represents the intended full task set and canonical repeats. Compare the
   materialized inventory with coverage and the aggregate's own scoring denominator. `+require_complete=true` checks
   settled collection work: a fully masked run or an unscored completion can still be complete. Explicit omissions
   can also satisfy the completion policy. Report these counts separately; they are not full measured coverage.
2. Inspect `run.artifacts.healthy` and `unhealthy_reasons` for run accounting. Read `run.verdicts`, per-rollout findings
   and task flags separately for evidence quality. An `unobserved` check lacked evidence; it did not pass.
3. Examine recorded verifier/grading operations and failure kinds. Judge, verifier, sandbox, harness and transport
   faults can invalidate measurement even when every selected rollout has a row. Settled model failures, including
   truncated or empty answers under the benchmark's limits, can remain scored model outcomes. Do not regenerate
   those answers or relax the protocol to raise the score.
4. If the artifacts contain `_ng_run_health_metrics` in the aggregate or `run.stats.acceptance_health` in the quality
   summary, check their source-artifact binding and known/count fields. `health/trial/known: 0` means evaluation health
   is unknown; it is not a passing zero failure rate. When known, read `health/trial/passed`, the failure count and
   its expected-trial denominator together; a recorded failed verdict must not be called healthy. Missing metadata
   also leaves that verdict unavailable.

Collection normally writes quality reports. A diagnostic recheck uses the actual filename and capture directory:

```bash
HEALTH_REVIEW=$(mktemp -d "$PWD/results/ifeval/health-review.XXXXXX")
cp -R results/ifeval/full/artifacts "$HEALTH_REVIEW/artifacts"
gym eval health-check "$HEALTH_REVIEW/artifacts" --rollouts-file rollouts.jsonl \
  --model-call-capture-dir "$PWD/results/ifeval/full/model-calls" --json
```

This refreshes the copied quality reports but changes no rollout record or reward; the original reports remain saved.
Its successful process exit means the check ran, not that grading was verified. For task runs with relative artifact
references, preserve the complete run layout in the diagnostic copy.

Direct runs do not have a submitted-run recovery gate or job ID. Do not invent `gate_task.status: SUCCESS`,
`evaluation_health: PASS` or an export receipt. If exact evaluation-side fault accounting or grading traceability
cannot be established from the recorded evidence, report the measured score with that limitation and withhold a
verified claim. A submitted run instead uses the authoritative [Slurm acceptance checks](../SKILL.md#6-health-and-acceptance).

## Reconcile and report the score

Read the benchmark's documented headline key and scale from the matching aggregate entry. For IFEval in this
preview, select `agent_ref.name: ifeval_instruction_following_simple_agent` and `agent_metrics["mean/reward"]`.
With its binary all-instructions grading, independently compare the selected measured rewards with the saved mean;
report the numerator and denominator as well as the fraction or percentage. For another benchmark, follow its own
aggregation: mean reward is not universally the published headline metric.

IFEval's language checks can be nondeterministic at this preview revision. Retain their actual per-instruction
verdicts when comparing runs; identical answers need not produce identical language-check results.

Use only the selected attempts under the saved selection policy. Compare coverage `attempts` with `expected`;
JSONL line order alone does not determine the selected attempt. Gym's public `RolloutStore` reader
(`from nemo_gym.rollouts import RolloutStore`, with `Path` from `pathlib`) loads `store = RolloutStore.read(Path(output))`.
If `store` is `None`, stop: there is no saved manifest or materialized inventory for this reader. Otherwise,
`store.selected("success")` returns manifest-selected success payloads, including any masked ones; inspect selected failures and omissions separately
and apply the benchmark's mask/aggregation policy. Bind each row to its saved task/repeat identity
(`_ng_task_index`, `_ng_rollout_index`, and the run's agent/environment scope). Duplicate rows and superseded attempts are not
extra samples. Do not silently turn masked, unscored or omitted tasks into zeros, and account explicitly for any
failures the configured aggregation counts as zeros. A custom metric may declare a narrower population through its
`metric_coverage/*` fields; use that denominator instead of assuming all recorded rows contributed.

Return the exact Gym preview commit, model and harness, benchmark or task-set identity, task and repeat counts,
metric key and value, scoring denominator, expected/measured coverage, failure/mask/omission counts, evaluation-health
and quality-evidence limits, and artifact locations. A task limit, reduced repeat count or unnamed partial subset
supports a smoke/subset result, not a full benchmark score. A stub endpoint test demonstrates protocol wiring and
artifact arithmetic; it is not model-quality or benchmark-performance evidence.
