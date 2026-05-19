# Reward Profiling Output Format

Reward profiling is built around joining materialized inputs with rollout results by task and rollout identity.

## Files

- `*_materialized_inputs.jsonl`: expanded inputs after repeat expansion. Each row should have `_ng_task_index` and `_ng_rollout_index`.
- `rollouts.jsonl`: completed rollout results. Each row should have matching `_ng_task_index` and `_ng_rollout_index`.
- `*_reward_profiling.jsonl`: task-level summaries produced by `ng_reward_profile`.
- `*_agent_metrics.json`: agent/global aggregate metrics.

## Task Profile Rows

Task-level profile rows include:

- `_ng_task_index`: original task/sample id.
- `sample`: representative materialized input row for the task, with task/rollout ids removed from the sample copy.
- `num_rollouts`: number of rollout results summarized for the task.
- `expected_num_rollouts`: number of materialized rollout rows expected for the task.
- `missing_num_rollouts`: number of expected rollout rows missing from the profile.
- `reward_profile_completion_pct`: percent of expected rollout rows included for the task.
- `rollout_infos`: compact per-rollout records sorted by `_ng_rollout_index`.
- aggregate metric keys such as `mean/reward`, `max/reward`, `min/reward`, `median/reward`, and `std/reward`.
- token usage aggregate keys such as `mean/input_tokens`, `std/output_tokens`, and `mean/total_tokens`, when those fields are present in `response.usage`.

`rollout_infos` are intentionally compact. They can include:

- `rollout_id`, usually `task_idx:rollout_idx`
- `_ng_task_index`
- `_ng_rollout_index`
- `reward`
- token usage fields from `response.usage`, when present
- numeric verifier/result fields, when present

Full model responses stay in `rollouts.jsonl`. Join back to full rows with `(_ng_task_index, _ng_rollout_index)` when needed.

## Partial Profiles

Strict profiling is the default. If materialized inputs and rollout results do not have the same rollout keys, `ng_reward_profile` fails and suggests:

```bash
++allow_partial_rollouts=True
```

With partial profiling enabled, rows with no matching materialized input still fail, but missing rollout results are allowed. The profile includes original input tasks with at least one completed rollout and drops original input tasks with zero completed rollouts.

At the end, the command prints rollout completion and input-task status counts:

- completed rollout rows / expected rollout rows and percentage
- complete input tasks
- partial input tasks
- input tasks without rollouts that were dropped
