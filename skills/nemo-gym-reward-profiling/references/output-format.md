# Reward Profiling Files

`gym eval run --output results/rollouts.jsonl` and `gym eval profile` write these files. Every name starts with the rollouts file's stem (`rollouts` here).

| File | Written by | One line or entry per |
|---|---|---|
| `rollouts_materialized_inputs.jsonl` | `gym eval run` | Planned rollout: every task times every repeat |
| `rollouts.jsonl` | `gym eval run` | Saved rollout |
| `rollouts_failures.jsonl` | `gym eval run` | Recorded failure (see below) |
| `rollouts_reward_profiling.jsonl` | `gym eval profile` | Task |
| `rollouts_agent_metrics.json` | `gym eval profile` | Agent (or run label) |
| `rollouts_repeat_level_metrics.json` | `gym eval profile` | Repeat |

Two keys identify a rollout everywhere: `_ng_task_index` (the task, numbered from 0 in input order unless the input rows already carry one; identical input rows share one index) and `_ng_rollout_index` (the repeat; when several agents run the same task, it keeps counting across them). Join files on both.

A failed `/run` request stops the run unless `+route_failures_to_sidecar=true` is set (after 0.7.0 it is written to the failures file first); with the option, the failure goes to the failures file and the run continues. Failures that the agent or environment returns as a result are written there too. Rollouts in that file are not in `rollouts.jsonl`.

## Per-task rows: `<stem>_reward_profiling.jsonl`

- `_ng_task_index`: the task.
- `agent_ref`: present only when the same task ran under more than one run; the file then has one row per task and run. It names the agent, or the Environment Server when several servers front one agent. Masked rollouts are left out first, so a run whose every rollout was masked can remove `agent_ref` from the file or change how the other runs are named.
- `sample`: one materialized input row for the task, without the two index keys.
- `num_rollouts`: measured rollouts summarized in this row.
- `expected_num_rollouts`: planned rollouts for the task, from the materialized inputs, minus masked ones.
- `missing_num_rollouts`: `expected_num_rollouts - num_rollouts`, the rollouts that were never saved.
- `reward_profile_completion_pct`: `100 * num_rollouts / expected_num_rollouts`.
- `mean/reward`, `median/reward`, `min/reward`, `max/reward`, `std/reward`: reward across this task's measured repeats. `std/reward` is the sample standard deviation (n − 1 in the denominator) and is 0 when only one rollout was measured.
- The same five statistics for every other top-level numeric field of a rollout row: token counts from `response.usage` (`input_tokens`, `output_tokens`, `total_tokens`) and the numeric fields the verifier returns.
- `rollout_infos`: one compact record per measured rollout, sorted by `_ng_rollout_index`: `rollout_id` (`"<task>:<repeat>"`), the two index keys, `reward`, token usage, and the other numeric fields. Full responses stay in `rollouts.jsonl`.

There is no confidence interval per task: a single task's repeated outcomes are often 0 or 1, and a normal-theory interval does not fit them.

A task with no measured rollout:

- nemo-gym 0.7.0 and earlier write no row for it.
- Later versions (the main branch today) write a row with only `_ng_task_index` (and `agent_ref` when several agents ran the task), `num_rollouts: 0`, `expected_num_rollouts`, `missing_num_rollouts` and `reward_profile_completion_pct: 0.0`.
- In 0.7.0 and later, a task whose every planned rollout was saved and masked has no row, and when no rollout at all was measured the file is empty.

## Per-agent entries: `<stem>_agent_metrics.json`

- `agent_ref`: the agent, or the Environment Server as above.
- `mean/reward`, `median/reward`, `min/reward`, `max/reward`, `std/reward`: over all measured rollouts of the agent, each rollout counted once. Tasks with more measured repeats weigh more.
- `mean_across_repeats/mean/reward` and its `median_`, `std_`, `se_`, `min_`, `max_`, `ci_low_95_` and `ci_high_95_` variants: statistics of the per-repeat mean reward, each repeat counted once. Read the key from the outside in: `se_across_repeats/mean/reward` is the standard error, across repeats, of each repeat's mean reward. `mean_across_repeats/mean/reward` equals `mean/reward` when every repeat has the same number of measured rollouts.
- `coverage/measured_rollouts`, `coverage/masked_rollouts`, `coverage/measured_tasks`, `coverage/fully_masked_tasks`: present only when the environment masked some results, and counted only for rows that name an agent. Masked results are saved but are not measurements: they are left out of every statistic and every per-task count, while the completion line that `gym eval profile` prints still counts them.

## Per-repeat entries: `<stem>_repeat_level_metrics.json`

One entry per agent and `_ng_rollout_index`; the file holds `[]` when there is only one repeat.

- `sample_count`: tasks with a measured rollout in this repeat.
- `missing_count`: tasks measured in some other repeat but not in this one. A task missing from every repeat is not counted here, so this stays 0 for it.
- For every numeric field: `mean/`, `median/`, `std/` (n − 1), `sem/`, `min/`, `max/`, `p25/`, `p75/`, and `ci_low_95/` and `ci_high_95/` (Student's t, left out when only one task was measured).

## Warnings

- `UserWarning: N repeat(s) are missing rollouts for some tasks ...`: repeats cover different task sets, because of missing or masked rollouts, so their statistics are not directly comparable. It fires off `missing_count`, so a task missing from every repeat does not trigger it.
- A zero standard error (every value identical) collapses the interval to the mean and says so.
- A 95% interval on a 0 to 1 reward can fall below 0 or above 1 with few tasks. That signals too few tasks or repeats, not a bug.
- After 0.7.0, where wandb is optional, `wandb is not installed, so the histogram/* stats are being skipped` is harmless: those statistics never reach the profile files.

## What `gym eval profile` prints

```text
Reward profile completion: 41/48 rollout rows (85.42%)
Input rows: 12 total; 9 complete; 2 partial; 1 without rollouts dropped from output.
Masked from quality metrics: 0 rollout rows (kept in completion accounting above).
```

The completion lines are measured against the materialized inputs, so they are the trustworthy coverage numbers; they count masked rollouts as saved. "Dropped from output" means the task has no statistics; after 0.7.0 it may still have a row with `num_rollouts: 0`.
