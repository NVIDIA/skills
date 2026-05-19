---
name: nemo-gym-reward-profiling
description: >-
  Use to help users get started with Nemo Gym reward profiling. Covers the basic
  ng_run, ng_collect_rollouts, and ng_reward_profile workflow, repeated rollouts,
  materialized inputs, rollout JSONL artifacts, task and rollout identity, output
  inspection, partial profiling, and rollout_infos. For failed jobs, prefer
  nemo-gym-debugging.
---

# Nemo Gym Reward Profiling

## Invocation Check

Use this skill when the user wants to run, understand, or lightly modify Nemo Gym reward profiling. Keep the answer oriented around the normal workflow:

`ng_run` starts model/resource servers, `ng_collect_rollouts` writes rollout artifacts, and `ng_reward_profile` generates profiling output from those artifacts.

If the user is primarily debugging a failed job or stack trace, use the `nemo-gym-debugging` skill first.

## Basic Workflow

1. Identify the environment config paths and input JSONL.
2. Start Gym servers with `ng_run`.
3. Collect rollouts with `ng_collect_rollouts`; this writes `rollouts.jsonl` and `*_materialized_inputs.jsonl`.
4. Run `ng_reward_profile` on the materialized inputs and rollout JSONL to generate `*_reward_profiling.jsonl`.
5. Inspect line counts and profile rows.

Repeated rollouts are the main profiling lever. `num_repeats=1` is valid, but per-task averages and variance are only meaningful with multiple rollouts per task.

## Core Concepts

- `*_materialized_inputs.jsonl`: expanded collection inputs after repeat expansion, agent defaults, and task/rollout id assignment.
- `rollouts.jsonl`: one completed rollout/result per materialized input row.
- `*_reward_profiling.jsonl`: one summarized profile row per original task with at least one completed rollout.
- `_ng_task_index`: original task/sample id.
- `_ng_rollout_index`: repeated rollout id for that task.
- `rollout_infos`: compact per-rollout info inside each task profile row, including reward, token usage, and numeric rollout metrics when available.

Keep reward-to-length or reward-to-token analysis keyed by both `_ng_task_index` and `_ng_rollout_index`.

## Reference Loading

Load references only when the user needs that detail:

- Read `references/quick-start.md` for a generic command template and the minimal run sequence.
- Read `references/output-format.md` to explain materialized inputs, rollout JSONL, reward profile rows, `rollout_infos`, and partial profiling.

## Practical Defaults

- Treat `ng_reward_profile` as the reward profiling step; rollout collection does not write reward profile files.
- Run strict profiling by default. If rollout collection stopped early, use `++allow_partial_rollouts=True` to profile completed rollouts and drop original input rows with no completed rollout.
- Trust the target checkout's CLI help and `nemo_gym/reward_profile.py` over memory if flags differ.
