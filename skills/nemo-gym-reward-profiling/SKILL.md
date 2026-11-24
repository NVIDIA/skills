---
name: nemo-gym-reward-profiling
description: >-
  Use when deciding which NeMo Gym tasks to train on with RL: collecting
  repeated rollouts with gym eval run, profiling their rewards with gym eval
  profile, reading per-task pass rates and reward spread, finding tasks that
  never or always pass, and reporting coverage honestly for partial runs. Not
  for benchmark scoring runs, comparing runs or explaining score gaps, planning
  benchmark repeats, or debugging crashed runs.
license: Apache-2.0
metadata:
  author: NVIDIA
  version: "1.0.0"
  tags:
    - nemo-gym
    - reward-profiling
    - rl-training
    - rollouts
---

# NeMo Gym Reward Profiling

## Purpose

Before RL training, run every task several times and look at how its reward changes between repeats. Group-based RL such as GRPO learns from the differences between repeats of the same task, so a task that gets the same reward every time, whether always solved or never solved, teaches the model nothing. This skill turns `gym eval profile` output into that decision and keeps every number honest when some rollouts are missing.

## When to Use This Skill

- "Which tasks should I keep for RL?", "filter my dataset by difficulty", "is this environment ready to train on?"
- Per-task pass rates, reward spread or variance across repeats, tasks that never or always pass.
- Running `gym eval profile`, reading its output files, or fixing its errors.
- Deciding whether the numbers from a partial or interrupted run can be trusted.

Hand off when the main job is different:

- Running, resuming or monitoring an evaluation job, or producing a benchmark score: the evaluation-running skill (`run-and-babysit-eval` in the Gym repo).
- Comparing two runs, or explaining why a score differs from a published number: the results-analysis skill (`nemo-gym-score-triage` in the Gym repo).
- Choosing repeat counts to measure a benchmark score precisely: `nemo-gym-repeat-planning` in the Gym repo.
- A crash, a stack trace, or servers that never start: `nemo-gym-debugging` in the Gym repo.

## Prerequisites

- NeMo Gym (`pip install "nemo-gym>=0.7"`) for `gym env start`, `gym eval run` and `gym eval profile`.
- `uv` on the `PATH` (`pip install uv`) to start servers: `gym env start` builds each server's environment with it.
- A model endpoint and its API key, only when rollouts still have to be collected.
- Python for the bundled script. It needs no packages and no Gym checkout.

## Instructions

### 1. Collect repeated rollouts

Skip this step if the user already has a rollouts file and the `<stem>_materialized_inputs.jsonl` written next to it.

Put the model endpoint in `env.yaml` in the working directory. If the file already exists, add or update these keys rather than replacing the file:

```yaml
policy_base_url: https://integrate.api.nvidia.com/v1
policy_api_key: <your key>
policy_model_name: <model id>
```

Then start the servers, wait until every server shows ✓, and collect:

```bash
# Terminal 1 (or run it in the background). Use --resources-server <name> for an environment shipped with
# nemo-gym, or --config <file> for your own. vllm_model talks to any OpenAI-compatible chat endpoint
# (vLLM, NVIDIA API); openai_model needs the OpenAI Responses API.
gym env start --resources-server <name> --model-type vllm_model

# Terminal 2. --agent takes the first name on the agent's line in gym env status, such as mcqa_simple_agent.
gym env status
gym eval run --no-serve --agent <agent> \
    --input tasks.jsonl --output results/rollouts.jsonl \
    --num-repeats 8 --concurrency 16
```

`gym eval run` writes `results/rollouts.jsonl` (one line per saved rollout) and `results/rollouts_materialized_inputs.jsonl` (one line per planned rollout, that is every task times every repeat, with `_ng_task_index` and `_ng_rollout_index`). Without `--no-serve`, `gym eval run` starts the servers itself but only runs the environment's prepared data; it rejects `--input`. For long runs, cluster jobs and resuming, use the evaluation-running skill.

Use enough repeats. A task with pass rate p gets the same reward on all n repeats with probability pⁿ + (1 − p)ⁿ, so hard tasks look unsolvable when n is small:

| True pass rate | Looks constant at 4 repeats | at 8 | at 16 |
|---|---|---|---|
| 5% | 81% | 66% | 44% |
| 10% | 66% | 43% | 19% |
| 25% | 32% | 10% | 1% |
| 50% | 13% | 1% | <1% |

Use at least 8 repeats when choosing training tasks, and more when the environment has many hard tasks.

### 2. Profile

```bash
gym eval profile \
    --inputs results/rollouts_materialized_inputs.jsonl \
    --rollouts results/rollouts.jsonl
```

It writes three files next to the rollouts: `rollouts_reward_profiling.jsonl` (one row per task), `rollouts_agent_metrics.json` (one entry per agent) and `rollouts_repeat_level_metrics.json` (one entry per repeat). See `references/output-format.md` for every field.

If some planned rollouts were never saved, it stops with `Materialized input rows and rollout results do not have matching rollout keys. Missing rollout results for N materialized input rows.` Then either:

- finish the run: rerun the same `gym eval run` command with `--resume`. It runs the planned rollouts that have no saved result; which recorded failures it retries depends on the version and the kind of failure; or
- profile what was saved: add `++allow_partial_rollouts=True`, and report the gap as in step 3.

### 3. Check coverage before reading any reward

Run the bundled script from this skill's folder on the profile. It needs only Python and the files, so it also works where `gym` is not installed:

```bash
python3 <skill folder>/scripts/summarize_profile.py results/rollouts_reward_profiling.jsonl
```

It reads the materialized inputs and agent metrics beside the profile, prints coverage first (measured, never saved and masked rollouts), then sorts the tasks into groups (see Examples). It catches what the profile file alone hides:

- **Tasks with no measured rollout.** nemo-gym 0.7.0 and earlier leave them out of `_reward_profiling.jsonl` entirely. Later versions (the main branch today) keep a row with `num_rollouts: 0` and no reward fields, except for a task whose every planned rollout was saved and masked. The summary that `gym eval profile` prints says `N without rollouts dropped from output` either way. Count tasks from the materialized inputs, never from the profile file alone.
- **The headline mean.** `mean/reward` in `_agent_metrics.json` averages measured rollouts only. Report it with its coverage, for example "0.51 over 41 of 48 planned rollouts; task 10 not measured", never as the environment's pass rate. When several agents ran the same tasks, each agent has its own mean. The `_aggregate_metrics.json` that `gym eval run` writes can count some failed or missing rollouts as 0, depending on the version and settings such as `count_failure_classes_as_zero`, so its mean can differ from the profile's.
- **Per-repeat counts.** In `_repeat_level_metrics.json`, `missing_count` counts only tasks that were measured in another repeat. A task missing from every repeat is never counted there, so a repeat can cover fewer tasks than planned while its `missing_count` stays 0. Compare `sample_count` with the number of planned tasks.
- **Masked rollouts.** An environment can mark a saved result as not a real measurement. `gym eval profile` leaves masked rollouts out of every per-task count and statistic, so a row's `expected_num_rollouts` is its planned count minus its masked rollouts. The printed completion line still counts them as saved, and `_agent_metrics.json` reports `coverage/masked_rollouts`. In 0.7.0 and earlier, `--resume` treats a masked result as saved and does not run it again; collect those tasks in a new run with its own `--output`.
- **Failed rollouts.** A failed `/run` request stops the run unless `+route_failures_to_sidecar=true` is set (after 0.7.0 it is written to `<stem>_failures.jsonl` first); with the option, the failure goes to that file and the run continues. Failures that the agent or environment returns as a result go to that file too. Either way they are not in `rollouts.jsonl`, so the profile counts them as never saved. Read that file before rerunning: `--resume` skips failures recorded as final.

### 4. Choose the training tasks

| What the profile shows | What it means | What to do |
|---|---|---|
| Reward varies across repeats | The model solves it some of the time | Keep: these carry the learning signal |
| Same top reward on every repeat | Already solved | Drop from RL training |
| Same zero reward on every repeat | Never solved | Check before dropping (below) |
| Same middle reward on every repeat, such as 0.5 each time | No spread, even though the mean looks medium | No signal; treat like a constant task |
| One measured rollout | Spread unknown; `std/reward` is 0 for any single value | Collect more repeats |
| Partial (fewer measured than planned) | Statistics from fewer repeats: rollouts never saved or masked | Keep the decision tentative; collect the missing rollouts |
| No measured rollout | Not measured | Rerun before deciding; never count it as 0 or 1 |

Judge spread with `min/reward` and `max/reward` (or `std/reward` with `num_rollouts`), not with `mean/reward`: two tasks with mean 0.5 can be 1, 0, 1, 0 (useful) or 0.5 every time (useless).

Before dropping a task that never passes, read two or three of its rollouts. Join `rollout_infos` in the profile row to full rows in `rollouts.jsonl` on `_ng_task_index` and `_ng_rollout_index`.

- The model gives the same answer every time and it differs from the expected answer: check the label and the verifier first. A wrong label, or a verifier that rejects a correct answer in another format, looks exactly like a hard task.
- Outputs are empty, cut off at the token limit, or never reach a final answer: the setup is wrong (limits, prompt, parser), not the task.
- Answers are wrong in different ways: the task is probably too hard for this model now. It may become useful as the model improves.

### 5. Report

Lead with coverage, then the decision. Take every number from the files or the script output, and say when a number covers only part of the run. See Examples.

## Examples

A math environment, 12 tasks with 4 planned repeats each; the job stopped early and was profiled with `++allow_partial_rollouts=True`. The script prints:

```text
Coverage: 41 of 48 planned rollouts measured (85.42%): 7 never saved, 0 masked.
Tasks: 12; 9 complete, 2 partial, 1 with no measured rollout.
  Not in the profile file (no saved rollout): 10
Mean reward over measured rollouts: 0.5122 (says nothing about tasks with none).
* = partial: fewer measured rollouts than planned.

Reward varies across repeats (learning signal): 5, 6, 7, 8, 9*
Same reward (0.0) on every repeat (no signal): 3, 4, 11*
Same reward (1.0) on every repeat (no signal): 0, 1, 2
No measured rollout (not measured): 10
```

Task 4's rollouts all answer 400 while its expected answer is 401. The report:

```text
Coverage: 41 of 48 planned rollouts measured (85%). Task 10 has none; tasks 9 and 11 are partial.
Keep (reward varies): 5, 6, 7, 8, and 9 (only 2 of 4 repeats measured).
Drop, always solved: 0, 1, 2.
Never solved: 3, 4, 11 (3 of 4 measured). Task 4: the model answers 400 every time but the expected answer is 401; check the label before dropping it.
Not measured: 10. Rerun with --resume before deciding.
Mean reward 0.51 over the 41 measured rollouts only.
```

## Available Scripts

| Script | Purpose | Arguments |
|---|---|---|
| `scripts/summarize_profile.py` | Coverage and task groups from `gym eval profile` output, every number read from the files | `<stem>_reward_profiling.jsonl`, optional `--inputs <stem>_materialized_inputs.jsonl` (found automatically beside the profile), optional `--json` (lists every task; the text report lists up to 20 per group) |

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `gym env start` stops with `model endpoint(s) never answered` (after waiting up to 600 s by default) | Nothing is listening at `policy_base_url`: wrong URL, or the server is down or still starting | Check the URL in `env.yaml`, and check that `curl -i <policy_base_url>/models` gets an HTTP answer. A hostname that does not resolve is reported as such instead |
| `gym eval run` saves no rollout: it fails with `ClientResponseError: 500, message='Internal Server Error'` on `/run`, or, when failures are recorded instead of stopping the run (`+route_failures_to_sidecar=true`, or by default for many environments after 0.7.0), ends with `None of the N dispatched rollouts produced a result` | The model endpoint rejected the requests. The `gym env start` log shows `[upstream_request_failed] ... status=401` or `403` for a bad or missing API key | Fix `policy_api_key`, then rerun |
| The `gym env start` log shows `status=404` for `.../v1/responses` | The endpoint or model does not serve the Responses API (some NVIDIA API models serve only chat completions) | Use `--model-type vllm_model` |
| Sandboxed environment: the run stops, or rollouts go missing, with sandbox or container errors in the logs | The sandbox was not ready or not reachable (Docker not running, provider unreachable, container still starting). After 0.7.0, recorded failures carry `_ng_failure_class` values that start with `provider_` | Fix the sandbox provider, then rerun with `--resume`. Use `nemo-gym-debugging` in the Gym repo for the details |
| `gym eval profile`: `do not have matching rollout keys` | Planned rollouts were never saved | Rerun with `--resume`, or add `++allow_partial_rollouts=True` and report coverage |
| `gym eval profile`: `Rollout results contain rows with no matching materialized input rows` | The inputs file belongs to a different run | Use the `<stem>_materialized_inputs.jsonl` written next to the same rollouts file |
| `gym eval profile`: `Duplicate result row for rollout key` (or `Duplicate materialized input row`) | Two runs wrote into one file. Some versions after 0.7.0 also keep the superseded rows of retried masked results | Collect each run into its own `--output`. `gym eval profile` cannot read a run with superseded rows yet; say so rather than editing the file |
| With `++allow_partial_rollouts=True`: `Reward profile completion: 0/N rollout rows` and an empty profile | Nothing was saved | Fix collection first; there is nothing to profile |
| `UserWarning: N repeat(s) are missing rollouts for some tasks` | Repeats have different numbers of measured rollouts (missing or masked rollouts) | Expected for a partial or masked run; report coverage as in step 3 |
| `UserWarning: wandb is not installed, so the histogram/* stats are being skipped` | wandb is optional after 0.7.0 | Harmless: the profile files do not include those statistics anyway |
| `gym: command not found` | NeMo Gym is not installed in this environment | `pip install "nemo-gym>=0.7"` to collect or profile. The summary script works on existing profile files without it |

## Limitations

- The script summarizes `gym eval profile` output; it does not recompute statistics from the rollouts.
- Telling masked rollouts from rollouts never saved needs the materialized inputs and the masked total in `_agent_metrics.json`, which counts only rows that name an agent (rows of Environment Server tasksets do not). When either is missing or incomplete, the script reports that split as unknown.
- When the same task ran under several agents or Environment Servers, the script reports each task and run separately and names each run as the materialized inputs do: by its agent, or by its Environment Server when several servers front one agent. If masking removed every rollout of a run, `gym eval profile` can name the other runs differently, so the profile file and `_agent_metrics.json` may use other names. When a row with no measured rollout could belong to more than one run, the script reports whether those runs are in the profile file as unknown.
- A profile shows which tasks give a signal, not why a task fails. Read the rollouts for that.
