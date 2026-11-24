# Evaluation Notes

`evals.json` has 17 cases: 13 that need this skill and 4 near misses that belong to other skills. Every case is a full agent session in both arms (with and without the skill), so the set is kept small enough to fit the CI time limit. Expected outputs and assertions describe results that an agent without the skill could also reach.

## Questions

- Choosing RL training tasks from a partial run, and checking whether its headline pass rate can be quoted.
- What `missing_count` in the per-repeat metrics does and does not count.
- Two tasks with the same mean reward but different spread, and a task with a single saved rollout.
- The `gym eval profile` command, its output files, and its error for missing rollouts.
- Checking a new environment before RL from a pip install, and how many repeats hard tasks need.
- A question about one output file of a run that stopped early, where that file looks complete: the profile, the agent metrics or the per-repeat metrics.
- Choosing training tasks from a run whose environment masked some rollouts, while the completion line says 100%.

## Negative Cases

Near misses that mention NeMo Gym but belong elsewhere: explaining a score gap against a model card, a crashed run, planning repeats to compare checkpoints, and latency profiling of a server.

## Fixtures

`files/` holds four small synthetic runs in the format that `gym eval run` writes. All were profiled with `gym eval profile` from nemo-gym 0.7.0, which leaves tasks with no saved rollout out of the profile file; their JSON metrics files were then indented, since 0.7.0 writes them on one line. Each run keeps its rollouts and materialized inputs, even where no case stages them, so the profiles can be reproduced. Cases stage them into `/workspace/input/`.

- `math_run/`: 16 math tasks with 8 planned repeats, scored 0 or 1 by an exact string match of the final answer; the job stopped at 98 of 128 rollouts and was profiled with `++allow_partial_rollouts=True`. Tasks 0 to 3 always pass. Task 4 never passes, with different wrong answers. Task 5 never passes because the exact match rejects `\frac{1}{6}` for the label `1/6`. Tasks 6 to 10 vary. Task 11 varies with 3 of 8 repeats saved. Task 13 has one saved rollout. Task 15 never passes with 6 of 8 saved. Tasks 12 and 14 have no saved rollout.
- `if_run/`: 8 instruction-following tasks with 4 repeats, rewarded by the fraction of 4 instructions followed; complete. Task 1 scores 0.5 every time and task 2 alternates between 1 and 0.
- `code_run/`: 12 code tasks with 8 planned repeats, scored 1 when all unit tests pass. The job ran the tasks in order and stopped after task 9, so tasks 0 to 9 are complete and tasks 10 and 11 have no saved rollout (80 of 96 rollouts); it was profiled with `++allow_partial_rollouts=True`. Every row of its profile looks complete. Tasks 0 to 2 always pass, tasks 3 and 4 never pass, and tasks 5 to 9 vary. Cases stage one output file at a time.
- `tool_run/`: 8 sandboxed terminal tasks with 8 repeats, every rollout saved, but the environment masked 14 rollouts whose sandbox failed: 4 of task 2, all 8 of task 5 and 2 of task 6. Of the measured rollouts, task 0 always passes, task 2 passes all 4, task 3 never passes, and tasks 1, 4, 6 and 7 vary.

The example in `SKILL.md` uses a different run, so the skill text does not contain these answers.
