# Evaluation Notes

`evals.json` has 23 cases: 13 that need this skill and 10 near misses that belong to other skills or to no skill. Every case is a full agent session in both arms (with and without the skill). Every expected number can be read or computed from the files a case stages.

## Questions

- Whether a candidate is better when the interval for the difference contains 0.
- Whether a task that flipped from pass to fail shows a regression.
- A candidate whose gain comes from failed rollouts that its score leaves out, while its health summary says healthy.
- A baseline that counted infrastructure failures as zeros, against a candidate that had none.
- Two runs over different task sets (`--limit`).
- A comparison of single-repeat runs, which has no interval.
- A gap to a published number that comes from a low output-token limit.
- Tasks that never pass: a correct answer in a format the grader does not read, and genuine wrong answers.
- Tasks that sometimes pass because the answer sometimes lands only in the reasoning.
- A pass@8 of 100% that leaves out the tasks that lost rollouts.
- A health check that says healthy while every rollout is unobserved.
- Judge failures in a score (no files).
- A managed run whose rollouts file holds a late superseded result and an unanswered retry, so grouping its raw rows miscounts.

## Negative Cases

Near misses that mention NeMo Gym, evaluation or statistics but need a different skill: running a benchmark, choosing RL training tasks, planning repeats before a run, a run that cannot connect to its servers, adding a benchmark, adding metrics to an environment, a flat training reward, a Welch interval in Python, a slow evaluation, and resuming a killed run.

## Fixtures

`files/` holds real output of `gym eval run` and `gym eval compare` from nemo-gym's main branch (which reports its version as 0.7.0rc0). The runs used the `mcqa` resources server and its simple agent against a local stand-in for an OpenAI-compatible endpoint that answers from a fixed, seeded script: a correct or wrong `\boxed{X}` letter per task and repeat, `Answer: X` without a box, an answer only in the reasoning, output that stops at the token limit, or an HTTP 500 on chosen calls. There are 24 four-option questions graded with `strict_single_letter_boxed` (only `\boxed{X}` counts) and 8 repeats unless stated; each run used `gym eval run --no-serve --agent mcqa_simple_agent --num-repeats 8` plus the options below. Local paths were replaced with `results/<run>/...`. To keep the folder small, each case stages only the files it needs, only the JSON form of the comparison reports is kept, and null fields inside the `responses_create_params` and `response` objects of rollout rows, and inside `_ng_failure_partial_response` of failure rows, were dropped. Every other field is as written.

- `base/`: baseline, complete; tasks 0 to 7 always pass, 8 to 11 never pass, 12 to 23 sometimes pass.
- `cand/`: candidate on the same tasks, complete.
- `candfail/`: candidate collected with `+route_failures_to_sidecar=true` while the endpoint returned HTTP 500 for some calls; 19 rollouts on tasks 8 to 11 failed with `model_request_error` and are left out of its score.
- `base_zero/`: baseline collected with `+route_failures_to_sidecar=true` and `+count_failure_classes_as_zero=[model_request_error]`; 16 rollouts on tasks 0 to 3 failed and count as zeros.
- `limit/`: candidate collected with `--limit 18`.
- `single_a/`, `single_b/`: one repeat each.
- `trunc/`: baseline collected with `--max-output-tokens 512`; tasks 12 to 23 need longer outputs and stop at the limit.
- `diag/`: tasks 8 and 9 answer with the correct letter as `Answer: X`; in tasks 16 to 19, failing repeats put the answer only in the reasoning.
- `managed/`: an input of questions 0, 9, 16 and 20 (renumbered 0 to 3), collected with `--num-repeats 2` and `+route_failures_to_sidecar=true`, then edited as a collector timeout and a crash would leave the run: in task 1, repeat 1, the retry (attempt 1, reward 0) is followed by a late result of attempt 0 with reward 1, and in task 0, repeat 1, attempt 1 is reserved in the manifest and never answered. The coverage file was rewritten from Gym's `RolloutStore` after the edits. Its rows keep their null fields.
- `compare_base_cand/`, `compare_base_candfail/`, `compare_base_zero_cand/`, `compare_base_limit/`, `compare_single/`: `gym eval compare` reports for those pairs, at the default 95%.

No model calls were captured, so every health verdict is `unobserved`. The examples in `SKILL.md` use other runs, so the skill text does not contain these answers.
