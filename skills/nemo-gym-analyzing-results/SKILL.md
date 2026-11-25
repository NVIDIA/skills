---
name: nemo-gym-analyzing-results
description: >-
  Use when finished NeMo Gym evaluation results need explaining: whether a
  score change is real or noise, whether two runs or a run and a published
  number are comparable, what the confidence interval and task flips from gym
  eval compare mean, which tasks failed and why, and what to fix. Not for
  running, resuming or speeding up evaluations, adding benchmarks or metrics,
  choosing RL training tasks or fixing training, planning repeats before a
  run, or debugging crashed runs.
license: Apache-2.0
metadata:
  author: NVIDIA
  version: "1.0.0"
  tags:
    - nemo-gym
    - evaluation
    - score-comparison
    - confidence-intervals
    - failure-analysis
---

# NeMo Gym Analyzing Results

## Purpose

Answer two questions about evaluation results, in this order:

1. **Is the difference real?** Check that both numbers measure the same thing (same tasks, settings and accounting of failed rollouts) and that the gap is larger than run-to-run noise.
2. **Why did tasks fail, and what should change?** Read the rollouts behind the score and tie each finding to task-level evidence.

Most reported gaps come from comparing unlike things: different tasks, settings, code, failure accounting, or ordinary noise. Rule those out first. Diagnosing a gap between runs that are not comparable explains the wrong thing.

## When to Use This Skill

- "Is the new checkpoint better?", "did quantization hurt accuracy?", "why is my score lower than the model card?", "these two runs disagree".
- Reading `gym eval compare` output: deltas, confidence intervals and task flips.
- "Which tasks fail and why?", "what should I fix first?", or writing an analysis report of a run.

Hand off when the main job is different:

- Running, resuming or monitoring an evaluation: the evaluation-running skill (`nemo-gym-run-and-babysit-eval`, called `run-and-babysit-eval` in older Gym checkouts).
- Choosing RL training tasks from reward profiles: `nemo-gym-reward-profiling`.
- Choosing repeat counts before collecting runs: `nemo-gym-repeat-planning` in the Gym repo.
- A crash, a stack trace, an empty output file, servers that never start, or a slow run: `nemo-gym-debugging` in the Gym repo.
- Building a BLADE benchmark package (golden reports, anchor facts, judge calibration): `nemo-gym-blade-analysis` in the Gym repo.

## Prerequisites

- The run directories. Every `gym eval run` writes its files next to its `--output` file (`<stem>.jsonl`); see step 1.
- NeMo Gym (`pip install "nemo-gym>=0.7"`, which needs Python 3.13.14 or newer) for `gym eval compare`, `gym eval health-check`, `gym eval profile` and `gym eval reverify`. None of them call the policy model; `gym eval reverify` calls the judge on judged benchmarks. If `gym` is missing, read the files directly.
- "After 0.7.0" marks options and files that are on the main branch but not in the 0.7.0 release. An install from main reports its version as `0.7.0rc0`, so check for the option in `--help` or for the file instead of comparing version numbers.
- Python 3 for the snippets in the references. On a run without `<stem>_manifest.json` they need only the standard library. On a managed run (after 0.7.0) run them with the Python that has nemo-gym installed; the recipe that profiles selected rows always needs it.

## Instructions

### 1. Find the run files

| File, next to `<stem>.jsonl` | What it answers |
|---|---|
| `<stem>.jsonl` | The saved rollouts: `reward`, the model's output, `_ng_task_index` (task) and `_ng_rollout_index` (repeat). One row per rollout in 0.7.0; a managed run can also hold superseded attempts |
| `<stem>_materialized_inputs.jsonl` | What was planned: one row per task and repeat, with the prompt and the verifier's inputs such as the expected answer |
| `<stem>_failures.jsonl` | Failed attempts, with `_ng_failure_class` and the error in `_ng_failure_message`. Not in the score by default |
| `<stem>_aggregate_metrics.json` | The scores per agent: `mean/reward`, per-task (`group_level_metrics`) and per-repeat (`repeat_level_metrics`) statistics, and `*_across_repeats` keys when there are 2 or more repeats |
| `<stem>_coverage.json` (after 0.7.0) | Planned, measured, masked, failed and missing rollout counts, and failures counted as zeros |
| `<stem>_provenance.json` (after 0.7.0) | Gym version and commit, the served model's id and checkpoint, a hash of the materialized inputs, sampling settings and seed |
| `<stem>_resolved_config.yaml` (after 0.7.0) | The run's merged config, secrets masked. After `gym eval run --no-serve` it holds only the collection settings (task limit, repeats, output tokens, failure accounting); model, judge, tool and timeout settings are in the config given to `gym env start` |
| `<run_dir>/quality_summary.json`, `<run_dir>/rollout_verdicts.jsonl` | Health reports: run accounting in `run.artifacts` (after 0.7.0) and a verdict for every saved rollout |
| `<output_dir>/compare_report.md`, `<output_dir>/compare_report.json` | Output of `gym eval compare` |

Ask for any file that is missing rather than guessing its content. A published number needs its source and its settings: repeats, sampling, token limits, prompt and judge.

### 2. Check each run is complete before reading its score

A score covers the measured rollouts and any failures counted as zeros. Count before comparing:

- **Planned** is the number of rows in `<stem>_materialized_inputs.jsonl`. **Measured** is the number of rollouts whose selected attempt succeeded with a numeric `reward` and no `mask_sample`. After 0.7.0 read it as `measured` in `<stem>_coverage.json`: a managed run's `<stem>.jsonl` can also hold superseded attempts and results that Gym counts as failures. A 0.7.0 run saves one row per rollout, so count its rows in `<stem>.jsonl` with a numeric `reward` and no `mask_sample`.
- **Failed rollouts are left out of the score by default**, not counted as 0. The exception is a failure row that a benchmark or agent policy marks `_ng_failure_counts_as_zero` (after 0.7.0): it enters the score (see failure accounting in step 3). A run that lost its hardest tasks to failures scores higher than it earned. Count the failures by `_ng_failure_class`, including `judge_failed`; a rollout that failed and later succeeded on `--resume` appears in both files and is not failed.
- Results that the benchmark marked `mask_sample` (not a valid measurement) also leave the score.
- With failed or missing rollouts, one run's metrics disagree: `mean/reward` weights rollouts, `pass@1[avg-of-K]` for the full repeat count K weights tasks equally, and `pass@k` leaves out tasks with fewer than k measured rollouts, so it can read 100% while some tasks never pass. Decide that rollouts are missing from the counts above, not from a gap between metrics. A gap points to uneven coverage only when both numbers score the same reward and differ only in weighting, as MCQA's `pass@1[avg-of-K]/accuracy` divided by 100 and `mean/reward` do. Other metrics differ even in a complete run: BEAM's `pass@1[avg-of-K]/overall` is already on 0–1 and averages its abilities, and `pass@1[avg-of-k]` and `majority@k` with a smaller k use only each task's first k repeats (`pass@k` for 0/1 rewards uses all of them).
- `run.artifacts.healthy: true` in `<run_dir>/quality_summary.json` only means that nothing is missing or recorded twice. Failure limits are off by default, so read `run.artifacts.failed` too.
- A rollout verdict of `unobserved` is not `healthy`: at least one check lacked the evidence it needs, such as captured model calls. Read each check's `evaluated` count in `run.artifacts.coverage`: a check such as `model_call_runaway_generation` with `evaluated: 0` never ran, so it flagged nothing even if answers were cut off.

`<stem>_coverage.json` and `run.artifacts` hold these counts after 0.7.0. For 0.7.0 runs, which have no coverage file and only record and per-check counts in `run.artifacts`, use the counting snippet in `references/score-comparability.md`. Report coverage next to every score, for example "0.62 over 180 of 192 planned rollouts; 12 failed (`model_request_error`), left out of the score".

### 3. Check the two numbers measure the same thing

Compare both sides on each item. A setting that the reference does not publish is unknown, not a match.

- **Tasks.** Same task count, task indices and prompts in both `<stem>_materialized_inputs.jsonl` files, or the same `materialized_inputs_sha256` in both provenance files. `--limit`, a different data version or a different prompt template changes the task set even on the same code. `gym eval compare` matches tasks by `_ng_task_index` only and does not check that the datasets match.
- **Settings.** Diff the two runs' configs: temperature and other sampling, maximum output tokens, reasoning settings, system prompt, judge model, agent harness, tools, timeouts and step limits.
- **Failure accounting.** `count_failure_classes_as_zero` adds failed rollouts to the score, as 0 unless the failure row carries its own `reward`, and `count_missing_rollouts_as_zero` scores missing rollouts as 0. They are not recorded in `<stem>_aggregate_metrics.json`; look for them in the resolved config or the log line `Counting N failure row(s) as scored zeros`. In `<stem>_coverage.json`, `failures_counted_as_zero` also counts failures that a benchmark's own policy scores as 0 (`final_zeros`, `counted_zeros_in_successful`), and `imputed` counts missing rollouts scored as 0. When failures sit on one side only, no accounting setting makes the runs comparable: counting them as zeros charges the model for infrastructure, and leaving them out drops whatever those tasks would have scored. Resume or rerun them.
- **Metric and scale.** pass@1 averaged over repeats, pass@k and majority vote are different numbers. `mean/reward` keeps the benchmark's reward units: 0–1 for pass/fail rewards, but other benchmarks use other ranges, some negative. Gym's shared `pass@k` and `majority@k` keys and many published numbers are percentages, but a benchmark can rescale its own (BEAM's `overall` keys are on 0–1). Check the metric's definition and bounds before converting.
- **Code and serving.** Gym version or commit, benchmark version, model checkpoint, precision and serving engine; `<stem>_provenance.json` records several of these.

`references/score-comparability.md` has the full checklist, where each factor lives in a run, and `references/benchmark-notes.md` lists known differences between Gym benchmarks and the published methods they follow.

### 4. Check whether the gap is bigger than noise

For two independent Gym runs with at least 2 repeats each, use `gym eval compare`. It reads only the saved files, so it needs no running servers and makes no model calls:

```bash
gym eval compare \
    --baseline results/baseline/rollouts.jsonl \
    --candidates results/candidate/rollouts.jsonl \
    --output-dir results/comparison
```

It prints the key metrics and writes `<output_dir>/compare_report.md` and `<output_dir>/compare_report.json`. Read them this way:

- **Delta is candidate minus baseline**, using each run's mean across repeats when available. The interval is a two-sided Welch t interval over the per-repeat scores, 95% by default (`--confidence-level 0.90` after 0.7.0). It treats the runs as independent samples of the same fixed task set.
- **Improvement** needs the whole interval above 0 (for a metric where higher is better). An interval that contains 0 shows neither an improvement nor a regression, and it does not show that the runs are equivalent.
- **"No worse than"** needs a largest acceptable loss chosen before looking at the result: the lower end of the interval must be above minus that margin.
- Overlapping intervals of the two separate runs are not the test. Use the interval of the difference.
- No interval (`—` in Markdown, `null` in JSON) means a side has fewer than 2 usable repeats, or the metric has no per-repeat values.
- The report does not check the runs for you. A warning about uneven task coverage across a run's repeats goes only to the terminal, and nothing warns about different task sets or failures counted as zeros. The task counts per side and `baseline_only_task_count` / `candidate_only_task_count` in the JSON show a task-set difference. Do not trust an interval whose runs failed step 2 or 3.

Other cases:

- **One repeat per run.** There is no interval. One run's standard error is at most `sqrt(p * (1 - p) / n)` for n tasks at pass rate p, about 5 points for 100 tasks at p = 0.5. Collect 2 or more repeats per side before deciding.
- **A published number.** The Gym run's standard error `se_a` is `se_across_repeats/mean/reward`. Estimate the reference's as `se_b = sd1 / sqrt(k_ref)`, where `sd1` is `std_across_repeats/mean/reward` of the Gym run and `k_ref` is the reference's repeat count (1 if unknown). Put both on the same scale. A gap under about `2 * sqrt(se_a^2 + se_b^2)` is within noise; check settings (step 3) whatever the size of the gap.
- **Several runs of the same setup.** After 0.7.0, `--baseline-runs` and `--candidate-runs` pool independent runs into one sample. A resumed run, a copy or a re-aggregation of the same rollouts is not an independent run.
- **Several benchmarks.** After 0.7.0, `gym eval compare-suite --suite <suite.yaml>` combines normalized benchmark scores per model, with a 90% interval by default.
- **More repeats.** For a standard error s you need about `(sd1 / s)^2` repeats. To plan repeat counts for a future comparison, hand off to `nemo-gym-repeat-planning` in the Gym repo.

### 5. Find the tasks that changed or fail

- **Task flips.** For rewards of 0 or 1, the flips table lists tasks whose majority outcome changed: a task passes when its mean reward over repeats is above 0.5, and exactly 0.5 is a tie. A flip on a task that sometimes passes in both runs is usually repeat noise; check each flipped task's pass counts in both runs before explaining it. The Markdown report shows at most 10 flips per direction; the JSON has all of them.
- **Outcome groups.** Sort every planned task by its rewards across repeats: always passes, sometimes passes, never passes, or not measured. Use the snippet in `references/failure-diagnosis.md`, which counts the attempt Gym selected for each rollout. `gym eval profile` gives per-task statistics but reads every saved row, so profile a managed run's selected rows as shown there (it needs `++allow_partial_rollouts=True` when rollouts failed or are missing, and writes its files next to the rollouts file). Count tasks from the materialized inputs, not from the rollouts, so a task with no measured rollout is not lost.

### 6. Diagnose why tasks fail

Load `references/failure-diagnosis.md` for the full method. In short:

- Start with tasks that sometimes pass. Compare a passing and a failing repeat of the same task and find where they diverge: the answer, its format, where the answer was written, a tool call, or where the output stopped.
- For a task that never passes, check the verifier before blaming the model. Compare the extracted answer with the expected answer, and look for a correct answer in another format, an answer only in the reasoning, or output cut off at the token limit.
- Keep infrastructure out of model quality: retryable failures, HTTP 429 or 5xx errors and sandbox errors are not wrong answers. Some endings are settled as the model's 0 by the benchmark itself, such as a Harbor verifier timeout or an out-of-memory kill during the agent's work; count those separately, and call them model errors or infrastructure only with evidence.
- Give each failing task one root-cause label with evidence (task and rollout ids, the relevant output, the verifier fields), then map labels to fixes.
- To size a verifier or answer-extraction fix, re-score the saved rollouts with `gym eval reverify` and count the rollouts and tasks whose reward changes. First check that every measured rollout got a valid re-score: re-scores that fail go to the failures file next to the new output. On the same rollouts the difference is exact; the interval from `gym eval compare` assumes independent runs and does not apply.

### 7. Report

Lead with the verdict, then coverage, then evidence. Take every number from the files. For a full written report, such as a BLADE golden report, use the layout in `references/failure-diagnosis.md`.

```text
Verdict: real difference | within noise | not comparable | explained by <cause> | not yet explained
Compared: <baseline> vs <candidate>: <tasks> tasks x <repeats> repeats each, <source and settings>
Coverage: <measured>/<planned> per side; <failed> failed (<class>), <masked> masked
Difference: <delta> [<low>, <high>] at <level>%: <what the interval supports>
Explained by: <finding>, with evidence: <file, key, task ids, rollout ids>
Tasks: <n> always pass, <n> sometimes pass, <n> never pass, <n> not measured; root causes <label: task ids>
Next step: <rerun or resume X, align setting Y, add repeats, fix verifier Z>
```

## Examples

**Two runs.** A team compares a BF16 baseline with an FP8 candidate on a 150-task tool-use benchmark, 4 repeats each. `gym eval compare` prints `mean/reward` 0.712 → 0.694, delta −0.018 [−0.041, 0.005]. The candidate's `<run_dir>/quality_summary.json` shows 600 expected rollouts and 11 failed (`model_request_error`, HTTP 429 from the endpoint); the baseline shows none. The two runs' configs differ only in the model path. The report:

```text
Verdict: within noise so far: the interval contains 0, and the candidate lost 11 rollouts to rate limits
Compared: BF16 vs FP8, 150 tasks x 4 repeats each, same Gym commit and config apart from the checkpoint
Coverage: baseline 600/600; candidate 589/600, 11 failed (model_request_error, HTTP 429), left out of the score
Difference: -1.8 points [-4.1, +0.5] at 95%: neither a loss nor equivalence is shown
Explained by: not explained yet; the failures sit on 9 tasks, 7 of them always-pass in the baseline, so the
  candidate's score leaves out mostly easy rollouts and reads low
Next step: resume the candidate to finish the 11 rollouts, then compare again. A "no worse than 2 points" claim
  needs the lower end above -2; at this spread that takes far more repeats, so plan them before collecting
```

**A published number.** A run scores 0.703 on a 1,000-question benchmark with 4 repeats (`std_across_repeats/mean/reward` 0.008); the model card reports 74.1 from 16 repeats at temperature 0.6 with a 32K output limit. The run's config shows temperature 1.0 and an 8,192-token limit, and 9% of its rollouts stop at that limit with no answer. The standard error of the gap is about `sqrt((0.008/sqrt(4))^2 + (0.008/sqrt(16))^2)` = 0.45 points, so the 3.8-point gap is not noise; the settings differ, so the verdict is "not comparable" until the run is repeated with the card's temperature and output limit.

**Tasks that never pass.** In a 6-repeat math run, two tasks never pass. Their rollouts give the right value as a fraction while the expected answer is the decimal `0.75`, and the verifier's extracted answer is that fraction: a verifier issue (`TI`), not a knowledge gap. Re-score the saved rollouts with `gym eval reverify` using a verifier that accepts equivalent forms, and report how many rollouts change reward.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `gym eval compare` shows `—` (JSON `null`) for the delta interval | A side has fewer than 2 usable repeats, or the metric exists only for the whole run (such as pass@k) | Collect 2 or more repeats per side; read per-repeat metrics such as `mean/reward` |
| `gym eval compare` cannot find `<stem>_aggregate_metrics.json` | The run used `--disable-aggregation`, the metrics file is elsewhere, or a strict run ended incomplete (after 0.7.0 it writes no metrics then) | Run `gym eval aggregate` (it needs the servers running), pass the file with `--baseline-agg-metrics` / `--candidates-agg-metrics`, or resume the incomplete run |
| `gym eval compare` fails with `No agent name is present in every run, so there is nothing to compare.` | The runs used different agent names | Pick them with `--baseline-agent` and `--candidate-agents` |
| `UserWarning: ... has incomplete or unequal task coverage across repeats` | Some repeats of one run measured fewer tasks: failed or missing rollouts | Check step 2; resume the run before trusting the interval |
| `gym eval profile` fails with `Missing rollout results for N materialized input rows` | Rollouts failed or are missing | Add `++allow_partial_rollouts=True` and report coverage with the numbers |
| `gym eval profile` fails with `Duplicate result row for rollout key` | The rollouts file of a managed run holds more than one attempt of a rollout | Profile the run's selected rows, as shown in `references/failure-diagnosis.md` |
| Failure rows whose `_ng_failure_class` starts with `provider_` (after 0.7.0), or other sandbox errors | The sandbox was not ready or not reachable | Infrastructure, not model quality. Fix the sandbox; a resume reruns these while `_ng_failure_terminal` is false and attempts remain |
| Unmasked zeros in `<stem>.jsonl` from verification that did not finish (`failure_kind` `harbor:verifier_timeout`, `harbor:missing_reward` or `harbor:invalid_reward`, or `evaluation_completed: false` with no `mask_sample`), or an out-of-memory kill during the agent's work counted as 0 | The benchmark's own settled grade: the run counts these as measured, so a resume does not redo them | Keep them in the score and report how many there are. If the evidence shows a grade is invalid, such as verifier timeouts on an overloaded cluster, re-verify where the benchmark supports it or collect a new run, and report both |
| `evaluation_completed: false` on a masked row or a `judge_failed` failure | Not a grade: the row is left out of the score | When the benchmark's resources server runs the judge, recover judge failures from the saved answers with `gym eval reverify --judge-failed-only`. Benchmarks that judge inside their Environment Server, such as SWE-Together, stop with `reverify: no resources server found in the config`: report those rows as left out; only new rollouts of those tasks recover them. Resume other retryable failures |
| `judge_failed` or other failure rows whose `_ng_failure_message` shows HTTP 401 or 403 | A bad or missing API key for the policy or judge endpoint | Fix the key, then recover judge failures as in the row above; resume policy failures |
| `gym eval reverify` cannot reach the judge, or fails with connection errors | The judge endpoint is unreachable or the URL is wrong | Check the judge URL and key in the config passed to `--config`, then rerun with `--resume` |
| `gym eval health-check` replaced reports in a shared folder | It writes `<run_dir>/quality_summary.json` and `<run_dir>/rollout_verdicts.jsonl` | Copy a shared or read-only run before checking it |
| `gym: command not found` | NeMo Gym is not installed here | `pip install "nemo-gym>=0.7"`, or read the files directly with the snippets in the references |
| `pip install "nemo-gym>=0.7"` fails with `No matching distribution found` | The Python is older than 3.13.14, so pip only sees old releases | Use Python 3.13.14 or newer, for example `uv venv --python 3.13.14` |

## Limitations

- The interval from `gym eval compare` covers repeat-to-repeat noise on a fixed task set. It does not cover sampling different tasks, and it does not pair repeats by index.
- `gym eval compare` trusts that both runs used the same tasks and settings; it matches tasks only by `_ng_task_index`.
- Health checks look at whether the saved evidence is complete and consistent, not at whether an answer is correct.
- A root-cause label is a judgment from the evidence available. Say what evidence is missing rather than filling the gap.
