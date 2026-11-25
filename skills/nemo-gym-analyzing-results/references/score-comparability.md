# Is the Difference Real?

Use this reference to decide whether two numbers can be compared, how failed and missing rollouts enter each one, and whether a gap is larger than run-to-run noise. Code and CLI help in the installed nemo-gym win over this file when they disagree.

## Comparability Checklist

Each row is one factor that can change a score by itself, even with the same model. Check every row for both sides. When a reference does not publish a setting, write it down as unknown instead of assuming it matches.

| Factor | Why it moves the score | Where to check in a Gym run |
|---|---|---|
| Gym version and images | Scorer, agent and failure-handling fixes change results between versions. | `gym_version` and `gym_commit` in `<stem>_provenance.json` (after 0.7.0; a `-dirty` commit had local changes); the launcher log; for `gym eval submit` runs, `driver.gym_install.ref` in `<run_dir>/resolved-config.yaml` (`gym_commit` in `<run_dir>/gym-job.json` is the Gym that submitted the job, not necessarily the one that ran it); container image tags. |
| Dataset version and subset | Different questions, date windows, or a public subset of a larger set. Data is prepared separately from the rollouts, so it can differ between runs on the same version. | Task count, task indices and prompts in both runs' `<stem>_materialized_inputs.jsonl`, or `materialized_inputs_sha256` in both provenance files; the benchmark's prepare step and its revision pins. |
| Task limit | `--limit N` runs the first N tasks only, usually not a random sample. | The task count in the materialized inputs; `limit` in the resolved config; the launch command. |
| Prompt and system prompt | The instructions and answer format decide what the verifier can read. | `responses_create_params.input` in the materialized inputs; prompt and template hashes in the provenance file. |
| Metric and aggregation | pass@1 averaged over k repeats, pass@k and majority@k are different numbers. Rollout-weighted and task-weighted means differ once rollouts are missing. `mean/reward` keeps the benchmark's reward units (0–1 for pass/fail rewards, other ranges elsewhere, some negative), while Gym's shared `pass@k` and `majority@k` keys are percentages unless the benchmark rescales them (BEAM's `overall` keys are on 0–1). | Key names in `<stem>_aggregate_metrics.json`; the benchmark's metric definition and bounds. |
| Repeats | Fewer repeats give a wider spread, and pass@k needs k rollouts per task. | Count rows per task in the materialized inputs rather than trusting a flag. After 0.7.0, `gym eval run --num-repeats` replaces a benchmark's own repeat count; in 0.7.0 the two multiplied. |
| Sampling | Temperature, top_p, top_k and seeds change outputs. | `responses_create_params` in the materialized inputs; `sampling_parameters` and `seed` in the provenance file; sampling overrides on the model server. |
| Token limits and context length | Truncated reasoning or answers score as wrong. | Maximum output tokens in the request or the resolved config; the serving engine's maximum model length; output tokens against the limit in the rollouts or in `max/output_tokens` of the aggregate metrics. |
| Reasoning settings | Answers that land only in the reasoning usually score 0. A reasoning parser can move part of an answer there. | Reasoning effort, chat-template arguments and the serving engine's reasoning parser. |
| Judge | Different judge models or prompts grade differently, and unparseable judge replies can count as wrong. | The judge model server in the config; parse-failure metrics; `judge_failed` rows in the failures file. |
| User simulator and tools | The simulator model and the search or tool provider change agentic outcomes. | Agent and resources server config. |
| Timeouts and step limits | Wall-clock and step caps decide how much work fits into a task, so slower serving lowers agentic scores. | Agent and sandbox timeouts, `max_steps` or turn limits. |
| Failure accounting | Failures can be left out, counted as 0, or retried. | The next section. |
| Agent harness | Different harnesses solve different fractions of the same tasks and treat timeouts differently. | The agent named in the config and in the rollouts (`agent_ref`). |
| Serving | Precision, engine version, parsers, parallelism and throughput all change results. | The served model id and checkpoint in the provenance file; serving config and logs. |

After 0.7.0, `gym eval run` saves its merged config, secrets masked, as `<stem>_resolved_config.yaml`. It keeps `${...}` interpolations as written, so values read from environment variables are not in it. With `--no-serve` it holds only the collection settings; the model, judge, tool and timeout settings live in the config given to `gym env start`. For older runs, rebuild the config with `gym env resolve --config <file> ...`, which resolves today's files rather than what ran.

## How Failed and Missing Rollouts Enter a Score

- A failed rollout is written to `<stem>_failures.jsonl` with `_ng_failure_class` and its error in `_ng_failure_message`, and is **left out of the score** by default.
  - Classes after 0.7.0 include `model_request_error` (the model endpoint answered with an error such as HTTP 401, 429 or 500), `model_transport_error`, `model_context_limit`, `agent_run_error`, `agent_timeout`, `environment_server_failed`, `judge_failed`, `verifier_error`, `provider_*` sandbox failures, and benchmark-specific classes.
  - 0.7.0 records a model error as `agent_run_error` (the agent answers HTTP 500) or `environment_server_failed`; its `agent_request_failed` means the agent answered 429, 502, 503 or 504, or not at all.
  - `_ng_failure_terminal: false` means a resume runs the rollout again, up to its attempt cap (`_ng_failure_max_attempts`, 3 by default).
  - The stage matters too: the default policy settles an out-of-memory kill during the agent's own work (`provider_oom_killed`) as the model's 0, which the coverage file lists under `counted_zero_kinds`, but retries the same kill during sandbox setup.
- The failures file has one row per failed attempt. A rollout that failed and then succeeded on `--resume` appears in both files; only rollouts with no success are failed.
- By default the first failed agent `/run` ends the whole run. With `+route_failures_to_sidecar=true`, the run records the failure and continues, and the score covers fewer rollouts than were planned.
- `+count_failure_classes_as_zero=[<class>,...]` adds rollouts whose last attempt failed with one of those classes to the metrics: as 0, unless the failure row carries its own `reward`, which is kept. `+count_missing_rollouts_as_zero=true` scores planned rollouts that left no row anywhere as 0. Both change only the metrics. Neither is recorded in `<stem>_aggregate_metrics.json`, so check the command, the resolved config, or the log lines `Counting N failure row(s) as scored zeros` and `Counting N materialized rollout(s) with no row as scored zeros`. In `<stem>_coverage.json` (after 0.7.0), `failures_counted_as_zero` also includes failures that a benchmark's own policy scores as 0 (`final_zeros`, `counted_zeros_in_successful`, for example a sandbox out-of-memory kill), and `imputed` counts missing rollouts scored as 0. With either on, `mean/reward` is the safest number to compare, because some benchmarks' own metric code cannot score the added zeros.
- When failures sit on one side only, neither setting makes two runs comparable: counting them as zeros charges the model for infrastructure, and leaving them out drops whatever those tasks would have scored. Resume or rerun them.
- A result the benchmark marked `mask_sample` is saved but is not a valid measurement, so it leaves every quality metric. A failed judge call is not masked this way: it becomes a `judge_failed` failure row, which the coverage file counts under `failed` and, when its producer masked it, `failed_masked`.
- With failed or missing rollouts, the metrics of one run disagree. `mean/reward` averages measured rollouts, so a task that lost rollouts counts less. The shared `pass@1[avg-of-K]` for the full repeat count K averages each task's measured rollouts and then weights tasks equally, and `pass@k` leaves out tasks with fewer than k measured rollouts, so it can read 100% while some tasks never pass. A gap between `pass@1[avg-of-K]` and `mean/reward` shows uneven coverage only when both score the same reward and differ only in weighting, as MCQA's `pass@1[avg-of-K]/accuracy` divided by 100 and `mean/reward` do. A benchmark's own metric can score something else, use another scale or average over other groups, so it differs even in a complete run: BEAM's `pass@1[avg-of-K]/overall` is already on 0–1 and averages its ten abilities. `pass@1[avg-of-k]` and `majority@k` with a smaller k use only each task's first k repeats, so they differ too, while `pass@k` for 0/1 rewards is estimated from all of a task's measured repeats (for rewards between 0 and 1 it takes the best of the first k). To decide that rollouts are missing, use the counts below.
- A judge reply that arrives but cannot be parsed usually counts as a wrong answer, not as `judge_failed`. After 0.7.0, `gym eval run` grades saved judge failures once more at the end of the run; rows that fail again stay out of the score. When the benchmark's resources server runs the judge, recover them later with `gym eval reverify --judge-failed-only`, without new policy calls. Benchmarks that judge inside their Environment Server, such as SWE-Together, have no resources server for it (`reverify: no resources server found in the config`): report those rows as left out of the score, and use the benchmark's own recovery or a new run.
- `require_complete` is off by default, so a partial run still publishes a score. `gym eval submit` turns it on for Slurm runs. A strict run that ends incomplete fails with `EVAL FAILED: <done>/<expected> samples completed`; after 0.7.0 it writes no `<stem>_aggregate_metrics.json` until a resume completes it (0.7.0 wrote metrics for the finished rollouts first). Check the job status, not only for a metrics file.
- Some benchmarks score a verification that did not finish as 0 inside the score instead of failing the rollout, for example Terminal-Bench verifier timeouts and missing reward files. An infrastructure problem then lowers the score instead of shrinking it. To size it, count measured rows (no `mask_sample`, no failure class) by `failure_kind` `harbor:verifier_timeout`, `harbor:missing_reward` or `harbor:invalid_reward` on benchmarks that run through the Harbor environment server, or with `evaluation_completed: false` on benchmarks that run through their own resources server. A masked row or `judge_failed` failure can also carry `evaluation_completed: false`, but it is left out of the score, not settled. These zeros are settled: the coverage file counts them as measured, and a resume does not redo them. Keep them unless the evidence shows a grade is invalid; then re-verify where the benchmark supports it, or collect a new run. Other benchmarks mask such endings instead: after 0.7.0, a SWE-bench evaluation that fails for a reason other than the model's patch is masked as a `verifier_error` and leaves the score, while a model patch that does not apply stays a measured 0. Both carry `evaluation_completed: false`, so use `mask_sample` and `failure_kind` to tell them apart.

## Counting Planned, Measured and Failed Rollouts

After 0.7.0, `<stem>_coverage.json` gives `expected`, `measured`, `masked`, `failed` (with `failed_by_class`, `failed_final` and `failed_not_final`), `missing`, `not_started` and more, and `<run_dir>/quality_summary.json` repeats the run accounting in `run.artifacts`. It is the run's own accounting: for a managed run (one with `<stem>_manifest.json`) it follows the attempt the run selected for each rollout, which raw rows cannot show. `gym eval status <run_dir>` shows only completed and remaining rollouts and failures by class. The snippet below prints that file when it exists, and counts from the raw files only for runs without one, such as 0.7.0 runs. It assumes the run wrote `results/rollouts.jsonl`; Gym names the other files from that stem:

```bash
python3 - results/rollouts.jsonl <<'EOF'
import collections, json, sys
from pathlib import Path
out = Path(sys.argv[1]).resolve()  # Read a symlinked run beside its target, as Gym does after 0.7.0.
coverage = out.with_name(out.stem + "_coverage.json")
if coverage.exists():
    # The run's own accounting: for a managed run it follows the attempt the run selected for each rollout.
    c = json.loads(coverage.read_text())
    print({k: c.get(k) for k in ("expected", "measured", "masked", "failed", "failed_by_class", "missing",
                                 "not_started", "failures_counted_as_zero", "imputed", "complete")})
    sys.exit(0)
if out.with_name(out.stem + "_manifest.json").exists():
    sys.exit("a managed run without a coverage file: with nemo-gym installed, read its counts from "
             "nemo_gym.rollouts.RolloutStore.read(Path(path)).coverage(); raw rows cannot show which attempt counts")
def rows(p, required=False):
    if not p.exists():
        if required:
            sys.exit(f"missing {p}: point this at the run's rollouts file, next to its materialized inputs")
        return []
    result = []
    with p.open("rb") as f:
        for line in f:
            try:
                row = json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue  # Blank lines, and the cut-off last line a killed writer can leave.
            if isinstance(row, dict):
                result.append(row)
    return result
key = lambda r: (r.get("_ng_task_index"), r.get("_ng_rollout_index"))
done = rows(out, required=True)
saved = {key(r) for r in done}
measured = {key(r) for r in done if isinstance(r.get("reward"), (int, float)) and not r.get("mask_sample")}
# Skip rows that only record evidence: a rollout a dispatch budget never started, or an unrouted run error.
fails = [r for r in rows(out.with_name(out.stem + "_failures.jsonl"))
         if not (r.get("_ng_omitted_rollout") and (r.get("_ng_dispatch_drained") or r.get("_ng_unrouted_run_error") is True))]
last = {key(r): r for r in fails}
lost = {k for k, r in last.items() if r.get("_ng_omitted_rollout")}
planned = {key(r) for r in rows(out.with_name(out.stem + "_materialized_inputs.jsonl"), required=True)}
failed = collections.Counter(r.get("_ng_failure_class") for k, r in last.items() if k not in saved | lost)
print(f"planned={len(planned)} measured={len(measured)} masked={len(saved - measured)} "
      f"failed={sum(failed.values())} no_row={len(planned - saved - (last.keys() - lost))} {dict(failed)}")
EOF
```

## Health Reports

At the end of a run, `gym eval run` writes `<run_dir>/quality_summary.json` and `<run_dir>/rollout_verdicts.jsonl` next to the rollouts: after 0.7.0 also for a failed run, in 0.7.0 only after a completed collection. `gym eval health-check <run_dir>` rebuilds both, with `--rollouts-file <name>` when the rollouts file has another name; it writes them into the run directory, so copy a shared or read-only run first.

- `run.artifacts` is, after 0.7.0, the run accounting against the materialized inputs: `expected`, `completed`, `scored`, `failed` (`total`, `final`, `not_final`, `by_class`), `missing`, `not_started`, `healthy` and `unhealthy_reasons`. In 0.7.0 it holds only `records`, `captures` and a per-check `coverage` of how many rollouts each check evaluated. `final` failures are ones a resume does not retry; the split says nothing about whether the model or the infrastructure caused them.
- `run.verdicts` counts saved rollouts that are `healthy`, `unhealthy` or `unobserved`. Failed and missing rollouts get no verdict. `run.issues` counts findings per check. `run.stats` counts model call errors by HTTP status, tokens and, after 0.7.0, timeouts; `run.stats.model_call_errors` covers only saved rollouts, so a run whose failed rollouts all ended on model errors can show none there.
- `tasks` gives verdict counts and flags per task, such as `task_consistently_unhealthy` and `task_no_successful_model_calls`.
- `<run_dir>/rollout_verdicts.jsonl` has one row per saved rollout, keyed by `_ng_task_index` and `_ng_rollout_index`, with its verdict and findings.

Where they mislead:

- `run.artifacts.healthy: true` means only that nothing is missing or recorded twice and that no failure limit was reached. The limits are off by default, so a run with many failed rollouts can still be healthy.
- `unobserved` is not healthy: at least one check lacked the evidence it needs, such as captured model calls. Read each check's `evaluated` count in `run.artifacts.coverage`; a check with `evaluated: 0` never ran. An agent that records no trajectory leaves `model_call_runaway_generation` at `evaluated: 0`, so cut-off or empty answers go unflagged.
- A health verdict says nothing about whether an answer is correct: a correct rollout can be unhealthy and a wrong one healthy. Findings such as `model_call_runaway_generation` (output stopped at its token limit with no answer, or, after 0.7.0, stopped for repetition) and `rollout_ended_on_failed_model_call` point at causes of low scores.
- Each run or `--resume` that writes the reports replaces them, so after a failed 0.7.0 run or resume the reports can be from an earlier attempt. After 0.7.0, a job stopped by SIGTERM leaves a provisional summary (`partial: true`). Check names change between versions, so rebuild both runs' reports with the same nemo-gym before comparing their health.

## Noise and Confidence Intervals

### Where Gym Reports Spread

- With 2 or more repeats, `<stem>_aggregate_metrics.json` holds `repeat_level_metrics` (one entry per repeat, across that repeat's tasks) and, in each agent's `agent_metrics`, `mean_across_repeats/<metric>`, `std_across_repeats/<metric>`, `se_across_repeats/<metric>` and `ci_low_95_across_repeats/<metric>` / `ci_high_95_across_repeats/<metric>`.
- `std_across_repeats/mean/reward` is the spread of a single repeat's score, called `sd1` below. A repeat's own `std/reward` is the spread across tasks and is much larger; do not use it as noise.
- Servers that compute pass@k may also write `.../std_dev_across_runs` and `.../std_err_across_runs`, on the scale of their metric: 0–100 for percentages, 0–1 for BEAM's `overall` keys.
- `group_level_metrics` has per-task statistics but no interval. `missing_count` in a repeat entry counts only tasks measured in another repeat, so a task missing from every repeat is never counted there.

### Reading `gym eval compare`

- Delta = candidate − baseline, from `mean_across_repeats/<metric>` when recorded, else the point value. JSON keeps `value` and `mean_across_repeats`; they differ when repeats cover different tasks.
- The delta interval is a two-sided Welch t interval over the per-repeat values (each repeat is one observation), 95% by default. It needs at least 2 finite per-repeat values on each side. After 0.7.0, `--confidence-level` changes it, and the JSON records it as `confidence_level`.
- **Improvement**: the whole interval is above 0 (below 0 for a lower-is-better metric). **Regression**: the whole interval is on the other side. An interval containing 0 establishes neither, and does not establish equivalence.
- **No worse than a margin**: choose the largest acceptable loss before looking. The lower end of the interval must be above minus that margin. An interval that crosses the margin is inconclusive, not a failure.
- The two runs' separate 95% intervals overlapping is not a test of the difference.
- Using the lower end of a two-sided 95% interval is a one-sided test at 2.5%. The interval covers repeat noise for this fixed task set only, and assumes the two runs are independent samples; restarted or resumed rollouts and shared outages break that.
- The interval does not apply to two scorings of the same rollouts, such as an original run and its `gym eval reverify` re-score. Their difference is exact for those rollouts; report how many rollouts and tasks changed reward instead.
- The report files do not check the runs. A warning about uneven task coverage across a run's repeats goes to the terminal only, nothing warns about different task sets, and failures counted as zeros are not flagged. In the JSON, the task count of each side and `baseline_only_task_count` / `candidate_only_task_count` in the flips show a task-set difference. Check coverage before reading the delta.
- Flips are for rewards of 0 or 1: a task passes when its mean reward over repeats is above 0.5, and exactly 0.5 is a tie. Flips are observations, not proof of a change; a task that passes 5 of 8 repeats in one run and 3 of 8 in the other flips by chance. The Markdown report lists at most 10 flips per direction.
- Pooling (after 0.7.0): `--baseline-runs a.jsonl,b.jsonl` and `--candidate-runs ...` concatenate the per-repeat values of independent runs into one sample. Copies, symlinks, resumed history and re-aggregations of the same rollouts are not independent. Gym rejects the same file given twice, through a symlink too, but accepts a copy, so check the sources yourself.
- `gym eval compare-suite` (after 0.7.0) normalizes each selected metric to 0–100 using its declared bounds, averages metrics within each repeat, weights benchmarks equally within a model, and reports a 90% Welch interval by default. It excludes a model–benchmark pair with fewer than 2 repeats on either side, with a warning.

### Rules of Thumb

- One run's standard error on n independent tasks is at most `sqrt(p * (1 - p) / n)` at pass rate p: about 5 points for 100 tasks at p = 0.5, 2.2 for 500 and 1.1 for 2,000. Shared failures, such as an outage that hits many tasks in one run, push it higher, so measure spread with repeats.
- The standard error of the gap between two independent numbers is `sqrt(se_a^2 + se_b^2)`, both on the same scale. A gap under about twice that is not evidence of a difference. With only a few repeats, the multiplier is larger than 2; use the Welch interval.
- A published number without a standard error: estimate it as `sd1 / sqrt(k_ref)`, with `sd1` from the Gym run and `k_ref` the reference's repeat count (1 if unknown). A reference measured on a different sample of tasks adds the task-sampling error above.
- About `(sd1 / s)^2` repeats give a standard error s. A benchmark whose single repeat varies by 4 points needs about 16 repeats for a 1-point standard error. Use `nemo-gym-repeat-planning` in the Gym repo to plan a comparison in advance.
- Agentic benchmarks with tens of tasks move several points between single runs. Average several runs before calling a regression.

### Spread That More Repeats Won't Remove

- Different clusters or load: wall-clock timeouts bite harder on slower serving.
- Resumes and retries: `--resume` runs retryable failures again, and a later success counts. A reference that allows one attempt per rollout is not counting the same thing.
- Infrastructure nondeterminism: environment setup, patch application and sandbox restarts. A deterministic verifier must give identical outputs identical rewards; check that across runs.
- Judges: an unpinned judge model or a changed judge prompt shifts every score. Judge sampling at a nonzero temperature adds noise that repeats do average out. Re-judge a sample with `gym eval reverify` and measure agreement.
