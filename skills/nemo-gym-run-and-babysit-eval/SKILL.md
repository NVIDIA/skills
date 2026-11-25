---
name: nemo-gym-run-and-babysit-eval
description: >
  Take a NeMo Gym evaluation from a request to a trusted result: compose the eval config when none
  exists (by following add-model-eval-config, in the same task), submit it, let the resumable Slurm
  chain continue through walltime limits, preemption and node loss, inspect and recover the run with
  `gym eval resubmit --recover`, check health and the acceptance gates, and decide when to stop a
  chain or start a fresh run. The entry point for "run evaluations on model Y on benchmark Z", with
  or without a ready config. Not for building a benchmark or harness (use add-benchmark or
  add-harness). Triggered by: "run the eval", "run evaluations on model Y on
  benchmark Z with serving params X", "submit this benchmark", "the job timed out", "the chain
  stopped", "resume the run", "recover the run", "is the run done", "inspect the run", "gate the
  score", "restart the evaluation".
  Also covers local runs, existing API endpoints, supported agents and Gym task folders.
---

# Run a Gym Evaluation to Its Result

## Select an execution path

- For a local run or an existing model endpoint, follow [local and API-endpoint runs](references/local-and-api.md).
  It covers pip preview installation, built-in benchmarks, and Responses, Chat Completions and vLLM clients.
- To evaluate an agent Gym already supports, first read [supported agents](references/supported-agents.md)
  for harness selection and model wiring, then follow the matching local or cluster path.
- For tasks already in Gym's task-folder format, follow [task folders](references/task-folders.md).
- For a submitted Slurm run or a self-hosted model on a cluster, follow the existing procedure below,
  including [model configuration](references/add-model-eval-config.md) when needed.

Finish direct runs with the [saved-result checks](references/local-results.md): require complete coverage,
read health and grading evidence, and reconcile the reported score with the saved benchmark aggregate.
Evaluation against Gym servers already started with `gym env start` follows [supported agents](references/supported-agents.md);
the authoring links below apply when building a benchmark or harness.
The remaining sections describe submitted Slurm evaluations; direct runs use the references above.

The Gym pages are the reference; when they and this skill disagree, the pages win:
[Configuration](https://github.com/NVIDIA-NeMo/Gym/blob/main/fern/versions/latest/pages/orchestration/configuration.mdx),
[Resumable Runs](https://github.com/NVIDIA-NeMo/Gym/blob/main/fern/versions/latest/pages/orchestration/how-to/resumable-runs.mdx),
[Job Records](https://github.com/NVIDIA-NeMo/Gym/blob/main/fern/versions/latest/pages/orchestration/job-records.mdx) and
[Resume](https://github.com/NVIDIA-NeMo/Gym/blob/main/fern/versions/latest/pages/evaluation/resume.mdx). For servers you started yourself
(`gym env start`, then `gym eval run --no-serve`) and for reading health while you build a benchmark or
a harness, follow [add-benchmark](https://github.com/NVIDIA-NeMo/Gym/blob/main/.agents/skills/add-benchmark/SKILL.md) and [add-harness](https://github.com/NVIDIA-NeMo/Gym/blob/main/.agents/skills/add-harness/SKILL.md).
The page links point into a full Gym checkout (`fern/...`); a copy of this skill installed on its own does not
contain those pages, so read them in the Gym repository.

## How to act

0. **No config yet: compose it** (section 0) in the same task; the user does not call a second skill.
1. **Submit pinned and resumable** (section 1), a small smoke first.
2. **Inspect, never guess** (section 3): read `decision.action` and `gate_task.status`.
3. **Do the one action the table names** (section 4): run the same command without `--inspect`, or add the
   one option the row names, with its reason. A decision with no row, or a STOP you cannot explain:
   stop and report it.
4. **Never** use plain `gym eval resubmit` or `gym job resubmit`, edit a run's files, or regenerate a saved answer.
5. **Done** only when the inspect says `SUCCESS` (section 6). Report the health, the corrections of
   section 7 and every operator action with its reason.

## 0. No config yet: compose it

A request such as "run evaluations on model Y on benchmark Z with serving params X" often comes without a
config. Then write it yourself before section 1: read [add-model-eval-config](references/add-model-eval-config.md)
and follow its sections 1 to 5 (the model card, the policy service with the serving params the request gives,
the reasoning toggle, each benchmark's own config from `benchmarks/<name>/`, validation without GPUs). A
quantized export of a model that also runs unquantized follows its section 9. Ask the user only for what that
skill says to ask for (the cluster and its login host, the account and partition, the images, the checkpoint
path, the output directory), never for a second skill to be run. Its smoke (section 6) is the smoke of
section 1 here; then continue with section 1. A config the user hands you is used as it is: do not re-derive
its model settings. A deployment handed over as a `services.policy` block is used as it is too
(add-model-eval-config, section 2).

## 1. Submit

The `gym` commands below need the `gym` CLI on the host you submit from. Install it at the commit in the
config's `driver.gym_install.ref`, in a directory that persists (not `/tmp`), with `uv` on `PATH`:

```bash
SHA=<the full 40-character commit>
git clone -q https://github.com/NVIDIA-NeMo/Gym.git ~/gym-${SHA:0:12} && cd ~/gym-${SHA:0:12} && git checkout -q $SHA && uv venv --seed .venv && . .venv/bin/activate && uv pip install -e ".[mlflow]" && gym --help
```

Without `uv`, install it with `curl -LsSf https://astral.sh/uv/install.sh | sh` and
`export PATH="$HOME/.local/bin:$PATH"`. The submitted run installs its own extras and `ray`; the launcher
needs only the `mlflow` extra, which the MLflow delivery check in `--inspect` uses.

Submit with `gym eval submit --config <config> [overrides]`; the config may be a Hydra defaults document
that carries its own `hydra.searchpath`; overrides are Hydra (`++k=v` set, `+k=v` add, `~k` delete, e.g.
`'~cluster'` when an imported fragment brings a key the submit config rejects); `--json` prints the
submission record.

```bash
gym eval submit --config submit.yaml --resolve-only   # the resolved config, validated
gym eval submit --config submit.yaml --dry-run        # the job scripts it would queue
gym eval submit --config submit.yaml                  # --select a,b submits only those
```

The job id is `gym_job_id` in the run directory's `gym-job.json`. The `Run directory:` line ends with it
but wraps when stdout is not a terminal. The same file has each benchmark's Slurm `job_id` and the Gym
commit that submitted the run.

Before you submit, check the config has:

1. Gym pinned: `driver.gym_install.ref` is a full commit, not a branch. Recovery refuses an unpinned ref.
2. Every long benchmark resumable under `driver.benchmarks.<name>`: `resumable: true`, or
   `resumable: {max_retries: N, max_walltime: "D-HH:MM:SS"}`, where `max_walltime` is the whole chain's
   budget (for example 3 x `compute.cluster.walltime`), not one job's.
3. A small smoke first: `run: {limit: 5, num_repeats: 1}`. `--limit` counts tasks, so rows = tasks ×
   repeats. A smoke that must stay in one allocation also sets `resumable: {max_retries: 0, max_walltime:
   <the walltime>}`. A named, versioned subset runs with `run: {task_ids_fpath: <file, one id per line>}`
   (or `task_ids: [...]`) instead: it keeps every copy of the named tasks, the run writes
   `rollouts_task_selection.json` (source, ids, sha256) when it starts, resume and recover keep it, and the
   inspect (`task_selection`), the MLflow run (`gym_task_selection=subset`) and `gym eval compare` label the
   score a subset score.
4. From a machine that is not the login node: `compute.cluster.hostname`, and every `runtime:` credential
   in `compute.cluster.forward_env`; only listed names reach the job. Site `#SBATCH` directives go in
   `compute.cluster.extra_args` without leading dashes (`{qos: <qos>, exclude: <node>}`).
5. `mlflow_tracking_required: false` in each benchmark's `run` when the run is not tracked in MLflow on
   purpose; the delivery state is then `not_required`.
6. A raised `run.server_spinup_timeout_seconds` (default 600 s) for a large model, a fresh install or a
   loaded cluster; a spin-up timeout is an infra stop, never a zero (section 4).

## 2. Let the chain run

A resumable benchmark is a chain of Slurm jobs. Each job queues its successor (`afternotok`) and a
sentinel (`afterany`), then reads its predecessor's final state:

| Predecessor ended | The chain |
| --- | --- |
| `TIMEOUT` | Continues from the saved rows, within `max_walltime`. A rollout in flight when an allocation ends is rerun from the start: the first such end of an in-flight rollout is free; from the second end on, each one charges the affected rows one attempt (`rollout_allocation_ended`, evaluation-side, counted in health; retried under the attempt cap, and final at it). The Store cannot tell a walltime cut from a preemption or a node failure, so the rule covers any ended allocation. |
| `PREEMPTED`, `NODE_FAIL`, `BOOT_FAIL` | Retries, spending one of `max_retries`. |
| `COMPLETED` | Ends. The job that finishes the run cancels the follow-up and sentinel it queued itself and journals both in `chain-end-journal.jsonl`. The sentinel its predecessor queued for it is not cancelled: it starts once the job ends and exits with nothing to do. |
| `FAILED` | The saved evidence decides: a driver that exited 75 (unfinished work) after saving progress continues as after a `TIMEOUT`; a retryable saved failure spends one of `max_retries`; anything else stops. |
| `CANCELLED`, `OUT_OF_MEMORY`, `DEADLINE`, unknown accounting | Stops. A segment the scheduler requeued has unknown elapsed time and stops this way. |
| `TIMEOUT` before the task inventory was written | Stops and names the fresh-restart command. |

The chain is over when none of its jobs is `PENDING` or `RUNNING` (`squeue --me`); a leftover pending
sentinel may be cancelled.

An empty `sacct` reply for the previous job is retried: the job asks up to 6 times, 10 seconds apart,
and then stops the chain with accounting unknown instead of resuming. A scheduler whose accounting lags
by about a minute therefore stops chains. The stop's log line names the continuation,
`gym eval resubmit <id> <benchmark> --recover --max-walltime HH:MM:SS --walltime-reason TEXT`, adding
`--extend-retries N` when the retries are spent, and `gym eval resubmit --recover --inspect` (section 3)
reports it too; the `.gym_chain_incomplete` marker keeps the bare reason.

When sbatch reports the scheduler's refusal of the follow-up (for example a QOS limit on submitted jobs),
the chain records `follow_up_submission_refused: sbatch exited <rc>: <error lines>` in
`.gym_chain_incomplete`, queues nothing, and its log line says to continue with
`gym eval resubmit --recover` once the scheduler accepts the job; a reply that is missing or says the
request was lost stays accounting unknown. When a later segment completes the run, it removes an earlier
stop marker, logs it, and journals `run_complete_stop_superseded` in `chain-end-journal.jsonl`.

A driver that stops on a typed condition exits at once and the next guard writes its kind and the way
out to `.gym_chain_incomplete`. The commands are in the decision table (section 4). Do not resubmit while a
job of the run is live.

While the chain runs, read its progress with the read-only inspection of section 3 (`gym eval resubmit
"$GYM_JOB_ID" <benchmark> --recover --inspect`, which reports `RUNNING` while a job is live) or with
`gym eval status "$GYM_JOB_ID"`, which reads `artifacts/rollouts_batch_status.json` (expected, completed and
remaining rollouts per agent, from the current attempts) without a scheduler. Counting the lines of
`artifacts/rollouts.jsonl` is only a fallback: it reads the whole store, which can be large and holds every
attempt.

## 3. Inspect first, always

```bash
gym eval resubmit "$GYM_JOB_ID" <benchmark> --recover --inspect \
  --score-agent <agent name in the aggregate> \
  --score-key '["agent_metrics", "mean/reward"]'
```

The agent name is `agent_ref.name` of the rows, the agent entry in
`artifacts/rollouts_aggregate_metrics.json`; without the score flags the score gate reports "score
unavailable". `--inspect` reads the saved evidence on the run's host and evaluates the gates; it
dispatches and writes nothing. The first read of a Gym pin installs it on that host (minutes; later
reads reuse it). The decision is the same without `--inspect`. A file that changes under the read stops
it ("unstable sources"), so never edit a run while you read it. Keep the JSON line: it is your record.

- **Another machine**: `No submission '<id>' in this machine's job index` → add `--record <run
  dir>/gym-job.json` (inspect only). To dispatch from a machine whose index lost the entry, add
  `--run-dir <run dir>`: the entry is restored from `gym-job.json` and journaled
  (`index-restores.jsonl`).
- **Older pin**: a run pinned at another Gym commit is planned with the installed Gym's rules, and when
  that plan needs the installed revision, the recovery moves the run to it (journaled in
  `repin-journal.jsonl`, shown in `pin_plan`). Runs you compare stay on one Gym commit: inspect each
  with `--no-auto-repin`.
- **`UNKNOWN`** with "nothing was read or decided" is a reader that did not run (reason on stderr), not a
  run fault; fix the reason and rerun the identical inspect; `NEMO_GYM_REMOTE_READER_SRUN` takes extra
  srun args if it needs a partition or QOS, and `NEMO_GYM_REMOTE_READER_TIMEOUT_S` raises its time bound
  (300 to 14400 s; by default it grows with the size of the rollout files). A big store needs memory
  (`--mem=<size>` in `NEMO_GYM_REMOTE_READER_SRUN`; on a login node the reader ends with `MemoryError`) and more
  time than a tool call that is cut at about 10 minutes allows: start the inspect or the recovery detached
  (`setsid nohup <command> > <log> 2>&1 &`) and poll the log. If a large-store reader still stalls or times out
  after one retry with the 14400 s bound and more memory, report the store size, command and reader diagnostics
  as a suspected Gym reader issue, and say the run is blocked on the reader. If inspection reports "saved rows
  are unreadable", check the named files' read permissions and storage errors before attributing the failure
  to Gym. Never report either condition as "nothing to recover".
- **Exit code**: 0 only when `gate_task.status` is `SUCCESS`. A dispatching call exits 1 when it carries
  a score, MLflow or health-limit option, and a call handed to a queue can exit 0 while the gate still
  reads `RESUMING`: read `ok`, `dispatched` and `job_id` from the JSON line, not the exit code.

The line's keys: `decision` (`action`, `reason`), `gate_task`, `ok`, `dispatched`, `job_id`,
`unsupported`, `operator_verdict`, `mlflow`, `reverify`, `judge_job`, `walltime_plan`, `pin_plan`,
`scheduler_refusal`, `reader_token`.

- `decision.action`: `noop`, `reverify`, `resubmit`, `fresh-restart`, `stop`, with its `reason` (it ends
  with `resumed across operational drift: <settings>` when the latest resume crossed such drift).
- `gate_task.status`: `RUNNING` (a job is live), `PLANNED` (eligible work, nothing dispatched; a
  dispatching call reports `RESUMING`), `FAILED` (stopped, unsupported or something failed: read
  `errors`), `UNVERIFIED` (complete, nothing failed, acceptance not proven: read `warnings`), `SUCCESS`
  (complete and every gate passed). A chain error turns `SUCCESS` or `UNVERIFIED` into `FAILED`; MLflow
  delivery never does.
- `gate_task.evaluation_health` (null until complete): `status` (`PASS`, `FAIL`, `UNVERIFIED`),
  `evaluation_failure_rate`, `evaluation` and `model` (`count`, `fraction`, `kinds`),
  `trajectory_evidence`. Health `PASS` means no evaluation-side fault was seen, not that trajectories
  were checked: read `trajectory_evidence`, and treat `in_session_unrecorded` as thin evidence.
- `gate_task.failures` (response failures: `truncated`, `empty`, `malformed`; trial failures by cause:
  `request`, `judge`, `sandbox`, `harness`, `timeout`, `allocation` (a rollout in flight at repeated ends
  of its allocation), `other`; `output_budget_refits`: model calls sent again with a lowered output budget;
  `repetition_stops`: policy calls the serving engine stopped for repetition, the model's own loops, scored
  as answered and never a failure;
  `agent_adaptations`: what an agent adapted on its own, for example a GDP-PDF input-limit rejection),
  `trial_counts` (`observed_healthy`,
  `unhealthy`, `unobserved`: unobserved trials are unverified, not healthy), `scored_samples`,
  `omitted_samples`, `score`, `errors`, `warnings`, `retry_budget` (`infra` and `attempts`, each with its
  `configured` cap and recorded `extension`), `observed_grading_recovery` (counts of recovered grading
  attempts; null when there are none), `health_verdict`, and `mlflow_delivery`.

- `operator_verdict` (also inside `gate_task`, and printed on stderr as `operator verdict: <state>: <reason>;
  action: <action>`): the one reading to act on. `accept` is `SUCCESS` with complete coverage, run health `PASS`
  and a `noop` decision. `needs_action` is any STOP, `FAILED`, incomplete coverage or failed run health. `unknown`
  is a live job, evidence the reader could not read, or any `UNVERIFIED`: by configuration (verification skipped),
  by a producer that records no grading (`gate_task.unverified_cause` and the one-sentence warning name it with
  counts; the full reason is in `evaluation_health.detail`), after a re-judgment whose grading was not recorded,
  or without a known cause. No recovery changes an `UNVERIFIED`, so its action starts `none:` and says what to
  report and where the fix is (references/benchmark-notes.md); failed run health on a `noop` decision is reported
  the same way, as no recovery reopens settled trials. Every command an action prints is complete: `gym eval
  resubmit <job id> <benchmark> --recover`, every option the inspected or recovered call was given (the pin
  choice and the gate options included), the typed stop's fix, then what the stop needs (an extension, say). The
  action names `gym eval submit` only where Gym's own stop text does (`environment_protocol_violation`,
  `collection_resume_refused`). An `accept` whose MLflow delivery is still pending or unverified names the export
  in its action, or reports it as unverified by configuration when the rollouts were not uploaded. Rollout quality
  never changes the gate's status, but a run whose every rollout ended on a failed model call (or whose every task
  had no successful model call) is not an `accept`: the verdict is `needs_action`, because the score most likely
  does not measure the model (serving or credentials broken); its action is to check the served configuration and
  read the rollout quality before accepting or reporting the score.

Run accounting and rollout quality are two things to read. Run accounting (`run.artifacts.healthy` and
`unhealthy_reasons` in `quality_summary.json`) counts expected, completed, missing and failed rollouts; a failure
counts whatever its cause, so a counted model zero is a failure row and can reach a configured failure limit.
Rollout quality (`run.verdicts`, per-rollout `findings`, task flags such as `task_no_successful_model_calls`)
says how far to trust each measurement and fails nothing. Accept or promote a smoke or a run only on the
recovery gate's `gate_task.status` and `evaluation_health.status` from `--recover --inspect`, as in section 6;
a healthy accounting with every rollout measured can still be `UNVERIFIED`.

Where the evidence lives, relative to the run directory: `gym-job.json` (the submission record),
`resolved-config.yaml`, the journals (`walltime-extension-journal.jsonl`, `repin-journal.jsonl`,
`unsafe-resume-journal.jsonl`); in `<benchmark>/`, `logs/driver-<job id>.log` and
`logs/<service>-<job id>.log`, the chain files (`.gym_job_chain`, `.gym_followup_queued.<job id>`,
`.gym_followup_sentinel.<job id>`, `.gym_chain_incomplete`, the last holding a provisional "Continuation
<job> started and had not queued its follow-up" line while a continuation starts: act on it only if it is
still there after the segment ended), `retry-extension-journal.jsonl`; in `artifacts/`, `rollouts.jsonl`,
`rollouts_failures.jsonl`, `rollouts_materialized_inputs.jsonl`, `rollouts_coverage.json`,
`rollouts_aggregate_metrics.json`, `quality_summary.json` and `rollouts_mlflow_export_done`. Read them;
never edit or delete them. The run directory also holds chain bookkeeping (`chain-end-journal.jsonl`,
`.gym_accumulated_walltime`, `.gym_infra_retries`, `.gym_rows_at_start.<job id>`, `.gym_walltime_before.<job id>`):
read it, do not edit it. `quality_summary.json` and `rollouts_aggregate_metrics.json` are written readable by the
submitting user only (mode 0600) in the runs read: as another user, say they are unreadable rather than absent.

## 4. Act on the decision

When the operator has asked you to complete or recover a run, inspect it and, if no continuation is live,
run the recovery the decision permits in the same turn and babysit it. Keep an intentionally stopped run
stopped unless the operator asks to continue it. A STOP that needs an owner's approval or a fix still waits.

Plain `gym eval resubmit` without `--recover`, and `gym job resubmit`, which is the same plain command, reset the budgets and the per-rollout attempt allowance:
it is not a substitute for any row below. Every row is the same command as the inspect, without
`--inspect` and with the option the row names. Facts about single benchmarks (agent time limits,
GDP-PDF, Stirrup, the benchmarks that read `UNVERIFIED` by construction) are in
[references/benchmark-notes.md](references/benchmark-notes.md).

| The inspection says | Do |
| --- | --- |
| `operator_verdict` is `needs_action` | Find the row below that matches the stop reason or the typed failure and follow it; the printed action is the default when no row matches. The action prints one complete command: `gym eval resubmit <job id> <benchmark> --recover`, every option the inspected or recovered call was given (`--no-auto-repin`, `--judge-partition`/`--judge-qos`, `--server-spinup-timeout-seconds`, the score and threshold options, ...), the fix of the typed failure the chain or the rows recorded before it (fix the credentials, raise the start-up limit, wait for the quota reset), then what the stop needs (`--extend-attempts`, `--extend-retries`, `--max-walltime` with `--walltime-reason`, `--rerun-masked-without-artifact`, `--repin`, or `--acknowledge-failed-judge-jobs` after reading the failed judge-only jobs). An option the run refused (`--extend-attempts` on a run without a rollout store, say) is dropped when the run can resume without it. When it first asks to wait for the scheduler, a live job or the allocation, to reconcile a follow-up claim, or to make unreadable files readable, do that and inspect again before any recovery. A stopped recovery records none of its options, so repeat them all. When it says no recovery command is known, read the stop reason and the rows below; never fall back to `gym eval submit` while the saved run can continue: a fresh submit samples every policy rollout again. |
| NOOP, `RUNNING` (`continuation <job> live`) | Wait for the job to end, then inspect again. |
| Eligible work, `PLANNED` | Run it, with the variables the saved config forwards (`compute.cluster.forward_env`) set in that shell; a missing one stops the dispatch. A run submitted from the login node usually forwards none: its credentials are the `runtime:` entries of the saved `resolved-config.yaml`, which the dispatch does not check, so set those variables (an `HF_TOKEN`, say) in the dispatching shell too, or the resumed job starts without them. |
| `reverify`, `PLANNED` (only rows with a saved answer remain: a judge failure, or a verify call lost in transport) | Run it: the saved answers are judged again and the policy is not called. A run that served its own model, and nothing else beside it, gets a judge-only job on one CPU node. Its partition is the first of: `--judge-partition <partition>`, a node pool of the saved config with `gpus_per_node: 0`, or the partition `NEMO_GYM_REMOTE_READER_SRUN` names. Its QOS is the first of: `--judge-qos <qos>`, the QOS of the saved CPU pool, the `-q`/`--qos` of `NEMO_GYM_REMOTE_READER_SRUN` when it names this partition or none, the partition's default. Slurm refuses a GPU QOS on a CPU partition (`QOSMinGRES`): pass `--judge-qos`. With no CPU pool and no reader placement, pass `--judge-partition`, or the recovery stops with nothing queued (next row). A run that serves its judge on a node pool of its own gets a judge-only job on that saved pool instead and keeps the run's QOS; `--judge-partition` and `--judge-qos` are not used (when you pass them the report says `--judge-partition/--judge-qos apply to the CPU judge profile only; not used`). A run with both judge failures and policy work is resubmitted; it re-judges the saved failures before its policy work and again at the end. |
| STOP on a `reverify` of a GPU-served run: `... has no existing CPU-only judge execution profile and none can be derived: no CPU partition is known` | Nothing was queued and no row was lost. Run it again with `--judge-partition <cpu partition>` (and `--judge-qos <qos>` on `QOSMinGRES`). If the message says `this submission has no served-judge execution profile: service(s) <names> run on the policy's nodes (...)`, no partition helps: the judge-only job does not start the policy's allocation and policy services are not restarted. Give each of those services a node pool of its own, beside the policy's, in the submission config, or report that the saved answers cannot be re-judged. |
| STOP on a `reverify`: `the last 2 judge-only job(s) (<ids>) ended without settling a saved answer; read their receipts and logs ...` | Nothing was queued. Read those jobs' receipts and logs under `<job dir>/reverification` (the message prints the path; in a finished run it is
`<run dir>/<benchmark>/artifacts/reverification`), then allow exactly one more with `--recover --acknowledge-failed-judge-jobs --acknowledge-reason "<why>"`; it is journaled with the job ids it covers in `judge-job-acknowledgement-journal.jsonl`, never covers a later pair of failures, and `--inspect` names the block without writing anything. Never edit the judge-job journal. |
| STOP typed `server_spinup_timeout` | `--recover --server-spinup-timeout-seconds N --spinup-timeout-reason "<why>"` (N above the recorded limit, 0 for none; 600 s unless the benchmark's `run` sets it). |
| STOP typed `driver_install_failed` (the stop omits the installer's error text, which can carry a credentialed URL; `logs/driver-<job id>.log` has it) | Fix the cause (the ref, the network, the image), then `--recover`. |
| STOP typed `provider_credentials_rejected` (the sandbox provider answered 401 or 403) | Fix the credentials, then `--recover`. |
| FAILED, the driver log ends `ValueError: Environment server must return an episode result or failure.` (or `InvalidRolloutResult`), the chain did not continue | A server replied in a shape the collector cannot use. It is deterministic: retrying cannot help. Fix the server or the benchmark pairing (the benchmark config names which server), optionally set `+route_failures_to_sidecar=true` to keep the other tasks' results while diagnosing, then `gym eval submit` again. Never `--recover`. |
| STOP typed `environment_protocol_violation` (the first results of one environment server were all replies the collector cannot use) | Same: fix the server or the benchmark pairing, then `gym eval submit` again. Never `--recover`. |
| STOP typed `collection_resume_refused`, or a **semantic** configuration difference (data, limits, repeats, model, sampling, judge or grading) | Usually a fresh run (section 8). Only when the operator accepts that exact difference: `--recover --allow-unsafe-resume --unsafe-resume-reason "<why>"` (records the reason and the user; never bypasses corrupt history). Operational differences (timeouts, retries, ports, concurrency, paths, tracking) resume by themselves with a warning. If the saved fingerprint no longer matches, continue at the run's own pin and report it; a resume on a newer Gym can be refused after setup, which costs an allocation. |
| STOP: a resumed segment saved no rollout (its walltime was shorter than setup plus one rollout) | `--recover --max-walltime HH:MM:SS --walltime-reason "<why>"`, when the partition allows a longer segment. Each segment requests the smaller of `--max-walltime` and the partition's MaxTime (`walltime_plan` shows `each segment requests ...`), and the recovery keeps the recorded partition: a raise past MaxTime lifts only the chain cap, so a segment that saved no rollout twice at MaxTime will do it again. Then submit a new run on a partition with a longer MaxTime (the compute pool's `partition` in the submit document), and raise `compute.<cluster>.walltime` above setup plus one rollout. If the benchmark sets `resumable.max_walltime`, make sure that chain cap permits the full segment too. Changing the partition alone does not raise the requested walltime. Follow section 8, keep the old run directory and name it in the report, because nothing links the two runs; when earlier segments saved no rollout there is nothing to lose. |
| The chain's walltime cap is spent, or the chain's accounting is unknown (a requeued segment) | `--recover --max-walltime HH:MM:SS --walltime-reason "<why>"`, never below the recorded walltime; journaled. For unknown accounting the explicit budget is what continues it, optionally with `--extend-retries`. When both budgets are spent, give `--extend-retries N` and `--max-walltime HH:MM:SS --walltime-reason "<why>"` in one call. |
| The infra-retry budget is spent | `--recover --extend-retries N`; the spent count is kept; journaled. It raises infra retries only, not the per-row attempt cap. When both budgets are spent, give `--extend-retries N` and `--max-walltime HH:MM:SS --walltime-reason "<why>"` in one call. |
| Rows used every attempt (`exhausted`, `max_rollout_attempts`) | `--recover --extend-attempts N`. Only rollouts the Store would run again are reopened (infrastructure failures, allocation cuts); a model failure or a saved result never is. Recorded in `artifacts/rollouts_attempt_extension`. |
| A Gym fix is needed to continue, or the launcher's Gym differs from the run's pin | Install that commit as the launcher, then `--recover --repin <full sha> --repin-reason "<why>"`. Saved rows and budgets stay. |
| STOP: N completed result(s) are masked and not measured | They are kept and a resume would run nothing; the run is not complete. Report the count and the reason. Nothing re-grades a masked row that has a saved answer but neither a gradeable artifact (no intact patch, say) nor a recorded grade; `--inspect` names those rows apart ("without a gradeable artifact") and keeps them masked. Only when the operator wants them sampled again: `--recover --rerun-masked-without-artifact --rerun-reason "<why>"` (below the table). The rerun covers only those rows: any other masked row stays masked and the run stays incomplete, as the verdict says. |
| `fresh-restart` (no task inventory) | Section 8. |
| NOOP, `SUCCESS` or `UNVERIFIED`, and `gate_task.mlflow_delivery.state` is `pending` or `unverified` (warning `MLflow delivery <state>: <reasons>`) | The results are complete; delivery is reported beside the gate, never inside it. Close it: `gym eval export mlflow --rollouts <run dir>/<benchmark>/artifacts/rollouts.jsonl +mlflow_tracking_uri=<uri> +mlflow_experiment_name=<name> +mlflow_run_name=<name>` with the current Gym, the three values from the run's `artifacts/rollouts_resolved_config.yaml`, and the token as `+mlflow_tracking_token=<token>` or `MLFLOW_TRACKING_TOKEN`. A repeat on an exported run verifies without touching the run lock; `--recover` on a complete run that requires tracking and has no receipt runs the export too. "although the saved config snapshot names a tracking URI...": tracking was required and the export did not finish: read the exporter's failure in the driver log, repair, export. "does not name all of the tracking URI, experiment and run name": nothing tracked the run; set `mlflow_tracking_required: false` next time. From a host without access to the tracking server the inspect reports `unverified` with `status query failed: ...`: inspect with `--mlflow-run-id <receipt run_id> --mlflow-status FINISHED` once you observed that status yourself (`GET <tracking uri>/api/2.0/mlflow/runs/get?run_id=<id>`, field `info.status`); `ModuleNotFoundError` means the reader's environment lacks `mlflow`. When the reason says the rollouts were not uploaded and the run sets `upload_rollouts: false`, the delivery stays `unverified` by configuration: report it; an export does not change it. |
| `SUCCESS` | Done. Keep the run directory as the record. The finishing job cancelled the follow-up and sentinel it queued (`chain-end-journal.jsonl`); a pending sentinel its predecessor queued may be cancelled. |

A judge-only re-judge takes the judge key and the other forwarded credentials from the environment of the shell
that dispatches it; the run keeps none of them. Export the valid key before dispatching: a stale key in that
shell fails every judge call again.

A rollout cut by a segment end restarts from its first step; there is no step-level resume. So the
segment walltime (`compute.cluster.walltime`) must exceed the longest single task's runtime: shorter
segments restart the same task every time, and chaining more of them repeats the same work.

Three limits, each counting something different: the **chain cap** (`max_walltime`, summed over
segments) is raised with `--max-walltime`; the **infra-retry budget** (preemptions, node failures) with
`--extend-retries`; the **per-row attempt cap** (`NEMO_GYM_MAX_ROLLOUT_ATTEMPTS`, default 3, read from the
environment of the process that runs `sbatch`, recorded in the manifest as `max_rollout_attempts`) with
`--extend-attempts`. A cut that ends an allocation while a row is in flight charges that row an attempt
(except the first), so a long task spanning several segments can reach the cap with no model failure:
such a row is masked, never scored as the model's, and a resubmit does not revive it. `--max-walltime`
takes `--walltime-reason`; `--extend-retries` and `--extend-attempts` take none, and their journal line
records the count and its kind, so put the reason in your report. A long-task benchmark whose 16-hour cap
was spent with rows missing continues with `--max-walltime 24:00:00 --walltime-reason "<why>"`.

A masked result that kept its saved answer is never sampled again by default, and when it has neither a gradeable artifact nor a recorded grade nothing can measure it. `--recover --rerun-masked-without-artifact --rerun-reason "<why>"` is the explicit opt-in: it appends one line (time, reason, each identity with its attempt count) to `artifacts/rollouts_rerun_masked_journal.jsonl`, and each listed row then counts one failed attempt, runs again within its attempt cap and is superseded by the new attempt. The reason is required, `--inspect` never writes the journal, and only rows the plan finds qualifying at that moment are approved.

## 5. Retriable and final

A saved failure is retriable or final; the run's recovery outcome policy decides, one action per
failure: `retry` (another attempt, within the attempt cap and the infra-retry cap), `verify` (re-judge
the saved answer without calling the policy) or `fail` (final and ungraded: recovery stops for an owner
decision). The action comes from the failure's HTTP status, else its kind, else the default. Defaults:
`judge_failed` is re-judged; `collection_dispatch_failed` and `harness_termination_unconfirmed` are
retried; every other saved failure is retried (`default_action: retry`). Settled results are final and
never reopened: a scored answer or zero (including a typed answer failure: truncated, empty or
malformed), and a clean agent timeout, which is the model's result under the benchmark's own time limit.
An `agent_timeout` that kept no answer is a saved failure the policy decides, retried by default
(`recovery_outcome_policy: {by_failure_kind: {agent_timeout: fail}}` keeps it final instead). A
failure the Environment Server marked terminal (a typed model failure such as `model_context_limit`) is
never retried. `--inspect` lists the identities in each state and the policy in force (`outcome_policy`,
with its `source`: `default`, `config` or `python`).

Retries happen in three layers before a failure reaches you: **in a call** (the policy client retries
HTTP 429, 502, 503, 504, 520, 404, 408 and 500; the judge client 408, 429, any 5xx, a timeout or a lost
connection, three attempts in all; the sandbox client 408, 409, 425, 429 and the 5xx codes); **in a run**
(a resumed segment dispatches every unsettled identity again, within the attempt cap); **in the chain**
(`PREEMPTED`, `NODE_FAIL` and `BOOT_FAIL` spend an infra retry, `TIMEOUT` continues, `FAILED` goes by its
saved evidence (section 2), `OUT_OF_MEMORY`, `DEADLINE` and `CANCELLED` stop it; `--recover` is the way on
and dispatches only the missing and retriable identities).

- **A judge failure** (`judge_failed`) or a verify call lost in transport is re-verified from its saved
  answer, never regenerated, and is never the model's zero: it is masked, out of the score, counted on the
  evaluation side.
- **`judge_unparseable`**: a row whose judge returned no usable verdict and was saved as a scored 0.0 with
  no failure record is planned for a re-judge.
  The inspect reads `reverify`, `PLANNED` and exits 1 until the re-judge settles it; meanwhile
  health still counts the row as an evaluation-side fault, `scored_samples` leaves it out, `errors` lists
  it and the MLflow export waits. Run the `reverify` row. A judge served on a node pool of its own is
  re-judged by a judge-only job on that pool; one served on the policy's own nodes still cannot be: report
  it with the health count.
- **An unsupported parameter value** rejected by the provider is an evaluation-side fault: it invalidates
  the run and is never the model's zero. Fix the parameter and run again.
- When the budget is spent or the cluster was down, run the recovery again after `--inspect`; running the
  same command does not extend an exhausted budget, the three options do.

What is retriable is one question; whether the score can be trusted is another: `evaluation_health`
(section 6) counts the evaluation-side faults and never fails on model faults.
The outcome policy never changes health, and health never changes what is retried.

A benchmark config changes what is retriable, before the run, with `recovery_outcome_policy` under
`resumable`:

```yaml
resumable:
  max_retries: 3
  max_walltime: "16:00:00"
  recovery_outcome_policy:
    by_failure_kind: {agent_timeout: fail}
    by_status_code: {429: retry}
    default_action: retry
```

Entries it does not name keep their defaults. Kinds must be registered (or `unknown`), statuses 400 to
599, `count_as_zero` and unknown keys are refused at config load (`invalid recovery_outcome_policy:
<field>: <problem>`). The run saves the policy in `resolved-config.yaml`; a change between the inspection
and the dispatch refuses the resubmit. The CLI extends budgets and has no flag for the policy; a Python
caller can pass `outcome_policy` to `recover(...)`, which replaces the config's section as a whole:

```python
from nemo_gym.agents import OutcomeAction
from nemo_gym.eval.recovery.policy import RecoveryOutcomePolicyConfig
from nemo_gym.orchestration.submit import recover

recover(job_id, benchmark, outcome_policy=RecoveryOutcomePolicyConfig(by_status_code={401: OutcomeAction.FAIL}))
```

| The product decides | The operator decides |
| --- | --- |
| Retrying the failures the outcome policy marks retriable, within the caps. | Raising the walltime cap or a budget, with a reason; what the config marks retriable, before the run. |
| That settled results are final. | Moving a run to another Gym commit (`--repin`). |
| Re-judging saved answers whose judge failed, without calling the policy. | Accepting a semantic configuration difference, or starting a fresh run. |
| Health verdicts and the gate results. | Whether a result is accepted, on the gate's facts. |
| Resuming across operational drift, and saying so. | When to stop a chain, and when to start over in a new location. |

## 6. Health and acceptance

A run can be healthy and still score low: that is the model or its deployment. Fix the deployment and run
again, not the evaluation. For the evaluation itself, only `num_repeats` and the client parallelism
(`--concurrency`) may change from the command line or a benchmark's `run`, and `num_repeats` only upward.
The benchmark's own count is `num_repeats` in its `benchmarks/<name>/manifest.yaml`; `--resolve-only`
shows only an override. A run with fewer repeats than that, or with `limit`, is a smoke (section 1):
report its score as a smoke, never as the benchmark's score. A score on part of the tasks needs a named,
versioned subset and is reported under its name and version; with no such subset, report no benchmark
score. Never change the benchmark, the judge, or
its token budgets and timeouts, which are the evaluation's identity. The model section is yours
([add-model-eval-config](references/add-model-eval-config.md)). A slow deployment produces timeouts and
truncated answers that count as model failures: a policy-model response timeout is the model's unless
the call sets `--policy-timeout-counts-as-model false` (for a self-hosted evaluation that owns its
serving; it moves the timeout to the evaluation side and changes nothing else). Keep the default whenever
two checkpoints, or a checkpoint and its quantized export, are compared: the deployment is part of what is
measured.

- The verdict is the inspection's output line. The persisted health record is `_ng_run_health_metrics` in
  the first entry of `artifacts/rollouts_aggregate_metrics.json`; `quality_summary.json` is the per-rollout
  summary (`gym eval health-check <copy>/artifacts` re-runs and rewrites it: run it on a copy).
- Acceptance needs complete results, the score gate (`--score-key` with `--score-agent`, and
  `--score-environment` when an agent has several entries), a readable health summary and evaluation
  health. Evaluation health is `PASS` when evaluation-side trial failures (judge, verifier, sandbox and
  harness faults, a server that did not start, transport faults, model-server 5xx) are strictly below 2%
  of the expected trials and every trial's grading is traceable; `FAIL` at or above 2%; `UNVERIFIED`
  below 2% when some grading is not traceable (the count is a lower bound). The 2% is fixed in Gym; no
  flag changes it. The classes are in `nemo_gym/core/failure_kinds.py`; every `provider_*` kind, any 5xx
  and another model's 4xx are evaluation-side. A verifier-timeout row can count as measured and healthy in
  `rollouts_coverage.json` and `quality_summary.json` while the recovery evidence marks it masked: trust the
  recovery evidence.
- A trial counts as observed when its server recorded the grading operation or saved a grading artifact
  bound to the answer, or the model's own failed answer settles it. A verifier that calls a judge without
  recording it is unobserved: its run reads `UNVERIFIED` with `trial_counts.unobserved_reason` naming the
  server (which benchmarks: [references/benchmark-notes.md](references/benchmark-notes.md)). That is not repairable by a resubmit or a flag; it needs a stack that records grading. Report it
  with the reason and do not present the run as verified.
- MLflow delivery is reported separately and never fails the gate (`gate_task.mlflow_delivery`).
- `SUCCESS` needs complete results and every gate. A NOOP, a finished job or a full score is not acceptance.

## 7. What the stack does not decide for you

A Gym run reports what it recorded. These outcomes are scored or retried in a way you must correct or
report yourself.

- **A sandbox killed for memory.** A kill during the agent's own command is the model's outcome, not a
  fault to rerun. Where the stack retries it (harness-agent rows, kind `provider_oom_killed`, the provider
  text in `failure_reason`) or scores it 0 (Terminal-Bench 4.0: read `sandbox_resource_limit`, resource
  `memory`, with the provider text in `detail`), count it as the model's. In Terminal-Bench 4.0,
  a process killed inside a live sandbox (exit 137) is a scored 0 marked interrupted. SWE-family and
  SkillsBench keep nothing typed about it.
- **A sandbox that ended for its own limit.** Terminal-Bench 4.0 scores 0 only when the provider text
  names the sandbox's own memory or storage limit; read `sandbox_resource_limit` before you accept that
  zero. Other sandbox deaths there are `infrastructure_error` and are retried. On SkillsBench's OpenHands
  path (`skillsbench/harbor*`) a sandbox that dies mid-trial is `harbor_failed` and is retried, not scored;
  it scores 0 only when it ended at the agent deadline with no test result. On the `harbor` environment
  server a sandbox lost before the reward is read is an infrastructure failure (`mask_sample`), not a score.
- **SkillsBench agent timeouts.** On the OpenHands path the row is final and scored. Many are zeros
  without a verifier result (the sandbox was discarded before verification, or the verifier produced
  nothing). Report the score over the rows the verifier graded beside the headline score. On the `harbor`
  environment server a harness that returns at the task's `[agent].timeout_sec` is verified and scored as
  usual, and the timeout signal is the harness's own: mini-SWE records
  `response.metadata.miniswe_termination = "timeout"`. Such a row carries no failure kind, neither
  `agent_timeout` nor `agent_limit` (the iteration cap), so don't count timeouts by kind or from reward 0. An
  episode that overruns the task's whole budget fails with `harbor:task_deadline` and is retried.
- **AutomationBench rows that ended on a serving timeout or a 5xx.** They are graded on the partial world.
  Count them from `rollout_error` before you publish a score; its `kind_candidate` is a candidate
  (`http_5xx`, `http_429`, `timeout`, `connection`, among others), and the verifier's own time limit leaves no
  `rollout_error`.
- **SciCode.** A killed grading subprocess shows as a wrong answer unless the step exit is recorded. Gym
  records it: look for `killed:<SIGNAL>` in `step_exits[].exit` (`killed:SIGKILL` for a memory kill). It is
  recorded for the run that decided the step, so a kill in another grading interpreter of the same row is
  not shown.

Name each correction in your report with its count.

## 8. Fresh run, and stopping a chain

A segment that ended before the task inventory was written has nothing to resume: recovery answers
`fresh-restart` and names the command, which restarts in the same directory with fresh budgets (saved
files kept, plan journaled):

```bash
gym eval resubmit "$GYM_JOB_ID" <benchmark> --recover \
  --max-walltime HH:MM:SS --walltime-reason "<why the longer walltime>"
```

`--max-walltime` must be longer than the recorded walltime. To evaluate something different, submit a new
run to a new location: a fresh `gym eval run` at an existing output path replaces it.

To stop a chain: `scancel` the running job, its successor (in `.gym_followup_queued.<job id>`) and the
sentinels (`.gym_followup_sentinel.<job id>`, and the one its predecessor queued for it, which becomes
eligible when you cancel the job; `squeue --me -o '%i %j %T %E'` shows it by its dependency). A successor
left to start sees `CANCELLED`, writes the stop to `.gym_chain_incomplete` and exits (it ends `FAILED`,
exit 1, by design: that is the stop, not a fault), after it got an
allocation; cancelling the successor and the sentinels saves that allocation and leaves no stop record
(the inspection reads the state from the scheduler either way). Then inspect (section 3) before deciding
anything else, and keep the run directory.

A segment whose driver exits 76 (nothing is eligible for another attempt) and left no failure kind the
next segment's guard handles stops the chain itself: it writes that stop to `.gym_chain_incomplete`, cancels the successor and
sentinel it queued, and journals the end in `chain-end-journal.jsonl`, so no allocation starts just to
write the stop. A typed stop (credentials rejected, resume refused, spin-up timeout, install failed) is
still written by the next segment's guard, with its own remedy.

## Exit checks

- [ ] `--inspect` shows `SUCCESS`, or the remaining STOP is explained and accepted by the operator.
- [ ] Every operator action (walltime, retries, repin, unsafe resume) has a reason in its journal or, for
  `--extend-retries` and `--extend-attempts`, in your report.
- [ ] `trial_counts` and `quality_summary.json` were read: unobserved and unhealthy counts are explained.
- [ ] Any `resumed across operational drift` settings are listed with the result.
- [ ] Each correction of section 7 that applies is in the report with its count.
- [ ] A score reported as the benchmark's comes from all its tasks at no fewer than its `num_repeats`
  (`manifest.yaml`); a smoke or subset score is labelled as one.
- [ ] The run is final on its receipts: `rollouts_coverage.json` says complete, the export receipt
  `rollouts_mlflow_export_done` is current and `gate_task.mlflow_delivery.state` is `verified` (or
  `not_required`, with `mlflow_tracking_required: false` set on purpose), and `.gym_chain_incomplete` is
  absent or explained.
