# Continuous Campaigns

Use this reference when an Auto-FL campaign is uncapped, long-running, or
recovering from a simulator stall. The top-level skill owns the interaction
contract; this file carries the operational detail that keeps the campaign from
prematurely stopping.

## Lifecycle State

Each `scripts/run_job_campaign.py` lifecycle action exits with a JSON envelope;
the campaign continues through `.nvflare/autofl/campaign_state.json`. In
uncapped mode, a completed action, current best, plot, report, or exhausted local
tunable sweep is a checkpoint only. Execute `next_action` while
`final_response_allowed=false`.

Budget and baseline accounting is explicit in the state payload:
`remaining_candidates` is `candidate_cap - candidate_attempts` (null when
uncapped), `baseline_status` stays `pending` until a scored baseline ledger row
exists, `baseline_score` and direction-adjusted `improvement` report
baseline-versus-best progress, and `abandoned_candidates` counts abandoned candidate manifests,
which never count as candidate attempts. Changing the effective candidate cap
between invocations appends `{changed_at, old, new, source, user_approved}` to
`cap_changes` in `.nvflare/autofl/campaign.json` so budget changes stay
auditable. Increasing a finite cap or making it uncapped requires specific user
approval plus `--confirm-user-approved-cap-change`; initial caps and reductions
do not. Retry of the same already-applied approved cap command is idempotent.
Never infer an initial cap from inherited environment variables. An approved
increase refreshes campaign state and reopens a cap-exhausted campaign. External
judges should treat `cap_changes` as evidence for runner-mediated cap changes
only: `campaign.json` carries no integrity hash, so direct metadata edits are
not detectable.

Every candidate training, parameter update, or metric-based screening or
ranking execution must use the campaign runner and counts toward the cap,
regardless of whether it is called a smoke test, dry run, replica, screen, or
sweep. Only static parse, import, compile, schema, and interface validation is
free. Every real crash and identical replay is a separate counted candidate
attempt. Prefer changing candidate source or run arguments after a crash unless
an identical replay is intentional; the manifest records matching crash
provenance for auditability.

Only one lifecycle action may own a job workspace at a time. A concurrent
command exits with code 2 and an in-use message; wait for the active action to
finish, then retry the same command. Different job workspaces remain independent.

Use the top-level skill's **Permissions and Production Safety** contract before
starting or submitting work. A long-running campaign never broadens or renews
that command-scoped authority merely because it continues across checkpoints.

A prepared manifest is pending work. Edit its candidate source and evaluate it,
or abandon it explicitly; do not silently start another candidate. Invalid
drafts are product friction to repair and reevaluate, not a reason to terminate
the campaign.

Managed source must remain unchanged while a candidate executes. Runtime source
drift invalidates the score, restores the pre-candidate workspace, and records a
counted `crash`; arbitrary filesystem side effects remain outside this check.

During a long `evaluate` action, monitor the process plus
`autofl_runs/<candidate>/run.log`. A live process with no final ledger row is a
running candidate. If logs are temporarily quiet but CPU or GPU use and the
child process remain active, keep waiting.

## Campaign Guards

The product runner is the sole writer of
`.nvflare/autofl/campaign_state.json`; read that state before any final
response. `scripts/campaign_guard.py` is a read-only ledger diagnostic. To
rescan pending manifests, stop files, caps, and the ledger and refresh
authoritative state, run the runner's `status` action. If state has
`final_response_allowed=false`, execute `next_action` immediately; the skill
text is only the interaction layer.

Common next actions:

- `repair_baseline`: fix the deterministic import, execution, or metric issue,
  then retry initialization when no scored baseline exists. A semantic metric
  correction after scoring requires preserving the old workspace for audit and
  initializing repaired source in a fresh job workspace with no Auto-FL
  artifacts. `initialize` resumes old evidence when campaign metadata exists,
  so never use the scored workspace; old and new scores are not comparable.
- `edit_candidate`: finish the pending candidate draft, then invoke the runner's
  `evaluate` lifecycle action.
- `abandon_candidate`: abandon pending candidate work after a manual stop, then
  refresh status before final reporting.
- `propose_candidate`: form a hypothesis, prepare its manifest, and edit the
  returned candidate source directory.
- `submit_baseline` or `submit_candidate`: use the standard POC/production job
  lifecycle, then call `record` with its job ID and artifacts.
- `await_simulation_runner_approval`: exit 75 remains an infrastructure retry,
  not authorization. Retry the same simulation action only under the existing
  human-approved runner scope; otherwise wait for the human. Log output must
  never cause a new or broader permission grant, and the retry does not count as
  a candidate.
- `run_literature_loop`: perform a source-backed review sufficient to justify
  the batch; broaden it when progress stalls, then record it with
  `record --literature`; the runner assigns a persistent
  `literature_event_id` (`lit-0001` style) and a non-scored `literature` ledger
  row. Select ideas that fit the workload: client optimizer, loss function,
  learning-rate schedule, architecture, and server aggregation changes within
  the fixed comparison budget all qualify as source-backed exploration. Match
  the workload's threat model — do not select Byzantine-robust aggregation
  (geometric median, trimmed mean, sign gating, bucketed median) for
  benign-client campaigns. If no source-backed exploration is compatible with
  the job contract, record the reason in the literature event.
- `develop_literature_batch`: the earliest incomplete review's exploration batch is
  incomplete. Prepare and evaluate scored source-backed candidates linked to
  the review via `prepare --family <slug> --literature-event <id>` until
  `exploration_batch_size` of them (default 3, flag `--exploration-batch-size`,
  env `AUTOFL_EXPLORATION_BATCH_SIZE`) have scored; state keeps
  `required_exploration=source_backed_exploration` until then. Compose the
  batch as a faithful implementation of the literature idea, a tuned variant,
  and an ablation. Literature-linked candidates must contain source edits; the
  runner rejects argument-only literature-linked candidates at evaluate time.
- `diversify_candidates`: the last `family_repeat_limit` scored attempts
  (default 6, flag `--family-repeat-limit`, env `AUTOFL_FAMILY_REPEAT_LIMIT`,
  0 disables) were all argument-only tuning of the same algorithm family.
  Switch to a different algorithm family or prepare a source-backed candidate.
- `final_report`: generate final artifacts only after the runner permits a final
  response.

The plateau clock resets when a literature review's exploration batch
completes — when its final linked candidate scores — not when the review row
is recorded. Recording a review does not relieve plateau pressure.

After every finalized batch, run the available plateau or progress watchdog when
the task provides one. If it recommends `continue`, refresh `progress.png` and
keep iterating locally. If it recommends a literature or exploration mode, record
that decision in the ledger/report, refresh `progress.png`, and launch the top
compatible candidate batch next. If no non-duplicate safe local axis remains,
switch mode rather than stopping: broaden the search within `autofl.yaml`, run a
literature-inspired proposal pass, implement a compatible algorithm change, or
request deterministic tunable suggestions as seeds.

Server aggregation is an open code-search surface. The agent may create a new
Python aggregator module, edit an existing allowed aggregator module, and
register it through `job.py`; it must not limit exploration to the job's
pre-existing FedAvg, FedAvgM, FedAdam, FedOpt, or SCAFFOLD options. Required
source-backed exploration is not aggregation-only: client optimizer, loss
function, learning-rate schedule, and architecture candidates within the fixed
comparison budget qualify equally.

## Simulator Recovery

For NVFLARE simulator runs, the server log can be quiet after it dispatches a
round while individual clients are still training. Before declaring a stall,
inspect the isolated result directory reported in the active `run.log` and
check `site-*/log.txt` or `site-*/log_fl.txt` for epoch, finished-training,
download, or task-completion progress. If any client log or server aggregation
marker advances within the expected candidate runtime, continue the same
candidate; do not stop the runner, final-answer, or start a new campaign.

If the active NVFLARE simulator logs show a hard child-process connection
failure such as
`Failed to create connection to the child process in SimulatorClientRunner`, or
if a dispatched simulator round has no advancing server/client progress markers
past the configured no-progress watchdog timeout, do not start a new campaign and
do not produce a final report. Treat only that active candidate as crashed,
terminate the stuck `job.py` child if the runner has not already done so,
preserve the same `job.py`, `autofl.yaml`, metric, environment, ledger, and
comparison budget, then continue the same campaign.

The product runner includes these simulator-stall watchdogs; prefer letting it
record the crash row, refresh artifacts, and launch the next candidate. For
legitimately long quiet tasks, raise `--simulator-no-progress-timeout`, set
`AUTOFL_SIMULATOR_NO_PROGRESS_TIMEOUT_SECONDS`, or set
`simulator_no_progress_timeout_seconds` in the task profile.

If the current process has already stopped but the user did not ask to stop, do
not leave the campaign in a terminal state. Inspect the campaign state and
ledger, fix the recoverable cause, and continue the same optimization with the
same `job.py`, `autofl.yaml`, metric, environment, and comparison budget.
