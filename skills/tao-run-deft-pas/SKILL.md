---
name: tao-run-deft-pas
description: >
  Run iterative improvement for NVIDIA TAO CLIP / SigLIP image-text retrieval
  on attribute-labelled data. Use when a request combines retrieval evaluation,
  weak-attribute or caption-pair mining, repeated retraining, and a stopping
  condition based on a retrieval KPI, validation plateau, or iteration budget;
  the customer need not know the DEFT or People Attribute Search (PAS)
  names. The self-contained workflow performs dataset preparation, zero-shot
  evaluation, attribute gap analysis, caption-space k-NN mining, history-aware
  selection, retraining, and re-evaluation. Treat
  `tao-deft-pas` as shorthand for this canonical `tao-run-deft-pas` workflow.
  Do not use for standalone CLIP training, one-off evaluation or embedding,
  generic k-NN mining, or AOI/ChangeNet DEFT workflows.
license: Apache-2.0 AND CC-BY-4.0
compatibility: Requires one supported TAO execution platform (Docker, SLURM, Kubernetes, Brev, or virtualenv), accessible NVIDIA GPUs, the two PAS dataset export archives, and Python 3.9+ for control; virtualenv execution additionally requires the documented CPython 3.12 pyt and ds profiles.
metadata:
  tags:
  - application
  - workflow
  - deft
  - pas
  - clip
  - retrieval
  - loop
  author: NVIDIA Corporation
  version: "0.4.0"
allowed-tools: Read Bash Write

---

# PAS DEFT Workflow

> **Standalone install?** If this session was not initialized by the TAO skill
> bank plugin, run the `tao-setup` skill first for host preflight, credential
> checks, and cross-skill discovery.

Run the canonical PAS flow as one resumable, disk-backed workflow. All PAS
workflow logic, templates, and host adapters ship with this skill; customers do
not need a separate source checkout. The bundled scripts make stage calls
deterministic without adding another orchestration layer.

This skill supports every packaged TAO execution platform: Docker, SLURM,
Kubernetes, Brev, and virtualenv. The workflow produces a platform-neutral,
schema-validated action bundle; the selected platform skill owns native
`submit`/`status`/`logs`/`cancel` and the job-record.
Virtualenv execution uses separate immutable `pyt` and `ds` runtime profiles;
the workspace control `.venv` is not an execution runtime.

## Entry Contract

`tao-deft-pas` and `tao-run-deft-pas` select this same workflow. If the user
did not choose a platform, ask once among Docker, SLURM, Kubernetes, Brev, and
virtualenv; never default to Docker. On resume, the immutable platform in
`deft_state.json` is already the selection and must not be changed.

Use two intake phases:

1. Perform only bounded, lightweight path discovery. Two explicit archive file
   paths win and suppress all archive searching. An explicit archive directory
   is treated as one approved search root. Otherwise, resolve the conventional
   `~/workspace` and use only these deduplicated search roots: `~/pas`, the
   workspace root, and its `pas/`, `input/`, and `inputs/` children. Check the
   workspace root itself only; beneath each other search root, inspect the root
   and directories at most two levels below it. This AOI-style bounded-subtree
   lookup supports nested export/drop directories without turning into a home
   or repository scan. Never follow symlinks or add the current checkout,
   source repositories, tutorial/notebook trees, or workspace dataset-output
   trees as implicit search roots.

   An archive candidate contains direct regular-file children
   `images_raw.tar` and `meta.tar.gz`; the workflow needs one such pair, not
   multiple dataset directories. Follow the classification and provenance
   format in `references/preflight.md` so the user sees every search root, its
   reason and depth bound, the directory in which a pair was found, and whether
   it is archive-only or mixed with extracted data.

   Enumerate run-state files only at
   `<workspace>/results/run_*/deft_state.json`. Read the minimal identity fields
   and present a resume candidate only when `workflow` is exactly
   `tao-run-deft-pas`; never offer AOI or unidentified DEFT state as PAS.
   Do not validate large archives to EOF or inspect platforms, images, GPUs, or
   credentials yet.
2. If `max_iterations` or a time budget is absent, ask one consolidated
   question for that required value and any genuinely ambiguous path/run
   choice. State the documented defaults and that a complete read-only
   preflight plus approval summary follows. Do not ask whether the user wants
   optional KPI, authentication, or parameter overrides; apply their defaults
   unless the prompt already supplies an override.

After required intake is resolved, discover and validate:

- workspace and either a new `${RESULTS_DIR}` or one existing run to resume;
- `images_raw.tar` and `meta.tar.gz`; `SHA256SUMS` is optional;
- `max_iterations`, or a user-supplied time budget from which an iteration
  limit can be estimated;
- metric name, query type, operator, and optional target;
- whether the deployment requires authenticated Hugging Face model access;
- selected TAO execution platform and its platform-specific prerequisites;
- any explicit epoch, GPU, mining, continual-learning, or visualization
  overrides.

For a new run, `RESULTS_DIR` must be a child of `WORKSPACE`. `DATASET_ROOT`
must be below a workspace data directory (for example
`$WORKSPACE/data/pas_v31_tao_ft`), not directly below the workspace; neither
path may contain the other. All approved paths are absolute and non-symlink.

Never replace an explicit value with a heuristic. Defaults apply only to
unspecified values and must be identified as defaults in the pre-flight
summary. `max_iterations` has no default. An absent metric target means an
ungated run that stops after `max_iterations`. Hugging Face token forwarding
defaults to disabled because the bundled model is public.

The authoritative parameter contract is the nested dataclass schema in
`scripts/pas_deft/config.py`, adapted from the PAS reference notebook. Read
that schema when a request needs the meaning, default, numeric bounds, or valid
options of a DEFT parameter. Do not infer an undocumented field or bypass its
metadata constraint. Config preparation, initialization, audit, and every
host-side stage validate the materialized bundle through that same schema;
the `pas` YAML section is validated through the typed PAS configuration model.

If required information remains missing after full discovery, ask one
consolidated follow-up. A normal invocation should need no knowledge of stage
modules, container mounts, state files, or bundled-runtime function
signatures.

## Safety Gate

Perform only read-only discovery before approval: resolve paths, inspect file
metadata and archives, check process-environment variable presence, inspect
local images, inspect GPUs, and audit an existing run. Credentials come only
from the launching process environment; never open or source a credential file,
and never print, grep, copy, or echo a credential value. If a required variable
is absent, tell the user which name
to export in the shell that launches the agent; never ask for its value in
chat. Do not inspect credential-file metadata when no credential is needed. If
the user explicitly asks for a file-permission check, `stat` only that named
file and warn about group/other readability.

Show the summary defined in `references/preflight.md`, including every
parameter and source, planned file creation/extraction, image pulls, estimated
runtime, and resume status. Wait for explicit approval before registry login or
pulls, platform submit, package installation, archive extraction, config/state
creation, or any write under the workspace.

If an approved parameter later changes, show the changed summary rows and get
approval again before continuing. No confirmation is needed between unchanged,
already-approved stages.

## Execution Contract

1. After approval, follow `references/preflight.md` to prepare the runtime,
   materialize the immutable run config, and initialize state once. Never
   reinitialize an existing run.
2. Before every stage, after context compaction, and before any completion
   claim, run:

   ```bash
   "$SKILL_ROOT/scripts/deft_python.sh" --workspace "$WORKSPACE" --runtime \
     "$SKILL_ROOT/scripts/audit_deft_run.py" --results-dir "$RESULTS_DIR"
   ```

   Continue only when the audit reports `IN_PROGRESS` and its `next_action`
   matches the intended stage. For `INVALID`, stop launching work and report
   the listed inconsistency. For nonterminal `FAILED` whose `next_action` is
   `loop_stop`, follow the pipeline reference and commit `hard_stop`; for
   terminal `FAILED`, report the failure and launch no more work. For
   `COMPLETE`, do not rerun a stage.
3. Read only the current stage reference named by `read_before_action`. Use
   `run_pas_stage.py` for bundled PAS host stages. For every TAO action, use
   `run_deft_action.py prepare`, reconcile any interrupted launch, bind the
   exact request-owned job-record before native submit, dispatch the emitted
   bundle through the selected platform's four verbs, synchronize remote
   outputs, capture native logs, then use `run_deft_action.py finalize`. Follow
   `references/platform-execution.md`; never assemble an untracked launch.
4. A command succeeds only when its exit status is zero and its documented
   output checks pass. Capture verbose output at the action-owned log path;
   inspect the final error block or at most the last 40 lines.
5. Commit each successful or terminally failed stage exactly once with
   `commit_stage.py`. It validates artifact structure, freshness, iteration
   scope, command evidence, leakage, and transition order before applying a
   recoverable, journaled update to `deft_state.json` and `loop_log.jsonl`.
   Never call
   `log_stage.py` or `record_metric_result.py` directly, and never hand-edit
   canonical state, log, metric, or history files.
6. Render the HTML report with `render_deft_report.py`. Reporting is a
   deterministic read of audited state; it does not require another agent.
7. Keep the driving turn attached through the complete bounded loop. After a
   stage reaches terminal status, finalize/commit it, re-audit, and immediately
   continue with the audit-selected next action. An in-progress update is
   commentary, not a final response and not a detach point.

All recorded artifact paths are absolute host paths under `${RESULTS_DIR}`.
Baseline artifacts live under `zs/`, iteration N under `iter_N/`, and run-wide
splits, source embeddings, and history files at the run root.

## Workflow

The normal path is linear, with only two meaningful decisions:

```text
pre-flight approval
  -> dataset_setup -> pool_embed -> baseline evaluate
  -> KPI met? yes: loop_stop
              no: baseline gap_analysis
  -> for N = 1..max_iterations:
       data_mining -> history_select -> visualize -> train -> evaluate
       -> KPI met? yes: loop_stop
                   no and N < max: gap_analysis -> next iteration
                   no and N = max: loop_stop
  -> final audit -> HTML report
```

Stage ownership and exact commands are in
`references/scripts-and-agents.md`. Read the detailed reference only when its
stage is next:

| Stage | Reference | Required result |
|---|---|---|
| dataset setup | `references/data-layout.md` | verified rebuilt dataset, transparent layout report, five split files, non-empty source-pool parquet |
| pool embedding and mining | `references/mining.md` | fresh command evidence plus non-empty, schema-checked parquet outputs |
| evaluate and train | `references/clip-train-eval.md`, `references/metric-contract.md` | successful TAO status, bound metric evidence; for train, a fresh best and normalized checkpoint |
| gap analysis | `references/gap-analysis.md` | non-empty iteration-scoped gaps parquet |
| history selection | `references/mining.md` | budgeted mined set, cumulative/history entry, zero eval leakage |
| visualization | `references/visualization.md` | enabled artifacts and command evidence, or a config-authorized skip |

`visualize` is the only optional stage. Commit it with `--skip` only when both
visualization settings are false in the approved config. Do not turn off a
failing visualization mid-run without revising and reapproving the config.

## Loop and Recovery Rules

- The loop is bounded by `max_iterations`; never create an iteration outside
  that range. Once the KPI passes, the only legal next transition is
  `loop_stop`. Do not mine or train again.
- Monitoring defaults to attached (`long_running_enabled=true`, five-minute
  updates). Use terminal-condition polling: retain the selected platform's job
  id and poll its native status no more often than every 30 seconds, continuing
  through finalize, commit, audit, and the next stage.
  The bounded workflow's terminal audit status is the end condition, so this is
  not open-ended polling.
- Never send a final response while an approved run is nonterminal. A final
  response ends chat-side execution and nothing can wake it automatically.
  Finalize the turn only after the audit reports `COMPLETE`, terminal `FAILED`,
  or `INVALID`, or when the user explicitly asks to stop or detach. If the
  runtime genuinely cannot keep a turn alive, say so before launch and provide
  the exact durable resume audit command; do not claim unattended monitoring.
- Never repeat an unchanged failed command speculatively. Classify the failure,
  inspect its final log block, make one evidence-based correction permitted by
  the stage reference, then retry once. Stable command status records persist
  attempt 1/2, and the wrappers refuse a third attempt after context loss. The
  audit must pass before advancing.
- A transient registry/network interruption or GPU contention may be retried
  once after evidence shows the condition changed. A second equivalent failure
  is terminal for the run.
- Never retry checksum or rebuild verification failure, zero-row mining,
  schema/cardinality failure, missing eval-split evidence, eval leakage,
  history conflict that fails documented resume, metric-contract mismatch,
  stale/cross-iteration output, or an unsupported GPU architecture. Commit an
  error when state permits and stop with the exact recovery action.
- If a command wrote outputs but crashed before stage commit, run the audit.
  Reuse outputs only when their command status is successful, fresh, correctly
  scoped, and all stage validators pass. History selection has one supported
  recovery: rerun the deterministic adapter with `--resume`; never delete or
  edit a history entry by hand.
- Manual modification or truncation of `deft_state.json` or `loop_log.jsonl`
  invalidates the run. Preserve it for diagnosis and start a new results
  directory.

## Metric and Stop Semantics

The approved metric contract is immutable for the run. Evaluation must parse
the exact iteration's `nvidia_pas_metrics_aggregate.csv`; the result records
its source path and is re-derived during commit and audit. Checkpoint ranking
and best-run reporting follow the approved operator (`>=`/`>` chooses the
higher value, `<=`/`<` the lower), not a hard-coded metric convention.

Successful completion is exactly one of:

- `loop_stop(reason=kpi_met)` after a bound evaluation satisfies its target;
- `loop_stop(reason=max_iterations)` after the final allowed evaluation when
  the target is unmet or absent.

`hard_stop` is a failed terminal outcome, not successful completion.

## Completion

At a normal loop boundary:

1. commit `loop_stop` with the audit-selected reason;
2. render `${RESULTS_DIR}/DEFT_Loop_Report.html` with trigger `loop-end`;
3. require this command to exit zero:

   ```bash
   "$SKILL_ROOT/scripts/deft_python.sh" --workspace "$WORKSPACE" --runtime \
     "$SKILL_ROOT/scripts/audit_deft_run.py" \
       --results-dir "$RESULTS_DIR" --require-complete
   ```

Report the stop reason, baseline and best metric, best iteration/checkpoint,
completed iteration count, report path, and any warnings. A checkpoint, CSV,
HTML file, or assistant statement alone is not completion evidence.

## Progressive References

| Need | Read |
|---|---|
| read-only checks, approval summary, initialization | `references/preflight.md` |
| stage commands and script interfaces | `references/scripts-and-agents.md` |
| platform staging, four verbs, job records, finalization | `references/platform-execution.md` |
| state transitions and resume behavior | `references/pipeline-and-state.md` |
| dataset/archive contract | `references/data-layout.md` |
| KPI parsing and evidence | `references/metric-contract.md` |
| gap generation | `references/gap-analysis.md` |
| embeddings, k-NN, selection | `references/mining.md` |
| contact sheets and t-SNE | `references/visualization.md` |
| train/evaluate/checkpoints | `references/clip-train-eval.md` |
