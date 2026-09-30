# Pipeline, State, and Resume

The workflow is a linear transaction log with KPI/max-iteration branching. It
does not require a general graph engine, scheduler, or multiple agents.

## Contents

- [Legal transitions](#legal-transitions)
- [Disk contract](#disk-contract)
- [Stage transaction](#stage-transaction)
- [Resume](#resume)
- [Terminal paths](#terminal-paths)

## Legal transitions

```text
baseline/dataset_setup
  -> baseline/pool_embed
  -> baseline/evaluate
       -> baseline/loop_stop                  KPI passed
       -> baseline/gap_analysis               otherwise
  -> iter1/data_mining

iterN/data_mining
  -> iterN/history_select
  -> iterN/visualize                          real or config-approved skip
  -> iterN/train
  -> iterN/evaluate
       -> iterN/loop_stop                     KPI passed or N=max
       -> iterN/gap_analysis                  otherwise; feeds iter N+1
  -> iterN+1/data_mining

any committed error -> same-label loop_stop(reason=hard_stop)
```

Baseline does not consume an iteration. Gap analysis is owned by the
evaluation it analyzes and writes the next iteration's input. Never create a
terminal gap artifact after the last evaluation.

The transition is decided from committed metric evidence and immutable
`max_iterations`, not prose or an in-memory loop counter. `commit_stage.py`
rejects duplicate, skipped-ahead, post-KPI, and out-of-range transitions.

## Disk contract

`deft_state.json` is a schema-v3 resume snapshot. It records:

- immutable workspace, archives, intended dataset root, bundled-runtime hash,
  selected TAO platform, pinned images/runtime, config paths and hashes;
- immutable loop/config values and metric contract;
- current iteration, gate flag, per-label stage proofs, and stop reason.

`loop_log.jsonl` is the ordered event history. It contains one gap-free
sequence entry per committed stage with label, stage, `ok|skip|error`, summary,
duration, and timestamp. Both files are initialized/updated only through the
bundled state scripts.

Artifact files are additional evidence, not alternate state. The audit checks
their exact paths, structure, row counts, provenance, freshness, and
cross-file invariants. It also reopens metric CSVs and selection history.

Do not display the full state or log in conversation. The audit's compact
fields are sufficient:

```bash
"$SKILL_ROOT/scripts/deft_python.sh" --workspace "$WORKSPACE" --runtime \
  "$SKILL_ROOT/scripts/audit_deft_run.py" --results-dir "$RESULTS_DIR"
```

Interpret statuses as follows:

| Status | Action |
|---|---|
| `IN_PROGRESS` | perform exactly `next_action` after reading `read_before_action` |
| `COMPLETE` | do not launch more work; render/report after final verification |
| `FAILED` | if nonterminal, perform only the selected `loop_stop` hard-stop commit; if terminal, render/report and launch no work |
| `INVALID` | stop all mutations; preserve the run and report audit errors |

## Stage transaction

For each stage:

1. Audit and confirm the selected next action.
2. Read only its named reference.
3. Run the documented host adapters or platform actions with verbose output in logs.
4. Validate and commit once with the exact artifact and command-status flags.
5. Audit again. Advance only when the committed event is accepted.
6. Tell the user one line: `[label · stage] outcome · next: <action>`.
7. When the audit remains nonterminal, continue with step 1 for its selected
   next action in the same attached turn. Do not convert the progress line into
   a final response or wait for a human `continue` message.

The driver stays attached from the first approved launch through a terminal
audit by default. Backgrounding a long container changes only how its status is
polled; it does not detach the driving turn. If the user explicitly requests a
detach, report the exact `audit_deft_run.py` resume command and make no promise
that the workflow will advance without a new driver invocation.

If execution fails and the documented single correction also fails, record
the expected current stage as an error when canonical state is still valid:

```bash
"$SKILL_ROOT/scripts/deft_python.sh" --workspace "$WORKSPACE" --runtime \
  "$SKILL_ROOT/scripts/commit_stage.py" \
    --results-dir "$RESULTS_DIR" --iter-label "$LABEL" --stage "$STAGE" \
    --status error --summary "<specific failed command/check and log path>"
```

Then use the hard-stop terminal path below. Do not commit an error merely to
paper over an `INVALID` audit; invalid canonical state is preserved for
diagnosis.

## Resume

On startup or after context compaction:

1. Resolve the existing results directory explicitly.
2. Run the audit with the workspace argument on `deft_python.sh`.
3. Trust its transition calculation, not remembered progress or output
   existence.
4. Read one reference and continue the selected stage.

Uncommitted outputs can be reused only when that stage's validators accept a
successful matching command status and fresh scoped artifacts. When a
container wrapper was interrupted and its status remains `running`, rerun the
exact wrapper command after the deterministic container exits. The wrapper
first reconciles durable Docker/log success evidence and fresh outputs; only
when that fails does it consume the bounded retry and delete the named fresh
outputs. For history selection, use its one `--resume` path. Never backfill a
log entry or manually mark a stage complete.

If an existing run's config hashes, inputs, metric, or requested parameters no
longer match, do not mutate it. Preserve the directory and initialize a new,
separately approved run.

## Terminal paths

### KPI or iteration budget

Use the label of the evaluation that selected the stop:

```bash
"$SKILL_ROOT/scripts/deft_python.sh" --workspace "$WORKSPACE" --runtime \
  "$SKILL_ROOT/scripts/commit_stage.py" \
    --results-dir "$RESULTS_DIR" --iter-label "$LABEL" --stage loop_stop \
    --reason "$REASON" --summary "$SUMMARY"
```

`REASON` is `kpi_met` only when committed evidence passed, otherwise
`max_iterations` only after final iteration evaluation. The validator rejects
premature or false stop reasons.

Render and prove completion:

```bash
"$SKILL_ROOT/scripts/deft_python.sh" --workspace "$WORKSPACE" --runtime \
  "$SKILL_ROOT/scripts/render_deft_report.py" \
    --results-dir "$RESULTS_DIR" --trigger loop-end

"$SKILL_ROOT/scripts/deft_python.sh" --workspace "$WORKSPACE" --runtime \
  "$SKILL_ROOT/scripts/audit_deft_run.py" \
    --results-dir "$RESULTS_DIR" --require-complete
```

An optional progress render after a successfully committed iterN evaluation
uses `--trigger iteration-complete`. It does not affect state and cannot prove
completion.

### Hard stop

After a committed stage error, commit the only legal transition:

```bash
"$SKILL_ROOT/scripts/deft_python.sh" --workspace "$WORKSPACE" --runtime \
  "$SKILL_ROOT/scripts/commit_stage.py" \
    --results-dir "$RESULTS_DIR" --iter-label "$LABEL" --stage loop_stop \
    --reason hard_stop --summary "run stopped after $LABEL/$STAGE failure"

"$SKILL_ROOT/scripts/deft_python.sh" --workspace "$WORKSPACE" --runtime \
  "$SKILL_ROOT/scripts/render_deft_report.py" \
    --results-dir "$RESULTS_DIR" --trigger loop-end

"$SKILL_ROOT/scripts/deft_python.sh" --workspace "$WORKSPACE" --runtime \
  "$SKILL_ROOT/scripts/audit_deft_run.py" \
    --results-dir "$RESULTS_DIR" --require-terminal
```

Report it as `FAILED`, never completed. Include the failed stage, command log,
auditor recovery statement, and preserved results directory.

### User-facing result

For a successful terminal run, report the stop reason, contract and target,
baseline value, operator-directed best value/label, best raw checkpoint and
normalized state (when an iteration trained), completed iterations, warnings,
HTML report path, and passing `--require-complete` evidence. For failure,
replace metric claims with the last validated metric and failure evidence.
