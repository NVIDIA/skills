# Bundled Scripts and Report Hook

## Using Bundled Scripts

Set `PYTHON=$(bash scripts/deft_python.sh)` before every bundled-script
command. The launcher selects
an explicit `DEFT_PYTHON` candidate when supplied, otherwise probes workspace
and preinstalled interpreters. It never invokes an installer, so it works
across isolated Claude, Codex, and OpenCode shell calls.
Resolve every path argument to an absolute host path before calling.

Commit state changes only through `commit_stage.py`. Never write
`deft_state.json` through inline Python, heredocs, an editor, echo, or jq.

## Available Scripts

| Script | Purpose | Arguments |
|---|---|---|
| `scripts/deft_python.sh` | Select an already-provisioned host Python with all core DEFT imports and execute it. Checks `DEFT_PYTHON`, workspace-local venvs, and preinstalled candidates; never installs packages. With no arguments, prints the selected absolute path. | `[PYTHON_ARG ...]`; env: `DEFT_PYTHON`, `WORKSPACE`, or `WORKSPACE_DIR` |
| `scripts/deft_context.py` | Re-read durable state and print the current iteration, immutable network policy, selected Python, and deterministic `next_stage`; optionally reject a stage mismatch. Run before every stage and finalization. | `--state PATH [--stage STAGE]` |
| `scripts/deft_exec.py` | Execute a command under `execution_policy`; in air-gap mode reject package/network/registry commands, inject offline variables, and enforce no-pull for direct containers. | `--state PATH -- COMMAND...` |
| `scripts/resolve_versions_key.py` | Resolve a dotted image key from the installed skill bank's `versions.yaml`; discovers the bank from `--skill-bank`, `TAO_SKILL_BANK_PATH`, or the script's ancestors. | `KEY [--skill-bank PATH]` |
| `scripts/metric_contract.py` | Shared stdlib parser/comparator for primary metrics, direction-aware best selection, constraints, and bundled-evaluator normalization. Import from sibling scripts; no CLI. | — |
| `scripts/record_metric_result.py` | Internal evaluate helper: validate standard evaluator JSON, enforce a configured artifact path, recompute `passed`, and write evaluate evidence. `commit_stage.py` calls it transactionally. | Internal; standalone CLI retained for compatibility |
| `scripts/commit_stage.py` | The only supported normal stage writer. Validate required evidence and append state atomically. Skip branches require zero-row routing proof and record `status=skipped` plus `skip_reason`; mining count must match the real parquet and logs cannot be placeholders. Terminal commits require an explicit reason and final handoff artifacts. | `--results-dir PATH --iter-label STR --stage STAGE --summary STR --duration-sec INT [stage artifact flags] [--status ok|error] [--skip]` (positive for executed stages; non-negative for skips) |
| `scripts/render_report.py` | Deterministically render the NVIDIA-styled, self-contained `DEFT_Loop_Report.html` from `deft_state.json` and recorded artifacts. Read the source template fresh, escape file-derived text, embed optional thumbnails, validate every placeholder/section, and replace atomically. Called automatically after initialization and every successful stage commit. | `--results-dir PATH [--require-terminal]` |
| `scripts/align_token_usage.py` | Backfill per-stage LLM token usage into `deft_state.json.events` by parsing the Claude Code transcript JSONL. Run after the loop (or any time). Adds a `tokens` field per event and refreshes `context_tokens`. | `--state-path PATH [--cwd PATH \| --project-dir PATH \| --transcript PATH ...] [--dry-run]` |
| `scripts/analyze_kpi.py` | Bundled threshold-sweep evaluator: emit standard `metric_result.json` plus diagnostic CSV/plots. Other customer metrics use the adapter contract in `references/metric-contract.md`. | `csv_path` (positional) `[--output-dir PATH]` `[--label-column NAME=label]` `[--score-column NAME=siamese_score]` `[--pass-label NAME=PASS]` `[--bins INT=40]` |
| `scripts/validate_training_csv.py` | Validate an assembled ChangeNet training CSV before launching training. Checks required columns, every reconstructed `input_path` / `golden_path`, filename-shape mistakes, duplicates, and optional validation leakage; `--report-json` writes the required merge-validation proof. Stdlib only — no pandas required. | `--csv PATH --workspace-root PATH [--validation-csv PATH] [--report-json PATH] [--light NAME] [--image-ext EXT]` |
| `scripts/init_deft_state.py` | Write fresh version-4 state, including immutable execution policy and resolved mining image/CSV paths. Atomic; never use on resume. | Required workflow args plus `--network-mode airgap|network-enabled --network-mode-source STR --python-executable PATH [--mining-images-root PATH] [--resolved-mining-pool-csv PATH]` |
| `scripts/resolve_mining_pool.py` | Resolve every source CSV row and golden pair against the separately staged image root, write a canonical source `filepath`, and fail on missing or ambiguous files. | `--csv PATH --images-root PATH --output-csv PATH [--summary PATH] [--light NAME] [--image-ext EXT]` |
| `scripts/changenet_data_pair_prepare.py` | Build the ChangeNet `(input, golden, label, object_name)` CSV from `_ng/` + `_ok/` image directories. NV_PCB_Siamese mode (`--images-dir`) emits the 14-column siamese CSV and copies images into the staged tree. | `--input-dir PATH --golden-dir PATH` `[--output PATH=dataset.csv]` `[--label STR]` `[--images-dir PATH]` `[--subdir NAME=sdg]` `[--light NAME=SolderLight]` `[--image-ext EXT=.jpg]` |
| `scripts/prepare_inference_spec.py` | Write `best_model.json` + `best_model_inference_spec.yaml` from `deft_state.json` + the training spec. Run once at loop end. See `references/prepare-for-inference.md`. | `--results-dir PATH` |
| `scripts/finalize_run.py` | Generate the ChangeNet handoff first, validate the stop reason against final metric evidence, commit `loop_stop`, and record final artifacts. | `--results-dir PATH --iter-label LABEL --stop-reason metric_met|max_iterations --duration-sec INT` |
| `scripts/stage_backbone.py` | Stage the ChangeNet pretrained backbone locally (download from HF, copy into the workspace). Idempotent; reuses an existing staged file. Hard-fails (non-zero exit) if it cannot produce a staged file. Prints the staged absolute path as the last stdout line. | `(--workspace PATH \| --dest PATH) [--repo-id STR=nvidia/C-RADIOv2-B] [--filename STR=model.safetensors] [--stage-name STR=c_radio_v2_b.safetensors] [--force]` |

## Script Invocation

Use one portable form in every harness:

```bash
PYTHON=$(bash <skill_root>/scripts/deft_python.sh)
"$PYTHON" \
  <skill_root>/scripts/commit_stage.py \
  --results-dir /abs/path/results/run_YYYYMMDD_HHMMSS \
  --iter-label iter1 \
  --stage anomalygen \
  --anomalygen-sdg /abs/path/iter1/anomalygen/sdg/SDG_result.csv \
  --anomalygen-allocation /abs/path/iter1/anomalygen/sdg/allocation.json \
  --duration-sec "${STAGE_DURATION_SEC}" \
  --summary "generated 10 pairs"
```

Before an external/container command, validate the stage and execute under the
persisted policy:

```bash
PYTHON=$(bash <skill_root>/scripts/deft_python.sh)
"$PYTHON" <skill_root>/scripts/deft_context.py \
  --state "${RESULTS_DIR}/deft_state.json" --stage data_mining
"$PYTHON" <skill_root>/scripts/deft_exec.py \
  --state "${RESULTS_DIR}/deft_state.json" -- \
  docker run --user "$(id -u):$(id -g)" \
    -e USER="$(id -un)" -e LOGNAME="$(id -un)" -e HOME=/tmp ...
```

Set `STAGE_DURATION_SEC` from measured wall-clock evidence before committing:
use the selected backend's elapsed time for submitted jobs, or time an inline
host stage directly. Round a measured sub-second stage up to `1`; never invent
a duration. `commit_stage.py` rejects an omitted, zero, or negative value so the
report can always calculate per-stage and total runtime. Stage references list
the required artifact flags.

### Aligning Per-Stage Token Usage (Post-Loop)

`commit_stage.py` cannot measure LLM token usage at write time. Run `align_token_usage.py` after the loop (or on demand) to backfill real per-stage numbers from the Claude Code transcript JSONL:

```bash
PYTHON=$(bash <skill_root>/scripts/deft_python.sh)
"$PYTHON" <skill_root>/scripts/align_token_usage.py \
  --state-path /abs/path/results/deft_state.json \
  --cwd /abs/path/to/project-root
```

The script reads `~/.claude/projects/<slug>/*.jsonl` (slug derived from `--cwd`), attributes each assistant message's `usage` to the stage whose `(prev.ts, this.ts]` window contains it, and rewrites `deft_state.json` atomically with a per-event `tokens` field plus a refreshed `context_tokens`. The `tokens` field exposes `input`, `output`, `cache_read`, `cache_create` (and its `5m`/`1h` breakdown), `context_size_end`, and the list of `models` seen. Pass `--transcript PATH` (repeatable) or `--project-dir PATH` to override the auto-discovered location; `--dry-run` inspects output without rewriting state.

## Automatic report hook

`init_deft_state.py` writes the first report immediately after canonical state
is initialized. `commit_stage.py` then calls `render_report.py` after every
valid state commit, including failed-stage and `loop_stop` commits. The
hook is deliberately outside the state transaction: a presentation error is
printed as `report hook failed` but never rolls back a valid GPU-stage result.

Normally do not invoke the renderer separately. If a hook reports an error,
repair the named presentation input and run:

```bash
PYTHON=$(bash <skill_root>/scripts/deft_python.sh)
"$PYTHON" <skill_root>/scripts/render_report.py \
  --results-dir "${RESULTS_DIR}"
```

At loop end, add `--require-terminal` when verifying or manually rebuilding
the final artifact. `agents/reporter.md` remains only as a compatibility
wrapper for runtimes that explicitly invoke that legacy agent; it calls the
same script and contains no rendering logic.

## Stage Reference Modules

Each pipeline stage maps to one underlying skill in the bank. The matching
`references/*.md` file layers DEFT-loop conventions (mounts, output dirs,
`deft_state.json` updates, `commit_stage.py` arguments) on top of the
skill's generic instructions. **Read only the current stage's relevant
section, then invoke the skill via the Skill tool or the direct-container
fallback below. Never preload or `cat` all references/underlying skills.** If
a reference file is missing, stop and ask the user to reinstall the plugin.

| Stage(s) | Reference file | Underlying skill | Owns |
|---|---|---|---|
| `train`, `evaluate` | `references/visual-changenet.md` | `tao-skill-bank:tao-train-visual-changenet` | TAO training, inference, evaluation, checkpoint discovery, TAO spec edits, two-checkpoint compare, `${TAO_PYT_IMAGE}` (pinned in Pre-Flight step 5) invocation. |
| `anomalygen` | `references/tao-generate-anomalies.md` | `tao-skill-bank:tao-generate-anomalies` | AMP / AnomalyGen synthetic defect generation, `defect_spec.jsonl` routing, testcase prep, allocation recovery, and SDG output schema. |
| `rca` (VCN Classify) | `references/tao-analyze-gaps-visual-changenet.md` | `tao-skill-bank:tao-analyze-gaps-visual-changenet` | Threshold sweep, per-label weakness ranking, per-lighting expansion, `kpi_gaps.parquet` schema, and `deft_state.json` output for VCN Classify models. |
| `routing` | `references/tao-route-visual-changenet-samples.md` | `tao-skill-bank:tao-route-visual-changenet-samples` | VCN weak-sample routing to mining and/or AnomalyGen, `mining_gaps.parquet` + `anomalygen_gaps.parquet` outputs, dropped-label warnings. |
| `data_mining` (VCN path) | `references/tao-mine-aoi-images.md` | `tao-skill-bank:tao-mine-aoi-images` | Embed-then-mine workflow: target embedding, source-pool embedding, k-NN nearest-neighbour mining, `mined.parquet` output schema, encoder consistency requirement. |

### Direct-container fallback

Use this path only when the mapped Skill tool is unavailable and Docker, this
skill source tree, the mapped underlying skill source, and the current stage's
reference module are all present. Record `execution_path=direct-container` in
the transcript. Read only the current stage overlay and mapped underlying
skill, then execute their documented `docker run` or wrapper commands with the
same approved arguments and absolute output paths. Do not infer a CLI, module,
or reduced workflow from an error message; if the two documents do not provide
a complete direct invocation for the stage, hard-stop with the missing command.
In air-gap mode, apply `references/air-gap.md` and use only images already
returned by `docker image inspect`.

The fallback changes only the invocation mechanism. It must produce the same
required artifacts and commit them through `"$PYTHON" scripts/commit_stage.py`. Never
hand-edit state to reconcile an incorrectly mounted or incomplete run.

### Invariants

**Path rule.** Record absolute host paths under `${RESULTS_DIR}` in state. For
ChangeNet direct containers, use `-v "$WORKSPACE:/data/workspace"` and
`-v "$RESULTS_DIR:/results"`; specs write baseline stages below
`/results/baseline/<stage>` and iter N below `/results/iterN/<stage>`. Mining
and AnomalyGen retain their stage reference's same-path workspace mounts.
Never mount `${RESULTS_DIR}` at `/results/iterN`: that shifts ChangeNet outputs
into run-level `train/` and `inference/` directories and invalidates proof.

## Workflow-level Pitfall — AutoML policy in the spec

The only loop-owned trap: writing `automl_policy: off` (or any `workflow:`
block) into `baseline_spec.yaml` makes TAO fail at config-merge time with
`Error merging 'baseline_spec.yaml' with schema: Key 'workflow' not in
'ExperimentConfig'`. `automl_policy` is a workflow argument, not a TAO spec
field. For direct `docker run visual_changenet train -e <spec>` (the inline
path this workflow uses), the plain-training entrypoint is the default and
no policy override is needed — just don't add the key. Full discussion in
`## Train AutoML Policy` in SKILL.md.

Stage-specific pitfalls (RCA `--user` / `-e <spec>`, AnomalyGen `chmod` /
`HF_HUB_DOWNLOAD_TIMEOUT`, mining-pool `golden_path` rewrite, etc.) belong in
the underlying skill's own `Common pitfalls` section — see each entry in
`## Stage Reference Modules` and read the matching `skills/data/<name>/SKILL.md`.
