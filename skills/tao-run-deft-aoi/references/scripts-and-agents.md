# Bundled Scripts and Agents

## Using Bundled Scripts

Run bundled scripts from `scripts/` via `run_script()` when the harness provides it (it is a Claude Code plugin runtime helper, not a function defined in this repo); otherwise fall back to direct `python` invocation. Resolve every path argument to an absolute host path before calling. For invocation examples, see `references/SCRIPT_USAGE.md`.

Never write `loop_log.jsonl` via `echo` or inline `jq` — the `seq` invariant requires reading the live tail through `next_seq()`.

## Available Scripts

| Script | Purpose | Arguments |
|---|---|---|
| `scripts/log_stage.py` | Append a stage event to `results/loop_log.jsonl` (computes `seq` from disk; guarantees valid JSON). `--context-tokens` is an optional placeholder; real values come from `align_token_usage.py`. | `--log-path PATH --iter-label STR --stage {evaluate,rca,anomalygen,data_mining,train,loop_stop} --status {ok,error} --summary STR --duration-sec INT [--context-tokens INT]` |
| `scripts/align_token_usage.py` | Backfill per-stage LLM token usage into `results/loop_log.jsonl` by parsing the Claude Code transcript JSONL. Run after the loop (or any time). Adds a `tokens` field per entry and refreshes `context_tokens`. | `--log-path PATH [--cwd PATH \| --project-dir PATH \| --transcript PATH ...] [--dry-run]` |
| `scripts/analyze_kpi.py` | Compute FAR / threshold sweep on a ChangeNet inference CSV and pick the FAR @ 100%-recall operating point. | `csv_path` (positional) `[--output-dir PATH]` `[--label-column NAME=label]` `[--score-column NAME=siamese_score]` `[--pass-label NAME=PASS]` `[--bins INT=40]` |
| `scripts/validate_training_csv.py` | Validate an assembled ChangeNet training CSV before launching training. Checks required columns and that every `input_path` / `golden_path` exists on disk. Stdlib only — no pandas required. | `--csv PATH --workspace-root PATH` |
| `scripts/init_deft_state.py` | Write a fresh `${RESULTS_DIR}/deft_state.json` from CLI args. Guarantees unique top-level keys. Atomic write; refuses to overwrite without `--force`. Use only on fresh runs; never on resume. | `--results-dir PATH --workspace PATH --kpi-target STR --max-iterations INT --num-gpus INT --num-epochs INT --num-sdg INT --project STR --step INT [--batch-size INT] [--top-k-per-target INT] [--knn-metric STR] [--min-similarity FLOAT] [--train-container STR] [--ag-container STR] [--force]` |
| `scripts/changenet_data_pair_prepare.py` | Build the ChangeNet `(input, golden, label, object_name)` CSV from `_ng/` + `_ok/` image directories. NV_PCB_Siamese mode (`--images-dir`) emits the 14-column siamese CSV and copies images into the staged tree. | `--input-dir PATH --golden-dir PATH` `[--output PATH=dataset.csv]` `[--label STR]` `[--images-dir PATH]` `[--subdir NAME=sdg]` `[--light NAME=SolderLight]` `[--image-ext EXT=.jpg]` |
| `scripts/prepare_inference_spec.py` | Write `best_model.json` + `best_model_inference_spec.yaml` from `deft_state.json` + the training spec. Run once at loop end. See `references/prepare-for-inference.md`. | `--results-dir PATH` |
| `scripts/stage_backbone.py` | Stage the ChangeNet pretrained backbone locally (download from HF, copy into the workspace). Idempotent; reuses an existing staged file. Hard-fails (non-zero exit) if it cannot produce a staged file. Prints the staged absolute path as the last stdout line. | `(--workspace PATH \| --dest PATH) [--repo-id STR=nvidia/C-RADIOv2-B] [--filename STR=model.safetensors] [--stage-name STR=c_radio_v2_b.safetensors] [--force]` |

## Agents

| Agent | Purpose | Invoke when |
|---|---|---|
| `agents/reporter.md` | Render `results/DEFT_Loop_Report.html` from disk state (`deft_state.json` + `loop_log.jsonl` + iter summaries + RCA artifacts) following `references/REPORT_RENDERING.md`. Atomic write; verifies all placeholders filled. | After each iteration completes (with `trigger="after-iteration"`) and once more at loop end (with `trigger="loop-end"`). Note: a per-stage trigger existed in earlier revisions and is no longer recommended — the spawn cost dominated for short stages. |

Spawn via the Task tool. Pass paths only, never values — the agent reads disk as the single source of truth:

```
Task(
  description="Render DEFT report",
  subagent_type="general-purpose",
  prompt=(
    f"Read {skill_root}/agents/reporter.md and follow its instructions exactly.\n"
    f"Inputs:\n"
    f"  results_dir = {RESULTS_DIR}\n"
    f"  skill_root  = {skill_root}\n"
    f"  trigger     = after-stage   # or 'loop-end' at the very end\n"
  ),
)
```

The agent prints one status line and exits. Never render `DEFT_Loop_Report.html` inline in the parent — the whole point of this agent is to keep rendering alive when the parent's context is saturated.

## Stage Reference Modules

Each pipeline stage maps to one underlying skill in the bank. The matching
`references/*.md` file layers DEFT-loop conventions (mounts, output dirs,
`deft_state.json` updates, `log_stage.py` summary string) on top of the
skill's generic instructions. **Read the reference file first, then invoke
the skill via the Skill tool.** If a reference file is missing, stop and
ask the user to reinstall the plugin.

| Stage(s) | Reference file | Underlying skill | Owns |
|---|---|---|---|
| `train`, `evaluate` | `references/visual-changenet.md` | `tao-skill-bank:tao-train-visual-changenet` | TAO training, inference, evaluation, checkpoint discovery, TAO spec edits, two-checkpoint compare, `${TAO_PYT_IMAGE}` (resolved from `tao_toolkit.pyt` in `versions.yaml`) invocation. |
| `anomalygen` | `references/paidf-anomalygen.md` | `tao-skill-bank:paidf-anomalygen` | AMP / AnomalyGen synthetic defect generation, `defect_spec.jsonl` routing, testcase prep, allocation recovery, and SDG output schema. |
| `rca` (VCN Classify) | `references/tao-analyze-gaps-visual-changenet.md` | `tao-skill-bank:tao-analyze-gaps-visual-changenet` | Threshold sweep, per-label weakness ranking, per-lighting expansion, `kpi_gaps.parquet` schema, and `deft_state.json` output for VCN Classify models. |
| `routing` | `references/tao-route-visual-changenet-samples.md` | `tao-skill-bank:tao-route-visual-changenet-samples` | VCN weak-sample routing to mining and/or AnomalyGen, `mining_gaps.parquet` + `anomalygen_gaps.parquet` outputs, dropped-label warnings. |
| `data_mining` (VCN path) | `references/tao-mine-aoi-images.md` | `tao-skill-bank:tao-mine-aoi-images` | Embed-then-mine workflow: target embedding, source-pool embedding, k-NN nearest-neighbour mining, `mined.parquet` output schema, encoder consistency requirement. |

### Invariants

**Path rule.** Use absolute host paths under `${RESULTS_DIR}/iter${ITER}/` for every stage's output, mount `<workspace>` into the container at the same path, pre-create dirs world-writable, and reject any config containing `output: /results/...` or any path outside `<workspace>`.

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
