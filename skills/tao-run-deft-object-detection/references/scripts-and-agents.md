# Bundled Scripts and Agents

## Using Bundled Scripts

Run every bundled script through `scripts/deft_python.sh`. Resolve every path argument to an absolute host path before calling. Commit state and log changes only through `commit_stage.py` — never write `deft_state.json` or `loop_log.jsonl` with inline Python, heredocs, an editor, or jq.

## State scripts

| Script | Purpose | Arguments |
|---|---|---|
| `deft_python.sh` | Select an already-provisioned host Python with the required imports and execute it. Never installs packages. | `[PYTHON_ARG ...]`; env `DEFT_PYTHON`, `WORKSPACE` |
| `init_deft_state.py` | Write a fresh `deft_state.json`. Atomic; refuses to overwrite without `--force`. Fresh runs only. | `--results-dir --workspace --max-iterations ...` |
| `commit_stage.py` | The only supported state writer. Validates the ordered transition, updates state, appends one log event, audits, rolls back on failure. | `--results-dir --iter-label --stage --summary [artifact flags] [--status ok\|error]` |
| `audit_deft_run.py` | Read-only cross-check of state, log, and artifacts. Prints the safe next action and `read_before_action`. | `--results-dir [--require-terminal] [--require-complete]` |

## Pipeline glue scripts

These replace internal container images from the reference pipeline whose scripts are not published. All are pure host Python.

| Script | Stage | Purpose |
|---|---|---|
| `fetch_gdino_checkpoint.py` | Pre-Flight | Resolve the Grounding DINO zero-shot checkpoint, downloading `nvidia/tao/grounding_dino:...trainable_v1.1` from NGC when the user supplied none. `--plan` resolves the path and reports whether a download is needed without performing one, for use before the approval gate. Idempotent; prints the path on stdout. Verified bit-identical to the hand-staged copy it replaces. |
| `emit_default_spec.py` | any stage | Emit a stage's starting spec from TAO: `default_specs` for annotations/analytics, the Hydra schema dump for grounding_dino/codetr, a shipped asset for gap_analysis/tmm/embedding (TAO emits none). Reports mandatory fields grouped by block. |
| `apply_spec_overrides.py` | any stage | Apply the stage's `--apply-workflow-defaults` (settings this workflow requires that differ from TAO's defaults, from `assets/overlays/`) then `--set` (only what varies per run), by dotted key and YAML-typed. A `--set` colliding with a workflow-default key is an error unless `--allow-workflow-default-override`. Refuses unknown keys unless `--allow-new`; `--require-no-mandatory` blocks launching against any remaining `???`, and `--require-no-mandatory-under BLOCK` scopes that to the blocks a stage actually reads — a `default_specs` dump marks fields mandatory for every action it supports, so the unscoped check cannot pass for a single-action run; `--report-json` records every key and its source. |

### Stage overlays

`assets/overlays/<stage>.yaml` holds the settings a stage needs but that do not
vary per run. A field nobody mentions keeps whatever `default_specs` or the Hydra
schema emitted, and TAO's default is not always the value the stage wants:
`kpi.ignore_sqwidth` is 0 by default against the reference pipeline's 40, and the
only symptom of the difference is that a different set of boxes gets scored. Each
overlay line carries TAO's default in a comment, so the cost of removing it is
visible without diffing against a container.

| overlay | stage | notably pins |
|---|---|---|
| `kitti_to_coco.yaml` | `annotations convert` KITTI→COCO | `kitti.project: coco` — names the output file every later step looks for |
| `coco_to_odvg.yaml` | `annotations convert` COCO→ODVG | formats only |
| `codetr_inference.yaml` | pool pseudo-labelling | `num_select: 1000` (TAO 300 crowds rare classes out of the pool), `conf_threshold: 0.3`, and the ViT-L/16 architecture the checkpoint requires |
| `grounding_dino_inference.yaml` | baseline + per-iteration inference | `conf_threshold: 0.0` (keep the full PR curve), `log_scale: auto`, `class_embed_bias: true` |
| `kpi_analyze.yaml` | scoring | `num_recall_points: 11`, `ignore_sqwidth: 40` |

`grounding_dino train` has no overlay: it is built from the full
`assets/train_grounding_dino.yaml` template rather than emitted-then-overridden,
so its values are already in a file under version control.

Two things deliberately stay out of the overlays. Paths, checkpoints and GPU
counts vary per run. And `dataset.infer_data_sources.captions` is the label map —
a detection takes the class of the caption token it matched, by position — so it
must be derived from the run's classes, never pinned.
| `prepare_mapping_for_kpi_analyze.py` | `kpi_analyze` | Narrow the supplied KPI class mapping to the run's target classes, aliases verbatim. A class the model cannot predict would otherwise score a constant 0 and compress the mAP trend. |
| `prepare_input_for_image_embeddings.py` | `prep` | List the pool image directory into the `filepath` parquet `embedding image_embeddings` reads. Absolute paths, symlinks resolved, sorted — so the same directory always yields the same parquet. |
| `prepare_val_split_for_train.py` | `prep` | Carve a validation COCO from 10% of the prepared pool, rewriting category ids to **0-based**. `grounding_dino train` cannot run without a validation source, and its loader uses `category_id` verbatim as a dense label index, so a conventional 1-based COCO overflows on the last class. |
| `summarize_kpi.py` | after `kpi_analyze` | Recompute the aggregate mAP from `kpi_calc.csv` and write `kpi_summary.json` beside it. The stage prints mAP to stdout and writes it nowhere, so this removes the need to hold a stream open for the length of the stage. |
| `await_stage.py` | any long stage | Block until a stage finishes by watching its artifacts or `status.json`. **Never wait on a process name.** Pass `--newer-than <marker touched before launch>` so a retry is not satisfied by the previous attempt's leftovers. |
| `prepare_class_mappings_for_mining_data_prep.py` | `prep` | Translate one `classes.yaml` into the two mappings TAO folds with: the `category_mapping` block for the Co-DETR inference spec (the real fold, applied at detection time with per-category soft-NMS) and the identity `kitti.mapping` for `annotations convert`. Emits nothing else — TAO does the folding. |
| `verify_class_contract.py` | Pre-Flight, before every inference | Compare the class list as it appears in captions, the KPI mapping, `deft_state.json`, the ODVG labelmap and `classes.yaml`. Grounding DINO labels a detection by caption *position*, so a reordered or short list relabels every prediction while the run still exits 0. |
| `verify_pseudo_labels.py` | `prep` | With `--pool-images-dir` alone, refuse a pool holding duplicate basenames before labelling: labels are one flat file per basename, so same-named images share one set. With `--labels-dir`, fail when Co-DETR wrote label files carrying no boxes. `codetr inference` exits 0 and prints PASS when the spec's architecture does not match the checkpoint, emptying every file; this is the gate that names that cause. Directory walk only, so it runs before the conversion it protects. |
| `validate_pool_coco.py` | `prep` | Verify the converted pool: every target class carries annotations, case-only class mismatches are named, unmapped source classes are reported with counts, and image/annotation counts reconcile. Both TAO consumers drop unmatched names silently, so this is where a broken fold surfaces. |
| `prepare_budget_for_mining.py` | before `mine` | `desired_unique_count` = weak-image count × multiplier, with optional floor/ceiling. Point `--weak-parquet` at **iteration 1's** parquet on every iteration to hold the budget constant. Writes only the number to stdout. |
| `stage_mined_odvg.py` | `stage` | Copy mined images, look up ODVG records by basename, renumber `image_id`, remap labels, write `tmm_odvg.jsonl` + `labelmap.json`. **Truncates** the JSONL, so re-running is idempotent. |
| `validate_odvg_images.py` | `stage` | Hard-fail when an ODVG record references a missing image, when there are no usable records, or when records are duplicated. `--prune` deletes orphan images. Stdlib only. |
| `prepare_exclude_for_mining.py` | `stage` | Merge this iteration's mined set with the previous cumulative and de-duplicate. `--parquet-b` is optional at iteration 1 only; pass `--iteration N` so a missing previous cumulative is an error after that, rather than silently re-mining trained images. |
| `prepare_spec_for_train.py` | `train` | Copy the previous spec, append one `{image_dir, json_file, label_map}` entry to `dataset.train_data_sources`, set `train.num_epochs` and `train.optim.lr`. Lowers `checkpoint_interval` / `validation_interval` when they exceed the epoch count, and will not double-add a source already present. |

### Script invocation

```bash
<skill_root>/scripts/deft_python.sh \
  <skill_root>/scripts/commit_stage.py \
  --results-dir /abs/path/results/run_YYYYMMDD_HHMMSS \
  --iter-label iter1 \
  --stage mine \
  --mining-output /abs/path/iter1/mining/final_unique_files.parquet \
  --mining-summary /abs/path/iter1/mining/summary.json \
  --summary "mined 500 unique images"
```

If no reliable start time was captured, omit `--duration-sec`; it records `0`. Do not invent a duration.

## Agents

| Agent | Purpose | Invoke when |
|---|---|---|
| `agents/reporter.md` | Render `results/DEFT_Loop_Report.md` from disk state. Atomic write; no HTML template. | After each completed iteration (`trigger="after-iteration"`) and at loop end (`trigger="loop-end"`). |

Spawn via the Task tool, passing paths only — the agent reads disk as the single source of truth:

```
Task(
  description="Render DEFT OD report",
  subagent_type="general-purpose",
  prompt=(
    f"Read {skill_root}/agents/reporter.md and follow its instructions exactly.\n"
    f"Inputs:\n"
    f"  results_dir = {RESULTS_DIR}\n"
    f"  skill_root  = {skill_root}\n"
    f"  trigger     = after-iteration\n"
  ),
)
```

Never render the report inline in the parent — the agent exists so an end-of-loop render survives a saturated parent context.

## Stage Reference Modules

| Stage | Overlay | Underlying skill |
|---|---|---|
| `prep` (once) | `references/prep-source-pool.md` | `tao-skill-bank:tao-train-codetr` + `tao-generate-image-embeddings` (+ bundled glue) |
| `gap_analysis` | `references/tao-analyze-gaps-od-map.md` | `tao-skill-bank:tao-analyze-gaps-od-map` |
| `embed` | `references/tao-generate-image-embeddings.md` | `tao-skill-bank:tao-generate-image-embeddings` |
| `mine` | `references/tao-mine-od-images.md` | `tao-skill-bank:tao-mine-od-images` |
| `stage` | `references/stage-mined-data.md` | *(bundled glue)* |
| `train`, `inference` | `references/grounding-dino.md` | `tao-skill-bank:tao-train-grounding-dino` |
| `kpi_analyze` | `references/tao-analyze-detection-kpi.md` | `tao-skill-bank:tao-analyze-detection-kpi` |

**Read only the current stage's overlay.** If one is missing, stop and ask the user to reinstall the plugin — do not substitute generic shell commands.

### Direct-container fallback

Use only when the mapped Skill tool is unavailable and Docker plus the current overlay are present. Record `execution_path=direct-container` in the transcript, then run the overlay's documented `docker run` with the same arguments and absolute output paths. The fallback changes the invocation mechanism only: it must produce the same artifacts and commit them through `commit_stage.py`.

## Invariants

**Path rule.** Record absolute host paths under `${RESULTS_DIR}`. Mount `"$WORKSPACE:$WORKSPACE"` so host and container paths are identical.

**`results_dir` appends the task name.** TAO's `update_results_dir` turns `results_dir=X` into `X/train/` or `X/inference/`. Never append the subdirectory yourself.

**Format spelling differs by stage.** `gap_analysis object_detection` takes lowercase `kitti`/`coco`; `analytics kpi_analyze` takes uppercase `KITTI`/`COCO`. Both are correct for their own stage.

**Never put `automl_policy` or a `workflow:` key in a TAO spec.** TAO's Hydra schema rejects them at config-merge time. Plain `docker run … train` is already non-AutoML.
