# Pipeline, State, and Runtime Behavior

## Containers run as the calling user

Every TAO invocation in this skill passes `--user "$(id -u):$(id -g)"`.

Without it the containers write **root-owned** output, and the next host-side script cannot
read its own input — the loop stops between stages with a permission error that names a file
the previous stage just created successfully. Verified on both images: `annotations convert`
and `grounding_dino inference` both report `Execution status: PASS` under `--user` and leave
their output owned by the caller.

Do not substitute a `chown` fixup after each stage. It works, but it is a step every caller
has to remember and it leaves a window where the artifacts on disk are unreadable.

`--user` is not sufficient on its own. A uid with no `/etc/passwd` entry in the image has
no name, and the container then crashes at torch import with
`KeyError: 'getpwuid(): uid not found: <uid>'` before the action starts. Every launch
therefore carries `$DOCKER_IDENTITY` as well; `references/preflight.md` defines it.
Removing `--user` is not the way out of that — it reintroduces the root-owned output
above.

## Pipeline

All stages run inline in the parent context. For SKILL stages, read the matching `references/*.md` overlay first, then invoke the underlying `tao-skill-bank:*` skill. GLUE stages run a bundled script; there is no leaf skill.

### Prep (`prep`) — once, before baseline

Runs only when the source pool is not already labeled and embedded. Produces `source_pool/odvg/` and `source_pool/source_embeddings.parquet` — the two artifacts every iteration reads. Idempotent: each artifact is skipped when it already exists on disk.

Chain: Co-DETR pseudo-label the pool → fold predictions onto the user's target classes → KITTI→COCO → COCO→ODVG → embed the pool. Full detail, including the class-consistency contract and the hard-stop gates, is in `references/prep-source-pool.md`.

Cost is proportional to **pool size**, not to what mining later selects, and it must finish before the baseline starts. Report the pool image count in the Pre-Flight Summary.

### Baseline (`iter_0`) — no training

The loop never trains at baseline. It scores the checkpoint the user supplied:

1. **[SKILL — `tao-train-grounding-dino`] `inference`.** Run `grounding_dino inference` with `inference.checkpoint=<zero_shot_checkpoint>` and `results_dir=${RESULTS_DIR}/baseline`. TAO appends the task name, so labels land in `${RESULTS_DIR}/baseline/inference/labels/`. See `references/grounding-dino.md`.
2. **[SKILL — `tao-analyze-detection-kpi`] `kpi_analyze`.** Score those labels against the KPI ground truth. See `references/tao-analyze-detection-kpi.md`.

Also seed `${RESULTS_DIR}/train_grounding_dino.yaml` by copying the user's train-spec template. Iteration 1 extends that copy; it is never trained from at baseline.

### Iteration N — seven stages, in order

Each iteration's `gap_analysis` consumes the **previous** phase's inference labels: iteration 1 reads `baseline`, iteration N reads `iter{N-1}`.

1. **[SKILL — `tao-analyze-gaps-od-map`] `gap_analysis`.**
   Input: `state["iterations"][<prev_phase>]["inference_labels_dir"]` plus the KPI ground-truth label directory.
   Output: `weak_images.parquet` (with a `filepath` column), `box_gaps.parquet`, `image_metrics.parquet`, `gap_report.json`.
   Set `default_ap50_threshold: 0.0` and list every class explicitly in `weak_thresholds` — see the overlay for why. See `references/tao-analyze-gaps-od-map.md`.

   Every `ok` `gap_analysis` commit must carry `--weak-image-count <rows in weak_images.parquet>`; it is rejected without one.

   If `weak_images.parquet` has zero rows, no data can be mined this iteration. Commit `gap_analysis/status=ok --weak-image-count 0` and advance directly to `loop_stop` — there is nothing for the remaining six stages to consume. That zero is the only proof of the early stop the audit can read: without it, `--require-complete` reports a fully successful run as INCOMPLETE.

2. **[SKILL — `tao-generate-image-embeddings`] `embed`.**
   Embeds the weak images so they can be matched against the source pool. `input_parquet` is the `weak_images.parquet` from stage 1 (it already carries `filepath`, so no projection step is needed); `output_parquet` is `weak_images_embeddings.parquet`.
   The encoder **must** match the one used to build the source-pool parquet. See `references/tao-generate-image-embeddings.md`.

3. **[SKILL — `tao-mine-od-images`] `mine`.**
   `tmm unique_neighbor_matching` with `source_path=<source pool embeddings>`, `target_path=<weak_images_embeddings.parquet>`, and `desired_unique_count` from the mining budget.
   **`exclude_path` must be `null` on iteration 1.** TAO DS raises `FileNotFoundError` when `exclude_path` is set but is not a file; the cumulative parquet does not exist yet. From iteration 2 on, pass the previous iteration's `mined_cumulative.parquet`.
   Output: `final_unique_files.parquet`, `summary.json`. See `references/tao-mine-od-images.md`.

4. **[GLUE] `stage`.**
   Turn the mined filepath list into a trainable ODVG source, then extend the exclude set. Runs three bundled scripts in order:

   ```bash
   <skill_root>/scripts/deft_python.sh <skill_root>/scripts/stage_mined_odvg.py \
     --mined-parquet "${RESULTS_DIR}/iter${N}/mining/final_unique_files.parquet" \
     --annotations-base-dir "<source pool ODVG dir>" \
     --output-images-dir "${RESULTS_DIR}/iter${N}/tmm/images" \
     --output-annotations-dir "${RESULTS_DIR}/iter${N}/tmm/annotations" \
     --report-json "${RESULTS_DIR}/iter${N}/tmm/staging_report.json"

   <skill_root>/scripts/deft_python.sh <skill_root>/scripts/validate_odvg_images.py \
     --image-dir "${RESULTS_DIR}/iter${N}/tmm/images" \
     --odvg "${RESULTS_DIR}/iter${N}/tmm/annotations/tmm_odvg.jsonl" \
     --key-field file_name

   <skill_root>/scripts/deft_python.sh <skill_root>/scripts/prepare_exclude_for_mining.py \
     --parquet-a "${RESULTS_DIR}/iter${N}/mining/final_unique_files.parquet" \
     --parquet-b "<previous mined_cumulative.parquet; omit on iter 1>" \
     --iteration "${N}" \
     --output "${RESULTS_DIR}/iter${N}/mined_cumulative.parquet"
   ```

   Staging writes `tmm_odvg.jsonl` (one line per mined image, `image_id` renumbered sequentially, `instances[].label` remapped through the labelmap) and `labelmap.json`. `stage_mined_odvg.py` **truncates** `tmm_odvg.jsonl` before writing — the reference implementation opened it in append mode, so re-running an iteration silently duplicated every entry. See `references/stage-mined-data.md`.

5. **[SKILL — `tao-train-grounding-dino`] `train`.**
   First append the new ODVG source to the train spec:

   ```bash
   <skill_root>/scripts/deft_python.sh <skill_root>/scripts/prepare_spec_for_train.py \
     --previous-spec "<prev phase train spec>" \
     --output-spec "${RESULTS_DIR}/iter${N}/train_grounding_dino.yaml" \
     --num-epochs "<epochs>" --learning-rate "<lr>" \
     --tmm-image-dir "${RESULTS_DIR}/iter${N}/tmm/images" \
     --tmm-odvg-file "${RESULTS_DIR}/iter${N}/tmm/annotations/tmm_odvg.jsonl" \
     --tmm-label-map-file "${RESULTS_DIR}/iter${N}/tmm/annotations/labelmap.json" \
     --val-image-dir "<pool images>" \
     --val-json-file "${RESULTS_DIR}/val_coco.json" \
     --pretrained-model-path "<config.zero_shot_checkpoint>"
   ```

   `--pretrained-model-path` and the `--val-*` pair are not optional: without the
   checkpoint, training starts from no pretrained weights, reports success, and emits a
   model that detects nothing — visible only at KPI, a full training run later. The
   validation COCO comes from `prepare_val_split_for_train.py` and must be 0-based. See
   `references/grounding-dino.md`.

   This copies the previous spec and **appends** one `{image_dir, json_file, label_map}` entry to the `dataset.train_data_sources` list, then sets `train.num_epochs` and `train.optim.lr`. Growth is by list-append; earlier sources are never removed.

   `train.pretrained_model_path` is deliberately left untouched, so every iteration fine-tunes the **base** checkpoint on the accumulated dataset rather than continuing from `iter{N-1}`'s weights. See `references/grounding-dino.md`. Then run `grounding_dino train -e <spec> results_dir=${RESULTS_DIR}/iter${N} train.num_gpus=<N>`.

   Iteration N's committed checkpoint must be a newly emitted file under `${RESULTS_DIR}/iter${N}/train/`. A non-zero exit, or a run emitting no new checkpoint, is a hard stop — never evaluate a checkpoint written before the failure.

6. **[SKILL — `tao-train-grounding-dino`] `inference`.**
   Run `grounding_dino inference` with `inference.checkpoint=${RESULTS_DIR}/iter${N}/train/gdino_model_latest.pth` and `results_dir=${RESULTS_DIR}/iter${N}`. Labels land in `${RESULTS_DIR}/iter${N}/inference/labels/` and become the next iteration's `gap_analysis` input.

7. **[SKILL — `tao-analyze-detection-kpi`] `kpi_analyze`.**
   Score the new labels. Record `kpi_calc.csv` and the mAP parsed from stdout. mAP is **reported, not gated** — a regression does not stop the loop.

## State & Logging

Two artifacts persist loop state:

- `results/deft_state.json` — resume snapshot. Initialize once with `init_deft_state.py`, then mutate only through `commit_stage.py`.
- `results/loop_log.jsonl` — append-only event stream, one JSON line per stage:

```json
{
  "seq":            <int, monotonically increasing from 1>,
  "ts":             "<ISO-8601 UTC; stage end time>",
  "iter":           "baseline|iter1|iter2|...",
  "stage":          "prep|inference|kpi_analyze|gap_analysis|embed|mine|stage|train|loop_stop",
  "status":         "ok|error",
  "summary":        "<one-line outcome>",
  "duration_sec":   <int>,
  "context_tokens": <always 0; reserved, nothing populates it>,
  "tokens":         <object added at loop end>
}
```

**Disk is the source of truth.** Before every stage run the audit and use its `last_committed`, `next_action`, and `read_before_action` output. Do not print the full state or log into context.

**On startup / resume:** print the last 5 entries of `loop_log.jsonl`, then proceed from disk-loaded state.

### Ordered transitions the audit enforces

```
prep:      prep                                   (once; skipped when artifacts exist)
baseline:  inference -> kpi_analyze
iterN:     gap_analysis -> embed -> mine -> stage -> train -> inference -> kpi_analyze
```

`commit_stage.py` rejects an out-of-order commit. Do not repair a rejected commit by editing JSON.

## Stage Execution

Three stage types:

- **SKILL** — read the overlay, then invoke the matching `tao-skill-bank:*` skill.
- **GLUE** — parent runs a bundled script directly.
- **AGENT** — parent spawns a subagent. The only AGENT stage is `agents/reporter.md`.

### Post-stage check

After every stage, before advancing:

1. Verify the documented required artifacts exist.
2. Invoke `commit_stage.py` once with the documented artifact flags, and pass
   `--duration-sec` with the stage's measured wall clock. Take a timestamp before the
   stage and subtract:

   ```bash
   started=$SECONDS
   # ... run the stage ...
   commit_stage.py ... --duration-sec "$(( SECONDS - started ))"
   ```

   The field is in the log schema and the report renders a duration column from it, but
   nothing populates it unless the flag is passed — every stage then reads 0 and the
   timeline is empty. If no reliable start time was captured, omit the flag rather than
   inventing a number.
3. Run `audit_deft_run.py --results-dir ${RESULTS_DIR}`. If `INVALID`, halt and repair.
4. If the committed status is `error` — halt, surface the disk evidence, **do not auto-retry**.
5. If `ok` — print one status line: `[iter <N>/<max> · <stage>] <detail> · <duration> · next: <stage>`. Then advance. Render the HTML report only at iteration end and loop end.

## Reports

- `results/iter${N}_summary.md` — ≤300 words, readable after context compaction.
- `results/DEFT_Loop_Report.md` — re-rendered after each completed iteration and at loop end by the `reporter` subagent.

## Runtime Behavior

Run without pausing. Between stages, run the audit and print only its one-line next action. Spawn the `reporter` only after a full iteration and at loop end.

**Loop-end sequence** (in order):

1. Append the final event via `commit_stage.py --stage loop_stop`.
2. Spawn `reporter` with `trigger="loop-end"`.

Before telling the user the loop is complete:

```bash
<skill_root>/scripts/deft_python.sh <skill_root>/scripts/audit_deft_run.py \
  --results-dir ${RESULTS_DIR} --require-complete
```

**Stop conditions:**

- `max_iterations` reached → run the loop-end sequence and report the mAP trend across iterations.
- Zero weak images at any `gap_analysis` → commit `loop_stop`, report that the model met every configured per-class AP50 threshold.
- Unrecoverable gate failure → halt, report the exact missing artifact. Steps 1–3 of the loop-end sequence still apply.

## Output Layout

```
results/run_<YYYYMMDD_HHMMSS>/
├── deft_state.json
├── loop_log.jsonl
├── .deft_commit.lock                  # commit_stage.py: one writer at a time
├── .deft_commit.journal               # only while a commit is in flight; the
│                                      # next commit_stage.py run undoes an
│                                      # interrupted one from it
├── DEFT_Loop_Report.md
├── train_grounding_dino.yaml          # seeded at baseline from the user's template
├── prep/                              # only when prep ran
│   ├── inference/labels/              # Co-DETR pseudo-labels, already folded
│   ├── codetr_category_mapping.yaml   # the fold, applied at detection time
│   ├── kitti_mapping.yaml             # identity — Co-DETR already folded
│   ├── classmap_target.txt
│   └── pool_report.json               # validate_pool_coco.py's verdict
│                                      # NOTE: the KITTI->COCO output is NOT under prep/.
│                                      # It is retained at <workspace>/source_pool/coco.json
│                                      # as class_stratified mining's source_detection_file.
│                                      # See references/prep-source-pool.md step 3.
├── iter${N}_summary.md
├── baseline/
│   ├── inference/labels/*.txt         # zero-shot predictions (KITTI, 15 fields + score)
│   └── kpi/kpi_calc.csv
└── iter${N}/
    ├── gaps/                          # weak_images.parquet, box_gaps.parquet,
    │                                  # image_metrics.parquet, gap_report.json
    ├── embeddings/weak_images_embeddings.parquet
    ├── mining/                        # final_unique_files.parquet, summary.json
    ├── tmm/
    │   ├── images/                    # staged mined images
    │   └── annotations/               # tmm_odvg.jsonl, labelmap.json
    ├── mined_cumulative.parquet       # exclude set for the next iteration
    ├── train_grounding_dino.yaml      # prev spec + one appended ODVG source
    ├── train/                         # gdino_model_latest.pth, status.json
    ├── inference/labels/*.txt
    └── kpi/kpi_calc.csv
```
