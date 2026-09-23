# DEFT OD — Gap Analysis Stage Overlay

Layers loop conventions on top of `tao-skill-bank:tao-analyze-gaps-od-map`. Read that skill's `SKILL.md` for the full field reference.

## When to invoke

Iteration stage 1 — the first stage of every iteration. It analyzes the **previous** phase's inference labels: iteration 1 reads `baseline`, iteration N reads `iter{N-1}`. Gap analysis never runs at baseline.

## Inputs from state

| State field | Description |
|---|---|
| `iterations.<prev_phase>.inference_labels_dir` | `${RESULTS_DIR}/<prev_phase>/inference/labels` — committed by that phase's `inference` stage. Read it from state; never hardcode. |
| `config.ground_truth_labels_dir` | KPI ground-truth KITTI label directory. |
| `config.kpi_images_dir` | KPI image root — establishes the full image universe, including images with no annotations. |
| `config.ap50_thresholds` | Per-class AP50 thresholds, e.g. `{"car": 0.99, "bicycle": 0.7, "person": 0.7}`. |

## Threshold mapping — set `default_ap50_threshold: 0.0`

This is the one setting that silently changes which images are selected.

The reference pipeline's filter had **no fallback**: a class absent from the thresholds JSON got no threshold at all and could never mark an image weak. TAO DS instead falls back to `default_ap50_threshold`, which **defaults to `0.5`**. Leaving it unset means every class you did not list starts marking images weak at 0.5, quietly inflating the gap set and the mining budget derived from it.

To reproduce the reference behavior exactly:

1. Set `default_ap50_threshold: 0.0`. TAO DS guards the AP50 gate with `a_thresh > 0`, so a zero threshold disables it for unlisted classes.
2. Set `default_recall_threshold: 0.0` and `default_precision_threshold: 0.0` — the reference selected on AP50 only.
3. List **every** class you want gated explicitly in `weak_thresholds`.

Everything else already matches: an image is weak if it fails on **any** class (union, whole image selected), comparison is strict `<`, class-name matching is case-sensitive, and an AP50 of NaN — meaning no ground truth for that class in that image — passes rather than fails.

## Spec

Write per-iteration under `${RESULTS_DIR}/iter${N}/gaps/od_gap_spec.yaml` — the invocation below reads it from
`$OD_GAP_SPEC`, so bind the two:

```bash
OD_GAP_SPEC="${RESULTS_DIR}/iter${N}/gaps/od_gap_spec.yaml"
```

Build it the same way every other stage does — emit, then fill the run-specific fields:

```bash
<skill_root>/scripts/deft_python.sh <skill_root>/scripts/emit_default_spec.py \
  --stage gap_analysis --out "$OD_GAP_SPEC"

<skill_root>/scripts/deft_python.sh <skill_root>/scripts/apply_spec_overrides.py \
  --spec "$OD_GAP_SPEC" \
  --set ground_truth_ann_path="$GROUND_TRUTH_LABELS_DIR" \
  --set inference_ann_path="<previous phase's inference/labels>" \
  --set images_dir="$KPI_IMAGES_DIR" \
  --set results_dir="${RESULTS_DIR}/iter${N}/gaps" \
  --set kpi="iter${N}" \
  --set-from-file weak_thresholds="<the threshold mapping from the section above>"
```

`gap_analysis` is absent from TAO's `default_specs` list, so unlike the other stages
there is no container to emit from: `emit_default_spec.py` copies
`assets/gap_analysis_object_detection.yaml`, which is hand-maintained for exactly that
reason. That asset is a **starting spec, not an overlay** — do not pass it to
`--apply-workflow-defaults`. It carries a nested `weak_thresholds` mapping, and the
overlay mechanism takes one dotted key per leaf, so it is rejected with
`'weak_thresholds' has a mapping value`. There is no `--apply-workflow-defaults` step
for this stage; the emitted spec already holds every documented default.


```yaml
ground_truth_ann_path: <config.ground_truth_labels_dir>
inference_ann_path:    <iterations.<prev_phase>.inference_labels_dir>
images_dir:            <config.kpi_images_dir>
results_dir:           <absolute path to ${RESULTS_DIR}/iter${N}/gaps>
kpi:                   iter${N}
input_format:          kitti          # LOWERCASE — see below
iou_threshold:         0.5
conf_threshold:        0.0            # inference already filtered; see below
min_area:              0
weak_thresholds:
  car:      {ap50: 0.99}
  bicycle:  {ap50: 0.7}
  person:   {ap50: 0.7}
default_recall_threshold:    0.0
default_precision_threshold: 0.0
default_ap50_threshold:      0.0
class_mapping: {}
```

**`input_format` is lowercase here** (`kitti` / `coco`), unlike `analytics kpi_analyze` in the same container which requires uppercase. Both spellings in this loop are correct for their own stage.

**Leave `conf_threshold: 0.0`.** TAO inference already dropped detections below its own `inference.conf_threshold` when it wrote the labels. A second threshold here composes with that one rather than replacing it, so a non-zero value filters twice.

## Invocation

```bash
<skill_root>/scripts/deft_python.sh <skill_bank>/skills/data/tao-analyze-gaps-od-map/scripts/verify_object_detection_spec.py \
  --spec "$OD_GAP_SPEC"

docker run --rm --name "deft_iter${N}_gap" --gpus all --shm-size=8g --user "$(id -u):$(id -g)" $DOCKER_IDENTITY \
  -v "$WORKSPACE:$WORKSPACE" $EXTRA_MOUNTS -w "$WORKSPACE" \
  "$TAO_DS_IMAGE" \
  gap_analysis object_detection -e "$OD_GAP_SPEC"
```

## Outputs

| Artifact | Path |
|---|---|
| `weak_images.parquet` | `${RESULTS_DIR}/iter${N}/gaps/weak_images.parquet` |
| `box_gaps.parquet` | `${RESULTS_DIR}/iter${N}/gaps/box_gaps.parquet` |
| `image_metrics.parquet` | `${RESULTS_DIR}/iter${N}/gaps/image_metrics.parquet` |
| `gap_report.json` | `${RESULTS_DIR}/iter${N}/gaps/gap_report.json` |

All four are always written, even when no gaps are found.

`weak_images.parquet` is the artifact the loop consumes. It carries `filepath`, so it feeds the `embed` stage directly with no projection step. Columns: `kpi`, `image_id`, `filepath`, `weak_classes`, `weak_recall`, `weak_precision`, `weak_ap50`.

## Zero weak images ends the loop

If `weak_images.parquet` has zero rows, every class met its configured AP50 threshold on every image. There is nothing to embed, mine, or add. Commit `gap_analysis/status=ok` **with `--weak-image-count 0`**, then commit `loop_stop` and run the loop-end sequence — do not run the remaining six stages against an empty set.

`--weak-image-count 0` is the only machine-readable proof of this stop. It is what makes `audit_deft_run.py` answer `next_action=loop_stop` instead of `embed`, and it is what makes the loop-end `--require-complete` gate pass. A `--summary` that says "0 weak images" is free text that nothing parses: without the flag a successful run is reported to the user as INCOMPLETE.

## Commit

Count the rows of `weak_images.parquet` and pass that number. It is required on every `ok` gap_analysis commit — the commit is rejected without it.

```bash
<skill_root>/scripts/deft_python.sh <skill_root>/scripts/commit_stage.py \
  --results-dir "${RESULTS_DIR}" --iter-label "iter${N}" --stage gap_analysis \
  --weak-images "${RESULTS_DIR}/iter${N}/gaps/weak_images.parquet" \
  --gap-report "${RESULTS_DIR}/iter${N}/gaps/gap_report.json" \
  --weak-image-count <row count of weak_images.parquet> \
  --duration-sec "$(( SECONDS - started ))" \
  --summary "gap_analysis: <N> weak images across <M> classes"
```

Use `--zero-weak-images` in place of `--weak-image-count 0` only when the row count itself could not be read; passing both a positive count and that flag is rejected as contradictory.
