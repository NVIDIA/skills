---
name: tao-analyze-gaps-od-map
description: >-
  Run TAO Data Services object-detection gap analysis from ground-truth and inference annotations.
  Use when an object detection workflow needs to identify weak images by comparing model predictions
  against ground truth using per-class recall, precision, and AP50 thresholds.
  Use when the user asks to "analyze OD gaps", "find weak OD images", or "run mAP gap analysis".
license: Apache-2.0
compatibility: Requires docker, nvidia-container-toolkit, and the TAO data-services container pinned in versions.yaml.
metadata:
  author: NVIDIA Corporation
  version: "0.1.0"
allowed-tools: Read Bash
tags:
- tao
- data
- gap-analysis
- object-detection
- mAP
- rcca
---

# TAO Analyze Gaps OD mAP

Use this skill to run TAO Data Services object-detection gap analysis. The skill compares ground-truth and inference annotations, computes per-image per-class TP/FP/FN/AP50 metrics, and identifies weak images where any class metric falls below its threshold. It does not run inference; upstream steps must produce the inference annotations first.

The container entrypoint is:

```bash
gap_analysis object_detection -e /absolute/path/to/object_detection.yaml
```

## Inputs

Required spec fields:

| Field | Meaning |
|---|---|
| `ground_truth_ann_path` | KITTI label directory or COCO `.json` with ground-truth boxes. |
| `inference_ann_path` | KITTI label directory or COCO `.json` with model predictions. |
| `images_dir` | Root image directory. Establishes the full image universe including unannotated images. |
| `results_dir` | Output directory for all artifacts. |
| `kpi` | Identifier tag written to every output row. |
| `input_format` | `kitti` or `coco`. Must be declared explicitly; never inferred from the path. |

Common optional fields:

| Field | Default | Meaning |
|---|---:|---|
| `iou_threshold` | `0.5` | IoU at or above which a prediction is accepted as a true positive. |
| `conf_threshold` | `0.0` | Predictions below this confidence are dropped before matching. |
| `min_area` | `0` | Boxes whose pixel area (w × h) is strictly below this value are discarded. |
| `class_mapping` | `{}` | Maps raw annotation label strings to canonical class names. Absent labels are kept as-is. |
| `weak_thresholds` | `{}` | Per-class thresholds as `{class_name: {recall, precision, ap50}}`. Absent keys fall back to the `default_*_threshold` values. Reference ITS defaults: `car 0.99`, `bicycle 0.7`, `person 0.7` — a strict gate on the abundant, well-learned class and looser gates on the rare ones the loop exists to improve. |
| `default_recall_threshold` | `0.5` | Fallback recall threshold for classes not listed in `weak_thresholds`. |
| `default_precision_threshold` | `0.0` | Fallback precision threshold. Set to `0.0` to disable precision-based weak selection. |
| `default_ap50_threshold` | `0.5` | Fallback for classes absent from `weak_thresholds`. **Set `0.0`** so unlisted classes never mark an image weak — the reference filter had no fallback, and leaving TAO DS's `0.5` in place silently gates every class you did not list. |

Do not hand-write the spec. Copy the template and fill in the `null`s — every
tuning value it already carries is the one this stage wants — then validate:

```bash
cp skills/data/tao-analyze-gaps-od-map/assets/default_object_detection.yaml "$SPEC"
# fill ground_truth_ann_path, inference_ann_path, images_dir, results_dir, kpi, input_format
python3 skills/data/tao-analyze-gaps-od-map/scripts/verify_object_detection_spec.py --spec "$SPEC"
```

`verify` rejects the spellings that fail — uppercase `input_format`, relative or
missing paths, a `weak_thresholds` entry that is a bare number rather than a
mapping — and reports every gated class plus the `default_*` fallbacks, so the
selection criteria behind a weak set are recoverable from the run's output. It
warns when a fallback is above zero, since that gates classes you did not list.

## Quick Start

Run from the `tao-skill-bank` repo root.

**Write the spec into the results directory.** The run emits four artifacts and
does not retain the spec, so a completed gap analysis otherwise cannot tell you
which thresholds produced its weak set — and that weak set sizes the mining
budget downstream. Keeping them together makes the selection criteria recoverable
from the run alone.

```bash
RESULTS_DIR=/absolute/path/for/this/run          # results_dir in the spec
SPEC="$RESULTS_DIR/object_detection.yaml"        # spec lives beside its outputs
RUN_ROOT=/absolute/path/that/contains/annotations/images/and/results
GPU_COUNT=1

DS_IMAGE=nvcr.io/nvidia/tao/tao-toolkit:7.2.0-data-services  # versions-key: images.tao_toolkit.data_services

docker run --rm --gpus "$GPU_COUNT" --shm-size=8g --network=host \
  -v "$RUN_ROOT:$RUN_ROOT" \
  -w "$RUN_ROOT" \
  "$DS_IMAGE" \
  gap_analysis object_detection -e "$SPEC"
```

Do not pass `--user $(id -u):$(id -g)`; some TAO DS images call `getpass.getuser()` at startup and fail when the UID is not in `/etc/passwd`.

## Preflight

1. Verify Docker access:

```bash
docker info > /dev/null
```

2. Resolve and pull the data-services image if needed:

```bash
DS_IMAGE=nvcr.io/nvidia/tao/tao-toolkit:7.2.0-data-services  # versions-key: images.tao_toolkit.data_services
docker image inspect "$DS_IMAGE" > /dev/null || docker pull "$DS_IMAGE"
```

3. Confirm `RUN_ROOT` contains the spec, both annotation sources, and the image directory. Mount `RUN_ROOT` to the same absolute path inside Docker.

## Outputs

| Artifact | Location | Contents |
|---|---|---|
| FP/FN box gaps | `results_dir/box_gaps.parquet` | One row per unmatched box: `kpi`, `image_id`, `filepath`, `class`, `gap_type` (FP/FN), `bbox`, `confidence`, `best_iou`. |
| Per-image metrics | `results_dir/image_metrics.parquet` | Per-image per-class: `tp`, `fp`, `fn`, `precision`, `recall`, `ap50`. |
| Weak images | `results_dir/weak_images.parquet` | Images where any class metric falls below threshold: `filepath`, `weak_classes`, `weak_recall`, `weak_precision`, `weak_ap50`. Feed this into `tao-mine-od-images`. |
| Gap report | `results_dir/gap_report.json` | FP/FN counts by type and class, plus run settings. |

All four artifacts are always written, even when no gaps are found.

## Troubleshooting

**`The subtask object_detection requires -e/--experiment_spec_file`**: rerun with `gap_analysis object_detection -e "$SPEC"`.

**Input path not found inside Docker**: use a `RUN_ROOT` mount where host and container paths are identical.

**`input_format` error**: set `input_format: kitti` or `input_format: coco` explicitly — it is never inferred from the path.

**`weak_images.parquet` is empty**: all class metrics are above their thresholds. Lower `default_recall_threshold` / `default_ap50_threshold` or add per-class entries to `weak_thresholds`.

**Output directory not writable after Docker exits**: the container writes as root. Chown back with `docker run --rm -v "$RUN_ROOT:$RUN_ROOT" alpine chown -R "$(id -u):$(id -g)" "$RESULTS_DIR"`.
