---
name: tao-analyze-detection-kpi
description: >-
  Run TAO Data Services KPI analysis for object detection, comparing inference annotations against
  ground truth to compute per-class TP/FP/FN/TN, precision, recall, accuracy, and AP at a fixed IoU.
  Use when an object detection workflow needs per-class mAP reported after inference, or when the user
  asks to "run KPI analyze", "compute detection mAP", or "score my OD predictions against ground truth".
license: Apache-2.0
compatibility: Requires docker, nvidia-container-toolkit, and the TAO data-services container pinned in versions.yaml.
metadata:
  author: NVIDIA Corporation
  version: "0.1.0"
allowed-tools: Read Bash
tags:
- tao
- data
- kpi
- object-detection
- mAP
- analytics
---

# TAO Analyze Detection KPI

Use this skill to run TAO Data Services KPI analysis for object detection. The skill compares inference annotations against ground truth over one or more KPI sources and writes a per-class metrics CSV. It does not run inference; an upstream step must produce the inference annotations first.

The container entrypoint is:

```bash
analytics kpi_analyze -e /absolute/path/to/kpi_analyze.yaml
```

## Inputs

The user provides either a finished spec or the paths to fill into the template.

Required spec fields:

| Field | Meaning |
|---|---|
| `data.input_format` | `KITTI` or `COCO`. **Uppercase** — see Pitfalls. |
| `data.kpi_sources` | List of sources. Each entry requires `image_dir`, `ground_truth_ann_path`, and `inference_ann_path`; all three are asserted at startup. |
| `data.mapping` | Path to a class-mapping YAML: a list of single-key dicts whose value is a **LIST** of aliases. See `assets/example_mapping.yaml` — a bare string here silently zeroes every metric. |
| `results_dir` | Output directory for `kpi_calc.csv`. |

Common optional fields. The `Default` column is what TAO DS uses when the field is
absent; `assets/default_kpi_analyze.yaml` already carries the recommended value
for each, so filling the template needs none of them changed:

| Field | Default | Meaning |
|---|---:|---|
| `kpi.iou_threshold` | `0.5` | IoU at or above which a prediction counts as a true positive. |
| `kpi.conf_threshold` | `0.5` | Predictions below this are dropped. The template uses `0.0`, which keeps the whole PR curve so a threshold can be swept afterwards without re-running inference. On the pinned image that is safe: unmatched ground truth carries a `-1.0` sentinel and lands in FN at any threshold. On a build predating that fix, `0.0` scored every missed box as a true positive — `TP` became the ground-truth count and `FN` was always 0 — so use a small positive value there. |
| `kpi.num_recall_points` | `11` | Recall points for the interpolated PR curve. The template keeps `11` (VOC-style), matching the reference ITS pipeline. `101` selects COCO-standard sampling and reports different numbers for the same detections. |
| `kpi.ignore_sqwidth` | `0` | Boxes narrower than this are ignored. The template uses `40`, matching the reference ITS pipeline, which never counted boxes below that. `0` scores small objects the reference excluded, so the two are not comparable. |
| `kpi.filter` | `false` | Enable source filtering. |
| `kpi.is_internal` | `false` | When true, drops every class except `person` and appends a `Summary` row. |
| `visualize.platform` | `local` | `local` writes a PR-curve plot into `results_dir`; `wandb` logs a run and table instead. |
| `visualize.tag` | `null` | Tag recorded on every row. |

For a Grounding DINO loop, `inference_ann_path` is the `labels/` directory TAO inference writes under `{results_dir}/inference/labels/`, and `input_format` is `KITTI`.

The default template is `assets/default_kpi_analyze.yaml`.

## Quick Start

Run from the `tao-skill-bank` repo root.

**Write the spec into the results directory.** The run does not retain it, so a
completed run otherwise cannot tell you which settings produced `kpi_calc.csv`.
Keeping them together makes the result reproducible from the run alone.

```bash
RESULTS_DIR=/absolute/path/for/this/run          # results_dir in the spec
SPEC="$RESULTS_DIR/kpi_analyze.yaml"        # spec lives beside its outputs
RUN_ROOT=/absolute/path/that/contains/images/annotations/and/results

python3 skills/data/tao-analyze-detection-kpi/scripts/verify_kpi_analyze_spec.py \
  --spec "$SPEC"

DS_IMAGE=nvcr.io/nvidia/tao/tao-toolkit:7.2.0-data-services  # versions-key: images.tao_toolkit.data_services

docker run --rm --gpus all --shm-size=8g --network=host \
  -v "$RUN_ROOT:$RUN_ROOT" \
  -w "$RUN_ROOT" \
  "$DS_IMAGE" \
  analytics kpi_analyze -e "$SPEC"
```

**Pass `--gpus all` even though the analysis itself is CPU-only.** The TAO launcher calls
`nvidia-smi -L` unconditionally before dispatching any subtask, so a container started
without GPU access dies with `FileNotFoundError: 'nvidia-smi'` before `kpi_analyze` runs.

Do not pass `--user $(id -u):$(id -g)`; some TAO DS images call `getpass.getuser()` at startup and fail when the UID is not in `/etc/passwd`.

## Generate A Spec

If the user provides paths instead of a ready spec, copy the template and fill in
the `null`s. Every tuning value it already carries is the one this stage wants —
change one only deliberately.

```bash
cp skills/data/tao-analyze-detection-kpi/assets/default_kpi_analyze.yaml "$SPEC"
```

Fill `data.kpi_sources` (one entry per source), `data.mapping` and `results_dir`,
all as absolute paths, then validate:

```bash
python3 skills/data/tao-analyze-detection-kpi/scripts/verify_kpi_analyze_spec.py --spec "$SPEC"
```

```yaml
data:
  input_format: KITTI
  kpi_sources:
  - image_dir: /absolute/path/kpi/images               # no trailing slash
    ground_truth_ann_path: /absolute/path/kpi/labels
    inference_ann_path: /absolute/path/results/inference/labels
  mapping: /absolute/path/mapping.yaml
results_dir: /absolute/path/results/analyze_kpi
```

The template is the only place a default value lives, so nothing can disagree
with it. `verify` reports the three settings that change what the numbers mean —
`conf_threshold`, `num_recall_points`, `ignore_sqwidth` — so the spec that ran is
recoverable from its output.

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

3. Validate the spec:

```bash
python3 skills/data/tao-analyze-detection-kpi/scripts/verify_kpi_analyze_spec.py --spec "$SPEC"
```

4. Confirm `RUN_ROOT` contains the spec, every `image_dir`, both annotation paths per source, the mapping file, and the results directory. Mount `RUN_ROOT` to the same absolute path inside Docker.

## Outputs

| Artifact | Location |
|---|---|
| Per-class metrics CSV | `results_dir/kpi_calc.csv` |
| PR curve plot | `results_dir/` (only when `visualize.platform: local`) |

`kpi_calc.csv` columns: `Sequence Name`, `TP`, `FP`, `FN`, `TN`, `Pr`, `Re`, `Acc`, `AP` — one row per sequence per class. A per-class result table and the aggregate mAP are also printed to stdout; capture the log if the caller needs the mAP value, since it is not written to the CSV.

## Pitfalls

**Ground truth may be 15- or 16-field KITTI.** The parser names 15 columns for ground truth
and 16 for predictions, but reads with `index_col=False`, so a trailing `conf_score` on a GT
file is truncated rather than shifted. Feeding GT straight from tooling that writes a score
column is fine — verified byte-identical results either way. The `ParserWarning` about
"length of header or names does not match length of data" describes exactly that truncation
and is not a sign of corruption.


**`input_format` is uppercase here.** `analytics kpi_analyze` accepts only `KITTI` or `COCO`. This differs from `gap_analysis object_detection`, which takes lowercase `kitti` / `coco`. Passing lowercase to this action fails to construct the data object.

**`Sequence Name` is derived from the path, not configured.** It is `image_dir.split('/')[-2]` — the *second-to-last* component of `image_dir`. A trailing slash or a flat image directory shifts which component is picked, so two sources can collide under one name. Lay out `image_dir` so that component is the sequence identifier you want.

**`data.mapping` values are LISTS of aliases, not strings.** This is the single most
destructive thing to get wrong: the file is a YAML list of single-key dicts whose value is a
list of source names that fold into that canonical class.

```yaml
- bicycle:
    - Bicycle
    - Motorcycle
    - bicycle
    - twowheeler
- car:
    - car
    - Heavy Truck
    - Vehicle
```

`construct_category_map` stores the value verbatim (`cat_map[k] = v`), so writing
`- car: car` — a bare string — yields a value that downstream code iterates **character by
character**. Class matching then fails for every box, and the run still exits 0: the result
is `TP=0, FN=0`, every prediction counted a false positive, and `mAP: 0.0`, with no error and
no warning.

The tell is a perfect-looking run with all-zero metrics. Sanity-check by scoring a
ground-truth set against a copy of itself — with a correct mapping that returns TP = every
box and `mAP: 1.0`; anything else means the mapping, not the model.

**`data.mapping` is required.** The Hydra schema marks it mandatory even though the underlying category-map builder can derive classes from the label directory when it is absent. Supply the YAML.

**Two different `conf_threshold` defaults.** The dataclass default is `0.5`, the shipped spec template uses `0.0`. Whichever you rely on, set it explicitly — an unset value silently changes which predictions are scored.

**`is_internal: true` is destructive to the report.** It drops every class except `person` and appends a `Summary` row. Leave it false unless you specifically want the internal person-only KPI.

## Troubleshooting

**`<key> not found in kpi_sources`**: every source entry needs all three of `image_dir`, `ground_truth_ann_path`, `inference_ann_path`.

**Paths not found inside Docker**: use a `RUN_ROOT` mount where host and container paths are identical, and confirm the images and annotation directories are under that mount.

**Empty or all-zero metrics**: usually `conf_threshold` above the model's score range, or an `input_format` that does not match the annotations on disk.

**wandb errors or hangs**: set `visualize.platform: local` to write a PR-curve plot instead of logging to wandb.
