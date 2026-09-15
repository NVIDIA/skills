---
name: foundationpose-pipeline
description: Adapt BOP datasets, run the FoundationPose perception pipeline with TAO depth, and evaluate or re-score pose results. Use for dataset runs and result comparisons; environment installation belongs to foundationpose-setup.
license: Apache-2.0
metadata:
  author: "zwdoescode <zhengwang@nvidia.com>"
  version: "0.1.0"
---

# Run and evaluate the FoundationPose perception pipeline

## Purpose

Run depth, SAM3 segmentation, and FoundationPose on BOP-format datasets using the TAO Deploy
TensorRT depth engine. Adapt datasets, preserve run provenance, and interpret pose metrics.
For missing dependencies or engine construction, use `foundationpose-setup` if installed,
or the product checkout's README Install and Verify sections.

## Requirements

Locate the user's [product checkout](https://github.com/nvidia-isaac/foundationpose_perception_pipeline)
by `pyproject.toml` (project `foundationpose-perception-pipeline`), `script/run_pipeline.py`, and
`config/defaults.yaml`. Run commands from that root, not from this installed skill's directory.
A catalog install supplies instructions, not the product code, datasets, or weights. If execution
was requested and no checkout exists, obtain it from the URL above and complete setup first.
For advice or analysis of supplied artifacts, use those inputs without cloning or loading models.

Execution requires the product's Python 3.12 venv, authorized SAM3 checkpoint access, the built
FoundationPose library, an adapted dataset, a matching TAO engine with its sidecar, and sufficient
GPU memory. Read the checkout's README Configuration and Dataset adaptation sections for
profile paths; read ARCHITECTURE.md Outputs for the matching artifact schema.

Set absolute paths before GPU work:

```bash
export FOUNDATIONPOSE_ROOT="$(realpath ../foundation-pose-inference-library)"
PIPELINE_SITE="$(realpath .venv/lib/python3.12/site-packages)"
export LD_LIBRARY_PATH="${PIPELINE_SITE}/tensorrt_libs:${PIPELINE_SITE}/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}"
./.venv/bin/python -c "import ctypes; ctypes.CDLL('libcudart.so.13'); print('ok')"
```

Do not mix libraries from another venv into this path. Skipping the check can cause pose to fail
after depth has already completed.

## Instructions

### 1. Resolve the task and inputs

Identify the profile, dataset name, source or adapted scene paths, engine, ground-truth
availability, and output directory. `<profile>` and `<dataset>` may differ. `--config` selects a
profile; it does not replace a required `--dataset`. Same-named profiles can be inferred by
commands that take `--dataset`.

| Request | Entry point |
|---|---|
| Convert a supported BOP dataset | `tools/bop_adapt/adapt.py` |
| Inference without pose ground truth | `script/infer.py` |
| Inference plus scoring | `script/run_pipeline.py` |
| Score a completed run with new scoring parameters | `script/evaluate.py` |
| Sweep several datasets | `script/run_batch_eval.py` |

A capture without `scene_gt.json` can use inference only. `--no-depth-metrics` skips collected
sensor-depth scoring; it does not remove the pose-ground-truth requirement for evaluation.

### 2. Adapt before building or checking an engine

Skip adaptation only for the pipeline's rig layout:
`<split>/<scene>/rgb/<im_id>.png`, one `scene_camera.json` per scene, and im_ids representing
rig cameras (base camera 0 in the shipped profiles).

```bash
./.venv/bin/python tools/bop_adapt/adapt.py --config <profile> --src <downloaded-dataset>
```

The profile's `dataset.name` selects a registered adapter; `--help` exposes its flags. An unknown
adapter is not supported automatically. On static-scene datasets the adapter emits one scene
per usable (source scene, base frame) pair and reports skipped frames with no rectifiable partner.
Changing the baseline band changes the adapted data: rebuild GT caches and regenerate depth.

Engine building uses `tools/build_tao_engine.py --shape-from-scene <adapted-scene>`; see the
checkout's README Install section if the setup skill is unavailable. A raw BOP directory or raw
image dimensions do not establish the required rectified engine shape.

### 3. Validate inputs and prepare GT caches

```bash
./.venv/bin/python test/check_engine_depth_smoke.py \
  --config <profile> --dataset <dataset> --engine <engine-path>
```

Expect `backend=tao`, `normalization=imagenet`, a fixed shape, a plausible valid fraction, and no
`cropping N rows` warning. A stale sidecar, changed GPU/TensorRT/precision, changed max-width,
or cropping requires rebuilding the engine and regenerating depth. Do not bypass these checks.

For scoring runs with pose GT, precompute the cache:

```bash
./.venv/bin/python script/build_gt_cache.py --config <profile> --dataset <dataset>
```

Use `--config <profile> --all` for all matching datasets. Missing collected depth calls for
`--no-depth-metrics`; missing `scene_gt.json` calls for inference only. Check the resolved
`dataset.collected_depth_root` using the actual path, not a shell command substitution.

### 4. Run only the work needed

For a new end-to-end run:

```bash
./.venv/bin/python script/run_pipeline.py --config <profile> --dataset <dataset> \
  --output-dir output/<new-run> --foundation-stereo-model <engine-path> \
  --depth-backend commercial --no-depth-metrics
```

Omit `--no-depth-metrics` when collected sensor depth is available and should be scored.
Start with `--max-scenes 1` for a time/fit check before sizing a larger run.
For a capture with no pose ground truth:

```bash
./.venv/bin/python script/infer.py --config <profile> --dataset <dataset> \
  --output-dir output/<new-run> --foundation-stereo-model <engine-path> \
  --depth-backend commercial
```

The model path selects the backend. `--depth-backend commercial` asserts that selection; it
neither downloads a model nor establishes rights to the weights. Set the engine once in the
profile's `overrides.depth.engine` to avoid repeating the model-path flag.

Preserve existing results when comparing runs. Reuse cached depth only after checking its
metadata. `--overwrite-results` reruns segmentation and pose; `--overwrite-depth` additionally
regenerates depth. Regenerate depth after changes to the engine, rectified width, CLAHE,
working-distance bounds, or adapted data. Resume a pose-only failure without overwriting valid
depth. Working-distance bounds must be supplied together.

Do not repeat tuned defaults from `config/defaults.yaml` on every command; use profile overrides
for deliberate dataset-specific changes. Rebuild the engine if `foundation_stereo_max_width`
changes.

### 5. Re-score without repeating inference

For a rerank cutoff, IoU threshold, or visibility-band change, keep the completed predictions,
mask sidecars, and depth files and run:

```bash
./.venv/bin/python script/evaluate.py --config <profile> --dataset <dataset> \
  --run output/<completed-run> --output-dir output/<new-score-run> \
  --rerank-cutoff 4.5 --no-depth-metrics
```

Omit `--no-depth-metrics` when depth comparison is desired. A separate `--output-dir` preserves
the old report. No inference model is loaded; a GT cache miss can still require rasterization.
For an offline cutoff sweep, `tools/sweep_rerank_cutoff.py --config <profile> --results-root output
--datasets <dataset>` expects one dataset subdirectory under the results root.

### 6. Verify provenance and interpret results

Inspect `inference_config.json` for the engine and max-width, and each scene's
`depth/<scene>/metadata.json` for `backend: tao`, `normalization: imagenet`, and `model_fixed_hw`.
Inspect both before trusting cached depth. These establish execution provenance, not legal approval.

Read `report.md`, `pose_summary.json` (`overall`, `by_visibility`, `by_object`), and
`depth_summary.json` when depth was scored. Compare:

- `matched_predictions` first: a change in matched population can bias apparent accuracy gains.
- `max_vertex_error_within_threshold_rate` against the configured threshold and required rate.
- Median, p90, and p99 vertex error, ADD/ADD-S, and rotation error, including per-object results.
- Depth error on object pixels; whole-image error can be dominated by the table or background.

A higher success rate can coexist with a worse mean or tail. Report both, along with population
changes. Preserve a baseline before overwriting results; use separate run directories when
retaining predictions and provenance matters.

### 7. Batch runs

```bash
./.venv/bin/python script/run_batch_eval.py --config <profile> --output-root output/<batch-run> \
  --foundation-stereo-model <engine-path> --depth-backend commercial --no-depth-metrics
```

The no-depth flag also removes collected-depth filtering from batch dataset discovery. Add
`--continue-on-error` only when failed datasets should not stop the sweep. Run GPU datasets
sequentially; inspect `run_status.jsonl` before interpreting aggregate `summary.json` or `report.md`.

## Examples

- "Adapt T-LESS and run one scene with the TAO depth engine."
- "Re-score this finished FoundationPose run at cutoff 4.5 and preserve the old report."
- "Compare these pose summaries; did the 5 mm success rate improve at the same coverage?"

## Troubleshooting and limitations

A smoke check demonstrates backend operation, not pose accuracy. Accuracy requires a real
representative dataset and a retained baseline. Never report an unavailable metric as zero.
`invalid resource handle` points to pycuda context boundaries around TAO calls. Plausible depth
at roughly twice the expected scale calls for checking input normalization and calibration.
Report the command, dataset/profile, output paths, model provenance, completion status,
headline metrics with matched counts, and any unverified steps.
