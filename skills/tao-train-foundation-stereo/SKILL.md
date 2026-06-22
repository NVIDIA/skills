---
name: tao-train-foundation-stereo
description: Stereo depth estimation using FoundationStereo. Predicts disparity maps from stereo image pairs for 3D
  reconstruction. Use when training, evaluating, exporting, or running inference for a TAO FoundationStereo model. Trigger
  phrases include "train stereo depth", "FoundationStereo", "stereo disparity estimation", "3D reconstruction from stereo".
license: Apache-2.0
compatibility: Requires docker + nvidia-container-toolkit.
metadata:
  version: "0.1.0"
  author: NVIDIA Corporation
allowed-tools: Read Bash
tags:
- stereo
- depth
- estimation
---

# Depth Net Stereo

Stereo depth estimation using FoundationStereo architecture. Predicts disparity maps from stereo image pairs for 3D reconstruction.

Uses pretrained Depth Anything v2 and EdgeNeXt encoders. Set `model.stereo_backbone.depth_anything_v2_pretrained_path` and `model.stereo_backbone.edgenext_pretrained_path`.

The mono and stereo skills both invoke the unified TAO `depth_net` CLI inside the container; the mono/stereo family is selected via `model.model_type` (e.g., `FoundationStereo`).

PyT actions packaged by this model skill: `train`, `evaluate`, `inference`, `export`, and `quantize`. The PyT `depth_net` entrypoint does not accept a `gen_trt_engine` action in the current TAO image; build TensorRT engines only through the deploy workflow.

For TAO Deploy TensorRT actions (`gen_trt_engine`, TensorRT `evaluate`, and TensorRT `inference`), read `references/tao-deploy-foundation-stereo.md` first. The deploy spec template lives in this skill's `references/spec_template_deploy.yaml`.

## Train Action Policy

This model is AutoML-enabled at the model layer. Before handling any train-stage request, read `references/skill_info.yaml` and resolve the run override from either an explicit `automl_policy` value or the user's workflow request. Use `automl_policy: on` by default and only expose `on` / `off` in new launch prompts. Treat phrases like "turn off AutoML", "disable AutoML", "no HPO", or "plain training" as `automl_policy: off` for this run only. When `automl_policy: on`, `automl_enabled: true`, and both `schemas/train.schema.json` and `references/spec_template_train.yaml` are packaged, route the train action through `tao-skill-bank:tao-run-automl` by default with this model's `skill_dir`. Preserve workflow/application overrides for datasets, specs, output directories, GPU/platform settings, parent checkpoints, and `automl_policy`. Use direct model training only when `automl_policy: off` or the packaged train schema/template is missing; in the missing-schema case, report that AutoML is enabled but not runnable for this model until schemas are generated.

Non-train actions such as `evaluate`, `inference`, `export`, and deploy flows stay in this model skill. The per-run `automl_policy` override does not change model metadata.

## Workflow

### Prerequisites — data accessibility

Your dataset (left + right images + GT disparity) must be reachable from inside the container:
- **SDK runner**: place files at the S3 paths the runner resolves (the `S3_TRAIN` / `S3_EVAL` placeholders shown in the spec overrides). The runner handles S3 → container-path mounting transparently.
- **Direct `docker run`** (e.g. local testing): mount the host dataset root read-only at the same in-container path:

```
docker run ... -v <host_data_root>:<host_data_root>:ro <container> ...
```

The same accessibility requirement applies to the `<output_dir>` written by all actions.

### Step 1 — Annotation file

Per-line annotation file referenced by `data_sources[*].data_file`:

| Columns | Format | Use |
|---|---|---|
| 2 | `<left> <right>` | Stereo inference (no GT) |
| 3 | `<left> <right> <disparity>` | Stereo with GT |
| 4 | `<left> <right> <disparity> <occlusion_mask>` | Stereo with GT and occlusion mask |

If you already have one, point to it. Otherwise generate via `depth_net convert`:

```
depth_net convert -e <convert_spec.yaml>
```

`convert_spec.yaml` template (stereo):

```yaml
results_dir: <directory where generated annotation files are written>
data_root: <directory whose immediate children are scene folders that contain your image+depth files; convert walks data_root recursively but expects per-scene subdirectories at one level below>
image_dir_pattern: [<substring matching left image paths>]
right_dir_pattern: [<substring matching right image paths>]
depth_dir_pattern: [<substring matching GT disparity paths>]
nocc_dir_pattern: []                 # optional, occlusion mask paths
image_extension: '.png'  # always include the leading dot
depth_extension: '.png'  # form must match image_extension (the swap is a substring replace)
nocc_extension: ''
split_ratio: 0.0        # 0.0/1.0 = test-only; 0.8 = 80/20 train+val
```

`convert` walks `data_root` recursively, selects paths whose path-string contains *all* substrings in `image_dir_pattern` (AND-filter), then derives right / depth / mask paths by replacing `image_dir_pattern[0]` with the corresponding pattern's first element plus extension swap. Inspect your dataset's directory layout and identify the substrings distinguishing left, right, and GT (e.g. `im0` vs `im1` vs `disp0GT` for Middlebury).

### Step 2 — Pair `model_type` and `dataset_name` based on your data

Prefer the dataset-specific class when your layout matches a supported one — it applies class-specific path conventions, evaluation crops, and (where applicable) occlusion-mask handling. Fall back to `GenericDataset` only for layouts that do not match any registered class.

| Data category | `model_type` | `dataset_name` |
|---|---|---|
| Middlebury data | `FoundationStereo` | `Middlebury` |
| KITTI data | `FoundationStereo` | `Kitti` |
| ETH3D data | `FoundationStereo` | `Eth3d` |
| FSD synthetic data | `FoundationStereo` | `FSD` |
| IsaacReal synthetic data | `FoundationStereo` | `IsaacRealDataset` |
| Crestereo synthetic data | `FoundationStereo` | `Crestereo` |
| Other / non-canonical layout | `FoundationStereo` | `GenericDataset` |

Valid `dataset_name` values for stereo `data_sources` (case-insensitive): `FSD`, `IsaacRealDataset`, `Crestereo`, `Middlebury`, `Eth3d`, `Kitti`, `GenericDataset`.

The same `dataset_name` value applies across train and evaluate actions (all of which use 3-column or 4-column annotations with GT disparity). The deploy-side `evaluate` action follows the same rule — see `references/tao-deploy-foundation-stereo.md`. For inference with 2-column annotations (left + right, no GT), use `dataset_name: GenericDataset` regardless of data layout — the dataset-specific classes (`Middlebury` / `Kitti` / `Eth3d` / `FSD` / `IsaacRealDataset` / `Crestereo`) require 3-column input and reject 2-column annotations at the dataloader level. For inference with 3-column annotations (left + right + GT), the dataset-specific class is fine.

### Step 3 — Write spec yaml from the spec overrides

Copy the action block from `references/spec-overrides-foundation-stereo.md`. Replace:
- `model.model_type` from Step 2 (typically `FoundationStereo`)
- `dataset.<...>.data_sources[*].dataset_name` from Step 2
- `dataset.<...>.data_sources[*].data_file` with the path from Step 1
- For deploy-side `evaluate`: enforce `dataset.test_dataset.batch_size: 1` (see `references/tao-deploy-foundation-stereo.md`).

Shape consistency: the `crop_size` in `dataset.test_dataset.augmentation.crop_size` should match `export.input_height` / `input_width` so the trained-model evaluator and the deploy-side TensorRT evaluator operate at the same shape. Note that `crop_size` is decorative on the pyt `evaluate` path but authoritative on the deploy `evaluate` side — see `references/troubleshooting-foundation-stereo.md` and `references/tao-deploy-foundation-stereo.md`.

Fresh-install smoke runs are validated at `crop_size: [128, 128]` with `dataset.max_disparity: 128` and `model.max_disparity: 128`. Avoid 112×112 crops and avoid setting `max_disparity` smaller than the square crop side for smoke tests: those combinations can fail inside FoundationStereo with feature-map or loss-mask shape mismatches before a checkpoint is produced.

Data source overrides are **mandatory for every action**. Each `data_sources` entry is a dict with two mandatory fields: `data_file` and `dataset_name`. See `references/spec-overrides-foundation-stereo.md` for the per-action dataset-requirements table, every action's override block, and the `quantize` known-issue note.

### Step 4 — Run

Create writable home/cache directories inside the mounted output path before using
`--user`. Some TAO containers do not have an `/etc/passwd` entry for the host UID,
and PyTorch / matplotlib need writable cache paths when running as that UID.

```bash
mkdir -p <output_dir>/home \
         <output_dir>/.cache/matplotlib \
         <output_dir>/.cache/torchinductor \
         <output_dir>/.cache/xdg
```

```
docker run --gpus 'device=0' --shm-size 16G --ipc=host \
  --user "$(id -u):$(id -g)" \
  -e USER="$(id -un)" \
  -e LOGNAME="$(id -un)" \
  -e HOME=<output_dir>/home \
  -e MPLCONFIGDIR=<output_dir>/.cache/matplotlib \
  -e TORCHINDUCTOR_CACHE_DIR=<output_dir>/.cache/torchinductor \
  -e XDG_CACHE_HOME=<output_dir>/.cache/xdg \
  -v <data_root>:<data_root>:ro \
  -v <output_dir>:<output_dir> \
  <container> \
  depth_net <action> -e <spec.yaml>
```

Without `--user "$(id -u):$(id -g)"` the container writes outputs as `nobody:nogroup`, blocking host-side cleanup / retry.

### Step 5 — Verify

- Container exit code 0
- `status.json` `kpi` block populated
- For `train`: inspect per-step `train_loss` directly (the entrypoint reports `Execution status: PASS` even when loss is NaN)
- For `evaluate`: rely on `epe` / `bp1` / `bp2` / `bp3` / `d1` / `rmse` (the evaluator also emits `abs_rel` / `sq_rel` / `rmse_log` which are non-meaningful for stereo — see `references/parameters-foundation-stereo.md`)
- For `inference`: artifacts under `results_dir`

For TAO Deploy TensorRT actions (`gen_trt_engine`, TensorRT `evaluate`, and TensorRT `inference`), read `references/tao-deploy-foundation-stereo.md` first. Deploy spec templates live in this skill's `references/` folder with the `spec_template_deploy_*.yaml` prefix.

## Training Requirements

- **Monitoring metric:** val/loss
- **Eval dataset:** optional. Val dataset configured via `dataset.val_dataset.data_sources` (each entry needs `data_file` and `dataset_name`).

See `references/spec-overrides-foundation-stereo.md` for the per-action dataset-requirements table and every action's mandatory data-source override block.

## Parameters, Metrics, Multi-GPU, Export/TRT, Hardware

See `references/parameters-foundation-stereo.md` for the full Important Parameters list (incl. `model.encoder` `vits` override, `model.max_disparity` default 416, `model.volume_dim` no-op note, `dataset.baseline`, `dataset.focal_x`, `train.precision`, `export.batch_size`), the Evaluation Metrics table, Multi-GPU / Multi-Node launch keys, Export / TRT Defaults (`opset_version`/`on_cpu` pairing, NGC 576×960 settings), and Hardware requirements.

## Error Patterns and Troubleshooting

See `references/troubleshooting-foundation-stereo.md` for disparity overflow, smoke-test shape mismatch, missing pretrained paths, the `encoder` / `dataset_name` struct errors, the `depth_net_stereo: not found` entrypoint note, the pyt-vs-deploy `crop_size` discussion, and the deploy `evaluate` scalar-conversion failure.

## Spec Param / Parent Model Inference

See `references/checkpoint-inference-mappings-foundation-stereo.md` for the checkpoint-resolution rules (`model_epoch_<epoch>_step_<step>.pth`, `dn_model_latest.pth` policy), the absence of parent PyT `gen_trt_engine`, and the full per-action inference-mapping table from `depth_net_stereo.config.json` (including `parent_model` / `parent_job_id` resolution).

## Deployment

- [tao-deploy-foundation-stereo](references/tao-deploy-foundation-stereo.md)
