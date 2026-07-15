---
name: tao-train-bevfusion
description: BEVFusion for multi-sensor 3D object detection. Fuses LiDAR point clouds and camera images in bird's-eye-view
  (BEV) space, used in autonomous driving for robust 3D perception. Use when training, evaluating, or running inference for
  a TAO BEVFusion model. Trigger phrases include "train BEVFusion", "LiDAR + camera fusion", "BEV 3D detection", "multi-sensor
  3D perception".
license: Apache-2.0
compatibility: Requires docker + nvidia-container-toolkit.
metadata:
  version: "0.1.0"
  author: NVIDIA Corporation
allowed-tools: Read Bash
tags:
- multi
- sensor
- 3d
- detection
---

# BEVFusion

BEVFusion for multi-sensor 3D object detection. Fuses LiDAR point clouds and camera images in bird's-eye-view (BEV) space. Used in autonomous driving for robust 3D perception.

Set pretrained backbone paths for Swin image backbone.

BEVFusion requires the BEVFusion-specific TAO container
`nvcr.io/nvidia/tao/tao-toolkit:5.5.0-pyt`. The shared TAO PyTorch 7.0 RC image
does not package `mmdet3d` and fails before any BEVFusion action can parse its
spec. The model-skill action is named `dataset_convert`, but the 5.5 container
CLI subtask is `bevfusion convert -e <spec>`.

## Dataclass Schemas

Generated TAO Core schemas are packaged in `schemas/<action>.schema.json`, with `schemas/manifest.json` listing available actions. Each generated schema also emits `references/spec_template_<action>.yaml` from the schema top-level `default` field. AutoML enablement is declared at the model layer in `references/skill_info.yaml` via `automl_enabled`. Runnable AutoML still requires `schemas/train.schema.json` and `references/spec_template_train.yaml` to exist and parse. Use the packaged train schema for `automl_default_parameters`, `automl_disabled_parameters`, defaults, min/max bounds, enums, option weights, math conditions, dependencies, and popular parameters. Do not expect `~/tao-core` at runtime; maintainers regenerate schemas/templates before packaging the skill bank.

## Train Action Policy

This model is AutoML-enabled at the model layer. Before handling any train-stage request, read `references/skill_info.yaml` and resolve the run override from either an explicit `automl_policy` value or the user's workflow request. Use `automl_policy: on` by default and only expose `on` / `off` in new launch prompts. Treat phrases like "turn off AutoML", "disable AutoML", "no HPO", or "plain training" as `automl_policy: off` for this run only. When `automl_policy: on`, `automl_enabled: true`, and both `schemas/train.schema.json` and `references/spec_template_train.yaml` are packaged, route the train action through `tao-skill-bank:tao-run-automl` by default with this model's `skill_dir`. Preserve workflow/application overrides for datasets, specs, output directories, GPU/platform settings, parent checkpoints, and `automl_policy`. Use direct model training only when `automl_policy: off` or the packaged train schema/template is missing; in the missing-schema case, report that AutoML is enabled but not runnable for this model until schemas are generated.

Non-train actions such as `evaluate`, `inference`, `export`, and deploy flows stay in this model skill. The per-run `automl_policy` override does not change model metadata.

## Training Requirements

- **Dataset type:** bevfusion
- **Formats:** default
- **Monitoring metric:** AP11

### Per-Action Dataset Requirements

| Action | Spec Key | Source | Files | List? |
|---|---|---|---|---|
| dataset_convert | root_dir | id |  | No |
| evaluate | dataset.test_dataset | train_datasets | ann_file: results/{dataset_convert_job_id}/kitti_person_infos_val.pkl | No |
| inference | dataset.root_dir | train_datasets |  | No |
| inference | dataset.test_dataset | train_datasets | ann_file: results/{dataset_convert_job_id}/kitti_person_infos_val.pkl | No |
| train | dataset.train_dataset | train_datasets | ann_file: results/{dataset_convert_job_id}/kitti_person_infos_train.pkl | No |
| train | dataset.val_dataset | train_datasets | ann_file: results/{dataset_convert_job_id}/kitti_person_infos_val.pkl | No |
| train | dataset.test_dataset | train_datasets | ann_file: results/{dataset_convert_job_id}/kitti_person_infos_val.pkl | No |

### Typical Spec Overrides

Data source overrides are **mandatory for every action** — the agent MUST construct data source paths from the Per-Action Dataset Requirements table above and include them in `spec_overrides`.

```python
DATA_ROOT = "/path/to/kitti_root"
CONVERTED = DATA_ROOT  # BEVFusion 5.5 writes info pickles into root_dir.
DATA_PREFIX = {"pts": "training/velodyne_reduced", "img": "training/image_2"}
```

**dataset_convert (mandatory data sources):**
```python
{
    "root_dir": DATA_ROOT,
    "results_dir": DATA_ROOT,
    "mode": "training",
}
```

**train (mandatory data sources):**
```python
{
    "train.num_epochs": 30,
    "train.checkpoint_interval": 10,
    "train.validation_interval": 10,
    "train.num_gpus": 1,
    "dataset.root_dir": DATA_ROOT,
    "dataset.train_dataset": {"ann_file": f"{CONVERTED}/kitti_person_infos_train.pkl", "data_prefix": DATA_PREFIX},
    "dataset.val_dataset": {"ann_file": f"{CONVERTED}/kitti_person_infos_val.pkl", "data_prefix": DATA_PREFIX},
    "dataset.test_dataset": {"ann_file": f"{CONVERTED}/kitti_person_infos_val.pkl", "data_prefix": DATA_PREFIX},
}
```

**evaluate (mandatory data sources):**
```python
{
    "dataset.root_dir": DATA_ROOT,
    "dataset.test_dataset": {"ann_file": f"{CONVERTED}/kitti_person_infos_val.pkl", "data_prefix": DATA_PREFIX},
}
```

**inference (mandatory data sources):**
```python
{
    "dataset.root_dir": DATA_ROOT,
    "dataset.test_dataset": {"ann_file": f"{CONVERTED}/kitti_person_infos_val.pkl", "data_prefix": DATA_PREFIX},
}
```
## Eval Dataset

Optional. Val dataset split is configured via ann_file in dataset config.

## Important Parameters

- **dataset.classes**: List of detection classes. Default ["person"]. Must match the annotation categories.
- **dataset.type**: Dataset type. Options: KittiPersonDataset, TAO3DSyntheticDataset, TAO3DDataset.
- **dataset.root_dir**: Root directory of the KITTI-style dataset.
- **dataset.box_type_3d**: 3D box coordinate frame. Options: lidar, camera. Default lidar.
- **train.optimizer.lr**: Learning rate. Default 2e-4 (AdamW). Use AmpOptimWrapper for mixed precision via optimizer.wrapper_type.
- **input_modality**: Dict controlling sensor modalities. Keys: use_lidar (True), use_camera (True), use_radar (False), use_map (False).
- **model.img_backbone**: Image backbone. Default mmdet.SwinTransformer (Swin-Tiny). embed_dims=96, depths=[2,2,6,2].
- **model.view_transform.type**: View transform for BEV projection. Options: DepthLSSTransform, LSSTransform. Default DepthLSSTransform.
- **model.point_cloud_range**: Spatial extent of LiDAR. Default [0,-40,-3,70.4,40,1].
- **model.voxel_size**: Voxel dimensions. Default [0.05, 0.05, 0.1].
- **dataset.train_dataset.batch_size**: Per-GPU batch size. Default 4.

## Multi-GPU / Multi-Node

**Launch method:** `torchrun` (LIGHTNING_EXCLUDED_NETWORK). The entrypoint runs `torchrun --nnodes=N --nproc-per-node=M train.py`, NOT plain `python`.

| Spec Key | Description | Default |
|----------|-------------|---------|
| `train.num_gpus` | Number of GPUs per node | 1 |
| `train.gpu_ids` | GPU device indices | [0] |
| `train.num_nodes` | Number of nodes | 1 |

- `CUDA_VISIBLE_DEVICES` is explicitly set from `TAO_VISIBLE_DEVICES`
- BEVFusion uses mmdet3d-based distributed training, not Lightning DDP
- `NODE_RANK` is copied to `RANK` if `RANK` is unset

**Multi-node env vars** (set by orchestrator):

| Variable | Purpose |
|----------|---------|
| `WORLD_SIZE` | Number of nodes |
| `NODE_RANK` | This node's rank |
| `MASTER_ADDR` | Rank-0 node IP |
| `MASTER_PORT` | Rank-0 port (default 29500) |
| `NUM_GPU_PER_NODE` | GPUs per node |

## Hardware

Minimum 2 GPU(s), recommended 4 GPU(s). 24GB+ (A100 recommended) VRAM per GPU. BEVFusion is memory-intensive due to multi-sensor fusion. A100 GPUs strongly recommended. Multi-GPU training expected.

## Error Patterns

**dataset_convert required**: Run the model-skill `dataset_convert` action
(`bevfusion convert -e <spec>` in the BEVFusion 5.5 container) before training
to produce `kitti_person_infos_train.pkl`, `kitti_person_infos_val.pkl`, and
`training/velodyne_reduced`. For direct local-docker 5.5 runs, set
`results_dir` to the same mounted path as `root_dir`; the converter writes the
info pickles there and later expects them under `root_dir` while reducing point
clouds.

**KITTI directory names**: The BEVFusion 5.5 converter writes reduced point
clouds under `training/velodyne_reduced` and expects camera images under
`training/image_2`. Do not use the stale `training/lidar_reduced` or
`training/images/` defaults when chaining dataset_convert into train/evaluate or
inference.

**BEVFusion 5.5 config surface**: Use the 5.5 dataclass keys in packaged
templates. Remove newer top-level/action keys such as `model_name`,
`wandb.group`, `wandb.run_id`, `train.checkpoint_interval_unit`,
`evaluate.trt_engine`, `evaluate.batch_size`, `inference.trt_engine`, and
`inference.batch_size`. For train, evaluate, and inference specs, keep the
non-running action stubs (`train`, `evaluate`, and `inference`) present with
empty checkpoint strings where needed; the 5.5 runners materialize the full
experiment config before running the selected action. Use YAML null, not an
empty string, for `train.pretrained_checkpoint` and
`train.resume_training_checkpoint_path` when no checkpoint is intended.

**`ModuleNotFoundError: No module named 'mmdet3d'`**: The shared TAO PyTorch
7.0 RC image does not include the BEVFusion `mmdet3d` dependency. Use
`nvcr.io/nvidia/tao/tao-toolkit:5.5.0-pyt`; it contains `mmdet3d` and exposes
the BEVFusion `convert`, `train`, `evaluate`, and `inference` subtasks.

**Post-evaluation SIGSEGV in BEVFusion 5.5**: Some local-docker runs can write
checkpoints or prediction files and still finish with TAO `Execution status:
FAIL` after `Signal 11 (SIGSEGV)` in `cuMemRetainAllocationHandle`. Do not mark
the action successful from the Docker exit code alone; inspect the TAO log or
`status.json`. If a checkpoint was produced before this failure, use only the
exact intended checkpoint such as `epoch_1.pth` for downstream diagnostics and
do not treat `last_checkpoint` as a best checkpoint unless the action explicitly
requests the latest checkpoint.

**Missing modality data**: Ensure both camera images and LiDAR point clouds are present if using multi-modal fusion.

**Epoch numbering**: BEVFusion checkpoint epoch numbers may not follow standard zero-padded format.

**Checkpoint handoff**: Use the SDK/model checkpoint resolver for parent-model
selection. For direct local-docker chaining, inspect the train results and pass
the exact intended checkpoint path such as `epoch_1.pth`; use `latest.pth` only
when the user explicitly asks for latest. Resume/retrain must set
`train.resume: true` and `train.resume_training_checkpoint_path` to the exact
checkpoint being resumed.

## Spec Param / Parent Model Inference

Model-specific inference mappings belong in this MD file, not in `config.json`. Generated runners should read this section and apply the mappings with SDK helpers before `create_job()`. This mirrors the old microservices `infer_params.py` flow.

Inference mappings from TAO Core `bevfusion.config.json`:

| Action | Spec Field | Inference Function | Meaning |
|---|---|---|---|
| dataset_convert | `results_dir` | `output_dir` | current job results directory |
| evaluate | `encryption_key` | `key` | encryption key |
| evaluate | `evaluate.checkpoint` | `parent_model` | model file inferred from the parent job results folder |
| evaluate | `results_dir` | `output_dir` | current job results directory |
| inference | `encryption_key` | `key` | encryption key |
| inference | `inference.checkpoint` | `parent_model` | model file inferred from the parent job results folder |
| inference | `results_dir` | `output_dir` | current job results directory |
| train | `encryption_key` | `key` | encryption key |
| train | `results_dir` | `output_dir` | current job results directory |
| train | `train.pretrained_checkpoint` | `ptm_if_no_resume_model` | PTM when no resume checkpoint exists |
| train | `train.resume_training_checkpoint_path` | `resume_model` | model file inferred from the current job results folder |

For `parent_model` or `parent_model_folder`, pass the upstream train/export/AutoML child job id as `parent_job_id`. The SDK lists the parent result folder, filters checkpoint artifacts, and returns the selected model file or folder. Do not add these mappings back to `config.json` and do not patch generated runner scripts to guess checkpoint paths.
