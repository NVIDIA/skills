---
name: tao-train-pointpillars
description: PointPillars for 3D object detection from LiDAR point clouds. Encodes point clouds into a pseudo-image via a
  pillar-based representation, then applies 2D detection — used in autonomous driving and robotics. Use when training,
  evaluating, exporting, pruning, retraining, or running inference for a TAO PointPillars model. Trigger phrases include
  "train PointPillars", "LiDAR 3D detection", "point-cloud object detection", "pillar-based 3D detector".
license: Apache-2.0
compatibility: Requires docker + nvidia-container-toolkit.
metadata:
  version: "0.1.0"
  author: NVIDIA Corporation
allowed-tools: Read Bash
tags:
- point
- cloud
- 3d
- detection
---

# PointPillars

PointPillars for 3D object detection from LiDAR point clouds. Encodes point clouds into a pseudo-image via pillar-based representation, then applies 2D detection. Used in autonomous driving / robotics.

Typically trained from scratch. Provide train.resume_training_checkpoint_path to resume.

For TAO Deploy TensorRT actions (`gen_trt_engine`, TensorRT `evaluate`, and TensorRT `inference`), read `references/tao-deploy-pointpillars.md` first. Deploy spec templates live in this skill's `references/` folder with the `spec_template_deploy_*.yaml` prefix.

The packaged PyTorch PointPillars CLI supports `dataset_convert`, `train`, `evaluate`, `inference`, `export`, and `prune`. It does not expose a parent-model `gen_trt_engine` action; TensorRT engine generation is deploy-only. It also does not expose a separate `retrain` subcommand. Retraining from a pruned model uses `pointpillars train -e ...` with `train.pruned_model_path` populated.

## Dataclass Schemas

Generated TAO Core schemas are packaged in `schemas/<action>.schema.json`, with `schemas/manifest.json` listing available actions. Each generated schema also emits `references/spec_template_<action>.yaml` from the schema top-level `default` field. AutoML enablement is declared at the model layer in `references/skill_info.yaml` via `automl_enabled`. Runnable AutoML still requires `schemas/train.schema.json` and `references/spec_template_train.yaml` to exist and parse. Use the packaged train schema for `automl_default_parameters`, `automl_disabled_parameters`, defaults, min/max bounds, enums, option weights, math conditions, dependencies, and popular parameters. Do not expect `~/tao-core` at runtime; maintainers regenerate schemas/templates before packaging the skill bank.

## Train Action Policy

This model is AutoML-enabled at the model layer. Before handling any train-stage request, read `references/skill_info.yaml` and resolve the run override from either an explicit `automl_policy` value or the user's workflow request. Use `automl_policy: on` by default and only expose `on` / `off` in new launch prompts. Treat phrases like "turn off AutoML", "disable AutoML", "no HPO", or "plain training" as `automl_policy: off` for this run only. When `automl_policy: on`, `automl_enabled: true`, and both `schemas/train.schema.json` and `references/spec_template_train.yaml` are packaged, route the train action through `tao-skill-bank:tao-run-automl` by default with this model's `skill_dir`. Preserve workflow/application overrides for datasets, specs, output directories, GPU/platform settings, parent checkpoints, and `automl_policy`. Use direct model training only when `automl_policy: off` or the packaged train schema/template is missing; in the missing-schema case, report that AutoML is enabled but not runnable for this model until schemas are generated.

Non-train actions such as `evaluate`, `inference`, `export`, and deploy flows stay in this model skill. The per-run `automl_policy` override does not change model metadata.

## Training Requirements

- **Dataset type:** pointpillars
- **Formats:** default
- **Monitoring metric:** loss

### Per-Action Dataset Requirements

| Action | Spec Key | Source | Files | List? |
|---|---|---|---|---|
| dataset_convert | dataset.data_path | id |  | No |
| evaluate | dataset.data_path | train_datasets |  | No |
| evaluate | dataset.data_info_path | train_datasets | /results/{dataset_convert_job_id}/results_dir/data_info/ | No |
| export | dataset.data_path | train_datasets |  | No |
| export | dataset.data_info_path | train_datasets | /results/{dataset_convert_job_id}/results_dir/data_info/ | No |
| inference | dataset.data_path | train_datasets |  | No |
| inference | dataset.data_info_path | train_datasets | /results/{dataset_convert_job_id}/results_dir/data_info/ | No |
| prune | dataset.data_path | train_datasets |  | No |
| prune | dataset.data_info_path | train_datasets | /results/{dataset_convert_job_id}/results_dir/data_info/ | No |
| retrain | dataset.data_path | train_datasets |  | No |
| retrain | dataset.data_info_path | train_datasets | /results/{dataset_convert_job_id}/results_dir/data_info/ | No |
| train | dataset.data_path | train_datasets |  | No |
| train | dataset.data_info_path | train_datasets | /results/{dataset_convert_job_id}/results_dir/data_info/ | No |

### Typical Spec Overrides

Data source overrides are **mandatory for every action** — the agent MUST construct data source paths from the Per-Action Dataset Requirements table above and include them in `spec_overrides`.

```python
DATA_ROOT = "s3://bucket/data/pointpillars"
DATA_INFO = "/results/{dataset_convert_job_id}/results_dir/data_info"
CHECKPOINT = "/results/{train_job_id}/results_dir/checkpoint_epoch_1.pth"
PRUNED_MODEL = "/results/{prune_job_id}/results_dir/pruned_0.1.tlt"
```

The raw PointPillars data root must be an extracted folder containing matching `train/lidar`, `train/label`, `val/lidar`, and `val/label` subfolders before `dataset_convert` runs. If the source dataset is packaged as separate train/val archives, extract both under the same mounted data root and point `dataset.data_path` at that root.

**train (mandatory data sources):**
```python
{
    "train.num_epochs": 30,
    "train.checkpoint_interval": 10,
    "train.validation_interval": 10,
    "train.num_gpus": 1,
    "dataset.data_path": DATA_ROOT,
    "dataset.data_info_path": DATA_INFO,
}
```

**resume train (mandatory checkpoint):**
```python
{
    "dataset.data_path": DATA_ROOT,
    "dataset.data_info_path": DATA_INFO,
    "train.resume_training_checkpoint_path": CHECKPOINT,
}
```

**evaluate (mandatory data sources):**
```python
{
    "dataset.data_path": DATA_ROOT,
    "dataset.data_info_path": DATA_INFO,
    "evaluate.checkpoint": CHECKPOINT,
}
```

**export (mandatory data sources):**
```python
{
    "dataset.data_path": DATA_ROOT,
    "dataset.data_info_path": DATA_INFO,
    "export.checkpoint": CHECKPOINT,
    "export.onnx_file": "/results/{export_job_id}/results_dir/pointpillars.onnx",
}
```

**inference (mandatory data sources):**
```python
{
    "dataset.data_path": DATA_ROOT,
    "dataset.data_info_path": DATA_INFO,
    "inference.checkpoint": CHECKPOINT,
}
```

**prune (mandatory data sources):**
```python
{
    "dataset.data_path": DATA_ROOT,
    "dataset.data_info_path": DATA_INFO,
    "prune.model": CHECKPOINT,
}
```

**retrain (mandatory data sources):**
```python
{
    "dataset.data_path": DATA_ROOT,
    "dataset.data_info_path": DATA_INFO,
    "train.pruned_model_path": PRUNED_MODEL,
}
```

For local Docker, `DATA_INFO` must be visible inside every train/evaluate/export/prune/retrain container. Use the dataset_convert job from the same results root, or mount/copy the converted `results_dir/data_info` folder into the current run and set `dataset.data_info_path` to that mounted container path. If the host scratch root is mounted at `/results` and the conversion artifacts live under host `scratch/results/<job_id>/results_dir/data_info`, the direct-job container path is `/results/results/<job_id>/results_dir/data_info`. Do not reuse a `/results/<job_id>/...` path from another run root unless that folder is mounted into the current job.

For AutoML train workflows, perform this as a launch preflight before calling `AutoMLRunner.run`: create or materialize the `dataset_convert` output under the current run's `RESULTS_ROOT`, set `dataset.data_info_path` to that current-run container path, and verify `dbinfos_train.pkl`, `infos_train.pkl`, and `infos_val.pkl` are present from the train container's point of view. If a runner is cloned or adapted from a prior AutoML algorithm, update the conversion artifact in the new run root; a stale `CONVERT_JOB_ID` from another results mount is not valid.
## Eval Dataset

Optional. Validation data (val.tar.gz) is separate from training. Used for mAP evaluation.

## Important Parameters

- **train.num_epochs**: Default 80 (much higher than other TAO models). PointPillars needs more epochs for convergence on 3D detection.
- **train.lr**: Learning rate. Default 0.003 (adam_onecycle scheduler).
- **dataset.class_names**: List of 3D object classes. Default 7 classes (KITTI-style). Modify to match your dataset.
- **dataset.data_path**: Path to point cloud data directory.
- **dataset.data_info_path**: Path to data info files from dataset_convert step.
- **dataset.point_cloud_range**: Spatial extent of the point cloud to consider. Must match your sensor configuration.
- **model.dense_head.anchor_generator_config**: Anchor configurations per class. Must be tuned for your object sizes and the point cloud range.

## Multi-GPU / Multi-Node

**Launch method:** `torchrun` (LIGHTNING_EXCLUDED_NETWORK). Uses PyTorch native `DistributedDataParallel` (NOT Lightning Trainer).

| Spec Key | Description | Default |
|----------|-------------|---------|
| `train.num_gpus` | Number of GPUs per node | 1 |
| `train.gpu_ids` | GPU device indices | [0] |
| `train.num_nodes` | Number of nodes | 1 |

- `CUDA_VISIBLE_DEVICES` is explicitly set from `TAO_VISIBLE_DEVICES`
- Uses `nn.parallel.DistributedDataParallel` directly (not Lightning strategy)
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

Minimum 1 GPU(s), recommended 4 GPU(s). 16GB+ (V100 or A100) VRAM per GPU. PointPillars is relatively efficient for 3D detection. The main bottleneck is data I/O for large point cloud datasets.

## Error Patterns

**dataset_convert required**: Training will fail if `dataset.data_info_path` is not populated from a prior `dataset_convert` job. Always run convert first, and verify the train container can see `dbinfos_train.pkl` and `infos_train.pkl` under `dataset.data_info_path`. A common local-Docker failure is a stale `/results/<old_job_id>/...` path from a different results root.

**Point cloud range mismatch**: If point_cloud_range does not match the actual sensor data extent, detections will be poor or empty.

**Epoch numbering**: PointPillars checkpoint epoch numbers may be offset by 1 from status.json reported epochs.

**Checkpoint selection**: PointPillars training emits checkpoints named like `checkpoint_epoch_1.pth`. For evaluation, inference, export, prune, and resume, select the intended checkpoint through the model/job checkpoint resolver and pass that exact file to `evaluate.checkpoint`, `inference.checkpoint`, `export.checkpoint`, `prune.model`, or `train.resume_training_checkpoint_path`. Do not guess by taking the newest `model.pth`; this model does not use that filename.

**Prune/retrain key**: PointPillars prune writes an encrypted `.tlt` artifact. Keep a non-empty `key` in the prune and retrain specs; the packaged templates use the TAO default `tlt_encode`. If `key` is omitted or `null`, the toolkit can still exit with a container success code while logging a passphrase error and creating an empty `pruned_0.1.tlt`. Always verify the pruned model is nonzero before using it for retrain.

**Status files matter**: Some PointPillars failures can be followed by `Execution status: PASS` in the entrypoint footer and a Docker exit code of 0. Check `results_dir/status.json` and the expected artifact before marking an action as passed.

**Local results_dir wiring**: For direct local-Docker specs, set the top-level `results_dir` as well as any action-specific `*.results_dir` field. If only `evaluate.results_dir` is set and the top-level field is left blank, evaluate can try to write under `/opt/nvidia/eval` and then still print the generic PASS footer. Treat that as a failed action unless the expected result directory and status/artifact files exist.

## Spec Param / Parent Model Inference

Model-specific inference mappings belong in this MD file, not in `config.json`. Generated runners should read this section and apply the mappings with SDK helpers before `create_job()`. This mirrors the old microservices `infer_params.py` flow.

Inference mappings from TAO Core `pointpillars.config.json`:

| Action | Spec Field | Inference Function | Meaning |
|---|---|---|---|
| dataset_convert | `results_dir` | `output_dir` | current job results directory |
| evaluate | `evaluate.checkpoint` | `parent_model` | model file inferred from the parent job results folder |
| evaluate | `key` | `key` | encryption key |
| evaluate | `results_dir` | `output_dir` | current job results directory |
| export | `export.checkpoint` | `parent_model` | model file inferred from the parent job results folder |
| export | `export.onnx_file` | `create_onnx_file` | output ONNX path |
| export | `export.save_engine` | `create_engine_file` | output TensorRT engine path |
| export | `key` | `key` | encryption key |
| export | `results_dir` | `output_dir` | current job results directory |
| inference | `inference.checkpoint` | `parent_model` | model file inferred from the parent job results folder |
| inference | `inference.trt_engine` | `parent_model` | model file inferred from the parent job results folder |
| inference | `key` | `key` | encryption key |
| inference | `results_dir` | `output_dir` | current job results directory |
| prune | `key` | `key` | encryption key |
| prune | `prune.model` | `parent_model` | model file inferred from the parent job results folder |
| prune | `results_dir` | `output_dir` | current job results directory |
| retrain | `key` | `key` | encryption key |
| retrain | `results_dir` | `output_dir` | current job results directory |
| retrain | `train.pruned_model_path` | `parent_model` | model file inferred from the parent job results folder |
| train | `key` | `key` | encryption key |
| train | `model.pretrained_model_path` | `ptm_if_no_resume_model` | PTM when no resume checkpoint exists |
| train | `results_dir` | `output_dir` | current job results directory |
| train | `train.resume_training_checkpoint_path` | `resume_model` | model file inferred from the current job results folder |

For `parent_model` or `parent_model_folder`, pass the upstream train/export/AutoML child job id as `parent_job_id`. The SDK lists the parent result folder, filters checkpoint artifacts, and returns the selected model file or folder. Do not add these mappings back to `config.json` and do not patch generated runner scripts to guess checkpoint paths.

## Deployment

- [tao-deploy-pointpillars](references/tao-deploy-pointpillars.md)
