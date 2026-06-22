---
name: tao-train-pose-classification
description: Pose classification using ST-GCN (Spatial Temporal Graph Convolutional Network). Classifies skeleton sequences
  into action categories from pose-keypoint data. Use when training, evaluating, exporting, or running inference for a TAO
  pose-classification model. Trigger phrases include "train pose classification", "skeleton action recognition", "ST-GCN",
  "keypoint sequence classifier".
license: Apache-2.0
compatibility: Requires docker + nvidia-container-toolkit.
metadata:
  version: "0.1.0"
  author: NVIDIA Corporation
allowed-tools: Read Bash
tags:
- pose
- classification
---

# Pose Classification

Pose classification using ST-GCN (Spatial Temporal Graph Convolutional Network). Classifies skeleton sequences into action categories from pose keypoint data.

Typically trained from scratch on skeleton data.

The packaged PyTorch Pose Classification CLI supports `dataset_convert`, `train`, `evaluate`, `export`, and `inference`. `dataset_convert` is conditional: run it only when the input is raw DeepStream BodyPose JSON. If the dataset is already converted to TAO-ready `.npy` / `.pkl` files, start directly with `train` on those files and mark dataset conversion as `not run: preconverted dataset provided` in validation reports. This model does not expose deploy, prune, quantize, or standalone retrain actions. Resume/retrain behavior uses `pose_classification train -e ...` with `train.resume_training_checkpoint_path` populated.

## Dataclass Schemas

Generated TAO Core schemas are packaged in `schemas/<action>.schema.json`, with `schemas/manifest.json` listing available actions. Each generated schema also emits `references/spec_template_<action>.yaml` from the schema top-level `default` field. AutoML enablement is declared at the model layer in `references/skill_info.yaml` via `automl_enabled`. Runnable AutoML still requires `schemas/train.schema.json` and `references/spec_template_train.yaml` to exist and parse. Use the packaged train schema for `automl_default_parameters`, `automl_disabled_parameters`, defaults, min/max bounds, enums, option weights, math conditions, dependencies, and popular parameters. Do not expect `~/tao-core` at runtime; maintainers regenerate schemas/templates before packaging the skill bank.

## Train Action Policy

This model is AutoML-enabled at the model layer. Before handling any train-stage request, read `references/skill_info.yaml` and resolve the run override from either an explicit `automl_policy` value or the user's workflow request. Use `automl_policy: on` by default and only expose `on` / `off` in new launch prompts. Treat phrases like "turn off AutoML", "disable AutoML", "no HPO", or "plain training" as `automl_policy: off` for this run only. When `automl_policy: on`, `automl_enabled: true`, and both `schemas/train.schema.json` and `references/spec_template_train.yaml` are packaged, route the train action through `tao-skill-bank:tao-run-automl` by default with this model's `skill_dir`. Preserve workflow/application overrides for datasets, specs, output directories, GPU/platform settings, parent checkpoints, and `automl_policy`. Use direct model training only when `automl_policy: off` or the packaged train schema/template is missing; in the missing-schema case, report that AutoML is enabled but not runnable for this model until schemas are generated.

Non-train actions such as `evaluate`, `inference`, `export`, and deploy flows stay in this model skill. The per-run `automl_policy` override does not change model metadata.

## Training Requirements

- **Dataset type:** pose_classification
- **Formats:** default
- **Monitoring metric:** val_loss

### Per-Action Dataset Requirements

| Action | Spec Key | Source | Files | List? |
|---|---|---|---|---|
| dataset_convert (optional) | dataset_convert.data | id | DeepStream BodyPose JSON | No |
| evaluate | evaluate.test_dataset.data_path | train_datasets | val_data.npy | No |
| evaluate | evaluate.test_dataset.label_path | train_datasets | val_label.pkl | No |
| inference | inference.test_dataset.data_path | train_datasets | test_data.npy | No |
| train | dataset.train_dataset.data_path | train_datasets | train_data.npy | No |
| train | dataset.train_dataset.label_path | train_datasets | train_label.pkl | No |
| train | dataset.val_dataset.data_path | train_datasets | val_data.npy | No |
| train | dataset.val_dataset.label_path | train_datasets | val_label.pkl | No |

### Typical Spec Overrides

Data source overrides are **mandatory for every action being run** — the agent MUST construct data source paths from the Per-Action Dataset Requirements table above and include them in `spec_overrides`. Do not run `dataset_convert` when the supplied dataset is already converted to `.npy` / `.pkl` files.

```python
S3_TRAIN = "s3://bucket/data/purpose_built_models_pose_classification_train/nvidia"
CHECKPOINT = "/results/{train_job_id}/results_dir/model_epoch_000_step_00007.pth"
```

**dataset_convert (optional; raw DeepStream BodyPose JSON only):**
```python
{
    "dataset_convert.data": "s3://bucket/data/<deepstream-bodypose-output>.json",
}
```

**train (mandatory data sources):**
```python
{
    "train.num_epochs": 30,
    "train.checkpoint_interval": 10,
    "train.validation_interval": 10,
    "train.num_gpus": 1,
    "wandb.enable": False,
    "dataset.num_classes": 6,
    "dataset.label_map": {
        "class_0": 0,
        "class_1": 1,
        "class_2": 2,
        "class_3": 3,
        "class_4": 4,
        "class_5": 5,
    },
    "model.graph_layout": "nvidia",
    "dataset.train_dataset.data_path": f"{S3_TRAIN}/train_data.npy",
    "dataset.train_dataset.label_path": f"{S3_TRAIN}/train_label.pkl",
    "dataset.val_dataset.data_path": f"{S3_TRAIN}/val_data.npy",
    "dataset.val_dataset.label_path": f"{S3_TRAIN}/val_label.pkl",
}
```

**resume train (mandatory checkpoint):**
```python
{
    "train.num_epochs": 31,
    "train.resume_training_checkpoint_path": CHECKPOINT,
    "dataset.train_dataset.data_path": f"{S3_TRAIN}/train_data.npy",
    "dataset.train_dataset.label_path": f"{S3_TRAIN}/train_label.pkl",
    "dataset.val_dataset.data_path": f"{S3_TRAIN}/val_data.npy",
    "dataset.val_dataset.label_path": f"{S3_TRAIN}/val_label.pkl",
}
```

**evaluate (mandatory data sources):**
```python
{
    "evaluate.test_dataset.data_path": f"{S3_TRAIN}/val_data.npy",
    "evaluate.test_dataset.label_path": f"{S3_TRAIN}/val_label.pkl",
    "evaluate.checkpoint": CHECKPOINT,
}
```

**export (mandatory checkpoint and output):**
```python
{
    "export.checkpoint": CHECKPOINT,
    "export.onnx_file": "/results/{export_job_id}/results_dir/pose_classification.onnx",
}
```

**inference (mandatory data sources):**
```python
{
    "inference.test_dataset.data_path": f"{S3_TRAIN}/test_data.npy",
    "inference.test_dataset.label_path": f"{S3_TRAIN}/test_label.pkl",
    "inference.checkpoint": CHECKPOINT,
    "inference.output_file": "/results/pose_classification_inference.txt",
}
```
## Dataset Convert

Dataset conversion is optional for Pose Classification. Run `pose_classification dataset_convert` only when the user supplies raw DeepStream BodyPose JSON. For the common S3 validation dataset, the data is already converted to `train_data.npy`, `train_label.pkl`, `val_data.npy`, `val_label.pkl`, `test_data.npy`, and `test_label.pkl`; use those files directly for train/evaluate/inference/export flows and do not synthesize fake BodyPose JSON.

## Eval Dataset

Optional. Validation data is provided alongside training as val_data.npy / val_label.pkl. TAO training emits `val_loss` as the TensorBoard validation scalar for this model; use `val_loss` with minimize direction for AutoML selection unless a custom evaluation hook supplies a different metric.

## Important Parameters

- **dataset.num_classes**: Number of pose action classes. Default 6.
- **model.graph_layout**: Skeleton graph layout. Options: nvidia, openpose. Determines joint connectivity.
- **model.graph_strategy**: Graph partitioning strategy for GCN.
- **train.optim.lr**: Learning rate. Default 0.1 (SGD). Higher than vision models due to graph convolution properties.
- **model.dropout**: Dropout rate for regularization.

## Multi-GPU / Multi-Node

**Launch method:** Lightning-managed (single `python` process, Lightning spawns workers).

| Spec Key | Description | Default |
|----------|-------------|---------|
| `train.num_gpus` | Number of GPUs | 1 |
| `train.gpu_ids` | GPU device indices | [0] |

- Strategy: `auto` (Lightning picks best strategy automatically)
- No explicit `num_nodes` or `distributed_strategy` config — single-node only
- Lightweight model, single GPU typically sufficient

## Hardware

Minimum 1 GPU(s), recommended 1 GPU(s). 8GB+ VRAM per GPU. Pose classification is very lightweight — skeleton data is small. Single GPU is sufficient.

## Error Patterns

**Graph layout mismatch**: Ensure model.graph_layout matches the skeleton format in your .npy data files.

**Label shape mismatch**: train_label.pkl class indices must be in range [0, num_classes).

**Missing label map**: The training dataloader expects `dataset.label_map` to be a dictionary. If the dataset only supplies numeric class IDs, set a synthetic contiguous map such as `class_0: 0` through `class_5: 5` for the six-class NVIDIA sample data.

**Checkpoint handoff**: After AutoML/train, use the checkpoint resolver to select the intended saved `.pth` checkpoint under the parent result folder, such as `model_epoch_000_step_00007.pth`, and pass that exact file as `evaluate.checkpoint`, `export.checkpoint`, `inference.checkpoint`, or `train.resume_training_checkpoint_path`. `pc_model_latest.pth` is a latest-checkpoint symlink; use it only when the user explicitly asks for latest rather than a specific/best checkpoint. Keep the same `dataset.num_classes`, `dataset.label_map`, and `model.graph_layout` overrides for downstream actions.

**Dataset conversion source**: `dataset_convert` expects the raw JSON output from the DeepStream BodyPose app. The common NVIDIA sample S3 folder is already converted to `train_data.npy`, `train_label.pkl`, `val_data.npy`, `val_label.pkl`, `test_data.npy`, and `test_label.pkl`; skip conversion and start from the converted files when those are present.

**Action-specific dataset paths**: The evaluate and inference templates also contain the training `dataset.train_dataset` and `dataset.val_dataset` blocks. For evaluate, populate `evaluate.test_dataset.data_path` and `evaluate.test_dataset.label_path`. For inference, populate `inference.test_dataset.data_path` and set `inference.output_file`; do not stop after replacing the first `data_path` or `label_path` in the file.

**Output files**: Export needs an explicit `export.onnx_file` path. Inference must set `inference.output_file` to a writable file path; the packaged template default is an empty string, and the current PyTorch inference code opens that value directly.

## Spec Param / Parent Model Inference

Model-specific inference mappings belong in this MD file, not in `config.json`. Generated runners should read this section and apply the mappings with SDK helpers before `create_job()`. This mirrors the old microservices `infer_params.py` flow.

Inference mappings from TAO Core `pose_classification.config.json`:

| Action | Spec Field | Inference Function | Meaning |
|---|---|---|---|
| dataset_convert | `dataset_convert.results_dir` | `output_dir` | current job results directory |
| evaluate | `encryption_key` | `key` | encryption key |
| evaluate | `evaluate.checkpoint` | `parent_model` | model file inferred from the parent job results folder |
| evaluate | `results_dir` | `output_dir` | current job results directory |
| export | `encryption_key` | `key` | encryption key |
| export | `export.checkpoint` | `parent_model` | model file inferred from the parent job results folder |
| export | `export.onnx_file` | `create_onnx_file` | output ONNX path |
| export | `results_dir` | `output_dir` | current job results directory |
| inference | `encryption_key` | `key` | encryption key |
| inference | `inference.checkpoint` | `parent_model` | model file inferred from the parent job results folder |
| inference | `inference.output_file` | `create_inference_result_file_pose` | pose inference result file |
| inference | `results_dir` | `output_dir` | current job results directory |
| train | `encryption_key` | `key` | encryption key |
| train | `model.pretrained_model_path` | `ptm_if_no_resume_model` | PTM when no resume checkpoint exists |
| train | `results_dir` | `output_dir` | current job results directory |
| train | `train.resume_training_checkpoint_path` | `resume_model` | model file inferred from the current job results folder |

For `parent_model` or `parent_model_folder`, pass the upstream train/export/AutoML child job id as `parent_job_id`. The SDK lists the parent result folder, filters checkpoint artifacts, and returns the selected model file or folder. Do not add these mappings back to `config.json` and do not patch generated runner scripts to guess checkpoint paths.
