---
name: tao-train-optical-inspection
description: Optical Inspection for defect detection using Siamese networks. Compares image pairs to detect manufacturing
  defects, anomalies, or quality issues. Use when training, evaluating, exporting, or running inference for a TAO Optical
  Inspection model on AOI / quality-control data. Trigger phrases include "train optical inspection", "AOI defect
  detection", "Siamese defect classifier", "PCB / manufacturing inspection".
license: Apache-2.0
compatibility: Requires docker + nvidia-container-toolkit.
metadata:
  version: "0.1.0"
  author: NVIDIA Corporation
allowed-tools: Read Bash
tags:
- defect
- detection
---

# Optical Inspection

Optical inspection for defect detection using Siamese networks. Compares image pairs to detect manufacturing defects, anomalies, or quality issues.

Set train.pretrained_model_path for pretrained Siamese weights.

For TAO Deploy TensorRT actions (`gen_trt_engine`, TensorRT `evaluate`, and TensorRT `inference`), read `references/tao-deploy-optical-inspection.md` first. The parent PyT container does not expose `optical_inspection gen_trt_engine`; TensorRT engine generation is deploy-only. Deploy spec templates live in this skill's `references/` folder with the `spec_template_deploy_*.yaml` prefix.

## Dataclass Schemas

Generated TAO Core schemas are packaged in `schemas/<action>.schema.json`, with `schemas/manifest.json` listing available actions. Each generated schema also emits `references/spec_template_<action>.yaml` from the schema top-level `default` field. AutoML enablement is declared at the model layer in `references/skill_info.yaml` via `automl_enabled`. Runnable AutoML still requires `schemas/train.schema.json` and `references/spec_template_train.yaml` to exist and parse. Use the packaged train schema for `automl_default_parameters`, `automl_disabled_parameters`, defaults, min/max bounds, enums, option weights, math conditions, dependencies, and popular parameters. Do not expect `~/tao-core` at runtime; maintainers regenerate schemas/templates before packaging the skill bank.

## Train Action Policy

This model is AutoML-enabled at the model layer. Before handling any train-stage request, read `references/skill_info.yaml` and resolve the run override from either an explicit `automl_policy` value or the user's workflow request. Use `automl_policy: on` by default and only expose `on` / `off` in new launch prompts. Treat phrases like "turn off AutoML", "disable AutoML", "no HPO", or "plain training" as `automl_policy: off` for this run only. When `automl_policy: on`, `automl_enabled: true`, and both `schemas/train.schema.json` and `references/spec_template_train.yaml` are packaged, route the train action through `tao-skill-bank:tao-run-automl` by default with this model's `skill_dir`. Preserve workflow/application overrides for datasets, specs, output directories, GPU/platform settings, parent checkpoints, and `automl_policy`. Use direct model training only when `automl_policy: off` or the packaged train schema/template is missing; in the missing-schema case, report that AutoML is enabled but not runnable for this model until schemas are generated.

Non-train actions such as `evaluate`, `inference`, `export`, and deploy flows stay in this model skill. The per-run `automl_policy` override does not change model metadata.

## Training Requirements

- **Dataset type:** optical_inspection
- **Formats:** default
- **Monitoring metric:** val_acc

### Per-Action Dataset Requirements

| Action | Spec Key | Source | Files | List? |
|---|---|---|---|---|
| evaluate | dataset.test_dataset.images_dir | eval_dataset | images.tar.gz | No |
| evaluate | dataset.test_dataset.csv_path | eval_dataset | dataset.csv | No |
| inference | dataset.infer_dataset.images_dir | inference_dataset | images.tar.gz | No |
| inference | dataset.infer_dataset.csv_path | inference_dataset | dataset.csv | No |
| train | dataset.train_dataset.images_dir | train_datasets | images.tar.gz | No |
| train | dataset.train_dataset.csv_path | train_datasets | dataset.csv | No |
| train | dataset.validation_dataset.images_dir | eval_dataset | images.tar.gz | No |
| train | dataset.validation_dataset.csv_path | eval_dataset | dataset.csv | No |
| train | dataset.test_dataset.images_dir | eval_dataset | images.tar.gz | No |
| train | dataset.test_dataset.csv_path | eval_dataset | dataset.csv | No |

### Typical Spec Overrides

Data source overrides are **mandatory for every action** — the agent MUST construct data source paths from the Per-Action Dataset Requirements table above and include them in `spec_overrides`.

```python
S3_TRAIN = "s3://bucket/data/train"
S3_EVAL = "s3://bucket/data/eval"
S3_INFERENCE = "s3://bucket/data/inference"
```

**train (mandatory data sources):**
```python
{
    "train.num_epochs": 30,
    "train.checkpoint_interval": 10,
    "train.validation_interval": 10,
    "train.num_gpus": 1,
    "dataset.batch_size": 8,
    "dataset.train_dataset.images_dir": f"{S3_TRAIN}/images.tar.gz",
    "dataset.train_dataset.csv_path": f"{S3_TRAIN}/dataset.csv",
    "dataset.validation_dataset.images_dir": f"{S3_EVAL}/images.tar.gz",
    "dataset.validation_dataset.csv_path": f"{S3_EVAL}/dataset.csv",
    "dataset.test_dataset.images_dir": f"{S3_EVAL}/images.tar.gz",
    "dataset.test_dataset.csv_path": f"{S3_EVAL}/dataset.csv",
}
```

**evaluate (mandatory data sources):**
```python
{
    "evaluate.checkpoint": "<selected train/AutoML checkpoint>",
    "dataset.test_dataset.images_dir": f"{S3_EVAL}/images.tar.gz",
    "dataset.test_dataset.csv_path": f"{S3_EVAL}/dataset.csv",
}
```

Use the workflow's checkpoint resolver for downstream actions instead of guessing a filename. For Optical Inspection smoke runs, AutoML may produce `model_epoch_000_step_00006.pth`; resume can then produce `model_epoch_001_step_00012.pth`. Best-checkpoint actions should use the AutoML best child job's selected checkpoint, epoch-specific actions should pass the exact epoch/step checkpoint requested, and only explicit "latest" requests should resolve to the latest checkpoint.

**export:**
```python
{
    "export.checkpoint": "<selected train/AutoML checkpoint>",
    "export.onnx_file": "/results/optical_inspection.onnx",
    "export.input_width": 128,
    "export.input_height": 512,
    "export.batch_size": 1,
}
```

**inference (mandatory data sources):**
```python
{
    "inference.checkpoint": "<selected train/AutoML checkpoint>",
    "dataset.infer_dataset.images_dir": f"{S3_INFERENCE}/images.tar.gz",
    "dataset.infer_dataset.csv_path": f"{S3_INFERENCE}/dataset.csv",
}
```

## Dataset Convert

Dataset conversion is optional for Optical Inspection. If the dataset is already in TAO-ready Optical Inspection format, start directly from the `images.tar.gz` plus `dataset.csv` splits and run `train`, `evaluate`, `inference`, and downstream checkpoint/export/deploy actions on that converted data.

The PyT container exposes `optical_inspection dataset_convert`, but this model skill does not package a `dataset_convert` action/template. The converter expects the raw Factory PCB layout (`root_dataset_dir`, train/val/all PCB directories, `golden_csv_dir`, `project_name`, and `bot_top`). The S3 validation bucket currently contains preconverted Optical Inspection `images.tar.gz` plus `dataset.csv` splits, not the raw PCB/golden CSV source. Do not synthesize a fake PCB dataset. In model validation reports, mark dataset conversion as `not run: preconverted dataset provided` rather than failed or blocked when only converted data is available.

When using the preconverted S3 validation tarballs locally, verify the extracted directory before writing specs. The tarballs may unpack an `images/` wrapper directory; point `dataset.*.images_dir` at the inner directory that contains `golden/` and the board/image folders referenced by `dataset.csv`, for example `.../<split>/images/images`, not the outer wrapper.

## Eval Dataset

Optional. Eval dataset uses same format (images + CSV).

## Important Parameters

- **model.model_type**: Siamese variant. Options include Siamese, Siamese_3.
- **model.model_backbone**: Default custom.
- **model.embedding_vectors**: Number of embedding dimensions. Default 5.
- **train.optim.lr**: Learning rate. Default 5e-4.
- **dataset.batch_size**: Training batch size. Must be greater than 1; use `2` or higher for minimal smoke runs.
- **dataset.num_input**: Number of input images per comparison.
- **dataset.input_map**: Mapping of input channels / image pairs.

## Multi-GPU / Multi-Node

**Launch method:** Lightning-managed (single `python` process, Lightning spawns workers).

| Spec Key | Description | Default |
|----------|-------------|---------|
| `train.num_gpus` | Number of GPUs | 1 |
| `train.gpu_ids` | GPU device indices | [0] |

- Strategy: `auto` (Lightning picks best strategy automatically)
- No explicit `num_nodes` or `distributed_strategy` config — single-node only
- Lightweight Siamese network, single GPU typically sufficient

## Hardware

Minimum 1 GPU(s), recommended 1 GPU(s). 8GB+ VRAM per GPU. Siamese networks for inspection are lightweight. Single GPU sufficient.

## Error Patterns

**CSV format error**: Ensure dataset.csv has the correct column format for image pair paths and labels.

**Extracted image root mismatch**: If train, evaluate, or inference cannot find paths from `dataset.csv`, inspect the extracted `images.tar.gz` tree. The TAO-ready root must contain `golden/` plus the board folders referenced in the CSV. For validation S3 tarballs this can be one level below the extraction target, such as `images/images`.

**Training batch size assertion**: The Optical Inspection dataloader rejects
`dataset.batch_size: 1` for train. Keep the template default of 8 for normal
runs, or set `dataset.batch_size: 2` for minimal AutoML smoke validation.

**PyTorch checkpoint load failure on downstream actions**: For checkpoints
produced by the same trusted TAO train/AutoML workflow, set
`TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` for evaluate, inference, export, and
resume jobs if the current PyTorch default blocks loading the full checkpoint.
Do not use this env var for untrusted checkpoints.

## Spec Param / Parent Model Inference

Model-specific inference mappings belong in this MD file, not in `config.json`. Generated runners should read this section and apply the mappings with SDK helpers before `create_job()`. This mirrors the old microservices `infer_params.py` flow.

Inference mappings from TAO Core `optical_inspection.config.json`:

| Action | Spec Field | Inference Function | Meaning |
|---|---|---|---|
| evaluate | `encryption_key` | `key` | encryption key |
| evaluate | `evaluate.checkpoint` | `parent_model` | model file inferred from the parent job results folder |
| evaluate | `results_dir` | `output_dir` | current job results directory |
| export | `encryption_key` | `key` | encryption key |
| export | `export.checkpoint` | `parent_model` | model file inferred from the parent job results folder |
| export | `export.onnx_file` | `create_onnx_file` | output ONNX path |
| export | `results_dir` | `output_dir` | current job results directory |
| inference | `encryption_key` | `key` | encryption key |
| inference | `inference.checkpoint` | `parent_model` | model file inferred from the parent job results folder |
| inference | `inference.trt_engine` | `parent_model` | model file inferred from the parent job results folder |
| inference | `results_dir` | `output_dir` | current job results directory |
| train | `encryption_key` | `key` | encryption key |
| train | `results_dir` | `output_dir` | current job results directory |
| train | `train.pretrained_model_path` | `ptm_if_no_resume_model` | PTM when no resume checkpoint exists |
| train | `train.resume_training_checkpoint_path` | `resume_model` | model file inferred from the current job results folder |

For `parent_model` or `parent_model_folder`, pass the upstream train/export/AutoML child job id as `parent_job_id`. The SDK lists the parent result folder, filters checkpoint artifacts, and returns the selected model file or folder. Do not add these mappings back to `config.json` and do not patch generated runner scripts to guess checkpoint paths.

## Deployment

- [tao-deploy-optical-inspection](references/tao-deploy-optical-inspection.md)
