---
name: tao-train-deformable-detr
description: Deformable DETR for 2D object detection. Uses deformable attention for efficient multi-scale feature processing,
  lighter than DINO with competitive accuracy. Use when training, evaluating, exporting, quantizing, or running inference for
  a TAO Deformable-DETR model. Trigger phrases include "train deformable-detr", "Deformable DETR object detection",
  "lightweight DETR detector".
license: Apache-2.0
compatibility: Requires docker + nvidia-container-toolkit.
metadata:
  version: "0.1.0"
  author: NVIDIA Corporation
allowed-tools: Read Bash
tags:
- object
- detection
---

# Deformable DETR

Deformable DETR for 2D object detection. Uses deformable attention for efficient multi-scale feature processing. Lighter than DINO with competitive accuracy.

Uses pretrained weights. Set `model.pretrained_backbone_path` for backbone-only
loading or `train.pretrained_model_path` for full model initialization.

Supported parent model actions are `train`, `evaluate`, `inference`, `export`, and `quantize`. The PyT model container does not support a native `gen_trt_engine` subtask for this network. The `gen_trt_engine` action declared in `references/skill_info.yaml` must run with the TAO Deploy container. Deploy spec templates live in this skill's `references/` folder with the `spec_template_deploy_*.yaml` prefix.

## Dataclass Schemas

Generated TAO Core schemas are packaged in `schemas/<action>.schema.json`, with `schemas/manifest.json` listing available actions. Each generated schema also emits `references/spec_template_<action>.yaml` from the schema top-level `default` field. AutoML enablement is declared at the model layer in `references/skill_info.yaml` via `automl_enabled`. Runnable AutoML still requires `schemas/train.schema.json` and `references/spec_template_train.yaml` to exist and parse. Use the packaged train schema for `automl_default_parameters`, `automl_disabled_parameters`, defaults, min/max bounds, enums, option weights, math conditions, dependencies, and popular parameters. Do not expect `~/tao-core` at runtime; maintainers regenerate schemas/templates before packaging the skill bank.

## Train Action Policy

This model is AutoML-enabled at the model layer. Before handling any train-stage request, read `references/skill_info.yaml` and resolve the run override from either an explicit `automl_policy` value or the user's workflow request. Use `automl_policy: on` by default and only expose `on` / `off` in new launch prompts. Treat phrases like "turn off AutoML", "disable AutoML", "no HPO", or "plain training" as `automl_policy: off` for this run only. When `automl_policy: on`, `automl_enabled: true`, and both `schemas/train.schema.json` and `references/spec_template_train.yaml` are packaged, route the train action through `tao-skill-bank:tao-run-automl` by default with this model's `skill_dir`. Preserve workflow/application overrides for datasets, specs, output directories, GPU/platform settings, parent checkpoints, and `automl_policy`. Use direct model training only when `automl_policy: off` or the packaged train schema/template is missing; in the missing-schema case, report that AutoML is enabled but not runnable for this model until schemas are generated.

Non-train actions such as `evaluate`, `inference`, `export`, and deploy flows stay in this model skill. The per-run `automl_policy` override does not change model metadata.

## Training Requirements

- **Dataset type:** object_detection
- **Formats:** coco, coco_raw
- **Monitoring metric:** val_mAP50 for AP50; `val_mAP` for COCO/paper-style benchmark comparisons.

### Per-Action Dataset Requirements

| Action | Spec Key | Source | Files | List? |
|---|---|---|---|---|
| evaluate | dataset.test_data_sources.image_dir | eval_dataset | images.tar.gz | No |
| evaluate | dataset.test_data_sources.json_file | eval_dataset | annotations.json | No |
| export | dataset.train_data_sources | train_datasets | image_dir: images.tar.gz, json_file: annotations.json | Yes |
| export | dataset.val_data_sources | train_datasets | image_dir: images.tar.gz, json_file: annotations.json | Yes |
| inference | dataset.infer_data_sources.image_dir | inference_dataset | images.tar.gz | Yes |
| inference | dataset.infer_data_sources.classmap | inference_dataset | label_map.txt | No |
| quantize | dataset.train_data_sources | train_datasets | image_dir: images.tar.gz, json_file: annotations.json | Yes |
| quantize | dataset.val_data_sources | train_datasets | image_dir: images.tar.gz, json_file: annotations.json | Yes |
| quantize | dataset.quant_calibration_data_sources | train_datasets | image_dir: images.tar.gz, json_file: annotations.json | No |
| train | dataset.train_data_sources | train_datasets | image_dir: images.tar.gz, json_file: annotations.json | Yes |
| train | dataset.val_data_sources | train_datasets | image_dir: images.tar.gz, json_file: annotations.json | Yes |

### Typical Spec Overrides

Data source overrides are **mandatory for every action** — the agent MUST construct data source paths from the Per-Action Dataset Requirements table above and include them in `spec_overrides`.

```python
S3_TRAIN = "s3://bucket/data/train"
S3_EVAL = "s3://bucket/data/eval"
```

**train (mandatory data sources):**
```python
{
    "train.num_epochs": 10,
    "train.checkpoint_interval": 10,
    "train.validation_interval": 10,
    "train.num_gpus": 1,
    "train.gpu_ids": [0],
    "dataset.num_classes": "<object classes> + 1",
    "dataset.eval_class_ids": [1, 2, "..."],
    "dataset.train_data_sources": [{"image_dir": f"{S3_TRAIN}/images.tar.gz", "json_file": f"{S3_TRAIN}/annotations.json"}],
    "dataset.val_data_sources": [{"image_dir": f"{S3_TRAIN}/images.tar.gz", "json_file": f"{S3_TRAIN}/annotations.json"}],
}
```

**evaluate (mandatory data sources):**
```python
{
    "dataset.num_classes": "<object classes> + 1",
    "dataset.eval_class_ids": [1, 2, "..."],
    "dataset.test_data_sources.image_dir": f"{S3_EVAL}/images.tar.gz",
    "dataset.test_data_sources.json_file": f"{S3_EVAL}/annotations.json",
}
```

If the train or AutoML run changed architecture-affecting fields such as
`model.enc_layers`, `model.dec_layers`, `model.num_queries`, or
`model.num_select`, carry the same values into evaluate, export, inference, and
deploy actions with the selected checkpoint. In addition to the fields above,
carry `model.num_feature_levels`, `model.dim_feedforward`, input image
dimensions, and dataset class metadata when they were changed. Loading a
checkpoint into the default architecture can fail with tensor shape mismatches,
especially when smoke-test runs shrink the transformer for speed.

**export (mandatory data sources):**
```python
{
    "dataset.num_classes": "<object classes> + 1",
    "dataset.eval_class_ids": [1, 2, "..."],
    "dataset.train_data_sources": [{"image_dir": f"{S3_TRAIN}/images.tar.gz", "json_file": f"{S3_TRAIN}/annotations.json"}],
    "dataset.val_data_sources": [{"image_dir": f"{S3_TRAIN}/images.tar.gz", "json_file": f"{S3_TRAIN}/annotations.json"}],
}
```

**TensorRT engine generation:**

Use the deploy spec templates after `export`. Do not call `deformable_detr
gen_trt_engine` from the parent PyT model container; that CLI advertises
`convert`, `evaluate`, `export`, `inference`, `quantize`, `train`, and
`default_specs`, but not `gen_trt_engine`. The model action metadata selects the
TAO Deploy container for engine generation.

Deploy engine generation needs the exported ONNX file as input and creates the
engine at `gen_trt_engine.trt_engine`.

```python
{
    "gen_trt_engine.tensorrt.data_type": "FP16",
    "dataset.num_classes": "<object classes> + 1",
    "gen_trt_engine.tensorrt.calibration.cal_image_dir": [f"{S3_TRAIN}/images.tar.gz"],
}
```

**inference (mandatory data sources):**
```python
{
    "dataset.num_classes": "<object classes> + 1",
    "dataset.infer_data_sources.image_dir": [f"{S3_EVAL}/images.tar.gz"],
    "dataset.infer_data_sources.classmap": f"{S3_EVAL}/label_map.txt",
}
```

**quantize (mandatory data sources):**
```python
{
    "dataset.train_data_sources": [{"image_dir": f"{S3_TRAIN}/images.tar.gz", "json_file": f"{S3_TRAIN}/annotations.json"}],
    "dataset.val_data_sources": [{"image_dir": f"{S3_TRAIN}/images.tar.gz", "json_file": f"{S3_TRAIN}/annotations.json"}],
    "dataset.quant_calibration_data_sources": {"image_dir": f"{S3_TRAIN}/images.tar.gz", "json_file": f"{S3_TRAIN}/annotations.json"},
}
```
## Eval Dataset

Optional. If provided, validation mAP is computed at each checkpoint interval.

## Checkpoint Handling

Training emits epoch-and-step checkpoints using the pattern
`model_epoch_<epoch>_step_<step>.pth`, plus a `dd_model_latest.pth` symlink. For
dependent actions, use the model-specific or SDK-provided checkpoint resolver to
select the intended artifact. Evaluation, inference, export, and quantize should
receive the selected exact checkpoint path, not the `dd_model_latest.pth` symlink,
unless the user explicitly asked for latest. Resume/retrain should set
`train.resume_training_checkpoint_path` to the exact checkpoint being resumed
from.

## Important Parameters

- **dataset.num_classes**: Number of object classes plus the background class. Default 91 (COCO). Must match annotations.
- **dataset.eval_class_ids**: Foreground category ids to include in COCO metrics. Set this to every object category id in custom datasets; the template default evaluates class id 1 only.
- **model.backbone**: Default resnet_50. Supported: resnet_50, gcvit_tiny, gcvit_small, gcvit_base, gcvit_large, gcvit_large_384 (more limited than DINO).
- **train.optim.lr**: Learning rate. Default 2e-4 (AdamW). lr_backbone is 2e-5.
- **train.optim.lr_steps**: MultiStep LR schedule. Default [40]. For short runs, set to match ~80% of total epochs.
- **model.num_queries**: Number of object queries. Default 300. Valid range 100-900.
- **model.dropout_ratio**: Dropout in transformer layers. Default 0.3 (higher than DINO's 0.0). Reduce for large datasets, increase for small datasets.
- **model.dim_feedforward**: FFN hidden dim. Default 1024 (vs DINO's 2048). Increasing improves capacity but costs memory.

## Multi-GPU / Multi-Node

**Launch method:** Lightning-managed (single `python` process, Lightning spawns workers).

| Spec Key | Description | Default |
|----------|-------------|---------|
| `train.num_gpus` | Number of GPUs | 1 |
| `train.gpu_ids` | GPU device indices | [0] |
| `train.num_nodes` | Number of nodes | 1 |
| `train.distributed_strategy` | `ddp` or `fsdp` | `ddp` |

Same DDP/FSDP behavior as DINO. Multi-node requires `WORLD_SIZE`, `NODE_RANK`, `MASTER_ADDR`, `MASTER_PORT` env vars set by orchestrator.

When increasing `train.num_gpus`, also set `train.gpu_ids` to the same visible
device range. For example, an 8-GPU single-node Slurm run must include both
`"train.num_gpus": 8` and `"train.gpu_ids": [0, 1, 2, 3, 4, 5, 6, 7]`.

## Export / TRT Defaults

- Export input: 640x640, opset 17
- TRT data types: FP32, FP16, INT8
- TRT workspace: 1024 MB
- TRT max_batch_size: 1

## Hardware

Minimum 1 GPU(s), recommended 4 GPU(s). 16GB+ (V100 or A100) VRAM per GPU. Slightly lighter than DINO due to smaller FFN. batch_size=4 fits on most 16GB+ GPUs.

## Error Patterns

**CUDA out of memory**: Reduce batch_size (4 -> 2 -> 1).

**num_select must be < num_queries * num_classes**: Same constraint as DINO.

**return_interm_indices length must match num_feature_levels**: Default [1,2,3,4] with num_feature_levels=4.

**Dataset size smaller than total batch size**: Reduce batch_size or num_gpus.

**AutoML metric extraction**: Deformable DETR emits detection metrics in structured training status and logs. For COCO/paper-style benchmark comparisons, optimize `val_mAP` with `direction: maximize`; for explicit AP50 workflows, optimize `val_mAP50`. Prefer `results_dir/train/status.json` or AutoML result state before parsing raw logs. Do not optimize `val_loss` for default detection model invocations.

## Spec Param / Parent Model Inference

Model-specific inference mappings belong in this MD file, not in `config.json`. Generated runners should read this section and apply the mappings with SDK helpers before `create_job()`. This mirrors the old microservices `infer_params.py` flow.

Inference mappings from TAO Core `deformable_detr.config.json`:

| Action | Spec Field | Inference Function | Meaning |
|---|---|---|---|
| evaluate | `encryption_key` | `key` | encryption key |
| evaluate | `evaluate.checkpoint` | `parent_model` | model file inferred from the parent job results folder |
| evaluate | `evaluate.trt_engine` | `parent_model` | model file inferred from the parent job results folder |
| evaluate | `results_dir` | `output_dir` | current job results directory |
| export | `encryption_key` | `key` | encryption key |
| export | `export.checkpoint` | `parent_model` | model file inferred from the parent job results folder |
| export | `export.onnx_file` | `create_onnx_file` | output ONNX path |
| export | `results_dir` | `output_dir` | current job results directory |
| inference | `encryption_key` | `key` | encryption key |
| inference | `inference.checkpoint` | `parent_model` | model file inferred from the parent job results folder |
| inference | `inference.trt_engine` | `parent_model` | model file inferred from the parent job results folder |
| inference | `results_dir` | `output_dir` | current job results directory |
| quantize | `encryption_key` | `key` | encryption key |
| quantize | `quantize.model_path` | `parent_model` | model file inferred from the parent job results folder |
| quantize | `results_dir` | `output_dir` | current job results directory |
| train | `encryption_key` | `key` | encryption key |
| train | `model.pretrained_backbone_path` | `ptm_if_no_resume_model` | PTM when no resume checkpoint exists |
| train | `results_dir` | `output_dir` | current job results directory |
| train | `train.pretrained_model_path` | `ptm_if_no_resume_model` | full model PTM when no resume checkpoint exists |
| train | `train.resume_training_checkpoint_path` | `resume_model` | model file inferred from the current job results folder |

For `parent_model` or `parent_model_folder`, pass the upstream train/export/AutoML child job id as `parent_job_id`. The SDK lists the parent result folder, filters checkpoint artifacts, and returns the selected model file or folder. Do not add these mappings back to `config.json` and do not patch generated runner scripts to guess checkpoint paths.

## Deployment

- [tao-deploy-deformable-detr](references/tao-deploy-deformable-detr.md)
