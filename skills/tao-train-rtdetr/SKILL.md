---
name: tao-train-rtdetr
description: RT-DETR (Real-Time DEtection TRansformer) for 2D object detection. Designed for real-time inference with
  competitive accuracy and supports distillation and quantization for deployment optimization. Use when training, evaluating,
  distilling, quantizing, exporting, or running inference for a TAO RT-DETR model. Trigger phrases include "train RT-DETR",
  "real-time DETR", "low-latency object detection", "RT-DETR distillation / quantization".
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

# RT-DETR

RT-DETR (Real-Time DEtection TRansformer) for 2D object detection. Designed for real-time inference with competitive accuracy. Supports distillation and quantization for deployment optimization.

Set model.pretrained_backbone_path for backbone weights or train.pretrained_model_path for full model.

For TAO Deploy TensorRT actions (`gen_trt_engine`, TensorRT `evaluate`, and TensorRT `inference`), read `references/tao-deploy-rtdetr.md` first. Deploy spec templates live in this skill's `references/` folder with the `spec_template_deploy_*.yaml` prefix.

## Dataclass Schemas

Generated TAO Core schemas are packaged in `schemas/<action>.schema.json`, with `schemas/manifest.json` listing available actions. Each generated schema also emits `references/spec_template_<action>.yaml` from the schema top-level `default` field. AutoML enablement is declared at the model layer in `references/skill_info.yaml` via `automl_enabled`. Runnable AutoML still requires `schemas/train.schema.json` and `references/spec_template_train.yaml` to exist and parse. Use the packaged train schema for `automl_default_parameters`, `automl_disabled_parameters`, defaults, min/max bounds, enums, option weights, math conditions, dependencies, and popular parameters. Do not expect `~/tao-core` at runtime; maintainers regenerate schemas/templates before packaging the skill bank.

## Train Action Policy

This model is AutoML-enabled at the model layer. Before handling any train-stage request, read `references/skill_info.yaml` and resolve the run override from either an explicit `automl_policy` value or the user's workflow request. Use `automl_policy: on` by default and only expose `on` / `off` in new launch prompts. Treat phrases like "turn off AutoML", "disable AutoML", "no HPO", or "plain training" as `automl_policy: off` for this run only. When `automl_policy: on`, `automl_enabled: true`, and both `schemas/train.schema.json` and `references/spec_template_train.yaml` are packaged, route the train action through `tao-skill-bank:tao-run-automl` by default with this model's `skill_dir`. Preserve workflow/application overrides for datasets, specs, output directories, GPU/platform settings, parent checkpoints, and `automl_policy`. Use direct model training only when `automl_policy: off` or the packaged train schema/template is missing; in the missing-schema case, report that AutoML is enabled but not runnable for this model until schemas are generated.

Non-train actions such as `evaluate`, `inference`, `export`, and deploy flows stay in this model skill. The per-run `automl_policy` override does not change model metadata.

## Supported Actions

The packaged RT-DETR PyT CLI supports `train`, `distill`, `quantize`, `evaluate`, `export`, `inference`, and `default_specs`. This model skill exposes `train`, `distill`, `quantize`, `evaluate`, `export`, and `inference`; resume/retrain is performed through `train` with `train.resume_training_checkpoint_path`.

The parent PyT CLI does not expose `gen_trt_engine`. Use `models/rtdetr/deploy` for TensorRT engine generation, TensorRT evaluation, and TensorRT inference.

## Training Requirements

- **Dataset type:** object_detection
- **Formats:** coco, coco_raw
- **Monitoring metric:** mAP50 for quick operational checks; `val_mAP` for COCO/paper-style benchmark comparisons.

### Per-Action Dataset Requirements

| Action | Spec Key | Source | Files | List? |
|---|---|---|---|---|
| distill | dataset.train_data_sources | train_datasets | image_dir: images.tar.gz, json_file: annotations.json | Yes |
| distill | dataset.val_data_sources | eval_dataset | image_dir: images.tar.gz, json_file: annotations.json | No |
| evaluate | dataset.test_data_sources | eval_dataset | image_dir: images.tar.gz, json_file: annotations.json | No |
| inference | dataset.infer_data_sources | inference_dataset | image_dir: images.tar.gz, classmap: label_map.txt | Yes |
| quantize | dataset.train_data_sources | train_datasets | image_dir: images.tar.gz, json_file: annotations.json | Yes |
| quantize | dataset.val_data_sources | eval_dataset | image_dir: images.tar.gz, json_file: annotations.json | No |
| quantize | dataset.quant_calibration_data_sources | train_datasets | image_dir: images.tar.gz, json_file: annotations.json | No |
| train | dataset.train_data_sources | train_datasets | image_dir: images.tar.gz, json_file: annotations.json | Yes |
| train | dataset.val_data_sources | eval_dataset | image_dir: images.tar.gz, json_file: annotations.json | No |

### Typical Spec Overrides

Data source overrides are **mandatory for every action** — the agent MUST construct data source paths from the Per-Action Dataset Requirements table above and include them in `spec_overrides`.

```python
S3_TRAIN = "s3://bucket/data/train"
S3_EVAL = "s3://bucket/data/eval"
CHECKPOINT = "/results/{train_job_id}/results_dir/model_epoch_000.pth"
ONNX_FILE = "/results/{export_job_id}/results_dir/rtdetr.onnx"
```

**train (mandatory data sources):**
```python
{
    "train.num_epochs": 10,
    "train.checkpoint_interval": 10,
    "train.validation_interval": 10,
    "train.num_gpus": 1,
    "train.gpu_ids": [0],
    "dataset.num_classes": "<num_classes> + 1",
    "dataset.train_data_sources": [{"image_dir": f"{S3_TRAIN}/images.tar.gz", "json_file": f"{S3_TRAIN}/annotations.json"}],
    "dataset.val_data_sources": {"image_dir": f"{S3_EVAL}/images.tar.gz", "json_file": f"{S3_EVAL}/annotations.json"},
}
```

**resume train (mandatory checkpoint):**
```python
{
    "train.num_epochs": 11,
    "train.resume_training_checkpoint_path": CHECKPOINT,
    "dataset.num_classes": "<num_classes> + 1",
    "dataset.eval_class_ids": [1, 2, 3, 4],
    "dataset.train_data_sources": [{"image_dir": f"{S3_TRAIN}/images.tar.gz", "json_file": f"{S3_TRAIN}/annotations.json"}],
    "dataset.val_data_sources": {"image_dir": f"{S3_EVAL}/images.tar.gz", "json_file": f"{S3_EVAL}/annotations.json"},
}
```

**evaluate (mandatory data sources and checkpoint):**
```python
{
    "dataset.num_classes": "<num_classes> + 1",
    "dataset.eval_class_ids": [1, 2, 3, 4],
    "dataset.test_data_sources": {"image_dir": f"{S3_EVAL}/images.tar.gz", "json_file": f"{S3_EVAL}/annotations.json"},
    "evaluate.checkpoint": CHECKPOINT,
}
```

**export (mandatory checkpoint and output):**
```python
{
    "dataset.num_classes": "<num_classes> + 1",
    "export.checkpoint": CHECKPOINT,
    "export.onnx_file": ONNX_FILE,
    "export.input_height": 640,
    "export.input_width": 640,
}
```

**quantize (mandatory data sources):**
```python
{
    "dataset.num_classes": "<num_classes> + 1",
    "quantize.layers": [
        {
            "module_name": "*",
            "weights": {
                "dtype": "float8_e4m3fn"
            },
            "activations": {
                "dtype": "float8_e4m3fn"
            }
        }
    ],
    "dataset.train_data_sources": [{"image_dir": f"{S3_TRAIN}/images.tar.gz", "json_file": f"{S3_TRAIN}/annotations.json"}],
    "dataset.val_data_sources": {"image_dir": f"{S3_EVAL}/images.tar.gz", "json_file": f"{S3_EVAL}/annotations.json"},
    "dataset.quant_calibration_data_sources": {"image_dir": f"{S3_TRAIN}/images.tar.gz", "json_file": f"{S3_TRAIN}/annotations.json"},
    "quantize.model_path": CHECKPOINT,
}
```

**inference (mandatory data sources and checkpoint):**
```python
{
    "dataset.num_classes": "<num_classes> + 1",
    "dataset.infer_data_sources": {"image_dir": [f"{S3_EVAL}/images.tar.gz"], "classmap": f"{S3_EVAL}/label_map.txt"},
    "inference.checkpoint": CHECKPOINT,
}
```

**distill (mandatory data sources and teacher checkpoint):**
```python
{
    "dataset.train_data_sources": [{"image_dir": f"{S3_TRAIN}/images.tar.gz", "json_file": f"{S3_TRAIN}/annotations.json"}],
    "dataset.val_data_sources": {"image_dir": f"{S3_EVAL}/images.tar.gz", "json_file": f"{S3_EVAL}/annotations.json"},
    "distill.pretrained_teacher_model_path": CHECKPOINT,
}
```
## Eval Dataset

Optional. Provides validation mAP at each checkpoint if supplied.

## Important Parameters

- **dataset.num_classes**: Number of classes. Default 80 (MSCOCO 80-class). Must match your dataset annotations.
- **model.backbone**: Default resnet_50. Supported: ResNet variants, ConvNeXt, FAN, EfficientViT. RT-DETR is optimized for real-time with lighter backbones.
- **train.optim.lr**: Learning rate. Default 1e-4 (lower than DINO's 2e-4). lr_backbone defaults to 1e-5.
- **dataset.augmentation.train_spatial_size**: Training input size. Default [640, 640]. Smaller than DINO's multi-scale (up to 1333). Key to RT-DETR's speed.
- **model.num_feature_levels**: Default 3 (vs DINO's 4). return_interm_indices is [1,2,3].
- **train.enable_ema**: Exponential moving average. Default False. Enable for potentially smoother convergence.
- **dataset.remap_mscoco_category**: Default False. Set True only for original MSCOCO dataset with 91-to-80 category ID remapping.

## Multi-GPU / Multi-Node

**Launch method:** `torchrun` (LIGHTNING_EXCLUDED_NETWORK). The entrypoint runs `torchrun --nnodes=N --nproc-per-node=M train.py`, NOT plain `python`.

| Spec Key | Description | Default |
|----------|-------------|---------|
| `train.num_gpus` | Number of GPUs per node | 1 |
| `train.gpu_ids` | GPU device indices | [0] |
| `train.num_nodes` | Number of nodes | 1 |
| `train.distributed_strategy` | `ddp` or `fsdp` | `ddp` |

- When increasing `train.num_gpus`, also set `train.gpu_ids` to the same
  visible device range. For example, an 8-GPU single-node Slurm run must
  include both `"train.num_gpus": 8` and
  `"train.gpu_ids": [0, 1, 2, 3, 4, 5, 6, 7]`.
- `CUDA_VISIBLE_DEVICES` is explicitly set (unlike Lightning-managed models which use `TAO_VISIBLE_DEVICES`)
- `ddp` with activation checkpointing: `find_unused_parameters=False`
- `ddp` without: `find_unused_parameters=True`
- `fsdp` supported, forces FP16

**Multi-node env vars** (set by orchestrator):

| Variable | Purpose |
|----------|---------|
| `WORLD_SIZE` | Number of nodes (triggers multinode mode) |
| `NODE_RANK` | This node's rank (0-indexed) |
| `MASTER_ADDR` | Rank-0 node IP |
| `MASTER_PORT` | Rank-0 port (default 29500) |
| `NUM_GPU_PER_NODE` | GPUs per node (default: all visible) |

**CRITICAL:** `NODE_RANK` is copied to `RANK` if `RANK` is unset. This is required for torchrun multinode.

## Export / TRT Defaults

- Export input: 640x640, opset 17
- TRT data types: FP32, FP16, INT8
- TRT workspace: 1024 MB
- TRT max_batch_size: 4

## Distillation

RT-DETR supports knowledge distillation with a teacher model. Requires `distill` action with `distill.pretrained_teacher_model_path` and a distillation binding configuration.

Use the packaged `references/spec_template_distill.yaml` as the starting point. The validated default binding uses the RT-DETR distiller's explicit IOU feature path:

```yaml
distill:
  bindings:
  - student_module_name: srcs
    teacher_module_name: srcs
    criterion: IOU
    weight: 1.0
```

Do not substitute DINO-style output names such as `pred_logits` / `pred_boxes`, and do not bind arbitrary decoder heads unless you have verified the module returns captured feature lists. The RT-DETR distiller asserts that IOU bindings must use `srcs` or `dsrcs`.

## Hardware

Minimum 1 GPU(s), recommended 2 GPU(s). 16GB+ (V100 or A100) VRAM per GPU. RT-DETR is more memory-efficient than DINO/GDINO due to smaller input size (640x640) and fewer feature levels. Trains well on single GPU for small-medium datasets.

## Error Patterns

**CUDA out of memory**: Reduce batch_size. RT-DETR at 640x640 is lighter than DINO at 1333px, but batch_size > 8 may still OOM on 16GB GPUs.

**num_classes mismatch**: RT-DETR defaults to 80 (not 91 like DINO). Ensure dataset.num_classes matches your annotation categories.

**CUDA index assert from category IDs**: If COCO category IDs are one-based or otherwise not remapped to zero-based contiguous IDs, set `dataset.num_classes` to `max(category_id) + 1` and keep `dataset.eval_class_ids` aligned to the actual category IDs. For the packaged four-class S3 sample with IDs 1-4, use `dataset.num_classes: 5` and `dataset.eval_class_ids: [1, 2, 3, 4]`.

**return_interm_indices vs num_feature_levels**: Default is [1,2,3] with num_feature_levels=3. Must be consistent if changed.

**Export shape mismatch**: Keep RT-DETR export and deploy consumer input size at
the validated `640x640` default unless the model has been trained and checked
for a different shape. The older packaged `960x544` template shape can fail
during ONNX tracing with `The size of tensor a (...) must match the size of
tensor b (...)` in `hybrid_encoder.py` positional embedding addition.

**AutoML metric extraction**: RT-DETR emits detection metrics in structured training status and logs. For COCO/paper-style benchmark comparisons, optimize `val_mAP` with `direction: maximize`; for explicit AP50 workflows, optimize `mAP50`. Prefer `results_dir/train/status.json` or AutoML result state before parsing raw logs. Do not optimize `val_loss` for default detection model invocations.

**Checkpoint handoff**: For evaluate/export/inference/quantize/distill/resume, use the checkpoint resolver on the best AutoML child job's `results_dir/train/` folder and select the action-appropriate `model_epoch_*.pth` checkpoint. RT-DETR may also write a latest symlink, but that should only be used when a caller explicitly requests latest. Keep `dataset.num_classes`, `dataset.eval_class_ids`, `model.num_queries`, and `model.num_select` consistent with training.

**Parent `rtdetr gen_trt_engine` rejected by the PyT CLI**: In the validated 7.0.0 PyT container, `rtdetr gen_trt_engine` is not a valid parent-model subtask. Use the RT-DETR deploy workflow (`references/tao-deploy-rtdetr.md`) for TensorRT engine generation, TensorRT evaluation, and TensorRT inference.

## Spec Param / Parent Model Inference

Model-specific inference mappings belong in this MD file, not in `config.json`. Generated runners should read this section and apply the mappings with SDK helpers before `create_job()`. This mirrors the old microservices `infer_params.py` flow.

Inference mappings from TAO Core `rtdetr.config.json`:

| Action | Spec Field | Inference Function | Meaning |
|---|---|---|---|
| distill | `distill.pretrained_teacher_model_path` | `parent_model` | model file inferred from the parent job results folder |
| distill | `encryption_key` | `key` | encryption key |
| distill | `results_dir` | `output_dir` | current job results directory |
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
| train | `train.pretrained_model_path` | `ptm_if_no_resume_model` | PTM when no resume checkpoint exists |
| train | `train.resume_training_checkpoint_path` | `resume_model` | model file inferred from the current job results folder |

For `parent_model` or `parent_model_folder`, pass the upstream train/export/AutoML child job id as `parent_job_id`. The SDK lists the parent result folder, filters checkpoint artifacts, and returns the selected model file or folder. Do not add these mappings back to `config.json` and do not patch generated runner scripts to guess checkpoint paths.

## Deployment

- [tao-deploy-rtdetr](references/tao-deploy-rtdetr.md)
