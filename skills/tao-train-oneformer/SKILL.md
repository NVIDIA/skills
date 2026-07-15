---
name: tao-train-oneformer
description: OneFormer for universal image segmentation. Unifies panoptic, instance, and semantic segmentation with a
  single architecture using task-conditioned queries. Use when training, evaluating, exporting, quantizing, or running
  inference for a TAO OneFormer model. Trigger phrases include "train OneFormer", "universal segmentation",
  "task-conditioned segmentation", "panoptic / instance / semantic in one model".
license: Apache-2.0
compatibility: Requires docker + nvidia-container-toolkit.
metadata:
  version: "0.1.0"
  author: NVIDIA Corporation
allowed-tools: Read Bash
tags:
- segmentation
---

# OneFormer

OneFormer for universal image segmentation. Unifies panoptic, instance, and semantic segmentation with a single architecture using task-conditioned queries.

Set train.pretrained_backbone and/or train.pretrained_model.

For TAO Deploy TensorRT actions (`gen_trt_engine`, TensorRT `evaluate`, and TensorRT `inference`), read `references/tao-deploy-oneformer.md` first. Deploy spec templates live in this skill's `references/` folder with the `spec_template_deploy_*.yaml` prefix.

## Dataclass Schemas

Generated TAO Core schemas are packaged in `schemas/<action>.schema.json`, with `schemas/manifest.json` listing available actions. Each generated schema also emits `references/spec_template_<action>.yaml` from the schema top-level `default` field. AutoML enablement is declared at the model layer in `references/skill_info.yaml` via `automl_enabled`. Runnable AutoML still requires `schemas/train.schema.json` and `references/spec_template_train.yaml` to exist and parse. Use the packaged train schema for `automl_default_parameters`, `automl_disabled_parameters`, defaults, min/max bounds, enums, option weights, math conditions, dependencies, and popular parameters. Do not expect `~/tao-core` at runtime; maintainers regenerate schemas/templates before packaging the skill bank.

## Train Action Policy

This model is AutoML-enabled at the model layer. Before handling any train-stage request, read `references/skill_info.yaml` and resolve the run override from either an explicit `automl_policy` value or the user's workflow request. Use `automl_policy: on` by default and only expose `on` / `off` in new launch prompts. Treat phrases like "turn off AutoML", "disable AutoML", "no HPO", or "plain training" as `automl_policy: off` for this run only. When `automl_policy: on`, `automl_enabled: true`, and both `schemas/train.schema.json` and `references/spec_template_train.yaml` are packaged, route the train action through `tao-skill-bank:tao-run-automl` by default with this model's `skill_dir`. Preserve workflow/application overrides for datasets, specs, output directories, GPU/platform settings, parent checkpoints, and `automl_policy`. Use direct model training only when `automl_policy: off` or the packaged train schema/template is missing; in the missing-schema case, report that AutoML is enabled but not runnable for this model until schemas are generated.

Non-train actions such as `evaluate`, `inference`, `export`, and deploy flows stay in this model skill. The per-run `automl_policy` override does not change model metadata.

## Training Requirements

- **Dataset type:** segmentation
- **Formats:** coco_panoptic, coco
- **Monitoring metric:** mIoU

### Per-Action Dataset Requirements

| Action | Spec Key | Source | Files | List? |
|---|---|---|---|---|
| evaluate | dataset.train.images | train_datasets | images.tar.gz | No |
| evaluate | dataset.label_map | train_datasets | label_map.json | No |
| evaluate | dataset.train.annotations | train_datasets | annotations.json | No |
| evaluate | dataset.train.panoptic | train_datasets | images_panoptic.tar.gz | No |
| evaluate | dataset.val.images | eval_dataset | images.tar.gz | No |
| evaluate | dataset.val.annotations | eval_dataset | annotations.json | No |
| evaluate | dataset.val.panoptic | eval_dataset | images_panoptic.tar.gz | No |
| evaluate | dataset.test.images | eval_dataset | images.tar.gz | No |
| evaluate | dataset.test.annotations | eval_dataset | annotations.json | No |
| evaluate | dataset.test.panoptic | eval_dataset | images_panoptic.tar.gz | No |
| inference | dataset.train.images | train_datasets | images.tar.gz | No |
| inference | dataset.label_map | train_datasets | label_map.json | No |
| inference | dataset.train.annotations | train_datasets | annotations.json | No |
| inference | dataset.train.panoptic | train_datasets | images_panoptic.tar.gz | No |
| inference | dataset.val.images | eval_dataset | images.tar.gz | No |
| inference | dataset.val.annotations | eval_dataset | annotations.json | No |
| inference | dataset.val.panoptic | eval_dataset | images_panoptic.tar.gz | No |
| inference | dataset.test.images | inference_dataset | images.tar.gz | No |
| quantize | dataset.train.images | train_datasets | images.tar.gz | No |
| quantize | dataset.train.annotations | train_datasets | annotations.json | No |
| quantize | dataset.label_map | train_datasets | label_map.json | No |
| quantize | dataset.train.panoptic | train_datasets | images_panoptic.tar.gz | No |
| quantize | dataset.val.images | eval_dataset | images.tar.gz | No |
| quantize | dataset.val.annotations | eval_dataset | annotations.json | No |
| quantize | dataset.val.panoptic | eval_dataset | images_panoptic.tar.gz | No |
| quantize | dataset.test.images | eval_dataset | images.tar.gz | No |
| quantize | dataset.quant_calibration_dataset.images_dir | calibration_dataset | images.tar.gz | No |
| train | dataset.train.images | train_datasets | images.tar.gz | No |
| train | dataset.train.annotations | train_datasets | annotations.json | No |
| train | dataset.label_map | train_datasets | label_map.json | No |
| train | dataset.train.panoptic | train_datasets | images_panoptic.tar.gz | No |
| train | dataset.val.images | eval_dataset | images.tar.gz | No |
| train | dataset.val.annotations | eval_dataset | annotations.json | No |
| train | dataset.val.panoptic | eval_dataset | images_panoptic.tar.gz | No |
| train | dataset.test.images | eval_dataset | images.tar.gz | No |

### Typical Spec Overrides

Data source overrides are **mandatory for every action** — the agent MUST construct data source paths from the Per-Action Dataset Requirements table above and include them in `spec_overrides`.

```python
S3_TRAIN = "s3://bucket/data/train"
S3_EVAL = "s3://bucket/data/eval"
S3_INFERENCE = "s3://bucket/data/inference"
S3_CALIBRATION = "s3://bucket/data/calibration"
```

**train (mandatory data sources):**
```python
{
    "train.num_gpus": 1,
    "train.num_epochs": 10,
    "train.checkpoint_interval": 10,
    "train.validation_interval": 10,
    "model.sem_seg_head.num_classes": 133,
    "dataset.contiguous_id": True,
    "train.precision": "32",
    "dataset.train.images": f"{S3_TRAIN}/images.tar.gz",
    "dataset.train.annotations": f"{S3_TRAIN}/annotations.json",
    "dataset.label_map": f"{S3_TRAIN}/label_map.json",
    "dataset.train.panoptic": f"{S3_TRAIN}/images_panoptic.tar.gz",
    "dataset.val.images": f"{S3_EVAL}/images.tar.gz",
    "dataset.val.annotations": f"{S3_EVAL}/annotations.json",
    "dataset.val.panoptic": f"{S3_EVAL}/images_panoptic.tar.gz",
    "dataset.test.images": f"{S3_EVAL}/images.tar.gz",
}
```

**evaluate (mandatory data sources):**
```python
{
    "evaluate.checkpoint": "<selected train/AutoML checkpoint>",
    "model.sem_seg_head.num_classes": 133,
    "dataset.contiguous_id": True,
    "dataset.train.images": f"{S3_TRAIN}/images.tar.gz",
    "dataset.label_map": f"{S3_TRAIN}/label_map.json",
    "dataset.train.annotations": f"{S3_TRAIN}/annotations.json",
    "dataset.train.panoptic": f"{S3_TRAIN}/images_panoptic.tar.gz",
    "dataset.val.images": f"{S3_EVAL}/images.tar.gz",
    "dataset.val.annotations": f"{S3_EVAL}/annotations.json",
    "dataset.val.panoptic": f"{S3_EVAL}/images_panoptic.tar.gz",
    "dataset.test.images": f"{S3_EVAL}/images.tar.gz",
    "dataset.test.annotations": f"{S3_EVAL}/annotations.json",
    "dataset.test.panoptic": f"{S3_EVAL}/images_panoptic.tar.gz",
}
```

**export:**
```python
{
    "export.checkpoint": "<selected train/AutoML checkpoint>",
    "model.sem_seg_head.num_classes": 133,
    "model.export": True,
    "export.onnx_file": "/results/oneformer_export_640.onnx",
}
```

**inference (mandatory data sources):**
```python
{
    "inference.checkpoint": "<selected train/AutoML checkpoint>",
    "dataset.train.images": f"{S3_TRAIN}/images.tar.gz",
    "dataset.label_map": f"{S3_TRAIN}/label_map.json",
    "dataset.train.annotations": f"{S3_TRAIN}/annotations.json",
    "dataset.train.panoptic": f"{S3_TRAIN}/images_panoptic.tar.gz",
    "dataset.val.images": f"{S3_EVAL}/images.tar.gz",
    "dataset.val.annotations": f"{S3_EVAL}/annotations.json",
    "dataset.val.panoptic": f"{S3_EVAL}/images_panoptic.tar.gz",
    "dataset.test.images": f"{S3_INFERENCE}/images.tar.gz",
    "inference.images_dir": f"{S3_INFERENCE}/images.tar.gz",
}
```

**quantize (mandatory data sources):**
```python
{
    "quantize.model_path": "<selected train/AutoML checkpoint>",
    "dataset.train.images": f"{S3_TRAIN}/images.tar.gz",
    "dataset.train.annotations": f"{S3_TRAIN}/annotations.json",
    "dataset.label_map": f"{S3_TRAIN}/label_map.json",
    "dataset.train.panoptic": f"{S3_TRAIN}/images_panoptic.tar.gz",
    "dataset.val.images": f"{S3_EVAL}/images.tar.gz",
    "dataset.val.annotations": f"{S3_EVAL}/annotations.json",
    "dataset.val.panoptic": f"{S3_EVAL}/images_panoptic.tar.gz",
    "dataset.test.images": f"{S3_EVAL}/images.tar.gz",
    "dataset.quant_calibration_dataset.images_dir": f"{S3_CALIBRATION}/images.tar.gz",
}
```

## Checkpoint Selection

OneFormer training writes epoch-step checkpoints such as
`model_epoch_000_step_00017.pth` and may also write a
`oneformer_model_latest.pth` symlink. For checkpoint-dependent actions, use the
model-skill or SDK parent-model resolver and pass the exact selected checkpoint
path into `evaluate.checkpoint`, `inference.checkpoint`, `export.checkpoint`,
`quantize.model_path`, or `train.resume_training_checkpoint_path`. Do not pick
the `oneformer_model_latest.pth` symlink by name unless the user explicitly asks
for latest checkpoint behavior. If the resolver reports a best checkpoint, use
that best checkpoint for evaluation/export/inference; if the user asks for a
specific epoch or step, use the matching epoch-step checkpoint.
## Eval Dataset

Optional. Val data configured alongside train in the dataset config.

## Important Parameters

- **model.sem_seg_head.num_classes**: Number of segmentation class indices available to the head. Default 133 for COCO panoptic data when `dataset.contiguous_id: True` remaps raw category ids through the label map. Do not shrink this to a global workflow class count unless the label map and annotations have actually been reduced to that class set.
- **model.one_former.hidden_dim**: Keep at 256 for local smoke runs unless
  the text encoder width is changed in lock-step. Reducing hidden_dim alone
  causes a text feature/context dimension mismatch during training.
- **model.backbone.name**: Default D2SwinTransformer (Swin-based). embed_dim=192, depths=[2,2,18,2] by default.
- **train.num_epochs**: Default 50 — significantly higher than most TAO models. OneFormer needs more epochs for convergence.
- **train.optim.lr**: Learning rate. Default 1e-5. Lower than Mask2Former's 2e-4.
- **model.task_toggling**: Enable/disable specific tasks: semantic_on, instance_on, panoptic_on.
- **export.task**: Export task mode. Options: semantic, instance, panoptic. Default semantic. Export input defaults to 640x640.
- **inference.mode**: Inference mode. Options: semantic, instance, panoptic. Default semantic. image_size defaults to [1024, 1024].
- **evaluate.iou_per_class**: Report per-class IoU in evaluation. Default True.

## Multi-GPU / Multi-Node

**Launch method:** Lightning-managed (single `python` process, Lightning spawns workers).

| Spec Key | Description | Default |
|----------|-------------|---------|
| `train.num_gpus` | Number of GPUs | 1 |
| `train.gpu_ids` | GPU device indices | [0] |
| `train.num_nodes` | Number of nodes | 1 |

- Uses explicit `DDPStrategy` with `find_unused_parameters=True`, `gradient_as_bucket_view=True`, `process_group_backend="nccl"`
- `sync_batchnorm` is always enabled
- No fsdp support — DDP only

**Multi-node env vars** (set by orchestrator): `WORLD_SIZE`, `NODE_RANK`, `MASTER_ADDR`, `MASTER_PORT`, `NUM_GPU_PER_NODE`.

## Hardware

Minimum 2 GPU(s), recommended 4 GPU(s). 24GB+ (A100 recommended) VRAM per GPU. OneFormer is memory-intensive like Mask2Former. batch_size=1 is the default. Multi-GPU needed for reasonable training speed, especially with 50 epochs.

## Error Patterns

**CUDA out of memory**: batch_size is already 1. Reduce image resolution or use a smaller Swin configuration.

**Extracted S3 tarball points one level too high**: For local Docker runs,
`images.tar.gz` and `images_panoptic.tar.gz` may extract wrapper directories
such as `images/` and `images_panoptic/`. Set `dataset.*.images`,
`dataset.*.panoptic`, `inference.images_dir`, and quantization calibration
paths to the actual folder containing image or panoptic files, not the wrapper
directory. A one-level-too-high path fails with `FileNotFoundError` for the
first annotation image even though recursive file counts look correct.

**default_specs missing results_dir**: The CLI `default_specs` subtask ignores
`-e` experiment specs for `results_dir`; pass a Hydra-style override instead:
`oneformer default_specs results_dir=/path/to/default_specs`.

**Invalid Lightning precision `fp32`**: Use `train.precision: "32"` in
train/AutoML/evaluate/inference specs. The current Lightning stack rejects the
legacy `fp32` string.

**PyTorch 2.6 checkpoint load failure on downstream actions**: Current
OneFormer checkpoints include OmegaConf objects. For checkpoints produced by
the same trusted TAO train/AutoML workflow, set
`TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` in downstream evaluate, inference, export,
quantize, or resume job env vars so Lightning can load the full checkpoint.
Do not use this env var for untrusted checkpoints.

**CUDA device-side assert in matcher/class cost**: If training fails in
`oneformer/utils/matcher.py` while indexing `out_prob[:, tgt_ids]`, compare
the effective target ids with `model.sem_seg_head.num_classes`. The packaged
COCO panoptic sample has 133 compact classes after `dataset.contiguous_id:
True` remapping, so use `model.sem_seg_head.num_classes: 133` even when a
broader validation workflow passes a smaller generic `num_classes` value.
Only use a smaller class count when the label map and annotations are reduced
to that exact contiguous class set.

**Inference returns PASS with no predictions**: OneFormer prediction reads
`inference.images_dir`, not `dataset.test.images`. Declare and populate
`inference.images_dir` with the image folder or tarball for every inference
run. `dataset.test.images` may still be useful for shared dataset context, but
it does not drive the PyTorch predict dataloader.

**Export output path pre-created as a directory**: Do not declare
`export.onnx_file` as a file output. The OneFormer exporter asserts that the
ONNX path does not already exist, while the local runner pre-creates declared
output paths. Set `export.onnx_file` explicitly in the spec to a non-existing
file path under the mounted results tree. Keep the default 640x640 export
shape for smoke validation; very small export shapes can trigger PyTorch ONNX
shape-inference failures.

**Quantize cannot find the training label map from an AutoML checkpoint**:
OneFormer Lightning checkpoints retain train-time absolute dataset paths in
their saved hparams. When running downstream actions from an AutoML child
checkpoint, keep the parent AutoML job directory accessible at its original
`/results/<job_id>` path inside the action container in addition to passing the
resolved checkpoint path. Otherwise quantize can fail while loading checkpoint
hparams even when the current spec includes a valid `dataset.label_map`.

**Slow training**: 50 default epochs with batch_size=1 is slow on single GPU. Use multi-GPU distributed training.

## Spec Param / Parent Model Inference

Model-specific inference mappings belong in this MD file, not in `config.json`. Generated runners should read this section and apply the mappings with SDK helpers before `create_job()`. This mirrors the old microservices `infer_params.py` flow.

Inference mappings from TAO Core `oneformer.config.json`:

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
| gen_trt_engine | `encryption_key` | `key` | encryption key |
| gen_trt_engine | `gen_trt_engine.onnx_file` | `parent_model` | model file inferred from the parent job results folder |
| gen_trt_engine | `gen_trt_engine.trt_engine` | `create_engine_file` | output TensorRT engine path |
| gen_trt_engine | `results_dir` | `output_dir` | current job results directory |
| inference | `encryption_key` | `key` | encryption key |
| inference | `inference.checkpoint` | `parent_model` | model file inferred from the parent job results folder |
| inference | `inference.trt_engine` | `parent_model` | model file inferred from the parent job results folder |
| inference | `results_dir` | `output_dir` | current job results directory |
| quantize | `encryption_key` | `key` | encryption key |
| quantize | `quantize.model_path` | `parent_model` | model file inferred from the parent job results folder |
| quantize | `results_dir` | `output_dir` | current job results directory |
| train | `encryption_key` | `key` | encryption key |
| train | `results_dir` | `output_dir` | current job results directory |
| train | `train.pretrained_backbone` | `{'link': 'https://github.com/SwinTransformer/storage/releases/download/v1.0.8/swin_tiny_patch4_window7_224_22k.pth', 'destination_path': '/ptm/mask2former/swin_tiny_patch4_window7_224_22k/swin_tiny_patch4_window7_224_22k.pth'}` | {'link': 'https://github.com/SwinTransformer/storage/releases/download/v1.0.8/swin_tiny_patch4_window7_224_22k.pth', 'destination_path': '/ptm/mask2former/swin_tiny_patch4_window7_224_22k/swin_tiny_patch4_window7_224_22k.pth'} |
| train | `train.pretrained_model` | `ptm_if_no_resume_model` | PTM when no resume checkpoint exists |
| train | `train.resume_training_checkpoint_path` | `resume_model` | model file inferred from the current job results folder |

For `parent_model` or `parent_model_folder`, pass the upstream train/export/AutoML child job id as `parent_job_id`. The SDK lists the parent result folder, filters checkpoint artifacts, and returns the selected model file or folder. Do not add these mappings back to `config.json` and do not patch generated runner scripts to guess checkpoint paths.

## Deployment

- [tao-deploy-oneformer](references/tao-deploy-oneformer.md)
