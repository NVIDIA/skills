---
name: tao-train-nvdinov2
description: NVDINOv2 for self-supervised visual representation learning. Trains vision transformers via self-distillation
  (teacher-student) without labels and produces general-purpose visual features. Use when training, exporting, or
  running inference for a TAO NVDINOv2 backbone. Trigger phrases include "train NVDINOv2", "self-supervised ViT pretraining",
  "DINOv2 backbone", "visual representation learning".
license: Apache-2.0
compatibility: Requires docker + nvidia-container-toolkit.
metadata:
  version: "0.1.0"
  author: NVIDIA Corporation
allowed-tools: Read Bash
tags:
- self
- supervised
- learning
---

# NVDINOv2

NVDINOv2 for self-supervised visual representation learning. Trains vision transformers via self-distillation (teacher-student) without labels. Produces general-purpose visual features.

Set train.pretrained_model_path for pretrained ViT weights.

For TAO Deploy TensorRT actions (`gen_trt_engine`), read `references/tao-deploy-nvdinov2.md` first. Deploy spec templates live in this skill's `references/` folder with the `spec_template_deploy_*.yaml` prefix.

## Dataclass Schemas

Generated TAO Core schemas are packaged in `schemas/<action>.schema.json`, with `schemas/manifest.json` listing available actions. Each generated schema also emits `references/spec_template_<action>.yaml` from the schema top-level `default` field. AutoML enablement is declared at the model layer in `references/skill_info.yaml` via `automl_enabled`. Runnable AutoML still requires `schemas/train.schema.json` and `references/spec_template_train.yaml` to exist and parse. Use the packaged train schema for `automl_default_parameters`, `automl_disabled_parameters`, defaults, min/max bounds, enums, option weights, math conditions, dependencies, and popular parameters. Do not expect `~/tao-core` at runtime; maintainers regenerate schemas/templates before packaging the skill bank.

## Train Action Policy

This model is AutoML-enabled at the model layer. Before handling any train-stage request, read `references/skill_info.yaml` and resolve the run override from either an explicit `automl_policy` value or the user's workflow request. Use `automl_policy: on` by default and only expose `on` / `off` in new launch prompts. Treat phrases like "turn off AutoML", "disable AutoML", "no HPO", or "plain training" as `automl_policy: off` for this run only. When `automl_policy: on`, `automl_enabled: true`, and both `schemas/train.schema.json` and `references/spec_template_train.yaml` are packaged, route the train action through `tao-skill-bank:tao-run-automl` by default with this model's `skill_dir`. Preserve workflow/application overrides for datasets, specs, output directories, GPU/platform settings, parent checkpoints, and `automl_policy`. Use direct model training only when `automl_policy: off` or the packaged train schema/template is missing; in the missing-schema case, report that AutoML is enabled but not runnable for this model until schemas are generated.

Non-train actions such as `inference`, `export`, and deploy flows stay in this model skill. The per-run `automl_policy` override does not change model metadata.

## Training Requirements

- **Dataset type:** image_classification
- **Formats:** ssl
- **Monitoring metric:** train_loss

### Per-Action Dataset Requirements

| Action | Spec Key | Source | Files | List? |
|---|---|---|---|---|
| inference | dataset.test_dataset.images_dir | inference_dataset | images_test.tar.gz | No |
| train | dataset.train_dataset.images_dir | train_datasets | images_train.tar.gz | No |

### Typical Spec Overrides

Data source overrides are **mandatory for train and inference** — the agent MUST construct data source paths from the Per-Action Dataset Requirements table above and include them in `spec_overrides`.

```python
S3_TRAIN = "s3://bucket/data/train"
S3_EVAL = "s3://bucket/data/eval"
```

**train (mandatory data sources):**
```python
{
    "train.num_gpus": 1,
    "train.num_epochs": 10,
    "train.checkpoint_interval": 10,
    "dataset.train_dataset.images_dir": f"{S3_TRAIN}/images_train.tar.gz",
}
```

**local AutoML validation / smoke run:**
Use this shape when the goal is to confirm Bayesian launch, metric selection,
best-model choice, and checkpoint persistence on local Docker. It keeps the
run representative while avoiding the much slower ViT-Large default.

```python
{
    "wandb.enable": False,
    "model.backbone.teacher_type": "vit_s",
    "model.backbone.student_type": "vit_s",
    "model.backbone.img_size": 224,
    "dataset.batch_size": 8,
    "dataset.workers": 2,
    "train.num_epochs": 1,
    "train.checkpoint_interval": 1,
    "train.num_prototypes": 1024,
    "train.precision": "32-true",
    "train.use_custom_attention": False,
    "train.num_gpus": 1,
    "dataset.train_dataset.images_dir": f"{S3_TRAIN}/images_train.tar.gz",
}
```

**export (mandatory checkpoint handoff):**
```python
{
    "export.checkpoint": "<selected train/AutoML student_epoch_* checkpoint>",
    "export.onnx_file": "/path/to/results/nvdinov2.onnx",
}
```

**inference (mandatory data sources):**
```python
{
    "inference.checkpoint": "<selected train/AutoML student_epoch_* checkpoint>",
    "model.backbone.teacher_type": "<same value used for train>",
    "model.backbone.student_type": "<same value used for train>",
    "model.backbone.img_size": "<same value used for train>",
    "train.use_custom_attention": "<same value used for train>",
    "dataset.test_dataset.images_dir": f"{S3_EVAL}/images_test.tar.gz",
}
```
## Eval Dataset

Optional. SSL training does not use labels. Evaluation is downstream task-specific.

## Important Parameters

- **model.backbone.teacher_type**: Teacher ViT variant. Default vit_l (ViT-Large).
- **model.backbone.student_type**: Student ViT variant. Default vit_l. Typically matches teacher.
- **model.backbone.img_size**: Input image size. Default 518. Higher resolution produces better features but costs more memory.
- **model.backbone.patch_size**: ViT patch size. Default 14.
- **dataset.batch_size**: Per-GPU batch size. Default 4. SSL training is memory-intensive due to dual (teacher+student) forward passes.
- **train.layerwise_decay**: Layer-wise learning rate decay. Important for ViT fine-tuning.
- **train.clip_grad_norm**: Gradient clipping. Important for stable SSL training.

## Multi-GPU / Multi-Node

**Launch method:** Lightning-managed (single `python` process, Lightning spawns workers).

| Spec Key | Description | Default |
|----------|-------------|---------|
| `train.num_gpus` | Number of GPUs | 1 |
| `train.gpu_ids` | GPU device indices | [0] |
| `train.num_nodes` | Number of nodes | 1 |

- Strategy: `auto` (Lightning picks best strategy automatically)
- `sync_batchnorm` is always enabled — critical for SSL training with teacher-student framework
- Multi-GPU strongly recommended (4-8 GPUs) for meaningful SSL training

**Multi-node env vars** (set by orchestrator): `WORLD_SIZE`, `NODE_RANK`, `MASTER_ADDR`, `MASTER_PORT`, `NUM_GPU_PER_NODE`.

## Hardware

Minimum 4 GPU(s), recommended 8 GPU(s). 40GB+ (A100 recommended) VRAM per GPU. SSL with ViT-Large teacher+student is very memory-intensive. Requires A100 40GB+ GPUs. Multi-GPU strongly recommended.

## Error Patterns

**CUDA out of memory**: ViT-Large teacher+student with img_size=518 requires 40GB+ GPU memory. Reduce batch_size, img_size, or use smaller ViT variant.

**Inference checkpoint has unexpected Lightning keys**: For downstream
`inference`, pass the selected AutoML run's `student_epoch_*.pth` checkpoint,
not `nvdinov2_model_latest.pth`. The latest file is a training checkpoint and
the inference loader reports unexpected keys such as `state_dict`, optimizer
state, and scheduler state.

**Export checkpoint has unexpected Lightning keys**: Export also consumes the
selected `student_epoch_*.pth` checkpoint. Use the full `model_epoch_*.pth`
checkpoint only for resume/retrain via `train.resume_training_checkpoint_path`.

**TensorRT engine passed to PyT inference**: The packaged PyT `nvdinov2 inference`
implementation only loads `.pth` or `.tlt` model paths. TAO Deploy
`gen_trt_engine` builds a TensorRT engine for downstream consumers, but the PyT
inference action does not run on that engine.

**Separate distill action not available**: The current TAO PyT CLI exposes
`export`, `inference`, `train`, and `default_specs` for NvDINOv2. Do not launch
or advertise a standalone `nvdinov2 distill` action.

**AutoML metric not found**: TAO's status KPI reports the final training scalar
as `train_loss`. Use `train_loss` with minimize direction for AutoML selection.
Some Lightning progress lines also render the same scalar as
`train_loss_epoch`; treat that as a fallback alias only, not the primary
monitoring metric.

**Slow convergence**: SSL needs many epochs. Default 10 is for quick testing; production runs typically use 100+ epochs.

## Spec Param / Parent Model Inference

Model-specific inference mappings belong in this MD file, not in `config.json`. Generated runners should read this section and apply the mappings with SDK helpers before `create_job()`. This mirrors the old microservices `infer_params.py` flow.

Model-specific handoff mappings:

| Action | Spec Field | Inference Function | Meaning |
|---|---|---|---|
| export | `encryption_key` | `key` | encryption key |
| export | `export.checkpoint` | `parent_model` | selected `student_epoch_*.pth` checkpoint from the parent job results folder |
| export | `export.onnx_file` | `create_onnx_file` | output ONNX path |
| export | `results_dir` | `output_dir` | current job results directory |
| inference | `encryption_key` | `key` | encryption key |
| inference | `inference.checkpoint` | `parent_model` | selected `student_epoch_*.pth` checkpoint from the parent job results folder |
| inference | `results_dir` | `output_dir` | current job results directory |
| train | `encryption_key` | `key` | encryption key |
| train | `results_dir` | `output_dir` | current job results directory |
| train | `train.pretrained_model_path` | `ptm_if_no_resume_model` | PTM when no resume checkpoint exists |
| train | `train.resume_training_checkpoint_path` | `resume_model` | selected full `model_epoch_*.pth` training checkpoint from the current job results folder |

For `parent_model` or `parent_model_folder`, pass the upstream train/export/AutoML child job id as `parent_job_id`. The SDK lists the parent result folder, filters checkpoint artifacts, and returns the selected model file or folder. Do not add these mappings back to `config.json` and do not patch generated runner scripts to guess checkpoint paths.

## Deployment

- [tao-deploy-nvdinov2](references/tao-deploy-nvdinov2.md)
