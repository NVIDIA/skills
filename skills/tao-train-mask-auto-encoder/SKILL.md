---
name: tao-train-mask-auto-encoder
description: Masked Auto-Encoder (MAE) for self-supervised pretraining and fine-tuning. Masks random patches and reconstructs
  them to learn visual representations; supports pretrain and finetune stages. Use when training, evaluating, exporting, or
  running inference for a TAO MAE backbone. Trigger phrases include "pretrain MAE", "self-supervised vision pretraining",
  "Masked Autoencoder", "Mask Auto-Encoder", "MAE fine-tune".
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

# MAE

MAE (Masked Autoencoder) for self-supervised pretraining and fine-tuning. Masks random patches and reconstructs them to learn visual representations. Supports pretrain and finetune stages.

Set train.pretrained_model_path for pretrained MAE weights when fine-tuning.

For TAO Deploy TensorRT actions (`gen_trt_engine`), read `references/tao-deploy-mask-auto-encoder.md` first. Deploy spec templates live in this skill's `references/` folder with the `spec_template_deploy_*.yaml` prefix.

The parent PyTorch `mae` CLI supports `train`, `evaluate`, `inference`, and
`export`. Build TensorRT engines through the deploy workflow, not the model skill.

## Dataclass Schemas

Generated TAO Core schemas are packaged in `schemas/<action>.schema.json`, with `schemas/manifest.json` listing available actions. Each generated schema also emits `references/spec_template_<action>.yaml` from the schema top-level `default` field. AutoML enablement is declared at the model layer in `references/skill_info.yaml` via `automl_enabled`. Runnable AutoML still requires `schemas/train.schema.json` and `references/spec_template_train.yaml` to exist and parse. Use the packaged train schema for `automl_default_parameters`, `automl_disabled_parameters`, defaults, min/max bounds, enums, option weights, math conditions, dependencies, and popular parameters. Do not expect `~/tao-core` at runtime; maintainers regenerate schemas/templates before packaging the skill bank.

## Train Action Policy

This model is AutoML-enabled at the model layer. Before handling any train-stage request, read `references/skill_info.yaml` and resolve the run override from either an explicit `automl_policy` value or the user's workflow request. Use `automl_policy: on` by default and only expose `on` / `off` in new launch prompts. Treat phrases like "turn off AutoML", "disable AutoML", "no HPO", or "plain training" as `automl_policy: off` for this run only. When `automl_policy: on`, `automl_enabled: true`, and both `schemas/train.schema.json` and `references/spec_template_train.yaml` are packaged, route the train action through `tao-skill-bank:tao-run-automl` by default with this model's `skill_dir`. Preserve workflow/application overrides for datasets, specs, output directories, GPU/platform settings, parent checkpoints, and `automl_policy`. Use direct model training only when `automl_policy: off` or the packaged train schema/template is missing; in the missing-schema case, report that AutoML is enabled but not runnable for this model until schemas are generated.

Non-train actions such as `evaluate`, `inference`, `export`, and deploy flows stay in this model skill. The per-run `automl_policy` override does not change model metadata.

## Training Requirements

- **Dataset type:** image_classification
- **Formats:** ssl
- **Accepted dataset intents:** training, evaluation, testing
- **Monitoring metric:** train_loss

### Per-Action Dataset Requirements

| Action | Spec Key | Source | Files | List? |
|---|---|---|---|---|
| train | dataset.train_data_sources | train_datasets | images_train.tar.gz | No |
| train | dataset.val_data_sources | eval_dataset | images_val.tar.gz | No |
| evaluate | dataset.val_data_sources | eval_dataset | images_val.tar.gz | No |
| inference | dataset.test_data_sources | inference_dataset | images_test.tar.gz | No |

For SDK/app job inputs, the `images_*.tar.gz` archives are uploaded as the
action inputs. For direct local Docker runs against host-mounted data, extract
the archives first and point `dataset.train_data_sources`,
`dataset.val_data_sources`, and `dataset.test_data_sources` at the extracted
`images_train`, `images_val`, and `images_test` folders. Passing a local tar
path directly to the MAE CLI can produce a zero-sample dataloader because the
local dataloader does not unpack that archive path.

### Typical Spec Overrides

Data source overrides are **mandatory for every action** — the agent MUST construct data source paths from the Per-Action Dataset Requirements table above and include them in `spec_overrides`.

```python
S3_TRAIN = "s3://bucket/data/train"
S3_EVAL = "s3://bucket/data/eval"
```

**train (mandatory data sources):**
```python
{
    "dataset.train_data_sources": f"{S3_TRAIN}/images_train.tar.gz",
    "dataset.val_data_sources": f"{S3_EVAL}/images_val.tar.gz",
    "train.num_epochs": 10,
    "train.optim.lr": 2e-4,
}
```

**evaluate (mandatory data sources):**
```python
{
    "dataset.val_data_sources": f"{S3_EVAL}/images_val.tar.gz",
    "evaluate.checkpoint": "<selected train/AutoML checkpoint>",
    "train.stage": "finetune",
}
```

**inference (mandatory data sources):**
```python
{
    "dataset.test_data_sources": f"{S3_EVAL}/images_test.tar.gz",
    "inference.checkpoint": "<selected train/AutoML checkpoint>",
    "train.stage": "finetune",
}
```

## Eval Dataset

Optional. Pretraining does not need eval data. Fine-tuning optionally uses val set.

## Important Parameters

- **train.stage**: Training stage. Options: pretrain, finetune. Pretrain learns representations via masking. Finetune adds a classification head.
- **model.arch**: Architecture. Default convnextv2_base. For local smoke
  AutoML, use `convnextv2_atto` rather than unsupported names such as
  `vit_tiny_patch16`. Supported families include `vit_base_patch16` and larger
  ViTs, ConvNeXtV2 atto/femto/pico/nano/tiny/base/large/huge, and Hiera
  tiny/small/base/large/huge.
- **model.num_classes**: Number of classes for fine-tuning. Default 1000 (ImageNet). Only relevant in finetune stage.
- **model.mask_ratio**: Fraction of patches to mask during pretraining. Typically 0.75.
- **model.norm_pix_loss**: Whether to normalize pixel values in reconstruction loss.
- **dataset.augmentation.input_size**: Keep the local smoke profile at 224
  for ConvNeXtV2 MAE. Reducing to 112 can make the MAE mask grid incompatible
  with feature-map dimensions.
- MAE does not expose a `dataset.workers` spec field. Do not add it to
  smoke-test overrides; Hydra rejects unknown dataset keys before training.
- **train.optim.lr**: Learning rate. Default 2e-4.
- **dataset.augmentation**: Augmentation settings including mixup, cutmix for fine-tuning.

## Multi-GPU / Multi-Node

**Launch method:** Lightning-managed (single `python` process, Lightning spawns workers).

| Spec Key | Description | Default |
|----------|-------------|---------|
| `train.num_gpus` | Number of GPUs | 1 |
| `train.gpu_ids` | GPU device indices | [0] |
| `train.num_nodes` | Number of nodes | 1 |
| `train.distributed_strategy` | `ddp` or `fsdp` | `ddp` |

- `ddp` uses `find_unused_parameters=True`
- `fsdp` forces FP16
- Multi-GPU strongly recommended for pretraining (large batch sizes needed)

**Multi-node env vars** (set by orchestrator): `WORLD_SIZE`, `NODE_RANK`, `MASTER_ADDR`, `MASTER_PORT`, `NUM_GPU_PER_NODE`.

## Hardware

Minimum 2 GPU(s), recommended 8 GPU(s). 24GB+ (A100 recommended) VRAM per GPU. MAE pretraining benefits from large batch sizes across many GPUs. Fine-tuning is more modest in resource requirements.

## Error Patterns

**Stage mismatch**: Ensure train.stage matches your intent (pretrain vs finetune). Fine-tuning without a pretrained_model_path trains from scratch.

**Inference with pretrain checkpoints**: The MAE predict dataloader raises
`NotImplementedError` for `train.stage: pretrain`. Use a `finetune` checkpoint
for inference and classification-style evaluation, or restrict a pretrain-only
run to train/evaluate/export.

**num_classes mismatch (finetune only)**: Ensure model.num_classes matches your dataset class count when fine-tuning.

## Spec Param / Parent Model Inference

Model-specific inference mappings belong in this MD file, not in `config.json`. Generated runners should read this section and apply the mappings with SDK helpers before `create_job()`. This mirrors the old microservices `infer_params.py` flow.

Inference mappings from TAO Core `mae.config.json`:

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
| train | `encryption_key` | `key` | encryption key |
| train | `results_dir` | `output_dir` | current job results directory |
| train | `train.pretrained_model_path` | `ptm_if_no_resume_model` | PTM when no resume checkpoint exists |
| train | `train.resume_training_checkpoint_path` | `resume_model` | model file inferred from the current job results folder |

For `parent_model` or `parent_model_folder`, pass the upstream train/export/AutoML child job id as `parent_job_id`. The SDK lists the parent result folder, filters checkpoint artifacts, and returns the selected model file or folder. Do not add these mappings back to `config.json` and do not patch generated runner scripts to guess checkpoint paths.

When resolving checkpoints outside the SDK resolver, select the intended
epoch/step artifact exactly, for example `model_epoch_000_step_00099.pth`.
Use the `convnextv2_atto_latest.pth` or other latest symlink only when latest
is explicitly requested. Carry `train.stage`, `model.arch`, `model.num_classes`,
and export input size forward into evaluate, inference, export, and deploy
specs so the checkpoint and ONNX/engine shapes match.

## Deployment

- [tao-deploy-mask-auto-encoder](references/tao-deploy-mask-auto-encoder.md)
