---
name: tao-train-segformer
description: SegFormer for semantic segmentation. Lightweight transformer-based architecture with hierarchical feature
  extraction, efficient for real-time segmentation tasks. Use when training, evaluating, exporting, quantizing, or running
  inference for a TAO SegFormer model. Trigger phrases include "train SegFormer", "semantic segmentation", "lightweight
  transformer segmenter", "real-time semantic segmentation".
license: Apache-2.0
compatibility: Requires docker + nvidia-container-toolkit.
metadata:
  version: "0.1.0"
  author: NVIDIA Corporation
allowed-tools: Read Bash
tags:
- segmentation
---

# SegFormer

SegFormer for semantic segmentation. Lightweight transformer-based architecture with hierarchical feature extraction. Efficient for real-time segmentation tasks.

Set model.backbone.pretrained_backbone_path for backbone weights.

For TAO Deploy TensorRT actions (`gen_trt_engine`, TensorRT `evaluate`, and TensorRT `inference`), read `references/tao-deploy-segformer.md` first. Deploy spec templates live in this skill's `references/` folder with the `spec_template_deploy_*.yaml` prefix.

## Dataclass Schemas

Generated TAO Core schemas are packaged in `schemas/<action>.schema.json`, with `schemas/manifest.json` listing available actions. Each generated schema also emits `references/spec_template_<action>.yaml` from the schema top-level `default` field. AutoML enablement is declared at the model layer in `references/skill_info.yaml` via `automl_enabled`. Runnable AutoML still requires `schemas/train.schema.json` and `references/spec_template_train.yaml` to exist and parse. Use the packaged train schema for `automl_default_parameters`, `automl_disabled_parameters`, defaults, min/max bounds, enums, option weights, math conditions, dependencies, and popular parameters. Do not expect `~/tao-core` at runtime; maintainers regenerate schemas/templates before packaging the skill bank.

## Train Action Policy

This model is AutoML-enabled at the model layer. Before handling any train-stage request, read `references/skill_info.yaml` and resolve the run override from either an explicit `automl_policy` value or the user's workflow request. Use `automl_policy: on` by default and only expose `on` / `off` in new launch prompts. Treat phrases like "turn off AutoML", "disable AutoML", "no HPO", or "plain training" as `automl_policy: off` for this run only. When `automl_policy: on`, `automl_enabled: true`, and both `schemas/train.schema.json` and `references/spec_template_train.yaml` are packaged, route the train action through `tao-skill-bank:tao-run-automl` by default with this model's `skill_dir`. Preserve workflow/application overrides for datasets, specs, output directories, GPU/platform settings, parent checkpoints, and `automl_policy`. Use direct model training only when `automl_policy: off` or the packaged train schema/template is missing; in the missing-schema case, report that AutoML is enabled but not runnable for this model until schemas are generated.

Non-train actions such as `evaluate`, `inference`, `export`, and deploy flows stay in this model skill. The per-run `automl_policy` override does not change model metadata.

## Supported Actions

The packaged SegFormer PyT CLI supports `train`, `evaluate`, `export`, `inference`, `quantize`, and `default_specs`. This model skill exposes `train`, `evaluate`, `export`, `inference`, and `quantize`; resume/retrain is performed through `train` with `train.resume_training_checkpoint_path`.

The parent PyT CLI does not expose `gen_trt_engine`. Use `models/segformer/deploy` for TensorRT engine generation, TensorRT evaluation, and TensorRT inference.

## Training Requirements

- **Dataset type:** segmentation
- **Formats:** unet
- **Monitoring metric:** val_miou, maximize

### Per-Action Dataset Requirements

| Action | Spec Key | Source | Files | List? |
|---|---|---|---|---|
| evaluate | dataset.segment.root_dir | eval_dataset | extracted root containing `images/<split>` and `masks/<split>` | No |
| export | dataset.segment.root_dir | train_datasets | extracted root containing `images/<split>` and `masks/<split>` | No |
| inference | dataset.segment.root_dir | inference_dataset | extracted root containing `images/<split>` and `masks/<split>` | No |
| quantize | dataset.segment.root_dir | train_datasets | extracted root containing `images/<split>` and `masks/<split>` | No |
| quantize | dataset.segment.quant_calibration_dataset.images_dir | calibration_dataset | extracted image directory | No |
| train | dataset.segment.root_dir | train_datasets | extracted root containing `images/<split>` and `masks/<split>` | No |

### Typical Spec Overrides

Data source overrides are **mandatory for every action** — the agent MUST construct data source paths from the Per-Action Dataset Requirements table above and include them in `spec_overrides`.

```python
SEG_TRAIN_ROOT = "/data/segformer/train"
SEG_EVAL_ROOT = "/data/segformer/eval"
SEG_INFER_ROOT = "/data/segformer/infer"
CAL_IMAGES = f"{SEG_TRAIN_ROOT}/images/train"
```

**train (mandatory data sources):**
```python
{
    "train.num_gpus": 1,
    "train.num_epochs": 10,
    "train.checkpoint_interval": 10,
    "train.validation_interval": 10,
    "dataset.segment.batch_size": 4,
    "dataset.segment.root_dir": SEG_TRAIN_ROOT,
}
```

**evaluate (mandatory data sources):**
```python
{
    "evaluate.batch_size": 4,
    "dataset.segment.root_dir": SEG_EVAL_ROOT,
    "evaluate.checkpoint": CHECKPOINT,
}
```

**inference (mandatory data sources):**
```python
{
    "dataset.segment.batch_size": 1,
    "dataset.segment.root_dir": SEG_INFER_ROOT,
    "inference.checkpoint": CHECKPOINT,
}
```

**export (mandatory data sources):**
```python
{
    "dataset.segment.root_dir": SEG_TRAIN_ROOT,
    "export.checkpoint": CHECKPOINT,
    "export.input_height": 256,
    "export.input_width": 256,
    "export.onnx_file": ONNX_FILE,
}
```

**quantize (mandatory data sources):**
```python
{
    "dataset.segment.root_dir": SEG_TRAIN_ROOT,
    "dataset.segment.quant_calibration_dataset.images_dir": CAL_IMAGES,
    "quantize.model_path": CHECKPOINT,
}
```

If the source dataset is delivered as separate `images/*.tar.gz` and
`masks/*.tar.gz` archives, extract them before launch so `root_dir` contains
directories such as `images/train`, `images/val`, `images/test`, `masks/train`,
and `masks/val`. Do not point `dataset.segment.root_dir` at an archive staging
folder that still contains only tarballs.
## Eval Dataset

Optional. Validation data is typically part of the root_dir structure.

## Important Parameters

- **dataset.segment.num_classes**: Number of segmentation classes. Default 2 (binary). Must match the number of classes in your mask annotations.
- **model.backbone.type**: Default fan_small_12_p4_hybrid. Supported includes FAN variants, SegFormer MIT variants, and others.
- **dataset.segment.root_dir**: Root directory of the segmentation dataset.
- **dataset.segment.img_size**: Input image size. Default 256. Increase for finer segmentation at the cost of memory.
- **train.optim.lr**: Learning rate. Default 6e-5.
- **model.freeze_backbone**: Whether to freeze the backbone during training. Useful for fine-tuning with limited data.
- **dataset.segment.batch_size**: Per-GPU batch size. Default 8.
- **dataset.segment.label_transform**: Use the string `"None"` when no label
  transform is desired. Do not set this to JSON/YAML null; strict schema merge
  treats the field as a string enum.
- **dataset.segment.palette**: For grayscale masks, use one integer per RGB
  entry, for example `rgb: [85]`. Preserve the dataset's actual label ids and
  class names rather than normalizing them unless the user explicitly asks for a
  conversion.

## Multi-GPU / Multi-Node

**Launch method:** Lightning-managed (single `python` process, Lightning spawns workers).

| Spec Key | Description | Default |
|----------|-------------|---------|
| `train.num_gpus` | Number of GPUs | 1 |
| `train.gpu_ids` | GPU device indices | [0] |
| `train.num_nodes` | Number of nodes | 1 |
| `train.sync_batchnorm` | Sync BN across GPUs | configurable |
| `train.use_distributed_sampler` | Use distributed sampler | configurable |

- Multi-GPU strategy: `ddp_find_unused_parameters_true`
- No fsdp support

**Multi-node env vars** (set by orchestrator): `WORLD_SIZE`, `NODE_RANK`, `MASTER_ADDR`, `MASTER_PORT`, `NUM_GPU_PER_NODE`.

## Hardware

Minimum 1 GPU(s), recommended 2 GPU(s). 16GB+ (V100 or A100) VRAM per GPU. SegFormer is relatively lightweight. Default img_size=256 is memory-friendly. Increase img_size for higher resolution at the cost of memory and speed.

## Error Patterns

**CUDA out of memory**: Reduce batch_size or img_size. SegFormer memory scales quadratically with image size.

**num_classes mismatch**: Ensure dataset.segment.num_classes matches the actual number of classes in your mask annotations.

**TensorBoard unsupported for segmentation training**: Keep `train.tensorboard.enabled: false`. The SegFormer training entrypoint asserts that TensorBoard visualization is not supported for segmentation, so do not enable TensorBoard just to extract AutoML metrics; use log parsing or a post-train evaluator instead.

**AutoML metric extraction**: SegFormer train status files report `val_miou` alongside `val_loss`, `val_acc`, and other validation KPIs. Default AutoML train launches must optimize `val_miou` with `direction: maximize`; do not optimize `val_loss` for default model invocations.

For AutoML or long segmentation sweeps, read `val_miou` from
`results_dir/train/status.json` first. If the wrapper reports a terminal
failure but the structured status file reached the configured training budget
and contains finite `val_miou`, report the recovered metric with the wrapper
failure noted instead of discarding the measurement.

For high-resolution custom segmentation targets, keep dataset paths as per-run
inputs. Do not add customer/user-specific roots to this reusable skill. When the
user asks for a fixed full-budget search, remember that bracket algorithms
(`asha`, `bohb`, `dehb`, `hyperband`, `hyperband_es`, `pbt`) may intentionally
lower `train.num_epochs` for some recommendations; use Bayesian/BFBO or lock the
budget if every recommendation must run the full epoch count.

**Checkpoint handoff**: For evaluate/export/inference/quantize/resume, use the checkpoint resolver on the best AutoML child job's `results_dir/train/` folder and select the action-appropriate `model_epoch_*.pth` checkpoint, such as `model_epoch_000_step_00010.pth`. SegFormer may also write `segformer_model_latest.pth`, but that should only be used when a caller explicitly requests latest. Preserve `dataset.segment.num_classes`, `dataset.segment.img_size`, and `dataset.segment.root_dir` overrides for downstream actions.

**Resume/retrain checkpoint**: Resume uses `train.resume_training_checkpoint_path`.
Pass the exact resolved checkpoint from the previous train output, not a guessed
`model.pth` path. A resumed one-epoch run should produce the next checkpoint in
the new results directory, for example `model_epoch_001_step_00020.pth`.

**Export / TensorRT shape alignment**: Keep `export.input_height` and
`export.input_width` aligned with `dataset.segment.img_size` unless the trained
model and deploy specs have been validated at another resolution. The packaged
fresh-install path is validated at `256x256`, matching the default SegFormer
dataset and deploy templates.

**Parent `segformer gen_trt_engine` rejected by the PyT CLI**: In the validated 7.0.0 PyT container, `segformer gen_trt_engine` is not a valid parent-model subtask. Use the SegFormer deploy workflow (`references/tao-deploy-segformer.md`) for TensorRT engine generation, TensorRT evaluation, and TensorRT inference.

## Spec Param / Parent Model Inference

Model-specific inference mappings belong in this MD file, not in `config.json`. Generated runners should read this section and apply the mappings with SDK helpers before `create_job()`. This mirrors the old microservices `infer_params.py` flow.

Inference mappings from TAO Core `segformer.config.json`:

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
| train | `model.backbone.pretrained_backbone_path` | `ptm_if_no_resume_model` | PTM when no resume checkpoint exists |
| train | `results_dir` | `output_dir` | current job results directory |
| train | `train.pretrained_model_path` | `ptm_if_no_resume_model` | PTM when no resume checkpoint exists |
| train | `train.resume_training_checkpoint_path` | `resume_model` | model file inferred from the current job results folder |

For `parent_model` or `parent_model_folder`, pass the upstream train/export/AutoML child job id as `parent_job_id`. The SDK lists the parent result folder, filters checkpoint artifacts, and returns the selected model file or folder. Do not add these mappings back to `config.json` and do not patch generated runner scripts to guess checkpoint paths.

## Deployment

- [tao-deploy-segformer](references/tao-deploy-segformer.md)
