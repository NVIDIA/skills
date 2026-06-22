---
name: tao-train-dino
description: DINO (DETR with Improved DeNoising Anchor Boxes) for 2D object detection. Transformer-based detector with
  denoising training, multi-scale features, and optional distillation support. Use when training, evaluating, exporting,
  distilling, quantizing, or running inference for a TAO DINO detector. Trigger phrases include "train DINO", "DETR object
  detection", "TAO 2D detection", "DINO with distillation".
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

# DINO

DINO (DETR with Improved DeNoising Anchor Boxes) for 2D object detection. Transformer-based detector with denoising training, multi-scale features, and optional distillation support.

Uses pretrained backbone weights (e.g. ResNet-50 ImageNet). Set `model.pretrained_backbone_path` for backbone-only or `train.pretrained_model_path` for full model.

## When To Use

Train, evaluate, export, distill, quantize, or run inference for a TAO DINO 2D object detector.

For TAO Deploy TensorRT actions (`gen_trt_engine`, TensorRT `evaluate`, and
TensorRT `inference`), read `references/tao-deploy-dino.md` first. Deploy spec templates live
in this skill's `references/` folder with the `spec_template_deploy_*.yaml`
prefix.

## Reference Map

- `references/dino-data-specs.md` — dataset contracts, per-action dataset requirements, per-action spec-override examples (train, evaluate, export, deploy/gen_trt_engine, inference, quantize, distill), data-source arrays, checkpoint inference, and dataset layout.
- `references/dino-actions-errors.md` — important parameters, default values, evaluate/export defaults, hardware, and the full error-pattern catalog.
- `references/dino-tuning-multigpu.md` — full AutoML/HPO notes (metrics, hyperparameters, extractor) and multi-GPU spec consistency.
- `references/dino-automl-sdk.md` — AutoML metrics, SDK orchestration internals, data-source gap, and spec-param/parent-model inference.
- `references/tao-deploy-dino.md` — TensorRT deploy workflow.
- `references/detailed-guide.md` — map to the detailed model guide.

## Dataclass Schemas

Generated TAO Core schemas are packaged in `schemas/<action>.schema.json`, with `schemas/manifest.json` listing available actions. Each generated schema also emits `references/spec_template_<action>.yaml` from the schema top-level `default` field. AutoML enablement is declared at the model layer in `references/skill_info.yaml` via `automl_enabled`. Runnable AutoML still requires `schemas/train.schema.json` and `references/spec_template_train.yaml` to exist and parse. Use the packaged train schema for `automl_default_parameters`, `automl_disabled_parameters`, defaults, min/max bounds, enums, option weights, math conditions, dependencies, and popular parameters. Do not expect `~/tao-core` at runtime; maintainers regenerate schemas/templates before packaging the skill bank.

## Train Action Policy

This model is AutoML-enabled at the model layer. Before handling any train-stage request, read `references/skill_info.yaml` and resolve the run override from either an explicit `automl_policy` value or the user's workflow request. Use `automl_policy: on` by default and only expose `on` / `off` in new launch prompts. Treat phrases like "turn off AutoML", "disable AutoML", "no HPO", or "plain training" as `automl_policy: off` for this run only. When `automl_policy: on`, `automl_enabled: true`, and both `schemas/train.schema.json` and `references/spec_template_train.yaml` are packaged, route the train action through `tao-skill-bank:tao-run-automl` by default with this model's `skill_dir`. Preserve workflow/application overrides for datasets, specs, output directories, GPU/platform settings, parent checkpoints, and `automl_policy`. Use direct model training only when `automl_policy: off` or the packaged train schema/template is missing; in the missing-schema case, report that AutoML is enabled but not runnable for this model until schemas are generated.

Non-train actions such as `evaluate`, `inference`, `export`, and deploy flows stay in this model skill. The per-run `automl_policy` override does not change model metadata.

## Training Requirements

The agent MUST read this section before generating any training or AutoML script for DINO.

- **Dataset type:** object_detection
- **Formats:** coco, coco_raw
- **Accepted dataset intents:** training, evaluation, testing, calibration
- **Monitoring metric:** mAP50 for quick operational checks; `val_mAP` for
  COCO/paper-style benchmark comparisons.

**Required datasets — MUST resolve both:**

| Dataset | Required | Why |
|---|---|---|
| Train dataset URI | Yes | Training data (COCO format) |
| Validation dataset URI | **Yes — ALWAYS** | DINO unconditionally builds a val dataloader. Omitting `val_data_sources` causes `FileNotFoundError` at startup regardless of the metric or workflow. If the user has no separate eval split, reuse the train URI. |

**Required inputs before generating any training spec:**

1. **Train dataset URI** — S3 path to COCO-format training data
2. **Validation dataset URI** — S3 path to COCO-format val data (can be same as train)
3. **`num_classes`** — How many object classes? Default 91 (COCO). Must be >= `max(category_id) + 1`. Too low causes `CUDA error: device-side assert triggered`.

Resolve these from the user request or the default profile below. Prompt only
for values that are still missing after applying the profile rules.

**Bankable local default profile for DINO AutoML smoke runs:**

Use this profile only when the user asks to run DINO AutoML and does not provide
dataset or class-count inputs. This profile is intentionally small and local to
this skill bank; it is for smoke/iteration runs, not a production benchmark.
Do not search previous runners, logs, session state, shell history, or the home
directory to recover these values.

```python
DINO_AUTOML_PROFILE = {
    "train_dataset_uri": "s3://nvcf-storage-handling/data/tao_od_synthetic_subset_train_no_convert",
    "validation_dataset_uri": "s3://nvcf-storage-handling/data/tao_od_synthetic_subset_val_no_convert",
    "object_classes": 4,
    "dataset_num_classes": 5,
    "image_archive": "images.tar.gz",
    "annotation_file": "annotations.json",
    "max_recommendations": 10,
    "train_num_epochs": 10,
    "train_checkpoint_interval": 10,
    "train_validation_interval": 1,
    "train_num_gpus": 1,
}
```

If the user supplies any dataset URI or class-count value, prefer the user value
and ask for any remaining required DINO value. Do not partially mix a user's
custom dataset with this profile's class count unless the user confirms it.

**Do not prompt for image layout for the standard DINO dataset.** The standard
TAO DINO dataset artifact is `images.tar.gz` plus `annotations.json`. Use
`images.tar.gz` in the remote `image_dir` spec override. The SDK downloads the
archive and rewrites the runtime spec to the extracted folder named after the
archive stem (`images.tar.gz` -> `images`). Only deviate if the user explicitly
provides a different image artifact name.

## Core Workflow

DINO supports train, evaluate, export, distill, quantize, and inference. Data-source
overrides are **mandatory for every action** — DINO's `config.json` has empty
`data_sources` because the runner cannot auto-resolve array-of-objects spec keys.
The agent MUST construct data source paths and include them in `spec_overrides`.

See `references/dino-data-specs.md` for the per-action dataset requirements table,
the standard dataset artifact (`images.tar.gz` + `annotations.json`) and runtime
folder rewrite rules, and the complete per-action `spec_overrides` examples for
train, evaluate, export, deploy/gen_trt_engine, inference, quantize, and distill —
including checkpoint inference via `parent_model`, the `results_dir/train/`
checkpoint location, and the distillation FAN-teacher / student rules.

## Important Parameters And Defaults

Key defaults: `num_epochs=10`, `batch_size=4`, `learning_rate=2e-4`,
`lr_backbone=2e-5`, `num_classes=91`, `backbone=resnet_50`.

- **dataset.num_classes**: Default 91 (COCO). Must be >= `max(category_id) + 1`. Too low causes `CUDA error: device-side assert triggered`. Set as `<num_classes> + 1` in spec overrides.
- **num_epochs**: default 10 (quick iteration); real datasets typically need 30-50+ epochs for good mAP.

See `references/dino-actions-errors.md` for the full parameter list (backbone
options, `train.optim.lr`/`lr_steps`, `model.num_queries`, `batch_size`),
default values, evaluate defaults, export defaults (input 960x544, opset 17,
TRT data types, workspace 1024 MB), and hardware requirements.

## Multi-GPU And AutoML / HPO

When increasing `train.num_gpus`, also set `train.gpu_ids` to the same visible
device range, or distributed startup can be inconsistent.

AutoML runs training — all **Training Requirements** above apply. For no-input
local smoke runs, use `DINO_AUTOML_PROFILE`. Recommended metric is `mAP50`
(`val_mAP` for benchmark comparisons) with `direction="maximize"` and a custom
`metric_extractor`.

See `references/dino-tuning-multigpu.md` for the full multi-GPU spec-consistency
rule (8-GPU example, NCCL timeout note) and the full AutoML/HPO notes (metric
selection, `metric_extractor`, recommended hyperparameters, `weight_decay`
behavior, dense-dataset resume guidance). See `references/dino-automl-sdk.md` for
AutoML metric extractor code, SDK orchestration internals, and parent-model
inference mappings.

## Error Patterns

Common failures include CUDA OOM (reduce `batch_size`), missing `val_data_sources`
(`FileNotFoundError` at startup — always supply val), `num_classes` too low (`CUDA
device-side assert`), and the parent `dino gen_trt_engine` / `dino convert` PyT-CLI
restrictions.

See `references/dino-actions-errors.md` for the complete error-pattern catalog
with diagnostics and fixes.

## Spec Param / Parent Model Inference

Model-specific inference mappings belong in this MD file, not in `config.json`.
Generated runners read the mappings and apply them with SDK helpers before
`create_job()`. For `parent_model`/`parent_model_folder`, pass the upstream
train/export/AutoML child job id as `parent_job_id`; the SDK lists the parent
result folder, filters checkpoint artifacts, and returns the selected model.

See `references/dino-automl-sdk.md` for the full inference-mapping table (per
action: `parent_model`, `key`, `output_dir`, `ptm_if_no_resume_model`,
`resume_model`, `create_onnx_file`) and the TensorRT-mapping note. TensorRT
mappings live in the deploy workflow, not the PyT model skill.

## Optional: running via the TAO SDK

When running DINO through the TAO SDK (`script_runner` orchestration, S3 I/O
wrapping, AutoML), skills read `references/skill_info.yaml` for input and
spec-param mappings. See `references/dino-automl-sdk.md` for SDK orchestration
internals, including the data-sources gap and the `[0]`-indexed `inputs`
declarations. Skip this when running locally with `docker run`.

## Deployment

- [tao-deploy-dino](references/tao-deploy-dino.md)
