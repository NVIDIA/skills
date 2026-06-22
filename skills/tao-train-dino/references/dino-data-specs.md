# DINO Data And Specs

Dataset contracts, per-action data requirements, spec override examples, and dataset layout details from the pre-refactor guide.

Load this file only when the compact `SKILL.md` points here for the current task. If this reference conflicts with `SKILL.md`, `skill_info.yaml`, schemas, or platform/model skills, the compact/current source wins.

## Contents

- DINO
- Dataclass Schemas
- Train Action Policy
- Training Requirements
- Per-Action Dataset Requirements
- Typical Spec Overrides
- Standard DINO dataset artifact. Pass the archive path as the remote input.
- At runtime the SDK extracts it and points DINO at the extracted "images" folder.
- Dataset
- Train Data Sources
- Val Data Sources (ALWAYS required)
- Inference Data Sources
- Evaluate Data Sources

# DINO Detailed Guide

Preserves the detailed model guide from release/7.0.1. Read this when the compact SKILL.md does not contain enough detail for action-specific spec overrides, data-source arrays, checkpoint inference, AutoML metric extraction, or SDK orchestration internals. If this reference conflicts with the compact `SKILL.md`, packaged metadata, or platform skills, the compact/current source wins.

## Contents

- Original workflow and operating rules
- Detailed command snippets and examples
- Error handling and troubleshooting notes
- Reporting, handoff, and validation details


# DINO

DINO (DETR with Improved DeNoising Anchor Boxes) for 2D object detection. Transformer-based detector with denoising training, multi-scale features, and optional distillation support.

Uses pretrained backbone weights (e.g. ResNet-50 ImageNet). Set `model.pretrained_backbone_path` for backbone-only or `train.pretrained_model_path` for full model.

For TAO Deploy TensorRT actions (`gen_trt_engine`, TensorRT `evaluate`, and
TensorRT `inference`), read `tao-deploy-dino.md` first. Deploy spec templates live
in this skill's `references/` folder with the `spec_template_deploy_*.yaml`
prefix.

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
- **Monitoring metric:** mAP50

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

### Per-Action Dataset Requirements

| Action | Spec Key | Source | Files | List? |
|---|---|---|---|---|
| distill | dataset.train_data_sources | train_datasets | image_dir: images.tar.gz, json_file: annotations.json | Yes |
| distill | dataset.val_data_sources | train_datasets | image_dir: images.tar.gz, json_file: annotations.json | Yes |
| evaluate | evaluate.checkpoint | trained_model | DINO .pth/.tlt checkpoint | No |
| evaluate | dataset.test_data_sources.image_dir | eval_dataset | images.tar.gz | No |
| evaluate | dataset.test_data_sources.json_file | eval_dataset | annotations.json | No |
| deploy/gen_trt_engine | gen_trt_engine.tensorrt.calibration.cal_image_dir | calibration_dataset | images.tar.gz | Yes |
| inference | dataset.infer_data_sources.image_dir | inference_dataset | images.tar.gz | Yes |
| inference | dataset.infer_data_sources.classmap | inference_dataset | label_map.txt | No |
| quantize | dataset.train_data_sources | train_datasets | image_dir: images.tar.gz, json_file: annotations.json | Yes |
| quantize | dataset.val_data_sources | train_datasets | image_dir: images.tar.gz, json_file: annotations.json | Yes |
| quantize | dataset.quant_calibration_data_sources | train_datasets | image_dir: images.tar.gz, json_file: annotations.json | No |
| train | dataset.train_data_sources | train_datasets | image_dir: images.tar.gz, json_file: annotations.json | Yes |
| train | dataset.val_data_sources | train_datasets | image_dir: images.tar.gz, json_file: annotations.json | Yes |

### Typical Spec Overrides

Data source overrides are **mandatory for every action** — DINO's `config.json` has empty `data_sources` because the runner cannot auto-resolve array-of-objects spec keys (see Internal Details). The agent MUST construct data source paths from the Per-Action Dataset Requirements table above and include them in `spec_overrides`.

```python
S3_TRAIN = "s3://bucket/data/train"
S3_VAL = "s3://bucket/data/val"    # can be same as S3_TRAIN
S3_EVAL = "s3://bucket/data/eval"  # for evaluate/inference

# Standard DINO dataset artifact. Pass the archive path as the remote input.
# At runtime the SDK extracts it and points DINO at the extracted "images" folder.
IMAGE_ARCHIVE = "images.tar.gz"
```

**train (mandatory):**
```python
{
    "dataset.train_data_sources": [
        {"image_dir": f"{S3_TRAIN}/{IMAGE_ARCHIVE}", "json_file": f"{S3_TRAIN}/annotations.json"}
    ],
    "dataset.val_data_sources": [
        {"image_dir": f"{S3_VAL}/{IMAGE_ARCHIVE}", "json_file": f"{S3_VAL}/annotations.json"}
    ],
    "dataset.num_classes": "<num_classes> + 1",
    "train.num_epochs": 10,
    "train.checkpoint_interval": 10,
    "train.validation_interval": 10,
    "train.num_gpus": 1,
}
```

**evaluate (mandatory checkpoint + data sources):**
```python
{
    "evaluate.checkpoint": "<checkpoint_uri>",
    "dataset.test_data_sources.image_dir": f"{S3_EVAL}/{IMAGE_ARCHIVE}",
    "dataset.test_data_sources.json_file": f"{S3_EVAL}/annotations.json",
    "dataset.num_classes": "<num_classes> + 1",
    "model.backbone": "<backbone used for training>",
    "model.num_queries": "<num_queries used for training>",
    "model.dropout_ratio": "<dropout_ratio used for training>",
}
```

For standard DINO eval datasets, do not search S3 to discover filenames. Build
the eval image and annotation URIs directly from the eval dataset base URI using
`images.tar.gz` and `annotations.json`, unless the user explicitly provides a
different layout.

For a DINO model trained by this SDK or by an AutoML child train job, prefer
microservices-style parent model inference instead of hardcoding the checkpoint
URI. Use this model-MD inference mapping:

```json
"spec_params": {
  "evaluate": {
    "evaluate.checkpoint": "parent_model"
  }
}
```

Use the train job id, or the AutoML best child train job id, as
`parent_job_id`. The SDK will list the parent result folder, filter `.pth`
checkpoints, and select the model file:

```python
checkpoint_uri = sdk.resolve_spec_param(
    eval_job_id,
    "parent_model",
    network_arch="dino",
    parent_job_id=train_job_id,
)
```

Equivalently, when resolving the checkpoint outside a spec-param loop:

```python
checkpoint_uri = sdk.get_model_results_path(train_job_id, network_arch="dino")
```

If cloud listing is unavailable but only the training job id is known, list the
training job result folder and choose the intended epoch/step checkpoint under:

```python
checkpoint_prefix = f"s3://{S3_BUCKET_NAME}/results/{train_job_id}/results_dir/train/"
```

Do not use `s3://<bucket>/results/<train_job_id>/dino_model_latest.pth`; DINO
training uploads checkpoints under `results_dir/train/`. The
`dino_model_latest.pth` symlink under that folder is valid only when latest is
explicitly requested.

When evaluating an AutoML-trained model, carry forward the winning rec's
structural model settings into the eval spec. At minimum copy
`model.backbone`, `model.num_queries`, `model.dropout_ratio`, and
`dataset.num_classes`. If future HPO runs tune additional structural model
fields, copy those too so the checkpoint shape matches the evaluation model.

**export:**
```python
{
    "export.checkpoint": "<checkpoint_uri>",
    "export.onnx_file": "<output_onnx_path>",
    "dataset.num_classes": "<num_classes> + 1",
}
```

**deploy/gen_trt_engine (use `tao-deploy-dino.md`):**
```python
{
    "gen_trt_engine.onnx_file": "<exported_onnx_uri>",
    "gen_trt_engine.trt_engine": "<output_engine_path>",
    "gen_trt_engine.tensorrt.calibration.cal_image_dir": [f"{S3_TRAIN}/{IMAGE_ARCHIVE}"],
    "gen_trt_engine.tensorrt.data_type": "FP16",
    "dataset.num_classes": "<num_classes> + 1",
}
```

For deploy TensorRT evaluation, also read `tao-deploy-dino.md`; the deploy metric
path expects at least 100 selected detections per image.

**inference (mandatory data sources):**
```python
{
    "dataset.infer_data_sources.image_dir": [f"{S3_EVAL}/{IMAGE_ARCHIVE}"],
    "dataset.infer_data_sources.classmap": f"{S3_EVAL}/label_map.txt",
    "dataset.num_classes": "<num_classes> + 1",
}
```

**quantize (mandatory data sources):**
```python
{
    "dataset.train_data_sources": [
        {"image_dir": f"{S3_TRAIN}/{IMAGE_ARCHIVE}", "json_file": f"{S3_TRAIN}/annotations.json"}
    ],
    "dataset.val_data_sources": [
        {"image_dir": f"{S3_VAL}/{IMAGE_ARCHIVE}", "json_file": f"{S3_VAL}/annotations.json"}
    ],
    "dataset.quant_calibration_data_sources": {
        "image_dir": f"{S3_TRAIN}/{IMAGE_ARCHIVE}", "json_file": f"{S3_TRAIN}/annotations.json"
    },
    "dataset.num_classes": "<num_classes> + 1",
}
```

**distill (mandatory data sources):**
```python
{
    "distill.pretrained_teacher_model_path": "<fan_teacher_checkpoint_uri>",
    "distill.teacher.backbone": "fan_tiny",
    "distill.bindings": [
        {"student_module_name": "pred_logits", "teacher_module_name": "pred_logits", "criterion": "L2", "weight": 1.0},
        {"student_module_name": "pred_boxes", "teacher_module_name": "pred_boxes", "criterion": "L1", "weight": 1.0},
    ],
    "dataset.train_data_sources": [
        {"image_dir": f"{S3_TRAIN}/{IMAGE_ARCHIVE}", "json_file": f"{S3_TRAIN}/annotations.json"}
    ],
    "dataset.val_data_sources": [
        {"image_dir": f"{S3_VAL}/{IMAGE_ARCHIVE}", "json_file": f"{S3_VAL}/annotations.json"}
    ],
    "dataset.num_classes": "<num_classes> + 1",
}
```

DINO distillation uses a FAN-family teacher (`fan_tiny`, `fan_small`,
`fan_base`, or `fan_large`) and a supported student such as `resnet_50`. The
teacher checkpoint must match the teacher architecture. Do not point
`distill.pretrained_teacher_model_path` at a ResNet training checkpoint unless
`distill.teacher.backbone` is also a compatible ResNet teacher in a future SDK.

## Dataset

COCO JSON format. train_data_sources and val_data_sources are lists supporting multiple data source entries. Each entry has image_dir and json_file (COCO annotations JSON).

**`image_dir` remote path**: For the standard TAO DINO dataset, set
`image_dir` to the archive path, e.g. `s3://bucket/data/images.tar.gz`.
The SDK downloads and extracts it, then rewrites the runtime training spec to
the extracted folder path, e.g. `/mnt/lustre/.../images`.

Do not ask the user whether to use `images` or `images.tar.gz` for standard
DINO datasets. Use `images.tar.gz`. If the user explicitly supplies a different
archive filename, derive the runtime folder from the archive stem:
`<name>.tar.gz` -> `<name>`, `<name>.tgz` -> `<name>`, `<name>.tar` -> `<name>`.

Supported formats: coco, coco_raw.

### Train Data Sources

- **image_dir**: `images.tar.gz` remote archive; runtime folder is `images`
- **json_file**: `annotations.json`

### Val Data Sources (ALWAYS required)

- **image_dir**: `images.tar.gz` remote archive; runtime folder is `images`
- **json_file**: `annotations.json`

### Inference Data Sources

- **image_dir**: `images.tar.gz` remote archive; runtime folder is `images`
- **classmap**: `label_map.txt`

### Evaluate Data Sources

- **checkpoint**: `evaluate.checkpoint`, a `.pth` or `.tlt` model file. For SDK
  train jobs and AutoML child train jobs, resolve it with `parent_model`
  inference so the SDK lists the result folder and selects an actual checkpoint
  file. Prefer concrete epoch/step files such as
  `results_dir/train/model_epoch_000_step_00025.pth`. Use
  `dino_model_latest.pth` only when the user explicitly requests the latest
  checkpoint; it is a symlink alias and should not replace best/specific
  checkpoint resolution.
- **image_dir**: `images.tar.gz` remote archive; runtime folder is `images`
- **json_file**: `annotations.json`
