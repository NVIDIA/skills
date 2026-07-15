---
name: tao-train-visual-changenet
description: Visual ChangeNet for binary image classification and segmentation in AOI defect detection. Use when training,
  evaluating, exporting, or running inference for PCB defect detection or visual inspection, comparing image pairs for
  PASS/NO_PASS classification, or producing change-segmentation masks. Trigger phrases include "train Visual ChangeNet",
  "ChangeNet classify", "ChangeNet segment", "AOI defect detection", "PCB inspection model".
license: Apache-2.0
compatibility: Requires docker + nvidia-container-toolkit.
metadata:
  author: NVIDIA Corporation
  version: "0.1.0"
allowed-tools: Read Bash
tags:
- pcb
- aoi
- defect
- classification
- segmentation
- siamese
- visual-inspection
---

# Visual ChangeNet

Visual ChangeNet is a TAO Toolkit model for visual inspection and defect detection. It supports two tasks:

- **Classify** — Binary image classification using a siamese-style architecture with a shared backbone (C-RADIO ViT) and a learnable difference module. Compares image pairs to classify defects as PASS/NO_PASS.
- **Segment** — Pixel-level change segmentation using a ViT-Large NVDINOv2 backbone. Compares before/after image pairs to produce a binary change mask.

The backbone weight (`c_radio_v2_vit_base_patch16_224`) is the `nvidia/C-RADIOv2-B` model from HuggingFace, distributed as `model.safetensors` (~393 MB). **The TAO 7.0.0-rc container does not auto-fetch from HF URLs** — `ptm_utils.load_pretrained_weights()` hands the `pretrained_backbone_path` value to `torch.load(path)` / `safetensors.torch.load_file(path)` directly. Passing an `https://huggingface.co/...` URL or a repo id produces `FileNotFoundError` and the run fails with `Execution status: FAIL` within a few seconds. Stage the file locally before launch:

```bash
python3 -c "from huggingface_hub import hf_hub_download; import shutil; \
shutil.copy(hf_hub_download('nvidia/C-RADIOv2-B', 'model.safetensors'), '<workspace>/backbone/c_radio_v2_b.safetensors')"
```

Mount it into the container (`-v <workspace>/backbone/c_radio_v2_b.safetensors:/data/pretrained_models/C-RADIOv2_B.safetensors`) and set the spec `model.backbone.pretrained_backbone_path` to the container path. `HF_TOKEN` is only needed at staging time, not at training time.

Segment specs use `model.backbone.type: vit_large_nvdinov2` and the NVDINOv2
checkpoint family. Keep the checkpoint architecture aligned with the backbone
type: `NV_DINOV2_518_16_256.ckpt` is compatible with the packaged segment
templates, but it must not be used with `fan_small_12_p4_hybrid`. If you switch
to a different segment backbone, use a matching checkpoint or leave
`model.backbone.pretrained_backbone_path` empty for default initialization.

## Dataclass Schemas

Generated TAO Core schemas are packaged in `schemas/<action>.schema.json`, with `schemas/manifest.json` listing available actions. Each generated schema also emits `references/spec_template_<action>.yaml` from the schema top-level `default` field. AutoML enablement is declared at the model layer in `references/skill_info.yaml` via `automl_enabled`. Runnable AutoML still requires `schemas/train.schema.json` and `references/spec_template_train.yaml` to exist and parse. Use the packaged train schema for `automl_default_parameters`, `automl_disabled_parameters`, defaults, min/max bounds, enums, option weights, math conditions, dependencies, and popular parameters. Do not expect `~/tao-core` at runtime; maintainers regenerate schemas/templates before packaging the skill bank.

## Train Action Policy

This model is AutoML-enabled at the model layer. Before handling any train-stage request, read `references/skill_info.yaml` and resolve the run override from either an explicit `automl_policy` value or the user's workflow request. Use `automl_policy: on` by default and only expose `on` / `off` in new launch prompts. Treat phrases like "turn off AutoML", "disable AutoML", "no HPO", or "plain training" as `automl_policy: off` for this run only. When `automl_policy: on`, `automl_enabled: true`, and both `schemas/train.schema.json` and `references/spec_template_train.yaml` are packaged, route the train action through `tao-skill-bank:tao-run-automl` by default with this model's `skill_dir`. Preserve workflow/application overrides for datasets, specs, output directories, GPU/platform settings, parent checkpoints, and `automl_policy`. Use direct model training only when `automl_policy: off` or the packaged train schema/template is missing; in the missing-schema case, report that AutoML is enabled but not runnable for this model until schemas are generated.

Non-train actions declared by this model skill (`evaluate`, `inference`,
`export`, `quantize`, `segment_evaluate`, and `segment_inference`) stay in this
model skill. Do not present `segment_export` or `segment_quantize` as runnable
parent-skill actions until matching entries are packaged in
`schemas/manifest.json`. Prune and retrain are not declared in the current
parent `references/skill_info.yaml`; do not present them as runnable parent-skill
actions unless the metadata is extended with matching action wiring and schemas.
The per-run `automl_policy` override does not change model metadata.

For TAO Deploy TensorRT actions (`gen_trt_engine`, TensorRT `evaluate`, and TensorRT `inference` for classify and segment variants), read `references/tao-deploy-visual-changenet.md` first. Deploy spec templates live in this skill's `references/` folder with the `spec_template_deploy_*.yaml` prefix.
Deploy requires an exported ONNX artifact as `parent_model`. If no ONNX artifact exists and the main skill does not expose an export action, report deploy as blocked instead of inventing an artifact.

## Training Requirements

Visual ChangeNet has two separate task modes with different dataset types and data source structures.

### Classify

- **Dataset type:** visual_changenet_classify
- **Formats:** default
- **Accepted dataset intents:** training, evaluation, testing, calibration
- **Monitoring metric:** val_loss

#### Per-Action Dataset Requirements (Classify)

The `quantize` and `gen_trt_engine` rows below describe TAO spec data requirements only. They are not parent-skill actions unless the corresponding action is declared in `references/skill_info.yaml` or `deploy/skill_info.yaml`.

| Action | Spec Key | Source | Files | List? |
|---|---|---|---|---|
| train | dataset.classify.train_dataset.images_dir | train_datasets | images.tar.gz | No |
| train | dataset.classify.train_dataset.csv_path | train_datasets | dataset.csv | No |
| train | dataset.classify.validation_dataset.images_dir | eval_dataset | images.tar.gz | No |
| train | dataset.classify.validation_dataset.csv_path | eval_dataset | dataset.csv | No |
| quantize | dataset.classify.train_dataset.images_dir | train_datasets | images.tar.gz | No |
| quantize | dataset.classify.train_dataset.csv_path | train_datasets | dataset.csv | No |
| quantize | dataset.classify.validation_dataset.images_dir | eval_dataset | images.tar.gz | No |
| quantize | dataset.classify.validation_dataset.csv_path | eval_dataset | dataset.csv | No |
| quantize | dataset.classify.quant_calibration_dataset.images_dir | train_datasets | images.tar.gz | No |
| evaluate | dataset.classify.validation_dataset.images_dir | eval_dataset | images.tar.gz | No |
| evaluate | dataset.classify.validation_dataset.csv_path | eval_dataset | dataset.csv | No |
| evaluate | dataset.classify.test_dataset.images_dir | eval_dataset | images.tar.gz | No |
| evaluate | dataset.classify.test_dataset.csv_path | eval_dataset | dataset.csv | No |
| inference | dataset.classify.infer_dataset.images_dir | inference_dataset | images.tar.gz | No |
| inference | dataset.classify.infer_dataset.csv_path | inference_dataset | dataset.csv | No |
| gen_trt_engine | gen_trt_engine.tensorrt.calibration.cal_image_dir | calibration_dataset | images.tar.gz | Yes |

### Segment

- **Dataset type:** visual_changenet_segment
- **Formats:** default
- **Accepted dataset intents:** training, calibration
- **Monitoring metric:** val_loss

Segment uses a paired directory structure (`A/`, `B/`, `list/`, `label/`) instead of CSV + images. The `root_dir` spec key points to the top-level directory containing all four subdirectories.

**Required files per dataset:** `A.tar.gz`, `B.tar.gz`, `list.tar.gz`, `label.tar.gz`

#### Per-Action Dataset Requirements (Segment)

The `quantize` and `gen_trt_engine` rows below describe TAO spec data requirements only. They are not parent-skill actions unless the corresponding action is declared in `references/skill_info.yaml` or `deploy/skill_info.yaml`.

| Action | Spec Key | Source | Files | List? |
|---|---|---|---|---|
| train | dataset.segment.root_dir | train_datasets | (root directory) | No |
| quantize | dataset.segment.root_dir | train_datasets | (root directory) | No |
| quantize | dataset.segment.quant_calibration_dataset.images_dir | train_datasets | (root directory) | No |
| evaluate | dataset.segment.root_dir | train_datasets | (root directory) | No |
| inference | dataset.segment.root_dir | train_datasets | (root directory) | No |
| gen_trt_engine | dataset.segment.root_dir | train_datasets | (root directory) | No |
| gen_trt_engine | gen_trt_engine.tensorrt.calibration.cal_image_dir | calibration_dataset | images.tar.gz | Yes |

### Typical Spec Overrides

Data source overrides are **mandatory for every action** — the agent MUST construct data source paths from the Per-Action Dataset Requirements table above and include them in `spec_overrides`.

```python
S3_TRAIN = "s3://bucket/data/train"
S3_EVAL = "s3://bucket/data/eval"
```

**train (classify, mandatory data sources):**
```python
{
    "train.num_epochs": 30,
    "train.checkpoint_interval": 10,
    "train.validation_interval": 10,
    "train.num_gpus": 1,
    "train.use_distributed_sampler": False,
    "train.sync_batchnorm": False,
    "dataset.classify.train_dataset.images_dir": f"{S3_TRAIN}/images.tar.gz",
    "dataset.classify.train_dataset.csv_path": f"{S3_TRAIN}/dataset.csv",
    "dataset.classify.validation_dataset.images_dir": f"{S3_EVAL}/images.tar.gz",
    "dataset.classify.validation_dataset.csv_path": f"{S3_EVAL}/dataset.csv",
}
```

**train (segment, mandatory data sources):**
```python
{
    "train.num_epochs": 30,
    "train.checkpoint_interval": 10,
    "train.validation_interval": 10,
    "train.num_gpus": 1,
    "train.use_distributed_sampler": False,
    "train.sync_batchnorm": False,
    "dataset.segment.root_dir": f"{S3_TRAIN}",
}
```

**export (classify):**
```python
{
    "export.input_height": 896,
    "export.input_width": 224,
}
```

**export (segment):**
```python
{
    "export.input_height": 224,
    "export.input_width": 224,
}
```

**quantize (classify, mandatory data sources):**
```python
{
    "dataset.classify.train_dataset.images_dir": f"{S3_TRAIN}/images.tar.gz",
    "dataset.classify.train_dataset.csv_path": f"{S3_TRAIN}/dataset.csv",
    "dataset.classify.validation_dataset.images_dir": f"{S3_EVAL}/images.tar.gz",
    "dataset.classify.validation_dataset.csv_path": f"{S3_EVAL}/dataset.csv",
    "dataset.classify.quant_calibration_dataset.images_dir": f"{S3_TRAIN}/images.tar.gz",
}
```

**evaluate (classify, mandatory data sources):**
```python
{
    "dataset.classify.validation_dataset.images_dir": f"{S3_EVAL}/images.tar.gz",
    "dataset.classify.validation_dataset.csv_path": f"{S3_EVAL}/dataset.csv",
    "dataset.classify.test_dataset.images_dir": f"{S3_EVAL}/images.tar.gz",
    "dataset.classify.test_dataset.csv_path": f"{S3_EVAL}/dataset.csv",
}
```

**inference (classify, mandatory data sources):**
```python
{
    "dataset.classify.infer_dataset.images_dir": f"{S3_EVAL}/images.tar.gz",
    "dataset.classify.infer_dataset.csv_path": f"{S3_EVAL}/dataset.csv",
}
```

**gen_trt_engine (classify, mandatory data sources):**
```python
{
    "gen_trt_engine.tensorrt.calibration.cal_image_dir": [f"{S3_TRAIN}/images.tar.gz"],
}
```

**quantize (segment, mandatory data sources):**
```python
{
    "dataset.segment.root_dir": f"{S3_TRAIN}",
    "dataset.segment.quant_calibration_dataset.images_dir": f"{S3_TRAIN}",
}
```

**evaluate (segment, mandatory data sources):**
```python
{
    "dataset.segment.root_dir": f"{S3_TRAIN}",
}
```

**inference (segment, mandatory data sources):**
```python
{
    "dataset.segment.root_dir": f"{S3_TRAIN}",
}
```

**gen_trt_engine (segment, mandatory data sources):**
```python
{
    "dataset.segment.root_dir": f"{S3_TRAIN}",
    "gen_trt_engine.tensorrt.calibration.cal_image_dir": [f"{S3_TRAIN}/images.tar.gz"],
}
```
## Optional: running via the TAO SDK

When running without the TAO SDK (local docker), resolve the TAO pyt image from `versions.yaml` and invoke `visual_changenet <train|evaluate|inference|export|quantize>` directly. `--shm-size=8g` is required, the C-RADIO `.safetensors` must be mounted to `/data/pretrained_models/C-RADIOv2_B.safetensors`, and checkpoint/results_dir can be overridden on the command line. See `references/local-docker.md` for the full `docker run` command, mounts, and overrides.

## Tasks

### Classify (default)

Uses actions: `train`, `evaluate`, `inference`. Defaults template: `references/spec_template_train.yaml`.

### Segment

Uses skill action names `segment_train`, `segment_evaluate`, and
`segment_inference`. When invoking local Docker directly, run TAO CLI subcommands
`train`, `evaluate`, and `inference` with `task: segment` in the spec. The
schema-driven action templates are `references/spec_template_segment_train.yaml`,
`references/spec_template_segment_evaluate.yaml`, and
`references/spec_template_segment_inference.yaml`; the compact direct-Docker
example template is `references/spec_template_segment.yaml`.

Segmentation requires compiling custom CUDA ops (`MultiScaleDeformableAttention`) on first run, which takes ~5 minutes. The ViT adapter backbone uses these for multi-scale feature extraction.

Dataset structure for segmentation differs from classify — uses paired directories (`A/`, `B/`, `list/`, `label/`) instead of CSV files. See `dataset.segment.root_dir` in the defaults.

## Data Format

Classify needs a 4-column CSV (`input_path,golden_path,label,object_name`) plus an images directory; segment uses a paired directory structure (`A/`, `B/`, `list/`, `label/`) under `dataset.segment.root_dir` instead of CSV. The `image_ext` field (default `.jpg`) must match the actual file extensions; if images are `.png`, set `dataset.classify.image_ext: .png`. Multi-lighting input is configured via `dataset.classify.input_map` (each lighting name maps to a channel index) with `dataset.classify.num_input` set to match. See `references/data-formats.md` for the per-field input tables (classify train/eval/inference, segment), CSV column semantics, lighting/path-concatenation conventions, the segment directory layout, and `input_map`/`grid_map` examples.

## Important Parameters

Key knobs include `train.validation_interval` (default 50, must be ≤ num_epochs), `train.checkpoint_interval` (default 200, must be ≤ num_epochs), `train.num_epochs` (default 100), `model.classify.eval_margin` (default 0.3, the precision/recall threshold), `model.classify.train_margin_euclid` (default 2.0), `model.classify.embedding_vectors` (default 5), `dataset.classify.batch_size` (default 16, must be > 1), `dataset.classify.fpratio_sampling` (default 0.25), and `train.classify.cls_weight` (default [1.0, 10.0]). Hardware: minimum 1 GPU with 16GB+ VRAM, recommended 8 GPUs (DDP); do not set `gpu_spec_key` (GPU count is managed internally by TAO), `num_nodes` (default 1) controls multi-node. See `references/tuning-parameters.md` for the full per-parameter guidance and hardware detail.

## Error Patterns

For checkpoint-not-found, CSV format mismatch, image extension mismatch, OOM, low evaluation accuracy, the contrastive-loss `AssertionError`, checkpoint load key mismatch at evaluate/inference, non-convergence, segment-only backbone dimension mismatch, the `MultiScaleDeformableAttention` `OSError`, the Lightning `MisconfigurationException`, `ModuleNotFoundError: nvidia_tao_pytorch`, and epoch defaults, see `references/troubleshooting.md` for the full symptom-and-fix list.

## Spec Param / Parent Model Inference

Model-specific parent-model mappings are declared in `references/skill_info.yaml` under `spec_params`, so generated runners and agents resolve checkpoints before `create_job()` instead of guessing file names. For `parent_model` or `parent_model_folder`, pass the upstream train/export/AutoML child job id as `parent_job_id`; the SDK lists the parent result folder, filters checkpoint artifacts, and returns the selected model file or folder. See `references/parent-model-inference.md` for the full per-action spec-field-to-inference-function mapping table.

## Deployment

- [tao-deploy-visual-changenet](references/tao-deploy-visual-changenet.md)
