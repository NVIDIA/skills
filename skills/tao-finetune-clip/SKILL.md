---
name: tao-finetune-clip
description: CLIP vision-language model for image-text retrieval, zero-shot classification, embedding extraction, ONNX
  export, and TensorRT deployment. Use when fine-tuning or training CLIP, running zero-shot classification, computing image
  embeddings, or deploying CLIP to ONNX/TensorRT.
license: Apache-2.0
compatibility: Requires docker + nvidia-container-toolkit.
metadata:
  author: NVIDIA Corporation
  version: "0.1.0"
allowed-tools: Read Bash
tags:
- vision-language
- classification
- embedding
- zero-shot
- deployment
---

# CLIP

Contrastive Language-Image Pre-training model for zero-shot and fine-tuned image classification, image-text retrieval, and embedding extraction. Fine-tuning adapts CLIP's shared image-text embedding space to domain-specific image-caption data.

No default NGC pretrained checkpoint is required for spec construction, but unset checkpoint behavior is action-specific. In the validation-fixes PyTorch image, `export.checkpoint: null` exports the selected CLIP architecture and may initialize weights when pretrained weights are unavailable. Do not assume `inference.checkpoint: null` loads pretrained weights: `clip inference` currently calls the checkpoint loader with `None` and fails before embedding extraction. For PyTorch inference, checkpoint-backed evaluation/export, resume, and retrain flows, resolve and pass an exact checkpoint from the parent train output. For trusted TAO checkpoints produced by the current run or a known parent job, set `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` on checkpoint-dependent PyTorch actions so PyTorch 2.6 can load the Lightning checkpoint metadata; do not set this for untrusted checkpoints.

Supported actions: `train`, `evaluate`, `inference`, `export`, `gen_trt_engine`.

## Train Action Policy

This model is AutoML-enabled at the model layer. Before handling any train-stage request, read `references/skill_info.yaml` and resolve the run override from either an explicit `automl_policy` value or the user's workflow request. Use `automl_policy: on` by default and only expose `on` / `off` in new launch prompts. Treat phrases like "turn off AutoML", "disable AutoML", "no HPO", or "plain training" as `automl_policy: off` for this run only. When `automl_policy: on`, `automl_enabled: true`, and both `schemas/train.schema.json` and `references/spec_template_train.yaml` are packaged, route the train action through `tao-skill-bank:tao-run-automl` by default with this model's `skill_dir`. Preserve workflow/application overrides for datasets, specs, output directories, GPU/platform settings, parent checkpoints, and `automl_policy`. Use direct model training only when `automl_policy: off` or the packaged train schema/template is missing; in the missing-schema case, report that AutoML is enabled but not runnable for this model until schemas are generated.

The packaged CLIP train schema enables `train.optim.vision_lr` and `train.optim.text_lr` as default AutoML search parameters. For smoke tests, keep the search small by using the Bayesian algorithm with two recommendations and narrow LR ranges.

Non-train actions such as `evaluate`, `inference`, `export`, and deploy flows stay in this model skill. The per-run `automl_policy` override does not change model metadata.

## Instructions

Use this skill for NVIDIA TAO CLIP jobs: training, evaluation, embedding inference, ONNX export, and TensorRT engine generation. Start by identifying the requested action, then load only the referenced files needed for that action: `defaults.json` for default parameters, `config.json` for action/data-source wiring, `references/spec_template.yaml` for full spec shape, and `references/model_info.yaml` for SDK metadata.

For dataset-backed actions, collect the required image, caption, list, or prompt files from the user and place the resolved paths in `spec_overrides`. For local Docker runs, mount extracted folders in the container and point `image_dir` / `caption_dir` at those folders; if a data source provides `.tar.gz` archives, extract them before running the in-container CLIP commands. For `export` and `gen_trt_engine`, infer parent artifacts from the upstream job when available; otherwise require explicit checkpoint, ONNX, or engine paths. Run `gen_trt_engine`, TensorRT `evaluate`, and TensorRT `inference` in the TAO Deploy image.

For TAO Deploy TensorRT actions (`gen_trt_engine`, TensorRT `evaluate`, and TensorRT `inference`), read `references/tao-deploy-clip.md` first. Deploy spec templates live in this skill's `references/` folder with the `spec_template_deploy_*.yaml` prefix.

## Training Requirements

- **Dataset type:** image_text
- **Formats:** custom image/caption folders or WebDataset shards
- **Monitoring metric:** val/t2i_mAP

### Supported Models

- **OpenCLIP / NV-CLIP:** `ViT-L-14-SigLIP-CLIPA-224` (default), `ViT-L-14-SigLIP-CLIPA-336`, `ViT-H-14-SigLIP-CLIPA-224`, `ViT-H-14-SigLIP-CLIPA-336`, `ViT-H-14-SigLIP-CLIPA-574`
- **Radio-CLIP:** `c-radio_v3-b`, `c-radio_v3-l`, `c-radio_v3-h`, `c-radio_v3-g`
- **SigLIP2:** `siglip2-so400m-patch16-256`, `siglip2-so400m-patch14-224`, `siglip2-so400m-patch14-384`, `siglip2-so400m-patch16-384`, `siglip2-so400m-patch16-512`, `siglip2-so400m-patch16-naflex`

Radio-CLIP requires `model.adaptor_name` to be set to `siglip` or `clip`.

### Per-Action Dataset Requirements

| Action | Spec Key | Source | Files | List? |
|---|---|---|---|---|
| train | dataset.train.datasets | train_datasets | image_dir: images.tar.gz, image_list_file: image_list.txt, caption_dir: captions.tar.gz | Yes |
| train | dataset.train.wds.root_dir | train_wds_dataset | root directory containing `.tar` shards | No |
| train | dataset.train.wds.shard_list_file | train_wds_dataset | shards.txt listing shard paths | No |
| train | dataset.val.datasets | eval_dataset | image_dir: images.tar.gz, image_list_file: image_list.txt, caption_dir: captions.tar.gz | Yes |
| evaluate | dataset.val.datasets | eval_dataset | image_dir: images.tar.gz, image_list_file: image_list.txt, caption_dir: captions.tar.gz | Yes |
| inference | inference.datasets | inference_dataset | image_dir: images.tar.gz | Yes |
| inference | inference.text_file | inference_dataset | prompts.txt | No |
| export | export.checkpoint | parent train job or explicit checkpoint | checkpoint .pth, optional for pretrained export | No |
| gen_trt_engine | gen_trt_engine.onnx_file | parent export job or explicit ONNX | clip_model.onnx | No |

For custom training, set `dataset.train.type: custom` and provide `dataset.train.datasets` entries. Image and caption files must share the same base name. `caption_file_suffix` defaults to `.txt`, and `image_list_file` is optional.

When no native CLIP image-caption dataset is available, do not silently treat image-classification data as CLIP data. If the user explicitly allows a plumbing-only validation fallback, derive caption files from class labels, document that the captions are generated from labels, and keep each image/caption pair on the same base filename. Without an `image_list_file`, the TAO custom loader scans the configured image directory for image files; keep validation folders flat unless you provide a list file.

For WDS training, set `dataset.train.type: wds` and provide at least one of `dataset.train.wds.root_dir` or `dataset.train.wds.shard_list_file`. `root_dir` is scanned recursively for `.tar` shards. `shard_list_file` is a text file with one shard path per line; relative lines resolve under the list-file directory unless `root_dir` is also supplied, in which case they resolve under `root_dir`. Validation/evaluation data remains custom format via `dataset.val.datasets`.

### Typical Spec Overrides

Data source overrides are mandatory for dataset-backed actions. Construct paths from the Per-Action Dataset Requirements table and include them in `spec_overrides`. For inference, provide at least one of `inference.datasets` or `inference.text_file`.

```python
S3_TRAIN = "s3://bucket/data/train"
S3_WDS = "s3://bucket/data/wds"
S3_EVAL = "s3://bucket/data/eval"
S3_INFER = "s3://bucket/data/infer"
```

**train, custom dataset:**
```python
{
    "train.num_epochs": 10,
    "dataset.train.type": "custom",
    "dataset.train.datasets": [{"image_dir": f"{S3_TRAIN}/images.tar.gz", "image_list_file": f"{S3_TRAIN}/image_list.txt", "caption_dir": f"{S3_TRAIN}/captions.tar.gz"}],
    "dataset.val.datasets": [{"image_dir": f"{S3_EVAL}/images.tar.gz", "image_list_file": f"{S3_EVAL}/image_list.txt", "caption_dir": f"{S3_EVAL}/captions.tar.gz"}],
}
```

**train, WDS dataset:**
```python
{
    "train.num_epochs": 10,
    "dataset.train.type": "wds",
    "dataset.train.wds.root_dir": f"{S3_WDS}",
    "dataset.train.wds.shard_list_file": f"{S3_WDS}/shards.txt",
    "dataset.train.wds.samples_per_shard": 10000,
    "dataset.val.datasets": [{"image_dir": f"{S3_EVAL}/images.tar.gz", "image_list_file": f"{S3_EVAL}/image_list.txt", "caption_dir": f"{S3_EVAL}/captions.tar.gz"}],
}
```

**evaluate:**
```python
{
    "dataset.val.datasets": [{"image_dir": f"{S3_EVAL}/images.tar.gz", "image_list_file": f"{S3_EVAL}/image_list.txt", "caption_dir": f"{S3_EVAL}/captions.tar.gz"}],
}
```

Leave `evaluate.checkpoint` unset for zero-shot evaluation with pretrained weights. Set `evaluate.trt_engine` instead of `evaluate.checkpoint` for TensorRT evaluation.

**inference:**
```python
{
    "inference.datasets": [{"image_dir": f"{S3_INFER}/images.tar.gz"}],
    "inference.text_file": f"{S3_INFER}/prompts.txt",
}
```

Inference writes `image_embeddings.h5` and/or `text_embeddings.h5` under `results_dir`. The saved embeddings are L2-normalized.

**export:**
```python
{
    "export.onnx_file": "${results_dir}/export/clip_model.onnx",
    "export.encoder_type": "combined",
    "export.batch_size": -1,
}
```

Set `export.encoder_type: separate` when deployment should use independent vision and text encoders. Separate export writes `_vision.onnx` and `_text.onnx` variants derived from the base `export.onnx_file`.

For checkpoint-dependent actions, use the model-specific checkpoint resolver output from the parent train job. CLIP training writes checkpoints such as `model_epoch_000_step_00020.pth` and a `clip_latest.pth` symlink. Use the exact resolved checkpoint for `evaluate.checkpoint`, `inference.checkpoint`, `export.checkpoint`, and `train.resume_training_checkpoint_path`; use `clip_latest.pth` only when the user explicitly asks for latest.

When the resolved checkpoint is trusted TAO output, checkpoint-backed PyTorch `evaluate`, `inference`, `export`, and resume training should run with `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`. PyTorch 2.6 otherwise defaults checkpoint loading to weights-only mode and can reject CLIP Lightning checkpoints containing NumPy scalar metadata.

**gen_trt_engine:**
```python
{
    "gen_trt_engine.onnx_file": "${results_dir}/export/clip_model.onnx",
    "gen_trt_engine.trt_engine": "${results_dir}/deploy/clip_model.engine",
    "gen_trt_engine.batch_size": -1,
    "gen_trt_engine.tensorrt.data_type": "fp16",
    "gen_trt_engine.tensorrt.min_batch_size": 1,
    "gen_trt_engine.tensorrt.opt_batch_size": 1,
    "gen_trt_engine.tensorrt.max_batch_size": 16,
}
```

## Eval Dataset

Optional for training. If provided, validation metrics are computed at validation intervals. Required for `evaluate`.

## Deploy Workflow

The skill exposes `gen_trt_engine` as the deploy action. In generated SDK runners, use `model_info["actions"]["gen_trt_engine"]` and run it in the TAO Deploy image, not the PyTorch training image. The in-container command is `clip gen_trt_engine -e {config_path}`; direct TAO Launcher usage spells the same action as `tao deploy clip gen_trt_engine -e /path/to/spec.yaml`.

TAO Deploy inference can discover combined engines, paired separate engines, or single-pillar `_vision.engine` / `_text.engine` files. For full TensorRT retrieval evaluation or image+text TensorRT inference, export with `export.encoder_type: separate` and run `clip gen_trt_engine` twice: build `clip_model_vision.onnx` to an engine ending in `_vision.engine`, then build `clip_model_text.onnx` to the matching `_text.engine` in the same directory. For image-only TensorRT inference, building only the `_vision.engine` is sufficient and `inference.text_file` must be `null`. TensorRT `evaluate` and text inference require a text-capable engine; if only a vision engine is present, deploy evaluation fails because text embeddings cannot be extracted.

Use `evaluate.trt_engine` for TensorRT evaluation and `inference.trt_engine` for TensorRT embedding extraction. These TensorRT paths also run in the TAO Deploy image. Direct TAO Launcher usage spells these as `tao deploy clip evaluate` and `tao deploy clip inference`.

## Important Parameters

- **model.type**: Backbone family and resolution. Use a TAO-registered CLIP model ID such as `ViT-L-14-SigLIP-CLIPA-224`. Prefer the listed OpenCLIP / NV-CLIP IDs for AutoML smoke tests because the current TAO container registry routes them through the supported augmentation adapter.
- **model.adaptor_name**: Required for Radio-CLIP. Set to `siglip` or `clip`.
- **model.image_size**: Training transform image resolution. Keep it aligned with the selected fixed-resolution backbone.
- **train.num_epochs**: CLIP fine-tuning often converges quickly. Start with 10-20 epochs for domain adaptation, then increase only if validation loss is still improving.
- **train.optim.vision_lr / train.optim.text_lr**: Learning rates for the two encoders. CLIP is sensitive to high learning rates; reduce both if loss is unstable.
- **model.freeze_vision_encoder / model.freeze_text_encoder**: Defaults are false. Freezing one encoder can help when the dataset is small or only one modality needs adaptation.
- **train.loss_type**: `siglip` is recommended for SigLIP2 and Radio-CLIP. Use `clip` for CLIP-style softmax loss.
- **export.encoder_type**: `combined` exports one ONNX graph. `separate` exports independent vision and text graphs.
- **gen_trt_engine.tensorrt.data_type**: TensorRT deployment supports `fp16` and `fp32`.

## Hardware

Single-GPU training works for small datasets. Use 4+ GPUs for datasets with more than 100k images or large backbones. Use 16GB+ VRAM per GPU for small/fixed-resolution runs and larger GPUs for Radio-CLIP or high-resolution OpenCLIP variants.

## Error Patterns

See `references/error-patterns.md` for the full list of CLIP error symptoms and fixes (CUDA OOM, NaN loss, retrieval quality, dataset format/size, Radio-CLIP and model-ID validation, ONNX external data, TensorRT shape mismatch, PyTorch 2.6 checkpoint load, null-checkpoint inference, TensorRT text/retrieval failures, `attention_mask` handling, and spec/schema merge errors).

## Spec Param / Parent Model Inference

See `references/spec-param-inference.md` for the model-specific inference mappings (the full `clip.config.json` action/spec-field/inference-function table) that generated runners apply with SDK helpers before `create_job()`, plus `parent_job_id` resolution rules.

## Deployment

- [tao-deploy-clip](references/tao-deploy-clip.md)
