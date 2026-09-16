---
name: tao-train-dinov3
description: DINOv3 continual self-supervised pre-training. Domain-adapts public DINOv3 ViT backbones
  on unlabeled images via teacher-student self-distillation (DINO + iBOT + KoLeo, optional Gram anchoring)
  and converts the EMA teacher into a timm-format backbone for downstream tasks. Trigger phrases include
  "train DINOv3", "DINOv3 SSL", "domain-adapt a foundation backbone", "continual pretraining", "self-supervised
  finetune DINOv3".
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
- dinov3
---

# DINOv3

> **Standalone install?** If this session was not initialized by the TAO skill bank plugin, run the `tao-setup` skill first.

Use this skill to continue self-supervised training of a DINOv3 backbone on unlabeled images, then convert, export, or run inference with a trained teacher checkpoint.

## Quick Start (docker run)

Docker-native launch — no TAO SDK and no Python on the host. Use the local
Docker/platform skill instead when it gives a stricter environment-specific
command (non-root UID mapping, cache redirects, remote daemons).

```bash
TAO_PYT_IMAGE_DEFAULT=nvcr.io/nvidia/tao/tao-toolkit:7.2.0-pyt  # versions-key: images.tao_toolkit.pyt
TAO_PYT_IMAGE="${TAO_PYT_IMAGE:-$TAO_PYT_IMAGE_DEFAULT}"
RUN_ROOT="${RUN_ROOT:-$PWD}"
DOCKER_COMMON=(
  --rm --gpus all --shm-size=8g
  --shm-size=8g
  --ulimit memlock=-1
  --ulimit stack=67108864
  -v "$RUN_ROOT/data:/data:ro"
  -v "$RUN_ROOT/specs:/specs:ro"
  -v "$RUN_ROOT/results:/results"
)
```

Train:

```bash
docker run "${DOCKER_COMMON[@]}" "$TAO_PYT_IMAGE" \
  dinov3 train -e /specs/train.yaml
```

Inference:

```bash
docker run "${DOCKER_COMMON[@]}" "$TAO_PYT_IMAGE" \
  dinov3 inference -e /specs/inference.yaml
```

Export:

```bash
docker run "${DOCKER_COMMON[@]}" "$TAO_PYT_IMAGE" \
  dinov3 export -e /specs/export.yaml
```

Convert:

```bash
docker run "${DOCKER_COMMON[@]}" "$TAO_PYT_IMAGE" \
  dinov3 convert -e /specs/convert.yaml
```

Every action takes its spec with `-e`; `results_dir` is set in the spec or
overridden on the command line. Mount any pretrained-weights directory the spec
references, and keep every in-container path consistent across actions.

## Configuration

Use `schemas/<action>.schema.json` for supported fields, defaults, ranges, and options. Start from the matching `references/spec_template_<action>.yaml`.

Training is AutoML-enabled. Read `references/skill_info.yaml` and resolve an explicit `automl_policy` or user request before training. Default to `automl_policy: on`; requests such as "disable AutoML", "no HPO", or "plain training" mean `off` for that run. When it is on and the train schema and template are packaged, route training through `tao-skill-bank:tao-run-automl`. Use direct training only when the policy is off or those files are missing, and report the missing-file limitation. Non-train actions remain in this skill.

Treat `train_loss` as an optimization signal; use downstream evaluation when selecting a checkpoint for deployment.

For method background or tuning ideas, optionally read:

- `references/dinov3-method.md`
- `references/dinov3-recipes.md`

## Core workflow

1. Obtain a compatible DINOv3 checkpoint in timm or TAO format.
2. Point `dataset.train_dataset.images_dir` to a directory of unlabeled images.
3. Configure the backbone, training scale, and output location.
4. Run training and evaluate useful teacher checkpoints on data representative of the downstream task.
5. Use the selected teacher checkpoint with `convert`, `export`, or `inference`.

Provide either `train.pretrained_model_path` for a new run or `train.resume_training_checkpoint_path` to resume an existing run.

## Essential fields

| Purpose | Spec field |
|---|---|
| Training images | `dataset.train_dataset.images_dir` |
| Initial weights | `train.pretrained_model_path` |
| Resume checkpoint | `train.resume_training_checkpoint_path` |
| Backbone | `model.backbone.teacher_type`, `model.backbone.student_type` |
| Training scale | `dataset.batch_size`, `train.num_gpus`, `train.num_nodes` |
| Output directory | `results_dir` |
| Conversion input | `convert.checkpoint` |
| Export input | `export.checkpoint` |
| Inference inputs | `inference.checkpoint`, `dataset.test_dataset.images_dir` |

Example train settings:

```yaml
model:
  backbone:
    teacher_type: vit_b
    student_type: vit_b
    rope_theta: 100.0
dataset:
  train_dataset:
    images_dir: /path/to/unlabeled/images
  batch_size: 16
train:
  pretrained_model_path: /path/to/dinov3/weights
  num_epochs: 10
  num_gpus: 1
```

Supported image extensions include jpg, jpeg, png, ppm, bmp, pgm, tif, tiff, and webp. Keep evaluation data separate from training data when measuring downstream quality.

Every `dataset.*.images_dir` value must be an extracted image folder. Managed
archive-backed sources use `runtime: extracted_folder`, which unpacks the source
before injecting the directory into the spec. For direct Docker or TAO CLI runs,
extract `.tar`, `.tar.gz`, `.tgz`, or `.zip` inputs before launch.

## Checkpoints

| File | Use |
|---|---|
| `model_epoch_*_step_*.pth` | Resume training, or export its EMA teacher |
| `teacher_epoch_*_step_*.pth` | Convert, export, inference, or downstream evaluation |
| `student_epoch_*_step_*.pth` | Diagnostics |

For export, prefer a selected stripped `teacher_epoch_*_step_*.pth`. Export also accepts
a selected full `model_epoch_*_step_*.pth` and deterministically selects and logs
`teacher.backbone`. Inference requires a stripped teacher checkpoint. Do not
substitute `dinov3_model_latest.pth` without evaluating that milestone.

Training loss does not directly measure downstream representation quality. When possible, compare candidate teacher checkpoints using metrics relevant to the target task.

For `convert`, `export`, and `inference`, copy the training-time backbone configuration into the action spec, including `rope_theta`. Stripped checkpoints do not restore the RoPE frequency base, so falling back to the default can change the trained model's behavior.

## Key options

- `model.backbone.teacher_type` and `student_type` support `vit_s`, `vit_s_plus`, `vit_b`, `vit_l`, `vit_h_plus`, and `vit_7b`.
- When `model.distill.enable: false`, the teacher and student backbone types must match.
- `model.backbone.rope_theta` controls the rotary frequency base. It defaults to `100.0` and is tunable; changing it alters positional encoding, so evaluate the resulting checkpoint for the intended task.
- `model.centering_method` supports `sinkhorn` and `softmax`.
- `model.gram.*` controls optional Gram anchoring.
- `model.lora.*` enables parameter-efficient continual pre-training. The default targets the attention `qkv` and `proj` projections; `num_last_blocks: 0` adapts every transformer block.
- `model.preservation.*` adds CLS-token MSE and cosine losses against the frozen anchor teacher. It is recommended with LoRA when global embedding quality matters.
- `train.log_every_n_steps` controls step-level loss visibility and defaults to `1`, so short smoke runs emit component and preservation losses.
- `train.precision` defaults to `16-mixed`; choose precision based on hardware and measured quality.
- `train.use_custom_attention: false` uses native attention when custom attention is unavailable.
- Crop and export dimensions need to be divisible by `model.backbone.patch_size`.

## Distributed training

Use `train.num_gpus`, `train.gpu_ids`, `train.num_nodes`, and `train.distributed_strategy`. Start with `auto`; use `ddp` or `fsdp` when appropriate for the selected backbone, resolution, and available memory.

## Common issues

- **Checkpoint load failure:** verify that the checkpoint is DINOv3 and that its backbone matches the configured teacher and student types.
- **No images found or archive rejected:** pass an extracted image folder to `dataset.*.images_dir`, not an archive file.
- **Out of memory:** reduce batch size or image resolution, or use a sharded distributed strategy.
- **Custom-attention failure:** set `train.use_custom_attention: false`.
- **Deterministic-backward failure:** xformers memory-efficient attention does not provide a deterministic backward kernel; use `train.cudnn.deterministic: false`.
- **Grid-size error:** choose crop and export dimensions divisible by the patch size.
- **Unexpected inference keys:** use a stripped `teacher_epoch_*_step_*.pth` checkpoint.
- **Full-checkpoint export:** a selected `model_epoch_*_step_*.pth` is supported; export selects and logs its EMA `teacher.backbone`.
- **Resume mismatch:** the SSL dataloader is not step-resumable, so resume from an epoch/checkpoint boundary when reproducibility matters.
