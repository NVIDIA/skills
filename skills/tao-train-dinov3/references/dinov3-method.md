# DINOv3 method and configuration

DINOv3 continual pre-training adapts an existing DINOv3 checkpoint to unlabeled domain images. TAO trains a student against an exponential-moving-average teacher and writes stripped teacher and student checkpoints alongside full training checkpoints.

## Training objectives

| Objective | Role |
|---|---|
| DINO | Aligns global representations across augmented views |
| iBOT | Learns patch-level representations from masked inputs |
| KoLeo | Encourages diversity in global features |
| Gram anchoring | Optionally limits drift in patch-feature geometry |

The usefulness of each objective depends on the dataset and downstream task. Evaluate the resulting features with metrics relevant to that task.

## Backbone and position encoding

DINOv3 uses patch-16 vision transformers with register tokens and two-dimensional rotary position encoding.

- `model.backbone.teacher_type` and `student_type` select the backbone size.
- `model.backbone.img_size` controls the input image size.
- `model.backbone.rope_theta` controls the rotary frequency base and defaults to `100.0`.
- Crop and export dimensions need to be divisible by `model.backbone.patch_size`.

`rope_theta` is tunable. Because changing it changes positional encoding, compare the resulting checkpoint on the intended workload.

## Training controls

- `model.centering_method` selects Sinkhorn or softmax centering.
- `train.schedulers.momentum` controls how quickly the teacher follows the student.
- `train.schedulers.*.warm_up_steps` should be selected for the planned training length.
- `model.gram.*` enables and configures optional Gram anchoring.
- `train.precision`, batch size, image size, and attention implementation affect memory and throughput.

## Checkpoint compatibility

`train.pretrained_model_path` accepts compatible timm-format DINOv3 weights or TAO DINOv3 checkpoints. Match the configured backbone to the checkpoint. DINOv2, NVDINOv2, and unrelated ViT checkpoints are not compatible.

Use a full `model_epoch_*.pth` checkpoint to resume training. Use a stripped `teacher_epoch_*.pth` checkpoint for conversion, export, inference, or downstream evaluation.

## References

- DINOv3 (Meta AI, 2025)
- DINOv2 (Meta AI, 2023)
- DINO (Caron et al., 2021)
- `nvidia_tao_pytorch/ssl/dinov3/README.md` in tao-pytorch
