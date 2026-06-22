---
name: tao-train-nvpanoptix3d
description: NVPanoptix3D for panoptic 3D scene reconstruction from posed RGB images. Produces 3D panoptic segmentation
  (semantic, instance, and panoptic masks) with occupancy completion. Built on a VGGT backbone with a Mask2Former-style head
  and 3D frustum reconstruction. Use when training, evaluating, exporting, or running inference for a TAO NVPanoptix3D model.
  Trigger phrases include "train NVPanoptix3D", "panoptic 3D reconstruction", "3D scene segmentation", "occupancy completion".
license: Apache-2.0
compatibility: Requires docker + nvidia-container-toolkit.
metadata:
  version: "0.1.0"
  author: NVIDIA Corporation
allowed-tools: Read Bash
tags:
- panoptic
- 3d
- reconstruction
---

# NVPanoptix3D

NVPanoptix3D for panoptic 3D scene reconstruction from posed RGB images. Produces 3D panoptic segmentation (semantic, instance, and panoptic masks) with occupancy completion. Built on VGGT backbone with Mask2Former-style head and 3D frustum reconstruction.

Uses 2D and 3D stage checkpoints. Set train.checkpoint_2d and train.checkpoint_3d for staged initialization.

## Dataclass Schemas

Generated TAO Core schemas are packaged in `schemas/<action>.schema.json`, with `schemas/manifest.json` listing available actions. Each generated schema also emits `references/spec_template_<action>.yaml` from the schema top-level `default` field. AutoML enablement is declared at the model layer in `references/skill_info.yaml` via `automl_enabled`. Runnable AutoML still requires `schemas/train.schema.json` and `references/spec_template_train.yaml` to exist and parse. Use the packaged train schema for `automl_default_parameters`, `automl_disabled_parameters`, defaults, min/max bounds, enums, option weights, math conditions, dependencies, and popular parameters. Do not expect `~/tao-core` at runtime; maintainers regenerate schemas/templates before packaging the skill bank.

## Train Action Policy

This model is AutoML-enabled at the model layer. Before handling any train-stage request, read `references/skill_info.yaml` and resolve the run override from either an explicit `automl_policy` value or the user's workflow request. Use `automl_policy: on` by default and only expose `on` / `off` in new launch prompts. Treat phrases like "turn off AutoML", "disable AutoML", "no HPO", or "plain training" as `automl_policy: off` for this run only. When `automl_policy: on`, `automl_enabled: true`, and both `schemas/train.schema.json` and `references/spec_template_train.yaml` are packaged, route the train action through `tao-skill-bank:tao-run-automl` by default with this model's `skill_dir`. Preserve workflow/application overrides for datasets, specs, output directories, GPU/platform settings, parent checkpoints, and `automl_policy`. Use direct model training only when `automl_policy: off` or the packaged train schema/template is missing; in the missing-schema case, report that AutoML is enabled but not runnable for this model until schemas are generated.

For AutoML, use `train_loss` as the optimization metric with
`direction=minimize`, and set `train.optim.monitor_name: train_loss` in
`spec_overrides`. NVPanoptix3D train jobs emit `PRQ`, `RSQ`, and `RRQ` in
`status.json`, and the training progress log emits `train_loss`; short or
minimal jobs may not emit `val_loss`, including full-trial smoke runs.
Multi-fidelity AutoML algorithms such as Hyperband, ASHA, and BOHB may promote
a checkpoint to a resume job that completes without emitting a fresh
`train_loss` line. In that case, the AutoML metric is the carried-forward
metric from the source rung job that emitted `train_loss`; still verify the
promoted job resumed from the explicit epoch/step checkpoint, produced a real
checkpoint, and is usable for evaluate/inference.
Non-train actions such as `evaluate`, `inference`, `export`, and deploy flows stay in this model skill. The per-run `automl_policy` override does not change model metadata.

## Training Requirements

- **Dataset type:** nvpanoptix3d
- **Formats:** front3d, matterport
- **Monitoring metric:** train_loss
- **AutoML direction:** minimize
- For AutoML train jobs, use `train_loss`. For multi-fidelity resume jobs that do not emit a fresh `train_loss`,
  compare AutoML's carried metric to the source rung job that emitted it.
  Validation status KPIs are `PRQ`, `RSQ`, and `RRQ`; do not use `val_loss`
  unless a specific run is known to emit it.

### Per-Action Dataset Requirements

| Action | Spec Key | Source | Files | List? |
|---|---|---|---|---|
| evaluate | dataset.frustum_mask_path | eval_dataset | meta/frustum_mask.npz | No |
| evaluate | dataset.label_map | eval_dataset | meta/colormap.json | No |
| evaluate | dataset.val.json_path | eval_dataset | meta/val.json | No |
| evaluate | dataset.val.base_dir | eval_dataset |  | No |
| evaluate | dataset.test.json_path | inference_dataset | meta/test.json | No |
| evaluate | dataset.test.base_dir | inference_dataset |  | No |
| inference | dataset.frustum_mask_path | inference_dataset | meta/frustum_mask.npz | No |
| inference | dataset.label_map | inference_dataset | meta/colormap.json | No |
| inference | inference.images_dir | inference_dataset | flat folder of `.jpg`/`.png` RGB images | No |
| train | dataset.frustum_mask_path | train_datasets | meta/frustum_mask.npz | No |
| train | dataset.label_map | train_datasets | meta/colormap.json | No |
| train | dataset.train.json_path | train_datasets | meta/train.json | No |
| train | dataset.train.base_dir | train_datasets |  | No |
| train | dataset.val.json_path | eval_dataset | meta/val.json | No |
| train | dataset.val.base_dir | eval_dataset |  | No |
| train | dataset.test.json_path | inference_dataset | meta/test.json | No |
| train | dataset.test.base_dir | inference_dataset |  | No |

### Typical Spec Overrides

Data source overrides are **mandatory for every action** — the agent MUST construct data source paths from the Per-Action Dataset Requirements table above and include them in `spec_overrides`.
For packaged S3 folders that store scene data as `data/images.tar.gz`, the
skill metadata requests extraction into the parent `data/` directory because
the TAO loader expects `base_dir/data/<scene_id>/...`.

```python
S3_TRAIN = "s3://bucket/data/train"
S3_EVAL = "s3://bucket/data/eval"
```

**train (mandatory data sources):**
```python
{
    "train.num_epochs": 10,
    "train.checkpoint_interval": 10,
    "train.validation_interval": 10,
    "train.num_gpus": 1,
    "dataset.enable_3d": True,
    "dataset.contiguous_id": True,
    "model.sem_seg_head.num_classes": 13,
    "dataset.frustum_mask_path": f"{S3_TRAIN}/meta/frustum_mask.npz",
    "dataset.label_map": f"{S3_TRAIN}/meta/colormap.json",
    "dataset.train.json_path": f"{S3_TRAIN}/meta/train.json",
    "dataset.train.base_dir": f"{S3_TRAIN}",
    "dataset.val.json_path": f"{S3_EVAL}/meta/val.json",
    "dataset.val.base_dir": f"{S3_EVAL}",
    "dataset.test.json_path": f"{S3_EVAL}/meta/test.json",
    "dataset.test.base_dir": f"{S3_EVAL}",
}
```

**evaluate (mandatory data sources):**
```python
{
    "evaluate.checkpoint": "<selected train/AutoML checkpoint>",
    "dataset.enable_3d": True,
    "dataset.contiguous_id": True,
    "dataset.frustum_mask_path": f"{S3_EVAL}/meta/frustum_mask.npz",
    "dataset.label_map": f"{S3_EVAL}/meta/colormap.json",
    "dataset.val.json_path": f"{S3_EVAL}/meta/val.json",
    "dataset.val.base_dir": f"{S3_EVAL}",
    "dataset.test.json_path": f"{S3_EVAL}/meta/test.json",
    "dataset.test.base_dir": f"{S3_EVAL}",
}
```

**inference (mandatory data sources):**
```python
{
    "inference.checkpoint": "<selected train/AutoML checkpoint>",
    "dataset.enable_3d": True,
    "dataset.frustum_mask_path": f"{S3_EVAL}/meta/frustum_mask.npz",
    "dataset.label_map": f"{S3_EVAL}/meta/colormap.json",
    "inference.images_dir": "/path/to/flat_rgb_images",
}
```
## Eval Dataset

Optional. Val/test splits configured via dataset.val and dataset.test paths.

## Important Parameters

- **model.sem_seg_head.num_classes**: Number of semantic classes. Default 13.
- **model.mode**: Prediction mode. Options: panoptic, instance, semantic. Default panoptic.
- **model.backbone_type**: Backbone. Default vggt (only option in schema).
- **model.mask_former.num_object_queries**: Object queries. Default 100.
- **model.mask_former.dec_layers**: Decoder layers. Default 10.
- **model.frustum3d.truncation**: 3D frustum truncation. Default 3.
- **model.frustum3d.panoptic_weight**: Panoptic loss weight. Default 25.
- **model.frustum3d.completion_weights**: Completion loss weights. Default [50, 25, 10].
- **dataset.name**: Dataset name. Options: front3d, matterport, synthetic_hospital, synthetic_warehouse.
- **dataset.contiguous_id**: Set `True` when the label-map JSON already
  supplies `trainId` values for its category IDs; leaving the default `False`
  can synthesize placeholder categories without `trainId` and fail during
  metadata construction.
- **dataset.downsample_factor**: Image downsample factor. Default 1 (Front3D), 2 (Matterport).
- **dataset.target_size**: Target image size. Default [320, 240].
- **dataset.depth_min**: Min depth. Default 0.4 meters.
- **dataset.depth_max**: Max depth. Default 6.0 meters.
- **train.lr**: Learning rate. Default 2e-4. backbone_multiplier=0.1.
- **train.lr_scheduler**: Options: MultiStep, Warmuppoly. Milestones [88, 96].
- **train.precision**: Only `fp32` is supported by the current train code.
- **train.distributed_strategy**: Options: ddp, fsdp. activation_checkpoint=True by default.
- **train.clip_grad_norm**: Gradient clipping norm. Default 0.1.
- **export.onnx_file_2d**: ONNX path for 2D model component.
- **export.max_voxels**: Max voxels for engine input. Default 700000.
- **inference.mode**: Options: semantic, instance, panoptic.

## Multi-GPU / Multi-Node

**Launch method:** Lightning-managed (single `python` process, Lightning spawns workers).

| Spec Key | Description | Default |
|----------|-------------|---------|
| `train.num_gpus` | Number of GPUs | 1 |
| `train.gpu_ids` | GPU device indices | [0] |
| `train.num_nodes` | Number of nodes | 1 |
| `train.distributed_strategy` | `ddp` only | `ddp` |

- **`fsdp` is NOT supported** for NVPanoptix3D (code only handles `ddp`)
- `ddp` with activation checkpointing (enabled by default): `find_unused_parameters=False`
- `ddp` without: `find_unused_parameters=True`
- FAN backbones with 3D enabled auto-enable `sync_batchnorm`

**Multi-node env vars** (set by orchestrator): `WORLD_SIZE`, `NODE_RANK`, `MASTER_ADDR`, `MASTER_PORT`, `NUM_GPU_PER_NODE`.

## Export / TRT Defaults

- Exports the 2D ONNX model to `export.onnx_file_2d`. The current export
  entrypoint calls `export_2d_model`; `export.onnx_file_3d` is present in the
  schema but not produced by this toolkit image.
- TRT data types: FP32, FP16 only
- max_voxels: 700000 (engine input tensor limit)

## Hardware

Minimum 2 GPU(s), recommended 4 GPU(s). 40GB+ (A100 recommended) VRAM per GPU. 3D reconstruction is very memory intensive. Use `train.precision: fp32`; the current training entrypoint rejects fp16. activation_checkpoint enabled by default. FSDP for multi-node. AutoML is enabled at the model layer; preserve this GPU/VRAM guidance when routing train through AutoML.

## Error Patterns

**`nvpanoptix3d: not found` in the PyTorch image**: Use the packaged module
entrypoint command:
`python -m nvidia_tao_pytorch.cv.nvpanoptix3d.entrypoint.nvpanoptix3d <action> -e <spec>`.
The 7.0 PyTorch image contains the NVPanoptix3D package but does not expose a
`nvpanoptix3d` console script.

**Missing frustum mask**: Ensure meta/frustum_mask.npz is present in the dataset directory.

**Downsample factor mismatch**: Use downsample_factor=2 for Matterport3D, 1 for Front3D / synthetic datasets.

**3D occupancy OOM**: Reduce frustum_dims or grid_dimensions if running out of GPU memory during 3D reconstruction.

**fp16 precision rejected**: The schema advertises `fp16`, but the current
training entrypoint raises `ValueError: Only fp32 precision is supported.` Use
`train.precision: fp32` for train and resume/retrain.

**Inference dataloader length is zero**: `inference.images_dir` is scanned only
for top-level `.jpg` and `.png` files. If the S3 test archive extracts to
scene subdirectories, create or point to a flat folder of real RGB images before
running inference.

**3D ONNX missing after export**: The current export entrypoint only calls the
2D ONNX exporter and writes `export.onnx_file_2d`. Do not require
`export.onnx_file_3d` unless the toolkit image adds a 3D exporter.

**Resume stops at the epoch boundary**: A one-epoch smoke run writes an
end-of-epoch checkpoint such as `model_epoch_000_step_00020.pth`. Resuming with
`train.num_epochs` set only one epoch beyond the original run can restore the
checkpoint and stop without producing a new epoch checkpoint. When validating
actual retraining from an epoch-boundary checkpoint, set `train.num_epochs` at
least two epochs beyond the source smoke run and raise `train.optim.max_steps`
accordingly. For example, resuming from `model_epoch_000_step_00020.pth` needs
`train.num_epochs: 3` and enough max steps to produce a new exact epoch/step
checkpoint such as `model_epoch_001_step_00040.pth` before handing the model to
evaluate, inference, or export.

## Spec Param / Parent Model Inference

Model-specific inference mappings belong in this MD file, not in `config.json`. Generated runners should read this section and apply the mappings with SDK helpers before `create_job()`. This mirrors the old microservices `infer_params.py` flow.

Model-specific handoff mappings:

| Action | Spec Field | Inference Function | Meaning |
|---|---|---|---|
| evaluate | `encryption_key` | `key` | encryption key |
| evaluate | `evaluate.checkpoint` | `parent_model` | model file inferred from the parent job results folder |
| evaluate | `results_dir` | `output_dir` | current job results directory |
| export | `encryption_key` | `key` | encryption key |
| export | `export.checkpoint` | `parent_model` | model file inferred from the parent job results folder |
| export | `export.onnx_file_2d` | `create_onnx_file_2d` | output 2D ONNX path |
| export | `results_dir` | `output_dir` | current job results directory |
| inference | `encryption_key` | `key` | encryption key |
| inference | `inference.checkpoint` | `parent_model` | model file inferred from the parent job results folder |
| inference | `results_dir` | `output_dir` | current job results directory |
| train | `encryption_key` | `key` | encryption key |
| train | `results_dir` | `output_dir` | current job results directory |
| train | `train.checkpoint_2d` | `parent_model_or_ptm` | parent model if available, otherwise PTM |
| train | `train.checkpoint_3d` | `ptm` | pretrained model |
| train | `train.resume_training_checkpoint_path` | `resume_model` | model file inferred from the current job results folder |

For `parent_model` or `parent_model_folder`, pass the upstream train/export/AutoML child job id as `parent_job_id`. The SDK lists the parent result folder, filters checkpoint artifacts, and returns the selected model file or folder. Do not add these mappings back to `config.json` and do not patch generated runner scripts to guess checkpoint paths.
