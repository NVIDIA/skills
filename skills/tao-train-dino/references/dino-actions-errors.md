# DINO Actions And Error Patterns

Important parameters, defaults, evaluate/export defaults, hardware notes, and known error patterns.

Load this file only when the compact `SKILL.md` points here for the current task. If this reference conflicts with `SKILL.md`, `skill_info.yaml`, schemas, or platform/model skills, the compact/current source wins.

## Contents

- Default Values
- Evaluate Defaults
- Export Defaults
- Hardware
- Error Patterns

## Important Parameters

- **dataset.num_classes**: Number of object classes. Default is 91 (COCO). Must be >= `max(category_id) + 1`. Too low causes `CUDA error: device-side assert triggered`.
- **model.backbone**: Backbone architecture. Default resnet_50. Supported values include `resnet_34`, `resnet_50`, `fan_tiny`, `fan_small`, `fan_base`, `fan_large`, `gc_vit_xxtiny`, `gc_vit_xtiny`, `gc_vit_tiny`, `gc_vit_small`, `gc_vit_base`, `gc_vit_large`, `gc_vit_large_384`, `vit_large_nvdinov2`, `vit_large_dinov2`, `swin_tiny_224_1k`, `swin_base_224_22k`, `swin_base_384_22k`, `swin_large_224_22k`, `swin_large_384_22k`, and `efficientvit_b0` through `efficientvit_b3`.
- **train.optim.lr**: Learning rate. Default 2e-4 (AdamW). lr_backbone defaults to 2e-5 (10x lower). Reduce both if training diverges.
- **train.num_epochs**: DINO typically needs 30-50+ epochs for good mAP on real datasets. The default of 10 is suitable for quick iteration.
- **train.optim.lr_steps**: MultiStep LR decay schedule. Default [11]. For longer training, set to e.g. [30, 40] for a 50-epoch run.
- **model.num_queries**: Number of object queries. Default 300. Increase for dense scenes with many objects per image. num_select must be < num_queries * num_classes.
- **dataset.batch_size**: Per-GPU batch size. Default 4. Reduce to 2 if OOM on 16GB GPUs. Total batch = batch_size * num_gpus.

## Default Values

- **num_epochs**: `10`
- **batch_size**: `4`
- **learning_rate**: `2e-4`
- **lr_backbone**: `2e-5`
- **num_classes**: `91`
- **backbone**: `resnet_50`

## Evaluate Defaults

Use `references/spec_template_evaluate.yaml` (when present) as the base spec
for `action="evaluate"`, then apply the mandatory checkpoint and data-source
overrides above. `references/skill_info.yaml` declares the required evaluate
inputs so the SDK script runner downloads and rewrites them before running
the container. This model MD also documents
`evaluate.checkpoint = parent_model`, so generated runners should infer the
checkpoint from the parent job result files before submission:

```json
{
  "evaluate.checkpoint": {"type": "file"},
  "dataset.test_data_sources.image_dir": {"type": "file"},
  "dataset.test_data_sources.json_file": {"type": "file"}
}
```

## Export Defaults

- **input_width**: `960`
- **input_height**: `544`
- **opset_version**: `17`
- **trt_data_types**: `[FP32, FP16, INT8]`
- **trt_workspace_size_mb**: `1024`

## Hardware

- **Minimum**: 1 GPU
- **Recommended**: 4 GPUs
- **GPU Memory**: 24GB+ (A100 recommended)

Transformer-based detection is memory-intensive. batch_size=4 fits on 24GB GPUs. For 16GB GPUs, reduce to batch_size=2. Multi-GPU with 4+ GPUs recommended for datasets > 10k images.

## Error Patterns

**CUDA out of memory**: Reduce dataset.batch_size (4 -> 2 -> 1). DINO uses multi-scale features that consume significant GPU memory, especially with high-resolution images (default max 1333px).

**num_select must be < num_queries * num_classes**: Ensure model.num_select (default 300) is less than num_queries * dataset.num_classes.

**Error merging spec.yaml with schema**: Hydra/OmegaConf validation error. num_epochs and num_gpus must be under 'train.*', not at spec root. Use the SDK spec_shorthand_keys mapping.

**Dataset size smaller than total batch size**: Total batch = batch_size * num_gpus. If val dataset has fewer samples, reduce dataset.batch_size or num_gpus. The agent should proactively check this.

**return_interm_indices length must match num_feature_levels**: Default is [1,2,3,4] with num_feature_levels=4. If changing one, update the other.

**`FileNotFoundError` on images**: The archive extraction/cache and annotation paths are out of sync. For standard DINO datasets, pass remote `images.tar.gz`; the SDK should rewrite the runtime spec to `images`. If DINO looks under `/mnt/lustre/.../images/<file>.jpg` and files are missing, clear the stale `<images.tar.gz>.extracted` marker and re-extract/download the archive, or inspect the archive top-level layout.

**`FileNotFoundError` at startup (val)**: `val_data_sources` missing or pointing to non-existent data. DINO unconditionally builds a val dataloader — this is required even when only optimizing `train_loss`.

**`CUDA device-side assert`**: `num_classes` too low. Set `num_classes >= max(category_id) + 1`.

**S3 inputs not downloaded inside container**: When the agent invokes DINO via SDK orchestration, `references/skill_info.yaml` must declare `actions.train.inputs` with `[0]`-indexed spec keys (see "Optional: SDK orchestration internals"). Use `s3://...` for S3-compatible datasets; do not generate `aws://...` URIs.

**Evaluate checkpoint not found at result root**: DINO train jobs upload
checkpoints under `results_dir/train/`. If eval fails with `FileNotFoundError`
for a root-level checkpoint path, resolve an actual file under
`s3://<bucket>/results/<train_job_id>/results_dir/train/`, normally an exact
`model_epoch_<epoch>_step_<step>.pth` file selected by the resolver.

**Parent `dino gen_trt_engine` rejected by the PyT CLI**: In the validated
7.0.0 PyT container, `dino gen_trt_engine` is not a valid parent-model subtask.
Use the DINO deploy workflow (`tao-deploy-dino.md`) for TensorRT engine
generation, TensorRT evaluation, and TensorRT inference.

**`dino convert` fails before reading the spec**: In the validated 7.0.0 PyT
container, DINO dataset conversion fails during Hydra schema initialization
because `DINODatasetConvertConfig` declares string fields with `None` defaults.
Do not advertise DINO dataset conversion as a model-skill action until the SDK
schema is fixed.
