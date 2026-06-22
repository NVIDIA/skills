# FoundationStereo Parameters and Defaults

## Important Parameters

- **model.model_type**: Architecture. Default `FoundationStereo` for stereo. Only `FoundationStereo` is selectable in the current release.
- **model.encoder**: Backbone encoder (top-level `model` field, not nested under `stereo_backbone`). Options: `vits`, `vitb`, `vitl`, `vitg`. Schema default `vitl`; **FS small NGC ckpt requires `vits` — must override explicitly** (silent shape mismatch on `patch_embed` / ViT block keys without it).
- **model.max_disparity**: Maximum disparity range. Default 416, range 1-416.
- **model.hidden_dims**: Hidden dimensions in GRU refinement. Default `[128, 128, 128]`.
- **model.train_iters**: GRU refinement iterations during training. Default 22.
- **model.volume_dim**: Cost volume dimension. Schema default `32`, but the `FoundationStereo` class hardcodes `volume_dim = 28` at construction (`foundation_stereo.py:51`) — the schema field is currently a no-op for FS. Override is unnecessary; the model always builds at 28.
- **model.low_memory**: Memory optimization level. Range 0-4. Higher = less memory.
- **dataset.dataset_name**: Top-level dataset family identifier (e.g., `StereoDataset`).
- **dataset.baseline**: Stereo camera baseline. Default `193.001/1e3` meters.
- **dataset.focal_x**: Camera focal length X. Default `1998.842`.
- **dataset.{train,val,test,infer}_dataset.batch_size**: Per-split batch size.
- **dataset.{train,val,test,infer}_dataset.workers**: Per-split DataLoader worker count (the field name is `workers`, not `num_workers`).
- **dataset.{train,val,test,infer}_dataset.augmentation.crop_size**: Per-split crop size (e.g., `[320, 736]`). Match `export.input_height`/`export.input_width` and the deploy-side `evaluate` crop_size for end-to-end shape consistency (see `tao-deploy-foundation-stereo.md` for the deploy-side shape table).
- **dataset.{train,val,test,infer}_dataset.data_sources**: List of `{data_file, dataset_name}` dicts. Both fields are mandatory per entry.
- **train.optim.lr**: Learning rate. Default 1e-4 (AdamW).
- **train.precision**: Training precision. Options: fp32 (recommended), fp16. (bf16 is not supported by the FS trainer.)
- **train.distributed_strategy**: Distribution strategy. Options: ddp, fsdp.
- **export.batch_size**: ONNX batch size. `1` = static (matches NGC release), `-1` = batch axis dynamic (height and width are always taken from the trace shape; the DINOv2 + EdgeNeXt backbone constant-folds the patch count, so H/W dynamic is not supported). Default `-1`.

## Evaluation Metrics

`StereoDepthEvaluator` (`nvidia_tao_deploy/cv/depth_net/evaluation/stereo_evaluator.py`) emits a fixed metric set; only the disparity-domain metrics are meaningful for stereo:

| Metric | Meaning | Use |
|---|---|---|
| `epe` | mean End-Point-Error in pixels | primary stereo metric |
| `bp1` / `bp2` / `bp3` | fraction of pixels with EPE > 1 / 2 / 3 px | quality thresholds |
| `d1` | KITTI-style outlier rate (EPE > 3 px AND > 5% of GT disparity) | KITTI-comparable headline |
| `rmse` | RMSE on disparity values | sensitivity to large errors |

The same evaluator also emits `abs_rel`, `sq_rel`, `rmse_log`. These are formulated for monocular depth (relative-error normalised by GT depth in metres) and produce numerically large, **non-meaningful** values when applied to disparity tensors. Ignore them for stereo evaluation; rely on `epe` / `bp*` / `d1` / `rmse`.

## Multi-GPU / Multi-Node

**Launch method:** Lightning-managed (single `python` process, Lightning spawns workers).

| Spec Key | Description | Default |
|----------|-------------|---------|
| `train.num_gpus` | Number of GPUs | 1 |
| `train.gpu_ids` | GPU device indices | [0] |
| `train.num_nodes` | Number of nodes | 1 |
| `train.distributed_strategy` | `ddp` or `fsdp` | `ddp` |

Same DDP/FSDP behavior as depth-net-mono. Multi-node requires `WORLD_SIZE`, `NODE_RANK`, `MASTER_ADDR`, `MASTER_PORT` env vars.

## Export / TRT Defaults

- TRT data types: FP32, FP16.
- Static-shape ONNX (`export.batch_size: 1`): `fp16` supported (recommended, best EPE).
- Batch-only dynamic ONNX (`export.batch_size: -1`): `fp16` supported. Engine accepts variable batch size; height and width are pinned to the trace shape.
- Height and width are always pinned to the trace shape; H/W-dynamic engines are not supported. Build separate engines for different (H, W) targets.
- For the NGC release (576×960), set `export.batch_size: 1`, `export.opset_version: 17`, `export.on_cpu: True` (CPU export is required at 576×960 to avoid GPU OOM during the trace).
- For user-trained fp16 export, pair `opset_version` to `on_cpu`: `on_cpu: True` (CPU trace) accepts either opset 16 or 17 deterministically; `on_cpu: False` (GPU trace) accepts only opset 16 (opset 17 + on_cpu=False is broken on TRT 10.13 fp16). At `on_cpu=False + opset 16` the fp16 build is occasionally non-deterministic — re-run on a `costTensor::indexOfMin` or `optimizer::reduce` assertion. fp32 builds are unaffected. See `tao-deploy-foundation-stereo.md` for the validation table.
- `export.on_cpu` is driven by GPU trace memory: `False` for ≤320×736 (fits 47 GB VRAM), `True` for ≥480×736 (PyTorch trace OOMs at GPU). Prefer `on_cpu: True` whenever feasible — fp16 builds at `on_cpu=True` are empirically deterministic at every tested shape (including NGC release 576×960).
- See `tao-deploy-foundation-stereo.md` for the three supported deploy paths (NGC static / user-trained static / user-trained batch-only-dynamic).

## Hardware

Minimum 1 GPU(s), recommended 4 GPU(s). 24GB+ (A100 recommended) VRAM per GPU. Stereo matching is memory intensive due to cost volume. Use `model.low_memory > 0` for constrained GPUs. fp32 recommended for training.
