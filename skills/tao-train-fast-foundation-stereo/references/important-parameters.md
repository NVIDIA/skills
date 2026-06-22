# Important Parameters

- **model.model_type**: Must be `FastFoundationStereo` for this skill.
- **model.encoder**: ViT backbone size; bp2 ckpt was trained with `vitl`. Other sizes will fail to load the bp2 weights.
- **model.hidden_dims**: bp2 uses `[128]` (single-GRU). Do **not** use the full-FS default `[128, 128, 128]` — shape-mismatch on the GRU head.
- **model.n_gru_layers**: bp2 uses `1`. Pair with `hidden_dims: [128]`.
- **model.max_disparity**: bp2 commercial uses `192`. The TAO Core schema default for this field is `416` — if the spec yaml's `model:` block does not explicitly set `max_disparity: 192`, OmegaConf falls back to the schema default and the cost volume is built with 2× the correct number of disparity levels (~104 vs the bp2-trained 48 at 1/4 scale). The model still loads and runs, but per-pixel disparity drifts severely from upstream because the cost-volume softmax peak shifts out of the trained regime. **Always set `model.max_disparity: 192` explicitly in the spec for FFS-bp2 deploy** — do not rely on the schema default. The setting on `dataset.max_disparity` is a separate dataset-side knob and does not propagate to the model.
- **model.mixed_precision**: Recommend `false` for FFS-bp2 train and pyt eval. The bp2 commercial ckpt was distilled upstream with bf16 amp, but the FS trainer in TAO does not support bf16 (only fp32 and fp16). Using `mixed_precision: false` (= fp32 forward) gives the cleanest pyt-vs-deploy parity check.
- **model.gwc_feature_normalize**: Must be `true` for FFS-bp2. The bp2 model was trained with normalized group-wise correlation cost volume, and the model code without this flag produces broken disparity (negative values, large drift from upstream baseline). Required for both pyt and deploy paths.
- **model.train_iters**: GRU refinement iterations during training. Default 22.
- **model.valid_iters**: GRU refinement iterations during inference / eval. bp2 ckpt was distilled targeting `8`; values higher than 8 do not improve quality.
- **model.volume_dim**: Cost volume Conv output channels. Schema default `32` (full-FS); FFS bp2 ckpt requires `28` — must override explicitly. Changing breaks bp2 ckpt key-shape match.
- **model.low_memory**: Memory optimization level. Range 0-4. Higher = less memory, slower.
- **dataset.dataset_name**: Top-level dataset family identifier (`StereoDataset`).
- **dataset.{train,val,test,infer}_dataset.batch_size**: Per-split batch size. Use `1` for variable-aspect datasets (Middlebury / KITTI / ETH3D) and during eval / TRT comparison; larger batch sizes are fine for fixed-shape synthetic data.
- **dataset.{train,val,test,infer}_dataset.workers**: Per-split DataLoader worker count.
- **dataset.{train,val,test,infer}_dataset.augmentation.crop_size**: Per-split crop. Match `export.input_height` / `export.input_width` and the deploy-side `evaluate` crop_size for end-to-end shape consistency.
- **dataset.{train,val,test,infer}_dataset.data_sources**: List of `{data_file, dataset_name}` dicts.
- **train.optim.lr**: Learning rate. Default 1e-4 (AdamW). For bp2 finetune, prefer `1e-5` (matches upstream).
- **train.precision**: Training precision. Options: `fp32` (recommended for FFS-bp2), `fp16`. (bf16 is not supported by the FS trainer.)
- **train.distributed_strategy**: Distribution strategy. Options: ddp, fsdp.
- **inference.save_raw_pfm**: Pyt inference action only — when `true`, the per-image disparity is dumped as a raw `.pfm` next to the colorized `.png`. Deploy inference (TRT engine path) emits only the colorized `.png` under `predicted_depth/<scene>_im0.png`; the `save_raw_pfm` knob is not consumed there. Use the pyt inference path if raw `.pfm` output is required.

## Evaluation Metrics

`StereoDepthEvaluator` emits a fixed metric set; only the disparity-domain metrics are meaningful:

| Metric | Meaning | Use |
|---|---|---|
| `epe` | mean End-Point-Error in pixels | primary stereo metric |
| `bp1` / `bp2` / `bp3` | fraction of pixels with EPE > 1 / 2 / 3 px | quality thresholds |
| `d1` | KITTI-style outlier rate (EPE > 3 px AND > 5% of GT disparity) | KITTI-comparable headline |
| `rmse` | RMSE on disparity values | sensitivity to large errors |

The same evaluator also emits `abs_rel`, `sq_rel`, `rmse_log` — these are formulated for monocular metric depth and produce non-meaningful values on disparity. Ignore them for stereo evaluation.

## Multi-GPU / Multi-Node

**Launch method:** Lightning-managed (single `python` process, Lightning spawns workers). Same DDP / FSDP behavior as `depth-net-stereo`.

| Spec Key | Description | Default |
|----------|-------------|---------|
| `train.num_gpus` | Number of GPUs | 1 |
| `train.gpu_ids` | GPU device indices | [0] |
| `train.num_nodes` | Number of nodes | 1 |
| `train.distributed_strategy` | `ddp` or `fsdp` | `ddp` |

Multi-node requires `WORLD_SIZE`, `NODE_RANK`, `MASTER_ADDR`, `MASTER_PORT` env vars.

## Export / TRT Defaults

- TRT data types: FP32, FP16.
- Recommended TRT precision for FFS-bp2: `fp16` on the static-shape ONNX path (lowest drift). Dynamic-shape path supports both `fp32` (default; static-fp32 parity) and `fp16` (latency-critical multi-resolution; higher drift than static fp16, may NaN under some checkpoint states — fall back to fp32 if observed). See `references/tao-deploy-fast-foundation-stereo.md` deployment matrix.
- `export` always emits a **fp32 ONNX** regardless of `model.mixed_precision`. The fp16 vs fp32 selection happens at the `gen_trt_engine` step via `gen_trt_engine.tensorrt.data_type`.
- For static-shape FFS at 480×736: `export.batch_size: 1`, `export.opset_version: 17`, `export.on_cpu: False`.
- **`export.batch_size`**: positive int (default `1`) — static batch dimension; `-1` enables a dynamic batch axis on the ONNX input.
- **`export.dynamic_hw`**: bool (default `false`) — `true` enables dynamic H/W axes on the ONNX input. **FFS only.** FS / mono models ignore this flag with a warning and fall back to static H/W (their DINOv2 backbone constant-folds positional embeddings into the trace, so dynamic H/W at runtime would produce a wrong-shape pos-embed mismatching the actual patch tokens — silent crash). FFS uses EdgeNeXt only and is safe.

### Export use-case matrix

`export.batch_size` and `export.dynamic_hw` are independent. The four combinations:

| Use case | `batch_size` | `dynamic_hw` | Resulting ONNX |
|---|---|---|---|
| Fixed-batch fixed-resolution (most common, production fp16) | `1` (positive) | `false` | static `[1, 3, H, W]` |
| Variable-batch fixed-resolution | `-1` | `false` | dynamic batch only |
| Variable-resolution single-batch (FFS only) | `1` (positive) | `true` | dynamic H/W only |
| Variable-resolution + variable-batch (FFS only) | `-1` | `true` | both batch and H/W dynamic |

For FS / mono models, `dynamic_hw: true` is automatically ignored with a warning and the engine falls back to static H/W. Only `FastFoundationStereo` supports dynamic H/W due to its EdgeNeXt-only encoder.

## Hardware

- Minimum 1 GPU, 24 GB+ VRAM per GPU recommended (A6000 / A100). FFS is ~10× lower-memory than full FoundationStereo at the same input shape, but cost-volume convolution still dominates peak VRAM during training.
- For inference / deploy on edge: A2 / Orin-class GPUs handle FFS at 480×736 fp16 within real-time budget.
- `model.low_memory > 0` for constrained GPUs at training time.
- fp32 recommended for training (bf16 unsupported by FS trainer).
