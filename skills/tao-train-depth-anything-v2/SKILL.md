---
name: tao-train-depth-anything-v2
description: Monocular depth estimation using Metric Depth Anything v2 or Relative Depth Anything architectures. Predicts
  per-pixel depth from single RGB images. Use when training, evaluating, exporting, or running inference for a TAO
  monocular depth model. Trigger phrases include "train monocular depth", "DepthAnything v2", "metric depth from single
  image", "monocular depth estimation".
license: Apache-2.0
compatibility: Requires docker + nvidia-container-toolkit.
metadata:
  version: "0.1.0"
  author: NVIDIA Corporation
allowed-tools: Read Bash
tags:
- monocular
- depth
- estimation
---

# Depth Net Mono

Monocular depth estimation using Metric Depth Anything v2 or Relative Depth Anything architectures. Predicts per-pixel depth from single RGB images.

Pretrained checkpoint loading varies by model variant and use case — see the **Pretrained checkpoint loading — use case matrix** in `references/parameters.md`.

The mono and stereo skills both invoke the unified TAO `depth_net` CLI inside the container; the mono/stereo family is selected via `model.model_type` (see `references/parameters.md`).

For TAO Deploy TensorRT actions (`gen_trt_engine`, TensorRT `evaluate`, and TensorRT `inference`), read `references/tao-deploy-depth-anything-v2.md` first. The deploy spec template lives in this skill's `references/spec_template_deploy.yaml`.

PyT actions packaged by this model skill: `train`, `evaluate`, `inference`, `export`, and `quantize`. The PyT `depth_net` entrypoint does not accept a PyT-side `gen_trt_engine` action in the current TAO image. The `gen_trt_engine` action metadata must run with the TAO Deploy container, and the deploy workflow remains the deploy-specific entrypoint.

## Train Action Policy

This model is AutoML-enabled at the model layer. Before handling any train-stage request, read `references/skill_info.yaml` and resolve the run override from either an explicit `automl_policy` value or the user's workflow request. Use `automl_policy: on` by default and only expose `on` / `off` in new launch prompts. Treat phrases like "turn off AutoML", "disable AutoML", "no HPO", or "plain training" as `automl_policy: off` for this run only. When `automl_policy: on`, `automl_enabled: true`, and both `schemas/train.schema.json` and `references/spec_template_train.yaml` are packaged, route the train action through `tao-skill-bank:tao-run-automl` by default with this model's `skill_dir`. Preserve workflow/application overrides for datasets, specs, output directories, GPU/platform settings, parent checkpoints, and `automl_policy`. Use direct model training only when `automl_policy: off` or the packaged train schema/template is missing; in the missing-schema case, report that AutoML is enabled but not runnable for this model until schemas are generated.

Non-train actions such as `evaluate`, `inference`, `export`, and deploy flows stay in this model skill. The per-run `automl_policy` override does not change model metadata.

## Workflow

### Prerequisites — data accessibility

Your dataset (RGB images + GT depth files) must be reachable from inside the container:
- **SDK runner**: place files at the S3 paths the runner resolves (the `S3_TRAIN` / `S3_EVAL` placeholders shown in **Typical Spec Overrides**). The runner handles S3 → container-path mounting transparently.
- **Direct `docker run`** (e.g. local testing): mount the host dataset root read-only at the same in-container path:

```
docker run ... -v <host_data_root>:<host_data_root>:ro <container> ...
```

The same accessibility requirement applies to the `<output_dir>` written by all actions.

### Step 1 — Annotation file

Per-line annotation file referenced by `data_sources[*].data_file`:

| Columns | Format | Use |
|---|---|---|
| 1 | `<image>` | Mono inference (no GT) |
| 2 | `<image> <gt_depth>` | Mono with GT |

Do not pass stereo annotation rows such as `<left_image> <right_image>
<gt_depth>` directly to mono train/evaluate/inference. If only a stereo depth
dataset is available, derive a mono annotation file by keeping the left image
and GT depth columns, then mount or stage the image/depth archive at the same
container paths referenced by that derived annotation file.

If you already have one, point to it. Otherwise generate via `depth_net convert`:

```
depth_net convert -e <convert_spec.yaml>
```

`convert_spec.yaml` template:

```yaml
results_dir: <directory where generated annotation files are written>
data_root: <directory whose immediate children are scene/sample folders that contain your image+depth files; convert walks data_root recursively but expects per-scene subdirectories at one level below>
image_dir_pattern: [<substring matching left/RGB image paths>]
depth_dir_pattern: [<substring matching GT depth paths>]
image_extension: ''     # optional .endswith filter, e.g. '.jpg'
depth_extension: ''     # optional, swapped during depth derivation, e.g. '.png'
split_ratio: 0.0        # 0.0/1.0 = test-only; 0.8 = 80/20 train+val
```

`convert` walks `data_root` recursively, selects paths whose path-string contains *all* substrings in `image_dir_pattern` (AND-filter), then derives the depth path by replacing `image_dir_pattern[0]` with `depth_dir_pattern[0]` and `image_extension` with `depth_extension`. Inspect your dataset's directory layout and identify the substring distinguishing RGB images from depth files (e.g. `rgb_` vs `sync_depth_`).

`data_root` must point at the parent that contains the per-scene subdirectories (e.g. for NYU eval, use `/data/nyu_v2/eval/test`, not `/data/nyu_v2/eval/test/bathroom` — the latter limits the walk to a single scene). Always include the leading dot in `image_extension` / `depth_extension` (e.g. `'.jpg'` not `'jpg'`); the substring swap is form-sensitive and a mismatch silently corrupts derived paths.

### Step 2 — Pair `model_type` and `dataset_name` based on your data

Default — generic class for each task:

| Data category | `model_type` | `dataset_name` |
|---|---|---|
| Disparity-encoded data (pixels) | `RelativeDepthAnything` | `RelativeMonoDataset` |
| Metric depth (meters) | `MetricDepthAnything` | `MetricMonoDataset` |
| Mono inference (no GT, any image) | matches train choice | `RelativeMonoDataset` or `MetricMonoDataset` |

Dataset-specific class — switch when the data needs preprocessing the generic class does not perform:

| Special case | `model_type` | `dataset_name` | What the class adds |
|---|---|---|---|
| NYU `sync_depth_*.png` (raw uint16 millimetres) — relative | `RelativeDepthAnything` | `NYUDV2Relative` | mm→m unit conversion + Eigen evaluation crop |
| NYU `sync_depth_*.png` (raw uint16 millimetres) — metric | `MetricDepthAnything` | `NYUDV2` | same |

Using a generic class on data that requires unit conversion (e.g. raw NYU uint16 PNGs) results in an empty valid mask and silent `train_loss = NaN`. Match the class to your data's encoding.

For relative mono data (`RelativeMonoDataset` or `NYUDV2Relative`), leave `dataset.min_depth` and `dataset.max_depth` unset or set both to `null`. Non-null metric depth ranges are passed into the relative dataset constructor and fail with `BaseRelativeMonoDataset.__init__() got an unexpected keyword argument 'min_depth'`.

### Step 3 — Write spec yaml from Typical Spec Overrides

Copy the action block from **Typical Spec Overrides** (`references/spec-overrides.md`). Replace:
- `model.model_type` from Step 2
- `dataset.<...>.data_sources[*].dataset_name` from Step 2
- `data_sources[*].data_file` with the path from Step 1 (S3 path under SDK runner, host path for direct docker)
- For metric finetune: additionally apply the **Metric Variant Finetuning Recipe** in `references/finetuning-recipes.md`.

For mono training set `train.precision: fp32` (recommended) or `bf16` (Ampere SM80+, alternative).

### Step 4 — Run

Create writable home/cache directories inside the mounted output path before using
`--user`. Some TAO containers do not have an `/etc/passwd` entry for the host UID,
and PyTorch / matplotlib need writable cache paths when running as that UID.

```bash
mkdir -p <output_dir>/home \
         <output_dir>/.cache/matplotlib \
         <output_dir>/.cache/torchinductor \
         <output_dir>/.cache/xdg
```

```
docker run --gpus 'device=0' --shm-size 16G --ipc=host \
  --user "$(id -u):$(id -g)" \
  -e USER="$(id -un)" \
  -e LOGNAME="$(id -un)" \
  -e HOME=<output_dir>/home \
  -e MPLCONFIGDIR=<output_dir>/.cache/matplotlib \
  -e TORCHINDUCTOR_CACHE_DIR=<output_dir>/.cache/torchinductor \
  -e XDG_CACHE_HOME=<output_dir>/.cache/xdg \
  -v <data_root>:<data_root>:ro \
  -v <output_dir>:<output_dir> \
  <container> \
  depth_net <action> -e <spec.yaml>
```

Without `--user "$(id -u):$(id -g)"` the container writes outputs as `nobody:nogroup`, blocking host-side cleanup and retry.

### Step 5 — Verify

- Container exit code 0
- `status.json` `kpi` block populated
- For `train`: inspect per-step `train_loss` directly — the entrypoint reports `Execution status: PASS` even when `train_loss = NaN` (see the Metric Variant Finetuning Recipe → Sanity-run PASS criteria in `references/finetuning-recipes.md`)
- For `evaluate` / `inference`: artifacts under `results_dir`

For TAO Deploy TensorRT actions (`gen_trt_engine`, TensorRT `evaluate`, and TensorRT `inference`), read `references/tao-deploy-depth-anything-v2.md` first. Deploy spec templates live in this skill's `references/` folder with the `spec_template_deploy_*.yaml` prefix.

## Training Requirements

- **Valid `dataset_name` values for mono `data_sources`** (case-insensitive): `ThreeDVLM`, `FSD`, `NvCLIP`, `IssacStereo`, `Crestereo`, `Middlebury`, `NYUDV2`, `NYUDV2Relative`, `RelativeMonoDataset`, `MetricMonoDataset`. `NYUDV2` carries metric depth GT (meters) — pair with `MetricDepthAnything`; `NYUDV2Relative` is the same data with relative-depth conventions — pair with `RelativeDepthAnything`.
- **Monitoring metric:** val/d1, val/loss
- For AutoML sanity runs on the packaged relative-depth smoke data, use `val/d1` as the primary monitor. `val/loss` can be emitted as `NaN` even when the trainer exits successfully and writes a usable checkpoint, so it is not a reliable AutoML objective unless the run's status metrics show a finite value.

### Per-Action Dataset Requirements

| Action | Spec Key | Source | Files | List? |
|---|---|---|---|---|
| evaluate | dataset.test_dataset.data_sources | eval_dataset | data_file: annotations.txt + dataset_name | Yes |
| inference | dataset.infer_dataset.data_sources | inference_dataset | data_file: annotations.txt + dataset_name | Yes |
| quantize | dataset.train_dataset.data_sources | train_datasets | data_file: annotations.txt + dataset_name | Yes |
| quantize | dataset.val_dataset.data_sources | eval_dataset | data_file: annotations.txt + dataset_name | Yes |
| quantize | dataset.quant_calibration_dataset.images_dir | train_datasets | images.tar.gz | No |
| train | dataset.train_dataset.data_sources | train_datasets | data_file: annotations.txt + dataset_name | Yes |
| train | dataset.val_dataset.data_sources | eval_dataset | data_file: annotations.txt + dataset_name | Yes |

### Typical Spec Overrides

Data source overrides are **mandatory for every action** — construct data source paths from the Per-Action Dataset Requirements table above and include them in `spec_overrides`. Each `data_sources` entry is a dict with **two mandatory fields**: `data_file` and `dataset_name`. See `references/spec-overrides.md` for the full per-action override blocks (`train`, `evaluate`, `export`, `inference`, `quantize`), the `S3_TRAIN` / `S3_EVAL` placeholders, the relative-variant precision recommendation, and the `quantize` known-issue note.

## Eval Dataset

Optional. Val dataset configured via `dataset.val_dataset.data_sources` (each entry needs `data_file` and `dataset_name`).

## Important Parameters

See `references/parameters.md` for the full parameter glossary (model, train, dataset, export, and inference keys with options, defaults, and sources) and the **Pretrained checkpoint loading — use case matrix**.

## Finetuning Recipes

See `references/finetuning-recipes.md` for:
- **Relative Variant Finetuning Recipe** — finetune from a TAO-trained `RelativeDepthAnything` checkpoint (lr `5e-6`, `LambdaLR`, sanity-vs-convergent guidance, deploy LSQ alignment note).
- **Metric Variant Finetuning Recipe** — checkpoint compatibility, required overrides, the dataset normalization block (`normalize_depth`/`min_depth`/`max_depth`) required in train AND export specs, trainer-enforced defaults, precision, the 1-epoch sanity-run override, and the Sanity-run PASS criteria with the NaN-mitigation order.

## Multi-GPU / Multi-Node

**Launch method:** Lightning-managed (single `python` process, Lightning spawns workers).

| Spec Key | Description | Default |
|----------|-------------|---------|
| `train.num_gpus` | Number of GPUs | 1 |
| `train.gpu_ids` | GPU device indices | [0] |
| `train.num_nodes` | Number of nodes | 1 |
| `train.distributed_strategy` | `ddp` or `fsdp` | `ddp` |

- `ddp` with activation checkpointing: `find_unused_parameters=False`
- `ddp` without: `find_unused_parameters=True`
- `fsdp` forces precision to FP16

**Multi-node env vars** (set by orchestrator): `WORLD_SIZE`, `NODE_RANK`, `MASTER_ADDR`, `MASTER_PORT`, `NUM_GPU_PER_NODE`.

## Export / TRT Defaults

- TRT data types: FP32, BF16 (Ampere SM80+). FP16 is not supported for the ViT-L mono backbone.
- Fresh-install TRT precision: `fp32`. BF16 is supported on Ampere SM80+ hardware, but keep smoke tests on FP32 unless the user explicitly requests BF16.

## Hardware

Minimum 1 GPU(s), recommended 2 GPU(s). 24GB+ VRAM per GPU. ViT-Large encoder is memory intensive. Use `fp32` (recommended) or `bf16` (Ampere SM80+, alternative) for training. Activation checkpointing is available for larger inputs.

## Error Patterns

See `references/troubleshooting.md` for the full error-pattern catalog (depth range mismatch, relative dataset rejecting `min_depth`, missing pretrained weights, `encoder` key location, `dataset_name` not in struct, `depth_net_mono` not found, metric variant hyperparameter sourcing, and export ONNX overwrite).

## Spec Param / Parent Model Inference

See `references/spec-param-inference.md` for the model-specific inference mappings (the TAO Core `depth_net_mono.config.json` action table), checkpoint-file naming under `<results_dir>/train/`, the `dn_model_latest.pth` policy, the parent-`gen_trt_engine` rationale, and the `parent_model` / `parent_job_id` resolution rules.

## Deployment

- [tao-deploy-depth-anything-v2](references/tao-deploy-depth-anything-v2.md)
