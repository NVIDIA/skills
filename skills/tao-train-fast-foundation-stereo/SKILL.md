---
name: tao-train-fast-foundation-stereo
description: Real-time stereo depth estimation using FastFoundationStereo (FFS), the distilled bp2 commercial variant of
  FoundationStereo. Predicts disparity maps from stereo image pairs with ~10× lower latency than full FoundationStereo. Use
  when training, evaluating, exporting, or running inference for a TAO FastFoundationStereo (FFS) model. Trigger phrases
  include "train fast stereo", "real-time stereo disparity", "FastFoundationStereo", "distilled stereo depth".
license: Apache-2.0
compatibility: Requires docker + nvidia-container-toolkit.
metadata:
  version: "0.1.0"
  author: NVIDIA Corporation
allowed-tools: Read Bash
tags:
- stereo
- depth
- estimation
- realtime
- distilled
---

# Depth Net Fast Stereo

Real-time stereo depth estimation using **FastFoundationStereo (FFS)** — the bp2 commercial distilled variant of FoundationStereo. Predicts disparity maps from rectified stereo image pairs with per-layer pruned widths for real-time inference.

The mono / stereo / fast-stereo skills share the unified TAO `depth_net` CLI; FFS is selected via `model.model_type: FastFoundationStereo`. FFS differs from `FoundationStereo` only in pruned per-layer widths and a serialized forward path; everything else (entrypoint, action verbs, dataset classes, deploy chain) is identical to `depth-net-stereo`.

For TAO Deploy TensorRT actions (`gen_trt_engine`, TensorRT `evaluate`, TensorRT `inference`), read `references/tao-deploy-fast-foundation-stereo.md` first. The deploy spec template lives at `references/spec_template_deploy.yaml`.

## When to Use

Use this skill to train, evaluate, export, or run inference for a TAO FastFoundationStereo model. Two supported use cases:

FFS raw-deploy and bp2-finetune flows require a pre-trained bp2 commercial checkpoint (`model_best_bp2_serialize.pth`). The default PyT image does not guarantee that this file is present on disk, so treat the checkpoint path as a required user/registry artifact. If no bp2 checkpoint is available, scratch training is still usable for workflow validation, but the resulting metrics are not representative of the bp2 model.

1. **Raw deploy** — use the bp2 ckpt as-is. Skip `train`; run `inference` / `evaluate` / `export` / `gen_trt_engine` directly with the bp2 file as the action's checkpoint.
2. **Finetune on user data** — set `train.pretrained_model_path` to the bp2 file, train on user data, then verify + deploy on the resulting ckpt. The full 7-action sequence (train → evaluate pyt → inference pyt → export → gen_trt_engine → inference deploy → evaluate deploy) is supported.

## Train Action Policy

This model is AutoML-enabled at the model layer. Before handling any train-stage request, read `references/skill_info.yaml` and resolve the run override from either an explicit `automl_policy` value or the user's workflow request. Use `automl_policy: on` by default and only expose `on` / `off` in new launch prompts. Treat phrases like "turn off AutoML", "disable AutoML", "no HPO", or "plain training" as `automl_policy: off` for this run only. When `automl_policy: on`, `automl_enabled: true`, and both `schemas/train.schema.json` and `references/spec_template_train.yaml` are packaged, route the train action through `tao-skill-bank:tao-run-automl` by default with this model's `skill_dir`. Preserve workflow/application overrides for datasets, specs, output directories, GPU/platform settings, parent checkpoints, and `automl_policy`. Use direct model training only when `automl_policy: off` or the packaged train schema/template is missing; in the missing-schema case, report that AutoML is enabled but not runnable for this model until schemas are generated.

FFS shares the `depth_net_stereo` schema but its bp2 architecture widths are fixed invariants. For default AutoML, search only `train.optim.lr` and `train.optim.lr_decay` unless the user explicitly requests a wider search. Do not include FFS architecture fields such as `model.volume_dim`, `model.hidden_dims`, or other bp2 width settings in the default search space.
Non-train actions such as `evaluate`, `inference`, `export`, and deploy flows stay in this model skill. The per-run `automl_policy` override does not change model metadata.

## Workflow

### Prerequisites — data accessibility

Your dataset (left + right images + GT disparity for train / evaluate, left + right only for inference) must be reachable from inside the container:
- **SDK runner**: place files at the S3 paths the runner resolves (`S3_TRAIN` / `S3_EVAL` placeholders shown in spec overrides).
- **Direct `docker run`** (e.g. local testing): mount the host dataset root read-only at the same in-container path:

```
docker run ... -v <host_data_root>:<host_data_root>:ro <container> ...
```

The same accessibility requirement applies to the `<output_dir>` written by all actions, and to the bp2 checkpoint path.

### Step 1 — Annotation file

Per-line annotation file referenced by `data_sources[*].data_file`. Schema is identical to `depth-net-stereo`:

| Columns | Format | Use |
|---|---|---|
| 2 | `<left> <right>` | Stereo inference (no GT) |
| 3 | `<left> <right> <disparity>` | Stereo with GT |
| 4 | `<left> <right> <disparity> <occlusion_mask>` | Stereo with GT and occlusion mask |

Generate via `depth_net convert` if needed; see the `depth-net-stereo` skill for `convert_spec.yaml` template.

### Step 2 — Pair `model_type` and `dataset_name` based on your data

Use `model_type: FastFoundationStereo` for FFS. The `dataset_name` choice mirrors the stereo skill — pick the dataset-specific class when your layout matches a registered one, otherwise `GenericDataset`.

| Data category | `model_type` | `dataset_name` |
|---|---|---|
| Middlebury | `FastFoundationStereo` | `Middlebury` |
| KITTI | `FastFoundationStereo` | `Kitti` |
| ETH3D | `FastFoundationStereo` | `Eth3d` |
| FSD synthetic | `FastFoundationStereo` | `FSD` |
| IsaacReal synthetic | `FastFoundationStereo` | `IsaacRealDataset` |
| Crestereo synthetic | `FastFoundationStereo` | `Crestereo` |
| Other / non-canonical | `FastFoundationStereo` | `GenericDataset` |

For inference with 2-column annotations (left + right, no GT), use `dataset_name: GenericDataset` regardless of layout.

### Step 3 — Set the bp2 distilled width overrides

FFS requires 15 model-section width override fields whose values match the bp2 commercial checkpoint exactly. Omitting any field falls back to TAO defaults that do **not** match the bp2 ckpt and produce shape-mismatch errors at forward time. See `references/setup-and-run.md` for the full copy-as-is `model:` block and notes. The spec templates at `references/spec_template_*.yaml` carry this block as the canonical source.

### Step 4 — Write spec yaml from spec overrides

Copy the action block from `references/spec-overrides.md`. Replace:
- `model.model_type: FastFoundationStereo` (already set)
- `dataset.<...>.data_sources[*].dataset_name` from Step 2
- `dataset.<...>.data_sources[*].data_file` with the path from Step 1
- For raw deploy use cases (no train): set `<action>.checkpoint` to the bp2 file path
- For finetune use cases: set `train.pretrained_model_path` to the bp2 file path

For chained train → next-action checkpoint path resolution and shape-consistency notes, see `references/setup-and-run.md`. SDK-runner deploys resolve handoff automatically via `parent_job_id` — see `references/parent-model-inference.md`.

### Step 5 — Run

Create writable home/cache directories inside the mounted output path before using `--user`, then launch `docker run ... depth_net <action> -e <spec.yaml>`. See `references/setup-and-run.md` for the full `mkdir` + `docker run` command, the `--user` rationale, and the local bind-mount `__pycache__` tip.

### Step 6 — Verify

Check container exit code 0 and a populated `status.json` `kpi` block. For `train` inspect per-step `train_loss` directly (the entrypoint reports `Execution status: PASS` even when loss is NaN); for `evaluate` rely on `epe` / `bp1` / `bp2` / `bp3` / `d1` / `rmse`; for `inference` check artifacts under `results_dir`. The pyt-vs-deploy KPI namespace difference and the expected deploy drift are detailed in `references/setup-and-run.md`.

### 7-action deploy flow

```
train (optional)            → finetuned ckpt
evaluate (pyt)              → PyT eager EPE / bp on val GT
inference (pyt)             → PyT eager disparity samples (visual sanity)
export                      → static fp32 ONNX (recommended at 480×736 or 320×736)
gen_trt_engine             → fp16 TRT engine on static ONNX path
inference (deploy)         → TRT disparity samples
evaluate (deploy)          → TRT EPE / bp drift vs PyT eager fp32
```

Skip `train` for raw-bp2 deploy. The remaining 6 actions (or the 4 deploy-only verbs starting from `export`) cover both use cases.

## Training Requirements

- **Valid `dataset_name` values for stereo `data_sources`** (case-insensitive): `FSD`, `IsaacRealDataset`, `Crestereo`, `Middlebury`, `Eth3d`, `Kitti`, `GenericDataset`
- **Monitoring metric:** val/loss

### Per-Action Dataset Requirements

| Action | Spec Key | Source | Files | List? |
|---|---|---|---|---|
| evaluate | dataset.test_dataset.data_sources | eval_dataset | data_file: annotations.txt + dataset_name | Yes |
| inference | dataset.infer_dataset.data_sources | inference_dataset | data_file: annotations.txt + dataset_name | Yes |
| train | dataset.train_dataset.data_sources | train_datasets | data_file: annotations.txt + dataset_name | Yes |
| train | dataset.val_dataset.data_sources | eval_dataset | data_file: annotations.txt + dataset_name | Yes |

### Typical Spec Overrides

Data source overrides are **mandatory for every action**. Each `data_sources` entry is a dict with **two mandatory fields**: `data_file` and `dataset_name`. The `model.*` width fields are also mandatory — see Step 3. See `references/spec-overrides.md` for the `FFS_MODEL_BLOCK` and per-action (train / evaluate / inference / export) Python override dicts.

## Eval Dataset

Optional. Val dataset configured via `dataset.val_dataset.data_sources` (each entry needs `data_file` and `dataset_name`).

## Important Parameters

Key knobs include `model.model_type` (`FastFoundationStereo`), `model.encoder` (`vitl`), `model.max_disparity` (set `192` explicitly — schema default `416` causes severe drift), `model.mixed_precision` (`false`), `model.gwc_feature_normalize` (`true`), `model.volume_dim` (`28`), `model.valid_iters` (`8`), and per-split `batch_size` / `workers` / `crop_size` / `data_sources`. Full parameter reference, evaluation metrics, multi-GPU / multi-node spec keys, export / TRT defaults, the export use-case matrix, and hardware guidance are in `references/important-parameters.md`.

## Error Patterns

For `shape mismatch`, `gwc_feature_normalize` schema errors, `max_disparity` drift, negative disparity, `depth_net_stereo: not found`, the pyt-`evaluate` `crop_size` asymmetry, the `Failed to import SAM3` warning, and the dynamic-engine stride-incompatible silent failure, see `references/error-patterns.md`.

## Spec Param / Parent Model Inference

Model-specific inference mappings (per-action spec field → inference function) for train / evaluate / inference / export / gen_trt_engine, plus `parent_job_id` / `parent_model` resolution and raw-bp2 explicit-checkpoint handling, are in `references/parent-model-inference.md`. Generated runners should read that section and apply the mappings with SDK helpers before `create_job()`.

## Deployment

- [tao-deploy-fast-foundation-stereo](references/tao-deploy-fast-foundation-stereo.md)
