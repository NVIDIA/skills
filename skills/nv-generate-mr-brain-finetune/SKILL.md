---
name: nv-generate-mr-brain-finetune
description: Used for finetuning NV-Generate-CTMR MR-Brain v1 for T1, T2, FLAIR, SWI, or MRA data from a NIfTI datalist. Not for clinical or production data approval.
license: Apache-2.0
allowed-tools: Bash, Read, Write, WebFetch, Env
permissions: [env, file_read, file_write, network, shell]
metadata:
  author: NVIDIA MedTech Team
  tags:
    - MedTech
    - MRI
    - brain
    - finetune
---

# NV-Generate-MR-Brain-Finetune

## Purpose
- Used for finetuning the NV-Generate-CTMR `rflow-mr-brain` v1 diffusion UNet from user-supplied T1, T2, FLAIR, SWI, or MRA NIfTI training volumes.
- Not for clinical interpretation, regulatory use, or approving synthetic data for production training.
- The wrapper stages the config glue locally and delegates execution to existing upstream scripts: `scripts.diff_model_create_training_data`, `scripts.diff_model_train`, and optionally `scripts.diff_model_infer`. It does not execute the notebook.
- Manifest I/O: inputs are `datalist` and `data_base_dir`; outputs are `finetuned_checkpoint`, optional `inference_outputs`, and `result_json`.
- The underlying training contract is the upstream config/env JSON (the same one driven from cell `[10]` of `train_diff_unet_tutorial.ipynb`). The wrapper stages those JSON files for you and exposes the most-tuned fields as CLI flags; the sections below document the fields, their defaults, and how to monitor/tune a run.

## Instructions
- Read `skill_manifest.yaml` before changing arguments, side effects, or validation gates.
- Run `scripts/run_mr_brain_finetune.py` from the Medical AI Skills repo root.
- If a host agent exposes `run_script`, use `run_script("scripts/run_mr_brain_finetune.py", args=[...])`; otherwise run the Bash/Python command below.
- For a command-shape review, do not install packages, clone repositories,
  download weights, or start GPU training. Emit only the wrapper command with
  the supplied datalist, an explicit `--data-base-dir`, an explicit
  `--output-dir`, and the requested modality.
- When the user explicitly asks for a training-launch command, do not silently
  replace it with `--preflight`; include `--preflight` only for a preflight
  request.
- Use `--preflight` first when checking a new datalist; remove `--preflight` only when the user explicitly wants to launch GPU finetuning.
- For a staged preflight input bundle directory, use `BUNDLE/preflight_datalist.json` as the datalist and `BUNDLE/preflight_dataset` as `--data-base-dir` when those files are present.

## Examples

Validate and stage a preflight finetune check from an input bundle (the recommended first step — no GPU, no training). This is the single canonical command; replace `INPUT_BUNDLE` and `OUT_DIR` with your paths:

```bash
export NV_GENERATE_ROOT="${NV_GENERATE_ROOT:-$HOME/.cache/nvidia-skills/upstreams/NV-Generate-CTMR-da438fe}" && \
python skills/nv-generate-mr-brain-finetune/scripts/run_mr_brain_finetune.py \
  INPUT_BUNDLE/preflight_datalist.json \
  --data-base-dir INPUT_BUNDLE/preflight_dataset \
  --output-dir OUT_DIR \
  --modality mri_t1 \
  --preflight
```

For real GPU finetuning and other variations, see [Usage](#2-usage-one-line-training) below.

Command-shape review for a requested training launch (no setup or execution):

```bash
python skills/nv-generate-mr-brain-finetune/scripts/run_mr_brain_finetune.py \
  PATH_TO_DATALIST.json \
  --data-base-dir PATH_TO_DATA_ROOT \
  --output-dir runs/nv_generate_mr_brain_finetune \
  --epochs 2 \
  --modality mri_t1
```

## Available Scripts
| Script | Purpose | Arguments |
|---|---|---|
| `scripts/run_mr_brain_finetune.py` | Primary entrypoint declared by `skill_manifest.yaml`. | `DATALIST.json --data-base-dir DATA_DIR --output-dir OUT_DIR [--epochs N] [--modality mri_t1] [--num-gpus N] [--no-amp] [--model-config FILE] [--download-model-data] [--run-inference] [--preflight]` |

## Prerequisites
- An explicit `NV_GENERATE_ROOT` may point to the caller's local checkout and
  must contain `scripts/diff_model_create_training_data.py`,
  `scripts/diff_model_train.py`, and `scripts/diff_model_infer.py`. The result
  records its current commit.
- If `NV_GENERATE_ROOT` is unset, the wrapper searches `.workbench_data/upstreams/NV-Generate-CTMR`.
- `CUDA_VISIBLE_DEVICES` is optional and can be used to select the GPU for real training.
- Runtime requirements: NVIDIA CUDA GPU for real training, Python packages from the upstream `requirements.txt`, and downloaded MR-brain weights.
- Side effects: writes staged configs, embeddings, checkpoints, optional inference images, and logs under the caller-provided `--output-dir`; may write model caches under the upstream checkout and `~/.cache/huggingface/`; may contact `https://huggingface.co` for model assets and `https://github.com` for the upstream checkout.
- The datalist is a MONAI-style JSON object with `training[].image` paths relative to `--data-base-dir`. `training[].modality` is optional and defaults to `mri_t1`.

When no local checkout is supplied, create the recommended pinned default
checkout once:

```bash
if [ -z "${NV_GENERATE_ROOT:-}" ]; then
  export NV_GENERATE_COMMIT=da438fec6484cdb6f421f8c7051d954ebefff730
  export NV_GENERATE_ROOT="$HOME/.cache/nvidia-skills/upstreams/NV-Generate-CTMR-da438fe"
  if [ ! -d "$NV_GENERATE_ROOT/.git" ]; then
    git clone https://github.com/NVIDIA-Medtech/NV-Generate-CTMR.git "$NV_GENERATE_ROOT"
    git -C "$NV_GENERATE_ROOT" checkout --detach "$NV_GENERATE_COMMIT"
  fi
fi
```

The wrapper executes upstream code only when `NV_GENERATE_ROOT` is at the exact
manifest commit and its tracked files are clean. Supply custom training and
inference settings through the documented config flags rather than editing the
checkout. Child processes receive only an allowlist of runtime, CUDA, locale,
and certificate variables; API keys, tokens, passwords, and unrelated parent
environment values are not forwarded. The public v1 assets do not require a
credential; pre-download them if your network setup requires separate tooling.

Before a GPU run, download the exact autoencoder and MR-Brain v1 checkpoint
revisions declared by the manifest. Passing `--download-model-data` performs
these same two pinned downloads:

```bash
python -m huggingface_hub.commands.huggingface_cli download \
  nvidia/NV-Generate-CT models/autoencoder_v1.pt \
  --revision 75ac080fb1083c403793563477724c038e7d430c \
  --local-dir "$NV_GENERATE_ROOT"
python -m huggingface_hub.commands.huggingface_cli download \
  nvidia/NV-Generate-MR-Brain models/diff_unet_3d_rflow-mr-brain_v1.pt \
  --revision ef9759bf221265b2704569cdeeac20bbf03b62ee \
  --local-dir "$NV_GENERATE_ROOT"
```

## 1. Config and environment JSON (adapt to your data)

This is a thin wrapper around the upstream `train_diff_unet_tutorial.ipynb` flow. Each run performs four steps, delegating the heavy lifting to the model author's scripts:

1. **Stage configs** — copy the three config JSONs and rewrite only the run-specific paths and `n_epochs` (notebook cell 15).
2. `python -m scripts.diff_model_create_training_data` → latent `*_emb.nii.gz` embeddings (cell 17).
3. **Write embedding sidecars** — a `<emb>.nii.gz.json` per embedding with `spacing`/`modality` (and body-region indices when the model uses them). This is the one piece of glue that lives in the notebook (cell 19), not in upstream `scripts/`, and `diff_model_train` requires it; the skill owns it.
4. `python -m scripts.diff_model_train` (cell 21), optionally `python -m scripts.diff_model_infer`.

**Tune by editing the config JSON, not by adding flags.** All training/inference hyperparameters (`lr`, `batch_size`, `cache_rate`, inference `dim`/`spacing`/`num_inference_steps`/`cfg_guidance_scale`, …) live in `config_maisi_diff_model_rflow-mr-brain.json`. Edit the upstream copy, or pass your own with `--model-config FILE` (and `--env-config` / `--model-def` for the other two). The wrapper only ever rewrites the fields below.

Environment JSON (`environment_maisi_diff_model_rflow-mr-brain.json`) — fields the wrapper rewrites per run:

| Field | Set from | Notes |
|---|---|---|
| `data_base_dir` | `--data-base-dir` | Root for relative `training[].image` paths. |
| `json_data_list` | your datalist | Staged copy with per-entry `modality` filled in. |
| `embedding_base_dir`, `model_dir`, `output_dir` | `--output-dir` | Latent embeddings, checkpoints, inference images. |
| `modality_mapping_path` | upstream | Maps modality name → integer code. |
| `model_filename` | `--model-filename` | Output checkpoint name (default `diff_unet_3d_rflow-mr-brain_v1.pt`). |
| `existing_ckpt_filepath` | upstream weights / `--existing-ckpt-filepath` | Starting checkpoint; cleared by `--train-from-scratch`. |
| `trained_autoencoder_path` | upstream weights / `--trained-autoencoder-path` | VAE used to encode/decode latents. |

Model config (`config_maisi_diff_model_rflow-mr-brain.json`) — the only fields the wrapper touches:

| Field | Set from | Default | Notes |
|---|---|---|---|
| `diffusion_unet_train.n_epochs` | `--epochs` | `2` (upstream config ships `1000`) | Convenience override (cell 15 does the same); wrapper default is small for verification. |
| `diffusion_unet_inference.modality` | `--modality` | from `modality_mapping.json` | Kept consistent with the training modality for optional `--run-inference`. |

Everything else in that file (`lr`, `batch_size`, `cache_rate`, the rest of `diffusion_unet_inference`) is left exactly as written — edit the JSON to change it.

The pinned v1 inference block defaults to `dim=[256,256,128]`,
`spacing=[0.94,0.94,1.36]`, and `cfg_guidance_scale=2`. The wrapper preserves
those fields. Older v0 examples may show `256^3`, 1 mm spacing, and guidance
scale 10; use the staged v1 JSON as the execution source of truth.

Runtime flags (not config fields): `--num-gpus N` (`>1` launches `torch.distributed.run`), `--no-amp` (disable mixed precision, passed through to `diff_model_train`).

`--modality` selects the integer code from `configs/modality_mapping.json`.
Supported brain values include `mri` (8), `mri_t1` (9, default), `mri_t2`
(10), `mri_flair` (11), `mri_mra` (16), `mri_swi` (20), and the
skull-stripped values `mri_t1_skull_stripped` (29),
`mri_t2_skull_stripped` (30), `mri_flair_skull_stripped` (31),
`mri_swi_skull_stripped` (32), and `mri_mra_skull_stripped` (33). Per-case
`training[].modality` overrides `--modality`. The modality also feeds the
step-3 embedding sidecars. Upstream reports sparse MRA training coverage, so
MRA output quality is not guaranteed.

For an end-to-end reference including example data download and checkpoint loading, see the upstream tutorial `train_diff_unet_tutorial.ipynb`.

## 2. Usage (one-line training)

Preflight only:

```bash
export NV_GENERATE_ROOT="${NV_GENERATE_ROOT:-$HOME/.cache/nvidia-skills/upstreams/NV-Generate-CTMR-da438fe}" && \
python skills/nv-generate-mr-brain-finetune/scripts/run_mr_brain_finetune.py \
  PATH_TO_DATALIST.json \
  --data-base-dir PATH_TO_DATA_ROOT \
  --output-dir runs/nv_generate_mr_brain_finetune_preflight \
  --preflight
```

Preflight bundle input:

```bash
export NV_GENERATE_ROOT="${NV_GENERATE_ROOT:-$HOME/.cache/nvidia-skills/upstreams/NV-Generate-CTMR-da438fe}" && \
python skills/nv-generate-mr-brain-finetune/scripts/run_mr_brain_finetune.py \
  PATH_TO_INPUT_BUNDLE/preflight_datalist.json \
  --data-base-dir PATH_TO_INPUT_BUNDLE/preflight_dataset \
  --output-dir runs/nv_generate_mr_brain_finetune_preflight \
  --preflight
```

GPU finetuning:

```bash
export NV_GENERATE_ROOT="${NV_GENERATE_ROOT:-$HOME/.cache/nvidia-skills/upstreams/NV-Generate-CTMR-da438fe}" && \
python -m pip install -r "$NV_GENERATE_ROOT/requirements.txt" && \
python skills/nv-generate-mr-brain-finetune/scripts/run_mr_brain_finetune.py \
  PATH_TO_DATALIST.json \
  --data-base-dir PATH_TO_DATA_ROOT \
  --output-dir runs/nv_generate_mr_brain_finetune \
  --epochs 2 \
  --modality mri_t1 \
  --run-inference
```

Replace `PATH_TO_DATALIST.json` and `PATH_TO_DATA_ROOT` with the user's actual paths. Do not use the fixture datalist for real training; it is a preflight-only placeholder.

## 3. Monitor training (TensorBoard)

`scripts.diff_model_train` writes TensorBoard event files under the staged `model_dir` (`OUT_DIR/artifacts/models`). Launch TensorBoard against the output directory and watch the loss curve:

```bash
python -m pip install tensorboard && \
tensorboard --logdir runs/nv_generate_mr_brain_finetune/artifacts
```

The run summary is written to `OUT_DIR/artifacts/workflow_summary.json` (checkpoint path, embedding sidecars, inference outputs); the JSON the wrapper prints to stdout mirrors the same paths plus `exit_code` and a `stderr_tail` for quick triage.

## 4. Hyperparameter tuning and common pitfalls

- **Loss not decreasing / unstable** — lower `diffusion_unet_train.lr` (default `1e-5`) in the model-config JSON, or keep AMP on (default); `--no-amp` is slower but more numerically stable on older GPUs.
- **Out-of-memory** — keep `diffusion_unet_train.batch_size` at `1` and `cache_rate` at `0` in the config JSON, and confirm the autoencoder/UNet fit your GPU before scaling. Multi-GPU (`--num-gpus N`) shards the batch via `torch.distributed.run`.
- **Few cases / quick check** — keep `--epochs` small (the wrapper default `2` is for verification, not convergence; the upstream config ships `1000`).
- **Wrong modality conditioning** — set `--modality` or per-case `training[].modality` to a value present in `configs/modality_mapping.json`; a mismatch produces a clear error rather than silently mislabeling latents.
- **Slow startup on first run** — `diff_model_create_training_data` precomputes latent embeddings once; reuse the same `--output-dir` to avoid recomputing them.

## 5. Evaluate the finetuned model

Use the staged checkpoint (`OUT_DIR/artifacts/models/<model_filename>`) as the diffusion UNet for generation, then inspect the synthesized volumes:

- Pass `--run-inference` here for a quick built-in sanity render, or
- Point the [`nv-generate-mr-brain`](../nv-generate-mr-brain/SKILL.md) inference skill at the finetuned checkpoint to generate fresh brain MRI volumes for qualitative review.

This skill gates file accounting and command provenance only — anatomical realism and downstream utility must be judged by a domain expert on the generated images.

## Limitations
- Requires a current upstream `NV-Generate-CTMR` checkout with the existing diffusion training scripts. The skill itself stages the required config and datalist glue locally and does not depend on the notebook or PR #33.
- Full training can be expensive and is not deterministic across hardware, CUDA, and package versions.
- The wrapper gates file accounting and command provenance, not anatomical realism or downstream model utility.
- Not for clinical deployment, clinical interpretation, autonomous diagnosis, regulatory submission, or production training-data approval.

## Troubleshooting
| Error | Cause | Fix |
|---|---|---|
| `diffusion training scripts were not found` | `NV_GENERATE_ROOT` does not point at a current NV-Generate-CTMR checkout. | Clone or update `https://github.com/NVIDIA-Medtech/NV-Generate-CTMR` and set `NV_GENERATE_ROOT`. |
| `missing datalist image` | `training[].image` paths are not relative to `--data-base-dir` or files are absent. | Fix the datalist or pass the correct data root. |
| CUDA or MONAI import failure | Runtime environment lacks upstream dependencies. | Install `"$NV_GENERATE_ROOT/requirements.txt"` in the selected environment. |
