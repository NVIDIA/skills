---
name: nv-segment-ct-finetune
description: Runs standard or fixed-channel softmax finetuning of NV-Segment-CT VISTA3D on CT NIfTI image/label datasets, with optional MONAI-native MLflow tracking and checkpoint evidence. Uses softmax for predefined, mutually exclusive classes; keeps the standard workflow when point prompts or runtime-variable classes are needed. Not for clinical validation.
license: Apache-2.0
allowed-tools: Bash, Read, Write, WebFetch, Env
metadata:
  author: "NVIDIA MedTech <noreply@nvidia.com>"
  tags:
    - MedTech
    - CT
    - finetuning
    - segmentation
---

# NV-Segment-CT Finetune

## Purpose

- Used for smoke or dataset finetuning of NV-Segment-CT VISTA3D on CT NIfTI labels, including the upstream fixed-channel softmax workflow and optional MLflow tracking. Not for clinical validation.
- Wraps the upstream MONAI bundle entrypoint; do not replace it with handwritten training or inference code.
- Manifest inputs are `dataset_dir`, `datalist`, `target_anatomy`, `label_mapping`, `smoke`, `sanity`, `auto_seg`, `softmax`, `skip_formal_eval`, `mlflow_tracking_uri`, `mlflow_experiment_name`, and `mlflow_run_name`.
- Manifest outputs are `finetuned_ckpt` and schema-checked `result_json`.

## Instructions

- Run `scripts/run_finetune.py`; do not patch files under `bundle/` or upstream checkouts during normal skill use.
- For standalone Bash, include the fresh-environment setup line before the wrapper; benchmark venvs start empty.
- Run the committed script in place from the repo root. Do not copy this skill to a runtime directory, and do not use `rm` or cleanup commands in generated invocations.
- If a host exposes `run_script`, use `run_script("scripts/run_finetune.py", args=[...])`; otherwise run from the repo root.
- For the shortest workflow check, use `--smoke`; for MSD Task06 Lung Tumor reproduction, use `--sanity`.
- Choose between the standard and `--softmax` workflows using the criteria below. Do not combine `--softmax` with `--auto-seg` or `--sanity`.
- Set `--mlflow-experiment-name` to enable MLflow for the training phase of either workflow. `--mlflow-tracking-uri` and `--mlflow-run-name` require an experiment name. Formal pre/post evaluation does not receive MLflow credentials.
- Read `references/task06-and-results.md` only when you need Task06 reference details, output-field definitions, or manual bundle setup notes.

## Choosing the Workflow

Use `--softmax` only when all of these conditions hold:

- The complete class set is known before training and will not vary between inference requests.
- Labels are mutually exclusive: each voxel is background or exactly one foreground class.
- Every foreground dataset label maps to an existing VISTA3D class ID, and a conventional fixed-channel output is desired.

Keep the standard workflow if point prompts must remain available, classes are selected dynamically at inference, labels can overlap, or the Task06 `--sanity` reproduction is required.

For `--label-mapping '[[1,3],[2,13]]'`, channel 0 is background, channel 1 represents dataset label 1 initialized from VISTA3D class 3, and channel 2 represents dataset label 2 initialized from VISTA3D class 13. Preserve the entries and their order when using the resulting `model_softmax.pt` with upstream `configs/inference_softmax.json`. The `nv-segment-ct` and `nv-segment-ctmr` inference skills do not currently expose that fixed-channel inference path.

## Available Scripts

| Script | Purpose | Arguments |
|---|---|---|
| `scripts/run_finetune.py` | Primary entrypoint declared by `skill_manifest.yaml`; stages configs, runs MONAI, and writes `output.json`. | `[FIXTURE_OR_DATASET] --output-dir OUT_DIR [--smoke] [--sanity] [--auto-seg] [--softmax] [--dataset-dir DIR] [--datalist JSON] [--target-anatomy TEXT] [--label-mapping JSON] [--patch-size JSON] [--mlflow-experiment-name NAME] [--mlflow-tracking-uri URI] [--mlflow-run-name NAME]` |

## Prerequisites

- Python 3.10+ with CUDA-capable Torch for GPU runs.
- Runtime packages from `skill_manifest.yaml`, especially `monai==1.4.0`, `numpy<2`, `nibabel`, `scipy`, `typer`, `PyYAML`, `fire`, `pytorch-ignite`, `einops`, and `huggingface_hub`. Install `mlflow>=2.10,<4` when MLflow tracking is enabled.
- Optional environment variables: `CUDA_VISIBLE_DEVICES` restricts visible GPUs; `NPROC_PER_NODE` overrides GPU count and values `>=2` select multi-GPU mode for non-sanity runs; `NVSEG_FINETUNE_AUTO_VENV=0` disables the cached MONAI 1.4 compatibility environment. Remote tracking may use `DATABRICKS_CONFIG_PROFILE`, `DATABRICKS_HOST`, `DATABRICKS_TOKEN`, `MLFLOW_TRACKING_CLIENT_CERT_PATH`, `MLFLOW_TRACKING_INSECURE_TLS`, `MLFLOW_TRACKING_PASSWORD`, `MLFLOW_TRACKING_SERVER_CERT_PATH`, `MLFLOW_TRACKING_TOKEN`, or `MLFLOW_TRACKING_USERNAME`; these variables are forwarded only when MLflow is explicitly enabled, and unrelated credentials are not forwarded.
- `--softmax` also needs the pinned NVIDIA-Medtech source checkout. Set `NV_SEGMENT_CT_ROOT` to its `NV-Segment-CT` directory, or set `NV_SEGMENT_CTMR_ROOT` to the sibling `NV-Segment-CTMR` directory. The wrapper reads the official softmax config and implementation in place and writes generated overrides only under `--output-dir`.
- Side effects: writes generated bundle configs under `skills/nv-segment-ct-finetune/bundle/configs/`, including `skills/nv-segment-ct-finetune/bundle/configs/auto_override.json`, `skills/nv-segment-ct-finetune/bundle/configs/train_continual_task06_lung.json`, and `skills/nv-segment-ct-finetune/bundle/configs/dfw_no_logging.json`; writes checkpoints/evidence under `--output-dir` and local tracking data under `<output-dir>/mlruns` when enabled; may create the MONAI compatibility environment under `~/.cache/nvidia-skills/venvs/nv-segment-ct-finetune-monai14/`; may cache model assets under `~/.cache/huggingface/`; and may contact `https://huggingface.co`, `https://raw.githubusercontent.com`, or `https://<caller-provided-mlflow-or-databricks-workspace>` when remote tracking is explicitly enabled.

Fresh environment setup:

```bash
python -m pip install "monai==1.4.0" "numpy<2" pytorch-ignite einops nibabel scipy typer PyYAML fire huggingface_hub
```

When MLflow tracking is enabled, also install:

```bash
python -m pip install "mlflow>=2.10,<4"
```

Known upstream compatibility constraints:

- DFW Task06 reference: Python `3.10.16`, MONAI `1.4.0`, Torch `2.7.0+cu126`.
- Use exact `monai==1.4.0` for smoke, sanity, and evidence runs; MONAI 1.5.x can crash the upstream finetune loss on boolean labels.
- Do not float the dependency as `monai>=1.4,<1.6` in generated commands.
- The softmax workflow keeps the upstream defaults of 100 epochs and learning rate `1e-4` unless the caller overrides them.

One-time source setup for `--softmax`:

```bash
export NV_SEGMENT_CTMR_COMMIT=cb921f5c58837c0f42a713855d68b32af88e1cdd
export NV_SEGMENT_CTMR_CHECKOUT="$HOME/.cache/nvidia-skills/upstreams/NV-Segment-CTMR-cb921f5"
if [ ! -d "$NV_SEGMENT_CTMR_CHECKOUT/.git" ]; then
  git clone https://github.com/NVIDIA-Medtech/NV-Segment-CTMR.git "$NV_SEGMENT_CTMR_CHECKOUT"
fi
git -C "$NV_SEGMENT_CTMR_CHECKOUT" checkout --detach "$NV_SEGMENT_CTMR_COMMIT"
export NV_SEGMENT_CT_ROOT="$NV_SEGMENT_CTMR_CHECKOUT/NV-Segment-CT"
```

## Usage

Smoke-scale workflow check:

```bash
python -m pip install "monai==1.4.0" "numpy<2" pytorch-ignite einops nibabel scipy typer PyYAML fire huggingface_hub && \
python skills/nv-segment-ct-finetune/scripts/run_finetune.py \
  PATH_TO_DATASET \
  --smoke \
  --patch-size '[64,64,64]' \
  --output-dir runs/nvseg_smoke
```

Use the staged dataset as `PATH_TO_DATASET`. For the micro fixture, use `skills/nv-segment-ct-finetune/fixtures/spleen_micro`. Smoke mode proves wiring, config generation, checkpoint loading, and runtime compatibility; it is not a quality bar.

MSD Task06 Lung Tumor sanity reproduction:

```bash
python skills/nv-segment-ct-finetune/scripts/run_finetune.py \
  /path/to/Task06 \
  --sanity \
  --output-dir runs/nvseg_task06_sanity
```

The sanity preset follows the single-GPU DFW recipe: fold-0 validation, label mapping `[[1, 23]]` for `lung tumor`, automatic class-prompt segmentation, patch `[128,128,128]`, 5 epochs, and original-spacing `configs/evaluate.json` scoring before and after training. Expected reference range is pretrained Dice about `0.6697`, training-best Dice about `0.6905`, and fine-tuned formal Dice about `0.6836`.

User-data finetune:

```bash
python skills/nv-segment-ct-finetune/scripts/run_finetune.py \
  --dataset-dir /path/to/dataset \
  --datalist /path/to/datalist.json \
  --target-anatomy "lung tumor" \
  --auto-seg \
  --epochs 5 \
  --patch-size '[128,128,128]' \
  --output-dir runs/nvseg_user_finetune
```

Use `--label-mapping '[[1, 23]]'` when local label values are custom or the anatomy name is ambiguous.

Optional local MLflow tracking:

```bash
python skills/nv-segment-ct-finetune/scripts/run_finetune.py \
  --dataset-dir /path/to/dataset \
  --datalist /path/to/datalist.json \
  --target-anatomy "lung tumor" \
  --epochs 5 \
  --mlflow-experiment-name nvseg-finetune \
  --mlflow-run-name trial-01 \
  --output-dir runs/nvseg_mlflow
```

This uses MONAI's documented `--tracking mlflow` path and built-in rank-zero handlers. With no `--mlflow-tracking-uri`, data stays in `<output-dir>/mlruns`. Pass a caller-approved remote URI, including `databricks`, only when remote tracking is intended. MLflow does not change patch size, transforms, optimizer values, DataLoader settings, or other training configuration.

Fixed-channel softmax finetune for mutually exclusive labels:

```bash
export NV_SEGMENT_CT_ROOT="$HOME/.cache/nvidia-skills/upstreams/NV-Segment-CTMR-cb921f5/NV-Segment-CT"
python skills/nv-segment-ct-finetune/scripts/run_finetune.py \
  --dataset-dir /path/to/dataset \
  --datalist /path/to/datalist.json \
  --label-mapping '[[1,3],[2,13]]' \
  --softmax \
  --epochs 100 \
  --output-dir runs/nvseg_softmax
```

This delegates to upstream `configs/train_continual_softmax.json`. It produces
`checkpoints/model_softmax.pt`; the source `model.pt` initializes the network
but is not compatible with `configs/inference_softmax.json`. The wrapper
therefore recommends the produced softmax checkpoint after a successful run.

## Examples

Smoke run on a staged tiny dataset:

```bash
python skills/nv-segment-ct-finetune/scripts/run_finetune.py \
  runs/with_vs_without_nv/_inputs/nv_segment_ct_finetune/input_dataset \
  --smoke \
  --patch-size '[64,64,64]' \
  --output-dir runs/nvseg_smoke
```

Task06 sanity run on a local MSD cache:

```bash
python skills/nv-segment-ct-finetune/scripts/run_finetune.py \
  .workbench_data/datasets/Task06_Lung \
  --sanity \
  --output-dir runs/nvseg_task06_sanity
```

## Data Contract

- Preferred layout: `dataset/imagesTr/*.nii.gz` and `dataset/labelsTr/*.nii.gz`.
- Labels must align one-to-one with images by basename.
- The target label value must be present in the training labels.
- Use a datalist when patient-level splitting matters. The bundle default `fold` is `0`, so `fold: 0` entries are validation and all other folds are training.
- Every trained foreground label must map to an existing VISTA3D global class id from `bundle/label_dict.json`; this skill cannot invent a new class.
- In `--softmax` mode, the first mapping column is the saved dataset label and the second is the pretrained VISTA class ID. Mapping order fixes the channel layout and must remain unchanged during inference.

## Results

Check `output.json` in the run directory first:

- `formal_pretrained_val_dice` and `formal_finetuned_val_dice`: original-spacing pre/post scores when formal eval is enabled.
- `training_start_val_dice`, `val_dice_per_epoch`, and `training_best_val_dice`: training-time validation trace.
- `finetuned_ckpt_matches_pretrained_weights`: detects the standard workflow's epoch-0 checkpoint trap when `val_at_start=true`; softmax uses a different checkpoint architecture.
- `recommended_ckpt`: checkpoint to keep. Do not blindly use the last epoch, `model_finetune.pt`, or `model_softmax.pt` without checking the recorded workflow and metrics.
- `invocation.mlflow_tracking`: selected tracking URI, experiment name, and optional run name, or `null` when tracking was disabled.
- `runtime.oom`, `runtime.peak_gpu_mb`, and phase logs: distinguish OOM, slow validation, and process failure.

Decision rule: prefer formal original-spacing pre/post scores when present; reject tensor-identical "fine-tuned" checkpoints for sanity recovery; treat `improved: false` as valid evidence rather than a wrapper failure.

## Limitations

- Thin wrapper. Training, validation, transforms, and checkpointing are delegated to the upstream bundle in `bundle/`.
- Reproduction record only: the successful five-epoch Task06 run used Python
  `3.12.3`, PyTorch `2.12.0+cu130` with CUDA `13.0`, MONAI `1.4.0`, NumPy
  `1.26.4`, PyTorch-Ignite `0.5.4`, NiBabel `5.4.2`, SciPy `1.16.0`, einops
  `0.8.2`, Fire `0.7.1`, Hugging Face Hub `0.36.2`, Transformers `4.57.6`,
  Typer `0.25.1`, PyYAML `6.0.3`, and MLflow `3.14.0` on one NVIDIA RTX 6000
  Ada 48 GB GPU. These versions document the evidence environment; they are
  not additional package constraints or a claim that other versions cannot
  work.
- The auto-derived plan is heuristic; caller-provided `--patch-size`, `--cache-rate`, `--epochs`, and `--learning-rate` win.
- `--softmax` is not compatible with `--sanity`: the Task06 reference scores and original-spacing pre/post evaluation belong to the standard VISTA3D continual-learning workflow. Softmax runs record the training validation trajectory but need a separate task-specific evaluation before quality claims.
- The Task06 sanity recipe intentionally forces single-GPU execution to match the DFW reference. Multi-GPU mode for other datasets requires host `torchrun` support.
- The paired verifier is CPU-only and audits the evidence pack; it does not re-run GPU segmentation.
- MLflow support is optional and uses MONAI's built-in tracking handlers. Tracking errors are part of the upstream MONAI run and can therefore fail the finetune command.
- Not for clinical deployment, clinical interpretation, autonomous diagnosis, or regulatory submission.

## Troubleshooting

| Error | Cause | Fix |
|---|---|---|
| Missing dependency or import error | Runtime drift from `skill_manifest.yaml`. | Install the packages above or use the documented environment. |
| Low Task06 pretrained Dice | Wrong config, wrong checkpoint, data split drift, or dependency drift. | Compare environment fields and staged configs before changing training logic. |
| `model_finetune.pt` matches pretrained | `val_at_start=true` selected epoch 0 as best. | Use `recommended_ckpt`; treat sanity recovery as failed unless a changed checkpoint improves formal Dice. |
| Missing formal Dice fields | Formal eval failed or was skipped. | Inspect `eval_pretrained.log`, `eval_finetuned.log`, and `metrics.csv`. |
| GPU out of memory | Patch/cache settings too large. | Reduce `--patch-size`, lower `--cache-rate`, or reduce workers. |
| No validation cases | Datalist lacks `fold: 0`. | Provide at least one validation entry. |
| `--softmax requires the pinned ... checkout` | The August softmax config/implementation is absent or the checkout is at a different commit. | Check out `cb921f5c58837c0f42a713855d68b32af88e1cdd` and set `NV_SEGMENT_CT_ROOT` or `NV_SEGMENT_CTMR_ROOT`. |
| MLflow tracking fails | MLflow is absent, credentials are invalid, or the experiment is inaccessible. | Inspect `finetune.log`, fix the MLflow client configuration, and rerun; omit `--mlflow-experiment-name` to disable tracking. |

## Verification

Run the implemented verifier when quality gates matter:

```bash
python -m eval_engine.run_trusted skills/nv-segment-ct-finetune \
  --fixture skills/nv-segment-ct-finetune/fixtures/spleen_micro \
  --out runs/nvseg_trusted
```
