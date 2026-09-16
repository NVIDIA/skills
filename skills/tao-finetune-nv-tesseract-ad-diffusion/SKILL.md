---
name: tao-finetune-nv-tesseract-ad-diffusion
description: >-
  NV-Tesseract AD Diffusion — diffusion-based anomaly detection and fine-tuning
  for multivariate time series. Use when the user asks to "fine-tune NV-Tesseract",
  "run AD diffusion inference", "detect anomalies with diffusion", "time series
  anomaly detection", "finetune ad-diffusion", "use perform_anomaly_analysis_with_diffusion",
  "automl ad-diffusion", "hyperparameter search ad-diffusion", "hyperparameter optimization"
  or mentions "curriculum_medium.yaml", "final_model.pth",
  "nv-tesseract-ad-diffusion", "ad_diffusion", or "TSDiffuser_Generic".
license: Apache-2.0
compatibility: Requires Python 3.12+ and uv. CUDA GPU recommended; CPU-only supported.
metadata:
  author: NVIDIA Corporation
  version: "0.2.0"
allowed-tools: Read Bash
tags:
  - anomaly-detection
  - time-series
  - diffusion
  - finetune
  - inference
  - automl
  - nv-tesseract
---

# NV-Tesseract AD Diffusion

Diffusion-based anomaly detection and fine-tuning for multivariate time series. The model
reconstructs randomly masked segments and scores each timestep by MAE between reconstruction
and original signal; adaptive thresholding (SCS or MACS) converts scores to binary labels.

**Source code:** https://github.com/NVIDIA/NV-Tesseract
**Pretrained weights:** https://huggingface.co/nvidia/nv-tesseract-ad-diffusion

> **For the most up-to-date usage information**, refer to the README files in
> the NV-Tesseract repository:
>
> - **[`ad_diffusion/README.md`](https://github.com/NVIDIA/NV-Tesseract/blob/main/ad_diffusion/README.md)** — full SDK reference, model architecture, and API docs
> - **[`ad_diffusion/examples/datasets/README.md`](https://github.com/NVIDIA/NV-Tesseract/blob/main/ad_diffusion/examples/datasets/README.md)** — dataset format, synthetic data generation, and CSV conventions

## External dependencies

| Dependency | Purpose | Install |
|---|---|---|
| Python 3.12+ | Runtime | https://www.python.org/downloads/ |
| uv | Package + environment manager | `pip install uv` |
| CUDA toolkit (optional) | GPU acceleration | https://developer.nvidia.com/cuda-downloads |
| huggingface_hub | Weight download from HF | Bundled via `uv sync` |

## Credentials

`nvidia/nv-tesseract-ad-diffusion` is a public repo — no token required for downloading weights.
If you ever hit a `401`/`403` (gated access or private fork) or a `504` on first download, see the Known pitfalls section.

## Quick start

```bash
git clone --branch main --single-branch https://github.com/NVIDIA/NV-Tesseract
cd NV-Tesseract/ad_diffusion
uv sync                              # install dependencies (one-time)

# Inference — synthetic data, auto-downloads weights from HF on first run
uv run python examples/quick_example.py

# Inference — your own CSV
uv run python examples/quick_example.py \
  --model-path final_model.pth \
  --config-path curriculum_medium.yaml \
  --dataset-path /path/to/data.csv

# Pre-download weights only (warm the cache before going offline)
uv run python examples/quick_example.py --download-weights

# Fine-tune on your own normal-behavior data
uv run python examples/finetune_example.py \
  --csv /path/to/normal_training_data.csv \
  --timestamp-col timestamp \
  --label-col is_anomaly \
  --epochs 20 \
  --output-dir artifacts/finetune_my_data
```

## Inference

Use `perform_anomaly_analysis_with_diffusion` in `sdk/anomaly_analysis.py`. It validates
input, auto-dispatches across all visible GPUs, applies adaptive thresholding (SCS or MACS),
and returns the original DataFrame with `Anomaly` (0/1) and `MAE` columns appended.

```python
import sys, pandas as pd
sys.path.append("/path/to/NV-Tesseract/ad_diffusion")  # clone NV-Tesseract with --branch main
from sdk.anomaly_analysis import perform_anomaly_analysis_with_diffusion

df = pd.read_csv("your_data.csv")

# The API raises ValueError on non-numeric columns — drop timestamp, IDs, and labels first.
df = df.select_dtypes(include="number")

results = perform_anomaly_analysis_with_diffusion(
    df=df,
    threshold_strategy="scs",       # "scs" (fast) or "macs" (adaptive)
    model_path=None,                 # None → auto-download final_model.pth from HF
    config_path=None,                # None → auto-download curriculum_medium.yaml from HF
    nsample=15,                      # diffusion samples per window; ↑ accuracy, ↑ latency
    preprocess_model_dir=None,       # optional preprocessing model directory
)
# results columns: Anomaly (0/1), MAE (float), plus all original columns
print(results[["Anomaly", "MAE"]].describe())
```

### Inference CLI reference

| Argument | Default | Description |
|---|---|---|
| `--dataset-path` | synthetic | CSV with numeric feature columns |
| `--model-path` | auto-download | Path to `.pth` checkpoint |
| `--config-path` | auto-download | Path to `curriculum_medium.yaml` |
| `--download-weights` | — | Fetch weights from HF and exit |
| `--skip-download` | false | Require local weights; skip HF fetch |

## Fine-tuning

Fine-tune on your own data. The training CSV should contain mostly normal behavior. Validate the pretrained model on your domain before fine-tuning.

```bash
uv run python examples/finetune_example.py \
  --csv /path/to/normal_data.csv \
  --val-csv /path/to/val_data.csv \   # optional; otherwise --val-ratio splits --csv
  --pretrained-model final_model.pth \
  --epochs 20 --batch-size 16 --lr 1e-5 \
  --output-dir artifacts/finetune_my_data
```

### Fine-tuning arguments

| Argument | Default | Description |
|---|---|---|
| `--run-config` | — | JSON/YAML config file generated by AutoMLRunner (`{config_path}`). All fields below can be set here; explicit CLI flags override the file. |
| `--csv` | **required** | Training CSV (ideally containing normal behavior). Required if not supplied via `--run-config`. |
| `--val-csv` | — | Separate validation CSV |
| `--val-ratio` | `0.3` | Validation fraction when `--val-csv` not used (temporal split) |
| `--timestamp-col` | `timestamp` | Column to drop from features |
| `--label-col` | — | Label column to drop |
| `--drop-cols` | — | Comma-separated extra columns to drop |
| `--pretrained-model` | `final_model.pth` | Warm-start checkpoint (auto-downloaded if missing) |
| `--config` | `curriculum_medium.yaml` | Model config YAML |
| `--repo-id` | `nvidia/nv-tesseract-ad-diffusion` | HuggingFace repo for auto-download |
| `--no-download` | false | Fail if pretrained weights are not local |
| `--epochs` | `10` | Training epochs |
| `--batch-size` | `16` | Per-GPU batch size |
| `--lr` | `1e-5` | AdamW learning rate |
| `--weight-decay` | `1e-6` | AdamW weight decay |
| `--grad-clip` | `1.0` | Gradient norm clip |
| `--num-workers` | `0` | DataLoader workers |
| `--seed` | `42` | Random seed |
| `--output-dir` | `artifacts/finetune` | Output directory |
| `--window-length` | config (100) | Sliding window length in timesteps |
| `--window-stride` | `1` | Step between consecutive windows |
| `--split` | config (10) | Alternating mask segments per window |
| `--mask-ratio` | `0.7` | Fraction of each window masked during training |
| `--scale-factor` | config (1) | Scale multiplier after min-max normalization |
| `--num-gpus` | all available | Number of GPUs for DDP fine-tuning; set `1` to force single-GPU |

### Running inference with a fine-tuned checkpoint

```python
results = perform_anomaly_analysis_with_diffusion(
    df=df,
    threshold_strategy="scs",
    model_path="artifacts/finetune_my_data/best_finetuned_model.pth",
    config_path="artifacts/finetune_my_data/finetune_config.yaml",
    nsample=15,
)
```

## AutoML (HPO: hyperparameter optimization)

This skill supports AutoML for fine-tuning HPO and labeled inference HPO through `tao-skill-bank:tao-run-automl` with this model's `skill_dir`.

Read `references/automl.md` when the user asks for AutoML/HPO setup, tunable parameters, VirtualEnvSDK setup, config flow, window-length constraints, inference trial scripts, or AutoML result handoff details.

## Data requirements

| Property | Requirement |
|---|---|
| Rows | ≥ `window_length` (default **100**); ≥ `target_dim` (default **18**) for PCA |
| Columns | Must be numeric — the API raises `ValueError` on non-numeric columns; drop `timestamp`, IDs, and labels before calling |
| Values | No NaN / ±Inf — fill before passing to the API |
| Feature count > `target_dim` | PCA reduction to `target_dim`; needs ≥ `target_dim` rows |
| Feature count < `target_dim` | Zero-padded to `target_dim` |

```
timestamp,sensor_1,sensor_2,sensor_3
2024-01-01 00:00:00,0.42,1.10,-0.33
...
```

Pass only numeric feature columns to the inference API — it raises `ValueError` on
non-numeric columns rather than dropping them. Use `df.select_dtypes(include="number")`
or drop by name before calling. Fine-tuning handles this via `--timestamp-col`,
`--label-col`, and `--drop-cols` CLI args.

## Output structure

**Inference** (`examples/quick_example.py`):

```
examples/datasets/
└── anomaly_results.csv      # original columns + Anomaly (0/1) + MAE
```

**Fine-tuning** (`--output-dir artifacts/finetune_my_data`):

```
artifacts/finetune_my_data/
├── best_finetuned_model.pth     # checkpoint with lowest validation loss
├── final_finetuned_model.pth    # checkpoint from last epoch
├── metrics.json                 # scalar for AutoML: {"val_loss": <best>}
├── epoch_metrics.json           # per-epoch log: [{"epoch": N, "train_loss": …, "val_loss": …}]
└── finetune_config.yaml         # config used during training (for reproducibility)
```

## Model configuration (`curriculum_medium.yaml`)

| Field | Default | Description |
|---|---|---|
| `model.target_dim` | `18` | Internal feature dim; data is PCA'd/padded to this |
| `dataset.window_length` | `100` | Sliding window size in timesteps |
| `dataset.split` | `10` | Alternating mask segments per window |
| `dataset.scale_factor` | `1` | Scale multiplier after min-max normalization |
| `diffusion.num_steps` | `500` | Full diffusion steps (overridden by DPM-Solver) |
| `diffusion.channels` | `128` | Model hidden dimension |
| `diffusion.layers` | `6` | Transformer encoder layers |

## Hardware

| Tier | Setup | Notes |
|---|---|---|
| Minimum | 1× CPU | Functional; DPM-Solver reduces steps 500 → 20 |
| Recommended | 1× NVIDIA GPU (≥8 GB VRAM) | Strongly recommended for fine-tuning |
| Multi-GPU inference | 2–8× NVIDIA GPUs | auto-dispatched by `perform_anomaly_analysis_with_diffusion` |
| Multi-GPU fine-tuning | 2+× NVIDIA GPUs | Auto DDP via `--num-gpus` (defaults to all visible GPUs) |

## Known pitfalls

| Symptom | Cause | Fix |
|---|---|---|
| `HfHubHTTPError: 401` | Repo gated or token missing | `export HUGGINGFACE_HUB_TOKEN="hf_..."` or `huggingface-cli login` |
| `504` / timeout on first weight download | HF CDN throttles unauthenticated requests — public repos are still subject to this on first download | Set `export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"` before running; authenticated requests use a more reliable CDN path |
| `ValueError: No numeric columns` | All columns are strings/dates | Drop non-numeric columns before calling API |
| `ValueError: PCA needs at least target_dim rows` | Fewer rows than `target_dim` (18) | Provide a longer time series |
| `ValueError: Need at least N rows` (finetune) | Split shorter than `window_length` | Ensure each train/val split has ≥ 100 rows |
| `RuntimeError: CUDA out of memory` | Batch too large | Reduce `--batch-size` or `nsample` |
| All MAE scores identical | Constant-value columns | Drop zero-variance columns before calling API |
| `ModuleNotFoundError: sdk` | Wrong working directory | `cd ad_diffusion/` before `uv run`, or add it to `sys.path` |
| Slow inference on CPU | Many diffusion windows | Reduce `nsample` to 5–10 for smoke tests |
