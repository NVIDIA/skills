# Forecasting AutoML Details

Read this reference when the user asks for NV-Tesseract Forecasting AutoML/HPO setup, tunable parameters, runner examples, inference trial scripts, DARR HPO, or AutoML result handoff details.

## Contents

- [AutoML overview](#automl-hpo-hyperparameter-optimization)
- [Fine-tuning AutoML](#fine-tuning-automl)
- [Inference HPO](#inference-hpo)
- [Basic mode](#basic-mode-tune-model-parameters)
- [DARR mode](#darr-mode-tune-blending-parameters)

## AutoML (HPO: hyperparameter optimization)

This skill is AutoML-enabled for both fine-tuning and DARR inference. When an HPO request
arrives, route it through `tao-skill-bank:tao-run-automl` with this model's `skill_dir`.

> **Note:** `VirtualEnvSDK` and `python_script` execution require the AutoML
> control-plane dependencies managed by `tao-run-automl`. Construct the SDK as
> `VirtualEnvSDK(venv_path=..., work_dir=...)` -- see that skill's **Runner
> Construction** section and
> `skills/applications/tao-run-automl/references/automl-runner-configuration.md`
> section "VirtualEnvSDK (containerless venv runs)". **Always pass `work_dir=`**;
> the default is a hidden state directory under `$HOME`, and each fine-tuning
> trial writes roughly 1.4 GB of checkpoint there. This model reference lists
> NV-Tesseract Forecasting search spaces, overrides, and trial payloads.

### Fine-tuning AutoML

AutoML minimizes `val_mse` (and also reports `val_mae`) by tuning 5 training hyperparameters.
Epochs, architecture flags, and data paths are fixed as spec overrides — not searched.

| Parameter | Spec key | Default | Search range |
|---|---|---|---|
| Learning rate | `train.lr` | `1e-4` | `[1e-6, 1e-3]` log |
| Batch size | `train.batch_size` | `8` | `[4, 64]` int |
| Head dropout | `train.head_dropout` | `0.1` | `[0.0, 0.5]` |
| Weight decay | `train.weight_decay` | `0.0` | `[0.0, 0.01]` |
| Gradient clip | `train.max_norm` | `5.0` | `[1.0, 10.0]` |

**Fixed (not searched):** `train.epochs`, `model.seq_len`, `model.forecast_horizon`,
`model.use_cross_channel`, `model.ckpt_init`, `dataset.*`, `train.seed`, `train.output_dir`.

**`use_cross_channel` is a user choice before HPO** — it switches both the pretrained checkpoint
(`run8_best_model_cr.pt` vs `moment_head_512_6hr.pt`) and the model architecture. Bayesian
search across these two would compare incomparable weight initializations.

> **Fair comparison rule:** `perform_forecasting` defaults to `use_cross_channel=True`
> (loads `run8_best_model_cr.pt`). If you compare pretrained inference against a finetuned
> checkpoint, both must use the same base. Either:
> - Finetune with `--use-cross-channel` (warm-starts from `run8_best_model_cr.pt`) and
>   run inference with `use_cross_channel=True` (default), or
> - Run inference with `use_cross_channel=False` and finetune without `--use-cross-channel`
>   (warm-starts from `moment_head_512_6hr.pt`, the finetune default).
>
> Mixing architectures — cross-channel pretrained vs standard finetuned — conflates model
> quality with architectural differences and makes the comparison uninterpretable.

```python
from pathlib import Path

forecasting_dir = Path("/path/to/NV-Tesseract/forecasting")
skill_dir = Path("/path/to/tao-skill-bank/skills/models/tao-finetune-nv-tesseract-forecasting")

# Build sdk + runner per tao-run-automl section "Runner Construction":
# sdk = VirtualEnvSDK(venv_path=<venv>, work_dir=<scratch>)  # work_dir is not optional in practice

result = runner.run(
    automl_settings={
        "algorithm":                  "bayesian",
        "metric":                     "val_mse",
        "direction":                  "minimize",
        "automl_max_recommendations": 5,
    },
    # spec_overrides MUST use dotted keys.
    spec_overrides={
        "dataset.csv":            "/path/to/timeseries.csv",
        "dataset.timestamp_col":  "timestamp",
        "dataset.val_ratio":      0.1,
        "model.forecast_horizon": 72,
        "model.seq_len":          512,
        "model.use_cross_channel": False,
        "train.epochs":           10,
        "train.num_epochs":       10,
        "train.output_dir":       "automl_workspace/finetune/trials",
    },
    workspace_path="automl_workspace/finetune",
    execution={
        "type":          "python_script",
        "script":        str(forecasting_dir / "examples/finetune_example.py"),
        "args":          ["--run-config", "{config_path}"],
        "cwd":           str(forecasting_dir),
        "config_format": "yaml",
    },
)

best = result["best"]
print(f"Best val_mse: {best['metric_value']}")
print(f"Best specs:   {best['specs']}")
# best['specs'] uses the same dotted keys as spec_overrides.
# The best checkpoint lives in the VirtualEnvSDK managed job directory, not in train.output_dir.
# If locating it via glob, scope to result['history'] rec_ids and sort by metrics.json val_mse
# to avoid accidentally picking up a checkpoint from a prior experiment with lower val_mse.
```

### Inference HPO

AutoML can search inference parameters against a ground-truth eval window. Hold back the last
`forecast_horizon` rows as ground truth; the trial script runs inference and writes MSE and MAE
to `metrics.json`. Two modes are supported — use whichever matches your setup.

The trial scripts below are generated by the agent — they are not shipped with NV-Tesseract.
Save the template anywhere and point `execution.script` at it.

#### Basic mode (tune model parameters)

Use when you have no context history for DARR. Tunes parameters that affect the model's
direct forecast quality.

| Parameter | Spec key | Default | Search range |
|---|---|---|---|
| Context length | `hpo.seq_len` | `512` | `[128, 256, 512]` ordered_int |
| Cross-channel | `hpo.use_cross_channel` | `false` | `[true, false]` categorical |
| Attention heads | `hpo.cross_channel_heads` | `8` | `[4, 8, 16]` ordered_int |
| Head dropout | `hpo.cross_channel_dropout` | `0.1` | `[0.0, 0.1, 0.2]` |

**Trial script template (basic):**

```python
"""Single basic inference HPO trial — called by the AutoML runner with --run-config {config_path}."""
import argparse, json, sys
from pathlib import Path
import yaml, pandas as pd
import numpy as np

FORECASTING_DIR = Path("/path/to/NV-Tesseract/forecasting")  # adjust to actual clone path
sys.path.insert(0, str(FORECASTING_DIR))

from sdk.forecasting import perform_forecasting

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-config", required=True)
    args = p.parse_args()

    cfg = yaml.safe_load(Path(args.run_config).read_text())
    ds  = cfg["dataset"]
    inf = cfg["hpo"]
    out = Path(cfg.get("train", {}).get("output_dir", "artifacts/inference_hpo"))
    out.mkdir(parents=True, exist_ok=True)

    ts_col = ds.get("timestamp_col", "timestamp")
    df = pd.read_csv(ds["csv"], parse_dates=[ts_col])
    target_cols = [c.strip() for c in ds["target_cols"].split(",")] if ds.get("target_cols") else None

    horizon = inf["forecast_horizon"]
    if horizon <= 0 or horizon >= len(df):
        raise ValueError(f"hpo.forecast_horizon must be in [1, {len(df) - 1}], got {horizon}")
    df_input  = df.iloc[:-horizon]
    df_actual = df.iloc[-horizon:]

    result = perform_forecasting(
        df=df_input,
        timestamp_column=ts_col,
        forecast_horizon=horizon,
        seq_len=inf.get("seq_len", 512),
        ckpt=inf.get("ckpt") or None,
        standardizer_pkl=inf.get("standardizer_pkl") or None,
        use_cross_channel=inf.get("use_cross_channel", False),
        cross_channel_heads=inf.get("cross_channel_heads", 8),
        cross_channel_dropout=inf.get("cross_channel_dropout", 0.1),
        return_all_channels=True,
    )

    # Align predictions to actuals by column name, not position.
    if target_cols:
        wanted_pred_cols = [f"{c}_forecast" for c in target_cols]
        missing = [c for c in wanted_pred_cols if c not in result.columns]
        if missing:
            raise ValueError(f"target_cols columns not found in predictions: {missing}")
        preds       = result[wanted_pred_cols].values[:horizon]
        actual_vals = df_actual[target_cols].values[:horizon]
    else:
        pred_cols   = [c for c in result.columns if c.endswith("_forecast")]
        actual_cols = [c.replace("_forecast", "") for c in pred_cols]
        preds       = result[pred_cols].values[:horizon]
        actual_vals = df_actual[actual_cols].values[:horizon]

    val_mse = float(np.mean((preds - actual_vals) ** 2))
    val_mae = float(np.mean(np.abs(preds - actual_vals)))

    (out / "metrics.json").write_text(json.dumps({"val_mse": val_mse, "val_mae": val_mae}))
    print(f"seq_len={inf.get('seq_len', 512)}  use_cross_channel={inf.get('use_cross_channel', False)}"
          f"  cross_channel_heads={inf.get('cross_channel_heads', 8)}  MSE={val_mse:.6f}  MAE={val_mae:.6f}")

if __name__ == "__main__":
    main()
```

**Running basic inference HPO:**

```python
result = runner.run(
    automl_settings={
        "algorithm":                  "bayesian",
        "metric":                     "val_mse",
        "direction":                  "minimize",
        "automl_max_recommendations": 12,
    },
    spec_overrides={
        "dataset.csv":                "/path/to/eval_data.csv",
        "dataset.timestamp_col":      "timestamp",
        "dataset.target_cols":        "target",
        "hpo.forecast_horizon": 72,
        "train.output_dir":           "automl_workspace/basic_hpo/trials",
    },
    hyperparameters={
        "hpo.seq_len":               {"type": "ordered_int", "values": [128, 256, 512]},
        "hpo.use_cross_channel":     {"type": "categorical", "values": [True, False]},
        "hpo.cross_channel_heads":   {"type": "ordered_int", "values": [4, 8, 16]},
        "hpo.cross_channel_dropout": {"type": "uniform",     "min": 0.0, "max": 0.2},
    },
    workspace_path="automl_workspace/basic_hpo",
    execution={
        "type":          "python_script",
        "script":        "/path/to/basic_inference_hpo_trial.py",
        "args":          ["--run-config", "{config_path}"],
        "cwd":           str(forecasting_dir),
        "config_format": "yaml",
    },
)

best = result["best"]
print(f"Best val_mse: {best['metric_value']}")
print(f"Best params: seq_len={best['specs']['hpo.seq_len']}  "
      f"use_cross_channel={best['specs']['hpo.use_cross_channel']}  "
      f"cross_channel_heads={best['specs']['hpo.cross_channel_heads']}")
```

#### DARR mode (tune blending parameters)

Use when you have a context history CSV. Tunes the three kNN blending parameters that control
how much weight the retrieval signal gets versus the model's direct forecast.

| Parameter | Spec key | Default | Search range |
|---|---|---|---|
| Blend weight | `hpo.alpha` | `0.01` | `[0.001, 0.5]` |
| kNN count | `hpo.k` | `64` | `[8, 16, 32, 64, 128]` ordered_int |
| Temperature | `hpo.temperature` | `0.05` | `[0.01, 0.5]` |

**Trial script template (DARR):**

```python
"""Single DARR inference HPO trial — called by the AutoML runner with --run-config {config_path}."""
import argparse, json, sys
from pathlib import Path
import yaml, pandas as pd
import numpy as np

FORECASTING_DIR = Path("/path/to/NV-Tesseract/forecasting")  # adjust to actual clone path
sys.path.insert(0, str(FORECASTING_DIR))

from sdk.forecasting import perform_forecasting

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-config", required=True)
    args = p.parse_args()

    cfg = yaml.safe_load(Path(args.run_config).read_text())
    ds  = cfg["dataset"]
    inf = cfg["hpo"]
    out = Path(cfg.get("train", {}).get("output_dir", "artifacts/inference_hpo"))
    out.mkdir(parents=True, exist_ok=True)

    ts_col = ds.get("timestamp_col", "timestamp")
    df = pd.read_csv(ds["csv"], parse_dates=[ts_col])
    target_cols = [c.strip() for c in ds["target_cols"].split(",")] if ds.get("target_cols") else None

    horizon = inf["forecast_horizon"]
    if horizon <= 0 or horizon >= len(df):
        raise ValueError(f"hpo.forecast_horizon must be in [1, {len(df) - 1}], got {horizon}")
    df_input  = df.iloc[:-horizon]
    df_actual = df.iloc[-horizon:]

    context_df = pd.read_csv(ds["context_csv"], parse_dates=[ts_col])

    result = perform_forecasting(
        df=df_input,
        timestamp_column=ts_col,
        context_df=context_df,
        forecast_horizon=horizon,
        seq_len=inf.get("seq_len", 512),
        alpha=inf["alpha"],
        k=int(inf["k"]),
        temperature=inf["temperature"],
        context_stride=inf.get("context_stride", horizon),
        ckpt=inf.get("ckpt") or None,
        standardizer_pkl=inf.get("standardizer_pkl") or None,
        use_cross_channel=inf.get("use_cross_channel", False),
        return_all_channels=True,
    )

    # Align predictions to actuals by column name, not position.
    if target_cols:
        wanted_pred_cols = [f"{c}_forecast" for c in target_cols]
        missing = [c for c in wanted_pred_cols if c not in result.columns]
        if missing:
            raise ValueError(f"target_cols columns not found in predictions: {missing}")
        preds       = result[wanted_pred_cols].values[:horizon]
        actual_vals = df_actual[target_cols].values[:horizon]
    else:
        pred_cols   = [c for c in result.columns if c.endswith("_forecast")]
        actual_cols = [c.replace("_forecast", "") for c in pred_cols]
        preds       = result[pred_cols].values[:horizon]
        actual_vals = df_actual[actual_cols].values[:horizon]

    val_mse = float(np.mean((preds - actual_vals) ** 2))
    val_mae = float(np.mean(np.abs(preds - actual_vals)))

    (out / "metrics.json").write_text(json.dumps({"val_mse": val_mse, "val_mae": val_mae}))
    print(f"alpha={inf['alpha']}  k={inf['k']}  temp={inf['temperature']}  MSE={val_mse:.6f}  MAE={val_mae:.6f}")

if __name__ == "__main__":
    main()
```

**Running DARR inference HPO:**

```python
result = runner.run(
    automl_settings={
        "algorithm":                  "bayesian",
        "metric":                     "val_mse",
        "direction":                  "minimize",
        "automl_max_recommendations": 8,
    },
    spec_overrides={
        "dataset.csv":                "/path/to/eval_data.csv",
        "dataset.context_csv":        "/path/to/context_history.csv",
        "dataset.timestamp_col":      "timestamp",
        "dataset.target_cols":        "target",
        "hpo.forecast_horizon": 72,
        "hpo.seq_len":          512,
        "hpo.use_cross_channel": False,
        "train.output_dir":           "automl_workspace/darr_hpo/trials",
    },
    hyperparameters={
        "hpo.alpha":       {"type": "uniform",     "min": 0.001, "max": 0.5},
        "hpo.k":           {"type": "ordered_int", "values": [8, 16, 32, 64, 128]},
        "hpo.temperature": {"type": "uniform",     "min": 0.01,  "max": 0.5},
    },
    workspace_path="automl_workspace/darr_hpo",
    execution={
        "type":          "python_script",
        "script":        "/path/to/darr_inference_hpo_trial.py",
        "args":          ["--run-config", "{config_path}"],
        "cwd":           str(forecasting_dir),
        "config_format": "yaml",
    },
)

best = result["best"]
print(f"Best val_mse: {best['metric_value']}")
print(f"Best DARR params: alpha={best['specs']['hpo.alpha']}  "
      f"k={best['specs']['hpo.k']}  temp={best['specs']['hpo.temperature']}")
```
