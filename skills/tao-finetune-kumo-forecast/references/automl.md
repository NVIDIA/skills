# Forecasting AutoML Details

Read this reference when the user asks for Kumo-Forecast AutoML/HPO setup, tunable parameters, runner examples, inference trial scripts, DARR HPO, or AutoML result handoff details.

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
> Kumo-Forecast search spaces, overrides, and trial payloads.

> **Session gate (required).** `tao-run-automl` refuses to launch without an explicit
> `automl_settings["session_id"]` and a call to `validate_session_settings(...)` (from
> `skills/applications/tao-run-automl/scripts/resolve_automl_session.py`). Every example
> below builds `automl_settings` that way; copying a bare `automl_settings={...}` without
> `session_id` fails the gate.
>
> **No baseline eval by design.** This skill has no `evaluate` action: fine-tuning reports its
> own validation metric and inference HPO scores each trial directly. `tao-run-automl`
> therefore records the baseline as unavailable (see "Models with no `evaluate` action by
> design" in `skills/applications/tao-run-automl/references/automl-preflight-concepts.md`)
> and continues.

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
import secrets
import sys
from pathlib import Path

skill_bank = Path("/path/to/tao-skill-bank")
sys.path.insert(0, str(skill_bank / "skills/applications/tao-run-automl/scripts"))
from resolve_automl_session import validate_session_settings

forecasting_dir = Path("/path/to/Kumo-TS/Kumo-Forecast")
skill_dir = skill_bank / "skills/models/tao-finetune-kumo-forecast"

# Build sdk + runner per tao-run-automl section "Runner Construction":
# sdk = VirtualEnvSDK(venv_path=<venv>, work_dir=<scratch>)  # work_dir is not optional in practice

automl_settings = {
    "algorithm":                  "bayesian",
    "metric":                     "val_mse",
    "direction":                  "minimize",
    "automl_max_recommendations": 5,
    "session_id":                 secrets.token_hex(6),   # explicit, required
}
validate_session_settings(automl_settings, resume=False, workspace=None)

result = runner.run(
    automl_settings=automl_settings,
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

AutoML can search inference parameters against a ground-truth eval window. The last
`hpo.forecast_horizon` rows of `dataset.csv` are held back as ground truth; each trial runs
inference on the rest and writes `val_mse` / `val_mae` to `metrics.json`.

The trial script ships with this skill: **`scripts/inference_hpo_trial.py`**. Point
`execution.script` at it and set `execution.cwd` to your `Kumo-TS/Kumo-Forecast` clone. Mode is
chosen by the spec: a non-empty `dataset.context_csv` runs **DARR**, an empty one runs **basic**
(direct forecast). The script translates the nested AutoML run-config (`dataset:` / `hpo:` /
`train:`) into the flat `inference:` mapping that `sdk.forecasting.load_forecasting_config`
accepts, so keep tunables under `hpo:`.

> **Runner arguments.** Select what to search with `automl_hyperparameters=[...]` — a list of
> schema parameter names — and constrain values with `custom_param_ranges`. There is no
> `hyperparameters=` argument: unknown keyword arguments are forwarded to the platform SDK and
> every trial fails at job creation with `unexpected keyword argument`. A parameter can only be
> searched if its schema entry has `automl_enabled: true`
> (`schemas/inference_hpo.schema.json`).

#### Basic mode (tune model parameters)

Use when you have no context history for DARR.

`hpo.seq_len` is **not** tunable: the published checkpoints' forecast head is sized for
`seq_len=512`, and any other value fails every trial with
`size mismatch for head.linear.weight`. Leave it at the default `512`.

`hpo.cross_channel_heads` only changes predictions on **multivariate** eval data with
`use_cross_channel: true`; on a single-channel series every head count gives identical results.
`cross_channel_dropout` is intentionally not tunable here: it has no effect at inference
(predictions are identical for any value), so it would only waste trials.

| Parameter | Spec key | Default | Search range |
|---|---|---|---|
| Cross-channel | `hpo.use_cross_channel` | `false` | `[true, false]` |
| Attention heads | `hpo.cross_channel_heads` | `8` | `{4, 8, 16}` (must divide the embedding size) |

```python
import secrets
import sys
from pathlib import Path

skill_bank = Path("/path/to/tao-skill-bank")
sys.path.insert(0, str(skill_bank / "skills/applications/tao-run-automl/scripts"))
from resolve_automl_session import validate_session_settings

forecasting_dir = Path("/path/to/Kumo-TS/Kumo-Forecast")
skill_dir = skill_bank / "skills/models/tao-finetune-kumo-forecast"

automl_settings = {
    "algorithm":                  "bayesian",
    "metric":                     "val_mse",
    "direction":                  "minimize",
    "automl_max_recommendations": 12,
    "session_id":                 secrets.token_hex(6),
}
validate_session_settings(automl_settings, resume=False, workspace=None)

result = runner.run(
    automl_settings=automl_settings,
    spec_overrides={
        "dataset.csv":           "/path/to/eval_data.csv",
        "dataset.timestamp_col": "timestamp",
        "dataset.target_cols":   "target",
        "hpo.forecast_horizon":  72,
        "train.output_dir":      "automl_workspace/basic_hpo/trials",
    },
    automl_hyperparameters=["hpo.use_cross_channel", "hpo.cross_channel_heads"],
    workspace_path="automl_workspace/basic_hpo",
    execution={
        "type":          "python_script",
        "script":        str(skill_dir / "scripts/inference_hpo_trial.py"),
        "args":          ["--run-config", "{config_path}"],
        "cwd":           str(forecasting_dir),
        "config_format": "yaml",
    },
)

best = result["best"]
print(f"Best val_mse: {best['metric_value']}")
print(f"Best params:  {best['specs']}")
```

#### DARR mode (tune blending parameters)

Use when you have a context history CSV. Tunes the three kNN blending parameters that control
how much weight the retrieval signal gets versus the model's direct forecast.

| Parameter | Spec key | Default | Search range |
|---|---|---|---|
| Blend weight | `hpo.alpha` | `0.01` | `[0.001, 0.5]` |
| kNN count | `hpo.k` | `64` | `[8, 128]` int |
| Temperature | `hpo.temperature` | `0.05` | `[0.01, 0.5]` |

```python
automl_settings = {
    "algorithm":                  "bayesian",
    "metric":                     "val_mse",
    "direction":                  "minimize",
    "automl_max_recommendations": 8,
    "session_id":                 secrets.token_hex(6),
}
validate_session_settings(automl_settings, resume=False, workspace=None)

result = runner.run(
    automl_settings=automl_settings,
    spec_overrides={
        "dataset.csv":           "/path/to/eval_data.csv",
        "dataset.context_csv":   "/path/to/context_history.csv",   # non-empty => DARR mode
        "dataset.timestamp_col": "timestamp",
        "dataset.target_cols":   "target",
        "hpo.forecast_horizon":  72,
        "train.output_dir":      "automl_workspace/darr_hpo/trials",
    },
    automl_hyperparameters=["hpo.alpha", "hpo.k", "hpo.temperature"],
    workspace_path="automl_workspace/darr_hpo",
    execution={
        "type":          "python_script",
        "script":        str(skill_dir / "scripts/inference_hpo_trial.py"),
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
