# Kumo-Anomaly AutoML Details

Read this reference when the user asks for Kumo-Anomaly AutoML/HPO setup, tunable parameters, VirtualEnvSDK setup, config flow, window-length constraints, inference trial scripts, or AutoML result handoff details.

## Contents

- [AutoML overview](#automl-hpo-hyperparameter-optimization)
- [What AutoML tunes](#what-automl-tunes)
- [Runner setup](#runner-setup)
- [Fine-tuning AutoML](#fine-tuning-automl)
- [Inference HPO with AutoML](#inference-hpo-with-automl)

## AutoML (HPO: hyperparameter optimization)

> For algorithm selection, budget configuration, preflight checks, monitoring,
> and result handoff, use the **`tao-skill-bank:tao-run-automl`** skill. This
> section documents what Kumo-Anomaly tunes, the VirtualEnvSDK setup required
> for this containerless model, and the inference HPO trial script.

This skill supports AutoML for two use cases — both run locally via
VirtualEnvSDK with no container required:

| Use case | Action | Metric | Requires labels? |
|---|---|---|---|
| Fine-tuning HPO | `train` | `val_loss` (minimize) | No |
| Inference HPO | `inference_hpo` | `f1_score` (maximize) | Yes |

### What AutoML tunes

#### Fine-tuning

| Parameter | Spec key | Default | Search range | Why it matters |
|---|---|---|---|---|
| Learning rate | `train.lr` | `1e-5` | `[1e-7, 1e-3]` | Most impactful — diffusion backbone is sensitive to LR |
| Weight decay | `train.weight_decay` | `1e-6` | `[1e-9, 1e-3]` | Regularization; prevents overfitting on small datasets |
| Batch size | `train.batch_size` | `16` | `[4, 32]` | Larger batches stabilize gradients; capped at 32 (MPS OOM) |
| Gradient clip | `train.grad_clip` | `1.0` | `[0.1, 10.0]` | Prevents gradient explosion in diffusion fine-tuning |

**Fixed (not searched):** `dataset.mask_ratio`, `dataset.window_length`, `dataset.csv`, `dataset.val_csv`,
`dataset.split`, `dataset.window_stride`, `train.seed`, `train.output_dir`, `train.pretrained_model`,
`train.config`.

#### Inference HPO

| Parameter | Spec key | Default | Search range | Effect |
|---|---|---|---|---|
| Diffusion samples | `hpo.nsample` | `15` | `[5, 30]` | More samples → better MAE estimate, slower per trial |

**`threshold_strategy`** (`scs` or `macs`) is fixed per AutoML run — run once
per strategy and compare F1 to pick the winner.

**Fixed:** `hpo.model_path`, `hpo.config_path` — set these in
`spec_overrides`, not the search space.

### AutoML artifacts

| Use case | Schema | Spec template |
|---|---|---|
| Fine-tuning | `schemas/train.schema.json` | `references/spec_template_train.yaml` |
| Inference HPO | `schemas/inference_hpo.schema.json` | `references/spec_template_inference_hpo.yaml` |

### Runner setup

Use `tao-run-automl` for AutoML control-plane setup, preflight, runner
construction, `AutoMLRunner.run()` invocation, monitoring, and result handoff.
Build the containerless SDK exactly as its **Runner Construction** section
shows: `VirtualEnvSDK(venv_path=..., work_dir=...)`, documented in
`skills/applications/tao-run-automl/references/automl-runner-configuration.md`
under "VirtualEnvSDK (containerless venv runs)".

> **Always pass `work_dir=`.** Omitted, `VirtualEnvSDK` defaults to a hidden
> state directory under `$HOME` and every trial's checkpoints land in the home
> directory. Point it at a scratch filesystem with room for
> `automl_max_recommendations` checkpoints.

This model reference lists Kumo-Anomaly search spaces, overrides,
constraints, and trial payloads.

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

```python
import secrets
import sys
from pathlib import Path

skill_bank = Path("/path/to/tao-skill-bank")
sys.path.insert(0, str(skill_bank / "skills/applications/tao-run-automl/scripts"))
from resolve_automl_session import validate_session_settings

kumo_anomaly_dir = Path("/path/to/Kumo-TS/Kumo-Anomaly")
skill_dir = skill_bank / "skills/models/tao-finetune-kumo-anomaly"

# Build sdk + runner per tao-run-automl section "Runner Construction":
# sdk = VirtualEnvSDK(venv_path=<venv>, work_dir=<scratch>)  # work_dir is not optional in practice

automl_settings = {
    "algorithm":                  "bayesian",
    "metric":                     "val_loss",
    "direction":                  "minimize",
    "automl_max_recommendations": 10,           # adjust to your compute budget
    "session_id":                 secrets.token_hex(6),   # explicit, required
}
validate_session_settings(automl_settings, resume=False, workspace=None)

result = runner.run(
    automl_settings=automl_settings,
    # spec_overrides MUST use dotted keys ("dataset.csv"), NOT nested dicts
    # ({"dataset": {"csv": ...}}). Nested dicts replace the entire sub-tree in
    # the spec template, which removes sibling keys like window_length from the
    # search space — AutoML silently excludes them as "deleted parameters".
    spec_overrides={
        "dataset.csv":        "/path/to/normal_data.csv",
        "train.epochs":       20,
        "train.num_epochs":   20,
        "train.output_dir":   "automl_workspace/finetune/trials",
    },
    # dataset.mask_ratio and dataset.window_length are fixed (not searched), see "Why mask_ratio and
    # window_length are not searched" below. Set them via spec_overrides if the defaults do not fit.
    workspace_path="automl_workspace/finetune",
    execution={
        "type":          "python_script",
        "script":        str(kumo_anomaly_dir / "examples/finetune_example.py"),
        "args":          ["--run-config", "{config_path}"],
        "cwd":           str(kumo_anomaly_dir),
        "config_format": "yaml",
    },
)

best = result["best"]
print(f"Best val_loss : {best['metric_value']:.6f}")
print(f"Best params   : {best['specs']}")
```

The winning checkpoint is at `train.output_dir/rec_<N>/best_finetuned_model.pth`.
Pass it as `model_path` to `perform_anomaly_analysis_with_diffusion` for inference.

#### How the config flows (fine-tuning)

Each trial:
1. Samples hyperparameters from `schemas/train.schema.json`
2. Merges with `spec_template_train.yaml` defaults
3. Writes merged spec to `{config_path}` (YAML)
4. Calls `<venv>/bin/python finetune_example.py --run-config {config_path}`
5. Reads `val_loss` from `metrics.json` in the trial's `output_dir`

#### Why `mask_ratio` and `window_length` are not searched

The objective is the validation reconstruction loss, and both parameters change *what that loss
measures*: the validation set is masked and windowed with the same values. Measured on the
pretrained model with no training (lr ~ 0): `mask_ratio` 0.3 gives val_loss ~1.0-1.1, 0.7 gives
~0.035 and 0.9 gives ~0.057, while `window_length` 50/100/140 moves it by under 10%. Trials with
different `mask_ratio` are therefore not comparable, and the search would pick whichever value makes
the task easiest rather than the best model. Both are fixed; set them with `spec_overrides`
(`dataset.window_length` must be divisible by `dataset.split`, default `10`, otherwise the trial fails
with `ValueError: window_length must be divisible by split`).

#### Interpreting the result

`val_loss` only ranks trials against each other on this fixed protocol; it does not guarantee a better
detector. On a 2-epoch run the pretrained checkpoint is already close to optimal, and `lr` in
`[1e-7, 1e-5]` is practically "no training" (val_loss within ~5%). Before adopting the winner, check
it on labeled data: run the inference HPO action with `hpo.model_path` set to the trial's
`best_finetuned_model.pth` (and `hpo.config_path`), and compare F1 with the pretrained model.

### Inference HPO with AutoML

Requires **labeled data** — a CSV with a ground-truth `label` column (0/1) plus numeric feature
columns. AutoML tunes `nsample` to maximize F1 on that eval set.

The trial script ships with this skill: **`scripts/inference_hpo_trial.py`**. Point
`execution.script` at it and set `execution.cwd` to your `Kumo-TS/Kumo-Anomaly` clone. It drops
the label column and any non-numeric columns (for example a timestamp) before calling
`perform_anomaly_analysis_with_diffusion`.

#### Run AutoML (once per strategy, compare F1)

```python
import secrets
import sys
from pathlib import Path

skill_bank = Path("/path/to/tao-skill-bank")
sys.path.insert(0, str(skill_bank / "skills/applications/tao-run-automl/scripts"))
from resolve_automl_session import validate_session_settings

kumo_anomaly_dir = Path("/path/to/Kumo-TS/Kumo-Anomaly")
skill_dir = skill_bank / "skills/models/tao-finetune-kumo-anomaly"

# Build sdk + runner per tao-run-automl section "Runner Construction" (action="inference_hpo"):
# sdk = VirtualEnvSDK(venv_path=<venv>, work_dir=<scratch>)  # work_dir is not optional in practice

results_by_strategy = {}
for strategy in ["scs", "macs"]:
    automl_settings = {
        "algorithm":                  "bayesian",
        "metric":                     "f1_score",
        "direction":                  "maximize",
        "automl_max_recommendations": 8,          # nsample range is narrow; 8 covers it well
        "session_id":                 secrets.token_hex(6),   # one id per run
    }
    validate_session_settings(automl_settings, resume=False, workspace=None)

    result = runner.run(
        automl_settings=automl_settings,
        # spec_overrides MUST use dotted keys.
        spec_overrides={
            "dataset.csv":            "/path/to/labeled_data.csv",
            "hpo.threshold_strategy": strategy,
            "hpo.model_path":         "",   # "" => pretrained HF weights; or a fine-tuned .pth. Do not use None: the spec validator rejects null.
            "train.output_dir":       f"automl_workspace/inference_hpo_{strategy}/trials",
        },
        workspace_path=f"automl_workspace/inference_hpo_{strategy}",
        execution={
            "type":          "python_script",
            "script":        str(skill_dir / "scripts/inference_hpo_trial.py"),
            "args":          ["--run-config", "{config_path}"],
            "cwd":           str(kumo_anomaly_dir),
            "config_format": "yaml",
        },
    )
    best = result["best"]
    results_by_strategy[strategy] = best
    # best["specs"] uses the same dotted keys as spec_overrides
    print(f"{strategy:4s}  F1={best['metric_value']:.4f}  nsample={best['specs']['hpo.nsample']}")

winner = max(results_by_strategy, key=lambda s: results_by_strategy[s]["metric_value"])
print(f"\nBest: strategy={winner}  nsample={results_by_strategy[winner]['specs']['hpo.nsample']}")
```

**Timestamp / label columns:** the trial script removes them before inference.

**Labeled data requirement:** skip inference HPO when you have no ground-truth
labels — without them there is no meaningful objective. Use `nsample=15` + `scs`
as reasonable defaults.
