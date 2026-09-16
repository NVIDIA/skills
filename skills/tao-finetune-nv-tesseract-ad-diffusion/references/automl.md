# AD Diffusion AutoML Details

Read this reference when the user asks for NV-Tesseract AD Diffusion AutoML/HPO setup, tunable parameters, VirtualEnvSDK setup, config flow, window-length constraints, inference trial scripts, or AutoML result handoff details.

## Contents

- [AutoML overview](#automl-hpo-hyperparameter-optimization)
- [What AutoML tunes](#what-automl-tunes)
- [Runner setup](#runner-setup)
- [Fine-tuning AutoML](#fine-tuning-automl)
- [Inference HPO with AutoML](#inference-hpo-with-automl)

## AutoML (HPO: hyperparameter optimization)

> For algorithm selection, budget configuration, preflight checks, monitoring,
> and result handoff, use the **`tao-skill-bank:tao-run-automl`** skill. This
> section documents what AD Diffusion tunes, the VirtualEnvSDK setup required
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
| Mask ratio | `dataset.mask_ratio` | `0.7` | `[0.3, 0.9]` | Fraction of each window masked; shapes reconstruction difficulty |
| Window length | `dataset.window_length` | `100` | `[50, 140]` | Temporal context per training sample; must be divisible by `split` |

**Fixed (not searched):** `dataset.csv`, `dataset.val_csv`, `dataset.split`,
`dataset.window_stride`, `train.seed`, `train.output_dir`, `train.pretrained_model`,
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

This model reference lists NV-Tesseract AD Diffusion search spaces, overrides,
constraints, and trial payloads.

### Fine-tuning AutoML

```python
from pathlib import Path

ad_diffusion_dir = Path("/path/to/NV-Tesseract/ad_diffusion")
skill_dir = Path("/path/to/tao-skill-bank/skills/models/tao-finetune-nv-tesseract-ad-diffusion")

# Build sdk + runner per tao-run-automl section "Runner Construction":
# sdk = VirtualEnvSDK(venv_path=<venv>, work_dir=<scratch>)  # work_dir is not optional in practice

result = runner.run(
    automl_settings={
        "algorithm":                  "bayesian",
        "metric":                     "val_loss",
        "direction":                  "minimize",
        "automl_max_recommendations": 10,           # adjust to your compute budget
    },
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
    # window_length must be divisible by split (default 10). The schema allows
    # any integer in [50, 140], so restrict to multiples of 10 explicitly —
    # otherwise the Bayesian sampler proposes values like 81 or 99 and every
    # trial fails with "window_length must be divisible by split".
    custom_param_ranges={
        "dataset.window_length": {
            "valid_options": [50, 60, 70, 80, 90, 100, 110, 120, 130, 140],
            "value_type":    "ordered_int",
        },
    },
    workspace_path="automl_workspace/finetune",
    execution={
        "type":          "python_script",
        "script":        str(ad_diffusion_dir / "examples/finetune_example.py"),
        "args":          ["--run-config", "{config_path}"],
        "cwd":           str(ad_diffusion_dir),
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

#### Window length constraint

`window_length` must be divisible by `split` (default `10`). The schema allows
any integer in `[50, 140]`, so the Bayesian sampler will propose non-multiples
like 81 or 99 — causing every trial to fail immediately with
`ValueError: window_length must be divisible by split`.

Always pass `custom_param_ranges` to restrict candidates to valid multiples:

```python
custom_param_ranges={
    "dataset.window_length": {
        "valid_options": [50, 60, 70, 80, 90, 100, 110, 120, 130, 140],
        "value_type":    "ordered_int",
    },
}
```

If your dataset is larger and you want a wider window, add more multiples of 10
to `valid_options` rather than raising the schema cap.

### Inference HPO with AutoML

Requires **labeled data** — a CSV with a ground-truth `label` column (0/1).
AutoML tunes `nsample` to maximize F1 on that eval set.

The trial script is not shipped with NV-Tesseract — generate it for the user's
working directory based on the template below, then point `execution.script` at it.

**Trial script template** (save anywhere, e.g. `inference_hpo_trial.py`):

```python
"""Single inference HPO trial — called by the AutoML runner with --run-config {config_path}."""
import argparse, json, sys
from pathlib import Path
import yaml, pandas as pd
from sklearn.metrics import f1_score

AD_DIR = Path("/path/to/NV-Tesseract/ad_diffusion")   # adjust to actual clone path
sys.path.insert(0, str(AD_DIR))

from sdk.anomaly_analysis import perform_anomaly_analysis_with_diffusion

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-config", required=True)
    args = p.parse_args()

    cfg = yaml.safe_load(Path(args.run_config).read_text())
    ds  = cfg["dataset"]
    inf = cfg["hpo"]
    out = Path(cfg.get("train", {}).get("output_dir", "artifacts/inference_hpo"))
    out.mkdir(parents=True, exist_ok=True)

    df      = pd.read_csv(ds["csv"]).head(ds.get("eval_rows", 2000))
    labels  = df[ds["label_col"]].values
    df_feat = df.drop(columns=[ds["label_col"]], errors="ignore").select_dtypes(include="number")

    results = perform_anomaly_analysis_with_diffusion(
        df=df_feat,
        nsample=inf["nsample"],
        threshold_strategy=inf["threshold_strategy"],
        model_path=inf.get("model_path") or None,
        config_path=inf.get("config_path") or None,
    )
    f1 = f1_score(labels, results["Anomaly"].values, zero_division=0)
    (out / "metrics.json").write_text(json.dumps({"f1_score": f1}))
    print(f"nsample={inf['nsample']}  strategy={inf['threshold_strategy']}  F1={f1:.4f}")

if __name__ == "__main__":
    main()
```

#### Run AutoML (once per strategy, compare F1)

```python
from pathlib import Path

ad_diffusion_dir = Path("/path/to/NV-Tesseract/ad_diffusion")
skill_dir = Path("/path/to/tao-skill-bank/skills/models/tao-finetune-nv-tesseract-ad-diffusion")

# Build sdk + runner per tao-run-automl section "Runner Construction":
# sdk = VirtualEnvSDK(venv_path=<venv>, work_dir=<scratch>)  # work_dir is not optional in practice

results_by_strategy = {}
for strategy in ["scs", "macs"]:
    result = runner.run(
        automl_settings={
            "algorithm":                  "bayesian",
            "metric":                     "f1_score",
            "direction":                  "maximize",
            "automl_max_recommendations": 8,          # nsample range is narrow; 8 covers it well
        },
        # spec_overrides MUST use dotted keys — see fine-tuning section for why.
        spec_overrides={
            "dataset.csv":                  "/path/to/labeled_data.csv",
            "hpo.threshold_strategy": strategy,
            "hpo.model_path":         None,   # None → pretrained HF weights; or path to fine-tuned .pth
            "train.output_dir":             f"automl_workspace/inference_hpo_{strategy}/trials",
        },
        workspace_path=f"automl_workspace/inference_hpo_{strategy}",
        execution={
            "type":          "python_script",
            "script":        "/path/to/inference_hpo_trial.py",   # generated trial script
            "args":          ["--run-config", "{config_path}"],
            "cwd":           str(ad_diffusion_dir),
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

**Timestamp / label columns:** drop them before passing to the inference function — the template above does this via `errors="ignore"`.

**Labeled data requirement:** skip inference HPO when you have no ground-truth
labels — without them there is no meaningful objective. Use `nsample=15` + `scs`
as reasonable defaults.
