# Diagnostic workflow

## Public result and run identity

For a fresh fit in the existing Snowflake notebook:

```python
from pathlib import Path
import json
import math
import pandas as pd

metric_rows = []
for name, value in training_result.metrics.items():
    numeric = None if value is None else float(value)
    metric_rows.append({
        "metric": name, "value": numeric,
        "available": numeric is not None and math.isfinite(numeric),
    })
metrics = pd.DataFrame(metric_rows, columns=["metric", "value", "available"])
display(metrics)

run_dir = Path(training_result.bundle_dir).parent.resolve()  # fresh fit only
print("Run:", run_dir)
print("Bundle:", training_result.bundle_dir)
print("Published model:", training_result.published_model)
```

The public result fields include `metrics`, `artifact_paths`, `bundle_dir`, `published_model`,
`metric_history`.
There is no `.job_id`, `.history()`, `.model_uri`, `.refitted_trainval` or `.refitted_full` on
this local TrainingResult. Those similarly named internal/hosted fields are not public accessors.
`artifact_paths` identifies published model components; it is not an inventory of all trial files.

After a restart, set `run_dir` to the known retained run path instead. Do not derive the original
run from a downloaded bundle's parent directory. If only the bundle remains, explain what survives.
Final metrics, configuration, and `result.metric_history` survive (`None` for older bundles).
TensorBoard files and `analyze/` output do not.

## Discover, then select the AutoML directory

```python
if not run_dir.is_dir():
    raise FileNotFoundError(f"Training run is unavailable: {run_dir}")
progress_files = sorted(run_dir.rglob("auto_train_progress.json"))
for path in progress_files:
    print(path.relative_to(run_dir))
if not progress_files:
    print("No AutoML history found in this run; inspect logs and retained artifacts.")
```

Choose one exact candidate based on the requested fit, not on recency. Set `automl_dir` to that
file's parent. This may be below the run directory. Within it, inspect `best_trainer.json`,
`version_*/model_plan.json`, `version_*/val_result.csv`, and checkpoint directories if present.
The selection JSON records `log_dir`; copied runs can retain an original absolute path. Resolve
that against the selected run, never blindly traverse another host's or another run's path.

## Trial metrics without hiding incomplete results

```python
def read_trial_results(automl_dir):
    frames, issues = [], []
    for trial in sorted(Path(automl_dir).glob("version_*")):
        if not trial.is_dir() or not trial.name.removeprefix("version_").isdigit():
            continue
        path = trial / "val_result.csv"
        if not path.is_file():
            issues.append({"trial": trial.name, "issue": "validation result missing"})
            continue
        try:
            frame = pd.read_csv(path)
        except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError) as error:
            issues.append({"trial": trial.name, "issue": str(error)})
            continue
        if frame.empty:
            issues.append({"trial": trial.name, "issue": "validation result has no rows"})
            continue
        frame.insert(0, "trial", trial.name)
        frame.insert(1, "source_path", str(path))
        frame.insert(2, "checkpoint_present", any((trial / "checkpoint").glob("*.ckpt")))
        frames.append(frame)
    results = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return results, pd.DataFrame(issues, columns=["trial", "issue"])

trial_results, trial_issues = read_trial_results(automl_dir)
display(trial_issues)
if trial_results.empty:
    print("No readable, nonempty trial validation results.")
else:
    display(trial_results)
```

A validation CSV alone does not prove the trial completed successfully or was eligible for
selection. The reviewed selector requires a checkpoint and a non-NaN value for the validation
tuning metric, then sorts using that metric's direction. `best_trainer.json` is the recorded
selection evidence. Inspect resolved `model_plan.json` rather than assuming defaults or ranking
by whichever metric happens to look best. Missing copied checkpoints limit reconstruction of
selection; they do not prove original training failed.

## Learning curves and progress

```python
def read_learning_curves(progress_path):
    progress = json.loads(Path(progress_path).read_text())
    trials = progress.get("trial_progress")
    if not isinstance(trials, dict):
        raise ValueError(f"Unexpected trial_progress schema in {progress_path}")
    rows = []
    for trial_id, trial in trials.items():
        epochs = trial.get("metrics", {})
        if not isinstance(epochs, dict):
            raise ValueError(f"Unexpected epoch metrics for trial {trial_id}")
        for epoch, record in epochs.items():
            row = {"trial": str(trial_id), "epoch": int(epoch),
                   "time_taken": record.get("time_taken")}
            for field in ("train_metrics", "validation_metrics"):
                values = record.get(field) or {}
                if not isinstance(values, dict):
                    raise ValueError(f"Unexpected {field} for trial {trial_id}")
                row.update(values)
            rows.append(row)
    curves = pd.DataFrame(rows)
    if curves.empty:
        curves = pd.DataFrame(columns=["trial", "epoch", "time_taken"])
    else:
        curves = curves.sort_values(["trial", "epoch"]).reset_index(drop=True)
    return progress, curves

progress, curves = read_learning_curves(Path(automl_dir) / "auto_train_progress.json")
print({key: progress.get(key) for key in (
    "total_trials", "completed_trials", "elapsed_training_time", "estimated_training_time"
)})
if curves.empty:
    print("No completed epoch records are available.")
else:
    print("Recorded metrics:", [name for name in curves if name.startswith(("train_", "val_"))])
    display(curves)
```

If JSON is unreadable while training is active, report the path and read error and take another
bounded snapshot only when appropriate. Do not replace unreadable content with an empty success.
For completed runs, malformed JSON is an artifact/schema problem to investigate. Do not modify
the producer's files to make the analysis work.

Choose one trial and recorded numeric metric pair before plotting, for example `train_loss`
and `val_loss` if both exist. Preserve NaNs and gaps instead of replacing them with zero. Metric
names and availability depend on task and build. Epoch keys are sorted numerically; epoch zero
is a valid completed record. Individual `train_progress.json` files can expose per-trial/refit
histories, but must be read separately from the AutoML aggregate and labeled by their paths.
Recorded estimated times are estimates, not measured remaining duration or a live process status.

## Diagnosis limits

- Final aggregate metrics, trial validation CSVs, per-epoch curves, and refit test metrics answer
  different questions. Do not merge them into a single unlabeled score.
- No validation curve for a full-data refit can be expected. Absence is not evidence of failure.
- Trial pruning, exceptions, and incomplete copying can all leave missing results. Use the
  run logs to distinguish them. Do not infer OOM solely from a short or missing curve.
- A high score is not proof of no leakage. A low score does not establish which feature or
  encoder is responsible. State hypotheses and the evidence needed to test them.
- TensorBoard files and `analyze/` output are not model-bundle components. They require retained
  scratch files for this workflow. A restored model returns per-epoch search-trial metrics only
  through `result.metric_history`, without refit epochs. Do not promise other cold-load history.
  See [retention](../../kumo-model-persistence/references/retention.md) to preserve them.

## Source provenance

Source: Kumo modules, as shipped in the runtime:

- `kumo.engine.trainer.trainer.Trainer.fit`: public TrainingResult and synchronous fit.
- `kumo_core.callbacks.train_progress_logger`: AutoML and per-trial JSON writers, epoch metrics
  and timing fields.
- `kumoapi.jobs.EpochMetrics`, `kumoapi.jobs.TrainerProgress`, `kumoapi.jobs.AutoTrainerProgress`:
  progress schema. The local workflow does not expose the hosted job APIs in that module.
- `kumo_core.automl.auto_trainer.AutoTrainer.get_best_trials`: trial selection.
- `kumo_core.training.orchestration`: selected-trainer metadata, original-fit versus refit
  evaluation and `test_refit_*` naming.
- `kumo_core.artifacts.VitalArtifactSources`: publication allowlist.

These are source-verified contracts. Runtime files and metric names remain the evidence for
the particular Snowflake run; this reference is not a claim that every task/build produces
every diagnostic. No Linux-only operation or hosted Kumo service is needed for these readers.
