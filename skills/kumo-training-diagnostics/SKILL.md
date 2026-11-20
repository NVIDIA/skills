---
name: kumo-training-diagnostics
description: Evaluate an existing Kumo training run in a Python notebook using aggregate metrics, trial results, learning curves, refit evidence, and training progress. Use for model-quality questions or training diagnostics; inspect existing results without automatically tuning, rerunning, or deploying a model.
---

# Kumo evaluation and training diagnostics

Answer “how did this model perform?”, “is it overfitting?”, “which trial was selected?”, and
“what does this training progress show?” using evidence from the user's existing Snowflake
notebook run. Kumo executes synchronously in the notebook process; do not invent hosted job
IDs, status APIs, polling, or recovery operations.

## Bind the analysis to one run

Reuse the known task, target/positive class, time horizon, scoring cohort, and run. For a fresh
fit, begin with `training_result.metrics` and `training_result.bundle_dir`. If only retained
files remain, use the explicit run directory supplied. If a restored model has aggregate
metrics or `metric_history`, inspect them. A model bundle does not contain TensorBoard or
analysis files.

Resolve multiple candidate runs before drawing conclusions. Do not choose by modification
time, mix directories from separate fits, or reconfigure scratch during inspection. Record
the run path, task, metric, data split, and fit/refit for each comparison.

Read [the diagnostic workflow](references/diagnostics.md) for runnable inspection cells,
artifact contracts, failure cases, and provenance. Kumo is already present in the notebook.

## Evaluate in this order

1. **Aggregate results:** list actual metric keys and finite values from `.metrics`, an
   attribute, not a method. Do not assume AUROC, calibration, confusion matrices, or per-class
   measures were all computed. Missing values mean unavailable, not zero.
2. **Selection evidence:** examine trial validation results, resolved plans, checkpoint
   availability, and `best_trainer.json`. Identify the configured tuning metric and its
   direction before comparing trials. Do not choose a model using test-set performance.
3. **Learning curves:** inspect each trial's completed epoch records and compare compatible
   training/validation metrics. Keep trial and refit histories separate. Zero completed epochs
   is a valid diagnostic state, not a reason to fabricate a curve or crash on an empty dataframe.
4. **Progress and failures:** distinguish recorded elapsed/estimated time, completed trials,
   missing files, pruned/incomplete trials, and actual exceptions. A stale progress file is not
   proof the process is still running or dead. Diagnose the observed stage before suggesting
   any next experiment.

## Interpret metrics correctly

- For binary classification, AUROC measures ranking and AP/AUPRC depends on prevalence.
  Accuracy depends on class balance and threshold. Compare like populations and splits.
- For multiclass or multilabel tasks, preserve class labels and averaging conventions.
  An aggregate score does not establish quality for every class.
- For regression, preserve target units. Lower MAE/RMSE is usually better; losses with different
  definitions or target scales are not interchangeable. Respect the recorded metric direction.
- For ranking/link prediction, preserve K, candidate population, filters, and evaluation
  sampling. Precision@10 and MAP@100 are different measures.
- For forecasting, preserve horizon, temporal split and aggregation window. Do not present a
  temporal regression score as proof of multi-horizon performance.
- A widening train/validation gap can suggest overfitting. Distribution shift, split design,
  and sampling can also explain it. A flat curve does not by itself identify a failed encoder,
  inadequate data, or a specific hyperparameter fix.

Metrics named `test_refit_*` belong to a different evaluation from `test_*`. Keep both names
and provenance. A full-data refit is not an independent held-out quality estimate. Do not
replace SDK model selection with an agent's ranking of final epoch scores.

## Scope and response

Return a compact evidence-backed assessment: available metrics, best-trial selection evidence,
curve/progress observations, and the specific uncertainty or next diagnostic action. Reference
the files and metric names used. Plot only recorded metrics, with split, trial, epoch and units
visible; do not connect unrelated trials into one curve.

Segment analysis requires appropriate prediction rows, labels, and group membership. Aggregate
metrics alone cannot reveal which customer segment underperforms. For feature attribution or
cohort explanations use `kumo-model-explanations` when available. Neither skill claims causal
effects or automatically detects all leakage.

Inspect and interpret by default. Propose tuning changes as hypotheses; running experiments,
changing data/splits, retraining, or writing results to Snowflake is outside an inspection
request. No automatic retries. For retention, use
[model persistence](../kumo-model-persistence/SKILL.md). TensorBoard and analysis files
remain scratch diagnostics outside the published model bundle. Per-epoch search-trial metrics
are in the bundle as `metric_history`.
