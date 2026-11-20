---
name: kumo-model-explanations
description: Inspect and interpret gradient attribution (including requests called radiant attribution), column analysis and sampled explanation subgraphs from a Kumo model trained in a Python notebook. Locate the correct run's explanation artifacts, explain feature scores and cohorts, and identify missing or unsupported explanations without automatically retraining.
---

# Kumo model explanations

Use this skill for questions such as “which columns influenced this model?”, “show gradient
attribution”, or “compare predictions and observed outcomes by column cohort”. Kumo calls the
method **gradient attribution/scoring**, not Radiant. Work in the existing Snowflake notebook
with its in-process `kumo.engine` runtime and retained local run files.

## Establish the run and question

- Reuse the user's task, model, graph, and run identity. Do not restart training intake.
- For a fresh `training_result`, use `training_result.explanations`. Otherwise reopen the
  explicit retained run with `kumo.TrainingExplanations.load(run_dir)`. A downloaded model
  bundle is not the original training run.
- If several runs or trainer/refit directories exist, list candidates with their paths and ask
  which one the user means, before showing any score. Never silently select the newest
  directory or combine different runs.
- Identify the task type, target/positive class, explanation stage, and fit/refit used.
  Record unknowns explicitly. These determine what the scores can mean.

Read [the artifact workflow](references/artifacts.md) for discovery, executable readers,
generation behavior, and source provenance. Read only the selected run's files. Kumo is already
present; if importing it fails, report the unavailable runtime and stop that workflow.

## Generation and availability

On supported builds, `Trainer.fit(train_table, snapshot=snapshot)` automatically requests
gradient scoring, column analysis and sampled subgraphs after successful training when
`model_plan.training_job.disable_explain` is false. The selected trainer produces test-stage
artifacts. A failure in this post-training step can leave a usable model with missing or
partial explanations.

The shared explainer can limit the number of test batches. Describe coverage as the processed
test subset unless the run proves it covered the full test population.

This behavior is source-verified against the Kumo modules listed in the reference's *Evidence*
section; it is not a claim about every runtime build. Check the actual runtime version, producer code
when available, artifacts, and logs before promising generation. Do not infer support solely
from the existence of `disable_explain`.

- Column analysis is restricted to supported node-level tasks. It is not a link-prediction
  column-analysis workflow; forecasting is not in the reviewed producer's column-analysis list.
- Gradient scoring supports more task types, but the shared trainer skips TWO_TOWER link models.
  The runtime's explainer task list can narrow support further.
- `training_result.explanations` retrieves retained TEST-stage artifacts and provides gradient
  and cohort plots. There is no public `kumo.Trainer.explain()` or standalone regeneration API.
  Do not call private trainers or enable prediction-time XAI through invented options.
- An inspection request does not authorize another fit. If required files are absent, diagnose
  the reason first. Generate an additional fit only when the user has requested that run.

## Interpret the evidence

**Gradient scores** rank table-column contributions aggregated across the examples processed
by the explainer. The default producer normalizes them across the graph. They are not signed
effects, probabilities, percentages of model accuracy, causal effects, or a per-person explanation.
Retain the table name when ranking identically named columns. Report the actual coverage and
normalization observed; a missing column is not automatically a zero-contribution column.

**Column analysis** groups values into cohorts per table, column, and graph hop and records
population shares, mean predictions, and, when present, mean observed targets. Inspect a class's
own series for multiclass/multilabel tasks. Compare prediction and target patterns with their
population shares. Small groups and sampled neighborhoods limit interpretation.

`prediction_score` and `target_score` summarize between-cohort variation and are normalized
separately by the default producer. Neither is AUROC, an error metric, or the gradient score.
Do not rank these three score families in one list or treat their differences as calibration error.
Use the cohort means for descriptive prediction-versus-target comparisons. A cohort mismatch
is a lead to investigate, not proof of leakage or a causal mechanism.

## Return a useful answer

Every answer, however short, carries:

- the run, task, stage, selected trainer/refit and the exact file paths read;
- gradient scores named as Kumo gradient attribution over the processed test subset, normalized
  across the graph as observed (the default producer, when the total is positive), and never
  called drivers, probabilities or percentages;
- for cohorts, each cohort's population share beside its mean prediction and, when the file
  records one, mean target, read from column analysis;
- for a plot, the plotting code in the reply, reading the selected run's files, titled with the
  table, column, run and test stage;
- when evidence is missing, that no fit was started, and whether the user wants one.

Distinguish observations, hypotheses, and missing evidence. Do not promise individual prediction
explanations from aggregate files; sampled subgraphs cover only the sampled test entities (see the
reference's *Sampled subgraphs*). If only one explanation family is present, analyze that family
and state why the other could not be assessed.

Explain that these files live in local scratch and are excluded from the published model bundle.
Keep this task focused on inspection and interpretation; exporting or regenerating results
requires the user's corresponding request. For learning curves, trial comparisons, or aggregate
evaluation metrics, use `kumo-training-diagnostics` when available.
