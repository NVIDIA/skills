<!-- miagent-template-source: assets/challenge_template/baseline_template/runs/exp_template/experiment.md -->
<!-- miagent-template-contract: active; keep the miagent-template-source loaded for creation/material updates; reload after resume or compaction -->
<!-- miagent-template-schema: experiment-record-v2 -->
# Experiment: expXXX_<slug>

One decision-bearing scientific experiment owns one record for its full
lifecycle. Create it before launch. Keep retries, resumes, inference,
evaluation, visualization, packaging, submission, and execution repairs inside
this experiment; they do not create new experiments. Job IDs are execution
evidence, not research identities.

When initializing a planned record, copy only through `Human Steering`. Write
`Intent` before adding the experiment to `dashboard.md`. After the first job
launches, do not rewrite the intent to fit the result. Label a factual
correction; use a new experiment for a material scientific change. Append the
later sections from this raw template only after a valid result.

## Intent

Keep this compact for a simple experiment. Add detail only where it is needed
to make the comparison reproducible and interpretable.

| Field | Value |
|---|---|
| parent / comparator | `<expXXX and exact result/checkpoint; or none: first experiment>` |
| current focus / exploration item | `<FXXXX; E001 when the Internal Exploration item exists; how this evidence advances the focus. For the first decision-bearing run: FXXXX; seeds Internal Exploration after result>` |
| question / hypothesis | `<decision-bearing question and why it is worth testing>` |
| intervention | `<exact scientific change needed to distinguish this from the parent>` |
| reasoning / alternatives | `<why this test is informative; nearest alternatives or why none matter>` |
| distinguishing evidence / falsifier | `<evidence that would support, reject, or redirect the idea>` |
| training / evaluation contract | `<model/init; training data/split; evaluation cases/provenance; metric/evaluator; protected coverage>` |
| execution / evidence plan | `<commands/configs and required curves, per-case evidence, viewer, or task-specific diagnostics>` |

## Human Steering

Record only human commands that change this experiment's intent,
implementation, interpretation, evidence, or next action. Put baseline-wide
method guidance in `baseline.md`.

| Date | Human Decision | Reason | Applied Action |
|---|---|---|---|
| YYYY-MM-DD | Example: add attention visualization after adding the transformer block | The human wants evidence for the architecture change | Add it to the evidence plan and result viewer |

<!-- Append the sections below only after a valid, decision-bearing model
training or performance result exists. Routine failures, retries, timeouts, and
repairs remain in jobs/ and dashboard.md while the same experiment continues. -->

## Outcome

| Field | Value |
|---|---|
| result status / limitations | `<completed; or completed with limitations: reason>` |
| execution / artifact evidence | `<successful stages/job IDs, logs, config, checkpoint/artifact, and viewer paths>` |
| actual model / checkpoint / config | `<exact model, initialization, config, and checkpoint/artifact>` |
| actual training data / split | `<dataset and immutable split manifest>` |
| actual evaluation provenance | `<self-split / held-out / official / hidden; exact cases; evaluator/submission ID>` |
| numerical result / comparison | `<exact metrics, parent/comparator result, and expected result>` |
| result viewer / diagnostic evidence | `<result_viewer.html and supporting evidence; or no-prediction reason / human waiver>` |
| interpretation / uncertainty | `<what evidence supports, plausible alternatives, and what remains unknown>` |
| decision / next action | `<adopt / reject / refine / diagnose / defer; concrete next action>` |
| downstream updates | `<Internal Exploration IDs/judgment changes; state focus/judgment changes; TXXXX trajectory link when the decision changed; or none: reason>` |

Every experiment with predictions must run the shared visualizer and link its
initial `result_viewer.html`. Important full-baseline, high-upside, surprising,
poor, or suspicious results must follow `references/result-analysis.md`
and the applicable task-specific result-analysis skill.

For such a result, group the applicable component audit below. Every plausible
suspect or untested component must lead to a diagnostic, controlled follow-up,
or explicit reason to defer. Omit this section for a routine result only after
the required evidence is valid.

## Diagnosis

| Disposition | Components / Findings | Evidence / Follow-up |
|---|---|---|
| verified |  |  |
| suspect |  |  |
| untested / deferred |  |  |
