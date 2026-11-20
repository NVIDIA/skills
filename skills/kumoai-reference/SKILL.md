---
name: kumoai-reference
description: "Reference for translating a validated ML request into KumoAI graph configuration, PQL (predictive query language), and Kumo Predict SDK code. Normally loaded by the kumo-data-science skill after its intake checks pass, rather than invoked directly. Triggers: kumo graph, PQL, predictive query, kumo.Table, kumo.Graph, kumo.Trainer, kumo sdk."
---

# KumoAI Reference Skill

**Load this skill only after a request has passed all checks in `kumo-data-science/SKILL.md`.**

This skill provides reference docs for translating a validated, structured ML request into KumoAI graph configuration, PQL, and KumoSDK code.

---

## What's in this skill

| Document | Purpose |
|---|---|
| `kumoai-reference/references/graph_construction.md` | How to map tables → Kumo nodes and edges |
| `kumoai-reference/references/pql_writing.md` | Writing Prediction Query Language for the validated task |
| `kumoai-reference/references/kumo_finetuning_sdk.md` | Kumo Predict SDK API and version-matched code generation |
| `kumoai-reference/references/task_journeys.md` | Per-task flow: options for `predict`, forecasting's generation plan, embeddings |
| `kumoai-reference/references/outputs.md` | Write modes, APPEND columns, embeddings, read-back |
| `kumoai-reference/references/troubleshooting_signals.md` | Known errors at graph/PQL generation time |

---

## Provenance of every claim

**The published Kumo documentation is out of date for this version of the local SDK and is not a
source here.** Do not cite it, link to it, fetch it, or reconcile against it. If a claim in these
files has no `Source:` line, treat it as unverified and say so rather than filling the gap from
the public docs or from memory.

Every claim that is grounded carries:

- ``Source: shipped package (`<python.module.path>`)`` — verified against the source of the
  shipped Kumo package, at the named module. This is the only authoritative kind of source.

Anything else is marked inline as **Not verified**. When you hit one, ask the user rather than
guessing.

## Before anything: one API, one package

There is a single Kumo SDK, verified against the shipped package named in `kumoai-reference/references/kumo_finetuning_sdk.md` §0.
Older material describing two interchangeable "variants" was a misreading of one SDK before and
after a rename. If a user's code contains `set_backend`, `LocalBackend`, `create_identity`,
`OutputConfig`, `kumo.Stype`, or edges written as dicts, it targets a **superseded**
build: say so and rewrite it rather than copying its shape. See `kumoai-reference/references/kumo_finetuning_sdk.md` §11.1.

Connection is one step, written as the host's `runtime.md` says, from the connector facts in
`kumoai-reference/references/kumo_finetuning_sdk.md` §3 (Snowflake) and §3.4 (every other
connector). No credential literal belongs in code.

## Execution flow

Given the validated structured payload from `kumo-data-science/SKILL.md`:

### Step 1 — Graph construction
Reference `kumoai-reference/references/graph_construction.md`. Map each table in the payload:
- Every source table → a `kumo.Table` with typed columns, a primary key where one was confirmed, and a time column where one exists
- Link the tables into a `kumo.Graph` by declaring edges between matching key columns
- Specify join keys, timestamp columns, and any column-level annotations.

### Step 2 — PQL query
Reference `kumoai-reference/references/pql_writing.md`. Translate the prediction target into a PQL `predict` statement:
- Entity to predict on
- Label definition
- Prediction horizon
- Feature tables to include

### Step 3 — SDK code generation
Reference `kumoai-reference/references/kumo_finetuning_sdk.md`. Generate:
- `kumo.Graph` definition
- `kumo.PredictiveQuery` and `kumo.Trainer` configuration
- `graph.snapshot()`, then `generate_training_table` / `generate_prediction_table`
- `Trainer.fit(training_table, snapshot=snapshot)` and `Trainer.predict(prediction_table, snapshot=snapshot, output_config=...)` — **`snapshot=` is required on both**
- `kumo.PredictionOutputConfig(connector=, artifact_type=, table_name=, write_mode=)` built from the output destination in the validated payload

Then follow the task's section of `kumoai-reference/references/task_journeys.md` (forecasting sets
`forecast_length` in a generation plan, not the model plan), and
`kumoai-reference/references/outputs.md` for the write mode and requested embeddings, which are a
separate `predict` call.

Emit only signatures that appear in `kumoai-reference/references/kumo_finetuning_sdk.md`. Do not infer or invent them.

### Step 4 — Sanity check
Before returning code:
- Confirm that the selected column lists enforce every exclusion confirmed at the graph gate.
  The target column always stays selected — the query references it by name. Re-check that no
  feature is determined after its table's row time, and that no non-target feature is derived
  from the prediction target. Do not return code that omits `column_names` for a table that has
  exclusions.
- If the active host adapter restricts a semantic type, confirm the generated code includes the
  adapter's runtime type guard before `graph.snapshot()`.
- Confirm that the output table in the generated code matches the output location from the original request.
- Check against known error patterns in `kumoai-reference/references/troubleshooting_signals.md`.

---

## Scope boundary

This skill generates code. It does not:
- Run jobs or check job status
- Interpret evaluation metrics or model outputs
- Repair runtime errors — recognise known signals with `kumoai-reference/references/troubleshooting_signals.md`, report them, and stop
