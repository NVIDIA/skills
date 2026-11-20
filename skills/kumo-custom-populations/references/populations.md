# Supplied Snowflake population contract

The constructors take a `SourceTable`, not a pandas `data_df` and not a table name string.
Use a Snowflake connector scoped to the supplied table, which may differ from the graph's
source schema. The query `pq` and `snapshot` describe the already confirmed graph/task.

## Columns and task shapes

| Field | Contract |
| --- | --- |
| `ENTITY1` | Non-null entity key, with numeric/string type compatible with the graph entity PK |
| `TARGET1` | Required for training; scalar for binary, regression and multiclass. Multilabel uses a scalar string encoded with the target's multicategorical delimiter |
| `TIMESTAMP` | Required temporal anchor. For a static node task it is required, training or prediction, when the entity table has a time column, and omitted when it has none. For static link prediction, training never needs it; prediction needs it when the entity table has a time column |
| Named weight | Optional training-only finite nonnegative scalar for scalar targets; at least one positive weight is required |
| Named split | Optional training-only values 0=train, 1=validation, 2=test (numeric values or digit strings) |

Training link targets require a typed list of RHS IDs, not a scalar string or arbitrary JSON.
Link weights must be lists matching target lengths; temporal link tasks permit signed weights,
with at least one positive weight. Forecast rows require a scalar `TARGET1` plus precomputed `future_labels` for training,
`autoregression_labels` when lags are enabled, and `year_over_year_labels` when enabled.
The constructors preserve these sequences; they do not derive them from scalar anchors.

Snowflake ARRAY/VARIANT source columns are exposed as strings by this connector. Do not
promise that an ordinary ARRAY label or forecast-sequence column supplies the typed lists
these paths require. Use generated link/forecast populations for those shapes in this
Snowflake workflow, or report the supplied-shape limitation. Do not silently replace an
explicit exact-population request with generated rows. Scalar supplied populations and
entity/anchor prediction populations remain supported for tasks without sequence requirements.
Exact forecast prediction still needs enabled lag and year-over-year sequences.

A prediction population must not contain numbered target columns (`TARGET1`, `TARGET2`, etc.).
Keep evaluation labels separately rather than leaking them into the scoring population.
If the supplied table carries a target column, say that the scoring population must not carry it
and that evaluation labels stay separate, joined to the scores afterwards. Never pass the table as
is. Propose, for the user's approval, a read-only `connector.sql(...)` source (Snowflake,
Databricks and BigQuery only) that keeps only the entity column, the `TIMESTAMP` where the
task needs one (a temporal task, or a static task, node or link, on an entity table with a time
column), and any sequence columns the task requires
(`autoregression_labels`, `year_over_year_labels`), or ask for a table without the target. A temporal prediction population with no
`TIMESTAMP` per row cannot be built from the table as it stands: ask for the anchors, and do not
substitute a generated population with one `prediction_time`.
No public column-renaming map exists on these constructors. Fields need their contract names.

## Train with supplied labels, weights and manual splits

The following consumes an existing scalar-label table `CURATED_LABELS` containing `ENTITY1`,
`TARGET1`, the applicable `TIMESTAMP`, `WEIGHT` and `SPLIT`. Select the source only after its
semantics and split design are confirmed.

```python
import kumo.engine as kumo
from kumoapi.train import TrainingTableSpec
from kumoapi.model_plan import RunMode

spec = TrainingTableSpec(weight_col="WEIGHT", split_col="SPLIT")
training_table = kumo.TrainingTable.from_source_table(
    population_connector["CURATED_LABELS"], spec, pq=pq, snapshot=snapshot,
)
model_plan = pq.suggest_model_plan(snapshot, run_mode=RunMode.FAST, train_table_spec=spec)
trainer = kumo.Trainer(model_plan)
training_result = trainer.fit(training_table, snapshot=snapshot)
```

Use `TrainingTableSpec()` when no weights or manual splits are supplied. Without `split_col`,
normal generation-plan split inference applies; a public `TrainingTableGenerationPlan` may
be passed as `plan=`. Do not assume every task uses the same random split. A supplied split
column overrides split assignment with 0/1/2. Inspect required nonempty splits for the
requested evaluation. Generated training tables are a distinct path and do not accept a
caller-supplied weight column by passing `TrainingTableSpec` to generation.

## String or other non-numeric splits

`split_col` reads only 0 (train), 1 (validation), 2 (test), as numbers or digit strings. A column
holding labels such as `'train'` / `'val'` / `'test'` is accepted as a string column if passed
as is, but no row matches a split. Ingestion skips every row it cannot read, so the build then
fails with `custom-TT source kept no rows`. When only some labels are unreadable, those rows are
skipped: the build succeeds on fewer rows and logs a warning with the skipped count per reason
(also recorded in `row_failures`). So report the split column as the offending field, with its
labels and counts (from the host's SQL tool), and ask: a prepared table holding 0/1/2, or the
user's explicit approval of a mapping you show. Only after that approval, map in a read-only
query; the result is a source table, so nothing is written:

```sql
SELECT SPLIT_LABEL, COUNT(*) AS N FROM CURATED_LABELS GROUP BY SPLIT_LABEL;
```

```python
labeled = population_connector.sql(
    "SELECT ENTITY1, TARGET1, TIMESTAMP, WEIGHT, "
    "CASE SPLIT_LABEL WHEN 'train' THEN 0 WHEN 'val' THEN 1 WHEN 'test' THEN 2 END AS SPLIT "
    "FROM CURATED_LABELS"
)
training_table = kumo.TrainingTable.from_source_table(
    labeled, TrainingTableSpec(weight_col="WEIGHT", split_col="SPLIT"), pq=pq, snapshot=snapshot,
)
```

Any value in that result that matches no branch (including NULL) becomes a NULL `SPLIT` and the
row is skipped: ask the user how to map it before building the training table, and never leave it
out unreported. List the labels you mapped with their counts. The mapping stays in the read-only
`connector.sql(...)` source: write nothing back, and create no mapped copy or derived table.

## Derived or normalized weights

Weights follow the same rule as splits. Asked to derive, scale or normalize weights (for example
positives x10, or `WEIGHT / mean(WEIGHT)`), report the request, then show the exact formula, the
columns it reads and the resulting weight distribution (min, mean, max, and counts per distinct
weight or per label) from a read-only query, and stop for the user's explicit approval. An
approval given before the formula and distribution were shown is not approval. Unapproved, report
and ask; do not transform. Only after approval, apply it in a read-only `connector.sql(...)`
source, as for approved split mappings above; keep the table's `SPLIT` (or the generated
training table's, if exported), and write nothing back: no UPDATE, no CREATE TABLE AS, no
derived table. The result must still be finite and nonnegative, except the signed weights
temporal link prediction permits above, with at least one positive weight.

## Export a generated training table

`TrainingTable.export` writes the rows of a table Kumo generated, with a `SPLIT` column
(0 train, 1 validation, 2 test), so people can review or edit them and fit on the edited copy:

```python
training_table.export(kumo.TrainingTableOutputConfig(
    connector=population_connector, table_name="REVIEW_TABLE",
))
```

`training_table` is the table this session already generated. Export is not training: do not call
`fit`, regenerate the table or change the plan unless asked. If the session has no generated
table, say so and ask before generating one. `table_name` is the destination. `write_mode`
(`kumoapi.jobs.WriteMode`) defaults to `OVERWRITE`; replacing an existing table needs the user's
explicit approval. `APPEND` is warehouse-only, and only when they ask for it. A file connector
takes a directory and supports only `OVERWRITE`: to keep an existing export, write to a new
directory. Report the destination the call returns.

To refit on an edited copy, read it back with `TrainingTable.from_source_table` and
`TrainingTableSpec(weight_col=..., split_col="SPLIT")` as above. Keep the exported `SPLIT` values.
Call `suggest_model_plan` with that same spec, because only the weighted spec changes the plan.

## Score exact rows

`SCORING_POPULATION` contains entity IDs and row-specific timestamps for a temporal task,
without target columns. It can be used with either a fitted or loaded trainer:

```python
population = kumo.PredictionTable.from_source_table(
    population_connector["SCORING_POPULATION"], pq=pq, snapshot=snapshot,
    plan=trainer.training_table_generation_plan,
)
# Binary example; other task arguments follow kumoai-reference/references/task_journeys.md.
scores = trainer.predict(
    population, snapshot=snapshot, binary_classification_threshold=THRESHOLD,
)
print(population.row_count, scores.num_rows)
```

Do not assume output rows equal population rows: class, ranking and forecast outputs expand
rows. Preserve every requested anchor; do not replace row anchors with one global time.

## Actionable validation

Empty sources, missing required columns, incompatible key types and list/scalar target mismatch
are errors. Row-level problems are not: a null entity or anchor, a nonfinite or disallowed
negative weight, a null split, or a split value outside 0/1/2 skips that row. The build logs a
warning with the skipped count per reason, records it in `row_failures`, and continues on the
remaining rows; if no row remains it fails with `custom-TT source kept no rows`. Check these
values before building, and report the offending field and required representation before
proposing a correction.
Verify that non-null IDs actually exist in the intended graph when exact coverage matters;
schema validation alone does not guarantee membership or prediction coverage.

Population construction rejects a snapshot from another graph. Later fit/predict receives a
snapshot separately, so retain explicit lineage rather than assuming the population object
pins all graph data. A loaded trainer checks the target/entity query contract; do not bypass a
mismatch by rewriting the target. Inspect output key/anchor coverage as well as total rows.
