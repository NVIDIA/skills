# Predictions, embeddings and destinations

`Trainer.predict` selects one artifact per call. Without `output_config`, it produces local
prediction Parquet files. `num_rows` is the selected output row count; `.frame` collects all
files into pandas and should be used only for bounded results. Local files require explicit
retention to survive runtime teardown.

Inspect schema without collecting all rows:

```python
import pyarrow.parquet as parquet

for filename in predictions.files:
    print(filename, parquet.ParquetFile(filename).schema_arrow)
```

Binary/regression results have scalar output columns. Multiclass/multilabel and rankings
use long class rows. Forecasts use expanded timestamp rows. Embeddings include vector-valued
EMBEDDING columns and entity metadata; link embeddings can include ID, TYPE and NAME for
self-links. Preserve the actual schema and task rather than assuming one row per entity.

## Write predictions

Inputs: `output_connector` scoped to the confirmed destination database/schema, exact bare
`OUTPUT_TABLE`, explicit `WRITE_MODE`, and task-specific `PREDICT_OPTIONS`. WriteMode has
APPEND and OVERWRITE. Select the mode deliberately: File export OVERWRITE replaces existing data while retaining the table and grants when its schema matches, and fails on Snowflake when it does not; on BigQuery it removes the table's row access policies (below);
APPEND can duplicate outputs on retry and never changes the existing table's schema (below). Check destination
existence and access first.

```python
import warnings
import kumo.engine as kumo
from kumoapi.jobs import MetadataField, PredictionArtifactType, WriteMode

# APPEND only; JOB_TIMESTAMP alone for a query with no time window.
RUN_COLUMNS = [MetadataField.JOB_TIMESTAMP, MetadataField.ANCHOR_TIMESTAMP]
output = kumo.PredictionOutputConfig(
    connector=output_connector,
    artifact_type=PredictionArtifactType.PREDICTIONS,
    table_name=OUTPUT_TABLE, write_mode=WRITE_MODE,
    extra_fields=RUN_COLUMNS if WRITE_MODE == WriteMode.APPEND else [],
)
with warnings.catch_warnings():
    if WRITE_MODE == WriteMode.APPEND:   # no model id, no write: see below
        assert trainer.result.id, 'No model id: APPEND would write rows without BUNDLE_ID'
        warnings.filterwarnings('error', message='Predicting without a prediction manifest')
    written = trainer.predict(
        population, snapshot=snapshot, output_config=output, **PREDICT_OPTIONS,
    )
print(written.num_rows, written.files)
```

**APPEND rows say which run wrote them.** An APPEND write to a table adds `PREDICTION_ID` (this
prediction run) and `BUNDLE_ID` (the model that scored it), whatever `extra_fields` says (Source:
shipped package (`kumo.engine.trainer.output_config.PredictionOutputConfig`)). The `extra_fields`
above add `JOB_TIMESTAMP` (when the run started) and `ANCHOR_TIMESTAMP` (the time its predictions
are for), so the table's history can be ordered and filtered by run, and a retried run shows as a second
`PREDICTION_ID` and `JOB_TIMESTAMP`. `BUNDLE_ID` is the model id the run's prediction manifest
names. When there is none — `trainer.result.id` is `None` because the id could not be computed at
fit, or `predict` warns `Predicting without a prediction manifest` — the writer leaves the
`BUNDLE_ID` column out of the rows; `PREDICTION_ID` is still set and the row count still matches.
An existing table with the column gets nulls in it. A table the run creates has no `BUNDLE_ID`
column at all, so every later APPEND that names a model fails on BigQuery and Databricks, or loses
the id on Snowflake (below). The snippet above checks the model id before scoring and turns the
warning into an error, so APPEND writes neither; report it as a failed run. A query with no time
window has no anchor: pass `JOB_TIMESTAMP` alone, or the write fails after scoring.

The write never adds a column to an existing table. When an APPEND destination already exists,
read its columns before generating the artifact and check every column the run emits:
`PREDICTION_ID`, `BUNDLE_ID`, `JOB_TIMESTAMP`, and `ANCHOR_TIMESTAMP` when the query has a time
window. What a missing one does depends on the destination. Snowflake loads by column name and
drops a column the table lacks, silently: the rows land without it, and a row-count check still
passes. Databricks and BigQuery fail the write, after scoring. Either way, say so at the gate and
let the user pick: add the missing columns (`ALTER TABLE … ADD COLUMN`; earlier rows hold nulls),
a new table the first run creates, or OVERWRITE. Adding columns is a write, confirmed like any
other. Column types:

| Destination | `PREDICTION_ID`, `BUNDLE_ID` | `JOB_TIMESTAMP`, `ANCHOR_TIMESTAMP` |
|---|---|---|
| Snowflake | `VARCHAR` | `TIMESTAMP_NTZ` |
| Databricks | `STRING` | `TIMESTAMP_NTZ` |
| BigQuery | `STRING` | `TIMESTAMP` |

The timestamps take the type the destination's sink writes: on BigQuery a `DATETIME` column fails
the APPEND (Source: shipped package (`kumo_bigquery.write`, `kumo_databricks.write`)).

OVERWRITE onto an existing Snowflake table keeps the table and its grants, so the run's columns
must be exactly the table's: an OVERWRITE run writes no run columns, so a table that earlier
APPEND runs gave them fails, after scoring. Offer OVERWRITE there only when the columns match;
otherwise offer a new table, or dropping the existing one first (a write, confirmed like any
other). Databricks OVERWRITE replaces the table, columns included.

BigQuery OVERWRITE replaces the table's rows and schema with one `WRITE_TRUNCATE` job
(`kumo_bigquery.write`). It keeps column descriptions and policy tags, but BigQuery drops the
table's row access policies with it, so every reader of the table then sees every prediction row. Before offering OVERWRITE onto an existing BigQuery table, list its policies
(`bq ls --row_access_policies <project>:<dataset>.<table>`, or ask the user). When it has any, say
so at the gate and offer a new table, with the access controls it needs, instead.

The exact table name is used; no suffix is appended and its case is kept. Read back, with the
session taken in the first cell (`kumoai-reference/references/kumo_finetuning_sdk.md` §3.3), using
the quoted table name of §9.1 there, inspect schema and bounded values. For a new/replaced table compare its count
with `written.num_rows`. For APPEND compare the count delta against a recorded pre-write count
only without concurrent writers, or use a run-specific destination for unambiguous verification.
Check entity/anchor/class coverage too. A count alone does not prove semantic correctness.

## Requested embeddings

Predictions and embeddings are separate calls and destinations. For a fitted or loaded trainer,
using the same compatible snapshot and population:

```python
embedding_output = kumo.PredictionOutputConfig(
    connector=output_connector,
    artifact_type=PredictionArtifactType.EMBEDDINGS,
    table_name=EMBEDDING_TABLE, write_mode=WRITE_MODE,
)
embeddings = trainer.predict(
    population, snapshot=snapshot, output_config=embedding_output, **PREDICT_OPTIONS,
)
print(embeddings.num_rows, embeddings.files)
```

Preserve task-specific arguments, including the explicit binary threshold where the API
requires it. Do not promise embeddings for every module/task: report an unsupported selection
error without silently switching models or refitting. Link models can expose entity and item
vectors; inspect TYPE and identifiers before calling them customer/product embeddings.

## Serialization boundaries

On Snowflake, the public prediction path exports selected Parquet through Snowflake's file sink,
with schema inference and COPY. This differs from the separate in-memory DataFrame writer, which rejects
list/dict columns. Do not apply that rejection indiscriminately to prediction export, and do
not send nested cohort JSON through the scalar DataFrame path unchanged.

For vectors or other nested fields, inspect the destination's actual column type and a
bounded read-back value; inference is not a guarantee of the consumer's expected representation.
Use [stage retention](../../kumo-model-persistence/references/retention.md) to preserve the original
Parquet or JSON when a destination cannot represent the required shape. Conversion/flattening
requires an explicit schema decision. Do not claim that all artifact families fit an ordinary
table unchanged or automatically rerun prediction after an export failure.
