# Kumo SDK — Code Generation Reference

**Target: the local, in-process Kumo runtime.** Kumo executes inside the notebook's own Python
process. There is no hosted service, no project, no job registry, no API key, no Kumo UI.

---

## 0. Ground truth

The baseline API was verified against the bundle below; graph visualization was checked against source including [NVIDIA/kumo-engine#134](https://github.com/NVIDIA/kumo-engine/pull/134).

| | |
|---|---|
| Baseline bundle | `kumo_fat_bundle` `20261007+2e1eb12` (after the `kumo-engine` rename, kumo-ai/kumo#31382) |
| Graph visualization | Source build including #134 (merged at `fa46bed23fb01921955cb77bfec21e80ab7e0fc6`); unavailable in the baseline bundle and `kumo-engine` `1.0.0`, with no released build verified |
| Import name | `kumo.engine` (the `kumo-engine` package; `kumoai` / `kumoai-local` before the rename) |
| Python / torch | 3.12 / `2.9.1+cu129` |

Claims are marked Source: shipped package (`<python.module.path>`): verified against the shipped
Kumo package source in that module. They were verified on `20260918+fe89e19`, before the
rename. Between that build and `a5503fd` the public API changed only in its import name
(`kumoai` → `kumo.engine`) and three added exports
(`EmbeddingModelStore`, `GloVeVectors`, `TokenEmbeddingModel`); module paths below use the new name.
Between `a5503fd` and `2e1eb12` it gained `PredictionResult.export(output_config)` and
`PredictionOutputConfig.destinations()` (§9), `connector.sql()` / `SQLSourceTable`
(kumo-ai/kumo#31529), `TrainingTableOutputConfig` (kumo-ai/kumo#31525),
`Trainer.fit(warm_start=)` (kumo-ai/kumo#31507) and `PredictiveQuery.validate(verbose=True)`
returning a `ValidationResponse` (kumo-ai/kumo#31574), and nothing was removed.
Anything else is marked **Not verified** — surface it to the user rather than guessing.

**Published Kumo documentation is not a source.** It predates this SDK and contradicts it. Do not
cite, link, fetch, or reconcile against it, including from memory.

**There is exactly one API.** Earlier material described two interchangeable "variants" — that was
a misreading of one SDK before and after a rename. If you have seen `set_backend`, `LocalBackend`,
`create_identity` or `OutputConfig` described as a current alternative, that is the **old** SDK;
see §11 and never emit it.

Rules:
- Emit only what appears in this file. If an API is not here, do not call it.
- **Never invent a signature.**
- Ignore KumoRFM / `kumo_rfm` — different product, incompatible API.

---

## 1. The whole flow

```
configure -> connector -> Tables -> Graph -> validate -> PredictiveQuery -> validate
  -> snapshot -> training/prediction tables -> model plan -> Trainer.fit -> Trainer.predict
```

`snapshot` is threaded explicitly through `fit` and `predict`. That is the single most common
place to get the signature wrong.

Top-level names, complete (Source: shipped package (`kumo.engine.__all__`)):

```
configure  Connector  SnowflakeConnector  S3Connector  GCSConnector  BigQueryConnector
DatabricksConnector  FileConnector  SourceTable  SQLSourceTable  Identity  SnowflakeIdentity
S3Identity  GCSIdentity  BigQueryIdentity  DatabricksIdentity  Column  Table  Edge  Graph  GraphSnapshot
PredictiveQuery  TrainingTable  PredictionTable  Trainer  TrainingResult  PredictionResult
ModelOutputConfig  PredictionOutputConfig  TrainingTableOutputConfig  ModelRegistryLocation
EmbeddingModelStore  GloVeVectors  TokenEmbeddingModel  __version__
```

Anything not on that list is not `kumo.<something>`.

---

## 2. Environment

**Kumo is already present. Assume `import kumo.engine` works.** Setting up the environment is not this
skill's job and is not a topic to raise with the user. Never generate a cell that puts Kumo there.

```python
import kumo.engine as kumo

print('kumo.engine:', kumo.__version__)
```

If the import raises, the environment is not ready: report that plainly and stop.

Graph-building types `Column` and `Edge` are available from `kumo.engine`.
Other supporting imports come from `kumoapi`:

```python
from kumo.engine import Column, Edge
from kumoapi.jobs import PredictionArtifactType, WriteMode
from kumoapi.model_plan import RunMode
from kumoapi.typing import Dtype, Stype
```

**`kumo.Stype` / `kumo.Dtype` do not exist.** Always import them from `kumoapi.typing`.

Source: shipped package (`kumo.engine`, `kumoapi.typing`, `kumoapi.graph`, `kumoapi.jobs`,
`kumoapi.model_plan`)

### 2.1 `configure` — the one process-wide call

```python
kumo.configure(scratch_dir=run_dir / 'scratch')
```

- Keyword-only, and **optional**; default scratch root is `~/.cache/kumo-engine`.
- Call it **once, early** — before constructing any `Graph` or `Trainer`. Nothing re-reads it
  after a scratch path has been computed.
- Must be a local filesystem path. An `s3://`/`gs://` URI or a Snowflake stage raises `ValueError`.
- Set it when you want working data on a faster or larger disk. `graph.snapshot()` and
  `Trainer.fit`/`predict` read and write it directly.

Source: shipped package (`kumo.engine.config`)

---

## 3. Connection

The connector is built in one step, as the host's `runtime.md` says; everything after it is the
same for every connector. §3-§3.3 cover `SnowflakeConnector`; §3.4 covers every other connector.
§3.2 (reading the catalog) applies to all of them.

`SnowflakeConnector` is built by a **factory**, which constructs the connector and its identity
together from the environment. No password, key, or API credential literal belongs in a cell.
Priority: the runtime token (`from_spcs_env`), then a `connections.toml` entry
(`from_connection_toml`), then a key pair read from Snowflake secrets (§3.1).

```python
connector = kumo.SnowflakeConnector.from_spcs_env(
    warehouse=WAREHOUSE, database=SOURCE_DATABASE, schema_name=SOURCE_SCHEMA,
)
```

Source: shipped package (`kumo.engine.connector.snowflake`) — both factories are classmethods,
all arguments keyword-only:

```
from_spcs_env(*, warehouse=None, database=None, schema_name=None, role=None)
from_connection_toml(*, connection_name=None, warehouse=None, database=None,
                     schema_name=None, role=None)
```

Choose on the environment, in that order:

```python
import os
from kumo_snowflake.spcs import SPCS_TOKEN_PATH

TARGET = dict(warehouse='WH_S', database='KUMO', schema_name='"my_schema"')  # lower case: keep the quotes
if os.path.exists(SPCS_TOKEN_PATH):
    connector = kumo.SnowflakeConnector.from_spcs_env(**TARGET)
else:
    try:
        connector = kumo.SnowflakeConnector.from_connection_toml(
            connection_name='<connection-name>', **TARGET)
    except ValueError as err:
        if 'connections.toml has no connection' not in str(err):
            raise
        from snowflake.snowpark.secrets import get_generic_secret_string
        identity = kumo.SnowflakeIdentity(
            account='<account>', username='<key-pair-user>',
            private_key=get_generic_secret_string('<private-key-secret>'),
            private_key_passphrase=get_generic_secret_string('<passphrase-secret>'))
        connector = kumo.SnowflakeConnector(identity=identity, **TARGET)
```

`from_connection_toml` raises `ValueError("connections.toml has no connection ...")` when the
entry is absent; any other `ValueError` (a missing warehouse, a malformed entry) is re-raised
rather than hidden by the fallback. Source: shipped package (`kumo_snowflake.config`)

- **Always pass `warehouse`, `database`, `schema_name` explicitly.** They default to `None`, and
  the environment's values are not necessarily the ones the user wants.
- **Write a lower-case or mixed-case name with its double quotes, inside the string.** `database`
  and `schema_name` are read as declared Snowflake identifiers: an unquoted name is folded to upper
  case, so `schema_name='my_schema'` looks up `MY_SCHEMA`, and on a lower-case schema the
  connector reaches no tables (`table_names()` is empty). Pass `schema_name='"my_schema"'`. A
  name stored in upper case needs no quotes. Take the spelling from preflight and quote anything
  that is not all upper case.
  Source: shipped package (`kumo_core.sources.sql.resolve_snowflake_identifier`)
- `SPCS_TOKEN_PATH` is imported from `kumo_snowflake.spcs`; never hardcode it.
- `connector.identity` is a `SnowflakeIdentity`; `connector.id` and `identity.slug` are printable.

### 3.1 The direct constructor

```
SnowflakeConnector(*, warehouse, database, schema_name, identity=None, role=None)
```

Keyword-only. **There is no `name=` and no `account=` parameter** — passing either is a
`TypeError`. The account lives on the identity:

```
SnowflakeIdentity(*, account, username=None, password=None, private_key=None,
                  private_key_passphrase=None, host=None, token_path=None)
```

Prefer the factories. Build an identity by hand only when the user cannot use either factory, and
read every secret from `snowflake.snowpark.secrets` — never a literal.

### 3.2 Reading the catalog

```python
connector.table_names()          # list of table names
connector['transaction_events']  # -> SourceTable
connector.has_table('orders')    # -> bool
```

`SourceTable` exposes `.columns` and `.column_dict`, resolved lazily from the source. **It has no
`.head()`** — do not emit one.

**Assert `table_names()` before building tables.** It is the cheap proof the connector reaches the
data, and it fails early with a clear error instead of deep inside a snapshot. A rootless
`S3Connector` (`root_dir=None`) has no `table_names()`, so root it at the confirmed prefix.

**Use the catalog's exact spelling for table and column names.** `connector['<table>']` tries the
exact name, then its upper-cased form, so `connector['ENTITY_PROFILE']` never finds a lower-case
`entity_profile` (`SourceNotFoundError: Invalid table ENTITY_PROFILE: it has no columns`). Write
table names as `table_names()` returns them and column names as `.columns` returns them — in
`primary_key=`, `time_column=`, column lists and filters.

**Never exclude a column by a case-sensitive `!=` against a hand-typed literal.** A filter such as
`c.name != 'END_DATE'` silently keeps a lower-case `end_date` — the comparison never matches, the
column is never dropped, and nothing raises. Pass the confirmed `Excluded columns` from preflight
(`kumo-data-science/accept/preflight.md` §8) as a literal `column_names=` list built from the catalog's own spelling
— it has no hand-typed comparison to get wrong. If a filter is used instead, compare
case-insensitively: `c.name.lower() != 'end_date'.lower()`.

Source: shipped package (`kumo.engine.connector.snowflake`, `.base`, `.source_table`;
`kumo_snowflake.identity` for the lookup order)

### 3.3 The Snowpark session: take it once, first

Only where the host's `runtime.md` provides an active Snowpark session. A host without one
follows its `runtime.md` for SQL and read-back instead, and this section and the `session`
calls in §7.1 and §9.1 do not apply.

Kumo's Snowflake connector opens its own Snowpark session. Once Kumo has read or written
Snowflake, the process holds two sessions, and a later `get_active_session()` raises
`SnowparkSessionException: (1409): More than one active session is detected`.

```python
from snowflake.snowpark.context import get_active_session

session = get_active_session()   # first code cell, before any connector
```

- Call `get_active_session()` **once**, in the first code cell, before building any connector.
  Reuse that `session` for every later Snowflake call: the write-back check (§9.1) and any
  population table (§7.1).
- Never call `get_active_session()` again later in the notebook, and never open another session
  with `Session.builder`.

Source: shipped package (`kumo_snowflake.session`, which opens its session with
`Session.builder.configs(...).create()`)

### 3.4 Every other connector

Constructors and identities, all keyword-only. Source: shipped package
(`kumo.engine.connector.file`, `kumo.engine.connector.s3`, `kumo.engine.connector.gcs`,
`kumo.engine.connector.bigquery`, `kumo.engine.connector.databricks`,
`kumo.engine.connector.identity`):

```
FileConnector(*, root_dir)
S3Connector(*, root_dir=None, identity=None, region=None)
GCSConnector(*, root_dir=None, identity=None, requester_pays_project=None)
BigQueryConnector(*, project_id, dataset_id, identity=None, location='US',
                  billing_project_id=None, staging_uri=None, require_staged=False)
DatabricksConnector(*, catalog, schema_name, identity, warehouse_id=None, cluster_id=None,
                    staging_volume_path=None, require_staged=False)
DatabricksConnector.from_notebook(*, catalog, schema_name, warehouse_id=None, cluster_id=None,
                                  staging_volume_path=None, require_staged=False)

S3Identity(*, access_key_id=None, secret_access_key=None, session_token=None, role_arn=None,
           partition='aws', connection=None)
GCSIdentity(*, service_account_key=None, workload_service_account_email=None)
BigQueryIdentity(*, service_account_key=None, workload_service_account_email=None)
DatabricksIdentity(*, host=None, client_id=None, client_secret=None)
```

**Prefer ambient credentials.** Each connector resolves its provider's own credential chain when
no secret is passed, so the connect step names no secret at all:

| Connector | Ambient form | Credentials it resolves | Table names | `sql()` |
|---|---|---|---|---|
| `FileConnector` | `FileConnector(root_dir=...)` | none needed | relative paths below `root_dir`, optionally with `.csv` / `.parquet`, or a directory of parts in one format | no |
| `S3Connector` | `identity=None` | AWS chain: instance profile, `aws configure`, environment | relative paths below `root_dir`; with `root_dir=None`, full `s3://` URIs and no `table_names()` | no |
| `GCSConnector` | `identity=None` | Application Default Credentials | as S3, with `gs://` | no |
| `BigQueryConnector` | `identity=None` | Application Default Credentials | tables of `dataset_id` | yes |
| `DatabricksConnector` | `from_notebook(...)`, or `identity=DatabricksIdentity()` | the Databricks SDK's chain: a notebook's context, a CLI profile, environment | tables of `catalog.schema_name` | yes |

- The explicit identity forms carry secrets (an access key, a service-account key dict, an OAuth
  client secret). Use one only when the user asks for it, and read each secret from the
  environment's secret store at run time — never a literal, never a value asked for in chat.
- `FileConnector.root_dir` is a filesystem directory, not a URI, and must not overlap the
  scratch directory (§2.1). Absolute table paths, URIs, `..` and globs are rejected. A CSV time
  column needs `.with_column(col, dtype=Dtype.date, timestamp_format=...)`.
- `region` on `S3Connector` is optional and normally **omitted**: the ambient chain resolves the
  bucket's region where the code runs. Pass it only when the user named one, and only with
  `root_dir`. It is never a value to ask for, and never a placeholder.
- `DatabricksConnector` takes exactly one of `warehouse_id` and `cluster_id`.
- `require_staged=True` needs `staging_uri` (BigQuery) or `staging_volume_path` (Databricks).
- ADLS has no connector in this package.

Where each one writes is §9.2.

---

## 4. Tables

`Table` is **immutable**. `with_column(...)` returns a *new* table — assign or chain it, never
call it for its side effect.

```python
from kumoapi.typing import Dtype, Stype

account_profile = kumo.Table.from_source_table(
    connector['account_profile'], primary_key='account_id', time_column='created_date',
)

transaction_events = (
    kumo.Table.from_source_table(
        connector['transaction_events'], primary_key='event_id', time_column='event_date',
    )
    .with_column('account_id', stype=Stype.ID)
    .with_column('device_id', stype=Stype.ID)
)
```

Source: shipped package (`kumo.engine.table`):

```
Table.from_source_table(source_table, column_names=None, *, primary_key=None,
                        time_column=None, end_time_column=None)
Table.with_column(name, *, stype=None, dtype=None, timestamp_format=None,
                  multicategorical_delimiter=None)
Table.infer_metadata()
```

- **The source table and column list are positional**; everything else is keyword-only.
- Omit `column_names` to take every column. Pass a list to restrict.
- **Foreign keys infer as numerical.** FK-ness is a graph-level fact, so per-table inference sees
  only values. Correct every FK with `.with_column(col, stype=Stype.ID)`.
- **A TEXT time column needs `dtype=Dtype.date`** declared, or staging will not cast it.
- `infer_metadata()` takes no arguments.
- Read-back attributes live under `.definition`: `.definition.pkey`, `.definition.time_col`,
  `.definition.end_time_col`. `.table_name` is on the table itself.

> **Not verified:** `timestamp_format` and `multicategorical_delimiter` are in the signature but
> exercised nowhere.

Source: shipped package (`kumo.engine.table`)

---

## 5. Graph

```python
from kumo.engine import Edge

graph = kumo.Graph(
    tables=[account_profile, transaction_events, device_profile],
    edges=[
        Edge(src_table='transaction_events', fkey='account_id', dst_table='account_profile'),
        Edge(src_table='transaction_events', fkey='device_id', dst_table='device_profile'),
    ],
)
graph.validate()
```

- **Emit `tables` as a list.** Each table's graph name is then its source table name
  (`.table_name`), which is what edges and PQL reference.
- `tables` may also be a mapping `{alias: table}`. The alias then replaces the source name in edges
  and PQL; the connector still reads the source name. It is valid in a user's notebook — do not
  call it superseded — but emit the list form unless two tables share a source name, which the
  list form rejects with `Duplicate graph table name`.
- **`edges` are `Edge` objects** from `kumo.engine`, not dicts or tuples.
- Edges point **foreign key -> primary key**: `src_table.fkey` references `dst_table`'s primary key.
- **`graph.validate()` takes no arguments.** There is no `verbose=`.
- Emit `validate()` explicitly. It is a structural check needing only schemas — no snapshot yet,
  so it is cheap and catches edge typos before any data moves.

Surface includes `.tables`, `.edges`, `.id`, `.native_graph`, `.validate()`, `.visualize()`, `.snapshot()`.
**There is no `add_table`, `remove_table`, `link`, `unlink`, `infer_links`, `get_edge_stats`, or
`get_table_stats`.**

`Graph.visualize(path=None, *, show_cols=False, show_samples=False, layout='auto')` displays or exports the graph.
Follow `kumo-graph-visualization/SKILL.md` for generated notebook visuals, script exports, and schema inspection during model iteration.

Source: shipped package (`kumo.engine.graph`)

---

## 6. Predictive query

```python
pquery = kumo.PredictiveQuery(graph=graph, query=QUERY)
pquery.validate()
print('task type:', pquery.get_task_type())
```

- The graph is bound at construction; parsing is lazy.
- **Emit `validate()` with no arguments.** It raises on an invalid query; otherwise it returns a
  `ValidationResponse` and logs its warnings.
- **Always emit `validate()` then `get_task_type()`** before training. The task type is never
  written in the query — it is inferred from the target's shape — so print it and confirm it
  matches intent. A query that validates can still be the wrong task.
- Print `.query` back so the user can check the generated PQL.

See `kumoai-reference/references/pql_writing.md` for the query language itself.

Source: shipped package (`kumo.engine.pquery.predictive_query`)

---

## 7. Snapshot, training and prediction tables

Materialization is **explicit**. `fit` does not do it for you.

```python
snapshot = graph.snapshot()

training_table = pquery.generate_training_table(snapshot)
prediction_table = pquery.generate_prediction_table(
    snapshot, prediction_time=PREDICTION_TIME,
)
```

Source: shipped package (`kumo.engine.graph`, `kumo.engine.pquery.predictive_query`):

```
Graph.snapshot()
PredictiveQuery.generate_training_table(snapshot, plan=None)
PredictiveQuery.generate_prediction_table(snapshot, *, prediction_time=None, plan=None)
```

- `snapshot` is **positional** on both. `prediction_time` and `plan` are keyword-only on
  `generate_prediction_table`.
- `graph.snapshot()` reads every table through its connector and builds an on-disk DiskGraph index
  under the scratch directory. An unchanged valid snapshot is reused on later runs.
- `GraphSnapshot` exposes `.id`, `.index_path`, `.graph_config`.
- `TrainingTable` exposes `.train_table_path` (ordinary parquet), `.manifest_path`,
  `.required_stages`, `.generation_plan`.
- `PredictionTable` exposes `.table_path`, `.row_count`, `.timestamps`, `.required_stages`.
- **`prediction_time` must be a NAIVE `datetime` — passing a timezone-aware one raises**
  `ValueError: prediction_time must not include a timezone`. Kumo applies UTC itself
  (`kumo_core.prediction_options.prediction_timestamps`), so write `datetime(2026, 8, 1)` and
  **never** `datetime(2026, 8, 1, tzinfo=timezone.utc)`. Do not import `timezone` for this.
- It must not be earlier than the start of the data's event history (an earlier anchor raises
  `Invalid Prediction Anchor Time`; a later date is accepted), and it **cannot be computed from
  the connector** — `connector.sql(query)` returns a source table to feed `Table.from_source_table`,
  not rows. Hardcode a known in-range value and say why. Omitted, a temporal query anchors at
  the latest source time of its target; state that choice if you rely on it.
- `prediction_time` is rejected outright for a **static** query — it is temporal-only.

Source: shipped package (`kumo.engine.pquery.predictive_query`,
`kumo.engine.pquery.prediction_table`, `kumo_core.prediction_options`, `kumo_core.training.validation`)

### 7.1 A static query scores the rows whose target is NULL

In prediction mode a static query keeps only the rows whose target is NULL: Kumo trains on the
labeled rows and scores the unlabeled ones. If every row already has its target,
`generate_prediction_table` raises `RuntimeError: prediction query produced no rows`.

The scoring population confirmed at the task gate (`kumo-data-science/SKILL.md` Node 1d) decides
the path. The staging example below is Snowflake's; on any other connector the host's `runtime.md`
says how a population is staged, or that it is not covered there:

- **Rows whose target is NULL** → `pquery.generate_prediction_table(snapshot)`, no
  `prediction_time`.
- **Rows that already have a target** (for example "every order") → stage their keys as
  `ENTITY1` in the confirmed population table and pass it explicitly:

```python
def q(name):                                  # a quoted Snowflake identifier
    return '"' + name.replace('"', '""') + '"'

if f'{POP_DATABASE}.{POP_SCHEMA}.{POP_TABLE}' != POPULATION_CONFIRMED:   # approved at Node 1d
    raise RuntimeError('The population table changed after Node 1d: confirm the new one first.')
session.sql(                                  # the session from §3.3
    f'CREATE OR REPLACE TABLE {POP_DATABASE}.{POP_SCHEMA}.{POP_TABLE} AS '
    f'SELECT {q(ENTITY_KEY)} AS ENTITY1 FROM {SOURCE_DATABASE}.{SOURCE_SCHEMA}.{q(ENTITY_TABLE)}'
).collect()
population_connector = kumo.SnowflakeConnector.from_spcs_env(   # or from_connection_toml, §3
    warehouse=WAREHOUSE, database=POP_DATABASE, schema_name=POP_SCHEMA,
)
prediction_table = kumo.PredictionTable.from_source_table(
    population_connector[POP_TABLE], pq=pquery, snapshot=snapshot,
)
```

- `CREATE OR REPLACE` replaces an existing table; Node 1d names that before it is confirmed,
  and the approval covers that one name. The config cell carries it verbatim as
  `POPULATION_CONFIRMED`, and every staging recipe stops before writing when the name it would
  replace differs — an edit after confirmation needs a new confirmation, not a silent replace.
- Stage the population only where Node 1d confirmed it — never in the source schema. Write
  `POP_TABLE` in upper case, so the unquoted `CREATE` and the connector lookup agree.
- Never change the query, the entity or the target to get rows. An empty population is a fact
  about the data to report, not a query to rewrite.
- The population table must carry `ENTITY1` and, when the task samples against seed times (a
  temporal query, or a static one on a timestamped entity), a `TIMESTAMP` column; it must not
  contain a target column. The SQL above stages only `ENTITY1`, which suffices for a static query
  on an entity without a timestamp; otherwise also select the anchor time as `TIMESTAMP`.
- Rows the model trained on are scored too. Say so: those predictions are not out-of-sample.

Source: shipped package (`kumo_core.prediction.generate_prediction_table`,
`kumo.engine.pquery.prediction_table.PredictionTable.from_source_table`)

---

## 8. Model plan and training

```python
from kumoapi.model_plan import RunMode

model_plan = pquery.suggest_model_plan(snapshot, run_mode=RunMode.FAST)

# Smoke-test sizing — say so, and remove for a real run.
model_plan.training_job.num_experiments = 1
model_plan.training_job.refit_trainval = False
model_plan.training_job.refit_full = False
model_plan.optimization.max_epochs = 3

trainer = kumo.Trainer(model_plan)
training_result = trainer.fit(training_table, snapshot=snapshot)
print(training_result.metrics)
```

Source: shipped package (`kumo.engine.pquery.predictive_query`, `kumo.engine.trainer.trainer`):

```
PredictiveQuery.suggest_model_plan(snapshot, *, run_mode=RunMode.FAST, train_table_spec=None)
Trainer(model_plan)
Trainer.fit(train_table, *, snapshot, export_config=None, warm_start=None)
```

- **`suggest_model_plan` is implemented.** Prefer it over hand-building a plan: it inspects the
  materialized graph and query. `snapshot` is positional; `run_mode` is keyword-only.
- **`fit` requires `snapshot=` as a keyword.** `fit(training_table)` alone is a `TypeError`. This
  is the single easiest signature to get wrong.
- Mutate the returned plan in place for sizing. Always tell the user when numbers are smoke-test
  values rather than a real run.
- `TrainingResult` exposes `.metrics` (a `Dict[str, float]` **attribute**, not a call),
  `.artifact_paths`, `.bundle_dir`, `.published_model`.
- `export_config=ModelOutputConfig(...)` publishes the bundle during `fit`.
- Success is a blocking call returning without raising. **There is no job status enum — never
  branch on one.**

`Trainer` also exposes `.model_plan` (settable), `.training_table_generation_plan`, `.scratch_dir`,
and reloading: `Trainer.load(bundle_path, embedding_models=None)` and
`Trainer.load_from_connector(model_registry, *, path=None, location=None,
embedding_models=None)`.

Source: shipped package (`kumo.engine.trainer.trainer`)

---

## 9. Prediction and write-back

```python
from kumoapi.jobs import PredictionArtifactType, WriteMode

output_connector = kumo.SnowflakeConnector.from_spcs_env(
    warehouse=WAREHOUSE, database=OUTPUT_DATABASE, schema_name=OUTPUT_SCHEMA,
)
output_config = kumo.PredictionOutputConfig(
    connector=output_connector,
    artifact_type=PredictionArtifactType.PREDICTIONS,
    table_name=output_table,
    write_mode=WriteMode.OVERWRITE,
)

prediction_result = trainer.predict(
    prediction_table, snapshot=snapshot, output_config=output_config,
)
print('rows:', prediction_result.num_rows)
```

Source: shipped package (`kumo.engine.trainer.trainer`, `kumo.engine.trainer.output_config`):

```
Trainer.predict(prediction_table, *, snapshot, output_config=None,
                num_classes_to_return=None, num_workers=1,
                binary_classification_threshold=None)

PredictionOutputConfig(connector=, artifact_type=, table_name=, write_mode=, extra_fields=,
                       output_paths=None)
PredictionResult.export(output_config)
```

- `write_mode`: `OVERWRITE` replaces an existing table at the destination. Use the mode Node 1d
  confirmed: `OVERWRITE` for a new table or an approved replacement, `APPEND` when the user chose
  it (`kumoai-reference/references/outputs.md`).
- For Snowflake, always pass `artifact_type` and `table_name`. `output_paths` is for file and
  object-store connectors only; with a `SnowflakeConnector` it raises `ValueError`. Every
  connector's form is in §9.2.
- `extra_fields=[MetadataField.JOB_TIMESTAMP, MetadataField.ANCHOR_TIMESTAMP]` (`kumoapi.jobs`)
  adds those two columns: when the run started, and the time each prediction is anchored at.
  `ANCHOR_TIMESTAMP` exists only for a query with a time window; requesting it for one without
  raises `ValueError` after scoring. Source: shipped package (`kumo_core.train.output`,
  `kumo_core.prediction_options`).

- **`snapshot=` is required.** `output_config` is optional — omit it to keep results local.
- `predict` requires a trainer holding fitted state — from `fit` on that same `Trainer`, or from
  `Trainer.load` / `Trainer.load_from_connector`; there is no job registry to look a model up in.
- `num_classes_to_return` is the number of results returned per entity, and it is independent
  of the query's `RANK TOP K`: a `RANK TOP 10` model can return 50 results. Link prediction
  requires it whenever predictions are written, as an integer from 1 to 500
  (`kumo_core.prediction_options.MAX_LINKS_TO_PREDICT`); for more than 500 per entity, export
  embeddings instead. Multiclass and multilabel accept it optionally and fall back to the
  query's `RANK TOP K` when it has one. Regression and binary classification reject it.
- Binary classification: without `binary_classification_threshold` (and none recorded with the
  model), `predict` returns `True_PROB` and leaves `TARGET_PRED` null; it has no 0.5 default of
  its own. Always pass the one confirmed at Node 1d (§12), written or local alike.
- `binary_classification_threshold` and `num_classes_to_return` are arguments of `predict`, and
  only of `predict`. `PredictionOutputConfig` describes where to write, not how to predict: it
  takes neither, and passing one is a `TypeError`. Whatever the destination, the config carries
  only the fields in its signature above.
- `PredictionResult` exposes `.num_rows`, `.files`, `.frame`, `.selection`, `.artifact_type`.
  Prefer `.num_rows` and `.files` — `.frame` collects every row into pandas.
- `prediction_result.export(output_config)` writes an existing result to a destination. Passing
  `output_config=` to `predict` does the same write in one call; that is the form to emit.

### 9.1 The output table keeps its name

**The table is written with exactly the name you pass.** No suffix, no case change, no quoting
needed on the write. Verify by reading it back and comparing the row count. Quote the table name
in the read: `session.table` upper-cases an unquoted part, so `churn_scores` would look for
`CHURN_SCORES` and not find the table Kumo wrote. The database and schema stay unquoted; both
sides resolve those the same way.

```python
def quoted(name):
    return '"' + name.replace('"', '""') + '"'

destination = session.table([OUTPUT_DATABASE, OUTPUT_SCHEMA, quoted(output_table)])
assert destination.count() - before == prediction_result.num_rows
```

`before` is 0 for `OVERWRITE`. For `APPEND` it is the same count taken before `predict` — 0 when
the table does not exist yet, since the first APPEND creates it — so the check compares the rows
this run added. The delta holds only while nothing else writes to the table
(`kumoai-reference/references/outputs.md`).

Scope the output connector to the destination database and schema; `table_name` is a bare name.
Always read the table back — it is the only proof the write-back worked. `session` is the one
taken in the first cell (§3.3); calling `get_active_session()` here, after Kumo has written,
raises.

Source: shipped package (`kumo.engine.trainer`)

### 9.2 Destinations by connector

The output connector may differ from the source connector. Source: shipped package
(`kumo.engine.trainer.output_config`, `kumo.engine.connector.file`, `kumo_core.object_store_sink`,
`kumo_databricks.sink`, `kumo_databricks.output`):

| Connector | `PredictionOutputConfig` form | Rules |
|---|---|---|
| `SnowflakeConnector` | `artifact_type=` + `table_name=` (bare name) | connector scoped to the destination database and schema; §9.1 |
| `FileConnector`, `S3Connector`, `GCSConnector` | `output_paths={'predictions': ..., 'embeddings': ...}` — the keys select the artifacts — or `artifact_type=` + `table_name=` | do not combine the two forms; a path ending in `/` is a directory and must be empty; otherwise one file, written in the format its `.parquet` or `.csv` suffix names, and `.parquet` is added when it has neither — by all three connectors; paths are relative to a rooted connector, full URIs for a rootless one |
| `BigQueryConnector` | `artifact_type=` + `table_name=` (unqualified table ID in its dataset) | predictions and embeddings only — not a model `export_config` and not a `Trainer.load_from_connector` source |
| `DatabricksConnector` | `artifact_type=` + `table_name=`: a bare name, or `'<schema>__kumo__<table>'` | a bare name lands in the connector's schema; a qualified name must name the connector's own schema, or it is refused; writes stage through the volume `<catalog>.<schema>._kumo_staging`, created on first use |

`output_paths` with a non-file connector raises `ValueError`. Read every destination back after
the write, as the host's `runtime.md` says (§9.1 for Snowflake).

---

## 10. Complete verified example

On Snowflake, in a host whose `runtime.md` provides an active Snowpark session (§3.3). On any
other host — Snowflake reached from outside it included — drop the `get_active_session()` line and
replace steps 1, 7 and 8 with that host `runtime.md`'s connect step, the §9.2 destination and its
SQL read-back; nothing else changes.

Self-contained: `OUTPUT_DATABASE`, `OUTPUT_SCHEMA` and `OUTPUT_TABLE` are placeholders for the
confirmed output destination (§12).

```python
from datetime import datetime
from pathlib import Path

import kumo.engine as kumo
from kumo.engine import Edge
from kumoapi.jobs import PredictionArtifactType, WriteMode
from kumoapi.model_plan import RunMode
from kumoapi.typing import Stype
from snowflake.snowpark.context import get_active_session

OUTPUT_DATABASE, OUTPUT_SCHEMA, OUTPUT_TABLE = '<db>', '<schema>', '<table>'  # confirmed destination

session = get_active_session()   # §3.3: only where the host provides the session

# 1. Scratch directory, then connect (inside SPCS; elsewhere use the §3 order). No credentials in the notebook.
kumo.configure(scratch_dir=Path('/tmp/kumo-run') / 'scratch')
connector = kumo.SnowflakeConnector.from_spcs_env(
    warehouse='WH_S', database='SYNTHETIC_DATASETS', schema_name='"my_schema"',
)
assert connector.table_names(), 'Connector reached no tables.'

# 2. Tables — correct every foreign key to Stype.ID.
account_profile = kumo.Table.from_source_table(
    connector['account_profile'], primary_key='account_id', time_column='created_date',
)
transaction_events = (
    kumo.Table.from_source_table(
        connector['transaction_events'], primary_key='event_id', time_column='event_date',
    )
    .with_column('account_id', stype=Stype.ID)
    .with_column('location_id', stype=Stype.ID)
)
location_profile = kumo.Table.from_source_table(
    connector['location_profile'], primary_key='location_id',
)

# 3. Graph — list of tables, Edge objects, explicit validate.
graph = kumo.Graph(
    tables=[account_profile, transaction_events, location_profile],
    edges=[
        Edge(src_table='transaction_events', fkey='account_id', dst_table='account_profile'),
        Edge(src_table='transaction_events', fkey='location_id', dst_table='location_profile'),
    ],
)
graph.validate()

# 4. Query — validate, then confirm the inferred task type.
pquery = kumo.PredictiveQuery(
    graph=graph,
    query=('PREDICT LIST_DISTINCT(transaction_events.location_id, 0, 30, days) '
           'RANK TOP 10 FOR EACH account_profile.account_id'),
)
pquery.validate()
print('task type:', pquery.get_task_type())

# 5. Materialize. PREDICTION_TIME is NAIVE — a tz-aware datetime raises ValueError.
PREDICTION_TIME = datetime(2026, 8, 1)
snapshot = graph.snapshot()
training_table = pquery.generate_training_table(snapshot)
prediction_table = pquery.generate_prediction_table(
    snapshot, prediction_time=PREDICTION_TIME,       # NAIVE datetime, inside the event history
)

# 6. Train — snapshot= is required.
model_plan = pquery.suggest_model_plan(snapshot, run_mode=RunMode.FAST)
model_plan.optimization.max_epochs = 3               # smoke-test sizing
trainer = kumo.Trainer(model_plan)
training_result = trainer.fit(training_table, snapshot=snapshot)
print(training_result.metrics)

# 7. Predict and write back — snapshot= is required here too.
output_connector = kumo.SnowflakeConnector.from_spcs_env(
    warehouse='WH_S', database=OUTPUT_DATABASE, schema_name=OUTPUT_SCHEMA,
)
prediction_result = trainer.predict(
    prediction_table,
    snapshot=snapshot,
    output_config=kumo.PredictionOutputConfig(
        connector=output_connector,
        artifact_type=PredictionArtifactType.PREDICTIONS,
        table_name=OUTPUT_TABLE,
        write_mode=WriteMode.OVERWRITE,
    ),
)
print('rows:', prediction_result.num_rows)

# 8. Verify the write-back. The table name is quoted (§9.1).
destination = session.table([OUTPUT_DATABASE, OUTPUT_SCHEMA, '"' + OUTPUT_TABLE.replace('"', '""') + '"'])
assert destination.count() == prediction_result.num_rows
```

Source: shipped package (`kumo.engine`, `kumoapi`, `kumo_snowflake.spcs`)

---

## 11. Never emit

### 11.1 The old SDK — recognise and refuse

These were the **previous** Kumo SDK. They are gone from the current package; each one is an
`AttributeError` or `TypeError`. If a user's existing notebook contains them, say plainly that it
targets a superseded build and needs rewriting against §10 — do not copy its shape.

| Old | Now |
|---|---|
| `kumo.set_backend(...)`, `kumo.LocalBackend(...)`, `kumo.get_backend()` | nothing — there is no backend |
| `kumo.Storage(...)`, `kumo.FilesystemArtifactLocation(...)` | `kumo.configure(scratch_dir=...)` (§2.1) |
| `kumo.create_identity(CreateIdentityArgs(...))`, `kumo.test_identity_credentials(...)` | `SnowflakeConnector.from_spcs_env()`, else `from_connection_toml()`, else `SnowflakeIdentity` from secrets (§3) |
| `SnowflakeConnector(name=..., account=..., credentials={...})` | keyword-only ctor, no `name`/`account` (§3.1) |
| `kumo.Table(source_table=..., primary_key=...)` | `Table.from_source_table(...)` (§4) |
| `table['COL'].stype = kumo.Stype.ID` | `.with_column('COL', stype=Stype.ID)` (§4) |
| `kumo.Stype`, `kumo.Dtype` | `from kumoapi.typing import Dtype, Stype` |
| `kumo.Graph(..., edges=[{...}])` — edges as dicts | `Edge` objects (§5). A `tables={alias: t}` mapping is still valid |
| `trainer.fit(graph, pquery=...)` | `fit(train_table, snapshot=snapshot)` (§8) |
| `OutputConfig` from `kumoai.artifact_export` | `kumo.PredictionOutputConfig` (§9) |
| `.metrics()` as a call, `.job_id`, `.summary()` | `.metrics` attribute, `.num_rows` (§8, §9) |
| output table renamed to `"<name>_predictions"` | the name is kept verbatim (§9.1) |

### 11.2 Hosted Kumo — out of scope entirely

- `kumo.init(...)` in any form (`url=`, `api_key=`, `snowflake_credentials=`,
  `snowflake_application=`), API keys, `KUMO_API_KEY` / `KUMO_API_ENDPOINT`, Kumo Studio.
- `training_job.tracking_url` or any Kumo UI reference.
- `kumo.Session(...)`, `kumo.ComputeType`, RBAC groups, `project_id`.
- The Snowflake Native App and Marketplace deployment.
- `kumoai.codegen`, `DistillationTrainer`, online serving.

### 11.3 Also never

- **Any environment-setup step.** Kumo is already present (§2); begin at `import kumo.engine`.
- Branching on a Kumo job status string or enum — Kumo has none. A Snowflake ML Job's status
  is Snowflake's, read from `snowflake.ml.jobs`, and only the host's job file uses it.
- `connector['X'].head(...)` — it does not exist. `connector.sql(query)` exists but returns a
  custom-SQL source table, never rows; do not use it to read data (Source: shipped package
  (`kumo.engine.connector.source_table`)).
- **`import kumoai`.** That is the hosted SDK's name, and this SDK's name before the rename. Where
  it resolves at all, it is the wrong package or a pre-rename build; import `kumo.engine`.

---

## 12. Parameters to inject from the validated request

| Code location | Source |
|---|---|
| The connect step | The host's `runtime.md`, from the confirmed `source_scope` (§3, §3.4) |
| `TARGET = dict(warehouse=, database=, schema_name=)` (Snowflake) | Data location from the intake contract; a lower-case name keeps its double quotes (§3) |
| `from_connection_toml(connection_name=)` (Snowflake) | The user's `[connections.<name>]` entry; ask |
| `SnowflakeIdentity(account=, username=)`, secret names (Snowflake) | Account identifier, key-pair user, and the names of the attached key and passphrase secrets; ask only when neither factory applies, never for a secret's value |
| `configure(scratch_dir=)` | A local path; `/tmp/<run-id>` is fine |
| `SnowflakeConnector(role=)` (Snowflake) | Role from the intake contract |
| `connector['TABLE']` names, column names | Exactly as `table_names()` / `.columns` return them (§3.2) |
| Column selection, `stype` / `dtype` | Data structure from the intake contract |
| `Edge(src_table=, fkey=, dst_table=)` | Relationships from the intake contract |
| PQL `query=` | Generated per `kumoai-reference/references/pql_writing.md` |
| `prediction_time=` | A known in-range **naive** `datetime`; **cannot be computed** |
| `predict(num_classes_to_return=N)` | Results per entity the user asked for; defaults to `RANK TOP K` but may differ. Link prediction writing predictions requires it, 1–500; no cap otherwise (§9) |
| `predict(binary_classification_threshold=)` | Always passed for binary classification, written or local; never on `PredictionOutputConfig`. The threshold restated at Node 1d: the user's, else `0.5 [default]` (`kumo-data-science/accept/intake_contract.md`) |
| Population table (static task, rows that already have a target) | Confirmed at Node 1d (§7.1) |
| `PredictionOutputConfig(...)` + output connector | Output destination in its connector's form (§9.2); report it back unchanged — `DB.SCHEMA.TABLE`, or the path |

---

## 13. Pre-return checklist

- [ ] Generated code begins at `import kumo.engine as kumo`; `import kumoai` appears nowhere
      (§11.3).
- [ ] `kumo.init(...)` appears nowhere; no API keys; no `tracking_url`.
- [ ] No call from §11.1 appears anywhere.
- [ ] `Stype` / `Dtype` imported from `kumoapi.typing`; `Column` / `Edge` from `kumo.engine`;
      `PredictionArtifactType` / `WriteMode` from `kumoapi.jobs`; `RunMode` from
      `kumoapi.model_plan`.
- [ ] The connect step is the host `runtime.md`'s, with no credential literal (§3, §3.4).
- [ ] Snowflake: the factory the host's `runtime.md` names (§3), with explicit `warehouse`,
      `database`, `schema_name`; `SPCS_TOKEN_PATH` imported, never hardcoded.
- [ ] Snowflake: lower-case or mixed-case `database` / `schema_name` passed with their double
      quotes (§3).
- [ ] Table and column names in the catalog's exact spelling (§3.2).
- [ ] Snowflake, on a host that provides a Snowpark session: `session = get_active_session()`
      once, in the first code cell; never again, no `Session.builder` (§3.3). Elsewhere, the
      host `runtime.md`'s connection for SQL and read-back.
- [ ] `connector.table_names()` asserted before building tables.
- [ ] Every foreign key corrected with `.with_column(col, stype=Stype.ID)`.
- [ ] `with_column(...)` results assigned or chained — it returns a new Table.
- [ ] `Graph(tables=[...], edges=[Edge(...)])` — list form; `graph.validate()` emitted.
- [ ] `pquery.validate()` and `get_task_type()` run before training, both called with no arguments.
- [ ] `graph.snapshot()` before generating any table.
- [ ] **`trainer.fit(training_table, snapshot=snapshot)`** — `snapshot=` present.
- [ ] **`trainer.predict(prediction_table, snapshot=snapshot, ...)`** — `snapshot=` present.
- [ ] `binary_classification_threshold` / `num_classes_to_return` on `predict`, never on
      `PredictionOutputConfig` (§9).
- [ ] `prediction_time` hardcoded and **naive** — no `tzinfo`, no `timezone.utc` — with the reason stated.
- [ ] Static query: generated population only for rows whose target is NULL; otherwise the
      population table confirmed at Node 1d (§7.1).
- [ ] No code cell writes anywhere unless a destination was confirmed; an example of how to add
      one later is markdown text, not a code cell.
- [ ] Smoke-test sizing called out as such.
- [ ] `write_mode` and any population replacement as Node 1d confirmed them (§9, §7.1); the
      staging cell stops when the population name differs from `POPULATION_CONFIRMED`.
- [ ] Destination in its connector's form (§9.2), reported with its exact name — no
      `_predictions` suffix.
- [ ] Written destination read back and row count compared to `prediction_result.num_rows` —
      for APPEND, the count added since before `predict` (§9.1).
- [ ] Success judged by blocking calls returning without raising — not by a status value.

---

## 14. Known gaps

**Not verified:**

- `Table`'s `timestamp_format` and `multicategorical_delimiter` — in the signature, exercised
  nowhere.
- `train_table_spec=` on `suggest_model_plan`; `plan=` on `generate_training_table` /
  `generate_prediction_table`.
- `export_config=` on `fit`, and `ModelOutputConfig`'s `path` vs `location`.
- `Trainer.load` / `load_from_connector` round-trips — implemented in the package, exercised
  nowhere. `load_from_connector` needs a connector whose sink implements `ModelFetchingSink`
  (otherwise `TypeError`); `FileConnector` provides `fetch_model`, other sinks are unverified.
- `num_workers` above 1.
- `PredictionResult.export` against Snowflake — in the package, exercised nowhere.
- Whether a snapshot can be reused across graphs or persisted between processes.
- The written column schema for a ranking task.
- `FileConnector`, `S3Connector`, `GCSConnector`, `BigQueryConnector`, `DatabricksConnector`:
  §3.4 and §9.2 are read from the package source; none has been run end to end with this skill.

**Known absent** — do not reach for these: `SourceTable.head(...)`,
`graph.infer_links()`, `graph.get_edge_stats()`, `graph.get_table_stats()`, `table.validate()`,
`table.metadata`, `table.get_stats()`, `graph.validate(verbose=)`, any job-status enum.
