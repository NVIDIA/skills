# KumoAI Graph Construction

Reference for translating a confirmed relational schema into Kumo Predict SDK objects.

Grounded **only** in the shipped Kumo package source. The published Kumo documentation is
stale for this version of the local SDK — do not cite it, fetch it, or reconcile against it.
Where a claim is not verified against the shipped package it is flagged. Do not invent signatures.

Import convention: `import kumo.engine as kumo`.

There is no `KumoFineTuning`, `kumo.NodeType`, or `kumo.EdgeType` class. Do not emit them.
Do not emit `kumo.init` — hosted-edition only, out of scope.

---

## One API

There is a single Kumo SDK. Everything below targets the shipped package documented in
`kumoai-reference/references/kumo_finetuning_sdk.md` §0. Older material describing a second "variant" with `set_backend`,
`LocalBackend`, `create_identity`, `kumo.Stype` or dict-form edges refers to a
**superseded** build — recognise it, refuse it, and rewrite. See `kumoai-reference/references/kumo_finetuning_sdk.md` §11.1.

Setup, auth, training and prediction are **not** covered here — see `kumoai-reference/references/kumo_finetuning_sdk.md`.
Kumo is already present in the environment.

---

## Concept

A Kumo `Graph` is a connected set of `Table` objects. Tables are linked by
primary-key / foreign-key `Edge`s. There is no separate node/edge type system:
**every table is a table**, and relationships are declared as PK/FK edges.

Layers:
1. `Connector` → `SourceTable` (a view of data behind the connector; raw, untyped).
2. `Table` → adds metadata: selected columns, `dtype`, `stype`, `primary_key`,
   `time_column`, `end_time_column`.
3. `Graph` → a list of tables (or an `{alias: table}` mapping) plus a list of `Edge` objects.

Tables have no names of their own. A table is fully specified by its configuration in code;
identical configuration yields the same `Graph.id` (a content fingerprint of the definition and
table sources), and `with_column(...)` returns a new immutable `Table` rather than editing in place.

Source: shipped package (`kumo.engine.graph`, `kumo.engine.table`)

Tables have no names of their own beyond their source table name. `Graph(tables=[...])` has no
keys: edges and PQL reference the **source table names** — `Edge(src_table='Ratings', ...)` and
`FOR EACH Movies.MovieID` for `connector['Movies']`.

---

## Step 1: connect

**Setup and authentication are NOT documented here.** The host's `runtime.md` says how the
connector is built; the SDK facts are in `kumoai-reference/references/kumo_finetuning_sdk.md` §3
(Snowflake) and §3.4 (`FileConnector`, `S3Connector`, `GCSConnector`, `BigQueryConnector`,
`DatabricksConnector`). There is no backend and no identity registry.

For Snowflake, build the connector from a factory:

```python
connector = kumo.SnowflakeConnector.from_spcs_env(
    warehouse=..., database=..., schema_name=...)          # inside SPCS
connector = kumo.SnowflakeConnector.from_connection_toml(
    connection_name=..., warehouse=..., database=..., schema_name=...)   # outside
```

`database` and `schema_name` are declared Snowflake identifiers: a lower-case name keeps its double
quotes inside the string (`schema_name='"my_schema"'`), or it is folded to upper case and the
connector reaches no tables. Take the Snowpark session once, before any connector. Both rules:
`kumoai-reference/references/kumo_finetuning_sdk.md` §3 and §3.3.

Never emit `kumo.init(...)` — hosted editions only, out of scope. Never construct
`SnowflakeConnector` with `name=`, `account=` or `credentials=`; none is a parameter.

Everything after this step — tables, graph, query, training, prediction — follows the same
workflow for every connector. Two things still differ by connector: how a table is named
(an identifier, or a path below a file connector's root without its suffix) and `connector.sql()`,
which only Snowflake, Databricks and BigQuery run (`kumo_finetuning_sdk.md` §3.4).

### Accessing source tables

```python
print(connector.table_names())
orders_src = connector['orders']              # -> SourceTable
print(len(orders_src.columns))
for col in orders_src.columns:
    print(col.name, col.dtype, col.stype or '- (needs column statistics)')
```

`SourceTable` resolves its schema lazily and caches it on first access to `.columns` /
`.column_dict`. **There is no `.head()`** — do not emit one.

---

## Step 2: define tables

### Imports

```python
import kumo.engine as kumo
from kumoapi.typing import Dtype, Stype     # NOT kumo.Stype / kumo.Dtype
from kumo.engine import Edge
```

### Defining tables

The column selection is a **positional** second argument. `primary_key` / `time_column` are
keywords.

```python
movies = kumo.Table.from_source_table(
    connector['Movies'],
    ['MovieID', 'YearOfRelease'],        # positional column list
    primary_key='MovieID',
)
users = kumo.Table.from_source_table(connector['Users'], primary_key='userID')
```

`infer_metadata()` **takes no arguments**. It re-resolves using whatever column selection,
primary key and time column the table already carries, so it chains directly after
`from_source_table` and before `with_column(...)` corrections.

`with_column(name, ...)` is **immutable — it returns a new `Table`**. Always rebind the result of
the chain; calling it for its side effect applies nothing. There is no `table['col'].stype = ...`
assignment form in this SDK.

```python
ratings = (
    kumo.Table.from_source_table(connector['Ratings'], time_column='date')
    .infer_metadata()
    .with_column('movieID', stype=Stype.ID)
    .with_column('userID', stype=Stype.ID)
    .with_column('date', dtype=Dtype.date)
)
```

Inspect resolved types via the table definition:

```python
', '.join(f'{c.name}:{c.stype}' for c in ratings.definition.cols)
```

### Two rules the corrections above encode

1. **Correct foreign keys to `Stype.ID` explicitly.** A foreign key is a *graph-level* fact —
   it only exists because of an edge. Inference sees one table at a time, so it judges an FK
   column on its values alone and calls it `numerical`. It is a key; say so with
   `.with_column('<fk>', stype=Stype.ID)` for every FK column in every table.
2. **A time column must be a real date.** If the column is stored as text in the source (a
   Snowflake `TEXT`, a CSV string) the catalog reports a string, and the table will not have a usable timeline. Declaring
   `.with_column('<time_col>', dtype=Dtype.date)` is what makes staging cast it. When the strings
   are not ISO-8601 (`03/15/2024`), also pass their format:
   `.with_column('<time_col>', dtype=Dtype.date, timestamp_format='%m/%d/%Y')`. `Dtype.date` is
   Kumo's date-and-time type; `Dtype.time` is a time of day and is wrong here. Check the catalog
   dtype (section above) before assuming a `time_column=` argument is sufficient.

---

## Step 3: dtypes and semantic types

`Dtype` (physical): `bool`, `int`, `byte`, `int16`, `int32`, `int64`, `float`, `float32`,
`float64`, `string`, `binary`, `date`, `time`, `timedelta`, `floatlist`, `intlist`,
`stringlist`, `unsupported`.

`Stype` (semantic — how Kumo encodes the column):

| Stype | Use for |
|---|---|
| `numerical` | integers/floats measured as quantities |
| `categorical` | boolean or short string labels |
| `multicategorical` | string containing multiple concatenated labels |
| `ID` | entity identifiers (primary keys, foreign keys) |
| `text` | multi-token strings where language content carries meaning |
| `timestamp` | date/time columns |
| `sequence` | equal-length float lists / embeddings |
| `image` | image URLs |
| `unsupported` | not modelable |

Rules:
- Primary keys and foreign keys should carry `stype='ID'`.
- A column typed `unsupported` (either dtype or stype) cannot be modeled — drop it or retype it.
- Both `dtype` and `stype` must be resolved before a table is used in a `PredictiveQuery`;
  `infer_metadata()` resolves whatever is missing.

---

## Step 4: build the graph

Emit `tables` as a **list** (a `{alias: table}` mapping is also valid; see
`kumoai-reference/references/kumo_finetuning_sdk.md` §5). Edges are `Edge` objects from `kumo.engine`, in the SDK direction:
the table holding the foreign key points at the table holding the primary key.

```python
graph = kumo.Graph(
    tables=[movies, users, ratings],
    edges=[
        Edge(src_table='Ratings', fkey='movieID', dst_table='Movies'),
        Edge(src_table='Ratings', fkey='userID',  dst_table='Users'),
    ],
)
graph.validate()
print('graph valid. snapshot_id:', graph.id)
```

`graph.validate()` is a structural check over table **schemas** only — no materialized snapshot
is needed yet.

### Snapshot — an explicit, required step

```python
snapshot = graph.snapshot()
print('index_path:', snapshot.index_path)
```

`snapshot()` reads every table through its connector and builds the on-disk index. It is not
optional bookkeeping: the snapshot object is the argument passed to
`pquery.generate_training_table(snapshot)` and
`pquery.generate_prediction_table(snapshot, prediction_time=...)`. Training and prediction
themselves belong to `kumoai-reference/references/kumo_finetuning_sdk.md`.

**`Graph` has no editing surface.** Its API includes `.tables`, `.edges`, `.id`, `.native_graph`, `.validate()`, `.visualize()`, and `.snapshot()`.
There is no `add_table`, `remove_table`, `link`, `unlink`, `infer_links`, `get_edge_stats`, `get_table_stats`, or `save`/`load`.
To change a graph, rebuild it.

Use `kumo-graph-visualization/SKILL.md` to include an interactive graph in generated notebooks or an SVG export in scripts after graph validation.
It also covers requested visuals and Markdown inspection during model iteration.
Visualization adds no required review or pause.

---

## Step 5: time columns and temporal correctness

**Anchor time** is the boundary between information available to the model and the future
period it predicts. For a row with a time column, Kumo treats information at or before the
anchor time as known context; events after it are future information. The target window of a
temporal query is left-open, right-closed: with anchor `2025-10-01 00:00:00` and a 30-day
window, the target is `(2025-10-01 00:00:00, 2025-10-31 00:00:00]`.

Assigning time columns:
- Event tables (one row per discrete occurrence): use the time the event occurred, or when
  its information became available. Always set `time_column` on event/fact tables.
- Entity tables (persistent objects such as customers, products): use the entity creation
  time when known.
- Do **not** designate an administrative last-modified timestamp unless it represents when
  the row became valid for modeling.
- Tables without a meaningful timeline can legitimately have no time column.
- `end_time_column` exists as a `Table`/`from_source_table` parameter for rows with a validity
  interval.
  > Not verified: its exact semantics under this SDK. Do not describe or rely on its
  > filtering behavior.

Temporal splits: for a temporal predictive query, Kumo splits train/validation/holdout by the
time column of the target table, and at prediction time accesses only data with timestamps at
or before the anchor. A static (non-temporal) query does not require a time column; Kumo then
splits target rows randomly (e.g. 80/10/10).

**Mandatory table-construction rule:** a table's `time_column` is the availability timestamp for
every selected feature in that row. If it marks the start of an interval, a field that is only
determined when that interval later changes or ends — its endpoint, its realized duration, or a
summary of the completed interval — is future information and must not be used as a feature.

If such an interval endpoint is an ordinary candidate feature, omit it from the positional column
list. If it is configured as `end_time_column`, keep it selected as structural metadata and do not
present it as a feature; do not rely on its interval-filtering behaviour unless that behaviour has
been verified for the deployed bundle.

The same applies to the prediction target: keep the target column selected so the query and table
metadata can reference it, and exclude only the companion columns derived from it.

Leakage causes to screen for:
- A feature directly or indirectly contains the future target.
- An event timestamp marks the start of a period while another column summarizes what happened
  by the end of that period.
- A supposedly static entity column is overwritten over time, so the historical value is not
  recoverable.
- Training and holdout periods overlap or are ordered incorrectly.

Fix the source timeline or drop the leaking field. Do not compensate with model tuning.
Signals of leakage: implausibly strong evaluation, large offline/production gap, one column
dominating explanations.

Date-only values: `2025-10-01` is interpreted at the start of that day. To include all events
on Oct 1, use a boundary of `2025-09-30 23:59:59` rather than midnight on Oct 1.

If no table in the schema has a usable timestamp, a temporal query cannot be posed — surface
this to the user rather than proceeding.

---

## Constraints and common failure modes

Structural:
- At least one table in the graph must have a primary key; otherwise tables cannot be linked.
- `primary_key`, `time_column`, and `end_time_column` must each name a column that is included
  in the table's `columns`.
- `Edge.dst_table` must have a primary key. `Edge.src_table` must contain the `fkey` column.
- Edges are bidirectional; do not declare both directions.
- Timestamps do not need manual linking — Kumo detects them automatically.
- If two linked key columns have different types, both are cast to strings; verify values still
  match after casting (e.g. `16909.0` vs `16909`).
- Rows with missing key values are ignored when creating linkages.
- Rows with unmatched key values across tables are ignored.

Types:
- `Dtype.unsupported` / `Stype.unsupported` columns cannot be modeled.
- Columns left without `dtype` or `stype` block use in a predictive query; call
  `infer_metadata()` or set them explicitly.

Data quality:
- General modelling guidance, not SDK-verified: do not treat an opaque serialized payload as
  natural language merely because its physical type is a string. Machine-generated blobs and serialized vectors are modelable only when they already
  have a supported semantic representation and their meaning is established; if they would need
  parsing or transformation first, exclude them. Constant width or a large character count is
  evidence to inspect, not a threshold — genuine long-form language can be a valid feature where
  the host supports text.
- Blank entries are treated as missing. Kumo does **not** recognize `"NaN"`, `"none"`, `"N/A"`
  as missing — clean them first.
- Numeric placeholder fills such as `-1` are treated as real values; blank them if they mean
  missing.
- Exclude fake/synthetic data, other models' predictions, unreliable columns, and columns whose
  definition changed over time.
- Exclude duplicated/near-duplicate/irrelevant columns: graph size (total column count across
  all tables) drives compute cost.
- Include complete history over the relevant timeframe, not partial records.

Scale limits (defaults; admin-configurable, may be lower per account):

| Limit | Default |
|---|---|
| Rows per table | 15 billion |
| Columns per table | 2,048 |
| Table size | 2 TB |
| Row size | 10 KB |
| Tables in a graph | 12 |
| Rows in a graph | 30 billion |
| Graph size | 2 TB |
| Edges in a graph | 30 billion |
| Parallel jobs | 10 |

Deployment:
- Under Kumo SPCS, only `SnowflakeConnector` is supported.
- Fixing data issues after graph creation: correct the source data; Kumo re-ingests and
  reconnects tables on the next training run.

Self-loops (a table's foreign key referencing its own primary key):
> Not verified. Do not assert whether self-referencing edges are permitted.

Multi-hop / 3+ table paths: no special API. Declare each PK/FK edge; Kumo traverses the graph.
No known hop-count configuration at graph-construction time.

---

## Full worked example

Tables and graph only. Connection, training and prediction are in `kumoai-reference/references/kumo_finetuning_sdk.md` §10.

```python
import kumo.engine as kumo
from kumo.engine import Edge
from kumoapi.typing import Dtype, Stype

# `connector` is the connector built per Step 1; the table names below are placeholders
# for tables that exist behind it.

account_profile = kumo.Table.from_source_table(
    connector['account_profile'], primary_key='account_id', time_column='created_date',
)
location_profile = kumo.Table.from_source_table(
    connector['location_profile'], primary_key='location_id',
)
transaction_events = (
    kumo.Table.from_source_table(
        connector['transaction_events'],
        primary_key='event_id',
        time_column='event_date',
    )
    .with_column('account_id', stype=Stype.ID)     # FK -> ID, every time
    .with_column('location_id', stype=Stype.ID)
)

graph = kumo.Graph(
    tables=[account_profile, location_profile, transaction_events],
    edges=[
        Edge(src_table='transaction_events', fkey='account_id', dst_table='account_profile'),
        Edge(src_table='transaction_events', fkey='location_id', dst_table='location_profile'),
    ],
)
graph.validate()

snapshot = graph.snapshot()      # explicit; required before generating any table
```

Source: shipped package (`kumo.engine.table`, `kumo.engine.graph`, `kumoapi.graph`)

---

## Documented gaps

Do not fill these from memory. If the task requires them, say so.

- `end_time_column` semantics under the Kumo Predict SDK (parameter exists; behavior only
  documented for the separate KumoRFM product line, which is out of scope here).
- Whether self-referencing edges (self-loops) are permitted.
- Excluding the target label column from features at table-definition time: there is no known
  `features=` argument on `Table`. Control feature inclusion via `column_names` / `columns`.
- Any explicit multi-hop / neighbor-sampling configuration at graph-construction time.
- Whether the scale limits, data-quality rules and structural constraints above are enforced by
  this SDK. State them as general modelling guidance, not as hard limits.
