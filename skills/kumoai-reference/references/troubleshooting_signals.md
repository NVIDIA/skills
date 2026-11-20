# Troubleshooting Signals

Observable symptom → likely cause → fix. Ordered by when the agent hits them: table definition, graph construction, query validation, then job time.

This file is the signal set as previously compiled. It is **not** re-verifiable against the published Kumo documentation, which is stale for this version of the local SDK and is not a source here — do not cite it, fetch it, or reconcile against it.

> No verbatim Kumo exception text is established for this SDK. Entries below are keyed on the **observable condition**, not on a quoted message. **Do not fabricate Kumo error strings.**
> The five messages quoted in §1a are the exceptions: they are the SDK's own text, read from its source.

**Who applies a fix.** A fix that changes the data — casting a column, deduplicating, flattening,
filling or filtering values, combining columns into a key, normalising key types — is the user's
to make in their source. The agent reports the symptom, its cause in the data, and that fix, and
asks; it never applies it in the generated code, and never declares a `dtype` to make a data
problem pass (router *What this skill does NOT do*). Choosing a different primary key, time column
or link, or leaving a column out, is a graph change: propose it and let the user confirm it.

---

## 0. The validation calls — run these, do not guess

These exist in the current package. **Call every one with no arguments.**

| Call | Checks |
|---|---|
| `table.infer_metadata()` | Re-resolves `dtype` / `stype` from the source |
| `graph.validate()` | Graph structure, before any snapshot. Cheap — schemas only |
| `pquery.validate()` | Query syntax and its fit against the graph |
| `pquery.get_task_type()` | The inferred task type — confirms the ML problem matches intent |

Standard order: build tables -> `graph.validate()` -> `pquery.validate()` ->
`get_task_type()` -> `graph.snapshot()` -> generate tables -> train.

**These do not exist.** Do not emit them, and do not tell the user to run them:
`table.validate()`, `table.metadata`, `table.get_stats()`, `graph.infer_links()`,
`graph.get_edge_stats()`, `graph.get_table_stats()`, `connector.sql(...)`,
`SourceTable.head(...)`.

Inspect a table through `.definition` (`.pkey`, `.time_col`, `.end_time_col`) and a graph through
`.tables` / `.edges`. To look at data, read the parquet that `snapshot()` and
`generate_training_table()` leave under the scratch directory.

---

## 1. Table definition

### Symptom: `connector.table("X")` raises `ValueError`
**Cause:** The table name does not exist in the connector.
**Fix:** List what is actually there before constructing tables: `connector.table_names()`, or guard with `connector.has_table(name)`. Snowflake object names, for example, are typically uppercase (`connector.table("ORDERS")`).

### Symptom: a column's `stype` is unset, or the query fails because a column has no semantic type
**Cause:** `Table.from_source_table()` and the explicit `Table(...)` constructor do **not** fully populate `dtype`/`stype`. A partially specified table is legal until it is used in a predictive query.
**Fix:** Call `.infer_metadata()`, or set the type explicitly:
```python
table.column("col").stype = "categorical"
```
Fully specify tables in production code to avoid unexpected re-inference.

### Symptom: "invalid data type error" / type-mismatch recommendation on a column
**Cause:** The semantic type (`Type`) does not align with the data type (`Data Type`).
**Fix:** Align them. Valid `Stype` values: `numerical`, `categorical`, `multicategorical`, `ID`, `text`, `timestamp`, `sequence`, `image`, `unsupported`.

### Symptom: a column holds URLs, lat/long, IP addresses, or phone numbers
**Cause:** These types are **not supported** for preprocessing.
**Fix (the user's, upstream):** derive domain/path, geographic area, subnet, or area code in the source and use that as a categorical column. The agent can leave the column out of the graph meanwhile, and says so.

### Symptom: a column holds nested schemas, arrays, or maps
**Cause:** Not supported.
**Fix (the user's, upstream):** flatten to a delimited string in the source. `["TV", "electronics", "promotion"]` → `"TV, electronics, promotion"` (multicategorical).

### Symptom: timestamps arrive as strings or are misparsed
**Cause:** Non-ISO-8601 format, or Parquet columns not cast to a `DATE`/`TIME`/`TIMESTAMP` type.
**Fix:** Declare a TEXT date column with `dtype=Dtype.date` (a declaration the skill makes). Otherwise the user casts the column in the source (ISO 8601 or epoch time).

### Symptom: row count silently drops after ingestion
**Cause:** Duplicate primary key values — Kumo retains one row per key and drops the rest.
**Fix:** Report how many rows the key repeats on (the preflight probe, or the snapshot warning in
§1a) and put the choice to the user: a different, genuinely unique key, or the source deduplicated
upstream. The agent does not deduplicate.

### Symptom: a categorical column has very high cardinality
**Cause:** Categorical supports up to **4,000 unique values by default**.
**Fix:** Re-type the column (`ID` for identifiers, `text` for multi-token strings) — a declaration the agent proposes. Bucketing the values is a data change for the user to make upstream.

### Symptom: missing values are filled with `N/A`, `none`, `null`, or `-1`
**Cause:** These placeholders are interpreted as real values.
**Fix (the user's, upstream):** leave the field blank in the source. Report the placeholder to the user; a high-missingness column can still be useful when absence itself is informative.

---

## 1a. Kumo reports a problem in the data — stop and put it to the user

These messages are the SDK's own (`kumo.engine.graph`, `kumo_core.pquery`,
`kumo_core.diskgraph.validation`). Each says the data is not what the graph declares. The run may
still finish. Do not report success before the user has seen the message and decided, and do not
change data, types or the query to make it go away (*Who applies a fix*, above).

| Message (begins) | What it means | Put to the user |
|---|---|---|
| `Snapshot contains <n> duplicate primary-key row(s) for <table>.<key>` | `<n>` rows share a key; Kumo keeps one per key and drops the rest | a different key, or the source deduplicated; or continue knowing `<n>` rows are gone |
| `Foreign key <table>.<fk> matched <m> of <n> non-null rows` | only `<m>` of `<n>` rows reach the parent; the rest get no edge | the link may be wrong (wrong column, or values differing in format, e.g. `007` vs `7`), or the parent rows are missing |
| `<n> invalid values found in <table>.<column>` | `<n>` values did not parse as the column's type and are read as null (see §5) | whether those rows are bad data, or the column's type is wrong |
| `key <table>.<column> has a floating-point dtype` | an ID column is stored as a float (often an integer key with nulls) | the column fixed upstream as an integer or string key |
| `foreign key <table>.<fk> (dtype=…) cannot link to primary key <table>.<pk> (dtype=…)` | the two sides of a link hold different types | which side is right, fixed upstream; never cast one side in the notebook |

A float or mismatched key fails `graph.validate()` and stops the notebook. The other three are
warnings: the notebook keeps going and trains on what is left. Either way, the report the user
reads starts with the message, the table and column, and the count.

---

## 2. Graph construction (highest priority — fail here, not at job time)

### Symptom: a table cannot be linked into the graph at all
**Cause:** To connect tables into a graph, **at least one table must contain a primary key**. Edges map a `src_table`'s foreign key to a `dst_table`'s primary key.
**Fix:** Set a primary key on the entity/dimension table:
```python
table.primary_key = 'new_primary_key'   # must already be a column of the table
print(table.has_primary_key())
```

### Symptom: an edge is declared but the model learns nothing from it
**Cause:** FK values do not actually match PK values. Rows with missing or unmatched key values are
**silently ignored** when linkages are built, so a broken join looks like a sparse one. If the two
linked columns have different types, both are **cast to strings** — values equal as numbers may not
match as strings (`1.0` vs `1`).
**Fix:** There is **no `get_edge_stats()` in this SDK.** The snapshot reports the match rate
(§1a). Report it, and put a low one to the user before training: a wrong link to correct, or keys
to fix upstream so both sides hold the same values in the same type. Do not cast either side.
Set every foreign key to `Stype.ID`; left as numerical it is treated as a feature rather than a
link.

### Symptom: the edge direction looks wrong / no neighbors are sampled
**Cause:** `src_table` must be the table holding the **foreign key**; `dst_table` holds the matching **primary key**. Edges are bidirectional once declared, but the declaration itself is directional.
**Fix:**
```python
edges=[
    dict(src_table='transaction', fkey='StockCode', dst_table='stock'),
    dict(src_table='transaction', fkey='CustomerID', dst_table='customer'),
]
```

### Symptom: graph exceeds known limits
**Cause:** Default quotas: 12 tables per graph, 15B rows per table, 2,048 columns per table, 2 TB per table, 30B rows per graph, 2 TB per graph, 30B edges. Environments may be configured lower.
**Fix:** Reduce table count or prune columns. Limits are admin-configurable; a limit increase requires the Kumo point of contact.

### Symptom: an event table has no time column
**Cause:** Kumo uses time columns to enforce the anchor-time boundary. For event tables (each row a discrete occurrence), the time column should be when the event occurred or when its information became available.
**Fix:** Set `time_column` on the event table:
```python
transaction = kumo.Table.from_source_table(
    source_table=transaction_src,
    time_column='InvoiceDate',
).infer_metadata()
```
Entity tables may use the entity creation time. Tables with no meaningful timeline can remain without a time column — that is legal. Do **not** designate an administrative "last modified" timestamp unless it represents when the row became valid for modeling.

### Symptom: the user supplies pre-aggregated features (`logins_last_30_days`, etc.)
**Cause:** Aggregating before ingestion removes the sequence/relationship information the GNN learns from. Kumo reports aggregated features can perform **5–10% worse** than an equivalent GBDT model, while raw data performs better.
**Fix:** Ask for the raw timestamped event table instead. Do not do feature engineering on Kumo's behalf.

### Symptom: the user has downsampled tables before ingestion
**Cause:** Independently sampling rows from each table breaks FK coverage and leaves entities without their histories.
**Fix:** Sample around the prediction entity and keep all related rows across every table — e.g. "all rows in all tables where `user_id % 10 == 1`". Never sample each table independently.

---

## 3. Predictive query validation

### Symptom: `pquery.validate()` reports a syntax error
**Cause:** PQL is `PREDICT <target> FOR EACH <entity>` with an optional filter clause. The query must reference the **names of tables as registered in the graph's `tables` dict**, not the source table names.
**Fix:** Re-check table names against `graph.tables`. Minimal valid form:
```python
pquery = kumo.PredictiveQuery(
    graph=graph,
    query=(
        "PREDICT MAX(transaction.Quantity, 0, 30, days)\n"
        "FOR EACH customer.CustomerID\n"
        "ASSUMING SUM(transaction.UnitPrice, 0, 7, days) > 15"
    ),
)
pquery.validate()
```

### Symptom: aggregation window rejected
**Cause:** Both *start* and *end* must be **non-negative integers**, and *end* must be **greater than** *start*.
**Fix:** Correct the bounds. Semantics: start is exclusive, end is inclusive. `SUM(TRANSACTIONS.PRICE, 10, 30)` from a `2020-01-01` anchor aggregates `2020-01-11 00:00:00 < t <= 2020-01-31 00:00:00`.

### Symptom: the user asks for a horizon in an unsupported unit
**Cause:** The accepted time units are `days` (default), `months`, and `hours`.
**Fix:** Convert. Do not emit `weeks`, `quarters`, or `years` — they are not accepted.

> `weeks` is not an accepted unit. Convert to `days`.

### Symptom: `get_task_type()` returns a different task than intended
**Cause:** The PQL formulation determines the ML task, not the user's prose. The same business problem has several valid formulations.
**Fix:** Always call `get_task_type()` after `validate()` and confirm it against the user's intent before generating a training table. Reformulate the query if it disagrees.

### Symptom: unclear whether the generated training data is what the user meant
**Cause:** Validation checks syntax, not semantics.
**Fix:** Inspect the generated data before training:
```python
training_table = pquery.generate_training_table(non_blocking=False)
print(training_table.data_df().head())
```
`data_df()` and `data_urls()` both expose the generated data.

### Symptom: an `END_DATE` column is set on a non-entity table and appears to be ignored
**Cause:** End-date-aware sampling is currently supported **only** on the entity table (after `FOR EACH`) and on the RHS table in link prediction (after `LIST_DISTINCT`).
**Fix:** Do not rely on end dates elsewhere.

### Symptom: date-only boundary includes one day too few / too many
**Cause:** A date such as `2025-10-01` is interpreted at the **start** of that day. The target window is left-open, right-closed.
**Fix:** To include all of October 1, use `2025-09-30 23:59:59` rather than midnight on October 1.

---

## 4. Data leakage

### Symptom: evaluation results are unexpectedly strong, offline and production performance diverge, or one column dominates explanations
**Possible cause:** Data leakage; these symptoms alone do not establish it. Check for:
- A feature directly or indirectly contains the future target.
- An event timestamp marks the start of a period while another column summarizes what happened by the end of it.
- A supposedly static entity column is overwritten over time, so its historical value is unavailable.
- Training and holdout periods overlap or are ordered incorrectly.

**Fix:** Verify when each source value became available relative to the anchor time before
correcting the timeline or removing a confirmed leaking field. **Do not tune the model to
compensate for temporally invalid data.** For interpretation limits of recorded metrics and
explanations, follow the existing-run analysis route in `kumo-data-science/SKILL.md`.

### Pre-flight data quality checklist
Run before building the graph:
- [ ] A source table contains the label needed for positive and negative training examples.
- [ ] Enough history to learn seasonal patterns — typically 1–2 years, sometimes more.
- [ ] The data fits within the quotas and limits above.
- [ ] PII is hashed, obfuscated, or removed per policy.
- [ ] No target leakage — columns mutated/updated over time must carry the timestamp at which the value first became known.
- [ ] The dataset contains the signals a human would expect to be predictive.
- [ ] Tables can all be linked through stable primary/foreign keys.

Column-selection criteria: **clean** (no synthetic data, no other models' predictions, no columns whose definition changed over time), **relevant and mutually exclusive** (drop duplicate/irrelevant columns — graph size drives compute cost), **complete** (full history, not just first/last interaction).

---

## 5. Training and prediction time

### Symptom: `ValueError: prediction_time must not include a timezone`
**Cause:** `prediction_time` was passed as a timezone-aware `datetime`. `kumo_core` rejects any
`tzinfo` and applies UTC itself.
**Fix:** Pass a **naive** datetime.
```python
PREDICTION_TIME = datetime(2026, 8, 1)                      # correct
# PREDICTION_TIME = datetime(2026, 8, 1, tzinfo=timezone.utc)  # raises
```

### Symptom: `ValueError: prediction_time is not supported for static queries`
**Cause:** `prediction_time` was passed for a non-temporal query.
**Fix:** Drop the argument. Confirm the intended task with `pquery.get_task_type()`.

### Symptom: `fit()` or `predict()` raises `TypeError` about a missing argument
**Cause:** Both require `snapshot=` as a keyword. `fit(training_table)` and
`predict(prediction_table)` alone are errors.
**Fix:**
```python
training_result = trainer.fit(training_table, snapshot=snapshot)
prediction_result = trainer.predict(
    prediction_table, snapshot=snapshot, output_config=output_config)
```

### Symptom: code from an older notebook raises `AttributeError` on `kumo.<something>`
**Cause:** It targets the previous SDK — `set_backend`, `LocalBackend`, `create_identity`,
`OutputConfig`, `kumo.Stype`, edges written as dicts. A `Graph(tables={alias: table})` mapping
is current, not old.
**Fix:** Do not patch it call by call. Say it targets a superseded build and rewrite against
`kumoai-reference/references/kumo_finetuning_sdk.md` §10. The recognise-and-refuse table is §11.1 there.

### Symptom: `snapshot()` warns `<count> invalid values found in <table>.<column>`
**Cause:** Values in that column did not parse as its declared type. They are read as null; the
snapshot still completes.
**Fix:** Report the table, column and count to the user. A large count usually means a wrong
`dtype` (a TEXT date without `dtype=Dtype.date`, or a numeric column with string placeholders);
correct the declaration or the source and rebuild. Do not ignore it silently.

### Symptom: `snapshot()` warns the scratch directory is on network storage
**Cause:** The scratch root is on network-backed storage such as EBS. Sampling and training-table
generation run much slower there.
**Fix:** It is a warning, not an error — a small run can continue. For a large run, point
`kumo.configure(scratch_dir=...)` at local NVMe if the runtime has it.

### Symptom: link prediction job fails with an explicit out-of-memory error
**Cause:** Link prediction has **no** batch-size fallback. Node prediction, classification, regression, and forecasting catch OOM, halve the batch size, and retry; link prediction does not.
**Fix:** Reduce `output_embedding_dim` — the first remedy to try.

### Symptom: results change between two training runs on the same query
**Cause:** The source data moved underneath the graph.
**Fix:** `graph.snapshot()` locks in the current version of all tables so repeated runs use the same data. It also reduces resource usage across multiple jobs.

---

## 6. Known gaps

> Not verified:
> - Verbatim exception text and error codes raised by `Graph.validate()` and
>   `PredictiveQuery.validate()`. **Do not fabricate them.**
> - The exception types raised by graph or PQL validation failures.
> - Whether `pquery.validate()` catches nonexistent column references, or only syntax.
