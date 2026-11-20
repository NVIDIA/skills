# Host Adapter — Generic coding agent

For a coding agent working beside the user — Claude Code, Codex, or any agent with a shell and
file tools — rather than inside a data platform. The agent produces code that runs in **any**
Python environment where Kumo is present, against whichever of the six Kumo connectors the data
calls for. The agent does not know where that code will run; the user does.

Kumo is already present in the environment that runs the artifact; never raise environment setup
with the user. Generating the artifact — connect step, credentials, destinations and read-back —
is in `runtime.md` in this directory; read it at Step 2.

This host has no job file: it does not submit, schedule or monitor jobs. When asked for one, say
so and offer the notebook or script.

---

## 1. Source scope

`source_scope` names a system and a location in it. Map it to its connector
(`kumoai-reference/references/kumo_finetuning_sdk.md` §3, §3.4):

| The user names | Connector |
|---|---|
| A local directory of CSV or Parquet files | `FileConnector` |
| `s3://bucket/prefix` | `S3Connector` |
| `gs://bucket/prefix` | `GCSConnector` |
| A BigQuery project and dataset | `BigQueryConnector` |
| A Databricks catalog and schema | `DatabricksConnector` |
| A Snowflake database and schema | `SnowflakeConnector` |

- A dotted name with no system named (`SALES.EVENTS`) fits Snowflake, BigQuery and Databricks
  alike. Ask which system holds it — that one question, nothing else. A system already named in
  any turn, or evident from the user's tools or files, settles it.
- A connector name on its own is not a location on this host, which keeps no registry of named
  connectors. Ask what system and location it points at.
- A Snowflake semantic view resolves through `kumo-data-science/accept/semantic_view.md`, which
  needs Snowflake SQL (§2). Without it, the §2 details are the view's base tables, their keys and
  its relationships.
- Values only the runtime needs — a Snowflake warehouse, a Databricks SQL warehouse or cluster, a
  `connections.toml` entry — are not part of `source_scope` and are never asked for at intake.
  `runtime.md` §1 handles them.

ADLS has no Kumo connector. Data there is outside this host's reach: say so.

---

## 2. Access

Implements `kumo-data-science/accept/preflight.md` §0. Check, in this order, and stop at the
first that reads the **columns and types** inside `source_scope`. A listing of object names
(`aws s3 ls`, `gcloud storage ls`) proves access, not structure: keep going down the list for a
reader, and if none is found, send the ask-back below with **Details** as the way forward.

1. **A tool in this session** that reads the system — a server or connector tool for that
   warehouse or storage. Use only its read operations.
2. **A command-line tool on the `PATH`** that is already authenticated (table below). Find it
   with `command -v <tool>`, then run its check command against `source_scope` itself — a
   general "who am I" call proves the login, not access to the data.
3. **Python in this session** in which `import kumo.engine` works and the provider's ambient
   credentials resolve. Build the connector exactly as the artifact would (`runtime.md` §2) and
   read `connector.table_names()` and `connector['<table>'].columns`, which gives each column's
   name and source type. Catalog reads only: no `Table`, `Graph` or snapshot.

| Source | Tool | Check, inside `source_scope` | Columns and types | Declared keys | §3 probe |
|---|---|---|---|---|---|
| Local files | your file tools; `duckdb`, or Python with `pyarrow` | list the directory | CSV header line; Parquet schema (`duckdb -c "DESCRIBE SELECT * FROM '<file>'"`) | none | `duckdb`, or `pyarrow` |
| S3 | `aws` | `aws s3 ls s3://<bucket>/<prefix>/` | file names only — not enough on its own; for columns, a reader of `s3://` with the same credentials (`duckdb`, or check 3), else the ask-back | none | the same reader, else skip |
| GCS | `gcloud` | `gcloud storage ls gs://<bucket>/<prefix>/` | as S3, with `gs://` | none | as S3 |
| BigQuery | `bq` | `bq ls <project>:<dataset>` | `bq show --schema --format=prettyjson <project>:<dataset>.<table>` | `INFORMATION_SCHEMA.TABLE_CONSTRAINTS` and `KEY_COLUMN_USAGE` of the dataset | `bq query --nouse_legacy_sql` |
| Databricks | `databricks` | `databricks tables list <catalog> <schema>` | `databricks tables get <catalog>.<schema>.<table>` | the same output's table constraints | only through a SQL warehouse the user named; else skip |
| Snowflake | `snow` | `snow sql -q "<the preflight §1 scope query>"` | the preflight §2 query | `SHOW PRIMARY KEYS` / `SHOW IMPORTED KEYS IN SCHEMA <DB>.<schema>` | `snow sql -q` |

A skipped probe leaves its candidates `[inferred, unverified]`, as preflight §0 says. A BigQuery
probe is billed by the bytes it scans; probe only the candidate columns.

For BigQuery, also read each dataset's `location` with `bq show --format=prettyjson
<project>:<dataset>` — the source, and the population and destination datasets once confirmed.
A population dataset in a different location from the source cannot be filled by one query: say
so before accepting it. The notebook reads locations itself (`runtime.md` §2), so they are not
config constants.

**What a check may do.** Read catalog metadata and run the preflight §3 probe — nothing else.
Never run a command that writes, creates, copies or exports data; never sign in, switch a
profile or account, or edit a tool's configuration; never add or update software to get a
reader. Tool permission prompts are the user's to answer; a refused prompt means that path is
unavailable.

**When nothing reads the source's columns**, send the preflight §0 ask-back. State what you checked and
what happened — "no BigQuery tool in this session; `bq ls my-proj:sales` returned `Access
Denied`" — then offer:

1. **Access**: authenticate the named command-line tool as an identity that can read the scope,
   or connect a tool for that system to this session, and tell you when it is done. The user
   does this; you never handle the credential.
2. **Details**: paste the structure. No rows are needed. Useful forms: DDL, `DESCRIBE` output,
   `bq show --schema` JSON, a Parquet schema, CSV header lines — with any declared keys.

---

## 3. Preflight

Answer `kumo-data-science/accept/preflight.md` §1-§6 with the tool §2 found. Write table and
column names exactly as that tool returns them; the artifact must use the same spelling
(`kumoai-reference/references/kumo_finetuning_sdk.md` §3.2). Probe results are evidence for the
proposal only: never written, never saved to a file, never fed to the artifact.

For files and object stores, a table is a file or a directory of parts in one format, below the
source location. **Name it without its file suffix**: `accounts.parquet` is the table `accounts`.
The connector finds the file from the bare name, and the name you pass becomes the table's name
in the graph, the edges and the PQL, where a suffix's dot does not parse. Show the bare names in
the proposal.

---

## 4. Destinations

Any destination a Kumo connector writes
(`kumoai-reference/references/kumo_finetuning_sdk.md` §9.2): a table in Snowflake, BigQuery or
Databricks, or a path in a local directory, S3 or GCS. The destination may be in a different
system from the source; it gets its own connector. Not destinations here: ADLS, and BigQuery for
anything but predictions and embeddings.

A path destination ends in `/` (a directory) or in `.parquet` / `.csv` (one file). The sink adds
`.parquet` to any other path, so the file would not be where the user said. When the user gives
a path with neither, restate it with a trailing `/` at Node 1d and let them confirm or correct it.

---

## 5. Runtime column constraints

None known for this host. Do not exclude text columns pre-emptively. If a run fails while loading
a text encoder, report the error and the column; that finding belongs at Node 1c.

---

## 6. Capability skills

The capability skills were written for, and measured on, Snowflake. Their Kumo calls hold for
every connector. Where a step names a Snowflake registry, stage, session or table, follow it only
when the source or destination is Snowflake. For any other connector, say that the step is not
covered on this host; do not adapt it.

---

## 7. Other deployment modes — out of scope

Hosted Kumo — `kumo.init(...)`, an API key, `KUMO_API_KEY` / `KUMO_API_ENDPOINT`, Kumo Studio — and
the Snowflake Native App are not this SDK. If the user is on one of them, say so and stop. Never
blend their setup into generated code.
