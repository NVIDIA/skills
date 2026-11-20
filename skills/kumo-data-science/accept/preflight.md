# Read-only Preflight — inferring structure before the graph gate

Once the required fields are present, the agent resolves **structure** itself. It does not ask
the user for column names, keys, or joins, and it builds no Kumo table or graph at this point —
preflight is a catalog read of the source plus explicitly targeted, read-only probes against
source data, and it is read-only throughout. A probe exists only to produce evidence for the
proposal; it is never written anywhere and never becomes a model input.

**What** preflight establishes is portable and written here. **How** it reads the source — which
tool, which catalog query — belongs to the active host, in the *Access* and *Preflight* sections
of `hosts/<host>/adapter.md`. The examples below are SQL; a host whose source has no SQL answers
the same questions with its own reader.

Preflight produces the **graph proposal** the user confirms at Node 1c. Every value carries a tag:

| Tag | Meaning |
|---|---|
| `[supplied]` | the user stated it; never overridden |
| `[declared]` | read from a semantic view's definition (`semantic_view.md`); shown for confirmation |
| `[inferred]` | measured by preflight; shown for confirmation |
| `[inferred, unverified]` | a candidate preflight could not establish, only fail to disqualify; must be confirmed before use |
| `[default]` | a contract default (`table_mode: all-connected`); shown for confirmation |

If the harness or host has already run preflight and handed you a result (for example a
"Preflight result for <schema>" message), use that result as-is — do not ask the user to repeat it.

---

## 0. Access — can you read the source?

Before §1, establish whether **you**, in this session, can read `source_scope` with a read-only
tool you already have. The host adapter's *Access* section says which tools count and the check
that proves each one reaches the data.

- **You can read it** → run §1-§6 through that tool, and name the tool in the proposal's
  `Source scope` line.
- **You cannot** — no suitable tool, or the tool reaches the system but not this scope (an
  authorization error, an unknown schema or bucket) → stop before §1. In one message, tell the
  user exactly what you checked and what happened, quoting any error, and offer both ways forward:
  1. **Access** — they make a read tool available, as the adapter describes, and tell you; you
     then repeat the check.
  2. **Details** — they paste the structure: tables, columns with types, and any declared keys,
     in whatever form they have (the adapter lists useful ones). You build the proposal from it.

  This is one ask-back. Do not also ask for anything §1-§8 would infer.
- **Never** ask for a credential's value or invite one into the chat, sign in, change a tool's
  configuration, or set up a tool on the user's behalf. Never broaden the search beyond
  `source_scope` because the scope itself was unreadable.

A proposal built from the user's details follows §1-§8 unchanged, except that nothing that needs
row data can run: the §3 rule 5 probe is skipped, so its candidates stay `[inferred, unverified]`.
Say in the `Source scope` line that the structure came from the user, not from a read.

This check concerns your session only. The generated artifact authenticates at run time through
its own connect step (the host's `runtime.md`), so your access — or its absence — changes
neither the generated code nor any gate.

---

## 1. Resolve `source_scope`

If `source_scope` is a semantic view (`DB.SCHEMA.VIEW`), read `accept/semantic_view.md` and
resolve it there instead of here; then continue at §2.

```sql
SELECT SCHEMA_NAME FROM <DB>.INFORMATION_SCHEMA.SCHEMATA WHERE SCHEMA_NAME ILIKE '<schema>';
SELECT TABLE_NAME, ROW_COUNT FROM <DB>.INFORMATION_SCHEMA.TABLES WHERE TABLE_SCHEMA ILIKE '<schema>';
```

- Not found, or zero tables → **ASK BACK**: the source cannot be resolved; ask for a valid
  location. Do not guess a similarly named schema, dataset, bucket or directory.
- Found → enumerate tables **only inside this scope**. Never search other schemas, and never
  rank or curate tables by business relevance.

## 2. Columns and types

```sql
SELECT TABLE_NAME, COLUMN_NAME, DATA_TYPE
FROM <DB>.INFORMATION_SCHEMA.COLUMNS WHERE TABLE_SCHEMA ILIKE '<schema>'
ORDER BY TABLE_NAME, ORDINAL_POSITION;
```

## 3. Primary key per table

In order of preference, first match wins:
1. A declared primary key, where the source declares any (the adapter's preflight shows how to
   read them; files and object stores declare none). Declared is not enforced: it is probed
   before use (**Probe every key the proposal uses**, below).
2. `<TABLE_STEM>_ID`, where the stem is the table name without a `_DIM` / `_PROFILE` / `_EVENTS`
   suffix, singular or plural (`CUSTOMER_ID` in `CUSTOMERS`, `PAGE_VIEW_ID` in `PAGE_VIEW_EVENTS`).
3. A column named `ROW_ID`, `ID`, or `EVENT_ID`.
4. For dimension / profile tables only: `<TOKEN>_ID` for any token of the table name
   (`ACCOUNT_ID` in `ACCOUNT_PROFILE`). Never for event tables — there such a column is a link.
5. A column that one or more *other* tables in scope carry under the same name is a **candidate**
   only. Shared naming does not make it a key, and neither does uniqueness: `COUNTRY`, `STATUS` or
   `MERCHANT` can be ordinary attributes, and an accidentally unique attribute is still not an
   identifier. A duplicated inferred key silently discards rows. A column that rules 1-4 already
   established as another in-scope table's key is a link to that table, not a candidate here —
   skip it, as rule 4 does for event tables.

   Use a targeted read-only probe to **reject** candidates, never to establish identity:

   ```sql
   SELECT COUNT(*) AS N, COUNT(DISTINCT <COL>) AS N_DISTINCT, COUNT(<COL>) AS N_NON_NULL
   FROM <DB>.<SCHEMA>.<TABLE>;
   ```

   Anything other than `N > 0 AND N_DISTINCT = N_NON_NULL = N` disqualifies the candidate outright
   — an empty table returns three zeros and does not qualify. Surviving the probe makes it a
   **proposed** key, tagged `[inferred, unverified]`, which the user confirms or corrects at the
   gate before it is used as the `FOR EACH` entity or an edge destination. If several candidates
   survive on one table, or the same candidate survives on more than one side, report the
   ambiguity and propose none. If the probe cannot run, list the column under `Not found` as an
   unverified candidate.

   Rules 2-4 match names only, so run the same probe on the column they pick: a column that
   fails it is a link, not a key (`CUSTOMER_ID` in `CUSTOMER_EVENTS`); go on to the next rule.
   If the probe cannot run, the rule's pick is `[inferred, unverified]`, as here.

6. Otherwise preflight **inferred no primary key**. Report that as its own line and treat the
   table as keyless unless corrected. A keyless table may be an edge *source*, but not an
   `Edge.dst_table` and not the `FOR EACH` entity. Do not conclude from this failure alone that
   the table is an event or fact table — it may have a natural, composite or unconventional key
   the heuristic missed. Never promote a merely unique-looking column to primary key: a column
   unique in the rows you can see is not a declared key, and a child table may hold several rows
   per parent.

Rules 1-4 establish a key and name it `[inferred]` (rules 2-4 once their probe passes). Rule 5
never establishes one: it proposes a survivor as `[inferred, unverified]`.

**Probe every key the proposal uses**, not only rule-5 candidates. A declared key (rule 1) or a
`[supplied]` key that will be an edge destination or the `FOR EACH` entity gets the same read-only
probe; rules 2-4 picks already had it. Snowflake, BigQuery and Databricks record a declared
primary key without enforcing it, so declared does not mean unique. If `N_DISTINCT < N_NON_NULL`
or `N_NON_NULL < N`, keep it in the proposal with the counts and list it under `Not found` —
"`orders.order_id`: 792 repeated values, 0 nulls" — because Kumo keeps one row per key and drops
the rest without failing. The user decides: a different key, or the source fixed upstream. The
agent does not deduplicate or filter. Until the user confirms it, the table is treated exactly as
keyless — usable as an edge source, but not as an edge destination and not as the `FOR EACH`
entity. Confirmation is what lifts that restriction, not the existence of a candidate.

## 4. Time column per table

The first `TIMESTAMP*`, `DATE`, or `DATETIME` column, preferring names ending in `_AT`, `_TS`,
`_TIME`, `_DATE`, or named `EVENT_TIME` / `TIMESTAMP`. A table with none is static.

For every time-ranged target aggregation, the table it references must have a usable
`time_column`. If it does not, say so in the proposal and say what would make the task
expressible: a timestamp on some other table does not repair the target.

A catalog `TEXT` column may still be usable when its date semantics are established and the
generated code casts it with `dtype=Dtype.date`; the absence of a timestamp-typed catalog column
is not by itself grounds for rejection. Integers that merely order or describe events — a
sequence number, a day-of-week, an hour, a gap in days since a previous event — are not
timestamps and must not be treated as one.

## 5. Column availability

For each table, place every candidate feature in time relative to that table's proposed
`time_column`, then exclude the ones that are not yet knowable at it.

- When the time column marks the beginning of a validity period or interval, exclude a feature
  whose value is only determined once that interval later changes or ends: its eventual endpoint,
  its realized duration, or any summary of the completed interval.
- Exclude a feature computed from, or containing, the confirmed prediction target.
- Apply the active host adapter's runtime column constraints.

Decide by semantic role, never by a fixed column-name list. Record every exclusion and its reason
in the proposal. Where the role is genuinely ambiguous, do not invent one — show the column as an
exclusion the user can overturn at the gate, rather than asking the user for column names.

## 6. Links

A link is proposed when a column in table A has the **same name** as the primary key of table B
(`FRAUD_EVENTS.CUSTOMER_ID → CUSTOMERS.CUSTOMER_ID`). Foreign-key constraints, if declared, are
taken as well, where the source declares any. Nothing else is inferred — no fuzzy matching,
no value overlap checks, no joins executed.

## 7. Seeds and table mode

- **Seeds** are the tables behind `prediction_entity` and `prediction_target` (for link
  prediction: source entity, item entity, and the interaction table). Resolve them by name
  against the enumerated tables; if none matches, ask the user which table holds the entity or
  the outcome — that is intent, not structure.
- **Preflight runs only after Node 1a passes.** It resolves tables for fields the user already
  supplied; it never back-fills a field the user left out. If the required-fields check found
  `positive_interaction`, `recommended_item`, `positive_label_condition` or `time_window`
  missing, ask for them first — do not read them off the schema.
- **`table_mode: all-connected [default]`** — every table reachable from the seeds through the
  proposed links is included. List any table that is **not** reachable so the user can drop it
  or name a link.
- If the seeds cannot be connected through inferred links, say so in the proposal and ask
  which existing columns link them; a name mismatch can hide a real link. Generate nothing until
  the seeds connect. If no existing column pair connects them, **reject**: connecting them would
  require creating or transforming data.

## 8. The graph proposal

Present exactly this shape, then **stop and wait** for confirmation:

```
Source scope: USER_DATA.E_COMMERCE [supplied] — 7 tables found, read via SQL
Table mode: all-connected [default] — 7 of 7 tables reachable from the seeds

Metadata
- customers: primary key CUSTOMER_ID [inferred]
- fraud_events: primary key EVENT_ID [inferred]
- fraud_events: time column EVENT_AT [inferred]
- ...

Links
- fraud_events.CUSTOMER_ID -> customers.CUSTOMER_ID [inferred]
- ...

Excluded columns
- fraud_events.RESOLVED_AT [inferred] — not knowable at the row's EVENT_AT
- fraud_events.DAYS_TO_RESOLVE [inferred] — duration of an interval that has not ended yet
- ...

Not found
- payments: no primary key inferred [inferred] — treated as keyless; edge source only unless corrected
- events: no usable time column [inferred] — a time-ranged target cannot reference this table
- ...

Confirm, or correct any line (your values take precedence).
```

`Excluded columns` and `Not found` are local availability-gate extensions to the confirmation
format.

## 9. Never

- Prepare, transform, aggregate or materialize **model inputs** — joining, cleaning,
  deduplicating, creating labels. A diagnostic scalar aggregate over source rows is permitted
  solely as evidence for the proposal: row, distinct and null counts for a proposed or candidate key. Its
  result is reported and then discarded; it is never written anywhere and never reaches training.
- Rank tables by relevance or drop tables the user did not ask to drop.
- Proceed to PQL before the proposal is confirmed.
- Ask the user for columns, types, keys or joins — with one exception: when §0 finds you cannot
  read the source, offering to take its structure from the user is the §0 ask-back. Whatever the
  proposal is built from, it is still tagged, still confirmed, and contains no invented values.
