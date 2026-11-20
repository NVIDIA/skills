# Semantic view as `source_scope`

Read this file only when `source_scope` is a fully qualified semantic view (`DB.SCHEMA.VIEW`). It
replaces `preflight.md` §1 for that case; every other preflight section still applies. Nothing here
applies to a schema or connector.

## Resolve the view

A semantic view is not a schema to search. It is a definition that points at physical tables and
declares their keys and relationships. Read the definition; do not query the view itself.

```sql
DESCRIBE SEMANTIC VIEW <DB>.<schema>.<view>;
```

The result holds one `property` / `property_value` pair per object, not one column per field.
Group the rows by `object_name` for each `TABLE` and `RELATIONSHIP` object, then read each field
as "the row whose `property` is X, taking `property_value`": for a `TABLE` object read
`BASE_TABLE_DATABASE_NAME`, `BASE_TABLE_SCHEMA_NAME`, `BASE_TABLE_NAME` and `PRIMARY_KEY` this way;
for a `RELATIONSHIP` object read `TABLE`, `REF_TABLE`, `FOREIGN_KEY` and `REF_KEY` this way —
`TABLE` and `REF_TABLE` are the view's logical names too; map each to its physical table through
the extracted `TABLE` records before using it anywhere downstream.

- Not found or unreadable → **ASK BACK**, exactly as for a schema that does not resolve.
- **A logical table with no `BASE_TABLE_*` properties does not name a physical table.** One
  defined by a query carries its own definition instead. The SDK can read a query through
  `connector.sql(...)`, but this skill does not build graph tables from SQL it composed itself, so
  the view alone is not enough. **ASK BACK**: name the logical table, say it is defined by a query
  rather than a base table, and ask for a view whose tables are physical or for the physical table
  to use. Do not parse the definition or rebuild it as a query of your own.
- **Only a plain column-to-column equality relationship becomes an edge.** A Kumo `Edge` is
  foreign key to primary key, matched on equality. If a relationship declares any other kind of
  match, do not read its `FOREIGN_KEY` / `REF_KEY` as an ordinary key pair: a non-equality payload
  can read as a one-column key and become an edge that does not mean what the view says.
  **ASK BACK**: name the relationship and its kind, and say only equality joins become edges.
- **The base tables must all sit in one database and schema.** The connector reads from a single
  schema. If they span more than one, **ASK BACK**: name the schemas found and ask for a semantic
  view or a schema whose tables share one schema. Do not pick one schema, and do not drop tables
  to make it fit.
- That one schema becomes the effective scope: it is the `database` / `schema_name` passed to
  the connector, and `preflight.md` §2 reads columns from it.
- **The tables in scope are exactly the view's base tables** — not every table in that schema.
  The user chose them by naming the view; that is their selection, not curation by the agent.
- **Always use physical names.** The view's logical names (`USERS`, `TRANSACTIONS`) do not exist
  for Kumo. Every table, key, link, edge and PQL reference uses the base table and column names,
  with the same case and quoting the catalog reports.
- Each `PRIMARY_KEY` and each relationship is `[declared]`. They replace `preflight.md` §3 and §5
  for the tables they cover; those sections still apply to any base table the view gives no key
  or link for.
- A key with more than one column, on either side of a relationship, cannot become a Kumo edge.
  **ASK BACK**: name the relationship and say a single-column key is needed, the same way as a
  view that does not resolve — ask for a corrected view or schema. Do not offer to drop the
  table, concatenate the columns, or substitute a different link as options. Those are repairs,
  not a corrected source, and a menu of repairs is not an ask-back.
- A declared `PRIMARY_KEY` with more than one column only matters when it is what some
  relationship's `REF_KEY` points **at**. If no relationship's `REF_KEY` is this composite key, it
  needs no ask-back by itself — not even when this table is the **foreign-key** side of its own
  relationships and one of those `FOREIGN_KEY` columns happens to also be part of the composite
  key; pointing out of a table never depends on that table's own primary key. Leave the table in
  the graph exactly as `preflight.md` §3 treats a table with no usable primary key: keyless, or
  with a single-column `[inferred]` key (e.g. an `*_ID` column) offered for confirmation. That
  keyless or `[inferred]` treatment applies only while **no** relationship references this table
  at all. If some relationship's `REF_KEY` is a different single column of it, that column is the
  key, by the bullet below; do not infer another, because an edge points at the key Kumo is given
  and a guessed one joins the foreign key to the wrong column. Ask back only when some
  relationship's `REF_KEY` is the composite key — that is the previous bullet, not this one.
- A relationship whose **single-column** `REF_KEY` is not the referenced table's declared
  `PRIMARY_KEY` (for example it references a `UNIQUE` key instead) is still expressible. A Kumo
  primary key is declared by the caller in `Table.from_source_table(..., primary_key=)` and does
  not have to match the catalog, so that column becomes the table's Kumo primary key, shown
  `[declared]`. Say in the proposal that it differs from the catalog's primary key so the user can
  correct it. This is not a repair; it is reading the view as written.
- That holds only while one column can serve the whole table. When **two relationships reference
  two different columns of the same table**, a Kumo table has one primary key and cannot carry
  both. **ASK BACK**: name both relationships and the column each references, say only one can be
  an edge destination, and ask for a corrected view, or for explicit authorization to build without
  one named relationship. Asking only which column to key on is not enough, because either answer
  still leaves the other relationship unbuildable. Do not silently key on one and leave the
  other relationship unbuilt, and do not emit the table twice to satisfy both: separate copies are
  not joined to each other, so `A -> Entity <- B` would silently become two disconnected
  components, and an entity split this way is ambiguous as the `FOR EACH` target.
- Ignore the view's dimensions, facts and metrics. They describe aggregate reporting, not the
  graph. A metric is not the user's outcome: if the user has not described one, `prediction_target`
  is missing, as for any source. If they have ("completed orders", "fraud events"), map their
  words to columns as for any source and show the mapping `[inferred]` at the gate — do not ask
  which column or value implements it.

## Tag precedence

Precedence: `[supplied]` > `[declared]` > `[inferred]`. `[inferred, unverified]` is not a rung
on that ladder but a weaker form of `[inferred]`: whatever its source, a value carrying it must
be confirmed before the table can be an entity or an edge destination.

## In the graph proposal

When `source_scope` is a semantic view, the first line names the view and the schema it resolved
to, and keys and links read from it carry `[declared]`:

```
Source scope: semantic view SALES_DB.ML.CHURN_SV [supplied] — 3 base tables in SALES_DB.ML
...
- orders: primary key order_id [declared]
- orders.customer_id -> customer_profile.customer_id [declared]
```

## Examples

### Valid view → graph proposal

> "Train a churn model on the semantic view `SALES_DB.ML.CHURN_SV`. Entity: each customer.
> Outcome: no order in the next 60 days."

Node 1a: `source_scope` is the semantic view ✓ — nothing else is asked. At Node 1c the agent runs
`DESCRIBE SEMANTIC VIEW`, finds three base tables, all in `SALES_DB.ML`, and replies — then stops:

> Source scope: semantic view `SALES_DB.ML.CHURN_SV` [supplied] — 3 base tables in `SALES_DB.ML`
> Table mode: all-connected [default] — 3 of 3 tables reachable from the seeds
>
> **Metadata**
> - customer_profile: primary key customer_id [declared]
> - orders: primary key order_id [declared]; time column ordered_at [inferred]
> - support_tickets: primary key ticket_id [declared]; time column opened_at [inferred]
>
> **Links**
> - orders.customer_id -> customer_profile.customer_id [declared]
> - support_tickets.customer_id -> customer_profile.customer_id [declared]
>
> Confirm, or correct any line — your values take precedence.

The proposal uses the physical names (`customer_profile`, not the view's logical `CUSTOMERS`),
lists only the view's three tables even if `SALES_DB.ML` holds more, and the connector is built on
`SALES_DB` / `ML`. The view's metrics play no part — the outcome is still the user's words.

### Base tables span more than one schema → ASK BACK

> "Predict 30-day churn for each customer using the semantic view `SALES_DB.ML.CHURN_SV`.
> Churn means no order in the window."

Preflight: `DESCRIBE SEMANTIC VIEW` finds base tables in two schemas — `customer_profile` in
`SALES_DB.CRM`, `orders` in `SALES_DB.ML`.

> "`CHURN_SV` reads from two schemas — `SALES_DB.CRM` (customer_profile) and `SALES_DB.ML`
> (orders). I can only build a graph from tables in a single schema. Could you point me to a
> semantic view or a schema where these tables sit together?"

Not done: picking one schema, dropping a table, or copying data across schemas.
