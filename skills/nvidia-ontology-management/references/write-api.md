<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Ontology write API (index)

Canonical request/response shapes: `docs/openapi/auto-ontology-api.json`. Do not paste
schemas here. Permissions are `x-auto-ontology-permissions` on each operation.

Call the **Next.js** origin (`APP_URL`, local `:3000`), not FastAPI `:3001`.

## Terms

| Method | Path | Permission | Notes |
| --- | --- | --- | --- |
| GET | `/api/terms` | `catalog:read` | Query `query`, `skip`, `limit` |
| GET | `/api/terms/{term_id}` | `catalog:read` | |
| PATCH | `/api/terms/{term_id}` | `catalog:edit` | `TermUpdate`: `name`, `description`, `name_certified`, `description_certified`. Renaming invalidates cached SQL-attribute description suggestions. |

There is no public "create empty term" POST on `/api/terms`. New terms arrive
through compilation, ingest, or model import.

## Column attributes

| Method | Path | Permission | Notes |
| --- | --- | --- | --- |
| GET | `/api/terms/{term_id}/column-attributes` | `catalog:read` | Physical columns a term maps to |
| PATCH | `/api/terms/{term_id}/column-attributes/{attr_id}` | `catalog:edit` | Name/description/certified/sample_values. `{attr_id}` is a ColumnAttribute id, not a SqlAttribute id. Refreshes the semantic embedding. |

## SQL attributes

| Method | Path | Permission | Notes |
| --- | --- | --- | --- |
| POST | `/api/sql-attributes/validate` | `catalog:edit` | Body: `expression`, optional `term_id`, optional `attribute_id`. Parse failure is **422**, not `valid: false`. |
| POST | `/api/sql-attributes` | `catalog:edit` | `SqlAttributeCreate`: required `name`, `description`, `expression`, `term_id`; optional `source` (default `manual`). 201. |
| GET | `/api/sql-attributes/{attr_id}` | `catalog:read` | Also MCP `get_sql_attribute` |
| PATCH | `/api/sql-attributes/{attr_id}` | `catalog:edit` | Metadata only (name/description). Does **not** change the SQL expression. |
| PUT | `/api/sql-attributes/{attr_id}` | `catalog:edit` | Re-parses SQL and refreshes the embedding. Same required fields as create. |
| DELETE | `/api/sql-attributes/{attr_id}` | `catalog:edit` | Graph cleanup plus embedding deletion |
| GET | `/api/terms/{term_id}/sql-attributes` | `catalog:read` | List under one term |
| GET | `/api/sql-attributes/{attr_id}/description-suggestion` | `catalog:edit` | LLM suggestion for the edit flow |

## Catalog node descriptions

| Method | Path | Permission | Notes |
| --- | --- | --- | --- |
| PATCH | `/api/nodes/{node_id}` | `catalog:edit` | Mutable properties of a Database, Schema, Table, or Column node |

## Compilation

| Method | Path | Permission | Notes |
| --- | --- | --- | --- |
| GET | `/api/semantic-compilation/status` | `chat:use` | Global status. `calculated` only means at least one Term exists; `running: false` does not prove the ingestion service is reachable. |
| PUT | `/api/configurations/semantic-compilation` | `semanticCompilation:manage` | Set `enabled`; enabling triggers a run best-effort. |
| POST | `/api/semantic-compilation/reset` | `semanticCompilation:manage` | **Destructive and asynchronous.** Accepts a delete-and-rebuild of every database's compiled layer. 202 is not completion; poll and verify. Not cleanup after a small edit. |

## Model interchange

| Method | Path | Permission | Notes |
| --- | --- | --- | --- |
| POST | `/api/model/export` | `modelInterchange:export` | JSON body `ExportRequest`: catalog database **IDs** in `databases` (empty = all), `format` `auto_ontology` \| `ossie`. Response is YAML. |
| POST | `/api/model/import` | `modelInterchange:import` | YAML import; request forms and query parameters are in the OpenAPI spec. Pitfall: `replace` defaults to true, so confirm with the user before importing over existing definitions. |

For mutation and publication safeguards, use
[publication.md](publication.md). Request success is not persisted parity.

## Reusable analysis definitions

These publish reusable definitions, not arbitrary external result rows:

| Method | Path | Permission | Notes |
| --- | --- | --- | --- |
| POST | `/api/custom-analyses/validate` | `analysis:manage` | Validate SQL against catalogued tables; parse or resolution failure is 422. |
| POST/PUT/DELETE | `/api/custom-analyses` or `/api/custom-analyses/{analysis_id}` | `analysis:manage` | Create, replace, or delete supported SQL definitions. Name or SQL conflicts return 409. |
| GET | `/api/custom-analyses` | `analysis:read` | List reusable SQL definitions. |
| POST/PUT/DELETE | `/api/pql-analyses` or `/api/pql-analyses/{analysis_id}` | `analysis:manage` | Create, replace, or delete PQL definitions. PQL is validated at prediction time. |
| GET | `/api/pql-analyses` | `analysis:read` | List reusable PQL definitions. |

## Exploration (read; semantic relationships)

All `catalog:read`:

- `GET /api/exploration/tables/{table_id}/details`
- `GET /api/exploration/terms/{term_id}/details`
- `GET /api/exploration/terms/{term_id}/path/{other_term_id}`
- `GET /api/exploration/graph`
- `GET /api/exploration/semantic-graph`
- `GET /api/exploration/edges`
- `GET /api/exploration/nodes/{node_id}/relationships`

## Verify after a write

For model import, first re-export the same database-ID scope and compare exact
persisted structure and properties. Then use MCP `check_answerable` and
`ask_question` for positive and negative behavior checks. REST equivalents:

- `POST /api/question-entity-coverage`
- `POST /api/chat/completions` (SSE; `chat:use`)
