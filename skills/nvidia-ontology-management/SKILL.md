---
name: nvidia-ontology-management
description: >-
  Model and publish semantic definitions in Auto Ontology. Use for terms, relationships,
  measures, imports, and governed results—not deployment or querying.
license: Apache-2.0
metadata:
  version: "0.2.1"
  author: "NVIDIA <opensource@nvidia.com>"
  tags:
    - nvidia-ontology
    - ontology
    - glossary
    - sql-attributes
    - catalog
---

<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Auto Ontology management and modeling

## Purpose

Inspect and change Auto Ontology's semantic layer without confusing a successful API call
with correct business meaning or durable publication. The machine-readable
inputs, outputs, statuses, gates, and handoffs are in
[runtime-contract.yaml](assets/runtime-contract.yaml).

Choose the focused workflow before acting:

- routine inspection or metadata edits: continue below and use
  [write-api.md](references/write-api.md);
- concepts, relationships, measures, units, grain, or policies: use
  [modeling.md](references/modeling.md);
- model promotion or reusable analytical results: use
  [publication.md](references/publication.md).

MCP tools only **read**. Create, patch, import, and compile-reset go through
the Next.js `/api/...` gateway. Governed result rows require an approved source
writer; Auto Ontology has no generic result-row writeback endpoint. Request bodies and
field types live in `docs/openapi/auto-ontology-api.json`; this skill names operations and
permissions only.

## Prerequisites

Use a running, authenticated Auto Ontology deployment. Use `nvidia-ontology-setup` when
the deployment or semantic layer is not ready.

## Limitations

MCP is read-only, Auto Ontology has no generic result-row writeback endpoint, and current
model import and certification APIs do not prove atomic promotion. Follow the
publication gates below and fail closed when exact readback is unavailable.

## Auth

Scripts use an API token (`x-api-key` or `Authorization: Bearer`). Mint it in
the UI (user menu → API Tokens). A token **acts as its owner** — a viewer's
token cannot do admin things. Creating and revoking tokens requires a signed-in
session; a token cannot mint another token.

Writes below need `catalog:edit` unless noted. `403` means the owner's role
lacks that permission, not that the path is wrong.

## Instructions

1. **Discover current meaning** via MCP (`search_terms`, `get_term`,
   `get_term_columns`, `get_term_sql_attributes`, `describe_table`) or REST
   if MCP is absent (`GET /api/terms`, `GET /api/terms/{term_id}`,
   `GET /api/exploration/tables/{table_id}/details`).
2. **Resolve the semantic contract** when meaning changes. Record source
   binding, identity, population, relationship cardinality, measure expression,
   unit, grain, denominator, time, validity, and counterexamples; see
   [modeling.md](references/modeling.md).
3. **State the proposed change** to the user (term rename, new SQL attribute,
   import, and so on). Do not silently rewrite the glossary.
4. **Validate SQL** before create/update: `POST /api/sql-attributes/validate`.
   A parse failure is **HTTP 422**, not `valid: false`.
5. **Apply** the smallest write that matches the request (see
   [write-api.md](references/write-api.md)).
6. **Compilation is not a hidden side effect.** Check
   `GET /api/semantic-compilation/status`. Do **not** call
   `POST /api/semantic-compilation/reset` as cleanup — it is an asynchronous,
   destructive rebuild of every database's compiled layer, returns 202, and
   requires `semanticCompilation:manage`.
7. **Verify the persisted change**, not merely request success. Model imports
   require a scoped backup and exact re-export comparison; see
   [publication.md](references/publication.md). Then use MCP
   `check_answerable` / `ask_question` (or REST
   `POST /api/question-entity-coverage` and `POST /api/chat/completions`) for
   positive and negative behavior checks.

## Semantic relationships and "what does this dataset mean"

Stay on the semantic layer. Do not browse raw schemas to answer meaning.

- Term → columns: MCP `get_term_columns` or
  `GET /api/terms/{term_id}/column-attributes`
- Term → derived SQL: MCP `get_term_sql_attributes`
- Table → terms and SQL attributes: MCP `describe_table` or
  `GET /api/exploration/tables/{table_id}/details`
- Semantic hop chain between two terms:
  `GET /api/exploration/terms/{term_id}/path/{other_term_id}`
- Semantic graphs: `GET /api/exploration/graph`,
  `GET /api/exploration/semantic-graph`

These are semantic relationship paths, not generation provenance, source
revision, certification history, or version lineage.

## Bulk import / export

- `POST /api/model/export` — YAML of catalog + semantic layer.
  Permission `modelInterchange:export`. Body may set catalog database **IDs**
  in `databases` (empty = all) and `format` (`auto_ontology` or `ossie`).
- `POST /api/model/import` — multipart YAML; native Auto Ontology
  (`data_layer` / `semantic_layer`) or Apache Ossie (`version`, `name`,
  `datasets` at the root).
  Query `replace` (default true) and `embed` (default true). Permission
  `modelInterchange:import`. Can replace existing data — confirm with the
  user before `replace=true`, apply first in isolation, and use the exact
  readback workflow in [publication.md](references/publication.md).

## Examples

- To define a run-grain measure, follow the evidence and counterexample workflow
  in [modeling.md](references/modeling.md).
- To revise and promote a model, back up the exact scope and follow the staged
  readback workflow in [publication.md](references/publication.md).

## Troubleshooting

For write failures, verify the authenticated owner's permission, distinguish
HTTP 422 parse failures from validation results, and inspect compilation status
before changing the model or resetting compilation.

## See also

- [write-api.md](references/write-api.md)
- [modeling.md](references/modeling.md)
- [publication.md](references/publication.md)
- `nvidia-ontology-query` — how to call Auto Ontology and validate query results
- `nvidia-ontology-setup` — deployment not ready
- `mcp/auto_ontology_mcp/tools.py` — live read-tool allow-list
