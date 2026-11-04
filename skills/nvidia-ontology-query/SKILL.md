---
name: nvidia-ontology-query
description: >-
  Query Auto Ontology and validate generated SQL, rows, and
  answers. Use for MCP or REST access, readiness, authentication, conversations,
  and grounded questions.
license: Apache-2.0
metadata:
  version: "0.2.1"
  author: "NVIDIA <opensource@nvidia.com>"
  tags:
    - nvidia-ontology
    - mcp
    - api
    - agents
    - aiq
---

<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Auto Ontology grounded queries

## Purpose

Query the current Auto Ontology implementation from an agent without prescribing a
harness, SDK, or LangGraph template. Typed workflow artifacts, statuses, gates,
and handoffs are in [runtime-contract.yaml](assets/runtime-contract.yaml).

## Prerequisites

Use a running Auto Ontology deployment and an authenticated MCP or REST client. If Auto Ontology is
not running, use `nvidia-ontology-setup` first.

## Limitations

This skill reads and validates answers. It does not expose FastAPI directly,
write semantic definitions, publish governed results, or provide a raw-schema
browser. Hand mutations and publication to `nvidia-ontology-management`.

## Instructions

1. Call readiness before asking a question.
2. Discover terms and confirm answerability through the semantic layer.
3. Ask through MCP, or use the authenticated Next.js REST gateway as fallback.
4. Validate material intent, SQL, rows, truncation, and prose before presenting
   a claim-bearing answer.

## Prefer MCP for reads

When the harness speaks MCP, use it. The **handshake tool list** is the live
source of truth (`mcp/auto_ontology_mcp/tools.py`). Do not copy a tool table into the
session; if the handshake and this skill disagree, trust the handshake.

Sequence (the server also advertises this so models do not jump straight to
`ask_question`):

1. `check_readiness` — compiled semantic layer **and** a live DB connection.
2. `search_terms` — what the nouns mean here.
3. `check_answerable` — cheap coverage check.
4. `ask_question` — full text-to-SQL; tens of seconds; answer + SQL + rows
   (capped at 100 rows, with `row_count` / `truncated`).
5. For a claim-bearing answer, validate material intent, generated SQL, returned
   rows, truncation, and prose using
   [query-validation.md](references/query-validation.md).

Everything on MCP **reads**. There is no MCP tool for raw databases, schemas,
or columns on purpose. `describe_table` is the legitimate table view: terms
and SQL attributes the table participates in.

Connect: `AUTO_ONTOLOGY_API_URL` is the **web app**. Run `auto-ontology-mcp`, point the client at
`…/mcp`, user signs in. No token on the MCP server. Details:
repository `mcp/README.md` and `docs/mcp.md`.

## REST fallback and write handoff

When MCP is unavailable, public clients call the authenticated **Next.js
gateway**, not FastAPI
`:3001`. FastAPI trusts `x-auto-ontology-user-id` and must not be exposed. A direct
FastAPI request without `conversation_id` is stateless; with
`conversation_id` it requires that internal header.

Auth (any of these; resolved in one place):

- Browser session cookie
- API token: `x-api-key: $AUTO_ONTOLOGY_API_TOKEN` (`Authorization: Bearer` works for
  `auto_ontology_…` tokens too). Token acts as its owner.
- OAuth bearer issued by Auto Ontology (MCP sign-in)
- SSO id token (`Authorization: Bearer <jwt>`), e.g. AI-Q — see
  [stack.md](references/stack.md)

Shapes: `docs/openapi/auto-ontology-api.json`. Do not scrape all ~87 operations. A write
request hands off to `nvidia-ontology-management`; this skill does not turn a read-only
question into a mutation.

### Chat

`POST /api/chat/completions` — SSE `step` / `result` / `error` / `charts`,
then `[DONE]`. Permission `chat:use`; a request with `conversation_id` also
needs `conversation:write`, or it gets 403.

- Omit `conversation_id` for a one-shot (no history, no chart step).
- Supply a client-generated UUID to create or continue a thread (needs
  `chat:use` and `conversation:write`).
- Wait for `[DONE]` before the next turn; overlapping requests return
  **`409 Conversation in progress`**.
- A 404 on a conversation id means it belongs to another user.

Python (from the Auto Ontology README; no extra SDK):

```python
import os
import requests

session = requests.Session()
session.headers["x-api-key"] = os.environ["AUTO_ONTOLOGY_API_TOKEN"]

terms = session.get("https://ontology.example.com/api/terms").json()

answer = session.post(
    "https://ontology.example.com/api/chat/completions",
    json={"question": "How many orders shipped last week?"},
)
# response is SSE, not a single JSON object
```

Open an API-created thread in the UI at `/chat?focus=<conversation_id>` when
it belongs to the signed-in user.

### Discovery (semantic layer, not a catalog browser)

MCP first. REST fallback:

- `GET /api/terms` (`query`, `skip`, `limit`)
- `GET /api/terms/{term_id}`
- `GET /api/exploration/tables/{table_id}/details`
- `GET /api/exploration/graph`

Writes and compile reset: `nvidia-ontology-management`.

## Validate claim-bearing answers

Do not treat successful execution as sufficient evidence. Before presenting a
decision-facing result, apply
[query-validation.md](references/query-validation.md) to check population,
measure, unit, grain, joins, lineage, status, validity, rows, truncation, and
answer prose. If a material constraint cannot be verified, return the gap rather
than a stronger claim.

## Empty or failed answers

`ask_question` / chat returned nothing useful → `check_readiness` (or
`GET /api/semantic-compilation/status`) before rewriting the question. A
named database is not proof SQL can execute. See `nvidia-ontology-setup`
troubleshooting.

## Examples

- For a grounded count, run `check_readiness`, `search_terms`,
  `check_answerable`, and then `ask_question`.
- If a response is truncated or violates measure grain, refuse the complete
  claim and follow [query-validation.md](references/query-validation.md).

## Troubleshooting

For empty or failed answers, check readiness and compilation status before
rephrasing. For REST errors, verify the public web origin, authentication, and
conversation ownership before retrying.

## See also

- [stack.md](references/stack.md) — AI-Q, Nemotron, cuOpt (what exists vs
  what partners wire)
- [query-validation.md](references/query-validation.md) — material intent, SQL,
  rows, and answer checks
- `nvidia-ontology-management` — model, modify, and publish through the layer
- `nvidia-ontology-setup` — bring-up and MCP OAuth discovery
