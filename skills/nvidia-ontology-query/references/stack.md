<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Auto Ontology in a broader stack

Honest map of what this repository implements versus what a partner wires.
Do not invent a first-party blueprint that is not here.

## AI-Q / NemoClaw (auth contract)

Auto Ontology normally authenticates browsers with a Better Auth cookie. AI-Q (and
similar NVIDIA SSO callers) forward the **user's SSO id token** as
`Authorization: Bearer <jwt>`. Auto Ontology verifies signature, issuer, and expiry
against the configured SSO provider's JWKS — the same issuer as interactive
SSO. Audience is **not** checked: the token was minted for AI-Q's OAuth
client, so `aud` is AI-Q's client id, not Auto Ontology's. Trusting any token that SSO
provider signed is the intended model. No shared service secret.

The bearer is mapped to an Auto Ontology user by email, then by SSO account subject.
There must already be a matching Auto Ontology user.

This is an auth contract, not instructions for building an AI-Q blueprint or
NemoClaw skill pack.

Credential order on every `/api` route (`frontend/auth/resolve-user.ts`):

1. Session cookie
2. Auto Ontology API token (`auto_ontology_…`, `x-api-key` or Bearer)
3. OAuth bearer issued by Auto Ontology (MCP)
4. SSO id token (AI-Q)

A revoked API token is a 401; it is not retried as an SSO JWT.

## Nemotron / NIM (deployment config)

Auto Ontology calls NVIDIA NIM for chat, embeddings, and rerank. That is `.env` /
Helm values, not a Nemotron training or customize integration.

Triplets: `<PREFIX>_ENDPOINT`, `<PREFIX>_API_KEY`, `<PREFIX>_MODEL`. Unset
fields fall back to `DEFAULT_MODELS_*`. Prefixes: `DEFAULT_MODELS`,
`REASONING` (NL-to-SQL), `NON_REASONING` (entity extraction), `EMBED`,
`RERANK`. `*_API_KEY` also falls back to `NVIDIA_API_KEY`.

`EMBED_ENDPOINT` + `EMBED_MODEL` must be the same at ingest and query.
Defaults in `.env.example` include a Nemotron embed model id; changing it
without re-ingest breaks retrieval.

## cuOpt (partner-wired)

This repository has **no** cuOpt client, skill, or job bridge. Auto Ontology returns
natural-language answers, the SQL it ran, and rows. A partner agent may take
those numbers and call cuOpt. Use NVIDIA's `cuopt-*` skills for formulation
and solve. Do not claim Auto Ontology schedules routes or solves LPs itself.

## Calling rules that bite in a stack

- Partners call the Next.js gateway (`APP_URL` / `:3000`), never public
  FastAPI `:3001`.
- MCP holds no deployment-wide identity. Each caller signs in as themselves.
- `conversation_id` is per Auto Ontology user. Do not share UUIDs across users.
- Overlapping turns on one conversation: `409`.
