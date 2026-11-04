---
name: nvidia-ontology-setup
description: >-
  Set up or troubleshoot the Auto Ontology runtime. Use for Helm (the official
  install), Docker Compose, developer setup, and MCP connection to an existing
  deployment.
license: Apache-2.0
metadata:
  version: "0.3.1"
  author: "NVIDIA <opensource@nvidia.com>"
  tags:
    - nvidia-ontology
    - install
    - docker
    - helm
    - mcp
    - troubleshooting
---

<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Auto Ontology setup

## Purpose

Bring up the current Auto Ontology implementation, troubleshoot it, or connect an agent
to an instance that is already running. Do not invent a second installer:
the official installation is the Helm chart in repository-root `DEPLOYMENT.md`;
Docker Compose runs the same stack on one machine. `dev_tools/setup_env.sh` is a
developer tool, not an installer. Typed setup, connection, ingestion,
compilation, and readiness artifacts are in
[runtime-contract.yaml](assets/runtime-contract.yaml).

On failure, read [troubleshooting.md](references/troubleshooting.md) instead of
searching the web.

## Instructions

### Required questions

Ask these if not already clear. Do not guess a default and emit a command.

1. **Target** — Helm on Kubernetes (the official install, and the default
   recommendation), Docker Compose on one machine, a developer workflow
   (`--dev` / `--ds`, only when changing Auto Ontology itself), or "Auto
   Ontology is already up, I only need MCP"?
2. **NVIDIA NIM key** — is `DEFAULT_MODELS_API_KEY` (Helm:
   `defaultModelsApiKey`) available? Chat and ingest need it.
3. **Admin account** — the email and password for the bootstrap admin.
   Self-service sign-up is disabled, so this is the only way to sign in.
4. **Source database** — a connection string (`CONNECTION_STRINGS` /
   `connectionStrings`), or will connections be added in the UI?
5. **Helm only** — which chart version, and the URL users will browse to
   (`appUrl`).
6. **Compose only** — is host `5432` free? `POSTGRES_PORT` only remaps the
   host side; containers still use 5432 internally.

Never ask the user to paste secrets into the conversation, and never put them
in command arguments. Have the user write them into a values file or `.env`.

## Install with Helm (official)

Releases are published to the public NGC `nvidia` org (chart and images). Follow
`DEPLOYMENT.md` with the version the user named:

The user creates a values file in an editor (not on the command line, so
values stay out of shell history and process arguments), readable only by
them, and out of git:

```yaml
# auto-ontology-values.yaml
defaultModelsApiKey: <API-KEY>
postgresPassword: <POSTGRES-PASSWORD>
adminEmail: <ADMIN-EMAIL>
adminPassword: <ADMIN-PASSWORD>
connectionStrings: <CONNECTION-STRINGS>   # optional; or add connections in the UI
```

```bash
helm fetch https://helm.ngc.nvidia.com/nvidia/charts/auto-ontology-<VERSION>.tgz
helm install auto-ontology auto-ontology-<VERSION>.tgz -f auto-ontology-values.yaml
kubectl port-forward svc/frontend 3000:3000
```

- `adminEmail` and `adminPassword` are required; the install fails without them.
- `appUrl` must be the exact origin users browse to. The default
  (`http://localhost:3000`) only fits the port-forward above; set it for a
  NodePort, ingress, or HTTPS origin.
- `authSecret` is generated on first install and kept across upgrades; set it
  only to share one value across environments.
- If the chart fetch returns 404, that version is not published yet. Ask the
  user; do not fall back to the internal staging registry.

This skill does not cover Astra GitOps.

## Local: Docker Compose

For running the whole stack on one machine. In the repository root, the user
copies `.env.example` to `.env` and fills in `AUTH_SECRET`, `APP_URL`,
`AUTO_ONTOLOGY_ADMIN_EMAIL`, and `AUTO_ONTOLOGY_ADMIN_PASSWORD` (all required).
Then:

```bash
docker compose up -d --build
```

`AUTH_SECRET` and `APP_URL` are required. Without them every page fails with a
Better Auth "default secret" error. Fill `DEFAULT_MODELS_API_KEY` and a
`CONNECTION_STRINGS` value (or plan to add connections in the UI). The full
variable list is `.env.example`.

This builds `auto-ontology` and `auto-ontology-frontend`, starts Postgres,
pgAdmin, and the ingestion service, runs two one-shot migrate jobs, and starts
the app:

- `auto-ontology-migrate` applies Alembic migrations to the backend's `public`
  schema (catalog and semantic tier). The backend waits for it.
- `frontend-migrate` syncs the Prisma `frontend` schema: users, sessions, API
  keys, and the OAuth tables MCP sign-in uses. The frontend waits for it.

If sign-in or MCP login fails, check both with `docker compose ps -a` and
`docker compose logs frontend-migrate`. Read the `frontend-migrate` log even
when it exited 0: it has finished without an error before while leaving MCP
login broken.

| Service | URL |
| --- | --- |
| UI | http://localhost:3000 |
| FastAPI (internal) | http://localhost:3001 |
| Ingestion | http://localhost:3002 |
| pgAdmin | http://localhost:5050 |
| Postgres | localhost:`$POSTGRES_PORT` (from `.env`, default 5432) |

## Developer workflows (`--dev`, `--ds`)

Only for people changing Auto Ontology itself; these are not installations.
`dev_tools/setup_env.sh` starts part of the stack in Docker and leaves the rest
to run from the checkout. Both need the `.env` above plus:

```bash
cd frontend && pnpm install   # Next.js
# from repo root:
uv sync
```

`--dev` runs infra only (Postgres, pgAdmin, ingestion); you run Next.js and FastAPI:

```bash
./dev_tools/setup_env.sh --dev
cd frontend && pnpm dev
# repo root:
uv run uvicorn auto_ontology.server.__main__:create_app --factory --reload --host 127.0.0.1 --port 3001
```

The app factory is `create_app()` in `auto_ontology/server/__main__.py`. There is no
`auto_ontology/server/main.py`. `uv run python -m auto_ontology.server` is the same app without
reload (what the container runs).

`--ds` runs the frontend in Docker and FastAPI on the host:

```bash
./dev_tools/setup_env.sh --ds
uv run python -m auto_ontology.server
```

The frontend image bakes `PYTHON_API_URL=http://host.docker.internal:3001`, so
the API must accept connections from the container, not just loopback.
`python -m auto_ontology.server` binds `0.0.0.0:3001` (what `setup_env.sh --ds`
prints); a `--host 127.0.0.1` uvicorn is unreachable from the container on
Linux. Because FastAPI trusts `x-auto-ontology-user-id`, keep port 3001
firewalled from other machines. On native Linux Docker you may need
`--add-host=host.docker.internal:host-gateway`.

## Limitations

This skill uses the repository's existing Helm chart and Compose file and does
not cover Astra GitOps, change semantic definitions, or treat partial Vault
configuration as secret storage. Ask before destructive volume deletion.

## Auto Ontology is already up — MCP only

Do not reinstall. Point `AUTO_ONTOLOGY_API_URL` at the **web app** (Compose UI is
`:3000`, not FastAPI `:3001`):

```bash
AUTO_ONTOLOGY_API_URL=http://localhost:3000 uvx --from "git+https://github.com/NVIDIA/auto-ontology.git#subdirectory=mcp" auto-ontology-mcp
```

Until the package is on PyPI this needs GitHub credentials that can read
`NVIDIA/auto-ontology`. Client config uses the MCP server URL with a `/mcp` suffix.
People sign in through Auto Ontology; do not put a token on the MCP server.

Confirm the deployment is new enough to be an authorization server:

```bash
curl -s -o /dev/null -w '%{http_code}\n' "$AUTO_ONTOLOGY_API_URL/.well-known/oauth-authorization-server"
```

`200` is required. Anything else: run the frontend from the checkout
(`pnpm dev`) and point `AUTO_ONTOLOGY_API_URL` at that port.

## Examples

- Kubernetes: install the published Helm chart from `DEPLOYMENT.md` with the
  admin account and `appUrl` set.
- One machine: `docker compose up -d --build` after filling `.env`.
- Existing deployment: do not reinstall; configure `AUTO_ONTOLOGY_API_URL` and verify
  OAuth discovery before connecting the MCP client.

## Connect a source

Connections come from one of two places, and only one is used:

- **UI-managed** (Settings → Connections, or the API below). When any exist,
  `CONNECTION_STRINGS` is ignored.
- **`CONNECTION_STRINGS`** in `.env` is a fallback used only when there are no
  UI-managed connections. `GET /api/connections/source` reports whether it is
  set; it cannot carry schema or table filters.

To add a UI-managed connection (permission `connection:manage`):

1. `POST /api/connections/test` validates credentials and returns the schemas
   for the allowlist. A 422 carries the driver's error; fix it before creating.
2. `POST /api/connections` stores the connection. The 201 only means it was
   saved: ingestion is triggered best-effort and may fail or still be running.
3. Verify it landed: `GET /api/datasources/dbs` (`catalog:read`) lists the
   database with the expected tables, `GET /api/semantic-compilation/status`
   reports progress, and MCP `check_readiness` reports no blockers.

Scope is set per connection: a schema allowlist, plus `table_allow_regex` /
`table_deny_regex` applied at catalog extraction (deny wins; unqualified,
case-sensitive names). Env connections have neither. The full table-level scope
guarantee is still open in
[#255](https://github.com/NVIDIA/auto-ontology/issues/255), so if particular tables must never be
catalogued, sampled, or embedded, say that this is not yet enforceable
end-to-end rather than implying it is.

## Verify

1. UI loads at http://localhost:3000 (or the deployed `APP_URL` / Helm `appUrl`) and shows
   the left navigation. Missing nav → stale cookie; see
   [troubleshooting.md](references/troubleshooting.md).
2. `GET /api/semantic-compilation/status` on the **web** origin authenticates
   (cookie, `x-api-key`, or SSO bearer) and returns JSON. `calculated: false`
   means the glossary is empty — ingest / compile, do not keep retrying chat.
3. MCP: OAuth discovery returns 200 (above). Then `check_readiness` once a
   client is connected.

## Stop

```bash
helm uninstall auto-ontology  # Helm; also deletes the postgres-data PVC and its data
docker compose down           # Compose; keep volumes
docker compose down -v        # Compose; wipe Postgres / pgAdmin data
```

Both `helm uninstall` and `docker compose down -v` destroy data; ask first.

## Troubleshooting

Use [troubleshooting.md](references/troubleshooting.md) for stale sessions,
model-key failures, embedding mismatches, port conflicts, and partial Vault
configuration. Redact secrets before sharing command output as evidence.

## See also

- [troubleshooting.md](references/troubleshooting.md)
- [runtime-contract.yaml](assets/runtime-contract.yaml)
- `nvidia-ontology-query` — call Auto Ontology once it is running
- `nvidia-ontology-management` — edit the semantic layer
- `CLAUDE.md` — maintain Auto Ontology itself
