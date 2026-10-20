# Deployment endpoint resolution

`vss-build-vision-ai` owns publication of the public endpoints that operate
skills consume. Operate skills do not discover Helm releases, Services, or
hostnames; the completed deployment supplies the public origin as
`VSS_PUBLIC_URL`.

For the catalog-level overview, see `skills/README.md`. This file is the
canonical contract for mapping deployment settings to operator-facing public
endpoint variables.

## Deployment output: `VSS_PUBLIC_URL`

A Kubernetes deployment must publish one public Ingress origin, including the
scheme and any non-default port:

```bash
# base / lvs profile example (same main-host pattern)
VSS_PUBLIC_URL=https://vss.example.com
# search profile example
VSS_PUBLIC_URL=https://vss-search.example.com
```

The deployment workflow must surface this value to the operator. Operate skills
trim trailing slashes before building derived URLs. The variable name does not
encode the profile; the deployment publishes one origin and each skill uses the
routes that profile exposes.

### Shared Helm / runtime name mapping

Operate skills take **one** public origin: `VSS_PUBLIC_URL`. Helm and agent pods
may label that same host differently; treat those names as deploy-side aliases,
not additional operate inputs:

| Operate-skill name | Helm / runtime equivalents |
|---|---|
| `VSS_PUBLIC_URL` | `global.externalHost`, main Ingress host, `AGENT_BASE_URL`, `VSS_AGENT_EXTERNAL_URL`, and deploy-minted `VST_EXTERNAL_URL` when it equals that origin |
| `VSS_VIOS_URL` | `${VSS_PUBLIC_URL}/vst` |
| `VST_API_BASE` | `${VSS_VIOS_URL}/api/v1` — all VIOS `curl` targets |
| `VLM_ENDPOINT` | `${VSS_PUBLIC_URL}/rtvi-vlm/v1` on every profile — RT-VLM is mounted at its service name, never at the origin root |
| `LVS_BACKEND_URL` / `VIDEO_SUMMARIZATION_URL` | `${VSS_PUBLIC_URL}/lvs` on Kubernetes (the mount, **no** `/v1` suffix); Docker Compose remains `http://${HOST_IP}:38111` |
| `VSS_STREAMER_URL` | Separate streamer Ingress host (`streamer.<ip>.nip.io`); **not** under `/vst`; search (and other NvStreamer-bearing) profiles only |

Do not make operate skills invent a Brev or nip.io hostname. The deployment
workflow publishes the public origin; operate skills consume it as
`VSS_PUBLIC_URL` only. Do not require a second operate variable named
`VST_EXTERNAL_URL`. `VSS_ENDPOINT` is a legacy alias for `VSS_PUBLIC_URL` in
`vss-ask-video`; prefer `VSS_PUBLIC_URL`.

### Consumer-derived public endpoints

Operate skills may derive these values from the deployment output:

```bash
VSS_PUBLIC_URL="${VSS_PUBLIC_URL%/}"
VSS_VIOS_URL="${VSS_VIOS_URL:-${VSS_PUBLIC_URL}/vst}"
VST_API_BASE="${VST_API_BASE:-${VSS_VIOS_URL}/api/v1}"
```

RT-VLM is mounted at `/rtvi-vlm` on every profile, so a skill needing direct VLM
access has one candidate to confirm. Confirm it rather than assuming — a
deployment that does not run RT-VLM has no such route, and one predating the
shared route table published it at the origin root instead:

```bash
if [ -z "${VLM_ENDPOINT:-}" ] && [ -n "${VSS_PUBLIC_URL:-}" ]; then
  # The canonical mount. The bare root is a fallback for deployments older than
  # the shared route table; drop it once none are left in the field.
  for _candidate in "${VSS_PUBLIC_URL%/}/rtvi-vlm/v1" "${VSS_PUBLIC_URL%/}/v1"; do
    if _models=$(curl -sf --max-time 5 "${_candidate}/models") \
      && _model=$(printf '%s' "${_models}" | jq -r '.data[0].id // empty') \
      && [ -n "${_model}" ]; then
      VLM_ENDPOINT="${_candidate}"
      VLM_MODEL="${VLM_MODEL:-${_model}}"
      break
    fi
  done
fi
```

The model-id guard is what makes this safe against a catch-all: a UI page
answers `curl -sf` but yields no `.data[0].id`, so that candidate is rejected
rather than adopted.

Shareable media URLs from `/url` endpoints may embed an internal host at mint
time. On Kubernetes, compare and validate against `VSS_PUBLIC_URL`; do not
substitute localhost or reconstructed URLs for returned screenshot or clip
links.

### Base profile public routes

Main host pattern: `vss.<ip>.nip.io` (Helm `dev-profile-base`). Base does **not**
require a streamer or Kibana host. Default Ingress exposes Agent, UI, VIOS, and
RT-VLM:

| Capability | Public endpoint |
|---|---|
| Agent readiness (K8s) | `vss configure check` reports the agent group; raw fallback `GET ${VSS_PUBLIC_URL}/api/v1/videos` — the CLI's own agent probe. `/health` and `/openapi.json` are not readiness endpoints |
| Agent API / chat | `${VSS_PUBLIC_URL}/api/...`, `/websocket`, `/chat` |
| VIOS list/inspect/clips | `GET ${VST_API_BASE}/sensor/list`, storage `/url`, replay `/picture` |
| Direct VLM (ask / report Mode A) | `${VSS_PUBLIC_URL}/rtvi-vlm/v1` → `GET …/models`, `POST …/chat/completions` |
| Phoenix (optional) | `${VSS_PUBLIC_URL}/phoenix` |

Base operate skills for the quickstart walkthrough:

- `vss-manage-video-io-storage` — VIOS via `${VST_API_BASE}`
- `vss-ask-video` — VIOS clip URL + direct VLM at `${VLM_ENDPOINT}` (**not** Agent `/generate`)
- `vss-generate-video-report` Mode A — same VIOS + VLM path (**not** Agent `/generate`)

Do **not** use `${VSS_PUBLIC_URL}/vlm/v1`. Stock base Helm exposes RT-VLM under
**`/rtvi-vlm`**, the same mount as every other profile; neither base Helm nor
Docker HAProxy serves `/vlm`. On Docker Compose, the VLM is not published on the
public origin — use the host port (`:30082` for NIM or `:8018` for RT-VLM).

### Search profile public routes

Main host pattern: `vss-search.<ip>.nip.io` (Helm `dev-profile-search`). Search
adds archive search and a separate NvStreamer host:

| Capability | Public endpoint |
|---|---|
| Agent search (operate) | `POST ${VSS_PUBLIC_URL}/generate` with `{"input_message": "..."}` |
| Agent ingest/delete | `${VSS_PUBLIC_URL}/api/v1/...` |
| Agent readiness (K8s) | `vss configure check` reports the agent group; raw fallback `GET ${VSS_PUBLIC_URL}/api/v1/videos` — `/health` is not on search Ingress and `/openapi.json` is not a readiness endpoint |
| VIOS list/inspect | `GET ${VST_API_BASE}/sensor/list` |
| Direct VLM (ask / report Mode A) | `${VSS_PUBLIC_URL}/rtvi-vlm/v1` → `GET …/models`, `POST …/chat/completions` |
| Elasticsearch (host CLI) | `${VSS_PUBLIC_URL}/elasticsearch` — edge guard denies PUT/DELETE, cluster-admin and two-segment mutating paths; POST still reaches ES |
| RT-Embed / RT-CV (host CLI) | `${VSS_PUBLIC_URL}/rtvi-embed/v1`, `${VSS_PUBLIC_URL}/rtvi-cv/api/v1` |
| NvStreamer HTTP | `${VSS_STREAMER_URL}/api/v1/...` — separate host, no `/vst` prefix |

Every backend is mounted by service name rather than at the origin root, so these
are the same paths `vss configure` records (`vss_cli/config.py:INGRESS_SERVICES`)
and the same ones every other profile publishes. A route is present whenever the
profile deploys that backend — there is no per-route toggle to reason about, and
`global.rtviInternalIngress` no longer publishes a second Ingress of its own; it
only routes the agent's own RT-CV / RT-Embed calls through the controller for
stream affinity.

### LVS profile public routes

Main host pattern: `vss.<ip>.nip.io` (Helm `dev-profile-lvs`) — same family as
base, not `vss-search.*`. LVS and RT-VLM each sit under their own mount, so
neither owns the origin root and both are reached the same way as on any other
profile:

| Capability | Public endpoint |
|---|---|
| LVS readiness | `GET ${VSS_PUBLIC_URL}/lvs/v1/ready` → video-summarization. `/lvs/v1/live` answers sooner: `/v1/ready` is 503 through model warmup, which reads as an absent route on a deployment that is merely still starting |
| LVS summarize | `POST ${VSS_PUBLIC_URL}/lvs/v1/summarize` → video-summarization |
| RT-VLM models / chat | `GET ${VSS_PUBLIC_URL}/rtvi-vlm/v1/models`, `POST ${VSS_PUBLIC_URL}/rtvi-vlm/v1/chat/completions` |
| Elasticsearch (host CLI) | `${VSS_PUBLIC_URL}/elasticsearch` — same edge guard as every other profile |
| VIOS list/inspect/clips | `GET ${VST_API_BASE}/sensor/list`, storage `/url`, … |

Derive the LVS client base as the **mount**, not the origin:

```bash
# Kubernetes — append /v1/ready and /v1/summarize yourself. The gateway strips
# /lvs, so the backend sees the paths it serves on :38111.
# Force the public origin; a leftover Docker LVS_BACKEND_URL must not win.
LVS_BACKEND_URL="${VSS_PUBLIC_URL%/}/lvs"
VIDEO_SUMMARIZATION_URL="${LVS_BACKEND_URL}"
# Docker Compose (unchanged)
# LVS_BACKEND_URL=http://${HOST_IP}:38111
```

Do **not** set `LVS_BACKEND_URL=${VSS_PUBLIC_URL}/lvs/v1` — that yields
`/lvs/v1/v1/ready`. Do **not** probe bare `${VSS_PUBLIC_URL}/v1`: nothing is
mounted there, so it falls through to the UI catch-all.

Because `/lvs` is a Prefix mount, everything the LVS backend serves is now
reachable under it — `${VSS_PUBLIC_URL}/lvs/models`, `/lvs/openapi.json`,
`/lvs/recommended_config`, `/lvs/metrics` — where the previous Exact-path Ingress
published only readiness and summarize. Public `/openapi.json` (no prefix) is
still the **Agent** document; never treat it as the LVS schema. Discover the
summarize model via `${VSS_PUBLIC_URL}/lvs/models`, or `${VSS_PUBLIC_URL}/rtvi-vlm/v1/models`
for RT-VLM itself, or an explicit `VLM_NAME`.

LVS operate skills for the docs walkthrough:

- `vss-manage-video-io-storage` — VIOS via `${VST_API_BASE}`
- `vss-summarize-video` — `/lvs/v1/ready` + `/lvs/v1/summarize` on `${VSS_PUBLIC_URL}`
- `vss-generate-video-report` Mode A — when LVS `/lvs/v1/ready` is 200, delegates to
  `vss-summarize-video`; otherwise VLM-direct

Clip URLs passed into `POST /v1/summarize` must stay as VIOS minted them (LVS
fetches that URL). Browser/report links may still rewrite `/storage/...` under
`/vst` using the base-profile rule. Deploy must mint media URLs the LVS pod can
reach (typically `VST_EXTERNAL_URL` equal to the public origin).

### Alerts profile public routes

Main host pattern: `vss.<ip>.nip.io` (Helm `dev-profile-alerts`) — same family as
base/lvs, not `vss-search.*`. Stock Ingress publishes Agent, VIOS, Alert Bridge,
and Video Analytics API. VA-MCP is published only when the resolved service
graph explicitly selects the legacy MCP interface (path-rewrite strips the
public prefix):

| Capability | Public endpoint |
|---|---|
| Alert Bridge health | `GET ${VSS_PUBLIC_URL}/alert-bridge/health` (not `/api/v1/health`) |
| Realtime rules | `GET`/`POST`/`DELETE ${VSS_PUBLIC_URL}/alert-bridge/api/v1/realtime` |
| Realtime incidents | `GET ${VSS_PUBLIC_URL}/alert-bridge/api/v1/realtime/incidents` |
| On-demand / verifier config | `${VSS_PUBLIC_URL}/alert-bridge/api/v1/verification/...` |
| VA-MCP health (legacy service selected only) | `GET ${VSS_PUBLIC_URL}/va-mcp/health` (rewritten to `/health`; prefer over `/mcp` or `/`) |
| VA-MCP (legacy service selected only) | `${VSS_PUBLIC_URL}/va-mcp/mcp` (rewritten to `/mcp`) |
| VIOS list/inspect | `GET ${VST_API_BASE}/sensor/list`, … |
| Agent generate | `POST ${VSS_PUBLIC_URL}/generate` — **not** for rule CRUD |
| NvStreamer HTTP | `${VSS_STREAMER_URL}/api/v1/...` — separate `streamer.*` host |

Derive Alert Bridge from the **public origin** (force; ignore leftover Docker
host-port env). Derive VA-MCP only when `vss-va-mcp` is in the resolved graph:

```bash
# Kubernetes — path-rewrite strips /alert-bridge and /va-mcp on the Service.
ALERT_BRIDGE_URL="${VSS_PUBLIC_URL%/}/alert-bridge"
if docker compose -f "$BUILD_DIR/resolved.yml" config --services |
  grep -qx vss-va-mcp; then
  VA_MCP_URL="${VSS_PUBLIC_URL%/}/va-mcp"
fi
# Docker Compose (unchanged host ports)
# ALERT_BRIDGE_URL=http://${HOST_IP}:9080
# VA_MCP_URL=http://${HOST_IP}:9901  # explicit legacy selection only
```

**Not on stock alerts Ingress** (Docker host ports / private backends only):
Elasticsearch `:9200`, Kafka, Redis, RT-CV (Docker `:9000`), RT-VLM
`:8018`, and `alert-notify` `:9090`. Do not `kubectl port-forward` them for
operate checks. Workflow B's
interim ES verdict probe stays Docker-only unless a public route is added later.

Alerts operate skills for the docs walkthrough (real-time mode):

- `vss-manage-video-io-storage` — VIOS via `${VST_API_BASE}`
- `vss-manage-alerts` — Alert Bridge via `${ALERT_BRIDGE_URL}` (Workflows C/D; never Agent `/generate` for rules)
- `vss-query-analytics` — `vss configure check`, then the read-only
  `vss analytics` commands through the recorded `/video-analytics-api` route

## Docker Compose

Resolve the deployment once with `vss configure`, then read the endpoints back
from the recorded config. The profile must have both its checked-in `.env` and a
runtime `generated.env` from the profile deploy workflow.

```bash
VSS_ORIGIN="${VSS_ORIGIN:-http://${HOST_IP:-127.0.0.1}:${HAPROXY_HOST_PORT:-7777}}"
VSS=(uv run --project "${VSS_REPO_ROOT}/libs/vss" vss)

"${VSS[@]}" configure --base-url "${VSS_ORIGIN}"
DEPLOYMENT=$("${VSS[@]}" configure show)

VSS_ORIGIN=$(printf '%s' "${DEPLOYMENT}" | jq -er '.base_url')
VSS_VIOS_URL=$(printf '%s' "${DEPLOYMENT}" | jq -er '.services.vst.url')
VST_API_BASE="${VSS_VIOS_URL}/api/v1"
ES_URL=$(printf '%s' "${DEPLOYMENT}" | jq -er '.services.elasticsearch.url')
# `vss configure` records RT-VLM at `.services.rt_vlm.url` where the profile
# routes it, which is what discovery needs. Inference keeps using the host port:
# the ingress applies `timeout server 120s`, and a non-streaming completion or a
# caption over a whole video runs past that.
RTVI_VLM_URL="http://${HOST_IP:-127.0.0.1}:${RTVI_VLM_PORT:-8018}"
```

For VIOS-only operate work without the search CLI, the Compose fallback is:

```bash
VSS_VIOS_URL="http://${HOST_IP:-127.0.0.1}:${VST_INGRESS_HOST_PORT:-30888}/vst"
VST_API_BASE="${VSS_VIOS_URL}/api/v1"
NVSTREAMER_ENDPOINT="http://${HOST_IP:-127.0.0.1}:${NVSTREAMER_HTTP_PORT:-31000}"
```

For Docker shareable-media origin checks, read the deploy-minted
`VST_EXTERNAL_URL` from `generated.env` (Brev HTTPS, not localhost). That
variable is a Compose/deploy runtime detail, not a Kubernetes operate input;
when `VSS_PUBLIC_URL` is set, ignore `VST_EXTERNAL_URL` and use
`VSS_PUBLIC_URL`.

`vss configure` probes each known route on the origin and records only the ones
that answer, so a route the deployment does not expose is absent from the config
rather than present-but-broken. Index names come from that same probe, and they
are created by ingestion — re-run `vss configure` after ingesting, or the
recorded index list stays empty. Direct Elasticsearch and RTVI probes are valid
Docker readiness checks; they are not Kubernetes operate prerequisites.

### Read vs write path resolution (headless)

The two runtime paths resolve endpoints by **different** mechanisms, and the
asymmetry is deliberate — each matches the vantage it runs from:

- **Read / query → `vss configure`** against the build origin (the block above).
  The CLI takes no per-command endpoints, so ingress-routed URLs for VST,
  Elasticsearch, Video Analytics API, RT-Embed, and RT-CV all come from the
  recorded config. There is **no
  ingress-less read path**: a build must front the operate route-set (see
  `services/ingress.md`) to be queryable from the host CLI.
- **Write / provision → loopback host ports**, *not* `vss configure`. The caller
  reads the consumer ports from the build's `resolved.yml` `ports:` mappings
  (`http://localhost:<port>`; stock deploys fall back to profile defaults) and
  hands them to `vss-manage-video-io-storage` `provision-vios-source.md`. Loopback
  covers RT-VLM natively and keeps RT-Embed's live SSE stream off the proxy hop.
  `vss configure` records **ingress URLs, not loopback ports**, so the two
  mechanisms do not overlap — except on a build that fronts RT-VLM for the tagging
  leg, where the RT-VLM tagging call may also use the recorded `/rtvi-vlm` origin
  from a remote host; loopback stays the lower-latency choice from the deploy host.
- **RT-VLM is loopback-only by default**, but fronted at `/rtvi-vlm` on any build
  whose host drives it — hand-driven tagging, or Critic verification and visual
  Q&A (see `services/ingress.md`). `vss configure` then records `rt_vlm` present,
  which is what enables the search CLI's critic — it builds the critic stack from
  that recorded URL, so a build recording `absent` runs search without one rather
  than failing. Only a build that never drives RT-VLM from the host keeps it
  loopback-only (`http://${HOST_IP:-127.0.0.1}:${RTVI_VLM_PORT:-8018}`).

## Kubernetes consumer contract (no port-forward)

Operate skills require `VSS_PUBLIC_URL` and use the consumer-derived variables
above:

```bash
: "${VSS_PUBLIC_URL:?Provide the public VSS Ingress origin}"
VSS_PUBLIC_URL="${VSS_PUBLIC_URL%/}"
VSS_VIOS_URL="${VSS_PUBLIC_URL}/vst"
VST_API_BASE="${VSS_VIOS_URL}/api/v1"
# Resolve VLM_ENDPOINT only with the probe-before-adopt flow above.
# LVS client base is the /lvs mount (no /v1 suffix) — the bare origin is the
# UI catch-all; ignore Docker-derived values:
LVS_BACKEND_URL="${VSS_PUBLIC_URL}/lvs"
# Alerts — force public prefixes; ignore leftover Docker :9080:
ALERT_BRIDGE_URL="${VSS_PUBLIC_URL}/alert-bridge"
# Only an explicitly selected legacy MCP workflow defines:
# VA_MCP_URL="${VSS_PUBLIC_URL}/va-mcp"
```

The public Agent, VIOS (`/vst`), Video Analytics API
(`/video-analytics-api`), and — when the profile deploys them — RT-VLM
(`/rtvi-vlm`), LVS (`/lvs`), Alert Bridge (`/alert-bridge`), and explicitly
selected legacy VA-MCP (`/va-mcp`) routes are supported operate interfaces.
Operate skills do not read
Deployments, ConfigMaps, Services, Secrets, or Helm values, and do not use
Service DNS, NodePorts, guessed release names, `kubectl port-forward`, or
`kubectl`/`docker exec` into pods.

Backends the deployment does not run (and `alert-notify` `:9090`, which is never
published) remain agent-side or Docker-host dependencies. Do not expose or
forward them merely to satisfy host-side operate checks. Where a backend *is*
deployed it is on the public origin at its canonical mount, the same on every
profile: `${VSS_PUBLIC_URL}/rtvi-vlm/v1` is the supported public operate path for
`vss-ask-video` and `vss-generate-video-report` Mode A, and
`${VSS_PUBLIC_URL}/lvs/v1/ready` / `/lvs/v1/summarize` for
`vss-summarize-video`. On the alerts profile, Alert Bridge serves
`vss-manage-alerts`, while the configured `/video-analytics-api` route serves
`vss-query-analytics` through `vss analytics`.

## Authentication boundary

If the public Ingress requires authentication, use only the operator-approved
header or client mechanism. Never inspect Kubernetes Secrets or copy tokens
into prompts, logs, generated files, or final answers.

## Out of scope for single-origin operate mode

| Capability | Why separate |
|---|---|
| NvStreamer HTTP / RTSP | Separate Ingress host; API path is `/api/v1/`, not `/vst/api/v1/` |
| VIOS RTSP proxy output | `rtsp://...` ports are not HTTP-Ingress routed |
| WebRTC live/replay media | Browser-only; not curl-operable through Ingress |
| Host CLI K8s search with ES/RTVI preflight | Needs private backend access or future Ingress routes |
