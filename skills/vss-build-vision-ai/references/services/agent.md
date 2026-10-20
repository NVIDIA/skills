# Agent Capability Owner

## Capabilities and service keys

| Capability | Canonical service profile keys |
|---|---|
| Agentic (natural-language) orchestration and agent REST API | `vss-agent` |
| Explicit legacy video-analytics MCP surface | `vss-va-mcp` |
| Web UI | `vss-ui` |
| Tracing UI | `phoenix` |

## Access role

This owner backs the interactive surface: agentic (natural-language)
orchestration, its own agent REST API, video-analytics MCP, Web UI, and tracing.
Its public front door is a separate owner (`ingress.md`), so browse ingress can
be requested without this tier.

As a capability owner the agent is reached only by **agentic orchestration** — a
request that needs an LLM to plan across tools and services (the agent's
`/generate`), or an explicit request for the agent REST API, video-analytics
MCP, Web UI, or tracing. It is **not** reached by two verb classes that look
interactive but are served elsewhere: (a) structured or programmatic query,
retrieval, and browse — owned by Elasticsearch with the host-CLI read path
(`vss configure` / `vss search run`), Kibana for dashboards, and each backend's
own REST API (VIOS, RT-CV, RT-Embed, Alerts); and (b) single-model VLM inference
— VLM Q&A and dense captioning go directly to RT-VLM (`/v1/chat/completions`,
`/v1/generate_captions`). "REST" alone is therefore not an agent cue: the agent's
REST API is that orchestration endpoint, distinct from those service-native REST
surfaces. The agent may also be pulled in as another owner's declared Required
peer (for example `lvs-server` lists it). Absent both an agentic request and such
a peer, nothing else reaches the agent, so the whole owner (and the LLM peer only
`vss-agent` required) is pruned as unreachable. "Headless" is just the explicit
name for that pruning, not a separate trigger. This is ordinary capability
pruning only. It does not apply when Q3 is the sole customization: that
harness-only delta bypasses owner pruning and removes only the harness-owned
keys defined in `SKILL.md` Step 5.

Video-analytics MCP (`vss-va-mcp`), Web UI (`vss-ui`), and tracing (`phoenix`)
are independently gated during an ordinary capability-removal build: each is
reached only by an explicit request for that surface and carries no capability
another owner needs. Prune each unless itself requested. In a harness-only
delta, preserve `vss-ui` with the rest of the Foundation, while `phoenix` leaves with the agent
whose traces it collected, as does the `llm_*` key once no enabled service calls
the LLM (`lvs-server` does; `alert-bridge` does not); `vss-va-mcp` follows the
narrower legacy-MCP rule in `SKILL.md` Step 5. The
video-analytics MCP is an agent-tier tool surface;
browsing or operating analytics is served by the host-side `vss analytics` CLI
through `vss-video-analytics-api` and does not reach it. A request for read-only
incidents, analytics sensors or metrics is not an MCP or Agent request.

## Required peers

- `vss-agent` requires one reachable LLM and the VIOS endpoints used by its
  selected config.
- Add a VLM owner only when the config enables video understanding, critique,
  alerts, or summarization that calls it.
- Add `vss-va-mcp` only for agent configurations that use video-analytics MCP.
- When the interactive surface must be reachable through a single public origin,
  add the Ingress owner (`vss-haproxy-ingress`, `ingress.md`) as the front door.
- ELK, Alerts, Search, and LVS peers are capability-dependent, not universal
  Agent dependencies.

## Configuration knobs

| Environment variable | Use |
|---|---|
| `VSS_AGENT_CONFIG_FILE` | Select the checked-in agent config for the Foundation. |
| `VSS_AGENT_HOST_PORT`, `VSS_AGENT_PORT` | Publish and bind the agent API. |
| `VSS_VA_MCP_HOST_PORT`, `VSS_VA_MCP_PORT`, `VSS_VA_MCP_CONFIG_FILE` | Configure the optional analytics MCP. |
| `LLM_MODE`, `LLM_MODEL_TYPE`, `LLM_NAME`, `LLM_BASE_URL` | Configure the agent's LLM. |
| `VLM_MODE`, `VLM_MODEL_TYPE`, `VLM_NAME`, `VLM_BASE_URL`, `RTVI_VLM_BASE_URL` | Configure the agent's VLM. |
| `VST_BASE_URL`, `VST_INTERNAL_URL`, `VST_EXTERNAL_URL`, `VST_MCP_URL` | Configure video storage access. |
| `LVS_BACKEND_URL`, `COSMOS_EMBED_ENDPOINT`, `ELASTIC_SEARCH_ENDPOINT`, `ALERT_BRIDGE_URL` | Enable capability-specific tools. |
| `ENABLE_AUDIO` | Enable audio-aware flows. (Result critique has no build-time flag; it is chosen per request via the agent's `use_critic` option.) |
| `VSS_AGENT_REPORTS_BASE_URL`, `VSS_AGENT_EXTERNAL_URL` | Generate externally reachable links. |
| `VSS_UI_HOST_PORT` | Publish the Web UI (public-ingress knobs live with the Ingress owner, `ingress.md`). |

## Sources

- `deploy/docker/services/agent/compose.yml`
- `deploy/docker/services/ui/compose.yml`
- `services/agent/README.md`
