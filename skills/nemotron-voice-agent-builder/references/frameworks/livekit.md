# LiveKit

Read after the user approves the table and the framework row says LiveKit. Cascaded only:
STT → LLM → TTS through an `AgentSession`. Omni is not supported on LiveKit. If the table
also asked for omni, stop and ask which to keep (intake already requires this).

LiveKit owns the session transport. Do not apply Pipecat transport rows, `-t` flags, or
`:7860` client wiring here. The connect path is the LiveKit Agents console (or the path
the LiveKit docs MCP currently documents), not a Pipecat browser client.

## Code Comes from the MCP

This skill ships no `agent.py`, no skeleton, and no template. Agent code is generated from
the LiveKit docs MCP. Discover it by capability first. Endpoint to add if missing:
`docs.livekit.io/mcp`. Index for degraded fetch only: `docs.livekit.io/llms.txt`.

Query for the pieces you need. For a cascaded NVIDIA stack that means:

- Voice AI quickstart and Agents console
- `AgentSession` wiring
- NVIDIA STT and TTS plugins (`livekit-plugins-nvidia`)
- LLM plugin or OpenAI-compatible client pointed at the locked NVIDIA model id
- LiveKit Cloud API keys and `lk cloud auth`
- Deployment / worker run commands
- current prompt, text-output, metrics, tool, and exception hooks required by
  `domain/agent-behavior.md` and `operations/observability.md`

Never write a LiveKit import, class name, or API call from memory. If a query fails or
comes back empty, say so and stop. Do not fill the gap with a guess.

### When the MCP Is Not Available

Stop and ask the user to enable the LiveKit docs MCP, or explicitly approve the less
reliable `docs.livekit.io/llms.txt` fallback. On fallback, fetch one current example from
LiveKit docs or `livekit-examples`. Never invent structure.

## What Gets Written

Write one `agent.py`. Do not mix Pipecat (`bot.py`) into the project. Follow
`output-contract.md` for the remaining files and any generated Compose services. Model
ids are constants in code, not `.env` values. `.env` holds secrets only:
`NVIDIA_API_KEY`, `LIVEKIT_URL`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET`.

## LiveKit Server

Default is **LiveKit Cloud** (`wss://<project>.livekit.cloud`), not `localhost:7880`.

Before generating, `LIVEKIT_*` must already clear `preflight.md`. If any key is missing,
guide the user and wait. Prefer current steps from the MCP (`lk cloud auth` / project API
keys). Manual path starts at https://cloud.livekit.io/.

Self-hosted LiveKit server is an override only when the user asks for it by name. Still
require all three `LIVEKIT_*` variables.

## NVIDIA Models

Resolve exact ASR, LLM, and TTS ids through `models/catalog.md` → slot files before
writing constants. Streaming ASR first (`models/asr.md`).

- STT / TTS: NVIDIA plugin from the MCP (`livekit.plugins.nvidia` or current package name).
  Cloud uses NVIDIA endpoints and `NVIDIA_API_KEY`. Self-hosted points `server` (or the
  current equivalent) at the local speech endpoint, which is a Speech NIM started from that
  model’s build.nvidia.com `/deploy` page (`platforms/deployment.md`) or the single
  NeMo-Speech.cpp gRPC endpoint (`platforms/single-gpu.md`).
- LLM: use the MCP’s documented way to call the locked Nemotron model id (NVIDIA cloud or
  self-hosted OpenAI-compatible base URL). Do not invent a plugin class from memory.

A self-hosted LLM has two traps. The OpenAI-compatible client may refuse to construct
without an API-key argument even though a local NIM needs no authentication, so pass a
non-secret placeholder when the resolved version behaves that way. Reasoning-off must reach
the request nested inside `extra_body` (`models/llm.md`), so ask the MCP which plugin
argument carries provider-specific fields and confirm the shape in the smoke test. The
model string is the served id from `GET /v1/models`, not the catalog slug.

Function ids for cloud speech come from build.nvidia.com, not from `/v1/models`.

On the single-GPU stack, follow `platforms/single-gpu.md`. Point both NVIDIA speech plugins
at the one shared gRPC endpoint, leave function ids empty, and point the OpenAI-compatible
LLM client at vLLM. Resolve the current `server`, SSL, language, and voice arguments from
the LiveKit docs MCP, and note that this endpoint is plaintext gRPC, so the plugins must
not be configured for TLS unless a proxy terminates it.

## Run Shape

The LiveKit CLI (`lk`) runs the worker. Before handover, confirm it is installed. When it
is missing, give the current install steps from the LiveKit docs MCP and wait, because the
worker start command depends on it.

Credentials set → LiveKit CLI available → start self-hosted model services if the deployment
row needs them → run the worker the way the MCP documents (typically `lk agent dev` on
`agent.py`). Success is a spoken exchange, same bar as `operations/run.md`.

Do not tell the user to open a Pipecat WebRTC client or hit `:7860`, and never present the
worker’s local HTTP or health port as a client URL. LiveKit’s Agents Console (or the current
documented client) is the path.

## Connect and Verify

`scripts/smoke.sh` must pass before the worker joins a room.

Then, with the worker still running and its `registered worker` line shown:

1. open the LiveKit project whose credentials match `.env` `LIVEKIT_URL`
2. refresh its Agents page and select the generated agent by name
3. start a session, allow the microphone, and speak

Confirm the worker connects without auth errors and STT → LLM → TTS each produce a turn. End
the handover by pointing the user to that LiveKit URL to try the agent. Full handover steps
are in `operations/run.md`.

## After It Runs

Re-query the MCP before changing any LiveKit API. For NIM LLM profile changes, return to
**Select a NIM Model Profile** in `models/llm.md`. The single-GPU stack returns
to its model card and platform guide.

## Anti-Patterns

- Generating Omni on LiveKit, or generating `bot.py` for a LiveKit row.
- Pointing the user at Pipecat `:7860` / transport `-t` flags.
- Defaulting to self-hosted LiveKit server when Cloud was not overridden.
- Pinning LiveKit or plugin versions from memory instead of resolving them.
