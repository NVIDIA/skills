# Pipecat

Read after the user approves the table and the framework row says Pipecat. Covers both
pipeline shapes. Omni has extra requirements on top, in `frameworks/omni.md`.

## Code Comes from the MCP

This skill ships no `bot.py`, no skeleton, and no template. Pipeline code is generated from
the Pipecat docs MCP. Discover it by capability first. The server is commonly exposed as
`user-pipecat-docs`.

Query it for the pieces you actually need, rather than pulling the whole surface. For a
cascaded pipeline that means pipeline construction, the runner, the transports, and the
NVIDIA STT, LLM, and TTS services. Omni adds LLM service subclassing, speech input, smart
turn detection, and user turn strategies. Also query current prompt, text-filter,
observer, metrics, tool, and exception hooks required by `domain/agent-behavior.md` and
`operations/observability.md`.

Never write a Pipecat import, class name, or API call from memory. If a query fails or
comes back empty, say so and stop. Do not fill the gap with a guess, because a plausible
wrong import costs the user more time than a missing file.

### When the MCP Is Not Available

Ask the user to add `https://daily-docs.mcp.kapa.ai` to their coding agent's MCP
configuration as a remote server named `pipecat-docs`, enable it, and reload the agent
session. Wait for the user to continue, then discover the server by capability.

If the user cannot enable it, ask them to explicitly approve the less reliable
`docs.pipecat.ai/llms.txt` fallback. Never fall back silently.

## What Gets Written

Write one `bot.py`. Follow `output-contract.md` for the rest of the project and any
generated Compose services.

## Local LLM Wiring

Cascaded only, and it covers any local endpoint, whether that is a NIM or raw vLLM on the
single-GPU stack. Omni follows `frameworks/omni.md`. Two failures here cost a full connect
cycle each because both look like a model problem, so resolve the exact shapes from the
MCP before generating. Query it for the NVIDIA LLM service class and import, the local
base-URL argument, the API-key argument's behaviour, and the settings wrapper that carries
provider-specific request fields.

**The client still wants an API key locally.** A local server needs no authentication, but the
underlying OpenAI-compatible client can refuse to construct without a key argument. When the
resolved version behaves that way, pass a non-secret placeholder. Never require a real cloud
key to talk to a container on localhost, and never write one into `.env` for this.

**Reasoning-off has to survive the wrapper.** The provider needs the thinking flag nested
inside the request's `extra_body` (`models/llm.md`). Pipecat does not take a raw request
dict, so the flag must be placed in the current settings wrapper's provider-extra field
such that it arrives nested, not flattened to the top level. Confirm that field name in the
MCP, then verify the shape in the smoke test rather than at the browser.

The model string is the runtime model id from `GET /v1/models`, not the catalog slug
(`models/llm.md` §Resolve three names).

## Transport Wiring

The transport row is already settled by `intake.md`. It controls transport parameters,
package extras, and the runner `-t` flag. The runner flag is the usual failure.

| Transport | `transport_params` keys | Package extras | Run |
| --- | --- | --- | --- |
| WebRTC | `webrtc` | `webrtc` | pass `-t webrtc` |
| WebSocket | `websocket` | `websocket` | pass `-t websocket` |
| Both | `webrtc` and `websocket` | both | pass no `-t` at all |

Passing `-t webrtc` when the user chose both is the failure. The flag restricts the server
to that single transport, so a `/start` naming any other one is rejected outright and the
WebSocket path never appears in the client UI. Omitting the flag lets the runner serve every
transport at once and lets the caller pick one per session.

Beyond the transport extras, the project needs the NVIDIA services, the runner, and a
voice activity detector. Resolve the package version from the MCP or from PyPI rather than
pinning a number from memory.

Both is the right default when there is any doubt. A corporate network that blocks UDP can
fall back to WebSocket without rebuilding the project.

For remote browsers, NAT traversal, coturn, or a separate GPU host, follow
`networking/remote-webrtc.md`.

## Verify Before Handover

`scripts/smoke.sh` must pass before the runner starts.

Ask the runner what it registered rather than assuming. `GET /status` lists the accepted
transports, and for the both case that list must include `webrtc` and `websocket`. A
`POST /start` naming either one should open a session on it.

Full startup and connection steps are in `operations/run.md`.

## After It Runs

Re-query the MCP before changing any Pipecat API. NIM profile changes return to **Select a
NIM Model Profile** in `models/llm.md`. The single-GPU stack returns to its model card and
`platforms/single-gpu.md`.

## Anti-Patterns

- Passing `-t webrtc` when the user chose both transports.
- Generating a separate bot file per transport.
- Pinning a Pipecat version remembered rather than resolved.
- Demanding a real cloud key to reach a container on localhost.
- Passing the reasoning flag as a raw request dict, or flattening it out of `extra_body`.
- Opening the client to find out whether the services construct. Run the smoke test.
