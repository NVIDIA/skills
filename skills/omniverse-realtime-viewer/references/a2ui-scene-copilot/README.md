<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# A2UI Scene Copilot

## Triggers

Use this reference for agent-to-UI (A2UI) copilot, scene copilot, chat-driven
viewer control, LLM tool calling into the viewer, CopilotKit integration,
natural language USD scene interaction, AI assistant for the viewer, or any
request to build an agent that controls the Omniverse Realtime Viewer through
conversation.

## Overview

The A2UI Scene Copilot pattern adds an LLM-powered chat agent alongside the
Omniverse Realtime Viewer. The agent communicates with the viewer server over
a WebSocket protocol (the A2UI protocol), giving the user natural-language
control over scene exploration, property editing, AOV switching, selection,
and variant management — all reflected in real time through the WebRTC viewport.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          Browser (React + Vite)                          │
├───────────────────────────────────┬─────────────────────────────────────┤
│       WebRTC Viewport             │         Chat Panel (CopilotKit)     │
│   AppStreamer ↔ ovstream          │    user ↔ agent ↔ tools ↔ A2UI WS  │
│       (video + audio)             │                                     │
└──────────┬────────────────────────┴───────────────┬─────────────────────┘
           │ UDP :47998 media                       │ WS to agent backend
           │ TCP :8011 signaling                    │ (port 8123)
           ▼                                        ▼
┌───────────────────────────────────────┐  ┌──────────────────────────────┐
│       Application Runtime Server       │  │     Agent Backend (Python)   │
│  app-owned OVStage + runtime adapter   │  │  LangGraph / LangChain       │
│  attached ovrtx + ovstream WebRTC      │  │  Tools → A2UI WS client      │
│  A2UI endpoint (WS :3001/a2ui/agent)  │  │  System prompt from hello    │
└───────────────────────────────────────┘  └──────────────────────────────┘
```

### Component Responsibilities

| Component | Role |
|-----------|------|
| Application runtime server | Owns stage lifecycle, ordinals, OVStage query/write adapter, attached rendering, and WebRTC streaming |
| Attached ovrtx renderer | Consumes committed OVStage publications; owns render products, frame output, renderer-native picks, and selection visualization |
| A2UI endpoint | WebSocket server exposing the runtime adapter's viewer operations to agents |
| Agent backend | LLM orchestration, tool definitions, A2UI client |
| Frontend | WebRTC viewport + chat UI (CopilotKit or custom) |

## A2UI WebSocket Protocol

### Connection

Connect to `ws://<server-host>:<a2ui-port>/a2ui/agent` (default port 3001).

On connect the server sends an `agent_hello` notice:

```json
{
  "op": "agent_hello",
  "notice": true,
  "data": {
    "version": "1.0",
    "operations": ["open_stage", "list_prims", "get_properties", ...],
    "operation_schemas": [...],
    "app_state": { "stage_url": "...", "selection": [], "active_aov": "LdrColor", ... },
    "system_prompt": "..."
  }
}
```

**Critical:** Wait for `agent_hello` before sending any operation request.

### Request/Response Format

```json
// Request
{"op": "operation_name", "request_id": "unique-id", "params": {...}}

// Success response
{"op": "operation_name", "request_id": "unique-id", "result": {...}}

// Error response
{"op": "operation_name", "request_id": "unique-id", "error": {"code": "...", "message": "..."}}
```

### Server-Pushed Notices

The server may push notices (e.g., `selection_changed`) at any time:

```json
{"op": "selection_changed", "notice": true, "data": {"paths": ["/World/Cube"]}}
```

### Operations

| Operation | Category | Mutates | Description |
|-----------|----------|---------|-------------|
| `open_stage` | stage | yes | Load a USD stage by path or URL |
| `list_prims` | stage | no | List child prims under a root path |
| `get_properties` | stage | no | Return attributes and relationships for a prim |
| `get_prim_count` | stage | no | Return total prim count |
| `get_variants` | stage | no | Return variant sets for a prim |
| `set_variant` | stage | yes | Set a variant selection |
| `set_property` | stage | yes | Set a USD attribute value |
| `reset_stage` | stage | yes | Force-reload the current stage |
| `select` | selection | yes | Replace the current viewer selection |
| `get_selection` | selection | no | Return current selection |
| `change_aov` | render | yes | Switch the active display AOV |
| `get_available_aovs` | render | no | List available AOVs |
| `get_app_state` | app | no | Return full viewer snapshot |

### Parameter Aliases

Operations accept multiple parameter names for convenience:

- Path params: `prim_path`, `path`, `url`
- Selection: `paths`, `prim_path`, `path` (single string auto-wrapped in list)
- AOV: `aov`, `name`
- Property: `property_name`, `property`, `attr`
- Variant: `variant_selection`, `selection`

## Server-Side Implementation

### A2UI Endpoint Setup

The A2UI endpoint runs beside the application-owned runtime. Before exposing
any operation, create and populate one `ovstage.Stage`, attach the renderer to
that stage, and publish the initial ordinal. The endpoint dispatches commands
through the application's runtime adapter; it does not use renderer attribute
APIs or a second OpenUSD stage as an alternate scene authority.

```python
from a2ui_endpoint import A2UIAgentEndpoint

# In startup, after the runtime owns a populated stage and attached renderer:
a2ui = A2UIAgentEndpoint(runtime, host="0.0.0.0", port=3001, path="/a2ui/agent")
a2ui.start()
```

The endpoint owns:
- `A2UIOperationDispatcher` — maps operation names to runtime-adapter commands and queries
- Connection lifecycle — sends `agent_hello`, dispatches requests, broadcasts notices
- Thread isolation — runs on a dedicated asyncio event loop

### Operation Dispatcher Pattern

Each operation is a method on `A2UIOperationDispatcher` that:
1. Validates params (using aliases)
2. Calls the runtime adapter, which is the single authority for stage state
3. Returns `A2UIDispatchResult(result_dict, notices=[...])`

```python
def _op_select(self, params):
    paths = _paths_param(params)
    self._set_selection(paths)
    return A2UIDispatchResult(
        {"paths": paths},
        notices=[{"op": "selection_changed", "data": {"paths": paths}}],
    )
```

### Runtime Adapter — One Read/Write Authority

Implement `set_property`, `set_variant`, selection, hierarchy, and property
queries through a narrow application-owned adapter over the active OVStage.
Every mutating command must:

1. validate the canonical USD path and value against the active stage generation;
2. reserve the next application ordinal `N`;
3. enqueue the OVStage write or stage-management operation and wait for it to complete;
4. advance the stage write floor to `N`; and
5. return the committed ordinal and generation in the response or notice.

The render loop consumes that same publication with
`renderer.step(..., ordinal=N)` (or an explicit pre-step
`renderer.update_from_stage(N)` when required). Do not render above the write
floor, and do not mutate or replace the stage while a renderer step is in
flight.

Queries return copied DTOs such as canonical path, value/schema summary,
generation, and observed ordinal. They resolve paths through the OVStage path
dictionary. Treat stage replacement, `open_stage`, and `reset_stage` as
serialized lifecycle operations: stop new commands, detach or replace the
renderer-stage pairing as required, populate the new stage, publish its initial
ordinal, then resume requests with the new generation.

Do not implement a renderer-write or `pxr`-worker fallback. With an attached
renderer, direct `renderer.write_attribute()`, `query_prims()`, and
`read_attribute()` are not the authority for viewer scene state. Keep ovrtx
APIs for renderer-owned work: render products, output frames, pick queues, and
selection-outline configuration.

## Agent Backend Implementation

### Framework Choice

Any LLM orchestration framework works. Validated patterns:

- **LangGraph + LangChain** — Python, tool-calling agents, streaming
- **CopilotKit** — React-native chat UI with frontend actions
- **AG-UI** — Agent-to-UI protocol (CopilotKit's wire format)

### A2UI Client (Agent Side)

The agent needs a synchronous WebSocket client that:
1. Connects to the A2UI endpoint
2. Waits for `agent_hello`
3. Sends operations and waits for the matching `request_id` response

```python
import json
import websocket  # websocket-client library

class A2UIClient:
    def __init__(self, url="ws://127.0.0.1:3001/a2ui/agent"):
        self.url = url
        self._ws = None

    def connect(self):
        self._ws = websocket.create_connection(self.url, timeout=10)
        # Wait for agent_hello
        hello = json.loads(self._ws.recv())
        assert hello["op"] == "agent_hello"
        return hello["data"]

    def call(self, op: str, params: dict = None) -> dict:
        request_id = str(uuid.uuid4())
        self._ws.send(json.dumps({
            "op": op,
            "request_id": request_id,
            "params": params or {},
        }))
        while True:
            msg = json.loads(self._ws.recv())
            if msg.get("notice"):
                continue  # skip server-pushed notices
            if msg.get("request_id") == request_id:
                if "error" in msg:
                    raise RuntimeError(msg["error"]["message"])
                return msg["result"]
```

### LangChain Tool Definitions

Map each A2UI operation to a LangChain tool:

```python
from langchain_core.tools import tool

@tool
def list_prims(root: str = "/World", recursive: bool = False) -> str:
    """List prims under a root path in the USD stage."""
    result = a2ui_client.call("list_prims", {"root": root, "recursive": recursive})
    return json.dumps(result["flat_prims"], indent=2)

@tool
def select_prim(paths: list[str]) -> str:
    """Select prims in the viewer by their USD paths."""
    result = a2ui_client.call("select", {"paths": paths})
    return f"Selected: {result['paths']}"

@tool
def set_property(prim_path: str, property_name: str, value) -> str:
    """Set a USD attribute value on a prim (e.g., light intensity)."""
    result = a2ui_client.call("set_property", {
        "prim_path": prim_path,
        "property_name": property_name,
        "value": value,
    })
    return f"Set {result['property']} = {result['value']} on {result['prim_path']}"
```

### System Prompt Strategy

The `agent_hello` message includes a server-generated `system_prompt` that lists
available operations and current state. Use this as the agent's system prompt or
merge it with your own:

- Keep the prompt concise enough for the selected model and tool budget
- Always include current stage URL and available operations
- Instruct the agent to use read ops before mutating ops

## Frontend Implementation

### Split-View Layout

The canonical layout places the WebRTC viewport on the left and chat on the right:

```
┌────────────────────────────┬──────────────────┐
│                            │   Chat Panel     │
│     WebRTC Viewport        │   (420px fixed)  │
│     (flex-1)               │                  │
│                            │   [user input]   │
└────────────────────────────┴──────────────────┘
```

### WebRTC Viewport Integration (React)

When integrating the current `@nvidia/ov-web-rtc` `AppStreamer` client in React,
follow `streaming-client` for the current DirectConfig contract. Render the
video/audio elements before connecting, then keep one stable connection effect:

```tsx
import { AppStreamer, StreamType } from "@nvidia/ov-web-rtc";

function StreamingViewport({ server, signalingPort }: { server: string; signalingPort: number }) {
  const videoId = "remote-video";

  useEffect(() => {
    void AppStreamer.connect({
      streamSource: StreamType.DIRECT,
      streamConfig: { videoElementId: videoId, audioElementId: "remote-audio", server, signalingPort, width: 1920, height: 1080, codec: "H264", codecList: ["H264"] },
    });
    return () => { AppStreamer.terminate(); };
  }, [server, signalingPort]);

  return <div className="relative w-full h-full"><video id={videoId} autoPlay /><audio id="remote-audio" autoPlay /></div>;
}
```

### CopilotKit Frontend Actions (Interactive Widgets)

CopilotKit `useCopilotAction` renders inline widgets in the chat when the agent
calls a tool. Each action needs both `handler` AND `render` props:

```tsx
useCopilotAction({
  name: "show_exposure_control",
  description: "Show the light exposure control widget",
  parameters: [
    { name: "lightPath", type: "string", description: "USD path to the light" },
  ],
  handler: async (args) => "Showing exposure control",
  render: (props) => <ExposureWidget lightPath={props.args.lightPath} />,
});
```

**Key rules:**
- CopilotKit requires BOTH `handler` and `render` (crashes without handler)
- Parameter types: only `"string"`, `"number"`, `"boolean"`, `"object"` — NOT arrays
- Widget WebSocket commands must wait for `agent_hello` before sending
- Don't use `crypto.randomUUID()` over plain HTTP — use Math.random fallback

### Widget-to-Server Communication

Frontend widgets that need to send A2UI commands (e.g., slider → set_property)
use short-lived WebSocket connections:

```typescript
async function sendA2UICommand(op: string, params: Record<string, unknown> = {}): Promise<any> {
  return new Promise((resolve, reject) => {
    const ws = new WebSocket(`ws://${SERVER_IP}:${A2UI_PORT}/a2ui/agent`);
    const requestId = generateId();
    let gotHello = false;

    ws.onmessage = (evt) => {
      const msg = JSON.parse(evt.data);
      if (msg.op === "agent_hello" && !gotHello) {
        gotHello = true;
        ws.send(JSON.stringify({ op, request_id: requestId, params }));
        return;
      }
      if (msg.request_id === requestId) {
        ws.close();
        resolve(msg.result);
      }
    };
  });
}
```

## Port Map

| Port | Protocol | Purpose |
|------|----------|---------|
| 8011 | HTTP + WS | REST API + ovstream WebRTC signaling |
| 3001 | WS | A2UI agent protocol |
| 8123 | HTTP | Agent backend (LangServe / AG-UI) |
| 5173 | HTTP | Frontend dev server (Vite) |
| UDP 47998 | UDP | WebRTC media (ICE candidate) |

## Deployment Considerations

### Public IP / Tailscale

- Set `--public-ip <tailscale-ip>` on the OVRTX server for remote WebRTC media
- ICE candidate in SDP will use this IP for UDP media delivery
- Frontend `stream.config.json` needs `server` and `mediaServer` set to same IP
- `tailscale ping` the server before connecting to warm DERP relay path

### GPU Requirement

The OVRTX server requires an NVIDIA GPU. The agent backend and frontend can run
anywhere with network access to the server.

### Process Supervision

Run three long-lived processes:
1. OVRTX streaming server (GPU-bound, port 8011 + 3001)
2. Agent backend (CPU-bound, port 8123)
3. Frontend dev server (CPU-bound, port 5173)

Use a process supervisor (supervisord, systemd, or shell script with restart
loops) to keep all three alive.

## Validation

### E2E Test Pattern

Use Playwright to validate the full loop:

1. Navigate to the frontend URL
2. Type a natural-language query in the chat input
3. Assert the agent responds with tool call results
4. (Optional) Assert viewport frame is non-black after scene load

### A2UI Protocol Test (No GPU)

The `A2UIHeadlessViewer` class provides a no-GPU test harness:

```python
from a2ui_ops import A2UIOperationDispatcher, A2UIHeadlessViewer

viewer = A2UIHeadlessViewer(stage_path="samples/stage01.usda")
viewer.start()
dispatcher = A2UIOperationDispatcher(viewer)

result = dispatcher.dispatch("list_prims", {"root": "/World", "recursive": True})
assert len(result.result["flat_prims"]) > 0
viewer.stop()
```

## Dependencies

| Package | Purpose | Install |
|---------|---------|---------|
| current ovrtx and ovstage packages resolved through `references/dependencies` | Attached runtime rendering and scene data | Follow the current upstream agent guidance, skills, examples, and package metadata |
| current ovstream package resolved through `references/dependencies` | WebRTC streaming | Follow the current dependency guidance |
| websockets | A2UI server endpoint | `pip install websockets` |
| websocket-client | A2UI agent client | `pip install websocket-client` |
| langchain-core | Agent tool framework | `pip install langchain-core` |
| @copilotkit/react-core | Chat UI | `npm install @copilotkit/react-core` |
| @nvidia/ov-web-rtc | WebRTC viewport | Follow `references/dependencies` and `streaming-client` |

## Related References

- `references/streaming-server` — OVRTX server setup and ovstream configuration
- `references/streaming-lifecycle` — WebRTC signaling flow and callbacks
- `references/streaming-client` — Browser-side WebRTC integration
- `references/streaming-messages` — Data channel message protocol
- `references/viewer-input-routing` — Input event routing architecture
- `references/ovstage-runtime` — Stage ownership, ordinals, and lifecycle
- `references/ovstage-data-plane` — Runtime query/write adapter and publication rules
- `references/ovstage-ovrtx-integration` — Attached renderer update contract
