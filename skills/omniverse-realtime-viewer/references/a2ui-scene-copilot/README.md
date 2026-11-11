<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# A2UI Scene Copilot

Use this reference for agent-to-UI viewer operations, a scene copilot,
chat-driven scene control, or tools that let an agent inspect and operate a
viewer. Build the explicit operation interface first; add language-model
orchestration and a chat or tool UI when requested.

The viewer remains the scene authority. An agent uses the same application
commands as ordinary controls; it does not acquire a stage handle or a second
renderer. The request envelope below is an application protocol, not an
OVRTX API or a substitute for an agent framework's wire protocol.

## Compose the operation path

```text
operation form or agent tool
  -> correlated client request
  -> endpoint validation and bounded command queue
  -> existing runtime owner
       -> copied query state / worker preparation
       -> stage publication or renderer/app interaction
  -> completion result and copied state
  -> tool result, controls and displayed viewer frame
```

Keep WebSocket I/O on its event loop and SDK work on the viewer's existing
runtime owner. Calling an adapter method from a socket callback is safe only
when that method enqueues work. Loading and authored metadata can use isolated
workers; their results return to the owner before affecting the live scene.
Use [stage-management](../stage-management/README.md) for preparation, generation
invalidation, source lifetime and replacement, and
[ovstage-ovrtx-integration](../ovstage-ovrtx-integration/README.md) for publication.

For WebRTC, compose this endpoint beside the existing streaming server and use
[streaming-client](../streaming-client/README.md) for video connection. The same
operation adapter can serve a native viewer through its chosen transport.
Resolve runtime packages through [dependencies](../dependencies/README.md);
choose the WebSocket and optional agent packages for the app's language and
manifest. Keep the operation service opt-in when extending an existing viewer.

## Advertise capabilities before accepting commands

On connection, send a capability snapshot: supported operations, parameter
schemas, effects, relevant bounds, current scene generation and copied app
state. A client waits for this handshake before enabling tools or controls.
Advertise only operations the current implementation can complete.

| Capability | Contract to define |
|---|---|
| Scene and property queries | Canonical paths, pagination or bounded previews, authored versus live values, observed generation and ordinal |
| Selection | Supported selection cardinality/modes and the common tree/pick selection path |
| Output choice | Available display outputs and when a requested output becomes active |
| Property edits | Explicit writable attributes or semantic commands, value types/shapes, units, coordinate space and edit ownership |
| Variants, open and reset | Source types, composition policy, retained edits, camera policy and replacement completion |

A generic `set_property` name does not make arbitrary USD values writable.
Keep inspection schemas separate from write schemas. An authored attribute or
bounded array preview describes the source; it need not describe the current
simulation pose or be a lossless value that can be written back. Return live
transforms through the runtime adapter and label authored metadata separately.
Use [prim-info-display](../prim-info-display/README.md) for this split and
[prim-transform-safety](../prim-transform-safety/README.md) for coordinate-space
conversion. Define JSON encodings for supported values; reject unencodable
results with a correlated error instead of leaving requests pending.

Apply the same edit ownership as ordinary controls. For example, a simulator or
scientific producer can own a geometry subtree and its composition. The agent
should use an advertised workload command or switch scenes, rather than bypass
that owner's controls. Variant edits follow the application's non-destructive
composition policy, refresh metadata, and retain edits only for surviving paths.

## Correlate requests through completion

One possible envelope is:

```json
{"op":"select","request_id":"pick-7","generation":3,"params":{"paths":["/World/Cube"]}}
```

Replies carry the same operation and request ID, with either a result or a
structured error. Notices are separate messages and never complete a request.
Use request IDs unique within a connection session and keep pending IDs
protected until they finish, independently of any bounded completed-ID cache.

Use one receive loop per connection. It dispatches notices to state consumers
and responses to a pending-request map; each caller awaits its own completion.
The following is application pseudocode, not an SDK class:

```text
call(operation, parameters):
    wait for capability handshake
    validate against advertised schema
    register pending[id] with deadline before sending
    send operation, id, observed generation and parameters
    await pending[id]

receive(message):
    notice -> apply copied state if still current
    matching response -> resolve or reject pending[message.request_id], clear timer
    unknown or expired response -> discard

connection closes:
    reject pending requests and clear timers
    reconnect -> acquire fresh capabilities and state
```

This permits concurrent calls without one caller consuming another's reply.
Browser controls and widgets can share this client instead of opening a new
connection for every slider edit. Coalesce compatible drafts before dispatch;
keep Reset and scene replacement as explicit commands. Cap connections,
message/result sizes and pending work, and return an overload error when full.
State notices may be coalesced; correlated replies must retain their identity.
The selected WebSocket library owns its receive and buffering contracts; for
Python, consult the [public client reference](https://websockets.readthedocs.io/en/stable/reference/asyncio/client.html).

## Execute on the owner and report the actual effect

After admission, the owner rechecks the request's generation, deadline,
cancellation and edit permissions. Recheck after asynchronous preparation and
immediately before committing a scene change. Discard stale results even when
the same prim path exists in the replacement scene. Property panels also need
a request/selection identity so an older response cannot replace newer details.

Choose publication according to the work performed:

- **Stage writes:** reserve an ordinal, write and wait, advance the floor, then
  let the existing render loop consume it. Use
  [ovstage-data-plane](../ovstage-data-plane/README.md).
- **Selection:** update the common selection state and native renderer groups
  through [native-picking-selection](../native-picking-selection/README.md).
  Renderer-only group changes do not require an invented stage write.
- **Output changes:** follow [aov-switching](../aov-switching/README.md). This may
  publish RenderProduct attributes. Distinguish the desired output from the
  active output confirmed by its copied frame.
- **Scene composition:** follow the normal scene handoff and wait for its first
  valid frame before reporting replacement success.

Return the scene generation and observed state; include a committed ordinal
when a stage publication occurred. An application revision can order selection
or other app-only changes without advancing the stage floor. State whether a
result acknowledges a committed command or an observed frame. A tool response
alone does not establish that the browser has decoded and displayed that frame.

On disconnect or timeout, cancel uncommitted work and retire its workers and
sources through the normal lifetime rules. A lost response does not prove that
a command failed to commit. Requery state after reconnect before retrying a
mutation; cancellation is not rollback of an already performed edit. During
shutdown, stop accepting requests before retiring the runtime and its consumers.

## Add an agent and tool UI when requested

An operation form can use this interface without a language model. For a real
agent, choose the provider/orchestrator appropriate to the product, then bind
the advertised schemas as tools. Each tool calls the correlated client and
returns the structured result or error; it should not guess unadvertised
attributes, paths, aliases or writable types.

[LangChain tools](https://docs.langchain.com/oss/python/langchain/tools) provide
schema-based tool definitions for an agent backend. A LangGraph workflow can
compose those tools when orchestration is needed. For a React chat or inline
tool interface, use [CopilotKit's current reference](https://docs.copilotkit.ai/reference)
and its selected version's execution/rendering hooks. Execution and rendering
are separate roles: a widget renders status and sends edits through the same
operation client. [AG-UI](https://docs.ag-ui.com/introduction) can carry agent
interaction events between the agent and frontend; adapt that boundary to the
viewer operation protocol rather than treating the two as interchangeable.

Give the agent current capabilities and scene context, and refresh them after
scene changes. Keep scene strings and tool results as data; the application's
instructions and permissions govern execution. Enforce any product approval
policy at the command boundary shared by UI and agent tools.

Compose the chat or operation panel with
[viewer-layout-patterns](../viewer-layout-patterns/README.md) and preserve the
viewport's stable size and input routing. Show pending, completed and failed
operations honestly. A rejected edit should leave camera navigation available;
controls should refresh from authoritative state after successful mutations.

Bind a local control endpoint to the intended local interface and validate
browser origins. Remote access needs an explicit authentication, authorization
and secure transport design; follow the selected deployment and transport
references. Provider credentials belong on the agent backend, not in browser
capability messages. Ports and process placement are application configuration.

## Validate the selected product

Protocol tests can check correlation, concurrent notices, schema errors,
bounded queues, cancellation and stale-generation rejection without rendering.
Use the actual viewer for visible selection, property edits, output switching,
variant changes, source preservation and recovery while ordinary controls remain
usable. Compare reported live values and displayed effects; a nonblack frame or
a successful tool message alone does not validate an edit.

When a real agent is part of the product, also exercise natural-language input,
actual tool selection, parameter generation, errors and the resulting visible
state through that agent. Explicit operation tests validate the viewer protocol;
they do not validate a model provider, orchestration framework or chat interface.
Use [validation](../validation.md) for the chosen delivery path.
