# Browser WebRTC Viewer

For a new browser USD viewer, build one decoded OVRTX frame through ovstream,
then add the requested interactions. Keep an existing app's architecture.
Framework, module layout and app-event names are application choices.

## Route The Current Work

Read the relevant sections below as needed; the full document stack is not a
prerequisite. Each linked reference owns its implementation contracts.

| Work | Owning reference |
|---|---|
| Resolve current libraries | [Dependencies](../dependencies/README.md) |
| Server startup and GPU frame handoff | [Streaming server](../streaming-server/README.md#startup), including [RGBA to BGRA](../streaming-server/README.md#rgba-to-bgra) |
| Browser connection and decoded-frame readiness | [AppStreamer Direct API](../streaming-client/README.md#appstreamer-direct-api) |
| App commands and state | [Streaming messages](../streaming-messages/README.md) |
| Navigation | [Input routing](../viewer-input-routing/README.md), then [camera controls](../camera-controls/README.md) |
| Part inspection | [Native picking](../native-picking-selection/README.md) and [attribute reads](../stage-attribute-reads/README.md) |
| Scene switching | [Stage management](../stage-management/README.md) |
| Reconnect, idle state and shutdown | [Streaming lifecycle](../streaming-lifecycle/README.md) |
| Additional UI, settings or deployment | Select only requested capabilities through [routing](../routing.md) |

For a fuller assembly example, use `server-runtime.md`, `client-protocol.md` and
`interaction-features.md`. `project-structure.md` offers an optional React layout;
it is not a required file list or a requirement to build every example panel.

Validate the actual delivery path using [validation](../validation.md); use
`validation-build-order.md` for detailed checks of implemented capabilities.
Capture a real scene and advancing decoded video before expanding the app.
