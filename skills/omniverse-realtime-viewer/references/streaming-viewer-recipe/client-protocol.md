# Streaming Client And Protocol

The connection and SDK envelope contracts apply to the chosen browser framework.
React provider and widget examples below are optional assembly patterns. Implement
only messages/capabilities requested by the app; event names must match both ends.

## 7. Build The React WebRTC Client

Do this in `frontend/src/streaming/` and `frontend/src/App.tsx`:

- Create a streaming provider that resolves host and signaling port from URL parameters first, Vite build-time environment variables second, a runtime `stream.config.json` file third, and browser hostname plus default port last. Follow the per-field resolution and validation rules in [Config Resolution](../streaming-client/README.md#config-resolution).
- Render the video element with the exact id expected by AppStreamer before attempting to connect.
- Connect with the Direct stream mode, resolved signaling host/port, fixed
  width/height, and explicit H264 codec policy from a stable React effect.
- Before connection, record an embedded-host H264 preflight; after connection, verify an actual decoded frame rather than treating signaling or `onStart` as video proof.
- Style the video with `object-fit: contain` so the fixed-resolution stream scales to the available page area without distortion.
- Expose status, error message, connection lifecycle events, a guarded send helper, and custom-event subscription to the rest of the app.
- Mount the viewport video first, then overlay DOM controls such as toolbar, scene picker, tree, property panel, status, and settings.
- Clean up custom-event subscriptions on component unmount.

Critical contracts:

- Use `signalingServer` and `signalingPort` as in the [owning client example](../streaming-client/README.md#appstreamer-direct-api); `server` is a legacy fallback.
- Do not configure `mediaServer` or `mediaPort` for the browser client. SDP negotiates media.
- Do not construct sign-in URLs, append custom signaling paths, or add
  auth/session fields to the browser WebRTC config. Orchestrators may launch and
  route the session, but the frontend should receive a standalone `ovstream`
  Direct host and signaling port.
- The video element must exist before connect starts.
- Embedded hosts: before connecting, record iframe status and the availability/result of `navigator.mediaCapabilities.decodingInfo` for H264 when present. This is advisory: an unavailable capability API in Tableau or another host frame must not cause an automatic codec rejection. Continue with the explicit H264 DirectConfig, then require decoded-frame proof.
- Gate initial data-channel sends on connected status. Connected status means
  confirmed SDK start success, not just a resolved `IN_PROGRESS` result.
  Follow [client readiness](../streaming-client/README.md#appstreamer-direct-api).
- Keep the `streamer.connect()` effect dependent only on immutable connection config. Do not let stateful message routers recreate the effect after `openStageResult`, `getChildrenResult`, or status events arrive.
- Catch rejected `streamer.sendMessage()` Promises during connect/disconnect windows.
- Give DOM overlays explicit `z-index` above the video layer.
- Use modest reconnect settings to avoid repeated `previous session already running` churn.
- Let the streaming library forward mouse, keyboard, wheel, and touch input through NVST's native input channel. Only use `mouseInput` after the explicit native-callback failure diagnostic in `viewer-input-routing`; fallback and native input must never run together.
- Do not send viewport-size messages when CSS layout changes. Keep the stream fixed and rely on NVST letterbox coordinate mapping.

Decision points:

- If the app should auto-load a default scene, send `openStageRequest` only
  after confirmed SDK start success. Do not wait for a decoded scene frame;
  an idle server may need the request before it can render that frame.
- If the server loads an initial scene before the browser connects, rely on server initial-state push rather than requiring the frontend to guess.
- If the user wants a dense tool UI, keep it as DOM overlay around the video. Use server-side overlays only when they must be part of the streamed pixels.
- If the client runs from a different machine, expose host and signaling port through URL parameters.

### Production `stream.config.json`

For static production builds, fetch an app-owned `stream.config.json` at startup
before resolving the connection. Place it beside the served app entry point
(e.g., in the frontend `dist/` directory):

```json
{
  "signalingServer": "<server-ip-or-hostname>",
  "signalingPort": 49100
}
```

- `signalingServer`: Hostname or IP of the standalone ovstream signaling server,
  reachable from the browser; no URL scheme, path, or auth/session fields.
- `signalingPort`: Integer WebSocket signaling port (default `49100`).
- This JSON is application configuration, not an SDK connection profile. Map
  these two resolved values to `DirectConfig.signalingServer` and
  `DirectConfig.signalingPort` in the [Direct example](../streaming-client/README.md#appstreamer-direct-api).
  Keep its fixed resolution, explicit H264 policy, callbacks, and media element IDs.
  Omit `server`, `mediaServer`, and `mediaPort`; SDP negotiates media.

Resolve each field using URL parameters > Vite build-time environment variables >
runtime JSON > browser hostname/default port, as specified in
[Config Resolution](../streaming-client/README.md#config-resolution). Vite embeds
`VITE_*` values at build time; they work in production but changing a container's
environment after building does not update the bundle. Editing the runtime JSON
requires no rebuild and takes effect at the next app startup, provided a higher
priority URL or built-in environment value does not override that field. Serve
fresh configuration at startup (e.g., fetch with `cache: 'no-store'`). Dev-server
proxying does not apply to a static production server.

Common failure modes:

- An embedded host can hide `MediaCapabilities` even when the Direct H264 path is viable. Log that absence and validate the same viewer standalone and in the host frame; do not silently substitute a codec or browser renderer.
- A resolved `connect()` Promise or `onStart` without an advancing decoded-frame count is not a working video stream.

- Connecting before the video element exists produces a connection that cannot display video.
- Setting media port to the signaling port produces connected-with-no-video failures.
- Sending requests before data channel readiness drops initial scene loads.
- Waiting for a decoded scene frame before sending `openStageRequest` can deadlock idle
  servers because no stage frame is decoded until the request arrives.
- One rendered frame then black can be caused by React effect cleanup calling `streamer.terminate()` after normal app state updates.
- A frontend waiting for `getChildrenResponse` while the server sends `getChildrenResult` leaves the tree empty.

Read for depth: see `references/streaming-client` and `references/streaming-lifecycle` for the full React/AppStreamer contract.

## 8. Define The Data-Channel Protocol

Do this before wiring UI features:

- Define the app envelope as `event_type` plus `payload` in both directions.
- On WebRTC, register `on_app_message`: it already receives inner application
  JSON. Do not unwrap the SDK envelope again. Raw non-WebRTC transports use
  the explicit adapter in `references/streaming-messages/README.md`.
- Register handlers by exact event name.
- Validate each payload before mutating renderer or USD state.
- Send all responses and pushed events through one guarded send helper.

This shared example message set supports a complete viewer. Implement only the
requested capabilities; an existing app protocol may use other names:

| Flow | Client event | Required payload | Server event | Required payload |
|---|---|---|---|---|
| Open scene | `openStageRequest` | `url` | `openStageResult` | `url`, `result`, optional `error` |
| Reset scene | `resetStageRequest` | empty object | `openStageResult` or loading/error events | current URL/result |
| Loading state | `loadingStateQuery` | empty object | `loadingStateResponse` | `url`, `loading_state` |
| Progress amount | none | none | `updateProgressAmount` | `amount` |
| Progress activity | none | none | `updateProgressActivity` | `activity` |
| Get hierarchy children | `getChildrenRequest` | `prim_path`, optional `filters` | `getChildrenResult` | `prim_path`, `children` |
| Get properties | `getPropertiesRequest` | `prim_path` | `getPropertiesResponse` | `prim_path`, `properties` |
| Select prims | `selectPrimsRequest` | `paths` | `stageSelectionChanged` | `prims` |
| Make prims selectable | `makePrimsSelectable` | `paths` | optional status/error | implementation-specific |
| Make scene pickable | `makePrimsPickable` | optional filters | optional status/error | native pickability state |
| Get variants | `getVariantsRequest` | `prim_path` | `getVariantsResponse` | `prim_path`, `variants` |
| Set variant | `setVariantRequest` | `prim_path`, `variant_set`, `variant_selection` | `getVariantsResponse` plus reload/dirty events | updated variants |
| Set render setting | `setRenderSettingRequest` | setting key and value | `renderSettingsChanged` | full effective settings plus `result`, `applied`, `applies_at`, `requires_reload`, optional `message` |
| Query render settings | `getRenderSettingsRequest` | empty object | `renderSettingsChanged` | full effective settings plus supported-setting capabilities |
| Camera command | `cameraCommandRequest` | command and optional values | `cameraStateChanged` | current camera state |
| Fit camera | `fitCameraRequest` | optional target prim path | `cameraStateChanged` | current camera state |
| Error | none | none | `viewerError` | `code`, `message`, optional context |

Critical contracts:

- Client and server event names must agree exactly. The shared example uses
  `openStageResult`, `getChildrenResult`, and `stageSelectionChanged`; these are
  application events, not SDK-reserved names.
- `paths: []` in `selectPrimsRequest` clears selection.
- Children semantics must be stable: expandable-but-not-loaded is truthy, loaded children is an array, and leaf is null or absent.
- Properties and variants responses must include the requested `prim_path` so the frontend can ignore stale responses.
- Message handlers that load scenes, set variants, or change heavy settings must enqueue work for the render thread.
- Message handlers that only query cached state may respond immediately if they do not touch renderer-owned state.
- `setRenderSettingRequest` must reject keys that are not in the server capability list. Success means the active viewer state changed, or an explicit non-live action was accepted.
- For runtime prim discovery, use the application-owned OVStage adapter's
  `runtime.query_prims(...)` / `query_prims_async(...)` and return copied,
  resolved path strings as described in `references/stage-hierarchy`. Direct
  renderer queries are only for standalone renderer-owned compatibility paths.
  Use a `pxr` worker only for queries the runtime cannot supply; keep native
  picking and renderer-visible selection groups on OVRTX.
- Push `openStageResult` and root `getChildrenResult` after a client connects if a stage is already loaded.

Decision points:

- If stage hierarchy queries are slow or risky in the renderer process, move them behind a subprocess owned by `stage_queries.py`.
- If multiple UI panels can request the same data, make responses idempotent and keyed by prim path instead of relying on request ordering.
- Keep implemented message names in one app-owned definition; do not add
  unused optional handlers just to reproduce the example table.

Common failure modes:

- Using raw-envelope parsing on WebRTC `on_app_message` can drop already
  decoded application events. Keep transport adaptation at its boundary.
- A response name such as `getChildrenResponse` fails if the app client
  expects `getChildrenResult`; this is an application agreement, not SDK versioning.
- Sending app state before the data channel is ready causes permanent loading indicators unless initial state is pushed on connection.

Read for depth: see `references/streaming-messages`, `references/streaming-lifecycle`, and `references/stage-hierarchy` for the full protocol and query contracts.

## 14. Wire Frontend Components

Do this after the stream connects and core messages work:

- `Viewport` renders the video element and any DOM overlay controls.
- `Toolbar` sends fit camera, reset view, render settings toggle, and debug view commands.
- `ScenePicker` lists available scenes and sends `openStageRequest`.
- `StageTree` requests children lazily and sends `selectPrimsRequest` on row selection.
- `PrimInfoPanel` subscribes to selection and requests properties/variants for the active prim.
- `RenderSettingsPanel` renders backend-advertised capabilities, displays effective settings, and sends validated changes.
- `StatusBar` displays connection, loading state, current scene, FPS if available, and latest viewer error.

Critical contracts:

- The video element id must match AppStreamer config exactly.
- Keep data-channel sends behind the streaming provider's connected-state guard.
- Treat server events as authoritative for loaded scene, selection, settings, and errors.
- Do not duplicate pointer input handling in DOM unless it is for UI widgets outside the video viewport or the verified, mutually exclusive `mouseInput` fallback.
- Clean up event subscriptions when components unmount.

Decision points:

- If a panel is optional, still keep the provider and message types stable so the feature can be added later.
- If a UI widget overlaps the video, ensure it does not intercept pointer events meant for camera/selection unless the widget is actively being used.
- If the app needs keyboard shortcuts, decide which shortcuts are browser UI shortcuts and which should pass through to the streamed app input path.

Common failure modes:

- DOM overlays intercept all pointer events and camera controls stop working.
- Frontend local state diverges when it assumes requests succeeded instead of waiting for server events.
- Components leak subscriptions and process every server event multiple times after navigation.

Read for depth: see `references/streaming-client`, `references/streaming-messages`, and `references/streaming-lifecycle` for the full frontend contracts.
