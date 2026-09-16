# Troubleshooting

## Triggers

Use this skill for troubleshoot, debug Omniverse Realtime Viewer, server won't start, no video, data channel not working, wrong colors, black frame, scene won't load, camera doesn't move, picking broken, or WebRTC internals.

Use this when an Omniverse Realtime Viewer fails during startup, streaming, scene loading, rendering, input, selection, or UI state sync. Start by classifying the first broken boundary instead of chasing downstream symptoms.

## Triage Flowchart

```text
Server process does not start
  -> Server startup diagnostics
Server starts but browser cannot connect
  -> WebRTC signaling and port diagnostics
Browser connects but video is blank or frozen
  -> Video streaming and renderer step diagnostics
Embedded host hides media capabilities or connection does not decode a frame
  -> Embedded-host H264 preflight and decoded-frame diagnostics
Electron SHM viewer opens but viewport is black, disconnected, or stutters
  -> Electron SHM Viewer diagnostics
Tauri or desktop WebView drag loses release, sticks, or selects after a drag
  -> Desktop WebView Pointer Lifecycle in viewer-input-routing
Video streams but colors/materials/frame are wrong
  -> Render var, BGRA conversion, camera, MDL diagnostics
UI connects but buttons/tree/settings do nothing
  -> Data-channel diagnostics
Stage open fails or loads as empty
  -> Scene loading and asset path diagnostics
Scene loads but camera does not move
  -> Input callback and live camera write diagnostics
Camera works but click selection is wrong or empty
  -> Picking, coordinate, and segmentation diagnostics
Selection works but highlight/info/tree is stale
  -> Derived state reset and message routing diagnostics
Physics or runtime transforms crash, no-op, or desynchronize
  -> OVStage/OVPhysX boundary diagnostics
Scene switch crashes or hangs
  -> Renderer ownership and stage reset diagnostics
```

## Fast Rules

- Debug one boundary at a time: process startup, WebRTC connection, rendered frame, JSON message, USD query, then feature state.
- Keep one render thread as the owner of `renderer.step()`, `reset_stage()`, `open_usd()`, `open_usd_from_string()`, reference mutation, native pick queries, selection outline writes, and live `write_attribute()` calls.
- In OVStage-backed apps, the parent viewer runtime also owns stage lifetime,
  ordinals, write floors, renderer attachment, and publication.
- Keep `pxr` queries and OVPhysX USD population in workers unless the exact
  ABI/import path is verified. A failed `PhysX.attach_ovstage(stage, read_ordinal=...)` bridge must
  not fall back to parent-process OVPhysX population.
- In WebRTC streaming apps, mouse, wheel, keyboard, and touch input normally arrive through NVST/ovstream `InputEvent` callbacks; app state commands use the JSON data channel. If the data channel works but an explicit viewport test produces no native callback, use the documented mutually exclusive browser `mouseInput` fallback.
- A frontend "loading forever" state is usually either a missing `openStageResult`, a missed proactive state push, or a message-name mismatch.
- For an Electron app with a Python sidecar on Windows, first run the App-Local
  Desktop Preflight in `windows-native-setup`; then diagnose the process and SHM
  lifecycle through `electron-shm-viewer`.
- A local Omniverse Realtime Viewer skips WebRTC entirely. If the same renderer and scene code works locally but not in a browser, focus on ovstream, frame conversion, and the standalone ovstream Direct AppStreamer config.

## Scenario Playbooks

Detailed startup, streaming, scene, input, selection, hierarchy, and recovery playbooks live in `scenario-playbooks.md`.

## Common Error Map

| Message or symptom | Actual cause | Usual fix |
|---|---|---|
| `ModuleNotFoundError: ovstream` | Python binding missing or native lib path missing | Install ovstream and set `OVSTREAM_LIB_PATH` |
| `CRenderApi not found` | ovrtx plugin tree not resolved | Set `OVRTX_BIN_PATH` and plugin library path |
| `usd-core detected` | ovrtx USD check found another USD package | Set `OVRTX_SKIP_USD_CHECK=1` before ovrtx work |
| `multiple debug symbol definitions for SDF_ASSET` | Two USD registries loaded | Put ovrtx bundled libs first or split pxr into worker |
| `_tf` import failure | USD DLL/shared library conflict | Fix import order or use subprocess queries |
| Duplicate USD registry, `TfType::AddAlias`, or repeated USD plugin/type aliases | `pxr`/`usd-core` or another USD-populating runtime loaded beside OVRTX | Move OpenUSD queries or physics USD population behind a process boundary |
| `Default.mdl` parse crash | Renderer initialized after wrong USD registry | Fix import/construction order |
| `PhysX.attach_ovstage(stage, read_ordinal=...)` reports missing `ovstage_register_consumer`, `ovstage_register_output_buffer`, `ovstage_publish`, `ovstage_query_changes`, `ovstage_read_attribute`, or `ovstage_write_attribute` | OVPhysX was built against OVStage bridge symbols unavailable in the installed OVStage wheel | Use the bounded OVPhysX worker path and report the ABI mismatch |
| Native crash after parent-process OVPhysX population | Competing USD population entered the process that owns OVRTX/OVStage | Move OVPhysX population to a child worker and hand poses back as JSON |
| Magenta materials | MDL resolver path missing | Set `OVRTX_BIN_PATH` and library path |
| `Unable to find RenderProduct prim` | Inline/session render path missing or mismatched | Create the render pipeline and pass the exact path |
| Black frame, no exception | Camera, RenderProduct, resolution, or RenderVar invalid | Validate stage-loading data and camera relation |
| USD parse error near `RenderVar` inline braces | ovrtx parser rejected one-line `def RenderVar "X" { ... }` syntax | Use multi-line `RenderVar` definitions from `stage-loading` |
| `RenderProductSetOutputs` has no attribute `get` | `renderer.step()` output was treated as a dict | Use `with products as ctx:` and index with `ctx[render_product_path]` |
| Invalid output handle | Frame or render var view outlived its step result | Copy buffers before leaving the `RenderProductSetOutputs` context |
| First `renderer.step()` exceeds normal test timeout | Cold RTX shader or pipeline compilation | Use a 300s+ first-run timeout and inspect ovrtx logs |
| Red/blue swapped | RGBA submitted as BGRA | Convert ovrtx `LdrColor` before ovstream |
| Scene renders but appears sideways | Camera fit, orbit, or pan hard-coded Y-up for a Z-up stage | Query `UsdGeom.GetStageUpAxis` in the metadata worker, set the server camera's `world_up`, refit, and keep the source USD unchanged |
| Stage load reports success but first frame fails | load operation status or RenderProduct path was not checked | wait/check load status, then step the exact RenderProduct |
| `TypeError: a coroutine was expected` from `ui.run` | ovui run loop received a callback/function instead of an awaitable | Pass an async render loop coroutine and yield with `await asyncio.sleep(1.0 / 120.0)` |
| `VIEWPORT_CAMERA_POSE_SOURCE` import failure | stale data adapters installed with newer local UI packages | Install local UI packages from the same package set |
| `ovui-data-adapters` is not installable | selected package set lacks matching package metadata | Use a compatible package set from `references/dependencies` |
| Native UI package requires a compiler toolchain | package/build instructions require local tools | Follow the current `ovui` dependency guidance |
| `Previous session is already running` | Old WebRTC client/session still active | Close old tab, reduce reconnect storm, restart server if stuck |
| `Could not get encoded frame` with black video | Direct client codec default was unknown or unsupported | Set explicit `codec: 'H264'` and `codecList: ['H264']` in DirectConfig |
| Embedded host lacks `navigator.mediaCapabilities` | Host frame does not expose the optional capability API | Record iframe/preflight state, retain explicit H264 config, and require decoded-frame validation; do not reject H264 from API absence alone |
| `connect()` resolves but no decoded video frame | Signaling/data channel success was mistaken for media proof | Check video `readyState`, decoded-frame counter/currentTime, WebRTC internals, and host-frame policy |
| Connects then disconnects in 1-2 seconds before messages arrive | Server did not stream fallback/idle frames while waiting for a stage command | Stream a valid fallback or last-good BGRA frame every connected tick |
| Initial app messages never reach the server | Frontend gated `sendMessage()` on `onStart` instead of the `connect()` Promise | Send app commands after `AppStreamer.connect()` resolves; use `onStart` only for video-live state |
| Keyboard/data-channel controls work but mouse drag, button, and wheel produce no server `on_input` events | Native browser input forwarding is unavailable in this host/embed | Verify viewport ownership and callback logs, then enable only the mutually exclusive `mouseInput` fallback in `viewer-input-routing`; do not process native and fallback events together |
| `NVST_R_INVALID_OPERATION` on second `Server.start()` | `ovstream.shutdown()` ran in a same-process stop path | Use `server.stop()` and `server.close()` for restarts; reserve `ovstream.shutdown()` for process exit |
| Server sees `messageType` but no `event_type` | AppStreamer envelope not unwrapped | Parse nested `data` payload before dispatch |
| `POST /sign_in` returns HTTP 501 | For a standalone ovstream Direct deployment, frontend used a connection profile intended for a different streaming product or injected auth/session fields | Rebuild the frontend from `streaming-client` using `@nvidia/ov-web-rtc` Direct mode with only the exposed `server` and `signalingPort` |
| UI waits on `getChildrenResponse` | Protocol name mismatch | Use active `getChildrenResult` route |
| Picks return old prims | pending pick/selectable state survived scene reload | clear pick state and refresh native pickability |
| Highlight visible on load | native selection outline groups not cleared | write group `0` for stale selected paths and clear runtime selection |
| Textures missing only after composition | cache path or sublayer path broke relative references | preserve cache layout and quote the original asset path correctly |

## Streaming And Local Omniverse Realtime Viewer Paths

Streaming path:

- Use `streaming-server` for ovstream lifecycle, ports, input callbacks, and frame submission.
- Use `streaming-client` for standalone ovstream Direct config, video element setup, browser diagnostics, and guarded sends.
- Use `streaming-lifecycle` when the connection exists but state, messages, or reconnects are wrong.
- Use `streaming-messages` to verify exact JSON event names and payload shapes.

Local Omniverse Realtime Viewer path:

- Use `local-viewer` for ovui shell, image display, UI-thread rules, and coordinate mapping.
- Use `ovrtx-rendering` for renderer construction, frame extraction, and live attribute writes.
- Use `stage-loading` for render prim injection and RenderProduct failures.
- Use `viewer-input-routing`, `camera-controls`, `object-selection`, `stage-hierarchy`, and `prim-info-display` for feature-specific debugging once the frame renders correctly.
