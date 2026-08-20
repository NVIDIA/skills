# Streaming Validation And Build Order

## 15. Validate The Omniverse Realtime Viewer

Use this as a routine smoke checklist. Add focused proof only for features the
viewer actually exposes or for demo-critical claims.

Validate in this order:

1. Run the generated server setup wrapper or equivalent commands and record dependency installation results.
2. Confirm `ovrtx` imports, constructs a renderer on the target GPU, and reports a usable version.
3. In an isolated one-shot probe, confirm `ovstream.initialize()` and
   `ovstream.shutdown()` complete in the generated server environment. Do not
   reuse that shutdown pattern as the per-server stop path for same-process
   restarts.
4. Confirm `warp` initializes when CUDA conversion is enabled, confirm the
   selected OVStage path if used, and confirm the selected `pxr` subprocess path
   if USD queries are generated.
5. If physics is requested, confirm either a verified `PhysX.attach_ovstage(stage, read_ordinal=...)`
   compatibility probe or the bounded OVPhysX worker path. Do not validate
   parent-process OVPhysX population as an acceptable fallback.
6. Start the server with no client and confirm it initializes ovstream, constructs ovrtx, loads or waits for a scene, and exits cleanly on interrupt.
7. Wait for `/healthz` to return `200 ok`, or capture the server log and failing readiness command.
8. Confirm the server logs the first valid converted frame, for example `First BGRA frame ready: 1280x720`.
9. Before the frontend connects, record the H264 embedded-host preflight: iframe status, `MediaCapabilities.decodingInfo` availability/result when present, and the explicit H264 DirectConfig. In an embed, missing MediaCapabilities is diagnostic only, not a reason to reject H264.
10. Start the frontend and confirm the browser connects to the signaling port.
11. Confirm video appears and continues updating: after `connect()` resolves, the video element reaches `HAVE_CURRENT_DATA` and decoded-frame count (or `currentTime` when frame counters are unavailable) advances. If the server starts idle,
   confirm the connection stays alive on fallback frames long enough for the
   frontend to send `openStageRequest` after `AppStreamer.connect()` resolves.
12. Confirm colors are correct with a scene containing obvious red and blue objects.
13. Confirm `openStageRequest` loads a scene and returns `openStageResult`.
14. Confirm initial-state push works by loading a scene before connecting the browser, then refreshing the browser.
15. Confirm server logs do not show immediate `connected=True` then `connected=False` loops after normal app messages arrive.
16. Confirm camera orbit, pan, zoom, wheel zoom, and fit-to-stage update the streamed view.
17. Confirm click selection does not fire after a drag gesture.
18. Confirm selected prim state appears in the tree and info panel through `stageSelectionChanged`.
19. Confirm `getChildrenRequest`, `getPropertiesRequest`, and `getVariantsRequest` responses include the requested prim path and render in the correct UI row/panel.
20. Confirm scene switching clears stale selection, refreshes hierarchy, preserves render settings, and avoids concurrent render/reset.
21. For visible render settings, confirm either an immediate before/after effect,
    backend effective-state echo, explicit reload/profile behavior, or
    unsupported-key rejection. Do not require broad artifact capture for hidden
    or unimplemented settings.
22. Confirm `setRenderSettingRequest` rejects unsupported keys and that success responses include `applied`, `applies_at`, and `requires_reload`.
23. Confirm render settings persist after scene switch and server restart only for settings that were validated or accepted as non-live defaults.
24. For optional runtime transform, pick-effect, or physics controls, capture one
    focused before/after proof for each requested capability and include the
    backend capability/result message. For physics, include parent/child process
    model, OVStage write/consume ordinals, and nonzero motion when motion was
    requested.
25. Confirm frontend reconnect does not flood logs or leave `previous session already running` loops.
26. Confirm per-server shutdown calls ovstream stop and close. Confirm the
   process-exit path calls `ovstream.shutdown()` only after the final server is
   closed, and same-process restart tests do not call shutdown between starts.

Use these failure checks:

- Browser connected but no video: verify the video element exists before connect, frontend did not set media port, explicit H264 codec config is present, server public IP/ICE is valid, and frames are being submitted. In an embedded host, also record iframe context and `MediaCapabilities` availability, then require `HAVE_CURRENT_DATA` plus advancing decoded frames; missing capability APIs alone do not prove H264 failure.
- Video has swapped colors: verify CUDA RGBA-to-BGRA conversion.
- Black frame: verify render product path, camera path, render var source, resolution, and camera transform.
- Scene load works once but fails after switching: verify renderer reset/load serialization, inline sublayer paths, and operation error handling.
- Messages do nothing: verify browser envelope unwrapping, exact `event_type` names, data-channel readiness, and send guard.
- Camera moves incorrectly: verify row-major camera matrix layout, world-up convention, finite state, input button mapping, viewport ownership, and letterbox transform. If the data channel works but native `on_input` receives no pointer/button/wheel events during an explicit test, record that evidence and validate the mutually exclusive `mouseInput` fallback from `viewer-input-routing`.
- Picking fails: verify native pick query enqueue/step/result handling, input routing, pick coordinate transform, RenderProduct GPU pinning when required by the active ovrtx build, and no picking during load/reset.
- UI stays loading after refresh: verify server pushes initial `openStageResult` and root `getChildrenResult` after connection.
- Render setting appears to work but image does not change: verify the control came from the backend capability list and that `renderSettingsChanged.applied` is true for immediate settings.

Read for depth: see `references/streaming-lifecycle`, `references/streaming-server`, `references/streaming-client`, `references/stage-loading`, `references/viewer-input-routing`, `references/camera-controls`, and `references/object-selection` for full debugging contracts.

## Recommended Build Order For Agents

Follow this sequence when implementing from scratch:

1. Create the project skeleton and dependency files.
2. Create generated setup and run wrappers for the server environment.
3. Build server config, ovstream lifecycle, and render-loop shell with no scene features.
4. Add ovrtx renderer construction and a minimal user-provided or generated validation stage load.
5. Add inline root/session setup with `LdrColor` and confirm one streamed frame path.
6. Add CUDA RGBA-to-BGRA conversion and continuous frame streaming.
7. Build the React streaming provider and video viewport.
8. Connect browser to server and validate live video before adding UI panels.
9. Add message router with envelope unwrapping and guarded sends.
10. Add `openStageRequest`, loading state, and initial-state push.
11. Add normalized input routing, camera handling, and live camera writes.
12. Add hierarchy and properties queries.
13. Add selection and native selection outline feedback.
14. Add scene picker and scene switching.
15. Add render settings capabilities, immediate apply paths, and persistence only for validated settings.
16. Run the validation checklist and fix failures before adding deployment or optional overlays.

Do not skip the live-video milestone. If the first implementation includes every feature before video is proven, failures become hard to isolate.

Read for depth on streaming Omniverse Realtime Viewers: see `references/streaming-client`, `references/streaming-server`, `references/streaming-messages`, `references/ovrtx-rendering`, `references/viewer-input-routing`, and `references/camera-controls`.
