# Streaming Server Runtime

## 2. Install Dependencies And Configure The Environment

*Read `references/dependencies` FIRST.* Its `nvidia-runtime.md` file is
the source of truth for NVIDIA runtime locations. Do not guess install paths and
do not repeat `ovstream`, `ovui`, `ovrtx`, or `ov-web-rtc` client acquisition
details in this recipe.

Do this on the server side:

- Install `ovrtx` using the package guidance in `references/dependencies`.
- Install `ovstage` using the package guidance in `references/dependencies`
  when the viewer runtime uses OVStage as the live stage substrate.
- Install `ovstream` using the current package guidance in `references/dependencies`.
  If `ovstream.initialize()` reports missing native StreamSDK libraries, follow
  the current dependency guidance.
- Install `warp-lang` for CUDA-side channel conversion (`pip install warp-lang`).
- Install `numpy` for matrices and camera math (`pip install numpy`).
- Install `usd-core` only when the app creates a USD query worker. Resolve a release compatible with the selected runtime package set after reviewing the current upstream runtime documentation and release notes. Keep
  routine `pxr` imports out of the server process that owns OVRTX and live
  OVStage unless the exact wheel set and import path are verified.
- Install `ovphysx` only in a bounded physics worker or an isolated
  attached-stage compatibility probe. Do not add it to the parent server
  environment unless the selected architecture has verified
  `PhysX.attach_ovstage(stage, read_ordinal=...)` for the exact OVStage/OVPhysX ABI.
- If the streaming server also selects `ovui` or a headless overlay UI package,
  resolve the Python environment from the current upstream `ovui` docs and
  package metadata before creating the venv. Plain browser-streaming servers
  that use only `ovrtx`, `ovstream`, and `warp` should follow those runtime
  wheels' compatibility instead.

For generated browser viewers, create app-local setup and run wrappers rather
than pointing users at repo-level scripts. The setup wrapper should install the
server dependencies above, run the verification checks from
`references/dependencies/environment-validation.md`, and fail with the exact
command that did not pass. The run wrapper should set `OVRTX_SKIP_USD_CHECK`,
derive `OVRTX_BIN_PATH` from the installed `ovrtx` package when needed, preserve
the selected ovstream package layout, start the server, and write logs to a
known project-local path.

Do this on the frontend side:

- Create a Vite React app (`npm create vite@latest frontend -- --template react-ts`).
- Configure and install `@nvidia/ov-web-rtc` using `references/dependencies`.
  For Brev, OKAS, or another portable orchestrator, use the standalone `ovstream` Direct connection pattern from `streaming-client` and convert the exposed endpoint into Direct `server` and `signalingPort` values. For NVCF self-hosted deployment, read `cloud-deployment/nvcf-self-hosted.md` and follow the upstream workflow before selecting the browser connection model.
- Add normal UI dependencies only when they are part of the requested UI. Keep the core connection path dependency-light.

Set these environment contracts before starting the server:

- `OVRTX_SKIP_USD_CHECK=1` must be set before ovrtx is imported or the renderer is constructed.
- `OVRTX_BIN_PATH` must point at the ovrtx `bin` directory when materials or renderer plugins fail to resolve.
- The ovrtx plugin library path must be first in the dynamic library path if another USD build is present.
- `OVSTREAM_LIB_PATH` must point at the ovstream native library directory when the Python binding cannot find native libraries.
- If `ovstream` native libraries are not found, use the current runtime
  guidance from `references/dependencies` instead of copying stale fallback
  library paths from older recipes.

Decision points:

- If the target is local development, use WebRTC signaling port `49100`, stream port defaulting through ovstream, and public IP `127.0.0.1`.
- If the target is LAN access, expose signaling host and port through frontend URL parameters and environment variables.
- If the target is cloud deployment, do not invent a new deployment model in this recipe; read `references/cloud-deployment` and keep to supported paths.
- If the feature needs OpenUSD inspection, prefer a separate USD query
  subprocess rather than importing `pxr` in the OVRTX/OVStage parent process.
- If the feature needs OVPhysX, prefer a bounded child worker that returns pose
  samples as JSON unless the installed `PhysX.attach_ovstage(stage, read_ordinal=...)` bridge has been
  verified.

Common failure modes:

- `usd-core detected`, duplicate USD debug symbols, `_tf` import failures, or MDL resolver crashes usually mean import order or library path is wrong.
- Magenta materials usually mean `OVRTX_BIN_PATH` or plugin library path is missing.
- ovstream import or initialize failure usually means native StreamSDK libraries are missing, `OVSTREAM_LIB_PATH` points at the wrong layout, the platform build does not match, or GPU/driver support is missing.
- `No matching distribution found for ovrtx`: wrong package guidance, unsupported platform, or unsupported Python version; re-check `references/dependencies`.
- ovstream package not found: wrong package source, stale package metadata, wrong platform tag, or network/proxy issue; re-check `references/dependencies`.

Read for depth: see `references/dependencies` for install commands, `references/ovrtx-rendering` and `references/streaming-server` for the full environment and native library contracts.

## 3. Build The Server Runtime Shell

Do this:

- In `server/ov_web_viewer_server.py`, parse width, height, target FPS, signaling port, public IP, initial scene URL/path, asset root, and settings path.
- Construct a single application runtime object that owns the renderer runtime,
  OVStage/session state when selected, scene manager, stream server, message
  router, input router, settings store, and command queue.
- Set the environment before import; use
  [ovstage-ovrtx-integration](../ovstage-ovrtx-integration/README.md) and the
  selected upstream runtime example for schema registration, renderer/stage
  initialization, and attachment. Publish the initial scene, warm up the
  renderer, then initialize ovstream and register callbacks before starting it.
- Initialize ovstream once, create a WebRTC server, register `on_connection`, `on_app_message`, and `on_input`, then start the server.
- Enter one render loop that drains queued commands, updates camera/selection/settings state, steps ovrtx when a scene is loaded, converts the frame to BGRA, and submits video to ovstream.
- While a client is connected, submit a fallback or last-good BGRA frame on every
  tick even when no scene is loaded or a stage load lock is held.
- On per-server stop, stop streaming and close the ovstream server. Call
  `ovstream.shutdown()` only from the top-level process-exit path after the
  final server is closed; do not call it between same-process restarts.

Log elapsed time before and after renderer creation, schema registration,
stage creation/attachment, population/publication, first step, and output copy.
If a phase stalls, follow [startup diagnostics](../troubleshooting/scenario-playbooks.md#server-startup)
before adding transport or attributing the failure to the host. Keep server
frame readiness separate from data-channel readiness and browser decoded video.

Startup sequence (runtime initialization follows the selected SDK example):

```text
set OVRTX_SKIP_USD_CHECK=1 before import
start health endpoint with readiness false
start runtime owner:
  initialize renderer and stage with schemas registered before USD consumers
  populate and attach initial stage with viewer render configuration
  initialize camera and native selection policy/styles
  commit initial publication and warm up the displayed product
  validate outputs and copy a valid frame into owned display storage
  initialize transport and register connection/message/input callbacks
  start transport; report startup success and mark ready
  continue queued mutations, rendering and frame handoff on this owner
on stop or owner failure:
  withdraw readiness and stop accepting work
  stop submissions; join any separate submission worker
  release output buffers only after their consumers finish
  finish native operations; detach and destroy on the owner
  propagate completion or original failure to the supervising caller
```

Critical contracts:

- Register callbacks before `server.start()`.
- Keep callbacks fast. They may be called from StreamSDK internal threads.
- Do not call renderer load/reset/step/write APIs directly from ovstream callbacks. Enqueue work for the render loop.
- Callbacks enqueue commands. The runtime owner applies stage changes and the
  configured physics owner runs simulation; USD-only queries can use a worker.
- Use the app-message adapter from `streaming-messages`; restore viewer state
  on reconnect using `streaming-lifecycle`.
- Maintain explicit runtime states: starting, idle/no scene, loading, streaming, error, shutting down.
- Keep the most recent loaded stage URL, root hierarchy summary, selection, render settings, and loading state in server memory so a newly connected browser can receive initial state.
- Readiness is not liveness: `/healthz` must return `503 not ready` until the render loop has produced and copied the first valid frame, then `200 ok`.
- Readiness must not depend on an active browser client or on `stream_video()` succeeding. Before any client connects, frame submission can no-op or raise transient no-client/disconnect errors depending on the selected ovstream build.
- A connected idle client still needs continuous video. Stream a valid fallback
  frame until the first real stage frame exists, then keep streaming the last
  good frame during scene-load locks.

Decision points:

- If no initial scene is configured, start the server in idle state and let the frontend send `openStageRequest`.
- If an initial scene is configured, load it before or during the first render loop iteration, then push state when the client connects.
- If one client is already connected and a second browser connects, follow ovstream's one-client WebRTC constraint. Either reject the new client clearly or replace the old session intentionally.
- If the render loop falls behind target FPS, prefer dropping frames or reducing render quality over running concurrent renderer steps.

Common failure modes:

- Registering callbacks after `start()` causes missed connection and data-channel events.
- Calling `renderer.step()` during `reset_stage()`, `open_usd*()`, or reference mutation causes races, stale buffers, or crashes.
- Sending messages while no client is connected silently loses important state unless the runtime pushes initial state after connection.
- Calling `ovstream.shutdown()` in a per-request stop path can corrupt native
  state for later same-process `Server.start()` attempts. Use stop/close for
  restarts and reserve shutdown for process exit.

Read for depth: see `references/streaming-server` and `references/streaming-lifecycle` for the full lifecycle contract.

## 4. Construct The ovrtx Renderer

Do this:

- Create the renderer in `server/renderer_runtime.py` after environment variables are set.
- Use synchronous rendering first. Add asynchronous rendering only after buffer lifetime, readiness, and stream pacing are explicitly handled.
- Store the active render product path, stream width, stream height, current frame index, and whether a valid stage is loaded.
- Expose render-loop-only operations for loading a scene, resetting the stage, stepping a frame, mapping render vars, enqueueing and decoding pick queries, setting selection outline groups, and writing live attributes.

Critical contracts:

- The application calls `renderer.step()` explicitly. ovrtx does not run a hidden app loop for the viewer.
- Pass the exact viewer RenderProduct path to every step call.
- Extract `LdrColor` from the returned frame. This is RGBA8 from ovrtx.
- For AOVs, handle current single-tensor and multi-tensor render var outputs.
  Single-tensor outputs are consumed directly through DLPack; multi-tensor
  outputs must select a named image tensor and read params separately. Image
  tensors are channel-last (`H x W x C`), not channel-first.
- Map CUDA buffers for streaming. Avoid CPU round trips in the normal streaming path.
- Keep mapped or converted frame buffers alive until `stream_video()` returns.
- Write live camera and object transforms through the application-owned
  OVStage runtime adapter using local `omni:xform` values. Wait for writes,
  advance the write floor, then render the committed ordinal. See
  `references/prim-transform-safety` for the shared transform/publication flow.
- Direct `renderer.write_attribute` and renderer bindings are only for an
  explicitly standalone renderer-owned compatibility path. Follow
  `references/ovrtx-rendering` for its semantic and create-new prim mode; do not
  mix those writes with an attached OVStage. Keep native picking and selection
  outline groups on OVRTX.

Decision points:

- If the app only needs basic video streaming, start with `LdrColor` only.
- Object selection uses native pick queries. Do not add segmentation render vars just to make picking work.
- If render settings include debug segmentation view, keep both render vars available and switch the streamed image source intentionally.
- If the renderer reports stale GPU hangs after a crash, inspect running Python GPU processes before changing code.

Common failure modes:

- `Unable to find RenderProduct prim` means scene setup did not create the path used by `renderer.step()`.
- Black frame usually means camera relation, render product resolution, render var source, or camera transform is invalid.
- Red/blue color swap means ovrtx RGBA was submitted to ovstream without BGRA conversion.
- Live camera changes doing nothing usually means the app wrote
  `xformOp:transform` instead of `omni:xform`, used existing-only prim mode, or
  published above the OVStage write floor.

Read for depth: see `references/ovrtx-rendering` for the full renderer construction, frame extraction, and live attribute contract.

## 5. Implement Scene Loading

Do this in `server/scene_loader.py` and call it only from the render loop:

- Resolve the requested URL/path against the configured asset root, allowed schemes, and security policy.
- Create viewer-owned camera, RenderProduct, RenderVar, and RenderSettings data in one inline root USDA string when the user stage lacks viewer render config.
- Load the user stage without modifying it.
- Store the viewer camera path and render product path in runtime state.
- Reset selection, native selection outline groups, hierarchy cache, pending pick queries, and loading progress for the new stage.
- Fit the camera to the stage bounds unless the user requested preserving the current camera or using an authored stage camera.

Critical contracts:

- Every loaded stage needs Camera -> RenderProduct -> RenderVar -> RenderSettings wiring that ovrtx can find.
- The viewer camera path must be the same path used by camera controls when writing `omni:xform`.
- Do not inject lights unless the user explicitly asks for viewer-controlled lighting. User stages usually own their lighting.
- Include segmentation render vars only for explicit debug/AOV display modes, not for picking.
- Prefer OVStage `population.open_usd_from_string()` for inline roots that
  sublayer the user USD and author viewer render config. Commit its ordinal
  before the attached renderer consumes it; this preserves relative asset
  resolution without falling back to a renderer-owned stage.
- Do not call renderer-owned reference or layer-add APIs in an attached viewer.
  Use OVStage population reference operations through the serialized runtime
  path.

Decision points:

- Use a single inline root USDA string with `subLayers = [@user_scene@]` when the user file needs viewer camera/render-product/render-var data.
  Copy source `upAxis` and `metersPerUnit` onto that root as described in
  [stage-loading](../stage-loading/README.md#compose-viewer-owned-prims).
- If the user stage has an authored camera and the requested policy is `stage-camera`, copy its focal length, apertures, clipping range, and transform into the viewer camera.
- If the user requests persistent camera across scene switches, keep camera state but sanitize and refit only when the old state is invalid for the new bounds.
- If the user requests viewer lighting controls, add explicit viewer-owned light prims only with a verified live apply path or an explicit reload/profile workflow; otherwise leave lighting untouched and omit live lighting controls.

Common failure modes:

- Inline roots that omit or misquote the user sublayer path fail composition or break relative asset resolution.
- Camera path mismatch makes input appear connected but the view never moves.
- A stage-load operation that reports an error must not be treated as a successful load just because the enqueue call returned.

Read for depth: see `references/stage-loading`, `references/render-settings`, `references/selection-feedback`, and `references/stage-hierarchy` for the full contracts.

## 6. Build Frame Streaming

Do this in `server/stream_server.py` and `server/frame_converter.py`:

- Start ovstream in WebRTC mode.
- Configure width, height, target FPS, signaling port, optional public IP, and video codec policy.
- For each rendered frame, map `LdrColor` on CUDA, convert RGBA8 to BGRA8 on CUDA, wrap the CUDA buffer as an ovstream video frame, and submit it.
- Pace the render loop to target FPS while allowing command queue work to run between frames.
- Send loading and error state messages even when video frames are temporarily unavailable.
- Log the first valid BGRA8 CUDA mapping or completed required conversion into the app-owned BGRA stream buffer and set readiness before depending on browser connection state. This lets validation distinguish renderer/frame-conversion failures from WebRTC negotiation failures.

Critical contracts:

- ovstream expects BGRA8 for raw CUDA frames; ovrtx `LdrColor` is RGBA8.
- Avoid CPU readback in the normal path. CPU readback is acceptable only for debugging screenshots or tests.
- `stream_video()` does not own a deep copy of the source buffer. Keep the buffer alive until the call returns.
- `stream_video()` can fail during disconnect races. Catch and debug-log transient failures; do not crash the render loop.
- First-frame readiness follows a valid BGRA8 CUDA mapping or completed required conversion into the persistent app-owned BGRA stream buffer. The `LdrColor` path above requires RGBA-to-BGRA conversion; already-BGRA8 output must not be swizzled again. Preserve producer synchronization and mapping/buffer lifetime as described in [streaming-server](../streaming-server/README.md#rgba-to-bgra). Do not wait for an attached browser client.
- Stream width, height, frame pitch, RenderProduct resolution, and camera aspect must agree for the fixed stream size.
- The WebRTC signaling port is not the media port. The browser discovers media endpoints through SDP.
- For local WebRTC development, set the server public IP to `127.0.0.1` when ICE discovery otherwise hangs.

Decision points:

- Use raw frames first. Add H264, H265, AV1, or custom encoded frames only when latency, bandwidth, or deployment constraints require it.
- If using encoded frames, make the encoder own color format conversion and frame lifetime explicitly.
- If no client is connected, still run scene loading and state updates, but consider pausing expensive frame submission work.
- If the user asks for RTSP, treat it as a separate delivery mode; the browser-streamed Omniverse Realtime Viewer path should remain WebRTC.

Common failure modes:

- Connected browser with no video often means frontend configured a media port manually, server ICE/public IP is wrong, or frame submission never happens.
- One frame then black usually means the frontend connected, received state, rerendered, and accidentally terminated/restarted the AppStreamer connection. Inspect server logs for `connected=True` followed by `connected=False` within about a second.
- Wrong colors mean missing RGBA-to-BGRA conversion.
- GPU memory growth can mean mapped frame resources or conversion buffers are retained beyond the intended frame lifetime.

Read for depth: see `references/streaming-server` and `references/streaming-lifecycle` for the full frame streaming contract.
