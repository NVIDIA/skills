# Streaming Server

## Triggers

Use this skill for stream to browser, WebRTC server, ovstream server, stream_video, on_input, VideoFrame, RGBA BGRA, fixed resolution, or no video.

Use this for the Python server that renders with ovrtx and streams frames through ovstream.

## Runtime Ownership

The server owns the live OVStage, publication clock, and OVRTX renderer. Browser
callbacks and query workers exchange commands and structured results; mapped GPU
buffers and live stage handles stay with their owner. Establish the renderer
owner before construction: initialization, initial load, warmup, frame stepping
and teardown share that execution context. Use one runtime queue for scene,
camera, selection, and simulation changes.

Read [dependency guidance](../dependencies/README.md) for runtime setup and
[ovstream's transport contract](../dependencies/ovstream.md) before selecting
frame delivery. A headless streaming server has no inherent X11 requirement;
windowed UI backends have their own display setup.

## Startup

1. Set `OVRTX_SKIP_USD_CHECK=1` before import and start the runtime owner. On
   that owner, register schema paths before population and construct the runtime.
   Follow [stage loading](../stage-loading/README.md) to create the viewer-owned
   stage and attach its renderer.
2. Initialize the camera and render configuration, publish the initial ordinal,
   and warm up the displayed product. Discover only the AOVs the app exposes.
3. Prepare a valid BGRA8 CUDA frame before marking renderer readiness; retain
   an app-owned loading/last-good frame for idle startup or blocking scene work.
4. Initialize ovstream and register app, input, and connection callbacks before
   starting the server.
5. Continue stepping and applying queued mutations on that same owner. It
   converts and submits frames, or hands app-owned frames to a serialized
   submission loop when blocking scene work needs to run independently.

Startup reports success or the original error to its caller. On shutdown or
owner failure, withdraw readiness, stop accepting runtime work and stop frame
submissions. Join any separate submission worker before releasing its buffers.
Finish in-flight native work,
detach and destroy on the owner, attempting remaining cleanup after an earlier
cleanup failure. Propagate owner exit/failure to the process supervisor so a
stopped render loop cannot leave an apparently ready server.

A fixed-resolution WebRTC server using Warp-owned output buffers:

```python
import ovstream
import warp as wp


def start_stream(width, height, on_connection, on_app_message, on_input):
    # Call after process-level ovstream.initialize() and wp.init().
    server = ovstream.Server(ovstream.ServerType.WEBRTC)
    server.on_connection = on_connection
    server.on_app_message = on_app_message
    server.on_input = on_input
    config = ovstream.ServerConfig(
        width=width, height=height,
        video_input=ovstream.VideoInput.CUDA,
        cuda_device=0, cuda_context=int(wp.get_device("cuda:0").context),
        webrtc_resizable=False,
        webrtc_signal_port=49100,
    )
    try:
        server.start(config)
    except Exception:
        server.close()
        raise
    return server
```

The device and context must match the BGRA allocation. Leave endpoint and network
configuration to the deployment; for local loopback the public address can be
set to `127.0.0.1`. Register `on_unicode` as well when composed text input is needed.

## Transport Lifecycle

Initialize ovstream for the process lifetime. Register callbacks before starting
each server, submit frames from one loop, and stop/join that loop before
`server.stop()` and `server.close()`. Call `ovstream.shutdown()` after the final
server closes. See [streaming lifecycle](../streaming-lifecycle/README.md) for
connection state, reconnects, and restoring app state.

Use the [app-message adapter](../streaming-messages/server-handler-map.md) for
request/reply traffic. Classify disconnect failures separately from persistent
encoding errors: transient disconnects must not kill the renderer, while
persistent failures should move the session into a visible error state.

## Frame Loop And Continuity

Detailed frame-source, fixed-resolution, and stage-load continuity guidance lives in `frame-loop-and-continuity.md`.

## Readiness Health Gate

Expose readiness separately from process liveness. Orchestrators and load balancers must not send clients to the service until the renderer has produced a valid BGRA8 CUDA mapping or completed the required conversion into the app-owned BGRA stream buffer.

```python
from http.server import HTTPServer, BaseHTTPRequestHandler
from threading import Event, Thread

class HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path != "/healthz":
            self.send_response(404); self.end_headers(); return
        if self.server.ready_event.is_set():
            self.send_response(200); self.end_headers(); self.wfile.write(b"ok")
        else:
            self.send_response(503); self.end_headers(); self.wfile.write(b"not ready")

ready = Event()
httpd = HTTPServer(("0.0.0.0", 8081), HealthHandler)
httpd.ready_event = ready
Thread(target=httpd.serve_forever, daemon=True).start()

# In the render loop, after the first valid BGRA mapping or completion
# of the required conversion into the app-owned stream buffer:
ready.set()
```

Readiness follows a valid BGRA mapping or completed required conversion, independently of client
connection or submission. Clear readiness when the owner stops or fails. Log
the first valid frame to distinguish startup from transport negotiation.

## RGBA To BGRA

ovstream expects BGRA8. An RGBA8 render output needs the conversion below;
an already-BGRA output must not be swizzled again. The paired
[OVRTX streaming example](https://github.com/NVIDIA-Omniverse/ovstream/tree/main/examples/python/ovrtx_stream)
authors a `BgraColor` RenderVar with an SPG swizzle and submits its CUDA mapping
directly. For fixed-resolution streaming, follow its `CudaSync` producer-wait
and mapping lifetime: keep the mapping valid until `stream_video()` has staged the frame, then unmap. Use an
app-owned buffer when retaining a frame beyond that mapping's lifetime.
Resizable mode has different encoder lifetimes; follow
[frame ownership](../dependencies/ovstream.md#frame-ownership).

```python
import warp as wp

@wp.kernel
def rgba_to_bgra(src: wp.array3d(dtype=wp.uint8), dst: wp.array3d(dtype=wp.uint8)):
    i, j = wp.tid()  # Exactly one thread owns each destination pixel.
    dst[i, j, 0] = src[i, j, 2]
    dst[i, j, 1] = src[i, j, 1]
    dst[i, j, 2] = src[i, j, 0]
    dst[i, j, 3] = src[i, j, 3]
```

```python
# Index render outputs by their full authored RenderVar prim path. Keep
# `source_name` only for UI-facing AOV classification.
# Allocate stream_bgra_buffer once: uint8 [h, w, 4] on cuda:0.
# This example uses fixed-resolution streaming (webrtc_resizable=False).
with frame.render_vars["/Session/Render/Vars/LdrColor"].map(device=ovrtx.Device.CUDA) as var:
    var.wait()  # Renderer producer must finish before Warp reads the mapping.
    wp_array = wp.from_dlpack(var)
    wp.launch(rgba_to_bgra, dim=(h, w), inputs=[wp_array, stream_bgra_buffer], device="cuda:0")
    wp.synchronize_device("cuda:0")  # Finish source reads and destination writes.
server.stream_video(ovstream.VideoFrame.from_cuda_array(stream_bgra_buffer))
```

No CPU round trip is needed.

## Production AOV Conversion Before `stream_video()`

Outputs that are not already BGRA8 need conversion into an owned CUDA `uint8 [H,W,4]` BGRA buffer before creating `ovstream.VideoFrame`. Preserve the accepted frame if a requested output cannot be copied; an explicit fallback must update the active-state UI too.

Current ovrtx render vars can be single-tensor or multi-tensor outputs. For a
single-tensor render var, consume the mapped object directly with DLPack. For a
multi-tensor render var, choose the named tensor that represents the image
payload and read params separately. Image tensors are channel-last: `H x W`,
`H x W x 1`, `H x W x 3`, or `H x W x 4`. Do not assume `C x H x W`, and do not
use old `.tensor` access in new generated code.

| AOV | Expected input | Conversion rule |
|---|---|---|
| `LdrColor` | `uint8 [H,W,4]` RGBA | Copy to the stream buffer and swap R/B to BGRA. |
| `HdrColor` | `uint16 [H,W,4]` or float RGB/RGBA | Apply exposure/Reinhard tonemapping, gamma/sRGB display correction, clamp, output BGRA8. |
| `DistanceToImagePlaneSD` / `DistanceToCameraSD` | verify against the installed output | Normalize to a useful grayscale visualization, usually inverse-distance or min/max normalized, output BGRA8. |
| `NormalSD` | float RGB/RGBA or `uint32 [H,W,4]` packed float bits | Remap normal components from `[-1, 1]` to `[0, 255]`, output BGRA8. |
| `InstanceSegmentationSD` | `uint32 [H,W]` or `[H,W,1]` | Debug visualization only: hash each non-zero ID to a deterministic color; ID `0` is black/background. Native picking does not require this AOV. |
| `SemanticSegmentationSD` | `uint32 [H,W]` or `[H,W,1]` | Use the same deterministic colorization as instance segmentation. |
| `DiffuseAlbedoSD` | float RGB/RGBA or `uint8 [H,W,4]` | Gamma-correct linear albedo when needed, clamp, output BGRA8. |

Use [AOV switching](../aov-switching/README.md) for capability discovery,
active output requests, scratch-buffer conversion and correlated completion.
Only hand a successfully converted buffer to `stream_video()`; do not commit a
new mode or overwrite the last good image before that conversion succeeds.

Pick queries are independent of the displayed AOV. Do not update a
segmentation-derived pick buffer in new generated apps.

## Native Picking And Selection Outlines

Use ovrtx native pick queries and native selection outline state:

1. Convert the ovstream input coordinate to render-product pixel space.
2. Enqueue `renderer.enqueue_pick_query(...)` with normalized top-left bounds
   for a one-pixel click or a larger marquee rectangle.
3. Step the same RenderProduct. The pick result appears as the synthetic render var `ovrtx_pick_hit`.
4. Map the pick-hit output, validate its params such as `magic` and `version`, read the named `primPath` tensor, and resolve each non-zero path id with `renderer.resolve_prim_path_id(...)`.
5. Deduplicate resolved paths and publish `stageSelectionChanged`.
6. Clear previous outlines with group `0`, then assign group `1` or another
   styled group through `Renderer.set_selection_outline_group_strings()`.

Configure selection outlines at renderer creation with `RendererConfig(selection_outline_enabled=True, selection_outline_width=...)`. Configure per-group colors at runtime with `Renderer.set_selection_group_styles(...)`. Changing global width or fill mode requires recreating the renderer; changing per-group colors does not.

Do not create legacy segmentation picker modules, CPU ray fallback picker modules, segmentation ID maps, isolation ID discovery, or Warp outline compositors for new generated apps.

## Runtime Feature Commands

Advertise optional capabilities before showing their UI controls. Camera,
selection, scene, and render settings commands share the runtime queue; physics
commands target the configured simulator owner. Use the
[message protocol](../streaming-messages/README.md) for transport-neutral command
shapes and request/result correlation.

## Operation Status And Errors

Treat stage loads, render steps, and pick queries as operations whose status must be checked:

- In Python, blocking helpers such as `open_usd()` may raise; async variants return operations that must be `.wait()`ed and fetched before the result is trusted.
- In C, load errors are reported through `ovrtx_op_wait_result_t::error_op_ids`; the enqueue return value only says whether the work was accepted.
- A failed or timed-out stage load must send `openStageResult {result: "error"}` plus `viewerError`, keep or restore the previous valid frame when possible, and avoid marking readiness.
- A failed render step should keep the last good frame, emit a bounded error event, and stop retrying only after repeated non-recoverable failures.
- A failed or empty pick query should clear hover state or return no path without mutating the current selection unless the user explicitly requested clear-on-miss.

## Input Callback

Input is separate from JSON messages in the normal WebRTC path: NVST forwards mouse/keyboard/gamepad input as binary `InputEvent` structs that arrive through `server.on_input`. A browser-only JSON `mouseInput` fallback is permitted only after `viewer-input-routing` verifies that the data channel works while native callbacks are absent, and it replaces native input. For SHM Python clients, send the native input struct path through `ovstream.Client(ovstream.ClientType.SHM, stream_name="...").send_input_event(event)`; C clients use `ovstream_client_send_input_event(client, event)`. Read `viewer-input-routing` before implementing either browser input mode.

```python
def on_input(event):
    if not viewport_input_active:
        camera.cancel_interaction()
        return
    if event.type == ovstream.InputEventType.MOUSE:
        mouse = event.mouse
        if mouse.type == ovstream.MouseEventType.MOVE:
            handle_mouse_move(mouse.x, mouse.y, mouse.modifiers)
        elif mouse.type == ovstream.MouseEventType.BUTTON:
            button = camera_button_from_ovstream(mouse.data, ovstream)
            if button is not None:
                handle_mouse_button(button, mouse.button_state, mouse.x, mouse.y)
        elif mouse.type == ovstream.MouseEventType.WHEEL:
            handle_scroll(mouse.scroll_y or mouse.data, mouse.x, mouse.y)
    elif event.type == ovstream.InputEventType.KEYBOARD:
        handle_key(event.keyboard.key_code, event.keyboard.key_state, event.keyboard.modifiers)
    elif event.type == ovstream.InputEventType.GAMEPAD:
        handle_gamepad(event.gamepad.control, event.gamepad.position, event.gamepad.gamepad_id)
```

Use this native input path for orbit/pan/zoom and viewport picking. In browser
apps with DOM controls, maintain `viewport_input_active` from a lightweight app
message such as `setViewportInputActive {active}`. Disable it when the pointer is
over sidebars, trees, inspectors, menus, or top bars so UI clicks do not move the
camera or trigger picks.

`ovstream.MouseButton` values are not DOM button ids: `LEFT=1`, `MIDDLE=2`,
and `RIGHT=3`. Use `viewer-input-routing` to convert through the enum or an
explicit mapping before passing buttons to shared camera helpers that use
`0=left`, `1=middle`, `2=right`.

When the WebRTC stream surface is the only source of native input, initialize
`viewport_input_active = True` and let DOM panels turn it off with
`setViewportInputActive {active:false}`. If the server starts inactive, the
first mouse-down can race ahead of the React activation message, so a left-click
release is seen without a matching press and click picking never queues.

## ServerConfig Reference

| Field | Meaning |
|---|---|
| `width`, `height`, `target_fps` | stream dimensions and frame rate |
| `stream_port=0` | default media port: 47998 WebRTC, 47999 native, 8554 RTSP; SHM uses no media port |
| `webrtc_signal_port=0` | default signaling port: 49100 |
| `webrtc_public_ip=None` | use ICE; set `127.0.0.1` for local loopback |
| `video_input` | `CUDA`, `TENSOR`, `CUSTOM`, `H264`, `H265`, or `AV1` |
| `rtsp_pipeline`, `rtsp_mount_point` | RTSP custom pipeline/path |
| `shm_stream_name`, `shm_slot_count` | SHM stream identifier and ring depth for local shared-memory transport |

`ServerType.WEBRTC` is for browser streaming, `RTSP` for VLC/ffplay, `NATIVE` for native clients, and `SHM` for same-machine shared-memory readers. WebRTC supports one connected client at a time; guard unsolicited app messages with `server.is_client_connected`. Multiple network servers in one process need explicit unique ports; ovstream does not auto-increment conflicting defaults.

## Ports

Signaling is TCP/WebSocket on 49100 by default. WebRTC media is UDP and
negotiated by SDP; check the selected ovstream release notes for the current
default media port. Do not conflate these in frontend config.

## Generated Module Checklist - streaming server

- [ ] `main()` sets `OVRTX_SKIP_USD_CHECK=1` before ovrtx imports.
- [ ] `OVWebViewerServer.start()` or equivalent constructs `Renderer(RendererConfig(sync_mode=True, selection_outline_enabled=True))` when selection feedback is needed.
- [ ] `PxrWorkerClient.start()` is optional and used only for USD queries not covered by ovrtx native stage APIs.
- [ ] Initial stage load calls `ovrtx.register_schema_paths()`, then uses OVStage
      `population.open_usd_from_string()` with an inline root USDA, commits its
      ordinal, and attaches the renderer when viewer render config must be injected.
- [ ] Renderer warmup steps before `ovstream.Server.start()`.
- [ ] `on_app_message` routes unwrapped WebRTC JSON app messages to the message handler.
- [ ] `MessageHandler.on_input` is registered for raw ovstream input events.
- [ ] `MessageHandler.on_unicode` is registered when composed text input matters.
- [ ] `/healthz` returns `503` before first successful frame and `200` afterward.
- [ ] One runtime owner serializes `renderer.step()` and stage mutation.
- [ ] Render loop enqueues/decodes native pick queries and updates native selection outline groups.
- [ ] Submission uses valid BGRA8 with producer synchronization and buffer lifetime preserved; non-BGRA AOVs use the conversion buffer.
- [ ] The render loop streams a fallback or last-good frame while connected,
      even when idle or loading.
- [ ] Disconnects preserve renderer state; persistent submission errors reach session status.
- [ ] Same-process server restarts stop and close the server without calling
      `ovstream.shutdown()` until process exit.

See also: `streaming-client`, `streaming-messages`, `streaming-lifecycle`, `ovrtx-rendering`, `stage-loading`, `viewer-input-routing`, `camera-controls`, `object-selection`.
