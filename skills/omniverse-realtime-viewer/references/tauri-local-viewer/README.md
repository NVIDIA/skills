# Tauri + Rust OVRTX Viewer

Use this path for a desktop app with a React WebView and a Rust-owned OVRTX
renderer. Share UI components through `viewer-backend-interface`; keep native
handles and frame transport inside the Tauri backend. For Python desktop apps
use `local-viewer`; for a remote renderer use `streaming-viewer-recipe`.

## Ownership and frame flow

```text
React controls -> Tauri commands -> render-owner queue
                                      |
                           OVStage edits + publication
                                      |
                              OVRTX step + CPU map
                                      |
React canvas <- Tauri Channel <- owned RGBA frame
```

One Rust thread owns the renderer, attached OVStage, publication ordinals,
camera, and selection. Tauri commands enqueue work; they do not call renderer
APIs from the IPC thread. Coalesce continuous pointer motion so old input does
not delay the next frame. Preserve button transitions and discrete actions.
Use request IDs or reply channels for operations that need a result.

This presentation path copies pixels to the CPU and across IPC. Start with it
for a simple desktop implementation; measure readback, transport, and paint
cost before choosing a different presentation architecture. Send raw binary
frames rather than JPEG or base64 on this local path.

## Runtime and stage setup

Acquire matching SDK binaries and headers through `dependencies`. Generate
Rust C bindings from those headers and use the package's runtime loader setup.
Use `windows-native-setup` for Windows packaging. Keep strings, config arrays,
and tensor storage alive for the native operation that borrows them.

For attached mode, follow `stage-loading` and `stage-management`:

1. Populate an inline root that sublayers the user's USD and authors the app
   camera, RenderProduct, and RenderVars.
2. Complete population, advance and wait for its write floor, then attach the
   renderer to that stage.
3. Write camera and feature changes through OVStage; complete and publish them
   before `ovrtx_update_from_stage()` and `ovrtx_step_with_stage()` at that ordinal.

Use `ovstage-data-plane` for matrix semantics and column layout. A standalone
OVRTX backend instead uses its own reset/load/write/step APIs; keep that mode
separate from attached-stage mutation. Its camera input descriptor is:

```text
ovrtx_write_attribute:
  attribute:     "omni:xform"
  semantic:      XFORM_MAT4X4
  binding dtype: float64, lanes=16
  input tensor:  (1,4,4) float64, lanes=1, ndim=3, strides=NULL, CPU
  prim_mode:     CREATE_NEW
  access:        SYNC
```

Choose prim paths once and share them across authoring, camera writes, render
requests, and output lookup. For example:

```text
/Session/Cameras/Main       camera
/Session/Render/Viewport    display RenderProduct
/Session/Render/Vars/...    authored RenderVars
```

On scene replacement, stop stepping, finish outstanding work, release old
results, and replace the attached stage through the render owner. Invalidate
pending picks, hierarchy caches, and selection by stage generation. Show loading
until the new stage produces a displayable frame; preserve the last good frame
with a loading indicator while work is pending.

## Frame delivery

Map `LdrColor` to CPU, read its tensor dimensions and strides, and copy RGBA rows
into an owned buffer. Use RAII for map and result cleanup on both display and
pick paths. No mapped tensor view may outlive its native result.

Register a Tauri `Channel<InvokeResponseBody>` for binary frame delivery. One
possible app payload is:

```text
[u32 width LE][u32 height LE][u64 sequence LE][tightly packed RGBA8 pixels]
```

The WebView decodes the `ArrayBuffer` and displays it with `ImageData` and
`canvas.putImageData`. Bound queued frames and prefer the latest completed frame
when presentation falls behind. A new subscriber can receive the cached latest
frame immediately. Use ordinary events for lifecycle, errors, and selection.

Keep render resolution independent of window size. Letterbox the image and map
input through its content rectangle. If the product needs dynamic resolution,
update the RenderProduct through the stage owner, preserve camera state, and
reallocate presentation storage from the next frame's dimensions.

## Interaction and UI

Implement the shared `ViewerBackend` operations for scene loading, camera input,
picking, selection, hierarchy, properties, and exposed render settings. Advertise
only implemented capabilities; route real hierarchy and property queries through
`stage-hierarchy` instead of parsing USD text or returning placeholder values.
Cache query results for UI reads and refresh them on stage changes.

Follow `viewer-input-routing` for gestures and pointer capture. Pointer release
or cancellation ends the gesture even outside the viewport. Write finite camera
changes before the next render step; derive animation delta from elapsed time
and cap catch-up work so a stalled frame does not create an input backlog.

For a click:

1. Map the click through the displayed image rectangle.
2. Queue its request ID, stage generation, and displayed RenderProduct.
3. On the render thread, enqueue the normalized native pick query and step that
   product at the committed ordinal.
4. Decode its synthetic `ovrtx_pick_hit` output and return the canonical path.
5. Ignore stale-generation results; update the outliner, properties, and native
   selection outline together.

See `native-picking-selection` for result decoding and outline groups. Keep
optional material effects, animation, and physics behind the same runtime owner;
use `prim-pick-effects`, `selection-animation`, or `physics-simulation` when the
feature is requested.

## Errors and shutdown

Distinguish a pending frame from a failed render. Keep the UI responsive and
show the last good frame with an error state when rendering fails. Allow scene
reload or an explicit retry to recover; do not report readiness before a frame
arrives.

On close, stop accepting commands, finish native operations, release mappings
and results, detach the stage, and destroy the renderer on its owner thread.
Validate the packaged app through first frame, camera interaction, picking,
scene replacement, and close. Include a screenshot from the real WebView path;
a Rust build alone does not exercise SDK loading or presentation.

## Separate-process SHM variant

For a Tauri shell consuming a separate renderer, use the
[unified client and ownership contract](../electron-shm-viewer/electron-client.md#unified-native-client-and-ownership).
Generate complete `repr(C)` structs and tagged unions from the matching headers;
validate layout on the target platform. Stop and join frame waits before client
destruction, and keep callback data alive until destruction completes.
