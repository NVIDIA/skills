# Viewport Overlays

## Scope And Read Order

Use this reference for non-XR, server-side ovui gizmos and floating panels
rendered in headless Vulkan mode and alpha-composited over OVRTX output before
presentation or ovstream delivery. OVRTX remains the USD scene renderer.

For browser-only 2D handles over video, use
[transform manipulators](../transform-manipulator/README.md) and
[input routing](../viewer-input-routing/README.md); they need no headless ovui
compositor. Scene surfaces and section caps remain OVRTX-rendered geometry.

Read [dependencies](../dependencies/README.md) for acquisition and the selected
ovui release's upstream headless export/input guidance. Then read
[CUDA compositing](cuda-compositing.md) for allocation, frame ownership, alpha,
channel conversion, and a blend kernel. For interactive translate/rotate/scale
tools, also read [transform manipulators](../transform-manipulator/README.md)
and [transform safety](../prim-transform-safety/README.md). They own drag math
and live scene edits; this reference owns overlay construction and composition.

For a local inline widget layered over an image, use
[local viewer](../local-viewer/README.md). For OpenGL-rendered gizmos use
[GL viewport overlays](../gl-viewport-overlay/README.md). For app-owned Vulkan
raster overlays, alpha composition or scene-depth-tested gizmos, use
[Vulkan compositing](../vulkan-compositing/README.md). That path can compose
directly in the native Vulkan view; ovui's `headless_frame` API only exports
its own UI output and does not import an arbitrary `VkImage`.

## Components To Build In Your App

No overlay Python modules are supplied by this package. Implement these
responsibilities in your app; the names are illustrative, not imports:

| Component | Implementation contract |
|---|---|
| Overlay owner | Initialize standalone/headless export, own one transparent window, tick, acquire/copy/release exported frames, and shut down. Use the sequence below and [CUDA compositing](cuda-compositing.md). |
| Camera gizmo | Draw rings/axes in an ovui `SceneView` with a persistent `DragGesture`; convert drag deltas to the viewer camera's orbit operation. Use [camera controls](../camera-controls/README.md). |
| Transform gizmo | Draw handles, hit-test them before camera/picking, capture the drag, and enqueue transform intent. Follow [transform manipulators](../transform-manipulator/README.md). |
| Input adapter | Normalize viewport coordinates, decide pointer ownership, and inject only owned events through the selected ovui release's input API. Follow the capture rules below. |
| Floating prim panel | Build labels in a `ui.Placer`; store selected path, copied properties, current runtime world anchor, and visible bounds. Use [prim info display](../prim-info-display/README.md). |
| Message/runtime adapter | Queue show/hide/tool/selection changes on the overlay owner, update the view/projection before ticking, and composite before submitting the frame. Use [input routing](../viewer-input-routing/README.md) and [message handlers](../streaming-messages/server-handler-map.md). |

An optional package initializer only re-exports classes/functions you implement.
There is no required `OrbitGizmoOverlay`, `OvuiCudaComposite`, or
`OvuiInputBridge` class to locate or install.

## Construct The Headless Overlay

Set these before importing ovui or constructing the renderer:

```python
import os
os.environ["OMNIUI_HEADLESS"] = "1"
os.environ["OMNIUI_BACKEND"] = "vulkan"
os.environ["OVRTX_SKIP_USD_CHECK"] = "1"
```

On the same serialized owner that will tick/export/tear down the overlay:

1. Import `omni.ui as ui`, `standalone` from `omni.ui`, and
   `headless_frame` from `omni.ui.standalone`. Verify the selected release
   supplies transparent headless export; resolve missing capability through
   dependency guidance rather than substituting an opaque full-frame UI.
2. Call `standalone.init("Viewer overlay", width, height, max_fps=None)`,
   `standalone.set_software_cursor(False)`, then `headless_frame.init()`.
   Check initialization success, `headless_frame.format() == "rgba8"`, and
   `headless_frame.extent()` against the render size. Clean up partial startup
   on failure. Keep any release-specific/private entry points in one adapter.
3. Build one `ui.Window` at `(0, 0)` covering the output, with
   `fill_app_window=True` and no title, resize, move, or scrollbar flags.
   Set its frame's `Window` style to transparent background/border and zero
   border width. Verify the exported clear pixels have alpha zero: window
   styling alone does not prove the render target is transparent.
4. Inside `window.frame`, create one `ui.ZStack` containing the camera gizmo,
   transform overlay, and any floating panel. Use `ui.Placer` offsets for
   screen anchors. Retain the window, widget/model, manipulator, and gesture
   objects for their lifetime. Create drag gestures once and reuse them when
   rebuilding geometry.
5. Add visibility and tool state to the owner. Disabled/hidden overlays neither
   draw nor claim input. Hiding a dragged tool or replacing a scene cancels its
   capture and invokes the transform-safety drag-end/cancel path.

Keep OVRTX rendering and overlay work serialized with scene load/reset. Input
and transport callbacks enqueue work; they do not tick ovui or mutate the live
stage concurrently.

For a simple orbit widget, create three ring polylines in a small `SceneView`:
sample `t` from 0 through `2*pi` and use `(cos(t), sin(t), 0)`,
`(cos(t), 0, sin(t))`, and `(0, cos(t), sin(t))`. Join adjacent samples with
`omni.ui_scene.scene.Line` under a retained manipulator, attach the same
persistent `DragGesture` to the lines, and brighten the active ring during a
drag. Give the widget its own fixed view/projection framing the unit rings;
the viewer camera remains the orbit target. On gesture begin store the pointer,
on change convert the payload's documented coordinate units to incremental
render-pixel deltas, and enqueue the same orbit action used by viewport input.
On end clear capture. If the payload is NDC, scale X by half the widget width
and Y by half its height, converting its Y sign once to the camera input
convention. Do not pass NDC deltas into a camera method expecting pixels.

## Input Capture

Use the displayed image rectangle and the transport's mapping from
[input routing](../viewer-input-routing/README.md). Keep hit rectangles and
injected positions in the same top-left render-pixel coordinates. NVST already
maps browser letterboxing; do not apply that correction twice.

- On press, a visible gizmo handle or panel may claim the button. Give transform
  handles first refusal, then other overlay controls, then camera/picking.
- Retain owned buttons until release, even when the pointer leaves the widget.
  Deliver that release before clearing capture. Never allow a drag that began
  in the viewport to be stolen just because it moves over the panel.
- Inject hover moves only inside a visible hit region unless a drag is owned.
  Wheel input belongs to the hovered/captured control; other events continue
  to normal viewport handling. Track all pressed buttons to distinguish
  viewport drags from overlay drags.
- Normalize left/middle/right explicitly to the selected ovui input API's
  button codes. Verify the supported injection API in upstream guidance;
  `_ui._inject_mouse_move`, `_inject_mouse_button`, `_inject_mouse_scroll`,
  and `standalone._tick_one_frame` are private spellings in some releases,
  not an app-independent interface promised by this skill.
- On disconnect, focus loss, disable, or scene replacement, cancel gestures and
  deliver/reset pressed-button state so reconnect cannot leave a stuck drag.

## World Anchors And Camera Alignment

Implement the projection helper from
[prim info display](../prim-info-display/README.md), which supplies
`world_to_screen()` and the bottom-left to top-left conversion. Pass explicit
matrix conventions; do not guess row/column layout from translation magnitude.
Use the exact view, projection, aperture, and render aspect of the OVRTX camera.
Hide behind-camera/nonfinite anchors; clamp panels to the image bounds.

Refresh anchors from current OVStage runtime world transforms every displayed
frame, before ticking ovui. Authored USD metadata from a query worker can be
stale during animation, manipulation, or simulation. Clear selection/panel
state on scene replacement and discard delayed results from older generations.

## Frame And Shutdown Sequence

```text
apply queued input/scene/camera/tool changes on the runtime owner
  -> renderer.step() and copy/convert output into an app-owned display buffer
  -> update overlay camera, world anchors, labels, and visibility
  -> tick ovui; wait_ready -> copy_to_linear -> finish copy -> signal_consumed
  -> blend owned overlay pixels over the owned display buffer
  -> finish GPU work; submit/present; retain buffers through consumer completion
```

The full sequence, failure behavior, and buffers are defined in
[CUDA compositing](cuda-compositing.md). Overlay failure must not publish
uninitialized or stale scratch pixels. Surface a visible feature error when a
requested overlay cannot initialize; keep the requested functionality tracked.

Stop input and submissions, cancel captures, finish outstanding GPU/consumer
work, then call `headless_frame.shutdown()` followed by `standalone.shutdown()`
on the owner. Attempt both even if one fails. Keep shutdown idempotent and
handle initialization that stopped halfway through. Prefer fixed render size;
for a deliberate resize, drain consumers and recreate/revalidate the export
and CUDA buffers together before accepting new frames.

## Validation

Capture a transparent clear frame and red/green/blue markers over a known
background. Check alpha edges, orientation, colors, and projected alignment
while orbiting/panning/zooming. Exercise drag outside bounds, viewport drag
across a panel, hide during drag, scene replacement, disconnect/reconnect, and
shutdown. Record both the pre-delivery composite and the actual presented or
streamed image using [validation guidance](../validation.md).
