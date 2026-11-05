# Local Omniverse Realtime Viewer Shell

## Triggers

Use this skill for ovui window shell, standalone viewport shell, simple ovui app, `ImageBridge`, image display, resize handling, mouse capture surfaces, or requests that should avoid the full ovui editor shell.

For a complete local desktop viewer, read `ovui-local-viewer-recipe` first and use
this as the focused shell/display skill. Use this for a small single-window
viewer: header, RTX viewport, optional sidebar, and inline controls. Do not use
`ovui_widgets.app.application.Application`; that starts the full USD Viewer editor with docking, menus,
property panels, transform gizmos, and status bars.

For mouse button normalization, click-vs-drag handling, and dispatch to camera
or selection controllers, read `viewer-input-routing`.

For header/sidebar controls, toolbar actions, render settings widgets,
sliders, and destructive confirmations, read `viewer-control-patterns` and
translate its client-agnostic control guidance into native `ovui` widgets.

For ovui widget APIs or native UI behavior not covered here, read
`references/dependencies` for acquisition guidance and supplemental dependency
documentation.

For ovrtx renderer setup, frame extraction, or release-specific behavior not
covered here, read `references/dependencies` for acquisition guidance and
supplemental dependency documentation.

## Runtime Setup

Install and activate the selected `ovui` package through `references/dependencies`
before imports. The lightweight local path should keep the UI/render process
focused on `ovui`, OVRTX, and live OVStage/session state. Run `pxr` queries in a
subprocess unless the exact wheel set and import order have been verified for
the target machine.

```python
import asyncio
import os
os.environ.setdefault("OVRTX_SKIP_USD_CHECK", "1")

import ovrtx
ovrtx.register_schema_paths()
import omni.ui as ui
```

For OVStage-backed local apps, the parent desktop process owns stage lifetime,
ordinals, the write floor, renderer attachment, and publication. Query workers
and physics workers should exchange JSON or other DTOs, not live stage handles,
path tokens, DLPack objects, or mapped buffers. If OVPhysX is requested, prefer
`PhysX.attach_ovstage(stage, read_ordinal=...)` after verifying the exact
installed OVStage/OVPhysX ABI. If that bridge is unavailable or fails, use a
bounded worker only when its installed-package API is verified; otherwise report
the compatibility block.

Run with a real display from the activated environment:

```bash
DISPLAY=:99 OVRTX_SKIP_USD_CHECK=1 python3 -m local_app
```

If `renderer.step()` hangs after a crash, use `nvidia-smi` and kill only stale Python Omniverse Realtime Viewer processes that still hold CUDA/RTX state.

## ovui Window Shell

Use this as the canonical standalone `ovui` bootstrap for generated lightweight
local viewers in this skill package: initialize once, create one fill-window
`ui.Window`, paint through one provider, then run the update coroutine alongside
`ui.run_async()` in an application-owned asyncio loop. Do not generate
`hasattr(ui, "get_app")` or alternate bootstrap branches unless a targeted compatibility validation proves the selected wheel
changed the standalone contract.

Use `fill_app_window=True`; without it the GLFW window can resize while the UI frame remains stuck at the initial dimensions.

```python
import asyncio
import numpy as np
import omni.ui as ui


def synthetic_rgba(width: int, height: int) -> np.ndarray:
    x = np.linspace(0, 255, width, dtype=np.uint8)
    y = np.linspace(0, 255, height, dtype=np.uint8)
    rgba = np.zeros((height, width, 4), dtype=np.uint8)
    rgba[:, :, 0] = x[None, :]
    rgba[:, :, 1] = y[:, None]
    rgba[:, :, 2] = 64
    rgba[:, :, 3] = 255
    return np.ascontiguousarray(rgba)


WIDTH = 1280
HEIGHT = 720

ui.init("Omniverse Realtime Viewer", width=WIDTH, height=HEIGHT, max_fps=60)
provider = ui.ByteImageProvider()
window = ui.Window(
    "Omniverse Realtime Viewer",
    width=WIDTH,
    height=HEIGHT,
    fill_app_window=True,
    flags=ui.WINDOW_FLAGS_NO_TITLE_BAR,
)

with window.frame:
    with ui.ZStack(style_type_name_override="Local.Root"):
        ui.Rectangle(style_type_name_override="Local.Root")
        with ui.VStack(spacing=0):
            build_header(height=50)
            with ui.HStack(spacing=0, height=ui.Fraction(1)):
                with ui.Frame(width=ui.Fraction(1)):
                    ui.ImageWithProvider(
                        provider,
                        fill_policy=ui.IwpFillPolicy.IWP_PRESERVE_ASPECT_FIT,
                    )
                with ui.Frame(width=ui.Pixel(280)):
                    sidebar.build()

provider.set_data_array(synthetic_rgba(640, 360), [640, 360])


async def render_loop():
    while True:
        # Advance viewer state here; ui.run_async owns presentation.
        await asyncio.sleep(1.0 / 120.0)


async def run_until_closed(update, presentation):
    tasks = [asyncio.create_task(update), asyncio.create_task(presentation)]
    try:
        done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            task.result()  # propagate update or presentation failures
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


try:
    asyncio.run(run_until_closed(render_loop(), ui.run_async()))
finally:
    # Stop and join the render owner here once a runtime is connected.
    window.destroy()
    ui.shutdown()
```

Keep the surface focused: black header, large viewport, white/sidebar utility area, green `#76b900` accents. Avoid editor-only affordances unless requested.

`ui.init(...)` returns `None`. `ui.run_async()` presents frames until the native
window closes. When either task finishes, cancel and await the other before
stopping the render owner, destroying the window, and calling `ui.shutdown()`.
The update coroutine should yield after each state update. This gives its
`finally` blocks time to run before the event loop and native backend disappear.
Validate this bootstrap before adding renderer or USD code.

## Header And Load Controls

Keep scene loading boring and reliable. For lightweight local viewers, default
to one path field plus one `LOAD` button that calls the same serialized scene
load path used by command-line startup. Do not add a separate `OPEN` button or
native file dialog unless it is implemented and validated under the same
display/session environment as the viewer. A broken dialog next to a working
path loader creates needless ambiguity.

```python
path_model = ui.SimpleStringModel(str(initial_stage))
ui.StringField(model=path_model, height=30)
ui.Button("LOAD", clicked_fn=lambda: runtime.load(path_model.get_value_as_string()))
```

If a native file dialog is explicitly requested, keep it secondary and test it
with the target display stack (`DISPLAY`, Xvfb/VNC, Wayland/X11, container
permissions). On failure, report the dialog failure in the status bar and leave
the path-field loader usable.

## Image Display

Use the app's local image bridge helper when a plain image widget is enough.
For projects using the `ovui_widgets` helper, import
`ovui_widgets.viewport.image_bridge.ImageBridge`; otherwise use
`ui.ByteImageProvider` directly. The renderer may stay at fixed resolution while
`ImageWithProvider` stretches with preserve-aspect letterboxing.

```python
bridge = ImageBridge(render_width, render_height)
image = ui.ImageWithProvider(bridge.provider, fill_policy=ui.IwpFillPolicy.IWP_PRESERVE_ASPECT_FIT)
bridge.update(frame_rgba_uint8)
```

For a direct provider path:

```python
provider = ui.ByteImageProvider()
image = ui.ImageWithProvider(provider, fill_policy=ui.IwpFillPolicy.IWP_PRESERVE_ASPECT_FIT)

# arr is C-contiguous uint8 RGBA, shape H x W x 4
provider.set_data_array(arr, [arr.shape[1], arr.shape[0]])
```

Use `ui.color("#76b900")` or `ui.color(r, g, b, a)` for style colors. The helper
packs ovui's integer `0xAABBGGRR` format; image-provider pixels remain RGBA bytes.

Read `LdrColor` inside the map context and copy before returning:

```python
# products is the mapping-like result of the render owner's renderer.step().
# RenderVar output maps use the full authored prim path.
rv = products[RENDER_PRODUCT_PATH].frames[0].render_vars["/Session/Render/Vars/LdrColor"]
with rv.map(device=ovrtx.Device.CPU) as mapping:
    frame = np.from_dlpack(mapping).copy()  # owned pixels for UI presentation
```

## Display Smoke Test And Blank-Viewport Triage

Before debugging camera, lighting, or USD composition, prove the ovui image path
can paint a synthetic frame:

```python
def synthetic_rgba(width: int, height: int) -> np.ndarray:
    x = np.linspace(0, 255, width, dtype=np.uint8)
    y = np.linspace(0, 255, height, dtype=np.uint8)
    rgba = np.zeros((height, width, 4), dtype=np.uint8)
    rgba[:, :, 0] = x[None, :]       # red horizontal ramp
    rgba[:, :, 1] = y[:, None]       # green vertical ramp
    rgba[:, :, 2] = 64               # visible blue floor
    rgba[:, :, 3] = 255
    return np.ascontiguousarray(rgba)

provider.set_data_array(synthetic_rgba(640, 360), [640, 360])
```

Capture a desktop screenshot of the window and verify it is nonblank and
non-solid. If the synthetic image does not paint, debug ovui presentation first:
provider construction, widget visibility, frame sizing, style opacity, main-loop
stepping, and whether the active provider API updates in this ovui build.

Use this decision tree for black, blank, or solid-color viewports:

1. Capture `LdrColor` directly from ovrtx outside ovui and save it as an image.
2. If direct `LdrColor` is blank, debug scene loading, camera fit, render
   product path, render vars, lighting, and material/plugin resolution.
3. If direct `LdrColor` is nonblank but the window is blank, debug ovui
   presentation before touching camera or renderer state.
4. Test the chosen `ImageBridge` or `ByteImageProvider` path with the synthetic
   RGBA frame above.
5. If dynamic byte-provider updates do not paint in the active ovui build, use a
   known-good ovui-native presentation path for validation, such as a
   `RasterImageProvider` screenshot/frame fallback, then document the selected
   presentation path in the generated app.

For unstable startup ordering, load the scene and render one direct ovrtx frame
before entering the long-running ovui loop. Continuous rendering may run in a
dedicated render worker that owns `renderer.step()` and copies the latest RGBA
frame into an application buffer; the ovui/main loop should only present that
latest copied frame. Do not call `renderer.step()` while another thread is
loading, resetting, or mutating the stage.

## Resize And Letterbox Math

For picking, overlays, and camera input, compute the visible rendered image rect inside the viewport widget.

```python
def widget_size(hit_rect, image, fallback):
    w = int(float(getattr(hit_rect, "computed_width", 0.0) or 0.0))
    h = int(float(getattr(hit_rect, "computed_height", 0.0) or 0.0))
    if (w <= 0 or h <= 0) and image is not None:
        w = int(float(getattr(image, "computed_width", 0.0) or 0.0))
        h = int(float(getattr(image, "computed_height", 0.0) or 0.0))
    return max(1, w or fallback[0]), max(1, h or fallback[1])

def image_content_rect(widget_w, widget_h, image_w, image_h):
    image_aspect = image_w / max(1.0, float(image_h))
    widget_aspect = widget_w / max(1.0, float(widget_h))
    if widget_aspect > image_aspect:
        draw_h = float(widget_h); draw_w = draw_h * image_aspect
        return (widget_w - draw_w) * 0.5, 0.0, draw_w, draw_h
    draw_w = float(widget_w)
    return 0.0, (widget_h - draw_w / image_aspect) * 0.5, draw_w, draw_w / image_aspect
```

The ovui wheel callback is `(delta_x, delta_y, modifiers)`. Use `delta_y` for
vertical zoom, preserve the modifier bitmask, and take cursor position from
pointer events.

## Mouse Capture Surface

Attach callbacks to a transparent top-level `ui.Rectangle` over the image. Wrap callbacks; unguarded ovui callback exceptions can tear down the app loop.

```python
hit_rect = ui.Rectangle(style={"background_color": 0x00000000})
hit_rect.opaque_for_mouse_events = True
hit_rect.set_mouse_pressed_fn(on_mouse_pressed)
hit_rect.set_mouse_released_fn(on_mouse_released)
hit_rect.set_mouse_moved_fn(on_mouse_moved)
hit_rect.set_mouse_wheel_fn(on_mouse_wheel)

def local_render_coords(screen_x: float, screen_y: float, clamp: bool):
    x = float(screen_x) - float(hit_rect.screen_position_x)
    y = float(screen_y) - float(hit_rect.screen_position_y)
    off_x, off_y, draw_w, draw_h = image_content_rect(...)
    if not clamp and (x < off_x or y < off_y or x > off_x + draw_w or y > off_y + draw_h):
        return None
    u = (x - off_x) / max(1.0, draw_w)
    v = (y - off_y) / max(1.0, draw_h)
    if clamp:
        u, v = max(0.0, min(1.0, u)), max(0.0, min(1.0, v))
    return u * render_width, v * render_height
```

```python
def on_mouse_moved(*args):
    try:
        if dragging and len(args) >= 2:
            camera.on_mouse_move(*local_render_coords(args[0], args[1], clamp=True))
    except Exception:
        logger.exception("Mouse move failed")
        dragging = False
```

When a `SceneView` overlay and top-level mouse callbacks coexist, explicitly
arbitrate pointer ownership. A transform drag should suppress orbit and click
selection for that mouse-down; normal left-drag outside the transform handle can
still orbit. Do not let a gizmo drag also enqueue a pick on release.

## Local Transform Gizmo Wiring

Use one transform adapter for the gizmo, Inspector and operation commands.
The UI owns pointer gestures; the runtime owner reads and writes scene state.
For attached apps, write local `omni:xform` values through OVStage and publish
before the renderer consumes them. Follow
[prim transform safety](../prim-transform-safety/README.md) for current ancestry,
reset stacks, typed matrix writes and publication. Its standalone-binding path
applies only when the renderer owns the scene.

1. At drag start, capture the selected paths, active scene generation and current
   live local/world transforms through that adapter. Keep selected prims
   separate from descendant meshes used for outlines.
2. Give the gizmo first refusal on the pointer gesture. Use
   [transform manipulator](../transform-manipulator/README.md) for hit testing,
   projection and drag math; suppress camera movement and click-picking while
   the drag owns the pointer.
3. Compose each target from the drag-start snapshot in the chosen coordinate
   space and queue that intent to the runtime owner.
4. The owner checks generation and edit ownership, then converts a world target
   `W` to local `L = W @ inverse(P)` using the current effective parent-to-world
   matrix `P`. Reset stacks and simultaneous parent/child edits follow the
   canonical transform model. Apply the local matrices and publish with the
   other writes for that tick. Authored USD values are baselines, not replacements
   for current runtime transforms after live edits.
5. Refresh the gizmo and Inspector from the same live state. On release, commit
   the final drag value; on cancel, restore the captured baseline through the
   same owner path. Retire the drag when its scene is replaced.

Verify numeric local/world results under a transformed parent, cancel/restore,
and actual window input that moves the selected prim with its handle. A moving
handle alone does not prove scene publication. Keep renderer output verification
separate from transform readback.

## Context Menu (Right-Click)

ovui supports popup context menus via `ui.Menu`. Show it on RMB release only when the mouse did not drag (use the drag threshold from `viewer-input-routing` or `camera-controls`). If the user drags RMB, that's a camera look/dolly — suppress the menu.

```python
def on_mouse_released(x, y, button, modifier):
    if button == 1:  # RMB
        if not _exceeded_drag_threshold:
            _show_context_menu(x, y)
        return
    # ... other release handling

def _show_context_menu(screen_x: float, screen_y: float):
    """Show a popup context menu at the cursor position."""
    if hasattr(ui, "Menu"):
        menu = ui.Menu("Viewport")
        with menu:
            ui.MenuItem("Open File...", triggered_fn=_on_open_file)
            ui.MenuItem("Reload Stage", triggered_fn=_on_reload)
            ui.Separator()
            ui.MenuItem("Frame All", triggered_fn=_on_frame_all)
            ui.MenuItem("Reset Camera", triggered_fn=_on_reset_camera)
            ui.Separator()
            ui.MenuItem("Quit", triggered_fn=lambda: ui.shutdown())
        menu.show_at(int(screen_x), int(screen_y))
```

Key points:
- Create the `ui.Menu` fresh each time (ovui menus are lightweight)
- Use `menu.show_at(x, y)` with screen-space pixel coordinates
- Guard with the drag threshold check — if the mouse moved ≥5 px between press and release, it was a camera gesture, not a menu intent
- Wrap in try/except; older ovui builds may not support `show_at`

## Local Validation

Run `python3 -m compileall local_app server`, launch with `DISPLAY=:99`, prove
synthetic ovui presentation paints in the window, save a direct `LdrColor`
artifact, and capture a desktop screenshot showing that same rendered frame in
the ovui window. Then resize the window, switch Sample 1/Sample 2, verify stage01
renders without extra session lights, check tree/viewport selection, confirm
native selection outlines clear on load, verify prim animation returns on
clear/reset, ensure prim info follows orbit/pan/zoom/resize, and confirm
orbit/pan/right-drag zoom/wheel move the camera. Confirm selected-prim gizmo
drag changes the prim's live `omni:xform`; do not accept a result where the
gizmo appears but the prim is stationary. Left-click selection
must not fire after orbit drags.

See also: `ovrtx-rendering`, `stage-loading`, `viewer-input-routing`, `viewer-control-patterns`, `camera-controls`, `object-selection`, `selection-feedback`, `transform-manipulator`, `prim-transform-safety`, `prim-info-display`, `stage-management`, `render-settings`, `dependencies`.
