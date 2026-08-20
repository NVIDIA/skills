<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# CAE/CFD Visualization: ovui Window Shell And Interactive Controls

## Triggers

Use this reference for the ovui window shell plus the interactive header
controls that drive a CAE/CFD visualization pipeline: representation
(points/surface/isosurface/slice/streamlines) and field selectors, iso-value and
slice-position sliders, colormap choice, orbit-camera navigation over a rendered
viewport, and sample-preset one-click scenes. It documents the thread-safe path
from an ovui callback to a single serialized render loop that owns every stage
rebuild and `renderer.step`.

This is the interactive-controls surface for the CAE/CFD viewer. For the generic
window bootstrap read `local-viewer` and `ovui-local-viewer-recipe`; for button
normalization and click-vs-drag dispatch read `viewer-input-routing`; for control
widget selection read `viewer-control-patterns`; for renderer ownership,
ordinals, and the write floor read the `ovstage-*` references and
`conventions.md`.

## Ownership Model (Non-Negotiable)

One serialized loop owns every OVStage write and every `renderer.step`. ovui
callbacks run on the UI thread and MUST NOT touch OVStage or the renderer
directly. A callback does exactly one of two things:

- mutates local, numpy-only input/camera state (through the `InputController`), or
- enqueues a command onto a thread-safe queue that the loop drains.

Because the loop drains all queued commands — including stage rebuilds — before
it ever calls `renderer.step`, a step can never run while the stage is being
rebuilt. This is the same single-owner rule as `conventions.md`, applied to an
interactive CAE control surface.

## ovui Shell Considerations

Use these behavior checks with the currently installed `ovui` API. Do not generate defensive `hasattr`/alternate-bootstrap branches
around them unless a targeted compatibility check proves the wheel changed the
contract; if the wheel changed, update this reference instead.

- `ui.init("...", width=W, height=H, max_fps=60)` returns `None`. There is no app
  handle to hold, and in particular there is **no `app.step_and_present()`**.
- `ui.run(coroutine)` runs its own standalone loop, and that loop performs
  presentation. The render-loop coroutine only advances viewer state and pushes
  frames into the provider; `ui.run` presents them.
- The render-loop coroutine calls `runtime.tick(dt)` then
  `await asyncio.sleep(1.0 / 120.0)` — **not `sleep(0)`**. A busy spin (`sleep(0)`)
  pegs a CPU core and starves the OS message pump, which reads as a hung window.
  Yield at ~120 Hz instead.
- Create one fill-window: `ui.Window("...", fill_app_window=True,
  flags=ui.WINDOW_FLAGS_NO_TITLE_BAR)`. Wrap the `flags=` construction in
  try/except and fall back to a no-flags `ui.Window(..., fill_app_window=True)`;
  some wheel signatures reject the `flags` kwarg.
- Viewport = `ui.ByteImageProvider` drawn with
  `ui.ImageWithProvider(provider, fill_policy=ui.IwpFillPolicy.IWP_PRESERVE_ASPECT_FIT)`.
  Push frames with `provider.set_data_array(rgba_uint8, [w, h])` where the array
  is C-contiguous `HxWx4` uint8 RGBA (no B/R swap) and the size list is
  `[width, height]`.
- Layer a transparent `ui.Rectangle` with `opaque_for_mouse_events = True`
  ABOVE the image, inside the image's own `ui.ZStack`. The rect captures camera
  drags over the viewport while the header controls above the ZStack stay
  clickable.

```python
import asyncio, time
import omni.ui as ui

ui.init("CAE ovui Viewer", width=W, height=H, max_fps=60)   # returns None
provider = runtime.viewport.provider

try:
    window = ui.Window("CAE ovui Viewer", width=W, height=H,
                       fill_app_window=True, flags=ui.WINDOW_FLAGS_NO_TITLE_BAR)
except Exception:
    window = ui.Window("CAE ovui Viewer", width=W, height=H, fill_app_window=True)

with window.frame:
    with ui.VStack(spacing=0):
        build_header()                       # ComboBoxes + FloatSliders (below)
        with ui.ZStack(height=ui.Fraction(1)):
            ui.ImageWithProvider(provider,
                fill_policy=ui.IwpFillPolicy.IWP_PRESERVE_ASPECT_FIT)
            hit = ui.Rectangle(style={"background_color": 0x00000000})
            hit.opaque_for_mouse_events = True   # captures camera drags
            wire_hit_rect(hit, runtime)          # mouse callbacks -> InputController

async def render_loop():
    prev = time.time()
    while True:
        try:
            now = time.time(); dt = now - prev; prev = now
            runtime.tick(dt)                 # advances state, pushes frames
        except Exception as exc:             # never let one frame kill the loop
            print(f"[app] render_loop error: {exc}", flush=True)
        await asyncio.sleep(1.0 / 120.0)     # NOT sleep(0)

ui.run(render_loop())                        # coroutine, not a plain callback
```

ovui integer style colors are `0xAARRGGBB`. Swapping the byte order turns an
intended dark background into light red/brown and makes a working viewport look
like a renderer failure.

## Header Controls (ComboBoxes + FloatSliders)

The header carries: a representation ComboBox (points/surface/isosurface/slice/
streamlines), a field ComboBox (populated from the loaded dataset), an iso-value
FloatSlider (0–1 fraction of the field range), a slice-position FloatSlider (0–1
along the slice axis), and a colormap ComboBox. Each control's callback ONLY
enqueues a command — it never touches OVStage or the renderer. Wrap every callback
in try/except; an unguarded ovui callback exception can tear down the app loop.

- ComboBox: `combo = ui.ComboBox(default_idx, *items)`, then
  `combo.model.add_item_changed_fn(callback)`. Read the selection inside the
  callback with `model.get_item_value_model().as_int` and index into the item
  list; guard the index and fall back to a sensible default when out of range.
- FloatSlider: `slider = ui.FloatSlider(min=0.0, max=1.0)`, seed it with
  `slider.model.set_value(initial)`, then
  `slider.model.add_value_changed_fn(callback)`. Read with
  `model.get_value_as_float()`.
- Populate the field ComboBox from real dataset field names. Pre-load the dataset
  once before building the header (the reference calls `runtime._ensure_dataset()`
  and reads `runtime.fields`) so the dropdown shows actual fields rather than a
  guess; fall back to the current field name if the dataset has not loaded.

```python
def _on_representation_changed(model, *_):
    try:
        idx = model.get_item_value_model().as_int
        mode = REPRESENTATIONS[idx] if 0 <= idx < len(REPRESENTATIONS) else "surface"
        runtime.enqueue_representation(mode)          # enqueue only
    except Exception as exc:
        print(f"[app] representation callback error: {exc}")

def _on_iso_changed(model, *_):
    try:
        runtime.enqueue_iso(float(model.get_value_as_float()))   # 0..1 fraction
    except Exception as exc:
        print(f"[app] iso callback error: {exc}")
```

The enqueue methods (`enqueue_load`, `enqueue_colormap`, `enqueue_representation`,
`enqueue_field`, `enqueue_iso`, `enqueue_slice`) each `put` a `(kind, payload)`
tuple on the runtime's `queue.Queue`. They are the only UI-thread-safe way to
change pipeline state.

## The Single Serialized Loop

The UI thread enqueues; the loop thread drains, coalesces, rebuilds at most once,
and renders one frame. Two triggers cause a frame:

1. A drained command (representation/field/iso/slice/colormap/load) that requires
   a stage rebuild.
2. The camera-dirty flag, set by input callbacks; the loop commits the pose and
   steps.

`runtime.tick(dt)` (runtime.py) is one serialized iteration:

```
tick(dt):
    _drain_commands()          # applies coalesced commands; may rebuild stage once
    if first build:  _build_scene_and_attach()
    if camera dirty: clear flag; _render_current()   # commit pose @ordinal, step
```

`_drain_commands()` (runtime.py) drains the whole queue with `get_nowait()` into a
`latest[kind] = payload` dict, so only the **last value per kind** survives
(last-wins coalescing). This is what keeps a dragged iso/slice slider from queuing
dozens of stage rebuilds per frame — the loop applies only the latest value, sets
a single `rebuild` flag, and calls `_build_scene_and_attach()` at most once per
tick. A `refit` flag distinguishes a new dataset (reframe the camera) from a
colormap-only rebuild (preserve the user's current view).

`_render_current()` enforces the write-floor ordering from `conventions.md` and
the `ovstage-*` references: get the row-major camera xform, `commit_camera_pose`
writes `omni:xform` at a new ordinal and advances the write floor, then
`step_extract(ordinal=...)` renders at that ordinal — never above the floor — and
the resulting RGBA is pushed to the viewport provider.

## Camera Navigation

Camera math is pure NumPy (`OrbitCamera`, camera_controller.py) so it works before
the OV stack is installed. The `InputController` (input_controller.py) translates
ovui mouse events into camera moves and flags the runtime camera-dirty; it never
writes OV state.

- Gestures: `orbit` (spherical azimuth/elevation), `pan` (move target in the
  camera right/up plane), `dolly` (exponential distance change), `zoom` (wheel),
  and `fit` (frame an AABB: target = center, distance from max extent and FOV with
  a 1.5× margin, plus derived near/far clip).
- Button remap (matches `conventions.md`): `{0: orbit, 2: pan, 1: dolly}`; wheel =
  zoom (positive steps zoom in). A press becomes a drag only after motion exceeds
  a 1.0 px threshold, and the first post-threshold step is measured from the
  down-point so it is not jumpy.
- `get_camera_xform()` returns a row-major float64 4×4: row 0 = right, row 1 = up,
  row 2 = `-forward`, row 3 = eye. State is sanitized before matrices are built —
  distance/elevation clamped, non-finite lanes scrubbed — so the loop never
  commits a non-finite pose.

## Sample-Preset Pattern

To reproduce a named sample scene in one click, keep a small registry of presets,
each a dict of `{dataset, representation, field, colormap}` (optionally iso/slice
fractions). Applying a preset is a single enqueued command that the loop drains
and applies as one coalesced rebuild — the same path as the individual controls,
so no preset code touches OVStage directly. A preset button's callback resolves
the named entry and enqueues it; the loop reloads the dataset (refit) and rebuilds
the stage once.

```python
# presets.py — a registry; the app is adding this module.
PRESETS = {
    "StaticMixer temperature": dict(dataset="static_mixer.npz",
        representation="surface", field="Temp", colormap="viridis"),
    "Disk isosurface": dict(dataset="disk_out_ref.npz",
        representation="isosurface", field="Pres", colormap="turbo"),
}

def apply_preset(runtime, name):
    p = PRESETS[name]
    runtime.enqueue_load(p["dataset"], field=p["field"], colormap=p["colormap"])
    runtime.enqueue_representation(p["representation"])   # coalesced into one rebuild
```

Because commands coalesce last-wins per kind within a tick, enqueuing several
preset fields together still produces a single stage rebuild.

## Gotchas

- Buffered stdout when redirected: Python buffers stdout when it is not a TTY, so
  piping logs to a file can hide `[status]`/`[render]` lines until exit. Run
  `python -u -m <app>` (or `flush=True` on prints) to see logs live.
- Friendly field names: resolve a display name like `"Temperature"` to the real
  dataset field (e.g. `"Temp"`) against the actual field list on every rebuild, so
  coloring never silently falls back to the wrong field. Implement this as an application-owned case-insensitive alias map, then require a visible
  selection or error when no unambiguous field exists.
- First-frame RTX shader compile: a fresh process's first `renderer.step` compiles
  RTX shaders and can take minutes. The GPU may look idle and the window may look
  frozen — this is **not** a hang. Allow a long first-frame timeout (for example, 300 s) and poll rather than assuming failure.
- Never let one frame kill the loop: wrap `tick`, command application, and every
  mouse/control callback in try/except; log and continue so a single bad frame or
  callback does not tear down the window.

## Suggested Application Modules

- `app.py` — ovui shell: `ui.init`/`ui.Window`, header ComboBoxes + FloatSliders,
  ZStack image + transparent hit rect, render-loop coroutine, `ui.run`.
- `runtime.py` — serialized `ViewerRuntime`: enqueue methods, coalesced last-wins
  queue drain, tick, and write-floor commit + step.
- `camera.py` — orbit/pan/dolly/zoom/fit and a sanitized row-major camera xform.
- `input.py` — button remap, drag threshold, and camera-dirty state.
- `viewport.py` — `ByteImageProvider` wrapper and letterbox mapping.
- `config.py`, `__main__.py` — fixed render size, interactive/test modes, and preflight env order.

See also: `local-viewer`, `ovui-local-viewer-recipe`, `viewer-input-routing`,
`viewer-control-patterns`, `camera-controls`, `conventions.md`,
`ovstage-runtime`, `ovstage-ovrtx-integration`, `ovrtx-rendering`.
