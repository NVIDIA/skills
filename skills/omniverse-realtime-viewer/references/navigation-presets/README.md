<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Navigation Presets

Use this reference only when a prompt asks for switchable navigation, Blender-
or Revit-style controls, CAD/DCC muscle memory, grab-the-world navigation, or
fly mode with speed control. A viewer with no such request keeps the defaults
from `conventions.md`: left-drag orbit, middle-drag pan, right-drag dolly, and
wheel zoom.

Read `viewer-input-routing/README.md` first, then `camera-controls/README.md`.
Those references remain authoritative for transport normalization, viewport
ownership, camera state, sanitization, matrices, and commit paths. This
reference adds preset dispatch, switching, and the operations in
`preset-camera-ops.md`; it does not replace the default camera controller.

## Architecture

Keep one camera state and one operation implementation. A preset is a dispatch
table over the normalized helper buttons (`0=left`, `1=middle`, `2=right`) and
modifier bits (`SHIFT=1`, `CTRL=2`, `ALT=4`):

```python
PRESETS = {
    "default": {"drags": {(0, 0): "orbit", (1, 0): "pan", (2, 0): "dolly"},
                "wheel": "zoom"},
    "blender": {"drags": {(1, 0): "orbit", (1, 1): "pan", (1, 2): "dolly"},
                "wheel": "zoom", "keys": "blender_views"},
    "revit": {"drags": {(1, 0): "pan", (1, 1): "orbit"},
               "wheel": "zoom_at_cursor", "dblclick": {1: "frame_all"}},
    "grab": {"drags": {(0, 0): "pan_locked", (2, 0): "orbit"},
             "wheel": "zoom"},
    "fly": {"drags": {(2, 0): "look", (1, 0): "pan_locked"},
            "wheel": "fly_speed", "step": "fly_move"},
}
```

Resolve the exact `(button, modifiers)` entry on every move so a modifier
pressed during a drag takes effect on the next move. Do not use a separate
camera controller per preset. The default preset is the initial state unless
the product explicitly chooses another one.

Preset characteristics:

- Revit: bare middle drag pans, Shift+middle orbits, wheel zooms toward the
  cursor, and middle-button double-click frames all. Left click remains
  selection.
- Blender: middle orbits, Shift+middle pans, Ctrl+middle dollies, and the
  standard front/right/top and 15-degree step views are available from numpad
  keys. If emulate-numpad is offered, top-row digits may alias those views.
- Grab: bare left drag uses cursor-locked pan so the grabbed world point stays
  under the pointer; orbit is assigned to right drag.
- Fly: WASDQE is integrated once per frame in world units/second. W/S and A/D
  use gravity-aligned horizontal directions, E/Q move along world-up, and the
  wheel changes speed rather than zooming. Clear held keys on focus loss.

Axis snapping, double-click focus, and other product-specific gestures are
optional enhancements, not requirements for claiming Blender or Revit parity.

## Switching and UI

Switching changes only the active dispatch table. Cancel the current gesture
first, retain the same camera state, validate the name, and publish the
effective state. This prevents a half-held drag or a camera jump:

```python
def set_preset(name: str) -> None:
    if name not in PRESETS:
        raise ValueError(name)
    camera.cancel_interaction()
    state.preset = name
    broadcast("navPresetChanged", {"preset": name,
                                   "available": list(PRESETS)})
```

For streamed viewers, accept `{ "event_type": "setNavPreset", "payload":
{ "preset": "revit" } }` on the server and broadcast `navPresetChanged` on
change and client connect. Serialize switching with input dispatch. Local
viewers update their in-process state and toggle group.

In browser viewers, prefer an on-screen preset control. Do not reserve bare
digits for preset switching when Blender view keys are available; browser
shortcuts and keyboard focus also make global hotkeys fragile. If desktop
hotkeys are needed, use an explicit chord and document it.

## Existing behavior and validation

Keep click-vs-drag selection, transform-gizmo ownership, letterbox mapping,
fit-to-stage, world-up configuration, and renderer/OVStage write ownership
from the existing references. A preset must not bypass those contracts.

Add math-only tests for:

- cursor-locked pan and zoom-at-cursor fixed points;
- preset switching cancelling the old gesture without changing camera pose;
- Revit bare-middle pan versus Shift+middle orbit;
- Blender view keys and live modifier changes;
- fly displacement of `speed * dt`, with no world-up component for planar motion;
- elevation and minimum-distance clamps under every operation.

See `preset-camera-ops.md` for the renderer-independent math.
