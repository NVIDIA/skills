<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Spatial Anchoring And Interaction

## Convert Stage Space To OpenXR

Use the full-scale row-vector contract from
[spatial surfaces](spatial-surfaces-and-interaction.md). Let B convert tracking
metres/up-axis into stage units/up-axis, A be the active managed anchor, and M
the measured tracking-space-to-LOCAL transform at the projection display time:

```text
local = world * inverse(B * A) * M
```

Transform origins as points and tangent axes as vectors; tangent lengths define
physical extents. Preserve positive uniform anchor scale through inversion.
Reject unsupported shear rather than silently stripping scale. A fixed-LOCAL
panel bypasses this conversion; a stage-anchored surface does not. Tracking
space may be STAGE or LOCAL_FLOOR, so the inverse anchor alone is not LOCAL.

Test translation and rotation with non-identity tracking anchors. Record the
resulting local pose. Do not infer offset signs: project the quad and inspect a
live capture. Keep row-vector USD math separate from column-vector WebXR math.

Use `LOCAL` for stage-attached UI. Use `VIEW` only when deliberately head
locked.

Publish an unconfigured world panel's authoritative `LOCAL` pose before eye or
frame telemetry arrives. Do not derive its initial visibility or position from
the first head sample, a desktop camera, or a tracking anchor that has already
been applied. A face-the-viewer policy updates the stored `LOCAL` orientation;
it does not change the surface to `VIEW` space or recompute its translation
from each eye.

## Hit The Oriented Quad

Transform the controller ray into quad-local coordinates. Intersect local
`z=0`; reject parallel, behind-ray, and out-of-bounds hits. Calculate:

```text
u = hitX / width + 0.5
v = 0.5 - hitY / height
```

Convert accepted UVs to raster pixels, then test Yoga-generated bounds.

## Own Pointer State

```text
miss -> no UI ownership; continue to the scene layer
hover -> target only
press -> capture hovered target
drag -> retain capture
release -> finish the captured node's activation policy
focus/session loss -> cancel and rearm
```

Use separate analog press and release thresholds to avoid chatter. When the UI
owns a hit or capture, consume that controller's scene-select event.

Use native OpenXR aim/select actions by default. Use an opaque data channel only
for extra client metadata or when the client contract explicitly requires it.

Sample VIEW, grip, and aim against the same explicit `LOCAL` reference space at
the exact display time submitted to `xrEndFrame`. If an adapter receives that
time after the high-level render helper returns, latch the coherent input
snapshot for the following projection frame and label both times. Do not
extrapolate an assumed next time from the display period.

Grip drives controller-body geometry; aim drives rays, tooltips, and pointer
hit testing. A visual model calibration may move the model asset but must never
rewrite either measured pose. One model locator can determine translation, not
rotation. Author a rotational correction only from a documented model basis or
at least three non-collinear measured correspondences with repeatable live
evidence. Reject acquisition windows whose samples have not settled rather
than calibrating to the first tracked pose.

Route selection through one authoritative service:

```text
UI row activation (press for immediate-selection rows)
  -> semantic select-prim action
  -> runtime selection
  -> OVRTX native outline
  -> refreshed UI selection state
```

Define an outline policy for non-renderable Xform or Scope selections. Keep the
canonical logical selection while resolving explicit descendant renderable
prims for the OVRTX outline; do not silently replace the selected path.

Window chrome retains ray capture while moving and publishes only a newer
surface pose revision. Controller callouts and transform gizmos are pose-only
analytic surfaces in the same per-eye draw as the panel. Give gizmo picking a
larger analytic hit volume without enlarging its visible geometry. Depth-test
gizmo parts against each other; an always-on-top scene policy may bypass scene
depth, but must still preserve self-occlusion.

## Persistent grab bars and selection

A permanent highlighted grab bar is a selectable manipulation target, distinct
from direct-drag chrome and ordinary buttons. Keep hover, selected target, and
active drag separate. A select edge chooses the bar; the normal transform
gesture then captures that panel with its current hand-to-panel offset. The
pointer need not remain over the bar once movement owns it. Moving a panel
publishes pose only and must not repaint unchanged text.
Latch both the manipulating hand/controller and surface at capture start.
Another hand closing must not steal capture or reset its transform baseline.

For a single-target interaction policy, selecting a scene prim clears the panel
handle and its drag owner in the same transaction; selecting a different prim
replaces the old prim in one activation. Route a selected-panel fist/transform
to the panel before the scene tool consumes it. Never manipulate both owners
from one gesture. Make the selected color reflect the authoritative target,
including selection through stage rows or external controls.

On focus loss, tracking expiry, panel hide, relocation by another owner, stage
reload, or context replacement, cancel active capture and consume the eventual
release. Test select-bar -> move-off-bar -> fist-move -> select-scene-prim,
plus hand/controller changes and reconnect. Visual highlighting alone does
not establish that manipulation reached its target.
