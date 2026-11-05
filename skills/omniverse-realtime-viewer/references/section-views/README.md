<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Section Views

Use for clipping planes, cross-sections, capped cutaways, or interactive slice
placement in local and streamed OVRTX viewers. Read `render-settings` for
capability-driven controls and `viewer-input-routing` before adding a gizmo.

## Choose the geometry contract

Treat clipping, caps, and interaction as separate capabilities:

- Native RTX clipping changes visibility at one or more planes. Inspect the
  selected OVRTX release's [render-settings skill](https://github.com/NVIDIA-Omniverse/ovrtx/blob/main/skills/render-settings/SKILL.md),
  schemas, and [RTX section-plane documentation](https://docs.omniverse.nvidia.com/materials-and-rendering/latest/rtx-renderer_common.html#section-plane)
  for current attributes, plane layout, sign convention, and lighting policy.
  A clipping setting alone does not establish cap generation.
- Caps are optional viewer-owned surfaces rendered by OVRTX. Choose exact CAD
  sections or polygon-mesh intersections according to the requested fidelity.
  For exact modeling, start with `dependencies/README.md` → USD BRep and its
  upstream operation guidance. For polygon meshes, use the selected public mesh
  library's plane-intersection and contour APIs, such as [Trimesh sections](https://trimesh.org/section.html).
  Use its documented polygon triangulation path; a triangulator such as
  [earcut](https://github.com/skogler/mapbox_earcut_python) fills contours but does
  not compute the mesh/plane intersection. Upstream owns those algorithms.
- A gizmo edits the app's plane parameters. It may be a native/server-composited
  overlay or a browser 2D UI overlay over video. It need not select a USD prim.

Declare the supported source scope: static polygon meshes, evaluated deformed
meshes, instances, subdivision surfaces, or exact solids. Use the same evaluated
geometry and runtime transforms as the visible scene. A control-cage cap is not
an evaluated subdivision cap; a source-USD snapshot does not reflect subsequent
runtime motion. Report unsupported sources rather than inventing a solid section.

## Plane state and cap publication

Keep a canonical plane frame: origin, normalized normal, and tangent basis,
plus the retained half-space and cap/lighting preferences. Convert that model
once to the renderer's verified plane equation. Axis presets are optional;
normal-axis dragging does not imply free rotation support. Keep all plane and
mesh calculations in one declared coordinate space, preserving stage units.

For polygon caps, snapshot the source revision and current runtime poses, then
intersect each solid with the accepted plane. Stitch closed contours within a
scale-aware tolerance, preserve disjoint regions and nested holes, and triangulate
in plane coordinates. Define behavior for open, branched, self-intersecting, or
degenerate contours; skip with a diagnostic or reject the candidate. Do not fill
holes with a convex hull or claim arbitrary non-manifold input is supported.

Orient cap surfaces toward the removed half-space and verify both cut directions.
Handle on-plane geometry and clipping tolerances so caps survive native clipping
without a visible gap. Bind an explicit cap material; scene surfaces and caps
remain server/native OVRTX geometry even when handles are browser UI.

Use `ovstage-live-geometry` for slot allocation, numeric property mappings,
complete topology/bounds updates, and publication through the existing runtime
owner. Publish clipping state and its accepted caps as one logical revision before
rendering. Tag worker results with scene generation, geometry/pose revision,
and plane revision; reject superseded candidates. Coalesce interactive rebuilds
and retain the last accepted plane/caps while a candidate is computing. An empty
intersection, disabled plane/caps, or scene replacement must retire stale caps.

## Interactive placement

Use the shared projection/input model in `transform-manipulator`, with a plane
parameter command in place of its selected-prim transform command. Project the
plane origin and normal handle from the authoritative render camera into the
fixed render image; scale into the visible image rectangle after letterboxing.
Refresh on camera/plane changes, reconnect, and layout changes. Hide or offer an
alternate interaction when the normal projects to near-zero length or the handle
is behind the camera; a fabricated screen axis is not a world-space constraint.

Snapshot the plane and camera interaction frame at drag start. Derive each update
from that baseline, using a supported ray/constraint or projected-axis method,
rather than accumulating deltas from prior frames. Give the handle first refusal
on input through release/cancel, including outside the viewport. Send semantic
intent to the owner and display authoritative accepted state. Cancel/restore and
scene-generation rules follow `viewer-input-routing`; ordinary camera and picking
input must resume after the gesture. Keep source asset transforms unchanged when
only the viewing plane is being moved.

## Focused validation

Show uncapped and capped cross-sections of a closed solid, a nested hole, and
an open/unsupported case with its declared outcome. Move and reverse the plane,
move a supported source object, disable caps, and replace the scene with pending
work: stale surfaces must disappear and late results must be rejected. A local
frame must come from the same OVRTX path as display; streamed validation must
inspect decoded video, not only successful setting acknowledgements.

For a gizmo, record the plane change, unchanged camera and selection during the
drag, cancel/restore, resized/letterboxed alignment, and camera/pick input after
release. Store focused proof with the generated app using `validation.md`.
