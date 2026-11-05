<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Driving CAE Visualization Through OVStage

Use this reference to turn a field, iso-value, slice, colormap, timeline or
emitter control into a live visualization update. Choose the publication path
from the prims and attributes that change, rather than from the control's name.

Read `references/dependencies` for the selected public runtime contracts,
[data-and-operators.md](data-and-operators.md) for scientific computation, and
`references/ovstage-data-plane/README.md` for typed writes and tensor lifetime.
The viewer owns the live OVStage; UI callbacks and scientific workers return
intent or copied results to that owner.

## Choose the publication path

| Change | Publication |
|---|---|
| Camera or seed-marker pose | Write the existing prim's local `omni:xform`. |
| Colormap or display range | Recolor the current field and write `primvars:displayColor`; keep geometry and camera. |
| Timeline sample, field, iso-value or slice position on an existing Mesh/Points prim | Compute the new arrays, then update points, topology and primvars that changed. |
| Streamline seeds on an existing BasisCurves prim | Update points, curve counts, colors and other length-dependent arrays. |
| Representation that changes prim type, adds/removes prims, or changes composition | Replace the affected geometry subtree or reference. |
| New dataset requiring a new scene | Use the scene-replacement flow in `references/stage-management/README.md`. |

A different vertex count or connectivity does not by itself require a new
Mesh prim. Mesh face counts and indices are array attributes too. Conversely,
a Mesh-to-Points change cannot be implemented by merely writing differently
shaped arrays to the Mesh. A representation change that still produces the
same prim type can use array updates.

## Compute away from the renderer owner

Controls enqueue typed intent. Merge compatible rapid edits into the latest
requested configuration, then let the scientific worker compute that state.
Keep dataset replacement and Reset as explicit intent boundaries. The owner
continues rendering and camera interaction while the worker is busy; it accepts
only results for the active scene generation and current request.

Return copied geometry, scalar values, field association and display metadata.
Keep source datasets and pending array buffers alive until their consumers
finish. Source ingestion and field/time selection follow
[cae-data-ingestion.md](cae-data-ingestion.md); UI draft handling follows
[ovui-controls.md](ovui-controls.md) or
[streaming-cae-viewer.md](streaming-cae-viewer.md).

## Publish an internally consistent array set

Author the geometry prim, material binding and primvar interpolation once.
For each accepted update, prepare the arrays required by its existing type:

| Prim | Geometry arrays | Related values |
|---|---|---|
| Mesh | `points`, changed `faceVertexCounts` and `faceVertexIndices` | Colors/scalars consistent with vertex, uniform or face-varying interpolation |
| BasisCurves | `points`, `curveVertexCounts` | Colors/scalars and widths consistent with curve interpolation |
| Points | `points` | Colors/scalars and widths consistent with point interpolation |

Use OVStage `write_attribute` with `is_array=True`, the attribute's declared
dtype/lanes and semantic, and a query for the actual target prim. Follow the
selected SDK's DLPack shape contract and keep each source buffer alive until
its write completes. Local transforms remain fixed-size matrix attributes;
use `references/prim-transform-safety/README.md` for their coordinate space.

Batch all writes for one result at one new ordinal above the current write
floor. Wait for them, advance the floor once, and render that committed ordinal.
No frame may observe new points with old connectivity or an old-length color
array. Validate finite coordinates, index bounds, counts and field lengths
before publication. Reuse cached topology only when connectivity and ordering
are unchanged, not merely when the number of vertices matches.

For an empty operator result, publish a consistent empty geometry/field set
instead of retaining the previous surface. The next nonempty result must update
all dependent arrays. A constant width needs one value; a vertex width must
track the new point count. Choose constant interpolation only when a constant
quantity is intended, not as a substitute for missing field data.

Retain camera and selection identity when the same prim survives. A colormap
edit uses the existing field values and does not require refitting or reloading
the scene. Keep scientific values and their data range separate from the chosen
color domain. [usd-authoring-and-materials.md](usd-authoring-and-materials.md)
describes interpolation and material setup.

## Replace structure when structure changes

For added/removed prims, a new prim type or changed composition, keep the viewer
camera, RenderProduct and other persistent state while replacing the affected
geometry reference or subtree. Use the public population operations in
`references/ovstage-population/README.md`, wait for completion, and publish the
new ordinal. Rebuild queries and feature state for replaced prims; update
selection according to which paths still exist.

A whole-dataset replacement follows `references/stage-management/README.md`,
including preparation, generation invalidation, attachment and first-frame
completion. Preserve the live camera on field/representation changes unless
the user requests Fit; apply the selected camera policy for a new dataset.
Structural replacement and attribute publication both remain serialized with
render stepping on the runtime owner.

## Keep interaction responsive

Bound pending work and coalesce compatible edits before dispatch. A continuous
drag can use fewer streamline seeds or integration steps, followed by a full
quality result when it settles. Both results use the same publication path;
label any approximate preview that changes what the visualization means.

Measure from accepted intent through computation, publication and the displayed
frame. Operator time alone does not measure interaction latency. Check actual
rendered movement, field-color changes, empty/nonempty transitions and Reset
alongside copied numeric values; use matched camera views so a label update or
camera move is not mistaken for geometry publication.
