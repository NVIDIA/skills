<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Optional USD BRep Geometry Source

Use when stylized CAD/technical viewers start from exact BRep rather than an
existing USD Mesh. First read `dependencies/README.md` → USD BRep, then the
owning repository's current `AGENTS.md`, skills router, operation docs, and
package README. Upstream owns modeling, schema, tessellation, and diagnostics;
this reference defines the handoff to the existing style compositor.

## Public Artifact Boundary

Use the packaged `usd_brep` API and the selected artifact's stubs. Source-checkout
`_omni_solid` and debugger `_smlib_dev` examples are separate surfaces; do not
make developer Draw/topology APIs a public viewer requirement. Resolve package
and interpreter compatibility before choosing an in-process import. Prefer a
worker when coexistence with OVRTX/OVStage is unverified; exchange owned arrays
and files, never native BRep handles or OpenUSD objects.

## Surface And Feature Representations

Tessellate with the public API and validate complete output before publishing.
Current upstream supports binding-owned contiguous NumPy mesh buffers and
source-face corner normals. `to_numpy_arrays(face_varying_normals=True)` yields
float32 points/normals, an int32 count-prefixed polygon stream, and per-face
normals. Decode counts/indices with bounds checks, preserve corner-normal order,
and declare USD `faceVarying`. Verify the method in the resolved artifact; use
the documented list output adapter if that artifact lacks NumPy output.

Normalize to the triangle-only, subdivision-none supported profile in
`line-compositor-delivery.md`. Do not assume every tessellated polygon is already
a triangle. Preserve logical part and face provenance through triangulation;
exclude unsupported topology with the existing diagnostic and fidelity rules.
The exact BRep remains the editable source; OVRTX renders the derived Mesh.

The public `tessellate_boundaries` API samples trimmed BRep edges and returns
points, per-edge counts, and borrowed edges. It is an optional feature source,
not an OVRTX renderer or a silhouette solver. Copy sampled coordinates while
the body is alive, skip degenerate polylines, and reacquire topology after edits.
Sampling reports and partial-output policies belong to upstream guidance.

Kernel edges include periodic seams and smooth joins. Classify design features
using current upstream topology/adjacency guidance; do not draw every edge as a
crease. Borrowed edge handles never cross the process boundary. Assign logical
IDs scoped to a geometry revision and transform coordinates once into the same
world space as the rendered mesh.

Sampled boundaries are polylines, whereas the canonical compositor consumes
ordered independent segments. Convert them deterministically and apply its
existing segment ID/dash phase rule. Continuous arc-length dashes are an
explicit noncanonical polyline extension requiring separate temporal tests.
BRep boundaries do not replace the mesh adjacency/view test for silhouettes.
Only provide silhouette candidates when valid adjacent face data exists.

## Occlusion, Revisions, And Evidence

Feed the same accepted tessellation to OVRTX and to the SPG private geometry-depth
node. Exact edge samples can differ from the mesh by tessellation error; measure
that error and refine rather than hiding it with a large depth bias. Keep mesh,
feature arrays, transforms, selection provenance and exact asset on one revision.
Discard stale worker results and show the last accepted frame during rebuilds.
No browser 3D renderer or native BrepArray rendering capability is implied.

Run the existing preset and temporal gates on deterministic box, cylinder,
trimmed/filleted and multi-part fixtures. Test seam suppression, smooth joins,
sharp boundaries, silhouette transitions, hidden-line occlusion, mirrored/nonuniform
transforms, invalid or partial tessellation, and selection after topology edits.
Record package identity and whether lines use mesh inference or exact boundaries.
The optional adapter does not broaden fidelity claims for unsupported meshes.

## Provenance and independent fidelity checks

Record exact source identity and whether rendered meshes come directly from
accepted kernel handles or from re-imported exact assets. Bind owned geometry
arrays to SPG nodes; do not require CAD imports in the renderer just to shade
those meshes. Use actual hollow/trimmed solids to test rear feature strokes
and compare the old image equations on identical AOVs.

Validate exact export/import separately from tessellation and SPG processing.
Closed-manifold status alone is insufficient: compare volume and geometric
checks under a declared healing policy. If a round trip fails while direct
tessellation renders correctly, retain the latter's scoped rendering evidence
and report the exact-file failure. Neither path implies native analytic BREP
rendering or exact boundary extraction from image AOVs.
