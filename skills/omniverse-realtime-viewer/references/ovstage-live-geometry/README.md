<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# OVStage Live Geometry

Use for procedural meshes, CAD rebuilds, or simulation geometry updates. Read
`dependencies/nvidia-runtime.md` and the current OVStage agent guidance, skills,
and public write APIs first. For a modeling source, read `dependencies/README.md`
→ USD BRep and follow its upstream skills and docs before implementation.

## One Runtime Owner

The viewer owns the stage lifetime, PathDictionary, ordinals, renderer attachment,
and frame publication. Workers enqueue owned arrays and a geometry revision;
they never call stage writes or renderer stepping. Follow
`ovstage-data-plane/README.md` and `ovstage-ovrtx-integration/README.md` for the
selected release's batch writes, buffer retention, synchronization, and publish.

For a bounded assembly, predeclare Mesh slots at population time and reuse them.
Validate all candidate parts before writing. Hide unused slots and publish one
coherent revision after all mesh attributes and provenance are ready. If part
count exceeds capacity, use the supported population/stage replacement workflow
in `stage-management/README.md`; do not truncate the assembly silently.

Write points, polygon counts/indices, normals and their interpolation, extent,
visibility and subdivision policy as required by the current runtime. Match
array dtype and shape to the public API. Choose vertex or faceVarying normals
from the actual source layout; source-face corner normals can preserve CAD
creases without re-smoothing. Keep the exact model and its provenance revision
consistent with the accepted mesh. Clear stale topology selection on replacement.

Do not assume a fixed Z-up to Y-up rotation: carry source units and up-axis
metadata and apply one declared transform. Test picking, bounds, and camera fit
against the same world-space geometry used for display.

Before constructing write descriptors, inspect the selected release's
[population property mappings](https://github.com/NVIDIA-Omniverse/ovstage/blob/main/docs/scene/population.rst).
Rendering-domain properties can use pinned runtime representations that differ
from their USD declarations; bounds are one such case. Match semantic, dtype,
lane count, and array/fixed-size status rather than inferring them from USDA.

For mesh-based section caps, use [section views](../section-views/README.md) to
choose the geometry scope and contour policy, then reuse this publication path.

## Responsiveness And Validation

Coalesce interactive rebuild requests and reject superseded results. Continue
streaming the last accepted frame while modeling runs. Detect a completed render
using the selected renderer contract; a fixed BVH-settle delay is not a completion
fence. Follow public CPU/GPU data-plane contracts without private descriptor edits.

Capture repeated swaps, topology and part-count changes, hiding unused slots,
selection invalidation, failed candidate recovery, and worker shutdown. Check
array bounds, finite values, interpolation lengths, retained buffer lifetime,
publication ordering and nonblank frames. Measure update cost and camera latency
on the selected runtime rather than repeating historical speed guarantees.

See also: `subprocess-geometry-bridge/README.md`,
`exact-brep-in-usd/README.md`, `smlib-brep-pitfalls/README.md`.
