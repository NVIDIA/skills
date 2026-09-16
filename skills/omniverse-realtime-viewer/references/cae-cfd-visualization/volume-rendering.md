<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Volume Rendering: Unsupported in This Skill Package

## Triggers

Use this reference when a request includes volume rendering, NVIDIA IndeX,
`UsdVolVolume`, `UsdVol`, `OpenVDBAsset`, NanoVDB, `.nvdb`, `.vdb`, `.vti`, a
voxel field, fog, or a value-driven volume transfer function.

## Capability Boundary

**This skill package does not support NVIDIA IndeX, `UsdVol`, NanoVDB, or direct
volume rendering.** Do not author a `UsdVolVolume`, attach an `OpenVDBAsset`,
write or rewrite NanoVDB files, or claim that an OVRTX viewer can render these
assets through this package.

The absence of a supported path is deliberate. Volume rendering depends on
runtime-specific renderer and volume-stack compatibility that this viewer skill
package does not establish or validate. In particular, do not work around that
boundary with byte-level edits to a volume file, version matching, hidden
renderer plugins, or an undeclared service.

## Safe Alternatives

When a user wants to understand a CAE scalar or vector field, offer a supported
geometry representation instead:

- Surface or iso-surface geometry for thresholds and boundaries.
- Slice geometry for planar field inspection.
- Streamlines, glyphs, or points for vector fields.
- A derived mesh, points, or curves representation that is authored as
  `UsdGeom` and rendered through the normal OVRTX path.

Read [data-and-operators.md](data-and-operators.md) for field-to-geometry
operators and [usd-authoring-and-materials.md](usd-authoring-and-materials.md)
for authoring the resulting geometry. Tell the user explicitly when the desired
visual result requires true volume rendering rather than silently substituting a
different physical interpretation.

## Request Handling

1. Confirm whether a supported geometry representation meets the visualization
   goal.
2. If it does, implement that representation and label it accurately (for
   example, "iso-surface" rather than "volume").
3. If true volume rendering is a requirement, stop at this boundary and ask the
   user to choose a renderer/runtime with documented volume support. Do not
   invent setup, binary conversion, or compatibility instructions.

## See Also

- [README.md](README.md) — CAE/CFD family routing and supported representations.
- [data-and-operators.md](data-and-operators.md) — CAE field operators.
- [streaming-cae-viewer.md](streaming-cae-viewer.md) — streamed viewer lifecycle
  for supported geometry.
