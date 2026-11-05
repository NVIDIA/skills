<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Exact BRep In USD

Use when a viewer must preserve exact analytic/NURBS geometry and topology
alongside an OVRTX render mesh. Read `dependencies/README.md` → USD BRep first;
follow upstream schema, data-authoring, USD interchange, and validation guidance.

## Exact Asset And Display Representation

Keep exact BrepArray data in an asset or sidecar owned by the modeling worker.
Publish a standard USD Mesh to the OVStage/OVRTX display path. Native BrepArray
support in other upstream viewers does not establish support in OVRTX; test
population and rendering in the selected runtime before choosing direct display.
A historical Fabric population failure is not a universal schema limitation.

Maintain an explicit mapping from logical part ID to exact asset, BrepArray path
and element identity, render prim, and geometry revision. Verify which provenance
attributes survive the selected population path; retain an app-owned mapping if
necessary. Topology changes invalidate face/edge IDs and stale selections.

## Export And Round Trip

Use the packaged `usd_brep.usd` API, checking current stubs and operation docs.
Inspect generated paths and counts rather than assuming generated names or one
prim per part. Use whole-file imports when recovering all elements and the
single-prim importer only with its documented selection semantics.

Export a copy when export's retained trims, attributes, or caches must not alter
the working model. Serialize access to each body and write candidates to temporary
files. Validate before promoting them to the accepted asset; preserve the old
asset when export, import, or tessellation fails.

Compare component counts, bounds, units, transforms, topology and appropriate
solid/sheet invariants after import. Follow upstream guidance on healing,
normalization, trim curves, unbounded face ranges, and partial imports rather than
assuming bitwise identity. Run the public schema validator using the environment
specified by upstream; schema compliance and kernel validity are separate checks.

Capture a multi-part round trip, a curved trimmed face, an invalid asset, and a
failed-export recovery. Confirm the renderer receives the matching mesh revision
and that picking resolves to the correct exact part after reload.

See also: `smlib-brep-pitfalls/README.md`,
`subprocess-geometry-bridge/README.md`, `ovstage-live-geometry/README.md`.
