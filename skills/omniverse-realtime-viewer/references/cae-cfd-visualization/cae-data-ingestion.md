<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Native CAE File Ingestion

Use this reference to read EnSight Gold `.case` series or CGNS `.cgns` files
into Warp SimData datasets and fields. For arrays already supplied by a solver
or converter, use the model factories in
[data-and-operators.md](data-and-operators.md).

## Reader and runtime boundary

Use the CAE OpenUSD Plugins native-file readers with `warp_simdata.usd`.
Resolve the plugin, OpenUSD and Warp SimData combination through
`references/dependencies/README.md`; its upstream OpenUSD API and reader guides
own supported formats, schema layouts and compatibility.

Keep this source stage in the scientific ingestion worker when its OpenUSD
runtime differs from the viewer's renderer runtime. The worker owns native-file
reads and operators; return copied geometry, fields and provenance to the viewer
owner for publication into its live OVStage. Retain the source stage and backing
files while lazy array reads are outstanding.

Register plugins before importing scientific schema modules or opening a stage.
Use registration failure as a setup error; do not silently substitute an empty
dataset. Raw-array ingestion remains a separate route when the application
already has a verified reader or solver output.

```python
import cae_openusd_plugins

cae_openusd_plugins.register_usd_plugins()

from pxr import Usd
import warp_simdata.usd as simusd

stage = Usd.Stage.Open(asset_path)
if not stage:
    raise RuntimeError(f"Cannot open scientific asset: {asset_path}")

candidates = list(simusd.find_supported_prims(stage))
for prim in candidates:
    print(prim.GetPath())
```

## Choose a dataset and field

Discovery returns adapter candidates, not disjoint blocks or a guarantee that
conversion succeeds. Select the intended part, zone or section by path and
inspect its fields. A CGNS zone may aggregate its section datasets; do not load
the zone and concatenate those same sections again. Use the reader's zone/section
relationships to select nonoverlapping data, rather than choosing every candidate
or assuming the first prim represents a whole multiblock file. A conversion can
return a `Dataset` or `DatasetCollection`; check the chosen operator's support.

```python
prim = stage.GetPrimAtPath(dataset_prim_path)
if not prim:
    raise ValueError(f"Missing dataset: {dataset_prim_path}")

time_code = Usd.TimeCode.EarliestTime()
fields = {info.name: info for info in simusd.list_fields(prim, time_code=time_code)}
if field_name not in fields:
    raise ValueError(f"Missing field: {field_name}")

dataset = simusd.dataset_from_prim(prim, device="cpu", time_code=time_code)
field = simusd.field_from_prim(prim, field_name, device="cpu", time_code=time_code)
dataset.add_field(field_name, field)
```

Use `FieldInfo.name` as the loader key; its label is for display. Dataset loading
does not attach fields automatically. Use the same device, time code and any
adapter-specific representation for geometry and field loading. CPU execution
is useful for source-value checks; select a supported CUDA device for GPU work.

## Read transient samples together

Native-file arrays can be time sampled even for a single-state asset. Request
`EarliestTime()` or an explicit `Usd.TimeCode(sample_time)` instead of relying
on the loaders' `Default()` argument. Reload geometry and the selected fields
at that same time: EnSight geometry can deform while connectivity stays fixed,
and transient CGNS fields can change on a static grid.

Derive the timeline from the reader's time mapping and source metadata. CGNS and
EnSight readers can map sample indices or source time values; an index with the
default scale does not establish physical elapsed seconds. Do not infer a steady
file merely from sparse lazy-array metadata. Check explicit first, middle and
last samples against source values. Reuse topology only after verifying stable
connectivity and node ordering, not just equal point counts.

Keep source values, association and units with each sample. Display units and
colormap limits are application choices unless the source declares them; report
those choices separately from the data range. Feed these matched samples to
[temporal-playback.md](temporal-playback.md).

## Preserve association through surface extraction

Compare `field.association` with `warp_simdata.AssociationType.NODE` or
`ELEMENT`; these are scientific associations, not USD interpolation strings.
Validate field length against the selected dataset's node or element count.

Use the operator result's topology rather than assuming every dataset has SIDS
`grid_coords` and 1-based `element_connectivity` members. For an external surface,
`element_faces.compute(dataset, external_only=True)` provides points and
0-based face indices plus `node_idx` and `element_idx` source provenance. Gather
source field values through the corresponding provenance:

| Source association | Gather by | Surface primvar interpolation |
|---|---|---|
| `NODE` | output `node_idx` | `vertex`, one value per output point |
| `ELEMENT` | output `element_idx` | `uniform`, one value per output face |

For face-corner colors, expand the gathered face values by face counts or gather
node values through face indices, and use `faceVarying`. Slice and iso-surface
operators create new sample locations; use their supported field interpolation
or explicit sampling contract rather than treating new vertices as source nodes.
See [data-and-operators.md](data-and-operators.md) for operator selection and
[usd-authoring-and-materials.md](usd-authoring-and-materials.md) for publication.
