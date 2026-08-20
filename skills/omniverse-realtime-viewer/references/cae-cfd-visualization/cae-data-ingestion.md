<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Native CAE File Ingestion (EnSight Gold + CGNS, library-only)

## Triggers

Use this reference to read a **native CAE solver file** — an EnSight Gold `.case`/`.geo`
series or a CGNS `.cgns` file — into a `warp_simdata.Dataset` using `warp_simdata` directly (without `cae-openusd-plugins`). Trigger terms: EnSight Gold, `.case`, `.geo`, C Binary EnSight,
CGNS, `.cgns`, `warp_simdata.io.cgns`, `warp_simdata.data_models.ensight_gold`,
`create_dataset`, transient CGNS, `FlowSolution`, `SolutionVertex`, cell-centered vs
node field, GridLocation, per-timestep series, "read the bumper_beam case", "load the
thermal CGNS", "ingest EnSight/CGNS directly", NODE vs ELEMENT association.

Scope: getting native `.case`/`.cgns` files into a renderable `Dataset` + per-step field
arrays. The `.npz` (CGNS-coded arrays) path, the SIDS builder, and the GPU visualization
operators are in [data-and-operators.md](data-and-operators.md) — read that for the
operator stack; this doc is the two **native-file readers** that feed it.

## The critical caveat: `cae-openusd-plugins` may be absent — prefer library-direct readers

The "clean" story is that `cae-openusd-plugins` registers an `SdfFileFormat` plugin plus
OmniSci (`OmniSci*`/`Cae*`) USD schemas so a solver file composes as USD prims. **In
practice that package is frequently not installed in a working venv.** When it is missing,
every `warp_simdata.usd.*` adapter is dead — each one does `from pxr import OmniSci*` at
import and raises. There is also no guarantee VTK / pyvista / meshio are present.

**Do not build the ingestion path on the USD `SdfFileFormat` route.** Build it on the
**library-direct readers** that only need `warp_simdata` + `h5py` (CGNS) and pure Python
(EnSight). They are the durable path for a generated app.
Probe for the USD path if you want it, but always have the library-direct fallback:

```python
def openusd_plugins_available() -> bool:
    try:
        from pxr import OmniSciCae  # noqa: F401  (any OmniSci schema module)
        return True
    except Exception:
        return False   # -> use the library-direct readers below
```

Place the readers in the app's CAE ingestion module. Normalize their outputs to points,
0-based connectivity, field values, association, and optional timesteps before passing them
to the operator and USD-authoring modules.

## CGNS via `warp_simdata.io.cgns` (h5py)

`warp_simdata.io.cgns.read(path, zone=None, device="cpu")` returns
`{section_name: Dataset}` with grid coords, `element_connectivity` (**1-based**), and the
FlowSolution fields. It reads `GridLocation` to set each field's association: **NODE**
(vertex fields) or **ELEMENT** (cell-centered). Take the first section for the common
single-block case.

```python
from warp_simdata.io import cgns
blocks = cgns.read(path, device="cpu")     # requires h5py
ds = next(iter(blocks.values()))           # first section
```

Proven: `compute_thermal.cgns` = 71534 nodes, HEXA_8 (code 17), `Temperature` cell-centered
(ELEMENT, 24.4–39.1 °C); `bscw_cfd.cgns` = 4991 nodes, `Cp` node field.

### The transient gotcha: the builtin reader collapses to the LAST step

A transient CGNS file stores one `FlowSolution_t` per timestep, but **they are all named
the same field** (e.g. every step's solution is `Cp`). `cgns.read` calls `add_field` once
per solution, so each step **overwrites** the previous — you get only the final step's
values. The topology is fine; the field is wrong for any step but the last.

**Workaround — read the static grid once, pull each step's array via h5py:**

```python
import h5py, numpy as np

ds = next(iter(cgns.read(path, device="cpu").values()))   # static grid/topology
steps = []
with h5py.File(path, "r") as f:
    for base in f:                                          # e.g. "Base"
        for zname in f[base]:
            z = f[base][zname]
            if not hasattr(z, "keys"):
                continue
            sols = sorted(k for k in z.keys()
                          if k.startswith(("SolutionVertex", "SolutionCellCenter", "FlowSolution")))
            got = [np.asarray(z[f"{s}/{field}/ data"][...]).ravel()   # NB leading-space " data"
                   for s in sols if f"{s}/{field}/ data" in z]
            if got:
                steps = got
                break
        if steps:
            break
# steps[i] is the field array at timestep i, over the shared topology.
```

Two h5py naming traps that will silently return nothing if you get them wrong:

- **The HDF5 leaf dataset name has a leading space: `" data"`**, not `"data"`. Every CGNS
  array node stores its payload under a child literally named `" data"`.
- **Timestep values** live under `Base/TimeIterValues/TimeValues/ data` (also the
  leading-space form). Use them to label the timeline; an empty array means the file is
  steady (single state).

## EnSight Gold `.case` + `.geo` (C Binary) — a custom parser

There is **no installed EnSight reader**. EnSight Gold is a small, stable binary format,
so a direct parser is the reader. Two-stage: parse the ASCII `.case` for the geometry
pattern, variables, and TIME block; then parse each C-Binary `.geo` (and per-step variable
files) into points + element pieces.

**`.case` (ASCII):** `model:` gives the `.geo` filename pattern (a `******` wildcard →
zero-padded step number); `scalar per node:` / `scalar per element:` declare variables as
`<desc> <name> <pattern>`; the TIME section gives `number of steps:`,
`filename start number:`, `filename increment:` → the step index list.

**`.geo` (C Binary):** fixed 80-byte strings, then per part `coordinates` → `nn` nodes as a
`3×nn` float block **transposed** to `(nn,3)`, then per-element-type pieces
(`quad4`/`tria3`/`tetra4`/`hexa8`/…) each with an `int32` connectivity block. **Connectivity
is 1-based — subtract 1** before writing `faceVertexIndices`. If `node id`/`element id` is
`given` or `ignore`, skip that id block (`4*n` bytes) before the coordinates/connectivity.

Build the `Dataset` with the EnSight-specific model, mapping the shape token to the
`ensight_shapes` code and creating one piece handle per element-type block:

```python
import warp as wp
from warp_simdata.data_models.ensight_gold import ensight_shapes as ES
from warp_simdata.data_models.ensight_gold.unstructured_part import (
    create_dataset, create_piece_handle,
)

pts = wp.array(part.points.astype("float32"), dtype=wp.vec3f, device="cpu")
handles = []
for (etoken, npe, ne, conn) in part.pieces:                 # conn is (ne, npe), 1-based
    code = getattr(ES, {"quad4": "EN_quad4", "tria3": "EN_tria3", ...}[etoken])
    handles.append(create_piece_handle(
        element_type=code,
        connectivity=wp.array((conn - 1).reshape(-1).astype("int32"),  # 1-based -> 0-based
                              dtype=wp.int32, device="cpu"),
    ))
ds = create_dataset(points=pts, pieces=handles)
```

### Per-element variable blocks + deforming geometry

- **Per-element variables are written one float block PER element-type piece**, in piece
  order. To read a `scalar per element:` file you must know the element counts of each piece
  and concatenate the blocks in the same order the geometry pieces were built. A per-node
  variable is a single block of `num_nodes` floats.
- **Deforming geometry: re-parse the `.geo` every step.** The bumper_beam crash case moves
  its nodes each step (span deforms ~219 units by step 25) while the **topology stays
  stable**. Read points (and the field) fresh per step; author the mesh topology once.
  Validated: bumper_beam = 51 steps, 13882 nodes, quad4 + tria3 shells, `PlasticStrain`
  grows 0 → 0.54.

## NODE vs ELEMENT association → primvar interpolation

Both readers tell you whether a field is per-node or per-cell — carry that through to the
authored primvar interpolation, or the colors land on the wrong elements:

| Source signal | Association | `displayColor` interpolation | Array length |
|---|---|---|---|
| CGNS `GridLocation = Vertex`; EnSight `scalar per node:` | **NODE** (`vertex`) | `interpolation = "vertex"` | `N` (nodes) |
| CGNS `GridLocation = CellCenter`; EnSight `scalar per element:` | **ELEMENT** (`uniform`) | `interpolation = "uniform"` | `F` (faces/cells) |

`warp_simdata`'s `field.association` is `'vertex'` for NODE, `'uniform'` for ELEMENT. See
[usd-authoring-and-materials.md](usd-authoring-and-materials.md) for the primvar
interpolation rules the association drives.

## Bridge to render arrays

Both readers land on the same shape the USD authoring + colormap path consumes: points
`(N,3)`, **0-based** connectivity, element type, field values, and association.

```python
def dataset_render_arrays(ds, field_name=None):
    pts   = ds.handle.grid_coords.numpy()                              # (N,3)
    conn  = ds.handle.element_connectivity.numpy().astype("int64") - 1  # flat, 1-based -> 0-based
    etype = int(ds.handle.element_type)
    vals, assoc = None, None
    if field_name:
        f = ds.get_field(field_name)
        if f is not None:
            vals  = f.to_array().numpy()          # + f.get_range() for the colormap domain
            assoc = str(f.association)             # 'vertex' (NODE) | 'uniform' (ELEMENT)
    return pts, conn, etype, vals, assoc
```

Volumetric cells (HEXA_8, TETRA_4) still need an operator (`element_faces` /
`iso_surface` / `slice` / `streamlines`) to become a renderable surface; an EnSight
quad4/tria3 surface is already a boundary mesh → author `UsdGeomMesh` directly. See
[data-and-operators.md](data-and-operators.md).

## Gotchas

- **`cae-openusd-plugins` / OmniSci schemas may be absent.** Never assume the USD
  `SdfFileFormat` path is available — probe, and fall back to the library-direct readers.
- **Connectivity is 1-based** from both formats — subtract 1 for USD indices.
- **CGNS transient collapses to the last step** — the h5py per-step workaround with the
  leading-space `" data"` dataset name is mandatory for anything but the final frame.
- **EnSight per-element vars are per-piece blocks** — concatenate in piece order using the
  element counts, or the values misalign with the faces.
- **Re-parse `.geo` per step for deforming cases**, but author topology once (it is stable).
- **Carry NODE/ELEMENT association through to primvar interpolation** (`vertex` vs
  `uniform`), or coloring lands wrong.

## See also

- [data-and-operators.md](data-and-operators.md) — the `.npz`/SIDS builder and the
  `element_faces`/`iso_surface`/`slice`/`streamlines` operators these datasets feed.
- [temporal-playback.md](temporal-playback.md) — turning the per-step EnSight/CGNS series
  into a scrubbable transient clip.
- [usd-authoring-and-materials.md](usd-authoring-and-materials.md) — primvar interpolation
  (`vertex`/`uniform`) and colormap authoring.
- `references/dependencies/README.md` — acquiring `warp_simdata` (and, if present,
  `cae-openusd-plugins`).
