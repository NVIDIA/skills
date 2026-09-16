<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# CAE/CFD Data And Operators

## Triggers

Use this reference for CAE/CFD data ingestion and GPU visualization operators: `warp_simdata`, `warp-simdata`, `cae-openusd-plugins`, CGNS, OpenFOAM, VTK/VTU/VTI, EnSight, FLASH, SIDS, `create_dataset`, `element_faces`, `iso_surface`, `slice`, `streamlines`, `element_connectivity`, `element_type` (CGNS codes), TETRA_4/HEXA_8/TRI_3/MIXED, isosurface, planar slice, external faces, streamline seeds, or turning a `.npz`/CGNS mesh into a renderable surface/curve set.

Scope of this doc: get CAE solver data into a renderable `warp_simdata.Dataset`, then run operators and read back geometry + fields. Downstream USD authoring, colormap baking, and OVStage/ovrtx rendering are covered by the rendering references.

## The general pattern (before the specifics)

However your data arrives — CGNS, OpenFOAM, VTK/VTU, EnSight, FLASH, a `.npz` of
arrays, raw numpy in memory, in-memory solver output, or a streamed buffer — the
goal is always the same: **get it into a `warp_simdata.Dataset` (or the renderable
arrays that back one), then run operators and read geometry + fields back.** The
routes below (the `.npz` SIDS builder, the direct CGNS reader, and the native-file
readers in [cae-data-ingestion.md](cae-data-ingestion.md)) are **examples of that
one pattern**, not the only path. Whatever gets you node coordinates, element
connectivity + type, and per-node/element field arrays onto the GPU will feed the
same operators.

## Libraries

Referred to by **role** so the prose survives a rename or repository move.
Acquisition (both are public on GitHub) is centralized in
`references/dependencies/README.md` (CAE/CFD Visualization Libraries); do not
duplicate install steps or URLs here.

- **the `cae-openusd-plugins` USD reader/schema library** — read-only USD ingestion. Registers an `SdfFileFormat` plugin plus OmniSci (`Cae*`/`OmniSci*`) schemas that expose CGNS / OpenFOAM / VTK / EnSight / FLASH files as composed USD with **data-on-demand** field arrays (arrays are not read until an operator pulls them). Use this when the app should reference solver files directly as USD prims. Optional — frequently absent; keep the library-direct fallback in [cae-data-ingestion.md](cae-data-ingestion.md).
- **the `warp-simdata` GPU operator library** — the GPU operator stack (NVIDIA Warp kernels). Builds an in-memory `Dataset` from raw arrays and runs `element_faces` / `iso_surface` / `slice` / `streamlines` / points on it. Imports Warp lazily; requires CUDA. This doc's recipes target this library.

Import guard (so the package still imports without CUDA/warp):

```python
def available() -> bool:
    try:
        import warp, warp_simdata  # noqa: F401
        return True
    except Exception:
        return False
```

## npz Array Schema (CGNS-coded)

One convenient ingestion route — used here as the worked example — is a NumPy `.npz` of
CGNS-coded arrays. It is an example of the general pattern above, not a required format: any
source that yields these same arrays feeds `create_dataset` identically. Load with
`allow_pickle=False`. If an app needs to make such a file, implement a small converter that
normalizes its source format to the schema below; keep that converter with the app.

| key | shape / dtype | meaning |
|---|---|---|
| `coords` | `(N,3)` float32 | node positions (also accepts split `coords_0/1/2` or `x/y/z`) |
| `element_connectivity` | flat int, **1-BASED** | node ids per element, CGNS ordering (e.g. 8/hex, 4/tet, 3/tri) |
| `element_type` | `(1,)` int | CGNS SIDS type code: `5`=TRI_3, `10`=TETRA_4, `17`=HEXA_8, `20`=MIXED |
| `element_range` | `(2,)` int64 | `[1, nElem]` |
| `<field>` | `(N,)` or `(N,3)` numeric | per-node scalar or vector solution field |

MIXED (`20`) sections carry a `start_offsets` array (per-element type/offset table); prefer single-type sections (TRI_3 / TETRA_4 / HEXA_8) unless you specifically handle MIXED.

## Build A SIDS Volume Dataset

```python
import numpy as np, warp as wp
from warp_simdata import Field
from warp_simdata.data_models.sids import unstructured as ug
from warp_simdata.fields import AssociationType

wp.init()  # idempotent; safe to call every load

npz  = np.load(npz_path, allow_pickle=False)
coords = np.ascontiguousarray(npz["coords"], dtype=np.float32)          # (N,3)
et   = int(np.asarray(npz["element_type"]).ravel()[0])                  # CGNS code
conn = np.ascontiguousarray(npz["element_connectivity"]).astype(np.int32)  # flat, 1-BASED
er   = np.asarray(npz["element_range"]).astype(np.int64).ravel()

ds = ug.create_dataset(
    grid_coords=wp.array(coords, dtype=wp.vec3f, device="cuda:0"),
    element_type=et,
    element_range=wp.vec2i(int(er[0]), int(er[1])),
    element_connectivity=wp.array(conn, dtype=wp.int32, device="cuda:0"),
)
```

Attach solution fields (skip coord and id/index arrays — they are not physical fields):

```python
N = coords.shape[0]
for k in npz.keys():
    if k in {"coords", "element_type", "element_connectivity", "element_range"} or _is_id_like(k):
        continue
    arr = np.asarray(npz[k])
    if arr.shape[0] != N or not np.issubdtype(arr.dtype, np.number):
        continue
    if arr.ndim == 1:                                   # scalar, per node
        v = np.ascontiguousarray(arr, dtype=np.float32)
        ds.add_field(k, Field.from_array(wp.array(v, device="cuda:0"), AssociationType.NODE))
    elif arr.ndim == 2 and arr.shape[1] == 3:           # vector, per node
        v = np.ascontiguousarray(arr, dtype=np.float32)
        comps = [wp.array(np.ascontiguousarray(v[:, i]), device="cuda:0") for i in range(3)]
        ds.add_field(k, Field.from_arrays(comps, AssociationType.NODE))   # 3 component arrays
```

`_is_id_like(name)` excludes `ids`/`id`/`index`/`indices`, `*_id`, and any name containing `element`. Color a vector by magnitude with a synthetic `"<vec>|mag"` name computed in numpy (`np.linalg.norm(vec, axis=1)`); `add_field` itself takes only scalar or 3-component vector fields.

### Direct CGNS (no npz)

```python
import warp_simdata
ds = warp_simdata.io.cgns.read(path)   # requires h5py
```

Use this to skip the npz step. If an app needs a portable CGNS/HDF5 converter, read the
`Elements_t` sections, remap referenced nodes to a compact 0-based node set, and extract
`FlowSolution` vertex or cell fields into the schema above. A minimal smoke fixture can be a
single HEXA_8 cell with a three-component swirl velocity field; generate it in the app's test
suite rather than relying on an external sample file.

## Operators

Each `compute(...)` returns a `Dataset`. Read geometry off `result.handle` and fields off `result.get_field(name)`.

```python
h   = result.handle
pts    = h.points.numpy().reshape(-1, 3)          # (M,3) float32   — all operators
counts = h.face_vertex_counts.numpy().ravel()     # per-face vertex counts — surface/iso/slice
idx    = h.face_vertex_indices.numpy().ravel()    # flat face indices     — surface/iso/slice
cvc    = h.curve_vertex_counts.numpy().ravel()    # per-curve vertex counts — streamlines
vals   = result.get_field(name).data.numpy()      # per-vertex field (guard with result.get_field_names())
```

| operator | import | call | output | field-on-result |
|---|---|---|---|---|
| external faces | `from warp_simdata.operators import element_faces` | `element_faces.compute(ds, external_only=True)` | faces (`points`/`face_vertex_counts`/`face_vertex_indices`) | gather source field via the result's `node_idx` field: `src_vals[result.get_field("node_idx").data.numpy().astype(int)]` |
| isosurface | `from warp_simdata.operators import iso_surface` | `iso_surface.compute(ds, field, value, field_names=[field])` | faces | contoured on `field` at `value`; `field_names=[...]` scalars are interpolated onto the surface, read via `get_field` |
| planar slice | `from warp_simdata.operators import slice as slice_op` | `slice_op.compute(ds, origin, normal, field_names=[name])` | faces | `origin`/`normal` are 3-tuples; requested `field_names` transferred onto the cut, read via `get_field` |
| streamlines | `from warp_simdata.operators import streamlines` | `streamlines.compute(ds, vfield, seeds, initial_dt=0.1, min_dt=0.01, max_dt=0.3, max_steps=200, direction="both")` | curves (`points`/`curve_vertex_counts`) | color by the per-vertex `"times"` field (1:1 with points) |
| points | — | none (use `ds` coords directly) | point cloud | color by any source node field directly |

`external_only=True` keeps only boundary faces. `direction` is `forward` \| `backward` \| `both`. Streamline seeds are a **second dataset**:

```python
import warp as wp
from warp_simdata.data_models.custom import point_cloud
seeds = point_cloud.create_dataset(wp.array(seed_pts, dtype=wp.vec3f, device="cuda:0"))  # seed_pts (S,3)
```

## Minimal End-To-End (npz → dataset → external faces → arrays)

```python
import numpy as np, warp as wp
from warp_simdata import Field
from warp_simdata.data_models.sids import unstructured as ug
from warp_simdata.fields import AssociationType
from warp_simdata.operators import element_faces

wp.init()
npz = np.load("data/cube_swirl.npz", allow_pickle=False)   # HEXA_8 volume + 'speed', 'V'
coords = np.ascontiguousarray(npz["coords"], dtype=np.float32)
conn   = np.ascontiguousarray(npz["element_connectivity"]).astype(np.int32)  # 1-based
er     = np.asarray(npz["element_range"]).astype(np.int64).ravel()

ds = ug.create_dataset(
    grid_coords=wp.array(coords, dtype=wp.vec3f, device="cuda:0"),
    element_type=int(np.asarray(npz["element_type"]).ravel()[0]),
    element_range=wp.vec2i(int(er[0]), int(er[1])),
    element_connectivity=wp.array(conn, dtype=wp.int32, device="cuda:0"),
)
speed = np.ascontiguousarray(npz["speed"], dtype=np.float32)
ds.add_field("speed", Field.from_array(wp.array(speed, device="cuda:0"), AssociationType.NODE))

surf = element_faces.compute(ds, external_only=True)          # boundary surface
h = surf.handle
pts    = h.points.numpy().reshape(-1, 3)
counts = h.face_vertex_counts.numpy().ravel()
idx    = h.face_vertex_indices.numpy().ravel()
node_idx = surf.get_field("node_idx").data.numpy().astype(np.int64)
face_speed = speed[node_idx]                                  # source field gathered onto surface verts
# pts/counts/idx/face_speed now feed the USD authoring + colormap path.
```

## Gotchas

- **Connectivity is 1-BASED.** `element_connectivity` uses CGNS 1-based node ids. A portable converter may remap through 0-based indexes internally, but must add `+1` before calling `create_dataset`. Do not pass 0-based ids to `create_dataset`.
- **Volumetric reps need a volume.** `element_faces` / `iso_surface` / `slice` / `streamlines` require volumetric cells (TETRA_4 `10`, HEXA_8 `17`, MIXED `20`). A **TRI_3 (`5`) surface dataset** is already a boundary mesh — it can only do `points` and (identity) `element_faces`; iso/slice/streamlines have no interior to cut or integrate.
- **`wp.init()` is idempotent.** Call it at the top of every load; repeated calls are safe.
- **Exclude id/coord arrays from fields.** Coord components and `element_*`/id/index arrays are geometry/topology, not solution fields — attaching them pollutes the field list and breaks coloring. Filter with `_is_id_like` and the coord-key set.
- **`allow_pickle=False`.** npz files saved with pickled object arrays (some legacy exports) will raise on load; only plain array npz is accepted.
- **Read fields defensively.** Check `name in result.get_field_names()` before `get_field(name)`; iso/slice/streamlines only carry the fields you request via `field_names` (or the always-present `times`/`node_idx`). A vector result field is `(M,3)` — take its magnitude for scalar coloring.
- **Device consistency.** Keep `grid_coords`, `element_connectivity`, field arrays, and seed points on the same Warp device (e.g. `"cuda:0"`).
