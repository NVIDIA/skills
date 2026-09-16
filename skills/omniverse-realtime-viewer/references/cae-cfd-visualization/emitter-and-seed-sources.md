<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Emitter And Seed Sources (Movable Streamline Seeding)

## Triggers

Use this reference when streamlines need a **movable seed source** equivalent — or when streamlines **refuse to advect** (curves collapse to
their seed points, probe reads zero). Trigger terms: emitter, seed source, seed sphere,
icosphere seeds, "Unit Sphere source", movable seeds, drag the emitter, streamlines don't
move / all zero / stuck at seeds, `emitter_streamlines`, `icosphere_verts`,
`point_cloud.create_dataset`, `Direction.FORWARD`, voxelize a point cloud, aero car npz,
`gaussian_point_cloud`, interpolable velocity, hex lattice, HEXA_8 binning, seed from an
upstream plane.

Scope: seeding streamlines from a positionable source and — the headline of this doc —
guaranteeing the velocity field is **interpolable** so the integrator actually moves. Data
ingestion and the base `streamlines` operator are in
[data-and-operators.md](data-and-operators.md); USD `BasisCurves` authoring is in
[usd-authoring-and-materials.md](usd-authoring-and-materials.md). Implement the seed source
and streamline operation in the generated app's CAE operator module: keep the seed center in
the data frame, create the display marker from a unit-sphere mesh or USD sphere, and send the
operator output through the normal curve-authoring path.

## The seed-source concept

A movable sphere seed source uses one sphere vertex per seed
point. Our equivalent builds the sphere vertices in numpy, wraps them in a point-cloud
`Dataset`, and hands that to the `streamlines` operator as the seed set. Seeds are always a
**second dataset** — never coordinates passed inline.

```python
import warp as wp
from warp_simdata.data_models.custom import point_cloud
from warp_simdata.operators import streamlines, probe
from warp_simdata.operators.advection import Direction

verts    = icosphere_verts(center, radius, subdiv=2)          # (162,3) float32 seed points
seeds_ds = point_cloud.create_dataset(wp.array(verts, dtype=wp.vec3f, device="cuda:0"))
result   = streamlines.compute(
    volume_ds, "velocity", seeds_ds, direction=Direction.FORWARD,
    initial_dt=0.05, min_dt=0.01, max_dt=0.2, max_steps=300)
```

`volume_ds` must be a **volume** dataset with cell interpolation (see the gotcha below);
`"velocity"` is the vector field name. `Direction` is imported from
`warp_simdata.operators.advection` and is one of `FORWARD` / `BACKWARD` / `BOTH`. Read the
curves back off `result.handle`:

```python
pts = result.handle.points.numpy().reshape(-1, 3)              # (M,3) all curve vertices
cvc = result.handle.curve_vertex_counts.numpy().ravel()        # per-curve vertex counts
```

### The emitter icosphere (`icosphere_verts`)

An icosphere = an icosahedron subdivided `subdiv` times, each vertex projected onto the unit
sphere, then scaled by `radius` and translated to `center`. It distributes seeds far more
evenly than a lat/long sphere. **`subdiv=2` yields 162 unique seeds** — a good streamline
count (subdiv 0 = 12, subdiv 1 = 42, subdiv 3 = 642).

```python
import numpy as np

def icosphere_verts(center, radius, subdiv: int = 2) -> np.ndarray:
    """De-duplicated (M,3) float32 seed points on a sphere. subdiv=2 -> 162 verts."""
    center = np.asarray(center, dtype=np.float64).reshape(3)
    radius = float(radius)
    t = (1.0 + np.sqrt(5.0)) / 2.0
    verts = [np.array(v, dtype=np.float64) for v in (
        (-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0),
        (0, -1, t), (0, 1, t), (0, -1, -t), (0, 1, -t),
        (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1))]
    faces = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
             (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
             (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
             (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1)]
    for _ in range(max(0, int(subdiv))):
        mid, new_faces = {}, []
        def _midpoint(a, b):
            key = (a, b) if a < b else (b, a)
            if key in mid:
                return mid[key]
            verts.append((verts[a] + verts[b]) * 0.5)
            mid[key] = len(verts) - 1
            return mid[key]
        for a, b, c in faces:
            ab, bc, ca = _midpoint(a, b), _midpoint(b, c), _midpoint(c, a)
            new_faces += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
        faces = new_faces
    V = np.asarray(verts, dtype=np.float64)
    norms = np.linalg.norm(V, axis=1, keepdims=True); norms[norms == 0] = 1.0
    V = V / norms                                # project onto unit sphere
    V = np.unique(np.round(V, 6), axis=0)        # de-dupe the shared icosahedron verts
    V = V * radius + center
    return np.ascontiguousarray(V, dtype=np.float32)
```

### The "on move → rebuild → recompute → re-author" loop

The emitter is movable: when the user drags it (new `center`/`radius`), regenerate the seed
vertices, recompute the streamlines, and re-author the `BasisCurves` prim. Nothing is
incremental — seeds are cheap and the integrator is fast, so rebuild the whole set each move.
Keep every OVStage write inside the one serialized command loop (see
[ovstage-render-and-camera.md](ovstage-render-and-camera.md)).

```
on emitter move (center, radius):
    verts    = icosphere_verts(center, radius, subdiv=2)               # rebuild seeds
    seeds_ds = point_cloud.create_dataset(wp.array(verts, dtype=wp.vec3f, device=dev))
    result   = streamlines.compute(volume_ds, "velocity", seeds_ds,    # recompute
                                   direction=Direction.FORWARD, ...)
    pts = result.handle.points.numpy().reshape(-1, 3)
    cvc = result.handle.curve_vertex_counts.numpy().ravel()
    color = probe_speed(volume_ds, "velocity", result, pts)           # coloring, below
    author_basiscurves(pts, cvc, color)                                # overwrite the prim
```

### Coloring the curves

Preferred: **probe the velocity field at every curve vertex** and take its magnitude, giving
a real speed gradient along each line. Fall back to the per-vertex integration `"times"`
field (always present, 1:1 with points) if the probe is unavailable or length-mismatched.

```python
def probe_speed(ds, vfield, result, pts):
    try:
        pr = probe.compute(ds, vfield, result, output_field_name="Vp")
        vp = np.asarray(pr.get_field("Vp").data.numpy())
        speed = np.linalg.norm(vp, axis=1) if vp.ndim == 2 else vp
        if speed.shape[0] == pts.shape[0] and float(np.nanmax(speed)) > 0:
            return speed.astype(np.float32)
    except Exception:
        pass
    # fallback: integration time
    t = result.get_field("times").data.numpy() if "times" in result.get_field_names() else None
    if t is not None and t.shape[0] == pts.shape[0]:
        return np.ascontiguousarray(t, np.float32).ravel()
    return np.linspace(0.0, 1.0, pts.shape[0], dtype=np.float32)
```

`probe.compute` treats `result` (the curve vertices) as a point cloud and samples `ds`'s
velocity there; a vector result field is `(M,3)`, so take its norm. If `nanmax(speed) <= 0`
the field is not interpolating — that is the gotcha below, not a coloring bug.

## THE CRITICAL GOTCHA — streamlines need an *interpolable* velocity field

**This is the headline. A subagent burned hours here.** The streamline integrator advances
each seed by repeatedly asking the dataset "what is the velocity at this position?" That
query only works if the dataset can **interpolate** the field — i.e. it has *elements*
(cells) whose shape functions blend node values. Whether it does depends entirely on the
dataset type:

- **VOLUME mesh (tet/hex — e.g. `cube_swirl`, `disk_out_ref`): works natively.** TETRA_4 /
  HEXA_8 cells give trilinear interpolation. Seed inside the domain and streamlines advect.
- **Bare POINT CLOUD (e.g. the aero car `.npz`): silently produces an ALL-ZERO field.** A
  point cloud has **no elements**. Any voxelization / sampling calls
  `find_elem_containing_position`, which fails at *every* location because there are no cells
  to contain a point, so every voxel falls back to the background value (`0`). The velocity
  field is therefore uniformly zero: streamlines sample zero velocity and **never move**
  (curves collapse onto their seeds), and `probe` reads zero everywhere. There is no error —
  just stuck lines. That silence is what costs the hours.

> `gaussian_point_cloud` is a **red herring** here — in this build its DataModel lacks
> `DatasetAPI`, so it cannot back a `streamlines`/`probe` dataset. Do not reach for it as the
> point-cloud interpolation fix.

### THE FIX — bin the scattered velocities onto a regular hex lattice

Convert the point cloud into a real **HEXA_8 volume** so the operators get genuine trilinear
interpolation. Bin each scattered velocity onto its nearest lattice node (box-average per
node), **dilate-fill** the empty nodes from filled neighbours, then build a SIDS HEXA_8
`ug.create_dataset(...)`. Streamlines then advect through the reconstructed field.

```python
import warp as wp
from warp_simdata import Field
from warp_simdata.fields import AssociationType
from warp_simdata.data_models.sids import unstructured as ug

def build_hex_volume(coords, vel, voxel=0.18):
    """Scattered (N,3) coords + (N,3) velocity -> interpolable HEXA_8 Dataset."""
    mn, mx = coords.min(0), coords.max(0)
    span = mx - mn
    nc = np.maximum((span / voxel).astype(int), 1)      # cells per axis
    nn = nc + 1                                          # nodes per axis
    axes = [np.linspace(mn[i], mx[i], nn[i]) for i in range(3)]
    X, Y, Z = np.meshgrid(*axes, indexing="ij")
    node_coords = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()]).astype(np.float32)

    # Box-average each point's velocity into its nearest node.
    nsp = (span / (nn - 1)).astype(np.float64)
    ijk = np.round((coords - mn) / nsp).astype(int)
    for a in range(3):
        ijk[:, a] = np.clip(ijk[:, a], 0, nn[a] - 1)
    lin = (ijk[:, 0] * nn[1] + ijk[:, 1]) * nn[2] + ijk[:, 2]
    nnode = int(np.prod(nn))
    acc = np.zeros((nnode, 3), np.float64); cnt = np.zeros(nnode, np.float64)
    np.add.at(acc, lin, vel.astype(np.float64)); np.add.at(cnt, lin, 1.0)
    valid = cnt > 0
    nv = np.zeros((nnode, 3), np.float64); nv[valid] = acc[valid] / cnt[valid, None]

    # Dilate-fill empty nodes from filled neighbours so no cell samples the zero background.
    grid = nv.reshape(nn[0], nn[1], nn[2], 3); vg = valid.reshape(nn[0], nn[1], nn[2])
    it = 0
    while not vg.all() and it < max(nn) * 2:
        it += 1
        for ax in range(3):
            for s in (1, -1):
                rolled, rv = np.roll(grid, s, axis=ax), np.roll(vg, s, axis=ax)
                fill = (~vg) & rv
                grid[fill] = rolled[fill]; vg[fill] = True
    node_vel = grid.reshape(-1, 3).astype(np.float32)

    # HEXA_8 (CGNS type 17) connectivity, 1-BASED (see data-and-operators.md).
    ii, jj, kk = (a.ravel() for a in np.meshgrid(
        np.arange(nc[0]), np.arange(nc[1]), np.arange(nc[2]), indexing="ij"))
    nid = lambda i, j, k: (i * nn[1] + j) * nn[2] + k
    conn = np.stack([
        nid(ii, jj, kk),     nid(ii+1, jj, kk),     nid(ii+1, jj+1, kk),     nid(ii, jj+1, kk),
        nid(ii, jj, kk+1),   nid(ii+1, jj, kk+1),   nid(ii+1, jj+1, kk+1),   nid(ii, jj+1, kk+1),
    ], axis=1)
    conn1 = (conn + 1).astype(np.int32).ravel()          # +1 -> CGNS 1-based
    nelem = conn.shape[0]

    ds = ug.create_dataset(
        grid_coords=wp.array(node_coords, dtype=wp.vec3f), element_type=17,
        element_range=wp.vec2i(1, nelem),
        element_connectivity=wp.array(conn1, dtype=wp.int32))
    comps = [wp.array(np.ascontiguousarray(node_vel[:, i]), dtype=wp.float32) for i in range(3)]
    ds.add_field("velocity", Field.from_arrays(comps, AssociationType.NODE))
    return ds, mn, mx, span
```

Detecting which path you are on: `emitter_streamlines` guards with
`ds.get_num_elems() <= 0` (a point cloud reports zero elements) and raises
`NotImplementedError` rather than emitting stuck lines. Route point-cloud inputs through
`build_hex_volume` **before** seeding.

## Illustrative Tuning (Tune In The Generated App)

The values below are starting points for a moderate-scale `+X` flow dataset, not portable
defaults or evidence for a particular sample. Tune them against the dataset bounds, velocity
scale, seed density, and frame-time target, then record the chosen values with the app.

| knob | value | why |
|---|---|---|
| voxel size | `~0.18` | coarse enough that the box-average + dilate produce a fully filled, smooth interpolable volume (fine lattices leave holes and cost memory) |
| seed placement | **upstream plane/sphere** ahead of the body, traced **`Direction.FORWARD`** | seeds must start where the field is nonzero and flow *into* the body; `a practical starting point is a Y–Z plane near the upstream side of the bounds |
| `initial_dt` | `0.02` | first integrator step |
| `min_dt` | `0.004` | floor for the adaptive stepper |
| `max_dt` | `0.10` | ceiling per step |
| `max_steps` | `420` | enough to cross the ~13-unit domain at these dt/velocity |
| `tolerance` | `1e-2` (**loose on purpose**) | a tight tolerance makes the adaptive stepper clamp to tiny steps and stall; loosening it lets lines actually traverse the domain |
| color clamp | probed `\|velocity\|`, **3rd–97th percentile** | so near-body slowdown reads instead of the freestream (`\|v\| ~ 30`) collapsing to one color |

```python
res = streamlines.compute(
    ds, "velocity", seeds_ds, initial_dt=0.02, min_dt=0.004, max_dt=0.10,
    max_steps=420, tolerance=1e-2, direction=Direction.FORWARD)
...
vmin = float(np.percentile(speed, 3))     # robust clamp — near-body slowdown reads
vmax = float(np.percentile(speed, 97))
```

**Velocity/dt scale matters.** Total integration time ≈ `max_steps * mean_dt`. With
`|v| ~ 30` and a ~13-unit domain, the streamline needs total integration time ≈ `13/30 ≈
0.3 s` to cross; the dt schedule above supplies it. If your field's magnitude or domain size
differs by an order of magnitude, rescale `initial_dt`/`max_dt`/`max_steps` proportionally —
otherwise lines either stall short or shoot out of the domain in a few steps.

## Optional — mask interior nodes so streamlines deflect around the body

Currently streamlines can pass *through* the car body (the reconstructed hex field has no
notion of the solid). If the source `.npz` carries a signed-distance field, mask the lattice
nodes that fall **inside** the body (`sdf < 0`) — zero their velocity (or drop their cells)
so seeds deflect around the body instead of tunnelling through it. Sample the npz `sdf` at
each lattice node (nearest-node, same binning as the velocity) and gate:

```python
inside = sample_sdf_at_nodes(node_coords) < 0.0     # nearest-node sdf per lattice node
node_vel[inside] = 0.0                                # or exclude cells that touch inside nodes
```

This is an **honest current limitation**: masking improves realism near the surface but the
reconstruction is still a coarse box-average, so treat deflection as indicative, not
solver-accurate.

## Gotchas

- **Stuck lines mean a zero field, not bad seeds.** If curves collapse to their seed points
  and `probe` reads zero, the dataset is a point cloud with no elements — voxelize to HEXA_8
  first. Check `ds.get_num_elems()`; `<= 0` means no interpolation.
- **`gaussian_point_cloud` won't save you** in this build — its DataModel lacks `DatasetAPI`,
  so it can't back `streamlines`/`probe`. Use the hex-lattice reconstruction.
- **Seed where the field is nonzero.** An emitter sphere/plane outside the domain (or inside
  the masked body) yields empty or zero-length curves. `emitter_streamlines` raises
  `"emitter produced no streamlines"` when `pts.shape[0] == 0`.
- **Loosen `tolerance` (`~1e-2`).** A tight tolerance starves the adaptive stepper down to
  `min_dt` and the lines never traverse the domain.
- **Hex connectivity is 1-BASED.** Add `+1` to the computed node ids before
  `create_dataset` (CGNS convention, matching [data-and-operators.md](data-and-operators.md)).
- **Percentile-clamp the color.** Without a 3rd–97th percentile clamp the freestream
  saturates the colormap and the interesting near-body gradient is invisible.
- **Rebuild, don't patch, on emitter move.** Regenerate seeds → recompute → re-author the
  `BasisCurves` prim each move; keep the write in the serialized OVStage loop.
