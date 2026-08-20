<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Glyphs (Vector Arrows via UsdGeomPointInstancer)

## Triggers

Use this reference to render a **vector field as oriented glyphs**
representation (`CaeVizGlyphsAPI`). Trigger terms: glyphs, vector arrows, arrow glyphs,
velocity arrows, `UsdGeomPointInstancer`, point instancer, orient by velocity, scale by
field, cone/arrow prototype, quiver, hedgehog plot, show the flow direction, glyph the
velocity field.

Scope: sampling a vector field at points and authoring one oriented, field-scaled,
field-colored prototype per sample. Getting the field arrays is in
[data-and-operators.md](data-and-operators.md); lighting/colormap material is in
[usd-authoring-and-materials.md](usd-authoring-and-materials.md); up-axis rotation is in
[coordinate-systems-and-up-axis.md](coordinate-systems-and-up-axis.md). This is **pure USD
geometry** — no new renderer path, no warp operator required (you already have the vectors).

For a portable smoke test, generate a small regular point grid with a known vector field
(for example, circular flow around the origin). Assert that the authored arrays have matching
instance counts and capture a rendered frame from the app's normal render path.

## The shape of it

A `UsdGeomPointInstancer` places one **prototype** (an arrow — the simplest renderable is a
`UsdGeomCone` with `axis = "Z"`) at each sample point. Per instance you author:

- `point3f[] positions` — the sample points (display up-axis).
- `int[] protoIndices` — which prototype (all `0` for a single arrow prototype).
- `quath[] orientations` — a **half-precision quaternion `(w, x, y, z)`** rotating the
  prototype's local axis (`+Z` for a Z-axis cone) onto the unit velocity direction.
- `float3[] scales` — per-instance `(girth, girth, length)`; grow `length` with `|V|`.
- `color3f[] primvars:displayColor` `(interpolation = "vertex")` — one color per instance
  from the colormap of `|V|` (see caveat below).
- `rel prototypes = [ <path/to/Arrow> ]` — the prototype prim(s), conventionally under a
  child `Scope "proto"` so they are not themselves drawn.

## Building the per-instance arrays (numpy)

Sample the field (subsample for a legible glyph count — a few thousand), then compute the
orientation quaternion from `+Z` to each velocity direction. This is a rotation about the
axis `z × v̂` by the angle `acos(z · v̂)`, with a fallback axis for the (anti)parallel case.

```python
import numpy as np

pos  = coords[sel]                       # (N,3) display up-axis (rotate Z-up→Y-up first)
vel  = V[sel]                            # (N,3) same rotation applied
mag  = np.linalg.norm(vel, axis=1)
vhat = vel / np.where(mag > 1e-12, mag, 1.0)[:, None]

z    = np.array([0.0, 0.0, 1.0])         # cone axis = "Z"
dot  = np.clip(vhat @ z, -1.0, 1.0)
axis = np.cross(np.broadcast_to(z, vhat.shape), vhat)
alen = np.linalg.norm(axis, axis=1)
axis = np.where(alen[:, None] > 1e-8,
                axis / np.where(alen[:, None] > 1e-8, alen[:, None], 1.0),
                np.array([1.0, 0.0, 0.0]))          # fallback for parallel/antiparallel
half = np.arccos(dot) * 0.5
quat = np.column_stack([np.cos(half), axis * np.sin(half)[:, None]])   # (N,4) w,x,y,z

diag   = float(np.linalg.norm(pos.max(0) - pos.min(0)))
girth  = diag * 0.006
length = girth * 2.0 + (mag / mag.max()) * (diag * 0.05)
scales = np.column_stack([np.full_like(length, girth),
                          np.full_like(length, girth), length])
```

Two things that bite:

- **`orientations` is `quath` (half float), component order `(w, x, y, z)`** — real part
  first. Getting the order wrong points every arrow the wrong way.
- **Rotate the vectors, not just the points, into the display up-axis.** If the solver data
  is Z-up and the scene is Y-up, apply the same `(x, y, z) → (x, z, -y)` to `V` that you
  apply to `coords`, or the arrows point in physically wrong directions
  ([coordinate-systems-and-up-axis.md](coordinate-systems-and-up-axis.md)).

## The USD (authored as a string, published through OVStage)

```usda
def PointInstancer "Glyphs"
{
    point3f[] positions = [ (x0,y0,z0), ... ]
    int[] protoIndices = [ 0, 0, ... ]
    quath[] orientations = [ (w,x,y,z), ... ]
    float3[] scales = [ (g,g,l), ... ]
    color3f[] primvars:displayColor = [ (r,g,b), ... ] (interpolation = "vertex")
    rel prototypes = [ </Session/Glyphs/proto/Arrow> ]
    def Scope "proto"
    {
        def Cone "Arrow" (prepend apiSchemas = ["MaterialBindingAPI"])
        {
            double height = 1.0
            double radius = 0.5
            uniform token axis = "Z"
            rel material:binding = </Session/Looks/GlyphMat>
        }
    }
}
```

Bind a `UsdPreviewSurface` whose `diffuseColor` reads the `displayColor` primvar (the same
material pattern the surfaces use — see
[usd-authoring-and-materials.md](usd-authoring-and-materials.md)). The path tracer renders
unlit geometry **black**, so author a dome + distant light as usual. Note the primvar
metadata `(interpolation = "vertex")` goes **after** the value, or the inline USDA fails to
parse.

## Per-instance color caveat (verified, still rough)

`primvars:displayColor (interpolation = "vertex")` sized to the instance count is delivered
**per instance** and colors most glyphs correctly. In the reference render a **subset of
instances render magenta** — ovrtx's missing-material fallback (magenta is not a viridis
color) — accompanied by a benign `Ill-formed SdfPath <>` warning. Treat per-instance color
as **working but not yet hardened**. If you hit the magenta fallback, options in order of
preference:

1. Bind the colormap material at the `PointInstancer` level as well as on the prototype.
2. Fall back to a **single constant color** (`interpolation = "constant"`, one RGB) and
   encode magnitude through `scales` length only — always robust.
3. Split into a few PointInstancers, one per colormap band, each with a constant color.

Magnitude is always legible through arrow **length** regardless of the color path, so a
constant-color instancer is a safe default when fidelity of per-instance color matters less
than never showing a fallback.

## Interactive updates

Glyph arrays are ordinary attributes on a persistent prim, so a moving cutting plane or a
changed field updates them the same way streamlines do: prefer **data-plane array writes**
(`positions`/`orientations`/`scales`/`displayColor`) over a structural rebuild — see
[driving-cae-viz-via-ovstage.md](driving-cae-viz-via-ovstage.md). As with curve `widths`,
if the instance **count** changes, re-publish the arrays together at one ordinal; per-vertex
metadata that goes stale against a new count is rejected by RTX.

## See also

- [data-and-operators.md](data-and-operators.md) — get the vector field arrays.
- [usd-authoring-and-materials.md](usd-authoring-and-materials.md) — colormap material + lights.
- [coordinate-systems-and-up-axis.md](coordinate-systems-and-up-axis.md) — rotate points **and** vectors.
- [driving-cae-viz-via-ovstage.md](driving-cae-viz-via-ovstage.md) — live array writes on a persistent prim.
