<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# CAE/CFD USD Authoring And Materials

## Triggers

Use this reference when turning CAE/CFD operator output (a surface, iso-surface,
slice, point cloud, or streamlines plus a scalar field) into a renderable,
colored USD prim for an `ovrtx` viewer: composing the viewer-root USDA,
authoring `UsdGeomMesh`/`UsdGeomPoints`/`UsdGeomBasisCurves` geometry, baking a
colormap into `primvars:displayColor`, wiring a `UsdPreviewSurface`, and adding
the lights the path tracer needs. Read `references/conventions.md` and
`references/ovrtx-rendering/README.md` first for the render loop and import
order.

This document covers exactly one step: **operator output (points + a field) → a
renderable, colored USD prim** inside a composed viewer stage. It does not cover
solving, meshing, or field extraction; those produce the `points`,
`faceVertexCounts`/`faceVertexIndices` (or `curveVertexCounts`), and per-vertex
scalar field this document consumes.

## Author The Viewer USDA As A Text String

Compose the whole viewer-root stage as a **USDA text string** and hand it to
OVStage with `population.open_usd_from_string(stage, usda, ordinal=N,
domains=PopulationDomain.RENDERING)`. Do **not** import `pxr` / usd-core in the
render process to build the stage in memory. Mixing usd-core with the ovrtx
bundled USD in the wrong order causes duplicate-symbol crashes and MDL resolver
failures (see `references/ovrtx-rendering/README.md` import-order notes). Text
authoring keeps this process pxr-free and lets OVStage own composition and
population.

Implement a small application-owned USDA builder from numpy arrays. Keep it pxr-free and split
it into block builders for the session header, render/camera, lights/materials, and each geometry
prim. Its structure, top to bottom under one session scope:

```
/Session (Scope, defaultPrim, upAxis="Y")
  Cameras/Main         UsdGeomCamera   (resolution-derived aperture, xform)
  Render/Viewport      RenderProduct   (uniform int2 resolution)
  Render/Vars/LdrColor RenderVar       (sourceName="LdrColor")
  Render/Settings      RenderSettings
  Lights/Dome, Lights/Key   DomeLight + DistantLight
  Looks/FieldMat       Material        (UsdPreviewSurface + PrimvarReader)
  Geometry/<prim>      Mesh | Points | BasisCurves  (+ displayColor primvar)
```

Header and root scope:

```usda
#usda 1.0
(
    defaultPrim = "Session"
    doc = "cae viewer root (colormap=viridis, field=Pressure)"
    upAxis = "Y"
)

def Scope "Session"
{
    # ... camera, render, lights, looks, geometry blocks ...
}
```

### Camera And Render Blocks

The RenderProduct `resolution` is viewer-owned and must be set in this composed
layer before the stage loads; the RenderProduct path is the same path passed to
`renderer.step(render_products={...})`. Keep it fixed for the session (typically
1920x1080 for streaming). Camera matrices are row-major with translation in row
3 (right, up, `-forward`, eye).

```usda
    def Xform "Cameras"
    {
        def Camera "Main"
        {
            float2 clippingRange = (0.1, 100000)
            float focalLength = 24
            float horizontalAperture = 20.955
            float verticalAperture = 11.7872
            matrix4d xformOp:transform = ( (1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1) )
            uniform token[] xformOpOrder = ["xformOp:transform"]
        }
    }
    def Scope "Render"
    {
        def RenderProduct "Viewport"
        {
            rel camera = </Session/Cameras/Main>
            uniform int2 resolution = (1920, 1080)
            rel orderedVars = [ </Session/Render/Vars/LdrColor> ]
        }
        def Scope "Vars"
        {
            def RenderVar "LdrColor"
            {
                string sourceName = "LdrColor"
                token sourceType = "raw"
            }
        }
        def RenderSettings "Settings"
        {
            rel products = [ </Session/Render/Viewport> ]
            rel camera = </Session/Cameras/Main>
            uniform int2 resolution = (1920, 1080)
        }
    }
```

Request only `LdrColor` for a basic viewer. `verticalAperture` is derived from
`horizontalAperture * height / width` so pixels stay square.

## Geometry Prims By Representation

Pick the prim type from the operator representation:

| Representation | USD prim |
|---|---|
| external faces / iso-surface / slice | `UsdGeomMesh` |
| point cloud | `UsdGeomPoints` |
| streamlines | `UsdGeomBasisCurves` |

All three carry the same two field primvars (see next section):
`primvars:displayColor` (baked color) and `primvars:<field>` (raw scalar).

### Surface / Iso / Slice — UsdGeomMesh

```usda
    def Xform "Geometry"
    {
        def Mesh "Surface"
        {
            point3f[] points = [ (0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0) ]
            int[] faceVertexCounts = [ 3, 3 ]
            int[] faceVertexIndices = [ 0, 1, 2, 0, 2, 3 ]
            uniform token subdivisionScheme = "none"
            rel material:binding = </Session/Looks/FieldMat>
            float[] primvars:Pressure = [ 0.1, 0.4, 0.9, 0.5 ] (interpolation = "vertex")
            color3f[] primvars:displayColor = [ (0.27, 0, 0.33), (0.16, 0.47, 0.56), (0.99, 0.9, 0.14), (0.13, 0.57, 0.55) ] (interpolation = "vertex")
        }
    }
```

`subdivisionScheme = "none"` renders the authored triangles/polygons directly
(no Catmull-Clark). `faceVertexCounts` is one count per face (3 per triangle);
`faceVertexIndices` indexes into `points`. Use vertex interpolation for per-vertex values
`utils.py`: NODE-associated fields are `vertex`, ELEMENT/cell fields are
`uniform`.

### Point Cloud — UsdGeomPoints

Use when the dataset has points and fields but no cells. `widths` is per-point
(world units); estimate it from the bbox diagonal over `sqrt(N)` when the
operator does not supply one.

```usda
        def Points "Surface"
        {
            point3f[] points = [ (0, 0, 0), (1, 0, 0), (1, 1, 0) ]
            float[] widths = [ 0.02, 0.02, 0.02 ]
            rel material:binding = </Session/Looks/FieldMat>
            float[] primvars:Pressure = [ 0.1, 0.4, 0.9 ] (interpolation = "vertex")
            color3f[] primvars:displayColor = [ (0.27, 0, 0.33), (0.16, 0.47, 0.56), (0.99, 0.9, 0.14) ] (interpolation = "vertex")
        }
```

### Streamlines — UsdGeomBasisCurves

`type = "linear"` gives polyline curves. `curveVertexCounts` is the vertex count
per curve; `points` is the concatenation of all curve vertices in order.
`widths` is per-vertex here (`interpolation = "vertex"`).

```usda
        def BasisCurves "Streamlines"
        {
            uniform token type = "linear"
            int[] curveVertexCounts = [ 3, 2 ]
            point3f[] points = [ (0,0,0), (1,0,0), (2,0,0), (0,1,0), (1,1,0) ]
            float[] widths = [ 0.02, 0.02, 0.02, 0.02, 0.02 ] (interpolation = "vertex")
            rel material:binding = </Session/Looks/FieldMat>
            float[] primvars:Pressure = [ 0.1, 0.4, 0.9, 0.2, 0.6 ] (interpolation = "vertex")
            color3f[] primvars:displayColor = [ (0.27,0,0.33), (0.16,0.47,0.56), (0.99,0.9,0.14), (0.28,0.14,0.46), (0.13,0.57,0.55) ] (interpolation = "vertex")
        }
```

## Field → Color: Bake A Colormap LUT

The simplest guaranteed-to-render color path is to **bake the colormap into
per-vertex color** and let a `UsdPreviewSurface` read it. Two things are
authored per prim:

1. `color3f[] primvars:displayColor (interpolation = "vertex")` — the baked RGB.
2. `float[] primvars:<field> (interpolation = "vertex")` — the raw scalar, kept
   so downstream tools (legends, thresholds, re-coloring, probing) still have
   the physical values.

Implement the colormap in the generated app. Store a small RGB control-point table for each
supported name, interpolate it to a 256-entry LUT, then normalize and index it:

```python
import numpy as np

def apply_lut(values, vmin, vmax, lut):
    values = np.asarray(values, dtype=np.float32)
    if not np.isfinite(vmax) or not np.isfinite(vmin) or vmax <= vmin:
        return np.full((values.size, 3), 0.5, dtype=np.float32)
    unit = np.nan_to_num((values - vmin) / (vmax - vmin), nan=0.0, posinf=1.0, neginf=0.0)
    index = np.rint(np.clip(unit, 0.0, 1.0) * (len(lut) - 1)).astype(np.intp)
    return np.asarray(lut, dtype=np.float32)[index, :3]

# `lut` is a generated `(256, 3)` float32 table for the selected named colormap.
colors = apply_lut(surface.field_values, *surface.field_range, lut)
```

Use `viridis` as the default and document any additional maps that the app ships. Preserve raw
field values alongside the RGB result so legends, thresholds, and recoloring remain possible.

The `(N,3)` `colors` array becomes the `primvars:displayColor` values and the
raw `surface.field_values` becomes `primvars:<field>`.

## Material And Lights

### UsdPreviewSurface Reading displayColor

Author one `Material` under `Looks` whose surface shader's diffuse input is
**connected** to a `UsdPrimvarReader_float3` that reads the `displayColor`
primvar. Bind it on each geometry prim with `rel material:binding`.

```usda
    def Scope "Looks"
    {
        def Material "FieldMat"
        {
            token outputs:surface.connect = </Session/Looks/FieldMat/Surface.outputs:surface>
            def Shader "Surface"
            {
                uniform token info:id = "UsdPreviewSurface"
                color3f inputs:diffuseColor.connect = </Session/Looks/FieldMat/PvReader.outputs:result>
                color3f inputs:emissiveColor = (0, 0, 0)
                float inputs:roughness = 0.65
                float inputs:metallic = 0
                token outputs:surface
            }
            def Shader "PvReader"
            {
                uniform token info:id = "UsdPrimvarReader_float3"
                string inputs:varname = "displayColor"
                float3 inputs:fallback = (0.55, 0.55, 0.55)
                float3 outputs:result
            }
        }
    }
```

The example authoring code also drives `emissiveColor` at low weight from the same
reader as insurance so the colormap stays visible under weak lighting, while
diffuse still gives lit shading form; plain `emissiveColor = (0,0,0)` with real
lights is the baseline shown above.

### Lights — CRITICAL GOTCHA

**A path tracer renders unlit diffuse geometry BLACK.** A `UsdPreviewSurface`
with only a diffuse color contributes nothing until light hits it, so a scene
with no light produces a black frame that looks like a broken render. You MUST
author at least one light. Use a `DomeLight` (soft fill so the whole surface is
lit) plus a `DistantLight` (key, for shading form):

```usda
    def Scope "Lights"
    {
        def DomeLight "Dome"
        {
            float inputs:intensity = 550
            float inputs:exposure = 0
            color3f inputs:color = (1, 1, 1)
        }
        def DistantLight "Key"
        {
            float inputs:intensity = 3000
            float inputs:angle = 1.5
            double3 xformOp:rotateXYZ = (-45, 25, 0)
            uniform token[] xformOpOrder = ["xformOp:rotateXYZ"]
        }
    }
```

If a rendered CAE frame is black despite valid geometry and colors, check for a
missing light before anything else.

## Beauty-Grade Fidelity: MDL Materials + HDRI Lighting

The baked-`displayColor` + `UsdPreviewSurface` path above is the right choice for
**the CAE scalar field itself** — the streamlines/iso/slice/point-cloud geometry
whose color *is* the data. For those prims the color must map the field faithfully
and must **not** be reshaped by a physically-based lighting response, so keep them
on the baked path.

A realtime CAE viewer often also carries **non-data "beauty" geometry** in the same
frame: the car body under the streamlines, a wind-tunnel floor, chrome trim, a glass
canopy. Those surfaces should look physically real — reflect the environment, show
clearcoat and metallic flake — not read as a flat lit color. `ovrtx` has full
USD/MDL support and bundles a large MDL material library, so bind a real **MDL
material** to that geometry while the scientific field stays on the colormap path.
The two coexist in one stage: bake `displayColor` for the field, bind MDL for the
props. This section is the fidelity upgrade layered on top of the baseline above.

### MDL Materials Via UsdShade

MDL resolution depends on `OVRTX_BIN_PATH` pointing at `ovrtx/bin` (the bundled MDL
library lives under `ovrtx/bin/library/mdl/`). **Without it, MDL surfaces render
magenta** — this is the single most common MDL failure. See the MDL resolution and
magenta-material rows in `references/ovrtx-rendering/README.md`; that document owns
the `OVRTX_BIN_PATH` / `LD_LIBRARY_PATH` / import-order rules and this reference does
not repeat them.

An MDL material is authored with `UsdShade`, but its wiring differs from the
`UsdPreviewSurface` block above: the surface output is `outputs:mdl:surface` (not
`outputs:surface`), the shader's `info:implementationSource` is `"sourceAsset"`, the
`.mdl` module is named by `info:mdl:sourceAsset`, and the specific material inside
that module is selected by `info:mdl:sourceAsset:subIdentifier`:

```usda
    def Material "CarMat"
    {
        token outputs:mdl:surface.connect = </Session/Looks/CarMat/S.outputs:out>
        def Shader "S"
        {
            uniform token info:implementationSource = "sourceAsset"
            uniform asset info:mdl:sourceAsset = @OmniSurfacePresets.mdl@
            uniform token info:mdl:sourceAsset:subIdentifier = "OmniSurface_CarPaintMetallic"
            token outputs:out
        }
    }
```

Useful bundled modules and their `subIdentifier` materials (any MDL in the library
works — these are convenient beauty presets):

| `.mdl` module | `subIdentifier` (material) | Use |
|---|---|---|
| `OmniSurfacePresets.mdl` | `OmniSurface_CarPaint` | flat automotive paint |
| `OmniSurfacePresets.mdl` | `OmniSurface_CarPaintMetallic` | metallic-flake car paint |
| `OmniSurfacePresets.mdl` | `OmniSurface_TwoToneCarPaint` | flip/two-tone paint |
| `OmniSurfacePresets.mdl` | `OmniSurface_GlossyPaint` | glossy non-metallic paint |
| `OmniSurfacePresets.mdl` | `Chrome` | polished chrome trim |
| `OmniPBR_ClearCoat.mdl` | (module material) | clearcoat over a PBR base |
| `OmniSurface.mdl` / `OmniPBR.mdl` | (general) | general PBR body / props |
| `OmniGlass.mdl` | (general) | glass canopy / transparent parts |

Bind the material with `MaterialBindingAPI` + `rel material:binding`, exactly like
the field material. The key rule when mixing bound and inherited materials:

> **A `material:binding` authored *on a prim* overrides a binding inherited from an
> ancestor Xform or reference container.**

So the streamlines/field prims inherit `FieldMat` from the `/Session/Geometry`
container (no binding of their own), while the car mesh carries its **own**
`rel material:binding = </Session/Looks/CarMat>` and overrides that inherited
colormap material for just that prim:

```usda
        def Mesh "Car"
        {
            # ... points / faceVertexCounts / faceVertexIndices / normals ...
            uniform token subdivisionScheme = "none"
            rel material:binding = </Session/Looks/CarMat>
        }
```

**When to use which:** MDL for beauty surfaces (car body, floor, chrome, glass) that
should reflect the environment and read as physical materials; the baked
`displayColor` colormap for the CAE scalar field (streamlines/iso/slice/points),
where the color encodes the data and must survive unchanged.

### HDRI DomeLight For Real Reflections

A plain `color3f` DomeLight lights the scene, but it gives a metallic or clearcoat
surface **nothing to reflect** — the car paint looks flat and dead, defeating the
MDL upgrade. Load an **equirectangular HDRI** into the DomeLight so reflective MDL
materials mirror a real environment. For metallic/clearcoat/glass this is the single
**biggest fidelity jump** in the whole scene — without it the reflective lobe has no
content:

```usda
        def DomeLight "Env"
        {
            float inputs:intensity = 900
            asset inputs:texture:file = @environment.hdr@  # app-supplied asset URI
            token inputs:texture:format = "latlong"
            double3 xformOp:rotateXYZ = (-90, 0, 0)
            uniform token[] xformOpOrder = ["xformOp:rotateXYZ"]
        }
```

Use an application-supplied or user-selected equirectangular `.hdr`/`.exr` asset URI.
Do not rely on a package-layout-dependent bundled texture path.

**CRITICAL — orient the HDRI to the scene's up axis.** A latlong (equirectangular)
HDRI wraps around a specific up axis; loaded raw it can arrive rolled 90°, so
reflections and the horizon come from the wrong direction (e.g. the "ground" of the
environment appears overhead). The DomeLight needs a rotation to **level the horizon
for the scene's up axis**. For the **Y-up** viewer scene used throughout this
reference, `double3 xformOp:rotateXYZ = (-90, 0, 0)` levels it (this is a suitable starting value for Y-up scenes). Match the rotation to
whichever up axis your geometry uses — see `coordinate-systems-and-up-axis.md` for the
Z-up vs Y-up handling; the dome rotation must track the same up-axis choice
as the geometry, or the reflected world and the model disagree.

### Smooth Normals For Imported Meshes

Imported CAE/prop surfaces are usually **faceted** — an STL, or a vertex-clustered
decimation, ships triangles with no authored normals. The path tracer then shades
each triangle flat, and MDL car paint reads visibly **low-poly**: every facet edge
catches a different highlight. Author **computed area-weighted vertex normals** so
the faceted body shades as one smooth surface. Each vertex normal is the
(area-weighted) sum of its incident face normals, normalized, authored per vertex:

```usda
        def Mesh "Car"
        {
            point3f[] points = [ ... ]
            int[] faceVertexCounts = [ 3, 3, ... ]
            int[] faceVertexIndices = [ ... ]
            normal3f[] normals = [ (0, 1, 0), ... ] (interpolation = "vertex")
            uniform token subdivisionScheme = "none"
            rel material:binding = </Session/Looks/CarMat>
        }
```

Compute face normals with a cross product, accumulate them onto each vertex, then normalize. If you
also rotate the mesh between up axes (a Z-up STL into a Y-up scene), rotate the
normals with the **same** rotation — a pure rotation maps points and direction
vectors identically, so the smooth shading survives the reorientation.

### Curve Widths: Prefer Constant Interpolation For Live Geometry

The streamlines example earlier authors **per-vertex** widths
(`interpolation = "vertex"`), which is fine for a static curve set. But when the
streamlines are **updated live via the data plane** — an emitter move recomputes the
curves and the total point count changes — a per-vertex widths array goes stale the
instant the point count changes: its length no longer matches `points`, so the write
fails or renders wrong. Prefer **constant** interpolation (one width for the whole
prim) so a point-count change never invalidates the widths array:

```usda
        def BasisCurves "Streamlines"
        {
            uniform token type = "linear"
            int[] curveVertexCounts = [ ... ]
            point3f[] points = [ ... ]
            float[] widths = [ 0.02 ] (interpolation = "constant")
            color3f[] primvars:displayColor = [ ... ] (interpolation = "vertex")
        }
```

`displayColor` stays per-vertex (it *is* the field, one color per point). Only
`widths` moves to constant. See the live data-plane array-write path in
`driving-cae-viz-via-ovstage.md`, which rewrites `points` /
`curveVertexCounts` / `primvars:displayColor` in place as the emitter moves — a
constant `widths` is what lets that fast path skip re-authoring the curve every move.

## USDA Authoring Gotchas

These are common USDA parser constraints:

- **Attribute metadata goes AFTER the value.** Interpolation is trailing
  metadata: `float[] primvars:x = [ ... ] (interpolation = "vertex")`, not
  before the attribute. Same for other primvar metadata.
- **Use multi-line brace bodies.** Compact one-line prim bodies can trip the
  parser. Author each prim with `{` and `}` on their own lines and one
  attribute per line. Long arrays wrap across lines (the example authoring code wraps
  ~6 elements per line for readability and parser safety).
- **`float2 clippingRange = (0.1, 100000)`** — author the camera clipping range
  as a `float2` tuple. A far plane large enough to contain CAE bounds avoids
  clipped geometry.
- **`matrix4d xformOp:transform` is row-major** with translation in row 3
  (`M[0]=right, M[1]=up, M[2]=-forward, M[3]=eye`), matching the viewer camera
  convention in `references/conventions.md`.
- **Numeric literals** should be finite; substitute `0.0` for any NaN/inf, and
  emit compact but valid floats (e.g. `%.6g`).

## Supported Colormap Boundary

This package supports the baked `displayColor` path above. Do not depend on applied schemas, dynamic texture services, or undocumented MDL LUT conventions for a generated app.
If the requested result requires a dynamic opacity transfer function or a runtime-managed texture
LUT, describe that as an unsupported extension rather than implying it is available here.

See also: `references/conventions.md`, `references/ovrtx-rendering/README.md`,
`references/ovstage-population/README.md`, `references/render-settings/README.md`,
`references/usd-sample-data/README.md`.
