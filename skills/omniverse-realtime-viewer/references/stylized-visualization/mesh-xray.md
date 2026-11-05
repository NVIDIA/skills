# Mesh X-Ray Delivery Profile

Use this focused reference when the requested X-Ray/technical look must work on
ordinary `UsdGeomMesh` geometry: tessellated props, support slabs, fixtures, or
converted CAD meshes. It complements the canonical presentation recipes; it
does not replace a BREP-aware adapter when one is available, and it introduces
no modeling-kernel dependency.

## Select The Geometry Provider First

Inventory the source before choosing a line implementation. If an exact
BREP-aware provider is present and the task requires its authored topology, use
that provider. If the stage is `UsdGeomMesh`-only or no BREP provider is
available, use this profile. For a hybrid stage, select a provider per prim and
compose their segments through the same camera/presentation contract.

Do not infer this from the user’s word “CAD”: converted CAD is often ordinary
mesh geometry. Record the selected provider and the count of included/excluded
meshes in the effective style capability response.

## What Mesh X-Ray Can And Cannot Show

`DepthSD` and the standard color/normal/ID AOVs describe the foremost rasterized
surface. They cannot reveal arbitrary layers behind it. Treat them as a
surface-shading and visible-edge signal, not as hidden-line data or a substitute
for depth peeling.

To reveal topology through regular meshes, extract deterministic mesh segments,
then rasterize those segments against a private current-camera depth test. A
light shell makes the scene read as X-Ray; the topology pass carries the
through-geometry information. One depth test can classify front/rear segments,
but cannot truthfully reconstruct every deeper overlapping layer. Keep rear
strokes quieter, or use a deliberate all-feature continuation convention; do
not imply physical transmission or multilayer depth peeling.

## Recommended Mesh Render Graph

```text
LdrColor + NormalSD + InstanceSegmentationSD (+ enhanced DepthSD)
    -> SPG shell and visible-guide node(s)
    -> SPG current-camera mesh projection/private-depth/stroke nodes
       -> visible graphite features
       -> subdued hidden/continuation graphite features
    -> published RGBA output -> existing delivery-format conversion
```

Validate guide semantics, then wire them into the same-frame SPG graph. Keep
the shell, topology raster and RGBA composite on GPU. Map only the processed
output for presentation. An explicitly retained legacy GL readback/upload path
needs separate capability and timing evidence; caching an old camera overlay
does not establish current-camera SPG delivery.

## AOV Capability Tiers

| Tier | Required guides | Behavior when available | Fallback |
|---|---|---|---|
| `mesh_xray_surface` | `LdrColor`, `NormalSD`, `InstanceSegmentationSD` | normal-shaped shell, normal/ID visible accents, mesh topology marks | color shell plus mesh marks, then `LdrColor` |
| `mesh_xray_enhanced` | surface tier + full-resolution float `DepthSD [H,W,1]` | relative local-depth edge accent and depth-aware cleanup | remove only depth-derived accent; keep surface tier live and report `missing_DepthSD` |
| `mesh_xray_gpu_continuation` | enhanced tier + SPG + mesh topology | same-device shell plus current-camera continuation raster | surface/enhanced tier or an explicitly validated legacy path |

The canonical X-Ray recipe does not require `DepthSD`. The enhanced mesh tier
does, because its local-depth edge formula consumes it. Never silently enable a
depth-derived edge from an unvalidated or mismatched tensor.

## Mesh Topology Rules

Start from the `LineGeometry` adapter in `line-compositor-delivery.md`.

- Include every render-visible mesh in the style scope, including passive
  tables, plates, supports, and occluders. Do not limit line extraction to
  selectable, animated, or currently inspected prims.
- Keep boundary and sufficiently noncoplanar crease edges; reject coplanar
  tessellation diagonals. Preserve silhouette candidates separately.
- Sort and cache topology by mesh/transform revision. Reproject and rasterize
  it from the newest applied camera pose whenever that pose changes.
- Use the exact OVRTX viewport, camera transform, projection, depth convention,
  and top-left image origin. A line layer that is even one camera update late
  reads as outline latency during orbit.
- Prefer a same-device topology pass for dense meshes. If a GL-to-host-to-CUDA
  fallback misses the current-camera refresh budget, report deferred lines
  rather than presenting stale lines as live X-Ray.

Visible and hidden strokes may differ in value or dash pattern, but keep them
in one color family. A useful noncanonical graphite family is visible RGB
`(0.215, 0.240, 0.267)` with a modestly lighter, lower-alpha hidden variant.
This preserves depth readability without a competing cyan line color.

### Accurate Depth Versus Fast Continuation

Use the private-depth line pass in `line-compositor-delivery.md` when a style
claims visible/hidden depth classification. It rasterizes occlusion triangles
and is the portable accuracy profile.

Use a same-device continuation profile when the product prioritizes responsive
technical readability over exact hidden-line classification. Keep persistent
GPU arrays for world vertices `[V,3]`, edge vertex indices `[E,2]`, adjacent
face indices `[E,2]`, face normals/centers, projected vertices, validity flags,
and two `4x4` camera matrices. Per frame:

1. Copy only the committed view/projection matrices into persistent GPU arrays.
2. Project vertices, rejecting nonfinite points and clipping segments before
   the perspective divide.
3. Classify an edge from adjacent face/view signs as silhouette, front feature,
   rear continuation, or reject.
4. Raster visible marks then hidden/continuation marks into the app-owned BGRA
   buffer with a deterministic priority rule: visible wins over hidden and
   either wins over the shell.
5. Bound samples by the clipped projected length and a configured maximum; log
   and skip any over-budget span instead of creating unbounded work.

This direct GPU profile is not a private-depth substitute. Label it
`continuation`, not `depth_classified_hidden`, and retain a per-style switch to
the accurate depth path when the distinction matters.

## Explicit Style Scope

Use a viewer-owned scope predicate, evaluated after inherited visibility and
purpose. Include `default`/`render` `UsdGeomMesh` prims and passive supports by
default. Exclude lights, cameras, helpers, guide geometry, and environment or
backdrop meshes only through named application configuration or inherited
opt-out metadata. Never exclude an object merely because it is not selected,
animated, or editable. Recompute the scope on stage/visibility/purpose changes
and report the resulting mesh/edge counts.

## Surface-Forward Shell

Do not turn geometry-only mode into a white or uniform-blue fill. That makes
real mesh geometry read as pure outlines and removes the form cues that make
the X-Ray useful. Suppress only the shell coverage of a large foreground
occluder when necessary; retain normal-shaped blue fill on every surface.

For the optional mesh surface-forward profile, let
`rim = smooth((1 - abs(Nz) - 0.08) / 0.72)` and
`facing = smooth((Nz + 1) * 0.5)`. A proven display-referred shell is:

```text
R = 0.39 + 0.17*facing + 0.05*rim
G = 0.63 + 0.22*facing + 0.06*rim
B = 0.76 + 0.19*facing + 0.07*rim
```

Composite visible guide edges and explicit topology only after this shell. The
formula is a mesh delivery profile, not a change to the canonical X-Ray palette
in `style-recipes.md`.

## Guide Cleanup

Use ID boundaries, normal creases, and relative local depth discontinuity only
as visible-surface accents. Suppress guide pixels with ID `0`; do not promote
border-connected background or flat slab artifacts into object lines. Gate
fine depth contours by local normal/curvature evidence, and reject isolated
screen-space fragments that cannot be associated with a mesh feature or a
coherent silhouette. This prevents disconnected dark marks from becoming more
prominent than the topology drawing.

Do not infer hidden topology from a depth contour. When a contour is needed as
a stylistic accent, use it at a lower weight than the mesh line pass and test it
under camera motion.

## Interaction, Cost, And Proof

Measure an X-Ray-off and X-Ray-on orbit separately. Report server frame time,
stream submission, browser decode/presentation, and the current-camera line
refresh rate. Attribute cost to guide mapping, shell, topology raster, and any
host transfer independently; a fast renderer with a stale line layer is not a
real-time X-Ray result.

Capture these mesh-specific proofs in addition to the standard style matrix:

1. An above-support three-quarter frame showing shaded cube/sphere/cone-like
   forms, not only outlines.
2. A lateral orbit that begins from the fitted above-support camera.
3. A below/behind-occluder frame where a passive support edge remains visible
   through the foreground geometry.
4. A fixed-camera comparison with the shell enabled and with the
   surface-forward profile enabled, confirming that the latter retains form.

Reset or fit the camera explicitly before capturing the above-support orbit.
Reconnecting a streaming client does not guarantee a fresh camera state. Record
the camera generation used by the shell and topology pass; they must match for
every frame in the proof.

Label this result `presentation_xray_mesh`. It is a technical-illustration
composite over regular mesh geometry, not physical transparency and not a
claim of arbitrary-layer visibility.

## Full wireframe and feature cleanup

Keep smooth surfaces free of accidental tessellation spokes in feature mode.
Before adjacency counting, remove faces collapsed by welding, and suppress
duplicate-index smooth seams and numerical micro-segments. A nonmanifold
condition must be reported rather than assumed to express design intent.

Offer full triangle wireframe only as an explicit diagnostic mode: include all
accepted triangle edges and diagonals, declare whether rear edges remain visible,
and keep its optional no-fill background distinct from the X-Ray shell. Verify
coverage numerically and ensure line disable removes marks. See the SPG
resource and geometry contracts in `line-compositor-delivery.md`.
