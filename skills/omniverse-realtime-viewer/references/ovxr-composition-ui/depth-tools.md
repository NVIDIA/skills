# Depth-aware rays, teleport and foreground gizmos

Use the [primary compositor](primary-projection-compositor.md), not another USD
renderer. A retained panel and analytic tool geometry can share the same
per-eye projection, reference-space conversion, frame slots, scissor union,
alpha convention, and depth policies without sharing one raster texture.

## Geometry and hit authority

Convert ray endpoints and teleport polyline points once into compositor LOCAL
metres at the coherent input/display time. Carry visibility, capture generation,
radius, color, and depth policy with each bounded draw record. Skip degenerate
segments and reject nonfinite coordinates before native submission.

For volume, form a low-sided tube around each segment: normalize its tangent t,
choose a stable perpendicular n, set b = cross(t,n), and generate rings
`p + radius * (cos(angle)*n + sin(angle)*b)` at both endpoints. Join neighboring
ring vertices with indexed triangles. Use a fixed small side count and segment
budget, with reusable vectors/buffers; do not allocate a mesh or Vulkan pipeline
per segment. Parallel-transport the side basis along a curve to reduce flips.
Use analytic normals with a bounded light/gradient term for volume cues rather
than flat ribbons; this is tool shading, not a claim of scene PBR lighting.

Selection stops at the authoritative hit, and teleport markers/facing arrows
use the same accepted hit/yaw that navigation will commit. Moving visuals to the
compositor must not disable release-edge scene queries. A final asynchronous
result is accepted only if gesture, stage, anchor, sample time and age match;
no query, an attempted miss and a stale result are distinct states.

## Explicit ordering

The [worked pass sequence](compositor-passes-and-sync.md) supplies attachment
selection, barriers, resource retirement and the opaque/masked baseline.

Choose policy per surface. For the requested editor behavior:

1. Scene-depth tools test against original scene depth; opaque/masked panel
   surfaces use their declared scene-occlusion policy.
2. Selected-object gizmos overlay scene objects but retain self-occlusion.
   Use a separate bounded depth attachment for that batch, cleared only when
   used. Compare/write against that private depth, never clear scene depth.
3. Draw UI panels/chrome/handles above foreground gizmos where they overlap,
   retaining the UI's chosen relationship to scene depth. Use coverage-aware
   masks/order, not a blanket full-rectangle opaque quad.

Keep foreground gizmo depth separate from exported scene/UI depth so a remote
depth reprojection does not treat foreground handles as embedded in scene
objects. Define which covered fragments update submitted depth; transparent
fragments must not stamp an entire carrier triangle into depth. Test a tool
crossing the bottom grab bar as well as one crossing panel text and empty space.

Index ranges and scissor bounds must include every surviving surface. Count
actual draw calls, passes and uploads independently; a depth-policy split may
require multiple draws even inside one per-eye pass.

## Antialiasing without a full-eye quality change

Diagnose text, procedural icons, silhouettes and texture minification separately.
For rounded bars and line/icon strokes on a carrier surface, use signed distance
d in local units, screen derivative width `w = max(fwidth(d), epsilon)`, and
coverage `1 - smoothstep(-w/2, w/2, d)`. Pad carrier geometry so fragments exist
outside the nominal boundary; keep hit bounds at the intended semantic shape.
Premultiply RGB and alpha consistently by coverage and match the blend state.

Analytic shading inside a tube does not antialias its triangle silhouette. Use
a coverage fringe or expanded carrier with an analytic boundary, accounting
for depth at the covered surface. Avoid writing depth for zero coverage. Do
not claim a panel texture-resolution increase fixes analytic toolbar/gizmo
edges or that a fullscreen blur/MSAA change is free.

Measure both-eye edge quality at oblique angles and worst-case screen coverage.
Keep raster supersampling and bounded minification taps local to the affected
surface; never change headset scale/foveation to hide a tool-shader defect.
Only use sampler features enabled on the borrowed Vulkan device.

## Hot path

Pack dynamic records into a reusable contiguous float32 buffer with one native
call, not per-segment ctypes allocations. Reserve native mesh capacity for the
bounded worst case, then clear logical lengths without discarding capacity.
Remove incidental per-eye/per-segment scratch allocations after measuring them.
Compare geometry/indices against a scalar reference across clipping, subimages,
depth conventions and image domains before accepting a packing optimization.

Measure packing, mesh construction, native compose, queue submit, and GPU work
separately. Zero texture uploads does not mean zero geometry uploads. CPU mesh
work before the native compose timer must not disappear from the performance
report. No same-frame fence waits or full-eye copies belong in this path.
