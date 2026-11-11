# Rectilinear projection and the actual depth image

Use this reference to implement per-eye math, not to infer a warped image's
mapping. [Foveation prerequisites](foveation-and-edge-math.md) gate that route.
All equations below operate in eye metres, with +X right, +Y up and forward -Z.
First apply the measured LOCAL-to-eye rigid transform at the projection time.

## Resolve the depth encoding before drawing

The chained depth subimage identifies the attachment and declared range. It
does **not by itself** prove the pixel encoding for every producer. In the
supplied OVXR header, the `depth_near_z`/`depth_far_z` comment describes XRDepth
as reversed 32-bit float while requiring physical near < far. Inferring LESS
comparison from that ordering would be wrong for this path.

Keep a validated producer contract alongside every attachment:

```text
DepthEncoding { normal | reversed, physicalNear, physicalFar,
                minDepth, maxDepth, format, producerCapability }
```

Check the installed header and the actual native attachment path; verify known
near/far occluders against it. Use GREATER_OR_EQUAL for this verified reversed
OVXR path, not a universal override for other runtimes or re-encoded images.
Do not apply the client's compressed/streamed depth encoding to the server image.
Unknown or contradictory encoding means skip the spatial draw and report why.

The finite-plane implementation below requires 0 < n < f and
0 <= minDepth < maxDepth <= 1. Reject NaNs, unsupported formats/sample counts
and degenerate ranges. Infinite-far support requires a separately implemented
limit and tests; it must not accidentally pass the finite formula.

## Projection and clipping

Let l/r/b/t be tangents of the eye's left/right/down/up FOV angles. Require
finite values, r > l and t > b. For eye position (x,y,z), set d=-z. Compute clip
coordinates without first dividing by d, so near-plane-crossing polygons can
be clipped correctly:

```text
A = f/(f-n); B = -f*n/(f-n)
if encoding == reversed: A = 1-A; B = -B
zA = minDepth + (maxDepth-minDepth)*A
zB = (maxDepth-minDepth)*B

clipX = (2*x - (r+l)*d)/(r-l)
clipY = (2*y - (t+b)*d)/(t-b)
clipZ = zA*d + zB
clipW = d
```

For each triangle/polygon, clip against left/right/top/bottom homogeneous
planes (-w <= x,y <= w), physical n <= d <= f and positive w. Sutherland-Hodgman
clipping uses a signed plane distance L(v): for a crossing edge a→b, interpolate
at `s=L(a)/(L(a)-L(b))`. Interpolate clip position and vertex attributes with
the same s, then triangulate the surviving polygon. Do not drop a whole panel
merely because one corner crosses the near plane. Retain perspective-correct
GPU interpolation of UV/color; do not pre-divide UV as if it were screen-linear.

Use the formulas' depth directly with a Vulkan viewport depth range [0,1];
do not remap minDepth/maxDepth a second time. Normal maps n→minDepth and
f→maxDepth; reversed maps n→maxDepth and f→minDepth. The compare operation
must agree with the same encoding used to build clipZ.

## Subimage, orientation and scissor

For subimage offset (ox,oy), size (W,H), and divided NDC (X,Y):

```text
pixelX = ox + (X+1)*W/2
pixelY = oy + (1-Y)*H/2
```

Choose exactly one Vulkan Y inversion: negating clipY in the vertex shader
with a positive-height viewport is one portable policy. Do not also invert
the viewport or UVs unless their independent origin convention requires it.
Use each eye's own pose/FOV/subimage/array layer. Clip, then floor minima and
ceil maxima for conservative pixel bounds, expand for any AA carrier fringe,
intersect with the submitted subimage, and union all drawable surfaces.
A toolbar or gizmo outside a panel-only scissor must not disappear.

## Color, coverage and submitted depth

For premultiplied source RGB Cs and alpha a, use
`Cout = Cs + Cd*(1-a)` and `aout = a + ad*(1-a)` with matching Vulkan factors
ONE and ONE_MINUS_SRC_ALPHA for color and alpha. Straight-alpha inputs must
be converted once, not multiplied twice. Define texture/color-space conversion
explicitly; blending encoded sRGB as linear produces different edges.

LOAD/STORE the existing attachments. Zero-coverage fragments must not write
depth. Declare the covered-fragment depth-write policy; one flattened primary
depth cannot encode translucent UI and the scene behind it independently.
Foreground gizmos use the separate private-depth policy in
[depth tools](depth-tools.md), not a clear of the scene depth.

## Numerical and visual gates

Implement CPU fixtures in the generated application using the equations above.
Assert both normal and reversed near/far endpoints, a known intermediate depth,
asymmetric FOV edges, nonzero subimage offsets, near-plane clip intersections,
and opposite near/far occlusion inequalities for the two depth encodings.
Reject non-finite inputs, `near <= 0`, `far <= near`, degenerate FOV ranges,
and invalid depth ranges before perspective division.

Use n=0.1, f=100, min=0, max=1: normal depth at d=1 is about 0.9009009;
reversed is about 0.0990991. Endpoints must map as above. These are numeric
fixtures, not suggested camera defaults. Test asymmetric tangents, a nonzero
subimage origin, unequal eye sizes, triangles crossing each clip plane, and
normal/reversed occlusion against scene geometry. A passing formula test does
not prove that the selected runtime actually emits that encoding.
