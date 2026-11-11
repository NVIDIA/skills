# Implement a stylized SPG viewer

Use this build sequence with the selected viewer recipe. It needs ordinary USD
geometry, public OVRTX SPG support and the contracts in this package; no prior
viewer implementation or named sample checkout is required. The paths and node
names below are illustrative application-owned names, not SDK facilities.

## 1. Establish the input/output graph

Create a RenderProduct with native RGBA8 `LdrColor`. For guided styles add the
normal and instance-ID RenderVars from `aov-guide-contracts.md`. Keep the camera,
source RenderVars, shader nodes and processed output in the viewer-owned layer.

The smallest graph is one Shader and one separately published output:

```usda
# Within the application's RenderProduct:
def RenderVar "ColorInput"
{
    uniform string sourceName = "LdrColor"
    opaque omni:rtx:aov
}
def Shader "StyleImage"
{
    uniform token info:implementationSource = "sourceAsset"
    uniform asset info:spg:sourceAsset = @StyleImage.cu@
    uniform token info:spg:sourceAsset:subIdentifier = "style_image"
    opaque inputs:Color.connect = <../ColorInput.omni:rtx:aov>
    int inputs:Mode = 0
    opaque outputs:Result
}
def RenderVar "StyledOutput"
{
    uniform string sourceName = "LdrStylized"
    opaque omni:rtx:aov.connect = <../StyleImage.outputs:Result>
}
```

Declare `opaque omni:rtx:aov` on each source RenderVar consumed by an SPG
connection, as in the [official first-node example](https://nvidia-omniverse.github.io/ovrtx/spg/first_node.html).
A `sourceName` alone does not declare the port: a missing port can leave the
shader input unresolved and the processed output empty.

Add the source and final RenderVars to that product's `orderedVars` relationship.
Retain the product's existing camera and render settings. The surrounding USD
and source names must follow the installed OVRTX contract; relative paths above
resolve from each authored node. Bind `LdrStylized`, rather than `LdrColor`, at
the selected viewport/encoder handoff.

Create `StyleImage.cu.lua` beside the CUDA source:

```lua
function style_image(inputs, outputs)
    local source = inputs.Color
    assert(#source.shape == 2 and source.dtype == cuda.uchar4)
    local h, w = source.shape[1], source.shape[2]
    outputs.Result = cuda.image(w, h, cuda.uchar4)
    return cuda.kernel({
        args = {cuda.int(w), cuda.int(h), cuda.int(inputs.Mode),
                cuda.TextureObject(source), cuda.SurfaceObject(outputs.Result)},
        block = {16, 16}, grid = {math.ceil(w / 16), math.ceil(h / 16)}
    })
end
```

The matching initial CUDA node is an identity implementation:

```cuda
extern "C" __global__ void style_image(
    int width, int height, int mode,
    cudaTextureObject_t color, cudaSurfaceObject_t output)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    uchar4 rgba = tex2D<uchar4>(color, x, y);
    // Replace RGB with the selected recipe; retain rgba.w.
    surf2Dwrite(rgba, output, x * sizeof(uchar4), y);
}
```

Use the installed SPG compiler's CUDA environment, resource backing and launch
contract. If adding normal/ID resources, add matching USD inputs, Lua arguments
and kernel parameters in the same order. For texture-backed guides, bind texture
objects; for buffers, pass array pointers with explicit shapes/strides. A CUDA
kernel signature, Lua argument list and USD interface form one contract.

Before adding styles, run these three USD/Lua/CUDA identity blocks together.
Map source and processed output from the same render step; require both to be
nonempty RGBA uint8 arrays of shape `(height, width, 4)` with byte-for-byte
equality, including alpha. If SPG reports an unconnected input or an empty
output, check the source port declaration, connection path and `orderedVars`
before changing the shader equations.

## 2. Implement the image recipes

Assign explicit application mode IDs to `style-recipes.md`'s 14 preset names.
Do not assume that table order is an SDK enum. Implement a switch or separate
nodes whose RGB equations come directly from that reference:

- Studio leaves source color unchanged; Solid's stock replacement is in the
  geometry composite, not an invented material override.
- Toon, Blueprint, Halftone and Pixel use their per-pixel formulas.
- Clay, Hatch and Rim use immutable neighbor reads with clamped image borders.
- AOV Ink, Facets, Technical, Concept, X-Ray and guided Rim consume verified
  view normals and nonzero-ID surface masks.

Decode channels explicitly as R/G/B, normalize bytes only for equations that
expect `[0,1]`, and use the specified clamp/round-to-byte rule. Alpha passes
through unchanged. Add intermediate nodes where a later pass reads the result
of an earlier neighborhood pass; do not read partially overwritten pixels.

For X-Ray, implement the shell and normal/ID accents from the recipe. Those
accents cover visible surfaces only. Hidden geometry is a separate path below.
Resolve missing-guide variants through the recipe fallback names rather than
silently inventing zero-valued guides and calling the result the requested look.

## 3. Build a geometry provider

For each included USD Mesh, obtain accepted points, polygon topology and current
world transform through the app's scene owner or query worker. Resolve supported
polygons to triangles, including winding, before creating arrays. Use the same
accepted mesh for OVRTX presentation and line occlusion.

Retain source vertex/part identity for updates. Produce:

| Resource | Element contract |
| --- | --- |
| Triangles | three world-space XYZ positions and owning object index |
| Features | two endpoints, stable edge ID and owning object index |
| Silhouette candidates | endpoints, two adjacent normals and owning object index |
| Object state | base/current transform or a declared world-space delta, including normal-transform policy |

Weld by the declared scale-relative tolerance for feature classification.
Remove triangles collapsed by welding before counting adjacency. One valid face
owner yields a boundary; two yield a crease only when their normals exceed the
chosen threshold. Smooth duplicate-index seams and coplanar diagonals are not
features. Report higher adjacency separately. Keep noncoplanar two-face edges
as silhouette candidates even when also classified as a crease.

Wireframe uses every triangle's `(0,1)`, `(1,2)`, `(2,0)` pair, without crease
filtering. It may emit a shared edge twice if the composite's overlap rule is
idempotent. Derive those pairs in a separate GPU node from the triangle input
when duplicating a large CPU edge array would exceed the upload/script budget.

## 4. Project triangles and build private depth

Use a declared matrix convention. For column-vector projection:

```text
clip = projection * view * current_world_position
near_distance = clip.z + clip.w       # conventional -w..w clip depth
pixel_x = (clip.x / clip.w * 0.5 + 0.5) * width
pixel_y = (0.5 - clip.y / clip.w * 0.5) * height
depth = clip.z / clip.w * 0.5 + 0.5
```

For a USD row-vector transform, transpose/reorder appropriately before this
calculation. Clip triangle polygons and segments against the near plane before
perspective division. One near-clipped triangle may become a four-vertex polygon;
triangulate it deterministically for the private rasterizer.

A projected triangle record can hold vertex count, pixel-space bounding box and
up to four clip-space XYZW vertices. Keep its element stride explicit in Lua and
GPU code. Bin conservative pixel bounds into screen tiles. A useful starting
choice is a 16-pixel tile with a configurable index capacity; those values are
implementation parameters, not guaranteed SDK limits.

For each output pixel, inspect its tile's triangle candidates, evaluate projected
triangle barycentrics at the pixel center, reject samples outside the triangle,
and interpolate normalized endpoint depth. Write the minimum accepted depth,
or the far value if none. Add a documented slope/constant bias to avoid line
self-occlusion, with comparisons matched to the depth convention. A bin overflow
must scan all relevant triangles or spill safely; truncation produces false holes.

## 5. Project features, silhouettes and wire edges

For each silhouette candidate, classify its two adjacent faces against the
current camera in the same coordinate space. Opposite signs retain the visible
contour candidate; equal signs discard it. Transform normals correctly for the
supported object transforms.

Project and clip retained segments using the same camera as the triangle node.
Each record needs validity, expanded pixel bounds, endpoint pixel positions and
depths, silhouette/feature class, width, pattern, period, duty, phase and color.
Expand bounds by half-width plus feather; include hidden-stroke width as well.
Bin segments by those expanded bounds with a safe overflow path.

For a pixel P and projected endpoints A/B:

```text
axis = B - A; length = |axis|; tangent = axis / length
along = dot(P - A, tangent)
across = cross(tangent, P - A)
clamped_along = clamp(along, 0, length)
segment_distance = sqrt((along-clamped_along)^2 + across^2)
```

Reject zero-length segments. Interpolate depth under the selected clipped
screen-space convention; apply the same private-depth policy as the source
geometry. A sample at or in front of private depth is visible, and a deeper
sample is hidden. The stroke contract in `line-compositor-delivery.md` specifies
round caps, one-pixel feather, nearest dash/dot center and deterministic phase.
Implement those equations instead of drawing implementation-dependent wide lines.

## 6. Composite and publish

One pixel-owning kernel can consume image, private depth and the indexed segment
records without cross-pixel write races. Evaluate quiet hidden features first,
then visible features, then solid silhouettes. Preserve explicit ordering when
using ordered alpha blending. With transparent marks, accumulate the recipe's
componentwise MAX premultiplied layer, then combine once:

```text
result_rgb = line_premultiplied_rgb + image_rgb * (1 - line_alpha)
result_alpha = image_alpha
```

Solid first substitutes stock fill where private mesh coverage exists, then
uses ordered alpha-over for ink. An explicit no-fill wireframe begins with a
uniform background and draws all accepted edges; use a separate opacity for
rear edges if desired. It is not the normal X-Ray shell.

Publish the composite as the chosen processed RGBA AOV. Keep the original color
output selectable. Perform any BGRA conversion once in the existing delivery
adapter, and keep mapped/owned GPU resources alive until that consumer finishes.

## 7. Drive the graph from the application

The runtime owner processes commands before stepping. A bounded style command
carries preset identity, line enable, thickness and optional pattern/lighting
parameters; return the effective state to the client. Update camera/projection,
object transforms and style values in the graph using the installed stage-owner
attribute API. Scene/topology changes rebuild accepted geometry; camera motion
only updates projection/depth/strokes. Resolution changes rebind image resources
and invalidate published-output capabilities.

Start with brute-force candidate scans on a small scene if that makes the first
correct image simpler. Add bins, revision caches and bounded work once projection
and visibility are correct. Never equate cached descriptors with skipped uploads
or GPU work. Test identity, clean smooth features, real creases, complete wireframe,
hidden bores and live camera/scene changes using the package's generic checks.
