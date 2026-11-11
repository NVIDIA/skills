# SPG geometry, depth and stroke delivery

Use this contract for any style that promises explicit features, silhouettes,
dashes/dots, hidden strokes, Solid stock fill or full wireframe. Read
[SPG processing](../spg-processing/README.md) before authoring nodes.

## Path decision

The primary path is a complete SPG graph: geometry upload -> projection ->
private mesh depth -> feature/silhouette projection -> stroke composite ->
processed AOV. This feeds WebRTC, SHM, ovui or native presentation through their
existing output boundary. No GL context or host framebuffer readback is needed
for this custom style path.

A native C++ application may retain its validated
[GL overlay](../gl-viewport-overlay/README.md) for UI/gizmos or an explicit
legacy line adapter. An existing ModernGL readback/upload renderer is useful
as a migration oracle or separately measured compatibility implementation;
it is not the default SPG architecture or an excuse to report its pixels as
SPG. Do not assume a native OVRTX feature-line API exists without public evidence.

## Accepted geometry and scope

The geometry provider supplies owned occlusion triangles `[T,3,3]`, feature
segments `[E,2,3]`, silhouette candidates `[S,2,3]` with adjacent normals, stable
edge IDs/pattern origins, and geometry/transform revisions. Record matrix layout
and whether coordinates are object or world space. Transfer arrays and files
across workers, never borrowed SDK handles or stage objects.

The supported mesh profile is resolved finite triangles, subdivision `none`,
render-visible `default`/`render` purpose and current transforms. Deterministically
resolve polygons, holes, subdivision and instances before extraction, or exclude
and report unsupported data. Do not silently promise deformations or prototype
expansion when the adapter handles only static meshes.

Cache topology in deterministic prim/face/edge order. Canonical geometric
thresholds remain a weld step of `max(bounds_diagonal*1e-6,1e-7)`, a `30 degree`
crease threshold and coplanar rejection at `cos(1 degree)`, unless a deliberately
named source-compatible profile specifies others. Record the effective profile;
do not change thresholds invisibly during migration.

After welding, reject collapsed faces before building edge adjacency. Keep
valid boundaries and sufficiently sharp two-face creases; suppress coplanar
internal edges and duplicate-index seams on smooth geometry. Exactly two
noncoplanar faces provide a silhouette candidate. More than two accepted owners
is a reported nonmanifold condition, not an automatic design feature. Authored
features may explicitly override a mesh inference policy.

A separate full-wireframe mode projects all triangle edges regardless of crease
classification. Include diagonals and smooth regions consistently. It may derive
segments directly from triangles in its own node rather than add a large edge
array to every line node. Preserve shared-edge overlap semantics.

## Resource and node contracts

A practical graph separates:

1. Triangle clip/projection, emitting explicit vertex count and pixel bounds.
2. Spatial bins and private nearest mesh depth.
3. Feature/silhouette projection, emitting bounds, projected endpoints/depth,
   width, pattern, color, stable phase and validity.
4. Stroke bins and RGBA premultiplied composition over the style image.

An image output can use a texture/surface while projection, bins and depth use
buffers. Bind each backing according to the installed SPG contract; a published
buffer's byte view is not an automatically typed image. Allocate large outputs
on GPU and bind geometry without constructing full-image Lua tables.

For each kernel, verify the whole domain, final partial group and output stride.
Image-derived launch dimensions may be inappropriate for buffer rows. Verify a
known projected point and every matrix lane before treating the output as valid.

Spatial bins use conservative bounds that include antialiasing and maximum
visible/hidden width. A capacity limit must have a correctness-preserving
spill/full-scan strategy or reject the frame explicitly. Never silently drop
edges or triangles. Report overflow frequency and benchmark dense views. Preserve
source feature-before-outline ordering when the composite depends on it.

## Camera, clipping and depth

Use the exact committed OVRTX view/projection, focal/aperture model, viewport,
near/far planes and image origin. Distinguish row-vector object transforms from
column-vector clip projection and validate with independent endpoint calculations.
Clip triangles and segments before homogeneous division. Transform positions
and silhouette-facing data consistently under object edits; nonuniform or
mirrored transforms need a declared tested normal policy.

Clear private depth to the far value and rasterize the accepted occlusion mesh.
For conventional normalized depth, visible strokes use `<=` and hidden strokes
use `>`; adapt the convention rather than guessing for reversed-Z. Match the
source depth-bias behavior intentionally and test it near silhouettes and holes.
A custom analytic rasterizer is not automatically pixel-identical to hardware GL.

The foreground ID/depth guides alone are not hidden topology. Include accepted
passive supports/occluders even when they cannot be picked or animated.

## Stroke and compositing contract

Preserve the widths, colors, patterns and cadence from [style recipes](style-recipes.md).
Measure along/across distance in screen pixels after clipping; perspective must
not distort stroke width or period. Use round caps and a one-pixel feather, with
coverage `1-smoothstep(0,1,distance-half_width)`.

Canonical independent segments retain their stable world-endpoint seed and
per-segment phase. A continuous sampled curve may carry accumulated arc length,
but that is a named extension when it changes the canonical pixels. Advance
animated phase from monotonic time; Blueprint's decorative update is `12 Hz`,
while camera/geometry projection refreshes immediately. Silhouettes stay solid.

Compose in this order: style image, quiet hidden feature strokes for Technical/
X-Ray, visible features, visible solid silhouettes. Use the source premultiplied
RGBA convention and overlap rule. Transparent strokes use componentwise MAX;
Solid first produces stock RGB `(0.43,0.72,0.78)` and then uses ordered alpha
over for dark ink. Preserve image alpha through the final output. Wireframe can
use a uniform background with no surface fill and an explicit faint rear tier.

Keep source color and the final SPG output separate. Convert RGBA to BGRA only
at a delivery boundary that requires it; do not blend RGBA marks as if their
channels were BGRA. Native GPU fences and output mappings remain owned by the
selected presentation path.

## Revisions, fallback and acceptance

The runtime owner commits parameters and geometry before stepping. Invalidate
projection/depth/marks on camera, accepted geometry, transforms, viewport and
relevant style changes. Topology does not need rebuilding for a camera move.
Descriptor caching alone does not prove cached uploads or skipped GPU work.

If this line graph is unavailable or over budget, choose a declared image-only
or compatible legacy result and report the missing tier. Never present stale
camera marks as current SPG hidden lines. Benchmark base, image-only and full
geometry modes independently before making a real-time claim.

Acceptance includes independent triangle/edge projection, bin membership,
overflow stress, private depth, visible/hidden edge samples, deterministic
width/pattern/alpha, preserved real creases, clean smooth tips, whole-mesh
wireframe, line disable, scene/resolution rebuild and live camera alignment.
Compare prior GL strokes on matched geometry, camera and base image, and report
localized as well as mean differences separately from image-equation parity.
