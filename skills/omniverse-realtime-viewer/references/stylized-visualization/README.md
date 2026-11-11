# Stylized visualization with SPG

Use this reference for toon, ink, clay, blueprint, halftone, hatch, rim,
technical illustration, presentation X-Ray, feature/silhouette strokes and
explicit wireframe modes. Build their image and geometry processing on OVRTX
Sensor Processing Graphs. The UI selects parameters and displays the resulting
pixels; it does not render the USD scene.

## Build from this package

For a new implementation, follow [the implementation sequence](implementation.md).
It provides a minimal USD/Lua/CUDA node, resource schemas, projection/depth and
stroke algorithms, composition equations and runtime control flow. Use
`style-recipes.md` for the complete preset equations. No existing sample or
previous shader implementation is a prerequisite.

## Read order

1. [SPG processing](../spg-processing/README.md): owning SDK documentation,
   node interfaces, resources, publication and presentation boundaries.
2. [Style recipes](style-recipes.md): preserve the 14 named presets, numerical
   defaults, stroke language and intentional fallback identities.
3. [AOV guide contracts](aov-guide-contracts.md): accepted color, normal and ID
   semantics, capability probes and missing-guide behavior.
4. [Geometry and compositor delivery](line-compositor-delivery.md): complete
   SPG depth, feature/silhouette projection, stroke and composite contracts.
5. [Reproduction validation](reproduction-validation.md): numerical, temporal,
   current-camera and real-time evidence; genre resemblance is insufficient.

For ordinary tessellated USD X-Ray, add [mesh X-Ray](mesh-xray.md). For a new
look without a named recipe, add [style composition](style-composition-contract.md).
Read the selected viewer delivery recipe and [AOV switching](../aov-switching/README.md)
when the app exposes source/processed output selection.

For exact CAD, first resolve [dependencies](../dependencies/README.md) and
upstream USD BRep guidance, then read [BREP source adaptation](usd-brep-source.md).
BREP is optional: ordinary USD Mesh sources do not need a modeling kernel.

## The complete processing graph

```text
USD + committed camera/material/light state -> OVRTX RenderProduct
    color + verified normal/ID guides -> immutable image pass(es)
    accepted geometry -> clip/project triangles -> bounded bins -> private depth
    feature/silhouette data -> project strokes -> bounded bins
    image + depth + strokes -> premultiplied composite -> processed RGBA AOV
    -> existing viewport/encoder conversion and delivery
```

SPG owns the custom GPU processing, including geometry depth and stroke
composition when those layers are required. The normal implementation does not
map every guide into Python, run a Warp presentation chain, or round-trip a GL
framebuffer through the host. Warp remains useful for validation or the existing
transport conversion; it is not the primary style graph. A separately benchmarked
legacy compositor may remain an explicit compatibility path, not the SPG claim.

Preserve source `LdrColor`. A single processed RGBA8 output such as `LdrStylized`
can serve several style aliases through a mode parameter; aliases are app
metadata, not native AOVs or fourteen simultaneous image allocations. Validate
that the selected alias maps the processed resource at the actual delivery
boundary. Publish intermediate outputs only for a real consumer or probe.

## Keep the existing application owner

Author viewer state and graph wiring in the session/composite layer. Source USD
and existing stage lights remain authoritative; a viewer lighting rig is opt-in.
Materials and lighting are not migrated into image equations.

The render owner drains commands, commits the camera/transforms/style values,
steps the renderer, maps the final processed output, converts format if needed,
and hands owned pixels to the existing presentation path. Keep established
OVStage or legacy stage integration and synchronization intact. Callbacks enqueue
changes only. Do not substitute a new scene loader or renderer thread merely to
follow an SDK example.

On scene changes, rebuild accepted geometry, reject stale worker results and
invalidate parameter/resource capabilities. On resolution changes, rebuild
resolution-dependent resources and revalidate output shape. Camera-dependent
projection/depth/strokes must use the pose committed for that same frame.

## Image passes and preset policy

Make preset identity, guide requirements, numerical defaults, fallback and
stroke policy server-owned data. Keep bounded user adjustments separate.
Return requested/effective mode and degradation reasons; client state alone
cannot establish a running style.

Implement the recipe equations by named channels. The source recipes may describe
BGRA transport bytes; native color textures are RGBA. Retain original palettes,
thresholds and rounding while moving processing before the one transport swap.
Preserve alpha. Normal and ID inputs must meet their observed guide contract.

Neighborhood passes read immutable inputs and write separate outputs. A multistage
Rim or Clay effect needs explicit internal intermediates where its source has
scratch boundaries. Use stable screen/object/world pattern coordinates, nearest
ID reads and clamped image borders. Do not introduce frame-random marks. Animated
Blueprint phase follows monotonic time with its specified decorative cadence;
projection stays current on every applied camera change.

For reproduction, compare the previous equations to the SPG node on identical
raw inputs before comparing screenshots. Floating-point contraction can change
thresholded patterns; match the specified arithmetic and rounding when exact
byte parity is required. Choosing a new palette or stronger filter is a new
look, not a silent migration correction.

## Features, silhouettes and full wireframe

A feature line is a genuine boundary, authored design feature or selected sharp
crease. Duplicate vertex indices and tessellation diagonals on smooth surfaces
are not sufficient evidence. Weld carefully, discard triangles collapsed by
welding before counting adjacency, and reject numerical micro-segments. Report
nonmanifold topology instead of automatically turning it into decorative spokes.

Keep view-dependent silhouettes separate and solid. Include passive supports
and occluders within the declared style scope; selection/animation membership
must not determine which meshes receive the style.

An explicit Wireframe mode is separate from the canonical 14 style recipes. It
intentionally includes every accepted triangle edge, including smooth-surface
edges and triangulation diagonals. Declare visible-only versus faint rear edges
and whether surface fill is present. Full wireframe must not be a partial set
of topology accidents. Deriving edges from existing triangles in a separate
SPG node can avoid duplicating a large launch-script input. Duplicate edge
emissions need an overlap rule that does not darken shared edges twice.

## Technical and X-Ray presentation

The image pass creates the recipe's shell/form treatment. Rear topology comes
from the geometry pass; foreground color, normals, IDs and one depth AOV cannot
reconstruct hidden surfaces. Project the accepted occlusion mesh into a private
depth buffer, classify visible/hidden feature strokes against that same camera,
and compose quiet rear marks before visible features and solid silhouettes.
Preserve Solid's opaque stock-fill behavior and the source stroke ordering.

This is a presentation X-Ray, not physical transmission or analytic BREP
rendering. A cheaper continuation mode may show through-lines without correct
visibility classification, but must have a distinct capability and label.

## Capabilities, performance and proof

Resolve graph/guide support against the installed release before exposing
controls. Missing guides or geometry choose an explicit recipe fallback;
missing processed output keeps the app live on color while reporting failure.
An image-only fallback must not be advertised as hidden-line X-Ray.

Cache topology by revision, but measure descriptor reuse, uploads and GPU work
separately. Bounded bins require a correctness-preserving overflow strategy;
dense overflow scans can dominate the frame. A same-device SPG graph is not
inherently a real-time graph. Record warm base/style costs, memory, unique
presented motion and camera latency. Follow the declared performance profile
before claiming full-style real time.

Verify each graph stage, then the final viewport/stream. A correct image pass
can conceal a projection node that ran only its first row. Preserve raw-image
parity and replacement-rasterizer comparisons as separate results. Test clean
smooth tips, real sharp edges, support occluders, holes, complete wireframe,
line disable, camera motion, stage changes and actual render resolution changes.
