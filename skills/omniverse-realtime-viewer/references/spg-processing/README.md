# Sensor Processing Graphs for viewer outputs

Use this reference for SPG nodes, GPU post-processing, custom sampling, chained
AOV processing, and publishing processed pixels to a viewer. For toon, ink,
hatching, technical illustration, X-Ray or geometry strokes, continue to
[stylized visualization](../stylized-visualization/README.md). Ordinary raw AOV
selection follows [AOV switching](../aov-switching/README.md).

## Start with the owning SDK

Read the selected OVRTX dependency route in
[dependencies](../dependencies/README.md) and inspect the installed release's
SPG documentation and examples before selecting APIs. Acquisition stays in
that dependency reference. SPG is an OVRTX facility, not a Python graph package
to invent or a browser renderer.

Official references:

| Task | Read |
| --- | --- |
| Node model and first working output | [Overview](https://nvidia-omniverse.github.io/ovrtx/spg/overview.html), [first node](https://nvidia-omniverse.github.io/ovrtx/spg/first_node.html), [grayscale example](https://nvidia-omniverse.github.io/ovrtx/examples/python_spg_grayscale.html) |
| USD wiring and output publication | [USD interface](https://nvidia-omniverse.github.io/ovrtx/spg/ref/usd.html), [publish and read AOVs](https://nvidia-omniverse.github.io/ovrtx/spg/do/aovs.html) |
| Internal intermediates and multistage effects | [Chaining](https://nvidia-omniverse.github.io/ovrtx/spg/do/chaining.html), [pipeline example](https://nvidia-omniverse.github.io/ovrtx/examples/python_spg_pipeline.html) |
| CUDA or Slang resource/launch binding | [CUDA reference](https://nvidia-omniverse.github.io/ovrtx/spg/ref/lua_cuda.html), [Slang reference](https://nvidia-omniverse.github.io/ovrtx/spg/ref/lua_slang.html), [launch geometry](https://nvidia-omniverse.github.io/ovrtx/spg/do/launch_geometry.html) |
| Upload geometry or edit live parameters | [Upload data](https://nvidia-omniverse.github.io/ovrtx/spg/do/upload_data.html), [values](https://nvidia-omniverse.github.io/ovrtx/spg/do/values.html), [runtime changes](https://nvidia-omniverse.github.io/ovrtx/spg/do/runtime_changes.html) |
| Resource backing and reuse | [Textures/buffers](https://nvidia-omniverse.github.io/ovrtx/spg/do/textures_buffers.html), [caching](https://nvidia-omniverse.github.io/ovrtx/spg/do/caching.html) |
| Simple arithmetic versus a custom shader | [Built-in catalogue](https://nvidia-omniverse.github.io/ovrtx/spg/ref/builtin_catalogue.html), [adjustable blur example](https://nvidia-omniverse.github.io/ovrtx/examples/python_spg_blur.html) |

The website can describe a newer contract than the installed release. Resolve
names, value types, resources and launch behavior against that release; use the
corresponding examples as small capability probes rather than replacing the
app's architecture with their setup code.

## Node and application boundaries

A custom node has GPU source (`.cu` or `.slang`), a matching launch script
(`.cu.lua` or `.slang.lua`), and USD shader/interface wiring. The launch script
validates descriptors, allocates outputs and binds arguments; pixel or geometry
work belongs in GPU code. Keep large data tables out of the Lua sandbox.

Author the graph in the viewer-owned session/composite layer. Retain the source
color output and publish a separately named processed RenderVar, for example
`LdrGrayscale` or `LdrStylized`. Connect guide AOVs from the same RenderProduct
and frame. Verify fully qualified connection paths resolve in the installed
runtime; product-local source RenderVars are a useful minimal wiring probe.

Keep intermediate textures/buffers inside the graph unless a consumer or
validation probe needs publication. Neighborhood reads consume immutable input;
separate output resources prevent races. Declare channel order, image origin,
normal space, ID/background encoding, resource shape, dtype and backing. An
integer normal container requires a verified bit reinterpretation, not a
numeric conversion. ID textures require nearest sampling.

The existing runtime owner commits camera, stage, transforms and parameters,
then steps OVRTX. Retain its established OVStage integration when present.
Network/UI callbacks enqueue state changes; they do not mutate renderer or
graph resources. Rebind and invalidate capabilities after stage or resolution
changes, and discard results from older geometry revisions.

The delivery layer selects and maps the actual processed output, performs its
required format conversion, and presents it with the existing synchronization
and ownership contract. Preserve RGBA through the graph; convert once to BGRA
only if the selected encoder needs it. CSS effects, CPU image processing and
post-processing a saved screenshot do not establish SPG integration.

## Verify resource and launch contracts

For each buffer node, test first and last rows/elements and a partial final
workgroup. Buffer iteration domains can differ from image domains. Declare
launch geometry using the installed contract and confirm which invocations
actually ran; a correct first tile does not prove the entire output was written.

A published buffer can arrive as raw bytes. Interpret it only using the node's
explicit element type, stride, lane count and layout. Do not infer a float image
from the byte mapping's shape. Independently validate matrices with a known
world point and projected endpoint; distinguish USD row-vector transforms from
column-vector projection math.

Prove scalar and vector-array lane counts with nonuniform fixtures before
uploading full geometry. Size large outputs on the GPU rather than creating
full-resolution Lua tables. Separate node responsibilities or reuse existing
triangle data when an additional edge array exceeds the launch-script budget.

Treat launch-script descriptor reuse, host-to-GPU upload reuse and GPU work
reuse as separate measurements. A static descriptor alone does not prove that
geometry stays resident or that the shader skips a frame. Inspect cache/upload
warnings and profile the actual target before promising a transfer-free path.

## Capability and fallback

Advertise a processed output only after it maps with the intended format and
contains plausible results. Resolve required-guide failures before publishing
an effective style. Keep current color or the last good frame available, but
report requested/effective mode and the failed layer explicitly. A live fallback
is not successful execution of the requested graph.

## Validate the graph and the presentation

Start with distinct RGB surfaces and an identity or independently specified
transform. Compare source and output from the same step, before compression.
Record formula, channel order, rounding, tolerance and alpha policy. Verify
that input contains color and that the transform changes the expected pixels.

For geometry graphs, validate projection, bin membership and private depth
independently before comparing the final image. Check overflow and near-plane
cases. Compare against a pinned previous implementation on identical inputs;
separate image-equation parity from differences in a replacement rasterizer.

Exercise parameters, repeated source/processed toggles, orbit/pan/zoom, stage
reload, actual RenderProduct resize and shutdown/restart where supported.
Browser resize may change only presentation size. Capture the real viewport or
stream and verify changed pixels, not merely a selected menu or decoded-frame
counter. Measure warm baseline/graph costs, memory growth, current-camera
latency, and actual unique presented frames.

## Lens distortion and other sampling effects

Use a custom sampling node when the effect needs more than documented built-in
arithmetic. Define output-pixel to source-coordinate mapping, camera intrinsics,
radial/tangential coefficients, interpolation and outside-image policy. Test
identity and a calibration grid before enabling the effect on a viewer.

Color can use filtered sampling; IDs require nearest sampling and normals/depth
need validity rules. Apply matching inverse coordinates to picking and scene
marks. Compose screen UI after the warp when it must remain readable. A lens
warp is a separately validated effect, not a capability implied by an X-Ray
shader or generic SPG support.
