<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Depth-Aware Vulkan Overlays

Use when world-space gizmos, markers, or billboards must be hidden by OVRTX
scene geometry. Read [compositing choices](README.md) and
[interop and lifecycle](interop-and-lifecycle.md) first. Keep the selected
OVStage/OVRTX render owner; this adds application raster overlays after its
color output, without introducing a second USD renderer.

A color splat does not establish scene depth. Choose either manual
metric-distance rejection in the overlay fragment shader, or a fullscreen pass
that seeds an application-owned D32 attachment before normal Vulkan depth tests.
The inline shaders below implement both paths; no local sample files are needed.

## Inputs From The Same OVRTX Frame

Add these RenderVars to your viewer-owned RenderProduct's `orderedVars`, using
the selected OVRTX release's output authoring API. Do not edit the source USD.
The [official camera output table](https://github.com/NVIDIA-Omniverse/ovrtx/blob/b9ae5f0913847502ff5ed29ab45a6e9f28bec8f1/docs/sensors/cameras/outputs.rst)
and [camera output decoding documentation](https://github.com/NVIDIA-Omniverse/ovrtx/blob/b9ae5f0913847502ff5ed29ab45a6e9f28bec8f1/docs/sensors/cameras/decoding_outputs.rst)
describe the output contracts:

| Output | Vulkan sampling image | Meaning |
|---|---|---|
| `LdrColor` | Format compatible with the existing color path | Already displayed by your color splat. Preserve its color-space handling. |
| `DistanceToCameraSD` | `VK_FORMAT_R32_SFLOAT` | Float32 H×W×1 Euclidean distance from camera origin, in meters. This is not a hardware Z value. |
| `NonStableInstanceSegmentation` | `VK_FORMAT_R32_UINT` | UInt32 H×W×1: 0 is background; **1 is a real unlabeled hit**; higher values are labeled hits. Used only as a hit mask here. |
| `CameraParams`, or your exact accepted camera state | UBO built by the application | View, projection, resolution, scene-to-meter scale for this output frame. |

Find these handles within one `ovrtx_render_product_frame_output_t`, matching
full authored RenderVar paths/source names. Never pair the newest depth with an
older displayed color or use current interactive camera matrices with a frame
rendered before that camera change. Disable frame generation for the initial
integration: a generated color frame may have no corresponding depth.

Use a pinhole perspective camera with depth of field disabled for this approach.
Require equal output extents and exact 1:1 placement in a positive-height Vulkan
viewport; letterboxing is supported through its integer X/Y offset. Scaling,
nonlinear lenses, orthographic cameras, independently jittered outputs,
transparency, motion blur and antialiased silhouette coverage need an explicit
reprojection/coverage policy before extending this approach. Primary-hit depth
cannot exactly represent all contributors to a filtered or transparent pixel.

## Descriptor And Camera ABI

Use this host-side layout for a 240-byte std140 camera UBO and a 28-byte
triangle vertex. These are application-defined types, not OVRTX SDK types:

```cpp
#include <cstdint>
struct alignas(16) CameraBlock {
    float worldToView[16], viewToClip[16], clipToView[16];
    float viewport[4];       // x, y, width, height in destination pixels
    float metric[4];         // meters/scene-unit, bias meters, unused, unused
    std::uint32_t flags[4];  // unused, flip AOV rows, reversed Z, unused
};
static_assert(sizeof(CameraBlock) == 240);
struct Vertex { float worldPosition[3]; float linearColor[4]; };
```

Create descriptor set 0 with:

| Binding | Descriptor | Stages |
|---|---|---|
| 0 | Uniform buffer, `sizeof(CameraBlock)` | Vertex + fragment |
| 1 | Combined image sampler, distance `R32_SFLOAT` | Fragment |
| 2 | Combined image sampler, hit IDs `R32_UINT` | Fragment |

Both sampled image descriptors use `SHADER_READ_ONLY_OPTIMAL`. Use nearest
samplers; shaders use `texelFetch` at integer pixel coordinates, not filtered
values across geometry edges. Match the sampled extents to `viewport.zw`.

Fill the UBO as follows:

- `worldToView`: camera transform inverse with column-vector math and
  column-major storage; the camera basis must be rigid, not scaled.
- `viewToClip`: the same pinhole camera projection, adapted to Vulkan Z in
  [0,1], with any Y convention conversion already applied exactly once.
  `clipToView` is its inverse. These shaders and the color splat must agree on
  which scene ray corresponds to each destination pixel.
- `viewport`: destination image rectangle `(x, y, renderWidth, renderHeight)`.
- `metric[0]`: meters per scene unit, positive. `metric[1]`: nonnegative
  occlusion bias in meters, initially 0; tune against a known plane, then use
  the smallest value that avoids unwanted self-occlusion.
- `flags[0]`: unused in the inline shaders; set to 0.
  `flags[1]`: 1 only when the copied AOV row origin needs flipping relative to
  destination framebuffer coordinates. Flip distance and hit fetches together.
  `flags[2]`: 1 for reversed Z; also choose the matching pipeline compare/clear.
  Other fields are zero.

Use `CameraBlock.flags[2] != 0` as the single reversed-Z choice for the shader,
attachment clear, and pipeline depth-compare direction. Projection, shader background depth,
attachment clear value, and hardware compare direction must all agree.

OVRTX `CameraParams` uses a documented row-vector matrix convention. Convert
its mathematical matrices to the column-vector representation above once; do
not transpose an already converted buffer twice. Use the official record
decoder rather than assuming a packed C++ structure has the correct ABI.

## Shader Snippets To Create In Your Application

Create the shader files named below. For each seed/gizmo shader, start its
source with the **shared prelude**, then append the indicated shader body.
The fullscreen vertex shader is complete as shown.

Shared prelude for `seed_depth.frag`, `gizmo.vert`, and `gizmo.frag`:

```glsl
#version 450
layout(std140, set = 0, binding = 0) uniform CameraBlock {
    mat4 worldToView;
    mat4 viewToClip;
    mat4 clipToView;
    vec4 viewport;
    vec4 metric;
    uvec4 flags;
} camera;
layout(set = 0, binding = 1) uniform sampler2D sceneDistance;
layout(set = 0, binding = 2) uniform usampler2D sceneHit;

ivec2 scenePixel(vec2 framebufferPosition) {
    ivec2 p = ivec2(floor(framebufferPosition - camera.viewport.xy));
    if (camera.flags.y != 0u) p.y = int(camera.viewport.w) - 1 - p.y;
    return p;
}
float farDepth() { return camera.flags.z != 0u ? 0.0 : 1.0; }
bool invalidDistance(float d) { return isnan(d) || isinf(d) || d <= 0.0; }
```

Complete `fullscreen.vert` (no prelude needed):

```glsl
#version 450
void main() {
    vec2 p = vec2(float((gl_VertexIndex << 1) & 2), float(gl_VertexIndex & 2));
    gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}
```

Append this body to the prelude for `seed_depth.frag`. The inverse projection
constructs the camera ray for each destination pixel. Multiplying that ray by
OVRTX's metric distance reconstructs the scene point; projecting it gives the
normalized Vulkan depth to write:

```glsl
void main() {
    ivec2 p = scenePixel(gl_FragCoord.xy);
    if (texelFetch(sceneHit, p, 0).r == 0u) {
        gl_FragDepth = farDepth();
        return;
    }
    float meters = texelFetch(sceneDistance, p, 0).r;
    if (invalidDistance(meters)) {
        gl_FragDepth = 1.0 - farDepth(); // Fail closed; host reports invalid input.
        return;
    }
    vec2 ndc = ((gl_FragCoord.xy - camera.viewport.xy) / camera.viewport.zw) * 2.0 - 1.0;
    vec4 eyeH = camera.clipToView * vec4(ndc, 0.5, 1.0);
    vec3 ray = normalize(eyeH.xyz / eyeH.w);
    vec3 eye = ray * ((meters + camera.metric.y) / camera.metric.x);
    vec4 clip = camera.viewToClip * vec4(eye, 1.0);
    gl_FragDepth = clamp(clip.z / clip.w, 0.0, 1.0);
}
```

Append this body to the prelude for `gizmo.vert`. Vertices are already in world
space; apply your gizmo's model transform first if they begin in local space:

```glsl
layout(location = 0) in vec3 worldPosition;
layout(location = 1) in vec4 linearColor;
layout(location = 0) out vec3 viewPosition;
layout(location = 1) out vec4 color;
void main() {
    vec4 eye = camera.worldToView * vec4(worldPosition, 1.0);
    viewPosition = eye.xyz;
    color = linearColor;
    gl_Position = camera.viewToClip * eye;
}
```

Append this body to the prelude for `gizmo.frag` in the hardware-depth path:

```glsl
layout(location = 0) in vec3 viewPosition;
layout(location = 1) in vec4 color;
layout(location = 0) out vec4 outColor;
void main() { outColor = color; }
```

## Vulkan Attachment And Pipeline Setup

Allocate an app-owned `VK_FORMAT_D32_SFLOAT` image with
`DEPTH_STENCIL_ATTACHMENT` usage, matching the destination color extent and
sample count (one sample here). Create its depth-aspect image view. Do not alias
OVRTX-owned memory as this image.

Create two graphics pipelines with a common descriptor/pipeline layout,
triangle-list topology, no culling, dynamic viewport/scissor, one color
attachment of your destination format, D32 depth, and one sample. With Vulkan
1.3 dynamic rendering, enable `dynamicRendering` and provide the formats through
`VkPipelineRenderingCreateInfo`. An existing compatible render pass works too.

| Pipeline | Shaders | Vertex input | Depth state | Color writes |
|---|---|---|---|---|
| Seed | `fullscreen.vert` + `seed_depth.frag` | None; draw 3 vertices | Test on, compare ALWAYS, write on | Disabled; preserve splatted color |
| Gizmo | `gizmo.vert` + `gizmo.frag` | Binding 0 stride 28; location 0 `R32G32B32_SFLOAT` offset 0; location 1 `R32G32B32A32_SFLOAT` offset 12 | Test on, compare LESS_OR_EQUAL, write on | RGBA enabled |

For reversed Z, change the gizmo compare to `GREATER_OR_EQUAL` and clear depth
to 0 rather than 1. The projection and `camera.flags.z` must match that choice.
Essential state setup (standard Vulkan structures; `reversedZ` is your app's
camera choice):

```cpp
VkPipelineDepthStencilStateCreateInfo seedDepth{};
seedDepth.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
seedDepth.depthTestEnable = VK_TRUE;
seedDepth.depthWriteEnable = VK_TRUE;
seedDepth.depthCompareOp = VK_COMPARE_OP_ALWAYS;
auto gizmoDepth = seedDepth;
gizmoDepth.depthCompareOp = reversedZ ? VK_COMPARE_OP_GREATER_OR_EQUAL
                                     : VK_COMPARE_OP_LESS_OR_EQUAL;
VkPipelineColorBlendAttachmentState seedColor{};
seedColor.colorWriteMask = 0; // Critical: depth seeding must preserve existing color.
```

For gizmo straight-alpha blending, enable RGBA writes and use RGB factors
`SRC_ALPHA` / `ONE_MINUS_SRC_ALPHA`, alpha factors `ONE` /
`ONE_MINUS_SRC_ALPHA`, both with `ADD`. Opaque handles use alpha 1. Preserve
your color attachment's encoding: shader colors are linear; an sRGB attachment
performs its usual output encoding.

Record this order after the color splat. The handles below are resources your
application creates using the contracts above:

```cpp
// End the existing color-splat rendering scope and apply the barriers below.
// Begin a scope with color loadOp=LOAD/storeOp=STORE (preserve OVRTX color),
// depth loadOp=CLEAR/storeOp=STORE, clearDepth=reversedZ ? 0.0f : 1.0f.
// Set viewport/scissor to (imageX, imageY, renderWidth, renderHeight), depth [0,1].
vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, layout,
                        0, 1, &descriptors, 0, nullptr);
vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, seedPipeline);
vkCmdDraw(cmd, 3, 1, 0, 0);
vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, gizmoPipeline);
vkCmdBindVertexBuffers(cmd, 0, 1, &gizmoVertices, &vertexOffset);
vkCmdDraw(cmd, gizmoVertexCount, 1, 0, 0);
// End rendering, then transition/synchronize the color target for its consumer.
```

Use triangle meshes for handles (cylinders, cones, quads). Wide Vulkan lines
require device-specific feature/limit checks.

**Barriers and ownership:** make prior splat color writes visible to subsequent
color reads/writes (`COLOR_ATTACHMENT_OUTPUT` -> `COLOR_ATTACHMENT_OUTPUT`,
color write -> color read|write), even when its layout stays
`COLOR_ATTACHMENT_OPTIMAL`. Put D32 in
`DEPTH_STENCIL_ATTACHMENT_OPTIMAL` with early/late depth read|write access.
Complete the shared-texture wait/acquire from
[interop and lifecycle](interop-and-lifecycle.md), and make UBO/vertex
writes visible to shader/input reads. Do not clear depth between the seed draw
and gizmo draw. Finish normal presentation/consumer synchronization afterward.

## Manual Metric Path And UI Policy

For a first proof of concept, you can skip the depth seed. Keep the same UBO,
distance and hit textures and `gizmo.vert`; replace the gizmo fragment body
with this, appended to the shared prelude:

```glsl
layout(location = 0) in vec3 viewPosition; // Perspective-correct interpolation.
layout(location = 1) in vec4 color;
layout(location = 0) out vec4 outColor;
void main() {
    ivec2 p = scenePixel(gl_FragCoord.xy);
    if (texelFetch(sceneHit, p, 0).r != 0u) {
        float sceneMeters = texelFetch(sceneDistance, p, 0).r;
        if (invalidDistance(sceneMeters)) {
            outColor = vec4(1, 0, 1, 1); // Diagnostic; reject invalid frames on host.
            return;
        }
        float gizmoMeters = length(viewPosition) * camera.metric.x;
        if (gizmoMeters > sceneMeters + camera.metric.y) discard;
    }
    outColor = color;
}
```

This tests the same scene occlusion in meters. Use a separately cleared raster
depth attachment for gizmo self-occlusion if needed. Do not mark `viewPosition`
as `noperspective`: the fragment needs its perspective-correct view-space
position. `length(viewPosition)` is radial distance; `-viewPosition.z` matches
it only on the optical axis. This alternative does not populate a shared scene
depth buffer for other raster passes.

A corner orientation triad usually belongs in a later always-on-top UI pass
with scene-depth testing disabled. World-anchored billboards and gizmos can
use scene depth. X-ray handles can use a second styled hidden-fragment pass,
but transparent depth writes and sorting need an intentional policy. Ordinary
object selection should keep OVRTX's native selection groups/outlines; this
approach addresses custom raster overlays while OVRTX continues to render USD.

The depth-writing pipeline suits opaque handles. For cutout billboards,
sample the billboard texture and discard below an alpha cutoff before writing
depth. Fully translucent overlays need a separate sorting/blending policy and
usually disable depth writes; alpha zero alone does not prevent a depth write.

## Compile And Validate

Assemble the shader sources from the prelude and bodies above, then compile
with the selected Vulkan shader compiler. Keep the manual fragment alternative
as a separate module. Use [viewer validation](../validation.md) to record the
actual presentation path; shader compilation or a numerical projection check
alone does not establish a working interop implementation.

1. Render an opaque plane with a handle before, on, and behind it. The front
   portion must remain visible and the rear portion must disappear without
   changing the underlying OVRTX scene color.
2. Repeat off-center, then orbit/pan/zoom. Off-axis errors indicate radial vs
   axial distance, projection, or units. Compare a known distance against the
   actual mapped AOV before tuning the bias.
3. Verify asymmetric corner markers, row origin, integer viewport offset, and
   non-meter scene scale. Test normal/reversed Z and near/far clipping.
4. Switch scenes and move the camera with multiple frames in flight. Color,
   distance, hit mask and camera must advance together. Preserve the last
   accepted slot while a new frame is incomplete; reject invalid hit depths.
5. Validate handle self-occlusion, hide/cancel while dragging, and drained
   cleanup during resize, disconnect, and shutdown.
