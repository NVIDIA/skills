<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Vulkan Compositing Over OVRTX

## When To Use

Use for non-XR native Vulkan presentation of OVRTX output with raster gizmos,
orientation triads, billboards, annotations, or custom image overlays. This also
covers a viewer that calls its presentation wrapper a `VulkanSurface`: identify
its real color image, layout, owner and completion contract rather than assuming
that wrapper exposes OVRTX's internal render targets.

OVRTX continues to render the USD scene. Vulkan presents its pixels and draws
the requested overlays. Keep the application's existing shell and render owner;
adding a compositor does not require replacing either. For stereo/XR, use the
XR references for per-eye camera, image ownership and frame timing first.

## Read Order And Boundaries

1. Resolve packages through [dependencies](../dependencies/README.md), then
   read the selected OVRTX release's current agent guidance, output/mapping
   documentation and Vulkan interoperability contract. Dependency acquisition
   stays there; this family owns the viewer's composition boundary.
2. Use [interop and lifecycle](interop-and-lifecycle.md) for CUDA/Vulkan device
   identity, image ownership, frame slots, barriers, and resource reuse.
3. Add [depth-aware overlays](depth-overlays.md) only when overlays must be
   hidden by scene geometry. It includes the input formats, camera ABI, shader
   snippets and Vulkan depth state for manual rejection and depth seeding.
4. For interactive handles, also read
   [input routing](../viewer-input-routing/README.md),
   [transform manipulators](../transform-manipulator/README.md), and
   [transform safety](../prim-transform-safety/README.md). The compositor
   consumes accepted camera/tool state; it does not own scene mutations.

For headless ovui widget export followed by CUDA alpha blending, use
[viewport overlays](../viewport-overlays/README.md). For OpenGL overlays, use
[GL viewport overlays](../gl-viewport-overlay/README.md). For image/geometry
processing inside OVRTX's GPU graph, use [SPG processing](../spg-processing/README.md).
The choices have different resource owners; do not combine their interop APIs.

## Choose An Overlay Policy

| Need | Composition policy |
|---|---|
| Corner orientation triad, HUD, screen labels | Draw after the OVRTX color splat with scene-depth testing and scene-depth writes disabled. Use a separate small depth attachment if the triad's own geometry needs self-occlusion. |
| World-anchored gizmo, billboard or annotation | Use matching scene-distance output; choose manual fragment rejection or seed an app-owned depth attachment. |
| Several raster passes sharing scene occlusion | Seed D32 once, preserve it, then use compatible depth tests for each pass. |
| Hidden-handle/X-ray treatment | Make visible/hidden fragment policy explicit; use separate styles/passes without overwriting scene color or depth unintentionally. |
| Ordinary selected-object feedback | Use [native selection feedback](../selection-feedback/README.md); custom raster overlays do not replace picking or native selection groups. |
| Cutout billboard | Sample texture alpha and discard below a chosen cutoff before depth writes. |
| Translucent annotation | Choose sorting and blending deliberately; usually test depth but disable depth writes. Alpha zero does not suppress a depth write. |

Represent wide handles with triangle geometry such as cylinders, cones and
quads. Vulkan wide-line support depends on device features and limits. Derive
camera-facing billboard axes from the same displayed-frame camera used for the
scene; do not use newer UI camera state with older displayed pixels.

## Color Composition

Preserve the OVRTX image when beginning an overlay scope: color attachment
`loadOp=LOAD`, `storeOp=STORE`. Clear only resources whose prior contents are
unneeded. Allocate a separate target when the base image remains owned by a
producer or consumer; a mapped OVRTX output is not an app-owned render target.

Choose one alpha convention and match it to the pipeline. With `ADD` blending:

| Source color convention | RGB source factor | RGB destination factor | Alpha factors |
|---|---|---|---|
| Straight alpha | `SRC_ALPHA` | `ONE_MINUS_SRC_ALPHA` | `ONE`, `ONE_MINUS_SRC_ALPHA` |
| Premultiplied alpha | `ONE` | `ONE_MINUS_SRC_ALPHA` | `ONE`, `ONE_MINUS_SRC_ALPHA` |

Inspect actual exported pixels rather than inferring alpha convention from
`RGBA8`. Multiplying premultiplied RGB by alpha again darkens edges. Use a
transparent black clear for a separate overlay image. Clear/opaque pixels must
be tested, not reconstructed by treating black as transparent.

Keep shader math in linear color where appropriate. An sRGB sampled-image view
decodes RGB; an sRGB color attachment performs its defined blend/output
conversion. Check how the existing OVRTX splat handles display-encoded color
before adding any gamma conversion: double conversion changes the scene even
when no overlay is visible. Alpha is not gamma encoded. For HDR, define whether
overlays compose before tone mapping or in the display path; do not blend
unconverted HDR scene values with display-byte overlay values.

The [Vulkan framebuffer contract](https://docs.vulkan.org/spec/latest/chapters/framebuffer.html)
owns blending and sRGB behavior. The viewer owns the choice of image formats,
alpha convention, and the exact order of its color and overlay passes.

## Implementation And Evidence

Build app-local allocation/import, frame-slot, pipeline and draw components from
the contracts above. No compositor class, shader file or private sample checkout
is a prerequisite; generate the required modules in the host application's
normal layout. Keep scene state below the UI/compositor boundary, and keep
slow asset work off its input path.

Start with color plus an opaque corner marker, then alpha edges, then scene
occlusion if requested. Validate both the pre-presentation composite and actual
window/transport output. Use [validation](../validation.md) for artifacts and
[interop and lifecycle](interop-and-lifecycle.md) for stale-frame, resize and
shutdown acceptance checks.
