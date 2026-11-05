<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# In-Place Primary-Projection Compositor

## Frame And Image Ownership

OVXR owns instance, session, frame timing, projection construction, and frame
submission. A loader API layer or supported additive hook may augment a frame,
but it never becomes a second frame owner.

For an in-place Vulkan path:

1. Observe primary color and chained depth swapchains and their image metadata.
2. Defer release only for images that were acquired and successfully waited.
3. At `xrEndFrame`, match the projection views to exactly one deferred color
   and depth image per eye.
4. Record bounded draws into those images, batching compatible depth policies.
5. Release each deferred image exactly once.
6. Forward the original projection and call the next `xrEndFrame` exactly once.

Copy application frame headers and layer-pointer arrays for the duration of the
downstream call. Never retain caller-owned OpenXR pointers. Serialize session,
swapchain, and image-acquisition state. Ambiguous or multiply deferred images
fail closed.

## Per-Eye Geometry

Treat each projection view independently. Use that eye's pose, asymmetric FOV,
color and depth subimage offsets/extents, array layers, image extents, and depth
parameters. Never average stereo FOVs or derive one eye from the other.

Transform `LOCAL` surface vertices into the eye view, project with the eye's
four FOV tangents, clip in homogeneous space before division, and map NDC into
the submitted subimage. Vulkan viewport and scissor coordinates are expressed
in the full target image, including the subimage offset.

Build one effective scissor per eye from the union of every surviving surface.
A valid tooltip or gizmo outside a panel-only scissor is a compositor defect.
Keep each surface independently valid: clipping one auxiliary must not suppress
the panel or another auxiliary.

## Color, Alpha, And Depth

Preserve the existing scene with attachment `LOAD`/`STORE`. Draw only covered
fragments and leave the rest of the primary image unchanged. Match the source
raster's straight or premultiplied alpha to the Vulkan blend equation; do not
infer it from appearance.

Derive fragment depth from the verified producer encoding plus chained
`XrCompositionLayerDepthInfoKHR` range metadata. Physical near/far ordering is
not sufficient for OVXR's reverse XRDepth; see
[projection/depth equations](projection-and-depth.md). Validate that color and depth refer
to the matching eye/subimage, and reject unsupported formats, sample-count
mismatches, protected content, or missing attachment usage.

A flattened primary projection can represent an opaque or alpha-masked panel
at one depth. It cannot preserve translucent UI and the scene behind it at two
independent depths. Declare that limitation rather than faking glass.

## Synchronization And Resource Lifetime

- Reuse pipelines, descriptors, image views, framebuffers, staging buffers,
  command buffers, and a bounded set of fence-polled frame slots.
- Upload the retained panel texture only when its document revision changes.
- Update only small pose/mesh buffers for pose-only surfaces.
- Do not transfer or read back full primary images.
- Do not wait on a fence submitted by the same frame.
- If every bounded slot is busy, omit UI for that frame and retain honest
  counters.
- Preflight retirement with a bounded timeout; on failure retain owners rather
  than destroying resources still in flight.

An API layer must use the application's enabled Vulkan device features, not
assume it can enable anisotropy or another feature after device creation.

## Additive Layers Are A Different Contract

An `XrCompositionLayerQuad` is appropriate for intentional HUD content or a
runtime that gives that layer the required transport semantics. It does not
carry `XrCompositionLayerDepthInfoKHR`, and a remote client may reproject its
primary stream without equivalent per-fragment quad depth. Do not use a
successful quad submission as proof of scene occlusion or world-stable CloudXR
output.
