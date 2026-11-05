<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Vulkan Interop And Compositor Lifetime

Read [Vulkan compositing](README.md) first. Obtain the selected SDK through
[dependencies](../dependencies/README.md); upstream owns exact mapping, external
memory, semaphore and platform-handle APIs. Check that release's public headers
and matching documentation before implementing an adapter.

## Establish The Image Boundary

Distinguish the OVRTX output from the application's presentation image. The
verified output mapping path exposes CUDA arrays or mapped tensors, not an
assumed public export of OVRTX's internal `VkImage` or depth attachment. An
application wrapper named `VulkanSurface` does not change that ownership.

For a GPU-copy integration, Vulkan allocates exportable image memory and CUDA
imports compatible memory as arrays; copy OVRTX output into those images, then
sample them in the application's Vulkan pass. Verify external-memory support,
handle type, image tiling/format, allocation size, dedicated-allocation rules,
and handle ownership on the chosen platform. Extend the image adapter explicitly
for each channel count and scalar type. Never treat a linear device pointer as
a CUDA array, or reuse a four-channel color importer for one-channel distance.

Match the Vulkan physical device to the actual CUDA device that owns each
output, using the SDK's device UUID guidance; ordinal 0 in different APIs is
not an identity guarantee. Respect process-visible CUDA ordinals and MIG-aware
identity. Use a consistent CUDA context and keep each API's current-context
requirements on the threads that perform its operations.

## Define A Paired Frame Slot

A slot owns everything needed to present one accepted frame:

- copied color, and distance/hit images when scene occlusion is requested;
- matching camera/projection, viewport, scene units, render extent, and scene
  generation;
- immutable-for-flight descriptors and uniform data;
- CUDA copy completion, CUDA/Vulkan handoff values, and Vulkan submission
  completion.

Keep an app-owned depth attachment per in-flight raster submission or otherwise
serialize its reuse. Per-slot color does not make a shared D32 attachment safe.
Do not update an in-flight descriptor set or camera UBO just because CPU camera
state changed. Choose a free slot before copying; when no slot is free, preserve
the last completed image or apply bounded backpressure rather than overwriting
one still being sampled.

## Producer, Copy, Draw, Reuse

1. On the renderer owner, fetch a completed frame. Collect related outputs from
   that same render-product frame. Keep mappings alive until all source reads
   finish. Disable frame generation for initial depth integration because an
   interpolated color frame may lack matching distance and hit data.
2. Map each required output using the selected OVRTX contract. Validate dtype,
   shape, tensor choice, byte offset/layout and GPU. Wait each mapping's
   producer event on the copy stream before reading. Do not assume a successful
   map means asynchronous production has finished.
3. Before CUDA writes a destination slot, establish that its previous Vulkan
   use is complete and its images have been released to the external owner in
   the interop-required layout. Copy all slot images on the ordered stream.
4. Record a completion event after all copies; release OVRTX maps with the
   required consumer-completion sync. Retain that event until all users of it
   have finished. Follow the SDK's unmap timing: an immediate unmap may block
   the CPU even if the copy submission was asynchronous.
5. Signal the external handoff semaphore only after all destination writes.
   Vulkan waits at the first applicable consumer stages, acquires external
   queue ownership, and transitions shared textures from `GENERAL` to
   `SHADER_READ_ONLY_OPTIMAL` according to the selected interop protocol.
   Semaphores order work; they do not perform image layout/ownership barriers.
6. Splat color, composite/draw overlays, and submit using that slot's camera.
   Make prior color writes visible to an overlay scope's load/blend reads and
   writes, even if color layout remains `COLOR_ATTACHMENT_OPTIMAL`. Depth
   reads/writes use early/late fragment-test stages, not just fragment shading.
7. Track Vulkan completion through actual sampling and downstream consumers.
   Release shared images back to the external owner in the required layout
   before signaling reuse. Advance semaphore values monotonically and keep a
   clear association between each value and the slot it releases.

Use the [Vulkan synchronization guide](https://docs.vulkan.org/guide/latest/synchronization_examples.html)
for access/stage dependencies, together with the selected OVRTX/CUDA interop
contract for external memory and semaphore ordering. Do not replace explicit
handoffs with per-frame global device-idle waits in an interactive viewer.
A deliberately synchronized bring-up path can establish correctness before
introducing independent in-flight slots; measure any optimized replacement.

## Failure, Resize And Shutdown

Treat a slot as a transaction: incomplete or invalid color/distance/camera data
never replaces the accepted slot. Validate hit depths before enabling scene
occlusion; on a nonbackground pixel, a nonfinite/nonpositive distance is an
input failure, not permission to expose hidden handles. Do not mix outputs from
different scene generations while one is being recreated.

For resize, stop accepting new slot work, drain OVRTX maps/CUDA/Vulkan consumers,
then recreate shared images, depth targets, descriptors and viewport state as
one consistent set. Reject zero extents and verify every actual output extent
before sampling. Do not destroy the old presentation target while a downstream
encoder/window still uses it. Resume only after one complete new slot is ready.

For shutdown or partial startup failure, stop submissions and input capture,
finish or retire owned work, then release maps, imported CUDA views, Vulkan
resources and synchronization objects in their dependency order. Keep native
handles alive until both APIs stop using them. Cleanup must be safe when only
some resources initialized; a failed producer must not trigger a blocking wait
for a semaphore value it will never signal. Escalate owner/device failure to the
application lifecycle rather than racing destruction against active GPU work.

Validate color-only startup, stalled producer, failed copy, out-of-order scene
results, several frames in flight, resize, and shutdown. Record the frame/slot
identity with captures so a visually plausible stale image cannot count as a
successful depth-aware composite.
