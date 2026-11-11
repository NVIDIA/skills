<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Revisioned RGBA ABI

## Add An Image Contract

This is an application ABI to implement, not an OVXR SDK type/export. For
multiple surfaces, wrap each image in a versioned record carrying surface ID,
surface/stage generation and pixel/alpha/color-space format. The fields below
alone do not identify a window. Use [editor state](editor-document-model.md) and
[async publication](async-raster-pipeline.md) for atomic image/hit-map activation.

Add a raster entry point instead of changing an existing document ABI:

```c
typedef struct ovxr_server_ui_image_t {
    uint32_t struct_size;
    uint32_t width;
    uint32_t height;
    uint32_t pitch_bytes;
    uint32_t reserved;
    uint64_t revision;
    const uint8_t* pixels;
} ovxr_server_ui_image_t;
```

Use fixed-width fields and test native and foreign-language structure sizes.

## Validate Then Copy

Reject null pointers, undersized structures, unexpected dimensions or pitch,
stale or repeated revisions, and dimensions that overflow the expected byte
count. Copy `height * pitch_bytes` into native-owned memory before returning.
Never retain caller-owned Python, Node, or foreign-language memory.

Protect document and image state with one mutex. Compare a revision with the
active representation, accept only the newest, and deactivate the old one.

Distinguish source-model generation from monotonic native image and pose
revisions. Authoritative stage metadata may refine a pose without changing the
source generation; issue a newer native revision while rejecting strictly older
source generations.

## Upload And Present

- Select an RGBA format supported by the chosen presentation backend.
- For an additive composition layer, create the UI swapchain at raster
  dimensions and follow `acquire -> wait -> upload -> release -> submit`.
- For primary-projection presentation, keep one persistent Vulkan texture and
  upload only a new revision before the bounded in-place eye draws.
- Stop that surface's presentation on any failed result.
- Keep staging memory alive through the GPU copy.
- Match Vulkan layouts and synchronization to the graphics binding.
- Expose the format and last `XrResult` in status.

Cache the last uploaded revision per swapchain image. If that image already
contains the current revision, release it without a CPU copy, Vulkan command
recording, queue submission, or fence wait. Update the cache only after a
successful upload. Expose exact render, upload, and upload-skip counters.

Match pixel alpha to the presentation contract. For an additive OpenXR layer,
match straight or premultiplied pixels to its composition flag. For an in-place
Vulkan path, match them to the blend factors and shader output. Do not guess.

Test symbol presence, structure size, invalid metadata, stale revisions,
caller-buffer lifetime, format selection, image ordering, failure cleanup,
session recreation, and swapchain destruction.
