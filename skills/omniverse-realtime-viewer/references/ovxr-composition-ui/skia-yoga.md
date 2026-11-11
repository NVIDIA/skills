<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Skia And Yoga Rendering

Read `dependencies/skia-yoga.md` for the pinned public dependency path.

## Rendering Contract

Implement [editor state](editor-document-model.md) and the
[bounded raster handoff](async-raster-pipeline.md) before connecting interactive
updates. A persistent child process alone does not make its caller asynchronous.

1. Build one Yoga tree from panel data.
2. Calculate layout in raster pixels.
3. Paint backgrounds, text, icons, hover, press, and selection with Skia.
4. Export RGBA bytes with a documented alpha convention.
5. Return the exact Yoga hit rectangles with the same revision.
6. Free Yoga nodes after reading computed layout.

Use a real font renderer. Do not substitute hand-built bitmap glyphs.

## Keep Three Sizes Separate

- Logical layout units control arrangement.
- Raster dimensions control pixel detail.
- Physical metres control spatial size.

A validated 2:1 example uses 1024x512 pixels. Treat that as an example default,
not a specification. Choose resolution from angular size and viewing distance,
then confirm readability in the headset capture.

Increasing physical size does not add pixel detail. Increasing raster size does
not move or enlarge the spatial quad.

## Diagnose the surface that is actually aliased

Separate dense panel text, analytic toolbar/chrome edges, texture minification,
and whole-eye resolution. Increasing a panel raster cannot improve a toolbar
drawn by another shader. If only the UI is affected, inspect its paint and
sampling before changing headset render scale, foveation or stream resolution.

Use real glyph/vector coverage for text and icons, and screen-derivative edge
coverage for suitable analytic shapes. Keep coverage consistent with the
premultiplied-alpha blend. Shader antialiasing does not soften the outside of a
triangle that has no fragments there: provide a padded carrier quad for a
rounded/SDF edge while retaining semantic hit bounds.

Use bounded sampling for minified textures only when it improves measured
output. Do not treat extra taps as free or enable sampler features the existing
Vulkan device did not enable. Validate toolbar and panel independently at
oblique angles, near/far distance, and both eyes' edges; measure GPU cost at the
largest screen coverage, with raster uploads separated from draw cost.

## Revision Policy

Increment the raster revision when hierarchy, labels, selection, hover,
capture, theme, scale, or dimensions change. Do not repaint every XR frame when
the revision is unchanged.

## Tests

Assert raster dimensions and byte count, many distinct RGBA values,
antialiased coverage, deterministic Yoga bounds, state-driven raster revisions,
matching drawn and hit rows, and clean renderer subprocess shutdown.
