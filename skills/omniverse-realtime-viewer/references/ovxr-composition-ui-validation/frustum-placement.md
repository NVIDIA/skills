<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Per-Eye Frustum Placement

## Capture Live Eye State

For each WebXR eye, record its inverse view transform, projection matrix,
viewport, `depthNear`, and `depthFar`. WebXR uses column-major storage with
column vectors:

```text
eyePoint = eyeFromLocal * localPoint
clip = projection * eyePoint
ndc = clip.xyz / clip.w
```

Map NDC into that eye's viewport. Invert framebuffer Y when mapping into page
or screenshot coordinates. Never average stereo projection matrices.

Corner bounds are only a quick diagnostic when an edge crosses `clip.w <= 0`
or another clip plane. Homogeneously clip the quad polygon before perspective
division, then derive the visible per-eye ROI.

## Project Four Corners Per Eye

Build the quad's four local corners, transform them by the quad pose, and
project each independently for each eye. Classify behind-camera, before-near,
near-clipped, outside-frustum, partially visible, and fully visible.

Define:

```text
combinedVisible = leftVisible || rightVisible
stereoSafe = leftFullyVisible && rightFullyVisible
```

For a conventional projection matrix `P`, naive FOV diagnostics are:

```text
tanLeft  = (P[8] - 1) / P[0]
tanRight = (P[8] + 1) / P[0]
tanDown  = (P[9] - 1) / P[5]
tanUp    = (P[9] + 1) / P[5]
```

Use live depth state instead of inferring near and far planes from the matrix.

## Correlate Pixels

Capture settled hidden and visible frames. Difference the buffers and require
most changed energy inside predicted per-eye regions of interest. Compare
measured and predicted motion at left, center, and right head positions.

A fixed `LOCAL` quad moves opposite head translation on screen. A `VIEW` quad
stays screen fixed.

Capture a repeated-center control to measure decode jitter. Scale the measured
motion threshold from predicted pixel shift, framebuffer probe density, jitter,
and a small quantization floor; do not use a fixed probe-column cutoff.

Record runtime warp/PQW state. Linear NDC-to-framebuffer ROIs are invalid after
an unknown foveation warp; read
`references/ovxr-foveation-and-display/README.md`, apply the reported mapping,
or repeat with local foveation disabled.

Persist matrices, viewports, projected corners, classifications, ROIs,
measured difference bounds, pose revision, and frame counters.
