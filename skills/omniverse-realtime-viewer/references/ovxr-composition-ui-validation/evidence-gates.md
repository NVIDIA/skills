<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Composition UI Evidence Gates

## Visual

- Scene and UI share the same frame and session.
- The full panel is inside the useful stereo view.
- Text is readable and antialiased at the demonstrated distance.
- Scene outline appears before UI interaction.
- Hover and press are visibly distinct.
- The final outline appears on the UI-selected prim.
- Frames are nonblank, settled, ordered, and distinct.

## Interaction

- Controller pose uses the expected reference space.
- Hover agrees with the targeted Yoga hit bound.
- Press captures the node.
- Activation follows the node's press/release policy exactly once; release
  never repeats a press-activated selection.
- Selecting a scene prim clears panel selection; a selected grab bar can start
  transform capture away from the bar, without another hand stealing ownership.
- UI and runtime selected paths agree.
- A UI miss does not consume scene input.

## Spatial

- Anchor path is correct.
- A stage-attached quad uses `LOCAL`, not `VIEW`.
- Native and server pose revisions agree.
- Left/center/right head motion matches predicted per-eye motion.
- Pose/capture sequences reject stale frames and include repeated-center jitter.
- Framebuffer origin, actual reference space, and foveation warp state are recorded.
- Physical and raster sizes are separate.
- Quad corners are not near-clipped or outside both eyes.

## Runtime

- Layer state is visible.
- Last result is `XR_SUCCESS`.
- Clearly named frames-seen and frames-injected counters advance.
- Swapchain format is valid.
- Ownership flags say the client and OVRTX do not draw the UI.
- No session restart occurs during the proof.
- Settled static frames advance upload skips without advancing image uploads.

## Client-frame correlation

For each demonstrated pose or interaction, record one ordered client-frame
chain: the requested pose is accepted, the matching frame metadata is observed,
the client allocates or updates its XR texture for that frame, and the expected
UI pixels appear in the per-eye framebuffer regions. Correlate the identifiers
or timestamps at each link; a video callback's arrival time is not a substitute.

Decoded video, a playing media element, nonblank video samples, or advancing
decode/frame counters prove only that the transport produced media. They do not
prove that the XR client joined that media to the requested pose and metadata or
presented it through its XR framebuffer.

Every reconnect is a new client lifetime and must independently re-establish
this complete chain before it can pass. If an emulator continues to decode video
but cannot allocate/update its XR texture or produce correlated per-eye
framebuffer pixels, report that as a fail-closed emulator boundary. Preserve the
diagnostic and require native-device evidence; do not mask it with a fallback UI
path, a session/service restart, or a transport-only pass.

## Reject Despite Exit Zero

Reject separate UI-only and scene-only captures, late one-surface frames,
clipped or unreadable panels, enlarged low-resolution pixel art, broad color
tests without predicted ROIs, telemetry-only selection, or evidence produced by
an older API-layer binary than the tested source. Also reject decoded-video or
counter-only reconnect evidence that does not establish client-frame correlation.

Lead handoff with the WebP, then validation JSON and test summaries. State the
action path, selected prim, raster and physical sizes, and `XrResult`.
