<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Controlled WebXR spatial input

Read [the native input route](README.md) first. This is an optional controlled
browser telemetry adapter; it does not replace native OpenXR tracking.

This section applies only when the app owns a controlled WebXR client (for
example an XR UI or interaction proof client) and deliberately sends spatial
samples over the application channel. Ordinary browser mouse/touch camera input
still uses AppStreamer/NVST's native input path.

Keep the ownership boundary explicit:

```text
OVXR/OpenXR runtime: predicted eye poses, eye render products, submission
controlled browser: observable WebXR viewer/input/hand poses from XRFrame
application converter: tracking metres/up-axis -> stage units/up-axis/anchor
application semantics: navigation, picking, UI, dedupe, evidence labels
```

Do not claim that a controlled-client `getViewerPose()` sample is OVXR's private
predicted eye transform. Label it as a controlled WebXR viewer observation.
OVXR may remain a pixels-and-poses runtime while Python or another application
layer owns the canonical conversion and interaction semantics above it.

### Matrix layout contract

`XRRigidTransform.matrix` is a 16-element column-major WebXR/WebGL array. An
application that uses row vectors can interpret that same flat array as the
mathematical transpose: translation remains at indices 12-14, so do **not**
reorder the 16 values before passing them to a converter designed for this
contract. Reordering and then applying row-vector math transposes twice.

Pin this with fixtures containing both translation and rotation. The conversion
helper should:

- reject non-finite or non-16-element poses;
- conjugate pose operators through the tracking-to-stage basis so unit scaling
  affects translation, not the controller/hand axes;
- convert ray origins as points, ray directions as unscaled basis vectors, and
  normalize directions after conversion;
- use one implementation for viewer, controller, hand-wrist, velocity, and ray
  data rather than growing per-feature Y-up/Z-up variants.

If the destination math library instead expects row-major storage with column
vectors, adapt once at that boundary and name the adapter. Never rely on an
unwritten transpose assumption.

### Hand presentation is not contact geometry

Keep the tracked joint/contact radius as semantic telemetry. A visible hand
marker may apply a separate presentation scale or minimum visible radius, but
must not overwrite the contact radius merely to make the hand readable in the
viewport. Publish both values when evidence or debugging needs them:

```text
tracked/contact radius -> picking, proximity, physics/contact semantics
presentation radius    -> viewer-owned sphere/glyph size only
```

This separation allows legible hands without silently inflating collision or
pinch thresholds.

### Real XRFrame proof gate

Spatial proof must exercise the production sampling path inside a real XR
animation callback. From the same declared `XRReferenceSpace`, capture:

- `frame.getViewerPose(referenceSpace)` for the controlled viewer sample;
- `frame.getPose(inputSource.gripSpace, referenceSpace)` and/or target ray pose;
- `frame.getJointPose(inputSource.hand.get('wrist'), referenceSpace)` for hands;
- the ray origin/direction actually used by the production hit-test path.

Send the bounded/versioned production message and retain the accepted server
response. Do not substitute a fabricated matrix bundle, DOM click, or direct
handler call. If controller and hand modes cannot coexist in the emulator,
capture explicit phases with their frame time, input epoch, and reference space
instead of pretending they were simultaneous. A strong gate correlates these
samples with accepted canonical state and visible decoded pixels; it still does
not broaden the claim to OVXR-private eye conversion.
