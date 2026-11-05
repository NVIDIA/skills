<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Projection-Integrated Spatial UI

## Triggers

Use this reference for OVXR/OpenXR panels, controller callouts, transform
gizmos, movable window chrome, CloudXR spatial UI, Skia/Yoga surfaces, or UI
that must remain crisp and world locked inside an OVRTX scene.

Use **projection-integrated spatial UI** for the product capability. It
describes UI drawn into the submitted primary projection without the ambiguity
of "injected UI." Use **in-place primary-projection compositor** for the Vulkan
implementation.

Read the focused references needed by the feature:

- [Experience quality](experience-quality.md) for a coherent panel/toolbelt
  design, discoverable gestures, surface-specific sharpness and acceptance;
- [Editor document model](editor-document-model.md) for stage tree, settings,
  toolbar, stable window IDs and selection/capture state;
- [async raster pipeline](async-raster-pipeline.md) for worker deadlines,
  bounded multi-window scheduling and coherent image/hit-map publication;
- [native build/discovery](native-build-and-discovery.md) for the application
  layer's exports, build products and manifest;
- [projection/depth equations](projection-and-depth.md) and
  [worked passes/synchronization](compositor-passes-and-sync.md) for actual
  depth encoding, per-eye clip math, private gizmo depth and texture retirement;

- [SDK frame backend](sdk-frame-backend.md) to retain existing styling/input
  while adopting begin/bind/submit and application-owned primary depth;
- [depth tools](depth-tools.md) for volumetric rays/teleport, foreground gizmos,
  UI-on-top ordering and bounded analytic edge coverage;

- [primary-projection.md](primary-projection.md) for color/depth ownership,
  clipping, synchronization, and presentation tradeoffs;
- [foveation-and-edge-math.md](foveation-and-edge-math.md) for per-eye
  projection, subimages, PQW domains, and edge distortion;
- [skia-yoga.md](skia-yoga.md) and [raster-abi.md](raster-abi.md) for dense
  retained panels;
- [spatial-interaction.md](spatial-interaction.md) for anchors, rays, chrome,
  controller callouts, and gizmos;
- [performance-and-telemetry.md](performance-and-telemetry.md) for the hot-path
  budget and proof boundaries;
- [AR compositing](../ovxr-library-runtime/ar-compositing.md) for passthrough
  backgrounds and preserving renderer alpha;
- `references/xr-spatial-input/README.md` for controller, hand, navigation, and scene
  tool ownership;
- `references/ovxr-composition-ui-validation/README.md` before claiming live
  success.

## Choose A Presentation Policy Per Surface

Do not force every surface through one backend. Keep a renderer-independent,
revisioned document and choose presentation from its spatial contract.

| Policy | Best for | Important tradeoff |
| --- | --- | --- |
| Primary projection | World panels, gizmos, tooltips, chrome that need scene depth and CloudXR reprojection | Must match each eye's projection, depth, foveation domain, and frame time |
| OVRTX renderer tail | Scene-native geometry and tools when a supported renderer seam exists | Best scene integration, but do not invent a private renderer hook |
| Additional OpenXR layer | Intentional HUD or always-on-top content | A quad has no per-fragment projection depth and may not share streamed reprojection |
| Client-native surface | Maximum text fidelity on a controlled client | Requires a proven anchor transform, reconnect convergence, and explicit occlusion policy |

The primary-projection path is the default when a surface must occlude and be
occluded by the streamed OVRTX scene. An additive quad is not a safe fallback
for that contract. Hide the surface and report the failed boundary instead of
silently changing it to head locked or always on top.

## Preserve Ownership

```text
OVRTX / OVStage          scene, authoritative transforms, selection
OVXR                     session, frame loop, spaces, predicted timing
UI model                 semantic nodes, revisions, actions
Yoga + Skia              layout, hit bounds, retained RGBA
OpenXR integration       primary-image lifetime and final composition
CloudXR/runtime/client   transport, runtime foveation, display
```

Keep one OVXR-owned OpenXR session and frame loop. A composition integration
may defer release of the exact primary color/depth images, draw once, release
each image exactly once, and forward the original projection. It must not call
`xrWaitFrame`, `xrBeginFrame`, or create a competing projection owner.

Keep the semantic surface independent from presentation:

```text
SpatialSurfaceDocument
  id, documentRevision, presentationPolicy
  anchor { stagePath | localPose | trackingPose }
  physicalSizeMeters, rasterSize, occlusionPolicy, visibility
  nodes[] { stableId, role, bounds, labelOrIcon, style, action }
  hitRegions[]
```

Raster revision and pose revision are separate. Moving a panel, tooltip, or
gizmo must not rerasterize or upload unchanged pixels.

## SDK composition boundary

Read [the library composition-integration skill](../dependencies/ovxr.md).
The release supports borrowed Vulkan handles and additional native composition
layers. Use [the SDK integration boundary](sdk-frame-backend.md) for
viewer-owned resources and retained UI. Match the release submit signature;
additional layers and Vulkan access do not supply primary scene-depth composition.

## Build Workflow

1. Define a revisioned, renderer-independent surface document and semantic
   action protocol.
2. Establish one stage-world/tracking/OpenXR `LOCAL` conversion and test it
   with nonidentity anchors, stage units, and up axes.
3. Build one Yoga tree for both painting and hit bounds; rasterize real fonts
   and icons with Skia off the XR thread.
4. Publish immutable image and pose revisions through copied, bounded ABIs.
5. Intercept only the matching primary projection images and validate their
   color/depth subimages and acquisition state.
6. Clip and project each surface independently for each eye, then map through
   the proven submitted foveation domain.
7. Submit bounded scissored in-place batches per eye with attachment
   `LOAD`/`STORE`; account for depth-policy splits and do not copy full eyes.
8. Sample VIEW, grip, aim, and panel interaction against an explicit `LOCAL`
   space and the exact submitted display-time contract.
9. Route semantic selection and transform actions through the authoritative
   OVStage/OVRTX services.
10. Validate math and lifecycle without hardware, then correlate telemetry
    with decoded or headset pixels on the intended client.

## Fail Closed

Omit the spatial draw for a frame when the image lifecycle, synchronization,
depth convention, projection mapping, foveation domain, or pose-time contract
is unknown. Reusing the last complete retained UI generation is acceptable;
blocking the XR thread or guessing a coordinate domain is not.

Do not disable foveation, add a visual filter, perform a full-eye unwarp/copy,
or change world UI to a head-locked quad to conceal a math defect.

## Completion Gates

- User USD files remain unmodified; generated surfaces live in viewer-owned
  session/runtime state.
- One owner steps OVRTX, submits OpenXR frames, and publishes transform batches.
- Primary color/depth subimages, per-eye FOVs, foveation domain, and display
  time are correlated for the same frame.
- Stable raster revisions cause no texture upload; pose-only updates remain
  bounded.
- No primary full-eye copy, same-frame fence wait, stage traversal, or
  frame-rate logging exists in the steady path.
- A panel miss continues to scene interaction; captured UI input never leaks.
- Focus, tracking, owner, session, stage-generation, and transport loss cancel
  capture and hide stale controller-attached UI.
- Hardware-free tests cover asymmetric eyes, nonzero subimages, clipping,
  depth conventions, foveation mapping, interaction capture, and reconnect.
- Live pixels prove world lock, edge stability, depth, readability, tooltip and
  gizmo visibility, and selection convergence. Submission telemetry alone is
  not visual proof.

## Standalone construction

Start with [library-only compositor](library-only-compositor.md) for the
application-owned OpenXR layer, measured tracking-to-LOCAL conversion, full
anchor scale, and explicit rectilinear/PQW gates. Use
[spatial surfaces](spatial-surfaces-and-interaction.md),
[telemetry contract](telemetry-contract.md), and
[validation gates](validation-and-live-gates.md) for current implementation
contracts. OVXR supplies the XR session and frames; these UI modules must be
implemented by the application.
