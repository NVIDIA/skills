<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Validate Projection-Integrated Spatial UI

## Triggers

Use this reference to prove world panels, controller callouts, transform
gizmos, window chrome, scene depth, foveation-domain mapping, and interaction
through OVXR/OpenXR and CloudXR.

Telemetry establishes internal boundaries. Emulator or desktop evidence proves
automation. Only correlated decoded-client or headset pixels prove the final
display.

For an agent-owned IWER session, implement the application-owned supervisor
specified by [owned orchestration](owned-orchestration.md). For headset,
tracked-hand physics and desktop-mirror claims, also follow
[viewer evidence](viewer-evidence.md).

Read [library validation](../dependencies/ovxr.md) for SDK/ABI gates.
This reference owns integrated viewer acceptance.

Use [evidence gates](evidence-gates.md) for the required functional, visual,
and performance records and [frustum placement](frustum-placement.md) for
projection fixtures, spatial placement, and client-image regions of interest.

## Hardware-Free Gate

Before using a headset:

1. Record the source revision, native integration build identity, manifest or
   hook selection, and intended runtime configuration.
2. Run unit fixtures for asymmetric FOVs, nonzero subimages, unequal eye
   extents, homogeneous clipping, normal/reversed depth, explicit `LOCAL` rays,
   and nonidentity tracking anchors. When testing warped composition, also
   require PQW identity and round-trip fixtures using the supplied vendor contract.
3. Build the native integration with strict warnings and run ABI, discovery,
   swapchain lifecycle, image matching, and resource-retirement tests.
4. Exercise UI capture, chrome drag, scene miss fallthrough, focus/tracking
   cancellation, disconnect/reconnect, and selection-to-gizmo convergence.
5. Verify a settled unchanged raster causes no texture upload, primary-eye
   copy/transfer, same-frame fence wait, Fabric polling, or unbounded logging.
6. Confirm the launched process selects the exact tested build and profile.

Use a hardware emulator such as IWER when available to automate head and
controller paths, but do not make it a dependency of the generated app or
claim that its camera proves the physical client.

## Owned IWER Readiness

Order an owned run around the runtime's connection dependency:

1. Start and attribute the web endpoint and CloudXR listener.
2. Wait for the renderer to begin attaching OVStage.
3. Start the IWER driver so the client can connect.
4. Wait for renderer/stage readiness, then XR-context readiness.
5. Require one submitted successful frame before accepting driver evidence.

Do not wait for a client-bound XR context before starting its driver; that can
deadlock. Do not start the driver before the listener; that creates a retry
race and weakens process attribution. The owned supervisor records each sparse
transition, bounds each wait, stops only its recorded process groups, and
verifies that their listener ports are released.

## Continuous Interaction Story

For the combined editor workload, use the
[experience acceptance stories](../ovxr-composition-ui/experience-quality.md#accept-the-complete-experience)
as well. They cover selection replacement, both-hand cancel/rearm, fist-moving
grab bars, settings acknowledgement, passthrough materials and mirror cost;
record functional, visual and performance outcomes separately.

Capture one connected lifetime showing:

1. OVRTX scene and world panel together.
2. A center-to-edge head/gaze sweep in both eyes.
3. Controller ray miss continuing to scene selection.
4. Controller hover and activation of a panel node.
5. Authoritative selection and OVRTX outline convergence.
6. A prim-attached gizmo, enlarged invisible hit volume, self-occlusion, and a
   translate/rotate/scale interaction appropriate to the active tool.
7. Window-chrome capture moving the panel without rerasterizing it.
8. Focus or tracking loss canceling capture, followed by reconnect convergence.

Do not splice UI-only and scene-only sessions when the contract is their shared
interaction and depth.

## Live Client Gate

Launch the intended client profile with the selected production projection path
and record its negotiated foveation mode. Test enabled foveation when that is a
supported target; without the vendor mapping contract, warped-depth composition
remains blocked. An explicitly selected nonfoveated baseline can validate the
rectilinear feature set but does not prove warped support. See
[foveation prerequisites](../ovxr-composition-ui/foveation-and-edge-math.md).
A desktop mirror is optional diagnostic output, not the
authority and not a reason to render a second OVRTX product. If a runtime needs
a publisher for connection lifecycle, keep it control-only or strictly rate
limited and prove that it does not add an eye render.

Correlate each observation with the same source/input epoch, exact submitted
display time, surface revisions, per-eye domain, draw counters, and decoded
frame identity. Record missing measurements as unknown.

Exercise:

- 0.15 m through 5 m, including near-plane crossing;
- the outer 15 percent of both eye images;
- head translation/rotation and stationary-controller head sweeps;
- scene geometry in front of and behind opaque or alpha-masked UI;
- panel alpha outside covered geometry;
- controller model, grip, aim ray, and physical tooltip alignment separately;
- tooltip, gizmo, chrome, selection, manipulation, and hand/controller owner
  replacement when supported.

## Triage By First Failed Boundary

1. Application state accepted.
2. Surface pose and input located in the declared `LOCAL` space/time.
3. Finite clipped per-eye mesh composed in the negotiated domain, with verified
   PQW mapping when warped composition is enabled.
4. Combined draw/scissor included its indices and bounds.
5. Color/depth/alpha contract valid and measured fragments changed output.
6. CloudXR decoded the matching primary frame and depth.
7. Headset pixels showed the expected surface and alignment.

Do not jump from an accepted pose to a renderer rewrite or from a server draw
counter to a client pass.

## Evidence And Completion

Keep a sanitized validation record with versions, build identity, configuration,
poses/times, FOV/subimage/PQW/depth state, actions, revisions, counters, timing
windows, rejection reasons, and raw settled captures where permitted. Open and
inspect every image; reject clipped, unreadable, stale, blank, or one-surface
evidence even when automation exits zero.

Report pass only when pixels and telemetry agree. State skipped gates and their
concrete blockers. Stop the exact supervised client/runtime processes after the
run and verify their listener ports are released; never use broad process or
shared-memory deletion patterns.

## Library-only harness

Use [library-only validation](library-only-validation.md) to implement the
CXR.js/IWER sender, exact reference-space ownership, correlated applied-state
and decoded-pixel proof, mirror checks, and the three update lanes without a
sample checkout. Generate the application's supervisor from
[owned orchestration](owned-orchestration.md).
