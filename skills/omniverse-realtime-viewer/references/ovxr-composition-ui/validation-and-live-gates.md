# Validation and live gates

Version boundary: INSET refers to an older binding. Inspect the installed
`LocalFoveationMode` enum; current OVXR uses AUTO/DISABLED/WARPED. Never reuse old
integer values or treat an INSET→DISABLED compatibility alias as foveated proof.
Custom compositor exports are application APIs, not shipped OVXR entry points.


## Before headset use

1. Add or update the narrowest failing unit/native test before fixing a bug.
2. Run focused Python tests for server UI, controller runtime/input, native
   OpenXR input, and layer bridge.
3. Build the native layer and run all CTest targets, including ABI/discovery.
4. Run desktop tests and build. Use a workspace-local pytest base temp on
   restricted Windows hosts.
5. Confirm the exact staged DLL/manifest and configuration selected by the
   launched child process.

Math coverage must include asymmetric eyes, offsets/extents, clipping, both
depth conventions, rectilinear/PQW mapping, explicit LOCAL controller rays,
panel capture/drag, input ownership, tooltip loss, selection-to-gizmo
propagation, and reconnect convergence.

Interaction coverage must also prove: stage-row activation on press; no repeat
while held or on release; off-panel captured release remains UI-owned; release
over another node never retargets; focus/tracking/reconnect cancellation
requires rearm; and chrome drag emits no button action. Reattach must republish
an already-selected gizmo when its composed transform remains resolvable.

## CXR.js/IWER and physical evidence

For an available browser runtime, use CXR.js plus IWER to drive the actual XR
application, not a detached synthetic panel. Capture decoded stereo pixels
with the same accepted input epoch and server frame. Prove a fixed LOCAL panel
and a world-anchored panel separately under yaw, translation and non-unit user
scale; include a nonidentity tracking-to-LOCAL origin. Exercise hover, captured
release, scene selection, tooltip, gizmo and reconnect. Browser emulation proves
these application paths, not physical controller models, button labels, native
headset bindings or headset optics. Generic fallback controller presentation
must be identified as fallback; do not infer hardware from a Touch binding.

Mirror-only pixels cannot prove primary-projection UI. In the measured package
combination, disabled-foveation mirror capture worked while warped submission
with independent unwarped readback produced no capture. That readback limitation
is separate from the native projection compositor and must not silently change
the requested submission mode.

## Vision Pro live run

Launch the approved Vision Pro profile with native CloudXR, Vulkan,
`projection-depth`, the installed supported foveation mode, and desktop mirror
disabled. Record the exact mode and negotiated image domain.

During connection, correlate:

- controller owner/focus/tracked/aim-valid and required action activity;
- pointer LOCAL ray availability, hit, UV/distance, phase, and ownership;
- tooltip/gizmo enabled revisions and per-eye composed/clipped counters;
- each auxiliary's accepted -> mesh -> draw/scissor -> depth/fragments ->
  observed-output boundary, leaving unavailable measurements `unknown`;
- panel world lock, image/PQW domain, source depth, fallbacks;
- zero-copy/transfer/wait/upload invariants and timing windows.

Exercise both eyes' outer 15 percent with head and gaze motion, center-to-edge
sweeps, 0.15-5 m range and near crossing, scene occlusion, alpha outside the
panel, controller tooltip, scene selection/gizmo, chrome drag, focus loss, and
disconnect/reconnect.

A pass requires headset observation or decoded client pixels correlated with
the same server frame and pose. Submission telemetry alone proves only that
work was accepted. Record failures with the first contract boundary that did
not hold. Keep a user-requested interactive evaluation available; bounded automated runs
should clean up their owned processes after capturing evidence.
