# Spatial surfaces and interaction

## Surface model

The stage panel is one retained raster document plus an authoritative LOCAL
pose. The unconfigured panel publishes its fixed LOCAL default immediately and
is independent of eye telemetry, head motion, USD units, up-axis, camera pitch,
or desktop tracking-anchor transforms. Explicit prim anchoring uses the shared
measured stage-world-to-tracking conversion.

Controller tooltip and prim gizmo are pose-only analytic surfaces appended to
the same per-eye projection mesh. Tooltip pose follows a controller aim pose
located in explicit LOCAL and faces a VIEW pose located in that same LOCAL.
The gizmo follows the selected prim's current complete world matrix converted
through the effective tracking anchor. Missing transforms disable stale UI.

## Symptom triage

A visible controller model proves only that grip or aim supplied the model
policy. It does not prove aim validity. A missing ray and missing tooltip share
this decision tree:

1. Prove controller ownership, focus, aim action activity, and valid location
   flags independently from grip/model tracking.
2. Prove the aim action space was located relative to identity OpenXR LOCAL;
   never relabel a borrowed base-space pose as LOCAL.
3. Prove the LOCAL ray intersects the retained LOCAL panel and record rejection
   reason, UV, and distance.
4. If the ray hits, use per-eye auxiliary composed/clipped bounds, union scissor,
   and depth results to diagnose tooltip visibility.
5. Only after those pass, inspect PQW/domain and client output.

Window chrome owns ray/surface capture while dragging and publishes a newer
LOCAL panel pose without rerasterizing unchanged content. Selection and capture
are separate state machines.

Give each movable window a permanent, independently hit-testable grab bar below
its content. Its stable surface ID, selected highlight, and manipulation target
must agree; enlarging its invisible hit area must not obscure content actions.
For a selected bar, reuse the normal fist/controller transform gesture with
no initial jump, reversible reach gain, and timestamped position filtering.
The [gesture-filtering recipe](../xr-spatial-input/gesture-filtering.md) supplies
the equations and capture/reset rules; no sample helper import is required.

## Controller contract

For the Vision Pro PSVR2 Sense profile:

- R1/right squeeze reveals and updates the hit-bounded scene ray.
- Cross/right primary selects or toggles the scene hit; over the panel it
  activates a stage row on press. UI consumes only this selection edge.
- R2/right trigger moves the selected target. Ordinary panel widgets do not
  consume it. A selected panel grab bar is the explicit exception: its
  manipulation capture owns R2 and suppresses scene movement for that cycle.
  Otherwise route R2 to the selected allowlisted scene target.
- Right stick scrolls while UI-owned; otherwise it turns. During R2 movement,
  stick Y pushes/pulls along the current viewer-object/aim line.
- Left stick locomotes relative to VIEW yaw. Left primary previews and commits
  teleport.

If a deployed CloudXR binding does not deliver a required component, prove the
inactive action in native telemetry before adding a profile-specific fallback.
Do not silently reinterpret controls from button-state absence alone.

## Panel capture and activation

Activation is a per-node policy, not one global click rule:

- Stage rows activate exactly once on press. Retain their stable node capture
  through release for ownership and visual continuity, but never require the
  release ray to remain inside the row and never dispatch twice.
- Ordinary or destructive buttons activate on release over their captured
  stable node. A release over another node never retargets.
- Direct-drag chrome is capture-only and never emits a button action. A permanent
  grab bar may instead select a persistent panel target on press; a subsequent
  transform/fist gesture moves it while preserving the hand-to-panel offset.

For a single-target policy, scene selection clears the selected panel handle
and active panel drag; selecting a different prim replaces selection in one
activation. Route manipulation to the selected panel before scene tools consume
it. A selected highlight is not proof that fist movement is routed correctly.
Separate selection lifetime from pointer capture and raster revision. Test a
move after the pointer leaves the handle and after switching hands/controllers;
focus loss, tracking expiry, hide or relocation cancels capture without a click.
Latch the manipulating hand and surface at grab begin. Closing the other hand
must not steal the drag or reset its offset; test both left-held/right-close
and right-held/left-close before claiming two-hand support.
Deliver both hands' held/open/tracking state every update, including an explicit
invalid/absent observation. Missing tracking cancels panel capture under its
loss policy; it is not an observed open hand or release activation and cannot
satisfy rearm. Do not route only the preferred fist: that can hide the owning
hand's actual release or loss. A non-owning fist already held during a drag
must open before it can start a new drag; do not transfer capture on release.

Capture owns the full cycle even when the bounded ray misses the panel, so a
captured release cannot leak into scene selection. Separate press/release
thresholds require rearm before a new press. Focus, tracking, controller-owner,
session, stage-generation, or transport loss cancels once and consumes the
eventual release without action.

## Routing and lifecycle

Use one immutable controller snapshot. Route panel ownership before CUDA scene
tools. A panel miss leaves the scene event untouched and hides the native
pointer visual; hidden, unavailable, or non-active projection UI cannot consume
input. Preserve a final bounded query for release-edge actions when an async
ray result is needed.

Focus, tracking, controller owner, session, stage generation, or transport loss
cancels capture/latches/filter state and disables controller-attached UI within
one accepted frame. Physical controllers preempt hand gestures. Reconnect is a
new input lifetime and republishes the latest retained panel, selection, and
gizmo state once.

For a convenience adapter sampling after submit, label the input for a later
projection; never correlate it to the projection already submitted. For a split
adapter, sample using the plan's predicted time and record its mapping provenance. The public README specifies anchor latching at begin or render. Convert input after begin through that shared frame mapping and publish eligible scene changes before bind renders.
Follow [frame/input scheduling](../ovxr-library-runtime/frame-input-order.md).
A stationary-controller head sweep distinguishes static space error from
motion-only timing latency.

## Anchor changes and scale

For explicitly stage/world-anchored panels or gizmos, transform the complete
world pose through the full inverse tracking-to-stage transform, preserving
uniform scale in the inverse. A rotation-only inverse produces incorrect LOCAL
position and size after user scaling. Keep pose and physical extent conventions
explicit, applying units and scale once.

After a managed anchor update, republish world-anchored placement once for each
newly latched anchor revision. Do not use pending navigation as if it were the
active frame anchor, and do not rerasterize an unchanged UI document for a pose
change. The default fixed OpenXR LOCAL panel remains at its LOCAL pose; it must
not be moved simply because application navigation changed. Test both policies
with translated, yawed, non-unit-scale anchors and correlated decoded pixels.

## Tracking reference space is not necessarily LOCAL

OVXR's anchor tracking space can prefer STAGE or LOCAL_FLOOR, while the spatial
compositor uses OpenXR LOCAL. Inverting the managed anchor only returns a pose
to that tracking space; it does not remove the floor-to-LOCAL origin difference.
Do not guess a head-height offset or assume identity because both spaces track
the same device.

Resolve the exact shared tracking space (`xr_space`) selected by OVXR and
confirm the consumed projection uses that source. Locate it relative to LOCAL with `xrLocateSpace(source_space, local_space,
projection_display_time)`. Require valid location flags. Use this same measured
mapping for controller rays and world-anchored UI, preserving the full scale-
aware inverse tracking-to-stage conversion before applying tracking-to-LOCAL.
The pose time and source-space identity are part of the contract; a stale
mapping from another context cannot be reused.

Implement this through documented borrowed OpenXR handles/reference-space
metadata where available, or expose the exact mapping already consumed by an
application-owned OpenXR API layer. Such a getter is a custom application API,
not an OVXR library method. Borrowing handles never authorizes another session
or frame owner. If the actual source space/time cannot be established, omit the
world-anchored surface/pointer and report that boundary instead of guessing.
Fixed LOCAL surfaces remain fixed and need no world-to-tracking conversion.

On context/session replacement clear the mapping and await a valid new sample.
Test nonidentity STAGE/LOCAL_FLOOR-to-LOCAL translation with user scale/yaw and
both controller-ray and panel projection, using correlated decoded pixels.

For row vectors, let B be metres/up-axis conversion, A the active managed
anchor, and M the measured tracking-to-LOCAL transform in metres. Then
`world = trackingMetres * B * A` and `localMetres = trackingMetres * M`.
For a world pose, use `world * inverse(B * A) * M`. An existing converter
expecting a LOCAL-space anchor can instead receive
`inverse(B) * inverse(M) * B * A`. Preserve scale in these inverses.

An optional layer getter should report the measured mapping, source-space
handle, display time and frame counter. In the convenience schedule, consume
it after the successful frame for a subsequent frame's input. In the split
schedule, establish the mapping at the current plan time rather than relabeling
the previous submission's sample; follow [frame/input scheduling](../ovxr-library-runtime/frame-input-order.md).
Require its source to match the shared runtime `xr_space`. Capture
mapping even while drawing is gated, so readiness does not depend circularly
on UI visibility. Recompute placement when the effective matrix changes;
coalesce unchanged values rather than republishing on each frame counter.
Anchor revisions remain state-correlation markers, not forced raster uploads.
