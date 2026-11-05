# Viewer movement, yaw, and user scale

Read [library tracking-anchor-and-scale and stage-publication](../dependencies/ovxr.md) for SDK validation, units, latching and ordinal ownership. The library public README specifies latching at begin or render, including no-render boundaries; use [input scheduling](../ovxr-library-runtime/frame-input-order.md).

## Application navigation state

Store one canonical anchor matrix plus an absolute revision. Route desktop
controls, sticks, teleport and gestures through that owner. Preserve the source
USD and its metres-per-unit; user scale changes tracking-space size relative to
the scene, not authored scene units.

A scale-capable editor must create its context with
`ContextDesc.floor_aligned_anchor=False`. The default floor-alignment policy
keeps translation/heading but discards authored anchor scale, so a successful
setter with that default cannot establish working user-scale controls.
Resolve camera/prim spawn into a rigid stage-space Xform first: retain floor
heading and translation, remove source scale and pitch/roll, and convert the
camera's basis for the stage up axis. Then apply navigation's positive uniform
scale. A raw Camera anchor in a Z-up scene is not that stage-space Xform.
Persist this context policy across recreation. An application deliberately using
floor alignment must report user scale unavailable and disable its scale tool.

For a new application, store positive user scale and yaw/translation directly
in the anchor. For a legacy application whose navigation N used to transform
scene roots inversely, preserve its protocol with `A = spawn * inverse(N)`.
Stop writing N to scene roots when enabling this adapter. Applying both would
move twice. Picking now operates on the unchanged scene with rays transformed
through the active anchor; remove any extra legacy inverse-navigation ray step.

Capture `spawn` from the initialized anchor. Application Reset restores that
saved spawn matrix. Public `clear_anchor()` instead resets the managed anchor
to identity; those actions are different in a non-origin scene.

With row vectors, preserve a world-space head pivot h during a yaw/scale delta
by right-composing the current anchor with
`T(-h) * S(scale_ratio) * R(up_axis, yaw_delta) * T(h)`.
Translation-only walking is `A * T(delta_stage)`.
Use current VIEW forward projected onto the floor for walking; a pitch must not
move the user vertically. Record the pivot at gesture begin and calculate
absolute updates from the initial state to avoid packet-loss drift. Snap turns
need threshold/release hysteresis and tool ownership, so release cannot trigger
a queued accidental snap.

For user scale, define the product label explicitly: anchor scale 2 means one
tracked metre spans twice as many stage units, making the user larger relative
to unchanged geometry. A legacy inverse-scene scale has the opposite direction.
Clamp to a product-appropriate finite positive range and test the chosen UI
semantics, not merely matrix invertibility.

## Reproducible anchor math

This NumPy code is application math, not an OVXR import. It uses a rigid
stage-space spawn, positive uniform user scale, row vectors, and radians:

```python
import math
import numpy as np

def translation(v):
    out = np.eye(4)
    out[3, :3] = v
    return out

def yaw_matrix(angle, up_axis):
    c, s = math.cos(angle), math.sin(angle)
    out = np.eye(4)
    if up_axis == "Y":
        out[:3, :3] = ((c, 0, -s), (0, 1, 0), (s, 0, c))
    elif up_axis == "Z":
        out[:3, :3] = ((c, s, 0), (-s, c, 0), (0, 0, 1))
    else:
        raise ValueError("unsupported up axis")
    return out

def pivot_update(begin_anchor, pivot_world, scale_ratio, yaw_delta, up_axis):
    if not math.isfinite(scale_ratio) or scale_ratio <= 0:
        raise ValueError("positive finite scale required")
    h = np.asarray(pivot_world, dtype=float)
    stretch = np.diag((scale_ratio, scale_ratio, scale_ratio, 1.0))
    return (begin_anchor @ translation(-h) @ stretch
            @ yaw_matrix(yaw_delta, up_axis) @ translation(h))
```

For each gesture, retain `begin_anchor` and the measured world-head pivot.
`scale_ratio` and `yaw_delta` are absolute differences from that begin state.
Reject nonfinite pivots/angles and results before calling
the supported managed-anchor setter through the application's publication coordinator. Commit the
application revision only after the setter succeeds. After the next successful
`render_frame`, compare `get_anchor_transform()` with the expected candidate and
mark that revision applied. Preserve queued/applied records separately. Recheck
only a pending revision; do not make redundant native getter calls every frame.
If a newer command supersedes it, never mark the old revision applied from the
new matrix. Save both spawn policy and current user anchor across context-only
refresh; reinstall them on the replacement context before accepting new input.

Let `k = norm(A[0,:3]) / metersPerUnit` for validated uniform-scale A. A desired
physical travel of `speed_m_s * dt` has stage length `k * speed_m_s * dt`.
Convert the measured tracking forward through the up-axis basis and normalized
anchor rotation, project onto the stage floor, normalize, then multiply by that
length. Do not multiply a vector already converted with the scaled anchor by k
again. Use the same k for controller model metre units, marker radius, ray
reach, head height, and hand contact distances.

## Acceptance

Test non-origin spawn/head, non-identity source root, both supported up axes,
non-default metres-per-unit, translation, ±yaw, scale above/below 1, combined
updates, and reset. Invalid matrices must fail without advancing canonical state.
Assert the source-root baseline never changes. With IWER, require real head
motion before and after each operation; confirm controller/hand/ray alignment,
new decoded scene pixels, and unchanged context/service/renderer identities.
After latching requested scales 0.5 and 2, assert the active anchor basis norm
and converted tracked positions/radii change accordingly; verify eye separation
and decoded scene pixels under both Y-up and Z-up with a non-origin spawn.
Test pending versus latched anchor telemetry and reconnect restoration. Unit
math establishes convention; runtime calls plus pixels establish visible motion.

## Transported hand samples

Keep WebXR hand positions, joint radii, and linear velocities in tracking metres
on the wire. Convert one complete batch through one snapshot of the active
anchor before passing it to either gesture recognition or rendering/physics.
Apply stage units and positive user scale to positions/radii/linear velocity;
apply the normalized anchor rotation to orientations/angular velocity. Physical
contact thresholds use the same effective stage-units-per-metre. Packets
explicitly declared as stage-world input must bypass this conversion.

Do not strip uniform scale while inverting the anchor for a tooltip or other
stage-to-tracking placement. Test a translated, rotated, non-unit-scale anchor;
identity-only tests cannot expose double conversion or lost scale.

## Selecting and persisting a default anchor

Offer three application policies: desktop camera, scene origin, or an explicit
scene prim. Persist the selected policy and optional absolute prim path; these
are application configuration, not OVXR descriptor enum values. Resolve the
chosen policy into a stage-world spawn transform, set the managed anchor base,
and reset canonical navigation to identity (user scale 1). The reference app
extracts origin and floor heading from the complete composed transform; it does
not inherit a selected scene prim's scale.
Subsequent navigation is relative to that saved spawn.

Read a scene prim's complete composed world pose without modifying that prim.
Never write the sampled world matrix back as a local transform: parented prims
would apply their parent twice and the source scene would move. A camera spawn
must be converted to the stage-space anchor convention, including removal of
camera-specific up-axis basis adjustments where needed.

Make selection atomic. Validate mode, path, readability, finite rigid/uniform
scale policy, and all derived state before the one native managed-anchor write.
On rejection preserve the prior selected policy, saved spawn, navigation,
revision and source state. Persist only accepted state; a missing prim must not
silently become scene origin. Report accepted versus frame-latched state, and
refresh world-anchored visuals only when the new anchor actually latches.

### Scene hierarchy/world-pose fallback

Read the installed OVStage hierarchy at a coherent committed ordinal, adopting
its latest write floor first. A missing result is not automatically evidence
that the scene prim is absent: package/query semantics can differ. If native
hierarchy/world-pose reads are unavailable or unreliable, obtain a generation-
scoped snapshot from the exact final composed USD in an isolated `pxr` worker.
Include topology, complete world matrices, units/up-axis, source/composite
identity, and generation. Never import a conflicting USD ABI into the renderer.

Use native live data when valid. The worker snapshot is an explicit fallback
for unchanged composed source state, not a claim to observe live Fabric edits.
Overlay authoritative app-owned mutations or invalidate the snapshot when they
change a requested pose. Reject stale generations and rebuild on stage change;
never traverse or spawn a worker per tracking frame. Keep fallback provenance
visible so a lookup failure cannot authorize a fabricated anchor.

## Hand teleport: aim and facing are independent

Use the reconstructable [swing/twist and filter recipe](gesture-filtering.md)
for wrist-steered aim, relative twist facing, and physical-unit smoothing.

Keep the arc origin, heading and elevation responsive to current valid hand
geometry throughout the hold. Destination yaw is a separate scalar; it must
not freeze the arc's starting axis or rotate its aim vector. Avoid interpreting
a joint quaternion's local axis as a portable pointing direction without
runtime evidence.

Capture a measured wrist/palm twist basis when the teleport gesture begins.
Compute signed twist relative to that basis, then apply an explicit gain to
destination yaw and draw the destination arrow with the same result used at
commit. A non-neutral starting wrist pose is still zero for that gesture.
For example, a user-selected mapping of +30 degrees of relative wrist motion
to a full turn has gain 12; starting at -30 and ending at 0 must traverse the
full turn. Do not silently replace that with an absolute -30/+30 range.

Test wrist twist while aiming left/right and near/far, steering while holding a
nonzero yaw, release after motion, and a gesture begun at a non-neutral wrist
pose. Simple world-axis Euler roll mixes steering with twist; quantify that
coupling in measured joint traces before adding filters or constraining motion.

## Teleport preview and release

Use [teleport acceptance](teleport-acceptance.md) for a concrete bounded arc,
walkability/clearance policy, final-query freshness and combined facing/position
commit. Its conservative release policy resolves what happens when a final hit
differs from the last preview.

Native and opaque WebXR input use the same semantic preview/release pipeline.
While the mapped control is held, query the measured aim ray/arc and display a
marker only for a valid walkable hit. On release, retain a final bounded query
or use a correctly correlated result; commit only a current valid preview.
Focus/tracking loss, capture ownership, stage change and disconnect cancel
without teleport. A client-provided position is not evidence of a walkable hit.

Convert physical thresholds and head clearance with effective stage units per
metre, including positive user scale. The normalized aim direction, scene hit,
head pivot and floor normal must all use the same active anchor snapshot.
Apply user scale once; do not reuse pre-scale metre conversion for height,
arc reach or clearance. Test release after motion, scale above/below one,
non-origin/rotated anchors, invalid slopes and lost tracking. Opaque input must
not bypass native walkability or teleport on the initial press edge.

For a direct user-anchor implementation, with measured world head h, stage up u,
and measured head-to-tracking-floor height H in stage units:

```text
feet = h - u * H
translation_delta = accepted_hit_world - feet
candidate_anchor = active_anchor * T(translation_delta)
```

This places the user's floor point at the hit and retains physical head height,
yaw and scale. Do not negate the delta unless adapting an explicitly inverse
scene-navigation protocol. Require a floor-based source reference or an actual
located floor; LOCAL head y is not automatically standing height. A surface
marker may be lifted slightly along its normal to avoid z-fighting; that display
clearance must not move the user above the destination. Tag preview results with
stage generation, input epoch, active-anchor revision and capture frame. On
release, accept only the displayed preview from the current captured gesture;
a cancellation clears it so the next release cannot reuse an old destination.

## World-to-tracking is not automatically world-to-LOCAL

The inverse managed-anchor conversion returns the runtime's selected tracking
space, which may be STAGE or LOCAL_FLOOR. For a compositor or interaction API
using LOCAL, additionally apply the exact source-tracking-to-LOCAL transform
located at the projection display time. Follow the
[spatial UI space contract](../ovxr-composition-ui/spatial-surfaces-and-interaction.md).
Never hide a missing reference-space mapping with a guessed standing height.
