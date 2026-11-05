<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# XR physics interaction

Use [Newton integration](../newton-physics/README.md) for the engine adapter
and [XR simulation lifecycle](simulation-lifecycle.md) for execution ownership.
The simulation owner controls solver state; the XR frame owner publishes copied
completed snapshots and delegates rendering to OVXR.

Check [the ovnewton dependency gate](../dependencies/nvidia-runtime.md#ovnewton-and-newton)
before enabling these interactions. Resolve the selected public bridge against
the XR runtime package set and validate the intended simulation/render/input
path. Keep interactions disabled until that integration is qualified; the
patterns below and package compatibility alone do not qualify XR behavior.

## Discover dynamic interactables from the binding

Do not require a domino-style scene to enumerate an `interactable` list. At
attach, build a generation-scoped index from the binding's body labels and
Newton body flags, then associate each renderable descendant with its nearest
owned body ancestor. Dynamic bodies become interaction candidates by
capability; exclude static bodies, hand proxies, remote-anchor helpers, hidden
runtime geometry, and anything explicitly denied by application policy.

Resolve a ray hit by walking ancestors through that index. Rebuild it after a
stage generation or model topology change. Authored metadata can narrow the
candidate set, but manual per-prim marking must not be the prerequisite for
opening an arbitrary physics stage and interacting with its dynamic bodies.

## Remote and direct hand interaction are exclusive

Expose one authoritative interaction mode, not independent toggles that can
both become true:

| Mode | Input and Newton ownership |
|---|---|
| Remote ray/fist | A ray selects a dynamic body and a fist begins/updates a solver-owned remote hold. Direct hand colliders are disabled or parked. |
| Direct colliders | Tracked kinematic palm/finger proxies make Newton contacts. Ray/fist capture is disabled and any remote hold is released before proxies become collidable. |

A mode transition first releases ownership from the old mode on the simulation
owner, then enables the new one. Selection may persist as UI state, but a hold,
contact rig, or gesture latch does not cross the transition.

Implement a remote hold with a predeclared kinematic anchor and a predeclared
Newton fixed joint (or the installed solver's verified equivalent). Queue the
anchor target to the simulation owner; let Newton solve collisions while the
payload remains dynamic. On fist-open, tracking loss, target invalidation, mode
change, reset, or disconnect, disable the constraint and preserve the body's
solver-owned `body_q`, linear velocity, and angular velocity. That preserved
velocity is the throw/drop behavior; do not calculate a replacement from hand
motion after release.

Keep cleanup ownership-specific. Direct-proxy maintenance may clear direct
contact/grasp state and park direct colliders, but it must not call a shared
“end hold” path that cancels an active remote fixed joint. Remote cleanup owns
the remote constraint and may release it only on a remote release condition or
a lifecycle transition that ends all interaction ownership. Report active mode,
hold phase, target body, and rejection/release reason in bounded status.

A palm sphere is the smallest useful proxy for a push, but it is not a hand
grasp. For a grasp, build the smallest **articulated collision rig that can make
the required opposing contacts**: normally a palm plus spheres or capsules for
the thumb and the few finger segments that can close around the target. Derive
those bodies from real client joint poses; do not copy the visual hand mesh into
physics and do not invent a server-side closed-hand pose. Shape dimensions,
collision margin, and joint selection are scene-scale inputs that need local
measurement, not constants to promote from one demo. The tracked poses provide
the rig's articulation; independent kinematic phalanx proxies do not need
solver joints merely to reproduce that motion.

Keep every hand body and shape in a runtime/session layer so the authored USD
remains byte-identical. Give the proxies stable labels and create their Newton
topology before binding. Park them outside the scene while inactive, at finite,
moderately distant, **distinct** positions. Parking every proxy at the same far
coordinate can create an overlapping kinematic pile and non-finite XPBD state
during focus loss or reconnect. A latest-wins host DTO may feed poses into the
simulation-owner control hook, with
these invariants:

- sample canonical joint poses from the actual WebXR/OpenXR frame and retain
  source, sequence, and timestamp metadata through transport;
- perform the tracking-space-to-stage-space conversion exactly once, before
  the DTO enters Newton ownership, and reject a second conversion downstream;
- update the driven body pose in **both** Newton state buffers before the next
  step, with finite/bounds checks, so a later buffer swap cannot restore a stale
  hand; and
- validate each driven body with Newton's `BodyFlags.KINEMATIC` bit, not inverse
  mass. In the measured ovnewton 0.1.1 path an authored kinematic body can still
  report nonzero inverse mass.

If required joints disappear, become stale, or fail a capability check, release
the grasp through the selected physics mechanism and park the proxies. Do not
hold the last pose through focus loss, disconnect, or hand reacquisition.

## Grasp with contact and friction first

Prefer a grasp owned entirely by Newton contact: the domino or payload remains a
dynamic body while opposing kinematic finger/thumb proxies close, Newton
generates contacts, and the selected solver transfers force and friction. The
gesture classifier may gate *intent* using real joint distances, velocities,
and hysteresis, but a closed-hand boolean is not an attachment and cannot prove
a grasp. Require named, physically plausible opposing contacts before reporting
one.

This pattern has a public, installable precedent: the Newton 1.4 test suite
contains a kinematic gripper regression in which two moving kinematic box
fingers close on and lift a dynamic capsule through contact/friction. Treat
that as evidence for the pattern, not as universal tuning. Re-run the relevant
installed example with the chosen solver, gravity, object mass, timestep, and
scale, then tune material friction, contact stiffness/damping, substeps, and
iterations against measured slip, penetration, stability, and input latency.

Maximize Newton ownership, not the raw number of operators. More contact shapes
or constraints are not automatically more accurate; add only the native bodies,
shapes, materials, contacts, and solver controls needed to reproduce the tracked
motion and physical grip.

## Native constraint fallback is version-sensitive

If repeat trials show that a contact/friction grasp cannot meet the application's
declared retention and latency bounds, a public Newton joint or constraint may
be a fallback. Inspect the exact installed Newton examples and selected solver
before choosing it. The tested Newton 1.4 package exposes fixed/D6 joint builders
and exercises runtime `model.joint_enabled` changes, but that observation is not
a cross-version, cross-solver promise and does not establish an API for changing
joint frames at grasp time.

The ovnewton binding freezes body/joint counts and labels at attach. Therefore:

- predeclare any fallback joint, its labeled stage counterpart, and required
  buffers before binding; do not add or remove a joint when the fingers close;
- when that predeclared joint connects a contact proxy directly to the payload,
  keep parent/child collision enabled explicitly. In USD set
  `physics:collisionEnabled = true`; for a code-built Newton 1.4 joint use the
  public `collision_filter_parent=False` builder option after verifying the
  installed signature. Disabling `model.joint_enabled` does not undo a
  parent/child collision filter, so omitting this can make one side of the
  required opposing-contact pair impossible before acquisition;
- start it disabled and enable it only after the real closure gesture and
  qualifying Newton contacts, using a public mechanism verified for the
  installed version and selected solver;
- establish anchors without snapping the payload or injecting unbounded
  constraint error. If the installed public API cannot express safe anchors for
  the current grasp pose, fail closed instead of using an app-side attachment;
  and
- verify that the chosen graph-capture path observes enable/disable changes, or
  keep capture off for the grasp path.

Never parent the payload under a hand prim, copy the hand pose onto it, teleport
it each frame, or mark it kinematic while held. Those shortcuts remove the
payload from the physical problem and make release velocity artificial.

For a contact grasp, opening the tracked fingers should remove the retaining
contacts while the payload remains in the solver. For a verified native-joint
fallback, disable the joint on release. In both cases preserve the dynamic
body's solver-owned pose and `body_qd`; do not overwrite velocity at release.
Validate the resulting free flight, gravity, floor contact, and settle rather
than declaring success when an application grasp flag changes.

After opening, withdraw the real tracked hand from the drop path. Open collision
proxies are still physical geometry and can silently shelf the payload, turning
a correct release into an apparent gravity or floor-contact failure.

Keep hand/contact proof in a bounded ring distinct from dense background
contacts such as every domino touching the floor. A shared small ring can add
the fingertip pair first and then evict it with dozens of floor pairs in the
same substep, falsely reporting that the visible impulse had no named contact.
Canonicalize every retained contact normal to proxy-to-payload orientation
(Newton 1.4 reports `rigid_contact_normal` from shape 0 to shape 1), then require
a genuinely opposed thumb/finger normal pair from the **same physics frame and
input sequence**. Historical contacts from different frames can otherwise form
a fictitious pair. Merely seeing one thumb label and one finger label is not
enough: both proxies can touch the same face. Once a grasp target is selected,
retain that target's later static-world/floor contacts in a separate focused
ring so unrelated grounded bodies cannot evict the release evidence.

Bound diagnostic history at the producer and enforce a transport-size guard at
the final data-channel sender. Retain only the newest useful events/readbacks,
measure the encoded payload, and fail closed before transport. For the measured
CloudXR 6.2 budget and its scope, follow
`cloudxr-opaque-data-channel/README.md`.

Acceptance for an XR physics interaction is an end-to-end physical **and**
visual contract:

```text
real client joint samples and byte counts
  -> server receipt and bounded hand-proxy updates
  -> named Newton contact pairs (and verified native constraint, if used)
  -> measured payload lift with bounded relative slip
  -> live-stage pose publish at ordinal N
  -> renderer consumes ordinal >= N and submits successfully
  -> nonblank changing decoded pixels show the same lift
  -> real opening gesture releases the dynamic payload
  -> Newton gravity and floor contact produce a visible fall and settle
```

Capture open/contact/closed/lifted/release/floor-contact/settled frames and one
continuous animation. Correlate them with bounded world-transform readback,
contact labels, lift height, relative-pose slip, release linear/angular velocity,
physics and renderer ordinals, and decoded-pixel changes. Repeat with approach
jitter, and include negative controls: an open hand near the object and a closed
hand away from it must not grasp; one-sided contact must not masquerade as a
retained grip; hand loss must release safely. Exercise focus loss,
loss/reacquisition, disconnect, and fresh-client reconnect without using
OVRTX/OVStage teardown as a reset mechanism.

## Breakable cable pulls

Model a cable's compliance and its failure threshold as separate facts. For a
circular Newton rod, derive the native per-joint stiffness from the material and
discretization (`EA/L` stretch, `EI/L` bend, `GJ/L` twist) with the installed
`newton.utils.create_cable_stiffness_from_elastic_moduli` contract. Keep a
dataset row for material id, density, radius, segment rest length, Young's
modulus, Poisson's ratio or shear modulus, and an independently sourced ultimate
tensile stress, break force, or break strain. Do not rename stiffness as
"breaking power," and do not compare cables whose radius or discretization
changed without accounting for area and segment length. Perform the material
calculation in one declared unit system, convert its dimensional outputs to
stage units exactly once, and record that conversion with the dataset.

Do not assume Newton automatically fractures rods. In a Newton checkout where
the selected cable solver demonstrably honors runtime `model.joint_enabled`,
predeclare every cable body and joint before ovnewton attach and break the
selected link by disabling that existing joint on the simulation owner. This
preserves the frozen ovnewton topology; deleting a body, changing counts, or
rebuilding the binding at pull time does not. Verify the exact installed solver
with a headless enable/disable regression before exposing the interaction, and
fail closed if no public, solver-observed break operation is available.

For a maximal-coordinate CABLE joint, measure the selected joint itself. Build
each live world anchor by transforming public `model.joint_X_p` and
`model.joint_X_c` through the corresponding parent/child `state.body_q`, then
measure their separation. Body-center distance is not joint extension and can
move in the opposite direction around a pulley. Choose the monitored joint next
to the grasped segment unless the dataset names another load path explicitly;
a remote link can be compressed while the hauling end is in tension.

Evaluate failure from a named physical measurement, not hand travel or elapsed
time. Prefer a public solver reaction/tension signal when the installed solver
documents one. Otherwise define the actual material measurement before choosing
a constitutive model. Adjacent CABLE joint anchors can coincide at rest: their
gap is constraint extension, not a material span of positive rest length.
Do not evaluate `(gap - segment_length) / segment_length`, or divide by the
zero rest gap; neither computes material engineering strain.

For a joint-extension proxy, declare a positive material gauge length `Lg`,
a current unit tensile axis `t`, and the rest gap vector expressed in that same
frame. Measure axial extension as
`delta = max(0, dot(current_gap - rotated_rest_gap, t))`, then use
`epsilon = delta / Lg`. Define how the axis follows the material and how
compression, shear and bending are excluded. Only use `sigma = E * epsilon`
when the selected material/solver law supports that elastic approximation;
for a verified joint stiffness law, `F = k * delta` and `sigma = F / area`
may be the appropriate explicitly labelled proxy. A force/stress proxy is not
a solver-reported reaction force. If measuring actual separated material
endpoints with positive initial span `L0`, `(L - L0) / L0` applies to that span.

Before live tests, assert zero strain at rest, a known axial extension
(e.g. 1 mm over a 100 mm gauge gives 0.01), no tensile strain for compression,
and no tensile extension from pure joint rotation with coincident anchors.
Reject non-finite values and zero/negative gauge lengths. A bending-only case
must not trigger a tensile break just because centerline orientation changed.
Record whether the threshold is stress, force, or strain. Require it for a small
declared number of consecutive physics substeps to reject one-frame contact
spikes, then emit exactly one bounded break event containing the material id,
joint/body labels, threshold, measured value, physics frame, hand input
sequence, and subsequent publication ordinal.

Acquire one cable segment through the same real tracked-hand and simultaneous
opposing-contact gate as a rigid payload. A gesture alone must not select the
hauling end or whole cable. If contact/friction cannot retain the segment within
declared slip bounds, use only a predeclared native constraint whose installed
API and solver behavior were verified, and release it on hand loss, focus loss,
disconnect, or opening. Keep any FVDB component in its proven sample role and
Newton as the sole cable-mechanics owner unless the exact installed sample
documents a coupled ownership contract. If an FVDB-backed visual or field is
required for the result, probe its package, device/backend, and publication path
as an independent capability and disable interaction with a clear reason when
it is unavailable.

Validate each material below and above its declared threshold, plus an ordered
multi-material sweep. A complete causal record is:

```text
real XRFrame joint poses and transmitted bytes
  -> server receipt and one coherent hand-proxy update
  -> opposed native contacts in one physics frame/input sequence
  -> named cable segment acquisition with bounded slip
  -> measured strain/tension crosses the named material threshold
  -> the predeclared Newton joint becomes disabled and adjacent segments separate
  -> live-stage cable transforms publish at ordinal N
  -> OVRTX consumes ordinal >= N and submits successfully
  -> changing decoded pixels and one continuous animation show the same break
```

Capture open, contact, pull, threshold-crossing, first-separated, and released
frames. Include a below-threshold pull that does not break, an open/away hand
that does not acquire, repeated trials per material, bounded live-stage
centerline or adjacent-body readback around the failed joint, and the source
USD hash. Do not declare success from a `broken=true` flag, a disabled-joint
array, a changed curve attribute, or an animation without correlated ordinals
and decoded pixels.

Treat VBD substeps and solver iterations as coupled stability/latency controls:
raising either increases constraint work, and raising both can make input lag or
render backlog dominate the interaction. Measure simulation wall time separately
from render cadence and hand-to-contact latency, vary one control at a time, and
declare the application's numeric slip, contact-persistence, constraint-error,
velocity, freshness, and repeated-trial bounds before comparing configurations.
Select the smallest configuration that repeatedly preserves finite state,
bounded slip, stable contacts, and the dataset's below/above-threshold ordering.
Reject a grasp or break trial when state becomes non-finite, constraint error or
velocity grows without bound, contacts chatter across frames, or the owner can no
longer consume fresh hand input. A visually dramatic animation from an unstable
solve is not evidence.

Keep proof readability separate from mechanics. A side camera should be fitted
from the hand path, mechanism bounds, hauling segment, and candidate break point
so the full interaction remains visible. If the physical cable is too thin to
read, add a non-collidable renderer-only sheath or highlight in the runtime
session layer and drive it from the same segment transforms. Do not enlarge the
Newton collider, alter the material radius, or mutate authored USD merely to
make the recording clearer. Keep that presentation geometry out of the Newton
model and any stage parsing used to build its topology, and disclose the visual
aid in the proof report.

## Continuous tracked-hand input is a command stream, not a grab event

Treat tracked hands as continuous commands with stricter provenance than a
click. For browser WebXR, sample from `XRSession.requestAnimationFrame()` and
obtain each required joint with `XRFrame.getJointPose()`. For native OVXR, use
the current OpenXR frame poses through the upstream borrowed-handles contract. A browser-side timer, cached
pose, fabricated server-side hand packet, or synthetic `grasp=true` message is
not evidence that a live tracked hand drove the simulation.

Send one bounded hand snapshot, not one queued item per joint. Include the
source/session epoch, stage generation, monotonically increasing input
sequence, sample time, validity for every required joint, and payload byte
count. Prefer newest-wins coalescing when the owner falls behind; reject an old
generation, old session epoch, non-monotonic sequence, or expired sample rather
than replaying it into the current hand. The input callback validates and
queues only. The render/simulation owner drains the latest accepted snapshot
and updates all kinematic proxies as one coherent command before stepping.

Derive `open`, `closing`, `closed`, and `opening` from the actual joint geometry
and joint velocity. Normalize distances by a stable hand measure, define enter
and exit thresholds separately, and require a small dwell interval so tracking
jitter cannot chatter the gesture. A grasp may become eligible only after the
physics runtime reports the required contacts or native constraint state; the
gesture classifier does not create attachment by itself.

Make interaction arbitration explicit. A test-oriented **physics-hand-only**
mode may mute ray selection and teleport tools, but it must not mute joint
sampling, the hand command stream, or physics contacts. Publish the active mode
and rejected-tool reason in status, and make the mode reversible without
restarting the renderer, stage, simulation, or input session. Route ownership
at the same first-refusal boundary described in
`viewer-input-routing/README.md`; do not let multiple tools consume one gesture
and then infer which one won.

Treat hand loss, focus loss, and client disconnect as explicit input events.
Queue a release and proxy-park command for the active input epoch, then let the
render/simulation owner apply it. Clear pending pose deltas and require a fresh,
complete joint snapshot before reacquisition becomes active. Never preserve a
grasp, replay the last pose, or teleport a reacquired proxy across the scene.

## Remote hold and direct collision are one mode choice

Use one authoritative mode value with exactly two valid states. In remote
mode, ray selection plus the mapped fist control drives a predeclared
solver-owned fixed-joint hold and direct colliders stay parked. In direct mode,
tracked kinematic colliders drive contact and the remote ray/fist hold path is
disabled. Do not model these as independent UI toggles; simultaneous ownership
lets one hand path cancel or fight the other.

Queue remote begin/update/end commands to the simulation owner. Keep the target
dynamic, move only the kinematic anchor, and let the solver produce contacts and
the held body's velocity. Release disables the constraint without overwriting
linear or angular velocity, so the body drops or throws naturally. Parenting,
per-frame payload transform copies, kinematic payload conversion, and synthetic
release velocity are grasping shortcuts, not valid Newton interaction.

Give direct and remote cleanup separate ownership tokens or state machines.
Parking an unused direct proxy every frame must not end the active remote hold;
only fist-open, tracking/target loss, mode transition, reset, disconnect, or an
explicit cancel may do that. Include mode, owner, target, phase, and rejection
reason in status so a gesture that reached input but failed at the constraint
boundary is diagnosable.
