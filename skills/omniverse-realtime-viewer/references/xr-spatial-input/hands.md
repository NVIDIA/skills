# Hand tracking

For implementation equations use [hand recognizers](hand-recognizers.md);
for the editor tool priority/state table use [gesture arbitration](gesture-arbitration.md).
These are configurable application policies, not new SDK gesture APIs.

## Choose the input boundary explicitly

OVXR does not automatically make client hand joints available to the
application. Use one of these paths and label it honestly:

- Controlled client: sample the client's real hand API and send joint snapshots
  over the existing opaque data channel.
- Native OpenXR: use upstream openxr-input-and-extensions through
  [the SDK route](../dependencies/ovxr.md), then normalize the valid joint
  snapshot at the same time/space as the controller bridge.

Do not present controlled-client telemetry as server-native hand tracking. If
the active CloudXR runtime does not expose the extension or authoritative
tracking conversion, report that capability gap instead of guessing.

## Canonical frame

An active hand contains all 26 `XR_EXT_hand_tracking` joints in canonical order.
Keep each hand independent: active/tracked flags, pose validity, radius, and
optional velocities. A missing left hand cannot borrow right-hand validity.

Apply OpenXR's flag contract exactly: position and orientation `VALID` bits
authorize reading/rendering the pose. `TRACKED` bits report whether that valid
pose is actively tracked rather than inferred or last-known; their absence does
not invalidate the fields. Preserve both qualities in diagnostics. Never read a
position or orientation whose corresponding `VALID` bit is clear.

For opaque-channel transport, use bounded packed keyframes plus recoverable
deltas. The current sample's `openxr-i16-v1` form stores 26 `(x,y,z,radius)`
signed-int16 tuples in a 208-byte block per hand. Enforce the 880-byte envelope
ceiling, finite values, schema version, monotonic source/session/epoch/sequence,
and a periodic keyframe. A gap must either recover from a keyframe-relative
delta or fail closed; never apply an ambiguous incremental delta.

Coordinate space is a contract, not metadata decoration. Accept stage-space
frames only when the client has authoritative stage conversion. For tracking
space, convert server-side with `Context.transform_openxr_pose` using the same
active frame-latched anchor as controllers and VIEW. Do not derive alignment
from private eye prims when this public capability exists. Check pose validity
before conversion and keep display size separate from measured physical radius.

## Visuals and interaction

Author 52 hidden joint leaves in a viewer-owned session layer. Publish both
hands in one bounded scene transaction per accepted frame on the render owner,
using upstream's scene-transform/publication contract or the shared OVStage
writer. Verify that head tracking continues with the feature enabled.
Scale visual geometry from measured radius while keeping semantic/contact
radii untouched. Freeze only a short documented number of missing received
sequences, then hide once.

Derive pinch from measured thumb/index geometry with start/end hysteresis. A
pinch does not imply a grab: begin grab only when the interaction target and
policy allow it, preserve target/controller offset, and emit end/cancel only for
states actually begun. Tracking loss, source replacement, stage reload, and
transport disconnect cancel active interactions and reset reconstruction state.

For native server-side gesture tools, recognize gestures from the same
stage/world frame published for hand visuals. Scale thresholds by measured hand
geometry and runtime-reported radii; never use fixed scene-unit distances. Build
an aim basis only from valid measured joints, then normalize gestures into the
existing controller-tool DTO and reuse its persistent scene raycaster,
selection, no-jump movement, and teleport state machines. Do not add a second
picker, navigation model, or transform writer.

For movement on a general stage, promote only the exact render leaf returned by
the persistent scene raycaster. At the selection edge, capture its live Fabric
world matrix and recover local bounds from already-extracted vertices. Keep that
write permission stage-generation scoped, preserve the captured baseline on the
first fist frame, and refit the BVH only on release/cancel. Never traverse USD or
read Fabric every frame.

Arbitrate overlapping gestures contextually and document the capture rule. In
this sample, a right pinch owns the selection ray; a left pinch edge selects at
that ray while it is active, otherwise left pinch captures teleport until
release. A selected-object right fist captures movement, and right pinch cancels
an active teleport. Keep hand availability, hysteresis, consumed edges, and
open-hand rearm independent per side. Loss of a non-owning hand must not cancel
the current action; loss of the owning hand, focus, source, stage generation, or
tool ownership cancels exactly once. Physical controllers take priority and the
gesture recognizer must cancel/rearm when that ownership changes.
Distinguish an active hand with temporarily invalid joints from an inactive
hand, focus loss, and a scene-ray miss. A grace branch that requires an active
hand does not tolerate `isActive=false`. State and test each policy separately;
hold the last measured observation only within the declared grace without
emitting synthetic edges. Expiry or a lifecycle boundary cancels and rearms.

If a separate OVPhysX worker is used, keep ownership isolated and terminology
precise. Successful write/step/read does not prove contact pairs. Label analytic
sphere overlap as analytic unless a public contact-pair query returned evidence.

## Gesture ownership and forgiving holds

Treat the gesture map as application policy. A system-reserved palm-up gesture
can conflict with the headset shell; use a simpler mapping when the product
calls for it instead of assuming the app can suppress the system gesture.

Separate a strong start threshold from a more permissive hold/release threshold.
Expose per-hand pinch thresholds in physical or hand-normalized units, and
check that a secondary joint-radius rule does not broaden a deliberately
strict teleport start. A fist can place thumb and index tips close together:
arbitrate fist versus pinch before emitting semantic edges.

For the sample's right-pinch cancel policy, cancel teleport without committing,
observe both hands open, then require a fresh left pinch. The cancelling right
pinch must not also select or immediately restart teleport. Track each side's
rearm state explicitly.

During a held preview, use a finite tracking/joint-validity grace and preserve
ownership without inventing new motion or press/release edges. Retained
observations are stale, not a current walkable hit; a release during stale
tracking must not commit a cached destination. Cancel on grace expiry or
focus/source/session loss. Prefer an elapsed-time bound as well as a frame
bound so a frame-rate drop does not lengthen a sticky hold indefinitely.
Carry the observation timestamp, per-hand staleness and actual scene-query
attempt/result through every runtime call site, not only recognizer tests.
An intentionally omitted query is different from an attempted miss, a failed
query or pending async work. Missing evidence must not authorize a teleport.
Retain async submission time: a newly delivered result is not a fresh ray.
Test release after a miss, stale tracking and delayed results end to end.

## Performance and proof

Decode/validate only accepted messages. Do not copy full hand history, inspect
Fabric, serialize proof, or run physics while hands are inactive. Keep proof
readback finite and explicitly armed.

Hardware-free tests should cover 26/52 joint completeness, byte ceilings,
quantization bounds, keyframe/delta gaps, stale sequences, independent hand
loss/reacquisition, visibility transitions, scale-relative pinch/fist
hysteresis, contextual gesture arbitration, non-owning-hand continuity,
owning-hand/focus/ownership cancellation, open-hand rearm, grab offset under a
non-identity navigation transform, and source/stage resets.

Live proof requires actual joint motion from the declared source, correlated
request/sequence and consumed ordinal, rendered joint geometry, and ordered
pixels. For a native headset run, also record extension support, tracker create
status, locate result/flags, and both hands independently.

## Reconstructable hand encoding and scaling

For a library-only client, agree joint order explicitly: palm, wrist, thumb
metacarpal/proximal/distal/tip, then index/middle/ring/little each with
metacarpal/proximal/intermediate/distal/tip. WebXR supplies the 25 non-palm
joints. One simple palm approximation uses the middle-metacarpal pose and a
radius derived from the wrist/metacarpal samples; label it synthesized instead
of native. Do not transmit an invented tracked orientation where only position
was observed.

For the position/radius codec, choose q in metres (for example 0.0001), round
`[x,y,z,radius] / q` to signed int16, check range, and serialize little-endian.
Decode only exactly 26×4×2 bytes for a full hand. At q=0.0001 the representable
coordinate range is about ±3.2767 m; for larger reference-space travel choose a
larger explicit q or a declared keyframe origin and a new protocol version.
Never wrap overflow. Quantization error is at most q/2 per rounded component.

A delta contains the full keyframe's sequence ID, an unsigned 26-bit changed
joint mask, and four **absolute quantized values** for each set bit in increasing
joint order. Copy the named cached keyframe and replace those joints; do not
add values and do not reconstruct from the preceding delta. Missing/wrong-epoch
base means inactive until a fresh keyframe. Two-hand base64 keyframes plus JSON
can exceed a small opaque channel: publish independently correlated per-hand
keyframes or a versioned compact envelope, preserve inactive/release states,
and enforce the final serialized byte ceiling before send.

Convert positions once using the same frame-latched anchor as controllers.
For validated uniform anchor scale s, radius and linear-speed magnitudes scale
by `s / metersPerUnit`; directions use only basis and normalized anchor rotation.
Derive velocities from converted positions or transform supplied velocities,
never both. Keep explicitly declared stage-world input unchanged. Gesture
thresholds, hand rendering and physics must consume that same converted batch.
