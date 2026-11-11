# Native and opaque-channel controllers

## Architecture

Use one immutable controller snapshot and semantic tool pipeline for both input
sources:

1. Native CloudXR: follow upstream openxr-input-and-extensions through
   [the SDK route](../dependencies/ovxr.md) for shared-session input startup.
2. CloudXR.js/IWER: receive bounded, correlated snapshots through
   `XR_NV_opaque_data_channel`.
3. Normalize both into an application-owned snapshot and semantic state machine
   (`ControllerSystem`/`ControllerToolState` are sample names, not SDK APIs).
   Publish documented scene/anchor operations from the render owner.

Do not add a second `xrWaitFrame`/`xrBeginFrame`/renderer-step/submit path.
Read [the library input contract](../dependencies/ovxr.md) for SDK calls.
Use [frame/input scheduling](../ovxr-library-runtime/frame-input-order.md):
the split path locates at the plan's predicted time, and converts after begin through the same frame-latched mapping as the eyes; a
convenience after-submit adapter feeds a later frame and must identify its
actual timing source. QPC-to-`XrTime` converts clocks, not current time into a
predicted display time. Head tracking must behave identically with the action
consumer enabled and disabled.

## Identity and bindings

Preserve these independent values:

- `reportedInteractionProfile`: the real runtime binding contract, such as the
  Touch-compatible profile CloudXR exposes.
- `controllerHardware`: advisory physical identity declared once per client
  session/input epoch, such as `sony-psvr2-sense`.

Use a session-scoped capability handshake. Validate schema, lengths, session ID,
and monotonic epoch; reject stale input without damaging the selected profile.
Unsupported or absent handshakes retain the generic binding/profile fallback.
Never invent an OpenXR PSVR2 path.

Profiles own logical controls. Keep the physical label, normalized semantic
action, and submitted OpenXR component together so diagnostics can prove the
mapping. Reset semantic edge latches when a profile/input lifetime changes.

### Profile example and visual assets

An application may choose the following Touch-compatible semantics: right Grip
(`squeeze/value`) toggles its ray; right A (`a/click`) selects; right Trigger
(`trigger/value`) moves a selected object; left X (`x/click`) previews teleport
while held and commits on release. Left stick walks; right stick turns unless
an active UI/object capture owns it. These are **application bindings**, not
OVXR defaults. Keep side, physical label, normalized semantic action, OpenXR
component and optional model locator together. Reset edge latches on profile
replacement and show tooltips only for actions actually supported.

`/interaction_profiles/meta/touch_controller_plus` and a CloudXR-reported
`/interaction_profiles/meta/touch_plus_controller` are different profile strings;
retain the runtime's exact value rather than silently rewriting it. A physical
hardware declaration is a separate validated session capability and may have a
per-profile headset allowlist. Headset identity alone cannot prove which hand
controllers are connected.

The libraries do not supply a calibrated model for every physical device. Begin
with clearly labelled application-authored primitive visuals, or accept a
licensed local pre-converted asset. Never call generic geometry official vendor
art or hardware-fit. A portable local-model manifest should record:

- profile/model identity, independent left/right source hashes and output hash;
- source licence/provenance, units, up-axis and normalized root;
- material-bound renderable leaf paths and physical bounds;
- separately audited control locator positions in measured grip coordinates;
- displayed action-to-locator mapping and explicit pose calibration policy.

Generate that asset and manifest outside the render process with USD/DCC tools.
Validate hashes/metadata before reference composition. No sample asset script,
`--controller-model-manifest` flag or sample model class is SDK functionality.
Do not create fake button meshes merely because a tooltip needs a locator.
Do not mirror a left-hand asset when handedness is not validated. A visual
calibration may position a mesh relative to measured grip; it must not change
input grip/aim orientation or the ray used for scene tools.

Flatten referenced renderable leaves if publishing full world matrices through
OVStage. Do not transform a whole controller reference and independently apply
world transforms to nested controls: inherited transforms can be applied twice.
For a visual leader terminating on the shell, precompute the nearest-triangle
projection of an audited locator during asset preparation. Retain original
locator, projected point and distance as provenance; shell contact proves a
visual attachment, not a measured physical button center.

## Spaces and controller geometry

Treat grip and aim as independently measured spaces:

- grip orientation drives the controller model when it visually matches;
- aim orientation drives interaction direction;
- neither pose may be replaced by a visual calibration.

Only for an explicitly supported legacy package without managed pose conversion,
if the application derives a shared tracking-to-world alignment from live OVXR
eye leaves, do not cache the first matrices merely because they are finite and
rigid. Require a physically plausible measured stereo separation; freshly
authored coincident eye defaults otherwise map every spatial input to scene
origin for the connection. Fail closed and retry until the pair is cooked, but
back off failed live-stage reads after a short eager window. Cache the solved
rigid transform for the complete input lifetime and clear it on detach/reload.

Extract a user-supplied controller asset in a disposable `usd-core` worker.
Flatten meshes relative to the selected model root, normalize units/up-axis,
preserve provenance, and create renderable left/right leaves. Never import
`pxr` into the OVRTX render process or redistribute an unlicensed source/cache.

An asset root is usually a layout transform, not a tracking locator. Exclude it
when its descendants already contain physical-size numeric coordinates and the
resulting dimensions validate in metres. Do not infer a rotation by eye.

When the asset authors a named trigger mesh and the measured OpenXR aim origin
is the intended emitter, derive alignment from geometry:

1. Compute the named L2 trigger mesh's root-relative bounds center in the USD
   worker; mirror X for R2 when right geometry is mirrored.
2. Translate the flattened model by the negative trigger center.
3. Compose that local translation with stage-units conversion and the model
   pose (measured aim translation plus measured grip orientation).

The trigger center then equals the measured aim origin by construction. The
visible ray can keep the aim pose's translation and direction. Record the
derived locator and source hash in the cache manifest; fail to generic geometry
if the expected named trigger is absent or ambiguous.

That single-anchor route does not establish the asset's coordinate frame. A
left/right pair of trigger centers and grip-to-aim offsets is still only two
points; it does **not** establish a safe controller basis or roll. For the
supported live path:

1. Keep the controller body at the exact OpenXR grip orientation.
2. After the strict bounded acquisition-spread gate succeeds, derive only a
   per-hand grip-local translation that maps the named trigger center to the
   measured aim origin. Do not scale the mesh.
3. Publish geometry as `model * translationOnlyModelFromGrip * gripPose`;
   continue publishing interaction rays from the independently measured aim
   pose.

Clear this translation on disconnect/profile replacement and never run it as
an unbounded frame-loop companion. Do not infer rotation from an asset locator,
an inferred pose, appearance, or bilateral trigger spacing. Any future rotated
model requires a versioned multi-landmark physical calibration artifact with
its own acceptance criteria and review; it is not part of the live fallback.

## Rays and scene tools

Build scene geometry once per stage generation in an isolated USD worker and
upload one persistent Warp/CUDA mesh/BVH. During frames:

- launch only fixed preallocated ray slots while mapped controls are active;
- convert measured aim once into the BVH's declared stage-world space using
  the active managed anchor; do not apply an additional inverse navigation;
- keep hit position/normal in that same stage-world contract (a legacy
  scene-root adapter needs an explicitly separate conversion);
- update pre-authored beam/arc/marker leaves, never create geometry per frame;
- refit transformed source vertices/BVH only at object release/cancel/disconnect.

Selection is not capture. Toggle selection on its mapped edge, begin movement
only on the move control, preserve the object's acquisition transform, and
apply controller deltas relative to the captured controller/object pair.
Push/pull uses the intended depth axis explicitly. For “toward the viewer,” use
the normalized full 3D viewer-to-object vector sampled for that increment; do
not flatten the up axis or reuse a pitched/laterally offset controller ray.

For a general-stage movement target, register only the exact extracted render
leaf returned by the CUDA hit. On that selection edge, read its exact live
Fabric matrix once and recover local bounds from the persistent mesh. Scope the
permission to the current stage generation, preserve the baseline on capture,
and refit the affected vertices/BVH at release or cancellation. Do not authorize
an arbitrary ancestor and do not add per-frame scene reads.

Teleport previews only measured walkable hits. One hardware profile may replace
the generic straight beam with its arc/marker, but do not show both. Validate
head height only at commit if preview needs only controller tracking.

## Navigation

Use [the managed-anchor contract](navigation.md) for movement, yaw, scale and
teleport. It replaces the older inverse scene-root workaround. Preserve head
pivot, stick hysteresis and tool ownership; leave source scene roots unchanged.
Use the public pose conversion API instead of deriving alignment from eye prims
when that capability is present. The measured-eye alignment discussion above is
only a legacy fallback for older packages, not the current preferred route.

## Filtering and lifecycle

Filter controller poses in raw tracking space before units/anchor conversion.
Use timestamp-aware One Euro position and quaternion/log-map orientation filters.
Filter signed velocity before magnitude; reset filters on tracking loss.

At attach/reconnect, create a new monotonic input lifetime and force one complete
visual republish. On loss/detach, cancel captures, hide visible leaves once,
reset sequence/profile state, and preserve the renderer/headset lifecycle.
Emit pose-boundary diagnostics on validity transitions and retain the latest
facts for status queries; do not copy unchanged diagnostic dictionaries into
every native frame payload.

## Verification

Hardware-free tests must cover coordinate transforms, trigger-to-aim alignment,
player-pivot yaw away from the origin, selection vs capture, no-jump movement,
depth direction, teleport preview/commit, action hysteresis, stale frames,
tracking loss, reconnect, and event-boundary BVH refit.

Physical claims additionally require the bounded native visual audit: submitted
leaf matrices, consumed ordinals, live Fabric world matrices, valid mesh data,
and correlated nonblank headset pixels. Action telemetry alone is insufficient.

## Controlled-client reference-space identity

Sample head, grip, aim and hands relative to the exact reference space used by
CloudXR's production pose submission, including offsets. A separately requested
`local` space is not interchangeable with the stream's `local-floor` space.
Do not infer the active space from the most recently requested WebXR object or
fit away an origin mismatch with calibration offsets. Carry explicit source-
space identity through transport/diagnostics and use the same frame/reference
for validation projection. The public managed anchor maps a coherent tracking
input; it does not repair mismatched input origins.

## Optional compact controller wire format

The transport schema belongs to the application, not OVXR. Preserve semantic
fields when fitting a bounded channel; do not drop released buttons, head pose,
frustum data or haptic acknowledgements to make a packet fit. The reference
application may define a versioned pose object or a compact alternative:

- Pose: `[flags, base64]`. Bits 0–1 encode source `a/g/i/f` as `0/1/2/3`;
  bit 2 selects tracking space (unset means stage space); bit 3 marks approximate
  data. Reject unknown flag bits and malformed tuples.
- The payload is exactly seven little-endian signed int32 values: position
  x/y/z followed by quaternion x/y/z/w, each quantized with scale 100000.
  That is 28 decoded bytes or 40 base64 characters. Validate length, numeric
  ranges and quaternion validity; retain the chosen five-decimal precision.
- Crowded input may use `[trigger, squeeze, stickX, stickY, buttonBits]`.
  Primary/secondary/menu are bits 1/2/4 respectively. Preserve analog values
  and explicit zero/released states; reject malformed or unknown bit masks.

Short correlation IDs are opaque identifiers, not commands. They can combine
hand/frame identity and a bounded reason code while remaining unique in the
session. The receiver must not infer feature behavior from their spelling.
Measure actual UTF-8 serialized bytes after all fields and acknowledgements are
present. Enforce the final 1024-byte ceiling even with custom IDs; reject excess
rather than truncating behavior. Test both hands, head/frustum, all buttons,
nonzero analogs, two acknowledgements, legacy objects, malformed compact data,
quantization round trips and explicit release frames.

When source poses first round to six decimals and the wire budget then rounds
to five, the total position-component error bound is 5.5 micrometres relative
to the raw source, not 5 micrometres. Test that combined bound separately from
lossless analog input transport.

Decode packed aim/grip through one strict shared decoder before *every* consumer,
including UI pointers, tooltips and raw transport adapters. Expanding only in
the controller semantic pipeline lets tools work while UI silently loses poses.
Apply the declared tracking-to-LOCAL mapping after decoding, not by changing
packed positions or inventing a fixed floor-height offset.

## Minimal application protocol from scratch

Choose and document your own schema rather than requiring the reference app's
message names. One interoperable semantic snapshot contains:

```text
version, sessionId, epoch, stageGeneration, sequence, sourceTime,
focused, referenceSpaceIdentity,
head = {positionMetres, orientationXYZW, measured:true},
controllers[hand] = {tracked, profile, grip, aim,
  trigger, squeeze, stick:[x,y], primary, secondary, menu}
pose = {positionMetres:[x,y,z], orientationXYZW:[x,y,z,w],
        source:aim|grip|inferred|fallback, valid, approximate}
```

In one WebXR RAF callback, use the **same XRFrame and exact reference-space
object supplied to the CloudXR pose submission** for `getViewerPose`,
`getPose(inputSource.gripSpace, referenceSpace)` and
`getPose(inputSource.targetRaySpace, referenceSpace)`. Copy values before the
callback returns. A grip-only fallback can keep a model visible but must remain
marked inferred; it is not a measured aim ray. Normalize the floor projection
of head forward from the actual head orientation, not controller heading.

Receiver order: decode bounded envelope → preview freshness → decode all poses
and controls → validate finite positions/unit quaternions/current source and
stage → convert a batch with one active-anchor snapshot → commit freshness →
compute button edges and semantic ownership → publish one accepted update.
Decoded raw tracking poses also feed the UI adapter; that adapter applies the
measured tracking-to-LOCAL transform when the compositor uses LOCAL. Do not
feed it the already converted stage-world poses.

A pose-free loss/release packet must still clear buttons and captures. Sample
controllers continuously; rate-limit transport independently from sample time,
with a guaranteed terminal released/inactive snapshot. Test reconnect with a
new epoch and a sequence reset. Native action support, visual art, semantic
bindings and compact-wire interoperability are separate capabilities.
