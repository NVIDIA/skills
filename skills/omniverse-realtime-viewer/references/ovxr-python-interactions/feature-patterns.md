# Feature patterns

## Controllers

- Sample both IWER controllers every animation frame on the controlled-client
  path; sample native actions through public borrowed handles after OVXR startup.
- Freeze grip/aim pose, profile, trigger, squeeze, thumbstick, and focus.
- Enforce the client 1024-byte wire ceiling and the server decoded-size ceiling.
- Normalize physical components before semantic actions.
- Cancel active actions on blur, visibility loss, tracking loss, stop, or
  profile replacement.
- Extract composed scene triangles once per stage generation in a disposable
  USD worker and upload one persistent Warp/CUDA mesh. Cast only fixed
  preallocated right-grab and left-teleport slots while their mapped controls
  are active. Refit a moved allowlisted target only at release/cancel; never
  traverse USD, rebuild a mesh, mint an ordinal, or launch an idle query per
  frame. Unsupported gprims/instances fail closed.
- Keep reported interaction profile, declared physical hardware, measured grip,
  measured aim, and profile-scoped ray policy separate. Model/trigger alignment
  may derive from authored locator geometry; never rewrite poses by eye.

## Hands

- Sample `XRFrame.getJointPose` for WebXR's 25 joints and synthesize only the
  OpenXR palm, yielding the canonical 26-joint order.
- Use periodic int16 keyframes and keyframe-relative deltas. A lost delta must
  not corrupt later frames; reacquisition requires a new full keyframe.
- Publish all 52 proxy leaves in one documented scene transaction on the
  render owner; native source support is conditional on the installed lifecycle
  capability. Do not treat the camera-update helper as a general hand API.
  Visual size may differ from measured contact radius, but never feed display
  scaling into contact logic.
- Position/orientation VALID bits authorize a joint pose. Preserve TRACKED bits
  as active-versus-inferred quality telemetry; do not discard valid inferred
  poses and do not read invalid fields.
- Use hand-scale-relative pinch hysteresis; validate open→pinch→hold→open.
- Keep a validation fixture absent until the first pinch creates it. Once a
  valid grab owns that fixture, retain ownership and follow the pinch midpoint
  while the pinch remains active; do not require overlap with the fixture's
  previous-frame position after motion begins.
- Treat the OVPhysX child write/step/read proof separately from contact-pair
  verification. If the bridge lacks contact reads, label overlap analytic.

## Navigation and anchors

- Store absolute translation, floor yaw, scale, and revision.
- Capture the pivot at gesture begin; compute later updates from the begin state
  so packet loss cannot accumulate drift.
- Route controller locomotion, teleport, desktop movement, and client gestures
  into one state stream.
- Reset and increment revision on stage change. Focus loss clears latched input.
- Keep source USD and `metersPerUnit` unchanged.
- Left-stick forward comes from current VIEW forward projected to the floor.
  Use the live managed anchor for user movement, yaw and scale; preserve the
  head pivot and source scene roots. See the self-contained
  [navigation contract](../xr-spatial-input/navigation.md).

## XR UI

- Keep UI out of USD and OVRTX scene geometry.
- Prefer the controlled-client renderer when runtime composition layers are not
  public. Use stable surfaces/nodes, revisions, ray/UV hit tests, capture, and
  cancel on focus or tracking loss.
- Correlate actions to server state; reject stale document revisions.
- Controller-pointer features inherit the same measured ray, source lifetime,
  capture cancellation, and navigation-space tests as scene tools even though
  client-native UI creates no USD geometry.

## Cross-feature acceptance

Use the spatial-input
[feature validation checklist](../xr-spatial-input/feature-validation.md)
for every new tracked pose, interaction tool, anchor, or generated visual. A
feature-specific happy-path test is not enough: preserve the disabled-feature
head-tracking control, transaction ownership, lifecycle cleanup, bounded hot
path, consumed-stage proof, and visible-pixel gate.

## Mirror

- Probe and exercise `/dev/shm` from the publisher/consumer execution namespace;
  do not infer native availability from a sandbox mount. Size the mount for the
  negotiated frame depth and resolution rather than silently redirecting to
  `/tmp`.
- Publish pixel order and exact per-frame foveation metadata atomically when the
  API permits it.
- When NPP is available, downscale on a private CUDA stream to an independent,
  aspect-preserving publication bound. Rescale pixel-space foveation metadata
  with the image and report source, requested bound, actual publication size,
  backend/fallback, pixel reduction, and unchanged headset resolution.
- Use a long-lived staging buffer and asynchronous synchronization.
- Letterbox to preserve aspect ratio. Bound publication and client draw cadence.
- Preview the cadence gate before acquiring an OVXR readback, then enforce it
  again under the publisher lock. The first check avoids needless context-lock
  traffic; the second remains authoritative when desktop and headset products
  share a stream.
- Treat connection and presentation demand separately. A minimized or hidden
  WebView can remain connected; report visibility over the control plane and
  use the idle heartbeat until it becomes visible again.
- Keep newest-frame-wins semantics at every handoff. If sequence ordering is
  already enforced before publication and RAF only draws newer sequences, one
  retained full-frame WebView buffer is enough.
- Correlate a published frame without changing OVStream's ABI by assigning a
  monotonic public `VideoFrame.start_timestamp_ns` and sending a bounded
  app-owned sideband keyed by the exact resulting SHM capture timestamp. Carry
  producer epoch/sequence, the ordinal actually consumed by OVRTX, source
  product, and live submission state as decimal strings. Copy and release the
  borrowed native pixels before a short exact-key wait; never join nearest
  timestamps or substitute the current stage floor for a missing mapping.
- Measure each boundary honestly: producer submit time/bytes, native SHM copy,
  response-body materialization, and WebGL upload. Do not label sequence gaps as
  queue depth or time only the callback after an `arrayBuffer()` transfer.
- Verify the installed ovstream version's buffer-lifetime contract before
  removing application staging; do not infer a newer contract from one machine
  while the sample still supports an older wheel.
- A performance proof must use the real SHM source, demonstrate advancing frame
  hashes or sequences, and measure foreground versus idle publication demand.
  A responsive-layout capture proves presentation and aspect preservation; it
  does not by itself prove producer throttling, GPU unwarp, or native resize.
- Capability-check `ContextDesc.readback_foveation_unwarp` in the installed SDK. When it
  produces rectilinear captures, omit client PQW inversion; applying it again
  distorts the preview. Distinguish configured readback domain from an actually
  received image. Some current warped-submission configurations produce no
  captures despite requesting unwarp: keep headset settings unchanged, report
  preview unavailable after sustained misses, label any retained image stale,
  and clear that diagnosis only after a fresh capture sequence. A working
  disabled-foveation mirror does not prove warped capture compatibility.
