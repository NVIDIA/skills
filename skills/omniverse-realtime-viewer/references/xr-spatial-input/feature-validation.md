# Spatial feature closure checklist

Use this checklist for every new controller, hand, tracked prop, gaze, anchor,
navigation, scene tool, or generated XR visual. Adapt irrelevant rows honestly;
for example, client-native UI has no generated USD geometry and must state that
instead of manufacturing an ordinal.

## 1. Preserve the control

- Run the same selected native or browser client with the feature disabled and enabled. VIEW
  translation and rotation must advance in both runs.
- Require one OVXR context, one OVRTX renderer, and exactly one supported frame
  path: high-level `Context.render_frame()` or the installed SDK's explicit
  begin/bind/submit path. Never call both for the same iteration or add a raw
  application-owned `xrWaitFrame`/`xrBeginFrame`/submit loop beside the SDK.
- Use documented anchor setters for navigation and documented scene-write APIs
  for application geometry on that same owner. The current camera helper is
  not a general object-transform contract. A feature must never introduce a
  concurrent scene writer or duplicate frame owner.

## 2. Prove the input boundary

- Record the actual source, session/input epoch, profile or capability, focus,
  side, sequence/time, pose-valid bits, and active/tracked quality separately.
- Test absent, malformed, stale, reordered, unsupported, unfocused, and lost
  input. A malformed future epoch must not poison a valid current source.
- Treat left and right independently. Reconnect, source replacement, profile
  replacement, and stage reload clear latches, filters, captures, generated
  visibility, reconstruction state, and pending proof.
- If native and opaque implementations exist, select one owner explicitly and
  test that they cannot drive the same semantic state simultaneously.

## 3. Prove coordinates with adversarial fixtures

- Preserve raw measured pose, reported interaction profile, physical hardware,
  authored model locator, and application navigation as separate values.
- Convert raw tracking poses once with `Context.transform_openxr_pose`, using
  the active frame-latched anchor. Do not duplicate units/up-axis/eye math.
  An application helper called `NavigationController` is not an SDK dependency.
- Use fixtures with non-identity tracking alignment, source-root transform,
  navigation scale/yaw/translation, non-origin viewer pivot, centimetre units,
  and every supported up-axis. Identity-only tests are insufficient.
- Assert physical relationships, not just finite matrices: emitter equals the
  measured/authored trigger locator, a grabbed object retains its acquisition
  transform, “toward viewer” push/pull is collinear with the full 3D
  viewer/object line (including height), forward locomotion follows VIEW, and
  player yaw preserves the current head pivot.

## 4. Prove generated scene state

- Author viewer-owned geometry in a content-addressed temporary composite;
  hash the source USD before and after and require no files beside it.
- Validate every renderable leaf has nonempty mesh or primitive data, plausible
  stage-unit dimensions, a hidden initial matrix, and a strict runtime
  allowlist entry. Dynamic native parents stay identity.
- At an interaction boundary, capture the exact submitted leaf matrices. After
  the renderer consumes that ordinal, read `omni:fabric:worldMatrix` once and
  compare world position, orientation/basis scale, visibility, and semantic
  target. An `omni:xform` echo alone is not rendered-stage proof.
- Correlate nonblank, changing headset or mirror pixels to that consumed state.
  Telemetry, an action edge, or an ordinal without pixels does not prove visible
  geometry.

## 5. Protect the hot path

- Build/extract scene data only at stage-generation boundaries; allocate GPU
  meshes, buffers, and graphs once. Refit only transformed instances at their
  release/cancel boundary.
- Launch GPU work only for an active mapped control. Hidden unchanged leaves are
  not resent every frame.
- Normal frames do not synchronously read Fabric, copy audit history, serialize
  proof, traverse USD, create prims, import `pxr`, or emit per-frame info logs.
- Validation readback is event-triggered or explicitly armed for a finite frame
  budget and automatically turns itself off. Add a test that steady state does
  not reschedule it.
- Bound transport/message drains per iteration and leave burst remainder queued.

## 6. Close lifecycle and evidence

- Exercise attach, disconnect, reconnect, one-side loss/recovery, focus loss,
  profile change, stage reload, cancel while captured, and shutdown. Head
  tracking and the long-lived renderer survive every input-only boundary.
- Require deterministic math/state tests plus a live physical or IWER path.
  Wait for a successful submitted frame whose consumed ordinal reaches the
  transaction before capture.
- Capture ordered visual evidence and report unsupported capabilities or
  unverified physical behavior as gaps. Never promote controlled input,
  analytic contact, or accepted haptic requests into stronger native claims.
