# Build a CloudXR.js/IWER proof without the sample

## Required inputs and capability scope

Use an application built with the sibling runtime and interaction skills, an
installed CloudXR runtime, matching CloudXR.js client generation, the public
IWER package, and a browser driver such as Playwright. Select the runtime/client
pair from current public vendor compatibility guidance and record its exact
identity together with the resolved IWER version.
A sample repository, Tauri shell, native UI layer, and native action callbacks
are not prerequisites for proving the opaque-channel path.

Inspect IWER's installed public declarations/examples to bind its actual remote
interface for device transforms, hand poses and selection controls. Use the
installed bundle consistently;
a sample CDN fallback may load a different version. Make the chosen IWER device
own `navigator.xr` before the CloudXR client starts. Do not fabricate WebXR
poses or dispatch server-only feature messages in place of a claimed input path.

## Implement the harness boundary

The production client samples `XRFrame` head/controller/hand poses using the
exact `XRReferenceSpace` object CloudXR uses for pose submission, including any
offset reference space. Capture that explicit owner reference; never select the
last space returned by an intercepted `requestReferenceSpace`. A diagnostic
request for `local` after the stream selected `local-floor` can introduce a head-
height offset even though both objects are valid. Use that same reference for
controller transport and pixel projection. Managed-anchor conversion cannot
correct a mismatched client-space origin. Send bounded snapshots over an
advertised CloudXR opaque channel with client-session ID, epoch, sequence,
source time, focus/tracking and stage generation. Decode/validate/enqueue on the
server; apply semantics and anchor changes on its render thread. Keep driver
control in the public IWER API rather than altering server state directly.

Expose an app-owned status/reply protocol containing at least:

```json
{
  "requestId": "move-1",
  "acceptedRevision": 7,
  "appliedRevision": 7,
  "contextGeneration": 1,
  "stageGeneration": 1,
  "rendererId": "renderer-1",
  "serviceId": "service-1",
  "frameSequence": 123,
  "submitted": true,
  "result": "SUCCESS"
}
```

Names are an example, not OVXR exports. Include committed/consumed ordinals for
scene writes when available. Anchor acceptance must also expose pending versus
frame-latched state; an accepted setter alone cannot prove camera motion. If
exact pixel-to-server-frame correlation is unavailable, state that limitation
and use a visible uniquely sequenced scene change plus post-change decoded
frame evidence rather than claiming an exact frame match.

## Bounded startup and proof

1. Allocate explicit unused ports and unique artifact directories. Record
   process identities and commands; implement the [application-owned supervisor](owned-orchestration.md) with your own
   readiness markers and commands. Never terminate an unrelated port owner.
2. Start the web client, then the CloudXR service and renderer. Start IWER after
   listener readiness; do not wait for an HMD system before connecting its
   client. That would be a circular wait.
3. Set the browser's actual connection UI to the same host/port, Quest 3,
   1024×896 per eye, immersive VR, and `XRWebGLLayer` alpha/depth false for the
   baseline. These are reproducible test choices, not universal runtime limits.
4. Require a real XR session/reference space/base layer, advertised opaque
   channel, advancing client frame serial, submitted server SUCCESS, and
   increasing decoded-frame stats before interaction assertions.
5. Drive IWER head/controller transforms through public APIs. Exercise the
   production snapshot sender, await correlated accepted and applied state,
   then capture decoded scene pixels after a new submitted frame.
6. Read the WebGL framebuffer only in a small armed budget after a known decode;
   disarm after capture. Never keep `readPixels` in every XR frame callback.
7. Capture ordered baseline/moved/rotated/scaled/reset images and a short
   animation, plus the exact requests, replies, negotiated extents and logs.
   Validate image dimensions, hashes and visible nonblank differences. DOM
   screenshots may show only controls; inspect actual decoded XR pixels.
8. Disconnect/reconnect, require a new input epoch and stable renderer identity.
   Verify context generation matches the declared lifetime policy: it advances
   for detach/recreate, but can stay stable for a supported persistent session.
   Test tracking/focus loss and rearm, then clean up
   owned groups and verify port release.

Use separate bounded cases for the three update lanes:

| Case | Required identity/state evidence |
|---|---|
| Anchor move/yaw/scale and documented product resolution | Same context, service, renderer; active new anchor/resolution; continued head motion and changed pixels |
| Foveation/readback configuration refresh | New context generation, same renderer/stage/service; reopened channels, resubmitted frames; record whether browser renegotiation was needed |
| CloudXR backend/port/create-time property change | New service and context; old listener released; browser reconnected to matching settings; retained renderer where supported |

A driver's timeout is a failed or blocked case, not permission to substitute a
simpler claim. Preserve its last boundary, logs, status and pixels. Report each
lane as verified, failed, or not exercised; do not generalize a resolution test
to arbitrary RenderProduct mutation.

## Hand and native boundaries

Public whole-hand presets can prove index pinch and root movement; they do not
prove an independently driven middle finger. Native startup and action attachment
follow upstream input guidance through [the library route](../dependencies/ovxr.md).
Browser proofs may deliberately disable native actions and must be labelled as
opaque-channel evidence; they do not establish physical controller behavior. A custom compositor has its own export/ABI gate only
when that compositor is being tested.

An animation can be encoded from owned PNGs with Pillow; verify `n_frames > 1`
and visually inspect first/middle/last frames. Record library hashes/revisions,
client/runtime/IWER versions, device profile, commands, deadlines, identities,
capability gaps, pass/fail assertions and cleanup alongside it. Do not upload or
commit artifacts unless that publication is part of the user's requested work.

## Measured lifecycle harness ordering

Pause controller/hand senders before sending a context-refresh or service-restart
intent. Reopen the advertised opaque channel after new handles exist, then
resume input; an ordinary send to a deliberately closed channel is a harness
ordering failure, not evidence that anchor movement failed. A service restart
requires the browser to exit/reconnect. Reconcile generation counters after
reconnection instead of treating a queued response as completion.

Record the exact input exercised: physical IWER head/controller pose and stick
changes differ from application-level navigation gesture intents sent through
the opaque channel. The latter can prove live scale rendering, but do not prove
a two-hand gesture recognizer or a physical headset. Include final orderly
shutdown and released listeners in the result.

## Mirror and desktop proof

Apply the runtime skill's
[mirror lifetime contract](../ovxr-library-runtime/mirror-lifetime.md)
for application-owned readback leases, copied staging, resize and reconnect.
A same-session desktop resize changes publication bounds, not headset source
resolution. Prove new real image dimensions and connection epoch where SHM is
recreated, retained pixel order/metadata, resumed demand, and changing pixels.

Record the exact UI surface tested. Real React controls through a development
SHM/HTTP adapter establish application behavior, not native Tauri shell behavior.
A `--no-mirror`/control-only run establishes no desktop image claim. Keep its
actual headset CXR/IWER pixel evidence separate. Fake native clients and unit
tests establish cleanup/ordering only; they cannot substitute for the GPU copy
and resize proof.
