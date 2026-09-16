<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Streaming CAE Viewer (ovrtx → ovstream WebRTC → React)

How to deliver the CAE/CFD viewer to a browser: server-side `ovrtx` RTX rendering, streamed
as an `ovstream` WebRTC H.264 video to a React client, with DOM controls that drive
server-side OVStage/CAE scene ops over a JSON data channel. This matches the RTWT demo's
browser delivery.

**The streaming shell is a known quantity — compose it, do not re-document it.** The generic
mechanics (server startup, client connect, message envelope, lifecycle, cloud deploy) live in
the streaming skills; this doc only covers what is *CAE-specific*: the server scene runtime
that reuses the CAE operator backend, the CAE message set, and how each React control maps to
a CAE OVStage op.

Read first for the generic pieces, then return here:

- `references/streaming-viewer-recipe/README.md` — the end-to-end browser-viewer recipe.
- `references/streaming-server/README.md` — server process, ovstream `Server(WEBRTC)`, BGRA.
- `references/streaming-client/README.md` — the React + `@nvidia/ov-web-rtc` client.
- `references/streaming-messages/README.md` — the `{event_type, payload}` protocol + envelope.
- `references/streaming-lifecycle/README.md` — connect/readiness/reconnect/teardown.
- `references/viewer-backend-interface/README.md`, `references/viewer-control-patterns/README.md`
  — the control → backend command pattern this specializes.
- `references/cloud-deployment/README.md` — container, ports, TURN, scaling.

This doc's CAE write pattern (build stage, publish, `omni:xform` data-plane commit, write
floor) is documented once in **[ovstage-render-and-camera.md](ovstage-render-and-camera.md)**;
cross-link that rather than repeating the OVStage write details here.

## 1. Architecture recap

The server process owns everything: `ovrtx` (RTX render), `ovstage` (live stage substrate),
`ovstream` (WebRTC video + data channel), and an application-owned CAE operator backend. The
browser is video + DOM only — it never renders USD, runs no visualization operators, and holds
no scene truth; it shows a `<video>` element plus React control panels, forwards mouse input
natively, and sends/receives JSON on the data channel. This is exactly the topology in
`references/streaming-viewer-recipe/README.md`; the CAE-specific substitution is *what the
single render-loop owner does per tick* (§2) and *which app messages it speaks* (§3).

## 2. The CAE server runtime

Create one `SceneRuntime` as the serialized owner of OVStage + ovrtx: **exactly one thread ever
calls `renderer.step()`, mutates the stage, or writes `omni:xform`.** The ovstream callbacks
(`on_message` / `on_input` / `on_connection`) run on StreamSDK threads and only *enqueue*
onto that runtime's command queue — they never touch OV state directly.

### Build the CAE operator backend in the app

Keep CAE code in application modules, for example:

```text
cae_runtime/
  ingestion.py       # native file / array normalization -> Dataset
  operators.py       # representation(dataset, request) -> geometry + fields
  usd_authoring.py   # geometry + fields -> USDA layer
  camera.py          # numpy orbit state -> row-major camera xform
  scene_runtime.py   # queue, publish/write-floor, renderer.step, frame output
```

`operators.py` loads the selected dataset, computes the requested representation, exposes its
field range and supported representations, and returns arrays to `usd_authoring.py`.
`scene_runtime.py` builds/publishes the USDA layer and commits the camera data-plane matrix.
Implement those contracts from [data-and-operators.md](data-and-operators.md),
[usd-authoring-and-materials.md](usd-authoring-and-materials.md), and
[ovstage-render-and-camera.md](ovstage-render-and-camera.md); do not import an external sample
application or modify `sys.path`.

### Startup order

Same order the streaming server skill mandates; the CAE-specific step is #4 (warm up the CAE
stage so first-use renderer work is paid before any client connects):

1. `os.environ.setdefault("OVRTX_SKIP_USD_CHECK", "1")` **before** importing `ovrtx`/`pxr`.
2. Construct `ovrtx.Renderer` **first** (`RendererConfig(sync_mode=True, active_cuda_gpus="0", keep_system_alive=True)`).
3. `import ovstream` (Warp is initialized lazily by the CAE backend on first use).
4. Build the initial CAE stage and **warm up**: rebuild → attach → `step()` in a loop until the
   first BGRA frame is produced (allocates the persistent BGRA + fallback buffers along the way).
5. `ovstream.initialize(...)` — ref-counted; the matching `shutdown()` runs **only at process exit**.
6. `stream = ovstream.Server(ovstream.ServerType.WEBRTC)`.
7. Register `stream.on_connection` / `on_message` / `on_input` **before** `start()`.
8. `stream.start(ovstream.ServerConfig(width, height, target_fps, video_input=VideoInput.CUDA, webrtc_signal_port=49100, webrtc_public_ip=...))`.
9. Start the `/healthz` gate (503 until the first converted frame, then 200).
10. Start **one** render-loop thread.

Use the current upstream `ovstream` documentation and the selected deployment guidance to configure signaling, media, health, and display requirements. Do not infer ports, client limits, or headless-display requirements from this reference.

### Per-tick contract (`render_tick`)

On the single render-loop thread, each tick:

1. **Drain the command queue** — apply camera events in order; coalesce rebuild-affecting
   commands last-wins (a dragged slider queues at most one rebuild). Returns whether a rebuild
   is needed.
2. **If rebuild needed:** reload data if the dataset changed, recompute the representation
   (`representation(...)` / `emitter_streamlines(...)`), detach the prior stage, author and publish
   the new stage layer, then `renderer.attach_ovstage(stage)`. At most once per tick.
3. **Commit the camera** at a new ordinal via the data plane (a scene-runtime camera commit
   → PathDictionary → `query_from_path_list` → `write_attribute(MATRIX, UPSERT)`), which advances
   the write floor. Committing every tick (even an unchanged pose) at a fresh ordinal keeps
   frames correctly ordered. See [ovstage-render-and-camera.md](ovstage-render-and-camera.md).
4. **Step + convert:** `renderer.step(render_products={rpp}, delta_time=dt, ordinal=N)` →
   map `LdrColor` on CUDA (`render_vars["LdrColor"].map(device=Device.CUDA)` → `wp.from_dlpack`)
   → copy into the persistent BGRA buffer + swap R/B → return it. `None` means no fresh frame
   this tick; the loop streams the last-good / fallback frame.

The render-loop thread (the streaming server, not the runtime) then calls
`stream.stream_video(ovstream.VideoFrame.from_cuda_array(bgra))`.

### LdrColor RGBA8 → BGRA8 (persistent CUDA buffer)

ovrtx emits `LdrColor` as `uint8 [H, W, 4]` **RGBA**; ovstream's CUDA video input needs
`uint8 [H, W, 4]` **BGRA**. Convert on-GPU with a Warp kernel — no CPU round-trip:

```python
@wp.kernel
def swap_rb(img: wp.array3d(dtype=wp.uint8)):
    i, j = wp.tid()                 # ONE thread per pixel (dim=(H, W))
    r = img[i, j, 0]; b = img[i, j, 2]
    img[i, j, 0] = b; img[i, j, 2] = r
```

Launch with `dim=(H, W)` (per-pixel). Launching with a channel axis `(H, W, 4)` runs the swap
for both `k==0` and `k==2` and **cancels out** — keep the 2-D form. Copy the mapped LdrColor
view into an **app-owned persistent BGRA buffer** and swap in place: the mapped view is only
valid inside its `map()` context and is reused by the next `step()`, and `stream_video()` does
**not** copy — it reads the buffer until the next call — so the buffer must stay alive and the
CUDA stream must be synchronized before handing it over.

Keep a **persistent fallback frame** (opaque black, A=255) and stream last-good / fallback every
tick while idle or loading — WebRTC drops the connection after ~7 s without frames. Run long
stage loads without blocking the heartbeat. Render is fixed 1920×1080 (ovrtx has no resize; the
client uses `object-fit: contain`).

### Render-loop stability (guardrails any streaming viewer needs)

A single render-loop thread that also serves a live client has failure modes a local window
never sees. Three guardrails, general to any renderer + client:

1. **Pre-warm each supported GPU feature once at startup, before the loop serves and before any client connects.** First use of a renderer feature may block frame production. Exercise only the supported geometry paths the app will use, then begin live streaming. This package does not support NVIDIA IndeX or direct-volume paths; do not add a volume pre-warm.

2. **Verify render-loop stability with a real client — a headless sequential harness does not
   reproduce live-streaming failures.** Some failures only appear under the **paced** render loop
   with the streaming path active and a client connected (live H.264 encode + GPU contention +
   per-frame CUDA interop). A feature-specific startup stall can be invisible to a "select example
   → render N frames sequentially" test and only surface with a browser client attached. **A passing
   headless test is necessary but not sufficient** — validate under production-like conditions (a
   continuous loop paced to the frame budget, streaming path live, a real or faithfully-stubbed
   client) before declaring a render-loop fix done. The `--selftest` headless path (§5) proves
   scene + frame + swap, but confirm the live leg with an actual client.

3. **Crash-isolate GPU tests in a child subprocess — native RTX asserts are modal.** An
   `ovrtx` / `rtx.scenedb` assertion calls `abort()`, which on Windows pops a **modal dialog**
   that blocks the process indefinitely and interrupts an interactive user. When exercising
   crash-prone GPU paths, run them in a **child subprocess** launched with
   `SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX | SEM_NOOPENFILEERRORBOX)` plus
   CRT `_set_abort_behavior(0, ...)` at process start, so an abort becomes a **silent non-zero
   exit** you detect programmatically. Detect hangs with a **heartbeat file** the child rewrites
   each rendered frame — no update within a timeout means a freeze (see guardrail 1). This keeps a
   GPU crash from hard-blocking CI or the developer's session.

## 3. The CAE message set

Extends the baseline `{event_type, payload}` protocol from `references/streaming-messages`.
Baseline events (`openStageRequest`/`openStageResult`, `getChildrenRequest`, `selectPrimsRequest`,
`changeAOVRequest`, `cameraCommandRequest`/`cameraStateChanged`, `setViewportInputActive`,
`viewerError`) are unchanged. The CAE extensions are app-specific — **reserve them centrally in
one message module and gate them via capabilities** so a generic client degrades cleanly:

| Client → server (command) | Payload | Server op (enqueued, coalesced) → `sceneState` |
| --- | --- | --- |
| `setRepresentation` | `{mode}` | set representation → rebuild |
| `setField` | `{name}` | set active field → rebuild |
| `setColormap` | `{name}` | set colormap LUT → rebuild |
| `setIso` | `{frac}` (0–1 of field range) | isosurface level → rebuild |
| `setSlice` | `{frac, axis?}` (0–1 along axis) | slice plane → rebuild |
| `setStreamlineSeed` | `{center:[x,y,z], radius}` | move emitter seed → `wo.emitter_streamlines` → rebuild |
| `setEmitterVisible` | `{visible}` | toggle the seed-sphere marker |
| `loadDataset` | `{id}` (registry id; `path`/`url` are legacy aliases) | resolve id → reload dataset (refit) |
| `getSceneState` | `{}` | push `sceneState` (no state change) |

Every command except `getSceneState` is validated, enqueued, and last-wins–coalesced on the render
loop; once accepted it triggers a (coalesced) `sceneState` push so all clients reconcile. **How each
becomes an OVStage write** (author → publish → advance write floor for geometry, or a data-plane
`omni:xform` / array write for the camera and emitter) is
**[driving-cae-viz-via-ovstage.md](driving-cae-viz-via-ovstage.md)** — do not restate it here. Keep
these CAE extensions in one message module so a generic client (which knows only the baseline events)
degrades cleanly.

Server → client is a **server-authoritative** state push — the client holds no scene truth, it renders
whatever the last `sceneState` gave it:

- `sceneState` — a COMPLETE snapshot in the **client-facing** shape:
  `{datasets:[{id,name}], fields:[{name,label?}], representation, field, colormap, iso, slice,
  emitter:{center,radius}, show_emitter, activeDataset, colormaps?, bounds?}`. Compose it by remapping
  the render-loop-owned runtime's `snapshot()`; the internal snapshot keys (`dataset`, `bbox`,
  `representations`) are **not** the wire keys — normalize to `activeDataset`, `bounds`,
  `datasets:[{id,name}]`, and tolerate legacy path-string dataset entries so the picker always populates.
- `cameraStateChanged`, `viewerError`.

### Message framing (the headline gotcha): wrap every server→client send

The single biggest trap, and the one that cost the most time: **`ovstream.Server.send_message(str)`
puts a raw string on the data channel, but `@nvidia/ov-web-rtc` only surfaces a message to
`onCustomEvent` if it arrives as a wire frame `{messageRecipient, messageType, data}`** — it routes by
`messageRecipient` and **drops a bare string silently** (no error, no console log). So
`send_message(json.dumps({"event_type": ..., "payload": ...}))` never reaches the client.

The two directions are symmetric: the client already frames its outgoing sends the same way, which is
exactly *why* the server unwraps `{messageType, data}` on receive. So **client→server worked from day
one; server→client failed silently until the server framed too.** The `sceneState` push, `viewerError`,
`openStageResult`, `cameraStateChanged` — all invisible until wrapped.

Server send — wrap the app envelope inside the wire frame:

```python
def send_event(self, event_type, payload):
    if self.server is None or not self.server.is_client_connected:
        return
    app = json.dumps({"event_type": event_type, "payload": payload}, default=str)
    frame = json.dumps({"messageRecipient": "app", "messageType": "json", "data": app})
    self.server.send_message(frame)          # send_message(app) alone is silently dropped
```

Server receive — unwrap the same frame the client wraps, then read the app envelope:

```python
def decode_app_message(raw):
    msg = json.loads(raw) if isinstance(raw, str) else raw
    if isinstance(msg, dict) and "messageType" in msg and "data" in msg:
        data = msg["data"]
        msg = json.loads(data) if isinstance(data, str) else data
    return msg if isinstance(msg, dict) and "event_type" in msg else None
```

### The control → server-op path (the core)

Each React control follows one path:

> React control → `{event_type, payload}` **wrapped in the `{messageRecipient,messageType,data}` frame**
> over the data channel → server unwraps + validates → `scene.enqueue_command(kind, payload)` → render
> loop drains (last-wins coalescing) → OVStage write / geometry rebuild at a new ordinal → advance write
> floor → next `renderer.step(ordinal=N)` → stream the next BGRA frame → server pushes a coalesced
> `sceneState` (wrapped) → client reconciles.

Concrete emitter example: React drags a seed sphere → `setStreamlineSeed{center, radius}` → the
`_h_streamline_seed` handler validates `center` is `[x,y,z]` and enqueues → the loop coalesces to
the latest, rebuilds via `wo.emitter_streamlines(wds, center, radius, color_field=field)`, authors
BasisCurves, publishes at ordinal N, steps, streams. `setRepresentation`/`setField`/`setIso`/
`setSlice`/`setColormap` map the same way — each an enqueued, coalesced command.

The CAE write details (author → `open_usd_from_string` → `advance_write_floor` → `attach_ovstage`,
and the data-plane `omni:xform` commit) belong to
**[ovstage-render-and-camera.md](ovstage-render-and-camera.md)** and
**[driving-cae-viz-via-ovstage.md](driving-cae-viz-via-ovstage.md)** — this doc reuses those patterns,
it does not restate them.

### sceneState hydration (server-authoritative)

Push a COMPLETE `sceneState` at three moments so every attached client converges on the truth:

1. **On connect** — from the short-delayed initial-state thread, ~0.3 s after the data channel is
   ready (alongside `openStageResult` + `cameraStateChanged`), so the dropdowns populate immediately.
2. **After `loadDataset`** — a dataset swap rebuilds the field list + bounds asynchronously on the
   loop, so push at **two horizons** (e.g. ~0.4 s and ~1.5 s) to catch the rebuilt fields.
3. **After any accepted state-changing command** — **coalesced** (at most one trailing push in flight)
   so a slider drag doesn't flood the data channel or fight the client's optimistic echo.

The client applies each `sceneState` to populate the dataset + field dropdowns and to clear its
"waiting for scene state" gate (the first push flips `ready`). The dataset picker sends
`loadDataset{id}`; the server resolves that id through its dataset registry to a concrete path.

### Example presets — the "uber-UI" walkthrough pattern

To turn a bare control panel into a guided walkthrough that reproduces named CAE examples,
drive the UI from **declarative example presets** the server advertises in `sceneState`.
Each preset binds a registered dataset to a representation, a default field/colormap/domain,
**the set of control groups the UI should surface** (`controls`), and a note on the exact
settings it replicates. Selecting a preset drives the scene *and* reconfigures the panel:

```python
# server preset (application-owned declarative data)
{
  "id": "time_varying_structural", "number": 1, "title": "Time-Varying Structural",
  "dataset": "bumper_beam", "representation": "transient",
  "field": "PlasticStrain", "colormap": "gist_rainbow",
  "domain": [0.0, 0.065], "domain_locked": True,
  "controls": ["timeline", "field", "colormap", "domain"],   # gates which UI groups show
  "example_notes": "Faces colored by PlasticStrain; locked domain 0-0.065; ...",
}
```

- **Server side:** ship the preset list (`list_examples()` / `get_example(id)`), add a
  `setExample {id}` command that loads the dataset, sets the representation/field/colormap/
  domain, and advertises the active example + its `controls`. Example 3 (Multi-Domain) is
  intentionally backlogged — omit it from the list rather than half-wiring it.
- **Client side:** render an **example picker** (chips) → `setExample`; then **gate control
  groups by `example.controls`** so a transient example shows the Timeline group, a volume
  example shows Volume/ROI/Context, an aero example shows the Emitter group. Show the
  `example_notes` as a concise explanation of the selected visualization.

Extend `sceneState` to carry the preset contract so the client can render the guided UI
without hardcoding it:

```
sceneState += {
  examples: [{id, number, title, subtitle, controls, example_notes}],
  activeExample: id | null,
  timeline: {t, n_frames, playing, fps, speed, interpolate},   # transient examples
  domain:   {min, max, locked},                                # colormap range + lock
  transform:{scale, upAxis, offset}, lineWidth,                # alignment controls
  context:  {available, visible},                              # USD context asset toggle
}
```

Each field is server-authoritative and echoed back after the corresponding `setX` command
(`setExample`, `setTimeline`, `setPlaying`, `setInterpolate`, `setDomain`, `setDomainLock`,
`setVolumeParams`, `setRoi`, `setContextVisible`, `setTransform`, `setLineWidth`). Keep them
in the one CAE message module (§3 head) so a generic client degrades cleanly.

### Robustness: a control-panel render error must never black the video

The single highest-severity frontend trap: **a React render error in the control panel can
unmount the whole app — including the `<video>` element — which drops the WebRTC session**
(the symptom is "flash one frame, then black; connect/disconnect cycling ~400 ms"). It looks
like a codec/transport failure (you may even see an unrelated `Unsupported Video Codec` log)
but it is a **crash in a sibling component tearing down the video**.

Root cause seen in practice: the server sends `emitter: {center: null, radius: 0}` for a
non-emitter dataset, and the panel does `center[axis].toFixed(...)` on `null` → throws →
with no error boundary the whole tree unmounts. Two defenses, use **both**:

1. **Null-guard every server-sent state field** the panel reads. Make nullable fields
   nullable in the type (`emitter.center: Vec3 | null`) and coalesce at the read site
   (`const center = scene.emitter.center ?? [0,0,0]`), in both the reducer and the panel.
2. **Wrap the control panel in a React error boundary** so a panel exception is caught and
   rendered as a fallback *without* unmounting the `<video>`. The video and the WebRTC
   connection must be structurally independent of the control panel's render success.

A single bad or missing field (a nullable emitter center, an empty timeline, a missing
domain) must never be able to take down the stream. Treat the video subtree as sacrosanct.

### Handler rules (from the baseline)

- Every callback runs on a StreamSDK thread: validate, enqueue, return fast; never step or write.
- **Unwrap** the browser frame `{"messageType":..,"data":"<json>"}` before `json.loads`, and
  symmetrically **wrap** every outgoing send in that frame (see *Message framing* above) — a raw
  `send_message` is dropped silently. Dispatch by exact `event_type` through a dict; unknown types
  warn and no-op.
- Send through one guarded helper that no-ops when `not server.is_client_connected` and swallows
  disconnect-race failures; cap data-channel messages at ~60000 bytes.
- On connect, push initial state on a short-delayed thread: `sceneState` + `openStageResult` +
  `cameraStateChanged` (~0.3 s after connect so the data channel is ready).

## 4. React client specifics that matter

The generic client (provider, endpoint resolution, video element, subscriber fan-out) is in
`references/streaming-client`. What is worth calling out here:

### Browser client dependency

Use the current `ov-web-rtc` acquisition and API guidance from the central `nvidia-runtime` dependency guide and the upstream package materials it links. Read the installed package types before choosing a connection API, event callback, config shape, or codec value. Keep this CAE guide focused on the app protocol rather than package-version behavior.

Verified-shape connect config (static/`DirectConfig` form, as built against the real package):

```ts
import { AppStreamer, StreamType, VideoCodec, type DirectConfig } from '@nvidia/ov-web-rtc';

const config: DirectConfig = {
  videoElementId: 'remote-video', audioElementId: 'remote-audio',
  server: host, signalingPort,                 // Direct fields: server + signalingPort ONLY
  width: 1920, height: 1080,
  codec: VideoCodec.H264, codecList: ['H264'],
  nativeTouchEvents: true, fps: 60, maxReconnects: 5, reconnectDelay: 3000,
  onStart: () => {}, onCustomEvent: dispatch, onStop: () => {},
};
await AppStreamer.connect({ streamSource: StreamType.DIRECT, streamConfig: config });
```

Do **not** set a media/`mediaPort` field — media is SDP-negotiated UDP.

### Receiving is shape-agnostic — unwrap recursively

Use the receive callback and payload contract documented by the currently installed browser client. Normalize the app envelope at one boundary; do not depend on undocumented transport-specific shapes.
Depending on the build / transport path it hands the callback the app message as any of:

- an already-parsed object `{event_type, payload}` (the documented `messageType:'json'` path), **or**
- a raw JSON **string** (unparsed), **or**
- `{data:"<json>"}` (legacy/GFN frame), **or**
- `{message:"<json>"}` (multi-part reassembly whose payload failed to parse) —

and combinations of these (e.g. a `{data}` whose value is itself a JSON string). So the client must
**unwrap recursively** — `JSON.parse` any string, then descend `data` / `message` — until it reaches an
object carrying `event_type`, then read `{event_type, payload}`; reject anything else. Do not assume the
documented `messageType:'json'` → `JSON.parse(data)` path is the only one that fires. Symmetrically, the
client wraps its own sends in the `{messageRecipient,messageType,data}` frame (§3 *Message framing*).

### Rules that avoid "one frame then black"

- Keep the `<video id="remote-video">` **React-rendered so it exists in the DOM before
  `connect()`** is called.
- The connect `useEffect` deps must be the **immutable host/port** (and the stable `dispatch`)
  only — never app state or the subscriber set. Tearing the connection down on ordinary re-renders
  is the classic one-frame-then-black bug.
- Two readiness signals are distinct: the **resolved `connect()` Promise** means the data channel
  is ready to send app commands (`status === 'connected'`); **`onStart`** means video is decoding
  only. Do **not** gate the first `getSceneState` / `openStageRequest` on `onStart`.
- Mouse / keyboard / wheel / touch are forwarded **natively** by the library — never send them as
  JSON. Only app state + commands go over the data channel. Arm/disarm native input by region:
  viewport `onPointerEnter` → `setViewportInputActive{true}`, control panels → `{false}`.
- Keep app state server-authoritative: the client optimistically echoes a `setX` for slider/select
  responsiveness, then reconciles on the next `sceneState`.

## 5. Deploy + honest verification note

For containerization, ports (49100 signaling / UDP media / health), TURN, and scaling, use
`references/cloud-deployment/README.md` — do not hardcode deployment-specific ports here.

Honest status: the **server side is verified end-to-end** — `ovstream.Server(WEBRTC).start()`
binds `:49100`, produces ~60 fps BGRA frames, `/healthz` returns 200, and the CAE scene builds,
attaches, renders, and converts via `swap_rb` (there is a `--selftest` path that proves scene +
frame + swap + `ovstream.start` headlessly). The **React client compiles and builds against the
real `@nvidia/ov-web-rtc`**, and the **message-framing fix (§3) is confirmed** — once the server
wrapped its sends, `sceneState` reached `onCustomEvent` and the dropdowns hydrated. Full **browser
WebRTC media negotiation still needs a real client** against a GPU host to confirm the video leg —
treat that leg as unproven until run.
