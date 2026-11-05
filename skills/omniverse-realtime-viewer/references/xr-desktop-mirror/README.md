<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# XR Desktop Mirror

## Triggers

Use this reference for: mirror the headset view on the desktop, spectator window
for an XR session, show the XR view in a Tauri or Electron app, desktop viewport
that becomes the headset preview, one renderer for both the headset and a window,
which render product feeds the stream, pixel order swapped in the mirror, red and
blue swapped, `--mirror-fps` measured lower than requested, publish rate gate,
two ovrtx sessions on one GPU, handoff when a headset connects or disconnects.

Read `ovxr-frame-loop-and-readback/README.md` for the readback API itself and
`ovxr-foveation-and-display/README.md` for what decides the capture's size and
warp state. This reference is about everything between that capture and a window.

## One session, one stream, and a product handoff

A second renderer can duplicate GPU allocations and render work. Reuse the
XR renderer for a desktop mirror and measure memory, render time and delivery
cost in the target application.

Build it as one process instead. One `ovrtx.Renderer`, one OVStage, one
`ovstream` SHM stream whose name never changes, and **exactly one render product
feeding that stream at a time**:

| Phase | Product feeding the stream | Viewport |
| --- | --- | --- |
| No headset | the app's own perspective-camera render product | interactive: orbit, pick, gizmo |
| Headset connected | ovxr's left-eye frame readback | display-only; that camera belongs to the headset |

Use one frame owner and one renderer step. When the headset owns the view,
feed the mirror from its readback instead of issuing another desktop step.
If a separate application render product is requested, use the selected SDK's
`set_frame_render_products` contract to join the same OVXR bind step. Multiple
products in that supported step do not create another renderer session; measure
their additional work and preserve OVXR's result lifetime.

Attach and detach are driven by the runtime's connect/disconnect events. **A
disconnect does not end the process**: destroy the context, resume the desktop
product, and let the next client get a fresh context. Keep "stop the XR session"
distinct from "stop the process" — the process owns the only renderer there is,
so killing it blanks the viewport and re-pays the PSO compile.

The desktop product is what makes the window useful during the several-second
renderer construction and the indefinite wait for a client, which is exactly when
the window would otherwise be empty.

## The consumer must not switch streams

A stream per producer looks tidier and is worse. The SHM client's only handle is
`connect`, which **drops the live client before trying a new name**, so a
switch initiated from the UI blanks the viewport and breaks input for the whole
wait — and a webview-side probe is the worst version of that.

Do the handoff in the producer. Because the two products are different sizes, the
producer restarts its SHM server across the swap, and a consumer that already
handles "the producer restarted" as an ordinary reconnect needs no new mechanism
at all. Do not add a stream-preference flag, a client-side probe, or a handoff
gated on scraping the producer's log tail.

## Both current producers emit RGBA; carry the actual order

ovstream's SHM ring has a single fixed format tag (`OVSTREAM_SHM_FORMAT_BGRA8`,
see `dependencies/ovstream.md`), so it does not independently describe the producer’s actual byte order.
Both current OVRTX LdrColor and OVXR readback emit RGBA8; neither requires a
producer-side swizzle when the consuming shader correctly honours RGBA.

Resolve the selected capture format through upstream frame-readback via
[the SDK route](../dependencies/ovxr.md), then match the consumer's byte order.
Use red/green/blue fixtures to verify the full display path. A matched consumer
may avoid a swizzle; standard consumers need their advertised format.

Whatever the answer, if the two producers can differ, the producer must *state*
which order is on the wire — the consumer cannot infer it:

- not from the stream name (there is only one),
- not from the process state (the process being up is not the headset owning the
  stream),
- and **not from an out-of-band announcement alone**.

### Why an announcement alone loses

An announcement and a frame race, and the frame always wins. The announcement
crosses the control channel, an `app.emit`, an IPC bridge and a React re-render;
the frame it applies to reaches the draw loop through an already-parked long-poll
that resolves instantly. Keying the shader off React state drew the first frames
of every handoff with the previous product's order, and a webview reload with a
headset attached reset that state to the default and painted swapped red and blue
until the next status poll landed.

Send an announcement and poll-visible status, but bind their metadata to a
producer/connection epoch and source revision. Feed a **host-side latch** and
stamp validated metadata into the application-owned frame envelope. This is
not a change to ovstream's fixed native SHM header. For example:

```
[epoch][source_revision][width][height][sequence][pixel_order][pixels...]
```

Define fixed field widths/versioning in that application protocol. The shader
reads `pixel_order` from the frame, not asynchronously updated React state.
An announcement sent before pixels is not proof that the host received it first.
Before accepting pixels, require metadata for the attached producer epoch and
source revision. Gate a source-format transition with an acknowledged metadata
handshake while publication is paused, or exactly join metadata to a frame key.
Without such correlation, use a fixed wire format (swizzle to BGRA for standard
consumers) instead of guessing from the latest announcement.

A webview reload can preserve the host's validated metadata for the **same**
producer epoch. A producer restart/SHM reconnect invalidates that metadata:
clear old image, sequence, dimensions, pixel order and foveation, await the new
producer's metadata, and reject old callbacks. Do not display frames with an
old/default order during the gap. This is the generation-aware contract in
[mirror lifetime](../ovxr-library-runtime/mirror-lifetime.md).

The cost of mislabelling is real and should be made deliberate: a stock SHM
client pointed at this stream shows red and blue swapped while a headset is
connected. Give the stream a distinct default name so that cannot happen by
accident, and if a third-party consumer is ever needed, swizzle in the producer
rather than teaching every client about the exception.

## Rate gates must advance from a deadline, not from completion

The mirror's cost lands on the **headset**, so the publish rate is the lever that
matters. Get the gate wrong and the cap is not the number you asked for.

Stamping "last publish" *after* the send folds two unrecoverable costs into every
interval:

- **the send itself** — a full-frame device-to-host copy into the SHM ring, ~5.6
  ms at an 18 MB readback size;
- **the wait for the next frame to be offered** — frames arrive on a grid (the
  headset's capture interval, and the publisher thread's poll), so a deadline
  missed by a microsecond costs a whole grid step.

33.3 + 5.6, rounded up to the next 11.1 ms capture or 15.6 ms poll, is 44-47 ms:
a requested **30 fps measured 21.5 fps**. Keep a `next_publish_at` deadline and
advance it *by the period*, so slack is spent rather than compounded. Re-base to
`now + period` only when more than a whole period late (cold start, rate change,
stall) — never burst to catch up, because a backlog of 18 MB frames is the one
thing a bandwidth cap must not emit.

### Windows timer resolution is the hidden grid step

That 15.6 ms poll is not a choice. A Windows process that has not called
`timeBeginPeriod` gets the default timer resolution, so
`threading.Event.wait(0.002)` — or any equivalent short sleep — actually sleeps
**~15.5 ms**. Assume the coarse value when reasoning about any pacing loop on
Windows, or raise the resolution deliberately and accept the system-wide cost.

Two more gates worth having:

- Publish nothing beyond a low-rate liveness heartbeat while no consumer is
  attached, and skip the copy entirely when nothing is waiting for it.
- Keep status/control reachable before the first capture, for example with a
  control-only SHM carrier. Report no image/source until real pixels arrive; a
  1×1 carrier is not a rendered frame. Resize/reconnect on the first real frame.

## The mirror must never take down the session

Every failure path in the mirror logs and returns a failure value. A missing
`ovstream`, a consumer that vanished, a size change mid-session — none of them may
propagate into the XR frame loop. The mirror is a convenience; the session is the
product.

Move slow transport work off the frame owner. Finish the source copy under the
[application capture lease](../ovxr-library-runtime/mirror-lifetime.md) before
publishing owned storage. Follow upstream frame-readback for concurrency and
pointer lifetime; a concurrent acquisition path does not reserve its pixels.

## Common mistakes

| Mistake | Consequence | Prevention |
| --- | --- | --- |
| A second process for the desktop viewport | Two ovrtx sessions on one GPU; ~9.5 GB VRAM, headset 33 → 21.7 fps | One process, one renderer, two render products |
| Calling a second renderer step for the mirror | Duplicate render work and conflicting frame ownership | Reuse eye readback, or join requested app products to the same OVXR bind step |
| A stream per producer, switched by the consumer | `connect` drops the live client first; viewport blanks for the whole wait | One stream name; swap the product in the producer |
| Inferring pixel order from stream name or process state | The process being up is not the headset owning the stream | State it explicitly and carry it in the frame header |
| Driving the swizzle from React state | First frames of every handoff drawn with the previous order; a reload paints swapped colours | Latch host-side, stamp into the frame header |
| Reusing old/default pixel order after producer reconnect | First frames can have swapped colors | Await new-epoch metadata; preserve the host latch only for a same-epoch webview reload |
| Stamping "last publish" after the send | 30 fps measured 21.5 fps; looks like a producer being slow | Advance a deadline by the period |
| Assuming a 2 ms sleep is 2 ms on Windows | Every pacing calculation is off by a 15.6 ms grid step | Default timer resolution unless `timeBeginPeriod` is called |
| Letting a mirror failure reach the frame loop | A vanished consumer takes down the headset session | Log and return; never propagate |
| Ending the process on client disconnect | Viewport blanks and the next session re-pays the PSO compile | Destroy the context, resume the desktop product |

## Related References

- `ovxr-frame-loop-and-readback/README.md` — `acquire_frame`, buffer lifetimes, consumer threading
- `ovxr-foveation-and-display/README.md` — what decides the capture's size and warp state
- `ovxr-openxr-runtime/README.md` — bring-up order, ordinals, context lifetime
- `dependencies/ovstream.md` — SHM frame contract, single format tag, no GPU-shareable handle
- `webgl-shm-transport/README.md` — texture upload and the shader-side swizzle
- `tauri-local-viewer/README.md` — moving frames across the webview boundary
- `xr-client-connection/README.md` — the connect/disconnect events that drive the handoff

## Independent readback, sizing and truthful availability

Use upstream frame-readback and foveation-and-display through
[the SDK route](../dependencies/ovxr.md) to request and inspect the capture
domain. Select the intended eye from the actual negotiated view configuration.
Do not override headset submission dimensions with desktop preferences.
After capture, resize on the GPU without upscaling, preserving aspect ratio
inside the requested bounds. For example, 1024×896 fits 1280×720 as 822×720.
Use a public GPU-resize path supported by the selected package, such as its
NPP scaler when available; do not assume an undocumented renderer helper. If resize fails and full-size fallback is allowed, report
that the requested bound was not met; otherwise omit that mirror frame.

If independent capture remains unavailable while the headset stream is live,
show a preview-unavailable warning while retaining the XR session and its
requested settings. Test disabled and warped modes separately; success in one
does not prove capture in the other. Record exact package identities in the
test evidence rather than treating one prerelease limitation as an SDK guarantee.

Expose requested readback domain independently of actual source, last sequence
and capture availability. With no capture, source remains none and image domain
unknown. SDK-unwarped capture carries no PQW correction metadata, preventing a
second inverse warp in the client. Clear stale misses/errors when fresh capture
recovers. Validate changing head motion and actual decoded desktop pixels,
not only SHM connection status or a control carrier.

Coordinate copy, resize and publication through
[mirror lifetime](../ovxr-library-runtime/mirror-lifetime.md); the transport
must receive application-owned storage, not a borrowed SDK pointer.

## High-rate XR mirror presentation

Measure WebView delivery separately from renderer cadence. A raw frame may take
several IPC hops; an acknowledgement gate can serialize them. If presentation
limits the requested mirror rate, use a supported custom-protocol pull path
with bounded concurrent requests and a single latest-frame slot. Responses may
arrive out of order, so gate drawing by the payload sequence number. Carry
pixel order and other frame-specific metadata in the same payload header.
See [SHM pixel transport](../webgl-shm-transport/README.md) and
[XR desktop mirror](../xr-desktop-mirror/README.md).
