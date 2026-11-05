<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Omniverse Realtime Viewer Conventions

These conventions are the shared behavior contract for all focused references in
this skill package. If a focused reference needs one of these values, use this file instead
of inventing a local rule.

## Architecture

- All USD and 3D rendering uses `ovrtx`.
- Browser-streamed apps display an `ovstream` video stream in a video element.
  The browser does not render USD geometry.
- Desktop apps display frames rendered by `ovrtx` in-process or through local
  pixel transport.

## Mouse And Input

These mouse bindings apply to desktop/browser viewports. XR uses
[spatial input](xr-spatial-input/README.md) and its capture/activation policies;
do not translate a hand pinch into desktop mouse orbit or force every XR node
to activate on release.

- Left mouse button drag: orbit.
- Middle mouse button drag: pan.
- Right mouse button drag: dolly/zoom.
- Scroll wheel: zoom; scroll up zooms in.
- Left click selection fires on mouse release, not press.
- A press becomes a drag when either axis moves by more than `1.0` pixel.
- Local ovui button IDs are remapped before calling shared camera code:
  `0 -> left/orbit`, `2 -> middle/pan`, `1 -> right/dolly`.
- WebRTC input uses the NVST native input channel. The browser streaming
  library forwards binary `InputEvent` structs to ovstream; React does not
  implement client-side 3D camera math or send JSON camera input.
- SHM input uses `ovstream.Client(ovstream.ClientType.SHM, stream_name="...").send_input_event(event)` from Python or
  `ovstream_client_send_input_event(client, event)` from C with `InputEvent` structs.
  Do not send JSON `mouseInput` for SHM camera control.
- In-process transports call the Python/C++ camera, selection, and settings
  APIs directly.

## Selection

- Default viewer behavior is single-select. Selecting a new prim replaces the
  previous selection; clicking empty space clears selection. If multi-select is
  requested, every subscriber must explicitly support mixed values and multiple
  highlighted prims.
- Viewport selection should use the selected delivery path's native picking
  route first, then a documented fallback when native picking cannot resolve a
  selectable prim.
- Selection state is keyed by stable USD prim paths and synchronized across the
  viewport, tree, property panel, and any status or info surfaces.
- Selection feedback should be renderer-visible and work for arbitrary valid
  USD scenes. Prefer native outlines or selection groups when the selected
  renderer path supports them.
- Material-driven glow, visibility changes, or shader-parameter effects are
  optional pick effects. Use them only when the active stage exposes compatible
  targets or the user explicitly asks for that behavior.
- Selection animation is optional and product-specific. If requested, keep it
  parameterized, reversible, and safe for the stage's units and coordinate
  system; do not assume a fixed lift direction, duration, or asset scale.

## Viewport And Rendering

- Choose render size from the delivery skill and product requirements. Keep it
  fixed for a session unless the `viewport-resize/README.md` skill is explicitly selected.
- UI resize scales the displayed image; it does not dynamically resize the
  render product unless the `viewport-resize/README.md` skill is explicitly selected.
- Browser video uses `object-fit: contain`.
- NVST maps pointer coordinates between the contained video and intrinsic stream
  resolution. Local apps use the visible image content rect for the same
  letterbox mapping.
- Standard ovstream consumers expect BGRA8; convert or colorize AOVs on the
  producer. An explicitly application-controlled XR SHM mirror may instead
  preserve RGBA with a matching consumer and generation-bound frame metadata,
  as specified in [XR mirror](xr-desktop-mirror/README.md). This exception is
  not compatible with stock SHM consumers; never swizzle twice or infer order.
- Shader bake/compile can take time on first load. Complete warmup before
  accepting client connections when startup latency matters.

For client-gated XR runtimes, renderer warmup and listener/client readiness are
separate: follow [XR bring-up](ovxr-library-runtime/library-only.md). Do not wait
for a client-bound XR context before allowing that client to connect.

## Camera

These camera controls concern application-owned desktop/browser cameras. OVXR
owns headset eye cameras and their frame schedule. Apply navigation through
the XR anchor contract; a desktop companion must not orbit the headset eyes.

- The camera is a USD prim, updated by writing `omni:xform`.
- Camera matrices are row-major: row 0 = right, row 1 = up, row 2 = `-forward`,
  row 3 = eye/translation.
- Fit the camera to the stage on initial load unless the app restores an
  explicit saved camera state.
- Camera gizmos are ovui overlays: local apps draw them in the viewport UI;
  streaming apps composite server-side ovui output into the BGRA stream.

## Scene Loading

- User USD files are not modified by viewer setup.
- Viewer camera, render product, render vars, settings, and selection metadata
  live in a session layer or composite wrapper.
- Clear selection, hover, temporary effects, and any viewer-authored runtime
  overrides on every stage load.
- Serialize population, scene attachment/replacement, runtime writes, and
  `renderer.step()` through the runtime owner; see `stage-management/README.md`.
- Detect the scene root dynamically on the server and pass `root_prim_path` to
  clients instead of hardcoding `/World`.

## Physics And Runtime Simulation

- Choose a supported shared-stage integration or an explicit worker boundary
  using `physics-simulation/README.md` or `newton-physics/README.md`.
- The application owns the simulation clock and publication. Give each animated
  transform one writer; camera/UI commands and physics updates meet at the
  runtime owner rather than racing independent stage mutations.
- Keep generated physics overlays and transient state separate from user USD.
  Worker boundaries carry copied poses/status, not borrowed native handles.

## Streaming Protocol

- WebRTC signaling and media ports are selected by the streaming/deployment
  skills. Do not hardcode deployment-specific ports unless the selected reference
  or hosting environment requires them.
- App messages travel over the data channel as JSON with
  `{ "event_type": "...", "payload": {...} }`.
- Receive application JSON through ovstream `on_app_message`; the transport
  removes its envelope before app dispatch. Native pointer/key input uses
  `on_input`. See `streaming-messages/README.md` for the shared app protocol.

## Environment

- Set `OVRTX_SKIP_USD_CHECK=1` before importing or constructing `ovrtx`.
- For a stage-first attached renderer, call `ovrtx.register_schema_paths()`
  after importing `ovrtx` but before constructing or first populating OVStage,
  and before any other OpenUSD consumer initializes. Importing `ovrtx` alone
  does not register those paths unless its explicit auto-registration option was
  enabled before import.
- Import/setup order for streaming servers is:
  environment variables -> `ovrtx` + schema registration -> `ovstage` create /
  population -> renderer attachment -> streaming helpers -> `pxr` only behind
  the chosen isolation boundary.
- Keep `renderer.step()` ownership on one render thread or UI loop.
