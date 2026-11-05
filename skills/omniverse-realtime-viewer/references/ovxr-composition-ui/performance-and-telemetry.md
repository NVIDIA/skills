<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Spatial UI Performance And Telemetry

Render FPS is the primary budget. Separate state-changing work from pose-only
work and keep both away from blocking XR-frame operations.

## Retained And Asynchronous Work

- Build hierarchy/property models only when stage generation or requested data
  changes.
- Run Yoga and Skia off the XR thread and publish the newest complete immutable
  revision through a bounded latest-wins handoff.
- Upload dense UI pixels only for a new document revision.
- Pre-author analytic controller, ray, tooltip, gizmo, and chrome geometry;
  update bounded pose/vertex data instead of creating prims.
- Extract stage meshes, bounds, and selectable mappings at generation
  boundaries. Keep USD traversal and `pxr` out of the frame loop.
- Use one immutable input snapshot and one bounded transform publication batch.
- Cache stable tracking-to-world and model metadata for the input lifetime;
  invalidate them on their explicit lifecycle boundaries.

Do not deep-copy status trees, serialize proof artifacts, poll Fabric, launch an
idle ray query, or log unchanged pose dictionaries every frame. Explicit
audits are finite and armed; they stop automatically after consuming their
bounded evidence.

A persistent raster subprocess is still synchronous if the XR owner writes to
its pipe and waits for the reply. Hover, capture and selection changes must
enqueue bounded latest-wins revisions; keep showing the previous complete
raster and its matching hit map until the new pair is ready. A response timeout
does not bound a blocking request-pipe write. Test interaction bursts and a
stalled worker, not only unchanged-panel steady state.
Keep at most one render in flight, one replaceable pending request, and one
complete result. Bind subprocess readers to their own process generation;
discard late output after reset. Give both request writes and responses a
worker-side deadline and interrupt a stuck child on shutdown. Retry only a
bounded number of times while retaining the last complete image/hit-map pair.

Account for bytes per revision: 2048x1024 RGBA8 is 8 MiB, four times 1024x512.
Raising raster resolution does not justify uploading it for pose changes.
Coalesce obsolete revisions and measure raster/IPC/upload latency separately.
Do not scan a full RGBA canvas for routine diagnostic coverage on the XR owner.
Compute bounded diagnostics in the raster worker or arm a finite audit. Track
geometry upload bytes separately from texture bytes; a narrowly defined
transfer counter is not necessarily total host-to-GPU traffic.

## Required Counters

- Raster: model generations, render revisions, texture uploads, unchanged
  revision upload attempts, renderer queue drops and latency.
- Primary path: injected/omitted frames, per-eye draws and scissors, full-eye
  copies/bytes, transfers/bytes, same-frame waits, slot exhaustion.
- Timing: bounded CPU compose and queue-submit windows; GPU timestamps read
  only after an existing frame-slot fence has signaled.
- Whole owner loop: include input, post-submit UI work, diagnostics and capture
  contention; a timer stopped at scene submission hides later work.
- Geometry: per-eye accepted, clipped, composed, and draw-included surfaces,
  bounds, index spans, and rejection reasons.
- Input: epoch, sample/display time, owner/focus/activity/validity, explicit
  reference space, ray hit/UV/distance, capture phase, and cancellation reason.
- Identity: native integration build/hash, manifest or hook selection, runtime
  versions, and process start time.

Unavailable measurements remain unknown, not passed.

## Connected-session measurement and targeted optimization

Bound connection waiting separately from the capture window. Exclude
disconnected pacing and shader warmup from steady headset FPS. Report frame
interval median/tails plus gesture-active versus idle windows; correlation is
not causation. Preserve scene/view, resolution, material policy, foveation and
mirror demand for A/B runs. Stop a one-shot collector after its requested
session, disconnect, deadline or error; do not silently rearm it.

Timing lanes can nest: SDK wall time includes runtime waits and child
begin/bind/submit lanes; controller time can contain gesture and tool work.
Never sum overlapping lanes. Native compositor CPU timing can exclude earlier
mesh construction. Report coordinate conversion, mesh packing/construction,
mirror capture/copy/publication and post-submit owner work explicitly; unknown
GPU or transport latency stays unknown.

Before rewriting a renderer, measure bounded CPU work: batch 26/52-joint
conversion against one active anchor snapshot, pack analytic strokes into
reusable contiguous buffers, preserve native mesh capacity, and avoid creating
unused frame-result wrappers. Keep a scalar reference for equivalence checks,
one immutable input snapshot, and a documented fallback for unavailable batch
capabilities. Do not cache an anchor across a real change or drop validity flags.
Disable unused fixture/physics work rather than merely hiding its visuals.
One-time hide/reset publication is acknowledged only after successful owner
publication; include activation generation so stale async results cannot revive
an earlier fixture. CPU microbench improvements are not headset FPS evidence.

## Proof Boundaries

Track these separately:

1. Application state was accepted.
2. A finite per-eye mesh was composed.
3. The submitted draw and scissor included its indices and bounds.
4. The depth/alpha contract was valid.
5. Measured fragments survived scissor/depth and changed the primary output.
6. Correlated decoded or headset pixels showed the surface.

A later boundary cannot be inferred from an earlier one. If bounded fragment
instrumentation is needed, consume results only after an already-signaled
production fence; do not add a second draw, full-eye readback, or same-frame
wait.

## Steady-State Gate

Across a settled interval with an unchanged panel raster, require zero
unchanged-revision texture uploads, primary full-eye copies/transfers,
same-frame fence waits, steady Fabric reads, and unbounded diagnostics. Require
an explicit draw/scissor budget per eye and bounded CPU/GPU timing samples.
Batch compatible surfaces, allowing measured foreground/depth-policy splits;
count actual draw commands separately from render passes and scissor changes.
Pose updates may upload bounded geometry even when texture uploads are zero.
Report texture, geometry and total host-upload bytes separately; do not redefine
zero-copy to mean no full-eye copy while hiding these smaller transfers.

Compare matched scene/view, resolution, foveation, mirror demand, material
policy, build and warmed shader state. Report steady and interaction-time
frame intervals separately. A short startup GPU timing window cannot explain
an unrelated end-to-end FPS difference. Isolate AR material/refraction/cutout
options individually while rechecking correct transparency and edge quality.

Count native eye capture independently from mirror publication. Disarming a
one-shot reader or hiding a preview does not disable create-time native
capture. End diagnostic-only capture through the documented context lifecycle;
do not rebuild a context merely because a shared mirror still needs capture.
For create-time-only capture, explicitly opt the diagnostic into a context
refresh afterward (which may require client reconnect), or require an existing
capture consumer. Coalesce cleanup with a pending lifecycle change. Cancel old
workers before detach and reject their completion against replacement contexts.
Attempt/time budgets bound application retries, not an unresponsive GPU driver.

Exercise retirement/disconnect/context refresh with GPU work in flight.
Fence polling in the steady loop does not prove bounded teardown. Define
timeout and device-loss handling without freeing resources the GPU still owns.
