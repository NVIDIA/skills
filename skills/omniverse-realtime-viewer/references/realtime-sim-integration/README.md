<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Simulation in the Viewer Loop

Use this reference to combine a stateful simulation with an OVRTX viewer through
OVStage. The application runtime owns scheduling and publication; scene adapters
supply models, solvers, asset mappings, and controls. Keep asset discovery and
scene catalogs outside the runtime so it can serve different scenes and engines.

Use `newton-physics` or `physics-simulation` for engine integration,
`interaction-round-trip` for user control, and `ovstage-data-plane` for writes.

## The frame contract

Choose scheduling deliberately. A synchronous loop finishes each physics step
before publishing and rendering; it is simple but its input latency includes
that physics work. An asynchronous loop polls a completion event and harvests
only completed simulation state. It renders the last accepted simulation pose
while camera and other viewer edits continue to publish.

```text
per viewer tick, on the runtime owner:
    drain input for the active stage generation
    poll outstanding simulation work without blocking
    if complete: harvest state and validate finite poses
    compose camera, animation, gizmo, and accepted simulation updates
    write all changed state at one owner-allocated ordinal K
    wait for those writes, then advance and wait for the floor once
    render the latest committed ordinal, even while physics is paused/pending
    enqueue bounded fixed-step work when running and no conflicting work is active
```

Only the runtime owner allocates publication ordinals and advances the floor.
Feature modules contribute updates to that transaction. Give a simulated body's
transform one owner, or compose its animation/gizmo offset explicitly before the
final write; combining dictionaries is not an ownership policy.

GPU state must be complete before readback, but this does not require blocking
every viewer tick. Use producer events and retained buffers to harvest finished
work. When overlap is unsupported by the engine, use the explicit synchronous
path and measure its latency. Drain outstanding work before reset or detachment.

Keep publication separate from stepping so attachment and reset can publish a
held pose. Never publish non-finite simulation state; preserve the last good
pose and pause physics while navigation remains usable.

## Attach, pause, and reset

Publish an initial held pose at attachment, even if the simulation starts paused.
Make publication independent of solver stepping. Generated body prims may start
at identity until the simulation's opening pose reaches the renderer.

Reset through the solver's supported reset API, restoring state buffers,
velocities, controller state, and solver history coherently. Clear queued input
and catch-up debt, then publish the reset pose. Define whether Reset returns to
the authored state or the warmed-up opening pose and save that snapshot once.
While paused, continue processing navigation and other viewer feature updates.

## Stage generations and shutdown

Scene replacement, detachment, or model rebuild retires a simulation attachment.
Tag requests and asynchronous results with the active stage generation:

```text
stop new work for generation G
finish its outstanding device operations
release its binding, publisher, and captured graph
discard queued commands and results for G
construct and attach generation G+1
publish its opening pose before reporting readiness
```

Ignore late results instead of applying them to matching paths in the new stage.
A reset can retain a binding if its model and topology are unchanged, but still
invalidates queued work. Keep control state unavailable during a transition and
allow navigation once the new committed stage is displayable.

## Publication cost

Batch transforms into one columnar write, ordered to match the stage query.
Complete all attribute writes for the tick, then call
`advance_write_floor(ordinal).wait()` once. Enqueueing alone does not complete
publication. Advances are monotonic: an older request leaves a higher floor
unchanged. Query the authoritative floor on attachment or for diagnostics rather
than rereading it after every completed advance.

Publish optional velocities, joint state, and telemetry only for active
consumers. Measure stage-call overhead as well as transferred bytes. Use
`prim-transform-safety` for local matrix conversion and the concrete writer.

## Pacing and GPU capture

Use a fixed simulation time step with bounded catch-up work. Measure step cost
and input latency to choose the budget; cap queued simulation work as well as
completed frames. Report lagged simulation state instead of delaying all viewer
input to repay accumulated simulation time.

Keep GPU stream and graph-capture ownership explicit in the process. Warm up
before capture, coordinate with other GPU libraries, and use producer completion
events before reading captured outputs. Invalidate capture after topology,
solver, device, or relevant scheduling changes. Retain an uncaptured path and
choose capture only when a matched measurement justifies it.

For pipelining, harvest completed work, compose and publish viewer changes,
render, then enqueue the next bounded simulation work. Draining is required for
reset/detach, not on every interactive tick. Engine integrations that cannot
safely overlap work should use the synchronous contract explicitly.

## Control and failure behavior

Call host controllers before each simulated frame's substeps, outside captured
graphs. Give controllers their own simulation-frame counter; a dropped display
frame must not alter a trajectory. Use engine-supported device control for
per-substep actuation. Surface controller failures and pause or hold according
to the product's interaction model.

Validate finite poses before publication. On divergence, retain the last good
pose and show a distinct paused/error state while the viewer remains usable.
Liveness counters alone do not establish simulation correctness: test the
requested physical behavior, such as contact or controlled displacement, in
addition to checking rendered motion.

## Scene preparation and validation

Check initial collision overlap, support, and joint travel across the intended
motion range. Convert physical parameters by their dimensions and stage units;
not every parameter scales like length. Derive framing and gesture magnitudes
from the selected subject's bounds rather than unrelated backdrop geometry.

Validate first pose, motion, pause, reset, and scene replacement. Exercise camera
and animation changes during a deliberately slow or paused simulation and check
that each rendered frame consumes the owner's latest committed publication.
Compare simulation, publication, rendering, and input latency on the same scene,
solver configuration, duration, and device placement. For nondeterministic
solvers, repeat behavior checks with appropriate tolerances.

For a solver running asynchronously beside OVXR, follow
[XR simulation lifecycle](../xr-spatial-input/simulation-lifecycle.md).
