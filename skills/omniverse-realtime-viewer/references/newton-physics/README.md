<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Newton Physics in a Realtime Viewer

Use Newton with ovnewton when a viewer needs continuous simulation of rigid
bodies, articulations, or deforming geometry. The application owns the simulation
clock; ovnewton connects the model to OVStage and does not step the solver.
Read `realtime-sim-integration` for frame scheduling, asynchronous work, and
capture lifetime, and `interaction-round-trip` for user control of simulation.

Use [Newton's documentation](https://newton-physics.github.io/newton/stable/)
and [examples](https://github.com/newton-physics/newton/tree/main/newton/examples)
for model construction, asset import, solvers, and tuning. Use the selected
ovnewton package's examples for its binding API and `dependencies` for acquisition.
This reference defines the viewer adapter rather than duplicating engine APIs.
For OVPhysX integration, see `physics-simulation`.

Before implementing this adapter, check the selected cohort through
[the ovnewton dependency gate](../dependencies/nvidia-runtime.md#ovnewton-and-newton).
Resolve compatibility from the selected public bridge release and validate its
simulation and renderer paths before enabling controls. Keep controls unavailable
when that combination is incompatible or unvalidated. A compatible package set
does not by itself qualify viewer or XR integration.

## Model, stage, and opening state

Choose the source of the simulation model before attaching:

- **Stage-first:** parse delivered USD physics content. Check that the resulting
  model supplies the topology and solver-specific data the selected solver needs.
- **Model-first:** construct or import a Newton model, then provide a matching
  USD stage for rendering. Reuse the asset's geometry instead of replacing it
  with primitive proxies unless that is the requested visualization.

Map simulated bodies to absolute prim paths in `model.body_label`, with the
physics schemas and joint prims required by the binding. Validate the mapping at
attachment, including any joint labels that resolve as USD paths. When a workflow
requires external schema fallbacks, register the owning package's schemas before
population; use `ovstage-population` for registration.

Preserve the intended opening state across attachment. A binding may synchronize
model state from the stage, so a generated stage must agree with the model's
opening pose. Reconcile generalized coordinates and body poses using the selected
Newton model's FK/IK path before stepping. Choose the authoritative representation
from the model's origin, rather than guessing from whether an array contains zeros.

Author stage units and up axis explicitly. Convert geometry, gravity, collision
margins, inertia, control targets, and camera framing consistently. Solver
coefficients have different physical dimensions; use the solver's unit contract
instead of multiplying every parameter by one scale factor.

## The application adapter

Keep the scene declaration separate from the runtime. A scene supplies its model,
solver factory, visual mapping, and optional controller. The runtime owns:

- model and state buffers, solver, controls, and contacts;
- the ovnewton binding and renderer transform publisher;
- the fixed simulation clock, publication ordinals, and stage generation;
- run, pause, step, reset, and attachment status exposed to the UI.

Keep simulation controls unavailable until the scene and runtime are compatible.
A lightweight availability check can explain missing packages or physics content
before a user presses Play; it should not attach or mutate the live stage.

The following uses **application-owned adapter methods**, not ovnewton API names:

```python
def tick(runtime, frame_dt):
    runtime.drain_inputs()  # camera, animation, control; active generation only
    completed = runtime.harvest_completed_step()  # nonblocking; None if pending
    if completed is not None:
        if not runtime.poses_are_finite(completed):
            runtime.pause_with_error("Simulation produced invalid poses")
        else:
            runtime.accept_simulation_snapshot(completed)

    # One transform model composes camera, animation, gizmo, and accepted poses.
    updates = runtime.compose_frame_updates(frame_dt)
    if updates:
        ordinal = runtime.next_ordinal()
        runtime.write_frame_updates(updates, ordinal).wait()
        runtime.stage.advance_write_floor(ordinal).wait()
        runtime.committed_ordinal = ordinal
    runtime.render_committed(runtime.committed_ordinal, frame_dt)
    if runtime.running:
        runtime.enqueue_steps_within_budget(frame_dt)
```

The update composer retains the last accepted simulation pose while processing
camera and animation changes on every tick. Writers participate in the owner's
transaction; they do not allocate ordinals or advance the floor themselves. Use
`prim-transform-safety` for the concrete OVStage matrix writer.

Harvest only completed GPU work; use the producer's stream/event contract before
CPU readback. Batch publication, then enqueue the next simulation work so the
renderer can overlap it where supported. Bound catch-up work to preserve input
responsiveness. Publish an initial held pose at attachment. Render the latest
committed snapshot while physics is paused or pending so camera and UI updates
remain independent of simulation progress.

## Publish renderer-visible state

Treat physics data and renderer transforms as separate outputs of the adapter.
Inspect the installed binding's output contract: publishing velocities or
`omni:fabric:worldMatrix` alone does not author the renderer's local transform.

Publish `omni:xform` through OVStage using the matrix semantic and layout in
`ovstage-data-plane`. Newton rigid body poses use position plus **xyzw**
quaternion; convert them to USD's row-vector matrix convention with translation
in row 3. Convert world poses into the body's parent space before the local
transform write. Preserve authored scale and reflection; rigid poses alone do
not contain them. Use `prim-transform-safety` for transform composition.

Source poses from the public simulation state, such as `state.body_q`, rather
than private binding buffers. Publish each body's transform once; child geometry
inherits it. Scope optional telemetry writes to actual consumers.

Use one ordinal allocator. Complete all transform and deformation writes for a
snapshot before advancing the floor, then render that ordinal. Feed user edits
into physics deliberately; do not blindly import the latest stage state every
frame and consume the simulation's own published poses as new input.

## Interaction, reset, and rebuild

Queue impulses and control changes to the runtime owner. Update host-side
controllers before the solver substeps, outside any captured CUDA graph. For
per-substep control, use an engine-supported device-side path. Keep simulation
time separate from publication counters so dropped display frames do not change
the controller's trajectory.

Reset both state buffers, velocities, controller state, and solver history through
the selected solver's reset API. Define whether Reset restores the authored pose
or a warmed-up opening pose, save that snapshot consistently, and publish it
before showing the app as reset.

Changes to body/joint counts or labels require a new binding and matching output
storage. Solver or substep changes may invalidate capture. On scene replacement,
stop new work, drain device operations, release the old publisher and binding,
and ignore results from the previous stage generation. Keep a non-captured path
so graph capture remains an optional optimization.

## Deforming geometry

Map simulation outputs to the corresponding USD attributes: mesh points,
point positions, or PointInstancer positions/orientations/scales. Author the
renderable topology and materials at scene setup. Publish related arrays
coherently: mesh topology must agree with points; instancer indices and per-instance
arrays must agree with instance count; primvar interpolation must match the data.

Write deformation and rigid transforms for the same simulation snapshot at the
same publication ordinal. Keep source CPU/GPU storage alive until writes finish.
When counts or topology change, rebuild the affected bindings and verify the
rendered result; do not infer resizing support from an accepted buffer write.

## Validate the integration

Compare the opening, moving, paused, and reset poses in the actual viewer. Check
that the selected body responds to input, render and physics poses agree, and
scene changes retire old work. Measure simulation, publication, rendering, and
input latency separately on the target scene and GPU. Use repeated runs with
solver-appropriate tolerances for nondeterministic behavior.

For tracked-hand grasps, remote holds and cable pulls in XR, read
[XR physics interaction](../xr-spatial-input/physics-interaction.md).
