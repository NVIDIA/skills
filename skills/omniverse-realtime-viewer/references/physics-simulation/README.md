<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Physics Simulation And OVPhysX Handoff

Use this reference when a realtime viewer needs pick-driven physics, drop tests,
impulses, rigid-body demos, or OVPhysX simulation results rendered in an `ovrtx` viewer.

This reference covers the viewer-side integration contract. Read
`dependencies/nvidia-runtime.md` first; it points to the current OVPhysX
repository and package guidance. Use the upstream OVPhysX `skills/`, examples,
and package metadata for exact tensor, lifecycle, worker, and release-specific
APIs. This package documents only the viewer-side ownership and handoff contract.

## Architecture

Preferred intent:

```text
viewer process
  owns ovrtx.Renderer + live OVStage/session state
  never runs competing USD population for physics

bounded OVPhysX worker process
  loads a physics-capable USD/overlay with its verified worker API
  steps simulation with tensor bindings
  returns pose samples as JSON

viewer process
  writes returned pose matrices through OVStage at monotonic ordinals
  calls renderer.update_from_stage / renderer.step through its normal render loop
```

OVStage is the interchangeable feature plane for the parent viewer. Native
OVRTX picking can choose the prim, OVPhysX can compute pose samples in a worker,
and OVStage carries the reversible runtime pose updates back to the renderer.

## Why subprocess handoff is the safe default

`ovphysx` exposes `PhysX.attach_ovstage(stage, read_ordinal=...)` on the current package contract, but the installed/public OVStage wheel may not export the bridge symbols OVPhysX expects. When attach fails with missing symbols such as:

```text
ovstage_register_consumer
ovstage_register_output_buffer
ovstage_publish
ovstage_query_changes
ovstage_read_attribute
ovstage_write_attribute
```

then do **not** populate USD in the already-running viewer process. That reintroduces competing USD population in the process that owns `ovrtx`/OVStage and can crash native code.

Instead, isolate OVPhysX in a child process only when the worker API is verified against the installed package. The parent must only consume JSON pose samples and write them through the viewer's live OVStage/session path. If no supported path is verified, report the compatibility block.

## Pick-Driven Impulse Pattern

Use the native picking/selection path to choose the rigid body, then hand only a
small request DTO to the physics worker:

```text
OVRTX pick result
  -> canonical selected prim path
  -> physics request DTO
  -> bounded OVPhysX subprocess with its verified worker API
  -> JSON pose samples
  -> parent OVStage omni:xform writes at ordinals N..N+k
  -> attached renderer consumes those ordinals
```

The pick handler should not call OVPhysX directly and should not write
transforms directly through OVRTX. It should enqueue intent for the runtime
owner so picking, selection outlines, physics scheduling, and render stepping
stay serialized.

## Worker contract

The worker request should be a small JSON object:

```json
{
  "stage_url": "./samples_data/stage01.usd",
  "path": "/World/Cube",
  "impulse": [260.0, 120.0, 40.0],
  "angular_velocity": [0.0, 7.5, 4.0],
  "angular_impulse": [0.0, 2.0, 3.0],
  "steps": 180,
  "steps_per_frame": 1,
  "dt": 0.0166666667,
  "device": "cpu",
  "timeout_s": 45.0,
  "physics_overlay_mode": "auto"
}
```

The worker response should include:

- `result` / `status`,
- `samples` with translations, quaternions, matrices, and rotation deltas,
- `child_add_usd_used: true` when the child loaded the USD,
- `parent_inprocess_add_usd_used: false`,
- `displacement_magnitude` and `max_rotation_delta_degrees`,
- binding counts/shapes and OVPhysX package version,
- child stdout/stderr/return code for diagnostics.

Keep worker stdout JSON-only or write results to an output JSON file. Native logs should go to stderr or copied log files so JSON parsing is deterministic.

## Hero stages without physics APIs

Most hero/demo USD stages are visual assets, not physics fixtures. Do not mutate the source USD to add physics APIs.

For a physics demo on a hero stage:

1. Keep the viewer rendering the original hero USD.
2. In the child process, generate a temporary physics-capable overlay/copy.
3. Apply rigid body, collision, mass, physics scene, and any child-only ground/collider to matching prim paths such as `/World/Cube`.
4. Run OVPhysX against the temporary worker stage.
5. Map returned pose samples back to the same live viewer prim path.
6. Delete/let temp files expire after the child process exits.

The report should explicitly state:

```text
parent stage mutated: no
child overlay created: yes/no
source stage: <hero USD>
worker stage: <temporary overlay/copy>
mapped parent prim: /World/Cube
```

## Parent render-loop handoff

The parent process must:

1. Queue the physics request from the UI/data channel.
2. Run the worker in a bounded child process with timeout.
3. On success, schedule returned matrices into the render loop.
4. Write each matrix through OVStage with monotonic ordinals.
5. Let the single renderer owner call `update_from_stage` / `step`.
6. Never use deprecated direct OVRTX prim transform writes for this proof path.
7. Do not route worker samples through selection animation or transform gizmo
   helpers unless those helpers are explicitly layered on top of the current
   physics pose and still publish one OVStage transform per frame.

Use the same row-major matrix convention as the camera and transform references: row 3 holds translation.

## Failure behavior

The viewer must survive worker failures:

- child timeout,
- nonzero child exit,
- invalid JSON result,
- missing rigid body bindings,
- zero displacement/rotation when motion was requested,
- native worker crash.

Report the blocker in UI/debug state and logs. Do not silently show a red X or claim physics passed.

## Validation scope

Keep routine validation light. Do not add broad physics/browser artifact generation to default CI unless the product explicitly requires it.

For demo/proof runs, capture one focused browser MP4 and a concise report proving:

- actual viewer UI/browser path,
- hero stage or requested stage opened,
- selected/clicked prim path,
- `parent_inprocess_attach_ovstage_used` is true, or
- `bounded_worker_used` is true with the worker API/version recorded,
- child overlay/copy status for hero stages,
- OVStage write ordinals and renderer consume ordinals,
- deprecated/direct OVRTX transform writes are `0`,
- nonzero displacement and rotation delta,
- server stayed alive.

Do not commit private scene captures, large validation reports, or environment-specific logs.
