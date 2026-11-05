<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Physics in a Realtime Viewer

Use OVPhysX for rigid-body simulation, pick-driven impulses, and drop tests.
This reference covers the connection between simulation, the viewer runtime,
and UI. Use `dependencies/nvidia-runtime.md` and the selected OVPhysX package's
skills and examples for compatible packages, session reads/writes, and engine APIs.
For Newton use `newton-physics`; both engines share the scheduling guidance in
`realtime-sim-integration`.

## Choose the integration boundary

Use caller-owned OVStage attachment for continuous simulation when OVPhysX and
the viewer use a supported compatible package set. One application owner orders
stage publications, physics stepping, and renderer consumption. Input callbacks
enqueue intent rather than stepping physics or mutating renderer state directly.

Use a separate worker when the feature is a bounded computation, such as a drop
test with replay, or when physics needs a separate supported runtime environment.
The worker owns its scene and simulation; the viewer consumes pose samples. A
worker is an application architecture choice, not an OVPhysX engine limitation.
Resolve incompatible packages through `dependencies`; do not invent a bridge or
mix independent USD population paths inside the active viewer process.

## Shared-stage flow

After resolving the external schema provider through `dependencies/nvidia-runtime.md`,
register both PhysX and deformable schema resource directories in each process
that reads USD schemas, including workers:

```python
from ovphysx.samples.python_samples.schema_support import physx_schema_paths
import ovstage

ovstage.population.register_usd_schemas([str(p) for p in physx_schema_paths()])
```

The helper is a shipped sample recipe, not a core OVPhysX API. It locates the
provider through distribution metadata without importing its stock-USD bindings
or loading `pxr` into the viewer process. See upstream
[Physics Schemas](https://github.com/NVIDIA-Omniverse/PhysX/blob/main/ovphysx/docs/physics_schemas.md)
for the resource layout and native equivalent. Include Newton schemas in the
same registration call when the scene authors `newton:*` attributes.

Registration precedes the first population/export that initializes the schema
registry, including scenes authored with core USD physics properties. Restart the
process after a registration-ordering error; a new Stage or PhysX instance cannot
recover dropped settings. Then use OVPhysX's documented
`PhysX.attach_ovstage(stage, read_ordinal=...)` and stepping flow for the selected
package pair. Collect successful task completion for attach, update, and detach;
submission alone is not success. Keep the Stage alive through successful detach
or instance destruction, and keep sealed input ordinals readable until their
consuming tasks complete.

```text
UI intent -> runtime-owner queue
                  |
       stage edits -> committed input ordinal
                  |
          physics update + fixed steps
                  |
       published poses -> renderer update + step
```

Use one ordinal allocator and obey each consumer's committed range. Physics
output must not be fed back as a new external edit to the same simulation.
Keep fixed-step catch-up bounded so navigation and rendering remain responsive.
See `realtime-sim-integration` for scheduling and GPU capture lifetime.

## Simulation state and cleanup

Use the selected upstream session read/write examples for new integrations.
Type-based synchronous conveniences use `read_sync()` and `write_sync()`; collect
write commits with `commit_sync()` or the operation's successful wait before
stepping. A context-manager exit alone does not collect a commit's outcome.

Copy pose/telemetry values that must outlive a read session. Release every
borrowed read array and downstream view before destroying PhysX, including
references retained by loop variables, UI caches, or pending results. Closing the
session alone does not drop those aliases; live borrows can block destruction
and scene replacement. Stage teardown uses detach plus successful completion;
restoring an initial simulation pose is a separate state-reset operation.

## Worker and replay flow

Capture coherent simulation input on the runtime owner: the source asset plus
current live overrides for participating bodies and colliders, including static
colliders. Include their current transforms and any supported geometry or physics
edits; reopening the source alone loses session edits. Send that copied snapshot
with the canonical target path, requested action, bounded duration, and stage
generation. Return target paths, sample times, poses, and operation status; choose
a transport that fits the sample volume rather than requiring a particular JSON
schema.

Keep the trajectory tied to that snapshot throughout computation and replay.
Either hold conflicting edits to its participants and ancestors, or invalidate
the job and trajectory when those inputs change within the same stage. Camera
navigation and unrelated edits can continue.

Run the worker off the UI/render thread. Provide pending, cancel, and failure
states while the viewer continues navigation. Cancel or ignore its result if
the scene changed. Keep native logs separate from a structured result channel.

Replay through the render owner's OVStage transform writer, one committed
publication per displayed update. `omni:xform` is a **local** transform;
convert world-space simulation poses into the target parent's space before
writing. `omni:fabric:worldMatrix` is derived state, not the renderer's transform
input. Use `prim-transform-safety` for transform composition and
`ovstage-data-plane` for the matrix write contract.

Give one subsystem ownership of a body's pose while simulation or replay is
active. Suspend selection animation and transform dragging on that body, or
compose them explicitly before the single final transform write. Preserve the
pre-simulation pose when the product offers reset/undo.

## Scene preparation and acceptance

A visual USD asset may lack bodies, colliders, mass, or a physics scene. Author
these in a session/overlay owned by the app so the source asset remains intact.
Keep prim paths consistent between simulation and display, and expose only
physics actions supported by the selected object.

When upgrading an existing scene, check the upstream migration guide for retired
joint schemas and changed solver defaults. A scene that loads successfully may
still lose joint settings or follow a different trajectory. Validate joint
limits, driven motion, contact, and reset against the intended behavior rather
than treating finite output or a successful step as sufficient proof.

Edit a body and a static collider before starting a worker job, then verify its
input matches their live state. Verify the requested action changes the intended
body's pose and is visible in the viewer. Also exercise cancellation, scene
replacement during work, and reset. Surface missing physics data or engine
failure as an actionable status; do not leave a successful-looking control
attached to a body that never moves.
