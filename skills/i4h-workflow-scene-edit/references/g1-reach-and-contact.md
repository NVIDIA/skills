# G1 Reach and Contact Authoring

Use this reference when a workflow asks Unitree G1 to face, walk toward, or stop safely near a named object.

## Reuse the locomotion Tasks

Compose `basic/g1_face_object` and `basic/g1_walk_to_object` as separate graph nodes. Do not create workflow-specific copies of facing, walking, heading math, WBC encoding, upright checks, or collision history.

Use `i4h_arena.assets.authoring_catalog.authoring_asset("g1").embodiment` for the registry name, action/joint widths, camera aliases, and verified collision-bearing body names. Use `spawn_pose_facing_rectangle()` for an initial edge-relative position and heading. Do not paste the G1 body list or hand-compute a one-off spawn quaternion into a generated Scene.

The G1 WBC action is 50-D:

- `[0:43]`: measured 43-joint posture, held while locomoting
- `[43:46]`: forward velocity, lateral velocity, yaw rate
- `[46]`: base-height command
- `[47:50]`: torso roll, pitch, yaw

Run it at the Scene manifest's declared control rate; maintained G1 scenes use 30 Hz. A face node commands yaw only. A walk node commands forward velocity only after alignment, tapers its approach, and sends zero navigation command inside the success band.

## Keep geometry in the Scene

The nearest-edge calculation consumes `SceneView.footprint_half_extents(name)`. Measure the object's task-relevant collision footprint from confirmed metric world bounds and return it from the Scene's `footprint_half_extents()` override. Do not duplicate those extents in a Workflow predicate or Task parameters.

Use an oriented-footprint implementation before applying this axis-aligned helper to a rotated obstacle whose world-aligned bounds would materially change the stopping distance.

## Configure robot-object collision sensing

Read the upstream `physics-simulation` and `isaac-sim-sensor` skills. Then inspect the spawned live stage, because the asset wrapper path is not necessarily its rigid-body path. The sensing and filtered paths must each resolve to exactly one supported rigid-contact entry. A path prefix that also matches nested collision or visual prims is not exact enough even when the wrapper itself carries `RigidBodyAPI`.

Isaac Lab's PhysX filtered contact sensor supports one sensing body to many filter bodies, not many-to-many filtering. To detect contact between an articulated G1 and one rigid table body:

1. Enable contact reporting on both asset spawners with `activate_contact_sensors=True`.
2. Identify the table's actual rigid body in the live stage and confirm its filter expression resolves exactly once in the PhysX tensor view.
3. If the imported table hierarchy makes an exact GPU filter impossible, author one invisible kinematic collision proxy with the confirmed table collision footprint. Use its unique rigid-body prim as the filter and keep its dimensions aligned with `SceneView.footprint_half_extents`; avoid leaving a second overlapping collision representation active when that would change task physics.
4. Build the family with `i4h_arena.assets.contact.filtered_contact_sensor_family()`, passing the catalog's G1 `contact_body_names`, the robot prim root, and the unique table body or proxy. Do not include IMU, camera, logo, visual-only, or other fixed decoration prims merely because they appear below the robot wrapper.
5. Name the family `contact_robot_table__<body>`. `ArenaSceneView.contact("robot", "table")` aggregates that family.

Do not attach one filtered sensor to `Robot/.*`; that is unsupported many-to-many filtering. Do not use a collision-mesh child as a filtered rigid body merely because it makes a prefix match narrower: PhysX can reject that collider for GPU contact filtering. Treat `filter_count` errors, “provided patterns ... did not match,” and errors for the configured sensor/filter prims as launch failures. Warnings for descendant visual prims are noisy but are not equivalent to the configured rigid-body prim failing; verify the sensor itself initializes and produces a filtered force.

## Define safe reach success

Require all of the following for a stable interval:

- outside the object's footprint and no farther than the requested edge distance
- upright within the requested tilt
- root linear speed below its threshold
- root angular speed below its threshold
- no robot-object contact at any earlier tick in the episode

Reset collision history on Task entry and episode reset. The Workflow success predicate must use the same reusable helper as the rule-based Task.

## Dynamic validation

Static lint proves only contract structure. A safety-qualified reach Task requires:

1. A normal visible rollout that reaches the band, stops, remains upright, and succeeds.
2. A forced-contact negative rollout that produces a non-zero filtered force, latches collision history, commands stop, and cannot succeed afterward.
3. A fresh reset proving collision history clears and the normal rollout can succeed again.

Record sensor names, force-matrix shape, maximum filtered force, final linear and angular speeds, tilt, edge distance, and success/failure. Idle launch alone is insufficient.
