# Interfaces and asset parity

## Policy and controller

The supported export observes `eef_pos`, `eef_rot_6d`, `socket_kp_pos`, and
`socket_kp_rot_6d` (18 values), and emits a six-dimensional relative pose action.
The stock trained action controls `flange` with no body offset. Observations use
TCP offset `[0, 0, 0.1925]`. These are distinct semantics.
Export also requires the stock observation functions and asset selectors:
EEF observations read `robot`; socket observations read `dp_socket`. Matching
dimensions alone do not establish a compatible trained interface.

The bundled adapter consumes the scaled Cartesian delta plus the observation
pose, subtracts the rotated observation TCP offset, and applies the delta at the
flange. Route the stock decoder output to an unused topic to avoid two competing
target publishers. Do not add another TCP conversion on top of the adapter.
The LEAPP model already scales actions; `action_blend=1` adds no further reduction.

The simulator controller uses the saved operational-space gains: stiffness
`[300,300,300,30,30,30]`, damping `[35,35,35,1.1,1.1,1.1]`, no inertia decoupling or
nullspace control. Fixed-base Jacobians omit the root body: use the flange body
index minus one when the Jacobian tensor has `max_links - 1` blocks. Validate the
scene script patch instead of assuming a matching source layout.

The published launcher uses CPU PhysX for this single scene; training uses its
own physics backend. Physics runs at 240 Hz and the policy at 30 Hz. Both ROS processes use simulation
time and the same dedicated ROS domain. At each 30 Hz boundary the scene publishes
its observation and clock, then holds physics until the matching ROS command
arrives. It applies that command before allowing the physics step to advance.
The wait is bounded to five wall-clock seconds; stale commands are drained and a
future timestamp or timeout invalidates the recording. Policy weights, OSC gains,
contact properties and the 240 Hz physics step are unchanged by this handshake.

Record `policy-sync.jsonl` with each boundary timestamp, command timestamp and
wall-clock wait. The recorder checks matching, consecutive 30 Hz responses and
near-complete coverage after the initial mode/gate transition. Do not replace the
handshake with a fixed sleep: host load, rendering and encoder speed otherwise
change which physics step receives an action. Triton readiness establishes that
the server is available; it does not establish per-step command synchronization.
Wait for Triton readiness before starting the scene; absence of policy targets is
an infrastructure failure, not a valid insertion result.

## Assets

Read URLs from **that checkpoint's saved `env.yaml`**, not the latest task source.
The public `release-r1` training defaults use these production-bucket references:

- Robot: `Assets/Isaac/6.1/Isaac/IsaacLab/Robots/Flexiv/Rizon4s/rizon4s_with_grav.usd`
  from NVIDIA's production bucket.
- Plug: `Assets/Isaac/6.1/Isaac/Props/Factory/display_port_cable_assets/displayport_plug.usd`.
- Socket: `Assets/Isaac/6.1/Isaac/Props/Factory/display_port_cable_assets/displayport_socket_no_protrusions.usd`.

The staging bucket remains allowlisted for historical checkpoints whose saved
`env.yaml` references it. It is not required by the current public defaults.
Preserve those saved references when replaying an older checkpoint; do not
silently substitute production assets or rewrite the checkpoint's provenance.

Open the published `flexiv_displayport_insertion.usd` scene for ROS integration,
replace all three asset references with saved training references, and apply saved
spawn, mass, articulation, actuator and material settings. Remove tutorial fixture
and support objects absent from training. Geometry, visual materials and collider
definitions must match the saved references; all authored properties must read back
correctly. Failure aborts before an insertion claim.

The scene contains an executable NVIDIA ROS ScriptNode. `scene_security.py`
pins the complete reviewed USD and embedded script SHA256 values, rejects changed
bytes and additional ScriptNodes, and checks integrity before opting into execution.
The local controller addition and final executable payload also have reviewed
hash pins. Setup reads articulation poses and Jacobians through public Isaac Sim
APIs; it does not separately execute a second copy of the scene script.
Training references are restricted to HTTPS NVIDIA production asset paths, with
staging paths retained for the historical checkpoint compatibility described above.
An upstream scene update requires review and a new pin; never bypass a mismatch.

Apply the saved material events, including finger friction 0.75, plug/socket
friction 0.001, and zero restitution for the supported task.

Cache flattened USD references and SHA256 identities per session. Preserve
upstream assets and source repositories. A saved URL does not prove that today's
remote asset is byte-identical to its training-time version.

## Success and scope

Use the saved task's terminal position criterion: plug/socket mate-point distance
strictly below 3 mm. Also report multi-keypoint distance and orientation. A brief crossing of the threshold and final
success are separate fields. Record the entire bounded attempt, including the
initial hold and final state, without teleporting during recorded policy control.
Resetting/prepositioning before recording is disclosed. No pickup mission, real
robot driver, success-rate benchmark or policy retraining is part of this skill.
