<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# The Interaction Round Trip (click → impulse → pixels)

Use this reference when a click, a button, or a script in a **streamed** viewport
has to change what a **running** simulation does — flick a body, jolt a scene,
drive a joint to a new target — and the change has to show up in the video the
user is watching.

Every other simulation reference in this skill set moves poses *out* of a solver
and onto the screen. This one is the return leg: getting user intent *in*, across
a thread boundary, without corrupting the solver or lying to the UI.


Use application-defined modules for the simulation runtime, input/protocol adapter, transform publisher, and optional client UI adapter. The names and directories in the generated app are its own design; the contracts below are the reusable requirement.

## Scope: what is here, and where the rest lives

Here: the **input-side contract** — how a viewport hit becomes a body index, how
input crosses onto the render thread, how a gesture is sized so it means the same
thing on any stage, how the UI learns what happened, and how controls get
described once instead of twice. It is written against ovnewton but every rule
below is engine-shaped, not Newton-shaped.

| For… | Go to |
|---|---|
| Newton itself — `ModelBuilder`, solvers, joints, `Control`, articulations | <https://newton-physics.github.io/newton/stable/> and its [`.claude/skills`](https://github.com/newton-physics/newton/tree/main/.claude/skills) |
| Warp kernels, CUDA graph capture mechanics, DLPack | <https://nvidia.github.io/warp/stable/> |
| Getting a pick at all — renderer setup, pickable flags, NDC contract, decoding pick results | `native-picking-selection/README.md` |
| Selection state, highlight, and the click→selection UX | `object-selection/README.md`, `selection-feedback/README.md` |
| Message envelopes, dispatch tables, request/response ids, data-channel limits | `streaming-messages/README.md` |
| The ovnewton attach contract, the publish path, ordinals, reset semantics, the scene declaration a runtime consumes | `newton-physics/README.md` |
| Widget vocabulary — sliders, toggles, confirmations — once the server has declared what to render | `viewer-control-patterns/README.md` |
| Engine-agnostic loop concerns — frame contract, divergence guards, catch-up budget, publish cost, graph lifetime, measurement | `realtime-sim-integration/README.md` |
| OVPhysX's bounded, request-scoped impulse handoff (a different shape entirely) | `physics-simulation/README.md` |
| Where input events enter the server and in what order handlers see them | `viewer-input-routing/README.md` |
| The wider simulation dependency and example matrix | the "Simulation Ecosystem" section of `routing.md` |

## Lead with this: a perturbation is not a command

Almost every "how do I make the user drive the sim" question is really this
question, and answering it wrong wastes the most time.

| | **Perturbation** | **Command** |
|---|---|---|
| What it is | One-shot disturbance | Standing change to *inputs* |
| Concretely | Body impulse `J`, applied as `body_f += J / h` for one substep | `control.joint_target_q = …` |
| Lifetime | Consumed by the next actual substep | Persists until rewritten |
| Applied | Drain once; consume on the next actual solver substep | **Every frame**, forever |
| Implemented as | a queue drained by the render loop | a `control_hook` called per simulated frame |
| Right for | flick, shake, knock over, throw | IK, teleop, joysticks, state machines, scripted choreography |
| Wrong for | anything that must *hold* a pose | anything that should decay naturally |

The trap: `model.control()` is built **once at attach** and a runtime that owns
the clock never rewrites `joint_target` on its own. A scene that only sets targets
at build time can express *constant* actuation and nothing else — periodic motion
has to come out of geometry (a slider-crank), and an upstream example driven by a
per-frame IK solve degrades to a statue with no error anywhere.

So a **command** needs a per-frame hook:

```python
def control_hook(model, state, control, frame_index, dt) -> None: ...
```

- Called **once per simulated frame**, immediately before the substep loop, on
  the render thread with the solver lock held. A tick that replays three catch-up
  frames calls it three times, so a scripted trajectory keeps **sim** time, not
  wall time.
- Called **outside the CUDA graph**, and structurally so: the capture records the
  substep loop and nothing else. That is what makes a host-side IK solve legal —
  the hook may call `.numpy()`, allocate, even capture a graph of its own.
- Write `control` **in place** (`wp.copy`, `assign`, a kernel launch). Rebinding
  an array (`control.joint_target_q = wp.array(...)`) silently does nothing under
  graph replay, because the captured graph holds the pointer the capture saw.
- `state` is the state the coming frame integrates *from*. Read it; do not hold
  it — the substep loop ping-pongs two buffers and the name is valid only for
  this call.
- An optional `hook.reset()` is called by the runtime's Reset, so a state machine
  rewinds with the world instead of lunging at a stack that no longer exists.
- A hook that raises is caught, rate-limited, and **never stops the sim**: first
  failure with a traceback, then one line per N calls (300 works). The symptom is
  a scene that steps but does not act, so surface a `control_hook_errors` count in
  status for exactly that case.

Wire one in two ways, and support both:

```python
# 1. the keyword — preferred
runtime.attach(stage, model=model, control_hook=controller, ...)

# 2. the fallback — tag the model, for a scene that cannot edit the server
model.ov_control_hook = controller          # CONTROL_HOOK_ATTR
```

The keyword wins; the attribute is read only if it is `None`. Refuse
non-callables loudly at attach rather than raising once per frame from inside the
step path. Note that a key in a scene's build dict is **not** forwarded
automatically — a server typically copies a named list of keys and then forwards
them explicitly, and a new key added to only one of those two places silently
does nothing. That is the reason for the model-attribute fallback.

## The round trip, hop by hop

```text
browser click
  → ovstream input event        → input-event handler
  → camera mouse-button-up      → was_click (drag rejected)
  → click handler               → native pick
  → [queue pick, render thread runs it, wait ≤300 ms]
  → gprim path
  → resolve_body                → body index          ← walk ancestors
  → queue_pick_impulse          → bounded queue       ← websocket thread ends here
  ────────────────────────────────────────────────────────────────────
  → render loop: tick()         → harvest previous frame
  → drain impulses              → accumulate pending J ← render thread starts here
  → control_hook                → control.joint_target_q
  → substep loop                → clear forces, add J/h once, solver.step
  → pose_matrices → transform relay writes omni:xform
  → ovstage → ovrtx → ovstream  → browser
  ────────────────────────────────────────────────────────────────────
  → client polls newtonSimRequest{status} at an application-defined interval → newtonSimState
```

| Hop | Owner | Must know |
|---|---|---|
| Input arrives | `input/protocol adapter` | Only a **non-drag** release picks: the camera's mouse-up returns `was_click`. Otherwise every orbit would flick whatever it passed over. |
| Overlays get first refusal | `input/protocol adapter` | Transform gizmo and ovui overlay consume the event before picking. See `viewer-input-routing/README.md`. |
| Pick | the render-loop owner | The pick is itself a queue-and-wait: enqueued for the render loop, waited on for ≤ **300 ms**, and executed just before `renderer.step`. Picking already crossed the same thread boundary the impulse is about to. |
| Pickability | render-loop owner | Include simulated bodies’ renderable gprim descendants in the scene selection policy. Refresh it when the scene or workload changes; hierarchy queries do not alter it. |
| Map hit → body | `simulation runtime` | Rule 1 below. |
| Ownership routing | `simulation runtime` | `owns(path)` is what keeps two engines off the same prim: an `applyPhysicsImpulseRequest` on a Newton body is re-routed to the Newton queue instead of starting an OVPhysX burst that would fight it. |
| Queue | `simulation runtime` | Rules 2–4 below. |
| Drain | `simulation runtime` | Between the harvest and the enqueue — device idle, nothing mid-read. |
| Step | `simulation runtime` | Hook first, then substeps. |
| Publish | `transform-publish adapter` | `omni:xform` (local), **not** `omni:fabric:worldMatrix` (derived). See `newton-physics/README.md`. |
| Reply | `input/protocol adapter` | Rule 6 below. |

## Minimum viable click-to-impulse

Five moving parts. Everything else in this file is a rule about one of them.

**1 — make the simulated geometry pickable.** Resolve each body's renderable
geometry from the live scene, including descendant gprims for container Xforms
and the body itself when it is renderable. Apply the complete scene/workload
policy through the renderer owner after population. Use segment-aware prim
ancestry, not substring matching, and retain ordinary selectable geometry that
the same policy allows. The set does not depend on the client's loaded tree rows.

**2 — map the hit back to a body.** Walk ancestors; never string-parse.

```python
def resolve_body(self, prim_path: str) -> int:
    path = str(prim_path or "").rstrip("/")
    labels = list(self._body_labels)                   # snapshot under the lock
    while path.startswith("/"):
        if path in labels:
            return labels.index(path)
        path = path.rpartition("/")[0]
    return -1
```

**3 — queue, do not touch the solver.** This runs on a websocket thread. Ask
`runtime.owns(hit_path)` first — a hit on a prop belonging to another engine is
a silent no-op, not an error — then call `queue_pick_impulse(hit_path)` and log
the rejection reason if it comes back unsuccessful.

Call it **before** the selection toggle, or a second click on an already-selected
ball only deselects it and the object stops responding.

**4 — drain on the render thread, before stepping.**

```python
def _enqueue_locked(self, catchup: int, frame_dt: float) -> None:
    self._drain_impulses_locked()     # after the harvest: device idle
    self._drain_shake_locked()
    for _ in range(catchup):
        self._run_control_hook_locked(frame_dt)
        self._step_frame_locked(frame_dt)
```

The following is an **application-owned, non-captured rigid-body adapter**.
Allocate `_pending_body_impulses` as a zeroed `(model.body_count, 6)` NumPy array
at attachment. Its lanes are world-space linear impulse followed by angular
impulse about the body's center of mass. Both click and shake drains accumulate
into it; neither drain edits solver velocities.

```python
# _drain_impulses_locked: index is a validated dynamic body in this generation.
for index, linear in pending:
    self._pending_body_impulses[index, 0:3] += linear

# _step_frame_locked: h is the actual positive solver substep duration.
h = frame_dt / self._substeps
for _ in range(self._substeps):
    self._state_0.clear_forces()
    self._add_continuous_forces_locked()  # application hook; may be a no-op
    forces = self._state_0.body_f.numpy().copy()
    forces += self._pending_body_impulses / h
    self._state_0.body_f.assign(forces)   # add, do not overwrite other forces
    self._solver.step(self._state_0, self._state_1, self._control,
                      self._contacts, h)
    self._state_0, self._state_1 = self._state_1, self._state_0
    self._pending_body_impulses.fill(0)   # consumed only after an actual step
```

Clear body forces on **every** substep, including the next frame. Applying
`J / h` for one substep integrates to `J`; replaying it for every substep or
catch-up frame multiplies the gesture. Retain the pending impulse if no step is
scheduled, and clear it on reset/detach or a stage-generation change. Count an impulse as
applied only after its consuming step succeeds; queue/drain counters describe
pending work separately. Reject
static/kinematic targets and unsupported solver/body types with a reason.

**5 — publish, then let the client poll.** The relay write is the render publish;
the reply is `newtonSimState`, and the client polls it at an application-defined interval.

### Solver state and impulse support

Check the installed Newton release's [state conventions](https://newton-physics.github.io/newton/stable/concepts/conventions.html)
and solver documentation. `State.body_f` uses world-space **force in lanes
0:3 and torque about the center of mass in lanes 3:6**. The one-substep external
wrench path above targets rigid bodies using `SolverFeatherstone` or
`SolverXPBD`; validate the selected solver/model before advertising it in the UI.
Other solvers and particle/deformable targets require their own verified adapter
or an explicit unsupported result.

Featherstone integrates `joint_q` / `joint_qd` and reconstructs body state, so
writing `body_qd` before stepping can be discarded. XPBD integrates body state,
but direct body-velocity edits are not a portable impulse recipe. The external
wrench path lets each solver enforce its constraints for free bodies and
articulations. Never index `joint_qd` by body index: generalized coordinates
have joint-specific dimensions and mappings. A constrained body's response can
differ from the unconstrained `Δv = J / m`, and a force along a locked direction
may yield no motion.

The shown click applies its impulse at the center of mass. For a world-space
hit point `p`, also accumulate angular impulse `(p - world_com) × J` after
resolving the live center of mass from the body pose and model COM. Do not use
the body origin as the COM unless they coincide.

For CUDA graph replay, implement this same schedule with persistent device
impulse storage: inject after force clearing, consume once in the captured step,
and zero that storage before the next substep/replay. Drain into existing storage
outside capture; do not capture the host NumPy readback above or a one-time
Python condition that replays the impulse forever. Use the non-captured step
path for this interaction until the captured path passes the same one-shot tests.


The following rules cover the remaining input and UI contracts.

## Rule 1 — a pick returns a gprim; the solver knows a body

`ovrtx` picking hands back the **renderable** prim — `/World/Cradle/Arm_2/Ball`.
The solver's `body_label` is the rigid-body `Xform` that carries
`PhysicsRigidBodyAPI` — `/World/Cradle/Arm_2`. An exact-match lookup therefore
misses **every click a user actually makes**, returns "not a simulated body", and
nothing logs why, because a miss is indistinguishable from clicking the floor.

Walk ancestors from the hit path to the stage root and stop at the first label
that matches. Two reasons to walk rather than parse:

- It survives a stage-authoring refactor that adds or removes a grouping level.
- It is the same operation `owns()` needs, so ownership routing and resolution
  cannot disagree.

Do not invert this by making the **body** prim pickable instead: an `Xform` has no
geometry to hit-test or outline.

**Generalisation.** Any engine binds to a coarser prim than the renderer draws.
The hit→entity map is a required component of the input path, not an
optimisation — build it before you build anything else, and log the resolve
failure, because "nothing happens and nothing logs" is the single most expensive
state this loop can be in.

## Rule 2 — input arrives on a websocket thread; the solver lives on the render thread

`queue_impulse` runs on whatever thread the websocket delivered the event on.
Solver state and pending device forces must not be written there. Three
independent reasons, all of them load-bearing:

1. **Device context.** The Warp device context belongs to the render thread. A
   host write from elsewhere is at best a stall and at worst wrong.
2. **Mid-read buffers.** Solver buffers may still be read by kernels the tick
   has already enqueued. Writing them from another thread lands in the middle
   of a frame.
3. **One writer.** This is the same rule OVStage imposes on its own columns: one
   owner per lane, sealed by ordinal. Input is not an exception to it.

So `queue_*` appends and returns; `tick()` drains. Placement inside `tick()` is
exact — **after** the harvest (device idle) and **before** the enqueue (so the
impulse lands on the frames this tick commits), not at queue time.

Budget for the latency this implies. Under the pipelined tick, one tick harvests
and publishes the frame the *previous* tick enqueued, so a click drained at tick
*N* first reaches `omni:xform` at tick *N+1* — roughly two frame periods plus the
pick's own round trip (up to the configured timeout) plus stream latency. That
is fine for a flick and unacceptable for a drag-to-move gizmo; if you need
tighter, the gesture belongs in a control hook, not the impulse queue.

**Bound the queue.** An unbounded queue turns a user hammering the mouse into
unbounded host memory and an unbounded velocity delta on the next drain.

| Knob | Value | Behaviour |
|---|---|---|
| `MAX_PENDING_IMPULSES` | 32 | At the cap, the **oldest** entry is dropped and the new one appended — a ring, so the most recent gesture always survives |
| Paused sim | — | **Rejected**, not queued: `tick()` returns before the drain while held, so a queued impulse would fire as a surprise on resume |
| Reset | — | Clears pending impulses and any pending batched gesture — a rewound world must not inherit input aimed at the old one |

**Generalisation.** Every input path that crosses into the sim needs: a bounded
buffer, an explicit drop policy (newest-wins or oldest-wins — pick one and say
which), a rejection path when the sim cannot accept input, and a clear on reset
and on detach. Clear accumulated impulse storage as well as the input queue.

## Rule 3 — specify a speed, derive the impulse, pin it to gravity

Two normalisations. Both are needed; each one alone is wrong.

**Mass-scale it.** Impulse is `J = m·Δv`. A constant `J` is mass-blind: one that
nudges a 1 kg cradle ball pleasantly *hurls* a 0.3 kg domino. So specify the
**speed** the gesture should impart and multiply by mass at dispatch:

```python
inv_mass  = self._body_inv_mass(index)
magnitude = speed / inv_mass if inv_mass and inv_mass > 1e-9 else speed
```

**Scale-normalise it.** ovnewton builds the model in **stage** coordinates and
never rescales lengths, so on a `metersPerUnit = 0.01` stage the velocity lanes
are cm/s and an SI-sized impulse is 100× too small — an invisible nudge with no
error anywhere. The inverse is worse: a constant hand-tuned in centimetres
launches bodies off-screen on a metre stage.

Gravity is the one quantity that already carries the scene's unit convention —
981 in centimetres, 9.81 in metres. Pin the gesture to it:

| Constant | Value | Meaning |
|---|---|---|
| `PICK_SPEED_GRAVITY_RATIO` | `0.1325` | A click's Δv = 0.1325·\|g\|. Reproduces a hand-tuned **130 cm/s** on a centimetre cradle and gives **~1.3 m/s** on a metre stage — a firm nudge, not a launch |
| `MAX_PICK_SPEED_GRAVITY_RATIO` | `0.6` | Safety clamp, applied to the resulting **speed**, not the impulse |
| `DEFAULT_IMPULSE_MAGNITUDE` | `180.0` | Explicit-vector fallback: a ~30° swing for an 18 cm 1 kg ball on a 120 cm rod |
| `OV_VIEWER_OVNEWTON_PICK_IMPULSE` | env var, stage units | Absolute speed override; **0 makes clicks select-only** |
| gravity-magnitude fallback | `981.0` | Centimetre-stage default when the model reports no gravity |

Clamping the **speed** rather than the impulse matters twice over: an absolute
impulse cap of, say, 500 kg·cm/s is a firm flick for a 1 kg ball and clamps a
legitimate nudge on a 502 kg cylinder down to a third of its intended effect.

**Direction, too, is a choice you must make explicitly.** Pushing along the body's
own local **+Z in world space**, read from its *live* orientation, topples a
domino along the chain instead of off the ring, and swings a cradle arm in its
swing plane instead of along the revolute axis, where the constraint would absorb
the push entirely and present as "nothing happened". Reading live orientation
rather than the authored layout keeps that correct after bodies have moved. The
cost is one device→host read on the click path, ordered behind whatever `tick()`
enqueued, so it can block for a frame — acceptable for a click, **wrong for
anything per-frame**.

**Two lane traps that both present as "nothing happened":**

- Confirm velocity layout, mutation APIs, and solver behavior against the installed engine documentation before applying an impulse.
- Treat a no-visible-effect result as a protocol or ownership failure until the command is observed at the simulation boundary.

## Rule 4 — a batched gesture takes a batched path, not N queued ones

A whole-scene gesture must use one bounded batch command, not one queued command per body. Per-body queues can silently drop work as scenes grow.

The batched path holds **one pending flag**, not N entries, and writes every
eligible body's pending impulse in a single pass:

```python
# queue: one clamped scalar
self._pending_shake = max(0.05, min(3.0, float(strength)))

# drain: stage one batch for the same one-substep force path as clicks
strength = self._pending_shake
if strength is not None:
    up = -g / mag
    speed = PICK_SPEED_GRAVITY_RATIO * mag * strength
    indices = self._dynamic_body_indices  # validated, supported, non-kinematic
    mass = self._body_mass[indices]       # host snapshot from model at attach
    linear = self._shake_directions(self._model.body_count, up)[indices] * speed
    self._pending_body_impulses[indices, 0:3] += mass[:, None] * linear
    self._pending_shake = None
```

Directions come from a golden-angle spiral about the up axis, **deterministic in
the body index**, so two runs of the same scene jolt identically — which is what
makes a shake usable as a before/after comparison rather than decoration.

Note the shake is also gravity-relative and up-axis-derived (`up = -g/|g|`), so it
needs no hard-coded scene layout. Use the same explicit zero-gravity magnitude
and up-axis fallback as the click path. That is the point: a whole-scene gesture
needs no selection and reads on any stage, where a "flick body N" button cannot,
because N is the runtime's interned-path order and means nothing to a viewer.

**Generalisation.** If a gesture's cost scales with body count, give it its own
drain. Per-entity queues are for per-entity intent. The tell that you have this
wrong is a gesture that works on the test scene and quietly half-works on the
real one.

## Rule 5 — see "perturbation vs command" above

It is the first section on purpose. One corollary belongs down here at the input
layer: a control hook is also how **interactive** actuation works, not just
scripted. A websocket handler writes a *target* into a plain Python object; the
hook — render thread, once per simulated frame — is the only thing that ever
touches `control`. Same queue-and-drain shape as Rule 2, different destination
and lifetime.

## Rule 6 — the reply path is what makes the UI honest

Nothing in the render loop pushes state. `newtonSimState` is **only ever a
reply**. A UI that updates only when the user clicks something cannot distinguish
a running sim from a stalled one, a diverged one, or one that never attached —
and all three look identical in a video stream of a scene that happens to be at
rest.

So the client polls, on the same pattern as stats:

```ts
const NEWTON_POLL_INTERVAL_MS = 500;
const poll = () => {
  if (status !== 'connected') return;
  sendMessage({ event_type: 'newtonSimRequest', payload: { action: 'status' } });
};
const interval = window.setInterval(poll, NEWTON_POLL_INTERVAL_MS);
poll();                                     // and immediately, not only after the configured interval
```

**A status probe must not change what it observes.** Dispatch `status` separately
from `start` at *both* layers — route it before the action table in the message
handler, and re-check it in the start entry point — precisely so a poll cannot
silently release a held sim. Getting this wrong produces "why is it already
running?", which is a very expensive question.

### State machine and snapshot contract

Every full state reply carries one explicit phase. Do not infer it from a zero
body count, a stopped frame counter, or a UI button label. The runtime owns
these transitions; the client renders them and never invents a transition
locally.

```text
unavailable → ready → attaching → running ↔ paused
                         │            │         │
                         └────────────┴─────────┤
                                      diverged   blocked
                                                   │
                         detaching ←───────────────┘
```

| Phase | Meaning | UI behaviour |
|---|---|---|
| `unavailable` | This scene or runtime cannot offer simulation. | Keep the feature discoverable but disabled; show the sanitized reason. |
| `ready` | Capability is known, but no live bridge is attached. | Enable attach or scene-selection actions only. |
| `attaching` | A new stage generation is being bound. | Show pending; reject control and impulse commands for both old and new generations. |
| `running` | The simulation can accept and drain commands. | Enable supported controls. |
| `paused` | The attached simulation is deliberately held. | Show held state; reject transient input instead of deferring it. |
| `diverged` | Runtime validity checks stopped publication or stepping. | Show a fault, preserve diagnostic reason, and require reset or detach. |
| `blocked` | A recoverable prerequisite prevents progress, such as a missing scene capability. | Show the reason and the action that can clear it. |
| `detaching` | The old bridge is releasing resources or changing stage generation. | Show pending; do not report the old state as live. |

Use application-defined event names, but make every state snapshot include this
minimum portable shape:

```json
{
  "phase": "running",
  "reason": null,
  "stage_generation": 12,
  "last_accepted_ordinal": 481,
  "command_result": {"request_id": "…", "accepted": true, "reason": null}
}
```

`stage_generation` fences scene lifetime. A client sends the generation it saw
with a mutating request; the runtime rejects a mismatch and returns a fresh full
snapshot. `last_accepted_ordinal` identifies the latest transform publication
that the current generation accepted. `command_result` reports the outcome of
the named request without pretending that an accepted command has already
become visible in the next rendered frame. Keep `reason` sanitized and
actionable; it is a user-facing explanation, not a traceback or environment
dump.

**Make the readout falsifiable.** Choose fields so a human can tell *which* stage
of the loop is broken:

| Field | Answers |
|---|---|
| `body_count` / `joint_count` | did the engine parse the stage at all |
| `frames_simulated` | is the solver stepping |
| `last_accepted_ordinal` | are poses being **published** for this generation — climbing in step with `frames` is the proof |
| `impulses_applied` | did an actual solver substep consume my click impulse (increment once per consumed command, never on drain alone) |
| `relay_active` | is the render publish path alive at all |
| `control_hook` / `control_hook_errors` | is the choreography installed, and is it silently throwing |
| `phase` + `reason` | why it is not running — `diverged` is not `paused` |
| `device`, `graph_mode`, `step_cost_ms` | is it slow, and where |

Show `reason` whenever the sim is **not running**, not only when unattached —
rendering a halted-on-divergence sim as "paused" describes a fault as a user
choice, which is exactly the confusion the halt exists to prevent.

**Not every reply is a state reply, and that is the point.** A
`newtonImpulseRequest` should answer with `newtonImpulseResult`, not
`newtonSimState` — deliberately, because the OVPhysX-shaped reply next to it is a
bounded-burst job report (steps, samples, write ordinals) and none of that applies
to a continuous sim that returns immediately. A client may then not subscribe to
`newtonImpulseResult` at all, learning that its click landed only from
`impulses_applied` on the next status poll. That is a working design and a warning
at the same time — an interaction whose only feedback is a reply nobody reads is
invisible; the poll is doing all the work.

**Reply shape.** Every request echoes its `request_id` and returns the **full**
state, never a delta. Make the demo-switch reply carry the post-load sim state so
the physics panel is correct across a scene change without waiting for the next
poll. A client that never has to merge a partial update into cached state cannot
get out of sync.

## Rule 7 — the server declares the controls; the client renders them

Rule 6 makes the *readout* honest. This one makes it cheap to add the next one.

A runtime knob is not a checkbox someone hand-wires on both sides. It is a
**descriptor the server publishes** and the client renders generically:

```json
{"controls": [
  {"name": "gravity_scale", "label": "Gravity", "kind": "float",
   "min": 0.0, "max": 2.0, "step": 0.05, "unit": "x g", "value": 1.0},
  {"name": "substeps", "label": "Substeps", "kind": "int",
   "min": 2, "max": 32, "step": 2, "unit": "", "value": 10,
   "note": "rebuild-only"}],
 "telemetry": [
  {"name": "step_cost_ms", "label": "Step cost", "unit": "ms", "value": 45.1},
  {"name": "realtime_factor", "label": "Realtime", "unit": "x", "value": 0.265}]}
```

The client loops over `controls`, maps `kind` to a widget, and sends
`{name, value}` back. It knows no physics vocabulary at all. **Adding the next
slider or plot then costs one server-side entry and zero client changes** —
which is the whole reason creative UI is cheap on this architecture and
expensive on the hand-wired one. It also means the panel is automatically
correct across a scene switch, because a new scene declares a different list.

Four consequences that are not optional:

- **Clamp on the server, and never trust the client.** `min`/`max` in the
  descriptor are for rendering a sensible widget, not for enforcement — a
  client is a message sender, and one out-of-range value can put a solver
  somewhere it cannot recover from. Clamp where the value is applied.
- **Echo the value you adopted, not the one you were sent.** The reply and the
  next status carry the post-clamp, post-quantisation value (an odd substep
  count snapped to even, a rebuild-only knob still showing its old value until
  re-capture). A UI that renders its own optimistic guess reports a state the
  simulation is not in, and the user tunes against a lie.
- **Hold local state for the duration of a drag.** A periodic status poll that
  overwrites the slider mid-gesture fights the pointer, visibly. Take the
  server's value as truth only when the control is idle: adopt on pointer
  release, or on the first poll whose echoed value matches what you last sent.
- **Throttle trailing-edge.** Leading-edge throttling drops the *end* of a
  gesture, which is the only value the user actually meant. Send at most one
  update per interval **and** always send the final value on release.

The optimistic-pending pattern below is not a counter-example: it is optimistic
about an action being *in flight*, never about the value it will settle on.

## The client half

| Piece | Where it belongs | Note |
|---|---|---|
| `newtonSim(action)` | the backend hook (`the client transport adapter`) | `start` / `stop` / `reset` / `shake` / `status` over one `newtonSimRequest` |
| `newtonImpulse(path?, impulse?)` | same | Both optional — server falls back to selection, then the far end body |
| `newtonShake(strength=1.0)` | same | Batched gesture; releases a held sim |
| `onNewtonState(cb)` | same | Subscribe + install the poll; replays the last state immediately so a late subscriber is not blank |
| `onNewtonDemoState(cb)` | same | Catalog is static, so **one** request on subscribe — no poll |
| Types | `the client protocol types` | `NewtonSimRequestPayload`, `NewtonImpulseRequestPayload`, `NewtonSimStatePayload`, demo payloads |
| Panel | `an optional simulation panel` | Return `null` when the sim is not enabled — a dead panel is worse than no panel |
| Demo switcher | `the scene-selection UI` | Optimistic `pending` on click, because a stage switch is seconds; unavailable demos shown **disabled**, not hidden |

Two client-side patterns worth stealing regardless of framework:

- **Optimistic pending state for slow actions.** A stage switch takes seconds; a
  UI that looks dead invites a second click, which queues a second stage load
  behind the first.
- **Physics types stay on the concrete backend hook**, not in the shared
  `ViewerBackend` interface — the payloads are one server's vocabulary, and
  shared UI is shared with clients that have no simulation at all.

## When it does not work

| Symptom | Cause | Check |
|---|---|---|
| Click does nothing, **nothing logs** | Hit path is a gprim; exact-match lookup missed | Rule 1 — `resolve_body`, ancestor walk |
| Click does nothing, "not a rigid body" | Body geometry not in the pickable set | The owner’s scene/workload pickability policy |
| Click selects but never flicks; second click only deselects | Impulse fired after the selection toggle | Order in the click handler |
| Every orbit flicks something | Ignoring `was_click` | The camera's mouse-button-up return value |
| Flick is invisible | Impulse sized in SI on a centimetre stage (100× small), or written into lanes 3:6 | Rule 3 |
| Flick launches the body off-screen | Constant tuned in cm applied to a metre stage, or mass-blind | Rule 3 |
| "Nothing happened" on a jointed body | Wrote `body_qd` with Featherstone, used a body index as a joint DOF, or pushed in a constrained direction | Solver state and impulse support; verify `body_f` at the first substep |
| Some bodies never shake | Batch passed through the per-body queue, or bypassed the solver-aware impulse path | Rule 4; report unsupported targets |
| Gesture grows with substeps or repeats on later frames | `J / h` applied more than once, or captured impulse storage never cleared | Consume after one actual step; clear forces every substep |
| Impulses land while paused, on resume | Queued instead of rejected when not running | Rule 2 |
| Robot holds its home pose forever | No `control_hook`; constant targets only | Rule 5, `control_hook` in status |
| Robot moves at a third speed under load | Hook called once per **tick** instead of once per simulated frame | The enqueue loop |
| Hook writes have no effect under graph replay | Rebound an array instead of writing in place | Rule 5 |
| Readout frozen; running sim looks stalled | No poll; server never pushes | Rule 6 |
| Status probe started a held sim | `status` folded into the action table | Rule 6 |
| Slider snaps back mid-drag | Status poll overwriting local state during the gesture | Rule 7 |
| Last value of a gesture never lands | Leading-edge throttle dropped the release | Rule 7 |
| UI shows a value the sim is not using | Client rendered its own guess instead of the echoed value | Rule 7 |
| Every new knob costs a client change | Controls hand-wired instead of server-declared | Rule 7 |
| Scene stops moving, every metric healthy | Solver diverged; the publish path would write NaN | `realtime-sim-integration/README.md` |

## Simulation validation ladder

Run this focused ladder only when the app exposes continuous simulation or
simulation controls. Each rung should leave a compact, sanitized result in the
generated app validation report. A failed rung is a diagnosis boundary; do not
claim a later rung as passed on its behalf.

1. **Headless model and solver.** Build the model and advance it without a
   renderer. Confirm the expected state changes are finite and a reset restores
   the declared initial state.
2. **Attach and initial publish.** Attach to one stage generation and confirm
   the initial held-state transform publication is accepted before simulation
   input is enabled.
3. **First renderer-visible transform.** Advance the simulation, publish a
   changed transform, and capture proof from the same output path the delivery
   client displays.
4. **Reset after running.** Run, reset, and verify that queued inputs,
   controller state, telemetry baselines, and transform state belong to the
   reset world rather than the prior run.
5. **Stage switch while work is outstanding.** Start a scene change while a
   command or publish is pending. Verify the old generation detaches, its work
   is discarded, and the new generation becomes ready without stale motion.
6. **Stale command rejection.** Send a mutating request tagged with a prior
   generation. Verify it is rejected with a full current snapshot and does not
   affect the active world.
7. **Client confirmation.** Verify the delivery client shows the phase,
   sanitized reason when present, accepted command outcome, and current
   generation. Exercise unavailable, paused, and fault or blocked states when
   the application can create them safely.

## Checklist

- [ ] Simulated geometry remains pickable before and after expanding unrelated hierarchy branches
- [ ] `resolve_body` walks ancestors; `owns()` is derived from it, and routes between engines
- [ ] Input handlers **queue only**; the render loop drains between harvest and enqueue
- [ ] Queue is bounded, with a stated drop policy; rejects while paused; cleared on reset and detach
- [ ] Gestures specify a **speed**, mass-scaled at dispatch, expressed as a fraction of `|gravity|`
- [ ] The safety clamp is on speed, not impulse
- [ ] Velocity lanes verified against the engine version (for the installed engine version)
- [ ] Whole-scene gestures use a batched drain, deterministic in body index
- [ ] Persistent actuation goes through a per-frame hook, called once per **simulated** frame, outside the graph, writing in place
- [ ] Hook failures are caught, rate-limited, counted in status, and never stop the sim
- [ ] Every request replies with full state and echoes `request_id`
- [ ] `status` is side-effect free at every layer that dispatches it
- [ ] Full status snapshots carry `phase`, `stage_generation`, `last_accepted_ordinal`, and the outcome for the named command
- [ ] Client polls; readout distinguishes unavailable / ready / attaching / running / paused / diverged / blocked / detaching, and shows `reason`
- [ ] Runtime controls and telemetry channels are **declared by the server** — name, label, bounds, step, unit, value — and rendered generically
- [ ] Values clamped server-side at the point of application, and the **adopted** value echoed back
- [ ] Controls keep local state during a drag and reconcile on release or on a matching echo; throttling is trailing-edge

For native or controlled-WebXR hand commands and solver ownership in XR, use
[XR physics interaction](../xr-spatial-input/physics-interaction.md) and
[XR simulation lifecycle](../xr-spatial-input/simulation-lifecycle.md).

## Validate solver input before the viewer round trip

For every advertised solver, test a zero-gravity free body and an articulated
body with a force in a permitted direction. Compare no-input, one click impulse,
and batched shake cases. For a free body check the expected `Δv = J / m` and
continued displacement after the pulse; for an articulation check a finite,
constraint-consistent response against the no-input control. Include unequal
masses, static/kinematic rejection, reset, and an unsupported solver/body type.
Repeat with multiple substeps and catch-up frames to prove each impulse is
consumed once. If graph replay is enabled, run these checks with and without
capture. Then verify the browser pick, batch action, published transforms, and
visible stream response separately; CPU solver tests alone do not prove that
complete round trip.
