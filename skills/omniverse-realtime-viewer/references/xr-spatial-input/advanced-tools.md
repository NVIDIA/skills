# Advanced scene queries and simulated interaction

Read [navigation](navigation.md) for managed-anchor movement, scale, teleport,
and reference spaces; [controllers](controllers.md) for identity, measured
poses, transport and capture; [hands](hands.md) for hand lifetimes; and
[stage composition](stage-composition.md) for source-safe writes. Those contracts
also apply here. The mechanisms below are application patterns to build with
an available scene-query/physics library, not additional OVXR APIs.

## Asynchronous simulation ownership

An asynchronous simulation owns its clock and solver state, then offers
immutable completed snapshots through a bounded newest-wins handoff. The
OVXR owner publishes a ready snapshot through documented scene writes or renders
the last committed state. Head tracking and frame submission must not wait for
a newer solver result. A simulation worker never calls OVXR's frame helpers or
mutates the render owner's stage concurrently.

For a native input integration supported by the installed SDK, sample actions,
grip, aim and VIEW at a coherent runtime display time. If the high-level helper
exposes a submitted display time only after submission, treat that sample as
latched input for a subsequent update and retain both frame/time identities.
Do not label a QPC sample or a guessed predicted-period offset as the exact
submitted time. Delegate native action startup/teardown to upstream
openxr-input-and-extensions through [the SDK route](../dependencies/ovxr.md);
[controllers](controllers.md) defines the application snapshot and tool policy.

Apply input smoothing in raw tracking space before unit and anchor conversion.
When smoothing tool outputs, keep visual finger-joint latency distinct from ray
stability. All direct colliders, remote-hold anchors, rays and teleport previews
consume the same converted batch. A managed-anchor teleport moves the user;
it must not also translate simulation bodies or add the delta again to hands.

## Generation-scoped scene queries

Extract render-leaf triangles and local/world baselines once per stage
generation in an isolated USD worker. Upload a persistent mesh/BVH and
preallocate bounded ray slots. Launch queries only while a mapped control or
release edge needs them. Reuse buffers and pre-authored beam/arc/marker leaves;
do not construct geometry or traverse USD on each tracking frame.

A hit becomes writable only after its exact render leaf, stage generation,
authoritative current world matrix and local bounds are captured at an
interaction boundary. Do not authorize an arbitrary ancestor. For simulated
objects, resolve the nearest solver-owned body ancestor from the active
simulation bindings and retain that mapping per generation, instead of
requiring a hardcoded list of object names.

Refit moved geometry/BVH at an explicit boundary such as release, cancel or
disconnect. If other tools query while geometry moves, either maintain the
required incremental acceleration-structure updates or suppress those queries
until refit; a stale BVH must not authorize an unrelated hit. Unsupported
instances/gprims require an explicit query capability result, not guessed bounds.

## Remote and direct hand modes

Choose one authoritative interaction mode: remote ray/grasp or direct
colliders. Switching mode releases and disables its predecessor before accepting
new input. A selected object is not captured merely because it is highlighted.

For a dynamic simulated grasp, capture object/controller baselines together to
avoid an acquisition jump. A predeclared solver-owned constraint or fixed-joint
hold can keep the payload dynamic and colliding while following the hand.
Release that constraint and retain the solver's resulting linear/angular
velocity. Parenting, teleporting or making the object kinematic is a different
interaction model and must not be described as this dynamic grasp.

Direct mode enables tracked kinematic hand colliders and disables remote
capture. Parking direct proxies or cleaning up their contacts must not cancel
a remote hold while remote mode owns it. Test both direction changes between
modes, including tracking loss during capture. When contact-pair queries are
unavailable, label any sphere-overlap estimate analytic; write/step/read alone
does not prove physical contact.

## Quiescence and proof

On full disconnect or Stop XR, mark input quiesced, release remote/direct
ownership, clear queued input and pending simulation snapshots, drain submitted
device work, then detach. Make that transition idempotent. Loss of one hand
retires only its owned interaction; it does not destroy the shared renderer or
release an unrelated other-hand capture.

Keep pose-boundary diagnostics transition-driven and audits explicitly bounded.
After the renderer consumes an audited transaction, read its live world matrices
once, compare them with the expected batch and stop the audit automatically.
Use [feature validation](feature-validation.md) for shared evidence requirements.
Add tests for nonidentity anchors, centimetre stages, alternate up-axis,
teleport while a body is held, both hand-mode transitions, preserved release
velocity, stale-query rejection and shutdown with work in flight. Prove each
conversion occurs once and direct cleanup cannot release a remote hold.
