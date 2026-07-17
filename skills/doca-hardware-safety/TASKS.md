# DOCA hardware safety workflows

**Where to start:** The verbs run `configure → modify → run → test →
debug`. `## build` is a routing stub — hardware-touching changes do
not produce build artifacts. The change-application discipline is a
loop: pre-flight inventory and out-of-band reachability are captured
in `## configure`, the apply step lives in `## modify`, post-change
verification gates live in `## run`, replica-first smoke is the
loop's pre-prod leg in `## test`, and the rollback ladder + escalation
escape valve live in `## debug`.

Read this file when the loader sent you here from [SKILL.md](SKILL.md).
For the meta-policy surface (the change classes in scope, the failure
modes the discipline prevents, the observability gate), see
[CAPABILITIES.md](CAPABILITIES.md). For the JSON schemas the agent
prefers when present, see
[`doca-structured-tools-contract`](../doca-structured-tools-contract/SKILL.md).

Each verb below describes the **shape of the workflow**, not a copy-
paste recipe. The per-artifact skill names the specific change; this
skill names the discipline that wraps it. The agent's job is to walk
the operator through the discipline before the per-artifact action
is issued.

## configure

Goal: prepare the change-application session — capture the pre-flight
inventory, establish out-of-band reachability, and frame the
maintenance window. Nothing on the hardware changes during this verb.

Steps the agent should walk the operator through:

1. **Identify the change class.** Map the recommended next step (from
   the per-artifact skill) to one of the change classes in
   [CAPABILITIES.md ## Capabilities and modes](CAPABILITIES.md#capabilities-and-modes).
   `mlxconfig`-class write, firmware burn, BFB reflash, mode flip,
   kernel boot parameter change, PCIe rescan / link state change, and
   BlueField cold reboot each carry a different commit action and a
   different rollback shape. The agent does not begin the discipline
   without naming the class — the class determines what counts as
   "applied" later.
2. **Capture the pre-flight inventory.** Walk the inventory taxonomy
   in [CAPABILITIES.md ## Capabilities and modes](CAPABILITIES.md#capabilities-and-modes):
   PCIe topology, link state, running firmware level, BFB level (on
   BlueField), `mlxconfig`-class configuration snapshot, and host-side
   env snapshot. Prefer the structured one-shot via the
   `collect-host-state` and `collect-dpu-state` schemas in
   [`doca-structured-tools-contract ## Schemas`](../doca-structured-tools-contract/SKILL.md#schemas)
   when the host has them; otherwise walk the manual command chain
   the same schema section names. Record every captured value in the
   session — the rollback path will quote them.
3. **Record the four-way version anchor for the session.** Per
   [`doca-version ## configure`](../doca-version/TASKS.md#configure),
   capture `pkg-config --modversion doca-common`,
   `cat /opt/mellanox/doca/applications/VERSION`, `doca_caps --version`,
   and (on BlueField hosts) the BFB version. The pre-change state of
   the four-way match is part of the rollback baseline; a change that
   moves any of these anchors must be re-verified in
   [`## run`](#run) before any workload moves.
4. **Confirm the env-class preconditions hold.** Per
   [`doca-setup ## configure`](../doca-setup/TASKS.md#configure),
   confirm `pkg-config` resolves, hugepages are mounted, the
   `mlx5_*` modules are loaded, and the representors the change
   touches are visible. An env-class problem before the change is an
   env-class problem after the change; do not begin the hardware-
   touching discipline until env-class is clean.
5. **Establish out-of-band reachability (when the change is link-
   breaking).** For the link-breaking change classes named in
   [CAPABILITIES.md ## Capabilities and modes](CAPABILITIES.md#capabilities-and-modes)
   (mode flip, BAR change, port reassignment, BFB reflash, NIC mode
   change), the operator names which OOB class is in place per the
   OOB-classes table in the same section AND has confirmed the OOB
   path is reachable BEFORE the change is issued. The agent refuses
   to issue a link-breaking change without OOB; the operator
   provides the specific endpoint (BMC IP, console TTY, etc.), the
   agent never invents one.
6. **Frame the maintenance window.** Name the start time, the
   expected duration, the rollback decision point, and the
   stakeholders notified. *"I'll just do this right now"* is the
   anti-pattern; the agent refuses to issue a hardware-touching
   change outside an explicit, time-boxed window.
7. **Name the rollback plan and the verification gate up front.**
   Document the rollback path BEFORE the change is applied per
   [CAPABILITIES.md ## Safety policy](CAPABILITIES.md#safety-policy);
   the gate that will declare the change "applied and safe to move
   workload back to" is the post-change inventory + four-way match
   + per-artifact smoke from [`## run`](#run) and [`## test`](#test).

If any step fails (env-class not clean, OOB not in place, rollback
cannot be named), route through the matching skill (`doca-setup` for
env-class, the per-artifact skill for the rollback shape) before the
discipline continues. Do not begin [`## modify`](#modify) until every
step here is closed.

## build

> **Anchor exists for lint compliance.** Hardware-touching changes do
> not produce build artifacts; there is no compile-time output of a
> firmware burn, an `mlxconfig` write, a BFB reflash, or a kernel-boot-
> parameter change. The build-side discipline for DOCA-consuming
> applications lives at
> [`doca-programming-guide ## build`](../doca-programming-guide/TASKS.md#build);
> the env-side discipline for the build host lives at
> [`doca-setup ## build`](../doca-setup/TASKS.md#build). This anchor is
> a routing stub.

The one hardware-safety overlay on the build side is the
*replica build host* match: when the change is going to be tested on
a non-prod replica per [`## test`](#test), the replica's build host
must be in the same DOCA-version band as production, so the binary
the operator smoke-tests on the replica is the same binary they will
re-run after the production change. The mechanics of confirming the
build host's DOCA version live in
[`doca-version ## build`](../doca-version/TASKS.md#build); this skill
only surfaces that the replica and production must agree on it.

## modify

Goal: apply the hardware-touching change inside the framing
established in [`## configure`](#configure). The per-artifact skill
names the specific action; this skill names the discipline.

Steps the agent should walk the operator through (in order):

1. **Re-confirm the inventory and OOB are still in scope.** The
   pre-flight inventory was captured in [`## configure`](#configure);
   confirm it is still current (the operator did not run other
   changes in between) and that the OOB path is still reachable. A
   stale inventory is the discovered-during-failure rollback failure
   mode; the agent re-runs the inventory probe before the apply step
   when there is any doubt.
2. **Quiesce the affected workload.** When the change touches a port,
   a function, or a service currently carrying traffic, the operator
   drains or schedules around the workload before the change is
   issued. The agent does not silently bring a port down with traffic
   on it.
3. **Apply the change using the per-artifact skill's specific
   command.** This skill does NOT name the specific command; the
   per-artifact skill does. The agent quotes the per-artifact skill's
   verbatim command and passes the operator's captured pre-flight
   values to it; the agent never invents a parameter literal, a PCI
   address, a firmware version, or a kernel-command literal.
4. **Apply the documented commit action for the change class.** Per
   [CAPABILITIES.md ## Capabilities and modes](CAPABILITIES.md#capabilities-and-modes):
   - `mlxconfig`-class writes: a **cold power cycle** (full A/C
     power removal) of the NIC / DPU. A warm reboot does not commit
     firmware-stored configuration. The agent does not declare the
     change applied after a warm reboot; the agent walks the
     operator through the cold power cycle and confirms the
     post-change inventory before proceeding.
   - NIC firmware burn: the burn tool's documented completion
     handshake plus a power cycle if the tool requires it. The
     agent quotes the tool's own documented completion criterion.
   - BFB reflash: the BFB's documented post-flash boot + BlueField
     cold reboot; every hosted service container is re-deployed per
     [`doca-container-deployment ## run`](../doca-container-deployment/TASKS.md#run).
   - BlueField mode flip: cold power cycle + post-reboot
     verification of the new personality per the matching per-
     artifact skill.
   - Kernel boot parameter change: a **host reboot**. Confirm the
     post-reboot env-class state per
     [`doca-setup ## debug`](../doca-setup/TASKS.md#debug); a
     runtime helper alone is not sufficient.
   - Link state change / PCIe rescan / function rebind: the
     operator's documented procedure; the affected function will
     disappear and reappear, and any workload using it sees the
     disruption.
5. **Capture the post-change inventory.** Re-walk the same inventory
   taxonomy from [`## configure`](#configure) step 2 and compare
   against the captured baseline. Any unexpected delta is the
   trigger for the rollback ladder in [`## debug`](#debug). The
   agent does not declare the change applied until the post-change
   inventory matches the plan.

The anti-pattern to refuse: declaring the change applied after a
warm reboot when the change class commits at cold power cycle, or
declaring the change applied without re-capturing the inventory.

## run

Goal: gate the workload behind a post-change verification before any
production traffic moves. The verification is a *gate*, not a
formality.

Three runtime checks the agent walks in order:

1. **Post-change four-way match.** Per
   [`doca-version ## test`](../doca-version/TASKS.md#test), confirm
   `pkg-config --modversion doca-common`,
   `cat /opt/mellanox/doca/applications/VERSION`, `doca_caps --version`,
   and the BFB version (on BlueField) all agree on the post-change
   state. A change that leaves the four-way match in a partial state
   is a failed change; route to [`## debug`](#debug) before any
   workload moves.
2. **Per-artifact health metric.** Run the smoke test from the
   matching per-artifact skill (e.g. the service-container smoke from
   the service's `## test`, the library lifecycle smoke from the
   library's `## test`). The per-artifact smoke is the artifact's own
   *"did the artifact come back to health"* signal; this skill does
   not name the signal — the per-artifact skill does.
3. **Observability path proven end-to-end.** Confirm the
   observability surface in
   [CAPABILITIES.md ## Observability](CAPABILITIES.md#observability)
   — logs visible, counters readable, container / service stdout
   reachable. For the program-side cross-library observability
   (`--sdk-log-level`, the `doca-<library>-trace` build flavor,
   `DOCA_LOG_LEVEL`), see
   [`doca-debug ## configure`](../doca-debug/TASKS.md#configure). The
   agent does not declare the deployment "ready to carry workload"
   until the observability path is proven end-to-end on the post-
   change state.

Only after all three checks pass does workload move back; the agent
does not silently lift the workload gate.

## test

Goal: rehearse the change on a non-prod replica before production is
exposed to it. The replica is the loop's pre-prod leg; the production
change runs only after the replica leg passes.

This is **a loop, not a one-shot pass.** Each iteration runs the
change on the replica, confirms the per-artifact smoke, then
rehearses the rollback on the same replica. The loop terminates when
both the change AND the rollback succeed against the replica's
captured baseline.

Iteration shape:

1. **Confirm the replica's hardware class matches production.** The
   replica must match production on BlueField generation, running
   firmware level, host kernel version, loaded `mlx5_*` modules, and
   the set of representors / VFs / SFs the change touches. Mismatches
   on any of those axes mean the replica result is advisory rather
   than gating per
   [CAPABILITIES.md ## Capabilities and modes](CAPABILITIES.md#capabilities-and-modes).
2. **Run the same [`## configure`](#configure) discipline on the
   replica.** Capture the replica's pre-flight inventory; the
   replica's baseline is what the replica's rollback path quotes.
3. **Apply the change on the replica per [`## modify`](#modify).**
   The change uses the same per-artifact command and the same
   commit action as the production plan.
4. **Run the per-artifact smoke on the replica.** Confirm the
   artifact's own health metric on the post-change state per the
   per-artifact skill's `## test`. A replica smoke that fails is a
   blocking signal for production; do not schedule the production
   change until the replica smoke passes.
5. **Rehearse the rollback on the replica.** Apply the rollback path
   to the replica; confirm the replica returns to the captured
   baseline. A rollback that does not return the replica to the
   captured state is a broken rollback; the agent does not approve
   the production change until the rollback is known to work on the
   replica.
6. **Loop back if any iteration changes the picture.** A replica
   change that surfaces a new failure mode, or a replica rollback
   that does not return cleanly, re-opens the change plan; the
   agent does not schedule production until the loop converges.

Loop termination: stop iterating once both the replica change and
the replica rollback succeed against the captured baseline, AND the
operator can describe the rollback path in a way the agent can
reproduce. Replica passes that the operator cannot describe are not
rehearsed rollbacks; the agent loops back until the rollback is
explicit.

The anti-pattern to refuse: skipping the replica because the change
"is small". A change without a tested rollback is not a small change
— it is a change with an untested recovery surface.

## debug

Goal: when the production change goes wrong, walk the rollback ladder
in order. The rollback path was documented in [`## configure`](#configure)
step 7 and rehearsed on the replica in [`## test`](#test); the debug
verb's job is to apply it and route the residual to the right next
skill.

**The rollback ladder.**

1. **Confirm the failure against the captured baseline.** Re-run the
   post-change inventory from [`## modify`](#modify) step 5 and
   compare against the pre-flight baseline captured in
   [`## configure`](#configure) step 2. Name which dimension moved
   away from the plan — PCIe topology, link state, firmware level,
   BFB level, `mlxconfig`-class configuration, or host-side env.
2. **Apply the matching rollback class.** Each change class has a
   matching rollback class:
   - `mlxconfig`-class write → revert the `mlxconfig` values to the
     captured baseline, then cold-power-cycle. The agent quotes the
     per-artifact skill's `mlxconfig` command shape and passes the
     captured baseline values; the agent does not invent the
     parameter literals.
   - Firmware burn → reflash to the previously-captured firmware
     level. The captured baseline names the level; the agent does
     not invent a firmware version.
   - BFB reflash → reflash to the previously-captured BFB level and
     re-deploy every hosted service container per
     [`doca-container-deployment ## run`](../doca-container-deployment/TASKS.md#run).
     The captured baseline names the BFB level; the agent does not
     invent it.
   - Kernel boot parameter change → revert the host's kernel
     command line to the captured baseline and reboot. Confirm the
     post-reboot env-class state per
     [`doca-setup ## debug`](../doca-setup/TASKS.md#debug).
   - Mode flip / emulation-slot enable → flip back to the captured
     personality, cold-power-cycle. Confirm the host-visible
     representor topology returns to the captured baseline.
   - PCIe rescan / link state change → re-issue the operator's
     documented restore procedure for the affected function.
3. **Re-verify against the captured baseline.** After the rollback,
   re-run the inventory probe and the four-way match per
   [`doca-version ## debug`](../doca-version/TASKS.md#debug). The
   ladder is not done until the inventory matches the baseline AND
   the four-way match is clean.
4. **If the residual symptom is at a software layer, hand off.**
   Once the rollback has restored the captured hardware state and a
   symptom still remains, the symptom now lives at a software layer.
   Hand off to [`doca-debug ## debug`](../doca-debug/TASKS.md#debug)
   with the captured state as evidence; that ladder owns install /
   version / build / link / runtime / program / driver-below
   symptoms once the hardware state is known-good.

**The class-shape escape valve (refuse and escalate).** When no
rollback is documented, OR when the captured baseline cannot be
restored (e.g. the vendor states the firmware roll is one-way; a
board-level change has no reversal; the change had no OOB plan and
the link is now down with no recovery path), the agent does NOT
guess at a recovery. The agent surfaces the missing rollback as the
blocking issue, refuses to invent a workaround, and routes to the
operator's change-control / vendor-escalation process. The DOCA
Developer Forum at
<https://forums.developer.nvidia.com/c/infrastructure/doca/370> is
the public escalation channel; for vendor-specific recovery (BMC,
RShim, BFB recovery image), the operator's NVIDIA support contact
or the deployment's vendor support is the right path. Per
[`doca-public-knowledge-map`](../doca-public-knowledge-map/SKILL.md),
the public forum is the right *public* surface; vendor-specific
recovery is operator-specific and never invented.

The anti-pattern to refuse: an invented rollback. *"Try resetting
this register"* or *"reflash to whatever the docs say is latest"* in
the absence of a captured baseline is how a recoverable incident
becomes an unrecoverable one. The agent refuses and escalates.

## Deferred task verbs

The following verbs are out of scope for this skill but are commonly
asked in the same conversations. Route them as follows so the agent
does not invent guidance:

- **install / first-time setup.** Owned by
  [`doca-setup`](../doca-setup/SKILL.md). This skill assumes a working
  install is in place and a hardware-touching change is being
  contemplated on top of it.
- **per-artifact specifics** (which exact firmware slot to flip,
  which exact `mlxconfig` parameter to write, which container tag
  to roll back to). Owned by the matching per-artifact skill's own
  `## Safety policy` and `## TASKS` overlays. This skill names the
  discipline; the per-artifact skill names the specific change.
- **program-side debug** (lifecycle order, `DOCA_ERROR_*`
  interpretation, capability discovery). Owned by
  [`doca-programming-guide ## debug`](../doca-programming-guide/TASKS.md#debug)
  and [`doca-debug ## debug`](../doca-debug/TASKS.md#debug). Once the
  hardware state is known-good (the rollback returned the inventory
  to the captured baseline), symptoms at the software layer hand off
  to those skills.
- **change-control process (ticketing, approval gates, ops
  notifications).** Operator-specific; this skill names the
  *discipline* (maintenance window framed, stakeholders notified)
  but does not own the operator's change-management tooling.
