# DOCA bare-metal deployment — Tasks

**Where to start:** The verb order is `configure → build → modify →
run → test → debug`. For bare-metal deployment, `build` and `modify`
are *routing stubs* — building the binary and modifying its source
both live in
[`doca-programming-guide`](../doca-programming-guide/SKILL.md); this
skill owns deployment of an already-built binary on the operator's
hardware. The `## test` verb is an iterative smoke-before-bulk loop,
not a one-shot pass.

These verbs cover the in-scope cross-cutting bare-metal-deployment
workflows for an external operator launching any DOCA-linked
application binary on either supported host mode — host x86 with a
remote BlueField NIC over PCIe, or BlueField Arm bare-metal with the
binary on the DPU directly. Every step assumes the operator has
consulted the live public DOCA Programming Guide, the public
BlueField / DPU User Manual, and the Linux man pages for
`numactl(8)` / `taskset(1)` / `systemd.service(5)` /
`systemd.unit(5)` (all reachable through
[doca-public-knowledge-map ## Public documentation entry points](../doca-public-knowledge-map/SKILL.md#public-documentation-entry-points))
and is using them as the authoritative reference; this file
prescribes the *order* and *what to look up where*, not a copy-paste
runbook.

## configure

Preparing the host (or BlueField Arm) target, confirming every
precondition the bare-metal launch will rely on, and picking the
launch mode BEFORE any binary is invoked. This is also the verb
where the smoke-before-bulk posture is established up front — every
later verb assumes the operator has read it here.

1. **Confirm the env is healthy first.** This skill expects DOCA
   installed and healthy on whichever side the binary is built for
   (host x86 OR BlueField Arm). If install health is unverified,
   run
   [`doca-setup ## test`](../doca-setup/TASKS.md#test) on the
   target first. If the operator has no install at all, route to
   [`doca-setup ## no-install`](../doca-setup/TASKS.md#no-install)
   for the public NGC DOCA container path; the bare-metal launch
   itself cannot run inside the NGC container, but the operator
   may use the container to build the binary they will deploy.
2. **Recognise the host mode.** Confirm with the operator which
   side the binary is going to run on — host x86 against a remote
   BlueField NIC over PCIe, OR BlueField Arm cores directly. The
   distinction governs every later step (which install layout,
   which `devlink`, which NUMA topology). If the operator is
   ambiguous, derive the answer from `file <binary>` (x86_64 vs
   aarch64) and the install location (`/opt/mellanox/doca` on
   the host vs on the BlueField Arm). Do NOT guess.
3. **Confirm the bare-metal preconditions are closed.** Walk the
   precondition surface in
   [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes)
   against the operator's target:
    - DOCA installed and healthy per
      [`doca-setup ## test`](../doca-setup/TASKS.md#test).
    - Hugepages mounted and reserved per
      [`doca-setup ## configure`](../doca-setup/TASKS.md#configure)
      step 7 (or its BlueField-Arm equivalent on the BlueField OS
      image).
    - Devices and representors visible per
      [`doca-setup ## configure`](../doca-setup/TASKS.md#configure)
      step 8.
    - `LD_LIBRARY_PATH` set correctly per
      [`doca-setup ## configure`](../doca-setup/TASKS.md#configure)
      step 6 — the runtime install version MUST match the binary's
      link-time `pkg-config doca-*` version per
      [`CAPABILITIES.md ## Version compatibility`](CAPABILITIES.md#version-compatibility).
    - The four-way version match is closed per
      [`doca-version TASKS.md ## configure`](../doca-version/TASKS.md#configure).
4. **Pick the launch mode.** Per the launch-mode table in
   [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes),
   the operator picks ONE of: direct foreground (best for first
   launch, interactive debug, smoke); tmux / screen (best for
   long-running with manual reattach, no supervisor); systemd-
   supervised (best for restart-after-reboot, journald-integrated
   logs, documented `Restart=` policy). The choice feeds the
   observability surface in
   [`## run`](#run) step 4 and the restart / recovery surface in
   [`## debug`](#debug) layer 6. The agent does NOT pre-bake a
   sample systemd unit, a sample tmux invocation, or a sample
   direct command line — the operator authors the launch invocation
   against their environment.
5. **Plan the hardware-resource binding.** Per the
   hardware-binding rules in
   [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes),
   capture (do NOT invent):
    - The PCI BDF the binary should attach to, derived from
      `lspci -d 15b3:` on the target host.
    - The NUMA node owning that BDF's PCIe root complex, derived
      from `numactl --hardware` / `lscpu`.
    - The CPU set the binary's polling threads should pin to,
      chosen to be NUMA-local to the BDF.
    - The IRQ affinity mask that mirrors the CPU set on the
      `/proc/irq/<n>/smp_affinity` side, per the public
      BlueField / DPU User Manual reached through
      [`doca-public-knowledge-map`](../doca-public-knowledge-map/SKILL.md).
   Record each choice; the same record drives the launch
   invocation in [`## run`](#run) and the isolation primitives in
   [`### isolation`](#isolation).
6. **Plan the rollback path.** For any deployment that will
   exercise device state under load (production traffic on
   doca-flow, production RDMA queue pairs on doca-rdma, production
   I/O on a doca-comch channel), every launch on a live target
   must have: (a) the pre-launch device state captured (host
   networking, representor list, firmware-query output); (b) the
   previous-known-good launch invocation (or a no-binary baseline)
   ready to re-apply; (c) an out-of-band way to reach the target
   if the launch disrupts host connectivity (BlueField console,
   redundant management path, IPMI to the host); (d) a maintenance
   window agreed with whoever uses the host. For interactive
   first-launch on a non-production target, the rollback bar is
   lower but the *"be able to revert"* rule still applies. When
   the planned launch touches hardware state itself (not just runs
   a DOCA workload), route to
   [`doca-hardware-safety`](../doca-hardware-safety/SKILL.md)
   FIRST.
7. **Confirm the four-way version match.** Per
   [`CAPABILITIES.md ## Version compatibility`](CAPABILITIES.md#version-compatibility),
   record BOTH the host (or BlueField Arm) DOCA install version
   via
   [`doca-version TASKS.md ## configure`](../doca-version/TASKS.md#configure),
   the binary's link-time `pkg-config doca-*` version (captured
   when the binary was built per
   [`doca-programming-guide ## build`](../doca-programming-guide/TASKS.md#build)),
   the BlueField firmware version (per the documented firmware-
   query surface in the BlueField / DPU User Manual), and the
   DOCA-version-policy row for the operator's release. A mismatch
   among any two is the silent *"the docs say this should work"*
   trap; align them explicitly.

## build

Bare-metal deployment is the *deploy* verb for an already-built DOCA
binary. There is no *application* artifact for the operator to build
inside this skill — the binary the operator deploys here is the one
they built per
[`doca-programming-guide ## build`](../doca-programming-guide/TASKS.md#build),
quoting the canonical `pkg-config doca-<library>` + meson pattern
that lives there.

If the user is asking how to build the binary they want to deploy,
hand off to
[`doca-programming-guide ## build`](../doca-programming-guide/TASKS.md#build)
(the canonical pattern, with C/C++ + non-C language tracks). Once
that returns a binary, control returns here at
[`## configure`](#configure) step 2 (host-mode recognition) for the
deploy.

If the user is asking how to build a systemd unit, a `numactl`
invocation, or any other launch-shell artifact, the answer is *the
launch artifact is composed against the live env*, not built ahead
of time. See [`## run`](#run) for how the launch-mode choice from
[`## configure`](#configure) step 4 composes the launch invocation
against the operator's PCI BDF / NUMA node / CPU set / `Restart=`
policy choice.

If the user is asking how to build a *DOCA library* itself, that is
a DOCA contributor workflow — out of scope for this skill per the
audience boundary in [`SKILL.md`](SKILL.md).

## modify

Bare-metal deployment does not have a *modify a sample program*
workflow analogous to DOCA libraries; the deployment-side analog of
"modify" is **re-walk the deploy after the binary changes, after
the launch invocation changes, or after the underlying env changes**:

1. **A binary change is a deploy event.** Any rebuild of the
   binary (a fresh `meson compile` per
   [`doca-programming-guide ## modify`](../doca-programming-guide/TASKS.md#modify))
   re-opens the smoke loop in [`## test`](#test) — the new binary
   may link against a different DOCA version, may have a different
   env-var contract, may exit on different signals. Re-walk
   [`## configure`](#configure) step 7 (version anchors), then
   [`## run`](#run), then [`## test`](#test).
2. **A launch-invocation change is a deploy event.** Editing the
   CPU pin set, the NUMA node, the PCI BDF, the env vars, or the
   `Restart=` policy on a systemd unit changes the deployment
   contract; treat each edit as a fresh deploy. Re-walk
   [`## run`](#run) step 3 (launch) and
   [`## test`](#test) step 1 (smoke).
3. **An env change is a deploy event.** A hugepage reservation
   change, an `LD_LIBRARY_PATH` change, a kernel-module load /
   unload, a devlink mode flip, or a representor moved into a
   different netns all change the deploy preconditions; re-walk
   [`## configure`](#configure) step 3 (preconditions).
4. **A hardware-state change leaves this verb entirely.** Any
   change touching device firmware (`mlxconfig set`, BFB
   reflash, firmware burn, BlueField mode flip) is owned by
   [`doca-hardware-safety`](../doca-hardware-safety/SKILL.md);
   this skill does NOT walk that workflow. Control returns here
   at [`## configure`](#configure) step 3 once the hardware-state
   change is complete and verified per the meta-policy.
5. **Modify the binary's source only at the programming-guide
   layer.** If the operator is modifying the binary's source
   (adding a new pipe, changing a queue depth, adapting to a
   different DOCA library version), that change is owned by
   [`doca-programming-guide ## modify`](../doca-programming-guide/TASKS.md#modify)
   + the matching `libs/<library>` skill. Once the modified
   binary is rebuilt, control returns here at step 1 above.

The agent's anti-pattern alert: editing a launch invocation in
place without re-walking the smoke is the canonical *"my deploy
silently degraded after a small change"* failure. Treat every
launch-side change as a fresh deploy.

## run

Bringing up the DOCA-linked binary, confirming the device-attach
layer reaches a healthy state, and confirming the trivial-workload
liveness signal before layering any real workload on top. Every step
here assumes the prerequisites in [`## configure`](#configure) are
done.

1. **Confirm the binary is executable and the loader resolves
   DOCA.** From the launch host, confirm `file <binary>` reports
   the expected arch (matches `uname -m`), `chmod +x` is set, and
   `ldd <binary>` resolves every `libdoca_*.so` against the
   install on the runtime `LD_LIBRARY_PATH` per
   [`doca-setup ## configure`](../doca-setup/TASKS.md#configure)
   step 3. A failure here is a layer-1 symptom in
   [`CAPABILITIES.md ## Error taxonomy`](CAPABILITIES.md#error-taxonomy)
   and must be resolved before the binary is launched.
2. **Compose the launch invocation against the captured
   bindings.** Per
   [`## configure`](#configure) step 5, the launch invocation
   names: the PCI BDF (or representor) the binary should attach
   to, derived from live `lspci -d 15b3:` / `devlink dev show`
   output; the CPU set and NUMA node (via the documented
   `numactl --cpunodebind` / `--membind` form or `taskset -c
   <cpu-list>`); the env-var surface the binary's documented
   contract requires (per the public DOCA Programming Guide for
   the library in use). The launch invocation is composed from
   live values; do NOT substitute a BDF, a NUMA number, or a
   representor name from memory.
3. **Launch in the chosen launch mode.** Per
   [`## configure`](#configure) step 4:
    - **Direct.** Invoke the binary in the foreground from the
      shell session; stdout / stderr arrive on the terminal.
    - **tmux / screen.** Open a named tmux / screen session,
      invoke the binary inside it, detach; reattach with the
      documented commands when reading live output.
    - **systemd-supervised.** Drop the operator's authored
      `.service` unit into the documented systemd unit path
      (per `systemd.unit(5)`), `systemctl daemon-reload`,
      `systemctl start <unit>`. The `Restart=` policy in the
      unit must be one of the modes named in
      `systemd.service(5)`; do NOT invent a mode.
4. **Verify the process is up and the device attached.** Per the
   observability surface in
   [`CAPABILITIES.md ## Observability`](CAPABILITIES.md#observability):
    - Direct: the binary's own stdout / stderr show the
      documented bring-up lines from the public DOCA Programming
      Guide for the library in use; no documented error lines
      repeat.
    - tmux: reattach and read the live pane buffer.
    - systemd: `journalctl -u <unit> -f` shows the same
      documented bring-up lines.
   A process that is up but printing layer-3 errors *"no
   matching device"* / *"representor not found"* / *"PCI BDF not
   found"* per [`CAPABILITIES.md ## Error taxonomy`](CAPABILITIES.md#error-taxonomy)
   is NOT ready; drop to [`## debug`](#debug) layer 3 BEFORE
   proceeding.
5. **Verify the trivial-workload liveness signal.** The
   documented liveness signal for the library in use (a single
   matched packet for doca-flow, a single RDMA write-with-imm for
   doca-rdma, a single comch send for doca-comch, etc.) lives in
   the matching `libs/<library>` skill — read it now from there
   per [`doca-programming-guide CAPABILITIES.md ## Error taxonomy`](../doca-programming-guide/CAPABILITIES.md#error-taxonomy)
   for the cross-library escalation rules. If the per-library
   liveness signal is NOT healthy, drop to
   [`## debug`](#debug) layer 4 (library error) or to the
   per-library skill's debug ladder — not to *"restart the
   binary and hope"*.
6. **Smoke before bulk (next: [`## test`](#test) step 1).**
   Before driving any real workload, walk [`## test`](#test)
   step 1 once to confirm end-to-end readiness; only then layer
   the workload on top.

### isolation

This sub-anchor covers per-tenant isolation for multiple DOCA
processes co-tenant on the same BlueField. It is reached from
[`## run`](#run) step 2 when the launch invocation crosses a
tenant boundary, and from [`## debug`](#debug) layer 7 when the
symptom is co-tenant noise. The isolation rules layered here are
the bare-metal-specific overlay on the standard Linux primitives
named at class level in
[`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes).

1. **Compose the cgroup-v2 cpu / memory / io budget per
   tenant.** Per the kernel cgroup-v2 documentation, each
   tenant's process is placed in its own cgroup with
   `cpu.weight` (or `cpu.max` for hard caps), `memory.max`
   (sized to cover the binary's resident set PLUS the hugepage
   accounting noted in
   [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes)),
   and `io.weight` if the binary touches block I/O. Set the
   budget BEFORE the launch; verifying after the launch is too
   late.
2. **Compose the network-namespace per tenant when traffic
   isolation is required.** The standard Linux primitive is
   `ip netns add <name>` + moving the per-tenant representor
   into the netns per the documented `ip link set <repr> netns
   <name>` form. The DOCA-specific gotcha: the DOCA library is
   only able to see the representor inside the netns that owns
   it; launching the binary in a different netns (or in the
   root netns when the representor was moved out) reproduces
   as a layer-3 *"no matching device"* error.
3. **Compose the CPU / NUMA pinning per tenant via `numactl` /
   `taskset`.** Per `numactl(8)` and `taskset(1)`, the
   per-tenant CPU set is NUMA-local to the PCI BDF the binary
   attaches to (per
   [`## configure`](#configure) step 5). The DOCA-specific
   gotcha: a per-tenant CPU set that crosses the NUMA boundary
   owning the NIC cancels the isolation gain — both tenants
   then compete on memory bandwidth across the NUMA
   interconnect.
4. **Confirm hugepage accounting per tenant.** Hugepages
   reserved per
   [`doca-setup ## configure`](../doca-setup/TASKS.md#configure)
   step 4 are a global pool; each tenant's allocation is
   accounted against its cgroup's `memory.max` per the kernel
   cgroup-v2 hugepage accounting rules. A `memory.max` value
   that fits the binary's resident set but starves the
   hugepage pool surfaces as layer-2 *"process starts but
   exits immediately"* with an `EAL: Cannot get hugepage
   information` line.
5. **Verify the isolation primitives BEFORE the workload
   starts.** Per the safety rule in
   [`CAPABILITIES.md ## Safety policy`](CAPABILITIES.md#safety-policy):
   `systemd-cgls` (or `cat /sys/fs/cgroup/<path>/cpu.max`
   etc.) confirms the cgroup limits are in effect;
   `ip netns exec <name> ip link show` confirms the
   per-tenant representor is in the right netns; `taskset -p
   <pid>` / `numactl --show -p <pid>` confirms the running
   binary is pinned to the planned CPU set / NUMA node. Adding
   the verification after a co-tenant complains is the wrong
   order.

## test

Bare-metal deployment has no *compile and unit-test* workflow —
testing is operational and end-to-end against real hardware.

**`## test` is an iterative loop, not a one-shot pass.** Every
mutation (binary rebuild, launch-invocation edit, env change,
co-tenant addition) re-opens the smoke sweep. Skipping the re-run
after a mutation is the failure mode this loop replaces.

The smoke-before-bulk loop (rows apply to every DOCA-linked binary
on the bare-metal path, not just one library):

| Step | Why this is a loop, not a step | Where the substance lives |
| --- | --- | --- |
| 1 → 4 → 1 | Step 4 (per-library liveness probe) often reveals an as-launched gap in the binding (wrong representor, wrong NUMA node) that masquerades as a binary problem; loop back to step 1 | [`## test`](#test) step 4 |
| 2 → ## debug | When trivial-arg invocation fails, the binary cannot reach DOCA at all — escalate to [`## debug`](#debug) immediately, do not run later steps | [`## debug`](#debug) |
| 3 → ## configure → 3 | When the liveness-equivalent invocation shows a precondition was not closed (hugepages, devlink mode, representor visibility), loop back to [`## configure`](#configure) step 3 and re-walk the preconditions | [`## configure`](#configure) |
| 1..5 → ## run | Each loop iteration ends with a documented smoke; if all five pass, hand off to live [`## run`](#run) traffic | [`## run`](#run) |

The five steps of the smoke:

1. **Trivial-arg smoke.** Invoke the binary with its `--help` /
   `--version` equivalent (whichever the binary actually
   supports — the agent does NOT invent a flag). Confirm the
   binary executes, the dynamic loader resolves every
   `libdoca_*.so`, and the documented version / usage string
   appears. A failure here is layer 1 of
   [`CAPABILITIES.md ## Error taxonomy`](CAPABILITIES.md#error-taxonomy).
2. **Liveness-equivalent smoke.** Invoke the binary against the
   captured bindings from
   [`## configure`](#configure) step 5 but with NO real
   workload offered. Confirm the binary reaches the documented
   bring-up lines, opens the device, and exits cleanly on
   `SIGTERM`. A failure here is layer 2 or layer 3 of
   [`CAPABILITIES.md ## Error taxonomy`](CAPABILITIES.md#error-taxonomy).
3. **Trivial-workload smoke.** Drive ONE operation through the
   binary — one packet for doca-flow, one queue-pair operation
   for doca-rdma, one channel send for doca-comch, etc. (per
   the matching `libs/<library>` skill's documented liveness
   contract). Confirm the per-library counter advances by
   exactly one. A failure here is layer 4 of
   [`CAPABILITIES.md ## Error taxonomy`](CAPABILITIES.md#error-taxonomy).
4. **Capability + launch snapshot.** Save the *as-launched*
   answer to: which binary version was running, which DOCA
   install version it linked against (per
   [`doca-version TASKS.md ## run`](../doca-version/TASKS.md#run)),
   which BlueField firmware version the device was on, which
   PCI BDF / NUMA node / CPU set the binary attached to, the
   chosen launch mode, the `Restart=` policy if systemd-
   supervised, the captured stdout / journald lines from the
   bring-up. This snapshot is the artifact that lets future
   debug sessions skip rediscovery — and on a HIGH-STAKES
   deploy it is the rollback baseline.
5. **Multi-tenant smoke (only when co-tenants are planned).**
   Bring up each additional tenant AFTER this binary, one at a
   time, and re-run step 3 (trivial-workload smoke) against
   THIS binary between additions. A regression in this
   binary's per-library counter that only appears once a
   co-tenant is added is layer 7 of
   [`CAPABILITIES.md ## Error taxonomy`](CAPABILITIES.md#error-taxonomy);
   walk [`### isolation`](#isolation) before queueing more
   tenants.

Loop termination: stop iterating once two consecutive iterations
of the same kind don't change anything — that means the cause is
below the deployment runtime (per-library configuration, hardware
state, BlueField OS, host). Escalate to the per-library skill's
debug ladder plus
[`doca-debug TASKS.md ## debug`](../doca-debug/TASKS.md#debug)
with the captured layer evidence.

## debug

Layered diagnosis. Walk the layers in this order; do not skip down
without clearing the layer above. The seven layers match
[`CAPABILITIES.md ## Error taxonomy`](CAPABILITIES.md#error-taxonomy).

1. **Process-won't-start layer (layer 1).** Is the binary even
   executing? Symptoms: *"command not found"*, *"permission
   denied"*, *"cannot execute binary file: Exec format
   error"*, or `error while loading shared libraries:
   libdoca_*.so` from the dynamic loader. Resolution: confirm
   the binary's arch via `file <binary>` matches `uname -m`;
   confirm executability (`ls -l`); resolve every shared
   object with `ldd <binary>`; set `LD_LIBRARY_PATH` per
   [`doca-setup ## configure`](../doca-setup/TASKS.md#configure)
   step 3. Do NOT add a randomly-chosen `libdoca_*.so` to
   `LD_LIBRARY_PATH` from memory — quote from the live install
   layout per
   [`doca-public-knowledge-map ## Layout of an installed DOCA package`](../doca-public-knowledge-map/SKILL.md#layout-of-an-installed-doca-package).
2. **Process-starts-and-exits-immediately layer (layer 2).**
   Binary launches, runs for milliseconds, exits non-zero.
   Symptoms: short stderr message about a missing env var, a
   missing config file, or `EAL: Cannot get hugepage
   information`; supervisor reports the process restarted.
   Resolution: read the binary's stdout / stderr (terminal /
   tmux pane / `journalctl -u <unit>` per the launch mode);
   re-walk the documented env-var surface in the public DOCA
   Programming Guide for the library in use; re-run hugepage
   reservation per
   [`doca-setup ## configure`](../doca-setup/TASKS.md#configure)
   step 4. NEVER paste an env-var name from memory; quote from
   the per-library guide.
3. **Process-runs-but-cannot-find-the-device layer (layer 3).**
   Binary stays up but per-library bring-up reports *"no
   matching device"*, *"representor not found"*, or
   `DOCA_ERROR_NOT_FOUND` from the device-open call.
   Resolution: re-walk the device-visibility surface per
   [`doca-setup ## configure`](../doca-setup/TASKS.md#configure)
   step 5 — confirm the PCI BDF the binary was launched with
   matches a live `lspci -d 15b3:` entry; confirm the
   representor is enumerated in `ip link show` and is in the
   netns the binary launched in; confirm the eswitch is in
   the documented mode per the public BlueField / DPU User
   Manual. Do NOT substitute a BDF from a previous deploy.
4. **Library-error layer (layer 4).** Binary attaches to the
   device but the workload errors with a `DOCA_ERROR_*` per
   [`doca-programming-guide CAPABILITIES.md ## Error taxonomy`](../doca-programming-guide/CAPABILITIES.md#error-taxonomy).
   Resolution: defer to the cross-cutting debug ladder at
   [`doca-debug TASKS.md ## debug`](../doca-debug/TASKS.md#debug)
   FIRST, then to the matching `libs/<library>` skill for the
   library-specific overlay. Per-library error codes are owned
   by the per-library skill, not by this one.
5. **OOM / signal / resource-limit layer (layer 5).** Binary
   disappears with no log line, or supervisor reports an
   unexpected exit signal. Symptoms: `dmesg` shows an
   OOM-killer entry naming the binary; supervisor reports
   `SIGKILL` / `SIGTERM`; restart count climbs without a
   matching binary-side error log. Resolution: read the
   supervisor's exit-status record; check `dmesg` for OOM;
   re-check the cgroup-v2 budget per
   [`### isolation`](#isolation) step 1; confirm the binary's
   documented signal-handling contract from the public DOCA
   Programming Guide. A `memory.max` that fits the resident
   set but starves the hugepage pool is the canonical trap.
6. **Restart-loop layer (layer 6, HIGH-STAKES).** Supervisor
   keeps re-launching the binary; each launch exits with the
   same exit signature; the device the binary touches may be
   reporting odd errors caused by the loop itself. Per the
   HIGH-STAKES rule in
   [`CAPABILITIES.md ## Safety policy`](CAPABILITIES.md#safety-policy):
   STOP the supervisor (`systemctl stop <unit>` for systemd-
   supervised launches; manual termination for tmux); read
   the binary's LAST full log surface; walk layers 1-5 above
   against the captured evidence; only re-enable the supervisor
   once the root cause is identified. Letting the supervisor
   loop a known-broken binary is delayed diagnosis, not
   resilience.
7. **Co-tenant-noise layer (layer 7).** This binary behaves
   correctly in isolation; introducing a second DOCA process
   (or any other process) on the same BlueField makes this
   binary's counters degrade. Resolution: re-walk
   [`### isolation`](#isolation) for the second tenant's
   cgroup-v2 / netns / `numactl` configuration BEFORE drawing
   a per-library conclusion; if the symptom only reproduces
   under co-tenancy, the diagnosis is multi-tenant isolation,
   not a per-library bug.
8. **Cross-cutting host layer (last resort).** Deployment looks
   healthy at every layer above but a cross-cutting host issue
   (kernel version, driver loaded / not loaded, PCIe link
   state, hugepage allocation health beyond reserved-vs-used,
   BFB log surface) breaks the binary's downstream behavior.
   Resolution: drop to
   [`doca-debug TASKS.md ## debug`](../doca-debug/TASKS.md#debug)
   for the cross-cutting debug ladder; if the symptom is a
   hardware-state change the operator is contemplating, route
   to
   [`doca-hardware-safety`](../doca-hardware-safety/SKILL.md)
   instead of touching device state from this skill.

## bluefield-lifecycle

The BlueField **platform lifecycle** anchor — the workflow for taking
a BlueField from "powered card in the slot" to "Arm OS healthy,
TMFIFO up, host PFs bound, four-way version match closed, DOCA-linked
binary safely launchable per [`## run`](#run)". This anchor exists
because the bare-metal-deployment skill's downstream verbs (`## run`,
`## test`, `## debug`) all assume a working BlueField; when that
assumption breaks, the bundle previously routed the operator out to
[`doca-hardware-safety`](../doca-hardware-safety/SKILL.md) for the
mutating-change meta-policy AND out to
[`doca-public-knowledge-map ## Externally-productized DOCA software`](../doca-public-knowledge-map/SKILL.md#externally-productized-doca-software--not-in-this-bundle-but-here-is-where-to-route)
for the BSP/BFB documentation entry, but the **operational sequencing
ladder itself** (which evidence to collect in which order, which
failure mode each evidence pattern points to, which recovery action
lattices to which evidence) had no home. This section is that home.

This is a **reasoning ladder, not a script.** The agent does not ship
a `bfb-install-wrapper.sh` or a `classify-bluefield-state.sh`; it
prescribes the *order in which an operator should collect evidence
and the recovery action each evidence pattern lattices to*, quoting
documented commands from the public BlueField Platform Software
Manual (reached through
[`doca-public-knowledge-map`](../doca-public-knowledge-map/SKILL.md)),
the MFT manual (`flint`, `mlxconfig`, `mlxfwmanager`), and the Linux
man pages for `modprobe(8)`, `lspci(8)`, `ip-route(8)`. Every
mutating step (firmware burn, BFB reflash, `mlxconfig set`, kernel
boot parameter change) STILL routes through
[`doca-hardware-safety`](../doca-hardware-safety/SKILL.md) for the
meta-policy (preflight, OOB console, maintenance window, rollback)
— this section adds only the *bare-metal-deployment-specific
sequencing* on top.

### bfb-install lifecycle

The canonical "push a BFB image from the host to the BlueField" flow.
The agent walks the operator through this verb-by-verb; it does NOT
fabricate the host-side `bfb-install` flag set, the BFB image
filename, the RShim character-device path, or the `bf.cfg` schema —
all four come from live `--help` on the installed tool and the
public BlueField Platform Software Manual.

1. **Pre-flight inventory.** Capture, BEFORE any push:
    - The current BFB image / BSP version on the BlueField, from the
      BSP version-query path documented in the BlueField Platform
      Software Manual (do NOT guess a command name).
    - The current ConnectX firmware version on the BlueField's NIC
      side, from `flint -d <bdf> q` (per the MFT manual); on a
      BlueField this is the NIC PSID + firmware revision the new
      BFB will or will not match.
    - The host-side RShim userspace daemon state (`dpkg -s
      rshim` / `rpm -q rshim` for package install, `systemctl
      status rshim` for `active (running)`, `pgrep -a rshim` for
      a live `/usr/sbin/rshim` process, `ls /dev/rshim*` for the
      character-device tree — all per the BSP manual). On DOCA
      3.3+ there is NO `rshim` kernel module (the in-tree
      module was removed); `lsmod | grep rshim` is expected to
      be empty and is NOT failure evidence. If the daemon /
      `/dev/rshim*` tree is missing, the host has no path to
      push.
    - The OOB management path the operator will use if the push
      breaks the Arm OS: BMC console-over-Redfish, BMC IPMI
      serial-over-LAN, or the physical UART (per the
      [`BlueField BMC Software`](../doca-public-knowledge-map/SKILL.md#externally-productized-doca-software--not-in-this-bundle-but-here-is-where-to-route)
      row in the public-knowledge-map). Without one of these, the
      bar for proceeding is "stop, escalate" per
      [`doca-hardware-safety`](../doca-hardware-safety/SKILL.md).
    - The BFB image's SHA matches the SHA the operator
      downloaded from the documented DOCA Downloads page (per
      [`doca-public-knowledge-map`](../doca-public-knowledge-map/SKILL.md#public-documentation-entry-points))
      — pushing a corrupted BFB is the load-bearing first-run
      failure for the entire flow.
2. **Author `bf.cfg` from the documented schema.** The BFB-install
   path takes an installer configuration file (`bf.cfg`) that
   controls post-install state on the Arm side — root/ubuntu
   password, hostname, and `bfb_modify_os()` shell-script hooks
   that run during the install to seed any state the documented
   `bf.cfg` parameters do NOT cover directly. **Two
   operator-relevant rules**:
   (a) If the operator wants passwordless SSH to survive the
   install, the `bf.cfg` MUST set the documented `ubuntu_PASSWORD`
   parameter (or equivalent per the schema, e.g. `ROOT_PASSWORD`)
   AND seed the SSH public key via a `bfb_modify_os()` hook that
   writes `/mnt/home/ubuntu/.ssh/authorized_keys` on the
   to-be-installed rootfs (the rootfs is mounted under `/mnt`
   during install per the BSP manual). The default BFB install
   rewrites `/home/ubuntu/.ssh/`, so any pre-existing key on the
   previous image is GONE unless reseeded by this hook. Do NOT
   invent an `authorized_keys` top-level `bf.cfg` parameter — the
   schema does not have one; the mechanism is the
   `bfb_modify_os()` hook.
   (b) For separated-host / bump-in-the-wire / scalable-function
   deployments where the BlueField must boot in a specific
   internal-CPU / port-owner / SF mode (e.g. `SEPARATED_HOST(0)`,
   `EMBEDDED_CPU(1)`), the required `mlxconfig set` invocations
   are run from a `bfb_modify_os()` hook in `bf.cfg` at
   BFB-install time; reconfiguring the same modes after the
   install typically requires another BFB push (per the BSP
   manual). The agent quotes the `bf.cfg` parameter keys from the
   public schema (and references the `bfb_modify_os()` script
   pattern) and does NOT invent key names from memory.
3. **Push the BFB.** Run the host-side `bfb-install` invocation
   per its `--help` and the BSP manual. The push streams the BFB
   to the BlueField over RShim/PCIe; the Arm side reboots through
   UEFI → Linux up → first-boot init.
4. **Do not trust `bfb-install` exit code 0 alone.** This is the
   single most expensive failure mode in the operator's loop.
   `bfb-install` has been observed in the field to exit 0 while
   the Arm-side flow only partially completed — the canonical
   field-reported signature is *"Ubuntu installation completed"*
   (or *"Ubuntu installation finished"*, both phrasings have
   been seen in different BFB releases) followed by an
   `INFO[MISC]: NIC firmware update failed` line in the RShim
   console / log, meaning the OS image landed but the
   firmware-update sub-step silently failed. The agent ALWAYS
   parses the actual console / log output the installer wrote,
   in addition to the exit code, and looks for: (a) any `[MISC]`
   or `[ERR]` line whose text contains a failure verb
   ("failed", "error", "abort"), (b) the documented `Linux up`
   marker, (c) the documented `DPU is ready` marker. If any of
   `(a)` is present or `(b)` / `(c)` are absent, the install is
   treated as **partial**, not complete, and the agent advances
   to
   [`### bluefield-state-classifier`](#bluefield-state-classifier)
   instead of declaring success.
5. **Distinguish "BFB install completed" from "readiness wait
   failed".** "Completed" means the installer's I/O is done; it
   does NOT mean the Arm OS is up, TMFIFO is reachable, SSH is
   live, or host PFs are bound. The agent always runs the
   readiness sequence in
   [`### post-bfb-recovery`](#post-bfb-recovery) before re-declaring
   the BlueField healthy.
6. **Routing for the firmware burn itself.** A BFB push is a
   mutating change against live device state — the meta-policy in
   [`doca-hardware-safety`](../doca-hardware-safety/SKILL.md)
   governs the preflight / OOB-console / rollback discipline that
   wraps this step. This section adds only the *sequencing
   ladder*; the agent loads `doca-hardware-safety` ALONGSIDE this
   skill whenever the operator's question reaches step 3 above.

### rshim and tmfifo

RShim is the host-side surface that exposes the BlueField's
Arm-side console (`/dev/rshim<N>/console`) AND the host-side
network endpoint of the TMFIFO recovery interface (factory
defaults per the BlueField Platform Software Manual: host-side
address `192.168.100.1/30`, BlueField-side address
`192.168.100.2/30`; the agent does NOT fabricate the subnet from
memory). On DOCA 3.3+ hosts the RShim surface ships as a
**userspace daemon** (`/usr/sbin/rshim` started by
`rshim.service`); the legacy in-tree kernel module is no longer
shipped. The TMFIFO interface is the *recovery* path when the
BlueField's normal management network is broken; it is NOT a
primary data path.

1. **Verify RShim is attached on the host.** The agent runs all
   three of: `dpkg -s rshim` / `rpm -q rshim` (the userspace
   package is installed), `systemctl status rshim` (the daemon is
   `active (running)`), and `ls /dev/rshim*` (character-device
   tree is present per the BSP manual). On DOCA 3.3+
   `lsmod | grep rshim` is EXPECTED to be empty and is NOT
   evidence of failure — the in-tree kernel module is gone; the
   surface is delivered entirely by the userspace daemon. If
   `systemctl status rshim` is not `active (running)` OR the
   `/dev/rshim*` tree is missing, the host has no path to the
   BlueField's recovery surface and downstream TMFIFO checks are
   meaningless.
2. **Verify the TMFIFO network endpoint on the host.** `ip
   addr show tmfifo_net0` (per the BSP manual's documented
   interface name; verify the name on the operator's host —
   different driver versions have shipped slightly different
   names). The address should be the host-side documented address.
3. **Critical TMFIFO gotcha — ALWAYS `ip route get` before
   `ping`.** A real failure mode the bundle has hit in the wild
   is: the BlueField-side TMFIFO address (e.g. `192.168.100.2`)
   gets accidentally added to the **host's** loopback or to
   `tmfifo_net0` itself, so `ping 192.168.100.2` from the host
   succeeds — but it is pinging *the host*, not the BlueField.
   The diagnostic that catches this in one command is:
    - `ip route get <bf-tmfifo-address>` on the host. **Healthy
      outputs** the agent should accept as "the route is going to
      the BlueField": `<bf-addr> dev tmfifo_net0 src <host-addr>`
      (driver versions that expose the TMFIFO interface directly)
      OR `<bf-addr> dev tm-br src <host-addr>` (BSP / DOCA-host
      installs that bridge `tmfifo_net0` into a `tm-br` bridge —
      observed in the wild on DOCA 3.3 hosts where the BFB-side
      RShim driver is configured to bridge the TMFIFO endpoint
      with the host-side management bridge). **Broken output** is
      always: `<bf-addr> dev lo src 127.0.0.1` or
      `local <bf-addr> dev lo` — the address has been bound
      LOCALLY and *every* ping / ssh / curl to it is hitting the
      host, not the BlueField. The agent ALWAYS runs
      `ip route get` before trusting `ping` for TMFIFO recovery,
      and accepts either `tmfifo_net0` or `tm-br` as the egress
      interface on the host side.
4. **Soft-reset is a recovery action, not a routine action.** The
   `rshim` soft-reset path documented in the BSP manual is
   appropriate when the Arm side is stuck in a known-recoverable
   state (UEFI hang post-BFB-install, console responsive but
   userspace dead); it is NOT routine. Routing: the operator
   captures the BlueField state per
   [`### bluefield-state-classifier`](#bluefield-state-classifier)
   FIRST, then decides whether soft-reset, cold power cycle, or
   re-push BFB is the appropriate recovery — and loads
   [`doca-hardware-safety`](../doca-hardware-safety/SKILL.md)
   ALONGSIDE for the meta-policy on any of those.

### post-bfb-recovery

After a BFB push lands, the BlueField's downstream surface (Arm
OS health, TMFIFO reachability, SSH liveness, host PFs bound,
firmware version match) must be re-verified before the
bare-metal-deployment skill returns to [`## run`](#run). This is
the "did the install actually take?" gate.

1. **Wait for the documented readiness markers, not for a
   timer.** A wall-clock sleep is not equivalent to a readiness
   probe. The agent polls for the documented `Linux up` / `DPU is
   ready` markers in the RShim console buffer (per the BSP
   manual), AND for the documented Arm-side SSH endpoint
   responding, AND for the documented BMC health endpoint
   reporting `OK`. If any of these never report ready within the
   manual's documented bound, the BlueField is in a partial state
   and the agent advances to
   [`### bluefield-state-classifier`](#bluefield-state-classifier)
   rather than declaring success.
2. **Host PF rebind sequence.** A BFB push can leave the host
   side's mlx5 driver in a stale state: the PCI devices for the
   BlueField PFs are present (`lspci -d 15b3:` lists them) but
   `ip link show` does not enumerate the netdevs, RDMA enumeration
   is empty, and any DOCA program that attaches by representor
   name fails. The documented recovery is:
    - `modprobe mlx5_core` (no-op if already loaded; loads if not).
    - For each BlueField PF BDF captured at pre-flight (e.g.
      `0000:b3:00.0`, `0000:b3:00.1`): `echo <bdf> >
      /sys/bus/pci/drivers/mlx5_core/bind` per the kernel sysfs
      driver-binding documentation.
    - Re-verify: `ip link show` enumerates the BlueField netdevs;
      `ibv_devinfo` enumerates the BlueField RDMA devices;
      `devlink dev show` lists the BlueField devlink instance.
   The agent does NOT invent the BDF strings from memory; they
   come from the pre-flight `lspci -d 15b3:` capture.
3. **`/home/ubuntu` operational gotcha (Arm-side BlueField OS).**
   On certain BlueField OS images, `/home/ubuntu` ships owned by
   `root` rather than `ubuntu`, which breaks the normal pattern of
   the `ubuntu` user writing logs / scratch files under their own
   home directory. The agent: (a) checks `stat -c '%U:%G' /home/ubuntu`
   after first SSH; (b) if it is `root:root`, flags it to the
   operator and proposes the documented `chown -R ubuntu:ubuntu
   /home/ubuntu` fix (per the BSP manual) BEFORE the operator
   pastes any script that writes there. The fix itself is
   trivially documented Linux; the value is the recognition
   *during* lifecycle recovery instead of after a script fails.
4. **Log copy-back to host workspace.** All install / readiness /
   recovery evidence collected on the Arm side (the RShim console
   buffer, the cloud-init log, the documented BSP install log,
   the readiness probe output) should be copied BACK to the host
   workspace before the operator re-attempts the workload. Two
   rules: (a) use `scp ubuntu@<bf-mgmt-addr>:<path> .` (or the
   documented BSP log-export path) from the host side — pulling
   is safer than pushing host credentials onto the BF; (b) do NOT
   wrap the copy step in `sudo` on the host unless the operator
   explicitly entered `sudo` mode for this lifecycle session, per
   the smoke-before-bulk rule in
   [`## run`](#run) step 4.
5. **Four-way version-match re-close.** Once Arm OS is healthy
   and host PFs are bound, the BlueField's new BFB / firmware
   stack must satisfy the four-way version match owned by
   [`doca-version TASKS.md`](../doca-version/TASKS.md). The
   agent walks the four-way match against the new BlueField state
   BEFORE returning to [`## configure`](#configure) step 7. A
   skipped re-close after a BFB push is the most common cause of
   "ran fine yesterday, breaks today" symptoms.

### bluefield-state-classifier

When a BFB push, a soft-reset, or a host PF rebind has been done
and the BlueField is *not yet* confirmed healthy, the agent walks
this six-state classifier IN ORDER and stops at the first state
the evidence matches. Each state names the evidence that
identifies it, the most-likely root cause class, and the
*sequencing* of the recovery action (mutating steps still load
[`doca-hardware-safety`](../doca-hardware-safety/SKILL.md)
alongside; this ladder names the order, not the burns).

This is a **reasoning ladder, not a binary**: an Arm OS can be
"Linux up" AND "host PFs unbound" simultaneously; the agent
walks the ladder top-to-bottom and reports every state that
matches, not just the first. The point is the *sequencing of
evidence collection*, so two ops engineers reading the same
console output reach the same triage step.

1. **`installer-still-running`.** Evidence: `bfb-install` is still
   resident on the host (`ps -ef | grep bfb-install`), AND the
   RShim console buffer is still emitting documented installer
   progress lines per the BSP manual. Root cause class: install
   in flight, not failure. Recovery: WAIT, do not abort. The
   `bfb-install` push can take many minutes on first-flash;
   aborting it mid-write is what *creates* the next state down.
2. **`uefi-only`.** Evidence: the RShim console buffer reports
   `exit Boot Service` (or the BSP manual's equivalent UEFI-exit
   marker) but never reaches the documented `Linux up` marker.
   Root cause class: kernel did not hand off to userspace —
   common after a partial BFB-install with a firmware-update
   error (see [`### bfb-install lifecycle`](#bfb-install-lifecycle)
   step 4). Recovery sequencing: capture full RShim console
   buffer to host, capture host-side `dmesg`, do NOT immediately
   re-push BFB; first perform a    documented cold power cycle of
   the BlueField (via BMC per the
   [`BlueField BMC Software`](../doca-public-knowledge-map/SKILL.md#externally-productized-doca-software--not-in-this-bundle-but-here-is-where-to-route)
   row, not just `reboot` from the unresponsive Arm side); only
   if cold power cycle does NOT recover, re-push BFB under
   [`doca-hardware-safety`](../doca-hardware-safety/SKILL.md)
   meta-policy.
3. **`linux-up-tmfifo-down`.** Evidence: documented `Linux up`
   marker present in RShim console, BUT TMFIFO probe per
   [`### rshim and tmfifo`](#rshim-and-tmfifo) step 2 returns
   no host-side address. Root cause class: TMFIFO interface
   never came up (driver / udev / link-state). Recovery
   sequencing: re-check host-side RShim daemon, run the
   documented TMFIFO bring-up procedure from the BSP manual,
   THEN re-run [`### rshim and tmfifo`](#rshim-and-tmfifo)
   step 3 (the `ip route get` gotcha) — a freshly-bound TMFIFO
   on the host can land in the loopback failure mode and look
   like it is working.
4. **`tmfifo-up-ssh-down`.** Evidence: documented TMFIFO probe
   passes (and `ip route get` confirms the route is going to the
   BlueField, not local loopback), BUT SSH to the documented
   Arm-side management endpoint refuses or hangs. Root cause
   class: Arm-side sshd not yet listening (still in init), OR
   the operator's authorized_keys / `ubuntu_PASSWORD` was NOT in
   the `bf.cfg` per [`### bfb-install lifecycle`](#bfb-install-lifecycle)
   step 2. Recovery sequencing: wait the documented sshd-ready
   bound from the BSP manual; if still down, fall through to the
   RShim console (`/dev/rshim<N>/console`) for a userspace
   prompt and re-seed credentials there; on the next BFB push,
   put the `authorized_keys` IN the `bf.cfg`.
5. **`arm-ok-host-pfs-unbound`.** Evidence: Arm-side SSH alive,
   Arm OS reports healthy (uptime > a few seconds, `dmesg` clean),
   BUT host-side enumeration is broken — `lspci -d 15b3:` shows
   the BlueField PFs, `ip link show` does NOT show the BlueField
   netdevs, `ibv_devinfo` is empty, DOCA programs cannot attach
   by representor name. Root cause class: stale host mlx5
   driver-binding state post-BFB push. Recovery sequencing: run
   [`### post-bfb-recovery`](#post-bfb-recovery) step 2 (the
   documented PF-rebind sequence). Do NOT proceed to launch any
   DOCA-linked binary in this state — every device-open will
   fail with a misleading error.
6. **`host-bf-version-mismatch`.** Evidence: everything above
   looks healthy, but the four-way version match owned by
   [`doca-version`](../doca-version/SKILL.md) does not close —
   host DOCA install version, BlueField BFB / BSP version,
   ConnectX firmware version, and binary's link-time
   `pkg-config doca-*` version do not satisfy the documented
   compatibility matrix in the DOCA Compatibility Policy
   (linked from
   [`doca-public-knowledge-map`](../doca-public-knowledge-map/SKILL.md#public-documentation-entry-points)).
   Root cause class: the operator pushed a BFB but did not
   simultaneously align the host DOCA-Host install, OR the
   `/etc/apt/sources.list.d/doca.list` is pointed at a
      different release channel than what is now installed (see
      [`doca-version TASKS.md ## apt-source consistency`](../doca-version/TASKS.md#apt-source-consistency)).
   Recovery sequencing: walk
   [`doca-version TASKS.md`](../doca-version/TASKS.md) in full;
   resolve any apt-source / repo-pin drift BEFORE installing
   anything new; route any host-side DOCA reinstall through
   [`doca-setup`](../doca-setup/SKILL.md).

Two cross-cutting rules for this classifier:

- **Match multiple states; do not stop at the first.** A
  BlueField that just came back from a partial BFB-install can
  match `uefi-only` initially, then `arm-ok-host-pfs-unbound`
  after a cold power cycle. The agent reports the *current*
  state set and the most-recently-observed transition, not just
  the first match.
- **Never declare healthy from absence of evidence.** "TMFIFO
  ping succeeded" without `ip route get` is NOT evidence the
  BlueField is reachable (see [`### rshim and tmfifo`](#rshim-and-tmfifo)
  step 3). "Arm SSH connected" without `uptime` / `dmesg` / a
  documented health probe is NOT evidence the Arm OS finished
  initialising. "host-side `lspci` shows the PFs" is NOT
  evidence the PFs are usable (see
  [`### post-bfb-recovery`](#post-bfb-recovery) step 2). The
  classifier states are walked top-to-bottom precisely so the
  agent does not skip an unverified gate.

## Command appendix

Bare-metal-deployment commands the verbs above reach for, grouped
by purpose so the agent picks the right family without searching
prose. Every row is a CLASS — the agent must not invent flags or
specific values beyond what the row names; flag and value discovery
is `--help` on the installed tool, the public DOCA / BlueField docs,
or the Linux man pages, not prose recall.

**Infra-aware preamble (every row below).** Per the bundle's
detect → prefer → fall back → report contract documented in
[`doca-structured-tools-contract ## The agent behavior contract`](../doca-structured-tools-contract/SKILL.md#the-agent-behavior-contract),
the agent should:

1. Probe for the matching structured helper FIRST (`doca-env
   --json` for version + devices + libraries + drivers +
   hugepages in one shot; `collect-host-state` /
   `collect-dpu-state` for the host-side device topology;
   `version-matrix.json` for *"available since"* lookups).
2. If the probe succeeds, the structured tool's output is the
   authoritative answer and the agent SHOULD NOT also run the
   manual command in the row below. Report *"using structured
   `<tool>`"*.
3. If the probe fails, fall back to the manual command in the
   row. Report *"falling back to manual chain"*.
4. The schemas the structured tools emit are defined in
   [`doca-structured-tools-contract ## Schemas`](../doca-structured-tools-contract/SKILL.md#schemas);
   the version-handling semantics (four-way match, NGC,
   headers-win) are owned by
   [`doca-version`](../doca-version/SKILL.md).

| Purpose | Command (class shape) | Owning step | Reads as healthy when … |
| --- | --- | --- | --- |
| Binary arch + loader resolution | `file <binary>` ; `ldd <binary>` | [`## run`](#run) step 1; [`## debug`](#debug) layer 1 | Arch matches `uname -m`; every `libdoca_*.so` resolves to a path under the active DOCA install. |
| Binary executability | `ls -l <binary>` ; `chmod +x <binary>` if needed | [`## run`](#run) step 1; [`## debug`](#debug) layer 1 | The execute bit is set for the user that will launch the binary. |
| PCI function enumeration | `lspci -d 15b3:` (Mellanox vendor ID); the documented form lives in the public BlueField / DPU User Manual | [`## configure`](#configure) step 5; [`## debug`](#debug) layer 3 | Lists the BlueField PF (and any VFs / SFs the operator expects); BDFs match what the binary's launch invocation will quote. |
| Device + representor enumeration | `devlink dev show` ; `ip link show` ; `cat /sys/class/net/*/phys_port_name` | [`## configure`](#configure) step 3 + step 5; [`## debug`](#debug) layer 3 | Lists the expected BlueField devices, netdevs, and representors; representor naming matches the BlueField OS image's documented convention. |
| NUMA topology | `numactl --hardware` ; `lscpu` | [`## configure`](#configure) step 5; [`### isolation`](#isolation) step 3 | Identifies the NUMA node owning the BlueField's PCIe root complex; CPU IDs per NUMA node match what the launch invocation will pin to. |
| CPU / NUMA pinning at launch | `numactl --cpunodebind=<node> --membind=<node> -- <binary> [args]` ; OR `taskset -c <cpu-list> <binary> [args]` (per `numactl(8)` / `taskset(1)`) | [`## run`](#run) step 2 + step 3; [`### isolation`](#isolation) step 3 | The launched process's `numactl --show -p <pid>` / `taskset -p <pid>` matches the planned set. |
| IRQ affinity inspection | `cat /proc/irq/<n>/smp_affinity` for the IRQ(s) the BlueField NIC owns (per the public BlueField / DPU User Manual) | [`## configure`](#configure) step 5; [`## debug`](#debug) layer 7 | IRQ mask matches the planned CPU set on the same NUMA node. |
| Hugepage state | `mount \| grep huge` ; `cat /proc/meminfo \| grep -i huge` | [`## configure`](#configure) step 3; [`## debug`](#debug) layer 2 + layer 5 | hugetlbfs is mounted; `HugePages_Total` > 0; `HugePages_Free` > 0 even after the binary attaches. |
| Process output (direct launch) | The terminal the operator launched from | [`## run`](#run) step 4; [`## debug`](#debug) layers 2-5 | Documented bring-up lines from the public DOCA Programming Guide appear; no documented error lines repeat. |
| Process output (tmux / screen) | `tmux attach -t <session>` ; `screen -r <session>` (per the operator's session name) | [`## run`](#run) step 4; [`## debug`](#debug) layers 2-5 | Same as direct, read from the reattached pane buffer. |
| Process output (systemd-supervised) | `systemctl status <unit>` ; `journalctl -u <unit> -f` (per the operator's unit name) | [`## run`](#run) step 4; [`## debug`](#debug) layers 2-6 | Unit is `active (running)`; journald tail shows the documented bring-up lines; restart count is stable. |
| Firmware query (READ-ONLY) | The documented BlueField firmware-query command (`mlxconfig -d <pci> query`) per the public BlueField / DPU User Manual — QUERY only; `mlxconfig set` is a hardware-state change owned by [`doca-hardware-safety`](../doca-hardware-safety/SKILL.md) | [`## configure`](#configure) step 7; [`## debug`](#debug) layer 8 | Reports the firmware version anchor used by [`CAPABILITIES.md ## Version compatibility`](CAPABILITIES.md#version-compatibility). |
| cgroup-v2 budget inspection | `systemd-cgls` ; `cat /sys/fs/cgroup/<path>/cpu.max` ; `cat /sys/fs/cgroup/<path>/memory.stat` (per the kernel cgroup-v2 documentation) | [`### isolation`](#isolation) step 5; [`## debug`](#debug) layer 5 + layer 7 | Per-tenant cgroup limits match the planned budget; `memory.stat` does not show OOM evidence. |
| Network-namespace inspection | `ip netns list` ; `ip netns exec <name> ip link show` (per `ip-netns(8)`) | [`### isolation`](#isolation) step 2 + step 5; [`## debug`](#debug) layer 3 | The per-tenant representor is in the netns the binary launches into; the root netns does not own a representor a netns-scoped binary is trying to reach. |
| Version anchor — host DOCA install | `pkg-config --modversion doca-common` ; `doca_caps --version` on the active install | [`## configure`](#configure) step 7; [`## debug`](#debug) layer 4 | The two strings match each other and match the binary's link-time `pkg-config doca-*` capture from [`doca-programming-guide ## build`](../doca-programming-guide/TASKS.md#build). |
| Version anchor — BlueField firmware | The documented firmware-query command on the BlueField (per the BlueField / DPU User Manual) | [`## configure`](#configure) step 7; [`## debug`](#debug) layer 4 | Reports a firmware version the public DOCA Programming Guide certifies for the operator's DOCA release. |

Three cross-cutting rules for this appendix:

- **Never invent a PCI BDF, a representor name, a NUMA node
  number, a hugepage allocation amount, or a `Restart=` mode
  name.** The public DOCA Programming Guide, the public
  BlueField / DPU User Manual, the Linux man pages, and live
  `lspci` / `numactl --hardware` / `devlink dev show` output
  on the target host are the contract; prose-derived strings
  are the most common hallucination failure for this skill.
- **Process output before device state before per-tenant
  resource picture.** When triaging, read the binary's own
  stdout / stderr / journald surface first (did the binary
  parse its args and reach DOCA?); only then read the
  device-state surface (`devlink` / `ip link` / `mlxconfig
  query`); only then read the per-tenant resource surface
  (cgroup-v2 / `numactl --show`). Reading the per-tenant
  resource picture against a binary that never reached its
  device-open call is meaningless.
- **Cross-link instead of duplicate.** Cross-cutting env
  commands (`pkg-config --modversion`, `doca_caps --version`,
  `mount | grep huge`, `lsmod | grep mlx5`) live in
  [`doca-setup TASKS.md ## Command appendix`](../doca-setup/TASKS.md#command-appendix);
  cross-cutting debug commands (`gdb`, `valgrind`, `strace`,
  `--sdk-log-level`, the `doca-<lib>-trace` build flavor) live
  in
  [`doca-debug TASKS.md ## debug`](../doca-debug/TASKS.md#debug);
  this appendix names only the bare-metal-deployment-specific
  ones.

## Deferred task verbs

- **Container-path deployment of any DOCA service container** on
  the BlueField (kubelet-standalone mode, static-pod manifests
  directory, pod-spec YAML, image-pull from NGC) — out of scope
  here. Route to
  [`doca-container-deployment`](../doca-container-deployment/SKILL.md)
  for the sibling deployment path.
- **Full-Kubernetes-cluster operations** (cluster API,
  `kubectl`, `Deployment` / `Service` / `Ingress` objects,
  cluster-wide observability, CNI overlays across DPUs) — out
  of scope for this bundle per the AGENTS.md non-goals. For
  fleet-scale DPU provisioning, see the DOCA Platform
  Framework on GitHub via
  [`doca-public-knowledge-map ## Public source code: GitHub`](../doca-public-knowledge-map/SKILL.md#public-source-code-github).
- **Library-API questions** (constructing a DOCA-Flow pipe, an
  RDMA queue pair, a Comch channel; interpreting a specific
  `DOCA_ERROR_*` code; per-library capability discovery) — out
  of scope here. Route to the matching `libs/<library>` skill
  via the per-library entry under
  [`doca-public-knowledge-map ## Library- and module-specific guides`](../doca-public-knowledge-map/SKILL.md#library--and-module-specific-guides).
- **Env-preparation questions** (installing DOCA on a fresh
  host, mounting and reserving hugepages, setting
  `PKG_CONFIG_PATH` / `LD_LIBRARY_PATH`, flipping the eswitch
  mode, IOMMU posture) — out of scope here. Route to
  [`doca-setup ## configure`](../doca-setup/TASKS.md#configure)
  and [`doca-setup ## test`](../doca-setup/TASKS.md#test); the
  bare-metal launch assumes these are already done.
- **Hardware-state changes** (`mlxconfig set`, firmware burn,
  BlueField mode flip, kernel-boot-parameter changes for IOMMU
  or hugepage reservation) — the *change-application discipline*
  (preflight, OOB console, maintenance window, rollback) is
  meta-policy owned by
  [`doca-hardware-safety ## modify`](../doca-hardware-safety/TASKS.md#modify);
  this skill always loads `doca-hardware-safety` ALONGSIDE when
  a mutating step is on the table. The bare-metal **operational
  sequencing ladder** for BFB-install / RShim / TMFIFO / host PF
  rebind / post-BFB recovery — what evidence to collect in what
  order, which classifier state lattices to which recovery
  action — is **in scope** here, in
  [`## bluefield-lifecycle`](#bluefield-lifecycle); the meta-policy
  governing each mutating burn is still
  `doca-hardware-safety`'s.
- **Cross-library programming questions** (the canonical build
  pattern, the modify-a-sample first-app workflow, the
  cross-library `DOCA_ERROR_*` taxonomy, the program-side
  debug order, the trace-build flavor rationale) — out of
  scope here. Route to
  [`doca-programming-guide`](../doca-programming-guide/SKILL.md)
  for the cross-library shape and to the matching
  `libs/<library>` skill for the per-library overlay.
- **Building the binary the operator is deploying** — out of
  scope here. Route to
  [`doca-programming-guide ## build`](../doca-programming-guide/TASKS.md#build)
  for the canonical pattern (C/C++ + non-C language tracks).
  Once the binary is built, control returns here at
  [`## configure`](#configure) step 2.

## Cross-cutting

- The public DOCA Programming Guide is the single source of
  truth for the per-library device-open / capability / runtime
  contract that any DOCA-linked binary follows. The public
  BlueField / DPU User Manual is the single source of truth
  for the documented device-enumeration surface, representor
  naming, IRQ-affinity guidance, and firmware-query surface
  on the BlueField side.
- Two host modes (host x86, BlueField Arm bare-metal) and
  three launch modes (direct, tmux / screen, systemd-
  supervised) compose freely; the agent's job is to walk the
  combination the operator picks, not to push a default mode.
- Hardware-resource binding (PCI BDF, NUMA node, CPU set, IRQ
  affinity) is derived from LIVE output on the target host,
  never substituted from memory. The most common first-run
  failure is a stale BDF or NUMA number quoted from a previous
  deploy.
- Smoke before bulk. The trivial-arg → liveness-equivalent →
  trivial-workload smoke loop runs BEFORE any real workload,
  never after.
- Failed bare-metal process restart is HIGH-STAKES. A binary
  in an auto-restart loop is delayed diagnosis, not
  resilience; stop the supervisor, clear the root cause, only
  then re-enable.
- Per-tenant isolation is set up BEFORE the workload starts,
  not after a co-tenant complains. The standard Linux
  primitives apply; the DOCA-specific gotchas (NUMA-locality,
  hugepage accounting, representor-vs-namespace) are this
  skill's load-bearing contribution.
- For URL routing to the public DOCA Programming Guide, the
  public BlueField / DPU User Manual, and the per-library
  guides, see
  [`doca-public-knowledge-map ## Public documentation entry points`](../doca-public-knowledge-map/SKILL.md#public-documentation-entry-points)
  and
  [`doca-public-knowledge-map ## Library- and module-specific guides`](../doca-public-knowledge-map/SKILL.md#library--and-module-specific-guides).
