# DOCA hardware safety — capabilities, version compatibility, errors, observability, safety

**Where to start:** This file is the meta-policy surface. Pick the H2
anchor that matches the question (what changes are in scope / how
version pairing factors in / which failure modes the policy prevents
/ what the observability gate looks like / what the cross-cutting
safety meta-policy is) and read that section end-to-end. The tables
in each section are the load-bearing content; the prose is
interpretation.

Read this file when the loader sent you here from
[SKILL.md](SKILL.md). For the *how* of executing the change-application
discipline, jump to [TASKS.md](TASKS.md). For the JSON schemas of the
structured helpers the agent prefers when present, see
[`doca-structured-tools-contract`](../doca-structured-tools-contract/SKILL.md).

## Pattern overview

Every hardware-safety question this skill teaches resolves into one
of SIX patterns. The patterns are CLASSES — they apply across every
DOCA release, every BlueField generation, and every hardware-touching
change the per-artifact skills recommend.

| Hardware-safety pattern | Class shape | Where the substance lives |
| --- | --- | --- |
| 1. Identify the change class | Map the recommended action (firmware-slot write, mode flip, BFB reflash, kernel-boot-parameter change, link state change, PCIe rescan, BlueField cold reboot) to its in-scope change class | [`## Capabilities and modes`](#capabilities-and-modes) change-class table |
| 2. Take the pre-flight inventory | Capture the as-deployed PCIe topology, link state, firmware level, BFB level, and config snapshot BEFORE any change, so rollback is even possible | [`## Capabilities and modes`](#capabilities-and-modes) inventory taxonomy + [TASKS.md ## configure](TASKS.md#configure) |
| 3. Establish out-of-band access | When the change can break the link the operator is managing the BlueField over, require an out-of-band path BEFORE the change is issued; refuse otherwise | [`## Safety policy`](#safety-policy) out-of-band rule + [TASKS.md ## configure](TASKS.md#configure) |
| 4. Frame the maintenance window | Every hardware-touching change runs inside an explicit, time-boxed maintenance window with ops notified; the agent does not silently apply during business hours | [`## Safety policy`](#safety-policy) maintenance-window rule |
| 5. Apply, verify, observe | Apply the change with the right post-change action (cold power cycle for `mlxconfig`-class, host reboot for kernel boot parameters, BlueField cold reboot for BFB reflash), then verify against the captured baseline before any workload moves back | [`## Observability`](#observability) post-change gate + [TASKS.md ## modify](TASKS.md#modify) + [TASKS.md ## run](TASKS.md#run) |
| 6. Roll back or escalate | Every change has a documented, rehearsed rollback path; changes without a documented rollback are refused and escalated rather than guessed | [`## Safety policy`](#safety-policy) rollback rule + [TASKS.md ## debug](TASKS.md#debug) |

Two cross-cutting rules that apply to *every* pattern above:

- **Pre-flight inventory is a hard prerequisite.** Rollback is only
  possible against a captured baseline. The agent does not begin any
  hardware-touching change before the inventory in
  [`## Capabilities and modes`](#capabilities-and-modes) is recorded
  for the session, regardless of how small the change appears.
- **No documented rollback → refuse and escalate.** The class-shape
  escape valve in [`## Safety policy`](#safety-policy). Changes that
  the operator cannot describe a rollback for are not applied through
  this skill's discipline; the agent surfaces the missing rollback as
  the blocking issue and routes to the operator's change-control
  process.

## Capabilities and modes

### The class of changes in scope

This skill governs the change-application discipline for the
following classes of action. The class names are the load-bearing
piece; specific parameter names, version literals, PCI addresses, or
kernel-command literals are *not* in scope and never appear in this
skill — they live in the matching per-artifact skill or in the
operator's own runbook.

| Change class | What it touches | Why this skill governs it |
| --- | --- | --- |
| **`mlxconfig`-class firmware-level parameter write** | Firmware-stored configuration on the NIC / DPU (BAR window, BlueField NIC ↔ DPU mode toggle, SR-IOV count, device-emulation slot enablement). | Writes are committed at next cold power cycle; the link layer, PCIe surface, and host-visible function set can all shift on the next boot. The agent must surface this so the operator does not assume a warm reboot is enough. |
| **NIC firmware burn** | The NIC's firmware image. | Interrupts every device function on the NIC during the burn; an interruption mid-burn is the canonical bricking failure mode. |
| **BlueField BFB reflash** | The BlueField's OS image and DOCA install. | Every BlueField-hosted service container is removed; the host's view of the BlueField PCIe surface may change; the BFB-side version anchor moves. |
| **BlueField mode flip (NIC ↔ DPU)** | The BlueField's operating personality. | The host-visible representor topology, the DOCA install side that owns the dataplane, and the management surface all change shape. |
| **Kernel boot parameter change (IOMMU mode, hugepages, device pass-through)** | The host kernel's boot configuration. | Takes effect at host reboot, not at runtime. An installed runtime helper alone cannot make up for the wrong IOMMU mode or insufficient hugepages reserved at boot. |
| **PCIe link state down / up, PCIe rescan, function rebind** | The PCIe surface a workload is currently using. | The workload's PCIe function disappears mid-flight; representors backed by the function go away; in-flight queues stall or fail. |
| **BlueField cold reboot** | Everything the BlueField hosts. | Every container and service on the BlueField restarts; host workloads that depend on the DPU are interrupted. Often the necessary post-change step for the classes above. |

The agent's rule: **identify which class the recommended next step
falls into BEFORE issuing the change**. The per-artifact skill names
the action; this skill names the discipline that wraps it.

### Pre-flight inventory taxonomy

The pre-flight inventory is the captured baseline every rollback
depends on. The taxonomy is also a CLASS — specific commands belong
in the operator's runbook, not in this skill.

| Inventory dimension | What to capture | Why it is load-bearing for rollback |
| --- | --- | --- |
| PCIe topology | The current set of PFs, VFs, SFs, and representors as DOCA sees them, with their PCIe addresses and state | A change that moves the PCIe surface (mode flip, BAR change, BFB reflash) must reproduce the topology on the other side; without the baseline, the operator cannot tell what was lost |
| Link state | Per-port link up/down state and link speed | A change that drops a port flips the same field; the as-deployed state names what *should* come back |
| Running firmware level | The firmware version the NIC / DPU is running before the change | A firmware burn rolls forward this anchor; the rollback path is to reflash the prior level, which the agent cannot name without the baseline |
| BFB level (BlueField only) | The DOCA BFB version on the BlueField side | A BFB reflash moves this anchor; the rollback is to reflash the prior BFB, which the agent cannot name without the baseline |
| Configuration snapshot | The current `mlxconfig`-class parameters and any per-artifact service config the change is about to touch | A change to firmware-stored config rolls forward this anchor; the rollback path is to write the prior values, which the agent cannot reconstruct without the baseline |
| Host-side env snapshot | The host's kernel command line (IOMMU mode, hugepage reservations), the loaded `mlx5_*` modules, and the host kernel version | A kernel-boot-parameter change rolls forward this anchor; the rollback path is to revert the boot configuration to the captured baseline |

For the structured one-shot form of this inventory, see the
`collect-host-state` and `collect-dpu-state` schemas in
[`doca-structured-tools-contract`](../doca-structured-tools-contract/SKILL.md);
the agent prefers the structured tool when present and falls back to
the manual chain documented there otherwise.

## Out-of-band access classes

Several change classes can break the link the operator is using to
manage the BlueField. For those, the agent requires an out-of-band
(OOB) management path to be in place BEFORE the change is issued. The
specific OOB endpoints (which BMC IP, which TTY device) are operator-
specific and never invented by the agent — the load-bearing piece is
the class.

| OOB class | When this OOB class is appropriate | Operator responsibility |
| --- | --- | --- |
| BMC serial console | A BMC is present on the host or the BlueField management board and exposes a serial console the operator can reach independently of the data-plane link | The operator knows the BMC's reachable address (or local KVM path) and has confirmed access before the change |
| RShim USB / RShim-over-PCIe console | The operator has physical or PCIe-level access to the BlueField's RShim management interface | The operator has confirmed RShim is reachable and the console is responsive before the change |
| Operator-managed console (KVM, terminal server, OCC-attached console) | The deployment includes a dedicated console path the operator controls | The operator names the path and has confirmed it works before the change |

The agent's rule: when the recommended change is in the *link-
breaking* class (mode flip, BAR change, port reassignment, BFB
reflash, NIC mode change), refuse to issue it until the operator
names which OOB class is in place. *"I'll just do it over SSH"* is
the canonical failure mode; the change drops the SSH session and the
operator cannot recover.

### Apply / verify / observe shape

Each change class has a documented *commit* action. The class shapes:

| Change class | Commit action | What "applied" means |
| --- | --- | --- |
| `mlxconfig`-class parameter write | Cold power cycle of the NIC / DPU (full A/C power removal) | A warm reboot does NOT commit firmware-stored configuration. The agent does not declare the change applied after a warm reboot. |
| NIC firmware burn | The burn tool's documented completion handshake plus a power cycle if the tool requires it | The agent quotes the tool's own documented completion criterion; never invents a "looks done" judgment from log noise. |
| BFB reflash | The BFB's documented post-flash boot + BlueField cold reboot | The BlueField comes up on the new BFB; every hosted service container has to be re-deployed per [`doca-container-deployment ## run`](../doca-container-deployment/TASKS.md#run). |
| BlueField mode flip | Cold power cycle + post-reboot verification of the new personality (per the matching per-artifact skill) | Confirm the host-visible representor topology matches the post-flip plan before any workload moves. |
| Kernel boot parameter change | A host reboot. An installed DOCA runtime helper alone does NOT make up for a wrong IOMMU mode or for hugepages that were not reserved at boot. | The agent does not promise that a host-side runtime helper substitutes for the boot-time parameter. Cross-link to [`doca-setup`](../doca-setup/SKILL.md) for env-class verification of the post-reboot state. |
| Link state down / up, PCIe rescan | The operator's documented procedure for the affected function | The agent surfaces that any workload currently using the function will see the function disappear; the verification gate is that the function comes back in the expected state. |

### Replica-first validation

Every hardware-touching change is first applied on a non-prod replica
with the same hardware class (same BlueField generation, same
firmware band, same host kernel). The smoke test from the per-
artifact skill runs on the replica before the production change. The
agent does not skip the replica step *"because this is a small
change"* — the replica is the cheapest signal that the change does
what the operator expects on the actual hardware class, and it is the
last opportunity to discover that the rollback path does not work
before production is exposed to that risk.

The replica's hardware class must match production on:

- The BlueField generation (per
  [`doca-version`](../doca-version/SKILL.md), the BFB anchor must be
  in the same compatibility band).
- The running firmware level on the NIC / DPU.
- The host kernel version and the loaded `mlx5_*` modules.
- The set of representors / VFs / SFs that the change touches.

Differences against production along any of those axes mean the
replica is *not* a representative test of the change; the agent
surfaces the difference and treats the replica result as advisory
rather than gating.

## Version compatibility

For the canonical DOCA version-detection chain, the four-way match
rule, NGC container semantics, and the headers-win-over-docs rule,
see [`doca-version`](../doca-version/SKILL.md). The body lives there;
this skill does not duplicate it.

**The hardware-safety overlay** is:

- **Every hardware-touching change has a version dimension.** A
  firmware burn moves the running firmware level; a BFB reflash moves
  the BlueField BFB version; an `mlxconfig`-class write changes
  firmware-stored configuration that the BFB and the host packages
  jointly interpret. The pre-flight inventory captures every version
  anchor named by `doca-version`'s four-way match plus the firmware
  level and the BFB level, so the rollback path can quote them.
- **The version anchor is jointly bound by host package, BFB,
  firmware, and (when applicable) the per-artifact container tag.**
  After the change, the agent re-runs the four-way match per
  [`doca-version TASKS.md ## test`](../doca-version/TASKS.md#test); a
  change that leaves the four-way match in a partial state is a
  failed change and the rollback path applies.
- **Never quote a firmware version, BFB version, or DOCA version
  literal from agent memory.** The version handling rules in
  [`doca-version CAPABILITIES.md ## Safety policy`](../doca-version/CAPABILITIES.md#safety-policy)
  apply here without modification; the agent quotes the version
  observed from the operator's host, not from agent memory or from a
  public docs URL.

## Error taxonomy

The hardware-safety surface is anti-bricking, anti-silent-mismatch,
anti-discovered-during-failure-rollback. The failure modes below are
what the meta-policy exists to prevent. Each row pairs a failure
mode with the policy rule that prevents it.

| Failure mode | Class shape | Policy rule that prevents it |
| --- | --- | --- |
| Bricked link (the change drops the management link and the operator cannot recover) | A link-breaking change was applied without an out-of-band path | The OOB-precondition rule in [`## Safety policy`](#safety-policy) and the OOB classes in [`## Capabilities and modes`](#capabilities-and-modes) |
| Runaway firmware burn (a burn is interrupted mid-flight; the device is left in an indeterminate state) | A firmware burn ran outside a documented maintenance window, or against a host that lost power continuity | The maintenance-window rule + the firmware-burn precondition list in [`## Safety policy`](#safety-policy) and [TASKS.md ## modify](TASKS.md#modify) |
| Silent mode change (an `mlxconfig` write happens; a warm reboot occurs; the agent declares the change applied; the firmware-stored configuration in fact did not commit) | The operator was not told that `mlxconfig`-class writes commit at cold power cycle, not at warm reboot | The `mlxconfig`-class commit-action row in [`## Capabilities and modes`](#capabilities-and-modes) and the apply-with-cold-power-cycle workflow in [TASKS.md ## modify](TASKS.md#modify) |
| Missing rollback (the change is applied; it fails; the operator and the agent cannot agree on what state the system is supposed to revert to) | The pre-flight inventory was not captured, OR the change was applied without a documented rollback path | The pre-flight inventory rule + the rollback-must-be-documented rule in [`## Safety policy`](#safety-policy) |
| Discovered-during-failure rollback (the rollback path is invented mid-incident; the invented rollback does not work) | The rollback was not rehearsed on the replica before production was exposed to the change | The replica-first rule in [`## Capabilities and modes`](#capabilities-and-modes) and the replica-smoke step in [TASKS.md ## test](TASKS.md#test) |
| IOMMU / hugepages drift (host-side runtime helper is installed; the operator believes the env is configured; the change still fails because the IOMMU mode or hugepage reservation is wrong at boot) | A kernel-boot-parameter change was treated as if a runtime helper could substitute for it | The kernel-boot-parameter commit-action row in [`## Capabilities and modes`](#capabilities-and-modes); cross-link to [`doca-setup`](../doca-setup/SKILL.md) for env-class verification |
| Version regression (the change is applied; the four-way match no longer holds; the agent does not notice because no post-change verification ran) | The post-change verification gate in [`## Observability`](#observability) was skipped | The verification-before-workload rule in [`## Safety policy`](#safety-policy) and the post-change four-way match in [TASKS.md ## run](TASKS.md#run) |
| Workload-during-burn (a firmware burn or BFB reflash is started while real traffic is on the wire; the burn interrupts the workload, or the workload interrupts the burn) | The do-not-interrupt rule was not surfaced | The firmware-burn precondition list in [`## Safety policy`](#safety-policy) |

For DOCA-program-level errors that surface AFTER the rollback has
restored a known state (e.g. `DOCA_ERROR_DRIVER`, `DOCA_ERROR_NOT_SUPPORTED`),
hand off to
[`doca-debug TASKS.md ## debug`](../doca-debug/TASKS.md#debug) — the
hardware-safety ladder owns the change-application discipline; the
program-side debug ladder owns what comes after.

## Observability

The hardware-safety observability surface is the set of signals the
operator must be watching before, during, and after a hardware-
touching change. The signals are CLASSES — specific command names
belong in the operator's runbook or in the per-artifact skill's
`## Observability` overlay.

| Phase | Signal class | What it tells the operator |
| --- | --- | --- |
| Pre-change | Pre-flight inventory (PCIe topology + link state + firmware level + BFB level + config snapshot + host-side env snapshot) | The captured baseline every later check is compared against. Without this, the agent cannot detect a regression after the change. |
| Pre-change | Out-of-band reachability (the OOB class identified in [`## Capabilities and modes`](#capabilities-and-modes) is confirmed responsive) | The recovery path is in place before the operator issues a link-breaking change. |
| Pre-change | Workload state (is the affected NIC / DPU / port currently carrying traffic? Are workloads using representors on the affected function?) | The set of workloads that need to be drained or scheduled around the change. |
| During change | The commit tool's own output (firmware burn progress, BFB reflash progress, `mlxconfig` apply confirmation) | The documented signal of the change's progress. The agent quotes the tool's documented signal verbatim and never paraphrases it as "looks done". |
| Post-change | Re-captured inventory (the same set of signals as pre-flight) | Did the change land as expected? Specifically, did the PCIe topology / link state / firmware level / BFB level move to the post-change values the plan named? |
| Post-change | Post-change four-way match (per [`doca-version TASKS.md ## test`](../doca-version/TASKS.md#test)) | Is the install consistent on the new state? A change that breaks the four-way match is a failed change. |
| Post-change | Per-artifact health metric (the smoke test from the matching per-artifact skill) | Did the artifact this change was made for actually come back to a healthy state on its own observability surface? |
| Pre-workload | Observability path proven end-to-end (logs visible, counters readable, dmesg readable, container / service stdout reachable) | Can the operator see what the workload is doing once it moves? The agent does not declare the deployment "ready" until this gate passes. |

For the env-side observability primitives (PCIe scans, link
introspection, kernel-module presence), see
[`doca-setup CAPABILITIES.md ## Observability`](../doca-setup/CAPABILITIES.md#observability).
For the program-side cross-library observability (`--sdk-log-level`,
the `doca-<lib>-trace` build flavor, `DOCA_LOG_LEVEL`), see
[`doca-debug CAPABILITIES.md ## Observability`](../doca-debug/CAPABILITIES.md#observability).
This skill names the *gates*; the named skills carry the surface
itself.

## Safety policy

This skill **defines** the cross-cutting hardware-safety meta-policy
that every per-artifact `## Safety policy` overlays. The rules below
apply to every hardware-touching change in the bundle. Per-artifact
skills layer artifact-specific safety on top; they do not redefine
the meta-policy.

### Cross-cutting rules

- **Inventory before any change.** Capture the pre-flight inventory
  per [`## Capabilities and modes`](#capabilities-and-modes) before
  issuing any hardware-touching action. Rollback is only possible
  against a captured baseline; an agent that begins a change without
  the inventory is choosing to make rollback impossible. The
  inventory is a hard prerequisite, not a nice-to-have.
- **Out-of-band path is a precondition for link-breaking changes.**
  When the change is in the link-breaking class (mode flip, BAR
  change, port reassignment, BFB reflash, NIC mode change), require
  an OOB class per [`## Capabilities and modes`](#capabilities-and-modes)
  to be in place before issuing the change. Refuse the change if no
  OOB is named and reachable. Specific OOB endpoints belong to the
  operator; the agent never invents a BMC IP or a console TTY.
- **Maintenance window is the default frame.** Every hardware-
  touching change runs inside an explicit, time-boxed maintenance
  window with operations notified. The agent does not silently
  recommend applying a change "right now" during business hours. The
  agent names the rollback plan and the verification gate in the
  same breath as the change, not as an afterthought.
- **`mlxconfig`-class writes commit at cold power cycle.** The
  agent surfaces this rule explicitly whenever the recommended next
  step writes an `mlxconfig`-class parameter. *"`mlxconfig` ...
  then `reboot`"* is an anti-pattern: the warm reboot does not
  commit firmware-stored configuration. The agent does not declare
  the change applied until the cold power cycle has occurred and
  the post-change inventory confirms the new state.
- **Firmware burn is high-stakes.** A NIC firmware burn or BFB
  reflash requires (a) the pre-flight inventory captured (in
  particular, the running firmware / BFB level the rollback would
  need to reflash to), (b) confirmed power continuity (UPS or dual
  feed) for the duration of the burn, (c) the OOB console reserved,
  and (d) no ongoing workload on the affected NIC / DPU. An
  interrupted burn is the canonical bricking failure mode; the
  agent refuses to start the burn if any precondition is missing.
- **Kernel boot parameters require a host reboot.** Changes to
  IOMMU mode, hugepage reservations, or device pass-through take
  effect at host reboot. An installed runtime helper alone is not
  sufficient; the agent does not promise that a host-side helper
  substitutes for the boot-time configuration. The post-change gate
  is verified after the host reboot per
  [`doca-setup ## debug`](../doca-setup/TASKS.md#debug).
- **Replica-first validation.** Apply the change on a non-prod
  replica that matches production's hardware class first; run the
  per-artifact smoke on the replica; only then schedule the
  production change. The replica is the last opportunity to
  discover that the rollback path does not work before production
  is exposed to the risk; skipping the replica because the change
  "is small" is how a small change becomes a long incident.
- **Observability before workload.** The agent does not declare a
  deployment "ready" until the post-change observability path is
  proven end-to-end per [`## Observability`](#observability). A
  workload that moves before the observability surface is reachable
  is a workload moved blind; the next failure has no diagnostic
  signal to work from.
- **Rollback is documented and rehearsed.** Every change has an
  explicit rollback path (revert `mlxconfig` + cold power cycle;
  reflash the previously-captured firmware level; reflash the
  previously-captured BFB; revert the kernel boot parameter and
  reboot; remove the emulation slot; redeploy the previous
  container tag together with the matching DPU-side package). The
  rollback is documented BEFORE the change is applied and rehearsed
  on the replica BEFORE production is exposed. Discovered-during-
  failure rollback is too late.
- **No documented rollback → refuse and escalate.** Changes that
  the operator cannot describe a rollback for, or for which the
  vendor explicitly states no rollback exists (irreversible
  firmware rolls, board-level changes), are NOT applied through
  this skill's discipline. The agent surfaces the missing rollback
  as the blocking issue, refuses to proceed, and routes the
  operator to their change-control process or vendor escalation.
  Guessing at a rollback in the absence of one is a user-visible
  regression dressed up as helpfulness.

### Host-side cold power cycle — when it is indicated

A **host** cold power cycle (full AC removal, not a warm `reboot`)
is the right next action when any of these signals appear AFTER a
hardware-safety-class change on the host:

- An `mlxconfig` write succeeded but `mlxconfig -d <bdf> q` continues
  to show the prior value after a warm reboot — firmware-stored
  config commits only at cold cycle.
- A NIC firmware burn was reported successful but `flint -d <bdf> q`
  continues to show the previous FW version on a warm-rebooted host.
- After a host-side DOCA / OFED upgrade reboot, `lspci -d 15b3:`
  still lists devices but `devlink dev info` reports
  `Link has been severed` or kernel-side `mlx5_core` complains in
  `dmesg` about reset / probe failure that did not recur on the
  pre-upgrade kernel.
- Host PCIe bus enumeration is intermittently missing one or more
  ConnectX / BlueField PFs across warm reboots.

DPU-side cold-power-cycle criteria (RShim no-progress, BFB-completed-
but-no-`Linux up`, TMFIFO down, BF PFs visible but link severed) are
operationally similar but live in NVIDIA's BlueField BSP / DOCA
Platform Framework documentation, not this bundle. The agent routes
DPU-side recovery questions to those docs per
[`AGENTS.md ## Non-goals`](../../AGENTS.md#non-goals-questions-the-agent-should-recognize-and-refuse-politely)
item 7.

### Script hygiene for hardware-touching scripts the operator runs

When the answer to a hardware-safety scenario ends with the operator
running a shell script (preflight capture, upgrade wrapper, rollback
ladder), the agent's script-shaped guidance MUST cover three hygiene
rules. These are not aesthetic preferences — each closes a recurring
operational failure mode reported by real operators against earlier
bundle revisions:

- **Hand log ownership back to the invoking user.** Scripts that
  capture under `sudo` write logs as root; later non-sudo inspection
  by the same operator then fails with `Permission denied`. Every
  root-running script must end with `chown -R "${SUDO_USER:-$USER}":
  "${SUDO_USER:-$USER}" <log-dir>` (or the equivalent on the OS in
  scope) so the operator can read their own captures without a second
  sudo round-trip.
- **Probe shared-fd device readers BEFORE opening them.** Single-
  reader character devices (the canonical case is
  `/dev/rshim0/console` on a host with RShim, but the rule
  generalises to any device that supports only one open reader)
  silently fail to capture when another process already holds them.
  The agent's script must run `sudo fuser -v <device>` and surface
  the holder before opening the device for read, and SHOULD warn
  that opening the device under `cat` will starve other readers
  while the script runs.
- **Exit-zero is never the success signal alone.** An installer or
  burn tool that exits `0` while its own log contains
  `ERR[<class>]: ...` is NOT a successful run. Scripts must parse
  the captured log for the documented error-class prefix(es) the
  underlying tool emits and SHOULD return non-zero (or surface the
  finding under a distinct status code) when one is present. This is
  the universal-verification-contract `## step 5 — first-failure
  ladder` rule (see [`AGENTS.md ## The universal verification
  contract`](../../AGENTS.md#the-universal-verification-contract))
  applied to script-shaped outputs.

### Per-artifact overlay pattern

Per-artifact skills (services, libraries, tools) carry their own
`## Safety policy` anchor. Those anchors **overlay** this meta-
policy with artifact-specific safety; they do NOT redefine the
cross-cutting rules above. The convention is:

> A per-artifact skill's `## Safety policy` MUST cross-link to this
> skill for the cross-cutting hardware-safety rules and MUST add
> only the artifact-specific safety overlays (e.g. a service-
> specific firmware-slot precondition, a library-specific
> validate-before-commit requirement, a tool-specific
> read-only-by-default stance). It MUST NOT restate the cross-
> cutting rules.

Skills that overlay this meta-policy with artifact-specific
hardware safety are the in-bundle services, device-touching
libraries, and hardware-touching tools whose own `## Safety
policy` adds the per-artifact preconditions on top of the meta-
policy here. The canonical in-bundle overlays are:

- Services (kubelet-standalone container deployments that touch
  device state): [`doca-argus`](../services/doca-argus/SKILL.md),
  [`doca-dms`](../services/doca-dms/SKILL.md),
  [`doca-firefly`](../services/doca-firefly/SKILL.md),
  [`doca-urom-svc`](../services/doca-urom-svc/SKILL.md).
- Device-touching libraries:
  [`doca-flow`](../libs/doca-flow/SKILL.md),
  [`doca-rdma`](../libs/doca-rdma/SKILL.md),
  [`doca-eth`](../libs/doca-eth/SKILL.md),
  [`doca-pcc`](../libs/doca-pcc/SKILL.md),
  [`doca-rmax`](../libs/doca-rmax/SKILL.md).
- Hardware-touching tools:
  [`doca-spcx-cc`](../tools/doca-spcx-cc/SKILL.md),
  [`doca-pcc-counters`](../tools/doca-pcc-counters/SKILL.md),
  [`doca-flow-tune`](../tools/doca-flow-tune/SKILL.md).

Each of those is an overlay; this skill is the meta-policy they
share. **Externally-productized analogs** — `doca-virtio-net`,
`doca-snap`, `doca-hbn`, BlueMan, DPF — are NOT in-bundle skills;
their per-product safety policies live in the public product
documentation reached through
[`doca-public-knowledge-map ## Externally-productized DOCA software`](../doca-public-knowledge-map/SKILL.md#externally-productized-doca-software--not-in-this-bundle-but-here-is-where-to-route).
The meta-policy in this skill still applies operationally (OOB
console, preflight, maintenance window, rollback), but the
artifact-specific preconditions for those products are owned by
their public docs, not by this bundle.
