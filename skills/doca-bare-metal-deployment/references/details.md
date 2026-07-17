# doca-bare-metal-deployment — reference detail

Moved out of `SKILL.md` to keep the loader under the per-file size budget. This is supporting detail, not routing logic.

## Example questions this skill answers well

The CLASSES of bare-metal-deployment questions this skill is built
to answer, each with one worked example. The class is the
load-bearing piece; the worked example is one instance.

- **"I have a DOCA-linked binary I built. What does it actually
  take to run it correctly on real hardware — not inside a
  container?"** — worked example: *"I built a doca-flow
  application on my host with a BlueField-3 in the PCIe slot; how
  do I launch it the right way?"*. Answered by the pattern
  overview + launch-mode table in
  [`CAPABILITIES.md ## Pattern overview`](../CAPABILITIES.md#pattern-overview)
  + the step-by-step launch walkthrough in
  [`TASKS.md ## run`](../TASKS.md#run).
- **"I want to run my binary on the BlueField Arm cores
  themselves, not on the x86 host. Is that the same workflow or a
  different one?"** — worked example: *"the BlueField OS image
  has DOCA installed on the Arm side; I'd like to run my DOCA app
  directly on the DPU, talking to its local NIC"*. Answered by
  the two-host-modes contract in
  [`CAPABILITIES.md ## Capabilities and modes`](../CAPABILITIES.md#capabilities-and-modes)
  + the parallel walkthrough in
  [`TASKS.md ## configure`](../TASKS.md#configure) and
  [`TASKS.md ## run`](../TASKS.md#run).
- **"Should I just `./my-doca-app &` it, run it in tmux, or wire a
  systemd unit?"** — worked example: *"I want this binary to come
  back automatically after a host reboot, but I also want to be
  able to attach to it and see what it is doing right now"*.
  Answered by the three-launch-modes decision table in
  [`CAPABILITIES.md ## Capabilities and modes`](../CAPABILITIES.md#capabilities-and-modes)
  + the launch-mode-selection step in
  [`TASKS.md ## configure`](../TASKS.md#configure).
- **"How do I bind my DOCA process to the right PCIe function and
  the right NUMA node so it doesn't trip over itself?"** — worked
  example: *"the BlueField is on NUMA node 1; my app is being
  scheduled on cores from node 0 and performance is terrible"*.
  Answered by the hardware-binding rules in
  [`CAPABILITIES.md ## Capabilities and modes`](../CAPABILITIES.md#capabilities-and-modes)
  + the
  [`### isolation`](../TASKS.md#isolation) sub-anchor under
  [`## run`](../TASKS.md#run) (cgroup-v2 / namespaces / numactl per-tenant
  primitives).
- **"My binary won't start; or it starts but exits immediately; or
  it starts but can't see the device. How do I diagnose this
  without guessing?"** — worked example: *"my doca-flow binary
  exits with status 1 within a second of launch; I have no idea
  which layer broke"*. Answered by the seven-layer error taxonomy
  in
  [`CAPABILITIES.md ## Error taxonomy`](../CAPABILITIES.md#error-taxonomy)
  + the matching layered ladder in
  [`TASKS.md ## debug`](../TASKS.md#debug).
- **"systemd put my DOCA binary in a `Restart=always` loop because
  it keeps crashing. Should I let it keep restarting, or is that
  exactly the wrong thing?"** — worked example: *"the unit is
  auto-restarting my binary every five seconds and the device is
  reporting odd errors; should I just bump the restart limit?"*.
  Answered by the restart-loop-is-HIGH-STAKES rule in
  [`CAPABILITIES.md ## Safety policy`](../CAPABILITIES.md#safety-policy)
  + the *"clear the root cause first"* layer in
  [`TASKS.md ## debug`](../TASKS.md#debug).
- **"Two of my colleagues are running DOCA processes on the same
  BlueField. How do I make sure their workload doesn't crush
  mine?"** — worked example: *"I want one DOCA-Flow process per
  representor, one DOCA-RDMA process for the storage path, all on
  the same BlueField, without cross-tenant interference"*.
  Answered by the per-tenant isolation rules in
  [`CAPABILITIES.md ## Capabilities and modes`](../CAPABILITIES.md#capabilities-and-modes)
  + the [`### isolation`](../TASKS.md#isolation) sub-anchor.
- **"My host is fine and the BlueField was working last week, but
  after a BFB push it never came back. `bfb-install` exited 0,
  but I cannot ssh to the BF, `ping 192.168.100.2` works but
  feels wrong, and `ip link` doesn't show any BlueField netdev on
  the host any more."** — worked example: *"DOCA 3.3 host
  upgrade is fine; BFB install on the BlueField reported `Ubuntu
  installation completed` then `INFO[MISC]: NIC firmware update
  failed`, but `bfb-install` still exited 0; now the DPU never
  reaches `DPU is ready`, host PFs are present in `lspci -d 15b3:`
  but `ip link` doesn't list their netdevs."* Answered by the
  BlueField lifecycle anchor in
  [`TASKS.md ## bluefield-lifecycle`](../TASKS.md#bluefield-lifecycle)
  (the `bfb-install` partial-failure recognition + the
  `192.168.100.2` host-loopback `ip route get` gotcha + the host
  PF rebind sequence + the post-BFB four-way version-match
  re-close) and the six-state classifier in
  [`### bluefield-state-classifier`](../TASKS.md#bluefield-state-classifier).

## What this skill deliberately does not ship

This skill is **agent guidance**, not a templates / sample-binaries /
sample-units bundle. To keep the boundary clean, it deliberately
does not contain — and pull requests should not add:

- **Pre-baked binaries.** No DOCA application binary, no sample
  ELF, no statically-linked test program is shipped with this
  skill. The canonical artifact is the user's own DOCA-linked
  binary, built per
  [`doca-programming-guide ## build`](../../doca-programming-guide/TASKS.md#build).
- **Sample systemd units, sample `numactl` invocations, sample
  `taskset` invocations, or any other ready-to-copy launch
  recipe.** Bare-metal launch is deployment-specific (per-host
  PCI BDF, per-host NUMA topology, per-tenant CPU set, per-site
  systemd policy) and the safe answer for an external operator
  is to *derive* the launch recipe from the public DOCA
  Programming Guide and the public BlueField / DPU User Manual
  against their own target. The agent's job is to prescribe the
  *procedure* and quote the documented command shapes from
  [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md),
  not to ship a `.service` file or a `numactl --cpunodebind=...`
  line the user might run unmodified.
- **PCI addresses, NUMA node numbers, representor names, devlink
  paths, hugepage allocation amounts, or systemd `Restart=` mode
  names invented from generic Linux knowledge.** The public DOCA
  Programming Guide, the public BlueField / DPU User Manual, the
  Linux man pages (`numactl(8)`, `taskset(1)`, `systemd.service(5)`,
  `systemd.unit(5)`), and `--help` on the installed tool are the
  authoritative sources. Inventing a `0000:01:00.0` or a
  `Restart=on-failure-with-burst-cap` from memory is the
  load-bearing first-run failure for this skill.
- **A `samples/`, `templates/`, `units/`, or `reference/` subtree
  of any kind.** A mock or incomplete artifact in this skill's
  tree, even one labeled "reference", is misleading: operators
  will read it as production-ready.

## Related skills

- [`doca-container-deployment`](../../doca-container-deployment/SKILL.md)
  — the SIBLING path. Two parallel deployment shapes in this
  bundle: containers (that skill) vs bare metal (this one). The
  recognition step that picks between them lives in
  [`doca-setup`](../../doca-setup/SKILL.md). Once the shape is
  *bare metal*, the agent stays here; if it is *container*, the
  agent routes there.
- [`doca-setup`](../../doca-setup/SKILL.md) — env preparation
  (install verification, hugepages mount and reservation, IOMMU
  posture, devlink mode, pkg-config path, representor visibility,
  kernel module load state). This skill assumes its preconditions
  are satisfied at the bare-metal target. The recognition step
  that decides container-vs-bare-metal is in `doca-setup` per
  the bundle convention; load `doca-setup` in parallel when the
  user's situation is ambiguous.
- [`doca-hardware-safety`](../../doca-hardware-safety/SKILL.md) —
  the cross-cutting meta-policy for any change touching DPU / NIC
  hardware state. This skill's `## Safety policy` overlays that
  meta-policy with bare-metal-specific rules
  (smoke-before-bulk-for-binaries, restart-loop-is-HIGH-STAKES,
  do-not-invent-PCI-addresses-or-NUMA-numbers-from-memory) and
  does **not** redefine the meta-policy itself. When the change
  the agent is about to recommend writes `mlxconfig`, burns
  firmware, reflashes the BFB, flips the BlueField mode, or
  changes a kernel boot parameter, the agent leaves this skill
  for `doca-hardware-safety` and only returns once the
  hardware-state change is complete.
- [`doca-debug`](../../doca-debug/SKILL.md) — the cross-cutting
  layered debug ladder (install / version / build / link /
  runtime / program / driver). Bare-metal-deployment-specific
  debug (process didn't start, started and exited, couldn't find
  the device, OOM / signal, restart loop, co-tenant noise)
  layers on top of the cross-cutting ladder; this skill's
  `## debug` cross-links into `doca-debug` for the broader
  context.
- [`doca-programming-guide`](../../doca-programming-guide/SKILL.md)
  — canonical DOCA build / modify / first-app patterns and the
  cross-library `DOCA_ERROR_*` taxonomy. This skill assumes the
  user already has a built binary; questions about *building*
  the binary or interpreting library-specific errors route there.
- [`doca-version`](../../doca-version/SKILL.md) — the four-way
  version match rule (host package ↔ binary build ↔ BlueField
  firmware ↔ DOCA-version policy). This skill's
  `## Version compatibility` cross-links the body of the rule
  there and adds only the bare-metal-specific overlay (the
  binary's link-time `pkg-config doca-*` version must match the
  runtime `LD_LIBRARY_PATH`'d install).
- [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md)
  — the routing table to the public DOCA Programming Guide, the
  public BlueField / DPU User Manual, the public Installation
  Guide, and the NGC catalog. This skill does not duplicate
  URLs; it points at the map and adds the bare-metal-deployment
  overlay.
- [`doca-structured-tools-contract`](../../doca-structured-tools-contract/SKILL.md)
  — the bundle's structured-tools precedence rule
  (detect / prefer / fall back / report). The
  [`## Command appendix`](../TASKS.md#command-appendix) in
  [`TASKS.md`](../TASKS.md) honors this contract — the agent probes
  for the matching structured helper first (`doca-env --json`,
  `doca-capability-snapshot`, `version-matrix.json`) and falls
  back to the documented manual commands when the probe fails.

