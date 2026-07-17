---
license: Apache-2.0
name: doca-hardware-safety
description: >
  Use this skill whenever the agent is about to recommend or apply a
  change that touches DPU / NIC hardware state on a live system —
  mlxconfig firmware-parameter write, NIC firmware burn, BFB reflash,
  NIC ↔ DPU mode flip, SR-IOV or device-emulation slot enable, kernel
  boot-parameter change (IOMMU, hugepages, VFIO), PCIe rebind /
  rescan / link-state flip, or BlueField cold reboot. Wraps the
  change in pre-flight inventory, OOB reachability, a maintenance
  window, the mlxconfig cold-power-cycle rule, replica rehearsal, and
  rollback. Trigger even when the user does not say "hardware safety"
  — implicit phrasings: "flip BlueField mode over SSH", "enable
  SR-IOV and reboot", "burned firmware but mlxconfig shows old
  value", "reflashed BFB and lost representors", "reflash during
  business hours", "vendor says this is one-way". Refuse for general
  DOCA orientation (doca-public-knowledge-map), install or env debug
  (doca-setup), and program-side debug (doca-debug,
  doca-programming-guide) — those belong to other skills.
metadata:
  kind: library
compatibility: >
  No DOCA install required to read this skill (it is an overlay
  loaded against any DOCA artifact skill); the validation steps
  within DO require a live DOCA install at /opt/mellanox/doca with a
  BlueField DPU or ConnectX NIC, plus out-of-band console
  reachability (BMC, RShim, or operator-managed console) for any
  link-breaking change.
---

# DOCA hardware safety

**Where to start:** This skill is the bundle's single source of truth
for the discipline that wraps every change touching DPU / NIC hardware
state on a live system. Open
[`TASKS.md`](TASKS.md) when the operator is about to *apply* a
hardware-touching change and needs the change-application discipline
(pre-flight inventory → out-of-band path → window → apply → verify
→ rollback). Open [`CAPABILITIES.md`](CAPABILITIES.md) when the
question is *what does hardware-safety even cover* (the class of
changes in scope, the failure modes the policy prevents, the
observability surface that gates a change, and the meta-policy that
every per-artifact `## Safety policy` overlays).

Every per-artifact skill (services, libraries, tools) in the bundle
that recommends a hardware-touching action overlays this meta-policy
with artifact-specific safety. The per-artifact `## Safety policy`
anchors do NOT redefine the cross-cutting discipline — they layer the
artifact's own concerns on top of it. This skill is the layer they all
build on.

## Example questions this skill answers well

The CLASSES of hardware-safety questions this skill is built to
answer, each with one worked example. The agent should treat the
*class* as load-bearing — the worked example is a single instance.

- **"I'm about to apply a hardware-touching change. What do I have to
  capture *before* I touch anything?"** — worked example: *"the
  per-artifact skill told me to flip a firmware-level emulation slot;
  what do I capture first?"*. Answered by the pre-flight inventory
  in [`TASKS.md ## configure`](TASKS.md#configure) plus the
  inventory taxonomy in
  [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes).
- **"This change might drop the link I'm using to manage the
  BlueField. Is that safe?"** — worked example: *"I'm about to flip
  the BlueField between NIC and DPU mode over the same management
  link"*. Answered by the out-of-band access rule in
  [`CAPABILITIES.md ## Safety policy`](CAPABILITIES.md#safety-policy)
  plus the OOB-precondition gate in
  [`TASKS.md ## configure`](TASKS.md#configure).
- **"The per-artifact skill said to write an `mlxconfig` parameter,
  then reboot. Is that the right sequence?"** — worked example:
  *"the storage-emulation skill told me to enable a firmware slot
  via `mlxconfig` and then warm-reboot to apply it"*. Answered by
  the `mlxconfig`-class rule in
  [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes)
  plus the apply-with-cold-power-cycle workflow in
  [`TASKS.md ## modify`](TASKS.md#modify).
- **"My deployment plan reflashes the BlueField BFB during business
  hours. Is that OK?"** — worked example: *"I have a one-hour
  window during the day; can I reflash now?"*. Answered by the
  maintenance-window discipline in
  [`CAPABILITIES.md ## Safety policy`](CAPABILITIES.md#safety-policy)
  plus the firmware-burn workflow in
  [`TASKS.md ## modify`](TASKS.md#modify).
- **"How do I prove the change works *before* I touch production?"**
  — worked example: *"the change is small; can I skip the lab
  replica"*. Answered by the replica-first rule in
  [`TASKS.md ## test`](TASKS.md#test) plus the
  pre-hardware-validation pattern in
  [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes).
- **"How do I roll back if this change goes wrong?"** — worked
  example: *"I just reflashed the BFB and the host can't see the
  representors anymore"*. Answered by the rollback ladder in
  [`TASKS.md ## debug`](TASKS.md#debug) plus the
  rollback-must-be-documented rule in
  [`CAPABILITIES.md ## Safety policy`](CAPABILITIES.md#safety-policy).
- **"This change doesn't have a documented rollback. Should I still
  apply it?"** — worked example: *"the vendor says this firmware
  rev is one-way"*. Answered by the refuse-and-escalate rule in
  [`CAPABILITIES.md ## Safety policy`](CAPABILITIES.md#safety-policy)
  plus the escalation path in [`TASKS.md ## debug`](TASKS.md#debug).

## When to load this skill

Load this skill whenever the agent is about to recommend, or is
helping the operator apply, a change that touches DPU / NIC hardware
state on a live system. The decision must be made **before** the
agent composes its first sentence — the activation checklist below
is the same one referenced from
[`AGENTS.md ## Cross-cutting overlay activation triggers`](../../AGENTS.md#cross-cutting-overlay-activation-triggers),
mirrored here so a per-artifact skill that already loaded this skill
has the activation rule at hand.

## Agent activation checklist — load this skill at the START of the answer when any cell below is true

| Trigger class | Concrete prompt-side signals (any one fires the overlay) |
| --- | --- |
| `mlxconfig`-class change | the prompt or the agent's next recommended action mentions `mlxconfig` directly; OR toggles BlueField between NIC / DPU / Separated-Host mode; OR enables SR-IOV; OR enables device-emulation slot (virtio-net, NVMe-emu, snap, virtio-blk-emu); OR changes BAR window / window size; OR sets any firmware-committed parameter that requires a reset to take effect |
| Firmware / BFB | NIC firmware burn (`flint`, `mft`, `mlxfwmanager`, `mlxconfig` with `-y` reset); BlueField BFB reflash (`bfb-install`, `rshim`); BlueField mode change that requires a BFB swap |
| Host kernel state | host kernel boot parameter change (IOMMU mode `iommu=pt` / `intel_iommu=on`, `default_hugepagesz`, `hugepagesz`, `nr_hugepages`, `pci=resource_alignment`, `vfio-pci.ids`); requires host reboot |
| Hugepages | hugepage *reservation* change (`/sys/kernel/mm/hugepages/*/nr_hugepages`, `sysctl vm.nr_hugepages`); hugepage *mount* change (`mount -t hugetlbfs`); this is global state shared with every DOCA / DPDK process on the host |
| PCIe state | PCIe rebind / `echo > /sys/bus/pci/.../{bind,unbind,remove,rescan}`; representor enable/disable; eswitch mode change (`devlink dev eswitch set ... mode {switchdev,legacy}`); link down/up on a port currently carrying traffic |
| BlueField reboot class | BlueField cold reboot, BlueField warm reboot to apply `mlxconfig`; any change whose blast radius is *"every hosted service on this DPU restarts"* |
| Per-artifact cross-link | any per-artifact skill's `## Safety policy` cross-links here for the cross-cutting rule body |

When any cell above fires, the agent MUST load this skill **before** composing the first sentence of the answer, MUST walk the change-application discipline in [`TASKS.md ## configure`](TASKS.md#configure) → [`## modify`](TASKS.md#modify) → [`## test`](TASKS.md#test) → [`## debug`](TASKS.md#debug) in order, and MUST cite the activation explicitly in the answer (e.g. *"because this touches `mlxconfig`, the answer follows the `doca-hardware-safety` discipline …"*) so the user can audit the reasoning.

The activation is **mandatory, not advisory.** The most common failure mode this overlay prevents is *"the agent recommended a `mlxconfig` change with no maintenance window, no out-of-band path, and no rollback statement, the user applied it, the management link dropped, and the box was unrecoverable without a physical console."* The cost of one unjustified activation (a few extra paragraphs in the answer) is trivial compared to the cost of one missed activation.

## Refuse-and-escalate is a hard rule

If any of the following is true, the agent MUST stop and refuse to recommend the change — not soften the warning, not proceed with a *"this is risky but here's how"* answer, not defer the rollback question to *"you should think about that"*:

1. The change has no documented rollback path AND the user cannot provide one. (Per [`CAPABILITIES.md ## Safety policy`](CAPABILITIES.md#safety-policy) *rollback-must-be-documented* rule.)
2. The change is link-breaking AND the host has no out-of-band access path. (Per [`CAPABILITIES.md ## Safety policy`](CAPABILITIES.md#safety-policy) *out-of-band-precondition* rule.)
3. The change requires a cold power cycle AND the user has not confirmed a maintenance window. (Per [`CAPABILITIES.md ## Safety policy`](CAPABILITIES.md#safety-policy) *maintenance-window* rule.)
4. The change has not been validated against a non-prod replica AND the user is asking for direct application to a production box. (Per [`TASKS.md ## test`](TASKS.md#test) *replica-first* rule.)

In each of these cases the correct answer shape is *"this change requires X (here is why); the bundle refuses to recommend it without X; here is the route to obtain X"* — not silence and not improvisation. The refuse-and-escalate rule is what makes the bundle's hardware-safety guidance trustworthy to production operators.

Do **not** load this skill for general DOCA orientation (use
[`doca-public-knowledge-map`](../doca-public-knowledge-map/SKILL.md)),
for first-time install or env-class debug (use
[`doca-setup`](../doca-setup/SKILL.md)), or for purely program-side
debug that does not touch hardware state (use
[`doca-debug`](../doca-debug/SKILL.md) or
[`doca-programming-guide`](../doca-programming-guide/SKILL.md)).

## What this skill provides

This is a **thin loader**. The body keeps only the orientation needed
to pick the right next file. The substantive content lives in two
companion files:

- `CAPABILITIES.md` — the meta-policy surface: the class of changes
  in scope (the pre-flight inventory taxonomy, the
  `mlxconfig`-class / firmware-burn / kernel-boot-parameter
  groupings), the cross-cutting safety policy that every per-artifact
  `## Safety policy` overlays, the failure modes the policy prevents
  (bricked-link, runaway-burn, silent-mode-change, missing-rollback),
  the observability gate the operator must satisfy before any
  workload moves, and the thin version-compatibility overlay that
  redirects to [`doca-version`](../doca-version/SKILL.md).
- `TASKS.md` — the change-application workflows: `## configure` (the
  pre-flight inventory + out-of-band + maintenance-window plan),
  `## build` (routing stub — hardware-touching changes do not produce
  build artifacts), `## modify` (the apply-the-change discipline,
  including the `mlxconfig` cold-power-cycle rule and the
  firmware-burn discipline), `## run` (the post-change verification
  gate), `## test` (the replica-first smoke), `## debug` (the
  rollback ladder + the refuse-and-escalate escape valve), and the
  `## Deferred task verbs` block.

## Loading order

1. Read this `SKILL.md` first to confirm the user's question is in
   scope (the agent is about to recommend a change that touches
   hardware state on a live system).
2. **For the class of changes in scope, the meta-safety policy, the
   failure-mode taxonomy, the observability gate, and the
   version-overlay redirect, see [CAPABILITIES.md](CAPABILITIES.md).**
3. **For the apply-a-change workflow — pre-flight inventory →
   out-of-band → maintenance window → apply → verify → rollback —
   see [TASKS.md](TASKS.md).**
4. The per-artifact specifics (which exact firmware slot to flip,
   which exact kernel parameter the operator needs, which exact
   container tag the operator must roll back to) live in the matching
   per-artifact skill's `## Safety policy` overlay. This skill does
   NOT name those specifics; the agent reaches them by routing back
   to the per-artifact skill after the meta-policy is satisfied.

## Related skills

- [`doca-version`](../doca-version/SKILL.md) — the four-way match
  rule and the host ↔ BlueField BFB ↔ container-tag pairing. Every
  hardware-touching change has a version dimension; this skill's
  `## Version compatibility` overlay is a 3-5 line redirect to
  `doca-version` for the body.
- [`doca-setup`](../doca-setup/SKILL.md) — env-class checks that
  precondition a hardware-touching change (hugepages, IOMMU mode,
  `pkg-config`, representor visibility). The pre-flight inventory in
  this skill's `## configure` cross-links to `doca-setup` for the
  env-class half of the inventory.
- [`doca-debug`](../doca-debug/SKILL.md) — the cross-cutting layered
  debug ladder. When a hardware-touching change goes wrong, the
  rollback ladder in this skill's `## debug` hands off to
  `doca-debug` once the rollback has restored a known state and the
  symptom now lives at a software layer.
- [`doca-structured-tools-contract`](../doca-structured-tools-contract/SKILL.md) —
  the JSON schemas the agent prefers when present. The
  `collect-host-state` / `collect-dpu-state` schemas are the
  structured form of this skill's pre-flight inventory; the agent
  uses them as the one-shot answer when the host has the helpers
  installed.
- [`doca-container-deployment`](../doca-container-deployment/SKILL.md) —
  the canonical container-deployment recipe shared across DOCA
  services. Several hardware-touching changes (BlueField cold
  reboot, BFB reflash) interrupt every hosted service container on
  the BlueField; the rollback path quotes the
  `doca-container-deployment` re-deploy shape.
- [`doca-programming-guide`](../doca-programming-guide/SKILL.md) —
  program-side preconditions (capability discovery,
  validate-before-commit). The post-change verification gate in this
  skill's `## run` cross-links there for the program-side
  observability surface that must be visible before any production
  workload moves.
- Per-artifact `## Safety policy` anchors in each in-bundle
  service / library / tool skill — e.g. the firmware-slot
  precondition in [`doca-argus`](../services/doca-argus/SKILL.md),
  [`doca-dms`](../services/doca-dms/SKILL.md),
  [`doca-firefly`](../services/doca-firefly/SKILL.md),
  [`doca-urom-svc`](../services/doca-urom-svc/SKILL.md); the
  device-touching libraries
  ([`doca-flow`](../libs/doca-flow/SKILL.md),
  [`doca-rdma`](../libs/doca-rdma/SKILL.md),
  [`doca-eth`](../libs/doca-eth/SKILL.md),
  [`doca-pcc`](../libs/doca-pcc/SKILL.md),
  [`doca-rmax`](../libs/doca-rmax/SKILL.md)); and the
  hardware-touching tools (e.g.
  [`doca-spcx-cc`](../tools/doca-spcx-cc/SKILL.md),
  [`doca-pcc-counters`](../tools/doca-pcc-counters/SKILL.md)).
  Every in-bundle artifact skill's `## Safety policy` overlays
  this meta-policy with artifact-specific safety. The cross-link is
  intentionally bidirectional: per-artifact skills link here for
  the meta-policy; this skill enumerates the in-bundle overlays in
  [`CAPABILITIES.md ## Safety policy`](CAPABILITIES.md#safety-policy)
  as "skills that overlay this meta-policy". The externally-
  productized analogs (`doca-virtio-net`, `doca-snap`, `doca-hbn`,
  BlueMan, DPF) are NOT in-bundle skills — their safety policies
  live in product documentation reached through
  [`doca-public-knowledge-map ## Externally-productized DOCA
  software`](../doca-public-knowledge-map/SKILL.md#externally-productized-doca-software--not-in-this-bundle-but-here-is-where-to-route).
