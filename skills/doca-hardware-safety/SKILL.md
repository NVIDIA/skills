---
license: Apache-2.0
name: doca-hardware-safety
description: >
  Use this skill whenever the agent is about to recommend or apply a
  change that touches DPU / NIC hardware state on a live system —
  mlxconfig firmware-parameter write, NIC firmware burn, BFB reflash,
  NIC ↔ DPU mode flip, SR-IOV or device-emulation slot enable, kernel
  boot-parameter change (IOMMU, hugepages, VFIO), PCIe rebind / rescan /
  link-state flip, BlueField cold reboot, or remediation of a BlueField
  thermal shutdown. Wrap the change in
  pre-flight inventory, OOB reachability, a maintenance window, replica
  rehearsal, and rollback. Trigger on implicit phrasing: "flip BlueField
  mode over SSH", "enable SR-IOV and reboot", "burned firmware but
  mlxconfig shows old value", "BlueField thermal shutdown". Not for
  read-only inventory or version queries (doca-version, doca-caps). Refuse
  orientation (doca-public-knowledge-map), install or env debug
  (doca-setup), and program-side debug (doca-debug,
  doca-programming-guide).
metadata:
  author: NVIDIA <doca@nvidia.com>
  kind: library
  doca_version: "3.5"
  tags:
    - doca
    - hardware
    - safety
compatibility: >
  No DOCA install required to read this skill (it is an overlay
  loaded against any DOCA artifact skill); the validation steps
  within DO require a live DOCA install at /opt/mellanox/doca with a
  BlueField DPU or ConnectX NIC, plus out-of-band console
  reachability (BMC, RShim, or operator-managed console) for any
  link-breaking change.
---

# DOCA hardware safety

## Examples

- **"Enable SR-IOV and reboot."** — this is in scope. Inventory first,
  require out-of-band reachability, a maintenance window, replica
  rehearsal, then apply.
- **"Flip BlueField mode over SSH."** — refuse if the host has no
  out-of-band path. A link-breaking change without BMC / RShim /
  console is a hard stop.

## Quick answers (answer from this file)

- **Thermal shutdown or hardware safety limits.** Treat it as a hardware
  event: capture read-only evidence first (BMC/Redfish chassis thermal
  sensors and event log, the out-of-band console, `dmesg | grep -i mlx5`,
  and `mget_temp -d <mst device>` from MFT). Never disable or raise thermal
  protection; airflow and cooling are facility fixes. Any remediation that is
  a firmware update, an `mlxconfig` write or a cold power cycle goes through
  the contract below; with no root cause or no rollback, refuse and escalate
  to NVIDIA support.
- **Read-only inventory or version questions** do not need this skill: name
  `doca-version` (four-source chain) or `doca-caps` (`doca_caps --list-devs`)
  and answer there.
- Name sibling skills (`doca-version`, `doca-setup`,
  `doca-structured-tools-contract`) when they apply; do not open them to
  answer a safety question.

## Instructions

Follow Loading order below. Open the tasks companion before applying
a hardware-touching change. Open the capabilities companion for the
in-scope change class and refuse-and-escalate rules. Do not open every
companion file.

**Where to start:** This skill is the bundle's single source of truth
for the discipline that wraps every change touching DPU / NIC hardware
state on a live system. Open
the tasks companion when the operator is about to *apply* a
hardware-touching change and needs the change-application discipline
(pre-flight inventory → out-of-band path → window → apply → verify
→ rollback). Open the capabilities companion when the
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

## Mandatory answer contract for an `mlxconfig`-class production change

An answer about applying an `mlxconfig`-class write on production MUST
state every gate below explicitly; links to companion files do not
substitute for walking the gates:

1. Name the change class before any apply step and state that its
   firmware-stored value commits at the next **cold power cycle**, not
   a warm reboot.
2. Require a pre-flight capture of PCIe topology, link state, firmware
   level, BFB level, the current `mlxconfig` configuration, and the
   host kernel/modules/environment.
3. Treat the change as potentially link-breaking and refuse to issue
   it until a deployment-specific OOB console class is named and
   proven reachable. Never describe an `mlxconfig`-class production
   write as inherently non-link-breaking.
4. Require a time-boxed maintenance window with stakeholders,
   rollback decision point, rollback plan, and verification gate
   documented before application.
5. Rehearse both the forward change and the rollback on a matching
   non-production replica before production.
6. Keep production workload gated until the four-way version match,
   per-artifact health metric, and end-to-end observability path all
   pass.
7. Name the rollback ladder: compare with the captured baseline →
   restore the captured values → cold-power-cycle → repeat inventory
   and four-way checks → route residual software symptoms to
   `doca-debug`.
8. Include an explicit refusal and escalation statement in the answer:
   if no documented rollback exists, the captured baseline cannot be
   restored, or OOB access is unavailable for a potentially
   link-breaking change, **stop and refuse to proceed**. Escalate to
   the operator's change-control process, vendor support, or the public
   DOCA Developer Forum; never guess at recovery. Merely naming a
   "refuse-and-escalate rule" without stating this action and route does
   not satisfy the contract. Never invent a parameter, address,
   version, endpoint, or recovery value.

The replica gate is **required**, never merely recommended: say that
production is blocked until both the forward change and its rollback
have succeeded on the representative replica. Wording such as
*"recommended"*, *"highly recommended"*, or *"if available"* violates
this contract even when a replica is mentioned elsewhere in the answer.

The same framing applies to firmware burn, BFB reflash, and host
boot-parameter changes; only their documented commit action,
class-specific preconditions, and rollback mechanism differ.

## When to load this skill

Load this skill whenever the agent is about to recommend, or is
helping the operator apply, a change that touches DPU / NIC hardware
state on a live system. The decision must be made **before** the
agent composes its first sentence — the activation checklist below
is the same one referenced from
`AGENTS.md ## Cross-cutting overlay activation triggers` (bundle `AGENTS.md#cross-cutting-overlay-activation-triggers`),
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
| BlueField reboot class | BlueField cold reboot, BlueField warm reboot to apply `mlxconfig`; opening any RShim node other than `misc` (even `cat /dev/rshimN/boot` starts a boot cycle and resets the BlueField; read only `/dev/rshimN/misc`); any change whose blast radius is *"every hosted service on this DPU restarts"* |
| Per-artifact cross-link | any per-artifact skill's `## Safety policy` cross-links here for the cross-cutting rule body |

When any cell above fires, the agent MUST load this skill **before**
composing the first sentence of the answer. For production it MUST
walk the change-application discipline in this order:
## configure (plan) →
`## test` (representative replica change + rollback
rehearsal) → `## modify` (production apply) →
`## run` (production verification) →
`## debug` (debug / rollback). It MUST cite the
activation explicitly in the answer (e.g. *"because this touches
`mlxconfig`, the answer follows the `doca-hardware-safety`
discipline …"*) so the user can audit the reasoning.

The activation is **mandatory, not advisory.** The most common failure mode this overlay prevents is *"the agent recommended a `mlxconfig` change with no maintenance window, no out-of-band path, and no rollback statement, the user applied it, the management link dropped, and the box was unrecoverable without a physical console."* The cost of one unjustified activation (a few extra paragraphs in the answer) is trivial compared to the cost of one missed activation.

## Refuse-and-escalate is a hard rule

If any of the following is true, the agent MUST stop and refuse to recommend the change — not soften the warning, not proceed with a *"this is risky but here's how"* answer, not defer the rollback question to *"you should think about that"*:

1. The change has no documented rollback path AND the user cannot provide one. (Per ## Safety policy *rollback-must-be-documented* rule.)
2. The change is link-breaking AND the host has no out-of-band access path. (Per ## Safety policy *out-of-band-precondition* rule.)
3. The change touches hardware state AND the user has not confirmed an
   explicit, time-boxed maintenance window. (Per
   ## Safety policy
   *maintenance-window* rule.)
4. Production application is contemplated before the change and its
   rollback have passed on a representative non-prod replica. This
   refusal is intent-based: it applies to a plan, recommendation, or
   next action that would reach production early, not only when the
   user explicitly asks for "direct application." A replica
   mismatched on the required hardware, firmware, kernel, module, or
   function-topology axes does not satisfy the gate; obtain a
   representative replica or refuse and escalate. (Per
   ## test *replica-first* rule.)

In each of these cases the correct answer shape is *"this change requires X (here is why); the bundle refuses to recommend it without X; here is the route to obtain X"* — not silence and not improvisation. The refuse-and-escalate rule is what makes the bundle's hardware-safety guidance trustworthy to production operators.

Do **not** load this skill for general DOCA orientation (use
`doca-public-knowledge-map/SKILL.md`),
for first-time install or env-class debug (use
`doca-setup/SKILL.md`), or for purely program-side
debug that does not touch hardware state (use
`doca-debug/SKILL.md` or
`doca-programming-guide/SKILL.md`).

## What this skill provides

This is a **thin loader**. The body keeps only the orientation needed
to pick the right next file. The substantive content lives in two
companion files:

- the capabilities companion — the meta-policy surface: the class of changes
  in scope (the pre-flight inventory taxonomy, the
  `mlxconfig`-class / firmware-burn / kernel-boot-parameter
  groupings), the cross-cutting safety policy that every per-artifact
  `## Safety policy` overlays, the failure modes the policy prevents
  (bricked-link, runaway-burn, silent-mode-change, missing-rollback),
  the observability gate the operator must satisfy before any
  workload moves, and the thin version-compatibility overlay that
  redirects to `doca-version/SKILL.md`.
- the tasks companion — the change-application workflows: `## configure` (the
  pre-flight inventory + out-of-band + maintenance-window plan),
  `## build` (routing stub — hardware-touching changes do not produce
  build artifacts), `## modify` (the apply-the-change discipline,
  including the `mlxconfig` cold-power-cycle rule and the
  firmware-burn discipline), `## run` (the post-change verification
  gate), `## test` (the replica-first smoke), `## debug` (the
  rollback ladder + the refuse-and-escalate escape valve), and the
  `## Deferred task verbs` block.

## Loading order

1. Read this skill first to confirm the user's question is in
   scope (the agent is about to recommend a change that touches
   hardware state on a live system).
2. **For the class of changes in scope, the meta-safety policy, the
   failure-mode taxonomy, the observability gate, and the
   version-overlay redirect, see the capabilities companion.**
3. **For the apply-a-change workflow — pre-flight inventory →
   out-of-band → maintenance window → apply → verify → rollback —
   see the tasks companion.**
4. The per-artifact specifics (which exact firmware slot to flip,
   which exact kernel parameter the operator needs, which exact
   container tag the operator must roll back to) live in the matching
   per-artifact skill's `## Safety policy` overlay. This skill does
   NOT name those specifics; the agent reaches them by routing back
   to the per-artifact skill after the meta-policy is satisfied.

## Related skills

- `doca-version/SKILL.md` — the four-way match
  rule and the host ↔ BlueField BFB ↔ container-tag pairing. Every
  hardware-touching change has a version dimension; this skill's
  `## Version compatibility` overlay is a 3-5 line redirect to
  `doca-version` for the body.
- `doca-setup/SKILL.md` — env-class checks that
  precondition a hardware-touching change (hugepages, IOMMU mode,
  `pkg-config`, representor visibility). The pre-flight inventory in
  this skill's `## configure` cross-links to `doca-setup` for the
  env-class half of the inventory.
- `doca-debug/SKILL.md` — the cross-cutting layered
  debug ladder. When a hardware-touching change goes wrong, the
  rollback ladder in this skill's `## debug` hands off to
  `doca-debug` once the rollback has restored a known state and the
  symptom now lives at a software layer.
- `doca-structured-tools-contract/SKILL.md` —
  the JSON schemas the agent prefers when present. The
  `collect-host-state` / `collect-dpu-state` schemas are the
  structured form of this skill's pre-flight inventory; the agent
  uses them as the one-shot answer when the host has the helpers
  installed.
- `doca-container-deployment/SKILL.md` —
  the canonical container-deployment recipe shared across DOCA
  services. Several hardware-touching changes (BlueField cold
  reboot, BFB reflash) interrupt every hosted service container on
  the BlueField; the rollback path quotes the
  `doca-container-deployment` re-deploy shape.
- `doca-programming-guide/SKILL.md` —
  program-side preconditions (capability discovery,
  validate-before-commit). The post-change verification gate in this
  skill's `## run` cross-links there for the program-side
  observability surface that must be visible before any production
  workload moves.
- Per-artifact `## Safety policy` anchors in each in-bundle
  service / library / tool skill — e.g. the firmware-slot
  precondition in `doca-argus/SKILL.md`,
  `doca-dms/SKILL.md`,
  `doca-firefly/SKILL.md`,
  `doca-urom-svc/SKILL.md`; the
  device-touching libraries
  (`doca-flow/SKILL.md`,
  `doca-rdma/SKILL.md`,
  `doca-eth/SKILL.md`,
  `doca-pcc/SKILL.md`,
  `doca-rmax/SKILL.md`); and the
  hardware-touching tools (e.g.
  `doca-spcx-cc/SKILL.md`,
  `doca-pcc-counters/SKILL.md`).
  Every in-bundle artifact skill's `## Safety policy` overlays
  this meta-policy with artifact-specific safety. The cross-link is
  intentionally bidirectional: per-artifact skills link here for
  the meta-policy; this skill enumerates the in-bundle overlays in
  ## Safety policy
  as "skills that overlay this meta-policy". The externally-
  productized analogs (`doca-virtio-net`, `doca-snap`, `doca-hbn`,
  BlueMan, DPF) are NOT in-bundle skills — their safety policies
  live in product documentation reached through
  `doca-public-knowledge-map ## Externally-productized DOCA
  software` (`doca-public-knowledge-map/SKILL.md#externally-productized-doca-software--not-in-this-bundle-but-here-is-where-to-route`).
