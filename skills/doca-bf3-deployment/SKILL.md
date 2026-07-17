---
license: Apache-2.0
name: doca-bf3-deployment
description: >
  Use this skill for BlueField-3 (BF3) day-1 platform bring-up via
  the classic RShim/BFB path: pushing a BlueField bundle (BFB) to
  the DPU over RShim with bfb-install from the host, the host-to-DPU
  TMFIFO management channel (tmfifo_net0, the 192.168.100.x
  convention), RShim daemon state and console-over-rshim, DPU mode
  selection (DPU/embedded-function vs separated-host/NIC mode) via
  mlxconfig, post-BFB recovery, a six-state BlueField-state
  classifier, and verifying the install (cat /etc/mlnx-release plus
  version checks). Trigger even when the user does not say "BF3" —
  typical phrasings include {push a BFB to my BlueField-3},
  {bfb-install exited 0 but the DPU never came back}, {ping
  192.168.100.2 works but ssh fails}, or {is DOCA on the host or
  the Arm side?}. Mutating burns (BFB reflash, mlxconfig set, mode
  flip, firmware burn) are meta-policy owned by doca-hardware-safety,
  loaded alongside; app launch, container deploy, env install, and
  the BF4 BMC-Redfish path belong to other skills.
metadata:
  kind: library
compatibility: >
  No DOCA install required to read this skill (it is a
  platform-lifecycle overlay loaded against BF3 hardware); the
  bring-up and validation steps within DO require a real
  BlueField-3, host-side RShim access (PCIe or USB), the matching
  DOCA-Host install, and a BlueField bundle (BFB) image downloaded
  from the public DOCA Downloads page.
---

# DOCA BlueField-3 (BF3) deployment

**Where to start:** This skill is the bundle's home for **BlueField-3
day-1 platform bring-up** — taking a BF3 from "powered card in the
slot" (or a card that just came back broken from a BFB push) to
"Arm OS healthy, TMFIFO up, host PFs bound, four-way version match
closed, ready to run a workload". It owns the **classic RShim/BFB
path** that BF3 uses today; the newer BMC-Redfish provisioning path
is the sibling skill
[`doca-bf4-deployment`](../doca-bf4-deployment/SKILL.md) (the BF4
equivalent). If the user has a BF3 and needs to push a BFB, recover
a DPU that did not come back, or verify the install, open
[`TASKS.md`](TASKS.md) and start at
[`## configure`](TASKS.md#configure). If the question is *what shape
does the BF3 platform-bring-up surface even have*, start at
[`CAPABILITIES.md`](CAPABILITIES.md). Once the BF3 is healthy, this
skill routes **onward** to the deployment skills — running a binary
goes to
[`doca-bare-metal-deployment`](../doca-bare-metal-deployment/SKILL.md);
deploying a service container goes to
[`doca-container-deployment`](../doca-container-deployment/SKILL.md).

Every **mutating** burn invoked from a bring-up step — the BFB
reflash itself, any `mlxconfig set` (including a DPU/separated-host
mode flip), a firmware burn, or a kernel-boot-parameter change — is
governed by the change-application meta-policy in
[`doca-hardware-safety`](../doca-hardware-safety/SKILL.md), which
the agent loads ALONGSIDE this skill. This skill adds only the
**BF3-specific operational sequencing** on top; it does NOT redefine
the preflight / OOB-console / maintenance-window / rollback
discipline that meta-policy owns.

## Audience

This skill serves **external DOCA operators bringing up a real
BlueField-3** — i.e. people who already have:

- a physical BlueField-3 in a host (or a standalone BF3 they can
  reach over its console / management network),
- host-side RShim access to the DPU (the RShim userspace daemon and
  the `/dev/rshim*` character-device tree present over the PCIe or
  USB RShim interface), and
- a matching DOCA-Host install on the host plus a BlueField bundle
  (BFB) image downloaded from the public DOCA Downloads page.

It is **not** for:

- BlueField-4 bring-up (the BMC-Redfish provisioning path) — route
  to [`doca-bf4-deployment`](../doca-bf4-deployment/SKILL.md), the
  BF4 equivalent of this skill,
- kernel-driver or BlueField-OS developers contributing to `mlx5_*`
  or the BFB image itself (that is internal-tree work, not a
  field deployment),
- operators who already have a healthy BF3 and just want to *run a
  binary* (route to
  [`doca-bare-metal-deployment`](../doca-bare-metal-deployment/SKILL.md))
  or *deploy a service container* (route to
  [`doca-container-deployment`](../doca-container-deployment/SKILL.md)),
- fresh-no-hardware users with no DOCA install — route to
  [`doca-setup ## no-install`](../doca-setup/TASKS.md#no-install).

The skill teaches the agent the BF3 bring-up *procedure* and the
rules for quoting documented commands from the public BlueField
Platform Software Manual, the public DOCA Installation Guide, and
the MFT manual via
[`doca-public-knowledge-map`](../doca-public-knowledge-map/SKILL.md);
it does **not** invent `bfb-install` flag sets, BFB image filenames,
RShim character-device paths, `bf.cfg` schema keys, `mlxconfig`
parameter names, or TMFIFO subnets from memory. Where a fact is
already vetted in
[`doca-bare-metal-deployment ## bluefield-lifecycle`](../doca-bare-metal-deployment/TASKS.md#bluefield-lifecycle),
this skill reuses that exact fact rather than restating a new one.

## When to load this skill

Load this skill when the user is doing **hands-on BlueField-3
platform bring-up over the RShim/BFB path**, or asking a
cross-cutting BF3 lifecycle question that is not specific to one
library's API. Concretely:

- Pushing a BFB image to a BF3 for the first time (or re-pushing
  after a failed install), from the host over the RShim interface
  with `bfb-install`.
- Bringing up or recovering the host-to-DPU TMFIFO management
  channel (`tmfifo_net0`, the documented `192.168.100.x`
  convention) and the RShim console.
- Confirming RShim driver/daemon state on the host (the userspace
  `rshim` daemon and the `/dev/rshim*` tree) before any push or
  console capture.
- Deciding (and routing) a DPU-mode change — DPU / embedded-function
  mode vs separated-host / NIC mode — knowing the actual `mlxconfig
  set` burn leaves this skill for
  [`doca-hardware-safety`](../doca-hardware-safety/SKILL.md).
- Recovering a BF3 that did not come back after a BFB push:
  `bfb-install` exited 0 but the DPU never reached the documented
  `DPU is ready` marker; `ping 192.168.100.2` works but SSH refuses;
  host PFs are present in `lspci -d 15b3:` but their netdevs are
  gone.
- Verifying a BF3 install — `cat /etc/mlnx-release` on the Arm side,
  the four-way version match per
  [`doca-version`](../doca-version/SKILL.md) — and distinguishing
  the host-side DOCA install from the BlueField-Arm-side DOCA
  install.
- Cross-cutting questions: *"is DOCA on the host or on the Arm
  side, and which one do I install?"*, *"my BF3 was fine last week
  and after a BFB push it never came back — where do I start?"*,
  *"how do I tell which `/dev/rshim<N>` is which BlueField on a
  multi-DPU host?"*.

Do **not** load this skill for: BlueField-4 bring-up (route to
[`doca-bf4-deployment`](../doca-bf4-deployment/SKILL.md), the BF4
equivalent); running a DOCA-linked binary on a healthy BF3 (route to
[`doca-bare-metal-deployment`](../doca-bare-metal-deployment/SKILL.md));
deploying a DOCA service container (route to
[`doca-container-deployment`](../doca-container-deployment/SKILL.md));
env-preparation including hugepages, IOMMU, pkg-config, and devlink
mode flips (use [`doca-setup`](../doca-setup/SKILL.md)); the body of
the version-match rule (use [`doca-version`](../doca-version/SKILL.md));
or any hardware-state-changing burn itself — the change-application
discipline is meta-policy owned by
[`doca-hardware-safety`](../doca-hardware-safety/SKILL.md), loaded
ALONGSIDE this skill.

## What this skill provides

This is a **thin loader**. Substantive material lives in two
companion files:

- `CAPABILITIES.md` — the BF3 platform-bring-up contract: the
  RShim/BFB transport surface (the userspace RShim daemon, the
  `/dev/rshim*` tree, console-over-rshim, the BFB image as the unit
  of input), the TMFIFO management-channel surface (`tmfifo_net0` /
  `tm-br`, the documented `192.168.100.x` convention, the
  `ip route get`-before-`ping` loopback gotcha), the DPU-mode
  surface (DPU / embedded-function vs separated-host / NIC mode, set
  via `mlxconfig` at BFB-install time — a MUTATING burn routed to
  [`doca-hardware-safety`](../doca-hardware-safety/SKILL.md)), the
  host-side-vs-Arm-side DOCA install distinction, the
  BF3-version overlay on the four-way match owned by
  [`doca-version`](../doca-version/SKILL.md), the cross-cutting error
  taxonomy, the observability surface, and the safety policy
  (overlay on
  [`doca-hardware-safety`](../doca-hardware-safety/SKILL.md)).
- `TASKS.md` — step-by-step workflows for the in-scope BF3
  lifecycle verbs: `configure`, `build` (routing stub), `modify`,
  `run` (the BFB-install + RShim/TMFIFO bring-up sequence), `test`
  (the post-BFB readiness smoke), `debug` (the six-state
  `bluefield-state-classifier`), and the `Deferred task verbs`
  block routing app-launch / container / install / library-API /
  hardware-state-change / BF4 questions out to their owning skills.

The skill assumes a target where:

- a BlueField-3 is physically present and powered, reachable from a
  host that has the RShim daemon and `/dev/rshim*` tree available,
- the operator has a BFB image downloaded from the public DOCA
  Downloads page (route via
  [`doca-public-knowledge-map`](../doca-public-knowledge-map/SKILL.md)),
  and
- the operator has an out-of-band path (BMC console, serial-over-LAN,
  or physical UART) to reach the BF3 if a push breaks the Arm OS.

It does **not** cover installing DOCA on a host from scratch (that
goes through [`doca-setup`](../doca-setup/SKILL.md)), and it does
**not** cover BlueField-4 (that goes through
[`doca-bf4-deployment`](../doca-bf4-deployment/SKILL.md)).

## Loading order

1. Read this `SKILL.md` first to confirm the user's question is in
   scope (BF3 platform bring-up over the RShim/BFB path; NOT BF4,
   NOT app-launch, NOT a library-API question).
2. **For the bring-up contract (RShim/BFB transport, TMFIFO
   channel, DPU-mode surface, host-vs-Arm install distinction,
   BF3-version overlay, error taxonomy, observability surface, BF3
   safety overlay), see [CAPABILITIES.md](CAPABILITIES.md).**
3. **For step-by-step workflows — `configure`, `build` (routing
   stub), `modify`, `run` (BFB install + RShim/TMFIFO bring-up),
   `test` (post-BFB readiness smoke), `debug` (the six-state
   `bluefield-state-classifier`), plus the `Deferred task verbs`
   block — see [TASKS.md](TASKS.md).**

## Example questions this skill answers well

See [`references/details.md`](references/details.md#example-questions-this-skill-answers-well).
## What this skill deliberately does not ship

See [`references/details.md`](references/details.md#what-this-skill-deliberately-does-not-ship).
## Related skills

See [`references/details.md`](references/details.md#related-skills).
