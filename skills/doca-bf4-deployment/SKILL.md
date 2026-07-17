---
license: Apache-2.0
name: doca-bf4-deployment
description: >
  WARNING: guides potentially IRREVERSIBLE BlueField-4 hardware
  operations (PLDM firmware burns, ISO reflashes, power cycles, BMC
  factory resets) that can brick firmware, corrupt boot media, or
  cause outages — a maintenance window and rollback plan are required,
  and every mutating step is governed by doca-hardware-safety, loaded
  alongside. Use this skill for BlueField-4 (BF4) day-1 platform
  bring-up from the BMC: installing the BlueField/DOCA bundle
  ISO onto the DPU (Grace, the Arm complex) over UEFI HTTP
  Boot, PXE, or Redfish Virtual Media; the PLDM firmware-update flow
  (BMC, NIC firmware, SBIOS, ERoT) via the Redfish UpdateService and
  pldmtool; and a Grace Ubuntu image with optional cloud-init. Trigger
  on BlueField-4/BF4 bring-up phrasings even without "BF4": {bring up
  my new BlueField-4}, {the BlueField ISO will not boot over HTTP from
  the BMC}, {attach BF4 virtual media via Redfish}, {BF4 firmware Task
  stuck at Running}. BF3 bring-up, application launch, and library APIs
  belong to other skills.
metadata:
  kind: library
compatibility: >
  No DOCA install is required to read this skill; it teaches the
  documented BlueField-4 BMC-driven bring-up flows (UEFI HTTP Boot,
  PXE, Redfish Virtual Media, PLDM firmware update). Executing the
  steps requires a BlueField-4 with an out-of-band-reachable BMC, a
  host or HTTP/HTTPS server to host the bundle ISO, and the target
  versions from the public BlueField/DOCA release notes.
---

# DOCA BlueField-4 (BF4) deployment

> ⚠️ **WARNING — irreversible hardware operations.** This skill guides
> operators through potentially destructive, irreversible BlueField-4
> hardware operations: PLDM firmware burns, ISO reflashes, power
> cycles, and BMC factory resets. These can brick firmware, corrupt
> boot media, or cause production outages. Do **not** proceed without a
> maintenance window and a tested rollback plan. Every mutating step is
> governed by
> [`doca-hardware-safety`](../doca-hardware-safety/SKILL.md), which
> MUST be loaded alongside this skill before any destructive action.
>
> Before executing any mutating step — PLDM firmware burn, ISO reflash,
> power cycle, or BMC factory reset — the agent MUST show the exact
> command and its blast radius (which device, what becomes unavailable,
> whether it is reversible) and obtain the user's explicit confirmation
> for that specific action. Never chain destructive steps or run them
> speculatively as a side effect of another task.

**Where to start:** This skill is the bundle's deliberate in-bundle
home for **day-1 platform bring-up of a BlueField-4 DPU via the
BMC** — getting a powered-but-bare BF4 to "Grace OS installed,
firmware at the target level, ready to deploy a workload." It is the
upstream of the two application-deployment skills
([`doca-container-deployment`](../doca-container-deployment/SKILL.md)
and
[`doca-bare-metal-deployment`](../doca-bare-metal-deployment/SKILL.md)):
those skills assume a working BlueField; this skill is how the
BlueField-4 GETS to working. If the user has a fresh BF4 and wants to
install the OS or update firmware, open [`TASKS.md`](TASKS.md) and
start at [`## configure`](TASKS.md#configure). If the question is
*what bring-up methods even exist and what is the contract for each*,
start at [`CAPABILITIES.md`](CAPABILITIES.md).

> **Scope note — BF4 day-1 is in scope by directive.** The bundle's
> [`AGENTS.md ## Non-goals`](../../AGENTS.md#non-goals-questions-the-agent-should-recognize-and-refuse-politely)
> item 7 lists the BlueField BSP / BFB / RShim / TMFIFO layer and the
> BlueField BMC software as externally-productized. **BlueField-4
> day-1 bring-up via the BMC is carved into scope for this skill by
> directive** because day-1 has no other home in the bundle. The
> carve-out is narrow: this skill teaches the documented BMC-driven
> install and firmware-update FLOWS (the CLASS), routing every
> *mutating* step through
> [`doca-hardware-safety`](../doca-hardware-safety/SKILL.md) for the
> change-application meta-policy. It does NOT redefine that
> meta-policy, and it does NOT cover BF3 (route to
> `doca-bf3-deployment`), application launch, or library APIs.

## Audience

This skill serves **external operators standing up a new
BlueField-4** who already have:

- a BlueField-4 with its BMC reachable out-of-band (BMC SSH plus the
  documented Redfish endpoint), so the DPU can be driven without
  physical access,
- the BlueField/DOCA bundle ISO (and, for the Grace-Ubuntu path, a
  Grace Ubuntu image) downloaded from the public NVIDIA download
  surface, hosted at {iso-uri} on the operator's own HTTP/HTTPS
  server, and
- the target firmware and OS versions read from the **public
  BlueField/DOCA release notes** (this skill never quotes a specific
  pre-release firmware version).

It is **not** for:

- BlueField-3 (BF3) bring-up — route to `doca-bf3-deployment`,
- developers who want to RUN a DOCA service container or a DOCA-linked
  binary on an already-working BlueField — route to
  [`doca-container-deployment`](../doca-container-deployment/SKILL.md)
  or
  [`doca-bare-metal-deployment`](../doca-bare-metal-deployment/SKILL.md),
- the cross-cutting hardware-change meta-policy itself (preflight, OOB
  console discipline, maintenance window, rollback) — that is owned by
  [`doca-hardware-safety`](../doca-hardware-safety/SKILL.md) and this
  skill cross-links it, never duplicates it,
- fleet-scale / orchestrated DPU provisioning — that is DOCA Platform
  Framework territory, routed via
  [`doca-public-knowledge-map`](../doca-public-knowledge-map/SKILL.md).

The skill teaches the agent the documented bring-up *procedure* and
the rules for quoting Redfish / PLDM / UEFI standard operations and
public BlueField/DOCA documentation via
[`doca-public-knowledge-map`](../doca-public-knowledge-map/SKILL.md);
it does not invent BMC credentials, ISO URIs, firmware version
strings, EIDs, Redfish task IDs, or device names from memory.

## When to load this skill

Load this skill when the user is doing **hands-on day-1 bring-up of a
BlueField-4 via the BMC**, or asking a cross-cutting BF4-bring-up
question that is not specific to a later application-deployment step.
Concretely:

- Installing the BlueField/DOCA bundle ISO onto the DPU (Grace) for
  the first time, and choosing between the three documented install
  methods — UEFI HTTP Boot (recommended), PXE Boot, or Redfish
  Virtual Media.
- Running the PLDM firmware-update flow across the BMC / NIC firmware
  / SBIOS / ERoT components: pushing the `.fwpkg` bundle through the
  Redfish UpdateService multipart endpoint, monitoring the returned
  Task, verifying pending images with `pldmtool`, and activating with
  a power cycle.
- Installing a Grace Ubuntu image (with optional cloud-init via a
  CIDATA-labelled config ISO) through Redfish Virtual Media, with
  either local hosting on the BMC eMMC or remote hosting on an
  HTTPS server.
- Reaching the DPU's OOB serial console (BMC SSH plus
  `obmc-console-client`) to watch the installer or UEFI menus.
- Diagnosing a bring-up that is misbehaving — the ISO will not boot,
  virtual media will not attach, a firmware Task hangs or reports an
  Exception, a pending image never activates, cloud-init is ignored,
  or the DPU is stuck in a boot loop because media was never detached.
- Cross-cutting questions: *"HTTP Boot or Redfish Virtual Media — which
  do I use, and when do I actually need PXE?"*, *"how do I know the
  firmware update actually took effect?"*, *"the ISO landed but the
  NIC firmware update sub-step seems to have failed — what now?"*.

Do **not** load this skill for BF3 bring-up (route to
`doca-bf3-deployment`); for running an application on an
already-working BlueField (route to
[`doca-container-deployment`](../doca-container-deployment/SKILL.md)
or
[`doca-bare-metal-deployment`](../doca-bare-metal-deployment/SKILL.md));
for env preparation on the installed Grace OS such as hugepages /
pkg-config / devlink (use [`doca-setup`](../doca-setup/SKILL.md)); for
the cross-cutting hardware-change meta-policy (route to
[`doca-hardware-safety`](../doca-hardware-safety/SKILL.md)); or for
fleet-scale orchestrated provisioning (route via
[`doca-public-knowledge-map`](../doca-public-knowledge-map/SKILL.md)).

## What this skill provides

This is a **thin loader**. Substantive material lives in two
companion files:

- `CAPABILITIES.md` — the BF4 day-1 bring-up contract: the three OS
  install methods (UEFI HTTP Boot, PXE Boot, Redfish Virtual Media)
  and the Grace-Ubuntu-plus-cloud-init Virtual Media path; the PLDM
  firmware-update surface across BMC / NIC firmware / SBIOS / ERoT;
  the version-compatibility overlay on
  [`doca-version`](../doca-version/SKILL.md) (the install and firmware
  targets come from the public release notes, never from memory); the
  bring-up error taxonomy (boot-source -> virtual-media-attach ->
  firmware-Task -> activation -> cloud-init -> boot-loop); the
  observability surface (the OOB console, the Redfish Task resource,
  the Redfish FirmwareInventory, `pldmtool` GetFwParams, and the
  installed-build check `cat /etc/mlnx-release`); and the safety
  policy (an overlay on
  [`doca-hardware-safety`](../doca-hardware-safety/SKILL.md): every
  PLDM burn / ISO reflash / power cycle / BMC factory reset is a
  MUTATING hardware op; never print a real password; always detach
  virtual media to avoid boot loops; only public hosts for any
  NVIDIA URL).
- `TASKS.md` — step-by-step workflows for the in-scope bring-up verbs:
  `configure`, `build` (routing stub), `modify` (routing stub), `run`
  (the three install methods plus the PLDM firmware-update flow plus
  the Grace-Ubuntu cloud-init path, as `###` sub-anchors), `test`
  (the post-install / post-update verification sweep), `debug` (the
  layered bring-up diagnosis), and the `Deferred task verbs` block
  routing BF3 / application-launch / library-API / env-prep /
  hardware-meta-policy / fleet questions out to their owning skills.

The skill assumes a BlueField-4 target where:

- the BMC is reachable out-of-band and the operator has BMC
  credentials ({bmc-user} / {bmc-password}) they supply — never
  invented here,
- the bundle ISO (and any Grace Ubuntu image / cloud-init config ISO)
  is downloaded from the public NVIDIA download surface and hosted at
  {iso-uri},
- the operator has read the target firmware and OS versions from the
  public BlueField/DOCA release notes.

It does not cover installing DOCA tooling on the host — that path goes
through [`doca-setup`](../doca-setup/SKILL.md) — and it does not cover
running a workload once Grace is up — those paths go through the two
application-deployment skills.

## Loading order

1. Read this `SKILL.md` first to confirm the user's question is in
   scope (BF4 day-1 bring-up via the BMC; NOT BF3, NOT application
   launch, NOT a library-API question, NOT the hardware-change
   meta-policy itself).
2. **For the bring-up contract (the three install methods, the
   Grace-Ubuntu cloud-init path, the PLDM firmware-update surface, the
   version overlay, the bring-up error taxonomy, the observability
   surface, and the BF4 safety overlay), see
   [CAPABILITIES.md](CAPABILITIES.md).**
3. **For step-by-step workflows — `configure`, `build` (routing
   stub), `modify` (routing stub), `run` (with the three install
   methods, the PLDM flow, and the Grace-Ubuntu cloud-init path as
   `###` sub-anchors), `test`, `debug`, plus the `Deferred task verbs`
   block — see [TASKS.md](TASKS.md).**
4. **Load
   [`doca-hardware-safety`](../doca-hardware-safety/SKILL.md)
   ALONGSIDE** whenever the question reaches a mutating step (PLDM
   firmware burn, ISO reflash, power cycle, BMC factory reset).

## Example questions this skill answers well

See [`references/details.md`](references/details.md#example-questions-this-skill-answers-well).
## What this skill deliberately does not ship

See [`references/details.md`](references/details.md#what-this-skill-deliberately-does-not-ship).
## Related skills

See [`references/details.md`](references/details.md#related-skills).
