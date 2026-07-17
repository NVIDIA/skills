# DOCA BlueField-4 deployment — Capabilities

**Where to start:** The pattern overview below names the recurring
BF4 day-1 bring-up patterns the agent walks for a BlueField-4 driven
from its BMC. Pick the pattern first, then drill into the H2 that owns
the substance. For the *how* of executing each pattern, jump to
[TASKS.md](TASKS.md). For the application-deployment skills that take
over once Grace is up, see
[`doca-container-deployment`](../doca-container-deployment/SKILL.md)
and
[`doca-bare-metal-deployment`](../doca-bare-metal-deployment/SKILL.md).
For the cross-cutting hardware-change meta-policy this skill overlays,
see [`doca-hardware-safety`](../doca-hardware-safety/SKILL.md).

This file enumerates the BF4 bring-up contract as described in the
public **BlueField/DOCA documentation** (reachable through
[`doca-public-knowledge-map`](../doca-public-knowledge-map/SKILL.md))
plus the public industry standards the BMC path is built on:
**Redfish** (DMTF), **PLDM for Firmware Update** (DMTF), and **UEFI**
(HTTP Boot, PXE Boot, Boot Manager). Treat this file as a *map of what
is documented*, not a substitute for reading the live BlueField/DOCA
release notes and the Redfish / PLDM / UEFI specifications against the
operator's target BF4.

## Pattern overview

Every BF4-bring-up question this skill teaches resolves into one of
FOUR patterns. The patterns are CLASSES — they describe the documented
flow, not a specific lab BF4, a specific firmware drop, or a
site-specific BMC.

| BF4 bring-up pattern | Class shape | Where the substance lives |
| --- | --- | --- |
| 1. Install the bundle ISO onto Grace | Choose UEFI HTTP Boot (recommended), PXE Boot, or Redfish Virtual Media; reach Grace over the OOB console; point the chosen boot path at {iso-uri}; boot the installer | [`## Capabilities and modes`](#capabilities-and-modes) install-method table; [`TASKS.md ## run`](TASKS.md#run) |
| 2. Update platform firmware via PLDM | Push the `.fwpkg` bundle through the Redfish UpdateService multipart endpoint; monitor the Task; verify pending images with `pldmtool`; activate with a power cycle | [`## Capabilities and modes`](#capabilities-and-modes) PLDM table; [`TASKS.md ### pldm-firmware-update`](TASKS.md#pldm-firmware-update) |
| 3. Install a Grace Ubuntu image + optional cloud-init | Transfer the image (and a CIDATA config ISO) to the BMC via Redfish SimpleUpdate; attach via Redfish VirtualMedia; BootSourceOverride to USB then reset Grace; verify and detach | [`## Capabilities and modes`](#capabilities-and-modes) Virtual Media rules; [`TASKS.md ### grace-ubuntu-cloud-init`](TASKS.md#grace-ubuntu-cloud-init) |
| 4. Map a bring-up failure back to its layer | Boot-source -> virtual-media-attach -> firmware-Task -> activation -> cloud-init -> boot-loop; six layers, each with its own evidence | [`## Error taxonomy`](#error-taxonomy) layered split |

Two cross-cutting rules apply to *every* pattern above:

- **Operate the documented path; do not invent one.** BMC
  credentials, the ISO URI, firmware version strings, the PLDM EID,
  Redfish task IDs, and Grace device names all come from the
  operator's environment, the public BlueField/DOCA release notes, the
  Redfish / PLDM / UEFI standards, or live BMC output — never from
  memory. Inventing a credential, a version, or an endpoint is the
  most common hallucination failure for this skill.
- **Every state change is a MUTATING hardware op.** A PLDM firmware
  burn, an ISO reflash, a power cycle, and a BMC factory reset all
  change live device state. The change-application discipline
  (preflight, OOB console, maintenance window, rollback) is owned by
  [`doca-hardware-safety`](../doca-hardware-safety/SKILL.md), loaded
  ALONGSIDE; this skill adds only the BF4-bring-up-specific sequencing.

## Capabilities and modes

### OS install methods — three documented ways to land the bundle ISO on Grace

Grace is the Arm CPU complex on BlueField-4; installing the
BlueField/DOCA bundle ISO onto it is the day-1 task. The runtime
contract is uniform (reach Grace over the OOB console, point a boot
path at the ISO, boot the installer, do the shared post-install); the
boot mechanism differs by method.

| Property | Method A — UEFI HTTP Boot (recommended) | Method B — PXE Boot | Method C — Redfish Virtual Media |
| --- | --- | --- | --- |
| What hosts the ISO | An HTTP server the operator runs at {iso-uri} (no PXE/DHCP/TFTP needed) | A configured PXE server: DHCP + TFTP + the ISO | The BMC itself (ISO uploaded to BMC eMMC), or a remote HTTPS server |
| How Grace reaches the installer | Reach the OOB console via BMC SSH + `obmc-console-client`; reboot; in UEFI go Device Manager -> Network Device List -> select the OOB MAC -> HTTP Boot Configuration -> set Boot URI to {iso-uri} -> save -> Boot Manager -> select UEFI HTTP -> boot | In Boot Manager set next boot to the OOB IPv4 net device; select the DOCA arm64 menu, then the ISO | Upload the ISO to BMC eMMC via the Redfish SimpleUpdate API; attach via Redfish VirtualMedia (Grace then sees USB mass-storage devices); set BootSourceOverride to USB (Once / UEFI); reset Grace |
| When to use it | Default — fewest moving parts, no DHCP/TFTP infrastructure | When you already have PXE infrastructure or need a custom `bf.cfg` | Fully out-of-band, no physical access, when you do not want to stand up HTTP/PXE; requires a recent dpu-bmc version |
| Out-of-band? | Console is OOB; the ISO fetch is over the OOB network | Console is OOB; the ISO fetch is over the OOB network | Fully out-of-band end to end |
| Monitor via | `obmc-console-client` over BMC SSH | `obmc-console-client` over BMC SSH | `obmc-console-client` over BMC SSH |

**Post-install (shared across all three methods).** First login with
the image's documented default credentials, then **immediately change
them**; set a strong root password; validate the installed build with
`cat /etc/mlnx-release`; and **detach all virtual media afterward to
avoid boot loops** (Method C especially — a still-attached install ISO
re-enters the installer on the next reset). This skill never prints a
specific default-credential string — it tells the operator to consult
the image's documented defaults and change them at once.

### PLDM firmware update — BMC / NIC firmware / SBIOS / ERoT

BlueField-4 platform firmware (the BMC, the NIC firmware, the SBIOS,
and the ERoT root-of-trust) updates through the **PLDM for Firmware
Update** flow driven over Redfish. The class-level surface:

- **NIC firmware from the host (when required).** Some flows program
  NIC firmware from the host with `flint` before the platform bundle
  is applied; a power cycle activates it. `flint` is a MUTATING
  firmware op — route the burn through
  [`doca-hardware-safety`](../doca-hardware-safety/SKILL.md).
- **Push the bundle.** Upload the `.fwpkg` bundle ({fw-image}) via a
  Redfish multipart `POST` to the UpdateService update-multipart
  endpoint. The BMC stages the contained component images.
- **Monitor the Task.** The push returns a Redfish **Task** resource
  ({task-id}). Healthy progression is `TaskState` Running ->
  Completed with `PercentComplete` climbing to 100; a `Messages`
  Exception or a non-Completed terminal state is failure.
- **Verify the pending images.** `pldmtool fw_update GetFwParams -m
  {eid}` lists each component's `ActiveComponentVersionString` and
  `PendingComponentVersionString`. A staged-but-not-active update
  shows Pending differing from Active; this is expected before
  activation, not after.
- **Activate.** Many components activate only on a power cycle (e.g.
  `ipmitool power cycle`). After activation, re-run GetFwParams and
  confirm Active now equals the target (Pending cleared).
- **Optional clean state.** A BMC factory reset can return the BMC to
  a documented clean state; it is a MUTATING op and is never routine.

The target versions for every component come from the **public
BlueField/DOCA release notes** — this skill never cites a specific
pre-release / "dev drop" firmware version number.

### Grace Ubuntu image + optional cloud-init via Virtual Media

A Grace Ubuntu image can be installed — with an optional cloud-init
seed — entirely through Redfish Virtual Media, as a variant of
Method C above. The load-bearing constraints (the `CIDATA` seed
volume label; the local-eMMC vs remote-HTTPS hosting choice; the
BMC-fixed `image.iso` / `config.iso` URIs; and the
attach → boot → verify → **detach** order, never ejecting
mid-install) are carried inline in the procedure rather than
restated here. See
[`TASKS.md ### grace-ubuntu-cloud-init`](TASKS.md#grace-ubuntu-cloud-init)
for the full step-by-step flow and the cloud-init layer of
[`## Error taxonomy`](#error-taxonomy) for its failure modes.

## Version compatibility

For the canonical DOCA version-detection chain and the four-way match
rule, see [`doca-version`](../doca-version/SKILL.md). The body lives
there; this skill does not duplicate it.

**The BF4-bring-up-specific overlay** is:

- **Every install / firmware target comes from the public release
  notes, never from memory.** The bundle-ISO build, the NIC firmware
  level, the SBIOS / ERoT / BMC component versions are all read from
  the **public BlueField/DOCA release notes** (route via
  [`doca-public-knowledge-map ## Public documentation entry points`](../doca-public-knowledge-map/SKILL.md#public-documentation-entry-points)).
  Quoting a specific pre-release firmware version string from memory is
  the canonical hallucination failure for this skill and would also
  leak non-public information.
- **The Redfish FirmwareInventory is the post-update version anchor.**
  After a PLDM update, the per-component versions reported by the
  Redfish FirmwareInventory (and by `pldmtool fw_update GetFwParams -m
  {eid}`) are the authoritative "what is actually running" picture —
  compare them against the release-notes target, not against an
  assumed value.
- **The installed-build anchor on Grace is `cat /etc/mlnx-release`.**
  After the ISO install, the build string in `/etc/mlnx-release` is
  the Grace-side install anchor; it feeds the four-way match owned by
  [`doca-version`](../doca-version/SKILL.md) once the operator hands
  off to an application-deployment skill.
- **BF4 is not BF3.** The methods, the Grace naming, and the
  dpu-bmc / Redfish surface are BlueField-4-specific. Do not assume a
  BF3 procedure transfers; BF3 bring-up is `doca-bf3-deployment`.

## Error taxonomy

BF4-bring-up errors fall into SIX layers, each with its own evidence.
The agent walks the layers in this order; conflating them wastes time
and blames the wrong layer.

1. **Boot-source layer — the ISO never starts.** Symptoms: UEFI HTTP
   Boot reports it cannot fetch the Boot URI; the PXE menu never
   appears; the Virtual Media device is not offered in the Boot
   Manager. Causes: {iso-uri} unreachable from the OOB network; the
   wrong OOB MAC / net device selected; HTTP server not serving the
   ISO; (Method C) the media was not attached before the reset.
   Resolution: confirm {iso-uri} is reachable and serves the ISO;
   confirm the OOB MAC matches the device in Network Device List;
   re-check the Boot Manager entry. Evidence: the OOB console via
   `obmc-console-client`.
2. **Virtual-media-attach layer (Method C / Grace-Ubuntu).** Symptoms:
   the Redfish InsertMedia action fails; the media shows
   `"Inserted": false` after attach; Grace does not see a USB
   mass-storage device. Causes: the dpu-bmc version is too old to
   support Redfish Virtual Media; the image was never transferred to
   eMMC (Method C local); a remote HTTPS source lacks unauthenticated
   access, lacks HTTP `HEAD` support, or its certificate is not in the
   BMC truststore; the combined local payload exceeds 5 GB. Resolution:
   re-check the SimpleUpdate transfer, the hosting preconditions, and
   the `"Inserted"` field.
3. **Firmware-Task layer.** Symptoms: the multipart `POST` is
   accepted but the returned Task stays at `TaskState` Running, or it
   ends with a `Messages` Exception, or `PercentComplete` stalls.
   Causes: a corrupt or wrong-architecture `.fwpkg`; a component the
   bundle targets is not present; the BMC is busy with another update.
   Resolution: read the Task resource's `Messages`; do NOT re-push
   blindly; treat a stalled Task as HIGH-STAKES per
   [`## Safety policy`](#safety-policy).
4. **Activation layer — staged but not running.** Symptoms:
   `pldmtool fw_update GetFwParams -m {eid}` still shows Pending
   differing from Active after the Task reported Completed. Cause:
   the component activates only on a power cycle and the operator has
   not cycled yet. Resolution: power cycle (e.g. `ipmitool power
   cycle`), then re-run GetFwParams and confirm Active equals the
   target.
5. **Cloud-init layer (Grace-Ubuntu path).** Symptoms: the OS installs
   but the cloud-init customization (user, SSH key, hostname) did not
   apply. Causes: the config ISO volume label is not exactly `CIDATA`;
   the config ISO was not attached under the fixed `config.iso` URI;
   the seed was malformed. Resolution: re-check the CIDATA label and
   that both `image.iso` and `config.iso` were attached.
6. **Boot-loop layer — media never detached.** Symptoms: Grace
   re-enters the installer (or the Virtual Media boot) on every reset.
   Cause: the install ISO / virtual media was left attached and
   BootSourceOverride re-selects it. Resolution: detach all virtual
   media, confirm `"Inserted": false`, and clear / re-point the boot
   override. This is the single most common day-1 self-inflicted loop.

## Observability

Documented observability surfaces, in the order the agent reaches for
them. Healthy means the layers agree.

- **OOB console layer (FIRST).** The DPU's serial console reached via
  BMC SSH + `obmc-console-client` is the first place to look — it shows
  the UEFI menus, the installer progress, and the Grace boot. The
  agent does not invent console markers; it reads what the installer
  and UEFI actually print.
- **Redfish Task layer (firmware updates).** The Task resource
  ({task-id}) returned by the multipart push reports `TaskState`,
  `PercentComplete`, and `Messages`. Running -> Completed with no
  Exception is healthy; a non-Completed terminal state or an Exception
  message is the failure signal.
- **PLDM component layer.** `pldmtool fw_update GetFwParams -m {eid}`
  reports each component's Active vs Pending version string — the
  authoritative "did the update actually take" picture after a power
  cycle.
- **Redfish FirmwareInventory layer.** The FirmwareInventory reports
  the per-component running versions; compare against the
  release-notes target. The Virtual Media resource's `"Inserted"`
  field reports attach/detach state (`true` when mounted, `false` when
  detached).
- **Grace install layer.** Once Grace boots, `cat /etc/mlnx-release`
  reports the installed build string — the install-success anchor.

Cross-cutting host-side debug (kernel, drivers, PCIe link state) on
the host that talks to the BF4 lives in
[`doca-debug TASKS.md ## debug`](../doca-debug/TASKS.md#debug); this
skill names only the BF4-bring-up-specific surfaces.

## Safety policy

> **Overlay on the bundle-wide hardware-safety meta-policy.** The rules
> below are this skill's per-artifact overlay on the cross-cutting
> rules in
> [`doca-hardware-safety` CAPABILITIES.md ## Safety policy](../doca-hardware-safety/CAPABILITIES.md#safety-policy)
> (specifically
> [### Per-artifact overlay pattern](../doca-hardware-safety/CAPABILITIES.md#per-artifact-overlay-pattern)).
> When the two layers disagree, the stricter wins; when either layer
> says STOP, the agent stops.

- **Every PLDM burn, ISO reflash, power cycle, and BMC factory reset
  is a MUTATING hardware op.** The change-application discipline
  (preflight inventory, OOB console reachability confirmed BEFORE the
  change, maintenance window, rollback path) is owned by
  [`doca-hardware-safety ## modify`](../doca-hardware-safety/TASKS.md#modify);
  this skill loads it ALONGSIDE and never redefines it. The OOB-access
  precondition specifically maps to
  [`doca-hardware-safety ## Out-of-band access classes`](../doca-hardware-safety/CAPABILITIES.md#out-of-band-access-classes).
- **Never print a real credential.** No BMC password, no image default
  password, no recommended credential string is ever emitted. The
  agent may say "use the image's documented default credentials" and
  MUST tell the operator to change them immediately; it uses
  placeholders ({bmc-user}, {bmc-password}) for everything
  site-specific.
- **Always detach virtual media when done.** A still-attached install
  ISO re-enters the installer on the next reset (the boot-loop layer
  of the error taxonomy). Confirm `"Inserted": false` before declaring
  the bring-up complete, and never eject mid-install.
- **Public information only.** Any NVIDIA URL must be on a public host
  (`docs.nvidia.com`, `developer.nvidia.com`, `catalog.ngc.nvidia.com`,
  `ngc.nvidia.com`, `forums.developer.nvidia.com`, `nvcr.io`). Never
  emit an internal hostname, NFS path, or a pre-release firmware
  version string. Redfish / PLDM / UEFI are public standards and may
  be described freely.

## Public-source pointer

The canonical public sources for BF4 day-1 bring-up are:

- The **public BlueField/DOCA documentation and release notes** on
  `docs.nvidia.com`, reachable through
  [`doca-public-knowledge-map ## Public documentation entry points`](../doca-public-knowledge-map/SKILL.md#public-documentation-entry-points)
  for the install methods, the dpu-bmc Redfish surface, the Grace
  Ubuntu image path, and the per-component firmware targets. Do not
  invent exact doc anchors — point generally and route via the map.
- The **DMTF Redfish** and **PLDM for Firmware Update** specifications
  for the UpdateService, Task, VirtualMedia, and FirmwareInventory
  semantics and for `pldmtool` behavior.
- The **UEFI specification** for HTTP Boot, PXE Boot, the Boot
  Manager, and Device Manager / Network Device List behavior.

Verify that the guide and release-notes versions match the operator's
target BF4 — install methods, the dpu-bmc Redfish surface, and
firmware component layouts evolve, so anything quoted from memory is
suspect.
