# DOCA BlueField-4 deployment — Tasks

**Where to start:** The verb order is `configure -> build -> modify ->
run -> test -> debug`. For BF4 day-1 platform bring-up, `build` and
`modify` are *routing stubs* — there is no application artifact to
build or sample to modify in a platform bring-up; those questions live
in the application-deployment skills and
[`doca-programming-guide`](../doca-programming-guide/SKILL.md). This
skill owns getting a bare BlueField-4 to a Grace OS installed and
firmware at the target level, driven from the BMC. The `## test` verb
is the post-install / post-update verification sweep.

These verbs cover the in-scope BF4-bring-up workflows for an external
operator standing up a new BlueField-4 through its BMC. Every step
assumes the operator has consulted the live public BlueField/DOCA
documentation and release notes (reachable through
[`doca-public-knowledge-map ## Public documentation entry points`](../doca-public-knowledge-map/SKILL.md#public-documentation-entry-points))
and the Redfish / PLDM / UEFI standards; this file prescribes the
*order* and *what to look up where*, not a copy-paste runbook. Every
MUTATING step (PLDM burn, ISO reflash, power cycle, BMC factory reset)
loads [`doca-hardware-safety`](../doca-hardware-safety/SKILL.md)
ALONGSIDE.

## configure

Preparing to bring up the BlueField-4 and confirming every
precondition BEFORE any boot, transfer, or burn.

1. **Confirm out-of-band BMC reachability.** The whole flow is driven
   from the BMC. Confirm BMC SSH works ({bmc-user} / {bmc-password},
   supplied by the operator — never invented), and that the DPU's OOB
   serial console is reachable via `obmc-console-client`. Without an
   OOB path the bar for proceeding is "stop, establish OOB first" per
   [`doca-hardware-safety ## Out-of-band access classes`](../doca-hardware-safety/CAPABILITIES.md#out-of-band-access-classes).
2. **Host the ISO.** Download the BlueField/DOCA bundle ISO (and, for
   the Grace-Ubuntu path, the Grace Ubuntu image and any cloud-init
   config ISO) from the public NVIDIA download surface, and host it at
   {iso-uri} on your own HTTP/HTTPS server. For a remote HTTPS source
   used by Redfish Virtual Media, confirm it allows unauthenticated
   access, supports HTTP `HEAD`, and that the BMC has the server
   certificate in its truststore.
3. **Pick the install method.** Per the install-method table in
   [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes):
   UEFI HTTP Boot (recommended — fewest moving parts), PXE Boot (when
   you already have DHCP+TFTP or need a custom `bf.cfg`), or Redfish
   Virtual Media (fully out-of-band; needs a recent dpu-bmc version).
   The choice feeds [`## run`](#run).
4. **Capture the target versions.** Read the target bundle-ISO build,
   NIC firmware level, and SBIOS / ERoT / BMC component versions from
   the **public BlueField/DOCA release notes** (route via
   [`doca-public-knowledge-map`](../doca-public-knowledge-map/SKILL.md)).
   Do NOT quote a pre-release firmware version from memory; record the
   release-notes targets as the verification baseline used in
   [`## test`](#test).
5. **Plan the rollback and the maintenance window.** Every burn /
   reflash / power cycle is MUTATING. Capture the pre-change state
   (current FirmwareInventory, current `/etc/mlnx-release` if Grace is
   already up), agree a maintenance window, and confirm the OOB path
   survives a failed change — all per
   [`doca-hardware-safety ## modify`](../doca-hardware-safety/TASKS.md#modify),
   loaded ALONGSIDE.

## build

BF4 platform bring-up has **no application artifact to build**. The
binaries, service containers, and DOCA libraries a workload uses are
not built here.

- If the user is asking how to build a DOCA-linked binary or a service
  to deploy once Grace is up, route to
  [`doca-programming-guide ## build`](../doca-programming-guide/TASKS.md#build),
  then to
  [`doca-bare-metal-deployment`](../doca-bare-metal-deployment/SKILL.md)
  or
  [`doca-container-deployment`](../doca-container-deployment/SKILL.md)
  to deploy it.
- If the user is asking how to assemble the bundle ISO or the
  firmware `.fwpkg` themselves, that is NVIDIA-internal release
  engineering — out of scope. The operator downloads the published
  artifacts from the public NVIDIA download surface.
- The H2 stays so the verb order is intact; the answer is *there is no
  build step in platform bring-up — you download published artifacts
  and install them.*

## modify

BF4 platform bring-up has **no "modify a sample program" workflow**.
The bring-up analog of "modify" is **re-walking the bring-up after a
change**, and any change to firmware *configuration* leaves this verb:

1. **A firmware-config change is a hardware-state change.** Adjusting
   NV-config / mode toggles (e.g. via `mlxconfig set`) or burning new
   firmware is MUTATING and is owned by
   [`doca-hardware-safety ## modify`](../doca-hardware-safety/TASKS.md#modify);
   this skill does not redefine that discipline. Control returns to
   [`## configure`](#configure) once the change is complete and
   verified.
2. **A re-install or re-flash is a fresh bring-up.** Re-pointing the
   boot path at a new ISO, or re-pushing a `.fwpkg`, re-opens the
   verification sweep — re-walk [`## run`](#run) then
   [`## test`](#test).
3. **A change to the cloud-init seed is a fresh Grace-Ubuntu install.**
   Editing the CIDATA seed means re-attaching `config.iso` and
   re-installing; re-walk
   [`### grace-ubuntu-cloud-init`](#grace-ubuntu-cloud-init).
4. **Modifying a workload's source belongs elsewhere.** Source changes
   to a DOCA application route to
   [`doca-programming-guide ## modify`](../doca-programming-guide/TASKS.md#modify);
   they are not a platform-bring-up concern.
5. The H2 stays for verb-order parity; the substantive routing is
   above.

## run

Bringing the BlueField-4 up: install the bundle ISO onto Grace via the
chosen method, update platform firmware via PLDM when required, and
(optionally) install a Grace Ubuntu image with cloud-init. Each
sub-anchor below is one documented flow. Reach the OOB console via BMC
SSH + `obmc-console-client` before starting any of them, and load
[`doca-hardware-safety`](../doca-hardware-safety/SKILL.md) ALONGSIDE.

### method-a-uefi-http-boot

The recommended OS-install method — no PXE / DHCP / TFTP needed.

1. Host the bundle ISO at {iso-uri} on your HTTP server.
2. Reach the DPU OOB console via BMC SSH + `obmc-console-client`.
3. Reboot Grace and enter UEFI.
4. Go Device Manager -> Network Device List -> select the OOB MAC ->
   HTTP Boot Configuration -> set the Boot URI to {iso-uri} -> save.
5. Go Boot Manager -> select the UEFI HTTP entry -> boot the
   installer.
6. Watch the install on the OOB console; then do the shared
   post-install in [`### post-install`](#post-install).

### method-b-pxe

Use when you already have PXE infrastructure or need a custom `bf.cfg`.

1. Stand up / confirm a PXE server: DHCP + TFTP + the ISO.
2. Reach the DPU OOB console via BMC SSH + `obmc-console-client`.
3. In Boot Manager, set next boot to the OOB IPv4 net device.
4. Select the DOCA arm64 menu, then the ISO, and boot the installer.
5. Watch the install on the OOB console; then do
   [`### post-install`](#post-install).

### method-c-redfish-virtual-media

Fully out-of-band; needs a recent dpu-bmc version. No physical access.

1. Upload the ISO to the BMC eMMC via the Redfish SimpleUpdate API.
2. Attach it via Redfish VirtualMedia; Grace then sees USB
   mass-storage devices. Confirm the media reports `"Inserted": true`.
3. Set BootSourceOverride to USB (Once / UEFI).
4. Reset Grace; monitor via `obmc-console-client`.
5. Do [`### post-install`](#post-install), and **detach the media**
   when done.

### pldm-firmware-update

Update BMC / NIC firmware / SBIOS / ERoT via PLDM over Redfish. Every
step here is MUTATING — `doca-hardware-safety` is loaded ALONGSIDE.

1. **NIC firmware from the host (when required).** Program NIC
   firmware from the host with `flint` if the flow calls for it, then
   power cycle. Route the burn through
   [`doca-hardware-safety ## modify`](../doca-hardware-safety/TASKS.md#modify).
2. **Push the bundle.** Upload the `.fwpkg` ({fw-image}) via a Redfish
   multipart `POST` to the UpdateService update-multipart endpoint.
3. **Monitor the Task.** Poll the returned Task ({task-id}):
   `TaskState` Running -> Completed, `PercentComplete` to 100; a
   `Messages` Exception or a non-Completed terminal state is failure
   (drop to [`## debug`](#debug) layer 3).
4. **Verify pending images.** `pldmtool fw_update GetFwParams -m
   {eid}` and compare `ActiveComponentVersionString` against
   `PendingComponentVersionString` per component — Pending differs
   from Active before activation.
5. **Activate.** Power cycle to activate (e.g. `ipmitool power
   cycle`); re-run GetFwParams and confirm Active now equals the
   release-notes target. Optionally BMC factory reset for a clean
   state (MUTATING — route through `doca-hardware-safety`).

### grace-ubuntu-cloud-init

Install a Grace Ubuntu image (optionally seeded by cloud-init) via
Redfish Virtual Media.

1. **Prepare the cloud-init seed (optional).** Build the seed as an
   ISO with volume label `CIDATA`.
2. **Choose hosting.** Local — copy the image and seed onto BMC eMMC,
   up to 5 GB combined. Remote — host on an HTTPS server that allows
   unauthenticated access and supports HTTP `HEAD`, with the server
   certificate in the BMC truststore.
3. **Transfer + attach.** Transfer via Redfish SimpleUpdate; attach
   via Redfish VirtualMedia. The URIs are fixed to `image.iso` and
   `config.iso` — original filenames do not matter. Confirm
   `"Inserted": true`.
4. **Boot.** Set BootSourceOverride to USB and reset Grace, OR
   manually pick the OpenBMC Virtual Media Device from the UEFI Boot
   Manager.
5. **Verify + detach.** Check component versions via the Redfish
   FirmwareInventory; eject / detach when done and confirm
   `"Inserted": false`. **Never eject mid-install.**

### post-install

Shared across every OS-install method:

1. First login with the image's **documented default credentials**,
   then **immediately change them**.
2. Set a strong root password.
3. Validate the installed build with `cat /etc/mlnx-release`.
4. **Detach all virtual media** (Method C / Grace-Ubuntu) to avoid a
   boot loop; confirm `"Inserted": false`.
5. Hand off to [`## test`](#test) for the full verification sweep.

## test

BF4 bring-up has no compile-and-unit-test workflow — verification is
operational against the BMC and Grace. **`## test` is the sweep run
after every install or firmware change**; skipping it after a change
is the failure mode this verb replaces.

1. **OS install took.** `cat /etc/mlnx-release` on Grace reports the
   release-notes target build string, and the OOB console shows a
   clean boot to a login prompt. Default credentials have been
   changed.
2. **Firmware update took.** The Redfish Task reported `TaskState`
   Completed with `PercentComplete` 100 and no Exception; after the
   activation power cycle, `pldmtool fw_update GetFwParams -m {eid}`
   shows Active equal to the target (Pending cleared) for every
   updated component.
3. **Versions match the release notes.** The Redfish FirmwareInventory
   per-component versions equal the targets captured in
   [`## configure`](#configure) step 4 — not an assumed value.
4. **Virtual media detached.** Any Virtual Media resource reports
   `"Inserted": false`, and Grace does not re-enter the installer on a
   subsequent reset (no boot loop).
5. **Cloud-init applied (Grace-Ubuntu path only).** The customization
   from the CIDATA seed (user, SSH key, hostname) is present on the
   booted Grace OS.
6. **Capture the bring-up snapshot.** Record which method was used,
   the install build string, the per-component firmware versions, and
   the OOB-console / Task evidence — the artifact that lets a later
   debug session skip rediscovery and is the rollback baseline.

If any check fails, drop to [`## debug`](#debug) at the matching
layer; do not declare the bring-up complete on a single green signal.

## debug

Layered diagnosis. Walk the layers in this order; do not skip down
without clearing the layer above. The six layers match
[`CAPABILITIES.md ## Error taxonomy`](CAPABILITIES.md#error-taxonomy).

1. **Boot-source layer.** ISO never starts: UEFI HTTP Boot cannot
   fetch the Boot URI, the PXE menu never appears, or the Virtual
   Media device is absent from the Boot Manager. Confirm {iso-uri} is
   reachable from the OOB network and serves the ISO; confirm the OOB
   MAC matches the Network Device List entry; (Method C) confirm the
   media was attached before the reset. Evidence: the OOB console via
   `obmc-console-client`.
2. **Virtual-media-attach layer.** InsertMedia fails or
   `"Inserted": false`; Grace sees no USB mass-storage device. Check
   the dpu-bmc version supports Redfish Virtual Media; confirm the
   SimpleUpdate transfer completed (Method C local) or that the remote
   HTTPS source allows unauthenticated access, supports HTTP `HEAD`,
   and has its certificate in the BMC truststore; confirm the combined
   local payload is within 5 GB.
3. **Firmware-Task layer.** The multipart `POST` was accepted but the
   Task stays Running, ends with an Exception, or stalls
   `PercentComplete`. Read the Task `Messages`; verify the `.fwpkg`
   is the correct, uncorrupted bundle and targets present components;
   do NOT re-push blindly — a stalled firmware Task is HIGH-STAKES per
   [`CAPABILITIES.md ## Safety policy`](CAPABILITIES.md#safety-policy).
4. **Activation layer.** `pldmtool fw_update GetFwParams -m {eid}`
   still shows Pending differing from Active after the Task completed:
   the component needs a power cycle to activate. Power cycle (e.g.
   `ipmitool power cycle`), then re-verify.
5. **Cloud-init layer (Grace-Ubuntu).** The OS installed but the
   cloud-init seed did not apply: confirm the seed ISO volume label is
   exactly `CIDATA` and that both `image.iso` and `config.iso` were
   attached under their fixed URIs.
6. **Boot-loop layer.** Grace re-enters the installer / Virtual Media
   on every reset: the install media was never detached. Detach all
   virtual media, confirm `"Inserted": false`, and clear / re-point
   the boot override. This is the most common day-1 self-inflicted
   loop.

For any cross-cutting host-side issue on the machine that talks to the
BF4 (kernel, drivers, PCIe link state), drop to
[`doca-debug TASKS.md ## debug`](../doca-debug/TASKS.md#debug). For any
mutating recovery action (re-burn, reflash, factory reset), route to
[`doca-hardware-safety`](../doca-hardware-safety/SKILL.md).

## Deferred task verbs

- **BlueField-3 (BF3) bring-up** — out of scope here. This skill is
  BlueField-4-specific (Grace, the dpu-bmc Redfish surface, the BF4
  install methods). Route to `doca-bf3-deployment`.
- **Running an application on an already-working BlueField** —
  out of scope. A DOCA service container routes to
  [`doca-container-deployment`](../doca-container-deployment/SKILL.md);
  a DOCA-linked binary launched directly routes to
  [`doca-bare-metal-deployment`](../doca-bare-metal-deployment/SKILL.md).
  This skill hands off once Grace is installed and firmware is at the
  target level.
- **Library-API questions** (constructing a DOCA-Flow pipe, an RDMA
  queue pair, interpreting a `DOCA_ERROR_*` code) — out of scope.
  Route to the matching `libs/<library>` skill via
  [`doca-public-knowledge-map ## Library- and module-specific guides`](../doca-public-knowledge-map/SKILL.md#library--and-module-specific-guides).
- **Env preparation on the installed Grace OS** (hugepages,
  `PKG_CONFIG_PATH` / `LD_LIBRARY_PATH`, devlink mode, representor
  visibility) — out of scope here. Route to
  [`doca-setup ## configure`](../doca-setup/TASKS.md#configure); this
  skill stops at "Grace installed, firmware at target."
- **The hardware-change meta-policy itself** (preflight, OOB-console
  discipline, maintenance window, rollback for any PLDM burn / ISO
  reflash / power cycle / BMC factory reset) — owned by
  [`doca-hardware-safety ## modify`](../doca-hardware-safety/TASKS.md#modify).
  This skill always loads it ALONGSIDE and adds only the BF4
  sequencing; it never redefines the meta-policy.
- **Fleet-scale / orchestrated DPU provisioning** (DOCA Platform
  Framework, NVIDIA Network Operator) — out of scope per the bundle's
  non-goals. Route via
  [`doca-public-knowledge-map ## Externally-productized DOCA software`](../doca-public-knowledge-map/SKILL.md#externally-productized-doca-software--not-in-this-bundle-but-here-is-where-to-route).
