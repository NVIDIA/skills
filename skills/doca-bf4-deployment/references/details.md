# doca-bf4-deployment — reference detail

Moved out of `SKILL.md` to keep the loader under the per-file size budget. This is supporting detail, not routing logic.

## Example questions this skill answers well

The CLASSES of BF4-bring-up questions this skill is built to answer,
each with one worked example. The class is the load-bearing piece; the
worked example is one instance.

- **"I have a brand-new BlueField-4 and need to install the DOCA
  bundle ISO onto it. What are my options and which should I use?"** —
  worked example: *"I do not want to set up PXE; can I just point it at
  an ISO over HTTP?"*. Answered by the install-method table in
  [`CAPABILITIES.md ## Capabilities and modes`](../CAPABILITIES.md#capabilities-and-modes)
  (UEFI HTTP Boot is the recommended default) + the step-by-step flow
  in [`TASKS.md ### method-a-uefi-http-boot`](../TASKS.md#method-a-uefi-http-boot).
- **"I want to install the DPU completely out-of-band — no physical
  access, no HTTP server to stand up. Is there a way?"** — worked
  example: *"attach the ISO straight from the BMC via Redfish"*.
  Answered by the Redfish Virtual Media row in
  [`CAPABILITIES.md ## Capabilities and modes`](../CAPABILITIES.md#capabilities-and-modes)
  + [`TASKS.md ### method-c-redfish-virtual-media`](../TASKS.md#method-c-redfish-virtual-media).
- **"I pushed a firmware bundle through Redfish and the Task just sits
  at Running. Is it stuck?"** — worked example: *"the multipart POST
  was accepted, PercentComplete is not moving"*. Answered by the
  firmware-Task layer in
  [`CAPABILITIES.md ## Error taxonomy`](../CAPABILITIES.md#error-taxonomy)
  + [`TASKS.md ## debug`](../TASKS.md#debug) layer 3, with the HIGH-STAKES
  "do not re-push blindly" rule in
  [`CAPABILITIES.md ## Safety policy`](../CAPABILITIES.md#safety-policy).
- **"The firmware Task said Completed but the version did not change.
  What did I miss?"** — worked example: *"pldmtool shows a Pending
  version that never becomes Active"*. Answered by the activation layer
  in [`CAPABILITIES.md ## Error taxonomy`](../CAPABILITIES.md#error-taxonomy)
  + the power-cycle-to-activate step in
  [`TASKS.md ### pldm-firmware-update`](../TASKS.md#pldm-firmware-update).
- **"I want a Grace Ubuntu image with my SSH key and hostname baked in
  via cloud-init, installed over Redfish. How?"** — worked example:
  *"my CIDATA seed is not being picked up"*. Answered by the Virtual
  Media + cloud-init rules in
  [`CAPABILITIES.md ## Capabilities and modes`](../CAPABILITIES.md#capabilities-and-modes)
  + [`TASKS.md ### grace-ubuntu-cloud-init`](../TASKS.md#grace-ubuntu-cloud-init),
  with the CIDATA / fixed-`config.iso`-URI gotcha in the cloud-init
  layer of the error taxonomy.
- **"My BlueField-4 keeps re-entering the installer every time it
  resets. Why?"** — worked example: *"I finished the install but it
  loops back into setup"*. Answered by the boot-loop layer in
  [`CAPABILITIES.md ## Error taxonomy`](../CAPABILITIES.md#error-taxonomy)
  (media was never detached) + the detach-and-verify-`"Inserted":
  false` step in [`TASKS.md ### post-install`](../TASKS.md#post-install).

## What this skill deliberately does not ship

This skill is **agent guidance** for the documented BF4 day-1 bring-up
flows, not a credentials / images / firmware bundle. To keep the
boundary clean, it deliberately does not contain — and pull requests
should not add:

- **BlueField-3 bring-up.** BF3 is a different platform with different
  methods; route to `doca-bf3-deployment`. This skill is
  BlueField-4-specific (Grace, the dpu-bmc Redfish surface, the BF4
  install methods).
- **Container or bare-metal application launch.** Running a DOCA
  service container or a DOCA-linked binary on an already-working
  BlueField is owned by
  [`doca-container-deployment`](../../doca-container-deployment/SKILL.md)
  and
  [`doca-bare-metal-deployment`](../../doca-bare-metal-deployment/SKILL.md).
  This skill stops at "Grace installed, firmware at the target level."
- **The hardware-change meta-policy.** The preflight / OOB-console /
  maintenance-window / rollback discipline that wraps every PLDM burn,
  ISO reflash, power cycle, and BMC factory reset is owned by
  [`doca-hardware-safety`](../../doca-hardware-safety/SKILL.md). This
  skill cross-links it and never redefines it.
- **Real credentials, internal hostnames, internal paths, or
  pre-release firmware version strings.** This skill uses only
  placeholders ({bmc-ip}, {bmc-user}, {bmc-password}, {iso-uri},
  {http-server-ip}, {device-name}, {fw-image}, {eid}, {task-id}) for
  anything site-specific. It never prints a default-credential string,
  never names an internal hostname or NFS path, and never cites a
  specific pre-release firmware version — those come from the public
  release notes and the operator's own environment. This is the
  load-bearing public-safety rule for this skill.
- **A `samples/`, `templates/`, `images/`, or `firmware/` subtree of
  any kind.** A mock or partial artifact in this skill's tree, even one
  labeled "reference", is misleading: operators will read it as
  production-ready, and a stored credential or firmware blob would fail
  the bundle's public-release gates.

## Related skills

- [`doca-hardware-safety`](../../doca-hardware-safety/SKILL.md) — the
  CRITICAL cross-cutting meta-policy for any change touching DPU / NIC
  hardware state. Every PLDM firmware burn, ISO reflash, power cycle,
  and BMC factory reset in this skill is a MUTATING op; the
  change-application discipline lives there and this skill loads it
  ALONGSIDE, never duplicating it. This skill's `## Safety policy`
  overlays that meta-policy with BF4-specific rules (never print a
  credential, always detach media, public information only).
- [`doca-bare-metal-deployment`](../../doca-bare-metal-deployment/SKILL.md)
  — the downstream skill for launching a DOCA-linked binary directly
  on the BlueField Arm once this skill has brought Grace up. Its
  `## bluefield-lifecycle` anchor covers the BF3-era BFB / RShim /
  TMFIFO lifecycle; this skill is the BF4 BMC-driven day-1 analog.
- [`doca-container-deployment`](../../doca-container-deployment/SKILL.md)
  — the downstream skill for deploying a DOCA service container on the
  BlueField once Grace is up.
- [`doca-setup`](../../doca-setup/SKILL.md) — env preparation on the
  installed Grace OS (install verification, hugepages, pkg-config
  path, devlink mode, representor visibility). This skill hands off to
  it once the platform is up.
- [`doca-version`](../../doca-version/SKILL.md) — the four-way version
  match rule. This skill's `## Version compatibility` cross-links the
  body there and adds only the BF4 overlay (install / firmware targets
  come from the public release notes; the Redfish FirmwareInventory
  and `pldmtool GetFwParams` are the post-update anchors; `cat
  /etc/mlnx-release` is the Grace install anchor).
- [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md)
  — the routing table to the public BlueField/DOCA documentation,
  release notes, and download surface. This skill does not duplicate
  URLs or invent exact doc anchors; it points at the map. The
  BlueField BSP / BMC rows there are the route for any
  externally-productized layer beyond this skill's day-1 carve-out.
- `doca-bf3-deployment` — the sibling skill for BlueField-3 day-1
  bring-up. This skill is BF4-only; BF3 questions route there.
