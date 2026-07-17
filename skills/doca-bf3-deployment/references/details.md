# doca-bf3-deployment — reference detail

Moved out of `SKILL.md` to keep the loader under the per-file size budget. This is supporting detail, not routing logic.

## Example questions this skill answers well

The CLASSES of BF3-bring-up questions this skill is built to answer,
each with one worked example. The class is the load-bearing piece;
the worked example is one instance.

- **"I have a BlueField-3 and a BFB image — how do I actually push
  it to the DPU?"** — worked example: *"I downloaded the BFB from
  the DOCA Downloads page and have a host with the RShim daemon
  running; walk me through pushing it with `bfb-install` the right
  way"*. Answered by the RShim/BFB transport surface in
  [`CAPABILITIES.md ## Capabilities and modes`](../CAPABILITIES.md#capabilities-and-modes)
  + the push sequence in [`TASKS.md ## run`](../TASKS.md#run).
- **"Is DOCA supposed to be on the host or on the Arm side? Which
  one do I install?"** — worked example: *"I installed DOCA-Host on
  my x86 server but my DOCA app still cannot see the BlueField — did
  I install it in the wrong place?"*. Answered by the install-side
  table in
  [`CAPABILITIES.md ## Capabilities and modes`](../CAPABILITIES.md#capabilities-and-modes)
  + the recognition step in
  [`TASKS.md ## configure`](../TASKS.md#configure) step 1.
- **"`bfb-install` exited 0 but my BF3 never came back. Where do I
  start?"** — worked example: *"the console showed `Ubuntu
  installation completed` then `INFO[MISC]: NIC firmware update
  failed`, but the installer still returned 0; now the DPU never
  reaches `DPU is ready`"*. Answered by the exit-0-is-not-success
  rule in [`TASKS.md ## run`](../TASKS.md#run) step 3 + the six-state
  classifier in
  [`CAPABILITIES.md ## Error taxonomy`](../CAPABILITIES.md#error-taxonomy)
  and [`TASKS.md ## debug`](../TASKS.md#debug).
- **"`ping 192.168.100.2` works but I still cannot ssh to the
  BlueField — what is going on?"** — worked example: *"the TMFIFO
  ping succeeds so I assumed the DPU was reachable, but every ssh
  hangs"*. Answered by the `ip route get`-before-`ping`
  local-loopback gotcha in the TMFIFO surface of
  [`CAPABILITIES.md ## Capabilities and modes`](../CAPABILITIES.md#capabilities-and-modes)
  + [`TASKS.md ## run`](../TASKS.md#run) step 4.
- **"My host PFs vanished after a BFB push — `lspci` shows them but
  `ip link` does not."** — worked example: *"after the push the
  BlueField PFs are in `lspci -d 15b3:` but no netdevs appear and
  DOCA programs cannot attach by representor name"*. Answered by the
  `arm-ok-host-pfs-unbound` state in
  [`CAPABILITIES.md ## Error taxonomy`](../CAPABILITIES.md#error-taxonomy)
  + the host PF rebind sequence in
  [`TASKS.md ## test`](../TASKS.md#test) step 2.
- **"I need my BF3 in separated-host (NIC) mode instead of DPU
  mode."** — worked example: *"the card boots in DPU mode and I need
  the host to own the ports directly"*. Answered by the DPU-mode
  surface in
  [`CAPABILITIES.md ## Capabilities and modes`](../CAPABILITIES.md#capabilities-and-modes)
  — the decision belongs at BFB-install time in
  [`TASKS.md ## configure`](../TASKS.md#configure) step 7, and the
  `mlxconfig set` burn itself routes to
  [`doca-hardware-safety`](../../doca-hardware-safety/SKILL.md).
- **"How do I verify the install actually took?"** — worked example:
  *"the push finished; how do I confirm which BFB landed and that
  the versions line up?"*. Answered by `cat /etc/mlnx-release` +
  `bfver` and the four-way re-close in
  [`CAPABILITIES.md ## Version compatibility`](../CAPABILITIES.md#version-compatibility)
  + [`TASKS.md ## test`](../TASKS.md#test) steps 3 and 5.

## What this skill deliberately does not ship

This skill is **agent guidance**, not a templates / sample-BFB /
sample-`bf.cfg` bundle. To keep the boundary clean, it deliberately
does not contain — and pull requests should not add:

- **Pre-baked BFB images, `bf.cfg` files, or `bfb-install`
  command lines.** The BFB is the operator's own downloaded image;
  the `bf.cfg` is composed against the live BSP-manual schema at push
  time; the `bfb-install` invocation is derived from `--help` on the
  installed tool. Shipping a ready-to-run `bf.cfg` or a flag set the
  operator might paste unmodified is the load-bearing first-run
  failure for this skill.
- **`bfb-install` flags, BFB filenames, `/dev/rshim<N>` paths,
  `bf.cfg` schema keys, `mlxconfig` mode parameters, or the TMFIFO
  subnet invented from memory.** The public BlueField Platform
  Software Manual, the MFT manual, and `--help` on the installed
  tool are the authoritative sources. Inventing a flag, a key, or a
  subnet from generic intuition or a previous BF2's behaviour is the
  canonical hallucination failure mode.
- **BlueField-4 bring-up.** The BMC-Redfish provisioning path is the
  sibling skill
  [`doca-bf4-deployment`](../../doca-bf4-deployment/SKILL.md), the BF4
  equivalent of this skill; this skill is BF3-only (the classic
  RShim/BFB path).
- **The app-launch and container-deploy workflows.** Running a
  DOCA-linked binary on a healthy BF3 belongs to
  [`doca-bare-metal-deployment`](../../doca-bare-metal-deployment/SKILL.md);
  deploying a DOCA service container belongs to
  [`doca-container-deployment`](../../doca-container-deployment/SKILL.md).
  This skill hands off once the BF3 is healthy.
- **The hardware-state-change meta-policy.** The
  preflight / OOB-console / maintenance-window / rollback discipline
  wrapping any BFB reflash, `mlxconfig set`, or firmware burn is
  owned by
  [`doca-hardware-safety`](../../doca-hardware-safety/SKILL.md); this
  skill cross-links it and adds only BF3-specific sequencing, never
  redefining the meta-policy.
- **A `samples/`, `templates/`, or `reference/` subtree of any
  kind.** A mock or incomplete artifact, even one labeled
  "reference", is misleading: operators will read it as
  production-ready.

## Related skills

- [`doca-bf4-deployment`](../../doca-bf4-deployment/SKILL.md) — the
  BF4 equivalent of this skill. BF3 uses the classic RShim/BFB path
  (this skill); BF4 uses the BMC-Redfish provisioning path (that
  skill). Same lifecycle shape, different platform transport; route
  by which BlueField generation the operator has.
- [`doca-bare-metal-deployment`](../../doca-bare-metal-deployment/SKILL.md)
  — the app-launch sibling and the source where the BF3 lifecycle
  facts this skill consolidates were originally vetted (its
  [`## bluefield-lifecycle`](../../doca-bare-metal-deployment/TASKS.md#bluefield-lifecycle)
  anchor). Once a BF3 is healthy and the operator wants to *run a
  binary*, control routes there.
- [`doca-container-deployment`](../../doca-container-deployment/SKILL.md)
  — the container-path deployment sibling. Once a BF3 is healthy and
  the operator wants to *deploy a service container*, control routes
  there.
- [`doca-hardware-safety`](../../doca-hardware-safety/SKILL.md) — the
  cross-cutting meta-policy for any change touching DPU / NIC
  hardware state. This skill's `## Safety policy` overlays that
  meta-policy with BF3-specific rules (OOB path is a precondition,
  exit-0-is-not-success, do-not-invent-flags-or-subnets) and does
  **not** redefine the meta-policy itself. Every BFB reflash,
  `mlxconfig set`, mode flip, and firmware burn leaves this skill for
  `doca-hardware-safety` and returns only once the change is
  complete.
- [`doca-setup`](../../doca-setup/SKILL.md) — env preparation and
  host-side DOCA-Host install (install verification, hugepages,
  IOMMU, pkg-config path, devlink mode, kernel module state). BF3
  bring-up assumes the host RShim surface is present; any host-side
  reinstall routes there.
- [`doca-version`](../../doca-version/SKILL.md) — the four-way version
  match rule (host package ↔ binary build ↔ BlueField firmware ↔
  DOCA-version policy). This skill's
  [`## Version compatibility`](../CAPABILITIES.md#version-compatibility)
  cross-links the body there and adds only the BF3 three-leg overlay
  (host DOCA-Host, Arm-side BFB, NIC firmware).
- [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md)
  — the routing table to the public BlueField Platform Software
  Manual, the public DOCA Installation Guide, the MFT manual, and
  the public DOCA Downloads page. This skill does not duplicate URLs;
  it points at the map and adds the BF3-bring-up overlay.
