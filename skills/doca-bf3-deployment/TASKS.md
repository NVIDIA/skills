# DOCA BlueField-3 (BF3) deployment — Tasks

**Where to start:** The verb order is `configure → build → modify →
run → test → debug`. For BF3 platform bring-up, `build` is a
*routing stub* — there is no application or BFB artifact to build
inside this skill (building a DOCA-linked binary lives in
[`doca-programming-guide`](../doca-programming-guide/SKILL.md);
building the BFB image itself is internal-tree work, out of scope).
This skill owns taking a real BF3 from "powered card" to "Arm OS
healthy, host PFs bound, version match closed". The `## run` verb is
the BFB-install + RShim/TMFIFO bring-up sequence; `## test` is the
post-BFB readiness smoke; `## debug` is the six-state classifier.

Every step assumes the operator has consulted the live public
BlueField Platform Software Manual, the public DOCA Installation
Guide, and the MFT manual (all reachable through
[`doca-public-knowledge-map ## Public documentation entry points`](../doca-public-knowledge-map/SKILL.md#public-documentation-entry-points))
and is using them as the authoritative reference; this file
prescribes the *order* and *what to look up where*, not a copy-paste
runbook. Every mutating burn (BFB reflash, `mlxconfig set`, firmware
burn, kernel-boot-parameter change) STILL routes through
[`doca-hardware-safety`](../doca-hardware-safety/SKILL.md) for the
meta-policy — this file adds only the BF3-specific sequencing.

## configure

Preparing the BF3 bring-up, confirming every precondition the push
will rely on, and recognizing the install side BEFORE any BFB is
pushed.

1. **Recognise the install side.** Confirm with the operator which
   side the work touches — host-side DOCA-Host on the host OS, OR
   Arm-side DOCA baked into the BFB landing on the DPU cores. Per
   the install-side table in
   [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes),
   this is the FIRST question for any BF3 deploy. If the operator is
   ambiguous, derive it from where the install lives
   (`/opt/mellanox/doca` on the host vs on the BF3 Arm). Do NOT
   guess.
2. **Confirm the host RShim surface is present.** Run all of:
   `dpkg -s rshim` / `rpm -q rshim` (userspace package installed),
   `systemctl status rshim` (daemon `active (running)`), and
   `ls /dev/rshim*` (character-device tree present), per the BSP
   manual. On DOCA 3.3+ `lsmod | grep rshim` is EXPECTED empty (the
   in-tree module is gone) and is NOT failure evidence. If the
   daemon or `/dev/rshim*` tree is missing, the host has no path to
   push.
3. **Disambiguate which `/dev/rshim<N>` is which BF3.** On a
   multi-DPU host, read the `DEV_NAME` field of each
   `/dev/rshim*/misc` file and cross-match against
   `lspci -d 15b3: -nn` (the canonical one-liner is in
   [`doca-bare-metal-deployment CAPABILITIES.md`](../doca-bare-metal-deployment/CAPABILITIES.md#rshim-instance--bluefield-disambiguation-canonical-one-liner)).
   This is the precondition for every per-DPU `bfb-install`;
   skipping it is the #1 cause of *"I flashed the wrong DPU"*.
4. **Confirm the OOB path exists.** A BMC console-over-Redfish,
   BMC IPMI serial-over-LAN, or physical UART to reach the BF3 if
   the push breaks the Arm OS — per the safety precondition in
   [`CAPABILITIES.md ## Safety policy`](CAPABILITIES.md#safety-policy).
   Without one, the bar to proceed is "stop, escalate" per
   [`doca-hardware-safety`](../doca-hardware-safety/SKILL.md).
5. **Verify the BFB image.** Confirm the BFB image SHA matches the
   SHA the operator downloaded from the documented public DOCA
   Downloads page (route via
   [`doca-public-knowledge-map ## Public documentation entry points`](../doca-public-knowledge-map/SKILL.md#public-documentation-entry-points))
   — pushing a corrupted BFB is the load-bearing first-run failure.
   Do NOT invent the BFB filename.
6. **Capture the BEFORE version anchors.** Per
   [`CAPABILITIES.md ## Version compatibility`](CAPABILITIES.md#version-compatibility),
   record (read-only): the host-side DOCA-Host version
   (`pkg-config --modversion doca-common`), the current Arm-side BFB
   (`cat /etc/mlnx-release` + `bfver`), and the per-device NIC
   firmware (`flint -d <bdf> q`). This capture is the rollback
   anchor — without a BEFORE state, "rollback" is a phrase, not a
   thing the operator can do. Close the four-way match per
   [`doca-version TASKS.md ## configure`](../doca-version/TASKS.md#configure).
7. **Decide the DPU mode BEFORE the push.** If the BF3 must boot in
   a specific mode (DPU / embedded-function vs separated-host / NIC
   mode), that decision belongs at BFB-install time — the
   `mlxconfig` set is applied from a `bfb_modify_os()` hook in
   `bf.cfg`, and reconfiguring after install typically needs another
   push. The mode-set burn itself routes to
   [`doca-hardware-safety`](../doca-hardware-safety/SKILL.md); this
   step only records the decision so the `bf.cfg` authored in
   [`## run`](#run) carries it.

## build

BF3 platform bring-up is the *deploy-the-platform* path; there is no
application or BFB artifact for the operator to build inside this
skill.

- If the user is asking how to build a DOCA-linked binary they want
  to run on the healthy BF3, hand off to
  [`doca-programming-guide ## build`](../doca-programming-guide/TASKS.md#build)
  (the canonical `pkg-config doca-<library>` + meson pattern). Once
  the BF3 is healthy, control routes onward to
  [`doca-bare-metal-deployment`](../doca-bare-metal-deployment/SKILL.md).
- If the user is asking how to build the **BFB image** itself, that
  is internal BlueField-OS tree work — out of scope for this bundle;
  route to the public BlueField Platform Software documentation via
  [`doca-public-knowledge-map`](../doca-public-knowledge-map/SKILL.md).
- If the user is asking how to author `bf.cfg`, that is composed
  against the live BSP-manual schema at push time, not built ahead;
  see [`## run`](#run) step 1.
- The `bf.cfg` `bfb_modify_os()` hook is the documented mechanism
  for seeding post-install state; the agent quotes its parameter
  keys from the public schema and does NOT invent key names.

## modify

BF3 bring-up does not have a *modify a sample program* workflow; the
platform-side analog of "modify" is **re-walk the bring-up after the
BFB, the `bf.cfg`, or the underlying mode/version state changes**.

1. **A BFB change is a bring-up event.** Pushing a different BFB
   re-opens the readiness smoke in [`## test`](#test) — the new BFB
   may land a different Arm-side DOCA version, a different host PF
   layout, or a different mode. Re-walk [`## configure`](#configure)
   step 6 (version anchors), then [`## run`](#run), then
   [`## test`](#test).
2. **A `bf.cfg` change is a bring-up event.** Editing the password,
   the `authorized_keys` hook, or the mode-set hook changes the
   post-install state; treat each edit as a fresh push and re-walk
   [`## run`](#run) step 1 (author `bf.cfg`) and [`## test`](#test).
3. **A host-side change is a bring-up event.** A host-side DOCA-Host
   reinstall, an apt-source repo-pin change, or an `LD_LIBRARY_PATH`
   change re-opens the four-way match; re-walk
   [`## configure`](#configure) step 6 and route the host reinstall
   through [`doca-setup`](../doca-setup/SKILL.md).
4. **A mode flip or firmware burn leaves this verb entirely.** Any
   `mlxconfig set` (DPU/separated-host mode flip), firmware burn, or
   BFB reflash is owned by
   [`doca-hardware-safety ## modify`](../doca-hardware-safety/TASKS.md#modify);
   this skill does NOT walk that burn. Control returns here at
   [`## configure`](#configure) step 6 once the change is complete
   and verified per the meta-policy.

The anti-pattern alert: re-pushing a BFB or editing a `bf.cfg`
without re-running the readiness smoke is the canonical *"the DPU
came back wrong and I did not notice"* failure. Treat every
platform-side change as a fresh bring-up.

## run

The BFB-install + RShim/TMFIFO bring-up sequence — pushing the BFB
and getting the host-to-DPU channel healthy. Every step assumes the
preconditions in [`## configure`](#configure) are done.

1. **Author `bf.cfg` from the documented schema.** The BFB-install
   path takes an installer configuration file (`bf.cfg`) controlling
   post-install Arm-side state. Two operator-relevant rules: (a) for
   passwordless SSH to survive the install, set the documented
   password parameter AND seed the SSH public key via a
   `bfb_modify_os()` hook that writes the to-be-installed rootfs
   (mounted under `/mnt` during install per the BSP manual) — the
   default BFB install rewrites the home `.ssh/`, so a pre-existing
   key is GONE unless reseeded; there is NO top-level
   `authorized_keys` `bf.cfg` parameter, the mechanism is the hook.
   (b) for a specific DPU mode, the required `mlxconfig set`
   invocations run from a `bfb_modify_os()` hook (decision recorded
   in [`## configure`](#configure) step 7; the burn routes to
   [`doca-hardware-safety`](../doca-hardware-safety/SKILL.md)). Quote
   the `bf.cfg` keys from the public schema; do NOT invent them.
2. **Push the BFB over RShim.** Run the host-side `bfb-install`
   invocation per its `--help` and the BSP manual, targeting the
   `/dev/rshim<N>` disambiguated in [`## configure`](#configure)
   step 3. The push streams the BFB to the BF3 over RShim/PCIe; the
   Arm side reboots through UEFI to Linux up to first-boot init. The
   BFB reflash is a mutating change — load
   [`doca-hardware-safety`](../doca-hardware-safety/SKILL.md)
   ALONGSIDE for the preflight / OOB / rollback discipline.
3. **Do NOT trust `bfb-install` exit code 0 alone.** This is the
   single most expensive failure mode. `bfb-install` has been
   observed to exit 0 while the Arm-side flow only partially
   completed — the canonical field signature is *"Ubuntu
   installation completed"* followed by an `INFO[MISC]: NIC firmware
   update failed` line. Parse the actual console / log output for:
   (a) any `[MISC]` / `[ERR]` line with a failure verb, (b) the
   documented `Linux up` marker, (c) the documented `DPU is ready`
   marker. If (a) is present or (b)/(c) are absent, the install is
   **partial** — advance to [`## debug`](#debug) instead of
   declaring success.
4. **Verify the RShim console and TMFIFO channel.** Watch
   `/dev/rshim<N>/console` for the documented boot markers, then
   bring up TMFIFO: `ip addr show tmfifo_net0` (or the `tm-br`
   bridge) on the host, and — CRITICALLY — `ip route get
   <bf-tmfifo-address>` BEFORE any `ping`. Accept `dev tmfifo_net0`
   OR `dev tm-br` as healthy egress; `dev lo` / `local <bf-addr>
   dev lo` is the local-loopback failure that makes `ping` lie. The
   factory subnet (`192.168.100.1/30` host, `192.168.100.2/30` BF)
   comes from the BSP manual; do NOT fabricate it.
5. **Hand off to the readiness smoke.** "BFB install completed"
   means the installer's I/O is done; it does NOT mean the Arm OS
   is up, TMFIFO is reachable, SSH is live, or host PFs are bound.
   Always run [`## test`](#test) before re-declaring the BF3
   healthy.

## test

BF3 bring-up has no *compile-and-unit-test* workflow — testing is
the operational post-BFB readiness smoke against real hardware.

**`## test` is an iterative loop, not a one-shot pass.** Every push,
every `bf.cfg` edit, every mode/version change re-opens the smoke.
Skipping the re-run after a mutation is the failure mode this loop
replaces.

1. **Wait for documented readiness markers, not a timer.** Poll the
   RShim console buffer for `Linux up` / `DPU is ready` (per the BSP
   manual), AND the Arm-side SSH endpoint responding, AND (where
   present) the BMC health endpoint reporting `OK`. If any never
   reports ready within the manual's documented bound, the BF3 is
   partial — advance to [`## debug`](#debug).
2. **Run the host PF rebind sequence if netdevs are missing.** A
   push can leave the host `mlx5` driver stale: PFs present in
   `lspci -d 15b3:` but `ip link show` shows no netdevs and
   `ibv_devinfo` is empty. The documented recovery: `modprobe
   mlx5_core` (no-op if loaded); for each BF3 PF BDF captured at
   pre-flight, `echo <bdf> > /sys/bus/pci/drivers/mlx5_core/bind`
   per the kernel sysfs driver-binding docs; re-verify with
   `ip link show` / `ibv_devinfo` / `devlink dev show`. The BDF
   strings come from the pre-flight `lspci -d 15b3:` capture, NOT
   from memory.
3. **Verify the install landed.** `cat /etc/mlnx-release` + `bfver`
   on the Arm side confirm which BFB landed; `flint -d <bdf> q`
   reads the per-device FW version. These are the Arm-side and NIC
   legs of the four-way match.
4. **Check the `/home/ubuntu` ownership gotcha.** On certain BFB
   images `/home/ubuntu` ships owned by `root`, breaking the normal
   pattern of the `ubuntu` user writing under their own home. Check
   `stat -c '%U:%G' /home/ubuntu` after first SSH; if `root:root`,
   flag it and propose the documented `chown -R ubuntu:ubuntu
   /home/ubuntu` fix (per the BSP manual) BEFORE pasting any script
   that writes there.
5. **Re-close the four-way version match.** Once Arm OS is healthy
   and host PFs are bound, walk the four-way match owned by
   [`doca-version TASKS.md`](../doca-version/TASKS.md) against the
   new BF3 state BEFORE handing onward to a deployment skill. A
   skipped re-close after a push is the most common cause of "ran
   fine yesterday, breaks today".

Loop termination: stop iterating once two consecutive smokes of the
same kind change nothing — that means the cause is below the
platform layer (BFB image, host OS, silicon). Escalate to
[`doca-debug TASKS.md ## debug`](../doca-debug/TASKS.md#debug) with
the captured evidence. Once the smoke is green, hand off — running a
binary to
[`doca-bare-metal-deployment ## run`](../doca-bare-metal-deployment/TASKS.md#run),
a service container to
[`doca-container-deployment ## run`](../doca-container-deployment/TASKS.md#run).

## debug

The six-state `bluefield-state-classifier`. When a BFB push, a
soft-reset, or a host PF rebind has been done and the BF3 is *not
yet* confirmed healthy, walk this classifier IN ORDER and stop at the
first state the evidence matches — but report EVERY state that
matches (an Arm OS can be "Linux up" AND "host PFs unbound"
simultaneously). The full state evidence/recovery detail is in
[`CAPABILITIES.md ## Error taxonomy`](CAPABILITIES.md#error-taxonomy);
this is the walking order. Mutating recovery steps load
[`doca-hardware-safety`](../doca-hardware-safety/SKILL.md) alongside.

1. **`installer-still-running`.** `bfb-install` resident + console
   still emitting progress lines → WAIT, do not abort. Aborting a
   first-flash push mid-write creates the next state down.
2. **`uefi-only`.** UEFI-exit marker present but no `Linux up` →
   capture console + host `dmesg`, do a documented cold power cycle
   (via BMC, not `reboot` from the dead Arm side) BEFORE re-pushing.
3. **`linux-up-tmfifo-down`.** `Linux up` present but the TMFIFO
   probe returns no host-side address → re-check the host RShim
   daemon, run the documented TMFIFO bring-up, THEN re-run the
   `ip route get` loopback check (a freshly-bound TMFIFO can land in
   the loopback failure mode and look like it works).
4. **`tmfifo-up-ssh-down`.** TMFIFO route confirmed to the BF3 (not
   loopback) but SSH refuses/hangs → wait the documented
   `sshd`-ready bound; else fall through to the RShim console for a
   userspace prompt and re-seed credentials; put `authorized_keys`
   in the `bf.cfg` on the next push.
5. **`arm-ok-host-pfs-unbound`.** Arm SSH alive + OS healthy but
   host enumeration broken (`lspci` shows PFs, `ip link` does not
   show netdevs, `ibv_devinfo` empty) → run the documented PF rebind
   in [`## test`](#test) step 2. Do NOT launch any DOCA binary in
   this state — every device-open fails with a misleading error.
6. **`host-bf-version-mismatch`.** Everything above healthy but the
   four-way match does not close → walk
   [`doca-version TASKS.md`](../doca-version/TASKS.md) in full,
   resolve apt-source / repo-pin drift per
   [`doca-version TASKS.md ## apt-source consistency`](../doca-version/TASKS.md#apt-source-consistency)
   BEFORE installing anything, and route any host reinstall through
   [`doca-setup`](../doca-setup/SKILL.md).

Cross-cutting host-layer issues (kernel version, driver state, PCIe
link state, hugepage health) that survive a healthy classifier walk
drop to
[`doca-debug TASKS.md ## debug`](../doca-debug/TASKS.md#debug). Never
declare the BF3 healthy from absence of evidence — a TMFIFO `ping`
without `ip route get`, an SSH connect without `uptime`/`dmesg`, an
`lspci` listing without a usable netdev are each NOT proof.

## Deferred task verbs

- **BlueField-4 bring-up** (the BMC-Redfish provisioning path) — out
  of scope here. Route to
  [`doca-bf4-deployment`](../doca-bf4-deployment/SKILL.md), the BF4
  equivalent of this skill.
- **Running a DOCA-linked binary on a healthy BF3** (launch mode,
  PCI/NUMA/CPU binding, per-tenant isolation, the binary error
  taxonomy) — out of scope here. Route to
  [`doca-bare-metal-deployment`](../doca-bare-metal-deployment/SKILL.md)
  once the BF3 is healthy.
- **Deploying a DOCA service container on the BF3** (kubelet
  standalone, static-pod manifests directory, pod-spec YAML,
  image-pull from NGC) — out of scope here. Route to
  [`doca-container-deployment`](../doca-container-deployment/SKILL.md).
- **Env-preparation / host install** (installing DOCA-Host on a
  fresh host, hugepages, IOMMU, `PKG_CONFIG_PATH` /
  `LD_LIBRARY_PATH`, devlink mode flips) — out of scope here. Route
  to [`doca-setup`](../doca-setup/SKILL.md); BF3 bring-up assumes
  the host RShim surface is already in place.
- **Hardware-state changes** (`mlxconfig set` including DPU/
  separated-host mode flips, firmware burn, BFB reflash,
  kernel-boot-parameter changes) — the *change-application
  discipline* (preflight, OOB console, maintenance window, rollback)
  is meta-policy owned by
  [`doca-hardware-safety ## modify`](../doca-hardware-safety/TASKS.md#modify);
  this skill loads it ALONGSIDE whenever a mutating burn is on the
  table and adds only the BF3-specific sequencing.
- **Library-API and cross-library programming questions** (building
  a DOCA-Flow pipe, an RDMA queue pair, the cross-library
  `DOCA_ERROR_*` taxonomy) — out of scope here. Route to
  [`doca-programming-guide`](../doca-programming-guide/SKILL.md) and
  the matching `libs/<library>` skill via
  [`doca-public-knowledge-map ## Library- and module-specific guides`](../doca-public-knowledge-map/SKILL.md#library--and-module-specific-guides).
- **The version-match body** (four-way match rule, NGC semantics,
  headers-win) — out of scope here. Route to
  [`doca-version`](../doca-version/SKILL.md); this skill carries
  only the BF3 three-leg overlay.
