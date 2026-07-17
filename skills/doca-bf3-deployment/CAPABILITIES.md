# DOCA BlueField-3 (BF3) deployment — Capabilities

**Where to start:** The pattern overview below names the recurring
BF3 platform-bring-up patterns the agent walks for any BlueField-3.
Pick the pattern first, then drill into the H2 that owns the
substance. For the *how* of executing each pattern, jump to
[TASKS.md](TASKS.md). For the BF4 counterpart of the same lifecycle
(the BMC-Redfish provisioning path), see the sibling skill
[`doca-bf4-deployment`](../doca-bf4-deployment/SKILL.md) — the BF4
equivalent. For the cross-cutting safety meta-policy this skill
overlays (and which owns every mutating burn named below), see
[`doca-hardware-safety`](../doca-hardware-safety/SKILL.md).

This file enumerates the BF3 platform-bring-up contract as described
in the public **BlueField Platform Software Manual**, the public
**DOCA Installation Guide**, and the public **MFT manual** (`flint`,
`mlxconfig`, `mlxfwmanager`) — all reachable through
[`doca-public-knowledge-map`](../doca-public-knowledge-map/SKILL.md)
— plus the standard Linux primitives the RShim/TMFIFO path inherits
(`modprobe(8)`, `lspci(8)`, `ip-route(8)`). Treat this file as a
*map of what is documented*, not a substitute for reading the live
manuals against the operator's BF3. The BF3 lifecycle facts here are
the same facts vetted in
[`doca-bare-metal-deployment ## bluefield-lifecycle`](../doca-bare-metal-deployment/TASKS.md#bluefield-lifecycle);
this skill is their consolidated BF3 home, not a re-derivation.

## Pattern overview

Every BF3-bring-up question this skill teaches resolves into one of
SIX patterns. The patterns are CLASSES — they apply to any
BlueField-3 regardless of which DOCA release the operator is
landing.

| BF3 bring-up pattern | Class shape | Where the substance lives |
| --- | --- | --- |
| 1. Recognise the install side — host-side DOCA-Host vs Arm-side BlueField-OS DOCA | Which package the operator installs where; which side `pkg-config --modversion doca-common` is read on; which side the binary will eventually run | [`## Capabilities and modes`](#capabilities-and-modes) install-side table |
| 2. Push a BFB over RShim | `bfb-install` streams the BFB to the DPU over the RShim PCIe/USB interface; the Arm side reboots through UEFI to Linux up to first-boot init | [`## Capabilities and modes`](#capabilities-and-modes) RShim/BFB surface |
| 3. Bring up / recover the TMFIFO channel | `tmfifo_net0` (or the `tm-br` bridge), the documented `192.168.100.x` convention, the `ip route get`-before-`ping` loopback gotcha | [`## Capabilities and modes`](#capabilities-and-modes) TMFIFO surface |
| 4. Select the DPU mode | DPU / embedded-function vs separated-host / NIC mode, set via `mlxconfig` at BFB-install time — a MUTATING burn | [`## Safety policy`](#safety-policy) + [`doca-hardware-safety`](../doca-hardware-safety/SKILL.md) |
| 5. Map a not-yet-healthy BF3 back to its state | installer-running → uefi-only → linux-up-tmfifo-down → tmfifo-up-ssh-down → arm-ok-host-pfs-unbound → host-bf-version-mismatch | [`## Error taxonomy`](#error-taxonomy) six-state classifier |
| 6. Verify before declaring healthy | readiness markers (not a timer), `cat /etc/mlnx-release`, host PF rebind, the four-way version match re-close | [`## Observability`](#observability) + [`## Version compatibility`](#version-compatibility) |

Two cross-cutting rules apply to *every* pattern above:

- **Operate the documented path; do not invent one.** `bfb-install`
  flags, the BFB image filename, the `/dev/rshim<N>` path, the
  `bf.cfg` schema keys, the `mlxconfig` mode parameters, and the
  TMFIFO subnet all come from the public BlueField Platform Software
  Manual, the MFT manual, or `--help` on the installed tool.
  Inferring them from generic Linux intuition or from a previous
  BF2's behaviour is the most common hallucination failure for this
  skill.
- **Every mutating burn leaves this skill for the meta-policy.** The
  BFB reflash itself, any `mlxconfig set` (including a DPU-mode
  flip), a firmware burn, and any kernel-boot-parameter change are
  owned by
  [`doca-hardware-safety`](../doca-hardware-safety/SKILL.md) for the
  preflight / OOB-console / maintenance-window / rollback
  discipline. This skill adds only the BF3-specific *sequencing* on
  top.

## Capabilities and modes

### Install side — host-side DOCA-Host vs Arm-side BlueField-OS DOCA

A BlueField-3 deployment has DOCA on TWO distinct sides, and the
operator must be explicit about which one a given step touches.

| Property | Host-side (DOCA-Host) | Arm-side (BlueField-OS DOCA) |
| --- | --- | --- |
| What is installed | The DOCA-Host packages on the host x86/Arm OS, per the public DOCA Installation Guide host-side path (reached through [`doca-public-knowledge-map`](../doca-public-knowledge-map/SKILL.md)) | The DOCA userland baked into the BFB image, landed on the BF3 Arm cores by the BFB push |
| How it lands | `apt` / `dnf` install on the host, routed via [`doca-setup`](../doca-setup/SKILL.md) | `bfb-install` pushing the BFB over RShim (this skill, [`TASKS.md ## run`](TASKS.md#run)) |
| Version read with | `pkg-config --modversion doca-common` + `doca_caps --version` on the host | `cat /etc/mlnx-release` + `bfver` on the BF3 Arm console |
| What runs there | Host-side binaries/containers talking to the BF3 NIC over PCIe | Arm-side binaries/containers running on the DPU cores directly |
| Routes onward to | [`doca-bare-metal-deployment`](../doca-bare-metal-deployment/SKILL.md) / [`doca-container-deployment`](../doca-container-deployment/SKILL.md) once healthy | the same two deployment skills, Arm-side leg |

The load-bearing recognition: *"is DOCA on the host or on the Arm
side?"* is the FIRST question for any BF3 deploy. The agent confirms
which side a step touches before walking it; conflating them ("just
install DOCA") is the canonical BF3 first-contact failure.

### RShim / BFB transport surface

RShim is the host-side surface that exposes the BF3 Arm console
(`/dev/rshim<N>/console`) AND the host-side endpoint of the TMFIFO
recovery interface. The BFB (BlueField bundle) is the unit of input:
`bfb-install` streams it to the DPU over the RShim PCIe (or USB)
interface, and the Arm side reboots through UEFI to Linux up to
first-boot init. On DOCA 3.3+ hosts the RShim surface ships as a
**userspace daemon** (`/usr/sbin/rshim` started by `rshim.service`);
the legacy in-tree kernel module is no longer shipped, so
`lsmod | grep rshim` being empty is EXPECTED, not failure evidence.
On a multi-DPU host the agent disambiguates which `/dev/rshim<N>`
maps to which BF3 (via the `DEV_NAME` field of each `misc` file,
cross-matched against `lspci -d 15b3: -nn`) BEFORE any per-DPU
`bfb-install` — skipping it is the #1 cause of *"I flashed the wrong
DPU"* incidents. The agent does NOT fabricate the `bfb-install` flag
set, the BFB filename, or the `/dev/rshim<N>` path; all come from
`--help` and the BSP manual.

### TMFIFO management-channel surface

The TMFIFO interface is the host-to-DPU *recovery* path when the
BF3's normal management network is broken; it is NOT a primary data
path. Factory defaults per the BlueField Platform Software Manual:
host-side address `192.168.100.1/30`, BlueField-side address
`192.168.100.2/30` (the agent does NOT fabricate the subnet). The
host-side interface is `tmfifo_net0`, or `tm-br` on BSP/DOCA-host
installs that bridge `tmfifo_net0` into a `tm-br` bridge. The
load-bearing gotcha: a `ping 192.168.100.2` that "works" can be
pinging the HOST, not the BF3, when the BlueField-side address has
been bound locally; the agent ALWAYS runs `ip route get
<bf-tmfifo-address>` first and accepts `dev tmfifo_net0` OR `dev
tm-br` as healthy egress, while `dev lo` / `local <bf-addr> dev lo`
is the broken local-loopback signature.

### DPU-mode surface (MUTATING — routed out)

A BF3 boots in one of two operating modes: **DPU / embedded-function
mode** (the Arm cores own the data path; the host sees a managed
function) or **separated-host / NIC mode** (the host owns the ports
directly; the Arm side is out of the data path). The mode is
selected via `mlxconfig` NV-config parameters, typically applied
from a `bfb_modify_os()` hook in `bf.cfg` at BFB-install time;
reconfiguring the same modes after install typically requires
another BFB push. Because `mlxconfig set` is a hardware-state change,
the agent quotes the `bf.cfg`/`mlxconfig` parameter keys from the
public schema but routes the actual set/burn to
[`doca-hardware-safety`](../doca-hardware-safety/SKILL.md) — this
skill names the *sequencing* (mode decision belongs at BFB-install
time, not after), the meta-policy owns the burn.

## Version compatibility

For the canonical DOCA version-detection chain, the four-way match
rule (host package ↔ binary build ↔ BlueField firmware ↔
DOCA-version policy), and the headers-win-over-docs rule, see
[`doca-version`](../doca-version/SKILL.md). The body lives there;
this skill does not duplicate it.

**The BF3-bring-up-specific overlay** is:

- **The BF3 has THREE independent version legs, read on different
  sides.** (i) The host-side DOCA-Host install version
  (`pkg-config --modversion doca-common` + `doca_caps --version` on
  the host). (ii) The BFB-image DOCA version on the Arm side
  (`cat /etc/mlnx-release` + `bfver` on the BF3 Arm console — this
  is the DOCA userland inside the BFB). (iii) The NIC-side firmware
  version (`flint -d <bdf> q`, the `FW Version:` line — NOT
  `mlxconfig -d <bdf> q`, which returns the firmware *configuration*
  dump, not the version). A mismatch among any two is the canonical
  *"the docs say this should work but it does not"* trap.
- **`cat /etc/mlnx-release` is the BF3 install-verification anchor.**
  After a BFB push, this is the first read that confirms which BFB
  landed; it is the Arm-side leg of the four-way match and the
  rollback baseline for any subsequent re-push.
- **Push the BFB and the host-side DOCA-Host install together.** A
  BFB push that is not accompanied by a matching host-side
  DOCA-Host alignment lands the `host-bf-version-mismatch` state in
  [`## Error taxonomy`](#error-taxonomy); resolve any
  `/etc/apt/sources.list.d/doca.list` repo-pin drift per
  [`doca-version TASKS.md ## apt-source consistency`](../doca-version/TASKS.md#apt-source-consistency)
  BEFORE installing anything new, and route any host-side reinstall
  through [`doca-setup`](../doca-setup/SKILL.md).
- **Do NOT substitute `mlxconfig -d <bdf> q` for the FW-version
  leg, and do NOT invent `bfb-info`** (not a real
  NVIDIA-documented tool) for the BFB-image leg — both are common
  hallucinations banned in
  [`doca-version CAPABILITIES.md ## Version compatibility`](../doca-version/CAPABILITIES.md#version-compatibility).
  Capture per-device (`flint -d <bdf> q`), not globally — two BF3s
  from different procurement waves on the same host are independent
  silicon with independent FW levels.

## Error taxonomy

A BF3 that has had a BFB push, a soft-reset, or a host PF rebind and
is *not yet* confirmed healthy is classified by the six-state
`bluefield-state-classifier`, walked IN ORDER in
[`TASKS.md ## debug`](TASKS.md#debug). The states are the SAME six
vetted in
[`doca-bare-metal-deployment ### bluefield-state-classifier`](../doca-bare-metal-deployment/TASKS.md#bluefield-state-classifier);
this skill is their BF3 home.

1. **`installer-still-running`.** `bfb-install` still resident on
   the host AND the RShim console buffer still emitting documented
   installer progress lines. Root cause class: install in flight,
   not failure. Recovery: WAIT — aborting a first-flash push
   mid-write is what *creates* the next state down.
2. **`uefi-only`.** The RShim console reports the documented
   UEFI-exit marker but never reaches the documented `Linux up`
   marker. Root cause class: kernel did not hand off to userspace —
   common after a partial BFB-install whose firmware-update sub-step
   silently failed. Recovery: capture the full console buffer and
   host `dmesg`, do a documented cold power cycle (via BMC, not
   `reboot` from the dead Arm side) BEFORE re-pushing.
3. **`linux-up-tmfifo-down`.** Documented `Linux up` marker present,
   BUT the TMFIFO probe returns no host-side address. Root cause
   class: TMFIFO interface never came up (driver / udev / link
   state). Recovery: re-check the host RShim daemon, run the
   documented TMFIFO bring-up, THEN re-run the `ip route get`
   loopback check.
4. **`tmfifo-up-ssh-down`.** TMFIFO probe passes (and `ip route get`
   confirms the route goes to the BF3, not local loopback), BUT SSH
   to the documented Arm-side endpoint refuses or hangs. Root cause
   class: Arm-side `sshd` not yet listening, OR the operator's
   `authorized_keys` / password was not in the `bf.cfg`. Recovery:
   wait the documented `sshd`-ready bound, else fall through to the
   RShim console and re-seed credentials there.
5. **`arm-ok-host-pfs-unbound`.** Arm-side SSH alive and OS healthy,
   BUT host-side enumeration is broken — `lspci -d 15b3:` shows the
   PFs, `ip link show` does NOT show the netdevs, `ibv_devinfo` is
   empty, DOCA programs cannot attach by representor name. Root
   cause class: stale host `mlx5` driver-binding state post-push.
   Recovery: run the documented PF-rebind sequence in
   [`TASKS.md ## test`](TASKS.md#test); do NOT launch any DOCA
   binary in this state.
6. **`host-bf-version-mismatch`.** Everything above looks healthy,
   but the four-way version match owned by
   [`doca-version`](../doca-version/SKILL.md) does not close.
   Recovery: walk [`doca-version TASKS.md`](../doca-version/TASKS.md)
   in full, resolve apt-source / repo-pin drift first, route any
   host-side reinstall through [`doca-setup`](../doca-setup/SKILL.md).

Two cross-cutting rules: **match multiple states, do not stop at the
first** (a BF3 back from a partial install can be `uefi-only` then
`arm-ok-host-pfs-unbound` after a cold cycle); and **never declare
healthy from absence of evidence** (a TMFIFO `ping` without
`ip route get`, an SSH connect without `uptime`/`dmesg`, an `lspci`
listing without a usable netdev are each NOT proof).

## Observability

Documented observability surfaces for BF3 bring-up, in the order the
agent reaches for them. Healthy means all of them agree; a wall-clock
sleep is never a substitute for a documented readiness marker.

- **RShim console buffer (FIRST).** The Arm-side boot story lives in
  `/dev/rshim<N>/console`. The agent watches for the documented
  `Linux up` and `DPU is ready` markers (per the BSP manual) and for
  any `[MISC]` / `[ERR]` line containing a failure verb ("failed",
  "error", "abort") — the canonical partial-install signature is
  *"Ubuntu installation completed"* followed by an
  `INFO[MISC]: NIC firmware update failed` line while `bfb-install`
  still exits 0. Exit code 0 alone is NOT proof of success.
- **TMFIFO reachability (SECOND).** `ip addr show tmfifo_net0` (or
  the `tm-br` bridge) on the host, then `ip route get
  <bf-tmfifo-address>` BEFORE any `ping` — the `dev lo` /
  `local <bf-addr> dev lo` output is the local-loopback failure that
  makes `ping` lie.
- **Host PF enumeration (THIRD).** `lspci -d 15b3:` lists the BF3
  PFs; `ip link show` enumerates their netdevs; `ibv_devinfo`
  enumerates the RDMA devices; `devlink dev show` lists the devlink
  instance. PFs present in `lspci` but absent from `ip link` is the
  `arm-ok-host-pfs-unbound` state.
- **Install verification (FOURTH).** `cat /etc/mlnx-release` +
  `bfver` on the Arm side confirm which BFB landed;
  `mlxconfig -d <pci> query` (QUERY only) reads the firmware
  configuration; `flint -d <bdf> q` reads the FW version. Any
  `mlxconfig set` is a hardware-state change owned by
  [`doca-hardware-safety`](../doca-hardware-safety/SKILL.md).

Cross-cutting host-side debug (kernel version, driver loaded/not,
PCIe link state, `dmesg`) lives in
[`doca-debug TASKS.md ## debug`](../doca-debug/TASKS.md#debug); this
skill names only the BF3-bring-up-specific surfaces.

## Safety policy

> **Overlay on the bundle-wide hardware-safety meta-policy.** The
> rules below are this skill's per-artifact overlay on the
> cross-cutting rules in
> [`doca-hardware-safety` CAPABILITIES.md ## Safety policy](../doca-hardware-safety/CAPABILITIES.md#safety-policy)
> (specifically
> [### Per-artifact overlay pattern](../doca-hardware-safety/CAPABILITIES.md#per-artifact-overlay-pattern)).
> When the two layers disagree, the stricter wins; when either layer
> says STOP, the agent stops.

- **Every mutating burn leaves this skill.** A BFB reflash, any
  `mlxconfig set` (including a DPU/separated-host mode flip), a
  firmware burn, and any kernel-boot-parameter change are
  hardware-state changes. The agent hands the change-application
  discipline (preflight inventory, OOB console, maintenance window,
  rollback) to
  [`doca-hardware-safety ## modify`](../doca-hardware-safety/TASKS.md#modify)
  and returns here only once the change is complete and verified.
  This skill owns the *sequencing*, never the burn.
- **An OOB path is a precondition, not a nicety.** Before any BFB
  push the operator must have a BMC console / serial-over-LAN /
  physical UART path to reach the BF3 if the push breaks the Arm OS.
  Without one, the bar to proceed is "stop, escalate" per the
  meta-policy.
- **Never declare healthy from `bfb-install` exit 0 alone.** Parse
  the console / log output for the documented `Linux up` / `DPU is
  ready` markers and for any failure-verb line, and run the
  post-BFB readiness smoke in [`TASKS.md ## test`](TASKS.md#test)
  before re-declaring the BF3 ready. Do NOT invent `bfb-install`
  flags, BFB filenames, `/dev/rshim<N>` paths, `bf.cfg` keys, or the
  TMFIFO subnet from memory — quote the live `--help` and the BSP
  manual.

## Public-source pointer

The canonical public sources for BF3 platform bring-up are:

- The **BlueField Platform Software Manual** on `docs.nvidia.com`,
  reachable through
  [`doca-public-knowledge-map ## Public documentation entry points`](../doca-public-knowledge-map/SKILL.md#public-documentation-entry-points),
  for `bfb-install`, the `bf.cfg` schema, RShim/TMFIFO, and the
  readiness markers.
- The **DOCA Installation Guide** on `docs.nvidia.com`, reachable
  through the same routing table, for the host-side DOCA-Host
  install path and the BFB download location (the public DOCA
  Downloads page).
- The **MFT manual** (`flint`, `mlxconfig`, `mlxfwmanager`) for the
  firmware-query and NV-config surfaces.

Verify that the version of each manual matches the operator's DOCA
release and BFB image per
[`## Version compatibility`](#version-compatibility) — flag names,
`bf.cfg` keys, and readiness markers evolve, so anything quoted from
memory is suspect.
