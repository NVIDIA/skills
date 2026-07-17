# DOCA bare-metal deployment — Capabilities

**Where to start:** The pattern overview below names the recurring
bare-metal deployment patterns the agent walks for any DOCA-linked
binary. Pick the pattern first, then drill into the H2 that owns the
substance. For the *how* of executing each pattern, jump to
[TASKS.md](TASKS.md). For the container-path counterpart of the same
patterns (kubelet standalone on the BlueField Arm watching a
documented static-pod manifests directory), see the sibling skill
[`doca-container-deployment`](../doca-container-deployment/SKILL.md).
For the cross-cutting safety meta-policy this skill overlays, see
[`doca-hardware-safety`](../doca-hardware-safety/SKILL.md).

This file enumerates the cross-cutting bare-metal deployment runtime
contract as described in the public **DOCA Programming Guide**, the
public **DOCA Installation Guide**, and the public
**BlueField / DPU User Manual** (all reachable through
[`doca-public-knowledge-map`](../doca-public-knowledge-map/SKILL.md)),
plus the standard Linux primitives the bare-metal path inherits
(`numactl(8)`, `taskset(1)`, `systemd.service(5)`,
`systemd.unit(5)`, cgroup-v2, network namespaces). Treat this file
as a *map of what is documented*, not a substitute for reading the
live guides against the operator's target hardware.

## Pattern overview

Every bare-metal-deployment question this skill teaches resolves into
one of SIX patterns. The patterns are CLASSES — they apply across
both supported host modes (host x86, BlueField Arm bare-metal) and
across any DOCA library the binary may have linked against (DOCA-Flow,
DOCA-RDMA, DOCA-DMA, DOCA-Comch, DOCA-GPUNetIO, …), not just one
library.

| Bare-metal deployment pattern | Class shape | Where the substance lives |
| --- | --- | --- |
| 1. Recognise the host mode — host x86 with a remote BlueField NIC over PCIe, OR BlueField Arm bare-metal with the binary on the DPU itself | Where DOCA is installed (host side vs Arm side); which `devlink dev` the operator sees; which representor naming convention is in play; which CPUs the binary may bind to without crossing a NUMA boundary the NIC owns | [`## Capabilities and modes`](#capabilities-and-modes) host-mode table |
| 2. Pick the launch mode — direct foreground, tmux / screen, or systemd-supervised | Direct = interactive debug, foreground stdout/stderr; tmux = long-running with manual reattach; systemd = restart-after-reboot, journald-integrated logs, documented `Restart=` policy | [`## Capabilities and modes`](#capabilities-and-modes) launch-mode table |
| 3. Bind the process to hardware — PF / VF / representor, NUMA node, CPU set, IRQ affinity | The DOCA-linked binary touches a specific PCIe function and a specific NUMA-local memory pool; pinning it correctly is the difference between baseline performance and the symptom *"my app works but throughput is a third of the documented number"* | [`## Capabilities and modes`](#capabilities-and-modes) hardware-binding rules |
| 4. Isolate per tenant — cgroup-v2 cpu / memory / io, network namespaces, `numactl` / `taskset` | Multiple DOCA processes co-tenant on the same BlueField is a supported shape; the isolation primitives are standard Linux, the gotchas (NUMA-locality, hugepage accounting, representor-vs-namespace) are DOCA-specific | [`## Capabilities and modes`](#capabilities-and-modes) isolation rules |
| 5. Map a failure back to its layer | Won't start → exits immediately → can't find device → library error → OOM / signal → restart loop → co-tenant noise; seven layers, each with its own owner | [`## Error taxonomy`](#error-taxonomy) layered split |
| 6. Smoke before bulk — trivial-arg invocation, then liveness equivalent, then real workload | The binary's stdout / journald lines BEFORE the workload moves; same discipline as the container path, applied to a process instead of a pod | [`## Safety policy`](#safety-policy) smoke-before-bulk rule |

Two cross-cutting rules apply to *every* pattern above:

- **Operate the documented path; do not invent one.** PCI addresses,
  representor names, NUMA node numbers, devlink paths, hugepage
  allocation amounts, and systemd `Restart=` mode names all come
  from the public DOCA Programming Guide, the public BlueField /
  DPU User Manual, the Linux man pages, or `--help` on the
  installed tool. Inferring them from generic Linux intuition or
  from a previous host's PCI BDF is the most common hallucination
  failure for this skill.
- **The deployment shape branches early.** Container vs bare metal
  is a fork the operator picks once, in the recognition step
  routed by [`doca-setup`](../doca-setup/SKILL.md); once the
  operator is on the bare-metal path this skill owns the runtime
  contract end-to-end, and the per-library skill layers on top.
  Re-stating the bare-metal runtime inside a per-library skill is
  the failure mode this skill exists to prevent.

## Capabilities and modes

### Host modes — host x86 vs BlueField Arm bare-metal

A DOCA-linked binary on the bare-metal path runs in one of two host
modes. The runtime contract is uniform; the install location,
device-enumeration surface, and representor naming convention
differ.

| Property | Host x86 mode | BlueField Arm bare-metal mode |
| --- | --- | --- |
| Where DOCA is installed | On the host x86 OS, per the public DOCA Installation Guide's host-side install path (reached through [`doca-public-knowledge-map`](../doca-public-knowledge-map/SKILL.md)) | On the BlueField Arm side, per the BlueField OS image's documented DOCA install path |
| Where the binary runs | On the host x86 CPU; calls into DOCA libraries that talk to the BlueField NIC over PCIe | On the BlueField Arm cores; calls into DOCA libraries that talk to the local NIC silicon directly |
| Device enumeration | `devlink dev show` on the host lists the BlueField as a remote NIC; `lspci -d 15b3:` enumerates the BlueField PFs / VFs | `devlink dev show` on the BlueField Arm side lists the local NIC; representor naming follows the documented BlueField OS convention |
| NUMA topology to bind against | The host's NUMA topology; the NUMA node owning the PCIe root complex that the BlueField is plugged into is the load-bearing one | The BlueField Arm's NUMA topology per the BlueField OS image's documentation |
| Hugepage backing | Reserved on the host per [`doca-setup ## configure`](../doca-setup/TASKS.md#configure) step 7 | Reserved on the BlueField Arm side per the BlueField OS image's documented procedure (routed via the BlueField / DPU User Manual reached through [`doca-public-knowledge-map`](../doca-public-knowledge-map/SKILL.md)) |
| Sibling deployment shape | The container path for this mode is *also* host-side (a container running on the host x86 that talks to the BlueField NIC over PCIe); the kubelet-standalone shape covered by [`doca-container-deployment`](../doca-container-deployment/SKILL.md) is the BlueField-Arm analog | The container path for this mode is the kubelet-standalone shape covered by [`doca-container-deployment`](../doca-container-deployment/SKILL.md) — the operator chooses container or bare metal at the recognition step in [`doca-setup`](../doca-setup/SKILL.md), not after the fact |

Operators frequently have BOTH host modes in play (a host x86 with a
DOCA host install AND a BlueField with DOCA on its Arm side). When
that is the case, the agent's rule is to confirm WHICH side the
binary in question is built for (`pkg-config doca-common
--variable=arch` against the install that produced it) and route
the workflow at the matching install — not to assume the host x86
install is the *only* install in scope.

### Launch modes — direct, tmux / screen, systemd-supervised

A DOCA-linked binary on the bare-metal path can be launched in one
of three modes. The choice is made up front in
[`TASKS.md ## configure`](TASKS.md#configure) and feeds the
observability surface and the restart / recovery semantics below.

| Launch mode | When it fits | Observability surface | Restart / recovery posture |
| --- | --- | --- | --- |
| Direct (foreground CLI) | Interactive debug; first-launch smoke; one-shot evaluation against a trivial workload. The shell session owns the process | stdout / stderr arrive on the terminal the operator launched from; closing the shell terminates the process unless the operator backgrounded it | None: a crash exits the shell session; restart is a manual re-invocation. The operator gets full visibility but no resilience |
| tmux / screen (long-running, manual reattach) | Long-running workload where the operator wants the process to outlive the SSH session and be able to reattach to read live stdout / stderr | stdout / stderr arrive on the tmux / screen pane; `tmux attach` / `screen -r` reattaches; output may scroll out of the pane buffer | None beyond what direct gives: a crash leaves the pane with the failed process; restart is still a manual re-invocation. Useful when the operator wants the live terminal surface without a supervisor |
| systemd-supervised | Long-running production workload; restart-after-reboot required; journald-integrated logs and a documented `Restart=` policy are valuable | stdout / stderr captured into journald per the unit's `StandardOutput=` / `StandardError=` settings; `journalctl -u <unit> -f` is the live tail | systemd's `Restart=` policy applies — see the restart / recovery semantics below. The HIGH-STAKES rule in [`## Safety policy`](#safety-policy) explicitly covers the case where the documented `Restart=` mode auto-restarts a process whose underlying failure has not yet been cleared |

The agent does NOT pre-bake sample systemd units, sample tmux
invocations, or sample direct command lines. The launch-mode
decision is the operator's; the agent's job is to surface the trade-
offs and quote the documented mode names (per `systemd.service(5)`
and `systemd.unit(5)`) when the operator picks systemd.

### Hardware-resource binding

A DOCA-linked process is bound to a specific PCIe function (the PF
or VF the DOCA library is documented to attach to), and that
binding implies a specific NUMA-local memory pool and a specific
CPU set that should drive it. Getting these bindings right is the
difference between baseline performance and the canonical *"my
DOCA-Flow app builds and runs but throughput is a third of what
the docs claim"* trap.

The binding surface, named at class level:

- **PF / VF / representor enumeration.** `lspci -d 15b3:` lists
  Mellanox PCIe functions on the local PCI tree;
  `devlink dev show` lists the BlueField devices the kernel sees;
  the documented representor naming convention for the operator's
  BlueField OS image lives in the public BlueField / DPU User
  Manual. The agent does NOT quote a specific BDF (e.g.
  `0000:01:00.0`) or a specific representor name (e.g.
  `pf0vf0`) from memory; both vary per host and per image.
- **NUMA topology discovery.** `numactl --hardware` and
  `lscpu` describe the host's NUMA layout. The load-bearing fact
  for a DOCA process is *"which NUMA node owns the PCIe root
  complex the BlueField is plugged into"*; that is the NUMA node
  the process should pin to, and crossing it costs measurable
  latency and bandwidth. The exact NUMA number is per-host —
  derive it from the live output, do not infer it.
- **CPU pinning rationale.** A DOCA dataplane binary typically
  wants a fixed CPU set for its workers (the *"polling threads"*
  in DPDK-based libraries) and isolated from the kernel
  scheduler's general balancing. `taskset(1)` and `numactl(8)`
  are the standard primitives. The agent quotes the public DOCA
  Programming Guide's documented CPU-pinning guidance for the
  library in use rather than picking a CPU set from generic
  Linux intuition.
- **IRQ affinity rules.** Interrupts from the BlueField PCIe
  function should be steered to the same NUMA node the polling
  threads run on, per the public BlueField / DPU User Manual's
  documented IRQ-affinity guidance. The Linux-side mechanism is
  `/proc/irq/<n>/smp_affinity`; the rule is to mirror the
  CPU-pinning choice on the IRQ side, not to invent a new IRQ
  mask.

### Per-tenant isolation

Multiple DOCA processes co-tenant on the same BlueField is a
supported shape (e.g. one DOCA-Flow process per representor plus
one DOCA-RDMA process for storage). The isolation primitives are
standard Linux; the DOCA-specific gotchas are what this skill owns.

- **cgroup-v2 cpu / memory / io controllers.** Per-tenant cpu
  shares (or hard caps), memory limits, and io weights enforce
  the resource split. The Linux-side mechanics live in the
  kernel cgroup-v2 documentation; the DOCA-specific gotcha is
  that a DOCA process's hugepage backing is accounted against the
  cgroup's memory budget — a `memory.max` value that fits the
  binary's resident set but starves the hugepage pool surfaces
  as a startup failure that LOOKS like a config error.
- **Network namespaces.** A per-tenant netns is the standard
  Linux mechanism for traffic isolation. The DOCA-specific
  gotcha is that a representor moved into a tenant netns has
  documented enumeration semantics — the representor is only
  visible to DOCA inside the netns that owns it, and a DOCA
  library called from a process in a different netns will not
  see it. The agent walks the operator through which netns owns
  which representor; it does not guess.
- **`numactl` / `taskset` per-tenant binding.** Each tenant's
  process gets its own CPU set bound via `numactl
  --cpunodebind` + `--membind` (NUMA-local memory) or
  `taskset -c <cpu-list>` (CPU-list only). The DOCA-specific
  gotcha is the rule named under [hardware-resource
  binding](#hardware-resource-binding): the chosen CPU set must
  not cross the NUMA boundary the NIC owns, or the per-tenant
  isolation gain is canceled by a NUMA-crossing latency hit.

The agent's discipline: confirm the tenant-isolation primitives
are in place BEFORE the workload starts, not after a co-tenant
complains. The corresponding workflow step lives at
[`TASKS.md ### isolation`](TASKS.md#isolation).

### Restart / recovery semantics

How a failed bare-metal process recovers depends on the launch mode
picked at [`TASKS.md ## configure`](TASKS.md#configure):

- **Direct / tmux launch.** No supervisor. A crash terminates the
  process; restart is a manual re-invocation. This is fine for
  interactive debug; it is the WRONG mode for a production
  workload where the operator expects the binary to come back
  after a host reboot.
- **systemd-supervised launch.** systemd's `Restart=` modes
  (documented in `systemd.service(5)`) govern auto-restart on
  exit. The mode names the operator may choose are the ones
  listed in `systemd.service(5)`; the agent does NOT invent
  modes (e.g. `Restart=on-failure-with-burst-cap` is not a real
  mode, but is the canonical class-of-thing the agent
  hallucinates when working from memory).
- **Crash-and-investigate vs supervisor-driven restart.** The
  cross-cutting rule (from
  [`doca-hardware-safety`](../doca-hardware-safety/SKILL.md)
  via [`## Safety policy`](#safety-policy) below) is that a
  failed bare-metal DOCA process that touched the device is
  HIGH-STAKES — the root cause MUST be cleared before the
  operator allows the supervisor to restart-loop the process.
  The auto-restart is the right default for a transient failure
  and the WRONG default for a recurring one.

## Version compatibility

For the canonical DOCA version-detection chain, the four-way match
rule (host package ↔ binary build ↔ BlueField firmware ↔
DOCA-version policy), and the headers-win-over-docs rule, see
[`doca-version`](../doca-version/SKILL.md). The body lives there;
this skill does not duplicate it.

**The bare-metal-deployment-specific overlay** is:

- **The binary's link-time `pkg-config doca-*` version MUST match
  the runtime `LD_LIBRARY_PATH`'d install.** A DOCA-linked binary
  resolves DOCA symbols against whatever `libdoca_*.so` is on the
  runtime `LD_LIBRARY_PATH` (or the default loader search path).
  If the binary was built against (for example) `doca-common 3.3.x`
  headers but the runtime install is `doca-common 3.2.x` (or vice
  versa), the binary fails at the first symbol that moved
  between releases — sometimes immediately at `dlopen` time,
  sometimes at the first call site, sometimes silently with
  wrong-behavior. Diagnose with
  [`doca-version TASKS.md ## debug`](../doca-version/TASKS.md#debug)
  layer 2.
- **`LD_LIBRARY_PATH` vs runtime-loader configuration.** When the
  operator picks the trace flavor at runtime
  (`/opt/mellanox/doca/lib/<arch>-linux-gnu/trace/`) per
  [`doca-setup ## configure`](../doca-setup/TASKS.md#configure)
  step 3, the runtime `LD_LIBRARY_PATH` is the contract. A
  systemd unit that does not propagate `LD_LIBRARY_PATH` into
  the unit's `Environment=` will silently use the release
  flavor and the operator's *"I'm running the trace build"*
  assumption is false.
- **Host x86 install version and BlueField device anchors are
  BOTH version anchors — and the device side splits into TWO
  distinct legs.** Same four-way overlay every per-library skill
  carries. The host-side `pkg-config --modversion doca-common`
  is the binary's link / runtime anchor; the device side splits
  into (i) NIC *firmware version* read via `flint -d <bdf> q`
  (look for the `FW Version:` line — this is the firmware image
  the silicon runs) and (ii) BFB-image DOCA version read via
  `bfver` + `cat /etc/mlnx-release` on the BlueField Arm
  console (this is the DOCA userland inside the BFB). Do NOT
  substitute `mlxconfig -d <bdf> q` for the FW-version leg —
  `mlxconfig` returns the firmware *configuration* dump
  (NV-config toggles like `LINK_TYPE_P1`, `INTERNAL_CPU_MODEL`),
  NOT the firmware version. Do NOT substitute `mlxprivhost`
  (configures privileged-host mode, not BFB version) or
  `bfb-info` (not a real NVIDIA-documented tool) for the
  BFB-image leg — both are common hallucinations explicitly
  banned in [`doca-version CAPABILITIES.md ## Capabilities and
  modes`](../doca-version/CAPABILITIES.md#capabilities-and-modes).
  Mismatched anchors are the canonical *"the docs say this
  should work but it does not"* failure mode. Capture all three
  (host pkg-config, per-device `flint q`, per-BlueField
  `bfver`) before debugging — and capture per-device, not
  globally, because BF2 and BF3 (or two BF3s from different
  procurement waves) on the same host are independent silicon
  with independent FW levels.

## Host-side DOCA upgrade workflow

When the operator is moving an already-running host from one DOCA
release to another (the canonical case: 3.1 → 3.3), the agent walks
THIS ordered ladder before quoting any `apt install` line. The
host-side portion is fully in-scope for this skill; the **BlueField
BSP / BFB / RShim / TMFIFO portion is OUT OF SCOPE** for this
bundle (per [`AGENTS.md ## Non-goals`](../../AGENTS.md#non-goals-questions-the-agent-should-recognize-and-refuse-politely)
item 7) and the agent routes the operator to NVIDIA's BlueField
BSP / DOCA Platform Framework documentation for those steps.

1. **Audit the host's current four-source state.** Per
   [`doca-version CAPABILITIES.md ## Version compatibility`](../doca-version/CAPABILITIES.md#version-compatibility),
   capture all four legs. The audit is read-only and is the rollback
   anchor — without a captured BEFORE state, "rollback" is a phrase
   the operator says, not a thing they can do.
2. **Verify the target release line.** Pin to a specific DOCA `X.Y.Z`
   from NVIDIA's public release notes (route via
   [`doca-public-knowledge-map`](../doca-public-knowledge-map/SKILL.md)
   — do NOT invent version strings). Read off the target's required
   OFED / kernel module version, the target BFB-image DOCA, and the
   target NIC firmware level from the release-notes *Supported NICs
   and Firmware* table.
3. **Walk the apt-repo + OS-matrix preconditions BEFORE any install
   line.** See
   [`doca-version CAPABILITIES.md ## Apt-repo and OS-matrix preconditions`](../doca-version/CAPABILITIES.md#apt-repo-and-os-matrix-preconditions).
   Pin the apt repo to the explicit `X.Y.Z` URL form, NOT to `latest`.
   Run `apt-cache policy <pkg>` for every package about to be
   installed. Confirm the host OS family AND point release is inside
   the target's supported sub-range — if it's outside, STOP and
   surface to the operator, do not silently proceed.
4. **Host-side upgrade is the first mutating step.** Quote the
   release-pinned `apt install` / `apt upgrade` line. Do NOT issue
   `apt install doca-all` reflexively; prefer the granular per-
   package list per the four-source partial-install table in
   `doca-version` so absent-source recovery is a single targeted
   install. Do NOT auto-reboot — let the operator schedule the
   reboot inside the documented maintenance window per
   [`doca-hardware-safety CAPABILITIES.md ## Safety policy`](../doca-hardware-safety/CAPABILITIES.md#safety-policy).
5. **Reboot the host, then re-walk the four-source audit AFTER.**
   The host-side upgrade is not declared "done" until the AFTER
   audit shows (i) all four sources coherent at the target version,
   (ii) `mlx5_core` / OFED at the target level (`modinfo mlx5_core
   | grep ^version`), (iii) all host-side host-PF / representor
   PCIe devices still enumerated (`lspci -d 15b3:`), (iv) host-
   visible NIC firmware version matches the per-device target from
   the release notes (`flint -d <bdf> q`).
6. **Handoff to BFB install — OUT OF SCOPE here, route per the contract.**
   Once the host is green, the BFB-side install (push BFB to BlueField via
   RShim, eMMC image install, BFB-side firmware update, BFB-side OS bring-
   up to `Linux up` / `DPU is ready`, TMFIFO recovery, BFB-side apt
   vocabulary, `bf.cfg`) is **externally-productized** (BlueField BSP layer
   plus, for fleet-scale deployments, DOCA Platform Framework). The
   *operational sequencing ladder* for the single-host BFB lifecycle
   itself — what evidence to capture, in which order, and which recovery
   action each evidence pattern lattices to — is in scope and lives in
   [`TASKS.md ## bluefield-lifecycle`](TASKS.md#bluefield-lifecycle); the
   *productized framework / BSP / BFB image / firmware tooling* is out
   of scope for this bundle's strict-1:1 monorepo alignment, but the
   bundle's `AGENTS.md ## Non-goals #7` contract still applies: the agent
   MUST produce the three-part response shape (recognize + name boundary
   + **route with substance**). For step (c), consult the per-product
   rows in the `doca-public-knowledge-map` routing table:
   [BlueField BSP / BFB / `bfb-install` / RShim / TMFIFO / `bf.cfg`](../doca-public-knowledge-map/SKILL.md#externally-productized-doca-software--not-in-this-bundle-but-here-is-where-to-route)
   for the single-host case, and [DOCA Platform Framework (DPF)](../doca-public-knowledge-map/SKILL.md#externally-productized-doca-software--not-in-this-bundle-but-here-is-where-to-route)
   for fleet-scale K8s-driven DPU provisioning. If the BFB-install symptom
   touches firmware state (`mlxconfig` / `flint` / `mlxfwmanager`), also
   load the [NVIDIA Firmware Tools (MFT)](../doca-public-knowledge-map/SKILL.md#externally-productized-doca-software--not-in-this-bundle-but-here-is-where-to-route)
   row. If the BFB-install left the BlueField unreachable, the recovery
   path is the [BlueField BMC](../doca-public-knowledge-map/SKILL.md#externally-productized-doca-software--not-in-this-bundle-but-here-is-where-to-route)
   row. The agent **must** name the symptom-matching gotcha class from the
   relevant row (e.g. "`bfb-install` exit 0 with `NIC firmware update failed`
   is the BSP row's #1 partial-failure signature — bisect with `flint -d
   <bdf> q`") rather than handing the user a bare URL. The agent does NOT
   synthesize BFB install / RShim / TMFIFO / `bf.cfg` mechanics from
   training memory.
7. **End-to-end success contract.** A DOCA host-side upgrade is
   "done" only when, AFTER the BFB-side portion completes
   independently, all of the following hold:
   - Host four-source audit at target version, coherent.
   - BlueField-side four-source audit at target version, coherent.
   - Per-device `flint -d <bdf> q` FW Version on each BF / NIC matches
     the release-notes target.
   - `doca_caps --list-devs` enumerates every expected device with
     expected capabilities.
   - The operator's own canonical smoke (a doca-flow / doca-rdma /
     doca-comch sample) runs end-to-end against the target install.

The discipline mirrors the universal verification contract end-to-
end: every step's preconditions established by the prior steps, no
"upgrade complete" claimed on exit-code alone, the BFB / BSP
boundary explicitly named and routed out rather than silently
synthesized. See [`AGENTS.md ## The universal verification contract`](../../AGENTS.md#the-universal-verification-contract).

### RShim instance ↔ BlueField disambiguation (canonical one-liner)

Multi-DPU hosts expose multiple RShim character devices under
`/dev/rshim<N>/`. To map each RShim instance to a specific
BlueField (so `bfb-install --rshim /dev/rshim<N>` targets the
right DPU, and `cat /dev/rshim<N>/misc` reads the right
console), read the `DEV_NAME` field of each `misc` file:

```bash
for r in /dev/rshim*/misc; do
  echo "== $r =="
  grep -E '^DEV_NAME' "$r"
done
```

`DEV_NAME` returns the PCIe BDF that RShim instance is attached
to (e.g. `DEV_NAME pcie-0000:03:00.2`); cross-match against
`lspci -d 15b3: -nn` to identify which physical BlueField is on
which `/dev/rshim<N>`. This one-liner is the canonical
disambiguation step before any per-DPU `bfb-install`, `rshim`
config edit, or console capture on a multi-DPU host; skipping it
is the #1 cause of *"I flashed the wrong DPU"* incidents. The
mapping is consumed downstream by every step in
[`TASKS.md ## bluefield-lifecycle`](TASKS.md#bluefield-lifecycle)
that takes a `--rshim` or `/dev/rshim<N>` argument.

## Error taxonomy

Bare-metal-deployment errors fall into SEVEN layers, each with its
own owner. The agent walks the layers in this order; conflating
them wastes debug time and blames the wrong layer.

1. **Process won't start at all.** Symptoms: the shell reports
   *"command not found"*, *"permission denied"*, *"cannot
   execute binary file: Exec format error"*, or a missing
   shared object error from the dynamic loader (`error while
   loading shared libraries: libdoca_*.so: cannot open shared
   object file`). Causes: binary not on `PATH` or not executable
   (`chmod +x`); binary built for a different arch than the
   host (x86 binary on the BlueField Arm or vice versa);
   `LD_LIBRARY_PATH` not set so the loader cannot find
   `libdoca_*.so`. Resolution: confirm the binary's arch
   (`file <binary>`) matches the host's arch (`uname -m`);
   confirm the binary's required shared objects are reachable
   (`ldd <binary>`); set `LD_LIBRARY_PATH` per
   [`doca-setup ## configure`](../doca-setup/TASKS.md#configure)
   step 3. Owner: this skill + [`doca-setup`](../doca-setup/SKILL.md).
2. **Process starts but exits immediately.** Symptoms: the
   process runs for milliseconds, exits with a non-zero status,
   prints a short message about a missing config file, a
   missing environment variable, or `EAL: Cannot get hugepage
   information` from DPDK. Causes: a required env var (e.g.
   `DOCA_LOG_LEVEL`, a per-library env knob documented in the
   public DOCA Programming Guide) is unset; the config file the
   binary expects is missing or at the wrong path; hugepages
   are not reserved per [`doca-setup ## configure`](../doca-setup/TASKS.md#configure)
   step 4; the binary's argp surface rejected the operator's CLI
   args. Resolution: read the binary's stdout / stderr from
   the launch mode in use (direct = terminal; tmux = pane;
   systemd = `journalctl -u <unit>`); cross-check the
   documented env vars in the public DOCA Programming Guide;
   re-run hugepage reservation. Owner: this skill + the
   per-library skill (for library-specific env vars).
3. **Process runs but cannot find the device.** Symptoms: the
   binary stays up but the per-library bring-up reports *"no
   matching device"*, *"representor not found"*, or
   `DOCA_ERROR_NOT_FOUND` from the documented device-open call.
   Causes: PCI address wrong (the operator passed a BDF that
   does not exist or does not match the BlueField); representor
   not enumerated by the kernel (the eswitch is not in
   `switchdev` mode); devlink mode wrong; the binary is in a
   netns that does not own the representor. Resolution: confirm
   the device shape from `devlink dev show` and `lspci -d
   15b3:` per [`doca-setup ## configure`](../doca-setup/TASKS.md#configure)
   step 5; re-read the public DOCA Programming Guide's
   device-open contract for the library in use; do NOT
   substitute a BDF from memory. Owner: this skill +
   [`doca-setup`](../doca-setup/SKILL.md).
4. **Process attaches to the device but the workload errors.**
   Symptoms: the binary stays up, the device is open, the
   per-library bring-up looks clean, but a runtime call returns
   `DOCA_ERROR_*` per the cross-library taxonomy in
   [`doca-programming-guide CAPABILITIES.md ## Error taxonomy`](../doca-programming-guide/CAPABILITIES.md#error-taxonomy).
   Causes: per-library configuration error (a pipe / queue /
   buffer the per-library skill is the authority on); a
   capability the operator's BlueField does not have at the
   running firmware version. Resolution: defer to
   [`doca-debug`](../doca-debug/SKILL.md) for the cross-cutting
   ladder, then to the matching `libs/<library>` skill for the
   library-specific debug overlay. Owner: the per-library skill
   + [`doca-debug`](../doca-debug/SKILL.md).
5. **Process runs correctly but OOMs / gets killed /
   signal-mishandled.** Symptoms: the binary disappears with
   no log line; `dmesg` shows an OOM-killer entry naming the
   binary; the supervisor reports the process exited with
   `SIGKILL` / `SIGTERM`; counters reset because the process
   restarted. Causes: the cgroup-v2 `memory.max` is too small
   for the binary's working set plus hugepage backing; the
   host is over-committed; the binary does not handle `SIGTERM`
   from the supervisor cleanly; NUMA imbalance forces a
   memory blowup on one node. Resolution: read the supervisor's
   exit-status / signal record; check `dmesg` for OOM
   evidence; re-check cgroup limits per
   [`### isolation`](TASKS.md#isolation); confirm the binary's
   documented signal-handling contract from the public DOCA
   Programming Guide. Owner: this skill + the operator's host
   OS team.
6. **Process is in a restart loop.** Symptoms: the supervisor
   keeps re-launching the binary; each launch exits with the
   same exit signature; the device the binary touches is
   reporting odd errors that may be caused by the loop itself.
   Causes: a real underlying failure (any of layers 1-5) that
   the supervisor's `Restart=` policy is hiding behind
   automatic re-launches. Resolution: STOP the supervisor (per
   the HIGH-STAKES rule in [`## Safety policy`](#safety-policy));
   read the binary's LAST full log; walk the taxonomy from
   layer 1 against the captured evidence; only re-enable the
   supervisor once the root cause is identified. Owner: this
   skill (the cross-cutting rule) + whichever layer the root
   cause turns out to live in.
7. **Co-tenant noise — another bare-metal process on the same
   NUMA / same NIC is interfering.** Symptoms: the binary
   behaves correctly in isolation; introducing a second DOCA
   process (or any other process) on the same BlueField makes
   the first one's per-library counters degrade, latency
   climb, or throughput collapse; the symptom does NOT
   reproduce on a quiet host. Causes: cgroup-v2 limits not in
   place; CPU pinning not in place; the second process is
   cross-NUMA against the BlueField's PCIe root complex; the
   second process is competing for the same hugepage pool.
   Resolution: confirm tenant-isolation primitives per
   [`### isolation`](TASKS.md#isolation) BEFORE introducing
   the second tenant, not after; if the symptom only appears
   under co-tenancy, the diagnosis is multi-tenant, not
   per-library. Owner: this skill.

DOCA library calls inside the binary return `DOCA_ERROR_*`
according to the cross-library taxonomy in
[`doca-programming-guide CAPABILITIES.md ## Error taxonomy`](../doca-programming-guide/CAPABILITIES.md#error-taxonomy);
that taxonomy becomes relevant at layer 4 above and is owned by
[`doca-debug`](../doca-debug/SKILL.md) plus the matching per-library
skill, not by this one.

## Observability

Documented observability surfaces, in the order the agent reaches
for them. Three layers, each tied to a launch mode and a documented
host-side primitive. Healthy means all three agree.

- **Process-output layer (FIRST).** The binary's own stdout /
  stderr is the first place to look. Where it arrives depends on
  the launch mode: direct = the terminal the operator launched
  from; tmux / screen = the pane buffer (reattach with `tmux
  attach` / `screen -r`; output may have scrolled off if the
  pane buffer is small); systemd-supervised = captured into
  journald per the unit's `StandardOutput=` / `StandardError=`
  settings, tailed with `journalctl -u <unit> -f` (the unit name
  is the operator's choice and lives in their systemd policy).
  The agent does NOT invent a journald query against an
  unnamed unit; the unit name comes from the operator's
  deployment, not from memory.
- **Device-state introspection layer (SECOND).** The host-side
  picture of the device the binary is touching: `devlink dev
  show` lists the BlueField PCIe device; `ip link show` lists
  the device's netdevs and any representors; `mlxconfig -d
  <pci> query` is the QUERY-ONLY surface for the device's
  firmware configuration. The agent uses `mlxconfig query`
  freely as an observability primitive but treats `mlxconfig
  set` as a hardware-state change — it MUST defer to
  [`doca-hardware-safety`](../doca-hardware-safety/SKILL.md)
  before recommending any `mlxconfig set` invocation. The
  documented query commands the agent may quote live in the
  public BlueField / DPU User Manual reached through
  [`doca-public-knowledge-map`](../doca-public-knowledge-map/SKILL.md).
- **Per-tenant resource visibility layer (THIRD, load-bearing
  for multi-tenant).** When the binary is one of several DOCA
  processes on the same BlueField, the per-tenant resource
  picture is what tells the operator whether the tenant
  isolation primitives are still in effect. cgroup-v2 reports
  `cpu.stat` / `memory.stat` / `io.stat` per cgroup;
  `numactl --hardware` reports the NUMA topology and per-node
  memory pressure; `perf stat -p <pid>` reports per-process
  CPU / cache behavior. The agent quotes these tools at class
  level; the exact metric names live in the Linux man pages
  and in the kernel cgroup-v2 documentation.

Cross-cutting host-side debug (kernel version, driver loaded /
not loaded, hugepage allocation health beyond reserved-vs-used,
PCIe link state, `dmesg`) lives in
[`doca-debug TASKS.md ## debug`](../doca-debug/TASKS.md#debug)
and [`doca-setup TASKS.md ## Command appendix`](../doca-setup/TASKS.md#command-appendix);
this skill names only the bare-metal-deployment-specific
surfaces.

## Safety policy

> **Overlay on the bundle-wide hardware-safety meta-policy.** The
> rules below are this skill's per-artifact overlay on the
> cross-cutting rules in
> [`doca-hardware-safety` CAPABILITIES.md ## Safety policy](../doca-hardware-safety/CAPABILITIES.md#safety-policy)
> (specifically
> [### Per-artifact overlay pattern](../doca-hardware-safety/CAPABILITIES.md#per-artifact-overlay-pattern)).
> When the two layers disagree, the stricter wins; when either
> layer says STOP, the agent stops.

The cross-cutting safety surface for any DOCA-linked binary
deployed bare-metal. Per-library skills add their own overlays
(e.g. doca-flow's pipe-validate-before-commit rule); the
cross-cutting rules below apply across every library.

- **Smoke before bulk (load-bearing).** Before any real workload
  touches the device, the agent walks the smoke sequence in
  [`TASKS.md ## test`](TASKS.md#test): (a) trivial-arg
  invocation (the binary's `--help` or `--version` equivalent,
  proving the binary executes and the loader resolves DOCA
  symbols); (b) liveness-equivalent invocation (the binary
  starts, opens the device, prints its documented bring-up
  lines, exits cleanly on `SIGTERM`); (c) trivial-workload
  invocation (one packet / one operation through the binary,
  per the library's documented liveness contract); only then is
  the binary ready for production workload. Skipping this and
  going straight to bulk is the most common reason *"the binary
  works on the bench but blows up under load"*.
- **Failed bare-metal process is HIGH-STAKES — clear the root
  cause BEFORE restarting.** A bare-metal DOCA process that
  touched the device and then failed has potentially left the
  device in a documented-but-not-guaranteed-clean state. The
  operator's rule: do NOT let a supervisor's `Restart=` policy
  re-launch the binary while the underlying failure is
  uncleared. A binary in an auto-restart loop that keeps
  touching the device burns BlueField cycles, fills the log
  surface, and obscures the actual root cause. This rule is
  the bare-metal analog of the failed-pod-restart-is-HIGH-STAKES
  rule in [`doca-container-deployment`](../doca-container-deployment/SKILL.md);
  it applies just as strongly to systemd-supervised binaries.
- **Do not invent PCI addresses, NUMA node numbers, representor
  names, devlink paths, hugepage allocation amounts, or
  `Restart=` mode names.** All of these are per-host or
  per-image specifics. The PCI BDF the binary should attach to
  lives in `lspci -d 15b3:` output on the target host; the NUMA
  topology lives in `numactl --hardware`; the representor
  naming convention lives in the public BlueField / DPU User
  Manual for the operator's BlueField OS image; the `Restart=`
  mode names live in `systemd.service(5)`. Quote from these
  live sources; do NOT supply a BDF, a representor name, or a
  systemd mode name from memory. This is the load-bearing
  first-run failure mode for this skill — the operator pastes
  the agent's BDF into a launch line, the binary attaches to
  the wrong function (or to no function), and the resulting
  debug spiral is hours long.
- **Confirm tenant-isolation primitives BEFORE the workload
  starts, not after a co-tenant complains.** Per
  [`### isolation`](TASKS.md#isolation), the cgroup-v2 / netns /
  `numactl` configuration must be in place when the first
  tenant launches. Adding isolation after a co-tenant is
  already misbehaving means the first symptom the operator
  sees is co-tenant noise from layer 7 of the error taxonomy —
  which looks like *"the binary regressed"* and burns time on
  the wrong diagnosis.
- **Hardware-state changes leave this skill.** Any change that
  touches device firmware (`mlxconfig set`, BFB reflash,
  firmware burn, BlueField mode flip, kernel-boot-parameter
  changes for IOMMU or hugepage reservation) is OUT OF SCOPE
  for this skill. The agent MUST hand off to
  [`doca-hardware-safety`](../doca-hardware-safety/SKILL.md) for
  the change-application discipline and only return here once
  the hardware-state change is complete.
- **The per-library skill owns the per-library safety overlay.**
  Doca-flow's pipe-validate-before-commit rule, doca-rdma's
  queue-pair-state-machine rule, doca-comch's
  channel-handshake-discipline rule, and similar per-library
  safety rules are owned by the matching per-library skill,
  not by this one. The agent reads both layers in parallel and
  applies them; this skill names the cross-cutting baseline.

## Public-source pointer

The canonical public sources for the bare-metal deployment runtime
are:

- The **DOCA Programming Guide** on `docs.nvidia.com`, reachable
  through
  [`doca-public-knowledge-map ## Library- and module-specific guides`](../doca-public-knowledge-map/SKILL.md#library--and-module-specific-guides),
  for the per-library device-open / capability / runtime
  contracts.
- The **DOCA Installation Guide** on `docs.nvidia.com`, reachable
  through
  [`doca-public-knowledge-map ## Public documentation entry points`](../doca-public-knowledge-map/SKILL.md#public-documentation-entry-points),
  for the host-side and BlueField-Arm-side install layouts plus
  the documented env-var surface.
- The **BlueField / DPU User Manual** on `docs.nvidia.com`,
  reachable through the same routing table, for the documented
  device-enumeration surface, representor-naming convention,
  IRQ-affinity guidance, and firmware-query surface.
- The Linux man pages for `numactl(8)`, `taskset(1)`,
  `systemd.service(5)`, `systemd.unit(5)`, and the kernel
  cgroup-v2 documentation, for the standard primitives this
  skill inherits.

Verify that the version of each guide matches the host's DOCA
install version, the BlueField's firmware version, and the
binary's link-time version per
[`## Version compatibility`](#version-compatibility) — flag names,
documented device-enumeration commands, and `Restart=` modes
evolve, so anything quoted from memory is suspect.
