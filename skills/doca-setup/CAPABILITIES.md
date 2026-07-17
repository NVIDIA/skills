# DOCA setup — capabilities, version compatibility, errors, observability, safety

**Where to start:** Pick the H2 anchor that matches your question
(modes / version / errors / observability / safety) and read that
section end-to-end before issuing a command. Tables in each section
are the load-bearing content; the prose around them is interpretation.

Read this file when the loader sent you here from [SKILL.md](SKILL.md). For the env workflows that *use* the surface described here, see [TASKS.md](TASKS.md). For the program-side counterparts (build flavor selection rationale, the universal lifecycle, the cross-library `DOCA_ERROR_*` taxonomy, the program-side safety policy), see [`doca-programming-guide`](../doca-programming-guide/SKILL.md). For where to find official documentation, the on-disk layout of an installed DOCA package, or the official Installation Guide, route through [`doca-public-knowledge-map`](../doca-public-knowledge-map/SKILL.md).

This file describes the **install / build / runtime *environment*** that any DOCA program assumes has been verified before its own workflows begin. It is deliberately library-agnostic and program-agnostic; everything here is about the host the program runs on, not about the program itself.

> Lint note: this file references `/opt/mellanox/doca` and `docs.nvidia.com` paths several times. For most library skills the lint flags this and asks for cross-links to `doca-public-knowledge-map` instead, since URL/path duplication is normally drift waiting to happen. For `doca-setup` specifically the references are *intrinsic* — the skill's whole job is to operate on the install tree and verify it. The repeated paths are intentional; do not refactor them out.

## Pattern overview

Every env-class concern this skill teaches resolves into one of FIVE
patterns. Reach for the pattern first, then drill into the matching
H2 anchor; the patterns are CLASSES, not use cases.

| Pattern | When it applies (class shape)                              | Where it lives                                          |
|---------|------------------------------------------------------------|---------------------------------------------------------|
| 1. Reach an install | User has no DOCA reachable from where they are now | [TASKS.md ## no-install](TASKS.md#no-install) (NGC Path 0) |
| 2. Detect an install | User has *something* installed and must figure out what | [`## Version compatibility`](#version-compatibility) + [TASKS.md ## test](TASKS.md#test) |
| 3. Wire the build to the install | Headers / `*.pc` / `LD_LIBRARY_PATH` / build flavor | [`## Capabilities and modes`](#capabilities-and-modes) + [TASKS.md ## configure](TASKS.md#configure) |
| 4. Wire the runtime to the device | Hugepages / representors / devlink / kernel modules | [`## Observability`](#observability) + [TASKS.md ## configure](TASKS.md#configure) |
| 5. Diagnose env vs program | Symptom looks like a bug; agent must rule env out first | [`## Error taxonomy`](#error-taxonomy) + [TASKS.md ## debug](TASKS.md#debug) |

Two principles cut across all five patterns:

- **Env layers run in order: install → capability → runtime.** A
  hugepages question with an unreachable install is the install
  question. Skipping a layer is the single most common
  failure mode the env-debug ladder catches.
- **Env-class only.** The instant the answer becomes "rewrite the
  program", hand off to
  [`doca-programming-guide`](../doca-programming-guide/SKILL.md). The
  patterns above never tell the user to change their code.

## Capabilities and modes

DOCA install is layered into three orthogonal env axes. Pick the right combination *before* writing or building any code. Program-side selection of which mode the program initializes (host vs DPU vs switch) and which build flavor it links against is in [`doca-programming-guide CAPABILITIES.md ## Capabilities and modes`](../doca-programming-guide/CAPABILITIES.md#capabilities-and-modes); this section covers what the env supports.

**Install profiles.** Determined by the package profile selected at install time (see the Installation Guide via `doca-public-knowledge-map`).

| Profile | What it pulls in | When to use |
| --- | --- | --- |
| `doca-all` | Full superset: SDK + samples + applications + tools + `doca-ofed` + `doca-networking`. | Default for any developer host, BlueField, or CI machine. Recommended for first-time setup. The public NGC DOCA container ([TASKS.md ## no-install](TASKS.md#no-install) Path 0) ships this profile. |
| `doca-ofed` | OFED userspace + kernel modules only (RDMA stack). | Constrained host that only needs the underlying RDMA stack, no DOCA libraries. Rare in a development context. |
| `doca-networking` | The DOCA networking subset (Flow, telemetry, etc.) on top of `doca-ofed`. | Constrained images that ship only the networking libraries. Most agent users will instead want `doca-all`. |

**The agent's rule:** if a sample build fails with a missing `*.pc` file (`doca-flow.pc`, `doca-rdma.pc`, …), the most likely cause is the wrong install profile, not a code bug. Surface this hypothesis before any code-level diagnosis.

**Build flavor — env side.** Two `*.so` trees ship with every DOCA install. The choice of which one a program links against is a programming decision (see [`doca-programming-guide CAPABILITIES.md ## Capabilities and modes`](../doca-programming-guide/CAPABILITIES.md#capabilities-and-modes)); the env side is *where each tree lives on disk* and *how to make the runtime see it*.

| Flavor | On-disk location | How a program reaches it |
| --- | --- | --- |
| Release | `/opt/mellanox/doca/lib/<arch>-linux-gnu/` | Default; `pkg-config doca-<library>` resolves here. |
| Trace | `/opt/mellanox/doca/lib/<arch>-linux-gnu/trace/` | Either link with the `doca-<library>-trace` `pkg-config` module at build time, or set `LD_LIBRARY_PATH=/opt/mellanox/doca/lib/<arch>-linux-gnu/trace:$LD_LIBRARY_PATH` at runtime. |

**Runtime modes — env-side enablement.** Most DOCA libraries (Flow in particular) run in one of three host configurations. The env determines which modes are *available* on this host; the program selects one at init time.

| Mode | Env enablement check |
| --- | --- |
| Host | x86 / Arm host with BlueField visible as a SmartNIC. Confirmed by `devlink dev show` listing the BlueField PCIe device, plus PF/VF representors under `/sys/class/net/`. |
| DPU | Inside the BlueField OS itself (Arm). The env is the BlueField BFB image; confirmed by `lsb_release -a` matching the BFB release. |
| Switch | DPU-side, with the BlueField in switch (DPU) mode. Confirmed by `mlxconfig -d <pcie> q INTERNAL_CPU_MODEL` reporting `EMBEDDED_CPU(1)` and the eswitch in `switchdev` mode. |

The agent must clarify *which* mode the user expects before recommending any port-id or representor naming. Program-side mode-selection guidance is in [`doca-programming-guide`](../doca-programming-guide/SKILL.md).

## Version compatibility

For the canonical DOCA version-detection chain (`pkg-config --modversion doca-common` → `cat applications/VERSION` → `doca_caps --version` → BFB version on BlueField), the four-way match rule, NGC container semantics, the headers-win-over-docs rule, and the routing to the DOCA Compatibility Policy, see [`doca-version`](../doca-version/SKILL.md). The body lives there; this skill does not duplicate it.

**BFB-version probe ban (same as in `doca-version`).** When the user is configuring a BlueField host and the agent reaches for the BFB-image DOCA version (the (d) leg of the four-source chain), the only legitimate probes are `bfver` and `cat /etc/mlnx-release` on the BlueField Arm console. Do NOT substitute `mlxprivhost` (configures privileged-host mode, not version) or `bfb-info` (not a real NVIDIA-documented tool) — both are common hallucinations even when the visible binary is present on `/usr/bin/` of the x86 host. See [`doca-version CAPABILITIES.md ## Capabilities and modes`](../doca-version/CAPABILITIES.md#capabilities-and-modes).

**The env-side overlay** is responsible for making the detection chain *work* on this host before any version question can be answered:

- `PKG_CONFIG_PATH` must include `/opt/mellanox/doca/infrastructure/lib/pkgconfig` so that `pkg-config --modversion doca-<library>` resolves at all. The env-setup procedure in [`TASKS.md ## configure`](TASKS.md#configure) verifies this; partial-install diagnosis lives in [`doca-version TASKS.md ## debug`](../doca-version/TASKS.md#debug) layer 2.
- The on-disk paths the detection chain reads (`/opt/mellanox/doca/applications/VERSION`, `$(pkg-config --variable=includedir doca-common)/doca_version.h`, the `*.pc` directory) are env-side artifacts — see [`## Capabilities and modes`](#capabilities-and-modes) for the install-tree layout that places them where the chain expects.
- On the *no-install* path (NGC container, per [`TASKS.md ## no-install`](TASKS.md#no-install) Path 0), the env-side overlay is that the four-way match is *of the container tag* the user pulled; the container's headers, `*.so`, samples, and `doca_caps` are guaranteed consistent by construction. The agent must still report which path was used so the user knows which install they verified.

## Error taxonomy

Env-class errors that the agent should recognize and disambiguate before falling back to a program-internal diagnosis. The cross-library, program-side `DOCA_ERROR_*` taxonomy lives in [`doca-programming-guide CAPABILITIES.md ## Error taxonomy`](../doca-programming-guide/CAPABILITIES.md#error-taxonomy); the library-specific overlays live in the matching library skills.

| Error surface | Typical message | Most-likely cause | Next step |
| --- | --- | --- | --- |
| `pkg-config` | `Package 'doca-flow' was not found` | Wrong install profile (`doca-flow.pc` is missing) **or** `PKG_CONFIG_PATH` does not include `/opt/mellanox/doca/infrastructure/lib/pkgconfig`. | Run `ls /opt/mellanox/doca/infrastructure/lib/pkgconfig/`. If `doca-flow.pc` exists, fix `PKG_CONFIG_PATH` (see [`TASKS.md ## configure`](TASKS.md#configure)). If it doesn't, reinstall with `doca-all` — or, if no install is reachable, the user should reach one via [`TASKS.md ## no-install`](TASKS.md#no-install). |
| First-app workflow | `/opt/mellanox/doca/samples/` is empty, missing, or `ls` returns `No such file or directory` on a host where `pkg-config --modversion doca-common` succeeds | Partial install: the `doca-samples` package was not installed alongside `doca-common`. The bundle's canonical first-app workflow ([AGENTS.md `## Ground rules` rule 5](../../AGENTS.md#ground-rules-every-agent-must-follow)) requires *modify-from-shipped-sample*; with no `samples/` tree the workflow cannot apply on this host. | The agent MUST say so explicitly to the user, MUST NOT scaffold `main.c` / `Makefile` / `Dockerfile` from API memory, and MUST route to one of: (a) targeted `apt install doca-samples` (or platform equivalent), (b) pivot to the NGC DOCA container via [`TASKS.md ## no-install`](TASKS.md#no-install) Path 0, where samples are guaranteed present at the container tag, or (c) point the user at the [DOCA samples in the public archive on `docs.nvidia.com`](https://docs.nvidia.com/doca/sdk/) and have them install the `doca-samples` package before continuing. |
| `meson` configure | `Dependency doca-flow found: NO` | Same as above (meson uses `pkg-config` under the hood). | Same as above. |
| `meson` configure | `compiler not found` / `meson not found` | Build toolchain missing on this host (common on minimal BlueField images). | Install the appropriate `build-essential` / `meson` / `ninja-build` packages for the OS. The NGC DOCA container ships these by default. |
| Compile time | `error: unknown type name 'doca_flow_pipe_cfg'` | The header path is not on the include search path, or the headers are from an older DOCA than the library you intend to link against. | `pkg-config --cflags doca-flow` to obtain the canonical include flags; cross-check the version ([`TASKS.md ## test`](TASKS.md#test)). |
| Link time | `undefined reference to 'doca_flow_pipe_create'` | The library is not on the link line (`-ldoca_flow` missing), or the library found at link time is older than the headers. | `pkg-config --libs doca-flow`; verify `--modversion` of the package matches the headers' release. |
| Runtime | `EAL: No free 2048 kB hugepages reported` | Hugepages not mounted or insufficient. | Mount and reserve hugepages ([`TASKS.md ## configure`](TASKS.md#configure)). |
| Runtime | `Cannot find any working PCI driver` | Kernel modules for the device not loaded, or the BlueField is not in the expected mode (host vs. DPU vs. switch). | `lsmod | grep mlx5`; verify mode via `mlxconfig`. Inside an NGC container with no real NIC, this is *expected* and the right move is to graduate the user to a hardware path (Path A or Path C in [`TASKS.md ## no-install`](TASKS.md#no-install)). |
| Runtime | `representor X not found` | Representors not enabled on the PF (`devlink dev eswitch set ... mode switchdev`), or the application is asking for a representor index that doesn't exist on this hardware. | `devlink dev show`; cross-check requested index against `cat /sys/class/net/*/phys_port_name`. |
| Runtime | Application starts, exits cleanly, no traffic effect | Often a silent steering-mode mismatch (HWS expected, SWS active) or a switch-mode app run in non-switch mode. | Re-run `doca_caps`; cross-check the steering mode the application requested via library init args. |

The taxonomy above is **env-class only**. Program-internal `DOCA_ERROR_*` codes (`DOCA_ERROR_BAD_STATE`, `DOCA_ERROR_NOT_SUPPORTED`, etc.) live in [`doca-programming-guide CAPABILITIES.md ## Error taxonomy`](../doca-programming-guide/CAPABILITIES.md#error-taxonomy); the library-specific overlay (e.g. Flow's mapping from API call to which `DOCA_ERROR_*` it returns) lives in the matching library skill — for Flow, see [`doca-flow CAPABILITIES.md ## Error taxonomy`](../libs/doca-flow/CAPABILITIES.md#error-taxonomy).

## Observability

What *"healthy install"* looks like under observation. The agent should run these checks (or have the user run them) **before** recommending any code-level change. Program-side observability (DOCA log levels, capability snapshots, library counters) lives in [`doca-programming-guide CAPABILITIES.md ## Observability`](../doca-programming-guide/CAPABILITIES.md#observability).

**Install layer:**

```bash
ls /opt/mellanox/doca                                  # SDK root present
ls /opt/mellanox/doca/samples                          # samples shipped
ls $(pkg-config --variable=includedir doca-common)           # headers shipped
ls /opt/mellanox/doca/infrastructure/lib/pkgconfig/    # *.pc files shipped
pkg-config --list-all | grep -i doca                   # what the build env can find
pkg-config --modversion doca-common                    # the unified version
```

**Capability layer (DPU and Flow):**

```bash
doca_caps                                              # device capabilities snapshot
doca_caps --version                                    # runtime version
```

**Runtime layer (host or DPU):**

```bash
mount | grep huge                                      # hugepages mounted
cat /proc/meminfo | grep -i huge                       # hugepages reserved
devlink dev show                                       # devices visible
ls /sys/class/net/                                     # network interfaces
cat /sys/class/net/*/phys_port_name                    # representor names
```

The presence of all of the above (without errors or empty output where output is expected) is the precondition for *any* program-level work. The agent's investigation order on an env-class report is exactly **install → capability → runtime**, in that order — never start at the application layer when these have not been verified.

Inside the NGC DOCA container, the install and capability layers above will respond normally; the runtime layer will partially or fully report no real hardware (no hugepages mounted by default, no `devlink` devices visible, no representors). That is the expected state for the build / read / learn loop the container is for; for runtime against real hardware, the user has to graduate to a hardware path (see [`TASKS.md ## no-install`](TASKS.md#no-install) Paths A and C).

## Safety policy

DOCA install and runtime preparation are **shared system state**. Several env actions affect more than just the user's program; the agent must surface this before recommending them. The program-side safety policy (validate-before-commit, stage-first-widen-later) lives in [`doca-programming-guide CAPABILITIES.md ## Safety policy`](../doca-programming-guide/CAPABILITIES.md#safety-policy).

1. **Never modify `/opt/mellanox/doca/lib*/`.** The shipped libraries are the release; rewriting them voids the install and breaks any other DOCA application on the host. Build flavor changes go via `LD_LIBRARY_PATH` or the `-trace` `pkg-config` module, never by editing the `lib/` tree.

2. **Hugepage mounts and reservations are global.** Adding hugepages reduces memory available to the kernel and to other applications; removing or remounting hugepages while another DOCA / DPDK application is running will crash that application. Before recommending a hugepages change, ask the user whether anything else on this host is using DOCA or DPDK; if yes, do the change in coordination, not in isolation. The same applies to `mlxconfig` changes that require a reset, and to `devlink dev eswitch set ... mode switchdev`.

3. **Never auto-`reboot` or auto-power-cycle.** `mlxconfig` and BFB updates require a host reset to take effect. The agent should produce the *commands* and explain the requirement, but must not chain them with an unattended reboot — let the user confirm their environment can absorb the reboot.

4. **The NGC container is build / read / learn only — not a runtime substitute.** The public NGC DOCA container at `nvcr.io/nvidia/doca/doca` is the canonical Stage-1 path for any user on macOS, Windows, or Linux without DOCA, but it is *not* a substitute for running against real hardware. DPDK / DOCA calls that require a real NIC, real DPU, real hugepages on the host kernel, or real driver presence will fail inside the container — and that failure is correct. The agent must not recommend `--privileged` / `--device` workarounds to "make it run anyway"; the right move when the user needs runtime is to graduate them from Path 0 to Path A or Path C in [`TASKS.md ## no-install`](TASKS.md#no-install).

5. **Be explicit about what changes when the user installs DOCA.** Installing the host package modifies kernel modules (`mlx5_core`, OFED), adds udev rules, and (for `doca-all`) installs services that may auto-start. Surface this before recommending an install on a host that has other workloads.

## Universal verification contract

Every answer that recommends a change (build / deploy / configure / modify / install / upgrade) MUST end with the **5-step verification contract** below. This is the bundle-wide shape that every per-artifact `## test` anchor instantiates with artifact-specific commands. Skipping any step makes the answer ineligible to declare the task complete.

| Step | What the agent must do | Why this step exists | Forbidden shortcut |
| --- | --- | --- | --- |
| **1. Preconditions** | Name the exact preconditions that must be true *before* applying the change, and how to verify each. For DOCA work this is at minimum: (a) versions match (run the four-source detection chain from [`doca-version CAPABILITIES.md ## Capabilities and modes`](../doca-version/CAPABILITIES.md#capabilities-and-modes)); (b) required hardware is visible (`lspci -d 15b3:`, `devlink dev show`, representor enumeration); (c) required packages are installed (`pkg-config --modversion doca-<library>`); (d) the rollback path is documented (per [`doca-hardware-safety CAPABILITIES.md ## Safety policy`](../doca-hardware-safety/CAPABILITIES.md#safety-policy) if the change touches hardware state). | A change applied without verified preconditions is not reproducible. The user discovers the unmet precondition as a runtime symptom, which is the most expensive failure class to debug. | Skipping straight to *"run this command"*. |
| **2. Smoke build / smoke spawn** | Apply the change at the smallest observable scale: build one sample, spawn one replica, dry-run the manifest, start one container with `--replicas=1`, allocate one queue. This proves the change *can* be applied at all. For a build-class change this is the canonical [`doca-programming-guide ## build`](../doca-programming-guide/TASKS.md#build) Track 1 (`pkg-config + meson setup + ninja`). For a container deploy this is [`doca-container-deployment ## run`](../doca-container-deployment/TASKS.md#run) with a single replica. For a bare-metal launch this is [`doca-bare-metal-deployment ## run`](../doca-bare-metal-deployment/TASKS.md#run) with `--mode=direct` first. | A failure here exposes preconditions the agent missed in step 1. Catching it at smoke scale is cheap; catching it after bulk deploy is operational pain. | Going straight to bulk / production scale. |
| **3. Smoke probe** | Issue ONE read-only check that confirms the change took effect at the smoke scale: one packet on the wire, one query to `doca_caps`, one `kubectl get pods` showing `Ready 1/1`, one `systemctl status` showing `active (running)`, one `dmesg | tail` showing no errors since the change applied, one `pkg-config --modversion` returning the expected version. The probe must be **read-only** (no further mutation) and **named** (the agent says exactly which command produces the green signal). | A change that built and started is not yet a change that *works*. The probe distinguishes *"command did not error"* from *"the system is in the state we intended"*. | Inferring success from the absence of an error message. |
| **4. Bulk / production scale** | Apply at the real scale only after step 3 returns green. For a deploy: `--replicas=N` after `--replicas=1` returned `Ready`. For a build: deploy the binary to the operational target after the smoke binary built clean. For a config change: roll to all hosts after one host stayed healthy through a probe window. | Smoke-before-bulk is the cheap insurance against config-change blast radius. The agent that goes straight to bulk skipped paid-for risk reduction. | Telling the user *"do steps 1-3 yourself first, then run the bulk command"* without making the smoke-before-bulk gate explicit. |
| **5. Observability + declare done — sustained green under controlled traffic** | Name the observability surface the agent expects to see green AND name the *anti-coincidence shape* that proves it is green. For DOCA work this is typically: the per-library trace surface (`doca-<lib>-trace` via [`doca-debug CAPABILITIES.md ## Observability`](../doca-debug/CAPABILITIES.md#observability)), the runtime counter that should have moved (e.g. `doca_flow_query_entry` / `doca_flow_query_pipe_miss` for Flow, the per-QP byte counter for RDMA, the channel statistics for Comch — whichever the lib's `CAPABILITIES.md ## Observability` actually names), the container's `Ready` field, or the `systemd` `active` state. The agent MUST: (a) record the **baseline** value *before* injecting the test signal; (b) inject **controlled traffic** of known magnitude (`ping -c 100 -i 0.01`, `iperf3 -t 30 -b 100M`, `N` SHA enqueues from `samples/.../sha_create/`, etc.) — tailing a counter on idle hardware is NOT a green signal; (c) read the counter *again* after the controlled traffic window and assert **sustained monotonic growth proportional to the injected magnitude** across at least two reads separated by a documented interval — *"counter is non-zero once"* is NOT the green signal; (d) for filter / gate / router changes, also assert the **negative-side check** (the counter that should stay flat, or the host-side `RX_packets` on the unrelated path that must keep growing — a one-sided check is incomplete); (e) write an **auditable "done" sentence** that quotes baseline + post-traffic value + controlled-traffic command + timing window — *"counter grew from 0 to 100 ± 0 across the 100 ICMP packets injected by `ping -c 100 -i 0.01 10.x.y.z` over ~1 s, while host-side `enp23s0f0np0 RX_packets` continued growing — done"* satisfies; *"counter is non-zero, done"* does not. | *"Done"* without a named green signal is unfalsifiable; a green signal that can be satisfied by a single coincidental observation (a misdirected probe, a stray ARP, a CT helper firing, a health check) is equally unfalsifiable. The user has no way to tell whether the change actually works at sustained scale or whether they are about to discover a production problem. | Declaring the task complete on a single non-zero counter read; declaring it complete on uncontrolled / ambient traffic without baseline; declaring it complete on the *match*-side only when the change is a filter / gate / router (the negative-side / host-side check is mandatory in that case). |

**Stacking rule.** When a per-artifact skill's `## test` anchor is loaded (e.g. [`doca-flow ## test`](../libs/doca-flow/TASKS.md#test)), the artifact-specific steps **layer on top of** the universal contract — they do not replace it. The agent runs both: the universal 5 steps as the spine, the artifact-specific overlay as the per-step instantiation.

**Cross-cutting layering.** When `doca-hardware-safety` is also loaded (see the activation triggers in [`AGENTS.md`](../../AGENTS.md#cross-cutting-overlay-activation-triggers)), step 1 (Preconditions) MUST include the pre-flight inventory from [`doca-hardware-safety TASKS.md ## configure`](../doca-hardware-safety/TASKS.md#configure) and step 5 (Observability + declare done) MUST include the rollback path verification (*"the rollback for this change is X; the agent verified it works before applying"*). When `doca-version` is also loaded, step 1 MUST cite the four-source detection chain explicitly and step 3 MUST include a post-change `doca_caps --version` re-read to confirm the version state did not silently shift.

### Deploy-loop bridge — step 5 not-green is the debug-loop trigger

The verification contract describes a **convergent** shape: step 3 smoke probe returns green, step 5 observability surface stays green, the task is done. Real deploys frequently land on not-green at step 5 (`Ready 0/1`, port `Down`, counter flat, log line absent, `systemd active (running)` followed by repeated restarts) and the agent's failure mode is to *declare done anyway* because the change "looks applied." The deploy-loop bridge prevents that.

The agent MUST treat **"step 5 observability did NOT reach the named green signal within the expected window"** as the symptom that **fires the [universal debug-loop contract](../doca-debug/CAPABILITIES.md#universal-debug-loop-contract)** on the change-not-converging symptom. That is: the deploy is now in the debug-loop's phase 1 (layer identification) on a deploy-class symptom — the agent walks layer identification → triple capture → single-variable mutation → re-capture → exit, exactly as it would for a `DOCA_ERROR_*` symptom. The deploy-loop bridge is the named hand-off that prevents the *"applied but unverified"* outcome.

| Verification-contract stage | Bridge behaviour |
| --- | --- |
| Green at step 3 smoke, green at step 5 within window | Task complete; no bridge fires. |
| Green at step 3 smoke, NOT-green at step 5 within window | **Fire the bridge.** Treat the not-green observability surface as the debug-loop's layer-identification input (which layer is the *not-green* speaking from — kubelet / systemd / wire / driver / firmware?). Walk the 5-phase debug-loop on the not-green symptom. Single-variable mutation in phase 3 is **always smaller than the original change** (one replica scaled, one config key flipped, one rollback step partially applied). |
| NOT-green at step 3 smoke | Fire the bridge directly from step 3 — same shape, debug-loop on the smoke symptom (rather than the post-bulk observability symptom). Bulk step 4 is **not** attempted. |
| Bridge resolves to "unchanged" or "shape-changed without convergence" after one bounded loop | Walk the rollback path documented at step 1 (per [`doca-hardware-safety`](../doca-hardware-safety/SKILL.md) if loaded, or per the per-artifact `## flow-ct` / `## modify` rollback overlay), then re-evaluate at step 1. The bridge is **bounded** — the agent does NOT loop indefinitely; on the second failure, rollback is mandatory and the agent surfaces the unresolved gap to the user. |

The bridge converts a verification-only answer into a verification + iteration answer for *every* change-recommending prompt — not only error-symptom prompts. Deploy / configure / install / upgrade prompts that previously stopped at *"and watch the metric for green"* now must say *"and watch the metric for green; if the metric is not green within X, walk the debug-loop on the not-green symptom; if the loop's second iteration does not converge, walk the rollback path."* That is the cross-cutting auto-debug-loop coverage the bridge guarantees on change-recommending prompts where no failure has been reported yet.

## Hardware binding-layer command stanza

Any answer that touches hardware state — recognizing the system shape, configuring representors, pinning workloads to NUMA / IRQ / queues, debugging a *"does nothing on the wire"* runtime symptom, applying a `doca-hardware-safety` overlay — MUST instantiate the **binding-layer command stanza** below. The stanza is the bundle-wide hardware-enumeration shape; per-artifact skills (doca-bare-metal-deployment, doca-flow, libs/doca-rdma, …) layer their library-specific binding details on top.

The agent's failure mode the stanza replaces: mentioning hardware ("you'll want to pin this to the right NUMA node") without naming the specific commands that produce the enumeration. Pointing at *"check the binding"* without running the commands is the same failure mode the debug-loop contract replaces for symptoms — hand-waving instead of evidence.

| Stanza row | Read-only command | What it tells the agent | When the row is mandatory |
| --- | --- | --- | --- |
| **PCIe presence** | `lspci -d 15b3:` (Mellanox vendor ID 15b3); for a specific device, `lspci -s <bdf> -vvv`. | Confirms the BlueField NIC / DPU is visible to the host kernel and enumerates its function structure (PF / VF / SF). Empty output = the device is not bound, the kernel module is not loaded, or the device is in the wrong eswitch mode. | Every hardware-touching answer; this is the lowest-cost confirmation of hardware reachability. |
| **Driver / device state** | `devlink dev show`; for representors, `devlink port show`; for SR-IOV, `cat /sys/class/net/<pf>/device/sriov_numvfs` and `ls /sys/class/net/<pf>/device/virtfn*/net/`. | Confirms the driver has bound the device, lists ports the kernel sees, and reports SR-IOV state. A `devlink dev show` that lists the device with `flavour: physical` but no representors means SR-IOV has not been enabled or the eswitch is in `legacy` mode. | Any deployment-shape answer (recognize / configure), any answer that references representors, any *"interface X is missing"* runtime debug. |
| **NUMA topology** | `cat /sys/class/net/<iface>/device/numa_node` (the NUMA node the NIC is attached to); `numactl -H` (the host topology); `lscpu \| grep -i numa` (CPU NUMA enumeration). | Tells the agent which NUMA node the workload must be pinned to for line-rate. A NIC on node 0 with a worker on node 1 will eat ~30–60% of throughput to cross-socket traffic. | Any bare-metal performance answer; any `doca-bare-metal-deployment ## run` invocation with isolation; any *"my throughput is low"* runtime debug. |
| **IRQ affinity** | `cat /proc/interrupts \| grep <iface>` (current IRQ-to-CPU mapping); `ls /sys/class/net/<iface>/device/msi_irqs/`; for inspection of one IRQ, `cat /proc/irq/<n>/smp_affinity_list`. | Confirms which CPU cores the NIC's MSI-X interrupts land on. Default kernel placement is rarely optimal for DOCA workloads; the agent must check before recommending or before debugging interrupt-storm symptoms. | Any answer that recommends CPU pinning, IRQ steering, or polling-mode tuning. |
| **Firmware / configuration snapshot** | `mlxconfig -d <bdf> q` (current firmware NV config — read-only, side-effect-free); for a single key, `mlxconfig -d <bdf> q \| grep -i <key>`. | Reports the firmware-level configuration (eswitch mode, SR-IOV count, port-type, IB vs ETH, etc.). The four-way coherence rule from [`doca-version CAPABILITIES.md ## Version compatibility`](../doca-version/CAPABILITIES.md#version-compatibility) explicitly includes firmware; ignoring this row means missing the case where the host install is consistent but the firmware is on a different rev. | Any answer that touches `mlxconfig`-class settings (the trigger that loads `doca-hardware-safety`); any version-coherence answer where firmware is in scope; any *"the feature is supposed to be supported but isn't"* runtime debug. |
| **Kernel module state** | `lsmod \| grep -E 'mlx5_core\|mlx5_ib\|mlx_compat'`; `modinfo mlx5_core \| head -20` for version detail. | Confirms the kernel modules backing the device are loaded and at the expected version. A missing `mlx5_core` produces every higher-layer symptom as a cascade; a mismatched module version against the OFED stack produces the "looks fine but nothing works" class. | Any *"no devices visible"* runtime symptom; any cross-kernel-upgrade debug. |

**The stanza is read-only.** Every row is a side-effect-free enumeration. Mutating commands (`mlxconfig -d <bdf> set ...`, `devlink dev eswitch set ... mode switchdev`, kernel-module reload, BFB reflash) are NOT part of the stanza — they belong in the per-artifact `## modify` anchor and require the [`doca-hardware-safety`](../doca-hardware-safety/SKILL.md) overlay. Running the stanza is always safe; acting on what it reveals is what requires the safety overlay.

**Stacking rule.** When a per-artifact skill is loaded (e.g. [`doca-bare-metal-deployment`](../doca-bare-metal-deployment/SKILL.md), [`doca-flow`](../libs/doca-flow/SKILL.md)), the per-artifact `## configure` / `## run` anchors layer library-specific binding details on top — for Flow, the steering-mode capability query (`doca_caps`) and the representor-to-`port_id` mapping; for bare-metal, the cgroup-v2 + namespace + `taskset` instantiation. The cross-cutting stanza is the spine; the per-artifact overlay names the artifact-specific calls.

**Cross-cutting layering.** When the [`## recognize`](TASKS.md#recognize) front-door routing decision fires, the auto-detect block at step 1 of that anchor is a *subset* of this stanza (PCIe presence + DOCA install presence). The full stanza is what `## recognize` hands off to the routed skill so the downstream answer has the binding-layer picture in hand. When the universal verification contract fires, step 1 (Preconditions) MUST include the relevant rows of this stanza for any hardware-touching change.
