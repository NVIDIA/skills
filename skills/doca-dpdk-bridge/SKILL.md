---
license: Apache-2.0
name: doca-dpdk-bridge
description: >
  Use this skill when the user has an existing DPDK application and is
  adding DOCA capabilities in-place — most commonly DOCA Flow hardware
  steering — without rewriting the data-plane in DOCA-native form:
  binding a DPDK port id to a `doca_dev` (`doca_dpdk_port_probe` /
  `doca_dpdk_port_as_dev`), converting `rte_mbuf` ↔ `doca_buf`,
  querying `doca_dpdk_cap_is_rep_port_supported`, or debugging
  `DOCA_ERROR_*` from a bridge call. Trigger even without "DOCA DPDK
  Bridge": "how do I add DOCA Flow to my DPDK app", "make a DPDK port
  visible to DOCA", "the bridge loads but every operation returns
  errors", "pkg-config --exists doca-dpdk-bridge fails", or
  "DOCA_ERROR_NOT_FOUND on port registration". Route elsewhere for
  fresh DOCA-native packet I/O (doca-eth), flow-rule programming
  (doca-flow), DOCA or DPDK install (doca-setup), or RDMA data
  movement (doca-rdma).
metadata:
  kind: library
compatibility: >
  Requires DOCA SDK installed at /opt/mellanox/doca on Linux
  (Ubuntu 22.04/24.04 or RHEL/SLES) with a BlueField DPU or
  ConnectX NIC attached, plus a separate DPDK install whose
  version falls within the bridge's matched-pair window. Reads
  the user's local install via `pkg-config doca-dpdk-bridge` and
  `pkg-config libdpdk`, and inspects
  /opt/mellanox/doca/{lib,include,samples,applications}.
---

# DOCA DPDK Bridge

**Where to start:** This skill assumes DOCA is already installed,
DPDK is already installed, the user has an **existing DPDK
application**, and they want to **add DOCA capabilities to it
in-place** (most commonly DOCA Flow for hardware steering)
without migrating the data-plane to DOCA-native APIs. Open
[`TASKS.md`](TASKS.md) if the user wants to *do* something
(configure / build / modify / run / test / debug); open
[`CAPABILITIES.md`](CAPABILITIES.md) when the question is *what
the bridge can express* on this version. If the user has not
installed DOCA yet, route to
[`doca-setup`](../../doca-setup/SKILL.md) first. If the user is
**starting fresh** (no DPDK code yet) and just wants line-rate
packet I/O against DOCA, route to
[`doca-eth`](../doca-eth/SKILL.md) instead — the bridge exists
for the interop case, not the start-fresh case.

## Example questions this skill answers well

The CLASSES of DOCA DPDK Bridge questions this skill is built to
answer, each with one worked example. The agent should treat the
*class* as the load-bearing piece — the worked example is a
single instance.

- **"I have a DPDK app — how do I add DOCA Flow rules to it
  without rewriting the data-plane?"** — worked example: *"my
  packet-processing app already drives mbufs through `rte_eth_*`
  ports; I want to install DOCA Flow steering rules on those
  ports for HW offload"*. Answered by the bridge-vs-native
  selection rule in
  [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes)
  + the port-handover workflow in
  [`TASKS.md ## configure`](TASKS.md#configure) step 3.
- **"How do I make a DPDK port visible to DOCA?"** — worked
  example: *"I have a DPDK port id from `rte_eth_dev_*`; how does
  DOCA see it"*. Answered by the DPDK-port-id ↔ `doca_dev`
  mapping (`doca_dpdk_port_probe` / `doca_dpdk_port_as_dev`) in
  [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes)
  bridge-objects table + the binding workflow in
  [`TASKS.md ## configure`](TASKS.md#configure) step 4.
- **"How do I move packets between DPDK mbufs and DOCA bufs?"** —
  worked example: *"DPDK delivers an `rte_mbuf` to my fastpath; I
  want a DOCA library to operate on the payload"*. Answered by
  the mbuf ↔ DOCA-buf conversion shape in
  [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes)
  + the conversion-step workflow in
  [`TASKS.md ## modify`](TASKS.md#modify).
- **"Is the bridge even installed and is its DPDK compatible
  with my DOCA?"** — worked example: *"`pkg-config --exists
  doca-dpdk-bridge` returns failure on a host that has DPDK
  separately installed"*. Answered by the version-coupling rule
  in
  [`CAPABILITIES.md ## Version compatibility`](CAPABILITIES.md#version-compatibility)
  + the cap-check workflow in
  [`TASKS.md ## configure`](TASKS.md#configure) step 1.
- **"Should I be using `doca-dpdk-bridge` or `doca-eth`?"** —
  worked example: *"new project; I have not committed to DPDK
  yet"*. Answered by the path-selection table in
  [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes)
  bridge-vs-native row + the deferred-verbs note in
  [`TASKS.md ## Deferred task verbs`](TASKS.md#deferred-task-verbs).
- **"What does this `DOCA_ERROR_*` from a bridge call mean and
  which layer caused it?"** — worked example: *"`DOCA_ERROR_NOT_FOUND`
  on a port-registration call after the DPDK port came up
  cleanly"*. Answered by the bridge overlay on the cross-library
  taxonomy in
  [`CAPABILITIES.md ## Error taxonomy`](CAPABILITIES.md#error-taxonomy)
  + the layered ladder in
  [`TASKS.md ## debug`](TASKS.md#debug) that escalates to
  [`doca-debug`](../../doca-debug/SKILL.md).

## Audience

This skill serves **external developers who already maintain a
DPDK-based packet-processing application** — i.e., users whose
data-plane already drives `rte_eth_*` ports and `rte_mbuf`
buffers — and who want to **add DOCA capabilities** (most
commonly hardware steering via DOCA Flow) by linking the bridge
into the same process. It is *not* for users starting a fresh
DOCA-native project (route to [`doca-eth`](../doca-eth/SKILL.md))
and *not* for users running pure DPDK with no interest in DOCA
(no skill in this bundle applies). It is also not for NVIDIA
developers contributing to DOCA DPDK Bridge itself.

**Language scope.** DOCA DPDK Bridge ships as a C library with
`pkg-config` module name `doca-dpdk-bridge` (the agent must
confirm the spelling against the user's install via
`pkg-config --exists doca-dpdk-bridge`; some DOCA releases use a
slightly different module name and the agent must not guess).
The shipped samples are written in C. C and C++ consumers are
the canonical case; the worked examples in `TASKS.md` assume
that path. Other-language consumers (Rust, Go, Python, …)
typically do not use this bridge — DPDK itself is C-shaped, and
a non-C DPDK app is rare; if the user is in that minority, the
skill's contribution is to keep the port-handover, capability,
permission, and error-taxonomy guidance language-neutral and
route them to the public C ABI as the authoritative surface.

## When to load this skill

Load this skill when the user is doing hands-on DOCA DPDK Bridge
work, in any language. Concretely:

- The user has an EXISTING DPDK application (their data-plane
  already drives `rte_eth_dev_*` ports and `rte_mbuf` buffers),
  and they want to layer DOCA on top — adding DOCA Flow rules,
  feeding packets into a DOCA accelerator (Compress, AES-GCM, …),
  or wiring a DOCA service into the same process.
- The user needs to bind a DPDK port id to a `doca_dev` (via
  `doca_dpdk_port_probe` / `doca_dpdk_port_as_dev`) so DOCA Core
  / DOCA Flow can operate on the same physical port.
- The user needs to convert between DPDK mbufs and DOCA-bufs at
  the data-plane boundary, and is asking about the conversion
  helpers and their cost.
- The user is debugging a `DOCA_ERROR_*` returned from a bridge
  call (lifecycle vs. permission vs. capability vs. DPDK-port-not-
  registered vs. mbuf-conversion-failed) and the layered DPDK ↔
  DOCA stack underneath.
- The user is hitting cross-version pain: their DPDK install and
  their DOCA install are not the matched pair the bridge expects.
- The user is asking *"native `doca-eth` or `doca-dpdk-bridge`?"*
  and the agent needs the path-selection rule.

Do **not** load this skill for general DOCA orientation, install
of DOCA itself, fresh DOCA-native packet I/O (use
[`doca-eth`](../doca-eth/SKILL.md)), flow-rule programming on
its own (use [`doca-flow`](../doca-flow/SKILL.md)), host ↔ DPU
control messaging (use [`doca-comch`](../doca-comch/SKILL.md)),
or RDMA data movement (use [`doca-rdma`](../doca-rdma/SKILL.md)).
For DOCA documentation orientation, use
[`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md).

## What this skill provides

This is a **thin loader**. The body keeps only the orientation
needed to pick the right next file. The substantive bridge-
specific material lives in two companion files:

- `CAPABILITIES.md` — what the DOCA DPDK Bridge can express on
  this version: the bridge-vs-native selection rule (when to
  use `doca-dpdk-bridge` vs native `doca-eth`), the
  bridge-object surface (DPDK-port-id ↔ `doca_dev` mapping, the
  `doca_dpdk_mempool` mbuf ↔ DOCA-buf conversion), the
  capability-query surface (the single
  `doca_dpdk_cap_is_rep_port_supported`), the bridge error taxonomy
  (mapped onto the cross-library `DOCA_ERROR_*` set), the
  observability surface (DPDK-side counters + DOCA-side PE
  events), and the safety policy that gates the matched-pair
  DPDK ↔ DOCA version coupling, the EAL-must-be-up
  precondition, and DPDK-side privileges.
- `TASKS.md` — step-by-step workflows for the six in-scope
  bridge verbs: `configure`, `build`, `modify`, `run`, `test`,
  `debug`. Plus a `Deferred task verbs` block that points
  out-of-scope questions at the right next skill, and a
  `Command appendix` of the recurring commands the agent
  reaches for.

The skill assumes a host or BlueField where DOCA AND DPDK are
already installed at the standard locations and the user has the
privileges their public install profile expects (typically sudo
or `mlnx`-group membership for DOCA, plus the DPDK-side
privileges to mount hugepages, bind PCIe ports, and open the
target `rte_eth` device). It does not cover installing DOCA —
that path goes through [`doca-setup`](../../doca-setup/SKILL.md);
it does not cover installing or learning DPDK itself — that
belongs in upstream DPDK documentation reachable via
[`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md).

## What this skill deliberately does not ship

This skill is **agent guidance**, not a samples or templates
bundle. To keep the boundary clean, it deliberately does not
contain — and pull requests should not add:

- **Pre-written DOCA DPDK Bridge application source code, in any
  language.** The verified bridge source code is the bridge
  usage inside the shipped C DOCA Flow samples at
  `/opt/mellanox/doca/samples/doca_flow/` (e.g. `flow_common.c`,
  which calls `doca_dpdk_port_probe()` / `doca_dpdk_port_as_dev()`),
  plus the canonical reference applications that pair DPDK +
  DOCA Flow (see
  [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md)
  for the reference-applications index). The agent's job is to
  route the user to those files and prescribe a minimum-diff
  modification on them via the universal modify-a-sample
  workflow in
  [`doca-programming-guide`](../../doca-programming-guide/SKILL.md),
  layered with the bridge-specific overrides in
  [`TASKS.md ## modify`](TASKS.md#modify).
- **Standalone build manifests** (`meson.build`,
  `CMakeLists.txt`, `Cargo.toml`, …) parked inside the skill.
  The agent constructs the build manifest *in the user's project
  directory* against the user's installed DOCA + DPDK, where
  `pkg-config --modversion doca-dpdk-bridge` and
  `pkg-config --modversion libdpdk` are the source of truth.
- **A `samples/`, `bindings/`, or `reference/` subtree** of any
  kind. A mock or incomplete artifact in this skill's tree, even
  one labeled "reference", is misleading: users will read it as
  buildable.

## Loading order

1. Read this `SKILL.md` first to confirm the user's question is
   in scope (existing DPDK app + wants DOCA on top — not fresh).
2. **For the bridge capability matrix, the bridge-vs-native
   selection rule, port-handover surface, capability-query rules,
   error taxonomy, observability, and safety policy, see
   [CAPABILITIES.md](CAPABILITIES.md).**
3. **For step-by-step workflows — configure, build, modify, run,
   test, debug — see [TASKS.md](TASKS.md).**

Both companion files cross-link to each other,
[`doca-version`](../../doca-version/SKILL.md) for the canonical
DOCA version-handling rules,
[`doca-eth`](../doca-eth/SKILL.md) for the start-fresh path that
is the bridge's deliberate alternative,
[`doca-flow`](../doca-flow/SKILL.md) for the steering library
that is the most common reason to use this bridge in the first
place, and
[`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md)
whenever the right answer is "look it up in the public docs or
the installed package layout" rather than "bridge-specific
guidance".

## Related skills

- [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md) —
  the routing table for every public DOCA documentation source
  and the on-disk layout of an installed DOCA package. The
  bridge URL slug is `DOCA-DPDK-Bridge`. Reference applications
  that pair DPDK + DOCA Flow (the canonical adopters of this
  bridge) are reachable from the same map's reference-
  applications index.
- [`doca-setup`](../../doca-setup/SKILL.md) — env preparation,
  install verification, hugepage mounts (DPDK requirement),
  port-state checks (`devlink dev show`, `ip link`), permission
  and group-membership requirements for opening a `doca_dev`
  AND for binding PCIe ports under DPDK. This skill assumes its
  preconditions are satisfied.
- [`doca-version`](../../doca-version/SKILL.md) — canonical
  DOCA version-handling rules. This skill's `## Version
  compatibility` cross-links the four-way match rule and adds
  only the bridge-specific overlay (DPDK ↔ DOCA matched-pair
  coupling).
- [`doca-structured-tools-contract`](../../doca-structured-tools-contract/SKILL.md) —
  the bundle's structured-tools precedence rule (detect / prefer
  / fall back / report). The Command appendix in
  [TASKS.md](TASKS.md) honors this contract.
- [`doca-programming-guide`](../../doca-programming-guide/SKILL.md) —
  general DOCA programming patterns shared by every library:
  the canonical `pkg-config` + meson build pattern, the
  universal modify-a-shipped-sample first-app workflow, the
  universal lifecycle, the cross-library `DOCA_ERROR_*`
  taxonomy, and the program-side debug order. This skill layers
  bridge specifics on top.
- [`doca-eth`](../doca-eth/SKILL.md) — the **alternative path**
  for users starting fresh: pure DOCA-native queue I/O via
  `doca_eth_rxq` / `doca_eth_txq` with no DPDK in the picture.
  The bridge exists *because* migrating an existing DPDK app to
  `doca-eth` is often the wrong tradeoff; for a new project
  with no DPDK lock-in, `doca-eth` is the right choice.
- [`doca-flow`](../doca-flow/SKILL.md) — the steering library
  that is the most common reason a DPDK app reaches for this
  bridge. The bridge binds a DPDK port id to a `doca_dev`;
  DOCA Flow programs steering rules onto that `doca_dev`. The two
  libraries are designed to compose.
- [`doca-debug`](../../doca-debug/SKILL.md) — the cross-cutting
  debug ladder (install / version / build / link / runtime /
  program / driver). Bridge-specific debug (DPDK ↔ DOCA version
  drift, port-not-registered symptoms, mbuf-conversion failures)
  overlays on top of that ladder.
