---
license: Apache-2.0
name: doca-eth
description: >
  Use this skill for hands-on DOCA Ethernet packet-queue work
  on a BlueField DPU or ConnectX NIC — bringing up a
  `doca_eth_rxq` or `doca_eth_txq` on a port / representor /
  SF, picking among the four `enum doca_eth_rxq_type` values
  (`_REGULAR` / `_CYCLIC` / `_MANAGED_MEMPOOL` /
  `_SHARED_MEMPOOL`), sizing burst or scatter-gather length
  against the `_cap_*` queries, submitting
  `doca_eth_txq_task_send` / `_lso_send` (carrying packet
  `doca_buf`s — no `doca_eth_frame` struct exists), or
  debugging DOCA_ERROR_* from an Ethernet call. Trigger on
  implicit phrasings: "my RX queue is up but no packets
  arrive", "send-task returns AGAIN at line rate", "which
  queue type for fixed-MTU ingress", "device open fails
  without sudo", or "is L3 checksum offload available here".
  Refuse and route elsewhere for installing DOCA,
  flow-rule / steering programming, host↔DPU control
  messaging, or RDMA data movement.
metadata:
  kind: library
compatibility: >
  Requires DOCA SDK installed at /opt/mellanox/doca on Linux
  (Ubuntu 22.04/24.04 or RHEL/SLES) with a BlueField DPU or
  ConnectX NIC attached. Reads the user's local install via
  `pkg-config doca-eth` and inspects
  /opt/mellanox/doca/{lib,include,samples,applications}.
---

# DOCA Ethernet

**Where to start:** This skill assumes DOCA is already installed and
the user is doing **hands-on packet-queue work** on a host or
BlueField with DOCA. Open [`TASKS.md`](TASKS.md) if the user wants
to *do* something (configure / build / modify / run / test /
debug); open [`CAPABILITIES.md`](CAPABILITIES.md) when the question
is *what can a DOCA Ethernet queue express* on this version. If
the user has not installed DOCA yet, route to
[`doca-setup`](../../doca-setup/SKILL.md) first. If the user is
asking *"how do I get packets to land on my RX queue at all"*, the
answer lives in [`doca-flow`](../doca-flow/SKILL.md) — DOCA
Ethernet is the *queue* surface; DOCA Flow is the *steering*
surface, and they are independent libraries.

## Example questions this skill answers well

The CLASSES of DOCA Ethernet questions this skill is built to
answer, each with one worked example. The agent should treat the
*class* as the load-bearing piece — the worked example is a single
instance.

- **"How do I bring up an RX and TX queue on a representor or
  physical port?"** — worked example: *"set up a `doca_eth_rxq`
  plus a `doca_eth_txq` on a single BlueField representor for
  first-run testing"*. Answered by the queue-pair lifecycle in
  [`TASKS.md ## configure`](TASKS.md#configure) +
  [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes)
  RX / TX object table.
- **"Which RX type fits my data shape — regular, cyclic, or
  managed-recv?"** — worked example: *"line-rate ingress with
  fixed-size frames into a pre-allocated buffer ring"*. Answered
  by the RX-type taxonomy in
  [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes)
  + the capability-query rule (`doca_eth_rxq_cap_is_type_supported`
  against a `doca_devinfo`) in
  [`TASKS.md ## configure`](TASKS.md#configure).
- **"How do I send a packet from user code through `doca_eth_txq`?"**
  — worked example: *"allocate a packet `doca_buf`, attach the
  payload, submit one send-task, wait for the completion event"*.
  Answered by the TX submission shape in
  [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes)
  + the property-set workflow in
  [`TASKS.md ## modify`](TASKS.md#modify).
- **"My queue is up but no packets arrive — why?"** — worked
  example: *"`doca_eth_rxq` started cleanly but the recv callback
  never fires"*. Answered by the steering-dependency rule in
  [`CAPABILITIES.md ## Safety policy`](CAPABILITIES.md#safety-policy)
  + the env-prep checklist in
  [`TASKS.md ## configure`](TASKS.md#configure) step 1, which
  routes the steering side to
  [`doca-flow`](../doca-flow/SKILL.md) and the
  promiscuous-mode side to
  [`doca-setup`](../../doca-setup/SKILL.md).
- **"Is this Ethernet capability available on my device + my
  installed DOCA?"** — worked example: *"does this device
  advertise L3 checksum offload"*. Answered by the
  capability-query rule (`doca_eth_txq_cap_is_l3_chksum_offload_supported`
  against a `doca_devinfo`) in
  [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes)
  + the version-and-device overlay in
  [`CAPABILITIES.md ## Version compatibility`](CAPABILITIES.md#version-compatibility).
- **"What does this `DOCA_ERROR_*` from an Ethernet call mean and
  which layer caused it?"** — worked example: *"`DOCA_ERROR_AGAIN`
  on `doca_task_submit` for an `eth_txq` send-task at high rate"*.
  Answered by the Ethernet overlay on the cross-library taxonomy
  in
  [`CAPABILITIES.md ## Error taxonomy`](CAPABILITIES.md#error-taxonomy)
  + the layered ladder in
  [`TASKS.md ## debug`](TASKS.md#debug) that escalates to
  [`doca-debug`](../../doca-debug/SKILL.md).

## Audience

This skill serves **external developers building applications that
consume the DOCA Ethernet library** — i.e., users whose code calls
`doca_eth_rxq_*` / `doca_eth_txq_*` (directly in C/C++, or through
FFI/bindings from another language) to do line-rate packet I/O on
physical ports, representors, or SFs of a BlueField or ConnectX
device. It is *not* for NVIDIA developers contributing to DOCA
Ethernet itself.

**Language scope.** DOCA Ethernet ships as a C library with
`pkg-config` module name `doca-eth`. The shipped samples are
written in C. C and C++ consumers are the canonical case; the
worked examples in `TASKS.md` assume that path. Other-language
consumers (Rust, Go, Python, …) consume the same `*.so` through
FFI or language-specific bindings; the skill's contribution in
that case is to keep the queue-object split, lifecycle,
capability-discovery, permission, RX-type taxonomy, and
error-taxonomy guidance language-neutral, and to route the agent
to the public C ABI as the authoritative surface that any wrapper
will eventually call.

## When to load this skill

Load this skill when the user is doing hands-on DOCA Ethernet
work, in any language. Concretely:

- Initializing a `doca_eth_rxq` on a `doca_dev` opened against a
  physical port, a representor, or an SF — choosing among the
  regular / cyclic / managed-recv RX types based on data shape
  before `doca_ctx_start()`.
- Initializing a `doca_eth_txq` on the same or a different
  `doca_dev` and posting send-tasks against packet `doca_buf`s
  payload buffers.
- Reading or setting Ethernet queue properties via
  `doca_eth_rxq_set_*` / `doca_eth_txq_set_*` and querying device
  capability via `doca_eth_rxq_cap_*` / `doca_eth_txq_cap_*`
  (max burst size, RX-type support, max scatter-gather length,
  checksum-offload presence).
- Confirming the port is up and that traffic is actually steered
  to the chosen RX queue — either via DOCA Flow rules (the
  canonical path) or via kernel-side promiscuous mode (the
  expedient first-run path).
- Wiring an Ethernet queue to a higher-level data plane: GPU
  packet processing via DOCA GPUNetIO, custom user-space
  forwarding agents, or telemetry mirrors that snap a copy of
  every packet.
- Debugging a `DOCA_ERROR_*` returned from an Ethernet call
  (lifecycle vs. permission vs. capability vs. send-queue-full
  vs. driver-below) and the per-queue progress engine events.
- Designing or extending non-C bindings (Rust, Go, Python, …)
  that wrap the DOCA Ethernet C ABI — for the lifecycle, queue
  split, RX-type, capability, and permission rules the wrapper
  must honor.

Do **not** load this skill for general DOCA orientation, install
of DOCA itself, flow-rule programming (use
[`doca-flow`](../doca-flow/SKILL.md)), host ↔ DPU control
messaging (use [`doca-comch`](../doca-comch/SKILL.md)), or
RDMA data movement (use [`doca-rdma`](../doca-rdma/SKILL.md)).
For DOCA documentation orientation, use
[`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md).

## What this skill provides

This is a **thin loader**. The body keeps only the orientation
needed to pick the right next file. The substantive Ethernet-
specific material lives in two companion files:

- `CAPABILITIES.md` — what a DOCA Ethernet queue can express on
  this version: the RX / TX object split, the RX-type taxonomy
  (regular / cyclic / managed-recv), the send-task / frame
  submission surface, the capability-query surface
  (`doca_eth_rxq_cap_*` / `doca_eth_txq_cap_*`), the Ethernet
  error taxonomy (mapped onto the cross-library `DOCA_ERROR_*`
  set), the observability surface (per-queue progress engine
  events, capability snapshots), and the safety policy that
  gates the steering / permission / port-state preconditions.
- `TASKS.md` — step-by-step workflows for the six in-scope
  Ethernet verbs: `configure`, `build`, `modify`, `run`,
  `test`, `debug`. Plus a `Deferred task verbs` block that
  points out-of-scope questions at the right next skill, and a
  `Command appendix` of the recurring commands the agent
  reaches for.

The skill assumes a host or BlueField where DOCA is already
installed at the standard location and the user has the
privileges their public install profile expects (typically sudo
or `mlnx`-group membership to open a `doca_dev` against a port).
It does not cover installing DOCA — that path goes through
[`doca-setup`](../../doca-setup/SKILL.md).

## What this skill deliberately does not ship

This skill is **agent guidance**, not a samples or templates
bundle. To keep the boundary clean, it deliberately does not
contain — and pull requests should not add:

- **Pre-written DOCA Ethernet application source code, in any
  language.** The verified Ethernet source code is the shipped C
  samples at `/opt/mellanox/doca/samples/doca_eth/<name>/`. The
  agent's job is to route the user to those files and prescribe
  a minimum-diff modification on them via the universal
  modify-a-sample workflow in
  [`doca-programming-guide`](../../doca-programming-guide/SKILL.md),
  layered with the Ethernet-specific overrides in
  [`TASKS.md ## modify`](TASKS.md#modify).
- **Standalone build manifests** (`meson.build`,
  `CMakeLists.txt`, `Cargo.toml`, …) parked inside the skill.
  The agent constructs the build manifest *in the user's project
  directory* against the user's installed DOCA, where
  `pkg-config --modversion doca-eth` is the source of truth.
- **A `samples/`, `bindings/`, or `reference/` subtree** of any
  kind. A mock or incomplete artifact in this skill's tree, even
  one labeled "reference", is misleading: users will read it as
  buildable.

## Loading order

1. Read this `SKILL.md` first to confirm the user's question is
   in scope.
2. **For the Ethernet capability matrix, RX-type taxonomy, send
   surface, capability-query rules, error taxonomy,
   observability, and safety policy, see
   [CAPABILITIES.md](CAPABILITIES.md).**
3. **For step-by-step workflows — configure, build, modify, run,
   test, debug — see [TASKS.md](TASKS.md).**

Both companion files cross-link to each other,
[`doca-version`](../../doca-version/SKILL.md) for the canonical
version-handling rules,
[`doca-flow`](../doca-flow/SKILL.md) for the steering side
that the RX queue depends on, and
[`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md)
whenever the right answer is "look it up in the public docs or
the installed package layout" rather than "Ethernet-specific
guidance".

## Related skills

- [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md) —
  the routing table for every public DOCA documentation source
  and the on-disk layout of an installed DOCA package. The
  Ethernet URL slug is `DOCA-Ethernet`.
- [`doca-setup`](../../doca-setup/SKILL.md) — env preparation,
  install verification, port-state checks (`devlink dev show`,
  `ip link`), permission and group-membership requirements for
  opening a `doca_dev`. This skill assumes its preconditions
  are satisfied.
- [`doca-version`](../../doca-version/SKILL.md) — canonical
  DOCA version-handling rules. This skill's `## Version
  compatibility` cross-links the four-way match rule and adds
  only the Ethernet-specific overlay (RX-type availability
  windows, checksum-offload device-conditional support).
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
  Ethernet specifics on top.
- [`doca-flow`](../doca-flow/SKILL.md) — the steering surface
  that decides which packets land on which `doca_eth_rxq`. DOCA
  Ethernet does *not* program steering itself; an empty RX queue
  almost always means a missing or wrong Flow rule, not an
  Ethernet bug.
- [`doca-debug`](../../doca-debug/SKILL.md) — the cross-cutting
  debug ladder (install / version / build / link / runtime /
  program / driver). Ethernet-specific debug (RX-type
  mismatches, send-queue-full retries, steering-empty-queue
  symptoms) overlays on top of that ladder.
