---
license: Apache-2.0
name: doca-sta
description: >
  Use this skill when the user is doing hands-on NVMe-over-Fabrics
  storage-target work on a BlueField DPU or ConnectX NIC with DOCA STA —
  standing up a doca_sta DOCA Core context that accelerates the
  target-side NVMe-oF data path over RDMA, defining
  doca_sta_subsystem targets (NQN + namespaces) backed by local
  NVMe-PCI backend disks (doca_sta_be), checking device support via
  doca_sta_cap_is_supported, sizing the per-connection I/O queues,
  or debugging DOCA_ERROR_* from a STA call. Trigger even
  when the user does not say "DOCA STA" — typical implicit phrasings
  include "my NVMe-oF Connect never completes", "Identify Controller
  times out over RoCE", "16 I/O queues at depth 1024 — does this
  BlueField support that", "offload the nvmf target onto the DPU", or
  "DOCA_ERROR_IO_FAILED on an NVMe read". Refuse and route elsewhere
  for DOCA install, raw RDMA data movement, raw packet I/O,
  flow-rule programming, or initiator-side / host NVMe stack work
  — those belong to other skills.
metadata:
  kind: library
compatibility: >
  Requires DOCA SDK installed at /opt/mellanox/doca on Linux (Ubuntu
  22.04/24.04 or RHEL/SLES) with a BlueField DPU or ConnectX NIC
  attached. Reads the user's local install via `pkg-config doca-sta`
  (and `pkg-config doca-rdma` for the NVMe-over-RDMA transport) and
  inspects /opt/mellanox/doca/{lib,include,samples,applications}.
---

# DOCA STA (Storage Target Acceleration)

**Where to start:** This skill assumes DOCA is already installed and
the user is doing **hands-on NVMe-over-Fabrics storage-target work** on
a BlueField-class device with DOCA. Open [`TASKS.md`](TASKS.md) if
the user wants to *do* something (configure / build / modify / run
/ test / debug); open [`CAPABILITIES.md`](CAPABILITIES.md) when the
question is *what can DOCA STA express* on this version. If the
user has not installed DOCA yet, route to
[`doca-setup`](../../doca-setup/SKILL.md) first. If the user is
asking *"is this an NVMe-oF initiator/host transport?"*, the
answer is no — doca-sta accelerates the **target** side: it presents
NVMe-oF `doca_sta_subsystem` targets backed by local NVMe-PCI
disks; the model lives in
[`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes).

## Example questions this skill answers well

The CLASSES of DOCA STA questions this skill is built to answer,
each with one worked example. The agent should treat the *class*
as the load-bearing piece — the worked example is a single
instance.

- **"How do I bring up an NVMe-oF target that uses the BlueField
  to accelerate the storage data path?"** — worked example: *"define
  a `doca_sta_subsystem` (NQN) with one namespace backed by a local
  NVMe-PCI disk (`doca_sta_be`) and accept NVMe-over-RDMA
  connections from a remote initiator"*. Answered by the
  target-model-and-lifecycle
  workflow in [`TASKS.md ## configure`](TASKS.md#configure) +
  [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes)
  target-object table.
- **"Can this BlueField accelerate an NVMe-oF target at all?"** —
  worked example: *"my data center is RoCE end-to-end; does this
  device support DOCA STA target acceleration?"*. STA transport is
  RDMA-only (there is no NVMe-over-TCP path). Answered by the
  capability-query rule (`doca_sta_cap_is_supported` against a
  `doca_devinfo`) in
  [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes)
  + the discovery step in
  [`TASKS.md ## configure`](TASKS.md#configure).
- **"How deep can I size my I/O queues, and how many I/O queues
  per connection?"** — worked example: *"I want 16 I/O queues at
  depth 1024 each — does this device support that?"*. Answered by
  the queue-sizing capability surface in
  [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes)
  + the queue-sizing step in
  [`TASKS.md ## configure`](TASKS.md#configure) which gates on
  the matching `doca_sta_get_max_*` query (e.g.
  `doca_sta_get_max_qps`, `doca_sta_get_max_io_queue_size`).
- **"Which other DOCA libraries do I need alongside doca-sta?"** —
  worked example: *"do I need doca-rdma directly, or does doca-sta
  hide it from me?"*. Answered by the substrate-library rule in
  [`CAPABILITIES.md ## Safety policy`](CAPABILITIES.md#safety-policy)
  + the env-prep checklist in
  [`TASKS.md ## configure`](TASKS.md#configure) step 1, which
  routes the steering side to
  [`doca-flow`](../doca-flow/SKILL.md) and the RDMA substrate
  to [`doca-rdma`](../doca-rdma/SKILL.md).
- **"Is this STA capability available on my installed DOCA?"** —
  worked example: *"is STA target acceleration supported on this
  BlueField + DOCA version?"*. Answered by the version-and-device
  overlay in
  [`CAPABILITIES.md ## Version compatibility`](CAPABILITIES.md#version-compatibility),
  which cross-links the canonical detection chain in
  [`doca-version`](../../doca-version/SKILL.md) and adds the
  STA-specific cap-query rule (`pkg-config --modversion doca-sta`
  is the build-time anchor; the runtime `doca_sta_cap_is_supported`
  query is the truth).
- **"What does this `DOCA_ERROR_*` from a STA call mean and which
  layer caused it?"** — worked example: *"`DOCA_ERROR_IO_FAILED`
  on a submitted NVMe read I/O against a target I can ping"*.
  Answered by the STA overlay on the cross-library taxonomy in
  [`CAPABILITIES.md ## Error taxonomy`](CAPABILITIES.md#error-taxonomy)
  + the layered ladder in
  [`TASKS.md ## debug`](TASKS.md#debug) that escalates to
  [`doca-debug`](../../doca-debug/SKILL.md).

## Audience

This skill serves **external developers building NVMe-over-Fabrics
storage targets that consume DOCA STA on BlueField** — i.e., users
whose code calls `doca_sta_*` (directly in C/C++, or through
FFI/bindings from another language) to accelerate the target-side
data path of an NVMe-oF target on the BlueField hardware: presenting
`doca_sta_subsystem` targets (NQN + namespaces) backed by local
NVMe-PCI disks (`doca_sta_be`) to remote initiators over RDMA. The
skill is *not* for NVIDIA developers contributing to DOCA STA
itself, and it is *not* for initiator/host-side NVMe stacks.

**Language scope.** DOCA STA ships as a C library with
`pkg-config` module name `doca-sta`. DOCA STA ships **no public
samples** — it is absent from the DOCA libraries /
extension_libraries sample profiles — so the worked examples in
`TASKS.md` build against the public headers directly rather than
modify a shipped sample. C and C++ consumers are the canonical
case. Other-language
consumers (Rust, Go, Python, …) consume the same `*.so` through
FFI or language-specific bindings; the skill's contribution in
that case is to keep the target-model, lifecycle,
capability-discovery, queue-pair shape, substrate-dependency,
and error-taxonomy guidance language-neutral, and to route the
agent to the public C ABI as the authoritative surface that any
wrapper will eventually call.

## When to load this skill

Load this skill when the user is doing hands-on DOCA STA work,
in any language. Concretely:

- Initializing a `doca_sta` instance on a `doca_dev` opened
  against a BlueField PF / SF and configuring the NVMe-oF
  target subsystems before `doca_ctx_start()`.
- Defining target resources — `doca_sta_subsystem` (NQN +
  namespaces) and `doca_sta_be` backend controllers (local
  NVMe-PCI disks) — and accepting NVMe-oF connections (admin
  queue plus N I/O queues per connection) on the target side as
  remote initiators connect over RDMA CM.
- Reading or setting STA properties via the `doca_sta_set_*`
  family, checking device support via `doca_sta_cap_is_supported`,
  and querying sizing limits via the `doca_sta_get_max_*` family
  (max I/O queue depth, max number of queue pairs, max I/O size,
  max subsystems, max namespaces per subsystem, max backends).
- Wiring the **NVMe-over-RDMA** transport — STA's only transport;
  it lands on the `doca-rdma` substrate and uses RDMA CM for
  connection establishment — for the target's I/O queues.
- Wiring DOCA Flow rules so that NVMe-oF traffic actually
  reaches the STA-managed queues — the steering boundary is
  `doca-flow`, not `doca-sta`.
- Debugging a `DOCA_ERROR_*` returned from a STA call (lifecycle
  vs. capability vs. transport-layer I/O failure vs.
  driver-below) and the per-queue events on the DOCA Core
  progress engine.
- Designing or extending non-C bindings (Rust, Go, Python, …)
  that wrap the DOCA STA C ABI — for the lifecycle, queue-pair,
  cap-query, and substrate-dependency rules the wrapper must
  honor.

Do **not** load this skill for general DOCA orientation, install
of DOCA itself, raw RDMA data movement (use
[`doca-rdma`](../doca-rdma/SKILL.md)), raw packet I/O on
Ethernet queues (use [`doca-eth`](../doca-eth/SKILL.md)),
flow-rule programming (use [`doca-flow`](../doca-flow/SKILL.md)),
or initiator/host-side NVMe stack development
(SPDK or kernel-nvme own that, not this skill). For DOCA
documentation orientation, use
[`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md).

## What this skill provides

This is a **thin loader**. The body keeps only the orientation
needed to pick the right next file. The substantive STA-specific
material lives in two companion files:

- `CAPABILITIES.md` — what DOCA STA can express on this
  version: the target object model (`doca_sta_subsystem` /
  namespaces / `doca_sta_be` backend NVMe-PCI disks),
  the NVMe queue-pair shape (admin queue + I/O queues over RDMA),
  the RDMA-only transport,
  the capability-query surface (`doca_sta_cap_is_supported` plus
  the `doca_sta_get_max_*` sizing queries), the STA
  error taxonomy (mapped onto the cross-library `DOCA_ERROR_*`
  set), the observability surface (per-queue progress engine
  events, capability snapshots), and the safety policy that
  gates substrate-library, permission, and steering
  preconditions.
- `TASKS.md` — step-by-step workflows for the six in-scope
  STA verbs: `configure`, `build`, `modify`, `run`, `test`,
  `debug`. Plus a `Deferred task verbs` block that points
  out-of-scope questions at the right next skill, and a
  `Command appendix` of the recurring commands the agent
  reaches for.

The skill assumes a BlueField (with DOCA installed at the
standard location) plus a remote NVMe-oF initiator reachable on
the fabric to connect into the accelerated target, and one or
more local NVMe-PCI disks to back the target's namespaces. It
does not cover installing
DOCA — that path goes through
[`doca-setup`](../../doca-setup/SKILL.md). It does not cover
initiator/host-side NVMe stacks (SPDK `bdev_nvme`, kernel `nvme`
host) or NVMe protocol semantics above the accelerated target
data path — those are out of scope.

## What this skill deliberately does not ship

This skill is **agent guidance**, not a samples or templates
bundle. To keep the boundary clean, it deliberately does not
contain — and pull requests should not add:

- **Pre-written DOCA STA application source code, in any
  language.** DOCA STA ships **no public samples** — there is no
  `/opt/mellanox/doca/samples/doca_sta/` directory, and STA is
  absent from the libraries / extension_libraries sample
  profiles. The authoritative surface is the public headers under
  $(pkg-config --variable=includedir doca-common) plus the public
  DOCA STA guide; the agent builds against those directly rather
  than modifying a shipped sample, per the
  [`TASKS.md ## modify`](TASKS.md#modify) workflow.
- **Initiator/host-side NVMe stack glue.** SPDK `bdev_nvme`,
  the kernel `nvme` host, and any initiator-side NVMe stack are
  upstream projects out of scope for this skill — DOCA STA is
  target-side acceleration, not an initiator transport provider.
- **Standalone build manifests** (`meson.build`, `CMakeLists.txt`,
  `Cargo.toml`, …) parked inside the skill. The agent
  constructs the build manifest *in the user's project
  directory* against the user's installed DOCA, where
  `pkg-config --modversion doca-sta` is the source of truth.
- **A `samples/`, `bindings/`, or `reference/` subtree** of any
  kind. A mock or incomplete artifact in this skill's tree,
  even one labeled "reference", is misleading: users will
  read it as buildable.

## Loading order

1. Read this `SKILL.md` first to confirm the user's question
   is in scope.
2. **For the STA capability matrix, the target object model,
   queue-pair shape, RDMA-only transport, capability-query
   rules, error taxonomy, observability, and safety policy,
   see [CAPABILITIES.md](CAPABILITIES.md).**
3. **For step-by-step workflows — configure, build, modify,
   run, test, debug — see [TASKS.md](TASKS.md).**

Both companion files cross-link to each other,
[`doca-version`](../../doca-version/SKILL.md) for the canonical
version-handling rules,
[`doca-rdma`](../doca-rdma/SKILL.md) for the RDMA substrate
that NVMe-over-RDMA transport lands on,
[`doca-flow`](../doca-flow/SKILL.md) for the steering rules
that direct NVMe traffic to STA-managed queues, and
[`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md)
whenever the right answer is "look it up in the public docs or
the installed package layout" rather than "STA-specific
guidance".

## Related skills

- [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md) —
  the routing table for every public DOCA documentation source
  and the on-disk layout of an installed DOCA package. The
  STA URL slug is `DOCA-STA`.
- [`doca-setup`](../../doca-setup/SKILL.md) — env preparation,
  install verification, BlueField mode checks, and the
  permission / group-membership requirements for opening a
  `doca_dev`. This skill assumes its preconditions are
  satisfied.
- [`doca-version`](../../doca-version/SKILL.md) — canonical
  DOCA version-handling rules. This skill's `## Version
  compatibility` cross-links the four-way match rule and adds
  only the STA-specific overlay (STA target-acceleration
  availability windows, NVMe-oF feature-set device-conditional
  support).
- [`doca-structured-tools-contract`](../../doca-structured-tools-contract/SKILL.md) —
  the bundle's structured-tools precedence rule (detect /
  prefer / fall back / report). The Command appendix in
  [TASKS.md](TASKS.md) honors this contract.
- [`doca-programming-guide`](../../doca-programming-guide/SKILL.md) —
  general DOCA programming patterns shared by every library:
  the canonical `pkg-config` + meson build pattern, the
  universal modify-a-shipped-sample first-app workflow, the
  universal lifecycle, the cross-library `DOCA_ERROR_*`
  taxonomy, and the program-side debug order. This skill
  layers STA specifics on top.
- [`doca-rdma`](../doca-rdma/SKILL.md) — the RDMA substrate
  that NVMe-over-RDMA transport lands on. STA hides most of
  the RDMA queue-pair details from the consumer, but the user
  still needs `doca-rdma` linked in and the device's RDMA
  capabilities discoverable for the NVMe-over-RDMA path to
  work.
- [`doca-eth`](../doca-eth/SKILL.md) — the queue-pair
  shape that STA's per-connection queue model echoes. Reach
  here if the user is asking general questions about how
  DOCA exposes queue-pairs that don't have an STA-specific
  answer.
- [`doca-flow`](../doca-flow/SKILL.md) — the steering
  surface that decides which NVMe-oF packets land on which
  STA-managed queue. DOCA STA does *not* program steering
  itself; an NVMe-oF target whose connections never come up
  is often a missing or wrong Flow rule, not a STA bug.
- [`doca-debug`](../../doca-debug/SKILL.md) — the
  cross-cutting debug ladder (install / version / build /
  link / runtime / program / driver). STA-specific debug
  (transport-type mismatches, queue-depth oversize,
  IO-failed transport errors) overlays on top of that
  ladder.
