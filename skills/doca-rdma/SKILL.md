---
license: Apache-2.0
name: doca-rdma
description: >
  Use this skill when the user is doing hands-on DOCA RDMA programming
  on a BlueField DPU, ConnectX NIC, or DOCA host — bringing up an RDMA
  context on a doca_dev, picking a connection method (RDMA CM,
  bridge/OOB, or gRPC exchange of doca_rdma_export()), enabling one of
  the eleven task types (Send/Receive/Send-Imm, Read/Write/Write-Imm,
  Atomic CmpSwap/FetchAdd, Get/Set/Add Remote Sync Event), setting
  matching mmap + RDMA permissions, sizing queues and connections,
  querying doca_rdma_cap_*, or debugging DOCA_ERROR_* from an RDMA
  call. Trigger even when the user does not mention "DOCA RDMA" —
  typical implicit phrasings include "one-sided read returns permission
  denied", "completions never arrive after submit", "connection
  callback never fires", "how do I do atomic compare-and-swap over
  RoCE", or "send queue hits DOCA_ERROR_FULL under burst". Refuse and
  route elsewhere for general RDMA / ibverbs theory (queue pairs, MRs,
  RoCE vs IB), installing DOCA itself, or non-RDMA DOCA libraries.
metadata:
  kind: library
compatibility: >
  Requires DOCA SDK at /opt/mellanox/doca on Linux (Ubuntu 22.04/24.04 or
  RHEL/SLES) with a BlueField DPU or ConnectX NIC. Reads the local install
  via `pkg-config doca` (umbrella module shipping the RDMA library; no
  separate doca-rdma.pc — find the real name via `pkg-config --list-all |
  grep -i doca`, set PKG_CONFIG_PATH to
  /opt/mellanox/doca/lib/{arch}/pkgconfig if needed) and inspects
  /opt/mellanox/doca/{lib,include,samples,applications}.

---

# DOCA RDMA

## Non-negotiable: the deliverable uses DOCA RDMA, not raw verbs

When this skill is in scope, the user is asking for **DOCA RDMA**. The
program you produce **must link `libdoca_rdma` and call the
`doca_rdma_*` API** (directly in C/C++, or through a thin FFI/cgo shim
from another language). Do **NOT** implement the request with raw
`libibverbs` / `librdmacm` / RDMA-CM and call it done. Those move bytes
but completely bypass DOCA — which defeats the entire purpose of using
this library, loses the DOCA programming model (progress engine, task/
event lifecycle, capability discovery, portability across BlueField/
ConnectX generations), and is the single most common failure mode.

"Raw verbs is fewer lines" / "it avoids building a DOCA binding layer"
is **not** an acceptable reason to bypass DOCA. The correct low-friction
path for a non-C language (Go, Rust, Python, …) is **not** to re-bind
the whole API — it is to start from a **shipped DOCA RDMA sample** under
`/opt/mellanox/doca/samples/doca_rdma/` and wrap its entry functions in
a **thin** cgo/FFI shim built with `#cgo pkg-config: doca` (Go) or the
equivalent. That shim is a single small file, not "a large custom
binding layer". See [`TASKS.md ## build`](TASKS.md#build) Step 0 and
[`TASKS.md ## modify`](TASKS.md#modify).

If `pkg-config doca` or the DOCA build fails, **fix the build** (module
name, `PKG_CONFIG_PATH`, sample path) — do not silently fall back to
verbs. A binary whose `ldd` shows no `libdoca_rdma` is a failed
DOCA-RDMA task, regardless of whether bytes moved.

**Where to start:** This skill assumes DOCA is already installed and
the user is doing **hands-on RDMA work** on a BlueField / ConnectX /
host with DOCA. Open [`TASKS.md`](TASKS.md) if the user wants to *do*
something (configure / build / modify / run / test / debug); open
[`CAPABILITIES.md`](CAPABILITIES.md) when the question is *what can
RDMA express* on this version. If the user has not installed DOCA
yet, route to [`doca-setup`](../../doca-setup/SKILL.md) first.

## Example questions this skill answers well

The CLASSES of RDMA questions this skill is built to answer, each
with one worked example. The agent should treat the *class* as the
load-bearing piece — the worked example is a single instance.

- **"How do I bring up an RDMA context and connect two sides?"** —
  worked example: *"set up sender + receiver with RDMA CM on a single
  host for first-run testing"*. Answered by the lifecycle + connection
  workflow in [`TASKS.md ## configure`](TASKS.md#configure) +
  [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes)
  connection-method selection.
- **"Which RDMA task type fits this data-movement pattern?"** —
  worked example: *"one-sided write + completion via Send-with-Immediate
  for a small control message"*. Answered by the task taxonomy in
  [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes)
  + the task-config workflow in
  [`TASKS.md ## modify`](TASKS.md#modify).
- **"What mmap permissions does this task need? Do I have to export
  the mmap?"** — worked example: *"my Read task fails with insufficient
  permissions"*. Answered by the permission matrix in
  [`CAPABILITIES.md ## Safety policy`](CAPABILITIES.md#safety-policy)
  + the mmap-export checklist in
  [`TASKS.md ## test`](TASKS.md#test).
- **"Is this RDMA capability supported on my device + transport?"** —
  worked example: *"does this device support Atomic Compare-and-Swap
  over RoCE"*. Answered by the capability-query rule
  (`doca_rdma_cap_task_*_is_supported` against a `doca_devinfo`) in
  [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes)
  + the discovery step in
  [`TASKS.md ## configure`](TASKS.md#configure).
- **"Is this RDMA API available on my installed DOCA version?"** —
  worked example: *"is RDMA CM in DOCA 2.6.0"*. Answered by the
  version-compatibility section in
  [`CAPABILITIES.md ## Version compatibility`](CAPABILITIES.md#version-compatibility)
  + the version-discovery rule (`pkg-config --modversion doca`)
  pinned in [`TASKS.md ## configure`](TASKS.md#configure).
- **"What does this `DOCA_ERROR_*` from an RDMA call mean and which
  layer caused it?"** — worked example: *"`DOCA_ERROR_BAD_STATE` from
  `doca_rdma_connection_disconnect`"*. Answered by the RDMA overlay
  on the cross-library taxonomy in
  [`CAPABILITIES.md ## Error taxonomy`](CAPABILITIES.md#error-taxonomy)
  + the layered ladder in
  [`TASKS.md ## debug`](TASKS.md#debug) that escalates to
  [`doca-debug`](../../doca-debug/SKILL.md).

## Audience

This skill serves **external developers building applications that
consume the DOCA RDMA library** — i.e., users whose code calls
`doca_rdma_*` (directly in C/C++, or through FFI/bindings from
another language) to do RDMA data movement between two sides
(host↔host, host↔BlueField, DPU↔DPU, or SF↔SF on a BlueField). It
is *not* for NVIDIA developers contributing to DOCA RDMA itself.

**Language scope.** DOCA RDMA ships as a C library *inside the umbrella
`doca` pkg-config module* (public header `doca_rdma.h`, shared object
`libdoca_rdma.so`). Current DOCA installs do **not** ship a separate
`doca-rdma.pc`; `pkg-config doca` is what resolves the RDMA cflags/libs.
Always discover the module on the target (`pkg-config --list-all |
grep -i doca`) rather than assuming a per-library `.pc` exists. The
shipped samples are written in C
(NVIDIA's choice). C and C++ consumers are the canonical case and
the worked examples in `TASKS.md` assume that path. Other-language
consumers (Rust, Go, Python, …) consume the same `*.so` through FFI
or language-specific bindings; the skill's contribution in that case
is to keep the lifecycle, capability-discovery, permission-matrix,
error-taxonomy, and connection-method guidance language-neutral, and
to route the agent to the public C ABI as the authoritative surface
that any wrapper will eventually call. **The non-C deliverable is still
a DOCA program**: a thin cgo/FFI shim over the shipped `doca_rdma`
sample that links `libdoca_rdma` (`#cgo pkg-config: doca`) — never a
raw-libibverbs reimplementation chosen to avoid wrapping DOCA (see the
mandate at the top of this file).

## When to load this skill

Load this skill when the user is doing hands-on DOCA RDMA work, in
any language. Concretely:

- Initializing an RDMA context on a `doca_dev` and configuring at
  least one task type before `doca_ctx_start()`.
- Establishing a connection — picking between RDMA CM
  (`doca_rdma_connect_to_addr()` / `doca_rdma_start_listen_to_port()`
  / `doca_rdma_connection_accept()`), bridge / OOB
  (`doca_rdma_bridge_*`), or gRPC (out-of-band exchange
  of `doca_rdma_export()` output).
- Setting permissions on `doca_mmap` correctly for the chosen task
  type (Read needs RDMA-read + local read-write; Write needs
  RDMA-write; Atomic needs RDMA-atomic; Send needs only local
  read-write).
- Reading / setting library properties via `doca_rdma_set_*` and
  `doca_rdma_cap_get_*` to size queues, list lengths, and
  transport-type selection.
- Checking which RDMA task types and transport types are supported
  on the active `doca_devinfo`.
- Debugging a `DOCA_ERROR_*` returned from an RDMA call (lifecycle
  vs. permission vs. capability vs. driver-below) and the connection
  state-machine transitions (`doca_rdma_set_connection_state_callbacks`).
- Designing or extending non-C bindings (Rust, Go, Python, …) that
  wrap the RDMA C ABI — for the lifecycle, permission, and
  capability rules the wrapper must honor.

Do **not** load this skill for general DOCA orientation, install of
DOCA itself, or non-RDMA library questions. For those, use
[`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md).

## What this skill provides

This is a **thin loader**. The body keeps only the orientation
needed to pick the right next file. The substantive RDMA-specific
material lives in two companion files:

- `CAPABILITIES.md` — what RDMA can express on this version: the
  eleven task types and their permission matrix, the three
  connection methods, transport types (RC baseline and DC for
  the export/connect CPU-datapath flow — there is no UD) — note
  these are the per-QP service
  type controlled by `doca_rdma_set_transport_type()`, NOT the
  link-layer (IB vs RoCE) which is inherited from the device
  port configuration, the
  capability-query surface (`doca_rdma_cap_*`), the RDMA error
  taxonomy (mapped onto the cross-library `DOCA_ERROR_*` set), the
  observability surface (per-task events, connection state callbacks),
  and the safety policy that gates permission and export decisions.
- `TASKS.md` — step-by-step workflows for the six in-scope RDMA
  verbs: `configure`, `build`, `modify`, `run`, `test`, `debug`.
  Plus a `Deferred task verbs` block that points out-of-scope
  questions at the right next skill.

The skill assumes a host or BlueField where DOCA is already
installed at the standard location and the user has the privileges
their public install profile expects. It does not cover installing
DOCA — that path goes through
[`doca-setup`](../../doca-setup/SKILL.md).

## What this skill deliberately does not ship

This skill is **agent guidance**, not a samples or templates bundle.
To keep the boundary clean, it deliberately does not contain — and
pull requests should not add:

- **Pre-written DOCA RDMA application source code, in any
  language.** The verified RDMA source code is the shipped C
  samples at `/opt/mellanox/doca/samples/doca_rdma/<name>/`. The
  agent's job is to route the user to those files and prescribe a
  minimum-diff modification on them via the universal
  modify-a-sample workflow in
  [`doca-programming-guide`](../../doca-programming-guide/SKILL.md),
  layered with the RDMA-specific overrides in
  [`TASKS.md ## modify`](TASKS.md#modify).
- **Standalone build manifests** (`meson.build`, `CMakeLists.txt`,
  `Cargo.toml`, …) parked inside the skill. The agent constructs
  the build manifest *in the user's project directory* against the
  user's installed DOCA, where `pkg-config --modversion doca`
  is the source of truth (resolve the module per `TASKS.md ## build`
  Step 0 — there is normally no separate `doca-rdma.pc`).
- **A `samples/`, `bindings/`, or `reference/` subtree** of any
  kind. A mock or incomplete artifact in this skill's tree, even
  one labeled "reference", is misleading: users will read it as
  buildable.

## Loading order

1. Read this `SKILL.md` first to confirm the user's question is in
   scope.
2. **For the RDMA capability matrix, task taxonomy, connection
   methods, permission matrix, error taxonomy, observability, and
   safety policy, see [CAPABILITIES.md](CAPABILITIES.md).**
3. **For step-by-step workflows — configure, build, modify, run,
   test, debug — see [TASKS.md](TASKS.md).**

Both companion files cross-link to each other and to
[`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md)
whenever the right answer is "look it up in the public docs or the
installed package layout" rather than "RDMA-specific guidance".

## Related skills

- [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md) — the
  routing table for every public DOCA documentation source and the
  on-disk layout of an installed DOCA package. Always available
  alongside this skill.
- [`doca-setup`](../../doca-setup/SKILL.md) — env preparation,
  install verification, and the *I have no install yet* path with
  the public NGC DOCA container. This skill assumes its
  preconditions are satisfied.
- [`doca-programming-guide`](../../doca-programming-guide/SKILL.md) —
  general DOCA programming patterns shared by every library: the
  canonical `pkg-config` + meson build pattern, the universal
  modify-a-shipped-sample first-app workflow, the universal
  lifecycle, the cross-library `DOCA_ERROR_*` taxonomy, and the
  program-side debug order. This skill layers RDMA specifics on
  top.
- [`doca-debug`](../../doca-debug/SKILL.md) — the cross-cutting
  debug ladder (install / version / build / link / runtime /
  program / driver). RDMA-specific debug (state-machine transitions,
  permission failures, connection callbacks) overlays on top of
  that ladder.


