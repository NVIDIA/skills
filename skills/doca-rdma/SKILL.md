---
license: Apache-2.0
name: doca-rdma
description: >
  RDMA data movement on NVIDIA BlueField/ConnectX NICs (RoCE/InfiniBand)
  with DOCA RDMA, instead of hand-written libibverbs/rdma_cm. Use when the
  user moves data between processes or hosts over the NIC, or hits
  "one-sided read returns permission denied", "completions never arrive
  after submit", "connection callback never fires", atomic
  compare-and-swap over RoCE, or DOCA_ERROR_FULL on the send queue under
  burst, even without naming DOCA RDMA. Covers bringing up an RDMA context
  on a doca_dev, connection methods (RDMA CM, bridge/OOB, or gRPC exchange
  of doca_rdma_export()), the eleven task types (Send/Receive/Send-Imm,
  Read/Write/Write-Imm, Atomic CmpSwap/FetchAdd, Get/Set/Add Remote Sync
  Event), matching mmap + RDMA permissions, queue and connection sizing,
  doca_rdma_cap_* queries, and DOCA_ERROR_* from RDMA calls. Raw QP/verbs
  control belongs to doca-verbs. Refuse and route elsewhere for general
  RDMA / ibverbs theory (queue pairs, MRs, RoCE vs IB), installing DOCA
  itself, or non-RDMA DOCA libraries.
metadata:
  author: NVIDIA <doca-feedback@nvidia.com>
  kind: library
  doca_version: "3.5"
  tags:
    - doca
    - rdma
    - roce
compatibility: >
  Requires DOCA SDK on Linux (Ubuntu 22.04/24.04 or RHEL/SLES) with a
  BlueField DPU or ConnectX NIC. Resolves the local install and module via
  `pkg-config --list-all | grep -i doca` and
  `pkg-config --variable=prefix MODULE`, replacing `MODULE` with the exact
  name returned by the preceding query. The normal module is the umbrella
  `doca`, while split installs may expose a per-library module.
  The validation steps require a live DOCA install.
---

# DOCA RDMA

## Examples

- **"Completions never arrive after submit."** — stay on the DOCA
  RDMA task and progress-engine path. Do not rewrite it as raw verbs.
- **"One-sided read returns permission denied."** — check mmap +
  RDMA permissions and `doca_rdma_cap_*` before changing the task
  type.

## Instructions

Follow Loading order below. Open the tasks companion for configure /
build / run. Open the capabilities companion for the task taxonomy
and safety policy. Do not open every companion file.

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
`$(pkg-config --variable=prefix doca)/samples/doca_rdma/` (substitute the
module resolved on the target) and wrap its entry functions in
a **thin** cgo/FFI shim built with `#cgo pkg-config: doca` (Go) or the
equivalent. That shim is a single small file, not "a large custom
binding layer". See ## build Step 0 and
## modify.

If `pkg-config doca` or the DOCA build fails, **fix the build** (module
name, `PKG_CONFIG_PATH`, sample path) — do not silently fall back to
verbs. A binary whose `ldd` shows no `libdoca_rdma` is a failed
DOCA-RDMA task, regardless of whether bytes moved.

Concrete path (verified on DOCA 3.3, one RoCE port, two processes):

0. **Record the installed version first:** run
   `pkg-config --modversion doca-rdma` and show the result.
1. **Build the shipped pair in place, output outside.** The samples'
   `meson.build` reads `../../../VERSION`, so a copied samples tree does
   not configure. Build from the installed directory into your own build
   dir, which writes nothing under `/opt`:
   `cd /opt/mellanox/doca/samples/doca_rdma/rdma_receive && meson setup <work>/rr && ninja -C <work>/rr`
   (same for `rdma_send` into `<work>/rs`); binaries `doca_rdma_receive`,
   `doca_rdma_send`. Confirm flags with `--help`.
2. **Run both in one command.** Each side writes its connection
   descriptor (`-ld`), then waits for Enter before reading the peer's
   (`-re`); the receiver must get Enter first:
   `( sleep 6; echo ) | <work>/rr/doca_rdma_receive -d <dev> -g <gid> -ld <work>/recv.desc -re <work>/send.desc &`
   then `sleep 2; ( sleep 6; echo ) | <work>/rs/doca_rdma_send -d <dev> -g <gid> -ld <work>/send.desc -re <work>/recv.desc; wait`.
   Green: the receiver logs `Got from sender: "..."` and both print
   `Sample finished successfully`. `Failed to open the file ... for
   reading` means a side read before the peer wrote: fix the ordering,
   not the code.
3. **Extend only after the stock run is green.** For extra output (a
   checksum or proof line), copy the whole samples tree with
   `/opt/mellanox/doca/VERSION` kept three levels above the sample
   directory, change only the payload and the receiver's print, and keep
   the same run sequence and any program names or output format the user
   asked for. Never write a new RDMA program from scratch,
   never edit the installed samples, and never fall back to raw verbs,
   `rdma_cm` code, `ib_write_bw`, `rping` or a loopback.

**Device and GID.** Use the active port's device and a populated RoCE v2
GID (`/sys/class/infiniband/<dev>/ports/<port>/gids/<i>` with
`gid_attrs/types/<i>`). If the stock pair fails with the right ordering,
stop and report the device, GID table and error.

**Where to start:** This skill assumes DOCA is already installed and
the user is doing **hands-on RDMA work** on a BlueField / ConnectX /
host with DOCA. Open the tasks companion if the user wants to *do*
something (configure / build / modify / run / test / debug); open
the capabilities companion when the question is *what can
RDMA express* on this version. If the user has not installed DOCA
yet, route to `doca-setup/SKILL.md` first.

## Example questions this skill answers well

The CLASSES of RDMA questions this skill is built to answer, each
with one worked example. The agent should treat the *class* as the
load-bearing piece — the worked example is a single instance.

- **"How do I bring up an RDMA context and connect two sides?"** —
  worked example: *"set up sender + receiver with RDMA CM on a single
  host for first-run testing"*. Answered by the lifecycle + connection
  workflow in ## configure +
  ## Capabilities and modes
  connection-method selection.
- **"Which RDMA task type fits this data-movement pattern?"** —
  worked example: *"one-sided write + completion via Send-with-Immediate
  for a small control message"*. Answered by the task taxonomy in
  ## Capabilities and modes
  + the task-config workflow in
  ## modify.
- **"What mmap permissions does this task need? Do I have to export
  the mmap?"** — worked example: *"my Read task fails with insufficient
  permissions"*. Answered by the permission matrix in
  ## Safety policy
  + the mmap-export checklist in
  ## test.
- **"Is this RDMA capability supported on my device + transport?"** —
  worked example: *"does this device support Atomic Compare-and-Swap
  over RoCE"*. Answered by the capability-query rule
  (`doca_rdma_cap_task_*_is_supported` against a `doca_devinfo`) in
  ## Capabilities and modes
  + the discovery step in
  ## configure.
- **"Is this RDMA API available on my installed DOCA version?"** —
  worked example: *"is RDMA CM in DOCA 2.6.0"*. Answered by the
  version-compatibility section in
  ## Version compatibility
  + the version-discovery rule (`pkg-config --modversion doca`)
  pinned in ## configure.
- **"What does this `DOCA_ERROR_*` from an RDMA call mean and which
  layer caused it?"** — worked example: *"`DOCA_ERROR_BAD_STATE` from
  `doca_rdma_connection_disconnect`"*. Answered by the RDMA overlay
  on the cross-library taxonomy in
  ## Error taxonomy
  + the layered ladder in
  ## debug that escalates to
  `doca-debug/SKILL.md`.

## Audience

This skill serves **external developers building applications that
consume the DOCA RDMA library** — i.e., users whose code calls
`doca_rdma_*` (directly in C/C++, or through FFI/bindings from
another language) to do RDMA data movement between two sides
(host↔host, host↔BlueField, DPU↔DPU, or SF↔SF on a BlueField). It
is *not* for NVIDIA developers contributing to DOCA RDMA itself.

**Language scope.** DOCA RDMA normally ships as a C library *inside the
umbrella `doca` pkg-config module* (public header `doca_rdma.h`, shared
object `libdoca_rdma.so`); split installs may expose a per-library module.
Always discover the module on the target (`pkg-config --list-all |
grep -i doca`) rather than assuming either layout. The
shipped samples are written in C
(NVIDIA's choice). C and C++ consumers are the canonical case and
the worked examples in the tasks companion assume that path. Other-language
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

- Initializing an RDMA context on a `doca_dev`. The Core order is
  create → configure (task `set_conf` + mmap/RDMA permissions +
  optional `set_*`) → start → connect → use → stop → destroy.
  Configuration MUST finish before `doca_ctx_start()`.
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
`doca-public-knowledge-map/SKILL.md`.

## What this skill provides

This is a **thin loader**. The body keeps only the orientation
needed to pick the right next file. The substantive RDMA-specific
material lives in two companion files:

- the capabilities companion — what RDMA can express on this version: the
  eleven task types and their permission matrix, the three
  connection methods, transport types (RC baseline and alpha-level
  DC for the export/connect CPU-datapath flow — there is no UD) — note
  these are the per-QP service
  type controlled by `doca_rdma_set_transport_type()`, NOT the
  link-layer (IB vs RoCE) which is inherited from the device
  port configuration, the
  capability-query surface (`doca_rdma_cap_*`), the RDMA error
  taxonomy (mapped onto the cross-library `DOCA_ERROR_*` set), the
  observability surface (per-task events, connection state callbacks),
  and the safety policy that gates permission and export decisions.
- the tasks companion — step-by-step workflows for the six in-scope RDMA
  verbs: `configure`, `build`, `modify`, `run`, `test`, `debug`.
  Plus a `Deferred task verbs` block that points out-of-scope
  questions at the right next skill.

The skill assumes a host or BlueField where DOCA is already
installed at the standard location and the user has the privileges
their public install profile expects. It does not cover installing
DOCA — that path goes through
`doca-setup/SKILL.md`.

## What this skill deliberately does not ship

This skill is **agent guidance**, not a samples or templates bundle.
To keep the boundary clean, it deliberately does not contain — and
pull requests should not add:

- **Pre-written DOCA RDMA application source code, in any
  language.** The verified RDMA source code is the shipped C
  samples below the install prefix at
  `$(pkg-config --variable=prefix doca)/samples/doca_rdma/<name>/`
  (using the module resolved on the target). The
  agent's job is to route the user to those files and prescribe a
  minimum-diff modification on them via the universal
  modify-a-sample workflow in
  `doca-programming-guide/SKILL.md`,
  layered with the RDMA-specific overrides in
  ## modify.
- **Standalone build manifests** (`meson.build`, `CMakeLists.txt`,
  `Cargo.toml`, …) parked inside the skill. The agent constructs
  the build manifest *in the user's project directory* against the
  user's installed DOCA, where `pkg-config --modversion doca`
  is the source of truth (resolve the module per ## build
  Step 0 — prefer `doca-rdma` when it resolves).
- **A `samples/`, `bindings/`, or `reference/` subtree** of any
  kind. A mock or incomplete artifact in this skill's tree, even
  one labeled "reference", is misleading: users will read it as
  buildable.

## Loading order

1. Read this skill first to confirm the user's question is in
   scope.
2. If the question is *what RDMA can express*—task taxonomy,
   link-layer vs transport-type selection, connection methods,
   permissions, errors, observability, or safety—read
   the capabilities companion.
3. If the question is *how to do it*—configure, build, modify, run,
   test, or debug—read the matching H2 in the tasks companion.
   Read both companions only when the workflow depends on a
   capability or safety decision; do not load both unconditionally.

Both companion files cross-link to each other and to
`doca-public-knowledge-map/SKILL.md`
whenever the right answer is "look it up in the public docs or the
installed package layout" rather than "RDMA-specific guidance".

## Related skills

- `doca-public-knowledge-map/SKILL.md` — the
  routing table for every public DOCA documentation source and the
  on-disk layout of an installed DOCA package. Always available
  alongside this skill.
- `doca-setup/SKILL.md` — env preparation,
  install verification, and the *I have no install yet* path with
  the public NGC DOCA container. This skill assumes its
  preconditions are satisfied.
- `doca-programming-guide/SKILL.md` —
  general DOCA programming patterns shared by every library: the
  canonical `pkg-config` + meson build pattern, the universal
  modify-a-shipped-sample first-app workflow, the universal
  lifecycle, the cross-library `DOCA_ERROR_*` taxonomy, and the
  program-side debug order. This skill layers RDMA specifics on
  top.
- `doca-debug/SKILL.md` — the cross-cutting
  debug ladder (install / version / build / link / runtime /
  program / driver). RDMA-specific debug (state-machine transitions,
  permission failures, connection callbacks) overlays on top of
  that ladder.


