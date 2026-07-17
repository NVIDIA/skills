---
license: Apache-2.0
name: doca-common
description: >
  Use this skill whenever the user is doing hands-on DOCA programming
  on a BlueField DPU or ConnectX NIC and needs the foundation
  primitives every per-library context rests on — walking the
  doca_ctx lifecycle, discovering doca_dev / doca_devinfo and gating
  on doca_*_cap_* before trusting a feature, wiring doca_mmap /
  doca_buf_inventory / doca_buf for zero-copy I/O across libraries,
  driving doca_pe for completions, or DOCA Log's two-tier
  (--sdk-log-level vs app-side) model. Trigger even when the user
  does not say "DOCA Common" — typical implicit phrasings include
  "my tasks submit but nothing completes", "DOCA_ERROR_BAD_STATE
  from doca_ctx_start", "--sdk-log-level does nothing for my
  DOCA_LOG_DBG lines", "share a buf between doca_dma and doca_rdma",
  or "crashes far from the offending line". Refuse and route
  elsewhere for per-library questions in isolation (load doca-flow /
  doca-rdma / doca-eth alongside), installing DOCA (doca-setup), or
  doc lookup (doca-public-knowledge-map).
metadata:
  kind: library
compatibility: >
  Requires DOCA SDK installed at /opt/mellanox/doca on Linux (Ubuntu
  22.04/24.04 or RHEL/SLES) with a BlueField DPU or ConnectX NIC
  attached. doca-common is present on every healthy DOCA install;
  reads the user's local install via `pkg-config --modversion
  doca-common` and inspects /opt/mellanox/doca/{lib,include,samples,applications}.
---

# DOCA Common

**Where to start:** This skill is the **foundation every DOCA app
loads first** — before doca-flow, doca-rdma, doca-eth, doca-comch, or
any other higher-level library. Every `doca_<library>_*` context is
built on top of `doca_ctx`, every device handle is a `doca_dev`
discovered through `doca_devinfo`, every zero-copy buffer is a
`doca_buf` from a `doca_buf_inventory` over a `doca_mmap`, every
task completion drains through a `doca_pe`, and every log line emits
through `doca_log`. Open [`CAPABILITIES.md`](CAPABILITIES.md) when
the question is *what does Common express* on this install; open
[`TASKS.md`](TASKS.md) when the user wants to *do* something
(configure / build / modify / run / test / debug). If the user has
not installed DOCA yet, route to
[`doca-setup`](../../doca-setup/SKILL.md) first. If the user is
already past the foundation and asking a library-specific question
(e.g. *"how do I program a Flow pipe"*), load the matching per-library
skill alongside this one — they cross-link back here for the shared
primitives.

## Example questions this skill answers well

The CLASSES of doca-common questions this skill is built to answer,
each with one worked example. The agent should treat the *class* as
the load-bearing piece — the worked example is a single instance.

- **"What is the doca-common foundation I have to set up BEFORE I
  open a doca-flow / doca-rdma / doca-eth / … context?"** — worked
  example: *"I'm starting a brand-new DOCA Flow program on
  BlueField-3; what's the doca-common skeleton I need before I open
  the Flow port?"*. Answered by the universal foundation walk in
  [`TASKS.md ## configure`](TASKS.md#configure) +
  [`CAPABILITIES.md ## ctx`](CAPABILITIES.md#ctx) +
  [`CAPABILITIES.md ## dev`](CAPABILITIES.md#dev) +
  [`CAPABILITIES.md ## progress engine`](CAPABILITIES.md#progress-engine).
- **"How do I discover a device and gate on its capabilities before
  trusting the public docs?"** — worked example: *"I want to use
  `doca_eth_txq` but the docs hint at a feature only on certain
  firmware bands"*. Answered by the capability-discovery rule
  (`doca_devinfo_create_list` → `doca_*_cap_*` against the active
  `doca_devinfo` is the runtime authority) in
  [`CAPABILITIES.md ## dev`](CAPABILITIES.md#dev) +
  [`TASKS.md ## use`](TASKS.md#use).
- **"What's the doca_buf / doca_mmap / doca_buf_inventory wiring
  for zero-copy I/O, and what's the lifecycle order?"** — worked
  example: *"I want to register a user-space buffer with my device,
  carve it into N data-plane buffers, and reference-count them
  across multiple DOCA libraries"*. Answered by the zero-copy
  buffer model in
  [`CAPABILITIES.md ## buf`](CAPABILITIES.md#buf) +
  the buffer-lifecycle walk in
  [`TASKS.md ## configure`](TASKS.md#configure) +
  [`TASKS.md ## use`](TASKS.md#use).
- **"How does the progress engine work and where do I have to call
  it?"** — worked example: *"my doca_rdma task submits cleanly but
  nothing completes — what loop am I missing?"*. Answered by the PE
  surface in
  [`CAPABILITIES.md ## progress engine`](CAPABILITIES.md#progress-engine)
  + the run-loop pattern in [`TASKS.md ## run`](TASKS.md#run).
- **"Why don't my DOCA log lines appear at the level I expect, and
  what's the difference between `--sdk-log-level` and the app-side
  setter?"** — worked example: *"I set `--sdk-log-level DEBUG`, my
  own `DOCA_LOG_DBG` lines still don't print"*. Answered by the
  two-tier log model in [`CAPABILITIES.md ## log`](CAPABILITIES.md#log)
  + the tier-flip iteration in [`TASKS.md ## log`](TASKS.md#log).
- **"What does this `DOCA_ERROR_*` from a `doca_buf_*` / `doca_ctx_*`
  / `doca_dev_*` / `doca_pe_*` / `doca_log_*` call mean?"** —
  worked example: *"`DOCA_ERROR_BAD_STATE` from `doca_ctx_start`"*.
  Answered by the Common overlay on the cross-library `DOCA_ERROR_*`
  taxonomy in
  [`CAPABILITIES.md ## Error taxonomy`](CAPABILITIES.md#error-taxonomy)
  + the layered ladder in
  [`TASKS.md ## debug`](TASKS.md#debug) that escalates to
  [`doca-debug`](../../doca-debug/SKILL.md).

## Audience

This skill serves **every external developer building applications
that consume any DOCA library** — i.e., users whose code calls *any*
`doca_*` symbol (directly in C/C++, or through FFI/bindings from
another language). Whether the user's primary library is doca-flow,
doca-rdma, doca-eth, doca-comch, doca-dma, doca-rmax, doca-sha,
doca-aes-gcm, doca-erasure-coding, or any other, the doca-common
surface is *under* it and the user will hit `doca_buf`, `doca_ctx`,
`doca_dev`, `doca_pe`, and `doca_log` as part of the first-app
journey. It is *not* for NVIDIA developers contributing to DOCA
Common itself.

## Language scope

DOCA Common ships as a C library with `pkg-config` module name
`doca-common`. The shipped samples that demonstrate Common primitives
live *inside* every per-library samples tree (any
`/opt/mellanox/doca/samples/<library>/<sample>/*_main.c` is a worked
example of the universal foundation — `doca_devinfo_create_list`
→ `doca_dev_open` → per-library `doca_ctx` create → `doca_pe_create`
→ `doca_pe_connect_ctx` → `doca_ctx_start` → submit work → drive
`doca_pe_progress` → drain completions → `doca_ctx_stop` → destroy).
C and C++ consumers are the canonical case and the worked examples
in `TASKS.md` assume that path. Other-language consumers (Rust, Go,
Python, …) consume the same `*.so` through FFI or language-specific
bindings; the skill's contribution in that case is to keep the
universal foundation walk, the lifecycle, the capability-discovery
rule, the PE-drives-completion rule, and the two-tier log model
language-neutral, and to route the agent to the public C ABI as the
authoritative surface that any wrapper will eventually call.

## When to load this skill

Load this skill whenever the user is doing **any** hands-on DOCA
work — it is the foundation. Concretely:

- Setting up the universal DOCA-side skeleton before opening any
  per-library context (Flow, RDMA, Eth, Comch, DMA, Rmax, …).
- Discovering devices and representors and gating capability use
  on the active `doca_devinfo` via the `doca_*_cap_*` family.
- Wiring `doca_mmap` + `doca_buf_inventory` + `doca_buf` for
  zero-copy I/O that crosses libraries (e.g. doca-eth feeds
  doca-dma feeds doca-rdma — they share the same buf surface).
- Driving the progress engine (`doca_pe_create` /
  `doca_pe_connect_ctx` / `doca_pe_progress`) — the universal
  task-completion drain every DOCA Core context relies on.
- Wiring DOCA Log into a fresh app or modifying a shipped sample to
  add the user's own per-component log lines via the two-tier
  (`--sdk-log-level` vs app-side registry) model.
- Debugging a `DOCA_ERROR_*` returned from any `doca_buf_*` /
  `doca_ctx_*` / `doca_dev_*` / `doca_pe_*` / `doca_log_*` call —
  the Common surface is where most lifecycle / capability /
  permission errors first surface for higher-level libraries.
- Designing or extending non-C bindings (Rust, Go, Python, …) that
  wrap any DOCA library — for the universal foundation surface
  (buf, ctx, dev, pe, log) the wrapper has to expose first.

Do **not** load this skill for library-specific Flow, RDMA, Eth,
Comch, DMA, Rmax, … questions in isolation — load the matching
per-library skill alongside this one. Do **not** load this skill for
general DOCA orientation, install of DOCA itself, or
*"where do I find docs"*. For those, use
[`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md)
or [`doca-setup`](../../doca-setup/SKILL.md).

## What this skill provides

This is a **thin loader**. The body keeps only the orientation
needed to pick the right next file. The substantive Common material
lives in two companion files:

- `CAPABILITIES.md` — what doca-common expresses on this install:
  the `## Capabilities and modes` overview of the universal
  primitives every DOCA application touches; the five subsystem
  H2s (`## log`, `## buf`, `## ctx`, `## dev`, `## progress
  engine`) that own the per-primitive surface; the
  `## Version compatibility` doca-common-specific overlay;
  the `## Error taxonomy` Common-side view of the universal
  `DOCA_ERROR_*` set; the `## Observability` surface (logs, PE
  events, capability snapshots); and the `## Safety policy` overlay
  on the bundle-wide hardware-safety meta-policy.
- `TASKS.md` — step-by-step workflows for the universal verbs
  (`configure`, `build`, `modify`, `run`, `test`, `debug`, `use`)
  PLUS a `## log` verb-side that covers the two-tier log model in
  workflow form. Plus a `Deferred task verbs` block that points
  install / deploy / rollback questions at the right next skill.

The skill assumes a host or BlueField where DOCA is already
installed at the standard location and the user has the privileges
their public install profile expects. It does not cover installing
DOCA — that path goes through
[`doca-setup`](../../doca-setup/SKILL.md).

## What this skill deliberately does not ship

This skill is **agent guidance**, not a samples or templates
bundle. To keep the boundary clean, it deliberately does not
contain — and pull requests should not add:

- **Pre-written DOCA application source code, in any language.**
  The verified Common usage shows up inside every shipped DOCA
  sample at `/opt/mellanox/doca/samples/<library>/<sample>/*_main.c`
  and inside every shipped reference application. The agent's job
  is to route the user to those files and prescribe a minimum-diff
  modification on them via the universal modify-a-sample workflow
  in
  [`doca-programming-guide`](../../doca-programming-guide/SKILL.md).
- **Standalone build manifests** (`meson.build`,
  `CMakeLists.txt`, `Cargo.toml`, `setup.py`, `go.mod`, …) parked
  inside the skill. The agent constructs the build manifest *in
  the user's project directory* against the user's installed DOCA,
  where `pkg-config --modversion doca-common` is the source of
  truth.
- **A `samples/`, `bindings/`, or `reference/` subtree** of any
  kind. A mock or incomplete artifact in this skill's tree, even
  one labeled "reference", is misleading: users will read it as
  buildable.

## Loading order

1. Read this `SKILL.md` first to confirm the user's question is in
   scope.
2. **For the universal primitives (`log` / `buf` / `ctx` / `dev` /
   `progress engine`), the `pkg-config --modversion doca-common`
   anchor, the Common error overlay, observability, and safety
   policy, see [CAPABILITIES.md](CAPABILITIES.md).**
3. **For step-by-step workflows — configure, build, modify, run,
   test, debug, use, log — see [TASKS.md](TASKS.md).**

Both companion files cross-link to each other,
[`doca-version`](../../doca-version/SKILL.md) for the canonical
version-handling rules,
[`doca-programming-guide`](../../doca-programming-guide/SKILL.md)
for the universal modify-a-shipped-sample workflow and the
cross-library `DOCA_ERROR_*` taxonomy,
[`doca-debug`](../../doca-debug/SKILL.md) for the cross-cutting
debug ladder, and
[`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md)
whenever the right answer is "look it up in the public docs or the
installed package layout" rather than "Common-specific guidance".

## Related skills

- [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md) —
  the routing table for every public DOCA documentation source and
  the on-disk layout of an installed DOCA package. Always available
  alongside this skill; this skill expects to be able to defer
  documentation-finding and install-layout questions there instead
  of duplicating them.
- [`doca-setup`](../../doca-setup/SKILL.md) — env preparation,
  install verification, and the *I have no install yet* path with
  the public NGC DOCA container (`nvcr.io/nvidia/doca/doca`) as the
  universal Stage-1 fallback. This skill assumes its preconditions
  are satisfied.
- [`doca-version`](../../doca-version/SKILL.md) — canonical DOCA
  version-handling rules (four-way match, NGC semantics,
  headers-win-over-docs). `pkg-config --modversion doca-common` is
  the build-time anchor *every* DOCA install carries; this skill's
  `## Version compatibility` overlays the Common-specific notes on
  top.
- [`doca-programming-guide`](../../doca-programming-guide/SKILL.md) —
  general DOCA programming patterns shared by every library: the
  canonical `pkg-config` + meson build pattern, the universal
  modify-a-shipped-sample first-app workflow, the universal
  lifecycle, the cross-library `DOCA_ERROR_*` taxonomy, and the
  program-side debug order. This skill is the *primitives* layer
  the programming-guide patterns rest on.
- [`doca-debug`](../../doca-debug/SKILL.md) — the cross-cutting
  debug ladder (install / version / build / link / runtime /
  program / driver) and the verbosity-escalation surface. DOCA Log
  is the foundation `doca-debug` builds its runtime-debug story on;
  this skill is where the two-tier log model and the universal
  lifecycle errors first surface, and `doca-debug` cross-links here
  for both.
- [`doca-structured-tools-contract`](../../doca-structured-tools-contract/SKILL.md) —
  the bundle's structured-tools precedence rule (detect / prefer /
  fall back / report). The Command appendix in
  [TASKS.md](TASKS.md) honors this contract.
- [`doca-hardware-safety`](../../doca-hardware-safety/SKILL.md) —
  the cross-cutting hardware-safety meta-policy this skill's
  `## Safety policy` overlays.
- Per-library skills (`doca-flow`, `doca-rdma`, `doca-eth`,
  `doca-comch`, `doca-dma`, `doca-rmax`, `doca-sha`, `doca-aes-gcm`,
  `doca-erasure-coding`, `doca-compress`, `doca-dpa`, `doca-gpunetio`,
  `doca-pcc`, `doca-sta`, `doca-telemetry`, `doca-urom`,
  `doca-verbs`, `doca-argp`, `doca-devemu`,
  `doca-dpdk-bridge`, …) — every per-library skill cross-links
  back to this skill for the foundation primitives. Loading this
  skill alongside any of them is the recommended default.
