---
license: Apache-2.0
name: doca-dma
description: >
  Use this skill when the user is doing hands-on DOCA DMA
  programming — bringing up a doca_dma context, configuring the
  single doca_dma_task_memcpy task type, sizing buffers via the
  doca_dma_cap_task_memcpy_* queries, setting LOCAL_READ_ONLY /
  LOCAL_READ_WRITE permissions on source / destination doca_mmap
  regions (plus doca_mmap_export_* for cross-peer copies),
  driving the progress engine, or debugging DOCA_ERROR_* returns.
  Trigger even when the user does not explicitly mention "DOCA
  DMA" or "doca_mmap" — typical implicit phrasings include
  "memcpy host buffer to BlueField without using the CPU",
  "offload a bulk copy to the DPU", "copy returns NOT_PERMITTED
  on first submit", "buffer too big for one DMA task", "task
  submitted but no completion", or "scatter-gather copy between
  two memory regions". Refuse and route elsewhere for
  cross-network copies (DOCA RDMA), producer/consumer messaging
  (DOCA Comch), DOCA Core / progress-engine internals, or DOCA
  install — those belong to other skills.
metadata:
  kind: library
compatibility: >
  Requires DOCA SDK installed at /opt/mellanox/doca on Linux
  (Ubuntu 22.04/24.04 or RHEL/SLES) with a BlueField DPU or
  ConnectX NIC attached. Reads the user's local install via
  `pkg-config doca-dma` and inspects
  /opt/mellanox/doca/{lib,include,samples,applications}.
---

# DOCA DMA

**Where to start:** This skill assumes DOCA is already installed
and the user is doing **hands-on DMA work** on a BlueField /
ConnectX / host with DOCA. Open [`TASKS.md`](TASKS.md) if the user
wants to *do* something (configure / build / modify / run / test /
debug); open [`CAPABILITIES.md`](CAPABILITIES.md) when the question
is *what can DMA express* on this version. If the user has not
installed DOCA yet, route to
[`doca-setup`](../../doca-setup/SKILL.md) first. If the user is
not sure DMA is even the right library — the data has to traverse
the network, or the flow is small messages between two processes —
read the path-selection rule in
[`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes)
before configuring anything.

## Example questions this skill answers well

The CLASSES of DMA questions this skill is built to answer, each
with one worked example. The agent should treat the *class* as
the load-bearing piece — the worked example is a single instance.

- **"How do I bring up a DOCA DMA context and copy a buffer
  between host and DPU?"** — worked example: *"copy a 64 KiB
  buffer from host memory to DPU memory in one task, starting from
  the shipped DMA Copy reference application"*. Answered by the
  lifecycle + memcpy-task workflow in
  [`TASKS.md ## configure`](TASKS.md#configure) +
  [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes)
  task-type table.
- **"How big a buffer can I memcpy in one task on this device?"** —
  worked example: *"can I copy 16 MiB in a single
  `doca_dma_task_memcpy`"*. Answered by the capability-query rule
  (`doca_dma_cap_task_memcpy_get_max_buf_size`, plus
  `_get_max_buf_list_len` for scatter-gather) in
  [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes)
  + the discovery step in
  [`TASKS.md ## configure`](TASKS.md#configure).
- **"What permissions do my source and destination mmaps need?"** —
  worked example: *"my memcpy task returns
  `DOCA_ERROR_NOT_PERMITTED` on the first submit"*. Answered by the
  source / destination permission matrix in
  [`CAPABILITIES.md ## Safety policy`](CAPABILITIES.md#safety-policy)
  + the permission checklist in
  [`TASKS.md ## test`](TASKS.md#test).
- **"Should I use DOCA DMA or DOCA RDMA / Comch / a plain CPU
  memcpy for this copy?"** — worked example: *"I have a 1 MiB copy
  that has to go from a host process to a DPU process; do I want
  DMA or Comch fast-path"*. Answered by the *"when to use DMA"*
  vs *"when not to"* path-selection bullet in
  [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes)
  + the routing pointers in [`## Related skills`](#related-skills).
- **"Is DOCA DMA on my installed version, and is the memcpy task
  supported on my device?"** — worked example: *"is
  `doca_dma_task_memcpy` available on DOCA 2.6 against this
  ConnectX-6"*. Answered by the version-compatibility overlay in
  [`CAPABILITIES.md ## Version compatibility`](CAPABILITIES.md#version-compatibility),
  which cross-links the canonical detection chain in
  [`doca-version`](../../doca-version/SKILL.md), plus the
  capability-query rule in
  [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes).
- **"What does this `DOCA_ERROR_*` from a DMA call mean and which
  layer caused it?"** — worked example: *"`DOCA_ERROR_AGAIN` from
  `doca_task_submit` on a `doca_dma_task_memcpy`"*. Answered by
  the DMA overlay on the cross-library taxonomy in
  [`CAPABILITIES.md ## Error taxonomy`](CAPABILITIES.md#error-taxonomy)
  + the layered ladder in
  [`TASKS.md ## debug`](TASKS.md#debug) that escalates to
  [`doca-debug`](../../doca-debug/SKILL.md).

## Audience

This skill serves **external developers building applications
that consume the DOCA DMA library** — i.e., users whose code calls
`doca_dma_*` (directly in C/C++, or through FFI/bindings from
another language) to copy bytes between two `doca_mmap` regions
using the BlueField DMA engine instead of the host CPU. It is
*not* for NVIDIA developers contributing to DOCA DMA itself.

**Language scope.** DOCA DMA ships as a C library with
`pkg-config` module name `doca-dma`. The shipped samples are
written in C. C and C++ consumers are the canonical case and the
worked examples in `TASKS.md` assume that path. Other-language
consumers (Rust, Go, Python, …) consume the same `*.so` through
FFI or language-specific bindings; the skill's contribution in
that case is to keep the lifecycle, capability-discovery,
permission, error-taxonomy, and path-selection guidance
language-neutral, and to route the agent to the public C ABI as
the authoritative surface that any wrapper will eventually call.

## When to load this skill

Load this skill when the user is doing hands-on DOCA DMA work,
in any language. Concretely:

- Initializing a `doca_dma` context on a `doca_dev` and
  configuring the memcpy task type via
  `doca_dma_task_memcpy_set_conf` before `doca_ctx_start()`.
- Setting up the source and destination `doca_mmap` regions for
  a memcpy, including the per-side permission flags
  (`DOCA_ACCESS_FLAG_LOCAL_READ_ONLY` on the source,
  `DOCA_ACCESS_FLAG_LOCAL_READ_WRITE` on the destination) and,
  for cross-peer copies, the `doca_mmap_export_*` step.
- Reading the device capability surface for DMA via the
  `doca_dma_cap_task_memcpy_*` query family
  (`_is_supported`, `_get_max_buf_size`,
  `_get_max_buf_list_len`) before sizing any buffer or assuming
  scatter-gather is available.
- Submitting `doca_dma_task_memcpy` tasks against a DOCA progress
  engine and reacting to per-task completion events.
- Choosing between DOCA DMA and an adjacent option (DOCA RDMA when
  the data has to cross the network, DOCA Comch fast-path for
  message-oriented producer/consumer flows, plain CPU memcpy when
  the copy is tiny and one-shot).
- Debugging a `DOCA_ERROR_*` returned from a DMA call (lifecycle
  vs. permission vs. capability vs. would-block) and the
  per-task completion status reported on the progress engine.
- Designing or extending non-C bindings (Rust, Go, Python, …)
  that wrap the DMA C ABI — for the lifecycle, permission, and
  capability rules the wrapper must honor.

Do **not** load this skill for general DOCA orientation, install
of DOCA itself, or non-DMA library questions. For those, use
[`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md).

## What this skill provides

This is a **thin loader**. The body keeps only the orientation
needed to pick the right next file. The substantive DMA-specific
material lives in two companion files:

- `CAPABILITIES.md` — what DMA can express on this version: the
  single `doca_dma_task_memcpy` task type and its scatter-gather
  buffer-list shape, the capability-query surface
  (`doca_dma_cap_task_memcpy_*`), the DMA error taxonomy (mapped
  onto the cross-library `DOCA_ERROR_*` set), the observability
  surface (per-task completion events on the progress engine),
  the source / destination mmap permission policy, and the
  path-selection rule against the adjacent libraries
  (RDMA / Comch / CPU memcpy).
- `TASKS.md` — step-by-step workflows for the six in-scope DMA
  verbs: `configure`, `build`, `modify`, `run`, `test`, `debug`.
  Plus a `Deferred task verbs` block that points out-of-scope
  questions at the right next skill.

The skill assumes a host or BlueField where DOCA is already
installed at the standard location and the user has the
privileges their public install profile expects. It does not
cover installing DOCA — that path goes through
[`doca-setup`](../../doca-setup/SKILL.md).

## What this skill deliberately does not ship

This skill is **agent guidance**, not a samples or templates
bundle. To keep the boundary clean, it deliberately does not
contain — and pull requests should not add:

- **Pre-written DOCA DMA application source code, in any
  language.** The verified DMA source code is the shipped C
  samples at `/opt/mellanox/doca/samples/doca_dma/<name>/` and
  the DMA Copy reference application reachable via
  [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md).
  The agent's job is to route the user to those files and
  prescribe a minimum-diff modification on them via the universal
  modify-a-sample workflow in
  [`doca-programming-guide`](../../doca-programming-guide/SKILL.md),
  layered with the DMA-specific overrides in
  [`TASKS.md ## modify`](TASKS.md#modify).
- **Standalone build manifests** (`meson.build`, `CMakeLists.txt`,
  `Cargo.toml`, …) parked inside the skill. The agent constructs
  the build manifest *in the user's project directory* against
  the user's installed DOCA, where `pkg-config --modversion
  doca-dma` is the source of truth.
- **A `samples/`, `bindings/`, or `reference/` subtree** of any
  kind. A mock or incomplete artifact in this skill's tree, even
  one labeled "reference", is misleading: users will read it as
  buildable.

## Loading order

1. Read this `SKILL.md` first to confirm the user's question is
   in scope.
2. **For the DMA capability surface, the memcpy task type, the
   capability-query rule, the source / destination permission
   matrix, the error taxonomy, observability, and the
   path-selection rule against RDMA / Comch / CPU memcpy, see
   [CAPABILITIES.md](CAPABILITIES.md).**
3. **For step-by-step workflows — configure, build, modify, run,
   test, debug — see [TASKS.md](TASKS.md).**

Both companion files cross-link to each other,
[`doca-version`](../../doca-version/SKILL.md) for the canonical
version-handling rules, and
[`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md)
whenever the right answer is "look it up in the public docs or
the installed package layout" rather than "DMA-specific
guidance".

## Related skills

- [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md) —
  the routing table for every public DOCA documentation source
  and the on-disk layout of an installed DOCA package. The DMA
  URL is `https://docs.nvidia.com/doca/sdk/DOCA-DMA/index.html`;
  the canonical reference application is *DMA Copy*.
- [`doca-setup`](../../doca-setup/SKILL.md) — env preparation,
  install verification, and the *I have no install yet* path
  with the public NGC DOCA container. This skill assumes its
  preconditions are satisfied.
- [`doca-version`](../../doca-version/SKILL.md) — canonical DOCA
  version-handling rules. This skill's `## Version
  compatibility` cross-links the four-way match rule + detection
  chain and adds at most one DMA-specific overlay rule.
- [`doca-structured-tools-contract`](../../doca-structured-tools-contract/SKILL.md) —
  the bundle's structured-tools precedence rule (detect / prefer
  / fall back / report). The Command appendix in
  [TASKS.md](TASKS.md) honors this contract.
- [`doca-programming-guide`](../../doca-programming-guide/SKILL.md) —
  general DOCA programming patterns shared by every library: the
  canonical `pkg-config` + meson build pattern, the universal
  modify-a-shipped-sample first-app workflow, the universal
  lifecycle, the cross-library `DOCA_ERROR_*` taxonomy, and the
  program-side debug order. This skill layers DMA specifics on
  top.
- [`doca-rdma`](../doca-rdma/SKILL.md) — the right library when
  the copy has to traverse the network. This skill's
  path-selection rule routes to RDMA when DMA is *not* the
  answer.
- [`doca-comch`](../doca-comch/SKILL.md) — the right library
  when the flow is producer / consumer messaging between a host
  and DPU process pair, rather than a raw mmap-to-mmap copy.
- [`doca-debug`](../../doca-debug/SKILL.md) — the cross-cutting
  debug ladder (install / version / build / link / runtime /
  program / driver). DMA-specific debug (lifecycle violations,
  permission mismatches, oversize-buffer rejections) overlays on
  top of that ladder.
