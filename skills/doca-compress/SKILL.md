---
license: Apache-2.0
name: doca-compress
description: >
  Use this skill for hands-on DOCA Compress programming on a
  BlueField DPU, ConnectX NIC, or host with DOCA — enabling
  compress-deflate, decompress-deflate, decompress-lz4-stream,
  or decompress-lz4-block tasks on a doca_compress context
  (the hardware supports DEFLATE both directions plus LZ4
  decompress; LZ4 encode is NOT supported), sizing source /
  destination doca_buf against the per-task cap query, setting
  mmap permissions, deciding offload vs CPU zlib / zstd,
  validating with a round-trip smoke, or debugging
  DOCA_ERROR_* from a Compress call. Trigger on phrasings
  like "offload this gzip", "decompress incoming network
  data", "compress task returns INVALID_VALUE on alloc_init",
  "submitted a task but no completion arrives", or "decompress
  LZ4 on the BlueField." Refuse and route elsewhere for
  non-DEFLATE / non-LZ4 algorithms (zstd / Snappy / brotli),
  LZ4 encode (route to a CPU LZ4 library), pure mmap-to-mmap
  copies (doca-dma), or DOCA Core lifecycle internals.
metadata:
  kind: library
compatibility: >
  Requires DOCA SDK installed at /opt/mellanox/doca on Linux
  (Ubuntu 22.04/24.04 or RHEL/SLES) with a BlueField DPU or
  ConnectX NIC attached. Reads the user's local install via
  `pkg-config doca-compress` and inspects
  /opt/mellanox/doca/{lib,include,samples,applications}.
---

# DOCA Compress

**Where to start:** This skill assumes DOCA is already installed and
the user is doing **hands-on Compress work** (bulk DEFLATE
compression or decompression) on a BlueField / ConnectX / host
with DOCA. Open [`TASKS.md`](TASKS.md) if the user wants to *do*
something (configure / build / modify / run / test / debug); open
[`CAPABILITIES.md`](CAPABILITIES.md) when the question is *what can
DOCA Compress express* on this version. If the user has not
installed DOCA yet, route to
[`doca-setup`](../../doca-setup/SKILL.md) first. If the user is
asking *"should I even offload this compression to the
accelerator?"*, the size-threshold path-selection rule in
[`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes)
is the first stop — bulk compress is the canonical fit, tiny
one-shot is not.

## Example questions this skill answers well

The CLASSES of DOCA Compress questions this skill is built to
answer, each with one worked example. The agent should treat the
*class* as the load-bearing piece — the worked example is a single
instance.

- **"Should I offload this compression to DOCA Compress, or just
  zlib/zstd on the CPU?"** — worked example: *"I am compressing a
  4 MiB log buffer before writing it to storage; is doca-compress
  worth the setup vs zlib on the CPU?"*. Answered by the
  size-threshold path-selection rule in
  [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes)
  + the *"when NOT to use doca-compress"* bullets in
  [`CAPABILITIES.md ## Safety policy`](CAPABILITIES.md#safety-policy).
- **"Does my device support the compress (or decompress) task I
  want, and how big a buffer can it move per submission?"** —
  worked example: *"is `doca_compress_task_compress_deflate` on
  this BlueField, and what is the max source size per task?"*.
  Answered by the per-task capability-query rule
  (`doca_compress_cap_task_compress_deflate_is_supported`,
  `_decompress_deflate_is_supported`, the matching
  `_get_max_buf_size` queries) in
  [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes)
  + the discovery step in
  [`TASKS.md ## configure`](TASKS.md#configure).
- **"I just need to decompress incoming network data — is that a
  valid standalone use of doca-compress?"** — worked example:
  *"my client receives DEFLATE-compressed payloads and never
  produces any compressed output of its own"*. Answered by the
  *"decompression-only"* note in
  [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes)
  task-type table + the per-task configuration matrix in
  [`TASKS.md ## configure`](TASKS.md#configure) step 5.
- **"What permissions does the source / destination mmap need?"** —
  worked example: *"my `doca_compress_task_compress_deflate`
  returns `DOCA_ERROR_NOT_PERMITTED`"*. Answered by the permission
  matrix in
  [`CAPABILITIES.md ## Safety policy`](CAPABILITIES.md#safety-policy)
  + the mmap-set-permissions checklist in
  [`TASKS.md ## test`](TASKS.md#test).
- **"Is this DOCA Compress API available on my installed DOCA
  version?"** — worked example: *"is
  `doca_compress_task_decompress_deflate` in the DOCA I have
  installed?"*. Answered by the version-compatibility overlay in
  [`CAPABILITIES.md ## Version compatibility`](CAPABILITIES.md#version-compatibility),
  which cross-links the canonical detection chain in
  [`doca-version`](../../doca-version/SKILL.md) and adds the
  Compress-specific *"discover per-task support, do not assume"*
  bullets.
- **"What does this `DOCA_ERROR_*` from a Compress call mean and
  which layer caused it?"** — worked example: *"`DOCA_ERROR_INVALID_VALUE`
  on `doca_compress_task_compress_deflate_alloc_init`"*. Answered
  by the Compress overlay on the cross-library taxonomy in
  [`CAPABILITIES.md ## Error taxonomy`](CAPABILITIES.md#error-taxonomy)
  + the layered ladder in
  [`TASKS.md ## debug`](TASKS.md#debug) that escalates to
  [`doca-debug`](../../doca-debug/SKILL.md).

## Audience

This skill serves **external developers building applications that
consume the DOCA Compress library** — i.e., users whose code calls
`doca_compress_*` (directly in C/C++, or through FFI/bindings from
another language) to offload bulk DEFLATE compression or
decompression onto a BlueField DPU or ConnectX accelerator. It is
*not* for NVIDIA developers contributing to DOCA Compress itself.

**Language scope.** DOCA Compress ships as a C library with
`pkg-config` module name `doca-compress`. The shipped samples are
written in C. C and C++ consumers are the canonical case and the
worked examples in `TASKS.md` assume that path. Other-language
consumers (Rust, Go, Python, …) consume the same `*.so` through
FFI or language-specific bindings; the skill's contribution in
that case is to keep the lifecycle, capability-discovery,
permission, error-taxonomy, and compress-vs-decompress guidance
language-neutral, and to route the agent to the public C ABI as
the authoritative surface that any wrapper will eventually call.

## When to load this skill

Load this skill when the user is doing hands-on DOCA Compress
work, in any language. Concretely:

- Initializing a `doca_compress` context on a `doca_dev` and
  configuring at least one task type
  (`doca_compress_task_compress_deflate` and/or
  `doca_compress_task_decompress_deflate`) before
  `doca_ctx_start()`.
- Choosing between the **compress** task and the **decompress**
  task for the user's data flow. The two task types are
  independent: a consumer that only decompresses inbound DEFLATE
  payloads is a valid shape and does not need to enable the
  compress task.
- Setting permissions on `doca_mmap` correctly for the source
  buffer (`DOCA_ACCESS_FLAG_LOCAL_READ_ONLY` at minimum) and the
  destination buffer (`DOCA_ACCESS_FLAG_LOCAL_READ_WRITE`).
- Sizing the source buffer against the per-task
  `doca_compress_cap_task_*_get_max_buf_size(devinfo)` ceiling
  and the destination buffer against the worst-case output size
  the algorithm can produce on this input.
- Checking which task types this device's accelerator advertises
  via `doca_compress_cap_task_compress_deflate_is_supported` and
  `doca_compress_cap_task_decompress_deflate_is_supported`
  against the active `doca_devinfo`.
- Validating against a round-trip (compress → decompress, or
  decompress → compare against a known DEFLATE-encoded fixture)
  before pushing bulk input through the accelerator.
- Deciding *whether to offload at all* — the size-threshold rule
  in [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes)
  says doca-compress is the right answer only for bulk inputs
  (rule of thumb: ≥ a few KiB); below that, CPU compression beats
  the DMA-to-accelerator round-trip.
- Debugging a `DOCA_ERROR_*` returned from a Compress call
  (lifecycle vs. buffer-sizing vs. permission vs.
  unsupported-task) and the task-completion event on the progress
  engine.
- Designing or extending non-C bindings (Rust, Go, Python, …)
  that wrap the Compress C ABI — for the lifecycle, permission,
  capability, and compress-vs-decompress rules the wrapper must
  honor.

Do **not** load this skill for general DOCA orientation, install
of DOCA itself, non-DEFLATE compression libraries on CPU (use
zlib / zstd / similar), or other DOCA libraries. For those, use
[`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md).

## What this skill provides

This is a **thin loader**. The body keeps only the orientation
needed to pick the right next file. The substantive
Compress-specific material lives in two companion files:

- `CAPABILITIES.md` — what DOCA Compress can express on this
  version: the two task types (compress-deflate and
  decompress-deflate, each a valid standalone shape), the
  per-task capability-query surface (`doca_compress_cap_*` for
  task support and per-submission max buffer size), the Compress
  error taxonomy (mapped onto the cross-library `DOCA_ERROR_*`
  set), the observability surface (per-task completion events on
  the progress engine), the safety policy that gates source /
  destination mmap permission decisions, and the size-threshold
  path-selection rule (when to use doca-compress versus CPU
  compression or `doca-dma` for the no-compression copy case).
- `TASKS.md` — step-by-step workflows for the six in-scope
  Compress verbs: `configure`, `build`, `modify`, `run`, `test`,
  `debug`. Plus a `## rollback` overlay (Compress-specific
  five-step teardown that drains in-flight tasks before
  `doca_ctx_stop`, unregisters mmap regions in reverse-register
  order, and re-verifies the device path with the round-trip
  smoke) and the 5-phase universal debug-loop instantiation
  appended to `## debug`. Plus a `Deferred task verbs` block
  that points out-of-scope questions at the right next skill.

The skill assumes a host or BlueField where DOCA is already
installed at the standard location and the user has the
privileges their public install profile expects. It does not
cover installing DOCA — that path goes through
[`doca-setup`](../../doca-setup/SKILL.md).

## What this skill deliberately does not ship

This skill is **agent guidance**, not a samples or templates
bundle. To keep the boundary clean, it deliberately does not
contain — and pull requests should not add:

- **Pre-written DOCA Compress application source code, in any
  language.** The verified Compress source code is the shipped C
  samples at `/opt/mellanox/doca/samples/doca_compress/`, plus
  the File Compression reference application linked from the
  public DOCA Compress guide. The agent's job is to route the
  user to those files and prescribe a minimum-diff modification
  on them via the universal modify-a-sample workflow in
  [`doca-programming-guide`](../../doca-programming-guide/SKILL.md),
  layered with the Compress-specific overrides in
  [`TASKS.md ## modify`](TASKS.md#modify).
- **Pre-encoded DEFLATE fixtures for arbitrary inputs.** The
  skill tells the agent to use a *round-trip* validation
  (compress → decompress and compare to the original, or
  decompress a known DEFLATE blob produced by a published CPU
  encoder like `zlib`) as the known-good smoke; it does not ship
  a fixture bank of its own.
- **Standalone build manifests** (`meson.build`, `CMakeLists.txt`,
  `Cargo.toml`, …) parked inside the skill. The agent constructs
  the build manifest *in the user's project directory* against
  the user's installed DOCA, where `pkg-config --modversion
  doca-compress` is the source of truth.
- **A `samples/`, `bindings/`, or `reference/` subtree** of any
  kind. A mock or incomplete artifact in this skill's tree, even
  one labeled "reference", is misleading: users will read it as
  buildable.

## Loading order

1. Read this `SKILL.md` first to confirm the user's question is in
   scope.
2. **For the Compress capability matrix, the compress / decompress
   task split (including decompression-only as a standalone
   shape), per-task capability-query rules, permission matrix,
   error taxonomy, observability, and safety / path-selection
   policy, see [CAPABILITIES.md](CAPABILITIES.md).**
3. **For step-by-step workflows — configure, build, modify, run,
   test, debug — see [TASKS.md](TASKS.md).**

Both companion files cross-link to each other,
[`doca-version`](../../doca-version/SKILL.md) for the canonical
version-handling rules, and
[`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md)
whenever the right answer is "look it up in the public docs or
the installed package layout" rather than "Compress-specific
guidance".

## Related skills

- [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md) —
  the routing table for every public DOCA documentation source
  and the on-disk layout of an installed DOCA package. The DOCA
  Compress page lives at
  `docs.nvidia.com/doca/sdk/DOCA-Compress/`; the File Compression
  reference application is the canonical worked example.
- [`doca-setup`](../../doca-setup/SKILL.md) — env preparation,
  install verification, and the *I have no install yet* path
  with the public NGC DOCA container. This skill assumes its
  preconditions are satisfied.
- [`doca-version`](../../doca-version/SKILL.md) — canonical DOCA
  version-handling rules. This skill's
  [`## Version compatibility`](CAPABILITIES.md#version-compatibility)
  cross-links the four-way match rule and adds only the
  Compress-specific *"discover per-task support + per-task max
  buffer size via cap query"* overlay.
- [`doca-structured-tools-contract`](../../doca-structured-tools-contract/SKILL.md) —
  the bundle's structured-tools precedence rule (detect / prefer
  / fall back / report). The Command appendix in
  [TASKS.md](TASKS.md) honors this contract.
- [`doca-programming-guide`](../../doca-programming-guide/SKILL.md) —
  general DOCA programming patterns shared by every library: the
  canonical `pkg-config` + meson build pattern, the universal
  modify-a-shipped-sample first-app workflow, the universal
  lifecycle, the cross-library `DOCA_ERROR_*` taxonomy, and the
  program-side debug order. This skill layers Compress specifics
  on top.
- [`doca-dma`](../doca-dma/SKILL.md) — the right library when
  the data flow is a pure mmap-to-mmap copy with no compression
  required. This skill's path-selection rule routes to DMA when
  Compress is *not* the answer (e.g. the user only wants to move
  bytes, not encode them).
- [`doca-debug`](../../doca-debug/SKILL.md) — the cross-cutting
  debug ladder (install / version / build / link / runtime /
  program / driver). Compress-specific debug (task-not-supported,
  source-buffer-too-large, destination-buffer-too-small) overlays
  on top of that ladder.
