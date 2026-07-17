---
license: Apache-2.0
name: doca-sha
description: >
  Use this skill when the user is doing hands-on DOCA SHA programming —
  offloading SHA-1, SHA-256, or SHA-512 hashing onto a BlueField DPU or
  ConnectX accelerator, picking between one-shot `doca_sha_task_hash` and
  incremental `doca_sha_task_partial_hash`, querying `doca_sha_cap_*`
  for algorithm support and min destination / max source buffer sizes, setting
  source / destination `doca_mmap` permissions, or decoding DOCA_ERROR_*
  returns from the SHA API. Trigger even when the user does not explicitly
  mention "DOCA SHA" or "doca_sha_task" — typical implicit phrasings
  include "hash a multi-GiB file on the DPU", "offload SHA-256 to the
  BlueField", "streaming hash over chunks", "partial hash returns
  BAD_STATE", "destination buffer too small for digest", or "is SHA-512
  available on this card". Refuse and route elsewhere for general
  cryptographic-hash theory (collision resistance, SHA-3 selection), other
  DOCA crypto libraries (AES-GCM, Compress, DMA), or DOCA install / BFB
  bring-up — those belong to other skills.
metadata:
  kind: library
compatibility: >
  Requires DOCA SDK installed at /opt/mellanox/doca on Linux (Ubuntu
  22.04/24.04 or RHEL/SLES) with a BlueField DPU or ConnectX NIC attached.
  Reads the user's local install via `pkg-config doca-sha` and inspects
  /opt/mellanox/doca/{lib,include,samples,applications}.
---

# DOCA SHA

**Where to start:** This skill assumes DOCA is already installed and
the user is doing **hands-on SHA-acceleration work** on a BlueField
/ ConnectX / host with DOCA. Open [`TASKS.md`](TASKS.md) if the user
wants to *do* something (configure / build / modify / run / test /
debug); open [`CAPABILITIES.md`](CAPABILITIES.md) when the question
is *what can DOCA SHA express* on this version. If the user has not
installed DOCA yet, route to
[`doca-setup`](../../doca-setup/SKILL.md) first. If the user is
asking *"should I even use the accelerator for this hash?"*, the
path-selection rule in
[`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes)
is the first stop.

## Example questions this skill answers well

The CLASSES of DOCA SHA questions this skill is built to answer,
each with one worked example. The agent should treat the *class* as
the load-bearing piece — the worked example is a single instance.

- **"Should I offload this hash to DOCA SHA, or just compute it on
  the CPU?"** — worked example: *"I am verifying file integrity on
  a 4 GiB image; is doca-sha worth the setup vs OpenSSL on the
  CPU?"*. Answered by the path-selection table in
  [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes)
  + the *"when NOT to use doca-sha"* bullets in
  [`CAPABILITIES.md ## Safety policy`](CAPABILITIES.md#safety-policy).
- **"Does my device support the SHA algorithm I want?"** — worked
  example: *"is SHA-256 in the accelerator on this BlueField, and
  what is the minimum destination buffer size for it?"*. Answered
  by the algorithm + buffer-sizing capability-query rule
  (`doca_sha_cap_task_hash_get_supported(devinfo, algorithm)`
  for the one-shot path; `_task_partial_hash_get_supported(devinfo, algorithm)`
  for the partial-hash path; `doca_sha_cap_get_min_dst_buf_size`,
  `doca_sha_cap_get_max_src_buf_size`) in
  [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes)
  + the discovery step in
  [`TASKS.md ## configure`](TASKS.md#configure).
- **"How do I pick between the one-shot and the partial / incremental
  hash task?"** — worked example: *"my input is 1 GiB and the device
  cap says max source buffer is 64 MiB"*. Answered by the
  one-shot-vs-partial table in
  [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes)
  + the task-config workflow in
  [`TASKS.md ## modify`](TASKS.md#modify).
- **"What permissions does the source / destination mmap need?"** —
  worked example: *"my `doca_sha_task_hash` returns
  `DOCA_ERROR_NOT_PERMITTED`"*. Answered by the permission matrix
  in [`CAPABILITIES.md ## Safety policy`](CAPABILITIES.md#safety-policy)
  + the mmap-set-permissions checklist in
  [`TASKS.md ## test`](TASKS.md#test).
- **"Is this DOCA SHA API available on my installed DOCA version?"**
  — worked example: *"is `doca_sha_task_partial_hash` in the DOCA
  I have installed?"*. Answered by the version-compatibility
  overlay in
  [`CAPABILITIES.md ## Version compatibility`](CAPABILITIES.md#version-compatibility),
  which cross-links the canonical detection chain in
  [`doca-version`](../../doca-version/SKILL.md) and adds the
  SHA-specific *"discover, do not assume"* bullets.
- **"What does this `DOCA_ERROR_*` from a SHA call mean and which
  layer caused it?"** — worked example: *"`DOCA_ERROR_INVALID_VALUE`
  on `doca_sha_task_hash_alloc_init`"*. Answered by the SHA overlay
  on the cross-library taxonomy in
  [`CAPABILITIES.md ## Error taxonomy`](CAPABILITIES.md#error-taxonomy)
  + the layered ladder in
  [`TASKS.md ## debug`](TASKS.md#debug) that escalates to
  [`doca-debug`](../../doca-debug/SKILL.md).

## Audience

This skill serves **external developers building applications that
consume the DOCA SHA library** — i.e., users whose code calls
`doca_sha_*` (directly in C/C++, or through FFI/bindings from
another language) to offload SHA hashing onto a BlueField DPU or
ConnectX accelerator. It is *not* for NVIDIA developers contributing
to DOCA SHA itself.

**Language scope.** DOCA SHA ships as a C library with `pkg-config`
module name `doca-sha`. The shipped samples are written in C. C and
C++ consumers are the canonical case and the worked examples in
`TASKS.md` assume that path. Other-language consumers (Rust, Go,
Python, …) consume the same `*.so` through FFI or language-specific
bindings; the skill's contribution in that case is to keep the
lifecycle, capability-discovery, permission, error-taxonomy, and
one-shot-vs-partial guidance language-neutral, and to route the
agent to the public C ABI as the authoritative surface that any
wrapper will eventually call.

## When to load this skill

Load this skill when the user is doing hands-on DOCA SHA work, in
any language. Concretely:

- Initializing a `doca_sha` context on a `doca_dev` and configuring
  at least one task type (`doca_sha_task_hash` and/or
  `doca_sha_task_partial_hash`) before `doca_ctx_start()`.
- Choosing between the **one-shot** task (`doca_sha_task_hash` —
  input fits in a single source buffer, output digest lands in a
  single destination buffer) and the **partial / incremental** task
  (`doca_sha_task_partial_hash` — input streamed in chunks, finalized
  separately) for the user's data shape.
- Setting permissions on `doca_mmap` correctly for the source buffer
  (`DOCA_ACCESS_FLAG_LOCAL_READ_ONLY` at minimum) and the destination
  buffer (`DOCA_ACCESS_FLAG_LOCAL_READ_WRITE`).
- Sizing the destination buffer against
  `doca_sha_cap_get_min_dst_buf_size(devinfo, algorithm)` and
  the source buffer against
  `doca_sha_cap_get_max_src_buf_size(devinfo)`.
- Checking which SHA algorithm enums
  (`DOCA_SHA_ALGORITHM_SHA1`, `DOCA_SHA_ALGORITHM_SHA256`,
  `DOCA_SHA_ALGORITHM_SHA512`) the active device's accelerator
  advertises, via
  `doca_sha_cap_task_hash_get_supported(devinfo, algorithm)` and
  `doca_sha_cap_task_partial_hash_get_supported(devinfo, algorithm)` —
  both fold task-support and algorithm-support into one call.
- Validating a digest against a published test vector before pushing
  bulk input through the accelerator.
- Debugging a `DOCA_ERROR_*` returned from a SHA call (lifecycle vs.
  buffer-sizing vs. permission vs. unsupported-algorithm) and the
  task-completion event on the progress engine.
- Designing or extending non-C bindings (Rust, Go, Python, …) that
  wrap the SHA C ABI — for the lifecycle, permission, capability,
  and one-shot-vs-partial rules the wrapper must honor.

Do **not** load this skill for general DOCA orientation, install of
DOCA itself, non-SHA hashing libraries on CPU (use OpenSSL or
similar), or other DOCA libraries. For those, use
[`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md).

## What this skill provides

This is a **thin loader**. The body keeps only the orientation
needed to pick the right next file. The substantive SHA-specific
material lives in two companion files:

- `CAPABILITIES.md` — what DOCA SHA can express on this version:
  the two task types (one-shot hash and partial / incremental
  hash), the three algorithm enums, the capability-query surface
  (`doca_sha_cap_*` for algorithm support and buffer sizing), the
  SHA error taxonomy (mapped onto the cross-library `DOCA_ERROR_*`
  set), the observability surface (per-task completion events on
  the progress engine), the safety policy that gates source /
  destination mmap permission decisions, and the path-selection
  rule (when to use doca-sha versus a CPU hash or a different
  DOCA crypto library).
- `TASKS.md` — step-by-step workflows for the six in-scope SHA
  verbs: `configure`, `build`, `modify`, `run`, `test`, `debug`.
  Plus a `Deferred task verbs` block that points out-of-scope
  questions at the right next skill.

The skill assumes a host or BlueField where DOCA is already
installed at the standard location and the user has the privileges
their public install profile expects. It does not cover installing
DOCA — that path goes through
[`doca-setup`](../../doca-setup/SKILL.md).

## What this skill deliberately does not ship

This skill is **agent guidance**, not a samples or templates
bundle. To keep the boundary clean, it deliberately does not
contain — and pull requests should not add:

- **Pre-written DOCA SHA application source code, in any
  language.** The verified SHA source code is the shipped C samples
  at `/opt/mellanox/doca/samples/doca_sha/`, plus the File Integrity
  reference application linked from the public DOCA SHA guide. The
  agent's job is to route the user to those files and prescribe a
  minimum-diff modification on them via the universal
  modify-a-sample workflow in
  [`doca-programming-guide`](../../doca-programming-guide/SKILL.md),
  layered with the SHA-specific overrides in
  [`TASKS.md ## modify`](TASKS.md#modify).
- **Pre-computed digest tables for arbitrary inputs.** The skill
  tells the agent to use a *published* test vector (e.g. the NIST
  SHA test vectors for the empty string, "abc", and the
  million-`a` input) as the known-vector smoke; it does not ship a
  vector bank of its own.
- **Standalone build manifests** (`meson.build`, `CMakeLists.txt`,
  `Cargo.toml`, …) parked inside the skill. The agent constructs
  the build manifest *in the user's project directory* against the
  user's installed DOCA, where `pkg-config --modversion doca-sha`
  is the source of truth.
- **A `samples/`, `bindings/`, or `reference/` subtree** of any
  kind. A mock or incomplete artifact in this skill's tree, even
  one labeled "reference", is misleading: users will read it as
  buildable.

## Loading order

1. Read this `SKILL.md` first to confirm the user's question is in
   scope.
2. **For the SHA capability matrix, algorithm enums, one-shot vs
   partial task split, capability-query rules, permission matrix,
   error taxonomy, observability, and safety / path-selection
   policy, see [CAPABILITIES.md](CAPABILITIES.md).**
3. **For step-by-step workflows — configure, build, modify, run,
   test, debug — see [TASKS.md](TASKS.md).**

Both companion files cross-link to each other,
[`doca-version`](../../doca-version/SKILL.md) for the canonical
version-handling rules, and
[`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md)
whenever the right answer is "look it up in the public docs or the
installed package layout" rather than "SHA-specific guidance".

## Related skills

- [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md) —
  the routing table for every public DOCA documentation source and
  the on-disk layout of an installed DOCA package. The DOCA SHA
  page lives at `docs.nvidia.com/doca/sdk/DOCA-SHA/`; the File
  Integrity reference application is the canonical worked example.
- [`doca-setup`](../../doca-setup/SKILL.md) — env preparation,
  install verification, and the *I have no install yet* path with
  the public NGC DOCA container. This skill assumes its
  preconditions are satisfied.
- [`doca-version`](../../doca-version/SKILL.md) — canonical DOCA
  version-handling rules. This skill's
  [`## Version compatibility`](CAPABILITIES.md#version-compatibility)
  cross-links the four-way match rule and adds only the SHA-specific
  *"discover algorithms + buffer sizes via cap query"* overlay.
- [`doca-structured-tools-contract`](../../doca-structured-tools-contract/SKILL.md) —
  the bundle's structured-tools precedence rule (detect / prefer
  / fall back / report). The Command appendix in
  [TASKS.md](TASKS.md) honors this contract.
- [`doca-programming-guide`](../../doca-programming-guide/SKILL.md) —
  general DOCA programming patterns shared by every library: the
  canonical `pkg-config` + meson build pattern, the universal
  modify-a-shipped-sample first-app workflow, the universal
  lifecycle, the cross-library `DOCA_ERROR_*` taxonomy, and the
  program-side debug order. This skill layers SHA specifics on
  top.
- [`doca-debug`](../../doca-debug/SKILL.md) — the cross-cutting
  debug ladder (install / version / build / link / runtime /
  program / driver). SHA-specific debug (algorithm-not-supported,
  destination-buffer-too-small, partial-hash-out-of-order) overlays
  on top of that ladder.
