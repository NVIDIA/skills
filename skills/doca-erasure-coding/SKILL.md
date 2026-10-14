---
license: Apache-2.0
name: doca-erasure-coding
description: >
  Hardware-offloaded erasure coding with DOCA Erasure Coding
  (Reed-Solomon, doca_ec) on a BlueField DPU, ConnectX NIC, or host:
  generating parity for data blocks, rebuilding lost blocks from the
  survivors, and refreshing parity after a change. Trigger on "one data
  block changed, how do I refresh parity without re-encoding", "a disk
  failed and 2 parity blocks are gone, can I rebuild", "RAID-6 resilience
  across 12 disks", "my doca_ec_task_create returns NOT_PERMITTED", or "is
  this N+K layout still recoverable", even without naming DOCA Erasure
  Coding. Covers bringing up a doca_ec context, the create / recover /
  update tasks, matrix type / N / K / block size, doca_ec_cap_* queries
  before sizing, doca_mmap src/dst permissions, and DOCA_ERROR_* from
  doca_ec_task_*. Refuse and route elsewhere for non-Reed-Solomon codes
  (fountain / LDPC / raptor), pure-replication designs, network FEC, or
  other DOCA accelerator libraries (SHA / Compress / AES-GCM / DMA).
metadata:
  author: NVIDIA <doca@nvidia.com>
  kind: library
  tags:
    - doca
    - erasure-coding
    - reed-solomon
compatibility: >
  Requires DOCA SDK installed at /opt/mellanox/doca on Linux (Ubuntu
  22.04/24.04 or RHEL/SLES) with a BlueField DPU or ConnectX NIC
  attached. Reads the user's local install via `pkg-config
  doca-erasure-coding` and inspects
  /opt/mellanox/doca/{lib,include,samples,applications}.
  The validation steps require a live DOCA install.
---

# DOCA Erasure Coding

## Examples

- **"A disk failed; can I rebuild from parity?"** — this is in
  scope if the layout is Reed-Solomon N+K. Query `doca_ec_cap_*`
  before sizing recover.
- **"One data block changed; refresh parity without re-encoding."**
  — use the update task. Do not route this to replication or
  network FEC.

## Quick answer: capability check and sample choice

Before writing code, query the device: `doca_ec_cap_task_create_is_supported`,
`doca_ec_cap_task_update_is_supported` or `doca_ec_cap_task_recover_is_supported`
for the task you need, then `doca_ec_cap_get_max_block_size` and
`doca_ec_cap_get_max_buf_list_len` to size blocks. Start from the shipped sample
for that task under `/opt/mellanox/doca/samples/doca_erasure_coding/` (copy it;
never edit the installed one). Use the update task to refresh parity for a
changed block instead of re-encoding everything.

## Instructions

Follow Loading order below. Open the tasks companion for configure /
recover / debug. Open the capabilities companion for create /
recover / update and matrix policy. Do not open every companion file.

**Where to start:** This skill assumes DOCA is already installed and
the user is doing **hands-on erasure-coding work** on a BlueField /
ConnectX / host with DOCA. Open the tasks companion if the user
wants to *do* something (configure / build / modify / run / test /
debug); open the capabilities companion when the question
is *what can DOCA Erasure Coding express* on this version. If the
user has not installed DOCA yet, route to
[`doca-setup`](../../doca-setup/SKILL.md) first. If the user is
asking *"should I even use erasure coding here, or just replicate
the data?"*, the path-selection rule in
## Capabilities and modes
is the first stop — erasure coding is a **storage-resilience**
primitive (RAID-6 / distributed file system parity / object-storage
erasure-coded buckets), not a network primitive and not a
replication substitute.

**How to use this skill.** Read this file, then search the companion file
for the `## <name>` heading the question needs and read that whole section.
Answer completely from it, quoting only its symbols and commands. Do not
browse docs; use the entry points in `doca-public-knowledge-map`. Give
environment checks as commands for the user's DOCA host. For out-of-scope
requests, name the skill that owns them and stop.

## Example questions this skill answers well

The CLASSES of DOCA Erasure Coding questions this skill is built
to answer, each with one worked example. The agent should treat
the *class* as the load-bearing piece — the worked example is a
single instance.

- **"Should I offload this erasure coding to DOCA, or do it on the
  CPU — or just replicate the data instead?"** — worked example:
  *"I am writing a distributed object store that wants RAID-6-like
  resilience across 12 disks; is doca-erasure-coding the right
  primitive, or should I just keep three copies?"*. Answered by
  the path-selection table in
  ## Capabilities and modes
  + the *"when NOT to use doca-erasure-coding"* bullets in
  ## Safety policy.
- **"Does my device support the EC task type, block size, N+K
  layout, and matrix type I want?"** — worked example: *"is
  `doca_ec_task_create` on this BlueField, what is the maximum
  block size, what is the cap on N+K, and does the device
  advertise a Vandermonde Reed-Solomon matrix?"*. Answered by
  the per-task + matrix + size capability-query rule
  (`doca_ec_cap_task_create_is_supported`,
  `_task_recover_is_supported`, `_task_update_is_supported`,
  `_task_galois_mul_is_supported` — the 4th public task on the
  EC accelerator, `doca_ec_cap_get_max_block_size`,
  `doca_ec_cap_get_max_buf_list_len`, and per-variant
  `doca_ec_matrix_create()` constructor success — the public
  header does NOT ship a `doca_ec_cap_get_matrix_*` family) in
  ## Capabilities and modes
  + the discovery step in
  ## configure.
- **"One source block changed — do I have to recompute all the
  parity from scratch?"** — worked example: *"I have 8 data + 4
  parity blocks on disk; one data block was just rewritten. How
  do I express the parity update without re-running the full
  encode?"*. Answered by the `doca_ec_task_update` row in the
  task-type table in
  ## Capabilities and modes
  + the update-vs-re-encode rule in
  ## Safety policy
  + the modify-from-sample slot for switching between create and
  update tasks in
  ## modify.
- **"K of my N+K blocks went missing — how do I reconstruct
  them?"** — worked example: *"a disk failed; the 8 data blocks
  are intact but 2 of the 4 parity blocks are gone; can DOCA
  recover them?"*. Answered by the
  `doca_ec_task_recover` row in the task-type table in
  ## Capabilities and modes
  + the recover smoke (encode → drop a block → recover → bit-compare)
  in ## test.
- **"What permissions does the source / destination mmap need?"** —
  worked example: *"my `doca_ec_task_create` returns
  `DOCA_ERROR_NOT_PERMITTED`"*. Answered by the permission matrix
  in ## Safety policy
  + the mmap-set-permissions checklist in
  ## test.
- **"Is this DOCA Erasure Coding API available on my installed
  DOCA version?"** — worked example: *"is `doca_ec_task_update` in
  the DOCA I have installed, or do I have to recompute parity from
  scratch?"*. Answered by the version-compatibility overlay in
  ## Version compatibility,
  which cross-links the canonical detection chain in
  [`doca-version`](../../doca-version/SKILL.md) and adds the
  EC-specific *"discover per-task support + matrix type via cap
  query"* bullets.
- **"What does this `DOCA_ERROR_*` from an EC call mean and which
  layer caused it?"** — worked example: *"`DOCA_ERROR_INVALID_VALUE`
  on `doca_ec_task_create_allocate_init` with block_size = 4 MiB"*.
  Answered by the EC overlay on the cross-library taxonomy in
  ## Error taxonomy
  + the layered ladder in
  ## debug that escalates to
  [`doca-debug`](../../doca-debug/SKILL.md).

## Audience

This skill serves **external developers building applications that
consume the DOCA Erasure Coding library** — i.e., users whose code
calls `doca_ec_*` (directly in C/C++, or through FFI/bindings from
another language) to offload Reed-Solomon erasure coding onto a
BlueField DPU or ConnectX accelerator. It is *not* for NVIDIA
developers contributing to DOCA Erasure Coding itself.

The canonical fit is **distributed storage**: RAID-6-style block
layouts, distributed file system parity, object-storage
erasure-coded buckets, and any data-durability workload where the
N data + K redundancy block model lets the system tolerate K
simultaneous block losses without data loss. The skill keeps the
agent oriented to that domain: erasure coding is operationally
distinct from pure replication (which keeps M whole copies) and
from RAID-6 the disk layout (which is one specific instance of
the N=k, K=2 case); confusing them produces wrong recommendations.

**Language scope.** DOCA Erasure Coding ships as a C library with
`pkg-config` module name `doca-erasure-coding`. The shipped samples
are written in C. C and C++ consumers are the canonical case and
the worked examples in the tasks companion assume that path. Other-language
consumers (Rust, Go, Python, …) consume the same `*.so` through
FFI or language-specific bindings; the skill's contribution in
that case is to keep the lifecycle, capability-discovery,
permission, error-taxonomy, and create-vs-recover-vs-update
guidance language-neutral, and to route the agent to the public C
ABI as the authoritative surface that any wrapper will eventually
call.

## When to load this skill

Load this skill when the user is doing hands-on DOCA Erasure
Coding work, in any language. Concretely:

- Initializing a `doca_ec` context on a `doca_dev` and configuring
  at least one task type (`doca_ec_task_create`,
  `doca_ec_task_recover`, and/or `doca_ec_task_update`) before
  `doca_ctx_start()`.
- Choosing between the **create** task (encode N data blocks into
  K redundancy blocks), the **recover** task (reconstruct up to K
  missing blocks given the surviving subset), and the **update**
  task (incrementally update redundancy blocks when one source
  block changes — much cheaper than re-encoding from scratch).
- Picking the coding parameters: matrix type / coding scheme
  (typically Vandermonde or Cauchy for Reed-Solomon), N (number
  of data blocks), K (number of redundancy blocks), and block
  size (all blocks the same size).
- Setting permissions on `doca_mmap` correctly for the source
  buffers — data blocks for create, data + parity for recover —
  (`DOCA_ACCESS_FLAG_LOCAL_READ_ONLY` at minimum) and the
  destination buffers — redundancy blocks for create, recovered
  blocks for recover, updated parity for update —
  (`DOCA_ACCESS_FLAG_LOCAL_READ_WRITE`).
- Sizing blocks against
  `doca_ec_cap_get_max_block_size(devinfo)` and N+K against
  `doca_ec_cap_get_max_buf_list_len(devinfo, &max_buf_list_len)`.
- Checking which task types and matrix schemes this device's
  accelerator advertises via the
  `doca_ec_cap_task_*_is_supported` (including the 4th task
  `_task_galois_mul_is_supported`) and per-variant
  `doca_ec_matrix_create()` constructor success — the public
  header does NOT ship a `doca_ec_cap_get_matrix_*`
  families against the active `doca_devinfo`.
- Validating against a known-vector recover smoke (encode a small
  N + K layout, simulate one block loss, recover, bit-compare
  against the original block) before pushing bulk storage data
  through the accelerator.
- Recognising when **update** is the right task (single source
  block changed in an existing N + K layout — `doca_ec_task_update`
  is far cheaper than recomputing all K parity blocks via a fresh
  `doca_ec_task_create`).
- Debugging a `DOCA_ERROR_*` returned from an EC call (lifecycle
  vs. block-size vs. N+K vs. matrix-type vs. permission vs.
  queue pressure) and the task-completion event on the progress
  engine.
- Designing or extending non-C bindings (Rust, Go, Python, …)
  that wrap the EC C ABI — for the lifecycle, permission,
  capability, and create-vs-recover-vs-update rules the wrapper
  must honor.

Do **not** load this skill for general DOCA orientation, install
of DOCA itself, non-Reed-Solomon erasure codes (fountain codes,
LDPC — those belong on a CPU library, not the DOCA accelerator),
pure replication / mirroring designs (erasure coding is the wrong
primitive when one extra copy is the right answer), or other
DOCA libraries. For those, use
[`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md).

## What this skill provides

This is a **thin loader**. The body keeps only the orientation
needed to pick the right next file. The substantive
EC-specific material lives in two companion files:

- the capabilities companion — what DOCA Erasure Coding can express on
  this version: the three task types (create / recover / update,
  each independently capability-gated), the matrix-type + N + K +
  block-size configuration surface, the capability-query family
  (`doca_ec_cap_*` for task support, max block size, max N+K,
  supported matrix types), the EC error taxonomy (mapped onto
  the cross-library `DOCA_ERROR_*` set), the observability
  surface (per-task completion events on the progress engine),
  the safety policy that gates source / destination mmap
  permission decisions, and the path-selection rule (when to use
  doca-erasure-coding versus a CPU EC library versus pure
  replication versus RAID-6 on disk).
- the tasks companion — step-by-step workflows for the six in-scope EC
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

- **Pre-written DOCA EC application source code, in any
  language.** The verified EC source code is the shipped C
  samples at `/opt/mellanox/doca/samples/doca_erasure_coding/`.
  The agent's job is to route the user to those files and
  prescribe a minimum-diff modification on them via the universal
  modify-a-sample workflow in
  [`doca-programming-guide`](../../doca-programming-guide/SKILL.md),
  layered with the EC-specific overrides in
  ## modify.
- **Pre-computed parity tables for arbitrary N + K layouts.**
  The skill tells the agent to use a *known-vector* smoke
  (encode a tiny N + K layout from a fixed input, simulate
  losing one block, recover, bit-compare against the original)
  as the smoke before bulk; it does not ship a vector bank of
  its own.
- **Standalone build manifests** (`meson.build`,
  `CMakeLists.txt`, `Cargo.toml`, …) parked inside the skill.
  The agent constructs the build manifest *in the user's
  project directory* against the user's installed DOCA, where
  `pkg-config --modversion doca-erasure-coding` is the source of
  truth.
- **A `samples/`, `bindings/`, or `reference/` subtree** of any
  kind. A mock or incomplete artifact in this skill's tree, even
  one labeled "reference", is misleading: users will read it as
  buildable.

## Loading order

1. Read this skill first to confirm the user's question is in
   scope (storage-domain resilience — N data + K redundancy
   blocks — and not a network or replication question
   misclassified as erasure coding).
2. **For the EC capability matrix, the three task types
   (create / recover / update), the matrix-type + N + K +
   block-size configuration surface, the capability-query rules,
   permission matrix, error taxonomy, observability, and safety /
   path-selection policy, see the capabilities companion.**
3. **For step-by-step workflows — configure, build, modify, run,
   test, debug — see the tasks companion.**

Both companion files cross-link to each other,
[`doca-version`](../../doca-version/SKILL.md) for the canonical
version-handling rules, and
[`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md)
whenever the right answer is "look it up in the public docs or
the installed package layout" rather than "EC-specific guidance".

## Related skills

- [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md) —
  the routing table for every public DOCA documentation source
  and the on-disk layout of an installed DOCA package. The DOCA
  Erasure Coding page lives at
  <https://docs.nvidia.com/doca/sdk/DOCA-Erasure-Coding/index.html>;
  the install-tree layout (samples directory, header location)
  belongs to the public knowledge map, not to this skill.
- [`doca-setup`](../../doca-setup/SKILL.md) — env preparation,
  install verification, and the *I have no install yet* path
  with the public NGC DOCA container. This skill assumes its
  preconditions are satisfied.
- [`doca-version`](../../doca-version/SKILL.md) — canonical DOCA
  version-handling rules. This skill's
  `## Version compatibility`
  cross-links the four-way match rule and adds only the
  EC-specific *"discover per-task support + supported matrix
  types + max block size + max N+K via cap query"* overlay.
- [`doca-structured-tools-contract`](../../doca-structured-tools-contract/SKILL.md) —
  the bundle's structured-tools precedence rule (detect / prefer
  / fall back / report). The Command appendix in
  the tasks companion honors this contract.
- [`doca-programming-guide`](../../doca-programming-guide/SKILL.md) —
  general DOCA programming patterns shared by every library: the
  canonical `pkg-config` + meson build pattern, the universal
  modify-a-shipped-sample first-app workflow, the universal
  lifecycle, the cross-library `DOCA_ERROR_*` taxonomy, and the
  program-side debug order. This skill layers EC specifics on
  top.
- [`doca-debug`](../../doca-debug/SKILL.md) — the cross-cutting
  debug ladder (install / version / build / link / runtime /
  program / driver). EC-specific debug (task-not-supported,
  matrix-type-unsupported, block-size-over-max, N+K-over-max,
  recover-with-too-many-missing) overlays on top of that ladder.
