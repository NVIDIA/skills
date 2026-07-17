# DOCA Erasure Coding capabilities, version overlay, errors, observability, safety

**Where to start:** Pick the H2 anchor that matches your question
(capabilities / version / errors / observability / safety) and
read that section end-to-end. The tables in each section are the
load-bearing content; the prose around them is interpretation.

Read this file when the loader sent you here from
[SKILL.md](SKILL.md). For the *how* of executing each pattern
(the verbs `configure / build / modify / run / test / debug`),
jump to [TASKS.md](TASKS.md). For the canonical DOCA
version-handling rules that this skill layers an EC overlay on
top of, see [`doca-version`](../../doca-version/SKILL.md).

## Pattern overview

Every DOCA Erasure Coding question this skill teaches resolves
into one of FIVE patterns. The patterns are CLASSES — they apply
across every EC release and every BlueField / ConnectX device the
accelerator runs on.

| Pattern | When it applies (class shape) | Where the substance lives |
| --- | --- | --- |
| 1. Decide whether to offload (and whether EC is even the right primitive) | Compare doca-erasure-coding against a CPU Reed-Solomon library on the same workload; before that, confirm EC (N data + K redundancy block layout, tolerates K simultaneous losses) is the right primitive at all versus pure replication (M copies, much simpler) or RAID-6 on disk (one specific instance of EC) | [`## Capabilities and modes`](#capabilities-and-modes) path-selection table + [`## Safety policy`](#safety-policy) *"when NOT to use doca-erasure-coding"* bullets |
| 2. Pick the task type(s) to enable | Create (`doca_ec_task_create`) for encoding N data → K redundancy blocks; recover (`doca_ec_task_recover`) for reconstructing up to K missing blocks from the surviving subset; update (`doca_ec_task_update`) for incrementally updating K redundancy blocks when ONE source block changes — far cheaper than re-encoding from scratch; galois_mul (`doca_ec_task_galois_mul`) for the matrix-builder fast path on the EC accelerator | [`## Capabilities and modes`](#capabilities-and-modes) task-type table + [TASKS.md ## modify](TASKS.md#modify) |
| 3. Discover capabilities | Query `doca_ec_cap_task_*_is_supported` (create / recover / update / galois_mul — note `galois_mul` is a real 4th task, not just an internal helper), `_get_max_block_size`, and `_get_max_buf_list_len` against the active `doca_devinfo` BEFORE choosing N + K, sizing blocks, or picking a matrix variant. Matrix-variant discovery is done by attempting `doca_ec_matrix_create()` / `doca_ec_matrix_create_from_raw()` against the active device for each variant the user wants to use — there is no `doca_ec_cap_get_matrix_*` family in the public header. | [`## Capabilities and modes`](#capabilities-and-modes) capability-query rule + [TASKS.md ## configure](TASKS.md#configure) step 2 |
| 4. Honor source / destination permissions | Source mmaps (data blocks for create; data + parity blocks for recover) = `DOCA_ACCESS_FLAG_LOCAL_READ_ONLY` at minimum; destination mmaps (redundancy blocks for create; recovered blocks for recover; updated parity for update) = `DOCA_ACCESS_FLAG_LOCAL_READ_WRITE`; both must be set BEFORE the first task submission | [`## Safety policy`](#safety-policy) permission matrix + [TASKS.md ## test](TASKS.md#test) |
| 5. Diagnose an EC error | Map symptom (`DOCA_ERROR_BAD_STATE`, `_INVALID_VALUE`, `_NOT_SUPPORTED`, `_NOT_PERMITTED`, `_AGAIN`, `_DRIVER`) to root cause — lifecycle, block-size, N+K, matrix-type, permission, queue pressure, or driver layer — without leaving the EC layer prematurely | [`## Error taxonomy`](#error-taxonomy) + [TASKS.md ## debug](TASKS.md#debug) |

Two cross-cutting rules that apply to *every* pattern above:

- **Discover the version-installed surface, do not assume.** Every
  pattern above gates on
  `pkg-config --modversion doca-erasure-coding` and on the
  `doca_ec_cap_*` capability queries against the active
  `doca_devinfo`. Quoting a max block size, a max N+K, a matrix
  type as universally available, or asserting any of the three
  task types is present without checking is the most common
  hallucination failure mode.
- **Validate against a known-vector recover smoke BEFORE bulk.**
  The cheapest way to confirm the configured EC context produces
  correct parity is to encode a small fixed N + K layout from a
  known input, drop one block (e.g. write zeroes over it),
  submit a recover task with the same matrix parameters, and
  bit-compare the recovered block against the original. This
  catches matrix-type mis-selection, N + K mismatch, block-size
  inconsistencies, and source / destination wiring bugs before
  any user storage data flows through the accelerator.

## Capabilities and modes

DOCA Erasure Coding is a **DOCA Core Context**. Every `doca_ec`
instance follows the universal `cfg-create → cfg-set-* → init →
start → use → stop → destroy` lifecycle (see
[`doca-programming-guide CAPABILITIES.md ## Capabilities and modes`](../../doca-programming-guide/CAPABILITIES.md#capabilities-and-modes)).
On top of that lifecycle, DOCA Erasure Coding layers its own task
model with three independent task types and a matrix-type +
N + K + block-size configuration surface.

**Domain reminder — EC is a STORAGE primitive.** Erasure coding
encodes N original data blocks into K redundancy (parity) blocks,
producing N + K total blocks; the layout tolerates ANY K
simultaneous block losses without data loss. Canonical homes:
RAID-6 (which is one specific N + K = 2 instance), distributed
file system parity (Ceph, Lustre's PFL with parity, HDFS EC),
object-storage erasure-coded buckets, and any data-durability
workload where storing M whole copies (pure replication) is
wasteful relative to N + K coded fragments. The agent should NOT
recommend doca-erasure-coding for network workloads, control
plane, or any non-storage problem — those have their own DOCA
libraries.

**The three task types.** DOCA Erasure Coding exposes three task
types, each of which is independently enabled and independently
capability-gated. The agent must call the matching capability
query before assuming any task type is available on the user's
device.

| Task type | Class shape | Notes |
| --- | --- | --- |
| `doca_ec_task_create` (encode) | Take N source `doca_buf` data blocks → produce K destination `doca_buf` redundancy blocks, under the configured matrix type, N, K, and block size | Asynchronous; completion arrives via `doca_pe_progress`. All N + K blocks must share the configured block size; mismatched block sizes within a single task return `DOCA_ERROR_INVALID_VALUE` at submit. Source N + K must satisfy `N + K ≤ doca_ec_cap_get_max_buf_list_len(devinfo, &max)`; block size must satisfy `block_size ≤ doca_ec_cap_get_max_block_size(devinfo)`. |
| `doca_ec_task_recover` (reconstruct) | Take the surviving subset of an existing N + K layout (any (N + K − K′) blocks where K′ ≤ K are missing) plus the recovery descriptor → reconstruct the missing blocks in destination `doca_buf`s | Same matrix-type + N + K + block-size constraints as create. The recover descriptor identifies which blocks are missing. If more than K blocks are missing, the layout is unrecoverable and the agent must NOT invent a recovery; the right answer is to restore from another replica or accept data loss. |
| `doca_ec_task_update` (incremental parity update) | When ONE source data block changes in an existing N + K layout, update the K redundancy blocks without recomputing them from scratch from all N data blocks | Far cheaper than re-encoding via `doca_ec_task_create` whenever a single source block was edited. The agent must teach this as the first-class path for single-block edits and route any "I changed one block, how do I refresh parity?" question here, not back to create. |

**Path selection — create vs recover vs update.** All three task
types can coexist on the same context; enable at least one
before `doca_ctx_start()`. The right shape depends on the user's
intended *operation*, not on convenience:

| Path | What it is | Right shape for | Wrong shape for |
| --- | --- | --- | --- |
| Create-only | Enable `doca_ec_task_create`; leave recover / update disabled | A one-shot or write-only producer: encode N data → K parity, write all N + K to storage, done. Bulk encoding of large datasets is the canonical fit | A consumer that ever needs to reconstruct missing blocks (must enable recover) or update parity in place (must enable update) |
| Recover-only | Enable `doca_ec_task_recover`; leave create / update disabled | A reconstruction worker: given the surviving subset of an existing N + K layout, rebuild the missing blocks. Disk-failure / node-failure rebuild workers fit here | A producer that emits parity (must enable create); an in-place parity updater (must enable update) |
| Update-only | Enable `doca_ec_task_update`; leave create / recover disabled | An in-place storage writer that already has a valid N + K layout on disk and edits one data block at a time — update K parity blocks per edit. This is **the right answer for single-block changes; recomputing all parity via create is wasteful** | A first-time encoder (must enable create); a reconstruction worker (must enable recover) |
| Combined | Enable two or three of the tasks on the same context | A full storage stack that encodes, updates in place, and rebuilds on failure — all on one EC context, one device | Single-purpose pipelines — keep the unused tasks disabled to keep the configuration surface small |

**Update vs full re-encode — the agent rule.** The most common
mistake the agent should catch: a user describes *"I have an
existing N + K layout on disk and one data block was just
rewritten. How do I refresh the parity?"* and the wrong answer is
*"re-encode all N data blocks into K fresh parity via
`doca_ec_task_create`"*. The right answer is
`doca_ec_task_update` — it is purpose-built for that case and is
algebraically much cheaper (it touches only the delta between
the old and new source block, not all N source blocks). The
agent should teach this as a first-class path; mis-routing here
wastes accelerator cycles linear in N.

**Path-selection — when to use DOCA Erasure Coding at all.** The
agent's rule:

- **Use doca-erasure-coding when** the input is a bulk storage
  workload that benefits from RAID-6-style resilience: a
  distributed file system or object store that wants to tolerate
  K block failures without keeping K + 1 whole copies, a backup
  system that wants efficient durability across many disks, an
  archival system encoding cold data into N + K coded fragments.
  Bulk encode of large datasets is the canonical fit; the
  accelerator amortizes the matrix arithmetic over many blocks.
- **Do NOT use doca-erasure-coding when** the right answer is
  pure replication (e.g. a system with only 3 disks where
  three copies are simpler and the wire cost is acceptable —
  EC's reconstruction CPU is wasted) — replication and EC are
  orthogonal, not interchangeable; when the user needs a
  non-Reed-Solomon scheme (fountain codes / LDPC / raptor codes
  — those belong on a CPU library, the DOCA Erasure Coding
  accelerator commits to Reed-Solomon); when the dataset is
  small enough that CPU EC finishes before the
  DMA-to-accelerator round-trip even completes (a one-shot
  encode of a few KiB); or when the user is asking about
  network durability (FEC on the wire — different domain,
  different libraries).
- **Do NOT confuse EC with RAID-6.** RAID-6 is one specific
  instance of erasure coding (N data blocks + 2 parity blocks
  across N + 2 disks, recoverable from any 2 disk failures);
  doca-erasure-coding generalizes that to arbitrary N + K. A
  user asking *"I want RAID-6"* on N + 2 disks is asking for an
  EC layout with K = 2; teach the general N + K vocabulary and
  let them pick K to match their durability target.

**Coding parameters at create time.** Before `doca_ctx_start()`,
the user must commit four parameters that frame every subsequent
task on the context:

| Parameter | What it is | Constraint | How the agent verifies |
| --- | --- | --- | --- |
| Matrix type / coding scheme | The Reed-Solomon variant the device uses to derive parity. `enum doca_ec_matrix_type` defines exactly two: `DOCA_EC_MATRIX_TYPE_CAUCHY` and `DOCA_EC_MATRIX_TYPE_VANDERMONDE`. (`doca_ec_matrix_create_from_raw()` is an *alternative constructor* that builds a matrix from a caller-supplied raw buffer — it is NOT a third matrix type.) Different matrix types trade off encode vs recover cost and have different numerical properties under specific N + K choices | Must be one the device's accelerator advertises | Attempt `doca_ec_matrix_create()` (or `_from_raw()`) against the active `doca_dev` for the variant the user wants; constructor failure = "not supported on this device." The public header does NOT ship a `doca_ec_cap_get_matrix_*` family. |
| N (data blocks) | The number of original data blocks fed into create / referenced by recover and update | `N + K ≤ doca_ec_cap_get_max_buf_list_len(devinfo, &max)`; N ≥ 1 | Compute against `doca_ec_cap_get_max_buf_list_len(devinfo, &max)` |
| K (redundancy blocks) | The number of parity blocks produced by create / required surviving for recover / refreshed by update. The layout tolerates any K simultaneous block losses | `N + K ≤ doca_ec_cap_get_max_buf_list_len(devinfo, &max)`; K ≥ 1 | Same query as N |
| Block size | The size of every block (data and redundancy). ALL blocks in a single task — source and destination — must share this size | `block_size ≤ doca_ec_cap_get_max_block_size(devinfo)`; mismatched block sizes inside one task fail with `DOCA_ERROR_INVALID_VALUE` at submit | Run `doca_ec_cap_get_max_block_size(devinfo)` and confirm every source / destination `doca_buf` length equals the configured block size |

**Capability discovery — the only rule.** Before enabling any
task, picking a matrix type, or sizing N, K, or blocks, call the
matching `doca_ec_cap_*` query against the active `doca_devinfo`:

| Capability | Query | Why the agent must ask |
| --- | --- | --- |
| Create task supported | `doca_ec_cap_task_create_is_supported(devinfo)` | Create is the baseline encode task; if false, the device has no EC accelerator in encode-capable mode and the user must fall back to CPU EC |
| Recover task supported | `doca_ec_cap_task_recover_is_supported(devinfo)` | Recover is independently capability-gated; agent must not silently assume "create is on => recover is on" |
| Update task supported | `doca_ec_cap_task_update_is_supported(devinfo)` | Update is independently capability-gated. If a user needs the cheap single-block-change path and update is false, they must either upgrade the device / DOCA install or fall back to re-encoding via create |
| Galois-multiply task supported | `doca_ec_cap_task_galois_mul_is_supported(devinfo)` | The 4th public task on the EC accelerator — used by the matrix-builder fast path. Independently capability-gated; the bundle previously omitted this row. |
| Maximum block size | `doca_ec_cap_get_max_block_size(devinfo)` | Hardware-bound ceiling on per-block size; block sizes larger than this return `DOCA_ERROR_INVALID_VALUE` at submit |
| Maximum total blocks in one task (`N + K`) | `doca_ec_cap_get_max_buf_list_len(devinfo, &max_buf_list_len)` | Hardware-bound ceiling on the per-task buf-list length, which is the real upper bound on the total `N + K` block count in a single task; layouts with `N + K > max_buf_list_len` return `DOCA_ERROR_INVALID_VALUE` at configure / submit. **There is no `doca_ec_cap_get_max_num_blocks` symbol in the public header — do not invent one.** |
| Supported matrix variants | *(no public cap query)* | The public header does NOT ship a `doca_ec_cap_get_matrix_*` family. Matrix-variant discovery is done by attempting `doca_ec_matrix_create()` / `doca_ec_matrix_create_from_raw()` against the active device for each variant the user wants to use; treat constructor failure as "variant not supported on this device." |

**Configuration shape.** *Mandatory* configurations before
`doca_ctx_start()`: at least one task type enabled (each with its
matching `set_conf` call, success callback, error callback, and
max-num-tasks budget); the four coding parameters committed
(matrix type, N, K, block size); and the matching mmap
permissions set on both source and destination buffers per the
matrix in [`## Safety policy`](#safety-policy). *Optional*
configurations (queue sizing, per-task `set_conf` callback
budget) use the standard DOCA Core `set_*` family with defaults
coming from the library; query the active capability values via
`doca_ec_cap_*`.

## Version compatibility

For the canonical DOCA version-detection chain, the four-way match
rule, NGC container semantics, and the headers-win-over-docs rule,
see [`doca-version`](../../doca-version/SKILL.md). The body lives
there; this skill does not duplicate it.

**The EC-specific overlay** is:

- **Per-task support is device-bound, not version-bound.** The
  agent must not infer *"DOCA version X includes create + recover
  + update"* from release notes alone; the runtime authority is
  the matching
  `doca_ec_cap_task_create_is_supported(devinfo)` /
  `_task_recover_is_supported(devinfo)` /
  `_task_update_is_supported(devinfo)` against the active device.
  Per the cross-cutting cap-query rule in
  [`doca-version CAPABILITIES.md ## Observability`](../../doca-version/CAPABILITIES.md#observability),
  the cap query is the runtime authority — never quote any of
  the three task types as available from agent memory.
- **Matrix-type support is device-bound, not version-bound.** The
  device advertises which Reed-Solomon matrix variants its
  accelerator implements (Cauchy or Vandermonde — the two values
  of `enum doca_ec_matrix_type`) via the
  `doca_ec_matrix_create()` constructor success per variant
  (the public header does NOT ship a `doca_ec_cap_get_matrix_*`
  family). Asserting a matrix type from
  release notes alone is the second-most-common hallucination
  failure mode; the runtime query is the authority.
- **`doca-erasure-coding.pc` plus `doca-common.pc` must both
  match `doca_caps --version`** at the four-way-match check (per
  [`doca-version CAPABILITIES.md ## Version compatibility`](../../doca-version/CAPABILITIES.md#version-compatibility)).
  A common partial-install pattern on hosts where the EC package
  was installed separately from the rest of DOCA is that
  `doca-erasure-coding.pc` reports release *X* while
  `doca-common.pc` reports release *Y*; route to
  [`doca-version TASKS.md ## debug`](../../doca-version/TASKS.md#debug)
  layer 2 before any EC-layer diagnosis.
- **`doca_ec_task_update` is the youngest of the three task
  types.** When the user reports *"the update API I'm reading
  about isn't on my install"*, the first hypothesis is that the
  install pre-dates update support. Confirm via
  `doca_ec_cap_task_update_is_supported(devinfo)`; if false,
  route to
  [`doca-version TASKS.md ## debug`](../../doca-version/TASKS.md#debug)
  before invoking the update codepath, and fall back to recompute
  via `doca_ec_task_create` in the meantime — flag the cost
  difference so the user understands the workaround is not
  permanent.

## Error taxonomy

EC-specific overlays on the cross-library `DOCA_ERROR_*` taxonomy.
The cross-library taxonomy itself lives in
[`doca-programming-guide CAPABILITIES.md ## Error taxonomy`](../../doca-programming-guide/CAPABILITIES.md#error-taxonomy);
the rows below are the *EC surface* meaning that the agent must
disambiguate before falling back to the cross-library response.

| Error | EC context where it shows up | EC-specific cause |
| --- | --- | --- |
| `DOCA_ERROR_BAD_STATE` | Any call after `doca_ctx_stop()` or before `doca_ctx_start()`; destroying the source or destination `doca_mmap` before `doca_ctx_destroy()`; submitting any task before the matching `set_conf` ran | Lifecycle violation. Walk the call sequence against the lifecycle in [`doca-programming-guide CAPABILITIES.md ## Capabilities and modes`](../../doca-programming-guide/CAPABILITIES.md#capabilities-and-modes); the most common case is releasing a source / destination mmap underneath a still-running context. |
| `DOCA_ERROR_INVALID_VALUE` | `doca_ec_task_create_allocate_init`, `_recover_allocate_init`, `_update_allocate_init`, `_galois_mul_allocate_init`; submit time | One of: block size > `doca_ec_cap_get_max_block_size(devinfo)`; total block count > `doca_ec_cap_get_max_buf_list_len(devinfo, &max)`; mismatched block sizes between source / destination buffers within a single task; recover with more than K missing blocks; matrix type configured does not match the matrix type associated with the buffers being recovered. Re-run the matching cap query and re-check that EVERY block in the task is the same size. |
| `DOCA_ERROR_NOT_SUPPORTED` | `doca_ec_task_*_set_conf`; matrix-type selection; first submit | The requested task type is not in this device's accelerator (any of create / recover / update / galois_mul can be independently unsupported), OR the matrix variant the user is trying to construct via `doca_ec_matrix_create()` is not supported on this device, OR the user is asking for a coding scheme outside the public Reed-Solomon surface. Re-run the matching `doca_ec_cap_task_*_is_supported` and re-attempt `doca_ec_matrix_create()` for the variant; if either fails, that is the answer — fall back to CPU EC, change matrix variant, or change N + K. |
| `DOCA_ERROR_NOT_PERMITTED` | First task submission | One or more source mmaps do not have `DOCA_ACCESS_FLAG_LOCAL_READ_ONLY` (or a stronger flag that supersedes it), OR one or more destination mmaps are missing `DOCA_ACCESS_FLAG_LOCAL_READ_WRITE`. Re-check the matrix in [`## Safety policy`](#safety-policy) against EVERY source and destination buffer (EC tasks have multiple buffers on both sides; a missing permission on one block of a 12-block layout is enough to trip this). |
| `DOCA_ERROR_AGAIN` | `doca_task_submit` on any EC task type | The task queue is full. This is *not* a hardware error; the program must drain completions via `doca_pe_progress()` before re-submitting. Same as the cross-library *"would-block, retry after progress"* pattern. |
| `DOCA_ERROR_DRIVER` | Any submit / completion call | The layer below DOCA reported failure (driver / firmware / accelerator). Capture state and route to env-class debug ([`doca-setup ## debug`](../../doca-setup/TASKS.md#debug)) — the layer below DOCA is the suspect, not the EC program. |

The agent's rule: **never recommend a retry loop on `DOCA_ERROR_*`
without first identifying which of the rows above is the cause**.
`_AGAIN` is the only one that wants a retry (after
`doca_pe_progress()`); the others want investigation, not retry.

## Observability

DOCA Erasure Coding observability is **event-driven, not
poll-driven**. Every submitted task produces a completion event
on the DOCA Core progress engine; there are no EC-specific
counters the way Flow has per-pipe counters.

Three primary signals the agent should reach for:

1. **Task completion events on the PE.** Every submitted EC task
   (create, recover, or update) produces a completion event when
   it finishes (or errors). The completion carries the
   `doca_error_t` if it failed; the agent must inspect the
   per-task completion, not the `doca_task_submit()` return
   value alone. *Submitted but no completion* is almost always a
   missed `doca_pe_progress()` call in the user's main loop.
2. **Capability snapshot at configure time.** The output of
   every `doca_ec_cap_*` query is a snapshot of *what the
   device's accelerator said was possible* before any task was
   submitted (task support, max block size, max N + K, supported
   matrix types). Save it as the baseline; if a task later
   returns `DOCA_ERROR_NOT_SUPPORTED` or `DOCA_ERROR_INVALID_VALUE`
   the diff against this snapshot is the bug.
3. **Known-vector recover smoke (encode → drop block → recover →
   bit-compare).** The cheapest correctness signal for an EC
   context is to encode a small fixed N + K layout from a known
   input, simulate losing one block (zero it or otherwise mark
   it missing), submit a recover task, and bit-compare the
   recovered block against the original. If the recovered block
   does not match, the configuration is wrong; do not move on to
   bulk storage data. For an update task, the equivalent smoke
   is: encode → update one data block → recompute parity by
   create from the new state → compare to the parity produced
   by update. The two parity sets must match byte-for-byte.

For cross-cutting observability primitives (`--sdk-log-level`,
the `doca-<lib>-trace` build flavor, the `DOCA_LOG_LEVEL` env
var) see
[`doca-debug CAPABILITIES.md ## Observability`](../../doca-debug/CAPABILITIES.md#observability).
For the install-tree observability (logger names, package
layout) defer to
[`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md).

## Safety policy

> **Overlay on the bundle-wide hardware-safety meta-policy.** The rules below are this skill's per-artifact overlay on the cross-cutting rules in [`doca-hardware-safety` CAPABILITIES.md ## Safety policy](../../doca-hardware-safety/CAPABILITIES.md#safety-policy) (specifically [### Per-artifact overlay pattern](../../doca-hardware-safety/CAPABILITIES.md#per-artifact-overlay-pattern)). When the two layers disagree, the stricter wins; when either layer says STOP, the agent stops.

DOCA Erasure Coding's safety surface is **buffer-permission
discipline plus path-selection discipline**. The library reads
from multiple source buffers (N data blocks for create; data
plus surviving parity blocks for recover; the changed source
block for update) and writes into multiple destination buffers
(K redundancy blocks for create; the missing blocks for recover;
the K refreshed parity blocks for update); an incorrect
permission flag on ANY of those buffers surfaces as a submit-time
error rather than silent corruption — but only when the matrix
is correct. The agent's job is to verify both permission and
the *should I use erasure coding at all* question before any
submission.

The **permission matrix** the agent must walk for any new EC
setup. Note that EC tasks have MULTIPLE buffers on each side,
so the matrix applies per buffer:

| Buffer | Minimum mmap permission | Class shape | Common over-broad mistake |
| --- | --- | --- | --- |
| Source data blocks (create, update) | `DOCA_ACCESS_FLAG_LOCAL_READ_ONLY` | The accelerator only needs to read source data; granting write is unnecessary and a small attack-surface widening | Granting `DOCA_ACCESS_FLAG_LOCAL_READ_WRITE` "to keep things simple" — works, but the read-only flag is the safe minimum |
| Source surviving blocks (recover — data + parity) | `DOCA_ACCESS_FLAG_LOCAL_READ_ONLY` | The accelerator only reads from the surviving subset to reconstruct the missing blocks | Same as above; the read-only flag is the safe minimum across all surviving buffers |
| Destination redundancy blocks (create) | `DOCA_ACCESS_FLAG_LOCAL_READ_WRITE` | The accelerator writes the K parity blocks; the application reads them back to persist them to storage | Reusing a source `doca_mmap` for destination — fails at submit if the mmap is read-only |
| Destination recovered blocks (recover) | `DOCA_ACCESS_FLAG_LOCAL_READ_WRITE` | The accelerator writes the reconstructed missing blocks; the application reads them back to repair the layout | Same as above; the destination mmap must be read-write |
| Destination updated parity blocks (update) | `DOCA_ACCESS_FLAG_LOCAL_READ_WRITE` | The accelerator writes the refreshed parity blocks; the application reads them back to overwrite the stale parity | Pointing the destination at the same mmap as the source data block — works only if the mmap is read-write; usually a wiring bug |

**Path-selection — when NOT to use doca-erasure-coding.** Equally
important:

- **Pure replication is enough.** If the durability target is met
  by storing M whole copies of the data (e.g. a small cluster
  where one extra copy is the right answer), erasure coding is
  the wrong primitive. EC trades CPU / accelerator cycles
  (encoding, reconstruction, updates) for storage efficiency;
  on systems where the storage savings don't justify the
  reconstruction cost, replication wins.
- **One-shot tiny inputs.** Encoding a single small dataset that
  fits comfortably in CPU EC is faster on the CPU than the
  DMA-to-accelerator round-trip. Recommend doca-erasure-coding
  when the workload is bulk (many encodes, or large datasets)
  *or* when the data is already pinned in `doca_mmap` memory
  because another DOCA library produced it.
- **Non-Reed-Solomon coding scheme.** doca-erasure-coding is
  Reed-Solomon. If the user needs fountain codes, LDPC, raptor
  codes, or any other family, the answer is a CPU library
  implementing that family — not a DOCA workaround.
- **Network durability (FEC on the wire).** Forward error
  correction on a network channel is a different problem with
  different libraries; do not route network FEC questions
  through doca-erasure-coding.
- **Single-block change → re-encode is wasteful.** If the user
  is editing one source block in an existing layout and asks
  about refreshing parity, the right answer is
  `doca_ec_task_update`, not `doca_ec_task_create`. Re-running
  create on all N data blocks just to refresh K parity blocks
  is linear-in-N work the update task is designed to avoid.

**The mmaps must stay valid until the EC context is destroyed.**
Destroying any source or destination mmap before
`doca_ctx_destroy()` is a use-after-free on the library's
bookkeeping; symptoms include `DOCA_ERROR_BAD_STATE` from
subsequent calls and undefined behavior on outstanding tasks.
This applies to EVERY buffer in an EC task (N data + K parity =
N + K buffers per layout — all must remain valid).

**Validate against a known-vector recover smoke BEFORE bulk
submission.** A single smoke (encode a tiny N + K layout, drop
one block, recover, bit-compare against the original) catches
matrix-type mis-selection, N + K mismatch, block-size
inconsistencies, and source / destination wiring bugs before
they corrupt a multi-GiB storage layout. For update workflows,
add an update smoke (encode → update one block → re-encode and
compare parity) as the second smoke before any bulk update path.

**Recover with more than K missing blocks is unrecoverable.** If
K′ > K blocks are missing from an N + K layout, the math says
the layout is gone. The agent must NOT invent a recovery
workflow; the right answer is to restore from another replica
(if the system layers replication on top of EC) or accept the
data loss. Surfaces as `DOCA_ERROR_INVALID_VALUE` from
`doca_ec_task_recover_allocate_init` or at submit.

## Deferred topic boundaries

This skill scopes itself to the DOCA Erasure Coding library.
Adjacent topics the agent will get asked but should route
elsewhere:

- **General coding-theory background** (Galois field arithmetic,
  Vandermonde vs Cauchy matrix derivations, MDS code bounds, why
  Reed-Solomon achieves the Singleton bound) — outside this
  skill. Route to upstream coding-theory references; this skill
  assumes the user already knows they want Reed-Solomon erasure
  coding and is asking *how to express it through the DOCA EC
  API*.
- **DOCA Core context and progress engine internals** — owned by
  [`doca-programming-guide`](../../doca-programming-guide/SKILL.md).
  This skill *uses* the Core context lifecycle; it does not
  redefine it.
- **Cross-cutting `DOCA_ERROR_*` taxonomy** — owned by
  [`doca-programming-guide CAPABILITIES.md ## Error taxonomy`](../../doca-programming-guide/CAPABILITIES.md#error-taxonomy).
  This skill adds the EC overlay, not the taxonomy itself.
- **Storage stack design** (how the user's distributed file
  system or object store should layer replication on top of EC,
  rebalance after node failure, schedule rebuilds, choose
  N + K per durability target) — outside this skill. The DOCA
  Erasure Coding library is one primitive inside a storage
  stack; the stack design lives upstream of this skill.
- **Other DOCA accelerator libraries (DOCA SHA, DOCA Compress,
  DOCA AES-GCM, DOCA DMA)** — separate libraries with their own
  skills. Path-selection guidance in
  [`## Capabilities and modes`](#capabilities-and-modes) names
  them; the deep per-library substance lives in the matching
  skill.
- **Cross-library `doca_caps` invocation patterns** — owned by
  the cross-library [`doca-caps`](../../tools/doca-caps/SKILL.md)
  tool skill. This skill references the *EC capability query
  family* (`doca_ec_cap_*`), which is per-library; the
  *cross-library capability snapshot tool* (`doca_caps
  --list-devs`) is a separate surface.
