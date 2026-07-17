# DOCA SHA capabilities, version overlay, errors, observability, safety

**Where to start:** Pick the H2 anchor that matches your question
(capabilities / version / errors / observability / safety) and read
that section end-to-end. The tables in each section are the
load-bearing content; the prose around them is interpretation.

Read this file when the loader sent you here from
[SKILL.md](SKILL.md). For the *how* of executing each pattern (the
verbs `configure / build / modify / run / test / debug`), jump to
[TASKS.md](TASKS.md). For the canonical DOCA version-handling rules
that this skill layers a SHA overlay on top of, see
[`doca-version`](../../doca-version/SKILL.md).

## Pattern overview

Every DOCA SHA question this skill teaches resolves into one of FIVE
patterns. The patterns are CLASSES — they apply across every SHA
release and every BlueField / ConnectX device the accelerator runs
on.

| Pattern | When it applies (class shape) | Where the substance lives |
| --- | --- | --- |
| 1. Decide whether to offload | Compare the per-call setup cost of doca-sha against a CPU SHA on the same input; choose offload only when the input is bulk, repeated, or already pinned in `doca_mmap` memory | [`## Capabilities and modes`](#capabilities-and-modes) path-selection table + [`## Safety policy`](#safety-policy) *"when NOT to use doca-sha"* bullets |
| 2. Pick the task type | One-shot (`doca_sha_task_hash`) when the input fits in one source buffer; partial / incremental (`doca_sha_task_partial_hash`) when the input is larger than `max_src_buf_size` or arrives in chunks | [`## Capabilities and modes`](#capabilities-and-modes) one-shot-vs-partial table + [TASKS.md ## modify](TASKS.md#modify) |
| 3. Discover capabilities | Query `doca_sha_cap_*` for algorithm support, min destination buffer size, and max source buffer size, against the active `doca_devinfo` BEFORE choosing the algorithm or sizing buffers | [`## Capabilities and modes`](#capabilities-and-modes) capability-query rule + [TASKS.md ## configure](TASKS.md#configure) step 2 |
| 4. Honor source / destination permissions | Source mmap = `DOCA_ACCESS_FLAG_LOCAL_READ_ONLY` at minimum; destination mmap = `DOCA_ACCESS_FLAG_LOCAL_READ_WRITE`; both must be set BEFORE the first task submission | [`## Safety policy`](#safety-policy) permission matrix + [TASKS.md ## test](TASKS.md#test) |
| 5. Diagnose a SHA error | Map symptom (`DOCA_ERROR_BAD_STATE`, `_INVALID_VALUE`, `_NOT_SUPPORTED`, `_NOT_PERMITTED`, `_AGAIN`) to root cause — lifecycle, buffer sizing, algorithm support, permission, or queue pressure — without leaving the SHA layer prematurely | [`## Error taxonomy`](#error-taxonomy) + [TASKS.md ## debug](TASKS.md#debug) |

Two cross-cutting rules that apply to *every* pattern above:

- **Discover the version-installed surface, do not assume.** Every
  pattern above gates on `pkg-config --modversion doca-sha` and on
  the `doca_sha_cap_*` capability queries against the active
  `doca_devinfo`. Quoting an algorithm enum, a digest size, or a
  max source buffer size without checking is the most common
  hallucination failure mode.
- **Validate against a published test vector before bulk.** The
  cheapest way to confirm the configured SHA context produces a
  correct digest is to hash a small fixed input (e.g. the empty
  string, "abc", or the million-`a` input from the NIST SHA test
  vectors) and compare against the published digest, before
  submitting any user data.

## Capabilities and modes

DOCA SHA is a **DOCA Core Context**. Every SHA instance follows the
universal `cfg-create → cfg-set-* → init → start → use → stop →
destroy` lifecycle (see
[`doca-programming-guide CAPABILITIES.md ## Capabilities and modes`](../../doca-programming-guide/CAPABILITIES.md#capabilities-and-modes)).
On top of that lifecycle, DOCA SHA layers its own task model and
algorithm-selection surface.

**The two task types.** DOCA SHA exposes two task types; each has
its own `doca_sha_task_*_set_conf` (enable the task) and its own
matching `doca_sha_cap_task_*_get_supported(devinfo, algorithm)` (capability query)
entry point. The agent must call the capability query before
assuming the task type is available on the user's device.

| Task type | Class shape | Notes |
| --- | --- | --- |
| `doca_sha_task_hash` (one-shot) | Take a single source `doca_buf` + an algorithm enum, produce a digest in a single destination `doca_buf` | Asynchronous; completion arrives via `doca_pe_progress`. Source must be `≤ doca_sha_cap_get_max_src_buf_size(devinfo)`. |
| `doca_sha_task_partial_hash` (incremental) | Stream input across multiple submissions, finalize with a separate call | Use when the input is larger than `max_src_buf_size` OR arrives in chunks (e.g. streamed from disk or network). Marking the task final on the first buffer (before any intermediate submit) is **undefined behavior**, not a defined error — use a non-partial `doca_sha_task_hash` instead when a single buffer is all you have. |

**Path selection — one-shot vs partial / incremental.** Both task
types can coexist on the same context; choose at least one before
`doca_ctx_start()`.

| Path | What it is | Right shape for | Wrong shape for |
| --- | --- | --- | --- |
| One-shot (`doca_sha_task_hash`) | Single source buffer + single destination buffer + one task submission | Buffers that fit in `max_src_buf_size` and are available in full at submit time (file-integrity verify on a small file, content-addressed lookup of a known blob) | Inputs larger than `max_src_buf_size`; inputs that arrive in chunks (streaming I/O); per-call cost dominates if the input is tiny — prefer CPU |
| Partial / incremental (`doca_sha_task_partial_hash`) | Multiple intermediate submissions feeding the same digest state, finalized with a separate call | Bulk inputs (multi-GiB files, content addressing across large blobs); streaming inputs where chunks arrive over time | One-shot small inputs; CPU is faster than the producer / consumer ceremony for a single small hash |

**Path-selection — when to use DOCA SHA at all.** The agent's rule:

- **Use doca-sha when** the input is bulk (rule of thumb: ≥ a few
  hundred KiB), the throughput requirement is high enough that
  freeing the CPU for other work is valuable, or the input is
  already pinned in `doca_mmap` memory because another DOCA library
  produced it. File integrity verification across many files,
  content addressing in a deduplication preprocessing pass, and
  storage workloads are the canonical fits.
- **Do NOT use doca-sha when** the input is a tiny one-shot hash
  (the per-call DMA-to-accelerator setup cost dominates; CPU is
  faster); when the user needs an algorithm not in the device's
  accelerator (use OpenSSL or a similar CPU library); or when the
  hash is part of a larger crypto flow that another DOCA library
  already covers (e.g. AES-GCM authentication tags via
  doca-aes-gcm).

**Algorithm enum — the only public enum.** DOCA SHA's algorithm
enum surface advertises:

| Enum | Notes |
| --- | --- |
| `DOCA_SHA_ALGORITHM_SHA1` | Legacy 160-bit digest; still supported for compatibility scenarios |
| `DOCA_SHA_ALGORITHM_SHA256` | The canonical 256-bit SHA-2 digest; the most common choice for file integrity |
| `DOCA_SHA_ALGORITHM_SHA512` | 512-bit SHA-2 digest |

The agent must NOT add algorithm names beyond this enum (no MD5, no
SHA-3, no Keccak, no Blake) unless `doca_sha_cap_task_hash_get_supported(devinfo, algorithm)`
returns true for them on the user's device — the enum above is the
public surface this skill commits to.

**Capability discovery — the only rule.** Before choosing an
algorithm or sizing any buffer, call the matching `doca_sha_cap_*`
query against the active `doca_devinfo`:

| Capability | Query | Why the agent must ask |
| --- | --- | --- |
| One-shot hash + algorithm supported | `doca_sha_cap_task_hash_get_supported(devinfo, algorithm)` | Returns `DOCA_SUCCESS` if both the one-shot hash task *and* the given algorithm are supported on this device. Any other return is "not supported." The "task supported" and "algorithm supported" axes are folded into this one call — there is no separate `doca_sha_cap_task_hash_is_supported` or `doca_sha_cap_is_algorithm_supported` in the public header. |
| Partial / incremental hash + algorithm supported | `doca_sha_cap_task_partial_hash_get_supported(devinfo, algorithm)` | Same shape, for the partial-hash task. Partial hash is not on every device-and-algorithm pair; agent must not silently assume it for streaming inputs. |
| Minimum destination buffer size | `doca_sha_cap_get_min_dst_buf_size(devinfo, algorithm)` | Algorithm-dependent; a destination buffer smaller than this returns `DOCA_ERROR_INVALID_VALUE` at submit time |
| Maximum source buffer size | `doca_sha_cap_get_max_src_buf_size(devinfo)` | Hardware-bound ceiling on one-shot input size; inputs larger than this force the partial-hash path |

**Configuration shape.** *Mandatory* configurations before
`doca_ctx_start()`: at least one task type enabled via
`doca_sha_task_hash_set_conf` or `doca_sha_task_partial_hash_set_conf`,
and the matching mmap permissions set on both source and
destination buffers per the matrix in [`## Safety policy`](#safety-policy).
*Optional* configurations (queue sizing,
`doca_sha_task_*_set_conf` queue depth) use the standard DOCA Core
`set_*` family with defaults coming from the library; query the
active value with `doca_sha_cap_get_*`.

## Version compatibility

For the canonical DOCA version-detection chain, the four-way match rule, NGC container semantics, and the headers-win-over-docs rule, see [`doca-version`](../../doca-version/SKILL.md). The body lives there; this skill does not duplicate it.

**The SHA-specific overlay** is:

- **Algorithm enum membership is device-bound, not version-bound.** The agent must not infer *"DOCA version X includes SHA-Y"* from release notes alone; the runtime authority is `doca_sha_cap_task_hash_get_supported(devinfo, DOCA_SHA_ALGORITHM_*)` against the active device (which folds task-support and algorithm-support into a single call). Per the cross-cutting cap-query rule in [`doca-version CAPABILITIES.md ## Observability`](../../doca-version/CAPABILITIES.md#observability), the cap query is the runtime authority — never quote an algorithm as available from agent memory.
- **`doca_sha_task_partial_hash` is newer than the one-shot task.** When the user reports *"the partial-hash API I'm reading about isn't on my install"*, the first hypothesis is that the install pre-dates partial-hash support. Confirm via `doca_sha_cap_task_partial_hash_get_supported(devinfo, algorithm)` for each algorithm the user needs; if it returns anything other than `DOCA_SUCCESS`, route to [`doca-version TASKS.md ## debug`](../../doca-version/TASKS.md#debug) before invoking the partial-hash codepath.
- **`doca-sha.pc` plus `doca-common.pc` must both match `doca_caps --version`** at the four-way-match check (per [`doca-version CAPABILITIES.md ## Version compatibility`](../../doca-version/CAPABILITIES.md#version-compatibility)). A common partial-install pattern on hosts where the SHA package was installed separately from the rest of DOCA is that `doca-sha.pc` reports release *X* while `doca-common.pc` reports release *Y*; route to [`doca-version TASKS.md ## debug`](../../doca-version/TASKS.md#debug) layer 2 before any SHA-layer diagnosis.

## Error taxonomy

SHA-specific overlays on the cross-library `DOCA_ERROR_*` taxonomy.
The cross-library taxonomy itself lives in
[`doca-programming-guide CAPABILITIES.md ## Error taxonomy`](../../doca-programming-guide/CAPABILITIES.md#error-taxonomy);
the rows below are the *SHA surface* meaning that the agent must
disambiguate before falling back to the cross-library response.

| Error | SHA context where it shows up | SHA-specific cause |
| --- | --- | --- |
| `DOCA_ERROR_BAD_STATE` | Any call after `doca_ctx_stop()` or before `doca_ctx_start()` | Lifecycle violation. Walk the call sequence against the lifecycle in [`doca-programming-guide CAPABILITIES.md ## Capabilities and modes`](../../doca-programming-guide/CAPABILITIES.md#capabilities-and-modes). Note: marking a `doca_sha_task_partial_hash` task final on the first buffer (before any intermediate submit) is **undefined behavior**, not a defined `BAD_STATE` error — use a non-partial `doca_sha_task_hash` instead when a single buffer is all you have. |
| `DOCA_ERROR_INVALID_VALUE` | `doca_sha_task_hash_alloc_init`; `doca_sha_task_partial_hash_alloc_init`; submit time | The destination buffer is smaller than `doca_sha_cap_get_min_dst_buf_size(devinfo, algorithm)`, OR the source buffer is larger than `doca_sha_cap_get_max_src_buf_size(devinfo)`. Re-run the matching cap query, then resize. |
| `DOCA_ERROR_NOT_SUPPORTED` | `doca_sha_task_*_set_conf`; submit with an algorithm enum | The algorithm enum is not in this device's accelerator, OR `doca_sha_task_partial_hash` is requested on a device that advertises only the one-shot task for that algorithm. Re-run `doca_sha_cap_task_hash_get_supported(devinfo, algorithm)` / `doca_sha_cap_task_partial_hash_get_supported(devinfo, algorithm)` against the active `doca_devinfo`; if either returns anything other than `DOCA_SUCCESS`, that is the answer — fall back to CPU or change algorithm. |
| `DOCA_ERROR_NOT_PERMITTED` | First task submission | The source mmap does not have `DOCA_ACCESS_FLAG_LOCAL_READ_ONLY` (or a stronger flag that supersedes it), OR the destination mmap is missing `DOCA_ACCESS_FLAG_LOCAL_READ_WRITE`. Re-check the matrix in [`## Safety policy`](#safety-policy). |
| `DOCA_ERROR_AGAIN` | `doca_task_submit` on either SHA task type | The task queue is full. This is *not* a hardware error; the program must drain completions via `doca_pe_progress()` before re-submitting. Same as the cross-library *"would-block, retry after progress"* pattern. |
| `DOCA_ERROR_DRIVER` | Any submit / completion call | The layer below DOCA reported failure. Capture state and route to env-class debug ([`doca-setup ## debug`](../../doca-setup/TASKS.md#debug)) — the layer below DOCA is the suspect, not the SHA program. |

The agent's rule: **never recommend a retry loop on `DOCA_ERROR_*`
without first identifying which of the rows above is the cause**.
`_AGAIN` is the only one that wants a retry (after
`doca_pe_progress()`); the others want investigation, not retry.

## Observability

DOCA SHA observability is **event-driven, not poll-driven**. Every
submitted task produces a completion event on the DOCA Core
progress engine; there are no SHA-specific counters the way Flow
has per-pipe counters.

Three primary signals the agent should reach for:

1. **Task completion events on the PE.** Every submitted SHA task
   (one-shot or partial) produces a completion event when it
   finishes (or errors). The completion carries the `doca_error_t`
   if it failed; the agent must inspect the per-task completion,
   not the `doca_task_submit()` return value alone. *Submitted but
   no completion* is almost always a missed `doca_pe_progress()`
   call in the user's main loop.
2. **Capability snapshot at configure time.** The output of every
   `doca_sha_cap_*` query is a snapshot of *what the device's
   accelerator said was possible* before any task was submitted.
   Save it as the baseline; if a task later returns
   `DOCA_ERROR_NOT_SUPPORTED` or `DOCA_ERROR_INVALID_VALUE` the diff
   against this snapshot is the bug.
3. **Known-vector digest comparison.** The cheapest correctness
   signal is to hash a small fixed input (a published NIST SHA
   test vector — the empty string, "abc", or the million-`a`
   input) and compare against the published digest. If the
   accelerator produces a different digest, the configuration is
   wrong; do not move on to bulk input.

For cross-cutting observability primitives (`--sdk-log-level`, the
`doca-<lib>-trace` build flavor, the `DOCA_LOG_LEVEL` env var) see
[`doca-debug CAPABILITIES.md ## Observability`](../../doca-debug/CAPABILITIES.md#observability).
For the install-tree observability (logger names, package layout)
defer to
[`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md).

## Safety policy

> **Overlay on the bundle-wide hardware-safety meta-policy.** The rules below are this skill's per-artifact overlay on the cross-cutting rules in [`doca-hardware-safety` CAPABILITIES.md ## Safety policy](../../doca-hardware-safety/CAPABILITIES.md#safety-policy) (specifically [### Per-artifact overlay pattern](../../doca-hardware-safety/CAPABILITIES.md#per-artifact-overlay-pattern)). When the two layers disagree, the stricter wins; when either layer says STOP, the agent stops.

DOCA SHA's safety surface is **buffer-permission discipline plus
path-selection discipline**. The library reads from a source buffer
and writes a digest into a destination buffer; an incorrect
permission flag on either side surfaces as a submit-time error
rather than silent corruption — but only when the matrix is
correct. The agent's job is to verify both permission and the
*should I use the accelerator at all* question before any
submission.

The **permission matrix** the agent must walk for any new SHA
setup:

| Buffer | Minimum mmap permission | Class shape | Common over-broad mistake |
| --- | --- | --- | --- |
| Source (input) | `DOCA_ACCESS_FLAG_LOCAL_READ_ONLY` | The accelerator only needs to read the input; granting write is unnecessary and a small attack-surface widening | Granting `DOCA_ACCESS_FLAG_LOCAL_READ_WRITE` "to keep things simple" — works, but the read-only flag is the safe minimum |
| Destination (digest) | `DOCA_ACCESS_FLAG_LOCAL_READ_WRITE` | The accelerator writes the digest; the application reads it back | Reusing the source `doca_mmap` as the destination — fails at submit if the mmap is read-only; if both buffers share an mmap, the mmap must be read-write |

**Path-selection — when NOT to use doca-sha.** Equally important:

- **Tiny one-shot hashes.** A single 64-byte hash on the CPU is
  faster than the DMA-to-accelerator setup. The agent should
  recommend doca-sha only when the input is ≥ a few hundred KiB
  *or* repeated, *or* the input is already pinned in `doca_mmap`
  memory because another DOCA library produced it.
- **Unsupported algorithm.** If `doca_sha_cap_task_hash_get_supported(devinfo, algorithm)`
  returns false for the user's intended algorithm on the active
  device, the answer is to use OpenSSL (or a similar CPU library),
  not to invent a partial-hash workaround.
- **Hash is part of a larger crypto flow.** If the user is
  computing the hash only to feed it into AES-GCM or a similar
  authenticated-encryption mode that DOCA already implements,
  doca-aes-gcm covers the combined surface; layering doca-sha on
  top is redundant.

**The mmap must stay valid until the SHA context is destroyed.**
Destroying either the source or destination mmap before
`doca_ctx_destroy()` is a use-after-free on the library's
bookkeeping; symptoms include `DOCA_ERROR_BAD_STATE` from
subsequent calls and undefined behavior on outstanding tasks.

**Validate against a known vector BEFORE bulk submission.** A
single known-vector smoke (one short input with a published
digest) catches algorithm mis-selection, endian assumptions, and
buffer-sizing bugs before they corrupt a multi-GiB result.

## Deferred topic boundaries

This skill scopes itself to the DOCA SHA library. Adjacent topics
the agent will get asked but should route elsewhere:

- **General cryptographic hash function background** (collision
  resistance, length extension, why SHA-1 is deprecated for
  signatures, when SHA-3 is preferred) — outside this skill. Route
  to upstream NIST / IETF cryptography documentation; this skill
  assumes the user already knows which algorithm they want and is
  asking *how to express it through the DOCA SHA API*.
- **DOCA Core context and progress engine internals** — owned by
  [`doca-programming-guide`](../../doca-programming-guide/SKILL.md).
  This skill *uses* the Core context lifecycle; it does not
  redefine it.
- **Cross-cutting `DOCA_ERROR_*` taxonomy** — owned by
  [`doca-programming-guide CAPABILITIES.md ## Error taxonomy`](../../doca-programming-guide/CAPABILITIES.md#error-taxonomy).
  This skill adds the SHA overlay, not the taxonomy itself.
- **Other DOCA crypto-acceleration libraries (DOCA AES-GCM, DOCA
  Compress, DOCA DMA)** — separate libraries with their own skills
  (when they ship). Path-selection guidance in [`## Capabilities
  and modes`](#capabilities-and-modes) names them; the deep
  per-library substance lives in the matching skill.
- **Cross-library `doca_caps` invocation patterns** — owned by
  the cross-library `doca-caps` tool skill (when it ships). This
  skill references the *SHA capability query family*
  (`doca_sha_cap_*`), which is per-library; the *cross-library
  capability snapshot tool* (`doca_caps --list-devs`) is a
  separate surface.
