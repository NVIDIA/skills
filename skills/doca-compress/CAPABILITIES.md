# DOCA Compress capabilities, version overlay, errors, observability, safety

**Where to start:** Pick the H2 anchor that matches your question
(capabilities / version / errors / observability / safety) and
read that section end-to-end. The tables in each section are the
load-bearing content; the prose around them is interpretation.

Read this file when the loader sent you here from
[SKILL.md](SKILL.md). For the *how* of executing each pattern
(the verbs `configure / build / modify / run / test / debug`),
jump to [TASKS.md](TASKS.md). For the canonical DOCA
version-handling rules that this skill layers a Compress overlay
on top of, see [`doca-version`](../../doca-version/SKILL.md).

## Pattern overview

Every DOCA Compress question this skill teaches resolves into one
of FIVE patterns. The patterns are CLASSES — they apply across
every Compress release and every BlueField / ConnectX device the
accelerator runs on.

| Pattern | When it applies (class shape) | Where the substance lives |
| --- | --- | --- |
| 1. Decide whether to offload | Compare the per-call setup cost of doca-compress against a CPU compressor (zlib / zstd) on the same input; choose offload only when the input is bulk (rule of thumb: ≥ a few KiB), repeated, or already pinned in `doca_mmap` memory | [`## Capabilities and modes`](#capabilities-and-modes) path-selection table + [`## Safety policy`](#safety-policy) *"when NOT to use doca-compress"* bullets |
| 2. Pick the task type(s) to enable | Compress task (`doca_compress_task_compress_deflate`) for outbound encoding; decompress task (`doca_compress_task_decompress_deflate`) for inbound decoding. Either is a valid standalone shape — a decompress-only consumer does NOT need to enable the compress task | [`## Capabilities and modes`](#capabilities-and-modes) compress-vs-decompress table + [TASKS.md ## modify](TASKS.md#modify) |
| 3. Discover capabilities | Query `doca_compress_cap_task_*_is_supported` and `_get_max_buf_size` per task type, against the active `doca_devinfo` BEFORE assuming the task is available or sizing buffers | [`## Capabilities and modes`](#capabilities-and-modes) capability-query rule + [TASKS.md ## configure](TASKS.md#configure) step 2 |
| 4. Honor source / destination permissions | Source mmap = `DOCA_ACCESS_FLAG_LOCAL_READ_ONLY` at minimum; destination mmap = `DOCA_ACCESS_FLAG_LOCAL_READ_WRITE`; both must be set BEFORE the first task submission | [`## Safety policy`](#safety-policy) permission matrix + [TASKS.md ## test](TASKS.md#test) |
| 5. Diagnose a Compress error | Map symptom (`DOCA_ERROR_BAD_STATE`, `_INVALID_VALUE`, `_NOT_SUPPORTED`, `_NOT_PERMITTED`, `_AGAIN`, `_DRIVER`) to root cause — lifecycle, buffer sizing, task support, permission, queue pressure, or driver layer — without leaving the Compress layer prematurely | [`## Error taxonomy`](#error-taxonomy) + [TASKS.md ## debug](TASKS.md#debug) |

Two cross-cutting rules that apply to *every* pattern above:

- **Discover the version-installed surface, do not assume.** Every
  pattern above gates on `pkg-config --modversion doca-compress`
  and on the `doca_compress_cap_*` capability queries against the
  active `doca_devinfo`. Quoting a per-task max buffer size, or
  asserting either task type is universally available, without
  checking, is the most common hallucination failure mode.
- **Validate against a round-trip BEFORE bulk.** The cheapest way
  to confirm the configured Compress context produces a correct
  encoding is to compress a small fixed input and decompress the
  output on the accelerator (or against a published CPU encoder
  like `zlib`), compare to the original byte-for-byte, before
  submitting any bulk user data. For decompression-only consumers,
  the equivalent smoke is to decompress a small fixture produced
  by a published CPU encoder.

## Capabilities and modes

DOCA Compress is a **DOCA Core Context**. Every `doca_compress`
instance follows the universal `cfg-create → cfg-set-* → init →
start → use → stop → destroy` lifecycle (see
[`doca-programming-guide CAPABILITIES.md ## Capabilities and modes`](../../doca-programming-guide/CAPABILITIES.md#capabilities-and-modes)).
On top of that lifecycle, DOCA Compress layers its own task model
with two independent task types.

**The two task types.** DOCA Compress exposes two task types, each
of which is independently enabled and independently
capability-gated. The agent must call the matching capability
query before assuming either task type is available on the user's
device.

| Task type | Class shape | Notes |
| --- | --- | --- |
| `doca_compress_task_compress_deflate` | Take one source `doca_buf` (raw input) → produce one destination `doca_buf` (DEFLATE-encoded output) | DEFLATE is the most common compression algorithm (the same algorithm used by zlib and gzip). Asynchronous; completion arrives via `doca_pe_progress`. Source must be `≤ doca_compress_cap_task_compress_deflate_get_max_buf_size(devinfo)`; destination must be sized for the worst-case compressed output (a poorly-compressible input can produce slightly more bytes than the input). |
| `doca_compress_task_decompress_deflate` | Take one source `doca_buf` (DEFLATE-encoded input) → produce one destination `doca_buf` (decoded output) | Decompresses DEFLATE-encoded payloads. **Valid as a standalone shape** — a consumer that only decompresses inbound network data does NOT have to enable the compress task. Asynchronous; completion arrives via `doca_pe_progress`. Source must be `≤ doca_compress_cap_task_decompress_deflate_get_max_buf_size(devinfo)`; destination must be sized for the worst-case decompressed output the input could expand to. |

**Path selection — compress vs decompress, or both.** Both task
types can coexist on the same context; enable at least one before
`doca_ctx_start()`. Enabling both is appropriate when the user is
running a round-trip (compress on the producer side, decompress on
the consumer side, same process or same machine); enabling only
one is the right shape for a one-direction pipeline.

| Path | What it is | Right shape for | Wrong shape for |
| --- | --- | --- | --- |
| Compress-only | Enable `doca_compress_task_compress_deflate`, leave decompress disabled | A producer / writer that emits DEFLATE-encoded output (e.g. compress-before-write storage daemon, compress-before-network sender, deduplication preprocessing) | A consumer that ONLY decodes inbound DEFLATE — enable the decompress task instead |
| Decompress-only | Enable `doca_compress_task_decompress_deflate`, leave compress disabled | A consumer / reader that decodes inbound DEFLATE-encoded data (e.g. decompress-on-read storage daemon, decompress-on-network receiver) — fully valid standalone shape | A producer that ONLY encodes outbound payloads — enable the compress task instead |
| Both | Enable both tasks on the same context | A round-trip flow (compress on one side, decompress on the other, same process) or a content-addressing pipeline that both encodes new blobs and decodes existing ones | Single-direction pipelines — keep the unused task disabled to keep the configuration surface small |

**Path-selection — when to use DOCA Compress at all.** The
agent's rule:

- **Use doca-compress when** the input is bulk (rule of thumb:
  ≥ a few KiB, with the canonical fit being inputs of file-size
  scale — typically ≥ 4 KiB and upwards), the throughput
  requirement is high enough that freeing the CPU for other work
  is valuable, or the input is already pinned in `doca_mmap`
  memory because another DOCA library produced it. File
  compression, deduplication preprocessing, and content-addressed
  storage prep are the canonical fits.
- **Do NOT use doca-compress when** the input is tiny one-shot
  (e.g. 64-byte buffers — the per-call DMA-to-accelerator setup
  cost dominates; CPU compression with zlib / zstd is faster);
  when the user needs **outbound** compression in an algorithm
  other than DEFLATE (use a CPU compression library — DOCA
  Compress exposes ONLY DEFLATE on the compress / encode side).
  However, DOCA Compress **does** support **LZ4 decompress** on
  the inbound / decode side, via two distinct task families
  (`doca_compress_task_decompress_lz4_stream` and
  `doca_compress_task_decompress_lz4_block`), each with its own
  `_cap_task_decompress_lz4_{stream,block}_is_supported(devinfo)`
  + `_get_max_buf_size` + `_get_max_buf_list_len` queries (and
  corresponding shipped samples under
  `/opt/mellanox/doca/samples/doca_compress/`). Do NOT blanket-
  refuse LZ4 — only outbound LZ4 encode is out of scope. The
  rule of thumb is: DEFLATE both directions; LZ4 decompress
  only; everything else (zstd / Snappy / brotli / …) → CPU.
  public task names); or when the user only wants to move bytes
  without encoding them (use [`doca-dma`](../doca-dma/SKILL.md) —
  pure mmap-to-mmap copy is DMA's job, not Compress's).

**The size-threshold the cap query does NOT report.** The
capability queries below report the per-task **maximum** input
size; they do NOT report the per-task **minimum** below which CPU
compression beats accelerator submission. That floor is
device-dependent and workload-dependent, and the agent must own
it as a path-selection rule rather than read it from the device.
A small smoke benchmark on representative inputs (CPU `zlib`
elapsed vs doca-compress elapsed end-to-end including
submit + progress + completion drain) is the only authoritative
answer for *"is offload worth it for inputs of size N"* on a
given host.

**Capability discovery — the only rule.** Before enabling either
task or sizing any buffer, call the matching `doca_compress_cap_*`
query against the active `doca_devinfo`:

| Capability | Query | Why the agent must ask |
| --- | --- | --- |
| Compress-deflate task supported | `doca_compress_cap_task_compress_deflate_is_supported(devinfo)` | The compress task is not on every device; some accelerators ship decompress-only. If false, the device cannot encode in hardware; the user must fall back to CPU compression or pick a different device. |
| Decompress-deflate task supported | `doca_compress_cap_task_decompress_deflate_is_supported(devinfo)` | Symmetric to the above. Independent of the compress query; agent must not assume support for one implies the other. |
| Decompress-lz4-stream task supported | `doca_compress_cap_task_decompress_lz4_stream_is_supported(devinfo)` | The hardware **does** decompress LZ4 stream-format payloads. Independent of the DEFLATE queries. The bundle previously refused this; the real surface is shipped. |
| Decompress-lz4-block task supported | `doca_compress_cap_task_decompress_lz4_block_is_supported(devinfo)` | The hardware also decompresses LZ4 block-format payloads. Independent of the LZ4-stream query and the DEFLATE queries. |
| Maximum source size, compress | `doca_compress_cap_task_compress_deflate_get_max_buf_size(devinfo)` | Device-bound ceiling on raw input size per compress submission. Inputs larger than this must be fragmented at the application layer; the library does not auto-chunk. |
| Maximum source size, decompress | `doca_compress_cap_task_decompress_deflate_get_max_buf_size(devinfo)`, `doca_compress_cap_task_decompress_lz4_stream_get_max_buf_size(devinfo)`, `doca_compress_cap_task_decompress_lz4_block_get_max_buf_size(devinfo)` (per algorithm) | Device-bound ceiling on compressed input size per decompress submission. The decompressed output can be substantially larger than this; the destination buffer sizing is the application's responsibility. |

**Configuration shape.** *Mandatory* configurations before
`doca_ctx_start()`: at least one task type enabled via
`doca_compress_task_compress_deflate_set_conf` or
`doca_compress_task_decompress_deflate_set_conf`, and the matching
mmap permissions set on both source and destination buffers per
the matrix in [`## Safety policy`](#safety-policy). *Optional*
configurations (queue sizing, per-task `set_conf` callback
budget) use the standard DOCA Core `set_*` family with defaults
coming from the library; query the active capability values via
`doca_compress_cap_*`.

## Version compatibility

For the canonical DOCA version-detection chain, the four-way match
rule, NGC container semantics, and the headers-win-over-docs rule,
see [`doca-version`](../../doca-version/SKILL.md). The body lives
there; this skill does not duplicate it.

**The Compress-specific overlay** is:

- **Per-task support is device-bound, not version-bound.** The
  agent must not infer *"DOCA version X includes the compress
  task and the decompress task"* from release notes alone; the
  runtime authority is
  `doca_compress_cap_task_compress_deflate_is_supported(devinfo)`
  and `_decompress_deflate_is_supported(devinfo)` against the
  active device. Per the cross-cutting cap-query rule in
  [`doca-version CAPABILITIES.md ## Observability`](../../doca-version/CAPABILITIES.md#observability),
  the cap query is the runtime authority — never quote either
  task as available from agent memory.
- **`doca-compress.pc` plus `doca-common.pc` must both match
  `doca_caps --version`** at the four-way-match check (per
  [`doca-version CAPABILITIES.md ## Version compatibility`](../../doca-version/CAPABILITIES.md#version-compatibility)).
  A common partial-install pattern on hosts where the Compress
  package was installed separately from the rest of DOCA is that
  `doca-compress.pc` reports release *X* while `doca-common.pc`
  reports release *Y*; route to
  [`doca-version TASKS.md ## debug`](../../doca-version/TASKS.md#debug)
  layer 2 before any Compress-layer diagnosis.
- **Per-task max buffer size is device-bound, not version-bound.**
  The `_get_max_buf_size` queries are the runtime authority; the
  matching `version-matrix.json` row records the *promise*. If
  the two disagree, the cap-query value wins, per the
  headers-win-over-docs rule in
  [`doca-version`](../../doca-version/SKILL.md).

## Error taxonomy

Compress-specific overlays on the cross-library `DOCA_ERROR_*`
taxonomy. The cross-library taxonomy itself lives in
[`doca-programming-guide CAPABILITIES.md ## Error taxonomy`](../../doca-programming-guide/CAPABILITIES.md#error-taxonomy);
the rows below are the *Compress surface* meaning that the agent
must disambiguate before falling back to the cross-library
response.

| Error | Compress context where it shows up | Compress-specific cause |
| --- | --- | --- |
| `DOCA_ERROR_BAD_STATE` | Any call after `doca_ctx_stop()` or before `doca_ctx_start()`; destroying the source or destination `doca_mmap` before `doca_ctx_destroy()` | Lifecycle violation. Walk the call sequence against the lifecycle in [`doca-programming-guide CAPABILITIES.md ## Capabilities and modes`](../../doca-programming-guide/CAPABILITIES.md#capabilities-and-modes); the most common case is releasing a source / destination mmap underneath a still-running context. |
| `DOCA_ERROR_INVALID_VALUE` | `doca_compress_task_compress_deflate_alloc_init`; `doca_compress_task_decompress_deflate_alloc_init`; submit time | The source buffer is larger than the matching `_get_max_buf_size` for the task type, OR the destination buffer is too small for the worst-case output (compress: input that does not compress well can produce slightly more bytes than the input; decompress: output can be substantially larger than the compressed input). Re-run the matching cap query, then resize. |
| `DOCA_ERROR_NOT_SUPPORTED` | `doca_compress_task_*_set_conf`; first submit | The requested task type is not available on this device's accelerator. Re-run the matching `doca_compress_cap_task_*_is_supported(devinfo)` query; if false, that is the answer — the user's device cannot offload this task type. Fall back to CPU compression or pick a different device. |
| `DOCA_ERROR_NOT_PERMITTED` | First task submission | The source mmap does not have `DOCA_ACCESS_FLAG_LOCAL_READ_ONLY` (or a stronger flag that supersedes it), OR the destination mmap is missing `DOCA_ACCESS_FLAG_LOCAL_READ_WRITE`. Re-check the matrix in [`## Safety policy`](#safety-policy). |
| `DOCA_ERROR_AGAIN` | `doca_task_submit` on either Compress task type | The task queue is full. This is *not* a hardware error; the program must drain completions via `doca_pe_progress()` before re-submitting. Same as the cross-library *"would-block, retry after progress"* pattern. |
| `DOCA_ERROR_DRIVER` | Any submit / completion call | The layer below DOCA reported failure (driver / firmware / accelerator). Capture state and route to env-class debug ([`doca-setup ## debug`](../../doca-setup/TASKS.md#debug)) — the layer below DOCA is the suspect, not the Compress program. |

The agent's rule: **never recommend a retry loop on `DOCA_ERROR_*`
without first identifying which of the rows above is the cause**.
`_AGAIN` is the only one that wants a retry (after
`doca_pe_progress()`); the others want investigation, not retry.

## Observability

DOCA Compress observability is **event-driven, not poll-driven**.
Every submitted task produces a completion event on the DOCA Core
progress engine; there are no Compress-specific counters the way
Flow has per-pipe counters.

Three primary signals the agent should reach for:

1. **Task completion events on the PE.** Every submitted Compress
   task (compress or decompress) produces a completion event when
   it finishes (or errors). The completion carries the
   `doca_error_t` if it failed AND the actual output size
   produced; the agent must inspect the per-task completion, not
   the `doca_task_submit()` return value alone. *Submitted but no
   completion* is almost always a missed `doca_pe_progress()`
   call in the user's main loop.
2. **Capability snapshot at configure time.** The output of every
   `doca_compress_cap_*` query is a snapshot of *what the
   device's accelerator said was possible* before any task was
   submitted. Save it as the baseline; if a task later returns
   `DOCA_ERROR_NOT_SUPPORTED` or `DOCA_ERROR_INVALID_VALUE` the
   diff against this snapshot is the bug.
3. **Round-trip validation on a known input.** The cheapest
   correctness signal is to compress a small fixed input and
   decompress the output (either on the accelerator if both tasks
   are enabled, or against a published CPU decoder like `zlib`)
   and compare to the original byte-for-byte. If the round-trip
   does not match, the configuration is wrong; do not move on to
   bulk input. For decompression-only consumers, the equivalent
   smoke is decompressing a small DEFLATE blob produced by a
   published CPU encoder and comparing to the original input.

For cross-cutting observability primitives (`--sdk-log-level`,
the `doca-<lib>-trace` build flavor, the `DOCA_LOG_LEVEL` env
var) see
[`doca-debug CAPABILITIES.md ## Observability`](../../doca-debug/CAPABILITIES.md#observability).
For the install-tree observability (logger names, package
layout) defer to
[`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md).

## Safety policy

> **Overlay on the bundle-wide hardware-safety meta-policy.** The rules below are this skill's per-artifact overlay on the cross-cutting rules in [`doca-hardware-safety` CAPABILITIES.md ## Safety policy](../../doca-hardware-safety/CAPABILITIES.md#safety-policy) (specifically [### Per-artifact overlay pattern](../../doca-hardware-safety/CAPABILITIES.md#per-artifact-overlay-pattern)). When the two layers disagree, the stricter wins; when either layer says STOP, the agent stops.

DOCA Compress's safety surface is **buffer-permission discipline
plus path-selection discipline**. The library reads from a source
buffer and writes encoded (or decoded) output into a destination
buffer; an incorrect permission flag on either side surfaces as a
submit-time error rather than silent corruption — but only when
the matrix is correct. The agent's job is to verify both
permission and the *should I use the accelerator at all*
question before any submission.

The **permission matrix** the agent must walk for any new
Compress setup:

| Buffer | Minimum mmap permission | Class shape | Common over-broad mistake |
| --- | --- | --- | --- |
| Source (input) | `DOCA_ACCESS_FLAG_LOCAL_READ_ONLY` | The accelerator only needs to read the input; granting write is unnecessary and a small attack-surface widening | Granting `DOCA_ACCESS_FLAG_LOCAL_READ_WRITE` "to keep things simple" — works, but the read-only flag is the safe minimum |
| Destination (output) | `DOCA_ACCESS_FLAG_LOCAL_READ_WRITE` | The accelerator writes the encoded / decoded output; the application reads it back | Reusing the source `doca_mmap` as the destination — fails at submit if the mmap is read-only; if both buffers share an mmap, the mmap must be read-write |

**Path-selection — when NOT to use doca-compress.** Equally
important:

- **Tiny one-shot inputs.** A single 64-byte compress on the CPU
  is faster than the DMA-to-accelerator setup. The agent should
  recommend doca-compress only when the input is bulk (rule of
  thumb: ≥ a few KiB, typically ≥ 4 KiB and upwards) *or*
  repeated, *or* the input is already pinned in `doca_mmap`
  memory because another DOCA library produced it.
- **Non-DEFLATE / non-LZ4 algorithm.** The public task names
  commit doca-compress to DEFLATE on the **encode** side
  (`doca_compress_task_compress_deflate`) and to DEFLATE +
  **LZ4 stream / LZ4 block** on the **decode** side
  (`doca_compress_task_decompress_deflate`,
  `doca_compress_task_decompress_lz4_stream`,
  `doca_compress_task_decompress_lz4_block`). The same algorithm
  underlying zlib and gzip). If the user needs a different
  algorithm (zstd, Snappy, brotli, …, or LZ4 on the **encode**
  side), the answer is to use a CPU
  library that implements it, not to invent a DOCA workaround.
- **Pure copy, no compression.** If the user only wants to move
  bytes between two `doca_mmap` regions without encoding them,
  [`doca-dma`](../doca-dma/SKILL.md) is the right library — its
  memcpy task is purpose-built for that pattern. Layering
  doca-compress on top wastes accelerator cycles on encoding
  output the user does not want.

**The mmap must stay valid until the Compress context is
destroyed.** Destroying either the source or destination mmap
before `doca_ctx_destroy()` is a use-after-free on the library's
bookkeeping; symptoms include `DOCA_ERROR_BAD_STATE` from
subsequent calls and undefined behavior on outstanding tasks.

**Validate against a round-trip BEFORE bulk submission.** A
single round-trip smoke (compress one short input and decompress
the output, or decompress a single small published DEFLATE
fixture) catches algorithm mis-selection, endian assumptions, and
buffer-sizing bugs before they corrupt a multi-GiB result.

## Deferred topic boundaries

This skill scopes itself to the DOCA Compress library. Adjacent
topics the agent will get asked but should route elsewhere:

- **General compression algorithm background** (entropy coding,
  why DEFLATE produces sub-optimal ratios on already-compressed
  inputs, when zstd is preferred to gzip, compression-level
  trade-offs in CPU libraries) — outside this skill. Route to
  upstream zlib / DEFLATE specification documentation; this
  skill assumes the user already knows they want DEFLATE and is
  asking *how to express it through the DOCA Compress API*.
- **DOCA Core context and progress engine internals** — owned by
  [`doca-programming-guide`](../../doca-programming-guide/SKILL.md).
  This skill *uses* the Core context lifecycle; it does not
  redefine it.
- **Cross-cutting `DOCA_ERROR_*` taxonomy** — owned by
  [`doca-programming-guide CAPABILITIES.md ## Error taxonomy`](../../doca-programming-guide/CAPABILITIES.md#error-taxonomy).
  This skill adds the Compress overlay, not the taxonomy itself.
- **Pure mmap-to-mmap copies** — owned by
  [`doca-dma`](../doca-dma/SKILL.md). When the user wants bytes
  moved without encoding, DMA is the answer; Compress is not.
- **Other DOCA accelerator libraries (DOCA SHA, DOCA AES-GCM,
  DOCA Erasure Coding, …)** — separate libraries with their own
  skills (when they ship). Path-selection guidance in
  [`## Capabilities and modes`](#capabilities-and-modes) names
  them; the deep per-library substance lives in the matching
  skill.
- **Cross-library `doca_caps` invocation patterns** — owned by
  the cross-library [`doca-caps`](../../tools/doca-caps/SKILL.md)
  tool skill. This skill references the *Compress capability
  query family* (`doca_compress_cap_*`), which is per-library;
  the *cross-library capability snapshot tool* (`doca_caps
  --list-devs`) is a separate surface.
