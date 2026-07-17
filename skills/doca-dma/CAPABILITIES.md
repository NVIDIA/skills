# DOCA DMA capabilities, version overlay, errors, observability, safety

**Where to start:** Pick the H2 anchor that matches your question
(task type / capability discovery / path selection / version /
errors / observability / safety) and read that section
end-to-end. The tables in each section are the load-bearing
content; the prose around them is interpretation.

Read this file when the loader sent you here from
[SKILL.md](SKILL.md). For the *how* of executing each pattern (the
verbs `configure / build / modify / run / test / debug`), jump to
[TASKS.md](TASKS.md). For the canonical DOCA version-handling
rules that this skill layers a DMA overlay on top of, see
[`doca-version`](../../doca-version/SKILL.md).

## Pattern overview

Every DMA question this skill teaches resolves into one of FIVE
patterns. The patterns are CLASSES — they apply across every DMA
release and every host / DPU / ConnectX combination, not just the
worked examples shown.

| Pattern | When it applies (class shape) | Where the substance lives |
| --- | --- | --- |
| 1. Decide DMA is the right library | The copy is between two `doca_mmap` regions on the same host or across a host ↔ DPU PCIe boundary; the data does NOT need to traverse the network and is NOT message-oriented producer/consumer | [`## Capabilities and modes`](#capabilities-and-modes) path-selection bullet |
| 2. Stand up the context | DOCA Core lifecycle: create → configure (memcpy task, properties) → start → submit memcpy tasks → progress → stop → destroy | [TASKS.md ## configure](TASKS.md#configure) |
| 3. Discover capabilities | `doca_dma_cap_task_memcpy_is_supported`, `_get_max_buf_size`, `_get_max_buf_list_len` against the active `doca_devinfo` BEFORE sizing buffers or assuming scatter-gather | [`## Capabilities and modes`](#capabilities-and-modes) capability-query rule + [TASKS.md ## configure](TASKS.md#configure) step 2 |
| 4. Honor permissions | Source mmap → `DOCA_ACCESS_FLAG_LOCAL_READ_ONLY` (and `doca_mmap_export_*` for cross-peer); destination mmap → `DOCA_ACCESS_FLAG_LOCAL_READ_WRITE` | [`## Safety policy`](#safety-policy) permission matrix + [TASKS.md ## test](TASKS.md#test) |
| 5. Diagnose a DMA error | Map symptom (`BAD_STATE`, `INVALID_VALUE`, `AGAIN`, `NOT_PERMITTED`, `NOT_SUPPORTED`, `DRIVER`) to root cause without leaving the DMA layer prematurely | [`## Error taxonomy`](#error-taxonomy) + [TASKS.md ## debug](TASKS.md#debug) |

Two cross-cutting rules that apply to *every* pattern above:

- **Cap query before payload, every time.** Do not size a buffer
  or assume scatter-gather is supported without first reading
  `doca_dma_cap_task_memcpy_get_max_buf_size` and
  `_get_max_buf_list_len` against the active `doca_devinfo`.
  Quoting a number from memory is the most common hallucination
  failure mode for this library.
- **Source / destination permission pair before submit, every
  time.** The `DOCA_ACCESS_FLAG_LOCAL_READ_ONLY` /
  `DOCA_ACCESS_FLAG_LOCAL_READ_WRITE` pair must be set on the
  matching mmap *before* the task is submitted. Skipping the
  permission set returns `DOCA_ERROR_NOT_PERMITTED` that
  masquerades as a hardware bug.

## Capabilities and modes

DOCA DMA is a **DOCA Core Context**. Every `doca_dma` instance
follows the universal `cfg-create → cfg-set-* → init → start →
use → stop → destroy` lifecycle (see
[`doca-programming-guide CAPABILITIES.md ## Capabilities and modes`](../../doca-programming-guide/CAPABILITIES.md#capabilities-and-modes)).
On top of that lifecycle, DMA layers a single task type and a
small capability-query family.

**The single task type.** DMA exposes ONE task type. The agent
must not invent additional ones; the public surface is closed.

| Task type | Direction | Notes |
| --- | --- | --- |
| `doca_dma_task_memcpy` | one source `doca_mmap` region → one destination `doca_mmap` region, with a length in bytes | Asynchronous; completion arrives as an event on the DOCA progress engine. Both regions must be valid `doca_mmap`s with the matching permission flag set; for cross-peer copies the source region must be exported via the appropriate `doca_mmap_export_*` so the DMA engine on the other side can read from it. |

**Capability discovery — the only rule.** Before sizing any
buffer or assuming scatter-gather is supported, call the matching
`doca_dma_cap_task_memcpy_*` query against the active
`doca_devinfo`:

| Capability | Query | Why the agent must ask |
| --- | --- | --- |
| Memcpy task supported on this device | `doca_dma_cap_task_memcpy_is_supported(devinfo)` | Some `doca_dev` entries do not advertise DMA at all; recommending the task without the probe is the silent-fail case |
| Maximum per-task buffer size | `doca_dma_cap_task_memcpy_get_max_buf_size(devinfo)` | Device-dependent; submitting a task whose source or destination buffer exceeds this returns `DOCA_ERROR_INVALID_VALUE` |
| Maximum scatter-gather buffer-list length | `doca_dma_cap_task_memcpy_get_max_buf_list_len(devinfo)` | Some devices cap the buffer-list at 1 (no scatter-gather); a multi-element list past the cap also returns `DOCA_ERROR_INVALID_VALUE` |

**Path selection — DMA vs the adjacent libraries.** DMA is for
mmap-to-mmap copies through the BlueField DMA engine. It is not
the answer for every copy; the agent must walk this rule before
recommending DMA setup.

| Use DOCA DMA when … | Use a different library when … |
| --- | --- |
| Bulk copy between two `doca_mmap` regions on the same host, between host and DPU memory, or between two DPU memory regions, where the goal is to free CPU cycles or hit lower latency than a CPU memcpy can provide | The data has to traverse the network — use [`doca-rdma`](../doca-rdma/SKILL.md) (its Read / Write task types are the cross-network analog of DMA's memcpy) |
| The user already holds (or can register) the source and destination as `doca_mmap` regions and the copy length is large enough that DMA submit overhead is amortized | The data flow is producer / consumer message-oriented between a host process and a DPU process — use [`doca-comch`](../doca-comch/SKILL.md) fast-path, whose producer / consumer surface is purpose-built for that pattern |
| The copy is part of a chain of accelerator offloads (DMA → SHA → AES-GCM → Compress) where keeping the bytes inside DOCA accelerator queues avoids extra CPU round-trips | The copy is tiny and one-shot (a few hundred bytes, executed once) — a CPU `memcpy` finishes faster than the DMA submit + completion-drain round-trip |

**Configuration shape.** *Mandatory* configurations before
`doca_ctx_start()`: enable the memcpy task via
`doca_dma_task_memcpy_set_conf` (with completion callbacks +
max-num-tasks); set permissions on the source and destination
`doca_mmap` regions to match the matrix in
[`## Safety policy`](#safety-policy). *Optional* configurations
(per-task user-data, completion-batch tuning) follow the standard
DOCA Core surface; defaults come from the library and the active
`doca_devinfo`.

## Version compatibility

For the canonical DOCA version-detection chain, the four-way match rule, NGC container semantics, and the headers-win-over-docs rule, see [`doca-version`](../../doca-version/SKILL.md). The body lives there; this skill does not duplicate it.

**The DMA-specific overlay** is:

- **Use the `doca_dma_cap_task_memcpy_*` query family at runtime.** Per the cross-cutting rule in [`doca-version CAPABILITIES.md ## Observability`](../../doca-version/CAPABILITIES.md#observability), the cap-query is the runtime authority for *"is the memcpy task supported on this device + this DOCA version, and how big a buffer can it move"*. The version-matrix is the *promise*; the cap query is the *reality*. Use `pkg-config --modversion doca-dma` as the build-time anchor (per [`doca-version TASKS.md ## configure`](../../doca-version/TASKS.md#configure)) when the user asks which DMA features ship on this install.
- **`doca-dma.pc` plus `doca-common.pc` must both match `doca_caps --version`** at the four-way-match check (per [`doca-version CAPABILITIES.md ## Version compatibility`](../../doca-version/CAPABILITIES.md#version-compatibility)). A common partial-install pattern after a DOCA upgrade is that `doca-dma.pc` lingers from the previous release; route to [`doca-version TASKS.md ## debug`](../../doca-version/TASKS.md#debug) ladder step 2 before any DMA-layer diagnosis.
- **Headers in $(pkg-config --variable=includedir doca-common) win over public docs.** Per the headers-win-over-docs rule in [`doca-version`](../../doca-version/SKILL.md), if a public DMA doc page mentions a `doca_dma_*` symbol that is not in the installed `doca_dma.h`, the headers describe what *this* install can call; the docs describe what *some* release shipped. The agent must quote the headers, not the docs URL, when the two disagree.

## Error taxonomy

DMA-specific overlays on the cross-library `DOCA_ERROR_*`
taxonomy. The cross-library taxonomy itself lives in
[`doca-programming-guide CAPABILITIES.md ## Error taxonomy`](../../doca-programming-guide/CAPABILITIES.md#error-taxonomy);
the rows below are the *DMA surface* meaning that the agent must
disambiguate before falling back to the cross-library response.

| Error | DMA context where it shows up | DMA-specific cause |
| --- | --- | --- |
| `DOCA_ERROR_BAD_STATE` | Any submit before `doca_ctx_start()` or after `doca_ctx_stop()`; destroying the source or destination `doca_mmap` before `doca_ctx_destroy()` | Lifecycle violation. Walk the call sequence against the lifecycle in [`doca-programming-guide CAPABILITIES.md ## Capabilities and modes`](../../doca-programming-guide/CAPABILITIES.md#capabilities-and-modes); the most common case is releasing an mmap underneath a still-running context. |
| `DOCA_ERROR_INVALID_VALUE` | `doca_dma_task_memcpy_alloc_init` with an oversized buffer or an oversized buffer-list | The length exceeds `doca_dma_cap_task_memcpy_get_max_buf_size(devinfo)`, or the scatter-gather list exceeds `_get_max_buf_list_len(devinfo)`. The fix is to fragment at the application layer; do not retry without re-reading the cap. |
| `DOCA_ERROR_AGAIN` | `doca_task_submit` on the memcpy task | The DMA task queue is full. This is *not* a hardware error; the program must drain completions via `doca_pe_progress()` before re-submitting. Same as the cross-library *"would-block, retry after progress"* pattern. |
| `DOCA_ERROR_NOT_PERMITTED` | First memcpy submit, or first cross-peer submit | The mmap permission flag is missing for the direction of access — source mmap missing `DOCA_ACCESS_FLAG_LOCAL_READ_ONLY`, destination missing `DOCA_ACCESS_FLAG_LOCAL_READ_WRITE`, or a cross-peer source not exported via `doca_mmap_export_*`. Re-walk the matrix in [`## Safety policy`](#safety-policy). |
| `DOCA_ERROR_NOT_SUPPORTED` | `doca_dma_task_memcpy_set_conf` or first submit | The active `doca_devinfo` does not advertise the memcpy task at all. Re-run `doca_dma_cap_task_memcpy_is_supported(devinfo)`; if false, that is the answer — the user's device cannot offload this copy. |
| `DOCA_ERROR_DRIVER` | Any submit / completion call | The layer below DOCA reported failure (driver / firmware). Capture state and route to env-class debug ([`doca-setup ## debug`](../../doca-setup/TASKS.md#debug)) — the layer below DOCA is the suspect, not the DMA program. |

The agent's rule: **never recommend a retry loop on
`DOCA_ERROR_*` without first identifying which of the rows above
is the cause**. `_AGAIN` is the only one that wants a retry
(after `doca_pe_progress()`); the others want investigation, not
retry.

## Observability

The DOCA Core progress engine (PE) is the single source of
observability for DMA: every memcpy task completion (success or
failure) arrives as an event on the PE. DMA does not maintain
external counters; its observability surface is entirely
event-driven, not poll-driven.

Three primary signals the agent should reach for:

1. **Per-task completion events on the PE.** Every submitted
   `doca_dma_task_memcpy` produces a completion event when it
   finishes (or errors). The completion carries the
   `doca_error_t` if it failed; the agent must inspect the
   per-task completion, not the `doca_task_submit()` return value
   alone.
2. **Capability snapshot at configure time.** The output of every
   `doca_dma_cap_task_memcpy_*` query is a snapshot of *what the
   library said was possible* before any task was submitted.
   Save it as the baseline; if a later submit returns
   `DOCA_ERROR_INVALID_VALUE` or `_NOT_SUPPORTED` the diff
   against this snapshot is the bug.
3. **Lifecycle / state transitions.** Trace-level DOCA logs
   (`DOCA_LOG_LEVEL=trace`) show when the context moved from
   `INIT` to `STARTING` to `RUNNING`. A DMA submit that *appears*
   to silently disappear is almost always the context not being
   in `RUNNING` at submit time.

For the cross-cutting observability primitives
(`--sdk-log-level`, the `doca-<lib>-trace` build flavor, the
`DOCA_LOG_LEVEL` env var) see
[`doca-debug CAPABILITIES.md ## Observability`](../../doca-debug/CAPABILITIES.md#observability).
For the install-tree observability (logger names, package
layout) defer to
[`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md).

## Safety policy

> **Overlay on the bundle-wide hardware-safety meta-policy.** The rules below are this skill's per-artifact overlay on the cross-cutting rules in [`doca-hardware-safety` CAPABILITIES.md ## Safety policy](../../doca-hardware-safety/CAPABILITIES.md#safety-policy) (specifically [### Per-artifact overlay pattern](../../doca-hardware-safety/CAPABILITIES.md#per-artifact-overlay-pattern)). When the two layers disagree, the stricter wins; when either layer says STOP, the agent stops.

DMA's safety surface is **mmap-permission-driven**. The library
moves bytes between two memory regions exposed to the BlueField
DMA engine; an incorrect permission either silently fails the
task or, worse, allows the engine to access memory the user did
not intend to share with the engine.

The **source / destination permission matrix** the agent must
walk for any new DMA setup:

| Region | Required permission flag | Cross-peer rule | Verifies via |
| --- | --- | --- | --- |
| Source `doca_mmap` (local memcpy) | `DOCA_ACCESS_FLAG_LOCAL_READ_ONLY` (at minimum) | n/a | Set via `doca_mmap_set_permissions` before `doca_mmap_start` |
| Source `doca_mmap` (cross-peer copy) | Same as above on the local side; additionally the source must be exported via `doca_mmap_export_*` so the DMA engine on the other peer can read it | The peer side imports the export and uses it as the source `doca_buf` for the memcpy task | Confirmed end-to-end by a successful first memcpy on a small buffer |
| Destination `doca_mmap` | `DOCA_ACCESS_FLAG_LOCAL_READ_WRITE` (at minimum) | The destination is always local to the side submitting the task; no export required | Set via `doca_mmap_set_permissions` before `doca_mmap_start` |

- **The mmap must stay valid until the DMA context is destroyed.**
  Destroying the source or destination mmap before
  `doca_ctx_destroy()` is a use-after-free on the library's
  bookkeeping; symptoms include `DOCA_ERROR_BAD_STATE` from
  subsequent calls and undefined behavior on outstanding tasks.
- **Validate permissions BEFORE the bulk run.** A small-buffer
  smoke memcpy (a few KiB) is the cheapest way to confirm the
  source / destination permission pair is correct. If the smoke
  passes, the permission set is right; any subsequent
  oversize-buffer failure narrows cleanly to the cap-query / size
  axis instead of the permission axis.

## Deferred topic boundaries

This skill scopes itself to the DOCA DMA library. Adjacent topics
the agent will get asked but should route elsewhere:

- **General BlueField DMA-engine internals** (queue arbitration,
  per-channel scheduling, hardware-side error counters) — outside
  this skill. Route to the public DOCA DMA guide reachable
  through
  [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md).
- **DOCA Core context, progress engine, and `doca_mmap`
  internals** — owned by
  [`doca-programming-guide`](../../doca-programming-guide/SKILL.md).
  This skill *uses* the Core context lifecycle and the
  `doca_mmap` permission flags; it does not redefine them.
- **Cross-network copies** — owned by
  [`doca-rdma`](../doca-rdma/SKILL.md). When the user's data
  flow has to traverse IB / RoCE, DMA is not the answer.
- **Producer / consumer message flows between host and DPU** —
  owned by [`doca-comch`](../doca-comch/SKILL.md). When the user
  is moving messages, not raw mmap regions, Comch fast-path is
  the answer.
- **Cross-cutting `DOCA_ERROR_*` taxonomy** — owned by
  [`doca-programming-guide CAPABILITIES.md ## Error taxonomy`](../../doca-programming-guide/CAPABILITIES.md#error-taxonomy).
  This skill adds the DMA overlay, not the taxonomy itself.
- **Cross-cutting debug ladder** (install / version / build /
  link / runtime / program / driver) — owned by
  [`doca-debug ## debug`](../../doca-debug/TASKS.md#debug). This
  skill's `## debug` overlays the runtime + program layers.
