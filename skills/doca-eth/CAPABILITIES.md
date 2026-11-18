# DOCA Ethernet capabilities, version overlay, errors, observability, safety

**Where to start:** Pick the H2 anchor that matches your question
(queue objects / RX-type taxonomy / TX submission / capabilities /
errors / safety) and read that section end-to-end. The tables in
each section are the load-bearing content; the prose around them
is interpretation.

Read this file when the loader sent you here from
[SKILL.md](SKILL.md). For the *how* of executing each pattern
(the verbs `configure / build / modify / run / test / debug`),
jump to [TASKS.md](TASKS.md). For the canonical DOCA
version-handling rules that this skill layers an Ethernet overlay
on top of, see `doca-version/SKILL.md`. For
the steering side (which Flow rule sends which packet to which
RX queue), defer to `doca-flow/SKILL.md`;
DOCA Ethernet is the *queue* surface, not the *steering*
surface.

## Pattern overview

Every DOCA Ethernet question this skill teaches resolves into one
of FIVE patterns. The patterns are CLASSES — they apply across
every Ethernet release and every host / BlueField pair.

| Pattern | When it applies (class shape) | Where the substance lives |
| --- | --- | --- |
| 1. Pick the queue side | Decide whether the user needs RX, TX, or both; instantiate as separate `doca_eth_rxq` and `doca_eth_txq` contexts (each with its own DOCA Core lifecycle) | [`## Capabilities and modes`](#capabilities-and-modes) RX / TX object table + [TASKS.md ## configure](TASKS.md#configure) step 2 |
| 2. Pick the RX type | Choose one of the **four** real `enum doca_eth_rxq_type` values — `DOCA_ETH_RXQ_TYPE_CYCLIC`, `_MANAGED_MEMPOOL`, `_REGULAR`, `_SHARED_MEMPOOL` — based on per-packet vs ring-buffer vs library-managed vs library-shared-pool data shape; cap-query before assuming the chosen type works on this device | [`## Capabilities and modes`](#capabilities-and-modes) RX-type taxonomy + [TASKS.md ## configure](TASKS.md#configure) step 3 |
| 3. Discover capabilities | Query `doca_eth_rxq_cap_*` / `doca_eth_txq_cap_*` for max burst size, supported RX types, scatter-gather depth, and checksum-offload presence — against the active `doca_devinfo` | [`## Capabilities and modes`](#capabilities-and-modes) capability-query rule + [TASKS.md ## configure](TASKS.md#configure) step 3 |
| 4. Honor preconditions | Verify the device opens under the user's access (sudo or `mlnx` group), the port reports linkup, and traffic is steered to the RX queue (via DOCA Flow or kernel promiscuous mode) | [`## Safety policy`](#safety-policy) precondition matrix + [TASKS.md ## configure](TASKS.md#configure) step 1 |
| 5. Diagnose an Ethernet error | Map symptom (`DOCA_ERROR_BAD_STATE`, `_AGAIN`, `_NOT_PERMITTED`, `_NOT_SUPPORTED`, `_NO_MEMORY`, `_DRIVER`) to root cause without leaving the Ethernet layer prematurely | [`## Error taxonomy`](#error-taxonomy) + [TASKS.md ## debug](TASKS.md#debug) |

Two cross-cutting rules that apply to *every* pattern above:

- **RX and TX are independent contexts.** A `doca_eth_rxq` and a
  `doca_eth_txq` are two separate DOCA Core contexts with two
  separate lifecycles; they share nothing except the underlying
  `doca_dev`. Conflating them — sizing TX queues from RX cap
  queries, or expecting one `doca_ctx_start()` to cover both —
  is the most common first-app design error and the cleanest
  place to fail fast.
- **An RX queue without steering is silent, not broken.** Bringing
  a `doca_eth_rxq` to STARTED state does not by itself cause
  packets to arrive. The user must either program a DOCA Flow
  rule that steers matching packets to this queue, or enable
  kernel-side promiscuous mode on the underlying interface so
  every packet on the wire reaches the queue. The agent must
  surface this before any code change when the user reports
  *"no packets arriving"*.

## Capabilities and modes

The two orthogonal selection axes for any DOCA Ethernet design
are *queue side* (RX vs TX) and, for RX, *RX type* (regular vs
cyclic vs managed-mempool vs shared-mempool). Choose both before
writing any code, then drill into the relevant capability-query.

**Queue object split — RX vs TX.** Each is its own DOCA Core
context; the public header exposes `doca_eth_rxq_as_ctx()` and
`doca_eth_txq_as_ctx()` to obtain the underlying `struct doca_ctx *`.

| Object | What it does | Key calls | Sizing inputs |
| --- | --- | --- | --- |
| `doca_eth_rxq` | Receive queue on a physical port, representor, or SF; surfaces inbound packets to user code as DOCA tasks or library-managed buffers depending on RX type | `doca_eth_rxq_create`, `_set_type`, `_set_cqe_format`, `_set_shared_mempool`, `_set_max_inflight_packets`, `doca_ctx_start` | `doca_eth_rxq_cap_is_type_supported(devinfo, type)`; `doca_eth_rxq_cap_is_cqe_format_supported(devinfo, format)`; `doca_eth_rxq_cap_get_max_inflight_packets(devinfo, &max_inflight_packets)` |
| `doca_eth_txq` | Transmit queue on the same device class; user submits send-tasks that carry a packet `doca_buf` plus the target queue | `doca_eth_txq_create`, `_set_max_send_buf_list_len`, `_set_max_inline_data_size`, `doca_ctx_start`, send-task allocator + submit | `doca_eth_txq_cap_get_max_send_buf_list_len(devinfo)`; `doca_eth_txq_cap_get_max_inline_data_size(devinfo)`; `doca_eth_txq_cap_get_max_inflight_packets(devinfo, max_send_buf_list_len, max_lso_header_size, &max_inflight_packets)` |
| Packet payload | The shipped public surface carries packet bytes as a plain `doca_buf *` attached to the send-task. There is **no** `doca_eth_frame` struct family in the public header — do not invent one. | `doca_buf_*` allocators + the send-task packet setter | — (sized by the user's MTU and scatter-gather plan) |

**RX-type taxonomy.** Picking the right type is data-shape-dependent;
the cap query is the only authority on what the device supports.

| RX type (`enum doca_eth_rxq_type`) | What the user sees | Right shape for | Wrong shape for |
| --- | --- | --- | --- |
| `DOCA_ETH_RXQ_TYPE_REGULAR` (the default) | One DOCA task per inbound packet; the user posts a recv buffer per packet via the recv-task allocator. | Small fan-out, per-packet user logic, debug-friendly first-run path | Line-rate ingress — per-packet task overhead dominates |
| `DOCA_ETH_RXQ_TYPE_CYCLIC` | A ring of fixed-size buffers preallocated by the user; the library writes packets in order; the user reads the ring head. | Max packet-rate ingress with fixed-MTU traffic; GPU packet-processing on a pre-pinned buffer | Variable-sized packets bigger than the ring slot, or workloads that need per-packet completion semantics |
| `DOCA_ETH_RXQ_TYPE_MANAGED_MEMPOOL` | The library allocates and reclaims the receive buffers from an internal mempool; the user just consumes completion events. Completion events expose per-packet metadata through the managed-mempool receive getter families: single-event getters (`doca_eth_rxq_event_managed_mempool_recv_get_flow_tag`, `_get_l3_ok`, `_get_l4_ok`, `_get_metadata_array`, `_get_rx_hash`, `_get_status`, `_get_timestamp`) and batch getters (`doca_eth_rxq_event_batch_managed_mempool_recv_get_flow_tag_array`, `_get_l3_ok_array`, `_get_l4_ok_array`, `_get_metadata_array`, `_get_rx_hash_array`, `_get_timestamp_array`). | Hands-off line-rate ingress with no preallocated buffer plan; lowest-friction first app | Workloads that need to control buffer placement (GPU pinned memory, NUMA-affinity, registered memory regions for downstream RDMA) |
| `DOCA_ETH_RXQ_TYPE_SHARED_MEMPOOL` | The library manages a **shared** memory pool across multiple RX queues / consumers; receive buffers are drawn from the shared pool. See `doca_eth_rxq_shared_mempool.h` for the shared-mempool surface. | Multi-queue ingress where memory must be shared across RX queues; the appropriate type when several `doca_eth_rxq` instances on the same device want a unified buffer plan. | Single-queue ingress (use `_REGULAR` / `_CYCLIC` / `_MANAGED_MEMPOOL` instead) |

**CQE format selection.** The public API exposes the
`DOCA_ETH_RXQ_CQE_FORMAT_*` constant family (regular and
compressed variants) in `doca_eth_rxq.h`. Query support with
`doca_eth_rxq_cap_is_cqe_format_supported(devinfo, format)` and
select the format with `doca_eth_rxq_set_cqe_format(eth_rxq,
format)`. The agent must not assume a compressed format is
available; the capability query is the only authority.

**TX submission shape.** A `doca_eth_txq` submission is always
*one send-task = one packet `doca_buf`*. The `doca_buf` wraps the
packet payload; the send-task carries it onto the queue. The
agent should not invent scatter-gather "list" submission beyond
what `doca_eth_txq_cap_get_max_send_buf_list_len(devinfo)`
reports — that cap is the only authority on whether and how
deep scatter-gather works on the active device.

**Capability discovery — the only rule.** Before sizing any
queue, picking any RX type, or assuming any offload feature,
call the matching `doca_eth_<side>_cap_*` query against the
active `doca_devinfo`:

| Capability | Query | Why the agent must ask |
| --- | --- | --- |
| RX type supported on this device | `doca_eth_rxq_cap_is_type_supported(devinfo, type)` | Not every device supports every RX type; agent must not silently assume managed-mempool or cyclic is present |
| CQE format supported on this RX queue | `doca_eth_rxq_cap_is_cqe_format_supported(devinfo, format)` | Not every device supports every CQE format; agent must not assume compressed formats are present |
| Maximum RX inflight packets | `doca_eth_rxq_cap_get_max_inflight_packets(devinfo, &max_inflight_packets)` | Device-dependent; the agent must query and then set via `doca_eth_rxq_set_max_inflight_packets()` instead of assuming a fixed inflight limit |
| Maximum send scatter-gather list length | `doca_eth_txq_cap_get_max_send_buf_list_len(devinfo)` | Used to size the scatter-gather plan; assuming "1" works everywhere but "N" does not |
| Maximum TX inline data size | `doca_eth_txq_cap_get_max_inline_data_size(devinfo)` | Device-dependent; setting above the cap fails at start |
| Maximum TX inflight packets | `doca_eth_txq_cap_get_max_inflight_packets(devinfo, max_send_buf_list_len, max_lso_header_size, &max_inflight_packets)` | Device-dependent and depends on the send scatter-gather length and LSO header size; the agent must query with the actual planned values rather than assume a fixed limit |

**Configuration shape.** *Mandatory* configurations before
`doca_ctx_start()`: an RX queue needs its type set via
`doca_eth_rxq_set_type()` plus a max inflight packets value at or
below the cap set via `doca_eth_rxq_set_max_inflight_packets()`,
and either a recv-task allocator (regular type), the appropriate
buffer wiring per the chosen type (cyclic or managed-mempool), or
`doca_eth_rxq_set_shared_mempool()` for shared-mempool type; a TX
queue needs its send-task configuration plus, if used, a
scatter-gather length at or below the cap. *Optional*
configurations (advanced flow-tag fields, RSS, CQE format, TX max
inline data size) gate on the matching `_cap_*` query and use the
matching `_set_*` setter.

**DOCA 3.6 API changes.** The create functions `doca_eth_rxq_create`
and `doca_eth_txq_create` now take `max_inflight_packets` instead of
`max_burst_size`; pass the inflight value directly. The GPU proxy and
GPU offload setters now require an explicit enable/disable boolean:
`doca_eth_rxq_gpu_set_uar_on_cpu`,
`doca_eth_txq_gpu_set_completion_on_gpu`,
`doca_eth_txq_gpu_set_uar_on_cpu`, and
`doca_eth_txq_set_wait_on_time_offload` all take an enable flag;
call them with the flag instead of as bare setters. The
`doca_eth_txq_task_batch_lso_send_set_conf` parameter
`num_task_batches` is now `task_batches_num`. Several other parameter
names changed: the `doca_eth_rxq_estimate_packet_buf_size` family now
uses `packet_max_time`, `max_inflight_packets`, and
`log_max_lro_packet_size`; the shared-mempool create function now
takes `max_packet_size` instead of `max_pkt_size`; the shared-mempool
mmap callback type is now `doca_eth_rxq_shared_mempool_alloc_mmap_cb_t`
and its getter/setter parameters are now `alloc_cb`/`alloc_cb_user_data`;
the shared-mempool event batch receive registration now takes event-count bounds (`events_number_max`/`events_number_min`) rather than packet-count bounds; size the batch by event count, not packet count; the GPU handle getters use
`gpu_handle`; and the GPU CPU proxy progress functions use
`eth_rxq`/`eth_txq`. See the installed `doca_eth_rxq.h`,
`doca_eth_txq.h`, and `doca_eth_rxq_shared_mempool.h` headers.

**Renamed in DOCA 3.6 — use the new name.** These entry points still exist
in the installed `doca_eth_rxq.h` / `doca_eth_txq.h` /
`doca_eth_rxq_shared_mempool.h`, under new names:

| DOCA 3.5 name | DOCA 3.6 name |
| --- | --- |
| `doca_eth_rxq_cap_get_max_burst_size` | `doca_eth_rxq_cap_get_max_inflight_packets` |
| `doca_eth_txq_cap_get_max_burst_size` | `doca_eth_txq_cap_get_max_inflight_packets` |
| `doca_eth_rxq_set_max_burst_size` | `doca_eth_rxq_set_max_inflight_packets` |
| `doca_eth_txq_set_max_burst_size` | `doca_eth_txq_set_max_inflight_packets` |
| `doca_eth_txq_cap_is_l3_chksum_offload_supported` | `doca_eth_txq_cap_is_l3_checksum_offload_supported` |
| `doca_eth_txq_cap_is_l4_chksum_offload_supported` | `doca_eth_txq_cap_is_l4_checksum_offload_supported` |
| `doca_eth_txq_set_l3_chksum_offload` | `doca_eth_txq_set_l3_checksum_offload` |
| `doca_eth_txq_set_l4_chksum_offload` | `doca_eth_txq_set_l4_checksum_offload` |
| `doca_eth_txq_cap_get_wait_on_time_offload_supported` | `doca_eth_txq_cap_get_wait_on_time_offload_supported_mode` |
| `doca_eth_rxq_gpu_enable_mcst_qp` | `doca_eth_rxq_gpu_set_mcst_qp` |
| `doca_eth_rxq_shared_mempool_allocate_non_library_buffer` | `doca_eth_rxq_shared_mempool_alloc_non_library_buf` |
| `doca_eth_rxq_shared_mempool_get_non_library_buffers_count` / `_set_` | `doca_eth_rxq_shared_mempool_get_non_library_bufs_count` / `_set_` |
| `doca_eth_rxq_shared_mempool_get_log_pkts_per_wqe_hint` / `_set_` | `doca_eth_rxq_shared_mempool_get_log_packets_per_wqe_hint` / `_set_` |
| `doca_eth_txq_gpu_event_error_send_packet_get_position` | `doca_eth_txq_gpu_event_error_send_packet_get_packet_index` |
| `doca_eth_txq_gpu_event_notify_send_packet_get_position` | `doca_eth_txq_gpu_event_notify_send_packet_get_packet_index` |
| `doca_eth_rxq_set_pkt_buf` | `doca_eth_rxq_set_packet_buf` |
| `doca_eth_rxq_task_recv_as_doca_task`, `doca_eth_txq_task_send_as_doca_task`, `doca_eth_txq_task_lso_send_as_doca_task` | `doca_eth_rxq_task_recv_as_task`, `doca_eth_txq_task_send_as_task`, `doca_eth_txq_task_lso_send_as_task` |
| `doca_eth_rxq_task_recv_get_pkt` / `_set_pkt`, `doca_eth_txq_task_send_get_pkt` / `_set_pkt` | `doca_eth_rxq_task_recv_get_packet` / `_set_packet`, `doca_eth_txq_task_send_get_packet` / `_set_packet` |
| `doca_eth_txq_task_lso_send_get_pkt_payload` / `_set_pkt_payload` | `doca_eth_txq_task_lso_send_get_packet_payload` / `_set_packet_payload` |
| `doca_eth_txq_task_send_num_expand`, `doca_eth_txq_task_lso_send_num_expand`, `doca_eth_txq_task_batch_send_num_expand`, `doca_eth_txq_task_batch_lso_send_num_expand` | `doca_eth_txq_task_send_expand_num`, `doca_eth_txq_task_lso_send_expand_num`, `doca_eth_txq_task_batch_send_expand_num`, `doca_eth_txq_task_batch_lso_send_expand_num` |
| `DOCA_ETH_TXQ_OL_FLAGS_L3_CSUM` / `DOCA_ETH_TXQ_OL_FLAGS_L4_CSUM` | `DOCA_ETH_TXQ_OL_FLAG_L3_CHECKSUM` / `DOCA_ETH_TXQ_OL_FLAG_L4_CHECKSUM` |

Check each signature in the installed header before use; renamed functions
may also have renamed parameters (see the parameter changes above).

**Removed in DOCA 3.6 — do not call.** No longer declared in the installed
headers:

- The managed-receive event getter/registration family
  (`doca_eth_rxq_event_managed_recv_*`): there is no managed-recv
  RX type in the current API; use `DOCA_ETH_RXQ_TYPE_MANAGED_MEMPOOL`
  and its `doca_eth_rxq_event_managed_mempool_recv_*` families instead.

**Task-based send/receive is still available in DOCA 3.6.** The receive,
send, LSO send, batch send and batch LSO send task families are declared in
`doca_eth_rxq_cpu_data_path.h` / `doca_eth_txq_cpu_data_path.h`; their allocators are now named
`doca_eth_rxq_task_recv_alloc_init`, `doca_eth_txq_task_send_alloc_init`,
`doca_eth_txq_task_lso_send_alloc_init`, `doca_eth_txq_task_batch_send_alloc`
and `doca_eth_txq_task_batch_lso_send_alloc` (3.5 used `_allocate_init` /
`_allocate`), and `doca_eth_txq_task_batch_lso_send_set_conf` takes
`task_batches_num` (was `num_task_batches`). Check each signature in the
installed header.

Developers must use the current DOCA Ethernet public header
installed with the package and must not call the removed entry points
listed above.

## Version compatibility

For the canonical DOCA version-detection chain, the four-way
match rule, NGC container semantics, and the
headers-win-over-docs rule, see
`doca-version/SKILL.md`. The body lives
there; this skill does not duplicate it.

**The Ethernet-specific overlay** is:

- **RX-type availability and `doca_eth_*_cap_*` are the runtime
  authority, not the public docs.** Per the cross-cutting
  cap-query rule in
  `doca-version CAPABILITIES.md ## Observability` (`doca-version/CAPABILITIES.md#observability`),
  the agent must call `doca_eth_rxq_cap_is_type_supported(devinfo, type)`
  before promising the user that managed-recv or cyclic is on
  this device + DOCA version; checksum offload likewise gates on
  the current TXQ checksum-offload capability query in the public
  DOCA Ethernet header.
  Quoting an RX-type or offload from memory is the canonical
  hallucination failure mode for this library.
- **`doca-eth.pc` and `doca-common.pc` must both match
  `doca_caps --version`** at the four-way-match check (per
  `doca-version CAPABILITIES.md ## Version compatibility` (`doca-version/CAPABILITIES.md#version-compatibility`)).
  Use `pkg-config --modversion doca-eth` as the build-time
  anchor; disagreement with `doca_caps --version` is a
  partial-install hazard and must be routed to
  `doca-version TASKS.md ## debug` (`doca-version/TASKS.md#debug`)
  layer 2 before any Ethernet-layer diagnosis.
- **DOCA 3.5 added shared-mempool RXQ, CQE format selection, and TX max inline data size APIs; earlier releases do not provide them.** The agent must call `doca_eth_rxq_cap_is_cqe_format_supported(devinfo, format)` with the desired format from the `DOCA_ETH_RXQ_CQE_FORMAT_*` family before promising a CQE format, `doca_eth_rxq_cap_is_type_supported(devinfo, type)` with the shared-mempool RXQ type from the `DOCA_ETH_RXQ_TYPE_*` family before promising shared-mempool RXQ, and `doca_eth_txq_cap_get_max_inline_data_size(devinfo, &max_inline_data_size)` before promising TX inline data size. See the public DOCA Ethernet header for the constant families. Quoting these from memory is the same hallucination failure mode.
- **DOCA 3.6 renamed many Ethernet entry points and removed one family; there is no compatibility shim.** Code written against 3.5 must use the 3.6 names in the renamed table above (checksum offload, inflight-packet limits, shared-mempool non-library buffers and WQE hint, GPU multicast QP, GPU send-event index, task allocators) and must drop the removed event-managed-receive family (`doca_eth_rxq_set_pkt_buf` is now `doca_eth_rxq_set_packet_buf`). Task-based I/O and the queue context conversion (`doca_eth_rxq_as_ctx` / `doca_eth_txq_as_ctx`) remain. Quoting 3.5 names from memory is the same hallucination failure mode.

## Error taxonomy

Ethernet-specific overlays on the cross-library `DOCA_ERROR_*`
taxonomy. The cross-library taxonomy itself lives in
`doca-programming-guide CAPABILITIES.md ## Error taxonomy` (`doca-programming-guide/CAPABILITIES.md#error-taxonomy`);
the rows below are the *Ethernet surface* meaning that the
agent must disambiguate before falling back to the cross-library
response.

| Error | DOCA Ethernet context where it shows up | Ethernet-specific cause |
| --- | --- | --- |
| `DOCA_ERROR_BAD_STATE` | Any call after `doca_ctx_stop()` or before `doca_ctx_start()`; submitting a send-task on a `doca_eth_txq` that is not STARTED; posting a recv buffer on a `doca_eth_rxq` of the wrong type | Lifecycle violation. Walk the queue's own state against the universal lifecycle in `doca-programming-guide CAPABILITIES.md ## Capabilities and modes` (`doca-programming-guide/CAPABILITIES.md#capabilities-and-modes`); the most common case is treating RX and TX as one context. |
| `DOCA_ERROR_NOT_SUPPORTED` | `doca_eth_rxq_set_type` with a type the device does not advertise; `doca_eth_txq_set_*` for L3 checksum offload on a device that does not support it; oversized burst | Re-run the matching `doca_eth_<side>_cap_*` query against the active `doca_devinfo`; if the cap query says false, that is the answer — the user's device or DOCA version does not support the request. |
| `DOCA_ERROR_NOT_PERMITTED` | `doca_dev_open` for a port or representor that the user has no access to; queue create after a permission downgrade | The device was not opened with the required access. Confirm sudo or `mlnx`-group membership per [`## Safety policy`](#safety-policy); do not modify the program. |
| `DOCA_ERROR_AGAIN` | `doca_task_submit` for a TX send-task when the queue is full | The send queue is full. This is *not* a hardware error; drain completions via `doca_pe_progress()` before re-submitting. Same as the cross-library *"would-block, retry after progress"* pattern. |
| `DOCA_ERROR_NO_MEMORY` | RX recv-task allocator or managed-recv internal buffer pool; large TX scatter-gather attempts | The descriptor or buffer pool is exhausted. The fix is to raise queue / pool sizing (within the cap) or to drain inbound completions faster, not to retry the allocation in a tight loop. |
| `DOCA_ERROR_DRIVER` | Any submit / completion call | The layer below DOCA reported failure. Capture state (`dmesg | tail`, `mlxconfig -d <pcie> q`) and route to env-class debug (`doca-setup ## debug` (`doca-setup/TASKS.md#debug`)) — the layer below DOCA is the suspect, not the program. |
| Any non-success `doca_error_t` from the managed-mempool receive event status getter | Inside the success event handler for a managed-mempool receive event | The event status getter (`doca_eth_rxq_event_managed_mempool_recv_get_status` in the DOCA Ethernet public header) reports per-packet receive errors that the success/error handler split does not capture. Check it before consuming packet data; treat a non-success value as an Ethernet receive error, not as a reason to retry the registration. |

The agent's rule: **never recommend a retry loop on
`DOCA_ERROR_*` without first identifying which of the rows
above is the cause**. `_AGAIN` is the only one that wants a
retry (after `doca_pe_progress()`); the others want
investigation, not retry. For managed-mempool receive events,
the success/error handler split is not the full error surface:
check the event status getter before applying the rows above.
And *"no packets arriving"* is
**never** a `DOCA_ERROR_*` — it is the steering side; route to
`doca-flow/SKILL.md` or to the
promiscuous-mode workaround in
`doca-setup/SKILL.md`, not to the Ethernet
error taxonomy.

## Observability

The DOCA Core progress engine (PE) is the single source of
observability for each Ethernet queue: every recv-task or
send-task completion (success or failure) arrives as an event on
the PE that the queue's context is registered against. DOCA
Ethernet does **not** maintain per-queue counters of its own;
its observability surface is event-driven, not poll-driven, with
two add-on signals (capability snapshot + port-state).

Three primary signals the agent should reach for:

1. **Per-queue task completion events on the PE.** Each RX queue
   produces a completion event on every received packet (regular
   type) or per ring-buffer slot consumed (cyclic / managed-recv,
   shape-specific). Each TX queue produces a completion event on
   every send-task that finishes (success or error). Absence of
   completions on a started queue is *always* either a missing
   `doca_pe_progress()` call or — for RX — a steering / port-
   state precondition gap. Route to [`## Safety policy`](#safety-policy)
   first. For RX event payloads, the per-packet flow tag, L3/L4
   OK, metadata, RX hash, and timestamp are available through the
   `doca_eth_rxq_event_batch_managed_mempool_recv_get_*_array`
   family for batch receive and the
   `doca_eth_rxq_event_managed_mempool_recv_get_*` scalar family
   for single-event receive; see the DOCA Ethernet public header.
2. **Capability snapshot at configure time.** The output of every
   `doca_eth_rxq_cap_*` / `doca_eth_txq_cap_*` query is a
   snapshot of *what the library said was possible* before any
   task was submitted. Save it as the baseline; if a task later
   returns `DOCA_ERROR_NOT_SUPPORTED` the diff against this
   snapshot is the bug.
3. **Port and link state from the env side.** `devlink dev show`,
   `ip -j link show <dev>`, and `ethtool <dev>` together tell
   the agent whether the port the user opened is up at the
   driver layer at all. A linkdown port is a silent ingress
   queue; the agent must check it before any Ethernet-layer
   diagnosis. Defer to
   `doca-setup CAPABILITIES.md ## Observability` (`doca-setup/CAPABILITIES.md#observability`)
   for the env-side observability primitives.

For the cross-library debug-time observability
(`DOCA_LOG_LEVEL=trace`, `--sdk-log-level`, the trace build
flavor) see
`doca-debug CAPABILITIES.md ## Observability` (`doca-debug/CAPABILITIES.md#observability`).

## Safety policy

> **Overlay on the bundle-wide hardware-safety meta-policy.** The rules below are this skill's per-artifact overlay on the cross-cutting rules in `doca-hardware-safety` CAPABILITIES.md ## Safety policy (`doca-hardware-safety/CAPABILITIES.md#safety-policy`) (specifically ### Per-artifact overlay pattern (`doca-hardware-safety/CAPABILITIES.md#per-artifact-overlay-pattern`)). When the two layers disagree, the stricter wins; when either layer says STOP, the agent stops.

DOCA Ethernet's safety surface is **access, port state, and
steering**. The single most common first-app failure is *"my
queue is up but no packets arrive"* — and the agent's job is to
verify the three preconditions before any code change, not after
the first DOCA_ERROR_* (because there often is no error; an
empty RX queue is silent).

The **precondition matrix** the agent must walk for any new
DOCA Ethernet setup:

| Precondition | What must be true before `doca_ctx_start()` | How the agent verifies | Where to fix |
| --- | --- | --- | --- |
| Device access | The `doca_dev` was opened against a port / representor / SF the user has permission to use (typically requires sudo or `mlnx`-group membership) | `id` for group membership; the open call failing with `DOCA_ERROR_NOT_PERMITTED` is the runtime symptom | `doca-setup/SKILL.md` for the env-side; do not modify the program |
| Port up | The underlying port reports linkup at the driver layer (`devlink dev show` reports `state: PORT_ACTIVE`; `ip link` shows the device `UP,LOWER_UP`) | `devlink dev show`; `ip -j link show <dev>`; `ethtool <dev>` | `doca-setup CAPABILITIES.md ## Capabilities and modes` (`doca-setup/CAPABILITIES.md#capabilities-and-modes`) port-bring-up; do not modify the program |
| Traffic actually reaches the RX queue | Either a DOCA Flow rule steers matching packets to this `doca_eth_rxq`, OR the underlying interface is in promiscuous mode at the kernel layer so every packet on the wire reaches the queue | Inspect the flow rule programmed for this queue (or the absence of one); `ip link show <dev>` reports the `PROMISC` flag if promiscuous mode is on | `doca-flow/SKILL.md` for the steering side (canonical); `doca-setup/SKILL.md` for the promiscuous-mode workaround (expedient first-run path only) |

**RX and TX queues are independent — do not share lifecycles.**
A `doca_eth_rxq` and a `doca_eth_txq` each have their own
`doca_ctx_*` lifecycle. Treating them as one context (one
start, one stop) is caught by the runtime as
`DOCA_ERROR_BAD_STATE` on whichever side was skipped; the
user-visible symptom does not name the conflation.

**Smoke before scale-up.** Before driving traffic at line rate,
the agent must walk the user through a single-packet smoke
(one TX submission → loopback or external echo → one RX
completion). A failure here narrows cleanly: TX-side, network,
or RX-side. A failure at scale-up *without* the smoke pass is a
much harder bisection.
