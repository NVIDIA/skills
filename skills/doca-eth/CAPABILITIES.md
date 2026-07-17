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
on top of, see [`doca-version`](../../doca-version/SKILL.md). For
the steering side (which Flow rule sends which packet to which
RX queue), defer to [`doca-flow`](../doca-flow/SKILL.md);
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
cyclic vs managed-recv). Choose both before writing any code,
then drill into the relevant capability-query.

**Queue object split — RX vs TX.** Each is its own DOCA Core
context.

| Object | What it does | Key calls | Sizing inputs |
| --- | --- | --- | --- |
| `doca_eth_rxq` | Receive queue on a physical port, representor, or SF; surfaces inbound packets to user code as DOCA tasks or library-managed buffers depending on RX type | `doca_eth_rxq_create`, `_set_type`, `_set_max_burst_size`, `doca_ctx_start` | `doca_eth_rxq_cap_get_max_burst_size(devinfo)`; `doca_eth_rxq_cap_is_type_supported(devinfo, type)` |
| `doca_eth_txq` | Transmit queue on the same device class; user submits send-tasks (`doca_eth_txq_task_send` for regular packets, `doca_eth_txq_task_lso_send` for LSO) that carry a packet `doca_buf` (`doca_eth_txq_task_send_set_pkt`) or a payload `doca_buf` + headers `doca_buf` (`doca_eth_txq_task_lso_send_set_pkt_payload` + `_set_headers`) plus the target queue | `doca_eth_txq_create`, `_set_max_send_buf_list_len`, `doca_ctx_start`, send-task allocator + submit | `doca_eth_txq_cap_get_max_send_buf_list_len(devinfo)`; `doca_eth_txq_cap_is_l3_chksum_offload_supported(devinfo)` |
| Packet payload | The shipped public surface carries packet bytes as a plain `doca_buf *` attached to the send-task via the setters above. There is **no** `doca_eth_frame` struct family in the public header — do not invent one. | `doca_buf_*` allocators + `doca_eth_txq_task_send_set_pkt` (or `_lso_send_set_pkt_payload` + `_set_headers`) | — (sized by the user's MTU and scatter-gather plan) |

**RX-type taxonomy.** Picking the right type is data-shape-dependent;
the cap query is the only authority on what the device supports.

| RX type (`enum doca_eth_rxq_type`) | What the user sees | Right shape for | Wrong shape for |
| --- | --- | --- | --- |
| `DOCA_ETH_RXQ_TYPE_REGULAR` (the default) | One DOCA task per inbound packet; the user posts a recv buffer per packet via the recv-task allocator | Small fan-out, per-packet user logic, debug-friendly first-run path | Line-rate ingress — per-packet task overhead dominates |
| `DOCA_ETH_RXQ_TYPE_CYCLIC` | A ring of fixed-size buffers preallocated by the user; the library writes packets in order; the user reads the ring head | Max packet-rate ingress with fixed-MTU traffic; GPU packet-processing on a pre-pinned buffer | Variable-sized packets bigger than the ring slot, or workloads that need per-packet completion semantics |
| `DOCA_ETH_RXQ_TYPE_MANAGED_MEMPOOL` | The library allocates and reclaims the receive buffers from an internal mempool; the user just consumes completion events | Hands-off line-rate ingress with no preallocated buffer plan; lowest-friction first app | Workloads that need to control buffer placement (GPU pinned memory, NUMA-affinity, registered memory regions for downstream RDMA) |
| `DOCA_ETH_RXQ_TYPE_SHARED_MEMPOOL` | The library manages a **shared** memory pool across multiple RX queues / consumers; receive buffers are drawn from the shared pool. Bundle previously omitted this fourth real RX-type. | Multi-queue ingress where memory must be shared across RX queues; the appropriate type when several `doca_eth_rxq` instances on the same device want a unified buffer plan. See `doca_eth_rxq_shared_mempool.h` for the shared-mempool surface. | Single-queue ingress (use `_REGULAR` / `_CYCLIC` / `_MANAGED_MEMPOOL` instead) |

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
| Maximum RX burst size | `doca_eth_rxq_cap_get_max_burst_size(devinfo)` | Device-dependent; oversizing returns `DOCA_ERROR_NOT_SUPPORTED` at `doca_ctx_start()` |
| RX type supported on this device | `doca_eth_rxq_cap_is_type_supported(devinfo, type)` | Not every device supports every RX type; agent must not silently assume managed-recv or cyclic is present |
| Maximum send scatter-gather list length | `doca_eth_txq_cap_get_max_send_buf_list_len(devinfo)` | Used to size the scatter-gather plan; assuming "1" works everywhere but "N" does not |
| L3 checksum offload presence | `doca_eth_txq_cap_is_l3_chksum_offload_supported(devinfo)` | Offload is device-conditional; requesting it on a device that does not advertise it fails at start |

**Configuration shape.** *Mandatory* configurations before
`doca_ctx_start()`: an RX queue needs its type set via
`doca_eth_rxq_set_type()` plus a burst size at or below the
cap, and either a recv-task allocator (regular type) or the
appropriate buffer wiring per the chosen type; a TX queue needs
its send-task configuration plus, if used, a scatter-gather
length at or below the cap. *Optional* configurations
(checksum offload, advanced flow-tag fields, RSS) gate on the
matching `_cap_*` query and use the matching `_set_*` setter.

## Version compatibility

For the canonical DOCA version-detection chain, the four-way
match rule, NGC container semantics, and the
headers-win-over-docs rule, see
[`doca-version`](../../doca-version/SKILL.md). The body lives
there; this skill does not duplicate it.

**The Ethernet-specific overlay** is:

- **RX-type availability and `doca_eth_*_cap_*` are the runtime
  authority, not the public docs.** Per the cross-cutting
  cap-query rule in
  [`doca-version CAPABILITIES.md ## Observability`](../../doca-version/CAPABILITIES.md#observability),
  the agent must call `doca_eth_rxq_cap_is_type_supported(devinfo, type)`
  before promising the user that managed-recv or cyclic is on
  this device + DOCA version; checksum offload likewise gates on
  `doca_eth_txq_cap_is_l3_chksum_offload_supported(devinfo)`.
  Quoting an RX-type or offload from memory is the canonical
  hallucination failure mode for this library.
- **`doca-eth.pc` and `doca-common.pc` must both match
  `doca_caps --version`** at the four-way-match check (per
  [`doca-version CAPABILITIES.md ## Version compatibility`](../../doca-version/CAPABILITIES.md#version-compatibility)).
  Use `pkg-config --modversion doca-eth` as the build-time
  anchor; disagreement with `doca_caps --version` is a
  partial-install hazard and must be routed to
  [`doca-version TASKS.md ## debug`](../../doca-version/TASKS.md#debug)
  layer 2 before any Ethernet-layer diagnosis.

## Error taxonomy

Ethernet-specific overlays on the cross-library `DOCA_ERROR_*`
taxonomy. The cross-library taxonomy itself lives in
[`doca-programming-guide CAPABILITIES.md ## Error taxonomy`](../../doca-programming-guide/CAPABILITIES.md#error-taxonomy);
the rows below are the *Ethernet surface* meaning that the
agent must disambiguate before falling back to the cross-library
response.

| Error | DOCA Ethernet context where it shows up | Ethernet-specific cause |
| --- | --- | --- |
| `DOCA_ERROR_BAD_STATE` | Any call after `doca_ctx_stop()` or before `doca_ctx_start()`; submitting a send-task on a `doca_eth_txq` that is not STARTED; posting a recv buffer on a `doca_eth_rxq` of the wrong type | Lifecycle violation. Walk the queue's own state against the universal lifecycle in [`doca-programming-guide CAPABILITIES.md ## Capabilities and modes`](../../doca-programming-guide/CAPABILITIES.md#capabilities-and-modes); the most common case is treating RX and TX as one context. |
| `DOCA_ERROR_NOT_SUPPORTED` | `doca_eth_rxq_set_type` with a type the device does not advertise; `doca_eth_txq_set_*` for L3 checksum offload on a device that does not support it; oversized burst | Re-run the matching `doca_eth_<side>_cap_*` query against the active `doca_devinfo`; if the cap query says false, that is the answer — the user's device or DOCA version does not support the request. |
| `DOCA_ERROR_NOT_PERMITTED` | `doca_dev_open` for a port or representor that the user has no access to; queue create after a permission downgrade | The device was not opened with the required access. Confirm sudo or `mlnx`-group membership per [`## Safety policy`](#safety-policy); do not modify the program. |
| `DOCA_ERROR_AGAIN` | `doca_task_submit` for a TX send-task when the queue is full | The send queue is full. This is *not* a hardware error; drain completions via `doca_pe_progress()` before re-submitting. Same as the cross-library *"would-block, retry after progress"* pattern. |
| `DOCA_ERROR_NO_MEMORY` | RX recv-task allocator or managed-recv internal buffer pool; large TX scatter-gather attempts | The descriptor or buffer pool is exhausted. The fix is to raise queue / pool sizing (within the cap) or to drain inbound completions faster, not to retry the allocation in a tight loop. |
| `DOCA_ERROR_DRIVER` | Any submit / completion call | The layer below DOCA reported failure. Capture state (`dmesg | tail`, `mlxconfig -d <pcie> q`) and route to env-class debug ([`doca-setup ## debug`](../../doca-setup/TASKS.md#debug)) — the layer below DOCA is the suspect, not the program. |

The agent's rule: **never recommend a retry loop on
`DOCA_ERROR_*` without first identifying which of the rows
above is the cause**. `_AGAIN` is the only one that wants a
retry (after `doca_pe_progress()`); the others want
investigation, not retry. And *"no packets arriving"* is
**never** a `DOCA_ERROR_*` — it is the steering side; route to
[`doca-flow`](../doca-flow/SKILL.md) or to the
promiscuous-mode workaround in
[`doca-setup`](../../doca-setup/SKILL.md), not to the Ethernet
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
   first.
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
   [`doca-setup CAPABILITIES.md ## Observability`](../../doca-setup/CAPABILITIES.md#observability)
   for the env-side observability primitives.

For the cross-library debug-time observability
(`DOCA_LOG_LEVEL=trace`, `--sdk-log-level`, the trace build
flavor) see
[`doca-debug CAPABILITIES.md ## Observability`](../../doca-debug/CAPABILITIES.md#observability).

## Safety policy

> **Overlay on the bundle-wide hardware-safety meta-policy.** The rules below are this skill's per-artifact overlay on the cross-cutting rules in [`doca-hardware-safety` CAPABILITIES.md ## Safety policy](../../doca-hardware-safety/CAPABILITIES.md#safety-policy) (specifically [### Per-artifact overlay pattern](../../doca-hardware-safety/CAPABILITIES.md#per-artifact-overlay-pattern)). When the two layers disagree, the stricter wins; when either layer says STOP, the agent stops.

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
| Device access | The `doca_dev` was opened against a port / representor / SF the user has permission to use (typically requires sudo or `mlnx`-group membership) | `id` for group membership; the open call failing with `DOCA_ERROR_NOT_PERMITTED` is the runtime symptom | [`doca-setup`](../../doca-setup/SKILL.md) for the env-side; do not modify the program |
| Port up | The underlying port reports linkup at the driver layer (`devlink dev show` reports `state: PORT_ACTIVE`; `ip link` shows the device `UP,LOWER_UP`) | `devlink dev show`; `ip -j link show <dev>`; `ethtool <dev>` | [`doca-setup CAPABILITIES.md ## Capabilities and modes`](../../doca-setup/CAPABILITIES.md#capabilities-and-modes) port-bring-up; do not modify the program |
| Traffic actually reaches the RX queue | Either a DOCA Flow rule steers matching packets to this `doca_eth_rxq`, OR the underlying interface is in promiscuous mode at the kernel layer so every packet on the wire reaches the queue | Inspect the flow rule programmed for this queue (or the absence of one); `ip link show <dev>` reports the `PROMISC` flag if promiscuous mode is on | [`doca-flow`](../doca-flow/SKILL.md) for the steering side (canonical); [`doca-setup`](../../doca-setup/SKILL.md) for the promiscuous-mode workaround (expedient first-run path only) |

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
