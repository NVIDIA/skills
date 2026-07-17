# DOCA RDMA capabilities, version compatibility, errors, observability, safety

**Where to start:** The pattern overview below names the recurring
RDMA-class patterns. Pick the pattern first, then drill into the H2
that owns the substance. For the *how* of executing each pattern,
jump to [TASKS.md](TASKS.md).

Read this file when the loader sent you here from
[SKILL.md](SKILL.md). For step-by-step workflows that *use* these
capabilities (configure, build, modify, run, test, debug) see
[TASKS.md](TASKS.md). For where the underlying public documentation
and installed package paths live, defer to
[`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md) — do
not duplicate URLs or install paths in this file.

## Pattern overview

Every RDMA-class question this skill teaches resolves into one of
SIX patterns. The patterns are CLASSES — they apply across every
RDMA data-movement use case, not just the worked example shown.

| RDMA pattern | Class shape | Where the substance lives |
| --- | --- | --- |
| 1. Pick the connection method | RDMA CM (callback-driven) vs bridge / OOB vs gRPC; decide before quoting any connection-side code path | [`## Capabilities and modes`](#capabilities-and-modes) connection-method bullet |
| 2. Stand up the context | DOCA Core lifecycle: create → configure (tasks, permissions, properties) → start → connect → use → stop → destroy | [TASKS.md ## configure](TASKS.md#configure) |
| 3. Express the data-movement pattern as a task | Receive / Send / Send-Imm / Read / Write / Write-Imm / Atomic CmpSwp / Atomic FetchAdd / Get-Sync / Set-Sync / Add-Sync — pick by direction (one-sided vs two-sided) and semantics (data vs atomic vs sync-event) | [`## Capabilities and modes`](#capabilities-and-modes) task taxonomy + [TASKS.md ## modify](TASKS.md#modify) |
| 4. Set permissions correctly | mmap permissions + RDMA permissions paired per task type; export the mmap with `doca_mmap_export_rdma()` when the task requires the peer to access it | [`## Safety policy`](#safety-policy) permission matrix + [TASKS.md ## test](TASKS.md#test) |
| 5. Observe what the HW actually did | Per-task completion events on the progress engine + connection state callbacks; `doca_rdma_cap_*` for what was discoverable up front | [`## Observability`](#observability) + [TASKS.md ## debug](TASKS.md#debug) |
| 6. Interpret a `DOCA_ERROR_*` from an RDMA call | Map the error to a layer (env / build / link / runtime / program / driver), then route | [`## Error taxonomy`](#error-taxonomy) RDMA overlay + [TASKS.md ## debug](TASKS.md#debug) |

Two cross-cutting rules that apply to *every* pattern above:

- **Discover the version-installed surface, do not assume.** Every
  pattern above gates on `pkg-config --modversion doca` (the umbrella
  module that ships RDMA — there is normally no separate `doca-rdma.pc`;
  verify with `pkg-config --list-all | grep -i doca`) and on
  the `doca_rdma_cap_*` capability queries against the active
  `doca_devinfo`. Quoting a task type, a transport type, or a
  property value without checking is the most common hallucination
  failure mode.
- **Permissions before payload, every time.** The mmap permission
  + RDMA permission pair must be set on both sides *before* the
  task is submitted, and the mmap exported when the peer needs to
  access it. Skipping the permission set returns `DOCA_ERROR_*`
  that masquerades as a hardware bug.

## Capabilities and modes

DOCA RDMA is a **DOCA Core Context**. Every RDMA instance follows
the universal `cfg-create → cfg-set-* → init → start → use → stop →
destroy` lifecycle (see
[`doca-programming-guide CAPABILITIES.md ## Capabilities and modes`](../../doca-programming-guide/CAPABILITIES.md#capabilities-and-modes)).
On top of that lifecycle, RDMA layers its own connection
state-machine and task model.

**Two distinct, orthogonal axes — link-layer vs transport-type.**
RDMA consists of *two connected sides* passing data between one
another, and the bundle MUST keep the following two axes
separate (the bundle previously conflated them — that conflation
is a real bug):

- **Link-layer (network fabric):** the wire/physical fabric
  the connection runs over — either **InfiniBand (IB)** or
  **Ethernet using RoCE (v1 or v2)**. Link-layer is determined
  by the active `doca_dev`'s port configuration (e.g.
  `mlxconfig` `LINK_TYPE_P*`) and is **not** chosen via a DOCA
  RDMA API; the agent inherits whatever the device exposes.
- **Transport-type (RDMA QP service type):** the per-QP service
  type. The `enum doca_rdma_transport_type` defines exactly two
  values — **RC** (Reliable Connection) and **DC** (Dynamically
  Connected; supported only in the export/connect flow on the
  CPU datapath). There is NO UD (Unreliable Datagram) transport
  in DOCA RDMA. This IS the axis that
  `doca_rdma_set_transport_type()` controls. RC is the baseline;
  DC is restricted to the export/connect + CPU-datapath flow —
  gate task availability on
  `doca_rdma_cap_task_*_is_supported(devinfo)`.

The two axes are independent: RC over IB, RC over RoCE, DC
over IB, DC over RoCE are all valid combinations subject to
device + version caps. Treating "IB / RoCE" as the value the
agent passes to `doca_rdma_set_transport_type()` is a bug —
that setter takes a transport-type (RC or DC), not a
link-layer.

**Device support.** BlueField-2 and higher devices are supported.
On the host any `doca_dev` works; on the BlueField platform,
applications must provide the library with **SFs as a `doca_dev`**
(see the public *OpenvSwitch Acceleration — OVS in DOCA* and
*BlueField DPU Scalable Function* guides, both reachable through
[`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md)).
The exception is RDMA on the DPA datapath, which currently only
supports PFs.

**The eleven task types.** RDMA exposes the following tasks; each
has its own `doca_rdma_task_*_set_conf` (enable the task) and
`doca_rdma_cap_task_*_is_supported` (capability query) entry
points. The agent must call the capability query before assuming
the task type is available on the user's device.

| Task type | Direction | Notes |
| --- | --- | --- |
| Receive | two-sided | Always paired with a Send on the peer; destination buffer must be pre-posted |
| Send | two-sided | Local read-write mmap; peer must have a Receive posted |
| Send with Immediate | two-sided | Same as Send + 4-byte immediate value that surfaces on the peer's Receive completion |
| Read | one-sided | Peer's mmap must be RDMA-readable + exported |
| Write | one-sided | Peer's mmap must be RDMA-writable + exported |
| Write with Immediate | one-sided | Same as Write + immediate value delivered as a peer-side Receive completion |
| Atomic Compare and Swap | one-sided | Peer's mmap must be RDMA-atomic + exported |
| Atomic Fetch and Add | one-sided | Peer's mmap must be RDMA-atomic + exported |
| Get Remote DOCA Sync Event | one-sided | Same permissions as Read |
| Set Remote DOCA Sync Event | one-sided | Same permissions as Write |
| Add Remote DOCA Sync Event | one-sided | Same permissions as Atomic Fetch-and-Add |

**Connection methods.** Three documented ways to connect two RDMA
instances; pick one *before* writing any connection-side code:

| Method | Shape | When to pick |
| --- | --- | --- |
| RDMA CM (`doca_rdma_connect_to_addr()` on the client; `doca_rdma_start_listen_to_port()` + `doca_rdma_connection_accept()` / `_reject()` / `_disconnect()` on the server; plus `doca_rdma_set_connection_state_callbacks()`) | Server/client; non-blocking; event-driven via the progress engine | When the application can run the PE and wants the library to manage handshake retries / timeouts |
| Bridge / OOB (`doca_rdma_bridge_*`) | The user app does the TCP/etc. listen + accept itself, then hands the RDMA CM id to the library | When the application already has its own out-of-band control plane and wants RDMA to *consume* a connection it produced |
| gRPC / arbitrary OOB | The user app exchanges the output of `doca_rdma_export()` over any out-of-band channel and feeds it to `doca_rdma_connect()` | When the application has no RDMA CM presence at all; the channel can be anything (gRPC, file, MPI, …) |

**Multiple connections.** A single RDMA instance can hold multiple
parallel connections. Sizing is via
`doca_rdma_set_max_num_connections()`; per-connection identity is
the connection object returned by the connect / accept call. All
established connections are terminated when the context is stopped.

**Configuration shape.** *Mandatory* configurations before
`doca_ctx_start()`: at least one task type enabled via
`doca_rdma_task_*_set_conf`, and the matching permissions set on
the RDMA instance and the mmap. *Optional* configurations
(queue sizes, max buf-list lengths, transport type, connection
timeouts) use the `doca_rdma_set_*` family with defaults coming
from the library; query the active value with
`doca_rdma_cap_get_*`.

## Version compatibility

For the canonical DOCA version-detection chain, the four-way match rule, NGC container semantics, and the headers-win-over-docs rule, see [`doca-version`](../../doca-version/SKILL.md). The body lives there; this skill does not duplicate it.

**The RDMA-specific overlay** is:

- **DC transport is alpha.** The public DOCA RDMA guide explicitly notes that dynamically-connected (DC) transport support is alpha-level. The agent must surface this when the user asks about DC, not present it as production-stable. Use `pkg-config --modversion doca` as the build-time anchor (per [`doca-version TASKS.md ## configure`](../../doca-version/TASKS.md#configure)) when asked which RDMA features are on this install.
- **Use the `doca_rdma_cap_*` query family at runtime.** Per the cross-cutting rule in [`doca-version CAPABILITIES.md ## Observability`](../../doca-version/CAPABILITIES.md#observability), the cap-query is the runtime authority for *"is this task / transport supported on this device + this DOCA version"*. The version-matrix is the *promise*; the cap query is the *reality*.
- **The resolved DOCA pkg-config module must match `doca_caps --version`** at the four-way-match check (per [`doca-version CAPABILITIES.md ## Version compatibility`](../../doca-version/CAPABILITIES.md#version-compatibility)). On current installs RDMA ships inside the umbrella `doca` module (there is no separate `doca-rdma.pc`); `pkg-config --modversion doca` must agree with `doca_caps --version`. On the rarer split installs that *do* expose per-library `.pc` files, every resolved component `.pc` must agree; a mismatch (one component reporting release *X*, another release *Y*) is the classic partial-install pattern — route to [`doca-version TASKS.md ## debug`](../../doca-version/TASKS.md#debug) ladder step 2 before any RDMA-layer diagnosis.

## Error taxonomy

The cross-library `DOCA_ERROR_*` taxonomy (what each family means
and which debug layer it routes to) lives in
[`doca-programming-guide CAPABILITIES.md ## Error taxonomy`](../../doca-programming-guide/CAPABILITIES.md#error-taxonomy).
The RDMA-specific overlay names the families the agent will see
most often from RDMA calls and what they specifically indicate:

| Family | Most common RDMA cause | First action |
| --- | --- | --- |
| `DOCA_ERROR_BAD_STATE` | Connection state-machine misuse (e.g. `connection_disconnect` on a connection that's already gone); lifecycle out of order | Walk the state machine; confirm the `doca_ctx_state` is what the call expects |
| `DOCA_ERROR_NOT_PERMITTED` | mmap permissions / RDMA permissions don't cover the task type | Re-check the permission matrix in [`## Safety policy`](#safety-policy); confirm the mmap was exported when required |
| `DOCA_ERROR_NOT_SUPPORTED` | The task type / transport type is not supported on this device + DOCA version | Run the matching `doca_rdma_cap_*_is_supported` against the active `doca_devinfo`; if false, that is the answer |
| `DOCA_ERROR_FULL` | Send queue full; bulk-submit exceeded `max_send_buf_list_len * send_queue_size` | Drain completions on the PE before re-submitting; or raise the queue size at configure time |
| `DOCA_ERROR_DRIVER` | The layer below DOCA (driver / firmware) reported failure | Capture `dmesg | tail` and `mlxconfig -d <pcie> q`; route to [`doca-setup ## debug`](../../doca-setup/TASKS.md#debug) layer 5 |
| `DOCA_ERROR_TIME_OUT` | RDMA CM `connection_request_timeout` exceeded | Confirm peer reachability; consider raising the timeout via `doca_rdma_set_connection_request_timeout()` |

Quote `doca_error_get_descr()` verbatim — do not paraphrase. The
cross-cutting debug ladder ([`doca-debug ## debug`](../../doca-debug/TASKS.md#debug))
is the canonical layered diagnosis path that the agent escalates
to once the RDMA-specific cause has been narrowed.

## Observability

The DOCA Core progress engine (PE) is the single source of
observability for RDMA: every task completion (success or failure)
and every connection state transition arrives as an event on the
PE. RDMA does **not** maintain per-pipe counters the way Flow does;
its observability surface is event-driven, not poll-driven.

Three primary signals the agent should reach for:

1. **Task completion events on the PE.** Every submitted task
   produces a completion event when it finishes (or errors). The
   completion carries the `doca_error_t` if it failed; the agent
   must inspect the per-task completion, not the
   `doca_task_submit()` return value alone.
2. **Connection state callbacks.** Registered via
   `doca_rdma_set_connection_state_callbacks()`. The four
   transitions the agent should expect callbacks for: *connection
   request received (server side), connection established (both
   sides), connection failed, connection disconnected*. A
   debugging session without these callbacks wired up is blind to
   half the lifecycle.
3. **Capability snapshot at configure time.** The output of every
   `doca_rdma_cap_*` query is a snapshot of *what the library said
   was possible* before any task was submitted. Save it as the
   baseline; if a task later returns `DOCA_ERROR_NOT_SUPPORTED`
   the diff against this snapshot is the bug.

For cross-cutting observability primitives (`--sdk-log-level`, the
`doca-<lib>-trace` build flavor, the `DOCA_LOG_LEVEL` env var) see
[`doca-debug CAPABILITIES.md ## Observability`](../../doca-debug/CAPABILITIES.md#observability).
For the install-tree observability (logger names, package layout)
defer to
[`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md).

## Safety policy

> **Overlay on the bundle-wide hardware-safety meta-policy.** The rules below are this skill's per-artifact overlay on the cross-cutting rules in [`doca-hardware-safety` CAPABILITIES.md ## Safety policy](../../doca-hardware-safety/CAPABILITIES.md#safety-policy) (specifically [### Per-artifact overlay pattern](../../doca-hardware-safety/CAPABILITIES.md#per-artifact-overlay-pattern)). When the two layers disagree, the stricter wins; when either layer says STOP, the agent stops.

RDMA's safety surface is **permission-driven**. The library sits
on top of memory exposed directly to a remote peer; an incorrect
permission allows the peer to read or write memory the user did
not intend to share, and an over-broad export persists until the
mmap is destroyed.

Per the public RDMA guide:

- **In production environments, ensure RDMA operations are
  performed over a secure channel.** The agent must surface this
  warning when the user asks about production deployment; do not
  paraphrase or omit. The full text lives on
  [docs.nvidia.com/doca/sdk/doca-rdma/index.html](https://docs.nvidia.com/doca/sdk/doca-rdma/index.html).
- **Permission matrix is asymmetric and per-task.** The agent must
  consult the matrix before recommending permission flags:

| Task | Local-side mmap | Local-side RDMA | Peer-side mmap | Peer-side RDMA | Export required |
| --- | --- | --- | --- | --- | --- |
| Read / Get Remote Sync | local read-write | RDMA read | local read-write \| RDMA read | RDMA read | yes (peer exports) |
| Write / Write-Imm / Set Remote Sync | local read-write | RDMA write | local read-write \| RDMA write | RDMA write | yes (peer exports) |
| Atomic CmpSwap / Atomic FetchAdd / Add Remote Sync | local read-write | RDMA atomic | local read-write \| RDMA atomic | RDMA atomic | yes (peer exports) |
| Send / Send-Imm | local read-write | — | local read-write | — | no |
| Receive | depends on the received task | — | local read-write | — | not relevant |

- **The mmap must stay valid until the RDMA context is destroyed.**
  Destroying the mmap before `doca_ctx_destroy()` is a use-after-free
  on the library's bookkeeping; symptoms include
  `DOCA_ERROR_BAD_STATE` from subsequent calls and undefined
  behavior on the peer's outstanding tasks.
- **Validate permissions BEFORE submitting the first task.** The
  cheapest way to validate the matrix is a Send/Receive smoke
  pair: if the bidirectional control path works, the two-sided
  permissions are correct; one-sided permission errors then
  isolate to the Read/Write/Atomic permission set.

## Deferred topic boundaries

This skill scopes itself to the DOCA RDMA library. Adjacent topics
the agent will get asked but should route elsewhere:

- **General RDMA networking concepts** (RoCE vs IB; what a queue
  pair is; what a memory region is in the underlying ibverbs
  layer) — outside this skill. Route to the upstream RDMA / IB
  documentation; this skill assumes the user already understands
  the abstractions and is asking *how to express them through the
  DOCA RDMA API*.
- **DOCA Core context and progress engine internals** — owned by
  [`doca-programming-guide`](../../doca-programming-guide/SKILL.md).
  This skill *uses* the Core context lifecycle; it does not
  redefine it.
- **Cross-cutting `DOCA_ERROR_*` taxonomy** — owned by
  [`doca-programming-guide CAPABILITIES.md ## Error taxonomy`](../../doca-programming-guide/CAPABILITIES.md#error-taxonomy).
  This skill adds the RDMA overlay, not the taxonomy itself.
- **Cross-cutting debug ladder** (install / version / build / link
  / runtime / program / driver) — owned by
  [`doca-debug ## debug`](../../doca-debug/TASKS.md#debug). This
  skill's `## debug` redirects there for layer 1-4 and layer 7;
  layers 5-6 carry the RDMA-specific overlay.
- **Cross-library `doca_caps` invocation patterns** — owned by
  [`doca-caps`](../../tools/doca-caps/SKILL.md). This skill
  references the *RDMA capability query family*
  (`doca_rdma_cap_*`), which is per-library; the *cross-library
  capability snapshot tool* (`doca_caps --list-devs`) is a
  separate surface routed there.
