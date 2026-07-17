# DOCA STA capabilities, version overlay, errors, observability, safety

**Where to start:** Pick the H2 anchor that matches your question
(target object model / queue-pair shape / RDMA transport /
capabilities / errors / safety) and read that section end-to-end.
The tables in each section are the load-bearing content; the prose
around them is interpretation.

Read this file when the loader sent you here from
[SKILL.md](SKILL.md). For the *how* of executing each pattern (the
verbs `configure / build / modify / run / test / debug`), jump to
[TASKS.md](TASKS.md). For the canonical DOCA version-handling
rules that this skill layers a STA overlay on top of, see
[`doca-version`](../../doca-version/SKILL.md). For the RDMA
substrate that NVMe-over-RDMA transport lands on, see
[`doca-rdma`](../doca-rdma/SKILL.md). For the steering side
that decides which NVMe-oF packets land on which STA-managed
queue, defer to [`doca-flow`](../doca-flow/SKILL.md).

## Pattern overview

Every DOCA STA question this skill teaches resolves into one of
SIX patterns. The patterns are CLASSES — they apply across every
NVMe-over-Fabrics target deployment shape (single-subsystem smoke
vs multi-subsystem fan-out, single namespace vs many, one backend
disk vs several), not just the worked example shown.

| Pattern | When it applies (class shape) | Where the substance lives |
| --- | --- | --- |
| 1. Define the target object model | Lay out the NVMe-oF target: one or more `doca_sta_subsystem` (NQN) resources, their namespaces, and the `doca_sta_be` backend controllers (local NVMe-PCI disks) that back them; doca-sta accelerates the *target* data path | [`## Capabilities and modes`](#capabilities-and-modes) target-object table + [TASKS.md ## configure](TASKS.md#configure) step 1 |
| 2. Stand up the `doca_sta` context | DOCA Core lifecycle: create → configure (add devices, queue-pair sizing, subsystems/namespaces/backends) → start → use → stop → destroy, on the underlying `doca_dev` | [TASKS.md ## configure](TASKS.md#configure) + [`doca-programming-guide CAPABILITIES.md ## Capabilities and modes`](../../doca-programming-guide/CAPABILITIES.md#capabilities-and-modes) for the universal lifecycle |
| 3. Confirm device support (RDMA transport) | STA transport is RDMA-only (RDMA CM); the single capability gate is `doca_sta_cap_is_supported` against the active `doca_devinfo`, which is the only authority on whether this device can accelerate an STA target at all | [`## Capabilities and modes`](#capabilities-and-modes) transport section + [TASKS.md ## configure](TASKS.md#configure) step 3 |
| 4. Shape the NVMe queue pair | One admin queue plus N I/O queues per NVMe-oF connection; size N (number of queue pairs), the depth per queue, and the I/O size against what the `doca_sta_get_max_*` queries report | [`## Capabilities and modes`](#capabilities-and-modes) queue-pair table + [TASKS.md ## configure](TASKS.md#configure) step 4 |
| 5. Honor substrate and steering preconditions | The RDMA transport needs `doca-rdma` discoverable and the RDMA cap surface non-empty on the chosen device; NVMe-oF traffic only reaches the STA-managed queues when DOCA Flow rules (or the env-side equivalent) steer it there | [`## Safety policy`](#safety-policy) precondition matrix + [TASKS.md ## configure](TASKS.md#configure) step 1 |
| 6. Diagnose a STA error | Map symptom (`DOCA_ERROR_BAD_STATE`, `_NOT_SUPPORTED`, `_INVALID_VALUE`, `_AGAIN`, `_IO_FAILED`, `_NOT_PERMITTED`) to root cause without leaving the STA layer prematurely | [`## Error taxonomy`](#error-taxonomy) + [TASKS.md ## debug](TASKS.md#debug) |

Two cross-cutting rules that apply to *every* pattern above:

- **doca-sta is target-side acceleration, not an initiator stack.**
  doca-sta presents NVMe-oF target subsystems (NQN + namespaces)
  backed by local NVMe-PCI disks and accelerates the target-side
  data path over RDMA; it does not implement an NVMe-oF initiator /
  host, and the remote initiator (SPDK `bdev_nvme`, kernel `nvme`
  host, …) is out of scope. An agent that recommends doca-sta as an
  initiator transport provider is misleading the user.
- **Discover the version-installed surface, do not assume.** Every
  pattern above gates on `pkg-config --modversion doca-sta`, on the
  `doca_sta_cap_is_supported` device check, and on the
  `doca_sta_get_max_*` sizing queries against the active
  `doca_devinfo`. Quoting an I/O queue depth, a queue count, or
  an NVMe-oF sizing value without checking is the most common
  hallucination failure mode.

## Capabilities and modes

DOCA STA is a **DOCA Core context**. Every STA instance follows
the universal `cfg-create → cfg-set-* → init → start → use → stop
→ destroy` lifecycle (see
[`doca-programming-guide CAPABILITIES.md ## Capabilities and modes`](../../doca-programming-guide/CAPABILITIES.md#capabilities-and-modes)).
On top of that lifecycle, STA layers a target object model, the
RDMA transport, and a queue-pair shape.

**Target object model — what doca-sta presents.** doca-sta is
target-side acceleration: the BlueField presents one or more
NVMe-oF target subsystems, backed by local NVMe-PCI disks, to
remote initiators over RDMA. The single most important framing
the agent should surface before any code:

| Object | Created by | What it is |
| --- | --- | --- |
| `doca_sta` context | `doca_sta_create()` on a `doca_dev` | The STA engine instance; additional devices are added with `doca_sta_add_dev()` |
| `doca_sta_subsystem` | `doca_sta_subsystem_create()` (takes an NQN) | An NVMe subsystem — the target identity remote initiators connect to; holds namespaces and is bound to one or more network devices |
| Namespace | `doca_sta_subsystem_add_ns()` | A namespace exposed by a subsystem, with a logical block size validated by `doca_sta_is_logical_block_size_supported()` |
| `doca_sta_be` backend controller | `doca_sta_be_create()` | An abstraction of a backend device — a local NVMe-PCI disk that stores the namespace data |
| `doca_sta_io` + IO QP | the STA IO QP API (`doca_sta_io_qp.h`) | A per-EU IO context carrying the RDMA queue pairs that remote initiators connect into |

The remote initiator (the NVMe-oF host) and its NVMe stack are
out of scope: doca-sta does not implement initiator logic, and
there is no NVMe-over-TCP path — the only transport is RDMA.

**Transport — RDMA only.** STA's transport is NVMe-over-RDMA on
the `doca-rdma` substrate; connections are established through
RDMA CM (the IO QP is moved to connected on the
`RDMA_CM_EVENT_ESTABLISHED` event via
`doca_sta_io_qp_connect_established()`, per `doca_sta_io_qp.h`).
There is **no NVMe-over-TCP transport** in the public API. The
single device gate is `doca_sta_cap_is_supported`.

The agent's rule: **never promise STA acceleration without naming
the cap check.** Run `doca_sta_cap_is_supported` against the
active `doca_devinfo`; quote the queried result. Do not assume
from the docs page.

**NVMe queue-pair shape.** Each NVMe-oF *connection* (initiator
↔ target pair) carries:

| Queue | Count per connection | What it carries |
| --- | --- | --- |
| Admin queue | exactly 1 | NVMe admin commands (Identify, Set/Get Features, Connect, Discovery, …) plus the NVMe-oF Connect handshake itself |
| I/O queue | configurable, up to the device cap | NVMe Read / Write / Flush / Dataset Management I/O commands |

Sizing inputs (each gates on the matching `doca_sta_get_max_*`
query — these are distinct from the single
`doca_sta_cap_is_supported` device check):

| Sizing input | Query | Why the agent must ask |
| --- | --- | --- |
| Number of queue pairs (connections) | `doca_sta_get_max_qps` | Oversize fails at start; under-size limits fan-out |
| I/O queue depth (NVMeoF QP depth) | `doca_sta_get_max_io_queue_size` | Per-queue depth is device-bound; assuming 1024 works everywhere is wrong |
| Maximum NVMeoF I/O size | `doca_sta_get_max_io_size` | I/O larger than the reported max is rejected |
| Number of IO contexts (IO threads) | `doca_sta_set_max_sta_io` / `doca_sta_get_max_sta_io` / `doca_sta_get_max_io_threads` | The IO-thread count is bounded by the library maximum |
| Subsystems / namespaces / backends | `doca_sta_get_max_subsys` / `doca_sta_get_max_ns_per_subs` / `doca_sta_get_max_be` | The target topology is bounded; oversize fails at create |

The agent SHOULD verify the exact spelling of these
`doca_sta_get_max_*` queries and the single
`doca_sta_cap_is_supported` device check against the headers
shipped on the user's install
(`$(pkg-config --variable=includedir doca-common) doca_sta*.h`)
or in the public DOCA STA guide reachable via
[`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md).
Do **not** invent a multi-query `doca_sta_cap_*` family — only
`doca_sta_cap_is_supported` exists; everything else is a
`doca_sta_get_max_*` getter.

**Configuration shape.** *Mandatory* configurations before
`doca_ctx_start()`: the underlying `doca_dev` opened against a
device that passes `doca_sta_cap_is_supported`; the devices added
via `doca_sta_add_dev()`; the IO-context count set with
`doca_sta_set_max_sta_io()` at or below the library maximum; the
target subsystems, namespaces, and backends defined within the
reported `doca_sta_get_max_*` bounds. The queue-pair depth and
count must stay at or below the `doca_sta_get_max_*` reported
values.

## Version compatibility

For the canonical DOCA version-detection chain, the four-way
match rule, NGC container semantics, and the
headers-win-over-docs rule, see
[`doca-version`](../../doca-version/SKILL.md). The body lives
there; this skill does not duplicate it.

**The STA-specific overlay** is:

- **`doca_sta_cap_is_supported` and the `doca_sta_get_max_*`
  queries are the runtime authority, not the public docs.** Per
  the cross-cutting cap-query rule in
  [`doca-version CAPABILITIES.md ## Observability`](../../doca-version/CAPABILITIES.md#observability),
  the agent must call `doca_sta_cap_is_supported` against the
  active `doca_devinfo` before promising the user that STA target
  acceleration is on this device + DOCA version, and must call
  the matching `doca_sta_get_max_*` query before promising any
  queue depth, queue count, or topology size. Quoting a sizing
  value from memory is the canonical hallucination failure mode
  for this library.
- **`doca-sta.pc` and `doca-common.pc` must both match
  `doca_caps --version`** at the four-way-match check (per
  [`doca-version CAPABILITIES.md ## Version compatibility`](../../doca-version/CAPABILITIES.md#version-compatibility)).
  Use `pkg-config --modversion doca-sta` as the build-time
  anchor; disagreement with `doca_caps --version` is a
  partial-install hazard and must be routed to
  [`doca-version TASKS.md ## debug`](../../doca-version/TASKS.md#debug)
  layer 2 before any STA-layer diagnosis.
- **Substrate-library version match.** Because STA's transport is
  RDMA, `doca-rdma.pc` must also match the same
  `doca_caps --version` line — a STA install that compiles
  against one DOCA RDMA major and runs against another is a
  partial-install hazard. Route to
  [`doca-rdma CAPABILITIES.md ## Version compatibility`](../doca-rdma/CAPABILITIES.md#version-compatibility)
  for the RDMA-side overlay.

## Error taxonomy

STA-specific overlays on the cross-library `DOCA_ERROR_*`
taxonomy. The cross-library taxonomy itself lives in
[`doca-programming-guide CAPABILITIES.md ## Error taxonomy`](../../doca-programming-guide/CAPABILITIES.md#error-taxonomy);
the rows below are the *STA surface* meaning that the agent
must disambiguate before falling back to the cross-library
response.

| Error | DOCA STA context where it shows up | STA-specific cause |
| --- | --- | --- |
| `DOCA_ERROR_BAD_STATE` | Any call after `doca_ctx_stop()` or before `doca_ctx_start()`; submitting an I/O on a queue pair that has not completed the NVMe-oF Connect handshake; reconfiguring subsystems / backends after start | Lifecycle violation on either the `doca_sta` context itself or the per-connection queue-pair state machine. Walk the call sequence against the universal lifecycle in [`doca-programming-guide CAPABILITIES.md ## Capabilities and modes`](../../doca-programming-guide/CAPABILITIES.md#capabilities-and-modes); the most common case is submitting before the queue-pair reports CONNECTED. |
| `DOCA_ERROR_NOT_SUPPORTED` | Opening / using a device that does not pass `doca_sta_cap_is_supported`; a logical block size the device does not support; oversized queue count or depth | Re-run `doca_sta_cap_is_supported` against the active `doca_devinfo` (the device gate) and the matching `doca_sta_get_max_*` query (the sizing gate); if the device is unsupported or a getter returns a smaller bound, that is the answer — the user's device or DOCA version does not support the request. |
| `DOCA_ERROR_INVALID_VALUE` | `doca_sta_set_*` with a queue depth or count outside the `doca_sta_get_max_*` bound; a namespace block size rejected by `doca_sta_is_logical_block_size_supported()` | The fix is to re-read the relevant `doca_sta_get_max_*` value, lower the requested value, and re-run configure. Quote the queried value, not a value the user remembered. |
| `DOCA_ERROR_AGAIN` | I/O submission on a per-queue path when the in-flight budget is full | The I/O queue is full. This is *not* a hardware error; the program must drain completions via `doca_pe_progress()` before re-submitting, or raise the queue depth within the `doca_sta_get_max_io_queue_size` bound. Same as the cross-library *"would-block, retry after progress"* pattern. |
| `DOCA_ERROR_IO_FAILED` | Per-IO completion event reports failure; transport-layer error during the NVMe-oF Connect handshake | A transport-layer I/O error. Likely causes: link drop, RDMA peer disconnect, firmware fault, peer-side controller reset. Do not retry blindly — capture `dmesg | tail` and route to [`doca-setup TASKS.md ## debug`](../../doca-setup/TASKS.md#debug) and to [`doca-rdma CAPABILITIES.md ## Error taxonomy`](../doca-rdma/CAPABILITIES.md#error-taxonomy) (the RDMA substrate) before recommending a code change. |
| `DOCA_ERROR_NOT_PERMITTED` | `doca_dev_open` for a device the user has no access to; STA context create after a permission downgrade | The device was not opened with the required access. Confirm sudo or the appropriate group membership per [`## Safety policy`](#safety-policy); do not modify the program. |
| `DOCA_ERROR_DRIVER` | Any submit / completion call | The layer below DOCA reported failure. Capture state (`dmesg | tail`, `mlxconfig -d <pcie> q`) and route to env-class debug ([`doca-setup TASKS.md ## debug`](../../doca-setup/TASKS.md#debug)) — the layer below DOCA is the suspect, not the program. |

The agent's rule: **never recommend a retry loop on `DOCA_ERROR_*`
without first identifying which of the rows above is the cause**.
`_AGAIN` is the only one that wants a retry (after
`doca_pe_progress()`); the others want investigation, not retry.
And *"the connection won't establish"* is **rarely** a `_BAD_STATE`
on the STA side — it is more often a steering / fabric / peer
issue and the agent should walk the precondition matrix in
[`## Safety policy`](#safety-policy) before any code change.

## Observability

The DOCA Core progress engine (PE) is the single source of
observability for STA: every queue-pair state transition and
every I/O completion (success or failure) arrives as an event on
the PE that the `doca_sta` context is registered against. STA
does **not** maintain per-connection counters of its own; its
observability surface is event-driven, with one add-on signal
(capability snapshot at configure time).

Three primary signals the agent should reach for:

1. **Per-queue I/O completion events on the PE.** Each submitted
   I/O produces a completion event when it finishes (or errors)
   on the matching queue. The completion carries the
   `doca_error_t` if it failed; the agent must inspect the
   per-IO completion, not the submit-call return value alone.
   Absence of completions on a queue with submitted IOs is
   *always* either a missing `doca_pe_progress()` call or a
   transport-layer stall; route to [`## Error taxonomy`](#error-taxonomy)
   `_AGAIN` / `_IO_FAILED` rows.
2. **Queue-pair state transitions.** The NVMe-oF queue pair
   transitions from CREATED → CONNECTED (after the NVMe-oF
   Connect handshake) → DISCONNECTED. The agent must wire and
   inspect the transition events; submitting an I/O before
   CONNECTED returns `DOCA_ERROR_BAD_STATE`, and a session that
   *seems up* but never moves past CREATED is a fabric or
   peer-side problem, not a STA bug.
3. **Capability snapshot at configure time.** The output of the
   `doca_sta_cap_is_supported` device check plus the
   `doca_sta_get_max_*` queries is a snapshot of *what the
   library said was possible* before any I/O was submitted.
   Save it as the baseline; if an I/O later returns
   `DOCA_ERROR_NOT_SUPPORTED` or `_INVALID_VALUE` the diff
   against this snapshot is the bug.

For the cross-cutting debug-time observability primitives
(`DOCA_LOG_LEVEL=trace`, `--sdk-log-level`, the trace build
flavor) see
[`doca-debug CAPABILITIES.md ## Observability`](../../doca-debug/CAPABILITIES.md#observability).
For the env-side observability primitives (link state, RDMA
device enumeration) defer to
[`doca-setup CAPABILITIES.md ## Observability`](../../doca-setup/CAPABILITIES.md#observability).

## Safety policy

> **Overlay on the bundle-wide hardware-safety meta-policy.** The rules below are this skill's per-artifact overlay on the cross-cutting rules in [`doca-hardware-safety` CAPABILITIES.md ## Safety policy](../../doca-hardware-safety/CAPABILITIES.md#safety-policy) (specifically [### Per-artifact overlay pattern](../../doca-hardware-safety/CAPABILITIES.md#per-artifact-overlay-pattern)). When the two layers disagree, the stricter wins; when either layer says STOP, the agent stops.

DOCA STA's safety surface is **substrate-library presence,
device access, and steering**. The single most common first-app
failure for an NVMe-over-RDMA target is *"the initiator's Connect
handshake never completes"* — and the agent's job is to verify
the three preconditions before any code change, not after the
first `DOCA_ERROR_*`.

The **precondition matrix** the agent must walk for any new
DOCA STA setup:

| Precondition | What must be true before `doca_ctx_start()` | How the agent verifies | Where to fix |
| --- | --- | --- | --- |
| Substrate library present | STA's transport is RDMA-only: `doca-rdma.pc` resolves and `doca_rdma_cap_*` reports a non-empty surface on the chosen device | `pkg-config --modversion doca-rdma`; the matching cap-query call as documented in [`doca-rdma CAPABILITIES.md ## Capabilities and modes`](../doca-rdma/CAPABILITIES.md#capabilities-and-modes); `mlxconfig -d <pcie> q` for the firmware view | [`doca-setup`](../../doca-setup/SKILL.md) for the env-side; do not modify the program |
| Device access | The `doca_dev` was opened against a BlueField PF / SF the user has permission to use (typically requires sudo or the appropriate group membership) | `id` for group membership; the open call failing with `DOCA_ERROR_NOT_PERMITTED` is the runtime symptom | [`doca-setup`](../../doca-setup/SKILL.md) for the env-side; do not modify the program |
| NVMe-oF traffic actually reaches the STA-managed queue | Either a DOCA Flow rule (or the env-side equivalent) steers matching NVMe-oF 5-tuples to this STA instance's queues | Inspect the Flow rule programmed for this NVMe-oF connection (or the absence of one); confirm via the steering-rule listing on the user's setup | [`doca-flow`](../doca-flow/SKILL.md) for the steering side; do not invent a `doca_sta_*` steering call |

**The initiator side is out of scope.** The agent must surface —
early in any recommendation — that doca-sta accelerates the
NVMe-oF *target* data path only. The remote initiator / host NVMe
stack (SPDK `bdev_nvme`, kernel `nvme` host) is a separate peer on
the fabric and is not programmed through doca-sta. The skill does
NOT prescribe the initiator's deployment — that's the remote
peer's decision — but it must ensure the user knows doca-sta is
target-side acceleration, not an initiator transport provider.

**Smoke before scale-up.** Before driving production workloads,
the agent must walk the user through a single-IO smoke (one NVMe
admin command — typically Identify Controller — over the admin
queue, then one NVMe Read / Write I/O over a single I/O queue).
A failure here narrows cleanly: admin-queue side, fabric, or
I/O-queue side. A failure at production scale *without* the
single-IO smoke pass is a much harder bisection.
