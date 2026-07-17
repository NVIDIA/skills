# DOCA Comch capabilities, version overlay, errors, observability, safety

**Where to start:** Pick the H2 anchor that matches your question
(roles / slow-path / fast-path / capabilities / errors / safety) and
read that section end-to-end. The tables in each section are the
load-bearing content; the prose around them is interpretation.

Read this file when the loader sent you here from
[SKILL.md](SKILL.md). For the *how* of executing each pattern (the
verbs `configure / build / modify / run / test / debug`), jump to
[TASKS.md](TASKS.md). For the canonical DOCA version-handling rules
that this skill layers a Comch overlay on top of, see
[`doca-version`](../../doca-version/SKILL.md).

## Pattern overview

Every Comch question this skill teaches resolves into one of FIVE
patterns. The patterns are CLASSES — they apply across every Comch
release and every host / DPU pair.

| Pattern | When it applies (class shape) | Where the substance lives |
| --- | --- | --- |
| 1. Pick the role | Decide which side is server (DPU), which is client (host); how many clients to accept | [`## Capabilities and modes`](#capabilities-and-modes) role table + [TASKS.md ## configure](TASKS.md#configure) step 2 |
| 2. Pick the path | Choose slow-path (send-task / recv-callback) vs fast-path (producer / consumer) for the user's throughput profile | [`## Capabilities and modes`](#capabilities-and-modes) slow-vs-fast table + [TASKS.md ## configure](TASKS.md#configure) step 4 |
| 3. Discover capabilities | Query `doca_comch_cap_get_*` for max-msg-size, max-clients, transport-type on the active `doca_devinfo` | [`## Capabilities and modes`](#capabilities-and-modes) capability-query rule + [TASKS.md ## configure](TASKS.md#configure) step 3 |
| 4. Honor permissions | Verify representor visibility (DPU side) and PCIe address reachability (host side) before any object creation call | [`## Safety policy`](#safety-policy) permission matrix + [TASKS.md ## configure](TASKS.md#configure) step 1 |
| 5. Diagnose a Comch error | Map symptom (`DOCA_ERROR_BAD_STATE`, `_AGAIN`, `_NOT_PERMITTED`, `_CONNECTION_RESET`) to root cause without leaving the Comch layer prematurely | [`## Error taxonomy`](#error-taxonomy) + [TASKS.md ## debug](TASKS.md#debug) |

Two cross-cutting rules that apply to *every* pattern above:

- **Server is always on the DPU side; client is always on the host
  side.** The Comch channel runs over the RoCE/IB protocol (it is
  not part of the TCP/IP stack) between the host and the DPU, in an
  asymmetric server/client model; there is no symmetric
  peer-to-peer mode. An agent recommending the opposite is wrong
  for *every* version of Comch.
- **Slow-path and fast-path are independent objects.** A user can
  run slow-path only, fast-path only, or both — but each adds its
  own lifecycle to the channel. Conflating them is the most
  common first-app design error and the cleanest place to fail
  fast.

## Capabilities and modes

The two orthogonal selection axes for any Comch design are *role*
and *path*. Choose both before writing any code, then drill into the
relevant capability-query.

**Role split — server vs client.** Comch is asymmetric.

| Role | Side | What it does | Object | Key calls |
| --- | --- | --- | --- | --- |
| Server | DPU (BlueField Arm) | Listens on a host representor; accepts up to the device's per-server connection cap (query with `doca_comch_cap_get_max_clients(devinfo)` before start); owns the connection-accept callback | `doca_comch_server` | `doca_comch_server_create`, `doca_comch_server_event_connection_status_changed_register` |
| Client | Host (x86 / Arm) | Initiates a single connection to a server, identified by its `doca_dev_rep` (representor on the DPU side surfaced to the host) | `doca_comch_client` | `doca_comch_client_create` (no client connection-event register exists — the client tracks connection state via its context state and task callbacks) |

**Path selection — slow-path vs fast-path.** Both paths can coexist
on the same channel; choose at least one before `doca_ctx_start()`.

| Path | What it is | Right shape for | Wrong shape for |
| --- | --- | --- | --- |
| Slow-path | Message-oriented send-task / recv-callback API on the connection itself; one DOCA task per outbound message; per-message recv callback on the inbound side | Control plane, low-rate messages, configuration sync, command-response patterns | Bulk data, line-rate streams — per-message DOCA-task overhead dominates and limits throughput |
| Fast-path | `doca_comch_producer` and `doca_comch_consumer` are separate context objects on each side; submit/recv-batched asynchronous transfers with smaller per-transfer overhead | Bulk data, asynchronous streams, telemetry uploads, log shipping | One-shot config messages — the producer / consumer setup overhead is not worth it for a single small message |

**Capability discovery — the only rule.** Before sizing any queue
or assuming a message length, call the matching
`doca_comch_cap_get_*` query against the active `doca_devinfo`:

| Capability | Query | Why the agent must ask |
| --- | --- | --- |
| Maximum slow-path message size | `doca_comch_cap_get_max_msg_size(devinfo)` | Device-dependent; sending past the cap silently truncates or fails with `DOCA_ERROR_INVALID_VALUE` |
| Maximum server connections | `doca_comch_cap_get_max_clients(devinfo)` | DPU-side ceiling; oversizing returns `DOCA_ERROR_NOT_SUPPORTED` at start |
| Producer / consumer presence | `doca_comch_consumer_cap_is_supported(devinfo)` / `doca_comch_producer_cap_is_supported(devinfo)` | Fast-path is not on every device; agent must not silently assume it |
| Maximum send queue depth | `doca_comch_cap_get_max_send_tasks(devinfo)` | Used to size the slow-path outstanding-task budget |

**Configuration shape.** *Mandatory* configurations before
`doca_ctx_start()`: at least one of (slow-path recv callback,
producer attached, consumer attached) per side, and the connection
state callback per side. *Optional* configurations on the server
side include `doca_comch_server_set_max_msg_size(server, …)` (and
the matching `_client_set_max_msg_size`); the max-clients value is
a device capability (`doca_comch_cap_get_max_clients(devinfo)`)
and is not user-settable via a public setter — the cap is the
hard ceiling on what a server is allowed to accept. Query the
active capability values with the matching
`doca_comch_cap_get_*` / `doca_comch_(consumer|producer)_cap_*`
queries before start; do not assume a setter exists for every
capability.

## Version compatibility

For the canonical DOCA version-detection chain, the four-way match rule, NGC container semantics, and the headers-win-over-docs rule, see [`doca-version`](../../doca-version/SKILL.md). The body lives there; this skill does not duplicate it.

**The Comch-specific overlay** is:

- **The library was renamed in DOCA 2.5.** Old name: *DOCA Comm Channel*, `pkg-config` module `doca-comm-channel`. New name: *DOCA Comch*, `pkg-config` module `doca-comch`, URL slug `DOCA-Comch`. On installs ≥ 2.5 the agent must use the new name; on installs < 2.5 the agent must say so explicitly when the user's installed version (per [`doca-version TASKS.md ## configure`](../../doca-version/TASKS.md#configure)) predates the rename, and route the user to the legacy Comm Channel guide via [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md).
- **`doca_comch_producer` / `doca_comch_consumer` are newer than the slow-path.** When the user reports *"the API I'm reading about isn't on my install"*, the first hypothesis is that the install pre-dates the fast-path. Confirm via `doca_comch_consumer_cap_is_supported(devinfo)` / `doca_comch_producer_cap_is_supported(devinfo)` per the cross-cutting cap-query rule in [`doca-version CAPABILITIES.md ## Observability`](../../doca-version/CAPABILITIES.md#observability).
- **`doca-comch.pc` and `doca-common.pc` must both match `doca_caps --version`** at the four-way-match check (per [`doca-version CAPABILITIES.md ## Version compatibility`](../../doca-version/CAPABILITIES.md#version-compatibility)). A common partial-install pattern after a 2.4 → 2.5 upgrade is that `doca-comm-channel.pc` lingers alongside the new `doca-comch.pc`; the agent must surface that as a partial-install hazard, not as a *"both APIs are usable"* convenience.

## Error taxonomy

Comch-specific overlays on the cross-library `DOCA_ERROR_*`
taxonomy. The cross-library taxonomy itself lives in
[`doca-programming-guide CAPABILITIES.md ## Error taxonomy`](../../doca-programming-guide/CAPABILITIES.md#error-taxonomy);
the rows below are the *Comch surface* meaning that the agent must
disambiguate before falling back to the cross-library response.

| Error | Comch context where it shows up | Comch-specific cause |
| --- | --- | --- |
| `DOCA_ERROR_NOT_PERMITTED` | `doca_comch_server_create`, `_client_create` | The representor (DPU side) or PCIe address (host side) is not visible to this process / user. Route to [`doca-setup CAPABILITIES.md ## Observability`](../../doca-setup/CAPABILITIES.md#observability) for representor enumeration. |
| `DOCA_ERROR_NOT_SUPPORTED` | `doca_ctx_start()` on the server, fast-path `_create` | The implied clients count exceeds the device cap reported by `doca_comch_cap_get_max_clients(devinfo)`, or the device does not advertise consumer / producer support. The public Comch API does not ship a setter for max-clients — the device cap is the hard ceiling; agent must surface the cap value and reshape the program, not invent a setter. |
| `DOCA_ERROR_BAD_STATE` | Any call after `doca_ctx_stop()` or before `doca_ctx_start()`; submitting a `doca_comch_task_send` (via `doca_task_submit`) before the connection callback reports CONNECTED | Lifecycle violation. Walk the call sequence against the lifecycle in [`doca-programming-guide CAPABILITIES.md ## Capabilities and modes`](../../doca-programming-guide/CAPABILITIES.md#capabilities-and-modes); the most common case is sending before the server-side accept callback has fired. |
| `DOCA_ERROR_AGAIN` | slow-path `doca_comch_task_send` submit (`doca_task_submit`); producer submit fast-path | The send queue is full. This is *not* a hardware error; the program must drain completions via `doca_pe_progress()` before re-submitting. Same as the cross-library *"would-block, retry after progress"* pattern. |
| `DOCA_ERROR_CONNECTION_RESET` | Slow-path send-task completion, recv callback | The peer disconnected. The connection state callback will have fired with the matching DISCONNECTED transition; agent must not invent a reconnection policy — defer to the user's higher-level protocol. |
| `DOCA_ERROR_INVALID_VALUE` | `_task_send_alloc_init` with oversized message | The message exceeds `doca_comch_cap_get_max_msg_size(devinfo)`. The fix is to fragment at the application layer or to switch to the fast-path producer / consumer for bulk data. |
| `DOCA_ERROR_TIME_OUT` | Connection callback never fires after `doca_ctx_start()` | Most often a representor visibility issue on the DPU side, or a wrong PCIe address on the host side. Route to [`## Safety policy`](#safety-policy) permission matrix before any code change. |
| `DOCA_ERROR_DRIVER` | Any submit / completion call | The layer below DOCA reported failure. Capture state and route to env-class debug ([`doca-setup ## debug`](../../doca-setup/TASKS.md#debug)) — the layer below DOCA is the suspect, not the program. |

The agent's rule: **never recommend a retry loop on `DOCA_ERROR_*`
without first identifying which of the rows above is the cause**.
`_AGAIN` is the only one that wants a retry (after `doca_pe_progress()`);
the others want investigation, not retry.

## Observability

Comch observability surface is the set of callbacks and queries
that report channel state and per-message status. There is no
external "Comch counter" — the visibility comes from the per-API
events.

Three primary signals the agent should reach for:

1. **Connection state callbacks.** On the server side, registered
   via `doca_comch_server_event_connection_status_changed_register`
   (which takes the connect and disconnect callbacks). There is NO
   client-side connection-event register — the client observes
   connection state through its context state and task callbacks.
   These fire on CONNECTED / DISCONNECTED transitions and are the
   single most informative signal for *"is the channel up"*.
2. **Task completion callbacks (slow-path).** Configured via
   `doca_comch_{server,client}_task_send_set_conf` and allocated
   with `_task_send_alloc_init`; the send is submitted with the
   generic `doca_task_submit(doca_comch_task_send_as_task(task))`.
   The per-task success / error callback fires on every submitted
   send; absence of a completion is *always* a missing
   `doca_pe_progress()` call in the user's main loop.
3. **Producer / consumer completion callbacks (fast-path).** Same
   shape as the slow-path task callbacks, on the producer and
   consumer context objects. Per-transfer status flows through
   here; per-batch ordering is preserved.

For the cross-cutting env-side observability primitives
(representor enumeration, `devlink dev show`, `mlnx-bf-cfg`) see
[`doca-setup CAPABILITIES.md ## Observability`](../../doca-setup/CAPABILITIES.md#observability).
For the cross-library debug-time observability
(`DOCA_LOG_LEVEL=trace`, `--sdk-log-level`, the trace build
flavor) see
[`doca-debug CAPABILITIES.md ## Observability`](../../doca-debug/CAPABILITIES.md#observability).

## Safety policy

> **Overlay on the bundle-wide hardware-safety meta-policy.** The rules below are this skill's per-artifact overlay on the cross-cutting rules in [`doca-hardware-safety` CAPABILITIES.md ## Safety policy](../../doca-hardware-safety/CAPABILITIES.md#safety-policy) (specifically [### Per-artifact overlay pattern](../../doca-hardware-safety/CAPABILITIES.md#per-artifact-overlay-pattern)). When the two layers disagree, the stricter wins; when either layer says STOP, the agent stops.

Comch's safety surface is **permission and representor
visibility**. The single most common Comch first-app failure is
that the DPU-side server cannot see the host representor — and
the agent's job is to verify that before any object-creation
call, not after the first `DOCA_ERROR_NOT_PERMITTED`.

The **permission matrix** the agent must walk for any new Comch
setup:

| Side | What must be true before `doca_comch_<role>_create` | How the agent verifies | Where to fix |
| --- | --- | --- | --- |
| DPU (server) | The host representor for the target PF / VF / SF is visible to the DPU process (`/sys/class/net/<rep>` exists, owned by a user the process can access; user must typically be in the `mlnx` group or run with sudo) | `ls /sys/class/net/`; `id` to confirm group membership | [`doca-setup`](../../doca-setup/SKILL.md) for the env-side; do not modify the program |
| Host (client) | The PCIe address of the BlueField is enumerable from the host (`lspci | grep Mellanox`) and the corresponding `doca_dev_rep` is opened against it | `lspci | grep Mellanox`; programmatic via `doca_dev_rep_list_create` + the per-device filter | [`doca-setup`](../../doca-setup/SKILL.md) for the env-side; do not modify the program |
| Both | The DPU is in the correct mode for the host ↔ DPU pair the user expects (SmartNIC vs DPU vs switch) | `mlxconfig -d <pcie> q INTERNAL_CPU_MODEL` (sudo) | [`doca-setup CAPABILITIES.md ## Capabilities and modes`](../../doca-setup/CAPABILITIES.md#capabilities-and-modes) runtime-modes table |

**Do not invent a "fallback transport" for Comch.** The Comch
channel runs over the RoCE/IB protocol (not the TCP/IP stack) and
is reached only through the host ↔ DPU representor / device pair —
there is no separate Ethernet/TCP fallback path the agent can
substitute. When the representor or PCIe address is unavailable,
the right answer is to fix the env, not to point the user at a
different DOCA library.

**Slow-path vs fast-path queues are non-overlapping.** Submitting
a slow-path send-task on the producer context (or vice versa) is
caught at submit time by the runtime, but the user-visible
symptom (`DOCA_ERROR_BAD_STATE`) does not name the conflation.
The agent must trace the user's path-selection (per [`##
Capabilities and modes`](#capabilities-and-modes) slow-vs-fast
table) before any other diagnosis when the user reports
*"`_submit` works on one side and not the other"*.
