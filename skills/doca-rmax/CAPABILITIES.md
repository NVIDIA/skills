# DOCA Rivermax capabilities, version overlay, errors, observability, safety

**Where to start:** Pick the H2 anchor that matches your question
(input stream object / capabilities / Rivermax SDK
+ license / errors / safety) and read that section end-to-end.
The tables in each section are the load-bearing content; the
prose around them is interpretation.

Read this file when the loader sent you here from
[SKILL.md](SKILL.md). For the *how* of executing each pattern
(the verbs `configure / build / modify / run / test / debug`),
jump to [TASKS.md](TASKS.md). For the canonical DOCA
version-handling rules that this skill layers a Rivermax
overlay on top of, see
[`doca-version`](../../doca-version/SKILL.md). For the queue
surface that carries the packets a Rivermax stream produces or
consumes, defer to [`doca-eth`](../doca-eth/SKILL.md);
for the steering side (which Flow rule sends which packet to
which queue) defer to [`doca-flow`](../doca-flow/SKILL.md).
DOCA Rivermax is the *Rivermax integration* surface, not the
*queue* surface and not the *steering* surface.

## Pattern overview

Every DOCA Rivermax question this skill teaches resolves into
one of FIVE patterns. The patterns are CLASSES — they apply
across every DOCA Rivermax release, every host / BlueField, and
every supported Rivermax SDK version.

| Pattern | When it applies (class shape) | Where the substance lives |
| --- | --- | --- |
| 1. Confirm the Rivermax precondition | Before any other concern, verify that the NVIDIA Rivermax SDK is installed AND a valid Rivermax license is readable on the host; without both, `doca-rmax` cannot function and there is no in-library fallback | [`## Safety policy`](#safety-policy) precondition matrix + [TASKS.md ## configure](TASKS.md#configure) step 1 |
| 2. Stand up the receive stream | Initialize the global DOCA Rivermax engine (`doca_rmax_init()` / `doca_rmax_release()`), then create the receive-only `doca_rmax_in_stream` session object via `doca_rmax_in_stream_create()` and drive it through its own DOCA Core lifecycle. The public API is receive-only — there is no transmit/output stream | [`## Capabilities and modes`](#capabilities-and-modes) input stream table + [TASKS.md ## configure](TASKS.md#configure) step 2 |
| 3. Discover capabilities | Query the `doca_rmax_get_*_supported` family against the active `doca_devinfo` for PTP-clock and packet-placement-order support before assuming any timing-precise configuration is feasible on this device + this Rivermax SDK | [`## Capabilities and modes`](#capabilities-and-modes) capability-query rule + [TASKS.md ## configure](TASKS.md#configure) step 3 |
| 4. Honor preconditions | Verify the device opens under the user's access (DOCA-side: sudo or `mlnx` group; Rivermax-side: license file readable, real-time scheduling on streaming threads), the underlying port is up, and traffic is steered to the queue the Rivermax stream is attached to (via DOCA Flow) | [`## Safety policy`](#safety-policy) precondition matrix + [TASKS.md ## configure](TASKS.md#configure) step 1 |
| 5. Diagnose a Rivermax error | Map symptom (`DOCA_ERROR_BAD_STATE`, `_NOT_PERMITTED`, `_NOT_SUPPORTED`, `_INVALID_VALUE`, `_DRIVER`) to root cause WITHOUT collapsing the DOCA-side / Rivermax-side / license-side / driver-below layers into one | [`## Error taxonomy`](#error-taxonomy) + [TASKS.md ## debug](TASKS.md#debug) |

Two cross-cutting rules that apply to *every* pattern above:

- **DOCA Rivermax is a wrapper, not the engine.** The
  timing-precise media-streaming engine lives in the NVIDIA
  Rivermax SDK; `doca-rmax` is the DOCA-side integration
  that exposes Rivermax sessions through the DOCA Core context
  lifecycle. Treating `doca-rmax` as a self-contained
  library is the most common first-app design error: there is
  no fallback path inside DOCA that recovers timing precision
  without a real Rivermax install + license behind it.
- **Sub-microsecond jitter is the library's whole point.** If
  the user does not need timing-precise streaming, the right
  answer is to route them to
  [`doca-eth`](../doca-eth/SKILL.md) for best-effort packet
  I/O — not to layer `doca-rmax` on top of a workload it
  was not built for, and not to silently downgrade their
  timing expectations.

## Capabilities and modes

The DOCA Rivermax public API is **receive-only**: the one
selection axis for any design is *timestamp format / packet-
placement order / scatter type* (device- and
Rivermax-conditional). The engine itself is a process-global
singleton; choose stream parameters only after the matching
capability-query confirms they are supported.

**Object model — a global engine plus a receive-only input
stream.** There is no per-integration context object: the
library is initialized process-wide with `doca_rmax_init()` and
torn down with `doca_rmax_release()`. The only stream object is
the receive-side `doca_rmax_in_stream`; a `doca_rmax_flow`
attaches match criteria to it. Read exact symbol names from the
headers at
$(pkg-config --variable=includedir doca-common) rather than from
agent memory beyond the family pattern below.

| Object | What it does | Family of calls | Sizing inputs |
| --- | --- | --- | --- |
| Global Rivermax engine (process singleton) | Process-wide initialization of the underlying Rivermax SDK integration; must be initialized before any stream is created and released after all streams are destroyed. Holds no per-device state itself | `doca_rmax_init()` / `doca_rmax_release()` / `doca_rmax_interrupt()`, plus `doca_rmax_set_cpu_affinity_mask()` and `doca_rmax_set_clock()` | None — it is a singleton; the device is selected per input stream |
| `doca_rmax_in_stream` (receive) | The one Rivermax session object — a receive stream for SMPTE ST 2110 inbound video / audio, real-time market data feeds, instrument streams. Created on a `doca_dev`, converted to a DOCA Core context via `doca_rmax_in_stream_as_ctx()`, and surfaces inbound data via the Rx-data PE event with Rivermax doing the timing-precise reassembly | `doca_rmax_in_stream_create()` / `_destroy()` / `_as_ctx()`, `doca_rmax_in_stream_set_*` setters, `doca_rmax_in_stream_event_rx_data_register()` | A `doca_dev` opened against the target port / representor / SF (must have a valid IPv4 address); `doca_rmax_get_*_supported` queries against the active `doca_devinfo` |
| `doca_rmax_flow` (steering filter) | A match filter (src/dst IPv4, src/dst port, tag) attached to an input stream so only the intended media flow lands on it | `doca_rmax_flow_create()` / `_destroy()` / `_attach()` / `_detach()` + `doca_rmax_flow_set_{src,dst}_{ip,port}` / `_set_tag` | The IP/port 5-tuple of the inbound media flow |

The input stream sits on top of the underlying packet queue
(`doca_eth_rxq`); the queue itself is
programmed through [`doca-eth`](../doca-eth/SKILL.md), and
the steering that gets matching packets to it is programmed
through [`doca-flow`](../doca-flow/SKILL.md). Treating the
Rivermax stream as a stand-in for the queue / steering surfaces
is wrong: it conflates three independent libraries that each
own their own lifecycle.

**Capability discovery — the only rule.** Before assuming a
PTP-synced timestamp format or hardware-accelerated packet
placement order is feasible, call the matching
`doca_rmax_get_*_supported` query against the active
`doca_devinfo`:

| Capability | Query function | Why the agent must ask |
| --- | --- | --- |
| PTP clock supported on this device + Rivermax version | `doca_rmax_get_ptp_clock_supported` | PTP-synced timestamps require device + Rivermax support; assuming PTP is on every device is a canonical hallucination failure mode. Required before `doca_rmax_set_clock()` and the PTP-synced timestamp format. |
| Hardware-accelerated RTP sequence-number packet placement | `doca_rmax_get_packet_placement_order_rtp_seqn_supported` | RTP-seqn placement ordering is device- and Rivermax-version-conditional; query before enabling it via `doca_rmax_in_stream_set_packet_placement_order_rtp_seqn`. |
| Hardware-accelerated SMPTE ST 2110-20 sequence-number placement | `doca_rmax_get_packet_placement_order_st2110_20_seqn_supported` | ST 2110-20 placement ordering is device- and Rivermax-version-conditional; query before enabling it via `doca_rmax_in_stream_set_packet_placement_order_st2110_20_seqn`. |

The capability-query rule is **the runtime authority**: when
the cap query says false, that is the answer for this host,
regardless of what the public docs claim about the feature in
general. Quote the queried value back to the user; do not
substitute a value the user remembered from another install.

**Configuration shape.** *Mandatory* configurations before
`doca_ctx_start()` on the input stream: the scatter type,
timestamp format, and memory-block / element-count layout set
per the user's intent (timestamp format and packet-placement
order gated on the matching `doca_rmax_get_*_supported` query),
the underlying queue context wired in (allocated and started via
[`doca-eth`](../doca-eth/SKILL.md)), and the Rx-data event
registered via `doca_rmax_in_stream_event_rx_data_register()`.
*Optional* configurations (hardware-accelerated packet-placement
order, PTP-synced timestamps, receive timeout / min-max packet
batching) gate on the matching `doca_rmax_get_*_supported`
query and the corresponding `doca_rmax_in_stream_set_*` setter; the agent should consult
the public DOCA Rivermax guide (slug `DOCA-Rivermax` via
[`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md))
for the canonical setter names on the installed version rather
than quote them from memory.

## Version compatibility

For the canonical DOCA version-detection chain, the four-way
match rule, NGC container semantics, and the
headers-win-over-docs rule, see
[`doca-version`](../../doca-version/SKILL.md). The body lives
there; this skill does not duplicate it.

**The Rivermax-specific overlay** is:

- **The Rivermax SDK version is a second compatibility axis.**
  DOCA Rivermax is a wrapper; the underlying timing-precise
  engine ships with the NVIDIA Rivermax SDK and its own
  version. A capability advertised by `doca-rmax` headers
  but rejected at runtime usually means the installed Rivermax
  SDK is older than the feature requires. The agent's rule:
  always quote both the DOCA version (per
  `pkg-config --modversion doca-rmax`) **and** the
  Rivermax SDK version (per the Rivermax SDK's own version
  query, documented in the public Rivermax SDK guide via
  [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md))
  before promising a feature.
- **`doca_rmax_get_*_supported` is the runtime authority, not the
  public docs.** Per the cross-cutting cap-query rule in
  [`doca-version CAPABILITIES.md ## Observability`](../../doca-version/CAPABILITIES.md#observability),
  the agent must call the matching capability query against
  the active `doca_devinfo` before promising the user that a
  PTP clock or hardware-accelerated packet-placement order is on
  this device + DOCA version + Rivermax SDK version. Quoting a
  feature from memory is the canonical hallucination failure mode
  for this library.
- **`doca-rmax.pc` and `doca-common.pc` must both match
  `doca_caps --version`** at the four-way-match check (per
  [`doca-version CAPABILITIES.md ## Version compatibility`](../../doca-version/CAPABILITIES.md#version-compatibility)).
  Use `pkg-config --modversion doca-rmax` as the
  build-time anchor; disagreement with `doca_caps --version`
  is a partial-install hazard and must be routed to
  [`doca-version TASKS.md ## debug`](../../doca-version/TASKS.md#debug)
  layer 2 before any Rivermax-layer diagnosis. A missing
  `doca-rmax.pc` does NOT imply the Rivermax SDK is
  missing — those are two independent install layers and the
  agent must check both.

## Error taxonomy

Rivermax-specific overlays on the cross-library `DOCA_ERROR_*`
taxonomy. The cross-library taxonomy itself lives in
[`doca-programming-guide CAPABILITIES.md ## Error taxonomy`](../../doca-programming-guide/CAPABILITIES.md#error-taxonomy);
the rows below are the *Rivermax surface* meaning that the
agent must disambiguate before falling back to the
cross-library response.

| Error | DOCA Rivermax context where it shows up | Rivermax-specific cause |
| --- | --- | --- |
| `DOCA_ERROR_BAD_STATE` | Any call after `doca_ctx_stop()` or before `doca_ctx_start()`; using an input stream whose underlying `doca_eth_rxq` queue has not been started; calling a `doca_rmax_in_stream_set_*` setter after the context has left idle | Lifecycle violation. Walk the input stream's state against the universal lifecycle in [`doca-programming-guide CAPABILITIES.md ## Capabilities and modes`](../../doca-programming-guide/CAPABILITIES.md#capabilities-and-modes); the most common case is treating the input stream lifecycle as covered by the process-global `doca_rmax_init()` call. |
| `DOCA_ERROR_NOT_PERMITTED` | Device open behind `doca_rmax_in_stream_create()` | DOCA-side device access denied (no sudo / not in `mlnx` group), same as every DOCA library — confirm via the precondition matrix in [`## Safety policy`](#safety-policy). Note that a **missing / expired / unreadable Rivermax license** surfaces instead as `DOCA_ERROR_NOT_SUPPORTED` from `doca_rmax_init()` (per the header), so check the init return code separately. |
| `DOCA_ERROR_NOT_SUPPORTED` | `doca_rmax_init()` (invalid / missing Rivermax license); timestamp-format / packet-placement-order setters | Either the Rivermax license is invalid or missing (from `doca_rmax_init()`), or the chosen stream parameter is not supported on this device + this Rivermax SDK version. Re-run the matching `doca_rmax_get_*_supported` query against the active `doca_devinfo`; if it says not-supported, that is the answer — the device or the Rivermax version does not support the request. The fix is to change the request, supply a valid license, or upgrade the Rivermax SDK, not to retry. |
| `DOCA_ERROR_INVALID_VALUE` | `doca_rmax_in_stream_set_*` setters with an out-of-range element count, memory-block size, or packet batch bound | The user passed a value outside the valid set for the chosen stream configuration. The fix is at the configure step, not in the program's runtime loop. |
| `DOCA_ERROR_DRIVER` | Any submit / completion call | The layer below DOCA reported failure — which for Rivermax may be either the DOCA driver layer (mlx5 / ConnectX firmware) OR the underlying Rivermax driver / stack. Capture state from both sides (`dmesg | tail`, `mlxconfig -d <pcie> q`, and the Rivermax SDK's own log if its docs describe one) and route to env-class debug ([`doca-setup ## debug`](../../doca-setup/TASKS.md#debug)) — the layer below DOCA is the suspect, not the program. |

The agent's rule: **never recommend a retry loop on
`DOCA_ERROR_*` without first identifying which of the rows
above is the cause**. For DOCA Rivermax there is no retry-on-
error row at all — every Rivermax error wants investigation,
not retry. And *"no packets arriving"* is **never** a
`DOCA_ERROR_*` — it is one of: a missing Rivermax license, a
missing or wrong Flow rule on the steering side (route to
[`doca-flow`](../doca-flow/SKILL.md)), or an unstarted
underlying `doca_eth_rxq` (route to
[`doca-eth`](../doca-eth/SKILL.md)).

## Observability

The DOCA Core progress engine (PE) is the single source of
observability for the Rivermax input stream: every
Rx-data event (success or failure)
arrives as an event on the PE that the input stream context is
registered against. DOCA Rivermax does **not** maintain
per-stream counters of its own; its observability surface is
event-driven, not poll-driven, with three add-on signals
(capability snapshot + Rivermax-side state + port-state).

Four primary signals the agent should reach for:

1. **Rx-data event completions on the PE.** The input stream
   produces an Rx-data event (via
   `doca_rmax_in_stream_event_rx_data_register()`) on every
   received batch of packets per the chosen stream type. Absence
   of completions on a started stream is *always* either a
   missing `doca_pe_progress()` call or a
   missing-license / steering / underlying-queue precondition
   gap. Route to [`## Safety policy`](#safety-policy) first.
2. **Capability snapshot at configure time.** The output of
   every `doca_rmax_get_*_supported` query is a snapshot of *what
   the library + the Rivermax SDK said was possible* before
   any task was submitted. Save it as the baseline; if a task
   later returns `DOCA_ERROR_NOT_SUPPORTED` the diff against
   this snapshot is the bug (and usually the bug is that a
   *different* `doca_devinfo` was used at submit time than at
   cap-query time).
3. **Rivermax-side state.** The Rivermax SDK has its own
   logging / status surface documented in the public Rivermax
   SDK guide; the agent should route to it via
   [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md)
   when the cause is suspected to be Rivermax-side (license,
   stream config, scheduling jitter) rather than DOCA-side.
   The agent must NOT invent Rivermax-side commands or paths
   from memory.
4. **Port and link state from the env side.** `devlink dev
   show`, `ip -j link show <dev>`, and `ethtool <dev>`
   together tell the agent whether the port the user opened
   is up at the driver layer at all. A linkdown port is a
   silent Rivermax stream; the agent must check it before any
   Rivermax-layer diagnosis. Defer to
   [`doca-setup CAPABILITIES.md ## Observability`](../../doca-setup/CAPABILITIES.md#observability)
   for the env-side observability primitives.

For the cross-library debug-time observability
(`DOCA_LOG_LEVEL=trace`, `--sdk-log-level`, the trace build
flavor) see
[`doca-debug CAPABILITIES.md ## Observability`](../../doca-debug/CAPABILITIES.md#observability).

## Safety policy

> **Overlay on the bundle-wide hardware-safety meta-policy.** The rules below are this skill's per-artifact overlay on the cross-cutting rules in [`doca-hardware-safety` CAPABILITIES.md ## Safety policy](../../doca-hardware-safety/CAPABILITIES.md#safety-policy) (specifically [### Per-artifact overlay pattern](../../doca-hardware-safety/CAPABILITIES.md#per-artifact-overlay-pattern)). When the two layers disagree, the stricter wins; when either layer says STOP, the agent stops.

DOCA Rivermax's safety surface is **the Rivermax SDK + license
precondition first**, then DOCA-side access, port state, and
steering, and finally the real-time scheduling discipline on
streaming threads. The single most common first-app failure is
*"I'm calling `doca-rmax` but it returns `_NOT_PERMITTED` /
silently fails"* — and the answer is usually that the user
assumed DOCA bundles Rivermax (it does not) and never
installed the Rivermax SDK or arranged the license file.

The **precondition matrix** the agent must walk for any new
DOCA Rivermax setup. The first row is the gate that determines
whether the rest of the matrix is even relevant.

| Precondition | What must be true before `doca_ctx_start()` | How the agent verifies | Where to fix |
| --- | --- | --- | --- |
| **Rivermax SDK installed (hard external dep)** | The NVIDIA Rivermax SDK is installed on the host at the location Rivermax expects | Check the public DOCA Rivermax guide (slug `DOCA-Rivermax` via [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md)) for the install probe; ask the user to confirm — do not assume from a successful `pkg-config doca-rmax` (that only confirms the DOCA-side wrapper, not the SDK behind it) | Route to the public Rivermax SDK guide via [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md); this is OUT OF SCOPE for `doca-rmax`. If the SDK is not installed, `doca-rmax` cannot be used — there is no DOCA-side fallback |
| **Valid Rivermax license readable** | A current, unexpired Rivermax license file is present at the location the Rivermax SDK expects, readable by the user the streaming process will run as | Confirm via the Rivermax SDK's own license check (per its public docs); a runtime `DOCA_ERROR_NOT_PERMITTED` from a Rivermax create call with DOCA-side device access already validated is the classic symptom of license missing / expired / unreadable | Route to the public Rivermax SDK guide via [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md); this is OUT OF SCOPE for `doca-rmax` |
| DOCA-side device access | The `doca_dev` was opened against a port / representor / SF the user has permission to use (typically requires sudo or `mlnx`-group membership) | `id` for group membership; the open call failing with `DOCA_ERROR_NOT_PERMITTED` is the runtime symptom, distinct from the Rivermax-side license cause above | [`doca-setup`](../../doca-setup/SKILL.md) for the env-side; do not modify the program |
| Port up | The underlying port reports linkup at the driver layer (`devlink dev show` reports `state: PORT_ACTIVE`; `ip link` shows the device `UP,LOWER_UP`) | `devlink dev show`; `ip -j link show <dev>`; `ethtool <dev>` | [`doca-setup CAPABILITIES.md ## Capabilities and modes`](../../doca-setup/CAPABILITIES.md#capabilities-and-modes) port-bring-up; do not modify the program |
| Traffic reaches the underlying queue | A DOCA Flow rule steers matching packets to the `doca_eth_rxq` the Rivermax input stream is attached to (input direction only) | Inspect the flow rule programmed for the queue; programmatic via the [`doca-flow`](../doca-flow/SKILL.md) skill's `## test` workflow | [`doca-flow`](../doca-flow/SKILL.md) for the steering side; this is OUT OF SCOPE for `doca-rmax` |
| Real-time scheduling on streaming threads | The streaming thread runs at real-time priority (per the Rivermax SDK's recommended scheduling discipline) so timing-precise emission / reception holds | Route to the Rivermax SDK's own docs via [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md) for the canonical scheduling guidance | Out of scope for `doca-rmax`; the canonical scheduling discipline lives in the Rivermax SDK guide |

**Do not invent a "fallback transport" for Rivermax.** When
the Rivermax SDK or license is unavailable, the right answer
is to tell the user honestly that `doca-rmax` cannot be
used and to route them to
[`doca-eth`](../doca-eth/SKILL.md) for best-effort packet
I/O (or [`doca-gpunetio`](../doca-gpunetio/SKILL.md) for
GPU-initiated best-effort I/O) — **and to clearly state that
the best-effort path loses the sub-microsecond-jitter timing
guarantee that is the whole point of Rivermax**. Silently
substituting `doca-eth` for `doca-rmax` is a user-visible
regression dressed up as helpfulness.

**Smoke before scale-up.** Before driving traffic at full
stream rate, the agent must walk the user through a
single-frame smoke (one frame / chunk of the configured
stream type, one PE progress, one completion event). A
failure here narrows cleanly: license, steering, queue, or
program. A failure at full stream rate *without* the smoke
pass is a much harder bisection across the
DOCA-side / Rivermax-side / scheduling-discipline boundary.
