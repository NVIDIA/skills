# DOCA Telemetry Exporter capabilities, version overlay, errors, observability, safety

**Where to start:** Pick the H2 anchor that matches your question
(role-split / object family / capability discovery / path selection
/ version / errors / observability / safety) and read that section
end-to-end. The tables in each section are the load-bearing
content; the prose around them is interpretation.

Read this file when the loader sent you here from
[SKILL.md](SKILL.md). For the *how* of executing each pattern (the
verbs `configure / build / modify / run / test / debug`), jump to
[TASKS.md](TASKS.md). For the canonical DOCA version-handling rules
that this skill layers an exporter overlay on top of, see
[`doca-version`](../../doca-version/SKILL.md).

## Pattern overview

Every telemetry-exporter question this skill teaches resolves into
one of SIX patterns. The patterns are CLASSES — they apply across
every exporter release and every DOCA-using application, not just
the worked examples shown.

| Pattern | When it applies (class shape) | Where the substance lives |
| --- | --- | --- |
| 1. Pick exporter, not the service | The application is the PUBLISHER of telemetry events; the receiving / aggregating side is a separate DOCA service out of this skill's scope | [`## Capabilities and modes`](#capabilities-and-modes) role-split table |
| 2. Decide the exporter is the right tool | The need is structured event / counter / gauge publishing INTO the DOCA telemetry ecosystem; it is NOT plain stdout logging, NOT a non-DOCA Prometheus-style scrape, NOT a real-time event subscription back into the app | [`## Capabilities and modes`](#capabilities-and-modes) path-selection bullet |
| 3. Register the schema BEFORE the first emit | Every event the exporter publishes is shaped by a `doca_telemetry_exporter_schema` the app registers in advance; emitting before registration returns `BAD_STATE` / `NOT_FOUND` | [`## Capabilities and modes`](#capabilities-and-modes) object table + [TASKS.md ## configure](TASKS.md#configure) step 4 |
| 4. Stand up the source + schema + lifecycle | DOCA Core lifecycle: create exporter context → register one or more `_schema` → create one or more `_source` → start → emit events → stop → destroy | [`## Capabilities and modes`](#capabilities-and-modes) object table + [TASKS.md ## configure](TASKS.md#configure) |
| 5. Discover capabilities before assuming limits | `doca_telemetry_exporter_*_get_*` family for max schema fields, max event size, supported event types — call BEFORE assuming a particular shape fits this install | [`## Capabilities and modes`](#capabilities-and-modes) capability-query rule + [TASKS.md ## configure](TASKS.md#configure) step 3 |
| 6. Diagnose an exporter error | Map symptom (`BAD_STATE`, `INVALID_VALUE`, `AGAIN`, `NOT_PERMITTED`, `NOT_FOUND`, `DRIVER`) to root cause; in particular recognise `AGAIN` as a *transport-full, drop-or-buffer-bounded, never-block* signal, not a sleep-and-retry signal | [`## Error taxonomy`](#error-taxonomy) + [TASKS.md ## debug](TASKS.md#debug) |

Two cross-cutting rules that apply to *every* pattern above:

- **Telemetry must never block the application's data path.** When
  the transport is congested (`DOCA_ERROR_AGAIN`), the correct app
  behavior is to DROP the event or buffer it bounded — never block
  the hot path on a telemetry submit. An agent that recommends a
  blocking retry-loop on `AGAIN` is wrong for every release of the
  exporter; that pattern destroys the very application latency the
  user is presumably trying to measure.
- **The exporter is the publisher; the receiver is a separate
  artifact.** `doca-telemetry-exporter` is what the user's
  application links to *send* telemetry. The aggregating /
  collecting side (the DOCA Telemetry Service) is a separate DOCA
  service with its own public guide, reachable through
  [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md).
  Conflating the two is the #1 first-app confusion and the agent
  must surface the distinction before any code-level guidance.

## Capabilities and modes

DOCA Telemetry Exporter is a **DOCA Core Context** with one
exporter context object that owns one or more schemas, one or more
sources, and a transport to the receiving consumer. Every exporter
instance follows the universal `cfg-create → cfg-set-* → init →
start → use → stop → destroy` lifecycle (see
[`doca-programming-guide CAPABILITIES.md ## Capabilities and modes`](../../doca-programming-guide/CAPABILITIES.md#capabilities-and-modes)).
On top of that lifecycle, the exporter layers an asymmetric
publisher / receiver role split, a small object family, and a
capability-query family.

**Role split — exporter (publisher) vs telemetry service
(receiver).** The exporter is asymmetric and the asymmetry is the
#1 first-app confusion.

| Side | What it does | What it does NOT do | Where it lives |
| --- | --- | --- | --- |
| Exporter (this library) | Application-side publishing of structured telemetry events: register schemas describing event shape, create one or more sources, submit values into the transport | Aggregate, persist, query, fan-out, or downstream-route telemetry; subscribe back into the app on someone else's events | Linked INTO the user's DOCA-using application; runs as the application's user, in the application's process |
| Telemetry receiver (separate DOCA service, out of scope here) | Receive telemetry from one or more exporters, aggregate / persist / forward to downstream sinks (NetFlow / IPFIX / Prometheus / …) | Publish telemetry events itself | A separate DOCA service with its own public guide; reach via [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md). NOT part of this skill |

**Object family.** The exporter exposes ONE root context plus a
small set of cooperating object types. The agent must not invent
additional ones; the public surface is closed.

| Object | What it represents | Per-instance scope | Notes |
| --- | --- | --- | --- |
| `doca_telemetry_exporter_schema` | The shape of a telemetry event the app will publish — field names + field types | One per distinct event shape the application emits | MUST be registered with the exporter BEFORE the first emit referencing it; emitting against an unknown schema returns `DOCA_ERROR_NOT_FOUND` |
| `doca_telemetry_exporter_source` | One logical source of telemetry events from the app's perspective (per-worker, per-pipeline-stage, per-tenant) | The app may create multiple sources from one exporter context, e.g. one source per worker thread | Used at emit time as the *who-is-reporting* anchor; aggregated by the receiver for fan-out / filtering |
| `doca_telemetry_exporter_type` | The data type of a telemetry value: counter (monotonic), gauge (instantaneous), event (one-shot structured record) | Per field declared in a schema | Selection drives the receiver's downstream-rendering choice (counters → rate, gauges → snapshot, events → timeline). The right type is workload-bound and the agent should NOT default to one without asking |

**Capability discovery — the only rule.** Before sizing a schema,
choosing an event payload size, or assuming a particular event type
is on this install, call the matching
`doca_telemetry_exporter_*_get_*` query family (per the
cross-cutting cap-query rule in
[`doca-version CAPABILITIES.md ## Observability`](../../doca-version/CAPABILITIES.md#observability)).
The query is the runtime authority for *"is this exporter shape
supported on this install"*. Quoting *"the exporter supports X
fields"* from memory without the cap query is the silent-fail case
— when the install's actual cap is lower, the program returns
`DOCA_ERROR_INVALID_VALUE` at schema-register time and the user
has no idea why. The agent MUST quote the queried values back to
the user; the canonical list of `_get_*` queries that exist on a
particular install is in the public exporter headers (per the
headers-win-over-docs rule in [`doca-version`](../../doca-version/SKILL.md)).

**Path selection — exporter vs the adjacent options.** The exporter
is for structured telemetry publishing INTO the DOCA telemetry
ecosystem. It is not the answer for every "I want to emit
something" question; the agent must walk this rule before
recommending exporter setup.

| Use DOCA Telemetry Exporter when … | Use a different primitive when … |
| --- | --- |
| Structured application telemetry (counters / gauges / events) is the goal and the receiver is the DOCA telemetry ecosystem (DOCA Telemetry Service → downstream NetFlow / IPFIX / Prometheus integrations) | Simple stdout / per-line structured logging is enough — use `doca_log` (covered in [`doca-programming-guide`](../../doca-programming-guide/SKILL.md)); the exporter's schema discipline is overhead the user doesn't need |
| The user wants the receiver-side ecosystem (rate aggregation, downstream routing, multi-source fan-out) the DOCA Telemetry Service already provides | The user needs a custom non-DOCA-aware telemetry sink (raw Prometheus scrape endpoint, custom binary protocol) — use a Prometheus client library directly, not the DOCA exporter |
| The flow is one-way publishing from the app to one or more receivers | The user actually needs a real-time event subscription back INTO the application (events flowing app ← receiver) — the exporter is publish-only / one-way; route to [`doca-comch`](../doca-comch/SKILL.md) for a host ↔ DPU message channel the app can drive both ways |

**Configuration shape.** *Mandatory* configurations before
`doca_ctx_start()` on the exporter context: at least one
`doca_telemetry_exporter_schema` registered, and the transport
target the exporter will write to (the receiver consumer, reachable
on the host the application runs on, started BEFORE the exporter
per [`## Safety policy`](#safety-policy)). *Optional* configurations
(buffer sizing, per-source naming, additional schemas) follow the
standard DOCA Core surface; defaults come from the library and the
active install. The exact transport options surface (which
on-host transports the library supports today — local domain
socket / on-host telemetry agent / NetFlow consumer / …) is
install-bound and the agent MUST route the user to the public
exporter guide (reachable via
[`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md))
for the current matrix rather than committing to one transport.

## Version compatibility

For the canonical DOCA version-detection chain, the four-way match rule, NGC container semantics, and the headers-win-over-docs rule, see [`doca-version`](../../doca-version/SKILL.md). The body lives there; this skill does not duplicate it.

**The exporter-specific overlay** is:

- **Use `pkg-config --modversion doca-telemetry-exporter` as the build-time anchor.** Per [`doca-version TASKS.md ## configure`](../../doca-version/TASKS.md#configure), this MUST match the other version sources in the four-way match. The set of event types / max schema fields / max event size that the exporter actually supports on a given install is bound to BOTH the DOCA version AND the active `doca_devinfo`; agent-memory limits are not authoritative and MUST be replaced with a `doca_telemetry_exporter_*_get_*` query at runtime (per the cross-cutting cap-query rule in [`doca-version CAPABILITIES.md ## Observability`](../../doca-version/CAPABILITIES.md#observability)).
- **The exporter is distinct from the DOCA Telemetry Service across every release.** When the user reports *"the docs I'm reading talk about a telemetry SERVICE — is that this library?"*, the answer is no — that is a separate DOCA artifact. Route to [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md) for the service guide; the exporter is the library the user's application links to publish events INTO the service / ecosystem.
- **`doca-telemetry-exporter.pc` plus `doca-common.pc` must both match `doca_caps --version`** at the four-way-match check (per [`doca-version CAPABILITIES.md ## Version compatibility`](../../doca-version/CAPABILITIES.md#version-compatibility)). A common partial-install pattern after a DOCA upgrade is that `doca-telemetry-exporter.pc` lingers from the previous release while `doca-common.pc` was refreshed; route to [`doca-version TASKS.md ## debug`](../../doca-version/TASKS.md#debug) ladder step 2 before any exporter-layer diagnosis.

## Error taxonomy

Exporter-specific overlays on the cross-library `DOCA_ERROR_*`
taxonomy. The cross-library taxonomy itself lives in
[`doca-programming-guide CAPABILITIES.md ## Error taxonomy`](../../doca-programming-guide/CAPABILITIES.md#error-taxonomy);
the rows below are the *exporter surface* meaning that the agent
must disambiguate before falling back to the cross-library response.

| Error | Exporter context where it shows up | Exporter-specific cause |
| --- | --- | --- |
| `DOCA_ERROR_BAD_STATE` | Emit call before `doca_ctx_start()` on the exporter context; or schema-register call after the context is already in `RUNNING` and the library does not allow late registration on this install | Lifecycle violation. Walk the call sequence against the lifecycle in [`doca-programming-guide CAPABILITIES.md ## Capabilities and modes`](../../doca-programming-guide/CAPABILITIES.md#capabilities-and-modes); the most common case is emitting before the exporter context reaches `RUNNING`. |
| `DOCA_ERROR_INVALID_VALUE` | Schema-register call, emit call | An event field's type does not match the registered schema; the event payload is oversized relative to the install's max-event-size cap; a required field is missing from the emit. The fix is to re-read the registered schema and the matching `doca_telemetry_exporter_*_get_*` cap query against the active install. |
| `DOCA_ERROR_AGAIN` | Emit call (per-event submit), under load | The transport queue to the receiving consumer is full. This is *not* a hardware error and *not* a sleep-and-retry signal. The application's correct response is to DROP the event (or buffer it bounded at the app layer) and continue the data path. NEVER block the hot path on telemetry retries — that pattern destroys the latency the user is presumably trying to measure. See [`## Safety policy`](#safety-policy) hot-path rule. |
| `DOCA_ERROR_NOT_PERMITTED` | Exporter context create, first emit | The application's user cannot write to the configured telemetry transport (e.g. the local socket the receiver listens on is owned by a different user / permission set). Route to [`## Safety policy`](#safety-policy) permission matrix BEFORE any code change. The exporter itself does NOT need sudo as a rule; this error usually means the receiver's transport endpoint is what is locked down. |
| `DOCA_ERROR_NOT_FOUND` | Emit call referencing a schema or a source that was never registered with this exporter context | The schema (or the source) the emit names is unknown to the exporter. The fix is to register the schema with the exporter context BEFORE the first emit that references it (per the schema-register-before-emit rule in [`## Capabilities and modes`](#capabilities-and-modes)). |
| `DOCA_ERROR_DRIVER` | Exporter context create, first emit | The transport layer below DOCA reported failure (socket / network / on-host agent). Capture state and route to env-class debug ([`doca-setup ## debug`](../../doca-setup/TASKS.md#debug)) — the layer below DOCA is the suspect, not the exporter program. |

The agent's rule: **never recommend a retry loop on
`DOCA_ERROR_*` without first identifying which of the rows above
is the cause**. None of the exporter rows wants a blind retry:
`AGAIN` wants a DROP (or bounded buffer) at the app layer;
`NOT_FOUND` wants the schema registered first; the others want
investigation, not retry.

## Observability

The exporter's observability surface is per-emit + capability
snapshot + the consumer side. There is no PE-based completion
stream on the publisher itself the agent can subscribe to —
visibility comes from inspecting each emit's return value, the
configure-time cap snapshot, and the receiving consumer's own
log / counters.

Three primary signals the agent should reach for:

1. **Per-emit return.** Every `doca_telemetry_exporter_*` emit
   call returns a `doca_error_t`. The agent must inspect it on
   every emit: success means the event entered the transport;
   `DOCA_ERROR_AGAIN` means the transport was full and the app
   should drop / buffer (per [`## Safety policy`](#safety-policy));
   `DOCA_ERROR_*` of any other shape means the emit did not
   reach the transport and the row in
   [`## Error taxonomy`](#error-taxonomy) names the fix.
2. **Capability snapshot at configure time.** The output of every
   `doca_telemetry_exporter_*_get_*` query is a snapshot of *what
   the exporter said was possible* before any emit. Save it as
   the baseline; if a later emit returns `DOCA_ERROR_INVALID_VALUE`
   the diff against this snapshot is the bug, not the emit call
   itself. Cap-query at configure time is the cheapest way to
   make a later `INVALID_VALUE` self-explanatory.
3. **Consumer-side reception (end-to-end).** The exporter's
   *true* observability is at the receiver: did the consumer
   actually log the event? An exporter that returns success on
   every emit while the consumer log is empty is the canonical
   *"emit succeeded into a transport with no reader"* failure —
   route via the receiver's own guide (reach through
   [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md)).
   The smoke-emit / verify-reception loop in
   [`TASKS.md ## test`](TASKS.md#test) is the bundle's
   end-to-end check before any bulk emit.

For the cross-cutting observability primitives
(`--sdk-log-level`, the `doca-<lib>-trace` build flavor, the
`DOCA_LOG_LEVEL` env var) see
[`doca-debug CAPABILITIES.md ## Observability`](../../doca-debug/CAPABILITIES.md#observability).
For the install-tree observability (logger names, package layout)
defer to
[`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md).

## Safety policy

> **Overlay on the bundle-wide hardware-safety meta-policy.** The rules below are this skill's per-artifact overlay on the cross-cutting rules in [`doca-hardware-safety` CAPABILITIES.md ## Safety policy](../../doca-hardware-safety/CAPABILITIES.md#safety-policy) (specifically [### Per-artifact overlay pattern](../../doca-hardware-safety/CAPABILITIES.md#per-artifact-overlay-pattern)). When the two layers disagree, the stricter wins; when either layer says STOP, the agent stops.

The exporter's safety surface is **same-user-as-the-app
permissions + consumer-up-first staging + hot-path drop-not-block
invariant**. Each one is the source of a specific first-app
failure the agent must prevent.

The **permission + staging matrix** the agent must walk for any
new exporter setup:

| Prerequisite | Required state | How the agent verifies | Where to fix |
| --- | --- | --- | --- |
| Application user can write to the telemetry transport | The exporter runs as the application's normal user (no sudo as a rule); that user has write access to whichever transport endpoint the receiver listens on | `id` (confirm the running user); inspect the receiver's transport endpoint ownership (e.g. socket permissions on the local consumer); the first emit returns success rather than `DOCA_ERROR_NOT_PERMITTED` | If the user thinks the exporter needs sudo, that is the bug — the exporter does not require sudo; what is locked down is the receiver's transport endpoint. Fix on the receiver / env side via the matching receiver guide reached through [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md), or via [`doca-setup`](../../doca-setup/SKILL.md) for the env-side staging |
| Receiving telemetry consumer is running BEFORE the exporter starts | The downstream telemetry receiver (the DOCA Telemetry Service or whichever consumer the exporter is configured to write to) is started and reachable BEFORE the exporter context is started; otherwise emitted events may be queued or dropped depending on the transport, with no app-side error | The user can independently confirm the receiver is up (its own logs / status); the first emit-then-observe smoke (per [`TASKS.md ## test`](TASKS.md#test)) sees the event on the receiver side | Start the receiver first; only then start the exporter-using application. This staging is on the user — the exporter cannot start the receiver on the user's behalf |
| Hot-path drop-not-block invariant | Application code that calls `doca_telemetry_exporter_*` emit on the data path treats `DOCA_ERROR_AGAIN` as a SIGNAL to drop the event (or push to a bounded app-side buffer) — never to block, sleep, or busy-retry on the hot path | Code review at modify-time per [`TASKS.md ## modify`](TASKS.md#modify); structured log shows event counts dropping rather than data-path latency rising under load | Edit the app's emit-site to non-blocking-on-`AGAIN`; if the drop rate is unacceptable, the answer is to widen the receiver's throughput (separate problem on the receiver side), NOT to back-pressure the data path through the exporter |

- **The exporter does NOT need sudo as a rule.** Unlike
  introspection-style DOCA libraries that read host memory from
  the DPU side, the telemetry exporter runs in the application's
  own process as the application's own user. If the user's first
  reaction to `DOCA_ERROR_NOT_PERMITTED` is to add `sudo`, walk
  them back: the cause is almost always the receiver's transport
  endpoint permissions, not a missing capability on the exporter
  process.
- **The exporter is one-way.** Do not invent a `_subscribe()` or
  `_callback_on_consumer_event()` shape; the library is
  publish-only. If the user needs the application to react to
  events flowing IN, the right primitive is a host ↔ DPU message
  channel via [`doca-comch`](../doca-comch/SKILL.md) (or the
  receiver's own API on the receiver side), not the exporter.
- **Validate with a one-event smoke before any bulk emit.** A
  single emit + receiver-side reception confirmation is the
  cheapest way to prove the permission + staging + transport
  path are all correct. If the smoke passes, bulk emits inherit
  that confidence; if the smoke says success-on-publish but
  nothing on the receiver, the consumer staging is the prime
  suspect. The loop is described in
  [`TASKS.md ## test`](TASKS.md#test).

## Deferred topic boundaries

This skill scopes itself to the DOCA Telemetry Exporter library.
Adjacent topics the agent will get asked but should route
elsewhere:

- **The DOCA Telemetry Service (the receiver)** — separate DOCA
  service with its own public guide. Reach via
  [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md).
  This skill is publisher-side only.
- **DOCA Core context and progress engine internals** — owned by
  [`doca-programming-guide`](../../doca-programming-guide/SKILL.md).
  This skill *uses* the Core context lifecycle; it does not
  redefine it.
- **Downstream telemetry rendering / dashboards** — out of scope.
  Once events reach the receiver, the downstream sinks
  (NetFlow / IPFIX / Prometheus / Grafana) are governed by their
  own ecosystems and the receiver's own guide.
- **Real-time event subscription back into the app** — the
  exporter is publish-only / one-way. The right primitive is
  [`doca-comch`](../doca-comch/SKILL.md) for a bi-directional
  message channel, not the exporter.
- **Plain structured logging to stdout / files** — use `doca_log`
  (documented in [`doca-programming-guide`](../../doca-programming-guide/SKILL.md));
  the exporter's schema discipline is overhead the user does not
  need for plain logs.
- **Cross-cutting `DOCA_ERROR_*` taxonomy** — owned by
  [`doca-programming-guide CAPABILITIES.md ## Error taxonomy`](../../doca-programming-guide/CAPABILITIES.md#error-taxonomy).
  This skill adds the exporter overlay (including the
  `AGAIN`-means-drop-not-block rule), not the taxonomy itself.
- **Cross-cutting debug ladder** (install / version / build /
  link / runtime / program / driver) — owned by
  [`doca-debug ## debug`](../../doca-debug/TASKS.md#debug). This
  skill's `## debug` overlays the runtime + program layers.
