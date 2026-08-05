# DOCA Telemetry Exporter capabilities, publish surfaces, version overlay, errors, observability, safety

**Where to start:** Pick the H2 anchor that matches your question
(role-split / object family / publish-surface selection / version
/ errors / observability / safety) and read that section
end-to-end. The tables in each section are the load-bearing
content; the prose around them is interpretation.

Read this file when the loader sent you here from
[SKILL.md](SKILL.md). For the *how* of executing each pattern (the
verbs `configure / build / modify / run / test / debug`), jump to
[TASKS.md](TASKS.md). For the canonical DOCA version-handling rules
that this skill layers an exporter overlay on top of, see
[`doca-version`](../../doca-version/SKILL.md).

> **Ground-truth note.** Every symbol, macro, error code, default
> path, and lifecycle step in this file is quoted from the public
> exporter headers (`doca_telemetry_exporter.h`,
> `doca_telemetry_exporter_netflow.h`) and the shipped samples
> under `/opt/mellanox/doca/samples/doca_telemetry_exporter/`.
> The headers-win-over-docs rule
> ([`doca-version`](../../doca-version/SKILL.md)) applies: if the
> installed header disagrees with anything here, the header is
> authoritative and the agent must re-read it before answering.

## Pattern overview

Every telemetry-exporter question this skill teaches resolves into
one of SIX patterns. The patterns are CLASSES — they apply across
every exporter release and every DOCA-using application, not just
the worked examples shown.

| Pattern | When it applies (class shape) | Where the substance lives |
| --- | --- | --- |
| 1. Pick exporter, not the service | The application is the PUBLISHER of telemetry; the receiving / aggregating side (DOCA Telemetry Service, DTS) is a separate DOCA artifact out of this skill's scope | [`## Capabilities and modes`](#capabilities-and-modes) role-split table |
| 2. Pick the right publish surface | The library exposes FOUR distinct publish surfaces — typed structured events, opaque events, the Metrics API (counter/gauge/histogram), and the NetFlow/IPFIX API — plus the OTLP-logs API layered on a source. Picking the wrong one is the #1 first-app confusion | [`## Capabilities and modes`](#capabilities-and-modes) publish-surface table |
| 3. Register the schema/type BEFORE the first report | Typed events are shaped by a `doca_telemetry_exporter_type` added to a `doca_telemetry_exporter_schema` and finalized with `doca_telemetry_exporter_schema_start()` BEFORE any source is created; reporting against a source that was not started returns `DOCA_ERROR_BAD_STATE` | [`## Capabilities and modes`](#capabilities-and-modes) object table + [TASKS.md ## configure](TASKS.md#configure) |
| 4. Walk the schema → source lifecycle | `schema_init` → configure exporters (file / IPC / opaque) → `type_create` + `field_*` + `schema_add_type` → `schema_start` → `source_create` → `source_set_id/tag` → `source_start` → report → `source_flush` → `source_destroy` → `schema_destroy` | [`## Capabilities and modes`](#capabilities-and-modes) object table + [TASKS.md ## configure](TASKS.md#configure) |
| 5. Confirm the transport, not device caps | This library has NO per-`doca_devinfo` capability-query family and NO `doca_caps` data (its `doca_caps` dump is a stub). The real preconditions are transport ones: is a receiver up, is IPC connected (`doca_telemetry_exporter_check_ipc_status`), is the file-write data root writable | [`## Capabilities and modes`](#capabilities-and-modes) transport rule + [`## Observability`](#observability) |
| 6. Diagnose an exporter error | Map the actual documented codes (`DOCA_ERROR_BAD_STATE`, `DOCA_ERROR_INVALID_VALUE`, `DOCA_ERROR_NO_MEMORY`, `DOCA_ERROR_INITIALIZATION`, `DOCA_ERROR_UNKNOWN`) to root cause; there is NO `DOCA_ERROR_AGAIN` and NO `DOCA_ERROR_NOT_FOUND` on this API surface | [`## Error taxonomy`](#error-taxonomy) + [TASKS.md ## debug](TASKS.md#debug) |

Two cross-cutting rules that apply to *every* pattern above:

- **The exporter is the publisher; the receiver is a separate
  artifact.** `doca-telemetry-exporter` is what the user's
  application links to *publish* telemetry. The aggregating /
  collecting side (the DOCA Telemetry Service, DTS — reached over
  IPC) and any downstream OpenTelemetry Collector are separate,
  each with its own public guide reachable through
  [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md).
  Conflating the two is the #1 first-app confusion and the agent
  must surface the distinction before any code-level guidance.
- **Delivery is buffered and flush-based, not a blocking submit.**
  Reports accumulate in an internal buffer that is flushed when
  full or on an explicit `doca_telemetry_exporter_source_flush()`
  (or the Metrics API's `_metrics_flush` / auto flush interval).
  A `_source_report` does NOT block on a full transport and does
  NOT return `DOCA_ERROR_AGAIN` — there is no such return in this
  API. The NetFlow `_netflow_send` reports how many records it
  actually sent via its `records_sent` out-parameter; a short
  send is the backpressure signal on that surface.

## Capabilities and modes

DOCA Telemetry Exporter is a **CollectX-backed publisher library**,
not a DOCA Core Context. It does NOT use the
`doca_ctx_create → doca_ctx_start` lifecycle and there is no
`doca_ctx` handle for the exporter. The lifecycle is instead a
**schema → source** progression, with three optional per-source
publish contexts (metrics / OTLP logs) and one sibling top-level
API (NetFlow).

**Role split — exporter (publisher) vs telemetry service
(receiver).** The exporter is asymmetric and the asymmetry is the
#1 first-app confusion.

| Side | What it does | What it does NOT do | Where it lives |
| --- | --- | --- | --- |
| Exporter (this library) | Application-side publishing: define event/type schemas, create sources, report typed / opaque events, metrics, and NetFlow records into the configured exporters (file write, IPC to DTS, Prometheus for metrics, UDP collector for NetFlow) | Aggregate, persist, query, fan-out, or downstream-route telemetry | Linked INTO the user's DOCA-using application; runs as the application's user, in the application's process |
| Telemetry receiver (separate — DTS / OTel Collector / NetFlow collector) | Receive telemetry from one or more exporters, aggregate / persist / forward to downstream sinks | Publish telemetry itself | A separate DOCA service (DTS) or third-party collector with its own guide; reach via [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md). NOT part of this skill |

**Object family.** The public surface is closed; the agent must not
invent additional object types or a `doca_ctx`.

| Object / handle | What it represents | Created / finalized by | Notes |
| --- | --- | --- | --- |
| `struct doca_telemetry_exporter_schema` | The root publisher configuration: which exporters are enabled (file / IPC / opaque), buffer size, and the set of registered event types | `doca_telemetry_exporter_schema_init(name, &schema)`; finalized by `doca_telemetry_exporter_schema_start(schema)`; freed by `doca_telemetry_exporter_schema_destroy` | Configure exporters with the `doca_telemetry_exporter_schema_set_*` family BEFORE `schema_start`. Do NOT add types after `schema_start` |
| `struct doca_telemetry_exporter_type` | The shape (field list) of ONE structured event record | `doca_telemetry_exporter_type_create(&type)`, fields added via `doca_telemetry_exporter_type_add_field`, registered into a schema with `doca_telemetry_exporter_schema_add_type(schema, name, type, &type_index)` | `schema_add_type` returns a `doca_telemetry_exporter_type_index_t` — the handle you pass to `_source_report`. Duplicate type name or bad field type → `DOCA_ERROR_INVALID_VALUE` |
| `struct doca_telemetry_exporter_field` | One field of a type: name, description, type-name, array length | `doca_telemetry_exporter_field_create`, then `_field_set_name` / `_field_set_description` / `_field_set_type_name` / `_field_set_array_len` | Field type is one of the `DOCA_TELEMETRY_EXPORTER_FIELD_TYPE_*` string macros (e.g. `..._UINT64`, `..._INT32`, `..._CHAR`, `..._TIMESTAMP`). Ownership transfers to the type on `_type_add_field` |
| `struct doca_telemetry_exporter_source` | One logical reporter (per-worker / per-pipeline-stage / per-tenant) | `doca_telemetry_exporter_source_create(schema, &source)` (schema must be started first), `_source_set_id` / `_source_set_tag`, then `doca_telemetry_exporter_source_start(source)` | Report only after `source_start`. Multiple sources per schema are allowed. `_source_flush` forces a buffer flush; `_source_destroy` tears it down |

**Publish surfaces.** ONE source can drive several publish
surfaces. The agent MUST pick the surface that matches intent
before writing any report call.

| Publish surface | How to publish | Precondition | Stability |
| --- | --- | --- | --- |
| Typed structured events | `doca_telemetry_exporter_source_report(source, type_index, data, count)` where `data` is an array of `count` packed records matching the registered type | A `type` was added via `_schema_add_type` before `schema_start`; source started | `DOCA_EXPERIMENTAL` |
| Opaque events | `doca_telemetry_exporter_source_opaque_report(source, app_id, user_defined1, user_defined2, data, data_size)`; size bound via `doca_telemetry_exporter_source_get_opaque_report_max_data_size` | `doca_telemetry_exporter_schema_set_opaque_events_enabled(schema)` BEFORE `schema_start` | `DOCA_STABLE` |
| Metrics (counter / gauge / histogram) | Create a metrics context on a STARTED source with `doca_telemetry_exporter_metrics_create_context(source)`, then `_metrics_add_counter` / `_add_counter_increment` / `_add_gauge` / `_add_gauge_uint64` / `_add_histogram` / `_add_base_histogram` / `_metrics_histogram_observe` / `_metrics_base_histogram_observe`; labels via `_metrics_add_constant_label` / `_metrics_add_label_names`; flush via `_metrics_flush` or `_metrics_set_flush_interval_ms` | Source started; metrics context created after `source_start` | `DOCA_EXPERIMENTAL` |
| OTLP logs | Create an OTLP-logs context on a STARTED source with `doca_telemetry_exporter_otlp_logs_create_context(source)`, add resource(s) + scope(s), then `_otlp_logs_write_event` / `_write_event_with_severity`, `_otlp_logs_flush` | `doca_telemetry_exporter_schema_set_opaque_events_enabled(schema)` (OTLP logs ride the opaque path); source started | `DOCA_EXPERIMENTAL` |
| NetFlow / IPFIX records | Sibling top-level API in `doca_telemetry_exporter_netflow.h`: `_netflow_init(source_id)` → `_netflow_set_collector_addr/port` → `_netflow_source_set_id/tag` → build a `_netflow_template` (`_netflow_template_create` + `_netflow_field_*` + `_netflow_template_add_field`) → `_netflow_start` → `_netflow_send(template, records, count, &records_sent)` → `_netflow_flush` → `_netflow_destroy` | NetFlow field IDs are the `DOCA_TELEMETRY_EXPORTER_NETFLOW_*` macros (IPFIX element IDs) | `DOCA_STABLE` |

**Export destinations (transports).** Enabled on the schema (or via
the NetFlow API / env var) BEFORE start. Multiple may be enabled
at once.

| Destination | How to enable | Where it goes | Default path / knob |
| --- | --- | --- | --- |
| File write | `doca_telemetry_exporter_schema_set_file_write_enabled(schema)` | Binary telemetry files on local disk (dev / offline ingestion) | Data root default documented in the header as `/opt/mellanox/doca/services/telemetry/data/`; override with `_schema_set_buf_data_root`. File rotation via `_schema_set_file_write_max_size` / `_max_age` |
| IPC to DTS | `doca_telemetry_exporter_schema_set_ipc_enabled(schema)` | Streams to the DOCA Telemetry Service over a local socket | Socket dir default documented in the header as `/opt/mellanox/doca/services/telemetry/ipc_sockets`; override with `_schema_set_ipc_sockets_dir`. Reconnect knobs: `_set_ipc_reconnect_time` / `_reconnect_tries` / `_socket_timeout`. Liveness via `doca_telemetry_exporter_check_ipc_status` |
| Prometheus (metrics only) | Set `PROMETHEUS_ENDPOINT=host:port` in the environment before `schema_init` | Exposes a `/metrics` scrape endpoint | Env-var driven, per the metrics sample; not a `schema_set_*` call |
| NetFlow collector | `_netflow_set_collector_addr` / `_netflow_set_collector_port` (and/or `_netflow_set_ipc_enabled` / `_netflow_set_file_write_enabled`) | UDP NetFlow/IPFIX collector | Address/port are caller-supplied |

**Capability discovery — the honest rule.** Unlike accelerator
libraries (SHA, RDMA, Compress, Flow), DOCA Telemetry Exporter has
**no `doca_devinfo` capability-query family** and **no `doca_caps`
data** — the `doca_caps` telemetry-exporter dump is a stub that
returns success with no fields. Do NOT tell the user to run a
`doca_telemetry_exporter_*_get_*` "cap query" for max fields / max
event size / supported event types; those functions do not exist.
The only runtime introspection the library offers is:

- `doca_telemetry_exporter_check_ipc_status(source, &status)` —
  is the IPC transport `CONNECTED` / `FAILED` / `DISABLED`.
- `doca_telemetry_exporter_source_get_opaque_report_max_data_size(source, &max)` —
  the max opaque payload size for this source.
- the `doca_telemetry_exporter_schema_get_*` config getters —
  read back the buffer / file-write / IPC settings you set.

Everything else is validated the sample-driven way: enable file
write, run, and inspect the emitted binary data files (and/or
confirm receipt at the DTS / collector). See
[`## Observability`](#observability).

## Version compatibility

For the canonical DOCA version-detection chain, the four-way match rule, NGC container semantics, and the headers-win-over-docs rule, see [`doca-version`](../../doca-version/SKILL.md). The body lives there; this skill does not duplicate it.

**The exporter-specific overlay** is:

- **Use `pkg-config --modversion doca-telemetry-exporter` as the build-time anchor.** Per [`doca-version TASKS.md ## configure`](../../doca-version/TASKS.md#configure), this MUST match the other version sources. Because there is no `doca_caps` data for this library, the four-way match's `doca_caps --version` leg is a whole-install version check, NOT a per-artifact capability probe for the exporter.
- **The Metrics API, OTLP-logs API, typed-`_source_report`, and `_schema_set_indexes` / `_get_indexes` are `DOCA_EXPERIMENTAL`.** They require building with `-DDOCA_ALLOW_EXPERIMENTAL_API` (per [`doca-programming-guide`](../../doca-programming-guide/SKILL.md)); they may change across releases. The `DOCA_STABLE` surface includes the schema → source core (`schema_init` / `schema_start` / `source_create` / `source_start` / `source_destroy` / `schema_destroy`), typed-schema registration (`type_create` / `field_create` / `type_add_field` / `schema_add_type`), `source_opaque_report` / `source_flush` / `check_ipc_status` / `get_timestamp`, the full `schema_set_*` / `schema_get_*` config family, and the entire NetFlow sibling API in `doca_telemetry_exporter_netflow.h`. Confirm the `DOCA_STABLE` / `DOCA_EXPERIMENTAL` marker in the installed header before relying on any symbol — the shipped samples still set `-DDOCA_ALLOW_EXPERIMENTAL_API` blanket-style even when only stable symbols are used.
- **The exporter is distinct from the DOCA Telemetry Service across every release.** When the user reports *"the docs I'm reading talk about a telemetry SERVICE — is that this library?"*, the answer is no. Route to [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md) for the service guide.
- **`doca-telemetry-exporter.pc` plus `doca-common.pc` must both match `doca_caps --version`** at the version check (per [`doca-version CAPABILITIES.md ## Version compatibility`](../../doca-version/CAPABILITIES.md#version-compatibility)). A common partial-install pattern after a DOCA upgrade is that `doca-telemetry-exporter.pc` lingers from the previous release while `doca-common.pc` was refreshed; route to [`doca-version TASKS.md ## debug`](../../doca-version/TASKS.md#debug) ladder step 2 before any exporter-layer diagnosis.

## Error taxonomy

Only the `doca_error_t` codes the exporter headers actually document
appear below. There is **no `DOCA_ERROR_AGAIN`** (delivery is
buffered/flush-based, not a blocking submit) and **no
`DOCA_ERROR_NOT_FOUND`** on this API surface — do not invent them.
The cross-library taxonomy lives in
[`doca-programming-guide CAPABILITIES.md ## Error taxonomy`](../../doca-programming-guide/CAPABILITIES.md#error-taxonomy);
the rows below are the *exporter surface* meanings.

| Error | Exporter context where it shows up | Exporter-specific cause / fix |
| --- | --- | --- |
| `DOCA_ERROR_BAD_STATE` | `doca_telemetry_exporter_source_report` / `_source_opaque_report` when the source was never started; `_otlp_logs_*` when the OTLP context does not exist (or the source is not started, or context already exists on create); NetFlow `_netflow_*` when NetFlow was not initialized / not started / already initialized (per `doca_telemetry_exporter_netflow.h`) | Lifecycle order violation. For schema/source: `schema_start` before `source_create`, `source_start` before any report, and create the OTLP context only AFTER `source_start`. For NetFlow: `_netflow_init` → configure → `_netflow_start` before `_netflow_send` |
| `DOCA_ERROR_INVALID_VALUE` | NULL handle to almost any call; `_schema_add_type` with a duplicate type name or a field carrying an invalid `DOCA_TELEMETRY_EXPORTER_FIELD_TYPE_*`; a `_metrics_*` call with a NULL handle, mismatched label arity, or invalid histogram ID; an `_otlp_logs_*` call with a NULL or mismatched-arity argument | Re-read the registered type/field definitions and the label-set arity. For events, confirm the packed struct matches the field list you registered. For metrics, note the header documents `INVALID_VALUE` / `NO_MEMORY` on `_metrics_*` — not `BAD_STATE` |
| `DOCA_ERROR_NO_MEMORY` | `_schema_init`, `_type_create`, `_field_create`, `_schema_add_type`, `_source_create`, metrics/histogram allocation, NetFlow template build | Allocation failure — an env / resource problem, not an API-usage bug. Capture and route to [`doca-setup ## debug`](../../doca-setup/TASKS.md#debug) |
| `DOCA_ERROR_INITIALIZATION` | `_schema_init`, `_schema_start` | The schema could not be initialized / finalized (often a bad exporter configuration, e.g. an unwritable data root, or a CollectX-backend init failure). Re-check the enabled exporters and their paths before retrying |
| `DOCA_ERROR_UNKNOWN` | `_schema_set_indexes` / `_schema_get_indexes` and other experimental paths where the underlying CollectX operation failed for a non-specific reason | Treat as a CollectX-backend failure; capture the trace log and route to [`doca-debug ## debug`](../../doca-debug/TASKS.md#debug) |

The agent's rule: **identify which row above is the cause before
recommending any change; never invent an `AGAIN`/`NOT_FOUND` retry
loop.** A `BAD_STATE` wants the lifecycle fixed; an
`INVALID_VALUE` wants the type/field/label definition fixed.

## Observability

The exporter's observability is per-call return + IPC status +
the emitted data itself + the receiver side. There is no
progress-engine completion stream (this is not a Core context).

Four primary signals the agent should reach for:

1. **Per-call return.** Every `doca_telemetry_exporter_*` call
   returns a `doca_error_t`; inspect it. For NetFlow,
   ALSO inspect the `records_sent` out-parameter of
   `_netflow_send` — `records_sent < count` is a partial send
   (backpressure at the collector / transport), which the
   NetFlow sample treats as a failure.
2. **IPC status.** `doca_telemetry_exporter_check_ipc_status`
   returns `DOCA_TELEMETRY_EXPORTER_IPC_STATUS_CONNECTED` /
   `_FAILED` / `_DISABLED`. When IPC is enabled but the DTS is
   not up, this is where the agent sees it — BEFORE blaming the
   report calls.
3. **File-write inspection (the cheapest end-to-end check).**
   Enable `_schema_set_file_write_enabled`, run, and inspect the
   binary data files under the data root. The shipped samples
   themselves rely on this as the "did it work" signal. A file
   with the expected schema + rows is the green signal for the
   publish half of the contract.
4. **Receiver-side reception (true end-to-end).** For IPC → DTS,
   OTLP → OpenTelemetry Collector, or NetFlow → collector, the
   *true* signal is that the receiver logged / stored the events.
   An exporter returning success on every call while the receiver
   is empty is the canonical *"published into a transport with no
   reader"* failure — route via the receiver's own guide reached
   through [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md).

For the cross-cutting observability primitives
(`DOCA_LOG_LEVEL`, the `doca-<lib>-trace` build flavor) see
[`doca-debug CAPABILITIES.md ## Observability`](../../doca-debug/CAPABILITIES.md#observability).
For the install-tree layout defer to
[`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md).

## Safety policy

> **Overlay on the bundle-wide hardware-safety meta-policy.** The rules below are this skill's per-artifact overlay on the cross-cutting rules in [`doca-hardware-safety` CAPABILITIES.md ## Safety policy](../../doca-hardware-safety/CAPABILITIES.md#safety-policy). Telemetry publishing does not change hardware or fabric state (no `mlxconfig`, no firmware, no eswitch/representor change), so the hardware-safety triggers do not fire here. The safety surface is **transport staging + destination permissions + buffer-flush semantics**.

The **staging + permission matrix** the agent must walk for any new
exporter setup:

| Prerequisite | Required state | How the agent verifies | Where to fix |
| --- | --- | --- | --- |
| File-write data root is writable | If `_schema_set_file_write_enabled` is on, the application user can write to the data root (default `/opt/mellanox/doca/services/telemetry/data/`, or the `_set_buf_data_root` override) | `id` + a write test to the data root; `schema_start` returns success rather than `DOCA_ERROR_INITIALIZATION` | Point the data root at a writable directory via `_schema_set_buf_data_root`; do NOT reflexively add `sudo` |
| Receiver up BEFORE export starts | If IPC is on, the DTS is running and its socket dir is reachable; if OTLP, the OpenTelemetry Collector is up; if NetFlow, the collector is listening | `doca_telemetry_exporter_check_ipc_status` returns `CONNECTED` (IPC); the receiver's own log/status for OTLP/NetFlow | Start the receiver first, then the app. The exporter cannot start the receiver; route to the receiver's guide via [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md) |
| Opaque path enabled before use | Opaque events and OTLP logs require `_schema_set_opaque_events_enabled` BEFORE `schema_start` | The first `_source_opaque_report` / `_otlp_logs_*` call does not return `DOCA_ERROR_BAD_STATE` for a missing context | Enable opaque events on the schema before `schema_start`; you cannot enable it after |
| Buffer is flushed before teardown | Buffered reports are flushed when the buffer fills or on explicit flush; on shutdown the app must flush so trailing events are not lost | The receiver / data file contains the final events, not just the pre-last-flush ones | Call `_source_flush` (events) / `_metrics_flush` (metrics) / `_netflow_flush` (NetFlow) before destroy, or configure a flush interval |

- **No sudo as a rule.** The exporter runs in the application's
  own process as the application's user. If the first reaction to
  an init failure is to add `sudo`, walk it back: the cause is
  almost always an unwritable data root or a receiver that is not
  up, not a missing privilege on the exporter process.
- **Delivery never blocks the data path.** Because reporting is
  buffered and flush-based (not a blocking submit), there is no
  hot-path back-pressure knob and no `DOCA_ERROR_AGAIN` to handle.
  If loss under load matters, tune buffer size
  (`_schema_set_buf_size`) and flush cadence, or widen the
  receiver — do NOT invent a retry loop.
- **Validate with a file-write smoke before any bulk / IPC run.**
  Enabling file write and inspecting the emitted data files is the
  cheapest way to prove the schema/type/report path is correct
  independent of any receiver. The loop is in
  [`TASKS.md ## test`](TASKS.md#test).

## Deferred topic boundaries

This skill scopes itself to the DOCA Telemetry Exporter library.
Adjacent topics the agent will get asked but should route
elsewhere:

- **The DOCA Telemetry Service (DTS, the receiver)** — separate
  DOCA service with its own public guide. Reach via
  [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md).
  This skill is publisher-side only.
- **CollectX (`clx`) collection deployment** — the collector-side
  deployment lives in
  [`doca-collectx-deployment`](../../doca-collectx-deployment/SKILL.md).
- **Downstream rendering / dashboards** — once events reach the
  receiver, the downstream sinks (NetFlow / IPFIX / Prometheus /
  Grafana / OpenTelemetry Collector) are governed by their own
  ecosystems and guides.
- **DOCA Core context and progress-engine internals** — owned by
  [`doca-programming-guide`](../../doca-programming-guide/SKILL.md).
  Note the exporter is NOT a Core context; do not apply the
  `doca_ctx` lifecycle to it.
- **Real-time event subscription back into the app** — the
  exporter is publish-only / one-way. The right primitive is
  [`doca-comch`](../doca-comch/SKILL.md) for a bi-directional
  message channel, not the exporter.
- **Plain structured logging to stdout / files** — use `doca_log`
  (documented in [`doca-programming-guide`](../../doca-programming-guide/SKILL.md)).
- **Cross-cutting `DOCA_ERROR_*` taxonomy** — owned by
  [`doca-programming-guide CAPABILITIES.md ## Error taxonomy`](../../doca-programming-guide/CAPABILITIES.md#error-taxonomy).
- **Cross-cutting debug ladder** — owned by
  [`doca-debug ## debug`](../../doca-debug/TASKS.md#debug). This
  skill's `## debug` overlays the runtime + program layers.
