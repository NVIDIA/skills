# DOCA Telemetry Exporter workflows

**Where to start:** The verbs run `configure → build → modify →
run → test → debug`. Skip ahead only when the user is already past a
verb. The `## test` verb is an iterative loop (file-write smoke →
receiver-side reception check → multi-event smoke → loop back if the
transport or schema changes), not a one-shot pass — see the
eval-loop overlay in `## test` below.

Read this file when the loader sent you here from
[SKILL.md](SKILL.md). For the exporter capability surface, the
exporter-vs-service role split, the object family, the
schema → source lifecycle, the four publish surfaces (typed
events / opaque events / metrics / OTLP logs) plus the NetFlow
sibling API, the error taxonomy, observability, and safety policy,
see [CAPABILITIES.md](CAPABILITIES.md). For where to find docs, the
installed DOCA layout, or release notes, route through
[`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md).

Each verb below describes the **shape of the workflow**, not a
copy-paste recipe. The agent's job is to walk the user through the
steps in order, verifying preconditions before recommending the next
call.

## configure

Goal: stand up a `doca_telemetry_exporter_schema` and at least one
started `doca_telemetry_exporter_source` inside the user's
application, with a reachable destination (file write and/or IPC to
DTS), before any event is reported.

> This library is NOT a DOCA Core context. There is no
> `doca_ctx_start()`. The lifecycle is `schema_init` → configure
> exporters → register type(s) → `schema_start` → `source_create`
> → `source_start` → report → flush → destroy.

Steps the agent should walk the user through:

1. **Confirm the role: this is the PUBLISHER side.** Before any
   code change, surface the exporter-vs-service distinction per the
   role-split table in
   [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes).
   `doca-telemetry-exporter` is what the user's application links
   to publish; the aggregating / receiving side is the DOCA
   Telemetry Service (DTS), a separate DOCA service with its own
   public guide reachable via
   [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md).
   State the role first.
2. **Pick the publish surface.** Walk the publish-surface table in
   [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes)
   and choose ONE (or more) deliberately: typed structured events
   (`_source_report`), opaque events (`_source_opaque_report`),
   the Metrics API (`_metrics_add_counter` / `_add_gauge` /
   `_add_histogram`), OTLP logs (`_otlp_logs_*`), or the NetFlow
   sibling API (`_netflow_*`). The surface drives the rest of the
   lifecycle — e.g. opaque events and OTLP logs require
   `_schema_set_opaque_events_enabled` before `schema_start`;
   metrics/OTLP contexts are created only AFTER `source_start`.
3. **Confirm the installed DOCA version.** Use the procedure in
   [`doca-version TASKS.md ## configure`](../../doca-version/TASKS.md#configure).
   Quote the version observed (`pkg-config --modversion
   doca-telemetry-exporter`, then `doca_caps --version`); do not
   assume "latest". Note: there is NO per-device capability query
   and NO `doca_caps` data for this library — do not tell the user
   to run a `doca_telemetry_exporter_*_get_*` cap query for limits;
   those functions do not exist (see
   [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes)
   capability-discovery rule).
4. **Init and configure the schema.**
   `doca_telemetry_exporter_schema_init(name, &schema)`, then set
   the exporters you need BEFORE `schema_start`:
   `_schema_set_file_write_enabled` (file), `_schema_set_ipc_enabled`
   (IPC → DTS), `_schema_set_opaque_events_enabled` (if using
   opaque events or OTLP logs), and optionally `_schema_set_buf_size`
   / `_schema_set_buf_data_root` / the IPC reconnect knobs. Defaults
   for the data root and IPC socket dir are documented in the header
   (do not hardcode them as universal — read them back with the
   `_schema_get_*` getters if needed).
5. **Register the event type(s) — only for the typed-events
   surface.** `doca_telemetry_exporter_type_create(&type)`, add
   fields with `_field_create` + `_field_set_name` /
   `_set_description` / `_set_type_name` (a
   `DOCA_TELEMETRY_EXPORTER_FIELD_TYPE_*` macro) / `_set_array_len`
   + `_type_add_field`, then
   `doca_telemetry_exporter_schema_add_type(schema, type_name,
   type, &type_index)`. Keep the `type_index` — it is the handle
   for `_source_report`. Do NOT add types after `schema_start`.
6. **Finalize the schema.**
   `doca_telemetry_exporter_schema_start(schema)`. After this no
   new types may be added.
7. **Create and start the source(s).**
   `doca_telemetry_exporter_source_create(schema, &source)`,
   `_source_set_id` / `_source_set_tag`, then
   `doca_telemetry_exporter_source_start(source)`. One source per
   logical reporter (per-worker / per-pipeline-stage / per-tenant);
   do not default to one source for the whole app without asking.
8. **Create the per-surface context if needed.** For metrics:
   `doca_telemetry_exporter_metrics_create_context(source)` AFTER
   `source_start`, then add labels. For OTLP logs:
   `doca_telemetry_exporter_otlp_logs_create_context(source)` AFTER
   `source_start`, then add resource(s) + scope(s). For NetFlow the
   flow is the separate `_netflow_init` → `_netflow_start` sequence
   in `doca_telemetry_exporter_netflow.h`.

If any step fails with a `DOCA_ERROR_*`, route through the error
taxonomy in
[`CAPABILITIES.md ## Error taxonomy`](CAPABILITIES.md#error-taxonomy)
before retrying. The most common first-app failure is a lifecycle
order violation returning `DOCA_ERROR_BAD_STATE` (reporting before
`source_start`, or OTLP calls before the OTLP context exists). Metrics
misuse more often surfaces as `DOCA_ERROR_INVALID_VALUE` per the header.

## build

Goal: produce an application binary that links DOCA Telemetry
Exporter against the user's installed DOCA, using the canonical
cross-library build pattern.

The build pattern for any DOCA C/C++ consumer is **identical**
across libraries — `pkg-config` for include + link flags, meson or
CMake as the build system — and is fully documented in
[`doca-programming-guide TASKS.md ## build`](../../doca-programming-guide/TASKS.md#build).
This skill carries only the exporter-specific overlay:

| Slot | Value for Telemetry Exporter | Why it matters |
| --- | --- | --- |
| `pkg-config` module name | `doca-telemetry-exporter` | The library's `.pc` file installed by the DOCA host packages. Wrong module name = `pkg-config: Package 'doca-telemetry-exporter' was not found` (most often the user typed `doca-telemetry` and was reading the receiving service's guide by mistake — re-check role) |
| Include flags | `pkg-config --cflags doca-telemetry-exporter` | Resolves to the exporter headers (`doca_telemetry_exporter.h`, `doca_telemetry_exporter_netflow.h`) on this install |
| Link flags | `pkg-config --libs doca-telemetry-exporter` | Pulls in whatever `pkg-config --libs` resolves on this install (do not predict the `-l<name>` form by hand — `.so` basenames use underscores, `.pc` names use hyphens, and `pkg-config` is the only correct translator) plus the transitive set the resolver computes |
| Experimental-API flag | `-DDOCA_ALLOW_EXPERIMENTAL_API` when using `_source_report`, the Metrics API, the OTLP-logs API, or `_schema_set_indexes` / `_get_indexes` | Those symbols are `DOCA_EXPERIMENTAL` (per [`CAPABILITIES.md ## Version compatibility`](CAPABILITIES.md#version-compatibility)); without the flag they will not compile. The `DOCA_STABLE` surface — schema/type/field registration, schema → source lifecycle, opaque events, NetFlow (`doca_telemetry_exporter_netflow.h`), `source_flush`, `check_ipc_status`, `get_timestamp`, and the `schema_set_*` / `schema_get_*` config family — does not need it. Shipped samples still set the flag blanket-style; a NetFlow-only or opaque-only app does not require it on current headers |
| Header check | the artifact's public headers resolvable under whichever include directory `pkg-config --cflags` reports (do not hardcode the include path) | If `pkg-config --cflags doca-telemetry-exporter` resolves but the include is missing, the install is partial — route to [`doca-version TASKS.md ## debug`](../../doca-version/TASKS.md#debug) layer 2 |
| Minimum required DOCA version | Query with `pkg-config --modversion doca-telemetry-exporter`; never hardcode in build files | Cross-version build/runtime mixing breaks per [`CAPABILITIES.md ## Version compatibility`](CAPABILITIES.md#version-compatibility) |

For non-C consumers (Rust, Go, Python), the link surface is the
same `*.so` files; the FFI wrapper layer is the language-specific
binding and is out of scope for this skill — but the slots above
are still the load-bearing inputs the wrapper needs.

## modify

Goal: take a shipped DOCA Telemetry Exporter sample as the verified
starting point and apply a **minimum-diff modification** to express
the user's intent.

The universal modify-a-shipped-sample workflow lives in
[`doca-programming-guide TASKS.md ## modify`](../../doca-programming-guide/TASKS.md#modify).
Use it as-is. The exporter-specific overlay starts with picking the
RIGHT sample — four ship under
`/opt/mellanox/doca/samples/doca_telemetry_exporter/`, one per
publish surface:

| Sample directory | Publish surface it demonstrates | Start here when the user wants … |
| --- | --- | --- |
| `telemetry_export/` (`telemetry_export_sample.c`) | Typed structured events with a custom schema (`schema_init` → `type`/`field` → `schema_add_type` → `schema_start` → `source_*` → `source_report`) | To emit their own structured event records |
| `telemetry_export_metrics/` (`telemetry_export_metrics_sample.c`) | The Metrics API — labeled counters, gauges, histograms; file / IPC / Prometheus exporters | Prometheus-style counters/gauges from their app |
| `telemetry_export_otlp_logs/` (`telemetry_export_otlp_logs_sample.c`) | OTLP logs — resources, scopes, attributes, severity; direct-HTTP or IPC-to-DTS modes | To ship structured logs as OpenTelemetry records |
| `telemetry_export_netflow/` (`telemetry_export_netflow_sample.c`) | NetFlow/IPFIX records via the `_netflow_*` sibling API | To export NetFlow/IPFIX flow records to a collector |

Confirm the exact set on the user's install with
`ls /opt/mellanox/doca/samples/doca_telemetry_exporter/` before
naming one (the install profile can vary). Then fill the
*modify-from-sample schema* — the slots the agent must elicit
before recommending any code-level edit:

| Slot | What the agent asks the user | Exporter-specific consideration |
| --- | --- | --- |
| 1. Starting sample | Which of the four samples above is closest to the user's intended publish surface? | Pick by surface first (events / metrics / otlp / netflow), then by shape. A smaller diff is always safer than a re-architecture |
| 2. Schema / record shape | For events: which fields with which `DOCA_TELEMETRY_EXPORTER_FIELD_TYPE_*` each? For metrics: which counters/gauges/histograms and label sets? For NetFlow: which `DOCA_TELEMETRY_EXPORTER_NETFLOW_*` field IDs? | The packed event struct MUST match the registered field list, or `_source_report` payloads mis-parse. For metrics, label-value arity MUST match the registered label set or `_metrics_add_*` returns `DOCA_ERROR_INVALID_VALUE` |
| 3. Lifecycle order | Where in the app flow do `schema_start`, `source_start`, and (if used) `_metrics_create_context` / `_otlp_logs_create_context` fall? | Per [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes): types registered before `schema_start`; source created after `schema_start`; report only after `source_start`; metrics/OTLP context created only after `source_start`. Out-of-order source/report calls return `DOCA_ERROR_BAD_STATE`; metrics misuse more often surfaces as `DOCA_ERROR_INVALID_VALUE` per the header |
| 4. Destinations | File write, IPC to DTS, Prometheus (metrics only), NetFlow collector — which, and is each reachable? | Enable on the schema (`_set_file_write_enabled` / `_set_ipc_enabled`) BEFORE `schema_start`; `_set_opaque_events_enabled` if opaque/OTLP. Prometheus is the `PROMETHEUS_ENDPOINT` env var, not a `schema_set_*` call |
| 5. Flush + teardown | Does the modified app flush before destroy? | Call `_source_flush` / `_metrics_flush` / `_netflow_flush` before destroy so trailing buffered events are not lost, per [`CAPABILITIES.md ## Safety policy`](CAPABILITIES.md#safety-policy) |
| 6. Build manifest | Keep the sample's existing `meson.build` (which already wires `pkg-config doca-telemetry-exporter` and `-DDOCA_ALLOW_EXPERIMENTAL_API`)? | Yes. Do not switch to a hand-rolled Makefile for *"simplicity"* — it removes the version-check rail and the experimental-API flag |

The agent emits an *intent description + the filled slots*; it must
NOT scaffold a `main.c` from API memory (AGENTS.md ground rule 5).
Walk the user through the diff line-by-line against the sample
source they read on disk, and have them paste back the result for
validation.

## run

Goal: actually execute the built application against the user's
installed DOCA, with any receiver up first, the schema/type
registered, the source started, and the destination writable.

Steps the agent should walk the user through:

1. **Start the receiver first (if using IPC / OTLP / NetFlow).**
   The DTS (IPC), OpenTelemetry Collector (OTLP), or NetFlow
   collector must be up and reachable BEFORE the application
   reports — per the staging matrix in
   [`CAPABILITIES.md ## Safety policy`](CAPABILITIES.md#safety-policy).
   For file-write-only runs there is no receiver; the data files
   are the sink.
2. **Run as the application's normal user (no sudo as a rule).**
   The exporter runs in the application's process as its user. An
   init failure is almost always an unwritable data root or a
   receiver that is not up — fix that, not by adding `sudo`.
3. **Enable file write for the first run and inspect the data
   files.** Even when the real target is IPC/OTLP, enabling
   `_schema_set_file_write_enabled` and inspecting the emitted
   binary data files under the data root is the cheapest proof the
   schema → source → report path is correct, independent of any
   receiver.
4. **Check the IPC status when IPC is enabled.** Call
   `doca_telemetry_exporter_check_ipc_status` (or read the sample's
   log line) — `CONNECTED` means the DTS is reachable; `FAILED`
   means it is not up or the socket dir is wrong; `DISABLED` means
   IPC was never enabled on the schema.
5. **Capture the structured log.** Set `DOCA_LOG_LEVEL=trace` for
   the first run (see
   [`doca-debug CAPABILITIES.md ## Observability`](../../doca-debug/CAPABILITIES.md#observability)).
   This makes the schema/source lifecycle transitions and each
   report visible on first failure.
6. **Flush before exit.** Ensure the app calls `_source_flush`
   (events) / `_metrics_flush` (metrics) / `_netflow_flush`
   (NetFlow) before teardown so trailing events are not lost.

## test

Goal: prove the configured exporter can actually deliver telemetry
to the intended destination, end-to-end, before claiming the
*"build a first telemetry-emitting app"* journey is done.

This is **a loop, not a one-shot pass.** Each iteration narrows
either the destination staging, the schema/record shape, the
source layout, or the flush behavior. The loop terminates when
either (a) the user's intended emit runs end-to-end with the
expected events reaching the destination, or (b) the agent has
narrowed the failure cause to a layer outside the exporter itself
(receiver / transport) and escalated to the matching skill.

Iteration shape:

1. **File-write smoke first.** Enable `_schema_set_file_write_enabled`,
   run, and confirm the emitted binary data files under the data
   root contain the expected schema + rows. This proves the
   publish half of the contract WITHOUT depending on any receiver.
   If `schema_start` returns `DOCA_ERROR_INITIALIZATION`, the data
   root is likely unwritable — fix per
   [`CAPABILITIES.md ## Safety policy`](CAPABILITIES.md#safety-policy).
2. **Lifecycle-order pass.** Confirm types are registered before
   `schema_start`, the source is created after `schema_start`, and
   every report happens after `source_start`. A `DOCA_ERROR_BAD_STATE`
   from `_source_report`, `_source_opaque_report`, or an `_otlp_logs_*`
   call is the canonical out-of-order symptom; `_metrics_*` misuse more
   often surfaces as `DOCA_ERROR_INVALID_VALUE` per the header, per
   [`CAPABILITIES.md ## Error taxonomy`](CAPABILITIES.md#error-taxonomy).
3. **Single-event smoke.** Report ONE event (or one metric point /
   one NetFlow record) and confirm it lands — in the data file, or
   at the receiver. If publish returns success but the receiver is
   empty, the receiver staging is the prime suspect, not the report
   call.
4. **IPC / receiver reception.** With IPC enabled, confirm
   `check_ipc_status` returns `CONNECTED` and the DTS logs the
   events. With OTLP/NetFlow, confirm the collector received them.
5. **Multi-event smoke.** Loop a small N (say, 100) reports with
   the receiver reading concurrently; confirm the receiver / data
   file count matches the publisher count after a `_source_flush`.
   For NetFlow, confirm `_netflow_send`'s `records_sent`
   out-parameter equals the batch size (a short send is
   backpressure at the collector).

Eval-loop overlay — why this is a loop, not a one-shot pass:

| Iteration trigger | What it looks like | What changes next iteration |
| --- | --- | --- |
| `DOCA_ERROR_INITIALIZATION` on `schema_start` | Data root unwritable or a CollectX-backend/exporter-config failure | Point the data root at a writable dir via `_schema_set_buf_data_root`; re-check enabled exporters before retrying |
| `DOCA_ERROR_BAD_STATE` on a report / OTLP context-create | Source not started, OTLP context created before `source_start`, or OTLP context missing on write/flush | Fix the lifecycle order per [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes) before re-running |
| `DOCA_ERROR_INVALID_VALUE` on `_schema_add_type` / a report / a metric | Duplicate type name, bad field type, packed struct ≠ registered fields, NULL metrics handle, label-value arity ≠ label set, or invalid histogram ID | Re-align the type/field definition (or label set) with the report payload |
| Report returns success but receiver / data file is empty | Receiver not up, IPC `DISABLED`/`FAILED`, or nothing was flushed | Confirm `check_ipc_status` = `CONNECTED`, start the receiver first, and add a `_source_flush` before teardown |
| NetFlow `records_sent < count` | Collector slower than the sender, or transport dropping | Re-narrow to the collector / transport side; this is not an exporter bug |

Loop termination: stop iterating once two consecutive iterations of
the same kind don't change anything — that means the cause is below
the exporter (transport, receiver). Escalate to
[`doca-debug TASKS.md ## debug`](../../doca-debug/TASKS.md#debug) with
the captured evidence and the receiver-side log state.

## debug

Goal: when a DOCA Telemetry Exporter call returns a `DOCA_ERROR_*`
(or events do not show up at the destination), narrow the cause to
a specific layer and act on it.

The cross-library debug ladder lives in
[`doca-debug TASKS.md ## debug`](../../doca-debug/TASKS.md#debug).
Walk it in order — install → version → build → link → runtime →
program → driver — *before* recommending exporter-specific fixes.
This skill's overlay names the exporter-specific manifestation at
layers 5 (runtime) and 6 (program):

**Layer 5 (runtime) — exporter overlay.**

- Walk the role rule: did the user actually want the exporter
  (publisher) and not the receiving DOCA Telemetry Service? If they
  are reading the DTS guide, route to it via
  [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md),
  not deeper into the exporter.
- Walk destination staging: for IPC, is the DTS up and does
  `check_ipc_status` return `CONNECTED`? A passing build + a missing
  receiver + an empty dashboard is the canonical *"published into a
  transport with no reader"* symptom — fix at the receiver side.
- Walk file-write permissions: a `DOCA_ERROR_INITIALIZATION` on
  `schema_start` with file write enabled usually means the data root
  is unwritable — fix the path/permissions, do not add `sudo`.

**Layer 6 (program) — exporter overlay.**

- Lifecycle order (the #1 program bug): `schema_add_type` before
  `schema_start`; `source_create` after `schema_start`; report only
  after `source_start`; `_metrics_create_context` /
  `_otlp_logs_create_context` only after `source_start`. Out-of-order
  source/report calls return `DOCA_ERROR_BAD_STATE`. OTLP write/flush
  with no context returns `DOCA_ERROR_BAD_STATE`; metrics functions
  document `INVALID_VALUE` / `NO_MEMORY` instead.
- Opaque path not enabled: `_source_opaque_report` or any
  `_otlp_logs_*` returning `DOCA_ERROR_BAD_STATE` for a missing
  context often means `_schema_set_opaque_events_enabled` was not
  called before `schema_start`.
- Value / schema mismatch: a `DOCA_ERROR_INVALID_VALUE` on a report
  is a packed-struct-vs-registered-fields mismatch, a bad
  `DOCA_TELEMETRY_EXPORTER_FIELD_TYPE_*`, a duplicate type name, or a
  label-value arity mismatch against the registered label set. Fix
  the definition, not the call site blindly.
- There is NO `DOCA_ERROR_AGAIN` and NO `DOCA_ERROR_NOT_FOUND` on
  this API. If the user reports seeing one, they are on a different
  library or misreading the log — re-read the actual `doca_error_t`
  returned.

Once the layer is identified, route to the matching debug verb on
the matching skill: install / build / link / driver to
[`doca-setup ## debug`](../../doca-setup/TASKS.md#debug); version
to [`doca-version ## debug`](../../doca-version/TASKS.md#debug);
cross-cutting runtime to
[`doca-debug ## debug`](../../doca-debug/TASKS.md#debug);
program-layer patterns to
[`doca-programming-guide TASKS.md ## debug`](../../doca-programming-guide/TASKS.md#debug).

## Deferred task verbs

The following verbs are out of scope for this skill but are
commonly asked in the same conversations. Route them as follows so
the agent does not invent guidance:

- **install.** Installing DOCA, choosing packages, post-install
  verification, `pkg-config` wiring — defer to
  [`doca-setup`](../../doca-setup/SKILL.md) and to the install-tree
  layout in
  [doca-public-knowledge-map ## Layout of an installed DOCA package](../../doca-public-knowledge-map/SKILL.md#layout-of-an-installed-doca-package).
  This skill assumes DOCA is already installed.
- **receiver / collector / aggregating service.** Setting up the
  DOCA Telemetry Service (DTS), an OpenTelemetry Collector, or a
  NetFlow collector — out of scope. Route to the receiver's own
  public guide via
  [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md),
  and to
  [`doca-collectx-deployment`](../../doca-collectx-deployment/SKILL.md)
  for the CollectX collection path. This skill is publisher-side
  only.
- **deploy.** Deploying telemetry-emitting applications at scale
  across many hosts — out of scope and reserved for a future
  platform skill.
- **firmware burn / reset.** The exporter does not depend on
  firmware-layer state; telemetry publishing changes no hardware or
  fabric state. If the debug ladder lands on a driver-layer issue,
  route via [`doca-setup ## debug`](../../doca-setup/TASKS.md#debug).

## Command appendix

Every command below is **cross-cutting on DOCA Telemetry
Exporter** — it answers a recurring class of question that comes up
in the verbs above. The agent should treat the *class* as
load-bearing; the worked example is a single instance. Run-as user
is the application's normal unprivileged user unless noted.

**Infra-aware preamble (every row below).** Per the bundle's
detect → prefer → fall back → report contract documented in
[`doca-structured-tools-contract ## The agent behavior contract`](../../doca-structured-tools-contract/SKILL.md#the-agent-behavior-contract),
the agent should:

1. Probe for the matching structured helper FIRST (`doca-env --json`
   for version + libraries in one shot).
2. If the probe succeeds, the structured tool's output is the
   authoritative answer; report *"using structured `<tool>`"*.
3. If the probe fails, fall back to the manual command in the row;
   report *"falling back to manual chain"*.
4. The version-handling semantics (four-way match, NGC,
   headers-win) are owned by
   [`doca-version`](../../doca-version/SKILL.md).

| Command (worked example) | Owning step | Class of question it answers | What healthy output looks like |
| --- | --- | --- | --- |
| `pkg-config --modversion doca-telemetry-exporter` | `## configure` step 3; `## build` | What is the build-time DOCA Telemetry Exporter version? | A semver string matching `doca_caps --version`. Disagreement = partial install (route to [`doca-version TASKS.md ## debug`](../../doca-version/TASKS.md#debug) layer 2) |
| `pkg-config --cflags --libs doca-telemetry-exporter` | `## build` | What include + link flags does the linker need? | Trust whatever `pkg-config --cflags --libs` produces on this install. Do not hardcode the `-I` include path or the `-l<name>` form — both drift between DOCA profiles/majors; `.so` basenames use underscores, `.pc` names use hyphens, and `pkg-config` is the only correct translator |
| `ls /opt/mellanox/doca/samples/doca_telemetry_exporter/` | `## modify` slot 1 | Which exporter samples ship in this install, and which is the closest starting point? | The four sample dirs `telemetry_export`, `telemetry_export_metrics`, `telemetry_export_otlp_logs`, `telemetry_export_netflow` (subset possible per install profile) |
| `doca_caps --version` | `## configure` step 3; `## test` | What is the *runtime* DOCA version? | A semver string matching `pkg-config --modversion doca-telemetry-exporter`. Note: `doca_caps` has NO per-device data for this library — its telemetry-exporter dump is a stub |
| `id` | `## configure` step 4; `## run` step 2 | Can the application user write to the file-write data root / reach the IPC socket dir? | The user can write the data root and reach the socket dir. Failure = `DOCA_ERROR_INITIALIZATION` on `schema_start` (file) or IPC `FAILED` — fix the path/permission, not with sudo |
| `cat /opt/mellanox/doca/applications/VERSION` | `## configure` step 3; `## debug` layer 1 | What does the install tree itself claim its version is? | A semver string matching the other two version sources |
| `DOCA_LOG_LEVEL=trace ./<binary>` | `## run` step 5 | What did the structured DOCA logger emit for the first failing call? | A trace-level line on every schema/source lifecycle transition and every report. A `BAD_STATE` trace on the first report = reporting before `source_start` |

For commands shared across libraries (`pkg-config --modversion`,
`doca_caps`, `cat /opt/mellanox/doca/applications/VERSION`,
`DOCA_LOG_LEVEL`) the cross-library overlay is in
[`doca-debug TASKS.md ## Command appendix`](../../doca-debug/TASKS.md#command-appendix);
this table adds the exporter-specific rows on top.
