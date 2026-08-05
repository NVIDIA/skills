---
license: Apache-2.0
name: doca-telemetry-exporter
description: >
  Use this skill when the user is doing hands-on DOCA Telemetry
  Exporter programming on a host where DOCA is installed — defining
  a doca_telemetry_exporter_schema and event types, creating
  sources, picking a publish surface (typed events / opaque events
  / the metrics counter-gauge-histogram API / OTLP logs / NetFlow),
  walking the schema-then-source lifecycle, or debugging
  DOCA_ERROR_* failures from the exporter API. Trigger even when
  the user does not explicitly mention "DOCA Telemetry Exporter" or
  "doca_telemetry_exporter_*" — typical implicit phrasings include
  "publishing counters from my DOCA app", "BAD_STATE when I report
  an event", "consumer/DTS sees nothing but my report succeeded",
  "how do I export NetFlow/IPFIX records", or "should I link the
  exporter or the telemetry service". Refuse and route elsewhere
  for the receiving DOCA Telemetry Service (DTS), plain stdout
  logging via doca_log, or real-time event subscription back into
  the app via doca-comch — those belong to other skills.
metadata:
  kind: library
compatibility: >
  Requires DOCA SDK installed at /opt/mellanox/doca on Linux (Ubuntu
  22.04/24.04 or RHEL/SLES) with a BlueField DPU or ConnectX NIC
  attached. Reads the user's local install via `pkg-config
  doca-telemetry-exporter` and inspects
  /opt/mellanox/doca/{lib,include,samples,applications}.
---

# DOCA Telemetry Exporter

**Where to start:** This skill assumes DOCA is already installed and
the user is doing **hands-on telemetry-exporter work** — emitting
structured application telemetry (counters / events) from a
DOCA-using program to an external consumer. Open
[`TASKS.md`](TASKS.md) if the user wants to *do* something (configure
/ build / modify + rebuild / run / test / debug); open
[`CAPABILITIES.md`](CAPABILITIES.md) when the question is *what can
the exporter express* on this install. If the user has not installed
DOCA yet, route to [`doca-setup`](../../doca-setup/SKILL.md) first.
If the user is confused about whether they want this library or the
DOCA Telemetry Service (the receiver) — read the
exporter-vs-service rule in
[`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes)
before configuring anything.

> **This library is NOT a DOCA Core context.** There is no
> `doca_ctx_start()` for the exporter and no per-`doca_devinfo`
> capability-query family (its `doca_caps` dump is a stub). The
> lifecycle is `schema_init` → configure exporters → register
> type(s) → `schema_start` → `source_create` → `source_start` →
> report → flush → destroy.

## Example questions this skill answers well

The CLASSES of telemetry-exporter questions this skill is built to
answer, each with one worked example. The agent should treat the
*class* as the load-bearing piece — the worked example is a single
instance.

- **"Which library do I want — the exporter or the telemetry
  service?"** — worked example: *"I want my DOCA Flow program to
  publish a per-second packets-processed counter to a downstream
  collector — which DOCA artifact do I link?"*. Answered by the
  exporter-vs-service rule in
  [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes)
  role-split table + the path-selection bullet, both of which name
  `doca-telemetry-exporter` as the publisher the application links
  and route the receiving / consuming side away from this skill.
- **"How do I emit my first structured event from a DOCA program?"** —
  worked example: *"emit a `packets_processed` event record from my
  DOCA Flow application"*. Answered by the schema → source
  lifecycle in
  [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes)
  object table + the workflow in
  [`TASKS.md ## configure`](TASKS.md#configure) +
  [`TASKS.md ## run`](TASKS.md#run) step 3 (file-write smoke before
  bulk), starting from the `telemetry_export/` sample.
- **"Which publish surface do I want — typed events, metrics, OTLP
  logs, or NetFlow?"** — worked example: *"I want labeled
  per-interface packet counters and a bandwidth gauge"*. Answered
  by the publish-surface table in
  [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes)
  (that intent maps to the Metrics API — `_metrics_add_counter` /
  `_add_gauge` — and the `telemetry_export_metrics/` sample), plus
  the sample map in [`TASKS.md ## modify`](TASKS.md#modify).
- **"My report call returns `DOCA_ERROR_BAD_STATE` — what did I
  get wrong?"** — worked example:
  *"`doca_telemetry_exporter_source_report` returns `BAD_STATE` on
  the first call"*. Answered by the `BAD_STATE` row in
  [`CAPABILITIES.md ## Error taxonomy`](CAPABILITIES.md#error-taxonomy)
  (the source was never started, or an OTLP context is missing on
  write/flush) + the lifecycle order in
  [`TASKS.md ## configure`](TASKS.md#configure). Note there is NO
  `DOCA_ERROR_AGAIN` and NO `DOCA_ERROR_NOT_FOUND` on this API.
- **"My program reports, but the DTS / collector sees nothing —
  where do I start?"** — worked example: *"my report returns
  success, but the DTS log is empty"*. Answered by the
  receiver-up-first staging in
  [`CAPABILITIES.md ## Safety policy`](CAPABILITIES.md#safety-policy)
  + the file-write smoke and `check_ipc_status` steps in
  [`TASKS.md ## test`](TASKS.md#test) (prove the publish half with
  file write, then confirm IPC is `CONNECTED` and the receiver
  is up).
- **"How do I confirm the exporter is installed and my transport
  is live?"** — worked example: *"is the exporter on my DOCA 3.x
  install, and is IPC to DTS actually connected?"*. Answered by
  the version-compatibility overlay in
  [`CAPABILITIES.md ## Version compatibility`](CAPABILITIES.md#version-compatibility)
  (cross-linking the detection chain in
  [`doca-version`](../../doca-version/SKILL.md)) plus the honest
  introspection rule in
  [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes)
  (`doca_telemetry_exporter_check_ipc_status`, not a device
  cap-query).

## Audience

This skill serves **external developers building applications that
emit structured telemetry through DOCA Telemetry Exporter** — i.e.,
users whose application code calls `doca_telemetry_exporter_*`
(directly in C/C++, or through FFI/bindings from another language)
to publish counters, gauges, and events from their DOCA-using
program to an external telemetry consumer. It is *not* for NVIDIA
developers contributing to DOCA Telemetry Exporter itself, and it
is *not* for users building the receiving / aggregating telemetry
service (the DOCA Telemetry Service is a separate DOCA service with
its own public guide, reached via
[`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md)).

**Language scope.** DOCA Telemetry Exporter ships as a C library
with `pkg-config` module name `doca-telemetry-exporter`. The
shipped samples are written in C. C and C++ consumers are the
canonical case; the worked examples in `TASKS.md` assume that path.
Other-language consumers (Rust, Go, Python, …) consume the same
`*.so` through FFI or language-specific bindings; the skill's
contribution in that case is to keep the exporter-vs-service
distinction, the schema → source lifecycle, the transport-not-caps
discovery rule, the same-user-as-the-app permission rule, the
buffered flush-based delivery model, and the error-taxonomy
guidance language-neutral, and to route the agent to the public C
ABI as the authoritative surface that any wrapper will eventually
call.

## When to load this skill

Load this skill when the user is doing hands-on DOCA Telemetry
Exporter work, in any language. Concretely:

- Defining a `doca_telemetry_exporter_schema` for the events the
  application will emit (field names + field types), and
  registering it with the exporter BEFORE any event is published.
- Creating one or more `doca_telemetry_exporter_source` instances
  to represent distinct logical sources of telemetry inside the
  application (e.g. one source per worker thread / per pipeline
  stage).
- Picking the right publish surface — typed structured events
  (`_source_report`), opaque events (`_source_opaque_report`), the
  Metrics API (counter / gauge / histogram), OTLP logs, or the
  NetFlow sibling API — for what the application reports.
- Confirming the exporter's install + transport reality (there is
  NO `doca_devinfo` cap-query family and NO `doca_caps` data for
  this library): `doca_telemetry_exporter_check_ipc_status` for
  IPC liveness, `_source_get_opaque_report_max_data_size` for the
  opaque payload bound, and the `_schema_get_*` config getters.
- Debugging a `DOCA_ERROR_*` returned from an exporter call
  (`BAD_STATE` lifecycle-order vs. `INVALID_VALUE` type/label
  mismatch vs. `NO_MEMORY` vs. `INITIALIZATION` vs. `UNKNOWN`
  backend) and the per-call status returned to the application.
- Choosing between Telemetry Exporter and an adjacent option
  (`doca_log` when stdout / structured-log shipping is enough; a
  Prometheus client library when the user needs a non-DOCA-aware
  sink; [`doca-comch`](../doca-comch/SKILL.md) when the user
  needs a real-time event subscription back INTO the app — the
  exporter is publish-only / one-way).
- Designing or extending non-C bindings (Rust, Go, Python, …)
  that wrap the exporter C ABI — for the exporter-vs-service
  distinction, the schema → source lifecycle, the permission
  policy, the buffered flush-based delivery model, and the
  transport-introspection + error rules the wrapper must honor.

Do **not** load this skill for general DOCA orientation, install
of DOCA itself, the receiving telemetry service (the DOCA
Telemetry Service has its own public guide reachable through
[`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md)),
or non-exporter library questions. For those, use
[`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md).

## What this skill provides

This is a **thin loader**. The body keeps only the orientation
needed to pick the right next file. The substantive
exporter-specific material lives in two companion files:

- `CAPABILITIES.md` — what the exporter can express on this
  install: the exporter-vs-service role-split rule, the object
  family (`doca_telemetry_exporter_schema` → `_type`/`_field` →
  `_source` with the schema → source lifecycle), the four publish
  surfaces (typed events / opaque events / Metrics API / OTLP
  logs) plus the NetFlow sibling API, the transport-not-caps
  introspection rule (`check_ipc_status`,
  `_get_opaque_report_max_data_size`, `_schema_get_*` — NO
  `doca_caps` data, NO device cap-query), the exporter error
  taxonomy (mapped onto the cross-library `DOCA_ERROR_*` set, with
  the note that there is NO `AGAIN` and NO `NOT_FOUND` on this
  surface), the observability surface (per-call status + IPC
  status + file-write inspection + the receiver side as the
  end-to-end signal), the safety policy that gates the
  same-user-as-the-app permission and the receiver-up-first
  staging, and the path-selection rule against `doca_log` and
  `doca-comch`.
- `TASKS.md` — step-by-step workflows for the six in-scope
  exporter verbs: `configure`, `build`, `modify` (followed by a
  rebuild), `run`, `test`, `debug`. Plus a `Deferred task verbs` block that points
  out-of-scope questions at the right next skill.

The skill assumes a host where DOCA is already installed at the
standard location, the application runs as a user that can write
to the telemetry transport the exporter is configured for, and a
receiving telemetry consumer is reachable and started before the
exporter. It does not cover installing DOCA — that path goes
through [`doca-setup`](../../doca-setup/SKILL.md) — and it does
not cover configuring / operating the receiving telemetry service,
which is a separate DOCA service with its own public guide.

## What this skill deliberately does not ship

This skill is **agent guidance**, not a samples or templates
bundle. To keep the boundary clean, it deliberately does not
contain — and pull requests should not add:

- **Pre-written DOCA Telemetry Exporter application source code,
  in any language.** The verified exporter source code is the
  shipped C samples at
  `/opt/mellanox/doca/samples/doca_telemetry_exporter/`. The
  agent's job is to route the user to those files and prescribe a
  minimum-diff modification on them via the universal
  modify-a-sample workflow in
  [`doca-programming-guide`](../../doca-programming-guide/SKILL.md),
  layered with the exporter-specific overrides in
  [`TASKS.md ## modify`](TASKS.md#modify).
- **A telemetry consumer / collector / receiving service.** The
  DOCA Telemetry Service is a separate DOCA service with its own
  public guide; routing to it goes through
  [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md).
  This skill is about the publisher side only.
- **Standalone build manifests** (`meson.build`, `CMakeLists.txt`,
  `Cargo.toml`, …) parked inside the skill. The agent constructs
  the build manifest *in the user's project directory* against
  the user's installed DOCA, where `pkg-config --modversion
  doca-telemetry-exporter` is the source of truth.
- **A `samples/`, `bindings/`, or `reference/` subtree** of any
  kind. A mock or incomplete artifact in this skill's tree, even
  one labeled "reference", is misleading: users will read it as
  buildable.

## Loading order

1. Read this `SKILL.md` first to confirm the user's question is
   in scope.
2. **For the exporter-vs-service rule, the object family, the
   schema → source lifecycle, the four publish surfaces + NetFlow,
   the transport-not-caps introspection rule, the error taxonomy
   (note: no `AGAIN`, no `NOT_FOUND`), observability, the safety
   policy, and the path-selection rule against `doca_log` / comch,
   see [CAPABILITIES.md](CAPABILITIES.md).**
3. **For step-by-step workflows — configure, build, modify,
   rebuild, run, test, debug — see [TASKS.md](TASKS.md).**

Both companion files cross-link to each other,
[`doca-version`](../../doca-version/SKILL.md) for the canonical
version-handling rules, and
[`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md)
whenever the right answer is "look it up in the public docs or
the installed package layout" rather than "exporter-specific
guidance".

## Related skills

- [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md) —
  the routing table for every public DOCA documentation source
  and the on-disk layout of an installed DOCA package. The
  exporter's public guide URL is
  `https://docs.nvidia.com/doca/sdk/DOCA-Telemetry-Exporter/index.html`;
  the on-disk samples live under
  `/opt/mellanox/doca/samples/doca_telemetry_exporter/`. The
  DOCA Telemetry Service (the *receiver*, out of scope here) is
  a separate guide reachable through that same routing table.
- [`doca-setup`](../../doca-setup/SKILL.md) — env preparation,
  install verification, transport-side reachability checks, and
  the *I have no install yet* path with the public NGC DOCA
  container. This skill assumes its preconditions are satisfied
  (in particular, the application user can write to the
  telemetry transport).
- [`doca-version`](../../doca-version/SKILL.md) — canonical DOCA
  version-handling rules. This skill's `## Version compatibility`
  cross-links the four-way match rule + detection chain and adds
  the exporter-specific overlay rules.
- [`doca-structured-tools-contract`](../../doca-structured-tools-contract/SKILL.md) —
  the bundle's structured-tools precedence rule (detect / prefer
  / fall back / report). The Command appendix in
  [TASKS.md](TASKS.md) honors this contract.
- [`doca-programming-guide`](../../doca-programming-guide/SKILL.md) —
  general DOCA programming patterns shared by every library: the
  canonical `pkg-config` + meson build pattern, the universal
  modify-a-shipped-sample first-app workflow, the universal
  lifecycle, the cross-library `DOCA_ERROR_*` taxonomy, and the
  program-side debug order. This skill layers exporter specifics
  on top.
- [`doca-comch`](../doca-comch/SKILL.md) — the right primitive
  when the user needs a real-time event subscription back INTO
  the application (host ↔ DPU control-plane messaging). The
  exporter is publish-only / one-way; this skill's
  path-selection rule routes there when subscription is the
  actual requirement.
- [`doca-debug`](../../doca-debug/SKILL.md) — the cross-cutting
  debug ladder (install / version / build / link / runtime /
  program / driver). Exporter-specific debug (receiver not up,
  lifecycle-order `BAD_STATE`, type/label `INVALID_VALUE`,
  opaque-path-not-enabled) overlays on top of that ladder.
