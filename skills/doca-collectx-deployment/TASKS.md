# DOCA CollectX telemetry deployment — Tasks

**Where to start:** The verb order is `configure → build → modify →
run → test → debug`. For collector deployment, `build` is a
*routing stub* — there is no collector binary for the operator to
build inside this skill (the collector ships with DOCA; a *source
program* that feeds the collector is built per
[`doca-programming-guide`](../doca-programming-guide/SKILL.md) +
[`doca-telemetry-exporter`](../libs/doca-telemetry-exporter/SKILL.md)).
The `## test` verb is an iterative end-to-end smoke loop, not a
one-shot pass.

These verbs cover the in-scope cross-cutting collector-deployment
workflows for an operator standing up or running a CollectX-based
collector on either target (host x86 or BlueField Arm). Every step
assumes the operator has consulted the live public **DOCA
Telemetry guide** and **DOCA Telemetry Service (DTS) guide** on
`docs.nvidia.com` (reachable through
[`doca-public-knowledge-map ## Public documentation entry points`](../doca-public-knowledge-map/SKILL.md#public-documentation-entry-points))
and is using them as the authoritative reference; this file
prescribes the *order* and *what to look up where*, not a
copy-paste runbook. The agent does NOT invent clx provider names,
schema fields, exporter flags, or config paths.

## configure

Preparing the target, surfacing the scope boundary, and deciding
what the collector will sample and where it will ship — before any
config commits. This is also where the gate-before-commit and
smoke-before-bulk postures are established.

1. **Decompose the four surfaces FIRST.** Before any config-level
   guidance, surface the four-surface decomposition per
   [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes):
   the clx collection mechanism (this skill), the reader library
   [`doca-telemetry`](../libs/doca-telemetry/SKILL.md), the
   publisher library
   [`doca-telemetry-exporter`](../libs/doca-telemetry-exporter/SKILL.md),
   and the productized DTS container (Non-goal #7 → public docs).
   If the operator actually wants the productized DTS service,
   STOP and route per the
   [externally-productized DTS row](../doca-public-knowledge-map/SKILL.md#externally-productized-doca-software--not-in-this-bundle-but-here-is-where-to-route).
2. **Confirm the env is healthy first.** This skill expects DOCA
   installed and healthy on the side being collected from (host
   x86 OR BlueField Arm). If install health is unverified, run
   [`doca-setup ## test`](../doca-setup/TASKS.md#test) on the
   target first. If the operator has no install at all, route to
   [`doca-setup ## no-install`](../doca-setup/TASKS.md#no-install).
3. **Decide what the collector will sample.** Map the operator's
   goal to a provider / counter set per the pipeline table in
   [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes).
   Provider and counter-family names come from the public DOCA
   Telemetry / DTS guide and the live collector config — the agent
   does NOT supply them from memory. If a source is the operator's
   own DOCA program, that source side is the publisher library
   ([`doca-telemetry-exporter`](../libs/doca-telemetry-exporter/SKILL.md)),
   not this skill.
4. **Gate provider / counter support BEFORE committing the
   config.** Per the gate-before-commit rule in
   [`CAPABILITIES.md ## Safety policy`](CAPABILITIES.md#safety-policy),
   probe that THIS device actually exposes each counter under the
   chosen property dimensions (the same discipline
   [`doca-telemetry-utils`](../tools/doca-telemetry-utils/SKILL.md)
   owns for the exporter-config case). A counter the device does
   not expose is the canonical silently-dropped metric — gate it
   before the config commits, not after.
5. **Pick the export backend(s).** Per the export-backend table in
   [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes),
   choose the backend(s) whose ingest model fits the downstream
   consumer: Prometheus (pull), Fluent Bit (push), NetFlow (push),
   or file / IPC (local). The exact enable flag and config-path
   spelling come from the public DTS / Telemetry guide and the
   live config, not from memory.
6. **Confirm the version overlay.** Per
   [`CAPABILITIES.md ## Version compatibility`](CAPABILITIES.md#version-compatibility),
   record the collector's DOCA version (via
   [`doca-version TASKS.md ## configure`](../doca-version/TASKS.md#configure)),
   any source program's link-time version, and the downstream
   consumer's expected schema version. A skew among these is the
   silent "rows assembled but the consumer drops them" trap.

## build

Collector deployment is the *deploy* verb for a collector that
ships with DOCA — there is **no collector artifact for the
operator to build inside this skill**. This verb is a routing
stub:

- If the user is asking how to build a **source program** that
  feeds the collector (a DOCA program that publishes counters /
  events), hand off to
  [`doca-telemetry-exporter ## build`](../libs/doca-telemetry-exporter/TASKS.md#build)
  for the publisher library and to
  [`doca-programming-guide ## build`](../doca-programming-guide/TASKS.md#build)
  for the canonical `pkg-config` + meson pattern.
- If the user is asking how to build a program that **reads**
  hardware counters off a `doca_dev`, that is the reader library —
  route to
  [`doca-telemetry ## build`](../libs/doca-telemetry/TASKS.md#build).
- If the user is asking how to build the productized DTS container
  image, that is externally productized (Non-goal #7) — route to
  the [public DTS guide](../doca-public-knowledge-map/SKILL.md#externally-productized-doca-software--not-in-this-bundle-but-here-is-where-to-route).
- The collector's own config is *composed against the live env*,
  not built ahead of time — see [`## run`](#run).

## modify

Collector deployment does not have a *modify a sample program*
workflow; the deployment-side analog of "modify" is **re-walk the
deploy after the config, the provider set, or the env changes**:

1. **A collector-config change is a deploy event.** Editing the
   enabled providers, the schema, the sample cadence, or an
   exporter backend changes the deployment contract; treat each
   edit as a fresh deploy. Re-walk
   [`## configure`](#configure) step 4 (gate support), then
   [`## run`](#run), then [`## test`](#test).
2. **A provider / counter change re-opens the gate.** Adding a
   counter family re-opens the per-device support probe in
   [`## configure`](#configure) step 4 — a family present on one
   device may return unsupported on another.
3. **An export-backend change is a deploy event.** Switching from
   (say) Prometheus pull to Fluent Bit push, or adding a second
   backend, changes the downstream contract; re-walk
   [`## configure`](#configure) step 5 and re-run the end-to-end
   smoke in [`## test`](#test).
4. **A source-program change lives in the library skill.** If the
   operator is modifying a DOCA program that feeds the collector,
   that change is owned by
   [`doca-telemetry-exporter ## modify`](../libs/doca-telemetry-exporter/TASKS.md#modify);
   once the source is rebuilt, control returns here at step 1.
5. **A hardware-state change leaves this verb entirely.** Any
   change touching device state (`mlxconfig set`, firmware burn,
   BlueField mode flip) is owned by
   [`doca-hardware-safety`](../doca-hardware-safety/SKILL.md);
   control returns here at [`## configure`](#configure) step 2
   once the change is complete and verified.

The agent's anti-pattern alert: editing the collector config in
place without re-walking the end-to-end smoke is the canonical
"my pipeline silently went quiet after a small change" failure.

## run

Bringing up the collector, confirming it assembles schema rows
locally, and confirming the export backend ships them — before
declaring the pipeline deployed. Every step assumes
[`## configure`](#configure) is done.

1. **Launch the collector against the committed config.** Start
   the collector daemon with the config composed in
   [`## configure`](#configure); the launch shape (foreground,
   service-supervised, container) follows the operator's
   deployment posture per
   [`doca-bare-metal-deployment`](../doca-bare-metal-deployment/SKILL.md)
   (non-container) or
   [`doca-container-deployment`](../doca-container-deployment/SKILL.md)
   (container). Read the daemon's own startup output first per
   [`CAPABILITIES.md ## Observability`](CAPABILITIES.md#observability).
2. **Confirm providers are enabled and rows are assembling.** Per
   the schema / row layer in
   [`CAPABILITIES.md ## Observability`](CAPABILITIES.md#observability),
   confirm the collector is assembling rows locally for the
   enabled providers BEFORE looking downstream. No rows for a
   provider is a layer-2 symptom in
   [`CAPABILITIES.md ## Error taxonomy`](CAPABILITIES.md#error-taxonomy)
   (provider not enabled, or device does not expose it — back to
   [`## configure`](#configure) step 4).
3. **Confirm the export backend ships.** Confirm the chosen
   backend (Prometheus endpoint, Fluent Bit forward, NetFlow
   export, file / IPC sink) is actually emitting. An exporter that
   is silent while rows exist locally is a layer-4 symptom.
4. **Capture the as-deployed snapshot.** Record the collector's
   DOCA version (per
   [`doca-version TASKS.md ## run`](../doca-version/TASKS.md#run)),
   the device / generation, the enabled providers + resolved
   counter identities (not just names), the export backend(s) +
   sink endpoints, and the downstream consumer's schema version.
   This snapshot is the artifact future debug sessions skip
   rediscovery from.
5. **Smoke before bulk (next: [`## test`](#test) step 1).** Before
   trusting the pipeline, walk [`## test`](#test) step 1 once to
   confirm the consumer actually receives the expected rows.

## test

Collector deployment has no *compile-and-unit-test* workflow —
testing is operational and end-to-end.

**`## test` is an iterative loop, not a one-shot pass.** Every
mutation (config edit, provider change, export-backend change,
source-program rebuild) re-opens the smoke sweep. Skipping the
re-run after a mutation is the failure mode this loop replaces.

The end-to-end smoke (each step proves what the previous does
not):

1. **End-to-end row smoke.** Confirm the full chain: the device /
   source produces the counters, the collector assembles schema
   rows for them, the export backend ships them, AND the
   downstream consumer receives rows with the expected identities.
   Passing = the consumer shows the expected rows; failing =
   "daemon running but consumer silent" (drop to
   [`## debug`](#debug)).
2. **Provider-support re-check.** Re-probe that each committed
   counter is still exposed by THIS device under the chosen
   dimensions; a family that returns unsupported is the silent
   drop (back to [`## configure`](#configure) step 4).
3. **Schema-version sanity.** Confirm the collector's schema
   version and the consumer's expected schema version agree per
   [`CAPABILITIES.md ## Version compatibility`](CAPABILITIES.md#version-compatibility);
   a skew is the silent layer-5 drop.
4. **Cadence + under-load behavior.** Confirm rows continue to
   arrive at the configured cadence and the consumer keeps up
   under load, rather than the export queue backing up.
5. **Snapshot the passing config.** Save the as-deployed snapshot
   from [`## run`](#run) step 4 as the rollback / reproduction
   baseline.

Loop termination: stop iterating once two consecutive iterations
of the same kind change nothing — the cause is below the collector
runtime (device, host, network, or the productized DTS container).
Escalate to
[`doca-debug TASKS.md ## debug`](../doca-debug/TASKS.md#debug)
with the captured evidence, or to the public DTS guide if the
operator is on the productized container.

## debug

Layered diagnosis. Walk the layers in the order in
[`CAPABILITIES.md ## Error taxonomy`](CAPABILITIES.md#error-taxonomy);
do not skip down without clearing the layer above.

1. **Collector-won't-start (layer 1).** Daemon exits / never
   binds. Confirm DOCA install health
   ([`doca-setup ## test`](../doca-setup/TASKS.md#test)); read
   the daemon's own startup output; do NOT paste a config path
   from memory — resolve it from the live install per
   [`doca-public-knowledge-map ## Layout of an installed DOCA package`](../doca-public-knowledge-map/SKILL.md#layout-of-an-installed-doca-package).
2. **No-provider-rows (layer 2).** Daemon runs, no rows for a
   provider. Re-run the per-device support probe per
   [`CAPABILITIES.md ## Safety policy`](CAPABILITIES.md#safety-policy);
   the provider is not enabled, or the device does not expose it.
   This is the canonical silent drop, not a collector bug.
3. **Schema-mismatch (layer 3).** Rows assembled but malformed.
   Re-derive the schema from the live provider set; do not invent
   field names.
4. **Exporter-silent (layer 4).** Rows exist locally, export ships
   nothing. Confirm the backend is enabled and its sink is
   reachable; confirm the endpoint / forward target against the
   live config, not memory.
5. **Downstream-skew (layer 5).** Export ships, consumer shows
   nothing. Align the schema / counter-identity version between
   collector and consumer per
   [`CAPABILITIES.md ## Version compatibility`](CAPABILITIES.md#version-compatibility).
6. **Transport (layer 6).** Network / firewall / DTS-runtime
   failure. Route to
   [`doca-debug TASKS.md ## debug`](../doca-debug/TASKS.md#debug)
   for the cross-cutting host / network ladder; for the
   productized DTS container, route to the public DTS guide
   (Non-goal #7). If the symptom is a contemplated hardware-state
   change, route to
   [`doca-hardware-safety`](../doca-hardware-safety/SKILL.md)
   instead of touching device state from this skill.

Library-level `DOCA_ERROR_*` codes raised by a *source* program
that feeds the collector are owned by the matching library skill
([`doca-telemetry-exporter`](../libs/doca-telemetry-exporter/SKILL.md)
for the publisher) plus the cross-library taxonomy in
[`doca-programming-guide CAPABILITIES.md ## Error taxonomy`](../doca-programming-guide/CAPABILITIES.md#error-taxonomy),
not by this skill.

## Deferred task verbs

The following are out of scope here but commonly asked in the same
conversation. Route them so the agent does not invent guidance:

- **The hardware-counter reader API** (open a per-domain
  `doca_telemetry_<domain>` context, read a counter snapshot off a
  `doca_dev`) — route to
  [`doca-telemetry`](../libs/doca-telemetry/SKILL.md).
- **The application-side publisher API** (define a schema, create
  sources, emit counters / events from a DOCA program) — route to
  [`doca-telemetry-exporter`](../libs/doca-telemetry-exporter/SKILL.md).
- **Operate the productized DTS container** (its packaged config
  schema, built-in provider set, kubelet manifest, NGC image) —
  externally productized (Non-goal #7); route to the
  [public DTS guide](../doca-public-knowledge-map/SKILL.md#externally-productized-doca-software--not-in-this-bundle-but-here-is-where-to-route).
- **Install DOCA / prepare the env** (install, hugepages,
  pkg-config, devlink, representor visibility) — route to
  [`doca-setup`](../doca-setup/SKILL.md).
- **Hardware-state changes** (`mlxconfig set`, firmware burn,
  BlueField mode flip, kernel-boot-parameter changes) — the
  change-application discipline is meta-policy owned by
  [`doca-hardware-safety ## modify`](../doca-hardware-safety/TASKS.md#modify);
  load it ALONGSIDE this skill whenever a mutating step is on the
  table.
