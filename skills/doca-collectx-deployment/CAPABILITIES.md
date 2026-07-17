# DOCA CollectX telemetry deployment — Capabilities

**Where to start:** The pattern overview below names the recurring
collector-deployment patterns the agent walks for any
CollectX-based telemetry collector. Pick the pattern first, then
drill into the H2 that owns the substance. For the *how* of
executing each pattern, jump to [TASKS.md](TASKS.md). For the
two telemetry **library** API surfaces this skill routes to, see
[`doca-telemetry`](../libs/doca-telemetry/SKILL.md) (the
hardware-counter reader) and
[`doca-telemetry-exporter`](../libs/doca-telemetry-exporter/SKILL.md)
(the application-side publisher). For the productized DTS
container (out of scope here), see the
[`doca-public-knowledge-map` externally-productized routing row](../doca-public-knowledge-map/SKILL.md#externally-productized-doca-software--not-in-this-bundle-but-here-is-where-to-route).

This file enumerates the collector-deployment contract **as a
class**, as described in the public **DOCA Telemetry guide** and
the public **DOCA Telemetry Service (DTS) guide** (both reachable
through
[`doca-public-knowledge-map ## Public documentation entry points`](../doca-public-knowledge-map/SKILL.md#public-documentation-entry-points)).
Treat this file as a *map of what is documented*, not a substitute
for reading the live guides and the live collector config against
the operator's target. The agent does NOT supply clx symbol
names, provider names, schema field names, exporter flag names, or
config paths from memory — those are install-bound and
docs-bound.

## Pattern overview

Every collector-deployment question this skill teaches resolves
into one of SIX patterns. The patterns are CLASSES — they apply
across both collection targets (host x86, BlueField Arm) and
across whichever export backend the operator picks, not just one
worked example.

| Collector-deployment pattern | Class shape | Where the substance lives |
| --- | --- | --- |
| 1. Decompose the four telemetry surfaces FIRST | Separate the clx collection mechanism (this skill) from the reader library, the publisher library, and the productized DTS container; conflating them is the #1 first-touch error | [`## Capabilities and modes`](#capabilities-and-modes) four-surface table |
| 2. Map the collection pipeline | providers / counters → schema → collector daemon → exporters; each stage is a separate thing to configure and verify | [`## Capabilities and modes`](#capabilities-and-modes) pipeline table |
| 3. Gate provider / counter support BEFORE the config commits | A counter the device does not expose under the chosen dimensions is the canonical "silently dropped metric"; probe per-device before committing | [`## Safety policy`](#safety-policy) gate-before-commit rule |
| 4. Pick the export backend that fits the consumer | Prometheus pull vs Fluent Bit push vs NetFlow vs file / IPC — the consumer's ingest model drives the choice | [`## Capabilities and modes`](#capabilities-and-modes) export-backend table |
| 5. Map a failure back to its layer | collector won't start → no provider rows → schema mismatch → exporter silent → downstream skew → transport; each layer has its own first-stop check | [`## Error taxonomy`](#error-taxonomy) layered split |
| 6. Smoke end-to-end before declaring deployed | The daemon running is NOT proof; the consumer receiving the expected rows is the proof | [`## Safety policy`](#safety-policy) smoke-before-bulk rule |

Two cross-cutting rules apply to *every* pattern above:

- **Operate the documented surface; do not invent one.** clx
  provider names, schema field names, exporter flag names, and
  config file paths all come from the public DOCA Telemetry /
  DTS guides, the live collector config on the target, or
  `--help` on the installed tool. Inferring them from memory is
  the load-bearing hallucination failure for this skill.
- **The four surfaces branch early.** clx-collector vs reader
  library vs publisher library vs productized DTS is a fork the
  operator picks once, surfaced in
  [`TASKS.md ## configure`](TASKS.md#configure) step 1; once the
  operator is on the collector path this skill owns the runtime
  contract, and the library skills / public DTS docs own the
  rest.

## Capabilities and modes

### Four-surface decomposition — the load-bearing first move

The word "telemetry" maps to four distinct DOCA surfaces with
four distinct owners. The agent draws this BEFORE any
config-level guidance.

| Surface | What it is | Owner |
| --- | --- | --- |
| CollectX (clx) collection mechanism | The framework that gathers provider counters into a schema and ships them through exporters — the *collector runtime* the operator deploys | THIS skill (deployment / operation) |
| Hardware-counter reader library | The per-domain API a DOCA program links to *read* hardware counters off a `doca_dev` | [`doca-telemetry`](../libs/doca-telemetry/SKILL.md) |
| Publisher library | The application-side API a DOCA program links to *emit / publish* counters and events to a consumer | [`doca-telemetry-exporter`](../libs/doca-telemetry-exporter/SKILL.md) |
| Productized DTS container | The packaged, NGC-shipped / kubelet-started DOCA Telemetry Service as-deployed (config schema, built-in providers, manifest) | Out of bundle (Non-goal #7) — route to the [public DTS guide](../doca-public-knowledge-map/SKILL.md#externally-productized-doca-software--not-in-this-bundle-but-here-is-where-to-route) |

Conflating any two of these is the #1 first-touch telemetry
error. In particular: a collector that uses the operator's own
DOCA program as a *source* is wiring the publisher library
(`doca-telemetry-exporter`) on the source side and the collector
(this skill) on the sink side — two surfaces, two skills.

### Collection pipeline — the four stages

A CollectX-based collector is a pipeline of four stages. Each is a
separate thing to configure and verify; the agent walks them in
order and does NOT collapse them.

| Stage | What it does (class level) | What the operator configures (verify against live source) |
| --- | --- | --- |
| Providers / counters | The sources the collector samples (built-in providers plus external sources such as a DOCA program using the publisher library) | Which providers are enabled and which counter families they expose — names come from the public docs / live config, not memory |
| Schema | The structured description of the counters the collector emits, including their property dimensions | The schema is derived from the enabled providers; the agent does not invent field names |
| Collector daemon | The process that samples the providers on a cadence and assembles schema rows | The sample cadence and which providers are bound; the run mode (host vs BlueField) per [`## Version compatibility`](#version-compatibility) |
| Exporters | The backends that ship the assembled rows off the box | Which export backend(s) are enabled and where they ship — see the export-backend table below |

### Export-backend surface (class level)

A CollectX collector can ship its assembled rows through one or
more export backends. The agent names these at class level and
routes the operator to the public docs / live config for the
exact flag and config-path spelling.

| Export backend | Ingest model | When it fits |
| --- | --- | --- |
| Prometheus | Pull — the consumer scrapes an endpoint the collector exposes | A Prometheus / Grafana monitoring stack that scrapes targets |
| Fluent Bit | Push — the collector forwards rows to a Fluent Bit pipeline | A log/metrics-forwarding pipeline (e.g. into ELK / a cloud sink) |
| NetFlow | Push — flow-shaped export to a NetFlow collector | A NetFlow / flow-analytics consumer |
| File / IPC | Local — rows written to a local file or an IPC sink | Local capture, debugging, or a local downstream reader |

The agent does NOT assert which backends are enabled by default,
the exact config knob that turns each on, or the config file path
— those are install-bound and documented in the public DTS /
Telemetry guide reached through
[`doca-public-knowledge-map ## DOCA tools`](../doca-public-knowledge-map/SKILL.md#doca-tools)
and the
[externally-productized DTS routing row](../doca-public-knowledge-map/SKILL.md#externally-productized-doca-software--not-in-this-bundle-but-here-is-where-to-route).

## Version compatibility

For the canonical DOCA version-detection chain, the four-way
match rule, NGC container semantics, and the headers-win-over-docs
rule, see [`doca-version`](../doca-version/SKILL.md). The body
lives there; this skill does not duplicate it.

**The collector-deployment-specific overlay** is:

- **The collector, the providers it samples, and any source
  program that feeds it must share a coherent DOCA version.** A
  collector on one DOCA release sampling a provider schema from
  another release is the silent "rows assembled but the consumer
  drops them on a schema-version skew" trap. Anchor the
  collector's DOCA version per
  [`doca-version TASKS.md ## configure`](../doca-version/TASKS.md#configure).
- **The collector's schema version and the downstream consumer's
  expected schema version must agree.** A consumer (Prometheus
  scrape, Fluent Bit pipeline, NetFlow collector) configured for
  one schema version will silently ignore rows from another. This
  is a deployment-time alignment, not a collector bug.
- **The productized DTS container's version is its own anchor.**
  When the operator is actually running DTS-as-deployed, its
  version / image tag is owned by the public DTS guide, not by
  this skill — route per Non-goal #7. Do not infer the DTS image
  tag from the host DOCA version.
- **Provider / counter sets grow across releases.** A counter
  family present on a newer DOCA may be absent on the operator's
  install; the per-device support probe (see
  [`## Safety policy`](#safety-policy)) is the only safe way to
  discover what *this* install + device exposes. Never hardcode
  the available counter set.

## Error taxonomy

Collector-deployment errors fall into SIX layers, each with its
own first-stop check. The agent walks them in order; conflating
them blames the wrong layer. Library-level `DOCA_ERROR_*` codes
raised by a source program are owned by the matching library
skill, not by this one.

| Layer | Symptom | First-stop check / owner |
| --- | --- | --- |
| 1. Collector won't start | The daemon exits immediately, reports a missing config, or never binds | Confirm DOCA install health ([`doca-setup ## test`](../doca-setup/TASKS.md#test)); read the daemon's own startup output; do NOT paste a config path from memory |
| 2. No provider rows | The collector runs but assembles no schema rows for a provider | The provider is not enabled, or the device does not expose it — re-run the per-device support probe ([`## Safety policy`](#safety-policy)); this is the canonical silent-drop |
| 3. Schema mismatch | Rows are assembled but malformed / unexpected fields | The schema does not match the enabled providers; re-derive the schema from the live provider set, do not invent field names |
| 4. Exporter silent | Rows exist locally but the export backend ships nothing | The export backend is not enabled, or its sink is unreachable; confirm the backend is on and the sink endpoint is reachable |
| 5. Downstream skew | The export ships but the consumer shows nothing | Schema-version skew between collector and consumer, or a Data-ID / property-dimension mismatch — align per [`## Version compatibility`](#version-compatibility) |
| 6. Transport | Network / firewall / DTS-runtime failure between collector and consumer | Cross-cutting host / network layer; route to [`doca-debug TASKS.md ## debug`](../doca-debug/TASKS.md#debug) and, for productized DTS, the public DTS guide |

The agent's rule: **never declare the collector deployed on
"the daemon is running" alone.** A running daemon with a silent
downstream is a deployment failure at layer 4, 5, or 6 — walk the
ladder before claiming success.

## Observability

The collector's observability surface is three layers, reached in
order. Healthy means all three agree.

- **Daemon-output layer (FIRST).** The collector daemon's own
  startup / runtime output is the first place to look — did it
  parse its config, enable the providers, and assemble rows? Where
  this output lands (terminal, log file, journald, container log)
  depends on how the operator launched it; the agent reads it
  before anything downstream and does NOT invent a log path.
- **Schema / row layer (SECOND).** Whether the collector is
  actually assembling rows for the enabled providers — the
  local evidence that stages 1-3 of the pipeline succeeded before
  any export. The agent confirms rows exist locally before
  blaming the exporter or the consumer.
- **Downstream-consumer layer (THIRD, load-bearing).** Whether
  the consumer (Prometheus scrape target, Fluent Bit pipeline,
  NetFlow collector, file sink) actually receives the expected
  rows. This is the only end-to-end signal that the deployment
  works; a silent consumer with a happy daemon is a failure.

For the cross-cutting observability primitives
(`--sdk-log-level`, the `DOCA_LOG_LEVEL` env var, the
`doca-<lib>-trace` build flavor) that apply to a source program
feeding the collector, see
[`doca-debug CAPABILITIES.md ## Observability`](../doca-debug/CAPABILITIES.md#observability).
For the install-tree layout (where the collector and its config
live), defer to
[`doca-public-knowledge-map ## Layout of an installed DOCA package`](../doca-public-knowledge-map/SKILL.md#layout-of-an-installed-doca-package).

## Safety policy

> **Overlay on the bundle-wide hardware-safety meta-policy.** The
> rules below are this skill's per-artifact overlay on the
> cross-cutting rules in
> [`doca-hardware-safety` CAPABILITIES.md ## Safety policy](../doca-hardware-safety/CAPABILITIES.md#safety-policy)
> (specifically
> [### Per-artifact overlay pattern](../doca-hardware-safety/CAPABILITIES.md#per-artifact-overlay-pattern)).
> When the two layers disagree, the stricter wins; when either
> layer says STOP, the agent stops.

- **Gate provider / counter support BEFORE the config commits.**
  A counter name committed to the collector config that the
  device does not expose under the chosen property dimensions is
  the canonical "silently dropped metric" — the collector does
  not fail loudly, it just emits nothing for that counter. Probe
  per-device support first (the same gate-before-commit rule
  [`doca-telemetry-utils`](../tools/doca-telemetry-utils/SKILL.md)
  owns for the exporter-config case), and verify the probe result
  against the live device, not from memory.
- **The collector is read-only against the device.** Sampling
  provider counters does not change device state. Any step that
  would change device state (`mlxconfig set`, a firmware burn, a
  BlueField mode flip, enabling a privileged-data daemon that
  requires a hardware-state change) leaves this skill and routes
  to [`doca-hardware-safety`](../doca-hardware-safety/SKILL.md)
  for the change-application discipline.
- **Smoke end-to-end before declaring deployed, and do not invent
  clx names / paths.** Confirm the consumer receives the expected
  rows before claiming success; and never supply a clx provider
  name, schema field, exporter flag, or config path from memory —
  quote it from the live config or the public DOCA Telemetry /
  DTS guide reached through
  [`doca-public-knowledge-map`](../doca-public-knowledge-map/SKILL.md).

## Deferred topic boundaries

This skill scopes itself to deploying / operating a CollectX-based
collector. Adjacent topics the agent will get asked but should
route elsewhere:

- **The hardware-counter reader API** — owned by
  [`doca-telemetry`](../libs/doca-telemetry/SKILL.md).
- **The application-side publisher API** — owned by
  [`doca-telemetry-exporter`](../libs/doca-telemetry-exporter/SKILL.md).
- **The productized DTS container as-deployed** — externally
  productized (Non-goal #7); route to the
  [public DTS guide](../doca-public-knowledge-map/SKILL.md#externally-productized-doca-software--not-in-this-bundle-but-here-is-where-to-route).
- **Installing DOCA / preparing the env** — owned by
  [`doca-setup`](../doca-setup/SKILL.md).
- **Cross-cutting `DOCA_ERROR_*` taxonomy** raised by a source
  program — owned by
  [`doca-programming-guide CAPABILITIES.md ## Error taxonomy`](../doca-programming-guide/CAPABILITIES.md#error-taxonomy).
