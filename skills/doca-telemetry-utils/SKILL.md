---
license: Apache-2.0
name: doca-telemetry-utils
description: >
  Use this skill when the user is invoking `doca_telemetry_utils` on
  a host with DOCA installed — discovering the diagnostic-counter
  schema, translating counter names to binary Data IDs, validating
  per-device counter support before committing a DOCA Telemetry
  exporter config, or reverse-resolving a captured Data ID. Trigger
  even when the user does not explicitly mention "doca_telemetry_utils"
  or "Data ID" — typical implicit phrasings include "my exporter ships
  but the collector sees nothing", "this metric silently drops
  downstream", "which counters does this BlueField expose", "translate
  this 0x... back to a counter name", "what do node / pcie_index /
  depth mean here", or "is this counter supported on this device before
  I commit it". Refuse and route elsewhere for developer-side collector
  / exporter library programming, DTS deployment, or DOCA install /
  repair — those belong to doca-telemetry, doca-public-knowledge-map,
  and doca-setup.
metadata:
  kind: tool
compatibility: >
  Requires DOCA SDK installed at /opt/mellanox/doca on Linux (Ubuntu
  22.04/24.04 or RHEL/SLES) with the Telemetry optional component and
  a BlueField DPU visible to DOCA on a known PCI address. Invokes
  /opt/mellanox/doca/tools/doca_telemetry_utils; per-device probe
  typically requires elevated privileges.
---

# DOCA Telemetry Utils

**Where to start:** This is a tool skill for invoking
`doca_telemetry_utils` — the documented host-side CLI that
supports a DOCA Telemetry exporter / collector pipeline by
discovering the counter schema, translating counter names ↔
Data IDs, and probing per-device counter support. Open
[`TASKS.md`](TASKS.md) and start at
[`## install`](TASKS.md#install) for the host-side
prerequisites and [`## run`](TASKS.md#run) for the
three documented invocation classes
(enumerate / name→ID / ID→name). Open
[`CAPABILITIES.md`](CAPABILITIES.md) when the question is
*what does this tool actually discover about the telemetry
schema*, *how does it pair with the developer-side
[`doca-telemetry`](../../libs/doca-telemetry/SKILL.md) and
exporter libraries*, *how do I confirm a device supports a
counter before committing an exporter config to it*, or
*why does my exporter pipeline silently drop a metric*.

This skill is the **operator-side support tool** for a
DOCA Telemetry deployment. It is NOT the developer-side
collector library (that is
[`doca-telemetry`](../../libs/doca-telemetry/SKILL.md)),
NOT the developer-side publisher library (that is
`doca-telemetry-exporter` — see
[`doca-telemetry ## Related skills`](../../libs/doca-telemetry/SKILL.md#related-skills)),
and NOT a DOCA Telemetry Service (DTS) deployment guide
(route via
[`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md)).
Three separate surfaces; conflating them is the most
common telemetry first-touch error.

## Example questions this skill answers well

The CLASSES of `doca_telemetry_utils` questions this skill
is built to answer, each with one worked example. The
class is the load-bearing piece; the worked example is one
instance.

- **"My exporter says it's emitting `port_rx_bytes` but
  nothing shows up downstream — what did I get wrong?"** —
  worked example: *"my exporter config has a counter name
  string and my collector sees no events with that
  name"*. Answered by the name ↔ Data ID translation
  step in
  [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes)
  + the per-device-support probe in
  [`TASKS.md ## test`](TASKS.md#test): the exporter
  ships a Data ID, not a name; a name in the config that
  resolves to a Data ID the device does not support is
  silently dropped.
- **"Which DOCA diagnostic counters does this BlueField
  actually expose?"** — worked example: *"enumerate the
  full counter schema for my BlueField-3 before I write
  the exporter config"*. Answered by the schema-discovery
  invocation class in
  [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes)
  (`doca_telemetry_utils get-counters` lists every
  counter name the diagnostic-data surface knows about;
  pair with a per-device probe to confirm support).
- **"I have a Data ID in a captured log — what counter
  was that?"** — worked example: *"a downstream consumer
  emitted `Data ID: 0x1160000600030201` — translate it
  back so I can correlate against the public guide"*.
  Answered by the reverse-resolve invocation class in
  [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes)
  + the
  ID-encodes-properties rule in
  [`TASKS.md ## use`](TASKS.md#use) (a Data ID carries
  the counter's property dimensions; the reverse-resolve
  reports them).
- **"What property dimensions does this counter take and
  what values are valid?"** — worked example: *"I know
  the counter is `pcie_link_write_stalled_time_*` — what
  do `node` / `pcie_index` / `depth` mean and what
  values does the device accept?"*. Answered by the
  property-dimension table in
  [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes)
  + the `<name>` invocation without arguments which
  prints the documented property options + units +
  unit-specific axes.
- **"Is this counter supported on this device before I
  commit it to the exporter config?"** — worked example:
  *"validate that `port_rx_bytes` with `node=1` is
  exposed on the BlueField at PCIe address X before I
  write the exporter config"*. Answered by the
  per-device-support probe (`<device PCI> <name>
  [properties]`) in
  [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes)
  + the gate-before-commit rule in
  [`TASKS.md ## use`](TASKS.md#use).
- **"Is `doca_telemetry_utils` on my install, and is it
  paired with the matching `doca-telemetry` library
  version?"** — worked example: *"is the diagnostic-
  data counter set my exporter targets on this DOCA
  version?"*. Answered by the version-overlay in
  [`CAPABILITIES.md ## Version compatibility`](CAPABILITIES.md#version-compatibility),
  which redirects to the canonical
  [`doca-version`](../../doca-version/SKILL.md) chain
  and adds the *tool ↔ `doca-telemetry` library
  schema-version* match rule.

## Audience

This skill serves **external operators, developers, and
AI agents standing up or debugging a DOCA Telemetry
exporter / collector pipeline who need to confirm the
counter schema, validate per-device support, or
translate between human-readable counter names and the
binary Data IDs the exporter actually ships**.
Concretely:

- A platform operator standing up a new DOCA Telemetry
  exporter on a BlueField fleet who needs to confirm
  which counters the target devices actually expose
  before committing the exporter config.
- A developer of a downstream consumer (a collector
  app linking [`doca-telemetry`](../../libs/doca-telemetry/SKILL.md),
  or a third-party aggregator consuming via DTS) who
  has a captured Data ID stream and needs to translate
  IDs back to counter names + properties.
- An operator debugging a *"nothing is shipping
  downstream"* / *"this metric is silently missing"*
  report against a deployed exporter — the
  schema-discovery + per-device-support probes are the
  canonical *"is the counter even supposed to work on
  this device"* first step before suspecting the
  collector or the network path.
- An AI agent producing a *"validated exporter config
  for this BlueField + this DOCA version"* answer
  honestly — with each counter resolved to its Data ID
  and each Data ID confirmed against the per-device
  capability probe.

It is **not** for users debugging the `doca_telemetry_utils`
binary itself, **not** a substitute for the live public
DOCA Telemetry guides, and **not** the right place for
learning how to *write* an exporter application (that
audience belongs in
[`doca-telemetry-exporter` skill](../../libs/doca-telemetry-exporter/SKILL.md)
when present, or via
[`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md)).

The tool is shipped as a **CLI binary** under
`/opt/mellanox/doca/tools/`, not a library you link
against. The skill uses the same `kind: tool`
three-file shape as the rest of the bundle so the
agent's task-verb contract is uniform across libraries,
services, and tools.

## Language scope

`doca_telemetry_utils` is a C host-side CLI that uses
the documented `doca_telemetry_diag` surface to list
counter types and probe per-device support. Its inputs
are command-line arguments (counter name, property
values, Data ID, optional device PCI address); its
outputs are human-readable Data ID + properties
mappings. The skill keeps workflow guidance language-
neutral; downstream consumers in any language can use
the resolved Data IDs against their own collector code.

## When to load this skill

Load this skill when the user is — or the agent needs
to — invoke `doca_telemetry_utils` on a real host with
DOCA installed, alongside an exporter / collector
pipeline that needs schema discovery, per-device
support validation, or Data ID translation.
Concretely:

- Enumerating the full counter schema before writing
  a fresh DOCA Telemetry exporter config (`get-counters`).
- Validating that a chosen counter + property set
  resolves to a Data ID the target device actually
  supports, before committing the config.
- Reverse-resolving a Data ID captured from a
  downstream consumer back to a counter name +
  properties for correlation against the public guide.
- Debugging a *"exporter ships, collector receives
  nothing"* report (the canonical schema-mismatch /
  unsupported-counter failure mode).
- Producing a *"validated counter set for this
  BlueField + this DOCA version"* artifact as part of
  a structured exporter-config baseline.
- Migrating an exporter pipeline across DOCA versions
  and confirming each previously-supported counter
  still resolves cleanly on the new version.

Do **not** load this skill for general DOCA orientation,
collector / exporter library programming, DTS
deployment, or DOCA install. For those, route to
[`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md),
[`doca-telemetry`](../../libs/doca-telemetry/SKILL.md),
or [`doca-setup`](../../doca-setup/SKILL.md).

## What this skill provides

This is a **thin loader**. Substantive material lives
in two companion files:

- `CAPABILITIES.md` — what `doca_telemetry_utils`
  discovers + how it pairs with the developer-side
  surfaces: the three documented invocation classes
  (enumerate / name→ID / ID→name), the property-
  dimension model (counters carry property axes such
  as `node`, `pcie_index`, `depth`, plus per-unit
  axes), the optional per-device capability probe
  (`<device PCI> <name>` runs the resolved counter
  against the device), the operator-side support
  role (this is NOT the developer-side library; it
  exists to make exporter / collector setups
  honest), the version overlay (tool ↔
  `doca-telemetry` library schema version pairing),
  the layered error taxonomy (install / parse /
  unknown-counter / unknown-data-id / property-out-
  of-range / device-not-supported / version /
  cross-cutting), the observability surface
  (stdout-only), and the safety policy (the tool is
  read-only; mistakes appear downstream as silent
  metric drops, not crashes).
- `TASKS.md` — step-by-step workflows for the
  in-scope task verbs: `install` (host-side DOCA +
  telemetry component prerequisites), `configure`
  (axis decisions: which invocation class, which
  device, which counter), `build` (route to install
  — the binary is shipped), `modify` (refuse — do
  not patch the binary; modify the invocation and
  the exporter / collector config that consumes the
  resolved Data IDs), `run` (the three documented
  invocations), `test` (round-trip a chosen
  counter through name → Data ID → per-device
  probe → exporter config → collector receipt),
  `debug` (walk the error taxonomy), `use` (the
  hand-off into the developer-side `doca-telemetry`
  collector / exporter pipeline), plus a `Deferred
  task verbs` block.

The skill assumes a host where DOCA is already
installed with the telemetry component, a BlueField
the operator is targeting is visible to DOCA, and the
exporter / collector pipeline that consumes the
resolved Data IDs is either being authored or has
already been authored against the matching
[`doca-telemetry`](../../libs/doca-telemetry/SKILL.md)
library version.

## What this skill deliberately does not ship

This skill is **agent guidance**, not a samples or
scripts bundle. To keep the boundary clean, it
deliberately does not contain — and pull requests
should not add:

- **A verbatim counter inventory or Data ID
  table.** The counter set evolves across DOCA
  versions and BlueField generations; an
  inventory pinned in this skill would silently rot.
  `doca_telemetry_utils get-counters` on the
  installed binary is the authoritative source.
- **Pre-baked exporter / collector configs.** Configs
  are install-, device-, and use-case-specific; a
  packaged config in this skill would mislead
  operators on a different setup.
- **A DTS deployment recipe.** DTS is a separate DOCA
  service with its own public guide; route via
  [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md).
  This skill *resolves the counter names + IDs* a
  DTS pipeline consumes; it does not configure DTS.
- **Wrappers, parsers, or scripts** in any language
  that consume the tool's output. The output is a
  simple text mapping; users who want to script
  against it should read the live guide and write
  the parser against their installed version.
- **A `samples/` or `reference/` subtree.** This is
  a thin loader for a shipped CLI; substantive
  material lives on the public page, in `--help`,
  and in [`doca-telemetry`](../../libs/doca-telemetry/SKILL.md).

## Loading order

1. Read this `SKILL.md` first to confirm the user's
   question is in scope (operator-side support
   tooling for a telemetry pipeline, not the
   developer-side library programming).
2. **For the three invocation classes, the
   property-dimension model, the per-device support
   probe, the version overlay, the error taxonomy,
   the observability surface, and the safety
   policy, see [CAPABILITIES.md](CAPABILITIES.md).**
3. **For the documented invocations and the
   discover → resolve → validate → consume
   workflow — `install`, `configure`, `build`,
   `modify`, `run`, `test`, `debug`, `use` — see
   [TASKS.md](TASKS.md).**

## Related skills

- [`doca-telemetry`](../../libs/doca-telemetry/SKILL.md)
  — the developer-side collector library whose
  schema this tool helps the operator discover. Pair
  them in every exporter-pipeline triage session.
  The collector library skill teaches the
  collector-vs-exporter rule, the schema-must-match
  contract with the publisher, and the consumer-
  queue-full back-pressure rule; this tool's role
  is to make the *schema* half of that contract
  inspectable from the operator side. Conflating
  the library and the tool is the most common
  telemetry first-touch error.
- [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md)
  — routing to the public DOCA Telemetry guide and
  the DOCA Telemetry Service (DTS) page on
  `docs.nvidia.com`, plus the on-disk install
  layout for the tool.
- [`doca-version`](../../doca-version/SKILL.md) —
  canonical DOCA version-handling rules. The
  `## Version compatibility` section in
  [`CAPABILITIES.md`](CAPABILITIES.md) is a concise
  overlay that redirects here for the body and
  adds the *tool ↔ `doca-telemetry` library
  schema-version pairing* rule.
- [`doca-setup`](../../doca-setup/SKILL.md) — env
  preparation, install verification, and the *I
  have no install yet* path with the public NGC
  DOCA container. This skill assumes its
  preconditions are satisfied (DOCA installed
  with the telemetry component, BlueField visible
  to DOCA).
- [`doca-debug`](../../doca-debug/SKILL.md) — the
  cross-cutting debug ladder. Telemetry-utils
  feeds the cross-cutting ladder by surfacing the
  counter schema + per-device support truth at the
  runtime layer; exporter-pipeline regressions
  often resolve at the schema layer before
  touching the collector / network path.
- [`doca-structured-tools-contract`](../../doca-structured-tools-contract/SKILL.md)
  — the bundle's detect → prefer → fall back →
  report contract for structured helper tools. The
  command appendix in [`TASKS.md`](TASKS.md)
  honors this contract.
- [`doca-programming-guide`](../../doca-programming-guide/SKILL.md)
  — general DOCA programming patterns shared by
  every library / tool surface, including the
  cross-library `DOCA_ERROR_*` taxonomy this
  tool's host-side error layer overlays on top of
  when downstream collector / exporter code
  surfaces a related error.
