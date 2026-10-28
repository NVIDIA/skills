---
license: Apache-2.0
name: doca-telemetry-utils
description: >
  doca_telemetry_utils command line on a host with DOCA: discover the
  diagnostic-counter schema, translate counter names to binary Data IDs
  and back, and check per-device counter support before committing a DOCA
  Telemetry exporter config. Use this skill when telemetry counters vanish
  between the exporter and the collector: "my exporter ships but the
  collector sees nothing", "this metric silently drops downstream", "which
  counters does this BlueField expose", "translate this 0x... back to a
  counter name", "what do node / pcie_index / depth mean here", or "is
  this counter supported on this device", even without naming the tool.
  Route exporter or collector code to doca-telemetry-exporter /
  doca-telemetry, DTS deployment to doca-dts, and DOCA install or repair
  to doca-setup.
metadata:
  author: NVIDIA <doca-feedback@nvidia.com>
  kind: tool
  doca_version: "3.5"
  tags:
    - doca
    - telemetry
    - tools
compatibility: >
  Requires DOCA SDK installed at /opt/mellanox/doca on Linux (Ubuntu
  22.04/24.04 or RHEL/SLES) with the Telemetry optional component and a
  BlueField DPU visible to DOCA on a known PCI address. Invokes
  /opt/mellanox/doca/tools/doca_telemetry_utils; per-device probe
  typically requires elevated privileges. The validation steps require a
  live DOCA install.
---

# DOCA Telemetry Utils

## Examples

- **"Which diagnostic counters does this BlueField actually
  expose?"** — this is in scope. Enumerate then probe; do
  not start a collector app here.
- **"My exporter names `port_rx_bytes` but nothing shows up
  downstream."** — translate name to Data ID and probe
  per-device support before changing the exporter.

## Quick triage: the exporter ships, the collector sees nothing

1. `doca_telemetry_utils get-counters`: is the counter known to this release?
2. `doca_telemetry_utils <counter>`: print its property options.
3. `doca_telemetry_utils <counter> <properties>`: resolve the Data ID.
4. `doca_telemetry_utils <device PCI> <counter> <properties>`: "not
   supported" on this device means the exporter drops it silently; do not
   commit that counter.
5. `doca_telemetry_utils <0x... Data ID>`: reverse-resolve an ID captured on
   the wire.
6. Check host and BlueField release pairing with `doca-version`.
7. If every step resolves cleanly, hand off to `doca-telemetry` `## debug`.

Quote the tool's output verbatim; never construct Data IDs by hand.

## Instructions

Follow Loading order below. Open the tasks companion for
install / run / test. Open the capabilities companion for
schema discovery and pairing with the telemetry libraries.
Do not open every companion file.

**Where to start:** This is a tool skill for invoking
`doca_telemetry_utils` — the documented host-side CLI that
supports a DOCA Telemetry exporter / collector pipeline by
discovering the counter schema, translating counter names ↔
Data IDs, and probing per-device counter support. Open
the tasks companion and start at
`## install` for the host-side
prerequisites and `## run` for the
three documented invocation classes
(enumerate / name→ID / ID→name). Open
the capabilities companion when the question is
*what does this tool actually discover about the telemetry
schema*, *how does it pair with the developer-side
`doca-telemetry/SKILL.md` and
exporter libraries*, *how do I confirm a device supports a
counter before committing an exporter config to it*, or
*why does my exporter pipeline silently drop a metric*.

**How to use this skill.** Read this file, then search the companion file
for the `## <name>` heading the question needs and read that whole section.
Answer completely from it, quoting only its symbols and commands. Do not
browse docs; use the entry points in `doca-public-knowledge-map`. Give
environment checks as commands for the user's DOCA host. For out-of-scope
requests, name the skill that owns them and stop.

This skill is the **operator-side support tool** for a
DOCA Telemetry deployment. It is NOT the developer-side
collector library (that is
`doca-telemetry/SKILL.md`),
NOT the developer-side publisher library (that is
`doca-telemetry-exporter` — see
`doca-telemetry ## Related skills` (`doca-telemetry/SKILL.md#related-skills`)),
and NOT a DOCA Telemetry Service (DTS) deployment guide
(route to `doca-dts`).
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
  the capabilities companion
  + the per-device-support probe in
  the tasks companion: the exporter
  ships a Data ID, not a name; a name in the config that
  resolves to a Data ID the device does not support is
  silently dropped.
- **"Which DOCA diagnostic counters does this BlueField
  actually expose?"** — worked example: *"enumerate the
  full counter schema for my BlueField-3 before I write
  the exporter config"*. Answered by the schema-discovery
  invocation class in
  the capabilities companion
  (`doca_telemetry_utils get-counters` lists every
  counter name the diagnostic-data surface knows about;
  pair with a per-device probe to confirm support).
- **"I have a Data ID in a captured log — what counter
  was that?"** — worked example: *"a downstream consumer
  emitted `Data ID: 0x1160000600030201` — translate it
  back so I can correlate against the public guide"*.
  Answered by the reverse-resolve invocation class in
  the capabilities companion
  + the
  ID-encodes-properties rule in
  the tasks companion (a Data ID carries
  the counter's property dimensions; the reverse-resolve
  reports them).
- **"What property dimensions does this counter take and
  what values are valid?"** — worked example: *"I know
  the counter is `pcie_link_write_stalled_time_*` — what
  do `node` / `pcie_index` / `depth` mean and what
  values does the device accept?"*. Answered by the
  property-dimension table in
  the capabilities companion
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
  the capabilities companion
  + the gate-before-commit rule in
  the tasks companion.
- **"Is `doca_telemetry_utils` on my install, and is it
  paired with the matching `doca-telemetry` library
  version?"** — worked example: *"is the diagnostic-
  data counter set my exporter targets on this DOCA
  version?"*. Answered by the version-overlay in
  the capabilities companion,
  which redirects to the canonical
  `doca-version/SKILL.md` chain
  and adds the *tool ↔ `doca-telemetry` library
  schema-version* match rule.

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
`doca-public-knowledge-map/SKILL.md`,
`doca-telemetry/SKILL.md`,
or `doca-setup/SKILL.md`.

## What this skill provides

This is a **thin loader**. Substantive material lives
in two companion files:

- the capabilities companion — what `doca_telemetry_utils`
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
- the tasks companion — step-by-step workflows for the
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
`doca-telemetry/SKILL.md`
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
  `doca-public-knowledge-map/SKILL.md`.
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
  and in `doca-telemetry/SKILL.md`.

## Loading order

1. Read this skill first to confirm the user's
   question is in scope (operator-side support
   tooling for a telemetry pipeline, not the
   developer-side library programming).
2. **For the three invocation classes, the
   property-dimension model, the per-device support
   probe, the version overlay, the error taxonomy,
   the observability surface, and the safety
   policy, see the capabilities companion.**
3. **For the documented invocations and the
   discover → resolve → validate → consume
   workflow — `install`, `configure`, `build`,
   `modify`, `run`, `test`, `debug`, `use` — see
   the tasks companion.**

## Related skills

- `doca-telemetry/SKILL.md`
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
- `doca-public-knowledge-map/SKILL.md`
  — routing to the public DOCA Telemetry guide and
  the DOCA Telemetry Service (DTS) page on
  `docs.nvidia.com`, plus the on-disk install
  layout for the tool.
- `doca-version/SKILL.md` —
  canonical DOCA version-handling rules. The
  `## Version compatibility` section in
  the capabilities companion is a concise
  overlay that redirects here for the body and
  adds the *tool ↔ `doca-telemetry` library
  schema-version pairing* rule.
- `doca-setup/SKILL.md` — env
  preparation, install verification, and the *I
  have no install yet* path with the public NGC
  DOCA container. This skill assumes its
  preconditions are satisfied (DOCA installed
  with the telemetry component, BlueField visible
  to DOCA).
- `doca-debug/SKILL.md` — the
  cross-cutting debug ladder. Telemetry-utils
  feeds the cross-cutting ladder by surfacing the
  counter schema + per-device support truth at the
  runtime layer; exporter-pipeline regressions
  often resolve at the schema layer before
  touching the collector / network path.
- `doca-structured-tools-contract/SKILL.md`
  — the bundle's detect → prefer → fall back →
  report contract for structured helper tools. The
  command appendix in the tasks companion
  honors this contract.
- `doca-programming-guide/SKILL.md`
  — general DOCA programming patterns shared by
  every library / tool surface, including the
  cross-library `DOCA_ERROR_*` taxonomy this
  tool's host-side error layer overlays on top of
  when downstream collector / exporter code
  surfaces a related error.
