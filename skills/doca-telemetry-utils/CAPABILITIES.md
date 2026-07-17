# DOCA Telemetry Utils — Capabilities

**Where to start:** `doca_telemetry_utils` is the
operator-side support tool for a DOCA Telemetry exporter /
collector pipeline. The pattern overview below names the
recurring `doca_telemetry_utils`-class questions. Pick the
pattern first, then drill into the H2 that owns the
substance. For the *how* of executing each pattern, jump
to [TASKS.md](TASKS.md).

This file is loaded by [`SKILL.md`](SKILL.md). It documents
*what the tool discovers about the diagnostic-counter
schema*, *its three documented invocation classes*,
*the property-dimension model counters carry*, *the
optional per-device capability probe*, *the version
overlay against the developer-side
[`doca-telemetry`](../../libs/doca-telemetry/SKILL.md)
library*, *the layered error and observability surfaces*,
*and the safety posture* (the tool is read-only;
operator mistakes show up downstream as silent metric
drops, not as crashes). For step-by-step invocations and
the discover → resolve → validate → consume workflow,
see [`TASKS.md`](TASKS.md).

## Pattern overview

Every `doca_telemetry_utils`-class question this skill
teaches resolves into one of SIX patterns. The patterns
are CLASSES — they apply across every BlueField the tool
can probe and every DOCA Telemetry pipeline shape, not
just one.

| `doca_telemetry_utils` pattern | Class shape | Where the substance lives |
| --- | --- | --- |
| 1. Discover the counter schema | Enumerate every counter name the DOCA diagnostic-data surface knows about, before committing exporter / collector config. `doca_telemetry_utils get-counters` is the enumeration entrypoint. | [`## Capabilities and modes`](#capabilities-and-modes) invocation-class table + [TASKS.md ## run](TASKS.md#run) step 1 |
| 2. Resolve name → Data ID + properties | Translate a human-readable counter name into the binary Data ID an exporter actually ships, with the documented property dimensions (`node`, `pcie_index`, `depth`, …). The exporter publishes IDs, not names; a name in the config that resolves to the wrong ID is silently dropped downstream. | [`## Capabilities and modes`](#capabilities-and-modes) invocation-class table + [TASKS.md ## run](TASKS.md#run) step 2 |
| 3. Reverse-resolve Data ID → name + properties | Translate a captured Data ID from a downstream consumer back to its counter name + properties, for correlation against the public DOCA Telemetry guide or against the exporter's source-of-truth config. | [`## Capabilities and modes`](#capabilities-and-modes) invocation-class table + [TASKS.md ## run](TASKS.md#run) step 3 |
| 4. Validate per-device support | Run the resolved counter against a specific device's capability surface (`<device PCI> <name> [properties]`) before committing it to an exporter config. A counter that resolves cleanly but is not supported on the target device produces no metric. | [`## Capabilities and modes`](#capabilities-and-modes) per-device-probe rule + [TASKS.md ## test](TASKS.md#test) validation gate |
| 5. Diagnose a silent metric drop | A downstream consumer reports a metric missing; the exporter says it's emitting. Walk the schema-mismatch / unsupported-counter error taxonomy in [`## Error taxonomy`](#error-taxonomy) before suspecting the collector or network path. | [`## Error taxonomy`](#error-taxonomy) + [TASKS.md ## debug](TASKS.md#debug) |
| 6. Bake a validated exporter config | The hand-off into the developer-side pipeline. Each counter the operator wants in the exporter config is first run through name → ID → per-device probe; the resulting validated tuple is the artifact the exporter config references. | [TASKS.md ## use](TASKS.md#use) hand-off + [`doca-telemetry`](../../libs/doca-telemetry/SKILL.md) for the consumer side |

Two cross-cutting rules that apply to *every* pattern
above:

- **The exporter ships Data IDs, not names.** Operator-
  side confusion between *"the human-readable counter
  name"* and *"the Data ID the exporter actually
  publishes"* is the most common silent-metric-drop
  failure mode. The agent's rule: every counter in an
  exporter config is resolved through this tool before
  the config is deployed, and the resolved Data ID is
  the canonical reference in the config.
- **Per-device support is not assumed from the schema.**
  The diagnostic-counter schema is what DOCA *knows
  about*; per-device support is what *this BlueField at
  this firmware* actually exposes. The two are not the
  same; the operator's rule for any exporter config is
  to validate each counter against the device's
  capability surface before committing.

## Capabilities and modes

`doca_telemetry_utils` is shipped as a single CLI binary
under `/opt/mellanox/doca/tools/` on every DOCA install
that includes the Telemetry optional component. It is a
read-only CLI; it does not configure DTS, write exporter
configs, or ship telemetry itself. Its job is the
schema-discovery + per-device-support question — the
operator-side knowledge that exporter / collector setups
need to be honest.

**Three documented invocation classes.** Per the README
shipped under `doca/tools/telemetry_utils/` and the
public DOCA Telemetry guide:

| Invocation class | Shape | What it produces |
| --- | --- | --- |
| Enumerate the schema | `doca_telemetry_utils get-counters` | Prints every counter name the DOCA diagnostic-data surface knows about on this install. The output is the catalog of *what DOCA understands*; per-device support is checked separately. |
| Resolve name → Data ID | `doca_telemetry_utils [<device PCI>] <name> [relevant properties]` | Resolves the counter name + property values to a Data ID, prints the Data ID + name + unit + property values. If `<device PCI>` is provided, also probes the device's support for the resolved Data ID. Running `<name>` alone prints the documented property options for that counter. |
| Reverse-resolve Data ID → name | `doca_telemetry_utils [<device PCI>] <DATA_ID>` | Reverse-resolves the Data ID to a counter name + unit + property values. If `<device PCI>` is provided, also probes the device's support. |

Running the tool with no arguments prints a help
banner per the shipped README; the exact subcommand
strings and option names are authoritative on `--help`
on the installed version and on the public DOCA
Telemetry guide.

**Property-dimension model.** Diagnostic counters carry
documented property dimensions; the same counter name
can resolve to different Data IDs depending on which
property values the operator picks. The general property
axes (re-verify against `--help` on the installed
binary):

| Property axis | What it picks |
| --- | --- |
| `node` | Which logical node the counter is reported against — e.g. a per-port counter has a `node` that selects which port. |
| `pcie_index` | Which PCIe interface the counter refers to — e.g. for `pcie_link_*` counters that distinguish between the device's PCIe ports / lanes. |
| `depth` | Which depth-axis bucket the counter applies to — e.g. for queue-depth-conditioned counters. |
| Unit-specific axes (per the counter's `Unit:` line) | Each counter belongs to a `Unit` (e.g. `PCIE`, the host-RX-transport group, the TX-port group); each unit can expose unit-specific property axes beyond the generic three. |

The shipped README's worked example (re-verifiable on a
local DOCA install):

```
$ doca_telemetry_utils pcie_link_write_stalled_time_no_posted_header_credits_ns 1 2 3
Data ID: 0x1160000600030201
Name: pcie_link_write_stalled_time_no_posted_header_credits_ns
Unit: PCIE
node: 1
pcie_index: 2
depth: 3
```

The agent's rule for *any* counter the operator names:
run `doca_telemetry_utils <name>` once without
properties to print the documented property options for
that counter, then re-run with the chosen properties
to resolve the Data ID. Inventing property values from
generic CLI knowledge is the canonical hallucination
failure for this skill.

**Counter catalog scope.** The schema the tool exposes
is the DOCA Telemetry diagnostic-data surface (the
`doca_telemetry_diag` API on the developer side). The
documented surface covers RX / TX port counters, host
RX / TX transport-layer counters, PCIe link counters,
completion-engine counters, ICMC counters, sensor
counters (voltage / current / power / temperature), and
other groups documented in the public DOCA Telemetry
guide. The exact counter set evolves across DOCA
versions; the agent does NOT enumerate it from memory.

**Optional per-device capability probe.** Whenever
`<device PCI>` is provided as the first argument, the
tool runs the resolved counter against the device's
capability surface. The two pieces of information the
tool returns:

- The resolved Data ID + name + unit + property values
  (the schema-side answer).
- Whether the device supports the resolved Data ID
  (the per-device answer).

A counter that resolves on the schema but is not
supported on the device produces no metric when the
exporter ships it. The agent's rule for every counter
that will be committed to an exporter config: run the
per-device probe with the target device's PCI address,
and only commit if the device supports the Data ID.

**What this tool deliberately does not do.** The tool
is read-only; the operator-side support role is
schema discovery + per-device support probing, not
exporter configuration, not DTS deployment, not
metric shipping. Once the operator has the validated
Data ID for each counter, the hand-off is into the
developer-side
[`doca-telemetry`](../../libs/doca-telemetry/SKILL.md)
collector or the exporter library that consumes the
IDs.

## Version compatibility

For the canonical DOCA version-detection chain, the
four-way match rule, NGC container semantics, and the
headers-win-over-docs rule, see
[`doca-version`](../../doca-version/SKILL.md). The body
lives there; this skill does not duplicate it.

**The `doca_telemetry_utils`-specific overlay** is:

- **Tool ↔ library schema-version pairing.** The tool
  resolves names ↔ Data IDs against the
  `doca_telemetry_diag` schema the host-side DOCA
  install carries; the developer-side
  [`doca-telemetry`](../../libs/doca-telemetry/SKILL.md)
  library consumes those Data IDs. The two must come
  from the same DOCA release band per the
  [`doca-version CAPABILITIES.md ## Version compatibility`](../../doca-version/CAPABILITIES.md#version-compatibility)
  four-way match rule, OR the exporter / collector
  pipeline can ship Data IDs the consumer's library
  does not understand.
- **The counter catalog is install-versioned.** New
  DOCA versions add new counters; some older
  counters can be renamed or have new property
  dimensions added. The agent's rule when migrating
  an exporter config across DOCA versions:
  re-resolve every counter in the config against the
  new install's `doca_telemetry_utils
  get-counters`, before deploying. A name that
  resolved on the old install is not guaranteed to
  resolve identically on the new install.
- **Per-device support is install + firmware
  versioned.** Even at the same DOCA version, two
  BlueField generations or two firmware levels can
  differ on which counters are exposed. The per-
  device probe (`<device PCI>`) is the runtime
  authority on top of the schema-side catalog.
- **Public DOCA Telemetry page is the source of
  truth for the tool's command-line surface.** When
  that page disagrees with this skill, the live
  guide wins.

## Error taxonomy

`doca_telemetry_utils`'s error surface spans the host
install, command-line parse errors, schema-side
*"unknown counter"* / *"unknown Data ID"* errors,
property-value-out-of-range errors, per-device
*"not supported on this device"* errors, the cross-
cutting partial-install / mixed-version layer, and
the downstream-pipeline (exporter / collector)
errors the tool does NOT itself emit but routes
to. The agent distinguishes these layers in
escalating order; jumping layers wastes the user's
time on the wrong fix.

1. **Install.** The binary is missing from
   `/opt/mellanox/doca/tools/`, the Telemetry
   optional component is not installed, or the
   binary's loader-dependent shared libs are
   missing. Routing: route to
   [`doca-setup ## install`](../../doca-setup/TASKS.md#configure)
   to install / repair the host-side DOCA package
   selection; confirm the installed version per
   [`doca-version TASKS.md ## configure`](../../doca-version/TASKS.md#configure).
2. **Parse.** The invocation itself does not parse
   — unknown subcommand, missing required argument,
   Data ID in the wrong format (the documented
   format is `0x`-prefixed hex). Routing: re-read
   `--help` on the installed binary and the public
   DOCA Telemetry guide; do not guess at the
   syntax.
3. **Unknown counter.** The counter name passed
   does not exist in the schema on this DOCA
   version. Cause: typo, renamed counter across
   versions, or counter from a different group the
   user thought was supported. Routing: run
   `doca_telemetry_utils get-counters` to
   enumerate the actual set on this install;
   re-confirm against the public DOCA Telemetry
   guide for the version the user is targeting.
4. **Unknown Data ID.** The Data ID passed for
   reverse-resolve does not exist in the schema on
   this DOCA version. Cause: the ID is from a
   different DOCA version, a typo, or was
   constructed by hand (Data IDs encode property
   values and must come from a forward resolve).
   Routing: cross-check the Data ID against a fresh
   forward resolve from the same DOCA version.
5. **Property-value-out-of-range.** Counter is
   known, but a property value is outside the
   documented range. Cause: the operator passed
   `node`, `pcie_index`, `depth`, or a unit-
   specific axis value the documented set does not
   accept. Routing: run `doca_telemetry_utils
   <name>` without properties to print the
   documented options for that counter; re-run
   with a valid value.
6. **Device-not-supported.** Counter resolves on
   the schema but the device probe (`<device PCI>`)
   reports the device does not support the
   resolved Data ID. Cause: BlueField generation,
   firmware level, or device configuration does
   not expose this counter. Routing: this is the
   answer the user came for — do NOT commit this
   counter to the exporter config for this
   device. Re-check the public DOCA Telemetry
   guide for the per-platform support matrix.
7. **Version.** Cross-cutting partial-install /
   mixed-version per
   [`doca-version CAPABILITIES.md ## Version compatibility`](../../doca-version/CAPABILITIES.md#version-compatibility).
   Symptoms: the tool's reported counter set
   disagrees with the developer-side
   [`doca-telemetry`](../../libs/doca-telemetry/SKILL.md)
   library version's schema. Routing: walk
   [`doca-version TASKS.md ## debug`](../../doca-version/TASKS.md#debug)
   end-to-end.
8. **Cross-cutting (downstream pipeline).** The
   tool reports cleanly but the exporter / collector
   pipeline still fails to ship a metric. Cause is
   in the downstream pipeline, not in the
   telemetry-utils tool. Routing: hand off to
   [`doca-telemetry TASKS.md ## debug`](../../libs/doca-telemetry/TASKS.md#debug)
   for the collector-side ladder, plus
   [`doca-debug ## debug`](../../doca-debug/SKILL.md)
   for the cross-cutting layers (driver, firmware,
   network).

`doca_telemetry_utils` does not itself participate in
the cross-library `DOCA_ERROR_*` taxonomy that DOCA
libraries return through their C API; it is a CLI
driving the diagnostic-data surface. The cross-library
taxonomy is owned by
[`doca-programming-guide CAPABILITIES.md ## Error taxonomy`](../../doca-programming-guide/CAPABILITIES.md#error-taxonomy).

## Observability

`doca_telemetry_utils`'s observability surface is
**stdout** — the tool prints the resolved Data ID +
counter name + unit + property values, or the
enumerated counter set, in human-readable form. There
is no file output, no daemon, no live-streaming mode.

- **Stdout.** The primary surface. Per the README,
  the resolved output is the (Data ID, Name, Unit,
  property-axis values) tuple. Capture stdout for
  every invocation the operator commits to an
  exporter config; it is the audit trail.
- **Per-device probe outcome.** When `<device PCI>`
  is provided, the printed tuple is the union of
  the schema-side answer and the device-side
  *"supported"* / *"not supported"* answer; the
  agent's rule is to preserve the entire output as
  evidence for the exporter config decision.
- **No structured output.** The tool does not (per
  the public surface) emit JSON / CSV. Users who
  want a machine-parseable form should read the
  live guide and write the parser against their
  installed version, or use the structured helper
  per the
  [`doca-structured-tools-contract`](../../doca-structured-tools-contract/SKILL.md)
  preamble when present.
- **Captured-evidence tuple.** The minimum metadata
  to attach to a validated counter committed to an
  exporter config: (DOCA version, BlueField
  identity + firmware, counter name, property
  values, resolved Data ID, per-device probe
  result). Without these, an exporter config
  cannot be regression-tested across DOCA
  upgrades.

For the cross-cutting env-side observability
primitives (PCIe scans, `devlink`, `mlxconfig`), see
[`doca-setup CAPABILITIES.md ## Observability`](../../doca-setup/CAPABILITIES.md#observability).
For the developer-side collector observability
(per-consume status, schema-query, consumer-queue-
full handling), see
[`doca-telemetry CAPABILITIES.md ## Observability`](../../libs/doca-telemetry/CAPABILITIES.md#observability).

## Safety policy

> **Overlay on the bundle-wide hardware-safety meta-policy.** The rules below are this skill's per-artifact overlay on the cross-cutting rules in [`doca-hardware-safety` CAPABILITIES.md ## Safety policy](../../doca-hardware-safety/CAPABILITIES.md#safety-policy) (specifically [### Per-artifact overlay pattern](../../doca-hardware-safety/CAPABILITIES.md#per-artifact-overlay-pattern)). When the two layers disagree, the stricter wins; when either layer says STOP, the agent stops.

`doca_telemetry_utils` is a **read-only operator-side
support tool**. It does not load algorithms, configure
firmware, or change device state. Its safety surface is
instead about the *downstream consequence* of an
operator decision the tool informs:

- **A wrong Data ID in an exporter config silently
  drops the metric.** This is the load-bearing
  failure mode the tool exists to prevent. The
  agent's rule: every counter committed to an
  exporter config is first round-tripped through
  the tool (name → ID → per-device probe → commit),
  not committed from prose / blog / memory.
- **The schema-side answer is not the per-device
  answer.** A counter that the schema knows about
  is not the same as a counter the target
  BlueField at this firmware level supports. The
  per-device probe (`<device PCI>`) is mandatory
  before committing a counter to an exporter
  config; skipping it is the canonical *"my
  exporter ships nothing for this device"* failure.
- **Re-resolve on DOCA version upgrades.** When the
  user upgrades DOCA on the host, every counter in
  every exporter config must be re-resolved
  against the new install before the exporter
  pipeline is rolled forward. A counter that
  resolved on the old install is not guaranteed to
  resolve identically on the new install.
- **Telemetry data can leak operational
  information.** Resolved counters and their
  reported values may surface workload-sensitive
  information (port utilization, retransmit
  counts, sensor readings); apply the operator's
  data-handling policy to the resolved-counter
  tuples + any captured downstream telemetry.
- **Do not invent counter names, Data IDs, or
  property values.** The documented surface lives
  on the public DOCA Telemetry guide and in
  `--help` on the installed binary;
  prose-derived names / IDs / property values are
  the most common hallucination failure for this
  skill, and a wrong Data ID in an exporter
  config is exactly the kind of silent failure
  this tool exists to prevent.

This tool does NOT itself participate in the
bundle-wide hardware-touching change classes
(`mlxconfig`-class writes, firmware burns,
BlueField mode flips); the meta-policy applies
when the operator's downstream exporter / collector
configuration touches those, in which case the
hardware-safety overlays of the matching
per-artifact skill (e.g.
[`doca-pcc CAPABILITIES.md ## Safety policy`](../../libs/doca-pcc/CAPABILITIES.md#safety-policy)
when telemetry sits next to PCC) take over.

## Public-source pointer

The canonical public source for `doca_telemetry_utils`
is the **DOCA Telemetry** page on `docs.nvidia.com`
and the **DOCA Telemetry Service (DTS)** guide on
the same site, reachable through
[`doca-public-knowledge-map ## DOCA tools`](../../doca-public-knowledge-map/SKILL.md#doca-tools)
and
[`doca-public-knowledge-map ## DOCA services`](../../doca-public-knowledge-map/SKILL.md#doca-services).
The shipped README under
`/opt/mellanox/doca/tools/telemetry_utils/README.md`
on a real DOCA install carries the three documented
invocation classes and one worked example each. Do
not invent counter names, Data IDs, property values,
or unit-specific axes beyond what those public
sources document, and re-verify against `--help` on
the installed binary.
