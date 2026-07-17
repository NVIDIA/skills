# DOCA Telemetry Utils — Tasks

**Where to start:** The verbs that carry real workflow
content are `## install` (host-side DOCA + telemetry
component prerequisites), `## configure` (which
invocation class, which device, which counter),
`## run` (the three documented invocations:
enumerate / name→ID / ID→name with optional per-device
probe), `## test` (round-trip a chosen counter through
schema → per-device probe → exporter ship → collector
receive), `## debug` (layered diagnosis), and `## use`
(hand the validated counter set off to the
developer-side
[`doca-telemetry`](../../libs/doca-telemetry/SKILL.md)
collector / exporter pipeline). The two routing-stub
verbs (`build`, `modify`) are kept because the agent's
task-verb contract is uniform across the bundle and
each carries a meaningful pointer to where the user's
question actually belongs.

This file is loaded by [`SKILL.md`](SKILL.md) after
[`CAPABILITIES.md`](CAPABILITIES.md). It walks the
agent through the documented invocations of
`doca_telemetry_utils`, the schema-discovery + per-
device-support validation workflow, and the hand-off
into a developer-side telemetry pipeline.

## install

`doca_telemetry_utils` is **shipped pre-built** as part
of every DOCA install that includes the Telemetry
optional component, under `/opt/mellanox/doca/tools/`.
The operator-side install path:

1. **Confirm DOCA is installed with the Telemetry
   component.** If the binary is missing, route to
   [`doca-setup ## install`](../../doca-setup/TASKS.md#configure)
   to install or repair the host-side DOCA package
   selection; confirm the version per
   [`doca-version TASKS.md ## configure`](../../doca-version/TASKS.md#configure).
2. **Confirm the BlueField the operator is targeting
   is visible to DOCA.** Walk
   [`doca-setup ## test`](../../doca-setup/TASKS.md#test)
   for the device-visibility check; the per-device
   support probe (per
   [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes))
   requires the device to be visible to DOCA on the
   PCI address the operator passes.
3. **Confirm the developer-side
   [`doca-telemetry`](../../libs/doca-telemetry/SKILL.md)
   library is installed at the matching DOCA
   version** if the operator intends to consume the
   resolved Data IDs from a collector application on
   the same host. The two must come from the same
   DOCA release band per
   [`CAPABILITIES.md ## Version compatibility`](CAPABILITIES.md#version-compatibility).
4. **Confirm the operator has the privileges the
   public DOCA Telemetry guide requires** for the
   per-device probe (binding a `doca_dev` against a
   PCI address typically requires elevated
   privileges).
5. **Confirm the downstream pipeline shape.** Is the
   target an exporter shipping Data IDs to a local
   collector? An exporter shipping into a DTS
   instance? A bare schema-discovery exercise with
   no immediate consumer? The answer affects which
   counters the operator commits to and which
   per-device probe matters.

If the binary is not at the standard path, the fix is
to install / repair the host-side DOCA package
selection that includes the Telemetry component, not
to patch the file in place.

## configure

The tool's configuration is the invocation: there is
no config file, no daemon, no env knob the public
guide documents as required. Steps the agent walks
the user through, in order:

1. **Pick the invocation class.** Per
   [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes)
   three-class table:
    - **Enumerate the schema** — when the operator
      is starting a new exporter config and wants to
      see what counters DOCA understands on this
      install.
    - **Resolve name → Data ID** — when the operator
      knows the counter name (from the public guide,
      from an existing config, from a teammate's
      runbook) and needs the Data ID for the
      exporter config + the property options for
      that counter.
    - **Reverse-resolve Data ID → name** — when the
      operator has a captured Data ID (from a
      downstream consumer's log, from an existing
      exporter config the team inherited) and needs
      the counter name + properties for correlation.
2. **Pick the device (or skip it for schema-only
   resolution).** The optional `<device PCI>`
   leading argument runs the resolved counter
   against the device's capability surface. The
   agent's rule: for every counter that will be
   committed to an exporter config on a target
   device, run the per-device probe; for schema
   discovery without a specific device in mind,
   omit the PCI argument.
3. **Pick property values for the chosen counter.**
   If the operator does not yet know the counter's
   property options, run `doca_telemetry_utils
   <name>` once without property arguments to
   print the documented options (`node`,
   `pcie_index`, `depth`, plus unit-specific
   axes). The agent does NOT invent property values
   from generic CLI knowledge.
4. **Plan the captured-evidence tuple.** Per
   [`CAPABILITIES.md ## Observability`](CAPABILITIES.md#observability)
   evidence tuple — the operator captures (DOCA
   version, BlueField identity + firmware, counter
   name, property values, resolved Data ID,
   per-device probe result) for every counter
   committed to an exporter config. This is the
   audit trail for the config decision.

For the canonical DOCA universal lifecycle on the
developer-side collector / exporter this tool
informs, see
[`doca-programming-guide TASKS.md ## configure`](../../doca-programming-guide/TASKS.md#configure)
and
[`doca-telemetry TASKS.md ## configure`](../../libs/doca-telemetry/TASKS.md#configure).
This skill is concerned with the *operator-side*
configuration of the support-tool invocation.

## build

`doca_telemetry_utils` is **shipped pre-built** as
part of every DOCA install that includes the
Telemetry optional component
(`/opt/mellanox/doca/tools/doca_telemetry_utils`).
There is no source tree the external user is
expected to compile, no build flags, no `meson` or
`make` workflow for the tool itself.

Routing for nearby "build" questions:

- *"The binary isn't there — do I need to build it?"*
  → no. Route to
  [`doca-setup ## install`](../../doca-setup/TASKS.md#configure)
  to install / repair the host-side DOCA package
  selection that includes the Telemetry component.
- *"I want to build my own collector app that
  consumes the resolved Data IDs."* → not a
  `doca_telemetry_utils` question. Route to
  [`doca-telemetry TASKS.md ## build`](../../libs/doca-telemetry/TASKS.md#build)
  for the collector-side build pattern.
- *"I want to extend the tool with a new
  invocation class."* → out of scope here; this
  skill is for external operators consuming the
  shipped tool, not for contributors extending it.

The `## What this skill deliberately does not ship`
block in [`SKILL.md`](SKILL.md) explicitly forbids
adding a build recipe for the tool; revisit that
policy before changing this section.

## modify

**Do not modify the shipped `doca_telemetry_utils`
binary.** It is an NVIDIA-shipped CLI; there is no
documented public way to change its behaviour, output
format, or counter catalog, and none should be
invented.

What the agent *does* modify, every time, is:

1. The **tool invocation** — which invocation
   class, which device PCI, which counter name,
   which property values, which Data ID.
2. The **downstream exporter / collector config**
   that consumes the resolved Data IDs — that is
   where the operator's actual exporter pipeline
   lives. The skill names the requirements the
   config must satisfy (every committed counter
   round-tripped through this tool, every counter
   probed against the target device, re-resolved
   across DOCA upgrades) and refuses to pin a
   specific config format.

Routing for nearby "modify" questions:

- *"I want to override what counter set DOCA
  understands."* → not supported; the counter
  catalog is owned by the
  `doca_telemetry_diag` surface that the host-side
  DOCA install ships. Re-installing or upgrading
  DOCA is the only way to change the catalog.
- *"I want a JSON output from the tool."* → not
  supported on the documented surface; route
  through the
  [`doca-structured-tools-contract`](../../doca-structured-tools-contract/SKILL.md)
  preamble when a structured helper is present, or
  write a parser against the documented text
  output on the installed version per
  [`SKILL.md ## What this skill deliberately does not ship`](SKILL.md#what-this-skill-deliberately-does-not-ship).
- *"I want to change which device the tool probes
  against."* → that is the invocation's `<device
  PCI>` argument; the tool itself is not modified.

## run

The three documented invocations — every
`doca_telemetry_utils` session uses one or more of
them. The full invocation surface lives in the
public DOCA Telemetry guide and the shipped README
at `/opt/mellanox/doca/tools/telemetry_utils/README.md`;
this section names the *shape* of the flow, not
verbatim command lines beyond what the README
documents.

1. **Confirm prerequisites.** Per [`## install`](#install)
   and [`## configure`](#configure): binary
   present, DOCA version known, device visible (if
   per-device probe is planned), invocation class
   chosen.
2. **Schema discovery (enumerate).** Run
   `doca_telemetry_utils get-counters` to print
   the full counter set DOCA understands on this
   install. The agent's rule: capture the output;
   it is the catalog every subsequent name → ID
   resolve uses.
3. **Name → Data ID resolution.** For each
   counter the operator considers for an exporter
   config:
    - Run `doca_telemetry_utils <name>` once
      without property arguments to print the
      documented property options for that
      counter.
    - Run `doca_telemetry_utils <name>
      <property-values>` to resolve the Data ID
      with the chosen property values. The output
      tuple per the shipped README:
      `Data ID: 0x...`, `Name: ...`, `Unit: ...`,
      property-axis values.
    - Capture the resolved tuple as evidence per
      [`CAPABILITIES.md ## Observability`](CAPABILITIES.md#observability).
4. **Per-device support probe.** For each
   counter that will be committed to an exporter
   config on a target device:
    - Run `doca_telemetry_utils <device PCI>
      <name> <property-values>` (or the
      equivalent on `--help`) to run the resolved
      counter against the device's capability
      surface.
    - Confirm the device-side output reports
      *"supported"* for the resolved Data ID.
      If not, the counter is silently dropped at
      the exporter; do NOT commit it.
5. **Reverse-resolve a captured Data ID** (when
   the operator has a Data ID from a downstream
   log). Run `doca_telemetry_utils <DATA_ID>` to
   print the name + unit + property values.
   Capture the result as the correlation
   anchor.

When recording the run for downstream consumers
(the *validated counter set* artifact pattern),
write down: the DOCA version, the BlueField
identity + firmware, every counter the operator
plans to commit, each counter's chosen property
values, the resolved Data ID, the per-device
probe result, and the stdout of every invocation.
The downstream [`## test`](#test) and
[`## use`](#use) workflows depend on those
fields.

## test

`doca_telemetry_utils` is **a discovery + validation
tool**, so its `## test` verb is about *testing the
exporter / collector pipeline against the validated
counter set* — confirming that the counter the
operator resolved actually ships from the exporter
and arrives at the collector / downstream consumer
end-to-end — not unit-testing the tool itself.

**Validation gate (mandatory before any exporter /
collector config is rolled forward).**

1. **Resolve and probe each counter.** Per
   [`## run`](#run) steps 3–4 for every counter in
   the proposed exporter config. Reject any counter
   that does not resolve cleanly or is not
   supported by the target device.
2. **Author / update the exporter config with the
   resolved Data IDs.** The exporter ships Data
   IDs, not names. Per the developer-side library
   ownership in
   [`doca-telemetry`](../../libs/doca-telemetry/SKILL.md),
   the exporter side is owned by the publisher
   library + the user's exporter code, not by
   this tool.
3. **Stand up the collector / downstream
   consumer.** Per
   [`doca-telemetry TASKS.md ## configure`](../../libs/doca-telemetry/TASKS.md#configure)
   and [`## run`](../../libs/doca-telemetry/TASKS.md#run);
   confirm the collector is up and ready to
   receive events.
4. **Run a single-event smoke from the exporter.**
   Per
   [`doca-telemetry TASKS.md ## test`](../../libs/doca-telemetry/TASKS.md#test):
   one event for one of the resolved counters; the
   collector must receive the event with the
   correct Data ID + correct property values.
5. **Reverse-resolve the received Data ID via
   this tool** to confirm the value the collector
   sees matches the name + properties the
   operator expects. This is the *"end-to-end
   honest"* gate.
6. **Roll forward the exporter config** to the
   rest of the fleet only after the round-trip
   succeeds.

The iteration loop (apply to every exporter-config
rotation):

| Iteration trigger | What it looks like | What changes next iteration |
| --- | --- | --- |
| `get-counters` lists a counter the operator wants, but the per-device probe reports *"not supported"* | Schema-vs-device support gap | Pick a different counter, or pick a different device class; do NOT commit the unsupported counter. |
| Forward resolve prints a Data ID, but reverse-resolve on a different DOCA version produces a different name | DOCA version skew — the counter was renamed or its property axes changed | Walk [`CAPABILITIES.md ## Version compatibility`](CAPABILITIES.md#version-compatibility); re-resolve against the version the exporter is actually targeting. |
| Exporter ships a Data ID, collector receives nothing | Downstream pipeline issue; the schema-discovery tool's role ends here | Hand off to [`doca-telemetry TASKS.md ## debug`](../../libs/doca-telemetry/TASKS.md#debug) for the collector-side ladder; this tool already confirmed the schema half of the contract. |
| Same DOCA version, two different BlueField generations, same counter — one supports, one does not | Per-device support is install + firmware versioned | Resolve / probe separately per device class; the exporter config may need to be parameterised per device. |
| Operator passes a property value the documented set does not accept | Property-value-out-of-range error | Re-run `doca_telemetry_utils <name>` without properties to print the documented options; re-run with a valid value. |

The agent's rule: every change to the counter set,
the property values, or the DOCA version re-opens
the loop. Re-using a resolved Data ID across a
DOCA upgrade without re-resolving is exactly the
silent-metric-drop failure mode this tool exists
to prevent.

Loop termination: stop iterating once every
counter in the proposed exporter config has been
(a) resolved cleanly, (b) probed cleanly against
every target device class, and (c) round-tripped
end-to-end through one exporter → collector
smoke.

This skill does NOT ship a "test fixture" or
pre-recorded expected output. The expected output
is install-, device-, and version-specific.

## debug

When `doca_telemetry_utils` fails to enumerate, fails
to resolve a name or Data ID, or reports a counter
as not supported on the target device — or when the
downstream exporter / collector pipeline silently
drops a metric the operator thought was shipping —
walk the layered error taxonomy in
[`CAPABILITIES.md ## Error taxonomy`](CAPABILITIES.md#error-taxonomy)
in order. The shape of the diagnosis:

1. **Install.** Confirm the binary exists, the
   Telemetry component is installed, and DOCA is
   at a known version per
   [`doca-version TASKS.md ## configure`](../../doca-version/TASKS.md#configure).
2. **Parse.** Re-read `--help` on the installed
   binary and the shipped README at
   `/opt/mellanox/doca/tools/telemetry_utils/README.md`;
   confirm the invocation matches one of the
   three documented classes; confirm the Data ID
   format (the documented form is `0x`-prefixed
   hex).
3. **Unknown counter.** Run
   `doca_telemetry_utils get-counters` to
   enumerate the actual counter set on this
   install; cross-check the public DOCA Telemetry
   guide for the version the user is targeting.
4. **Unknown Data ID.** Cross-check the Data ID
   against a fresh forward resolve from the same
   DOCA version; Data IDs are not hand-constructible
   in general.
5. **Property-value-out-of-range.** Run
   `doca_telemetry_utils <name>` without
   properties to print the documented options;
   re-run with a valid value.
6. **Device-not-supported.** This is the answer
   the user came for — do NOT commit this counter
   to the exporter config for this device. Walk
   the per-platform support matrix in the public
   DOCA Telemetry guide; pick a different counter
   or a different device class.
7. **Version.** Walk
   [`doca-version TASKS.md ## debug`](../../doca-version/TASKS.md#debug)
   end-to-end; the tool ↔
   [`doca-telemetry`](../../libs/doca-telemetry/SKILL.md)
   library schema-version pairing is the common
   mismatch failure.
8. **Cross-cutting (downstream).** If the tool
   reports cleanly but the exporter / collector
   pipeline still fails, the cause is in the
   downstream pipeline, not in this tool. Hand
   off to
   [`doca-telemetry TASKS.md ## debug`](../../libs/doca-telemetry/TASKS.md#debug)
   and
   [`doca-debug ## debug`](../../doca-debug/SKILL.md).

In every case: **quote what the tool reported.** Do
not paraphrase the resolved tuple, do not invent a
Data ID, do not summarize a *"not supported"*
result into *"may not be supported"*.

## use

The validated counter set is consumed by the
**developer-side exporter / collector pipeline**.
The agent's hand-off:

1. **Author / update the exporter config with the
   resolved Data IDs**, per the developer-side
   ownership in
   [`doca-telemetry`](../../libs/doca-telemetry/SKILL.md)
   (collector) plus the publisher-side library
   that emits the IDs.
2. **Stand up the collector** per
   [`doca-telemetry TASKS.md ## configure`](../../libs/doca-telemetry/TASKS.md#configure)
   and
   [`## run`](../../libs/doca-telemetry/TASKS.md#run);
   confirm the collector is up before the
   exporter starts emitting (per the
   publisher-up-before-collector staging the
   collector-library skill enforces).
3. **Round-trip an end-to-end smoke** per
   [`## test`](#test); confirm the Data ID the
   collector sees reverse-resolves through this
   tool to the name + properties the operator
   committed.
4. **Roll forward only after the round-trip is
   clean.** Do NOT roll out an exporter config
   that has not been end-to-end smoked.
5. **Re-resolve every counter across DOCA
   upgrades** before the upgraded exporter
   pipeline is rolled forward, per
   [`CAPABILITIES.md ## Safety policy`](CAPABILITIES.md#safety-policy)
   re-resolve-on-upgrade rule.

The agent's rule: this tool produces the validated
counter set; the
[`doca-telemetry`](../../libs/doca-telemetry/SKILL.md)
library (and the publisher-side library) consumes
it. Conflating the two surfaces is the most common
telemetry first-touch error.

## Deferred task verbs

The verbs below are not `doca_telemetry_utils` work
and should be routed out before the agent does any
of them under this skill's name.

- **Developer-side collector / exporter
  programming** →
  [`doca-telemetry`](../../libs/doca-telemetry/SKILL.md)
  (collector library) + the publisher-library
  skill referenced from
  [`doca-telemetry ## Related skills`](../../libs/doca-telemetry/SKILL.md#related-skills).
  This tool informs those skills; it does not
  redefine them.
- **DTS deployment / configuration** → separate
  DOCA service with its own public guide; route
  via
  [`doca-public-knowledge-map ## DOCA services`](../../doca-public-knowledge-map/SKILL.md#doca-services).
- **DOCA install / repair / upgrade** → route to
  [`doca-setup ## install`](../../doca-setup/TASKS.md#configure).
- **Cross-cutting Telemetry pipeline observability
  (Prometheus / OpenTelemetry / fluentd
  integration)** → operator-owned downstream
  tooling; this skill names the validated counter
  surface that downstream tooling consumes, but
  does not pin a downstream tool.

## Command appendix

`doca_telemetry_utils`-specific invocation classes
the verbs above reach for. Every row is a CLASS —
the agent must not invent flags, counter names,
property values, or Data IDs beyond `--help` on
the installed binary, the shipped README at
`/opt/mellanox/doca/tools/telemetry_utils/README.md`,
and the public DOCA Telemetry page.

**Infra-aware preamble (every row below).** Per
the bundle's detect → prefer → fall back → report
contract documented in
[`doca-structured-tools-contract ## The agent behavior contract`](../../doca-structured-tools-contract/SKILL.md#the-agent-behavior-contract),
the agent should:

1. Probe for the matching structured helper FIRST
   (`doca-env --json` for version + devices +
   telemetry-component availability in one shot;
   `doca-capability-snapshot` for per-device
   capability flags).
2. If the probe succeeds, the structured tool's
   output is the authoritative answer.
3. If the probe fails, fall back to the manual
   command in the row.
4. The schemas the structured tools emit are
   defined in
   [`doca-structured-tools-contract ## Schemas`](../../doca-structured-tools-contract/SKILL.md#schemas).

| Purpose (class) | Invocation (shape) | Owning step | Reads as healthy when … |
| --- | --- | --- | --- |
| Print help banner / documented flag surface | `doca_telemetry_utils` (no args) + the shipped README + the public DOCA Telemetry page | [`## configure`](#configure); [`## debug`](#debug) layer 2 | Prints the documented inventory of subcommands and option shapes. |
| Enumerate the diagnostic-counter schema | `doca_telemetry_utils get-counters` | [`## run`](#run) step 2 | Prints every counter name DOCA understands on this install; the agent captures the output as the canonical catalog. |
| Resolve a name to a Data ID with properties | `doca_telemetry_utils <name> <property-values>` (re-run `doca_telemetry_utils <name>` first if property options are unknown) | [`## run`](#run) step 3 | Prints the (Data ID, Name, Unit, property-values) tuple per the shipped README's worked example. |
| Run a per-device support probe | `doca_telemetry_utils <device PCI> <name> <property-values>` | [`## run`](#run) step 4 | Reports the resolved tuple + a *"supported"* / *"not supported"* answer for this device. |
| Reverse-resolve a Data ID | `doca_telemetry_utils <DATA_ID>` (`0x`-prefixed hex per the shipped README's worked example) | [`## run`](#run) step 5 | Prints the counter name + unit + property values that the Data ID encodes. |

Three cross-cutting rules for this appendix:

- **Never invent a counter name, property value,
  or Data ID.** `--help` on the installed binary,
  the shipped README, and the public DOCA
  Telemetry page are the joint contract.
- **Round-trip before commit.** Every counter
  committed to an exporter config is first run
  through name → ID → per-device probe → exporter
  → collector receipt → reverse-resolve. Skipping
  any step is the canonical silent-metric-drop
  failure.
- **Cross-link instead of duplicate.** Cross-
  cutting commands (`pkg-config --modversion
  doca-telemetry`, `doca_caps --list-devs`,
  `lspci` for the BlueField PCI address) live in
  [`doca-debug ## debug`](../../doca-debug/SKILL.md) and
  [`doca-setup TASKS.md ## debug`](../../doca-setup/TASKS.md#debug);
  this appendix names only
  `doca_telemetry_utils`-specific invocation
  classes.

## Cross-cutting

A few rules that apply across every verb in this
file, restated here so they are visible at the
point of action and not buried in
[`SKILL.md`](SKILL.md):

- The **public DOCA Telemetry page** plus the
  installed `--help` and the shipped README are
  the joint source of truth. When they disagree,
  the *installed* output wins for the user's
  actual run.
- **Exporters ship Data IDs, not names.** Every
  counter in an exporter config is first
  resolved through this tool; a name in the
  config is a recipe for a silent metric drop.
- **Per-device support is mandatory before
  commit.** A counter that resolves on the
  schema but is not supported on the target
  device produces no metric.
- **Re-resolve on DOCA upgrade.** Every counter
  in every exporter config is re-resolved
  against the new install before the exporter
  pipeline is rolled forward.
- **Quote the (DOCA version, BlueField identity
  + firmware, counter name, property values,
  resolved Data ID, per-device probe result)
  tuple.** Evidence without this tuple is
  unreplicable.
- This skill **assumes a healthy DOCA install**
  with the Telemetry component and a BlueField
  visible to DOCA. If either is in doubt, route
  to [`doca-setup`](../../doca-setup/SKILL.md)
  before running anything else here.
