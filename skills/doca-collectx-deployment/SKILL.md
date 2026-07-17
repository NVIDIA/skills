---
license: Apache-2.0
name: doca-collectx-deployment
description: >
  Use this skill to deploy and operate a CollectX (clx) based
  DOCA telemetry collector on a host or BlueField — wiring
  providers / counters into the collector, running the
  collection daemon, and shaping its exporters (Prometheus
  pull, Fluent Bit push, NetFlow, file / IPC) so the metrics
  actually leave the box. Trigger even when
  the user never says CollectX or clx — implicit phrasings:
  {collector emits nothing downstream}, {add a provider to the
  clx collector}, {turn on the Prometheus endpoint}, {ship
  counters to Fluent Bit from the DPU}, {daemon starts but no
  schema rows appear}. This skill owns the CollectX collection
  mechanism plus the operator's own doca-telemetry /
  doca-telemetry-exporter usage; it ROUTES the
  productized DOCA Telemetry Service (DTS) to public docs
  (AGENTS.md Non-goal #7), the reader API to doca-telemetry, and
  the publisher API to doca-telemetry-exporter. Refuse to invent
  clx symbols, provider names, schema fields, flags, or config
  paths — describe the class and route to the live source.
metadata:
  kind: library
compatibility: >
  No DOCA install required to read this skill (it is a
  deployment / operation overlay over the DOCA telemetry
  libraries and the CollectX collection mechanism). The
  hands-on steps DO require a live DOCA install at
  /opt/mellanox/doca on a host or BlueField, an operator
  account that can run the collector and reach its exporter
  sinks, and the public DOCA Telemetry / DTS guides on
  docs.nvidia.com for any concrete provider name, schema field,
  flag, or config path (this skill never invents those).
---

# DOCA CollectX telemetry deployment

**Where to start:** This skill is the bundle's home for
*operating a CollectX (clx) based telemetry collector* — the
collection framework that gathers provider counters into a
schema and ships them out through one or more exporters. It is a
deployment / operation skill, parallel to
[`doca-bare-metal-deployment`](../doca-bare-metal-deployment/SKILL.md)
and [`doca-container-deployment`](../doca-container-deployment/SKILL.md):
it owns the *runtime shape* of a telemetry collector on the
operator's host or BlueField, not the library APIs the operator's
own program calls. If the user wants to stand up, wire, or debug
a collector and its exporters, open [`TASKS.md`](TASKS.md) and
start at [`## configure`](TASKS.md#configure). If the question is
*what surfaces does the collector even have and where is the
scope boundary*, start at [`CAPABILITIES.md`](CAPABILITIES.md).
If the user has not installed DOCA yet, route to
[`doca-setup`](../doca-setup/SKILL.md) first.

## The scope boundary (read this before anything else)

CollectX (clx) is NVIDIA's telemetry **collection** framework. It
underpins the **DOCA Telemetry Service (DTS)** — and DTS
*as-deployed* (the productized, NGC-shipped / kubelet-started
service container) is **out of scope** for this bundle per
[`AGENTS.md` Non-goal #7](../../AGENTS.md#non-goals-questions-the-agent-should-recognize-and-refuse-politely).
This skill therefore draws a hard line and the agent MUST state
it up front:

- **In scope here:** the CollectX *collection mechanism* as a
  class (providers / counters → schema → collector daemon →
  exporters), and the operator deploying / running / debugging a
  collector that they own, plus the operator's own usage of the
  two in-bundle telemetry **libraries** when those feed or
  consume the collector.
- **Routed to the DOCA telemetry libraries:** the
  hardware-counter **reader** API is owned by
  [`doca-telemetry`](../libs/doca-telemetry/SKILL.md); the
  application-side **publisher** API (emit counters / events from
  a DOCA program) is owned by
  [`doca-telemetry-exporter`](../libs/doca-telemetry-exporter/SKILL.md).
  This skill does not re-document either API surface.
- **Routed to public docs (Non-goal #7):** the productized
  **DTS container** — its packaged config schema, its built-in
  provider set, its kubelet manifest, its NGC image — is
  externally productized. Route every "operate the DTS service"
  question to the
  [`doca-public-knowledge-map` externally-productized routing row](../doca-public-knowledge-map/SKILL.md#externally-productized-doca-software--not-in-this-bundle-but-here-is-where-to-route)
  and the public DTS guide it points at. The agent must NOT
  synthesize DTS config file names, provider knob names, or
  paths from memory.

The load-bearing first-touch failure this skill exists to
prevent is **collapsing these four surfaces into "DOCA
telemetry"**: the clx collection mechanism, the `doca-telemetry`
reader library, the `doca-telemetry-exporter` publisher library,
and the productized DTS container are four different things with
four different owners. The agent surfaces the decomposition
BEFORE any config-level guidance.

## Audience

This skill serves **external operators standing up or running a
CollectX-based telemetry collector** on a host or BlueField they
administer — people who already have:

- a DOCA install on the side they are collecting from (host x86
  or BlueField Arm), verified per
  [`doca-setup ## test`](../doca-setup/TASKS.md#test),
- a goal of *getting counters off the box* through a collector +
  exporter, not of writing the reader / publisher library code
  (that is the two `libs/` skills above), and
- access to the public DOCA Telemetry and DTS guides on
  `docs.nvidia.com` as the authoritative source for any concrete
  provider name, schema field, flag, or config path.

It is **not** for:

- developers writing the hardware-counter reader API (route to
  [`doca-telemetry`](../libs/doca-telemetry/SKILL.md)) or the
  publisher API (route to
  [`doca-telemetry-exporter`](../libs/doca-telemetry-exporter/SKILL.md)),
- operators deploying / configuring the **productized DTS
  container** as a turnkey service — that is externally
  productized (Non-goal #7); route to the public DTS guide,
- fresh-no-install users — those belong on
  [`doca-setup ## no-install`](../doca-setup/TASKS.md#no-install).

The skill teaches the agent the *procedure and the scope
boundary*; it does not invent clx symbol names, provider names,
schema field names, exporter flag names, or config paths from
memory — those come from the live install and the public docs via
[`doca-public-knowledge-map`](../doca-public-knowledge-map/SKILL.md).

## When to load this skill

Load this skill when the user is doing hands-on **deployment or
operation of a CollectX-based telemetry collector** and the
question is about the collector runtime shape, not a library API.
Concretely:

- Standing up a collector that gathers provider counters into a
  schema and ships them out — and deciding which export backend
  (Prometheus pull, Fluent Bit push, NetFlow, file / IPC) fits
  the downstream consumer.
- Wiring a provider / counter family into the collector and
  confirming the device actually exposes it before the config
  commits (the gate-before-commit rule, shared with
  [`doca-telemetry-utils`](../tools/doca-telemetry-utils/SKILL.md)).
- Turning on / shaping an exporter so the metrics actually leave
  the box, and confirming the downstream consumer receives them
  end-to-end (not just "the daemon is running").
- Diagnosing a collector that starts but produces no schema rows,
  or ships nothing downstream, or whose exporter endpoint is
  silent — walking the layered ladder rather than guessing.
- Recognising when the user is actually asking about the
  productized DTS container (route to public docs, Non-goal #7),
  the reader library (route to `doca-telemetry`), or the
  publisher library (route to `doca-telemetry-exporter`) instead
  of the collection mechanism this skill owns.

Do **not** load this skill for: the hardware-counter reader API
(use [`doca-telemetry`](../libs/doca-telemetry/SKILL.md)); the
publisher API (use
[`doca-telemetry-exporter`](../libs/doca-telemetry-exporter/SKILL.md));
operating the productized DTS container (route via
[`doca-public-knowledge-map`](../doca-public-knowledge-map/SKILL.md)
Non-goal #7); installing DOCA or preparing the env (use
[`doca-setup`](../doca-setup/SKILL.md)); or any hardware-state
change (use
[`doca-hardware-safety`](../doca-hardware-safety/SKILL.md)).

## What this skill provides

This is a **thin loader**. The substantive material lives in two
companion files:

- `CAPABILITIES.md` — the collector deployment contract as a
  class: the four-surface decomposition (clx collection
  mechanism vs reader library vs publisher library vs productized
  DTS), the collection pipeline shape (providers / counters →
  schema → collector daemon → exporters), the export-backend
  surface (Prometheus pull, Fluent Bit push, NetFlow, file /
  IPC) at class level, the version overlay on
  [`doca-version`](../doca-version/SKILL.md), the error taxonomy
  (collector won't start → no provider rows → schema mismatch →
  exporter silent → downstream skew → transport), the
  observability surface, and the safety policy (gate provider
  support before commit; collector is read-only against the
  device; route any mutating step to
  [`doca-hardware-safety`](../doca-hardware-safety/SKILL.md); do
  not invent clx names / paths).
- `TASKS.md` — step-by-step workflows for the deployment verbs:
  `configure`, `build` (routing stub), `modify`, `run`, `test`,
  `debug`, plus a `Deferred task verbs` block that routes
  out-of-scope questions (the two libraries, the productized DTS
  container, env prep, hardware-state change) to their owners.

The skill assumes a host or BlueField where DOCA is already
installed and healthy (per
[`doca-setup ## test`](../doca-setup/TASKS.md#test)) and the
operator can run the collector and reach its exporter sinks. It
does not cover installing DOCA — that path goes through
[`doca-setup`](../doca-setup/SKILL.md) — and it does not cover
the reader / publisher library APIs or the productized DTS
container.

## Loading order

1. Read this `SKILL.md` first to confirm the user's question is
   in scope (operating a CollectX-based collector and its
   exporters — NOT the reader / publisher library APIs, NOT the
   productized DTS container).
2. **For the four-surface decomposition, the collection pipeline
   shape, the export-backend class surface, the version overlay,
   the error taxonomy, the observability surface, and the safety
   policy, see [CAPABILITIES.md](CAPABILITIES.md).**
3. **For step-by-step workflows — `configure`, `build` (routing
   stub), `modify`, `run`, `test`, `debug`, and the
   `Deferred task verbs` block — see [TASKS.md](TASKS.md).**

Both companion files cross-link to each other,
[`doca-version`](../doca-version/SKILL.md) for the canonical
version-handling rules, and
[`doca-public-knowledge-map`](../doca-public-knowledge-map/SKILL.md)
whenever the right answer is "read the live config / public docs"
rather than collector-specific guidance.

## Example questions this skill answers well

See [`references/details.md`](references/details.md#example-questions-this-skill-answers-well).
## What this skill deliberately does not ship

See [`references/details.md`](references/details.md#what-this-skill-deliberately-does-not-ship).
## Related skills

See [`references/details.md`](references/details.md#related-skills).
