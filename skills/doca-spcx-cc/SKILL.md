---
license: Apache-2.0
name: doca-spcx-cc
description: >
  Use this skill when the user is invoking `doca_spcx_cc` (the
  host-side CLI under /opt/mellanox/doca/tools/) to load,
  parameterize, start, observe, or stop a Programmable Congestion
  Control (SPCX) algorithm on a BlueField with a DPA processor
  against a live RDMA / RoCE fabric, or picking SPCX vs the
  established `doca-pcc` surface. Trigger even when the user does
  not say "DOCA SPCX" or "doca_spcx_cc" —
  typical implicit phrasings include "I want to write a custom
  RTT-based CC algorithm for my RoCE fabric", "my SPCX session
  loaded but throughput / latency didn't change", "doca_pcc status
  shows Active but factory CC seems to still be in charge",
  "DOCA_PCC_PS_ERROR on start", "is the programmable-CC surface
  available on my install", or "DPA-side algorithm image won't
  load". Refuse and route elsewhere for DPA-side algorithm
  authoring detail, factory PCC firmware configuration, read-only
  PCC counter inspection, raw DPA cycle profiling, RDMA library
  programming, or general DOCA install — those belong to other
  skills.
metadata:
  kind: tool
compatibility: >
  Requires DOCA SDK installed at /opt/mellanox/doca on Linux
  (Ubuntu 22.04/24.04 or RHEL/SLES) with the SPCX optional
  component, a BlueField exposing its DPA processor with the
  firmware custom-PCC slot enabled, the DPACC compiler installed
  and version-matched, and a non-prod RDMA / RoCE fabric with
  controllable contention reachable for evaluation. Probes via
  `pkg-config doca-pcc` and `doca_spcx_cc --help`.
---

# DOCA SPCX Congestion-Control Tool

**Where to start:** This is a tool skill for invoking
`doca_spcx_cc` — the documented host-side CLI that exercises
an SPCX-class Programmable Congestion Control algorithm on a
live RDMA / RoCE fabric driven by a BlueField with a DPA
processor. Open [`TASKS.md`](TASKS.md) and start at
[`## configure`](TASKS.md#configure) for the
SPCX-vs-PCC-vs-factory-firmware decision tree (load-bearing
gate before any code), the role decision (RP / NP), the
DPA-side algorithm authoring vs consumption split, and the
live-link / contention precondition. Open
[`CAPABILITIES.md`](CAPABILITIES.md) when the question is
*what does SPCX let me express that `doca-pcc` does not*,
*what is the SPCX-vs-PCC tradeoff*, *what runtime metrics
does the tool surface*, or *what is the safety posture for
loading a custom CC algorithm on a production fabric*. If
DOCA is not installed yet, route to
[`doca-setup`](../../doca-setup/SKILL.md) first.

This skill is the **next-gen programmable-CC surface**.
[`doca-pcc`](../../libs/doca-pcc/SKILL.md) is the established
PCC story; SPCX is the documented extension that authors
SPCX-class algorithms on the same DPA hardware substrate.
[`doca-pcc-ztr-rttcc-algo`](../../libs/doca-pcc-ztr-rttcc-algo/SKILL.md)
is one shipped reference algorithm (zero-touch RTT-based
CC) that can be loaded through either the PCC or SPCX path
depending on the install and the user's algorithm choice;
the agent surfaces this decision tree explicitly per
[`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes).

## Example questions this skill answers well

The CLASSES of `doca_spcx_cc` questions this skill is built
to answer, each with one worked example. The class is the
load-bearing piece; the worked example is one instance.

- **"Should I use SPCX or `doca-pcc` for my custom CC
  algorithm?"** — worked example: *"I want to write a new
  RTT-based congestion-control algorithm for my RoCE
  fabric — which surface do I target?"*. Answered by the
  SPCX-vs-PCC-vs-factory-firmware decision tree in
  [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes)
  + the
  *"is SPCX the right surface on this install?"* gate in
  [`TASKS.md ## configure`](TASKS.md#configure) step 1.
- **"How do I evaluate my authored SPCX algorithm on a
  real RDMA link before letting it touch production?"** —
  worked example: *"I have a DPACC-compiled SPCX algorithm
  and a non-prod BlueField pair; how do I run a
  contention-positive evaluation?"*. Answered by the
  authoring vs consumption split + the live-link
  precondition in
  [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes)
  + the replica-first evaluation flow in
  [`TASKS.md ## test`](TASKS.md#test).
- **"My SPCX algorithm appears to load cleanly but the
  link's throughput / latency curve is unchanged — what's
  going on?"** — worked example: *"the host-side `doca_pcc
  --status` reports `Active` and a stable session, but my
  RoCE flows look like the factory algorithm is still in
  charge"*. Answered by the live-link / contention rule
  in
  [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes)
  + the
  *"algorithm has no signal under no contention"*
  guidance in
  [`TASKS.md ## test`](TASKS.md#test) +
  [`TASKS.md ## debug`](TASKS.md#debug) (route through
  the layered error taxonomy before blaming the
  algorithm).
- **"My SPCX algorithm passed replica testing — what's
  the gate before I roll it forward to production?"** —
  worked example: *"my CC algorithm works on the two
  BlueField pairs in the lab; can I push it to the
  fleet?"*. Answered by the safety overlay in
  [`CAPABILITIES.md ## Safety policy`](CAPABILITIES.md#safety-policy)
  (heavily cross-linked to
  [`doca-hardware-safety CAPABILITIES.md ## Safety policy`](../../doca-hardware-safety/CAPABILITIES.md#safety-policy)):
  blast-radius bounded, observability gate proven, OOB
  reachable, factory-PCC rollback rehearsed, escalation
  path documented before any production cutover.
- **"Is `doca_spcx_cc` on my install, and is it paired
  with the matching `doca-pcc` library and DPACC compiler
  version?"** — worked example: *"is the SPCX surface
  available on my DOCA install?"*. Answered by the
  version-overlay in
  [`CAPABILITIES.md ## Version compatibility`](CAPABILITIES.md#version-compatibility),
  which redirects to the canonical
  [`doca-version`](../../doca-version/SKILL.md) chain
  and adds the *tool ↔ `doca-pcc` library ↔ DPACC
  compiler ↔ firmware custom-PCC slot* match rule.
- **"My SPCX session errored — is it the tool, the
  algorithm, the device, or the firmware?"** — worked
  example: *"`doca_spcx_cc` exits with
  `DOCA_PCC_PS_ERROR` on start"*. Answered by the
  layered error taxonomy in
  [`CAPABILITIES.md ## Error taxonomy`](CAPABILITIES.md#error-taxonomy)
  + the layered walk in
  [`TASKS.md ## debug`](TASKS.md#debug).

## Audience

This skill serves **external developers, platform
operators, and AI agents authoring, loading, and
evaluating an SPCX-class Programmable Congestion Control
algorithm on a BlueField with a DPA processor, against a
live RDMA / RoCE fabric**. Concretely:

- A CC researcher / developer who has authored an
  SPCX-class DPA-side algorithm (or who is consuming a
  documented shipped reference such as the zero-touch
  RTT-based algorithm via
  [`doca-pcc-ztr-rttcc-algo`](../../libs/doca-pcc-ztr-rttcc-algo/SKILL.md)
  on the SPCX path when the install + algorithm support
  it) and needs the operator-side harness to load,
  parameterize, start, observe, and stop the algorithm.
- A platform operator running a programmable-CC pilot
  on a non-prod RDMA fabric to characterise the
  algorithm's behaviour under controlled contention.
- An AI agent producing a *"is this SPCX algorithm safe
  to roll forward"* answer honestly — with evidence
  from a contention-positive evaluation, a documented
  rollback to the factory PCC, and an explicit blast-
  radius bound — instead of a guess from datasheet
  prose.

It is **not** for users debugging the `doca_spcx_cc`
binary itself, **not** a substitute for the live public
DOCA SPCX / DOCA PCC programming guides, **not** the
right place for the DPA-side algorithm authoring detail
(that path goes through the public DOCA SPCX programming
guide and the
[`doca-pcc`](../../libs/doca-pcc/SKILL.md) +
[`doca-dpa`](../../libs/doca-dpa/SKILL.md) skills), and
**not** the right place for default factory PCC
configuration (no host-side library or SPCX tool needed;
route via
[`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md)).

The tool is shipped as a **CLI binary** under
`/opt/mellanox/doca/tools/`, not a library you link
against. The skill uses the same `kind: tool`
three-file shape as the rest of the bundle so the
agent's task-verb contract is uniform across libraries,
services, and tools.

## Language scope

`doca_spcx_cc` is a C host-side CLI that links the
host-side [`doca-pcc`](../../libs/doca-pcc/SKILL.md)
library and loads a DPA-side SPCX algorithm image built
by the DPACC compiler. The algorithm body is a separate
DPA-side translation unit written in the language DPACC
accepts. The skill keeps workflow guidance
language-neutral and routes per-language questions to
the public DOCA SPCX / DOCA PCC / DPACC guides via
[`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md).

## When to load this skill

Load this skill when the user is — or the agent needs to
— invoke `doca_spcx_cc` on a real host with DOCA
installed, paired with a BlueField that has its DPA
processor exposed AND the firmware custom-PCC slot
enabled, against a port carrying RDMA / RoCE traffic with
**actual contention**. Concretely:

- Loading an authored SPCX algorithm (or a documented
  shipped reference) onto the BlueField via the SPCX
  surface, parameterizing it, starting it, and observing
  its effect on the live link.
- Evaluating an SPCX algorithm on a non-prod replica
  before any production rollout — capturing the
  contention-positive run as evidence.
- Comparing SPCX vs `doca-pcc` paths for the same
  algorithm class on the same install (where both paths
  are available) and deciding which to commit to.
- Producing a *"safe to roll forward"* recommendation
  with the documented evidence + rollback plan, or
  refusing the recommendation when the evidence /
  rollback is missing per
  [`CAPABILITIES.md ## Safety policy`](CAPABILITIES.md#safety-policy).
- Diagnosing a `DOCA_PCC_PS_ERROR` or a silent-no-effect
  symptom against the layered error taxonomy.

Do **not** load this skill for general DOCA orientation,
DPA-side algorithm authoring detail, raw cycle profiling
of the DPA, the factory PCC algorithm shipped in the
firmware, or DOCA install. For those, route to
[`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md),
[`doca-pcc`](../../libs/doca-pcc/SKILL.md),
[`doca-dpa`](../../libs/doca-dpa/SKILL.md), or
[`doca-setup`](../../doca-setup/SKILL.md).

## What this skill provides

This is a **thin loader**. Substantive material lives
in two companion files:

- `CAPABILITIES.md` — what SPCX expresses that PCC does
  not (and where the surfaces overlap), the
  SPCX-vs-PCC-vs-factory-firmware decision tree, the
  authoring vs consumption split, the role decision
  (RP / NP), the documented probe-packet format axis,
  the live-link / contention precondition rule (the
  load-bearing safety invariant — *"a CC algorithm has
  no signal under no contention"*), the runtime
  observability surface (per-port / per-flow trace
  formats + the host-side status surface), the
  triple-axis precondition rule
  ([`doca-pcc`](../../libs/doca-pcc/SKILL.md): DPA-capable
  BlueField + firmware custom-PCC slot enabled +
  `doca_pcc_cap_*` agreement) extended with SPCX-specific
  availability, the version overlay (tool ↔ library ↔
  DPACC ↔ firmware), the layered error taxonomy
  (install / device-binding / fw-slot / DPA-image /
  algorithm-precondition / live-link-precondition /
  runtime / version / cross-cutting), the observability
  surface, and the heavily-overlaid safety posture (a
  wrong CC algorithm on production can melt the
  fabric).
- `TASKS.md` — step-by-step workflows for the in-scope
  task verbs: `install` (host-side DOCA install + DPA
  prerequisites + firmware custom-PCC slot), `configure`
  (SPCX-vs-PCC decision + role + algorithm + parameters
  + probe-packet format), `build` (route to install —
  the host-side tool is shipped; the DPA-side algorithm
  is user-built by DPACC), `modify` (refuse — do not
  patch the binary; modify the invocation, algorithm,
  and parameters), `run` (the
  prepare → smoke → contention-positive evaluation
  flow), `test` (iterative loop on the replica before
  production), `debug` (walk the error taxonomy),
  `use` (the *"safe to roll forward"* decision with
  evidence + rollback + escalation), plus a `Deferred
  task verbs` block.

The skill assumes a host where DOCA is already
installed, a BlueField with a DPA processor and the
firmware-level custom-PCC slot enabled is present and
visible, the DPACC compiler is installed at a version
matched to the host-side DOCA, the user already knows
how (at sketch level) to write the DPA-side SPCX
algorithm (or has a shipped reference algorithm to
consume), and a non-prod RDMA / RoCE fabric with
controllable contention is available for evaluation.

## What this skill deliberately does not ship

This skill is **agent guidance**, not a samples or
scripts bundle. To keep the boundary clean, it
deliberately does not contain — and pull requests
should not add:

- **A specific congestion-control algorithm.** SPCX
  *loads* an algorithm the user supplies; the skill
  refuses to invent algorithm bodies and routes any
  *"what algorithm should I write"* question to the
  public DOCA SPCX / DOCA PCC programming guides and
  to the user's own domain expertise.
- **Pre-baked example output** (throughput / latency
  curves, per-flow counter snapshots). Output is
  device-, firmware-, fabric-topology-, and
  workload-specific; pinning one would mislead
  operators elsewhere.
- **Specific flag strings, subcommand names,
  probe-packet format tokens, or metric names beyond
  what the public DOCA SPCX page and `--help`
  document.** The SPCX surface is the newer of the
  programmable-CC surfaces and the documented flag
  set evolves; the installed `--help` is the
  authoritative inventory.
- **Wrappers, parsers, or scripts** in any language
  that consume the tool's output. The output format
  is documented; users who want to script against it
  should read the live guide and write the parser
  against their installed version.
- **A specific tuning recommendation derived from a
  single observation.** CC tuning on a live fabric
  is high-stakes; the skill prescribes how to
  *capture evidence and compare against the factory
  PCC baseline* and refuses to translate a single
  observation into a parameter-change recommendation
  without the user's own domain analysis.
- **A `samples/` or `reference/` subtree.** This is
  a thin loader for a shipped CLI; substantive
  material lives on the public page, in `--help`,
  and in
  [`doca-pcc`](../../libs/doca-pcc/SKILL.md) +
  [`doca-dpa`](../../libs/doca-dpa/SKILL.md).

## Loading order

1. Read this `SKILL.md` first to confirm the user's
   question is in scope (SPCX-side custom CC work,
   not factory firmware PCC, not raw DPA cycle
   profiling, and not algorithm authoring detail).
2. **For the SPCX-vs-PCC decision tree, the authoring
   vs consumption split, the live-link precondition,
   the version overlay, the error taxonomy, the
   observability surface, and the safety posture,
   see [CAPABILITIES.md](CAPABILITIES.md).**
3. **For the documented invocations and the
   prepare → smoke → contention-positive evaluation
   workflow — `install`, `configure`, `build`,
   `modify`, `run`, `test`, `debug`, `use` — see
   [TASKS.md](TASKS.md).**

## Related skills

- [`doca-pcc`](../../libs/doca-pcc/SKILL.md) — the
  established host-side library for Programmable
  Congestion Control. The SPCX tool builds on and
  links this library; the SPCX-vs-PCC decision tree
  in
  [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes)
  is the load-bearing routing rule. Conflating the
  two is the most common programmable-CC first-touch
  error.
- [`doca-pcc-ztr-rttcc-algo`](../../libs/doca-pcc-ztr-rttcc-algo/SKILL.md)
  — the shipped reference zero-touch RTT-based CC
  algorithm. When the user wants to deploy a
  documented reference algorithm via the SPCX path
  (rather than author one), this is the
  algorithm-side skill paired with this tool's
  operator-side workflow.
- [`doca-dpa`](../../libs/doca-dpa/SKILL.md) — the
  host-side DPA control library the SPCX algorithm's
  DPA-side body builds on. For DPA-level questions
  (kernel-launch model, DPACC build flags, DPA-side
  comms / verbs), this is the skill the agent loads
  alongside.
- [`doca-rdma`](../../libs/doca-rdma/SKILL.md) — the
  library whose RDMA / RoCE flows on the attached
  BlueField port the SPCX algorithm is controlling.
  Without RDMA traffic in flight and contention on
  the fabric, the algorithm has no signal — surface
  this precondition with the user before any
  evaluation.
- [`doca-pcc-counters`](../doca-pcc-counters/SKILL.md) —
  the sibling tool for read-only PCC counter
  inspection. SPCX exposes its own runtime
  observability surface; the PCC counter tool is the
  cheaper *"is anything happening on this port"*
  first step before / during an SPCX evaluation.
- [`doca-hardware-safety`](../../doca-hardware-safety/SKILL.md)
  — the bundle-wide hardware-safety meta-policy
  (pre-flight inventory, OOB requirement,
  replica-first, observability-before-workload,
  rollback discipline, escalation). The `## Safety
  policy` overlay in this skill is heavily layered
  on the meta-policy; deploying a wrong CC algorithm
  on a production fabric is a *meta-policy STOP*
  case.
- [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md)
  — routing to the public DOCA SPCX programming
  guide, the public DOCA PCC programming guide, the
  public DOCA DPA / DPACC guides, and the rest of
  the public DOCA documentation set.
- [`doca-version`](../../doca-version/SKILL.md) —
  canonical DOCA version-handling rules. The
  `## Version compatibility` section in
  [`CAPABILITIES.md`](CAPABILITIES.md) is a concise
  overlay that redirects here for the body and adds
  the *tool ↔ `doca-pcc` library ↔ DPACC ↔ firmware
  custom-PCC slot* matching rule.
- [`doca-setup`](../../doca-setup/SKILL.md) — env
  preparation, install verification, DPACC compiler
  install / verification, BlueField firmware
  configuration (custom-PCC slot enablement is a
  firmware-level setting), and the *I have no
  install yet* path with the public NGC DOCA
  container.
- [`doca-debug`](../../doca-debug/SKILL.md) — the
  cross-cutting debug ladder. SPCX-specific debug
  layers on top of that ladder; cross-link the
  captured runtime evidence + counter snapshots
  into a `doca-debug` session when the cause is
  below DOCA.
- [`doca-structured-tools-contract`](../../doca-structured-tools-contract/SKILL.md)
  — the bundle's detect → prefer → fall back →
  report contract for structured helper tools. The
  command appendix in [`TASKS.md`](TASKS.md)
  honors this contract.
- [`doca-programming-guide`](../../doca-programming-guide/SKILL.md)
  — general DOCA programming patterns shared by
  every library / tool surface, including the
  cross-library `DOCA_ERROR_*` taxonomy this
  tool's host-side error layer overlays on top of.

The default factory PCC algorithms shipped inside
ConnectX firmware are **not in scope** for this skill —
those work without `doca_spcx_cc` and are configured
through firmware-level knobs, not through any host-side
library or tool API. Route via
[`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md).
Conflating the factory PCC story with SPCX is the
single most common programmable-CC first-touch error.
