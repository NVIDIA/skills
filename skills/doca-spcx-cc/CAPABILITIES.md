# DOCA SPCX Congestion-Control Tool — Capabilities

**Where to start:** `doca_spcx_cc` is the operator-side CLI
for the documented SPCX-class Programmable Congestion Control
surface — the next-gen extension on top of the established
[`doca-pcc`](../../libs/doca-pcc/SKILL.md) story. The pattern
overview below names the recurring `doca_spcx_cc`-class
questions. Pick the pattern first, then drill into the H2
that owns the substance. For the *how* of executing each
pattern, jump to [TASKS.md](TASKS.md).

This file is loaded by [`SKILL.md`](SKILL.md). It documents
*what SPCX expresses that PCC does not (and where the
surfaces overlap)*, *the SPCX-vs-PCC-vs-factory-firmware
decision tree*, *the algorithm-authoring vs
algorithm-consumption split*, *the role decision (RP / NP)
and the documented probe-packet format axis*, *the live-link
/ contention precondition that makes any signal meaningful*,
*the runtime observability surface*, *the version overlay*,
*the layered error and observability surfaces*, *and the
heavily-overlaid safety posture* — deploying a wrong CC
algorithm on a production fabric can melt it. For step-by-
step invocations and the prepare → smoke → contention-
positive evaluation workflow, see [`TASKS.md`](TASKS.md).

## Pattern overview

Every `doca_spcx_cc`-class question this skill teaches
resolves into one of SIX patterns. The patterns are CLASSES
— they apply across every SPCX-class algorithm and every
RDMA / RoCE fabric topology the tool can drive, not just
one.

| `doca_spcx_cc` pattern | Class shape | Where the substance lives |
| --- | --- | --- |
| 1. SPCX vs PCC vs factory firmware | Picking the right programmable-CC surface for the user's algorithm + install. SPCX is the **newer, more flexible** extension (potentially preview / limited HW); [`doca-pcc`](../../libs/doca-pcc/SKILL.md) is the **stable, established** surface; factory PCC is the firmware-resident default that needs no host-side code. The agent surfaces all three before committing. | [`## Capabilities and modes`](#capabilities-and-modes) decision-tree table + [TASKS.md ## configure](TASKS.md#configure) step 1 |
| 2. Authoring vs consumption | The DPA-side algorithm body is either user-authored (the user wrote the algorithm and DPACC compiled it) or consumed (the user is deploying a documented shipped reference such as the zero-touch RTT-based algorithm in [`doca-pcc-ztr-rttcc-algo`](../../libs/doca-pcc-ztr-rttcc-algo/SKILL.md)). The tool's harness is the same; the algorithm-side discipline differs. | [`## Capabilities and modes`](#capabilities-and-modes) authoring-vs-consumption table + [TASKS.md ## configure](TASKS.md#configure) step 3 |
| 3. Role + probe packet format | The host-side role (Reaction Point / Notification Point per the public DOCA PCC documentation), the probe-packet format (e.g. CCMAD as the public default; the tool's `--probe-packet-format` axis exposes the documented set), and the threads / cores axis. | [`## Capabilities and modes`](#capabilities-and-modes) role + probe-packet table + [TASKS.md ## configure](TASKS.md#configure) step 4 |
| 4. Live link + contention | The load-bearing safety invariant. A CC algorithm has **no observable signal** without actual contention on the fabric. Evaluating SPCX on an idle link tells you nothing; evaluating on a saturated link is the test. The agent refuses any *"my algorithm is broken because nothing changed"* conclusion drawn from an idle link. | [`## Safety policy`](#safety-policy) live-link rule + [TASKS.md ## test](TASKS.md#test) |
| 5. Diagnose a session failure | Walk the error taxonomy in [`## Error taxonomy`](#error-taxonomy) — install / device-binding / fw-slot / DPA-image / algorithm-precondition / live-link-precondition / runtime / version / cross-cutting — instead of guessing. | [`## Error taxonomy`](#error-taxonomy) + [TASKS.md ## debug](TASKS.md#debug) |
| 6. Roll-forward vs roll-back decision | A CC algorithm that "works" on a replica is not the same as a CC algorithm that is safe on production. The skill's `## use` verb is the documented decision gate: blast-radius bounded, observability gate proven, OOB reachable, factory-PCC rollback rehearsed, escalation path documented. | [`## Safety policy`](#safety-policy) roll-forward gate + [TASKS.md ## use](TASKS.md#use) |

Two cross-cutting rules that apply to *every* pattern above:

- **A CC algorithm has no signal under no contention.** The
  agent's rule for any *"my SPCX algorithm changed nothing"*
  report: the FIRST question is *"was the link actually
  congested when you observed"*, not *"is the algorithm
  broken"*. Idle-link evaluations are non-results.
- **Production CC is high-stakes.** Deploying a wrong or
  unstable CC algorithm on a production RDMA fabric can
  cascade into flow collapse, queue blow-up, or persistent
  link instability. The bundle-wide hardware-safety
  meta-policy in
  [`doca-hardware-safety CAPABILITIES.md ## Safety policy`](../../doca-hardware-safety/CAPABILITIES.md#safety-policy)
  applies heavily here; the per-artifact overlay in
  [`## Safety policy`](#safety-policy) layers SPCX-specific
  rules on top.

## Capabilities and modes

`doca_spcx_cc` is shipped as a single CLI binary under
`/opt/mellanox/doca/tools/` on every DOCA install that
includes the SPCX optional component. It links the host-side
[`doca-pcc`](../../libs/doca-pcc/SKILL.md) library and loads
a DPA-side SPCX algorithm image produced by the DPACC
compiler onto the BlueField's DPA processor. The
interaction model is the
[`doca-pcc`](../../libs/doca-pcc/SKILL.md) two-side-program
model extended with the SPCX-side runtime; the agent does
not separately invent a new lifecycle for SPCX.

**SPCX-vs-PCC-vs-factory-firmware decision tree.** The
load-bearing routing rule before any code:

| Surface | When it is the right choice | Where it lives |
| --- | --- | --- |
| **Factory PCC** (firmware-resident default) | The user wants a documented, supported CC algorithm with no custom code; firmware-level configuration is sufficient; no programmable-CC slot is required. | Firmware configuration via `mlxconfig`-class knobs; route via [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md). |
| **[`doca-pcc`](../../libs/doca-pcc/SKILL.md)** (stable, established) | The user wants a custom CC algorithm on the established PCC surface; the install supports it; the BlueField generation + firmware exposes the custom-PCC slot. | [`doca-pcc`](../../libs/doca-pcc/SKILL.md) host-side lifecycle + DPA-side algorithm via DPACC. |
| **`doca_spcx_cc` / SPCX** (newer, more flexible extension) | The user wants the documented SPCX-class flexibility (the extension surface this tool drives), the install has SPCX exposed, and the user accepts that SPCX is the *newer* of the two surfaces (potentially preview / limited HW on some installs). | This skill (operator side) + the DPA-side SPCX algorithm via DPACC. |

The agent's rule: **probe before committing**. Confirm
which surfaces are exposed on the user's install (via
`doca_pcc_cap_*` + the SPCX availability per the public
guide + `doca_spcx_cc --help` on the installed version);
do not assume SPCX is available just because the user
asked for it.

**Authoring vs consumption table.** The DPA-side algorithm
body comes from one of two sources:

| Source | What the user has | Skill pairing |
| --- | --- | --- |
| User-authored algorithm | A DPA-side translation unit the user wrote and DPACC compiled. | This skill (operator harness) + [`doca-pcc`](../../libs/doca-pcc/SKILL.md) (library) + [`doca-dpa`](../../libs/doca-dpa/SKILL.md) (DPA programming model) + the public DOCA SPCX programming guide. |
| Shipped reference algorithm | A documented NVIDIA-shipped algorithm such as the zero-touch RTT-based CC reference in [`doca-pcc-ztr-rttcc-algo`](../../libs/doca-pcc-ztr-rttcc-algo/SKILL.md). | This skill (operator harness) + [`doca-pcc-ztr-rttcc-algo`](../../libs/doca-pcc-ztr-rttcc-algo/SKILL.md) (algorithm semantics). |

The operator-side flow is the same in both cases; the
algorithm-side discipline (review, parameter tuning,
rollback to the previous algorithm version) differs.

**Role + probe-packet format axis.** SPCX inherits the
PCC two-role model documented in the public DOCA PCC
programming guide:

| Role | What it is | When it is loaded |
| --- | --- | --- |
| Reaction Point (RP) | The role that REACTS to congestion signal by adjusting send rate. On DOCA 3.3 the shipped reference sample `doca_spcx_cc` hard-codes `cfg.role = PCC_ROLE_RP` in its `pcc.c`; the binary does NOT register a runtime `--role` flag. The agent does NOT quote `--role RP` against the binary's `--help`; it loads RP by running the shipped binary as-is. | The standard custom-CC role; loaded on the side of the fabric that is *sourcing* the RDMA / RoCE traffic and that needs to be rate-adjusted. |
| Notification Point (NP) | The role that GENERATES the congestion signal — typically the receive side or a switch ASIC. To run NP with the shipped reference sample, the user either rebuilds the sample with `cfg.role = PCC_ROLE_NP` or follows the public DOCA SPCX-CC guide's documented method for the user's installed version. The agent does NOT invent a runtime `--role` flag that the binary does not register. | When the SPCX algorithm needs a custom NP-side behaviour different from the documented factory behaviour. |

The probe-packet format axis (`--probe-packet-format`)
selects between the documented formats (CCMAD per the
public default; the full set is on the installed
`--help`). The two halves of the deployment must agree on
the probe-packet format.

The thread / core axis follows the PCC convention from
[`doca-pcc`](../../libs/doca-pcc/SKILL.md): host-side
threads pinned to specific cores process the host-side
half of the SPCX session. The shipped sample documents a
default thread count and core list; the agent
re-confirms against `--help` on the installed binary.

**Live-link + contention rule (load-bearing).** SPCX
algorithms are programmable congestion-control algorithms;
they only modulate behaviour when there *is* congestion to
respond to. The evaluation precondition:

- A live RDMA / RoCE link is up between two BlueField
  endpoints (or one BlueField endpoint + a documented
  peer).
- Actual contention exists — concurrent flows sharing
  a bottleneck, queue occupancy non-trivial, ECN /
  RTT signals firing. An idle link with no flows
  shows nothing regardless of how the algorithm is
  written.
- The contention pattern matches the algorithm's
  designed signal — e.g. an RTT-based CC algorithm
  needs RTT variation; a marking-based algorithm needs
  ECN-capable receivers.

The agent's rule for evaluation: refuse to draw a
conclusion from an idle-link or contention-absent run;
the run is uninformative by construction.

**SPCX runtime observability surface.** The host-side
tool prints a status indicator (per the
[`doca-pcc`](../../libs/doca-pcc/SKILL.md) status surface
the agent observed in the shipped sample: `Active`,
`Standby`, `Deactivated`, `Error`) and produces tracer-
style per-port / per-flow runtime captures the user can
correlate against the algorithm's intended behaviour.
The exact metric names + tracer formats are
authoritative on `--help` on the installed binary and the
public DOCA SPCX guide.

**Companion observation surfaces.** Beyond the tool's
own output, the agent typically pairs SPCX evaluation
with:

- [`doca-pcc-counters`](../doca-pcc-counters/SKILL.md) for
  read-only per-port / per-flow PCC counter snapshots
  (the cheaper *"is anything happening on this port"*
  surface).
- [`doca-rdma`](../../libs/doca-rdma/SKILL.md) for the
  RDMA-side observation primitives the algorithm is
  meant to modulate (throughput, latency, completion
  ordering).
- The host-OS RDMA observability surface (counters,
  `perfquery`, etc.) when the cause is below DOCA.

## Version compatibility

For the canonical DOCA version-detection chain, the
four-way match rule, NGC container semantics, and the
headers-win-over-docs rule, see
[`doca-version`](../../doca-version/SKILL.md). The body
lives there; this skill does not duplicate it.

**The `doca_spcx_cc`-specific overlay** is:

- **Tool ↔ library ↔ DPACC ↔ firmware quadruple
  pairing.** The host-side tool, the
  [`doca-pcc`](../../libs/doca-pcc/SKILL.md) library it
  links, the DPACC compiler that built the DPA-side
  algorithm image, and the firmware-level custom-PCC
  slot exposure must agree per the
  [`doca-version CAPABILITIES.md ## Version compatibility`](../../doca-version/CAPABILITIES.md#version-compatibility)
  four-way match rule plus the DPA-specific *DOCA must
  match DPACC* overlay
  [`doca-dpa`](../../libs/doca-dpa/SKILL.md) carries
  plus the firmware-slot precondition
  [`doca-pcc`](../../libs/doca-pcc/SKILL.md) carries.
- **SPCX availability is install-specific.** SPCX is
  the newer of the programmable-CC surfaces; the public
  DOCA SPCX guide documents the BlueField generations
  and DOCA versions on which it is available. The
  agent does NOT assume SPCX is available; the gate is
  `doca_spcx_cc --help` succeeding on the installed
  binary + the public guide naming the user's
  BlueField + DOCA combination.
- **Role / probe-packet format / metric names are
  versioned.** The documented role tokens
  (RP / NP), probe-packet format tokens, and metric
  names evolve across releases; re-verify against the
  installed `--help` and the public DOCA SPCX guide
  rather than against this skill's prose.
- **Public DOCA SPCX page is the source of truth for
  the tool's command-line surface.** When that page
  disagrees with this skill, the live guide wins.

## Error taxonomy

`doca_spcx_cc`'s error surface spans the host install,
the device / firmware-slot binding, the DPA-side
algorithm image, the algorithm's own preconditions, the
live-link / contention precondition, the runtime
session, and the cross-cutting partial-install /
mixed-version layer. The agent distinguishes these
layers in escalating order; jumping layers wastes the
user's time on the wrong fix.

1. **Install.** Binary missing under
   `/opt/mellanox/doca/tools/`; SPCX optional component
   not installed; the host-side
   [`doca-pcc`](../../libs/doca-pcc/SKILL.md) library
   not linked. Routing: route to
   [`doca-setup ## install`](../../doca-setup/TASKS.md#configure)
   and confirm the version per
   [`doca-version TASKS.md ## configure`](../../doca-version/TASKS.md#configure).
2. **Device-binding.** Tool runs but cannot bind the
   target BlueField device. Cause: device PCIe address
   / IB name does not exist; the BlueField is not
   visible to DOCA. Routing: confirm device visibility
   per
   [`doca-pcc TASKS.md ## configure`](../../libs/doca-pcc/TASKS.md#configure)
   step 1.
3. **Firmware custom-PCC slot.** Device bound but the
   firmware-level custom-PCC slot is disabled on the
   running firmware. Cause: the BlueField's firmware
   configuration does not have the custom-PCC slot
   enabled; `mlxconfig`-class configuration is
   required to flip it. Routing: walk
   [`doca-pcc TASKS.md ## configure`](../../libs/doca-pcc/TASKS.md#configure)
   step 1's env-precondition checklist + the
   bundle-wide hardware-safety overlay in
   [`doca-hardware-safety CAPABILITIES.md ## Safety policy`](../../doca-hardware-safety/CAPABILITIES.md#safety-policy)
   for the `mlxconfig`-class change discipline.
4. **DPA-image.** Custom-PCC slot enabled but the
   DPA-side SPCX algorithm image does not load. Cause:
   image was built against a different DPACC version
   than the host-side DOCA; image is corrupted; image
   is the wrong algorithm class (PCC-class loaded into
   the SPCX harness, or vice versa). Routing: confirm
   version pairing per
   [`## Version compatibility`](#version-compatibility);
   re-build the DPA-side image via DPACC if needed.
5. **Algorithm-precondition.** Image loaded but the
   algorithm refuses to start because its own
   preconditions are not met. Cause: the parameters
   passed are outside the algorithm's documented
   ranges; the algorithm requires a probe-packet
   format the install does not expose; the algorithm
   requires a peer NP that is not present. Routing:
   walk the algorithm's own documentation (the public
   DOCA SPCX programming guide for user-authored
   algorithms, or
   [`doca-pcc-ztr-rttcc-algo`](../../libs/doca-pcc-ztr-rttcc-algo/SKILL.md)
   for the shipped reference algorithm).
6. **Live-link / contention.** Algorithm running but
   no observable effect. Cause: the link is idle, the
   contention is below the algorithm's signal floor,
   or the contention pattern does not match what the
   algorithm is designed to react to. Routing: this is
   NOT a tool / algorithm failure — re-walk the
   live-link rule in
   [`## Capabilities and modes`](#capabilities-and-modes)
   and the evaluation flow in
   [TASKS.md ## test](TASKS.md#test); generate the
   appropriate contention before drawing a
   conclusion.
7. **Runtime.** Algorithm running on a live, congested
   link but the captured behaviour is wrong (rate
   collapse, oscillation, persistent under-utilisation,
   queue blow-up). Cause: algorithm bug, parameter
   mis-tuning, or the algorithm is correct but mis-
   matched to the fabric topology. Routing: do NOT
   roll forward; capture the runtime evidence and walk
   [`doca-pcc TASKS.md ## debug`](../../libs/doca-pcc/TASKS.md#debug);
   the rollback path
   ([`## Safety policy`](#safety-policy)) is the
   immediate operational response.
8. **Version.** Cross-cutting partial-install /
   mixed-version per
   [`doca-version CAPABILITIES.md ## Version compatibility`](../../doca-version/CAPABILITIES.md#version-compatibility).
   Routing: walk
   [`doca-version TASKS.md ## debug`](../../doca-version/TASKS.md#debug)
   before any further investigation.
9. **Cross-cutting.** Cause is below DOCA — driver,
   firmware, BlueField mode, NUMA, host kernel.
   Routing: hand off to
   [`doca-debug ## debug`](../../doca-debug/SKILL.md)
   and
   [`doca-setup ## debug`](../../doca-setup/TASKS.md#debug).

For the cross-library `DOCA_ERROR_*` taxonomy at the
program-side level (the host-side
[`doca-pcc`](../../libs/doca-pcc/SKILL.md) calls the SPCX
tool internally drives), see
[`doca-programming-guide CAPABILITIES.md ## Error taxonomy`](../../doca-programming-guide/CAPABILITIES.md#error-taxonomy).

## Observability

`doca_spcx_cc`'s observability surface is **the runtime
session state + the captured per-port / per-flow trace
behaviour + the host-side status surface**, paired with
read-only counter inspection via
[`doca-pcc-counters`](../doca-pcc-counters/SKILL.md) and
RDMA-side observation via
[`doca-rdma`](../../libs/doca-rdma/SKILL.md).

- **Host-side status indicator.** The host-side tool
  prints a status indicator following the
  [`doca-pcc`](../../libs/doca-pcc/SKILL.md) process
  state set: `Active`, `Standby`, `Deactivated`,
  `Error`. The agent treats `Active` as *"loaded and
  exercising the link"* and `Standby` / `Deactivated`
  / `Error` as states that need to be acknowledged
  before any evaluation conclusion.
- **Per-port / per-flow runtime captures.** The
  documented tracer-style runtime output the tool
  produces — re-verify field names + format against
  `--help` on the installed binary and the public
  DOCA SPCX page.
- **Companion PCC counter snapshots.** Per
  [`doca-pcc-counters`](../doca-pcc-counters/SKILL.md)
  — the cheaper *"is anything happening on this port"*
  surface. Useful as a sanity check before and during
  an SPCX evaluation.
- **RDMA-side metrics.** Per
  [`doca-rdma`](../../libs/doca-rdma/SKILL.md) — the
  throughput / latency / completion-ordering surface
  the algorithm is meant to modulate.
- **Captured-evidence tuple.** The minimum metadata to
  attach to any evaluation artifact: (DOCA version,
  DPACC compiler version, BlueField identity +
  firmware, algorithm name + version + parameter set,
  probe-packet format, role assignment per endpoint,
  fabric topology + contention shape, capture
  window). Without all of these the evidence is not
  comparable to the next run.

For the cross-cutting env-side observability primitives
(PCIe scans, `devlink`, `mlxconfig`), see
[`doca-setup CAPABILITIES.md ## Observability`](../../doca-setup/CAPABILITIES.md#observability).
For the program-side observability surface (DOCA log
levels, `DOCA_LOG_LEVEL`, `--sdk-log-level`), see
[`doca-programming-guide CAPABILITIES.md ## Observability`](../../doca-programming-guide/CAPABILITIES.md#observability).

## Safety policy

> **Overlay on the bundle-wide hardware-safety meta-policy.** The rules below are this skill's per-artifact overlay on the cross-cutting rules in [`doca-hardware-safety` CAPABILITIES.md ## Safety policy](../../doca-hardware-safety/CAPABILITIES.md#safety-policy) (specifically [### Per-artifact overlay pattern](../../doca-hardware-safety/CAPABILITIES.md#per-artifact-overlay-pattern)). When the two layers disagree, the stricter wins; when either layer says STOP, the agent stops.

`doca_spcx_cc` is a **high-stakes** tool: it loads a
custom congestion-control algorithm onto a BlueField
that participates directly in an RDMA / RoCE fabric's
rate-control loop. A wrong or unstable algorithm on
production can cascade into flow collapse, queue blow-
up, or persistent link instability — the kind of
failure mode that takes a fabric down and is expensive
to recover. The rules:

- **Replica-first is mandatory, not optional.** The
  meta-policy's replica-first rule applies here
  without modification: every SPCX algorithm is first
  exercised on a non-prod replica that matches
  production on BlueField generation, firmware level,
  host kernel, and fabric topology class, with a
  contention pattern that exercises the algorithm's
  design. Skipping the replica because the algorithm
  *"looks small"* is the canonical *"small CC change
  becomes a fabric outage"* failure.
- **No contention → no signal → no conclusion.** A
  CC algorithm has no observable behaviour without
  actual contention on the fabric. The agent refuses
  to declare an algorithm safe based on an idle-link
  or contention-absent run; the run is uninformative
  by construction.
- **Bounded blast radius before any production
  cutover.** The agent does not recommend loading an
  SPCX algorithm fleet-wide at first cutover. The
  rollout is incremental — one BlueField pair, then
  a small bounded set, then progressively larger
  bounded sets — with observability + rollback
  rehearsed at each step.
- **Factory-PCC rollback is the always-available
  escape hatch.** Every SPCX deployment plan names
  the rollback path as *"revert to the factory PCC
  algorithm in firmware"* (or to the previous
  programmable-CC algorithm if one was already in
  place). The rollback is documented and rehearsed
  on the replica before production is exposed. A
  plan without a documented rollback is refused per
  [`doca-hardware-safety CAPABILITIES.md ## Safety policy`](../../doca-hardware-safety/CAPABILITIES.md#safety-policy)
  no-rollback rule.
- **Observability-before-workload.** The captured
  per-port / per-flow tracer output and the companion
  PCC counter snapshots must be readable end-to-end
  *before* the algorithm is asked to handle production
  traffic. An algorithm running without observability
  is an algorithm whose next failure has no
  diagnostic signal to work from.
- **OOB management path required for production
  rollout.** If the cutover plan touches a BlueField
  that the operator is managing over the same RDMA
  link the SPCX algorithm controls, an out-of-band
  management path is mandatory per
  [`doca-hardware-safety CAPABILITIES.md ## Safety policy`](../../doca-hardware-safety/CAPABILITIES.md#safety-policy)
  OOB rule.
- **Maintenance-window discipline.** Production
  cutovers run inside an explicit, time-boxed
  maintenance window with operations notified, per
  the meta-policy. CC algorithm cutovers are NOT
  business-hours operations.
- **Do not invent algorithm parameters, role tokens,
  probe-packet format tokens, or metric names.** The
  documented surface lives on the public DOCA SPCX
  page and in `--help` on the installed binary;
  prose-derived names are the most common
  hallucination failure for this skill, and a wrong
  parameter on a production fabric is exactly the
  kind of failure the agent exists to prevent.
- **Refuse the cutover when any gate fails.** The
  agent surfaces failed gates as blocking and does
  NOT find a creative workaround. The escalation
  path is the operator's change-control process per
  [`doca-hardware-safety CAPABILITIES.md ## Safety policy`](../../doca-hardware-safety/CAPABILITIES.md#safety-policy)
  no-documented-rollback → refuse-and-escalate rule.

## Public-source pointer

The canonical public source for `doca_spcx_cc` is the
**DOCA SPCX** page on `docs.nvidia.com`, reachable
through
[`doca-public-knowledge-map ## DOCA tools`](../../doca-public-knowledge-map/SKILL.md#doca-tools).
The companion programming-model surface lives in the
public DOCA PCC, DOCA DPA, and DPACC guides on the
same site; the shipped reference zero-touch RTT-based
algorithm lives in
[`doca-pcc-ztr-rttcc-algo`](../../libs/doca-pcc-ztr-rttcc-algo/SKILL.md).
Do not invent flags, role tokens, probe-packet format
tokens, or metric names beyond what those public sources
document, and re-verify against `--help` on the
installed binary.
