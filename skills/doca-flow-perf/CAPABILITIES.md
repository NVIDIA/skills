# DOCA Flow Perf — Capabilities

**Where to start:** `doca_flow_perf` measures the host or
DPU-CPU execution path of a `doca-flow` pipeline. The
pattern overview below names the recurring `doca_flow_perf`-
class questions. Pick the pattern first, then drill into the
H2 that owns the substance. For the *how* of executing each
pattern, jump to [TASKS.md](TASKS.md). For the `doca-flow`
API surface behind the pipeline this tool drives, see
[`doca-flow CAPABILITIES.md`](../../libs/doca-flow/CAPABILITIES.md);
for the DPA-offloaded path measurement, see
[`doca-flow-dpa-perf CAPABILITIES.md`](../doca-flow-dpa-perf/CAPABILITIES.md);
for the optimization-vs-measurement boundary, see
[`doca-flow-tune CAPABILITIES.md`](../doca-flow-tune/CAPABILITIES.md).

This file is loaded by [`SKILL.md`](SKILL.md). It documents
*what `doca_flow_perf` is*, *what it measures and what it
deliberately does not measure*, *how its DPDK and DOCA
backends differ*, *the JSON policy contract surface*, *what
versions it ships in*, *what its layered error and
observability surfaces look like*, and *the safety posture*
the tool's role as a synthetic driver of dataplane state
forces.

## Pattern overview

Five recurring patterns drive every `doca_flow_perf`-class
question, and the H2s below own one each. Pick the pattern,
then jump to the owning H2.

| Pattern | Recognise it when … | Owning H2 |
| --- | --- | --- |
| **Methodology / perimeter** | "What does flow-perf actually measure?" "Does this tell me about dataplane throughput?" | [`## Capabilities and modes`](#capabilities-and-modes) |
| **Backend selection** | "DPDK or DOCA backend?" "Which one matches what NVIDIA published?" | [`## Capabilities and modes`](#capabilities-and-modes) (Backends) |
| **JSON policy design** | "Which `configs/` JSON do I start from?" "How do I model a custom traffic class?" | [`## Capabilities and modes`](#capabilities-and-modes) (JSON policy) |
| **Version compatibility** | "What flow-perf version comes with my DOCA install?" "Am I comparing apples to apples across releases?" | [`## Version compatibility`](#version-compatibility) |
| **Failure / interpretation** | Tool fails to start, reports `num_failed > 0`, or gives a Kops/sec number nothing like the reference. | [`## Error taxonomy`](#error-taxonomy) + [`## Observability`](#observability) |

Two non-patterns the agent must NOT collapse into the above:

- *"I want my deployed Flow application to go faster"* —
  that is optimization, not measurement. Route to
  [`doca-flow-tune`](../doca-flow-tune/SKILL.md). The
  flow-perf tool does not modify the live application; it
  drives a synthetic pipeline.
- *"I want to measure the DPA-offloaded Flow path"* — route
  to [`doca-flow-dpa-perf`](../doca-flow-dpa-perf/SKILL.md).
  flow-perf measures the path that executes on the host or
  DPU-CPU.

## Capabilities and modes

`doca_flow_perf` is a **synthetic driver** of a
JSON-described `doca-flow` pipeline that measures the rule
install / delete / (optional) query rate of the
host or DPU-CPU execution path. It has four orthogonal
decisions the operator commits to in [`## configure`](TASKS.md#configure):

### What is measured

`doca_flow_perf` measures CPU-cycle cost per iteration for
the following per-entry operations across a user-chosen
number of worker threads:

- **insert** (rule add) — the control-plane cost of
  installing a Flow rule that matches a (configurable)
  pattern and applies a (configurable) action;
- **delete** (rule remove) — the control-plane cost of
  removing a previously-installed rule;
- **query** (optional, per-entry, e.g. counter read) — the
  per-entry timed query path the tool exposes when the
  policy enables it.

The tool reports per-iteration cycles, the number of
successful operations (`num_pushed`), the number of failed
operations (`num_failed`), and from those numbers the
operator (or the tool's printed summary) derives the
operations-per-second rate. The exact field names and the
exact stdout format are documented by the public DOCA Flow
Perf guide on `docs.nvidia.com` and by the binary's own
runtime output — this skill does NOT invent additional field
names.

### What is NOT measured

Critical perimeter — the agent must say this explicitly
whenever the user uses the word "throughput" or "latency":

- **Dataplane packet throughput.** flow-perf does not send
  packets through the pipeline it programs and does not
  measure how many bits-per-second the resulting rules
  push. The rules can be hairpin / drop / forward to a
  port, but the tool's measurement is the cost of
  programming them, not the cost of running them.
- **End-to-end packet latency.** Same — flow-perf does not
  inject a packet and time it from RX to TX.
- **Hardware-side throughput at the steering engine.**
  flow-perf's number is a CPU-cycle-based control-plane
  rate; it is not a hardware-side rule-execution rate.
- **An apples-to-apples cross-tool comparison.** A
  flow-perf number is comparable to another flow-perf
  number ONLY when the four-tuple (DOCA version, BlueField
  generation, firmware version, JSON policy + worker /
  queue / burst configuration) is identical. The agent
  refuses to declare two flow-perf numbers comparable when
  the four-tuple differs.

### Backends — DPDK and DOCA, same JSON contract

`doca_flow_perf` ships TWO backends behind the same JSON
policy contract:

- a **DPDK backend** — exercises the Flow rule path via
  `rte_flow`, the DPDK Flow API; and
- a **DOCA backend** — exercises the Flow rule path via
  `doca-flow` directly.

The tool's source tree (per the shipped `src/doca/` and
`src/dpdk/` subtrees) shows both implementations side by
side. The agent must surface that:

- the JSON policy is shared between backends, so the
  *workload description* is identical;
- the *implementation path* and therefore the reported
  cycles are not — the DPDK backend reflects the cost of
  `rte_flow`-based Flow programming; the DOCA backend
  reflects the cost of the native `doca-flow` programming
  path;
- **the operator MUST report which backend they used.**
  Reporting a flow-perf number without identifying the
  backend is meaningless.
- the NVIDIA-published reference numbers on
  `docs.nvidia.com` are for ONE specific backend per
  release; the agent must not assume which one.

### JSON policy — the workload description

The pipeline shape is described in a JSON policy file. The
shipped `configs/` directory contains canned policies as
exemplars; the agent should treat them as the **schema by
example**, not as the schema itself. Recurring axes the
shipped policies vary across:

- **port count and fwd mode** — number of ports (typically
  2 for hairpin policies), and the forwarding action (drop,
  hairpin, encap, shared-RSS, etc.).
- **pipe inventory** — number of pipes, per-pipe priority,
  per-pipe `num_inserted_entries` (the operator's chosen
  rule count), ingress vs. egress.
- **match shape** — the matcher tree (Ethernet, IPv4, UDP /
  TCP, VXLAN, packet meta, etc.) and whether each match
  field is `fixed`, `increase`, or `decrease` per iteration.
- **action shape** — the action list (forward to port, drop,
  hairpin, RSS, encap, decap, modify, count, meter).
- **resource shape** — shared vs. non-shared counters and
  meters, RSS configuration.
- **CT pipes** (when present) — counter-tracking pipe
  configuration (IPv4 / IPv6 session counts, timeout).

Recurring rules for the JSON the agent must surface:

- the agent NEVER invents JSON keys that are not present in
  the shipped `configs/` files; if the operator needs a
  field that no shipped policy demonstrates, the agent
  refuses to invent the spelling and routes to the public
  DOCA Flow Perf guide;
- changing `mode: fixed` to `mode: increase` or
  `mode: decrease` is NOT a free change — it changes the
  per-entry uniqueness pattern and is the most common cause
  of "my new policy gives wildly different numbers than the
  exemplar";
- `num_inserted_entries` directly controls the iteration's
  workload size; varying it across runs to find a stable
  Kops/sec number is part of the methodology, not a
  diagnostic.

### Operating modes

The tool itself supports three operation modes per the
shipped headers (`enum flow_perf_mode` exposes `ADD`, `DEL`,
`QUERY`):

- **insert-only** — measure rule add rate;
- **insert + delete** — measure both, paired;
- **insert + query** — measure add plus per-entry query
  timing.

The exact CLI shape the operator picks each mode by lives in
the binary's `--help` and the public DOCA Flow Perf guide; the
agent must not invent flag spellings.

## Version compatibility

`doca_flow_perf` is **shipped with the DOCA release** and is
versioned *with the surrounding DOCA install*, not on a
separate cadence. Specifically:

- the binary, the `configs/` JSON library, and the
  `doca-flow` library it links against are all part of the
  same DOCA install — they are versioned together;
- a flow-perf number is meaningful ONLY against a specific
  DOCA release, a specific BlueField / ConnectX generation,
  and a specific firmware version; this is the **four-tuple
  capture rule** the agent enforces on every reported
  number;
- the JSON policy files in `configs/` may evolve across
  releases; an operator carrying a custom JSON forward
  across releases must re-validate it against the new
  release's exemplars.

This skill **does NOT** maintain its own version-handling
rules in parallel with
[`doca-version`](../../doca-version/SKILL.md). The agent
treats `doca-version` as the source of truth for:

- the four-way DOCA install match (host package, kernel
  module, firmware, target application's linked
  `doca-flow` version);
- the BlueField-mode / device-mode questions;
- the cross-release behaviour-change questions.

The flow-perf-specific overlay on top of that:

- **Report the backend.** A flow-perf number without a
  named backend (DPDK or DOCA) is not interpretable.
- **Report the JSON policy.** A flow-perf number without
  the exact JSON policy contents is not reproducible.
- **Report the worker / queue / burst / iteration config.**
  Number of workers, queue size, burst size, and iteration
  count are first-class methodology variables; changing any
  of them across runs invalidates the comparison.
- **Report the BlueField mode and firmware version.** Per
  [`doca-version`](../../doca-version/SKILL.md), the device
  mode and firmware are load-bearing.

Concretely the agent applies this rule in
[`TASKS.md ## test`](TASKS.md#test) (capture step) and in
[`TASKS.md ## debug`](TASKS.md#debug) (layer 6 — version).

## Error taxonomy

`doca_flow_perf` failures fall into seven layers. The
[TASKS.md `## debug`](TASKS.md#debug) verb walks them in
order; this section names them so the agent can route fast.

1. **Tool-not-installed.** The binary or the `configs/`
   JSON library is not present on the user's system —
   either DOCA isn't installed, or the user's DOCA install
   profile didn't include the flow-perf component. Route to
   [`doca-setup ## install`](../../doca-setup/TASKS.md#configure)
   and [`## no-install`](../../doca-setup/TASKS.md#no-install).
2. **JSON-policy-malformed.** The policy file doesn't
   parse, or references fields the running binary's version
   does not understand (the canonical
   "I edited the JSON and forgot a comma" / "the field
   moved between releases" failure). The tool reports a
   parse error on stdout / stderr; the agent quotes it
   verbatim and walks the user back to the shipped
   `configs/` exemplars.
3. **Pipeline-creation-failed.** The JSON parsed but the
   underlying `doca-flow` library rejected the pipe
   creation — a matcher or action is unsupported on the
   target device, the BlueField mode is wrong, or a Flow
   precondition is not met. Route the underlying library
   diagnosis to
   [`doca-flow TASKS.md ## debug`](../../libs/doca-flow/TASKS.md#debug).
4. **Insertion-rate-failure.** `num_failed > 0` during the
   measurement loop — some rules failed to install. This is
   a methodology finding, not a tool bug; common causes are
   the policy asking for more entries than the device's
   tables can hold or a duplicate match pattern at the
   chosen `mode`. The agent reports this as a finding, not
   silently.
5. **Iteration-variance-too-high.** The per-iteration
   numbers vary so much that the reported Kops/sec is not
   defensible. This is a methodology finding; route the
   user to increase iterations, increase worker count, or
   stabilize the host (CPU pinning, sibling-CPU isolation,
   interrupt routing). The agent does NOT silently average
   high-variance results.
6. **Version.** Walk
   [`doca-version TASKS.md ## debug`](../../doca-version/TASKS.md#debug)
   end-to-end. Common flow-perf-specific symptoms:
   firmware doesn't expose the matcher / action the JSON
   asks for, BlueField is in the wrong mode for the
   forwarding type, or the operator is comparing a number
   from one DOCA release with a number from another (which
   the four-tuple capture rule prevents).
7. **Cross-cutting.** Hand off to
   [`doca-debug ## debug`](../../doca-debug/SKILL.md) and
   [`doca-setup ## debug`](../../doca-setup/TASKS.md#debug)
   for the env-side layers (driver loaded, PCIe state,
   hugepages, IOMMU, ulimits, kernel module).

The full procedural ladder for each layer lives in
[`TASKS.md ## debug`](TASKS.md#debug); this section names
them so the agent can route on the FIRST symptom.

## Observability

Three kinds of evidence the tool emits, and how the agent is
expected to read each:

- **stdout per-iteration cycles + counts** (the primary
  signal). The tool prints per-iteration CPU cycles and the
  `num_pushed` / `num_failed` / `num_pending` counters. The
  agent treats the per-iteration numbers as the raw signal
  and the printed summary as a derived view; the exact
  format lives in the public guide, NOT in the agent's
  memory.
- **the surrounding `doca-flow` library's own state**
  during the run — pipe creation success / failure, port
  state, counter values. The `doca-flow` library exposes
  this programmatically per
  [`doca-flow CAPABILITIES.md`](../../libs/doca-flow/CAPABILITIES.md);
  flow-perf's measurement is more meaningful when the
  operator cross-checks against the library's own view
  (e.g. via the application's inspector). flow-perf does
  not duplicate the library's state surface.
- **the host-side machine state** the four-tuple capture
  requires: DOCA version, kernel + driver version, firmware
  version, BlueField mode, the JSON policy file as
  shipped / edited, the worker / queue / burst /
  iteration configuration. The agent captures all six
  alongside every reported number; partial captures are not
  defensible.

For the cross-cutting host / device observability surfaces
(driver, firmware, BlueField mode, PCIe), route to
[`doca-debug ## Observability`](../../doca-debug/SKILL.md)
and [`doca-setup TASKS.md ## test`](../../doca-setup/TASKS.md#test).

The skill explicitly does NOT add streaming-telemetry export
on top of flow-perf — the tool is a one-shot synthetic
measurement, not a daemon.

## Safety policy

> **Hardware-safety meta-policy applies.** Every operator
> action below inherits the safety contract defined in
> [`doca-hardware-safety ## Safety policy`](../../doca-hardware-safety/CAPABILITIES.md#safety-policy):
> the pre-flight inventory (DOCA version, firmware, kernel
> module, BlueField mode, OOB access, dataplane co-tenancy),
> the OOB / blast-radius rules, the change-class
> classification, and the smoke-before-bulk discipline.
> This section names ONLY the `doca_flow_perf`-specific
> overlay on top of that meta-policy.

`doca_flow_perf` is at the edge of a live dataplane. Even
though the tool's user-facing operation is *measurement*, it
is implemented by **synthetically driving real Flow rule
installation and deletion** on the device. That has three
operational consequences the agent must surface:

- **flow-perf programs real `doca-flow` rules on the device
  during the measurement.** If another `doca-flow`
  application is running on the same device, the
  measurement competes with it for Flow resources (table
  capacity, counters, meters, shared RSS slots) and can
  disrupt the production workload. The pre-flight
  inventory must confirm whether anything else is using
  Flow on the target device. When in doubt: don't run.
- **flow-perf can install very large numbers of entries.**
  The shipped exemplars routinely set
  `num_inserted_entries` in the millions. On a device
  whose Flow tables are sized for production, that can
  exhaust the steering tables and induce subtle
  rule-install failures on the production application as
  a side-effect. The agent walks the operator to a clean
  device whenever possible.
- **The smoke-before-bulk rule from
  [`doca-hardware-safety ## Safety policy`](../../doca-hardware-safety/CAPABILITIES.md#safety-policy)
  applies twice here.** First on the policy: validate one
  iteration on a small `num_inserted_entries` (e.g. a few
  thousand) before running the full-scale policy. Second on
  the platform: validate the host + device are stable
  before iterating. Skipping either smoke and going
  straight to a large-scale measurement is the canonical
  failure mode for this tool.

In addition:

- **OOB access.** Per the hardware-safety meta-policy: when
  the target is a BlueField, the operator must have OOB
  console / reset access before any large-scale run.
  flow-perf at scale CAN hang a Flow-busy device hard
  enough to require a reset; the OOB precondition is
  non-negotiable.
- **No mutation of the operator's live `doca-flow`
  application.** flow-perf does not patch the operator's
  source; it drives its own synthetic pipeline. If the
  operator wants to change the live application, that is
  [`doca-flow-tune`](../doca-flow-tune/SKILL.md) territory,
  not flow-perf.
- **Capture before retry.** Per
  [`doca-hardware-safety ## Error taxonomy`](../../doca-hardware-safety/CAPABILITIES.md#error-taxonomy):
  if a measurement run fails or returns surprising
  numbers, capture the failed run's full output and the
  four-tuple BEFORE re-running. Retrying without capture is
  the canonical lost-signal failure.

The full procedural application of the safety overlay (when
to abort, when to escalate, what to capture) lives in
[`TASKS.md ## debug`](TASKS.md#debug) and
[`TASKS.md ## test`](TASKS.md#test) plus the
[`doca-debug`](../../doca-debug/SKILL.md) cross-cutting debug
ladder. This section names the rules that constrain those
verbs for flow-perf specifically.
