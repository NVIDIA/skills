---
license: Apache-2.0
name: doca-flow-perf
description: >
  Use this skill when the user is measuring the host or DPU-CPU
  control-plane rate of a DOCA Flow pipeline with doca_flow_perf —
  picking a JSON policy from configs/, choosing the DPDK or DOCA
  backend, running the single-iteration smoke then the iterative eval
  loop, interpreting per-iteration CPU cycles and num_pushed /
  num_failed, or capturing the four-tuple (DOCA version,
  BlueField/firmware, JSON policy, worker/queue/burst config) that
  makes a Kops/sec number defensible. Trigger even when the user does
  not explicitly mention "doca-flow-perf" — typical implicit phrasings
  include "how many rules per second can my BlueField insert",
  "5-tuple hairpin rule rate", "Kops/sec for steering", "flow-perf
  number does not match release notes", "DPDK vs DOCA benchmark", or
  "rule-install variance too high". Refuse and route elsewhere for
  optimizing a live Flow app (doca-flow-tune), the DPA-offloaded path
  (doca-flow-dpa-perf), dataplane throughput or latency, or
  library-internal pipe semantics — those belong to other skills.
metadata:
  kind: tool
compatibility: >
  Requires DOCA SDK installed at /opt/mellanox/doca on Linux (Ubuntu
  22.04/24.04 or RHEL/SLES) with a BlueField DPU or ConnectX NIC
  attached. The doca_flow_perf binary plus its configs/ JSON exemplars
  must be present (the DOCA Flow Perf install component), with the
  underlying doca-flow library healthy. Reads `pkg-config doca-flow`
  and inspects /opt/mellanox/doca/{lib,include,samples,applications}.
---

# DOCA Flow Perf (`doca_flow_perf`)

**Where to start:** This is a tool skill for invoking
`doca_flow_perf`, the host-side / DPU-CPU-side DOCA Flow
performance measurement tool. Open [`TASKS.md`](TASKS.md) and
start at [`## configure`](TASKS.md#configure) to commit to
the three-axis decision (target Flow pipeline shape × traffic
class × measurement axis) and pick the JSON policy file that
expresses the workload, then [`## run`](TASKS.md#run) for the
single-iteration smoke, then [`## test`](TASKS.md#test) for
the iterative eval loop that produces a defensible
Kops/sec-class number. Open [`CAPABILITIES.md`](CAPABILITIES.md)
when the question is *what `doca_flow_perf` measures and what
it deliberately does not measure*, *how its DPDK and DOCA
backends differ behind the same JSON contract*, *how to
interpret the per-iteration CPU-cycle output*, or *how it
differs from `doca-flow-tune` (measurement vs. optimization)
and `doca-flow-dpa-perf` (host / DPU-CPU vs. DPA-offloaded
path)*. If DOCA is not installed, route to
[`doca-setup`](../../doca-setup/SKILL.md) first; if the
target measurement is the DPA-offloaded path, route to
[`doca-flow-dpa-perf`](../doca-flow-dpa-perf/SKILL.md)
instead; if the goal is to optimize an already-deployed Flow
pipeline rather than measure a synthetic one, route to
[`doca-flow-tune`](../doca-flow-tune/SKILL.md) — `flow-perf`
is a synthetic-driver microbenchmark, not a tuner of a live
Flow application.

## Example questions this skill answers well

- *"I want a defensible host-side baseline number for how
  many `doca-flow` rules per second a single BlueField-3 can
  insert for a 5-tuple match-and-hairpin workload. Which
  policy JSON do I start from, how do I make the result
  reproducible, and what do I have to capture alongside the
  number for it to be defensible?"* — class-shaped flow-perf
  baseline question; the agent walks the `configs/` library,
  the JSON contract, and the four-tuple capture rule.
- *"What is the difference between `doca-flow-perf`,
  `doca-flow-dpa-perf`, and `doca-flow-tune`? They all
  mention `doca-flow` and `perf` in their names — when do I
  reach for each?"* — measurement-vs-optimization plus
  host-vs-DPA-path; the agent surfaces the boundaries.
- *"My policy JSON looks like the example, but the reported
  Kops/sec is dramatically lower than the published numbers I
  see in NVIDIA's release notes. What variables do I have to
  control before I can trust the comparison?"* — methodology
  question; the agent walks the controllable axes (number of
  workers, queue depth, burst size, fixed-vs-incremented match
  fields, DPDK vs DOCA backend, BlueField mode, driver /
  firmware).
- *"I have a workload that does not match any of the shipped
  policy JSONs in `configs/`. How do I author a new policy
  JSON, what is the JSON schema in broad strokes, and what
  changes when I switch a match field from `mode: fixed` to
  `mode: increase`?"* — JSON authoring question; the agent
  walks the shipped configs as exemplars and refuses to
  invent schema fields not present in the source tree.
- *"What does the tool actually NOT measure? I am trying to
  understand whether a flow-perf number tells me anything
  about end-to-end traffic latency or just about the
  rule-programming control-plane rate."* — methodology
  perimeter question; the agent draws a hard line: this tool
  measures rule install / delete (control-plane) rate plus
  optional query rate, NOT dataplane latency, NOT dataplane
  throughput, NOT end-to-end application performance.
- *"I see two backends — DPDK and DOCA — behind the same
  JSON. When do I pick which, and what does the choice mean
  for the result I report?"* — backend choice question; the
  agent walks the DPDK-backend vs. DOCA-backend trade-off and
  insists the operator REPORT which one they used.

## Audience

Experienced AI agents and platform / network engineers who
are comfortable with the `doca-flow` programming model and
the DPDK control-plane, who want a *defensible* number for
the host-side / DPU-CPU-side Flow rule-install / rule-delete
rate. Readers are expected to know that the published
numbers in NVIDIA release notes are run with very specific
preconditions (specific DOCA version, specific BlueField
firmware, specific traffic class) and that any number they
produce locally must explicitly state those preconditions.

This skill is NOT for:

- operators who want to optimize an already-deployed
  `doca-flow` application — that is
  [`doca-flow-tune`](../doca-flow-tune/SKILL.md);
- operators measuring the DPA-offloaded Flow path — that is
  [`doca-flow-dpa-perf`](../doca-flow-dpa-perf/SKILL.md);
- operators measuring end-to-end dataplane throughput or
  latency — that is the application's responsibility,
  layered on
  [`doca-flow`](../../libs/doca-flow/SKILL.md);
- contributors authoring or modifying the tool itself.

## Language scope

User interaction with `doca_flow_perf` is via:

1. The shipped binary's command-line flags (documented by
   `doca_flow_perf --help` and the public DOCA Flow Perf
   guide on `docs.nvidia.com`).
2. A JSON policy file describing the pipeline (ports, pipes,
   matchers, actions, forwarding). The shipped `configs/`
   directory contains canned policies for the most common
   traffic classes; new policies are authored by copying and
   editing one of those.
3. The tool's per-iteration output (CPU cycles per iteration,
   number-processed, number-failed; reported via the tool's
   stdout — the exact format is the public guide and the
   binary's runtime output, NOT this skill's invention).

The skill itself is Markdown. There is no programmatic API on
top of `doca_flow_perf`; consumers of its results read its
stdout / captured logs.

## When to load this skill

Load `doca-flow-perf` when ANY of the following is true:

- the user mentions `doca_flow_perf`, `doca-flow-perf`, the
  `configs/` JSON library, or asks for a "host-side flow
  rules per second" number;
- the user wants to baseline an underlying Flow path (not
  optimize a live application);
- the user is comparing host-side / DPU-CPU-side Flow
  performance across DOCA releases, BlueField generations,
  or firmware versions;
- the user wants to design a new traffic class JSON and
  needs to know which canned `configs/` JSON to start from
  and which fields they can change.

Co-load this skill with:

- [`doca-flow`](../../libs/doca-flow/SKILL.md) (the
  underlying library; flow-perf programs the same
  matchers / actions / pipes the library exposes);
- [`doca-flow-tune`](../doca-flow-tune/SKILL.md) (the
  measurement-vs-optimization distinction is the most
  common confusion);
- [`doca-flow-dpa-perf`](../doca-flow-dpa-perf/SKILL.md)
  (the host-vs-DPA-path distinction is the second most
  common confusion);
- [`doca-version`](../../doca-version/SKILL.md) (the
  four-way version match every reported flow-perf number
  must carry);
- [`doca-debug`](../../doca-debug/SKILL.md) and
  [`doca-setup`](../../doca-setup/SKILL.md) for the
  env-side debug ladder.

Do NOT load this skill when the user wants to optimize a live
Flow application (route to
[`doca-flow-tune`](../doca-flow-tune/SKILL.md)) or measure
the DPA-offloaded path (route to
[`doca-flow-dpa-perf`](../doca-flow-dpa-perf/SKILL.md)).

## What this skill provides

Three companion files in this directory, each owning a
different question shape:

- [`SKILL.md`](SKILL.md) — this file. Audience, scope,
  loading order, related skills. Routes everything else.
- [`CAPABILITIES.md`](CAPABILITIES.md) — *what
  `doca_flow_perf` is*, what it measures, what it
  deliberately doesn't measure, the DPDK-vs-DOCA backend
  duality, the JSON contract surface, the per-iteration
  output interpretation, version compatibility (versioned
  with `doca-flow` and `doca-version`), the layered error
  taxonomy, observability, and the safety policy overlay.
- [`TASKS.md`](TASKS.md) — the procedural verbs (`configure`,
  `run`, `test`, `debug`, etc.) plus a `doca_flow_perf`-
  specific command appendix and the agent-side `use`
  workflow that consumes the captured per-iteration output.

The combined skill teaches an AI agent to drive the
*measurement-class* of `doca_flow_perf` questions: pick a
shipped or author-new policy JSON, run the single-iteration
smoke, run the iterative eval loop, capture the four-tuple
that makes the resulting number defensible, interpret the
output, and route every adjacent question (tune the live
app, measure the DPA path, optimize the firmware) to the
right neighbouring skill.

## What this skill deliberately does not ship

- **End-to-end dataplane throughput or latency
  measurement.** `doca_flow_perf` measures the
  *control-plane* rate of programming rules, plus optional
  per-entry query timing. It does NOT measure how fast
  packets traverse the resulting rules in the dataplane.
  That is the application's responsibility, layered on
  [`doca-flow`](../../libs/doca-flow/SKILL.md). The agent
  must say this explicitly when the operator asks for "Flow
  throughput".
- **DPA-offloaded Flow path measurement.** Route to
  [`doca-flow-dpa-perf`](../doca-flow-dpa-perf/SKILL.md).
- **Optimization of a deployed Flow application.** Route to
  [`doca-flow-tune`](../doca-flow-tune/SKILL.md). flow-perf
  is a synthetic driver of a JSON-described pipeline, not a
  tuner of a live one.
- **A canonical "right answer" Kops/sec number.** The agent
  refuses to quote published numbers from memory as
  authoritative; the published numbers live in NVIDIA's
  release notes per the DOCA version and BlueField
  generation, and the operator must reproduce on their own
  exact preconditions before comparing.
- **Invented JSON schema fields.** The agent does NOT invent
  policy JSON keys that are not present in the shipped
  `configs/` exemplars. If a key the operator wants is not
  in any shipped exemplar, the agent says so and routes to
  the public DOCA Flow Perf guide.
- **Library-internal `doca-flow` API explanations.** The
  underlying matchers and actions belong to
  [`doca-flow`](../../libs/doca-flow/SKILL.md); this skill
  references them but does not duplicate the library's API
  documentation.
- **Cross-tool benchmarking apples-to-apples claims** when
  preconditions differ. Two flow-perf numbers from different
  DOCA versions / BlueField generations / firmware versions
  are NOT directly comparable; the agent insists on the
  four-tuple capture so consumers can judge.

## Loading order

When a `doca_flow_perf` question arrives:

1. Confirm DOCA is installed and the binary plus the
   `configs/` JSON library are reachable — if not, route to
   [`doca-setup`](../../doca-setup/SKILL.md);
2. Confirm the underlying `doca-flow` library is healthy on
   the device — if not, route to
   [`doca-flow TASKS.md ## test`](../../libs/doca-flow/TASKS.md#test);
3. Confirm the user wants to *measure*, not *optimize* —
   if optimize, route to
   [`doca-flow-tune`](../doca-flow-tune/SKILL.md);
4. Confirm the target path is host / DPU-CPU, not DPA — if
   DPA, route to
   [`doca-flow-dpa-perf`](../doca-flow-dpa-perf/SKILL.md);
5. Read [`CAPABILITIES.md`](CAPABILITIES.md) to commit to
   the three-axis decision (pipeline shape × traffic class ×
   measurement axis);
6. Read [`TASKS.md`](TASKS.md) and walk
   `## configure → ## run → ## test → ## debug` in that
   order; do NOT start with `## run` without the
   `## configure` precondition step.

## Related skills

Cross-link conventions follow the bundle's relative path
contract from `tools/<X>/`:

- [`doca-flow`](../../libs/doca-flow/SKILL.md) — the
  underlying library. flow-perf programs Flow pipes, entries,
  matchers, and actions; the library is the source of truth
  for the API surface flow-perf exercises.
- [`doca-flow-tune`](../doca-flow-tune/SKILL.md) — the
  unified Flow tuning tool. **Measurement vs. optimization
  boundary** lives here. Ask: "do I want a number, or do I
  want to change the deployed pipeline?"
- [`doca-flow-dpa-perf`](../doca-flow-dpa-perf/SKILL.md) —
  the DPA-offloaded Flow performance tool. **Host /
  DPU-CPU vs. DPA path boundary** lives here. Ask: "am I
  measuring the path that executes on the CPU, or the path
  that executes on the DPA processor?"
- [`doca-version`](../../doca-version/SKILL.md) — every
  reported flow-perf number must come with the four-way
  match (host package, kernel module, firmware, target
  application's linked `doca-flow` version) and the BlueField
  / ConnectX generation. flow-perf overlays this rule, not
  contradicts it.
- [`doca-setup`](../../doca-setup/SKILL.md) — DOCA install
  posture; routing for "is the binary even here?" questions.
- [`doca-debug`](../../doca-debug/SKILL.md) — the
  cross-cutting debug ladder for env-side issues (driver,
  firmware, BlueField mode, kernel module).
- [`doca-bench`](../doca-bench/SKILL.md) — a peer
  benchmarking tool with a broader scope (multiple DOCA
  primitives, not just Flow). flow-perf is the Flow-specific
  microbenchmark; doca-bench is the broader workload
  benchmark.
- [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md)
  — routing to the public `docs.nvidia.com` DOCA Flow Perf
  page, release notes, and forums for release-specific
  published numbers and reproducibility notes.
- [`doca-structured-tools-contract`](../../doca-structured-tools-contract/SKILL.md)
  — the agent's detect → prefer → fall back → report contract
  for the structured helpers (`doca-env --json`,
  `doca-capability-snapshot`, `version-matrix.json`)
  flow-perf preconditions rely on.
- [`doca-hardware-safety`](../../doca-hardware-safety/SKILL.md)
  — the canonical hardware-safety meta-policy that
  [`CAPABILITIES.md ## Safety policy`](CAPABILITIES.md#safety-policy)
  overlays.

This skill assumes the surrounding doca-flow application is
the operator's existing source artifact; flow-perf does not
ship a sample doca-flow application of its own.
