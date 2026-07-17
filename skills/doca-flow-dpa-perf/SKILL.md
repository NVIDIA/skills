---
license: Apache-2.0
name: doca-flow-dpa-perf
description: >
  Use this skill when the user is invoking doca_flow_dpa_perf on
  DPA-capable hardware (ConnectX-7+ or BlueField-3) to measure rule
  update / disable rates on the DPA-offloaded DOCA Flow path —
  picking the active / passive device split, choosing workload-shape
  axes (burst, queue, completion threshold, workers, hash pipe algo,
  PSL tables), or reading Kops/sec iteration stats and the optional
  self-test. Trigger even when the user does not explicitly mention
  "doca_flow_dpa_perf" or "DPA Provider" — typical implicit phrasings
  include "how fast can the DPA program path-selector entries",
  "baseline rule-update rate on ConnectX-8", "tool reports zero ops
  on my BlueField", "self-test sentinel never shows on tcpdump", or
  "is my BlueField-2 DPA-capable". Refuse and route elsewhere for the
  host / DPU-CPU Flow path (doca-flow-perf), Flow pipeline tuning
  (doca-flow-tune), writing doca-flow / doca-dpa applications, or
  DOCA install — those belong to other skills.
metadata:
  kind: tool
compatibility: >
  Requires DOCA SDK installed at /opt/mellanox/doca on Linux (Ubuntu
  22.04/24.04 or RHEL/SLES) with a DPA-capable device attached —
  ConnectX-7+ or BlueField-3 (BlueField-2 and earlier ConnectX are
  unsupported). VNF Flow mode required; PF or VF only (SFs are not
  supported on the DPA path). Reads `pkg-config doca-flow` and the
  shipped `doca_flow_dpa_perf` binary plus its README on the user's
  install.
---

# DOCA Flow DPA Perf (`doca_flow_dpa_perf`)

**Where to start:** This is a tool skill for invoking
`doca_flow_dpa_perf`, the DPA-accelerated Flow performance tool.
Open [`TASKS.md`](TASKS.md) and start at
[`## configure`](TASKS.md#configure) to confirm DPA-capable
hardware + VNF Flow mode + the active / passive device split, then
[`## run`](TASKS.md#run) for the smoke-before-bulk flow with a
small operation count before any sweep, then
[`## test`](TASKS.md#test) for the eval-loop overlay that gates
defensible Kops/sec numbers. Open [`CAPABILITIES.md`](CAPABILITIES.md)
when the question is *what `doca_flow_dpa_perf` can measure*,
*what the DPA preconditions are*, *which devices it runs on*,
or *how to interpret update / disable / self-test output without
fooling yourself*. If DOCA is not installed yet, route to
[`doca-setup`](../../doca-setup/SKILL.md) first; if the device is
not DPA-capable (no ConnectX-7+ or BlueField-3+) then this tool is
the wrong surface and the right answer is
[`doca-flow-perf`](../doca-flow-perf/SKILL.md).

## Example questions this skill answers well

The CLASSES of `doca_flow_dpa_perf` questions this skill is built
to answer, each with one worked example. The class is the
load-bearing piece; the worked example is one instance.

- **"Should I measure the DPA-offloaded Flow path or the
  host / DPU-CPU Flow path for this question?"** — worked
  example: *"my workload programs path-selector entries via
  DOCA Flow; do I baseline with `doca_flow_dpa_perf` or with
  `doca_flow_perf`?"*. Answered by the *DPA-vs-host* boundary
  in
  [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes)
  and the device-preconditions table.
- **"What does the DPA-offload actually accelerate, and what
  doesn't it change?"** — worked example: *"if I move my Flow
  rule update path to the DPA, what changes in the data plane
  for the packets themselves?"*. Answered by the DPA-Provider
  scope in
  [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes).
- **"What hardware do I need to use this tool at all?"** —
  worked example: *"is my BlueField-2 DPA-capable?"*. Answered
  by the device-preconditions table in
  [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes)
  (BlueField-3 yes, BlueField-2 no; ConnectX-7 recommended-or-
  later, ConnectX-8 / 9 supported per the public guide and the
  shipped README on the user's install).
- **"How do I size my run — burst, queue, completion threshold,
  number of operations, iterations — to get a defensible
  Kops/sec number?"** — worked example: *"I want the median
  iteration time and standard deviation, not a single noisy
  first-iteration spike"*. Answered by the eval-loop overlay
  in
  [`TASKS.md ## test`](TASKS.md#test) and the iteration-stats
  rule in
  [`CAPABILITIES.md ## Observability`](CAPABILITIES.md#observability).
- **"My tool reports zero ops / hangs / fails the self-test —
  what does that mean?"** — worked example: *"the tool runs but
  the self-test step fails"*. Answered by the layered error
  taxonomy in
  [`CAPABILITIES.md ## Error taxonomy`](CAPABILITIES.md#error-taxonomy)
  + the debug ladder in
  [`TASKS.md ## debug`](TASKS.md#debug).
- **"How do I quote a DPA-perf number alongside a host-side
  Flow-perf number for the same workload, in a way the next
  engineer can actually compare?"** — worked example: *"two
  Kops/sec numbers for what is supposedly the same workload"*.
  Answered by the four-tuple capture rule in
  [`CAPABILITIES.md ## Safety policy`](CAPABILITIES.md#safety-policy)
  + the per-tool-name rule (the host tool and the DPA tool are
  different surfaces; their numbers are not interchangeable
  without naming which tool produced which).

## Audience

This skill serves **external operators, performance engineers,
DOCA Flow application developers, and AI agents who need a
defensible measurement of the DPA-offloaded Flow update path** on
DPA-capable hardware. Concretely:

- A platform operator deciding whether to move a path-selector
  workload onto the DPA versus keeping it on the host / DPU-CPU
  path, and wanting a number to compare.
- A performance engineer producing a *"DPA Kops/sec for update
  operation, queue-size X, burst-size Y, N workers"* baseline
  on a specific device + DOCA version so a downstream
  comparison is meaningful.
- A DOCA Flow application developer who has already used
  `doca-dpa` to land a DPA-offload of their Flow rule update
  path and wants to characterize what the device delivers.
- An AI agent answering *"what update rate should I expect from
  the DPA-offloaded Flow path on device Y?"* honestly — with a
  measured number, the command line that produced it, and the
  device + DOCA version + as-deployed environment that scopes
  it — instead of guessing from datasheet headlines.

It is **not** for users debugging the tool's source code,
**not** a substitute for the live public DOCA Flow DPA Perf guide
on `docs.nvidia.com`, **not** the place to learn the `doca-flow`
or `doca-dpa` APIs (that audience belongs in
[`doca-flow`](../../libs/doca-flow/SKILL.md) and
`doca-dpa`), and **not** the right
tool for the host / DPU-CPU Flow path (route to
[`doca-flow-perf`](../doca-flow-perf/SKILL.md)).

`doca_flow_dpa_perf` is shipped as a **single CLI binary** with
DPA-side device code linked in. The skill uses the same
`kind: tool` three-file shape as the rest of the bundle so
the agent's task-verb contract is uniform across the bundle.

## Language scope

This skill governs invocation, output interpretation, and
recommendation-of-routing for the `doca_flow_dpa_perf` CLI on
DPA-capable hardware. The tool itself has both a host-side
control (C-language ARGP + DOCA + DPDK code per the shipped
`flow_dpa_perf.c` / `flow_dpa_perf_core.c`) and a DPA-side device
component (DPA-side code on the shipped DPA device runtime).
External users do not link any of this; what they configure is
the JSON-config-or-CLI invocation surface. For the
`doca-dpa` programming model behind the DPA-side execution
engine, see
`doca-dpa`; for the `doca-flow`
API behind the pipeline the DPA path executes, see
[`doca-flow`](../../libs/doca-flow/SKILL.md).

## When to load this skill

Load this skill when the user is — or the agent needs to —
invoke `doca_flow_dpa_perf` on a real host with DOCA installed
and a DPA-capable device attached (or the public NGC DOCA
container with the equivalent device passthrough) to measure
update / disable rates on the DPA-offloaded Flow path.
Concretely:

- Confirming DPA preconditions (DPA-capable device class,
  VNF Flow mode, recommended PF use, no SFs) before invoking
  the tool.
- Picking the active / passive device split appropriate to the
  user's hardware (two-port BlueField-3 active + passive; one-
  port ConnectX-9 active only).
- Picking the workload-shape axes (burst size, queue size,
  completion threshold, hash pipe algorithm, work policy,
  number of PSL tables, table size, number of workers).
- Picking the operation axis (update or disable-enable) per the
  shipped README's documented operations.
- Producing a defensible Kops/sec number with iteration stats
  (median, max, standard deviation) captured.
- Diagnosing zero-ops / hung / failed-self-test runs through
  the layered error taxonomy.

Do **not** load this skill for general DOCA orientation, Flow
program API work, or installation. For those, use
[`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md),
the matching `libs/<library>` skill, or
[`doca-setup`](../../doca-setup/SKILL.md). Do not load it for
the host / DPU-CPU Flow path — that audience belongs in
[`doca-flow-perf`](../doca-flow-perf/SKILL.md).

## What this skill provides

This is a **thin loader**. Substantive material lives in two
companion files:

- `CAPABILITIES.md` — what `doca_flow_dpa_perf` measures
  (the DPA-Provider-on-DPA-device update / disable path
  specifically), the DPA-vs-host-path boundary, the
  device-preconditions table (ConnectX-7+ / BlueField-3+),
  the documented VNF-only Flow-mode rule, the PF-vs-VF-vs-SF
  rule (SFs not supported on DPA), the workload-shape axes
  (burst, queue, completion threshold, hash pipe algorithm,
  work policy, PSL tables, table size, workers), the
  operation axis (update vs disable-enable), the version
  overlay (this tool rides the `doca-flow` and `doca-dpa`
  versions it links against; the canonical rules live in
  [`doca-version`](../../doca-version/SKILL.md)), the layered
  error taxonomy
  (config-syntax / device-binding / dpa-precondition /
  workload-precondition / measurement-soundness / self-test /
  version / cross-cutting), the observability surface
  (iteration statistics, self-test path-selector verification,
  tcpdump-side traffic verification), and the safety posture
  (smoke-before-bulk, four-tuple capture, name the tool that
  produced the number).
- `TASKS.md` — step-by-step workflows for the in-scope task
  verbs: `install` (route to setup; the binary is shipped),
  `configure` (DPA-preconditions + active / passive device +
  workload-shape decision), `build` (route to install — the
  binary is shipped), `modify` (refuse — modify the invocation,
  not the binary), `run` (smoke before bulk), `test` (eval
  loop), `debug` (layered diagnosis), `use` (consume the
  captured number), plus a `Deferred task verbs` block routing
  out-of-scope questions and a `Command appendix`.

The skill assumes a host where DOCA is already installed (or
the NGC DOCA container is running) on a DPA-capable device and
the operator has the permissions to bind the device and allocate
the DPA execution resources the tool needs.

## What this skill deliberately does not ship

This skill is **agent guidance**, not a samples or scripts
bundle. To keep the boundary clean, it deliberately does not
contain — and pull requests should not add:

- **Verbatim default values for flag inventories beyond what
  the public guide and the shipped README document.** The
  flag surface is install-specific within the documented
  surface; the documented invocations + `--help` on the
  installed version are the authoritative answer. Inventing
  a flag is the most common hallucination failure.
- **Pre-baked example Kops/sec numbers or expected throughput
  numbers.** Output is device-, firmware-, DOCA-version-,
  workload-, and platform-specific; a pinned number for one
  platform misleads operators on a different platform /
  version. The shipped README's example numbers are
  *illustrative*, not a baseline the agent should quote as
  ground truth.
- **Wrappers, parsers, or scripts** in any language that
  consume the tool's stdout / CSV. The output format is
  documented; if a user wants to script against it, the
  right answer is "read the live guide, write the parser
  against your installed version".
- **A `samples/` or `reference/` subtree.** This is a thin
  loader for a documented CLI; substantive material lives on
  the public page, in `--help`, and in the shipped README on
  the user's install.

## Loading order

1. Read this `SKILL.md` first to confirm the user's question
   is in scope (the user actually wants to invoke
   `doca_flow_dpa_perf` on DPA-capable hardware, not measure
   the host / DPU-CPU Flow path).
2. **For what `doca_flow_dpa_perf` measures, the DPA-vs-host
   boundary, the device-preconditions table, the workload-
   shape axes, the version overlay, the error taxonomy, the
   observability surface, and the safety posture, see
   [CAPABILITIES.md](CAPABILITIES.md).**
3. **For the documented invocations and the smoke-before-bulk
   workflow — `install`, `configure`, `build`, `modify`,
   `run`, `test`, `debug`, `use` — see [TASKS.md](TASKS.md).**

## Related skills

- [`doca-flow`](../../libs/doca-flow/SKILL.md) — the **base
  library** whose pipeline this tool measures on the DPA
  path. The pipe / entry / rule surface this tool drives is
  created by `doca-flow` program code; the library's pipe
  attributes and capability surface are the upstream context.
- `doca-dpa` — the
  programming model behind the DPA execution engine the tool
  runs on. When the user's question goes from *"measure the
  DPA path"* to *"why is the DPA path doing this"*, that
  skill is the next stop.
- [`doca-flow-perf`](../doca-flow-perf/SKILL.md) — the
  host / DPU-CPU Flow performance tool. The cross-tool
  comparison rule lives in
  [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes):
  name which tool produced which number.
- [`doca-flow-tune`](../doca-flow-tune/SKILL.md) — the Flow
  tuning tool. A DPA-perf number is the kind of baseline
  `doca-flow-tune` then optimizes on top of, via a Flow-program
  modify-a-sample loop.
- [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md) —
  routing to the public DOCA Flow DPA Perf page on
  `docs.nvidia.com` and the rest of the public DOCA
  documentation set.
- [`doca-version`](../../doca-version/SKILL.md) — canonical
  DOCA version-handling rules. The
  [`## Version compatibility`](CAPABILITIES.md#version-compatibility)
  section in this skill is a thin overlay on top.
- [`doca-setup`](../../doca-setup/SKILL.md) — env preparation,
  install verification, hugepages, NUMA awareness, and the
  *I have no install yet* path with the public NGC DOCA
  container.
- [`doca-debug`](../../doca-debug/SKILL.md) — the cross-cutting
  debug ladder. DPA-perf surfaces *its own* error taxonomy;
  when the cause turns out to be below DOCA, the taxonomy
  hands off to `doca-debug`.
- [`doca-hardware-safety`](../../doca-hardware-safety/SKILL.md) —
  the cross-cutting hardware-safety meta-policy this skill's
  `## Safety policy` overlays.
