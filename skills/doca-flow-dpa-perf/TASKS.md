# DOCA Flow DPA Perf — Tasks

**Where to start:** The verbs that carry real workflow content
are `## configure`, `## run`, `## test`, and `## debug`. The
other verbs (`## install`, `## build`, `## modify`, `## use`)
carry routing stubs or a tightly-scoped agent-side workflow,
because `doca_flow_dpa_perf` is a shipped binary, not a source
artifact the user compiles or patches.

This file is loaded by [`SKILL.md`](SKILL.md) after
[`CAPABILITIES.md`](CAPABILITIES.md). It walks the agent through
the task verbs every artifact in this bundle exposes, then
explicitly defers verbs that do not belong here.

## install

`doca_flow_dpa_perf` is **shipped pre-built** with the DOCA
install when the Flow DPA Perf package profile is present.
There is no separate install workflow this skill owns.

Routing for nearby "install" questions:

- *"The binary isn't there — do I need to install something?"*
  → yes. Route to
  [`doca-setup ## install`](../../doca-setup/TASKS.md#configure)
  for the DOCA install or to
  [`doca-setup ## no-install`](../../doca-setup/TASKS.md#no-install)
  for the NGC DOCA container path.
- *"My device class is not DPA-capable — can I install a
  workaround?"* → no. Per the device-preconditions table in
  [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes),
  BlueField-2 and pre-ConnectX-7 are not in the supported set.
  The right answer is host-side
  [`doca-flow-perf`](../doca-flow-perf/SKILL.md), not a
  workaround.

## configure

The tool's *configuration* is the invocation itself: there is
no separate config file, no daemon, no env knob the public
guide documents as required beyond DOCA-wide variables. What
the agent has to *configure* is the precondition + device-split
+ workload-shape decision documented in
[`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes).

Steps the agent should walk the user through, in order:

1. **Confirm DOCA is installed and the device is DPA-capable.**
   Run [`doca-setup ## test`](../../doca-setup/TASKS.md#test);
   cross-check against the device-preconditions table in
   [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes).
   If the device is BlueField-2 or pre-ConnectX-7, this is
   the wrong tool — route to
   [`doca-flow-perf`](../doca-flow-perf/SKILL.md).
2. **Confirm VNF Flow mode and supported device target.** Per
   the shipped README, VNF mode is required and SFs are not
   supported. PF is recommended; VF is supported. Confirm the
   operator's intended PCI device is in the supported set.
3. **Pick the active / passive device split.** Per the shipped
   README's example invocations:
    - BlueField-3 two-port: active + passive PCI pair.
    - ConnectX-8 two-port: active + passive PCI pair.
    - ConnectX-9 one-port: active only.
   Confirm the operator's hardware matches one of these
   shapes; do not invent a hybrid invocation.
4. **Pick the operation.** Update or disable-enable per the
   shipped README. Different DPA-side paths; not
   interchangeable. The agent should name *why* it picked the
   operation so the user can challenge the framing.
5. **Pick the workload-shape axes.** Per the workload-shape
   table in
   [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes):
   burst size, queue size, completion threshold, number of
   PSL tables, table size, number of workers, hash pipe algo,
   work policy, warmup, pipeline policy. Defaults belong to
   the shipped README on the user's install; quote the README,
   do not infer.
6. **Pick the self-test posture.** The self-test path is
   off by default; enable it only when end-to-end correctness
   (not just throughput) is the question. The README documents
   the tcpdump-side verification step.
7. **Sanity check before any invocation.** Confirm with the
   user: what is the question (throughput? correctness? a
   sweep)? Which device-class? Which workload shape? If any
   answer is unclear, stop and ask.

For the canonical DOCA universal lifecycle that underlies
program-side configuration (which the tool runs internally),
see [`doca-programming-guide TASKS.md ## configure`](../../doca-programming-guide/TASKS.md#configure).
This skill is concerned with the *operator*-side configuration
of the tool invocation, not the program-side lifecycle of
`doca-flow` or `doca-dpa` underneath.

## build

`doca_flow_dpa_perf` is **shipped pre-built** as part of every
DOCA install with the Flow DPA Perf package profile present.
There is no source tree the external user is expected to
compile, no build flags, no `meson` or `make` workflow for the
tool itself.

Routing for nearby "build" questions:

- *"The binary isn't there — do I need to build it?"* → no.
  Route to [`## install`](#install).
- *"I want to build my own DPA-Flow perf harness."* → not a
  `doca_flow_dpa_perf` question. Route to
  [`doca-programming-guide ## build`](../../doca-programming-guide/TASKS.md#build)
  plus
  `doca-dpa ## build` and
  [`doca-flow TASKS.md ## build`](../../libs/doca-flow/TASKS.md#build)
  for the library-specific build overlays. The tool is the
  shipped harness; a bespoke harness is a different artifact.
- *"I want to extend the tool with a new workload axis."* →
  out of scope here; this skill is for external operators
  consuming the shipped tool, not contributors extending it.

## modify

**Do not modify the shipped `doca_flow_dpa_perf` binary.** It
is an NVIDIA-shipped CLI; there is no documented public way to
change its behavior, output format, or workload-axis surface.

What the agent *does* modify, every time, is the **invocation**
— the flags, the chosen device target, the workload-shape axes,
the operation. That is the configuration loop in
[`## configure`](#configure) above and the iteration loop in
[`## test`](#test) below.

Routing for nearby "modify" questions:

- *"Can I patch the tool to add a new workload axis?"* → out
  of scope.
- *"Can I post-process the stdout into a different format?"*
  → the documented surfaces are stdout and (per the README)
  the iteration-stats block; even a parser script is out of
  scope per
  [`SKILL.md ## What this skill deliberately does not ship`](SKILL.md#what-this-skill-deliberately-does-not-ship).
- *"I need a different metric than Kops/sec — can I change the
  tool to report something else?"* → no. The metric is what
  it is. If the user's question is genuinely outside the
  tool's surface, route to
  [`doca-flow-tune`](../doca-flow-tune/SKILL.md) for
  optimization-class questions, or to
  [`doca-flow-perf`](../doca-flow-perf/SKILL.md) for the host
  / DPU-CPU path's measurement surface.

## run

The smoke-before-bulk flow — every DPA-perf session goes
through it. The full invocation surface lives in the shipped
README on the user's install plus the public DOCA Flow DPA
Perf guide; this section names the *shape* of the flow, not
verbatim command lines (per
[`CAPABILITIES.md ## Safety policy`](CAPABILITIES.md#safety-policy)
*"do not invent flags"*).

1. **Confirm preconditions.** Per
   [`## configure`](#configure) steps 1-3. The device must
   be DPA-capable, VNF mode active, not an SF.
2. **Smoke run — small operations, few iterations, warmup
   enabled.** Pick the smallest defensible values per the
   shipped README's defaults (small `num-operations`, small
   `num-iterations`, warmup on, single worker, identity or
   random algo). Goal: *the tool binds the devices, the
   DPA path starts, and per-iteration stats land in a
   defensible range*. Not a usable performance number.
3. **Read the printed parameter set.** Per the shipped
   README's "Tool parameters are: ..." block, the tool
   echoes every effective parameter at the start. This is
   the user's chance to catch a defaulted value that does
   not match intent.
4. **Inspect iteration statistics.** A successful smoke
   shows median ≈ max (low variance), low standard
   deviation, and a Kops/sec rate consistent with the
   warmup having taken effect. If standard deviation is
   high, loop back to [`## debug`](#debug) layer 5
   before sinking time into a longer run.
5. **(Optional) self-test correctness path.** If end-to-end
   correctness is part of the question, enable self-test
   and follow the README's tcpdump-side verification.
6. **Plan the bulk run.** Only after the smoke is green.
   Increase `num-operations`, `num-iterations`, workers,
   table count / size per the question; estimate the wall-
   clock cost; confirm the operator is OK with it.
7. **For the exact, current invocation surface — flag names,
   default values, ranges, sweep families** read `--help`
   on the installed binary and the shipped README plus the
   public DOCA Flow DPA Perf guide via
   [`doca-public-knowledge-map ## DOCA tools`](../../doca-public-knowledge-map/SKILL.md#doca-tools).
   Do **not** invent any of these.

When recording the run for downstream consumers, write down:
the DOCA version, the host platform (host vs BlueField Arm,
OS, kernel, firmware), the exact command line used, the
device target (active / passive PCI addresses), the
configured workload-shape values (the tool's own echo line
covers this), and the full unredacted iteration-stats block
plus any self-test output. The downstream
[`## test`](#test) and [`## debug`](#debug) workflows depend
on those five fields.

## test

`doca_flow_dpa_perf` is **a measurement tool**, so its
`## test` verb is about *testing the measurement* — i.e.
confirming the Kops/sec number is sound and reproducible — not
unit-testing the tool itself.

**`## test` is an iterative loop, not a one-shot pass.** A
DPA-perf run that completes is not the same as a run that
produced a defensible number; each iteration tightens one
axis of measurement soundness (warmup, iteration variance,
self-test correctness, NUMA placement, cross-run
reproducibility, cross-version delta) and loops back to
[`## run`](#run).

The eval-loop overlay (rows apply to every DPA-perf run, not
just one workload):

| Iteration trigger | What it looks like | What changes next iteration |
| --- | --- | --- |
| Smoke completed; Kops/sec is far below datasheet headline | Could be cold pipeline, wrong workload shape, wrong device class for this DPA, or actually-right for this install. Do not assume datasheet first. | Confirm warmup applied per [`CAPABILITIES.md ## Error taxonomy`](CAPABILITIES.md#error-taxonomy) layer 5; re-check the workload-shape axes per [`## configure`](#configure) step 5; only then question hardware. |
| Per-iteration variance is high (standard deviation > a small fraction of median) | Steady-state not reached or outlier-dominated | Lengthen the run via `num-iterations`; if still volatile, isolate the device from other workloads; consider NUMA pinning. |
| Self-test path-selector sentinel does not appear on the wire | Traffic was not sent during the pause, or the configured ports do not match the operator's traffic generator | Re-walk the README's self-test section; confirm the tcpdump filter matches the configured device. |
| Same invocation produces different numbers on two hosts at the same DOCA version | NUMA / firmware / driver delta below DOCA | Walk axis 2 environment (cores / threads / NUMA) and the version layer per [`doca-version TASKS.md ## test`](../../doca-version/TASKS.md#test) before blaming the tool. |
| Same invocation produces different numbers on the same host across DOCA versions | This *is* a regression signal — provided both four-tuples are captured | Cross-link the two baselines, name the changed fields, route to [`doca-version TASKS.md ## debug`](../../doca-version/TASKS.md#debug) for the version-delta diagnosis. |
| DPA-perf number vs host-side Flow-perf number are compared without naming the tool | This is the cross-tool apples-to-oranges failure | Restate which number came from which tool; the two are not interchangeable per [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes). |

The agent's rule: every change to the invocation re-opens the
loop. Re-running with a tweaked flag and quoting the new
number without re-checking warmup / iteration variance /
self-test is exactly the failure mode this loop replaces.

**Baseline-capture rule.** When the goal of the session is a
baseline (vs an ad-hoc question), the captured artifact must
include the four-tuple — (command line + DOCA version + device
+ as-deployed environment) — alongside the iteration-stats
block AND the tool name (per
[`CAPABILITIES.md ## Safety policy`](CAPABILITIES.md#safety-policy)).
Without all of those, the baseline cannot be regression-tested
or compared to a host-side number later.

Loop termination: stop iterating once two consecutive runs
do not change the picture — the answer is now *"this is what
the DPA path delivers on this DOCA version / firmware /
driver stack"*.

This skill does **not** ship a "test fixture" or pre-recorded
expected output. The expected output is install-, device-,
firmware-, and tuning-specific.

## debug

When `doca_flow_dpa_perf` fails to start, fails to produce
numbers, or produces numbers that do not look defensible, walk
the [`CAPABILITIES.md ## Error taxonomy`](CAPABILITIES.md#error-taxonomy)
layers in order:

1. **Config-syntax.** Confirm the flag exists on the
   installed `--help` and in the shipped README; confirm
   units / values are in the documented form.
2. **Device-binding.** Confirm the active / passive PCI
   addresses exist on the host via
   [`doca-caps ## run`](../doca-caps/TASKS.md#run);
   confirm the device class is in the supported set; confirm
   the device target is not an SF.
3. **DPA-precondition.** Confirm VNF Flow mode; confirm the
   LAG / multi-port-Eswitch preconditions per the shipped
   README; confirm the DPA execution resources are present
   per
   `doca-dpa TASKS.md ## debug`.
4. **Workload-precondition.** Re-walk the workload-shape
   rules per the README (burst-size divisibility,
   queue-size power-of-2, workers range, algo / operation
   compatibility).
5. **Measurement-soundness.** Walk the three sub-layers
   per
   [`CAPABILITIES.md ## Error taxonomy`](CAPABILITIES.md#error-taxonomy)
   layer 5.
6. **Self-test.** If self-test was enabled and failed,
   re-walk the README's self-test section.
7. **Version.** Walk
   [`doca-version TASKS.md ## debug`](../../doca-version/TASKS.md#debug)
   end-to-end.
8. **Cross-cutting.** Hand off to
   [`doca-debug ## debug`](../../doca-debug/SKILL.md) and
   [`doca-setup ## debug`](../../doca-setup/TASKS.md#debug).

In every case: **quote what the tool reported**, including
the iteration-stats block and the self-test output if it
ran. Do not paraphrase or summarize.

## use

`## use` is the agent-side workflow for *consuming* a
captured `doca_flow_dpa_perf` number as evidence inside a
larger conversation with the user.

1. **Read the four-tuple plus the tool name first.** A
   captured DPA-perf number without the four-tuple OR
   without the tool name is unfalsifiable; the agent says
   so rather than draw inferences from it.
2. **Cross-check the device class against the supported
   set.** Per
   [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes),
   a DPA-perf number from BlueField-2 is structurally
   impossible; that is a red flag the agent surfaces.
3. **Cross-check the iteration stats.** Median ≈ max + low
   standard deviation + warmup-enabled is the soundness
   signal; high variance or warmup-off is a soundness flag
   the agent must surface before quoting the number.
4. **Cross-link to the host-side Flow-perf number** (if
   the user is comparing). Per
   [`doca-flow-perf`](../doca-flow-perf/SKILL.md), the host
   tool's number is *not* directly comparable to a DPA-perf
   number; they measure different paths.
5. **Route to [`doca-flow-tune`](../doca-flow-tune/SKILL.md)
   only after the agent confirms the baseline is sound.**
   Optimization on top of an unsound baseline is the
   canonical wasted-iteration failure mode.

## Deferred task verbs

The verbs below are not `doca_flow_dpa_perf` work and should
be routed out before the agent does any of them under this
skill's name.

- **measure the host / DPU-CPU Flow path** ⇒
  [`doca-flow-perf`](../doca-flow-perf/SKILL.md). This tool
  measures the DPA path only.
- **optimize a Flow pipeline against a baseline** ⇒
  [`doca-flow-tune`](../doca-flow-tune/SKILL.md). Optimization
  is downstream of measurement.
- **install DOCA** ⇒
  [`doca-setup ## install`](../../doca-setup/TASKS.md#configure)
  and [`## no-install`](../../doca-setup/TASKS.md#no-install).
- **write a doca-flow / doca-dpa application** ⇒
  [`doca-flow`](../../libs/doca-flow/SKILL.md) +
  `doca-dpa`, layered on
  [`doca-programming-guide`](../../doca-programming-guide/SKILL.md).
  The tool is the shipped harness; a bespoke application is
  a different artifact.
- **streaming telemetry / live metrics export** ⇒ not a
  feature of this tool. The DOCA Telemetry Service (DTS) is
  the documented telemetry surface; routing belongs in
  [`doca-public-knowledge-map ## DOCA services`](../../doca-public-knowledge-map/SKILL.md#doca-services).

## Command appendix

`doca_flow_dpa_perf`-specific invocation classes the verbs
above reach for. Every row is a CLASS — the agent must not
invent flags or default values beyond `--help` on the
installed binary, the shipped README, and the public DOCA
Flow DPA Perf guide.

**Infra-aware preamble (every row below).** Per the bundle's
detect → prefer → fall back → report contract documented in
[`doca-structured-tools-contract ## The agent behavior contract`](../../doca-structured-tools-contract/SKILL.md#the-agent-behavior-contract),
the agent should:

1. Probe for the matching structured helper FIRST
   (`doca-env --json`; `doca-capability-snapshot`;
   `version-matrix.json`).
2. If the probe succeeds, the structured tool's output is the
   authoritative answer.
3. If the probe fails, fall back to the manual command in
   the row.
4. The schemas the structured tools emit are defined in
   [`doca-structured-tools-contract ## Schemas`](../../doca-structured-tools-contract/SKILL.md#schemas);
   the version-handling semantics are owned by
   [`doca-version`](../../doca-version/SKILL.md).

| Purpose (class) | Invocation (shape) | Owning step | Reads as healthy when … |
| --- | --- | --- | --- |
| Discover the documented flag surface | `doca_flow_dpa_perf --help` plus the shipped README plus the public DOCA Flow DPA Perf guide | [`## configure`](#configure) step 1 + [`## debug`](#debug) layer 1 | Prints the documented flag inventory; the agent uses this as the only source of truth for flag names. |
| Confirm device class is DPA-capable | The device-preconditions table in [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes); cross-check via [`doca-caps ## run`](../doca-caps/TASKS.md#run) | [`## configure`](#configure) step 1 + [`## debug`](#debug) layer 3 | The device is ConnectX-7+ or BlueField-3; not BlueField-2 / earlier ConnectX. |
| Smoke run — single-worker update | The documented invocation against an active (or active + passive) PCI device with small `num-operations`, small `num-iterations`, warmup enabled | [`## run`](#run) steps 2-4 | The iteration-stats block reports low standard deviation; the Kops/sec rate is in a defensible order of magnitude for the device class. |
| Self-test correctness | The documented self-test invocation plus the README's tcpdump-side verification | [`## run`](#run) step 5 | The path-selector sentinel `65432` appears on the wire during the pause; the disable step removes it from the wire after the second pause. |
| Sweep a workload-shape axis | The documented sweep invocation across a planned range (burst, queue, workers, etc.) | [`## run`](#run) step 6 | The smoke for the boundary values passed first; the sweep completes; the resulting series has no implausible discontinuities. |
| Capture a baseline alongside the iteration stats | Redirect the tool's stdout to a file (`> dpa-perf-session.txt`) plus the four-tuple captured from `doca-env --json` (or its manual fallback) | [`## test`](#test) baseline-capture rule | A captured file is written; the stdout iteration-stats block is preserved verbatim; the four-tuple accompanies the file. |
| Cross-link to host-side Flow perf | Capture a matching session with [`doca-flow-perf`](../doca-flow-perf/SKILL.md) on the host / DPU-CPU path; quote each number with its tool name | [`## use`](#use) step 4 | The two captures live alongside each other in the bundle the agent saves; the cross-tool comparison is explicit, not implicit. |

Three cross-cutting rules for this appendix:

- **Never invent a flag, default value, workload-axis name,
  or output column header.** `--help` on the installed
  binary, the shipped README on the user's install, and the
  public DOCA Flow DPA Perf guide are the joint contract.
- **Smoke before bulk.** Every row above presumes the smoke
  row succeeded first.
- **Cross-link instead of duplicate.** Cross-cutting
  commands (`pkg-config --modversion`, `doca_caps --list-devs`,
  `dmesg`, `mlxconfig -d <bdf> q`, `numactl --hardware`) live
  in [`doca-debug TASKS.md ## Command appendix`](../../doca-debug/TASKS.md)
  and
  [`doca-setup TASKS.md ## debug`](../../doca-setup/TASKS.md#debug);
  the host-side Flow-perf commands live in
  [`doca-flow-perf`](../doca-flow-perf/SKILL.md); the DPA
  programming-side commands live in
  `doca-dpa`; this appendix
  names only DPA-Flow-Perf-specific invocations on top.

## Cross-cutting

A few rules that apply across every verb in this file:

- The **public DOCA Flow DPA Perf guide** + the installed
  `--help` + the shipped `README.md` are the joint source of
  truth. When they disagree, the *installed* `--help` plus
  the shipped README on the user's version win.
- `doca_flow_dpa_perf` *does* drive hardware and *does*
  allocate DPA resources; smoke-before-bulk is mandatory.
- **Quote the four-tuple AND the tool name.** Command line
  + DOCA version + device + as-deployed environment + the
  string `doca_flow_dpa_perf` is the minimum unit a number
  from this tool is meaningful in.
- This skill **assumes DPA-capable hardware and a healthy
  DOCA install**. If either is in doubt, route to
  [`doca-setup`](../../doca-setup/SKILL.md),
  [`doca-version`](../../doca-version/SKILL.md), or
  [`doca-flow-perf`](../doca-flow-perf/SKILL.md) (for the
  host / DPU-CPU path) before running anything else here.
