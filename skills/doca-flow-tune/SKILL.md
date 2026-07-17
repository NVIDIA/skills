---
license: Apache-2.0
name: doca-flow-tune
description: >
  Use this skill when the user is tuning a live or captured
  `doca-flow` pipeline with `doca_flow_tune` — snapshotting
  pipe / counter / KPI state, picking a tuning axis (rule
  placement, resource hints / table sizing, HW-offload mode)
  and a matching measurement (rule-install rate, lookup latency,
  hardware-counter delta), running offline or online (read-only
  or state-changing) modes, reading the dumper CSV / analyze
  JSON / visualize mermaid, or applying a recommendation back
  into the Flow program. Trigger even when the user does
  not explicitly mention "doca_flow_tune" — typical implicit
  phrasings include "Flow rule-install rate is low on
  BlueField", "table sizing looks wrong for this pipe", "tune
  visualize step is empty", "before/after
  counters don't move", or "which doca-flow knob does this
  recommendation hit". Refuse and route elsewhere for measuring
  baseline numbers (doca-flow-perf, doca-flow-dpa-perf), writing
  the doca-flow application, DOCA install, or streaming Flow
  telemetry — those belong to other skills.
metadata:
  kind: tool
compatibility: >
  Requires DOCA SDK installed at /opt/mellanox/doca on Linux
  (Ubuntu 22.04/24.04 or RHEL/SLES) with a BlueField DPU or
  ConnectX NIC attached, plus a running or captured `doca-flow`
  application to observe. Reads the user's local install via
  `pkg-config doca-flow` and the shipped `flow_tune_cfg*.json`
  templates and `scripts/` directory under /opt/mellanox/doca.
---

# DOCA Flow Tune (`doca_flow_tune`)

> **Subcommand surface correction (Run-12, verified Run-13
> against doca/tools/flow_tune/src/tune/common/tune_config.cpp).**
> `doca_flow_tune` is a single binary whose **role on a given
> invocation is determined by which of five top-level
> subcommands** the user picks — `dump`, `monitor`, `web`,
> `analyze`, `visualize` (case-insensitive on the CLI;
> uppercased in this skill for readability). All five names
> are registered via `doca_argp_cmd_set_name(...)` in
> `tune_config.cpp` (lines 1799 / 1860 / 1896 / 2074 / 2111);
> `analyze` further accepts `import` / `export` / `packet_trace`
> / `sim_timing` sub-subcommands. The `dump` / `monitor` / `web`
> subcommands run the binary in **server-attached online mode**
> against a live `doca-flow` application reached over a Unix-
> domain socket whose path lives in `network.server_uds` of the
> shipped `flow_tune_cfg*.json`; the `analyze` / `visualize`
> subcommands run in **offline / captured-snapshot mode** against
> JSON / CSV files the online modes previously dropped into the
> configured `outputs_directory`. The rest of this skill (and
> [`CAPABILITIES.md`](CAPABILITIES.md) / [`TASKS.md`](TASKS.md))
> uses the legacy *"server role / online mode / offline mode"*
> framing — that framing is internally consistent with the
> subcommand surface here: *server role* = a server-attached
> online subcommand (`dump`/`monitor`/`web`); *online mode* =
> any of `dump`/`monitor`/`web`; *offline mode* =
> `analyze`/`visualize`. Treat the subcommand name as the
> primary handle; treat *server/online/offline* as the
> downstream behavioral consequence of the subcommand pick.

**Where to start:** This is a tool skill for invoking `doca_flow_tune`,
the unified DOCA Flow tuning tool. Open [`TASKS.md`](TASKS.md) and
start at [`## configure`](TASKS.md#configure) to commit to the
three-axis decision (target Flow pipeline × tuning axis ×
measurement) and pick offline vs online vs server-attach mode, then
[`## run`](TASKS.md#run) for the snapshot → analyze → visualize
loop, then [`## test`](TASKS.md#test) for the smoke-before-bulk
overlay that gates any state-changing application of a tuning
recommendation back into the Flow application's code. Open
[`CAPABILITIES.md`](CAPABILITIES.md) when the question is *what
state `doca_flow_tune` can observe and recommend on*, *how its
server / client roles fit inside the single artifact*, *which DOCA
version the tool ships in*, or *how to interpret the dumper / monitor
/ analyze / visualize outputs without fooling yourself*. If DOCA is
not installed, route to
[`doca-setup`](../../doca-setup/SKILL.md) first; if the user has
no running `doca-flow` application yet, route to
[`doca-flow`](../../libs/doca-flow/SKILL.md) — flow-tune does not
create pipes, it observes and recommends on top of pipes the
library already created.

## Example questions this skill answers well

The CLASSES of `doca_flow_tune` questions this skill is built to
answer, each with one worked example. The class is the load-bearing
piece; the worked example is one instance.

- **"Should I reach for `doca-flow-tune` or `doca-flow-perf` for
  this question?"** — worked example: *"my doca-flow service runs
  on a BlueField-3 and I think the rule-install rate is below what
  the device can sustain; do I measure first or tune first?"*.
  Answered by the *tune vs perf* boundary in
  [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes)
  and the routing into
  [`doca-flow-perf`](../doca-flow-perf/SKILL.md) for baselines vs
  this skill for optimization on top of a measured baseline.
- **"Capture a snapshot of a live `doca-flow` pipeline's hardware
  and software counters without touching the dataplane."** — worked
  example: *"I want a side-effect-free dumper / monitor run against
  the running Flow ports for an operations-rate profile"*. Answered
  by the snapshot flow in
  [`TASKS.md ## run`](TASKS.md#run) plus the read-only-by-default
  posture in
  [`CAPABILITIES.md ## Safety policy`](CAPABILITIES.md#safety-policy).
- **"Pick the right tuning axis — rule placement, resource hints,
  or hardware-offload mode — for the question I actually have."**
  — worked example: *"my Flow pipe's rule-install rate is low; is
  this a placement question or a table-sizing question?"*. Answered
  by the three-axis configuration in
  [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes)
  + the configure walk in
  [`TASKS.md ## configure`](TASKS.md#configure).
- **"How do `doca_flow_tune`'s server role and client / consumer
  role fit together inside the single artifact?"** — worked
  example: *"I keep reading about a Flow Tune server and a Flow
  Tune client; which binary am I running?"*. Answered by the
  *one binary, two roles* breakdown in
  [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes)
  and the corresponding routing in
  [`TASKS.md ## configure`](TASKS.md#configure).
- **"How do I take a recommended parameter change from flow-tune
  back into my doca-flow application without breaking the
  dataplane?"** — worked example: *"the analyze step suggests a
  different table sizing for my pipe; how do I apply it?"*.
  Answered by the *recommendation → minimum-diff modification of
  the Flow program* loop in
  [`TASKS.md ## modify`](TASKS.md#modify) and the
  smoke-before-bulk rule in
  [`TASKS.md ## test`](TASKS.md#test).
- **"`doca_flow_tune` reports nothing / disagrees with the Flow
  app / cannot attach — what does that mean?"** — worked example:
  *"the tool runs but the visualize step produces an empty
  mermaid diagram"*. Answered by the layered error taxonomy in
  [`CAPABILITIES.md ## Error taxonomy`](CAPABILITIES.md#error-taxonomy)
  + [`TASKS.md ## debug`](TASKS.md#debug).

## Audience

This skill serves **external operators, performance engineers,
DOCA Flow application developers, and AI agents who need to
understand, characterize, or improve a running `doca-flow`
pipeline's behavior on the user's actual install and device**.
Concretely:

- A platform operator running a `doca-flow` service on BlueField
  who wants a read-only snapshot of which pipes exist and how
  their hardware / software counters are progressing before
  recommending any change.
- A performance engineer who already has a `doca-flow-perf`
  baseline number and wants to turn the *measurement* into an
  *optimization* — pick a tuning axis and identify which knob in
  the doca-flow program is the lever for it.
- A DOCA Flow application developer who wants the offline analyze
  + visualize loop to understand a pipe layout without
  re-instrumenting the Flow program.
- An AI agent driving the *"is this Flow pipeline behaving as
  expected, and would a non-mutating tuning hint help"* triage
  step before recommending any code change to the Flow program.

It is **not** for users debugging the `doca_flow_tune` source code,
**not** a substitute for the live public DOCA Flow Tune guide on
`docs.nvidia.com`, **not** the right place to learn the
`doca-flow` API (that audience belongs in
[`doca-flow`](../../libs/doca-flow/SKILL.md)), and **not** the
right place for baseline *measurement* methodology — that belongs
to [`doca-flow-perf`](../doca-flow-perf/SKILL.md).

`doca_flow_tune` is shipped as a **single tool** (one binary plus
its companion analyzer / visualizer scripts and JSON config
templates) — the historical *server* and *client* roles live
inside this one artifact, not in two separate executables. The
skill uses the same `kind: tool` three-file shape as the rest
of the bundle so the agent's task-verb contract
(`configure / build / modify / run / test / debug`) is uniform
across libraries, services, and tools.

## Language scope

This skill governs invocation, output interpretation, and
recommendation-to-code-change routing for the C / C++ DOCA Flow
application that `doca_flow_tune` observes. The tool itself is
not a programming target — there is no public API the agent is
supposed to link against; what the agent and the user do with the
tool is *configure JSON, run, read the outputs, propose minimum-
diff changes to the surrounding `doca-flow` program in the
program's own language*. For the `doca-flow` API the
recommendations route back into, see
[`doca-flow CAPABILITIES.md`](../../libs/doca-flow/CAPABILITIES.md);
for cross-language application patterns, see
[`doca-programming-guide`](../../doca-programming-guide/SKILL.md).

## When to load this skill

Load this skill when the user is — or the agent needs to — invoke
`doca_flow_tune` against a running or planned `doca-flow`
application (on host or BlueField Arm, or inside the public NGC
DOCA container with the matching Flow trace-build flavor) to
characterize, dump, visualize, analyze, or tune that pipeline.
Concretely:

- Picking *which* role of `doca_flow_tune` to engage (offline
  analyze / visualize on a captured config + state, online
  dumper / monitor against the live Flow app, or attach-to-app
  server-role usage when the Flow application links the
  documented tune server entry points).
- Picking *which* tuning axis to ask about (rule placement,
  resource hints / table sizing, or hardware-offload-mode) for a
  candidate workload.
- Picking *which* measurement axis to compare against (rule-install
  rate, lookup latency, hardware-counter delta) — the three are
  not interchangeable and the chosen axis should be the same one a
  prior `doca-flow-perf` baseline named.
- Capturing a documented before / after pair around a proposed
  Flow-program change (the documented JSON config file path, the
  command line, the DOCA version, the device, the as-deployed
  environment, the full unredacted dumper / analyzer / visualizer
  output).
- Diagnosing why a tune session produced empty output, a
  visualize step rendered a degenerate diagram, or an analyze
  recommendation does not match what the live counters say.

Do **not** load this skill for general DOCA orientation, Flow
program API work, install, or pure measurement methodology.
For those, route to
[`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md),
[`doca-flow`](../../libs/doca-flow/SKILL.md),
[`doca-setup`](../../doca-setup/SKILL.md), or
[`doca-flow-perf`](../doca-flow-perf/SKILL.md).

## What this skill provides

This is a **thin loader**. Substantive material lives in two
companion files:

- `CAPABILITIES.md` — what `doca_flow_tune` observes and
  recommends on: the unified-artifact decomposition (server role
  + client / consumer role inside one binary), the three-axis
  configuration model (tuning axis × measurement × scope: which
  pipe / port / app), the documented offline / online / attach
  modes, the JSON configuration-file shape (the publicly-shipped
  `flow_tune_cfg_public.json` template plus its hardware-only and
  software-only variants), the dumper / monitor / analyze /
  visualize output surfaces, the version overlay (this tool rides
  the `doca-flow` library version it observes; the canonical
  rules live in [`doca-version`](../../doca-version/SKILL.md)),
  the layered error taxonomy
  (config-syntax / attach-failed / pipe-not-found /
  measurement-unsound / recommendation-unactionable / version /
  cross-cutting), the observability posture (the tool *is* an
  observability primitive for the Flow pipeline), and the safety
  policy that makes any mutating application of a recommendation
  high-stakes because the recommendation lands in live Flow state.
- `TASKS.md` — step-by-step workflows for the in-scope task verbs:
  `install` (route to setup; the binary is shipped),
  `configure` (the three-axis decision + JSON config + mode pick),
  `build` (route to install; the binary is shipped), `modify`
  (apply a recommendation back to the Flow program via minimum-
  diff), `run` (the snapshot → analyze → visualize flow), `test`
  (the eval loop — warm-up, steady-state, before / after pair,
  client / server / Flow version match), `debug` (walk the error
  taxonomy layer by layer), `use` (the agent-side workflow for
  consuming flow-tune output), plus a `Deferred task verbs` block
  and a `Command appendix`.

The skill assumes a host where DOCA is already installed (or the
public NGC DOCA container is running) and a `doca-flow`
application is already created and validated per the
[`doca-flow`](../../libs/doca-flow/SKILL.md) skill. Without those
preconditions, the tune session has nothing to observe.

## What this skill deliberately does not ship

This skill is **agent guidance**, not a samples or scripts bundle.
To keep the boundary clean, it deliberately does not contain —
and pull requests should not add:

- **Verbatim flag inventories, subcommand names, JSON config field
  names, or default endpoint paths quoted as the contract.** The
  public DOCA Flow Tune guide on `docs.nvidia.com` (reached via
  [`doca-public-knowledge-map ## DOCA tools`](../../doca-public-knowledge-map/SKILL.md#doca-tools))
  and the installed `--help` on the user's version are the joint
  source of truth; the shipped `flow_tune_cfg*.json` templates on
  the user's install are the second source for the JSON schema.
  Copying them here pins the skill to one release and silently
  rots when the tool evolves.
- **Pre-baked example output (dumper CSV columns, analyzer JSON
  field names, visualizer mermaid output).** Output is install-,
  device-, firmware-, NUMA-, Flow-pipe-, and DOCA-version-specific;
  a captured example pinned to one platform misleads operators on
  a different platform / version.
- **Wrappers, parsers, or scripts** in any language that consume
  flow-tune output. The output formats are documented and the
  shipped `scripts/` directory on the user's install contains
  vendor-provided helpers (e.g. `flow_json_diff.py`,
  `flow_mermaid_diff.py`, `hw_counters_csv_analyzer.py`); if a
  user wants to script against the outputs, the right answer is
  *"read the shipped scripts on your installed version"*.
- **Pre-baked tuning recommendations.** Recommendations from this
  tool are install-, device-, firmware-, and workload-specific;
  shipping one for *"hairpin pipes"* or *"NAT pipes"* misleads
  operators applying it to a different pipe. The agent always
  re-derives the recommendation from the user's actual session.
- **A `samples/`, `templates/`, or `reference/` subtree.** Mock or
  incomplete tuning recipes in this skill's tree are misleading;
  operators read them as production-grade.

## Loading order

1. Read this `SKILL.md` first to confirm the user's question is
   in scope (the user actually wants to invoke `doca_flow_tune`
   against a `doca-flow` pipeline, not measure baseline perf or
   learn the Flow API).
2. **For what `doca_flow_tune` observes, the one-binary / two-role
   decomposition, the three-axis model, the version overlay, the
   error taxonomy, observability surface, and safety posture,
   see [CAPABILITIES.md](CAPABILITIES.md).**
3. **For the documented invocations and the snapshot → analyze →
   visualize → propose → smoke workflow — `install`, `configure`,
   `build`, `modify`, `run`, `test`, `debug`, `use` — see
   [TASKS.md](TASKS.md).**

## Related skills

- [`doca-flow`](../../libs/doca-flow/SKILL.md) — the **base
  library** whose pipeline this tool observes and tunes. The
  pipe / entry / rule surface flow-tune reports on is created by
  `doca-flow` program code; recommendations route back into that
  program via the universal modify-a-sample workflow.
- [`doca-flow-perf`](../doca-flow-perf/SKILL.md) — the sibling
  *measurement* tool. The rule is: `doca-flow-perf` measures
  baselines; `doca-flow-tune` recommends optimizations on top.
  An agent that reaches for tune without a baseline number from
  perf is optimizing in the dark; an agent that reaches for perf
  without a question is benchmarking for the sake of it.
- [`doca-flow-dpa-perf`](../doca-flow-dpa-perf/SKILL.md) — the
  DPA-offloaded variant of Flow perf. Relevant when the Flow
  pipeline the user is tuning runs through a DPA-offload path;
  the baseline comes from there, not from host-side
  `doca-flow-perf`.
- [`doca-flow-grpc-server`](../doca-flow-grpc-server/SKILL.md) —
  the remote-control gRPC surface for `doca-flow`. Programmatic
  Flow rule management lives there; flow-tune's recommendations
  may be applied through that surface when the operator's
  control plane is remote.
- [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md)
  — routing to the public DOCA Flow Tune page on `docs.nvidia.com`
  and the rest of the public DOCA documentation set.
- [`doca-version`](../../doca-version/SKILL.md) — the canonical
  version-detection chain, four-way match, NGC semantics, and
  headers-win-over-docs rule. The
  [`## Version compatibility`](CAPABILITIES.md#version-compatibility)
  overlay in this skill is a thin extension on top.
- [`doca-debug`](../../doca-debug/SKILL.md) — the cross-cutting
  debug ladder. Flow-tune surfaces *its own* error taxonomy; when
  the cause turns out to be below DOCA (driver, firmware, NUMA),
  the tune taxonomy hands off to `doca-debug`.
- [`doca-structured-tools-contract`](../../doca-structured-tools-contract/SKILL.md)
  — the bundle's detect → prefer → fall back → report contract.
  The Command appendix in [`TASKS.md`](TASKS.md) honors it.
- [`doca-setup`](../../doca-setup/SKILL.md) — env preparation,
  install verification, hugepages, NUMA, and the *I have no
  install yet* path with the public NGC DOCA container.
- [`doca-hardware-safety`](../../doca-hardware-safety/SKILL.md) —
  the cross-cutting hardware-safety meta-policy this skill's
  `## Safety policy` overlays.
