---
license: Apache-2.0
name: doca-flow-tune
description: >
  Use this skill when the user is tuning a live or captured `doca-flow`
  pipeline with `doca_flow_tune` — snapshotting pipe / counter / KPI
  state, picking a tuning axis (rule placement, resource hints / table
  sizing, HW-offload mode) and a matching measurement (rule-install
  rate, lookup latency, hardware-counter delta), running offline or
  online (read-only or state-changing) modes, reading the dumper CSV /
  analyze JSON / visualize mermaid, or applying a recommendation back
  into the Flow program. Trigger even when the user does not explicitly
  mention "doca_flow_tune" — typical implicit phrasings include "Flow
  rule-install rate is low on BlueField", "table sizing looks wrong for
  this pipe", "tune visualize step is empty", "before/after counters
  don't move", or "which doca-flow knob does this recommendation hit".
  Refuse and route elsewhere for measuring baseline numbers
  (doca-flow-perf), writing the doca-flow
  application, DOCA install, or streaming Flow telemetry — those belong
  to other skills.
metadata:
  author: NVIDIA <doca@nvidia.com>
  kind: tool
  doca_version: "3.5"
  tags:
    - doca
    - flow
    - tune
compatibility: >
  Requires DOCA SDK installed at /opt/mellanox/doca on Linux (Ubuntu
  22.04/24.04 or RHEL/SLES) with a BlueField DPU or ConnectX NIC
  attached, plus a running or captured `doca-flow` application to
  observe. Reads the user's local install via `pkg-config doca-flow` and
  the shipped `flow_tune_cfg*.json` templates and `scripts/` directory
  under /opt/mellanox/doca. The validation steps require a live DOCA
  install.
---

# DOCA Flow Tune (`doca_flow_tune`)

## Examples

- **"Flow rule-install rate is low on BlueField."** — this is
  in scope. Snapshot then pick a tuning axis; do not replace
  this with a flow-perf baseline run.
- **"Tune visualize step is empty."** — check the dumper JSON/CSV /
  analyze JSON before changing the Flow program.

## Instructions

Follow Loading order below. Open the tasks companion for configure /
tune / debug. Open the capabilities companion for online vs
offline modes. Do not open every companion file.

> **Command surface (DOCA 3.5 public Flow Tune guide).**
> `doca_flow_tune` is a single binary whose **role on a given
> invocation is determined by which of three top-level
> commands** the user picks — `monitor`, `analyze`,
> `visualize` (confirm with `--help` on the installed
> version). `dump`, `web`, `import`, and `sim_timing` are not
> in the 3.5 guide; do not invoke them.
> Counter dumping is part of `monitor` (`--enable-dump`, or the
> `background` sub-command for dumping without screen output),
> not a separate command; `analyze` accepts the `export` and
> `packet-trace` sub-commands (hyphenated). `monitor` and
> `analyze` run the binary in **server-attached online mode**
> against a live `doca-flow` application reached over a Unix-
> domain socket whose path lives in `network.server_uds` of the
> shipped `flow_tune_cfg*.json`; `visualize` (and offline
> analysis of a previously exported pipeline file) runs in
> **offline / captured-snapshot mode** against JSON / CSV files
> the online commands previously dropped into the
> configured `outputs_directory`. The rest of this skill (and
> the capabilities companion / the tasks companion)
> uses the legacy *"server role / online mode / offline mode"*
> framing — that framing is internally consistent with the
> command surface here: *server role* = a server-attached
> online command (`monitor`, `analyze export` / `analyze
> packet-trace`); *online mode* = any of those; *offline mode* =
> `visualize` over captured files. Treat the command name as the
> primary handle; treat *server/online/offline* as the
> downstream behavioral consequence of the subcommand pick.

**Where to start:** This is a tool skill for invoking `doca_flow_tune`,
the unified DOCA Flow tuning tool. Open the tasks companion and
start at `## configure` to commit to the
three-axis decision (target Flow pipeline × tuning axis ×
measurement) and pick offline vs online vs server-attach mode, then
`## run` for the snapshot → analyze → visualize
loop, then `## test` for the smoke-before-bulk
overlay that gates any state-changing application of a tuning
recommendation back into the Flow application's code. Open
the capabilities companion when the question is *what
state `doca_flow_tune` can observe and recommend on*, *how its
server / client roles fit inside the single artifact*, *which DOCA
version the tool ships in*, or *how to interpret the dumper / monitor
/ analyze / visualize outputs without fooling yourself*. If DOCA is
not installed, route to
`doca-setup/SKILL.md` first; if the user has
no running `doca-flow` application yet, route to
`doca-flow/SKILL.md` — flow-tune does not
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
  the capabilities companion
  and the routing into
  `doca-flow-perf/SKILL.md` for baselines vs
  this skill for optimization on top of a measured baseline.
- **"Capture a snapshot of a live `doca-flow` pipeline's hardware
  and software counters without touching the dataplane."** — worked
  example: *"I want a side-effect-free dumper / monitor run against
  the running Flow ports for an operations-rate profile"*. Answered
  by the snapshot flow in
  the tasks companion plus the read-only-by-default
  posture in
  the capabilities companion.
- **"Pick the right tuning axis — rule placement, resource hints,
  or hardware-offload mode — for the question I actually have."**
  — worked example: *"my Flow pipe's rule-install rate is low; is
  this a placement question or a table-sizing question?"*. Answered
  by the three-axis configuration in
  the capabilities companion
  + the configure walk in
  the tasks companion.
- **"How do `doca_flow_tune`'s server role and client / consumer
  role fit together inside the single artifact?"** — worked
  example: *"I keep reading about a Flow Tune server and a Flow
  Tune client; which binary am I running?"*. Answered by the
  *one binary, two roles* breakdown in
  the capabilities companion
  and the corresponding routing in
  the tasks companion.
- **"How do I take a recommended parameter change from flow-tune
  back into my doca-flow application without breaking the
  dataplane?"** — worked example: *"the analyze step suggests a
  different table sizing for my pipe; how do I apply it?"*.
  Answered by the *recommendation → minimum-diff modification of
  the Flow program* loop in
  the tasks companion and the
  smoke-before-bulk rule in
  the tasks companion.
- **"`doca_flow_tune` reports nothing / disagrees with the Flow
  app / cannot attach — what does that mean?"** — worked example:
  *"the tool runs but the visualize step produces an empty
  mermaid diagram"*. Answered by the layered error taxonomy in
  the capabilities companion
  + the tasks companion.

## Audience

This skill serves **external operators, performance engineers,
DOCA Flow application developers, and AI agents who need to
understand, characterize, or improve a running `doca-flow`
pipeline's behavior on the user's actual install and device**.
Concretely:

- An operator wanting a read-only snapshot of which pipes exist
  and how their hardware / software counters progress before
  recommending any change.
- A performance engineer who has a `doca-flow-perf` baseline and
  wants to turn *measurement* into *optimization* — pick a tuning
  axis and identify which knob in the doca-flow program is the
  lever.
- A Flow developer using the offline analyze + visualize loop to
  understand a pipe layout without re-instrumenting the program.
- An AI agent driving the *"is this pipeline behaving as expected,
  and would a non-mutating tuning hint help"* triage step.

It is **not** for users debugging the `doca_flow_tune` source code,
**not** a substitute for the live public DOCA Flow Tune guide,
**not** the place to learn the `doca-flow` API (see
`doca-flow/SKILL.md`), and **not** the
place for baseline *measurement* methodology — that belongs to
`doca-flow-perf/SKILL.md`.

`doca_flow_tune` is shipped as a **single tool** (one binary plus
companion analyzer / visualizer scripts and JSON config
templates) — the historical *server* and *client* roles live
inside this one artifact, not in two executables. It uses the
same `kind: tool` three-file shape as the rest of the bundle.

## Language scope

This skill governs invocation, output interpretation, and
recommendation-to-code-change routing for the DOCA Flow
application `doca_flow_tune` observes. The tool is not a
programming target — there is no public API to link against; the
workflow is *configure JSON, run, read outputs, propose minimum-
diff changes to the surrounding `doca-flow` program*. For the
`doca-flow` API the recommendations route back into, see
`doca-flow/SKILL.md`;
for cross-language application patterns, see
`doca-programming-guide/SKILL.md`.

## When to load this skill

Load this skill when the user is — or the agent needs to — invoke
`doca_flow_tune` against a running or planned `doca-flow`
application to characterize, dump, visualize, analyze, or tune
that pipeline. Concretely:

- Picking *which* role to engage (offline analyze / visualize on
  a captured config + state, online dumper / monitor against the
  live app, or attach-to-app server-role usage).
- Picking *which* tuning axis to ask about (rule placement,
  resource hints / table sizing, hardware-offload-mode).
- Picking *which* measurement axis to compare against
  (rule-install rate, lookup latency, hardware-counter delta) —
  not interchangeable; use the axis a prior `doca-flow-perf`
  baseline named.
- Capturing a documented before / after pair around a proposed
  Flow-program change (JSON config path, command line, DOCA
  version, device, environment, unredacted tool output).
- Diagnosing an empty session, a degenerate visualize diagram, or
  an analyze recommendation that contradicts live counters.

Do **not** load this skill for general DOCA orientation, Flow
program API work, install, or pure measurement methodology —
route to `doca-public-knowledge-map/SKILL.md`,
`doca-flow/SKILL.md`,
`doca-setup/SKILL.md`, or
`doca-flow-perf/SKILL.md`.

## What this skill provides

This is a **thin loader**. Substantive material lives in two
companion files:

- the capabilities companion — what `doca_flow_tune` observes and
  recommends on: the unified-artifact decomposition (server role
  + client / consumer role inside one binary), the three-axis
  configuration model (tuning axis × measurement × scope: which
  pipe / port / app), the documented offline / online / attach
  modes, the JSON configuration-file shape (the publicly-shipped
  `flow_tune_cfg_public.json` template plus its hardware-only and
  software-only variants), the dumper / monitor / analyze /
  visualize output surfaces, the version overlay (this tool rides
  the `doca-flow` library version it observes; the canonical
  rules live in `doca-version/SKILL.md`),
  the layered error taxonomy
  (config-syntax / attach-failed / pipe-not-found /
  measurement-unsound / recommendation-unactionable / version /
  cross-cutting), the observability posture (the tool *is* an
  observability primitive for the Flow pipeline), and the safety
  policy that makes any mutating application of a recommendation
  high-stakes because the recommendation lands in live Flow state.
- the tasks companion — step-by-step workflows for the in-scope task verbs:
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
`doca-flow/SKILL.md` skill. Without those
preconditions, the tune session has nothing to observe.

## What this skill deliberately does not ship

This skill is **agent guidance**, not a samples or scripts bundle.
To keep the boundary clean, it deliberately does not contain —
and pull requests should not add:

- **Verbatim flag inventories, subcommand names, JSON config field
  names, or default endpoint paths quoted as the contract.** The
  public DOCA Flow Tune guide (reached via
  `doca-public-knowledge-map ## DOCA tools` (`doca-public-knowledge-map/SKILL.md#doca-tools`))
  and the installed `--help` are the joint source of truth; the
  shipped `flow_tune_cfg*.json` templates are the second source for
  the JSON schema. Copying them here rots when the tool evolves.
- **Pre-baked example output (dumper CSV columns, analyzer JSON
  field names, visualizer mermaid output).** Output is install-,
  device-, firmware-, and version-specific; a captured example
  pinned to one platform misleads operators elsewhere.
- **Wrappers, parsers, or scripts** that consume flow-tune output.
  The shipped `scripts/` directory on the user's install contains
  vendor-provided helpers (e.g. `flow_json_diff.py`,
  `flow_mermaid_diff.py`, `hw_counters_csv_analyzer.py`); the right
  answer is *"read the shipped scripts on your installed version"*.
- **Pre-baked tuning recommendations.** Recommendations are
  install-, device-, firmware-, and workload-specific; the agent
  always re-derives them from the user's actual session.
- **A `samples/`, `templates/`, or `reference/` subtree.** Mock
  tuning recipes read as production-grade and mislead.

## Loading order

1. Read this skill first to confirm the user's question is
   in scope (the user actually wants to invoke `doca_flow_tune`
   against a `doca-flow` pipeline, not measure baseline perf or
   learn the Flow API).
2. **For what `doca_flow_tune` observes, the one-binary / two-role
   decomposition, the three-axis model, the version overlay, the
   error taxonomy, observability surface, and safety posture,
   see the capabilities companion.**
3. **For the documented invocations and the snapshot → analyze →
   visualize → propose → smoke workflow — `install`, `configure`,
   `build`, `modify`, `run`, `test`, `debug`, `use` — see
   the tasks companion.**

## Related skills

- `doca-flow/SKILL.md` — the **base
  library** whose pipeline this tool observes and tunes. The
  pipe / entry / rule surface flow-tune reports on is created by
  `doca-flow` program code; recommendations route back into that
  program via the universal modify-a-sample workflow.
- `doca-flow-perf/SKILL.md` — the sibling
  *measurement* tool. The rule is: `doca-flow-perf` measures
  baselines; `doca-flow-tune` recommends optimizations on top.
  An agent that reaches for tune without a baseline number from
  perf is optimizing in the dark; an agent that reaches for perf
  without a question is benchmarking for the sake of it.
- `doca-public-knowledge-map/SKILL.md`
  — routing to the public DOCA Flow Tune page on `docs.nvidia.com`
  and the rest of the public DOCA documentation set.
- `doca-version/SKILL.md` — the canonical
  version-detection chain, four-way match, NGC semantics, and
  headers-win-over-docs rule. The
  `## Version compatibility`
  overlay in this skill is a thin extension on top.
- `doca-debug/SKILL.md` — the cross-cutting
  debug ladder. Flow-tune surfaces *its own* error taxonomy; when
  the cause turns out to be below DOCA (driver, firmware, NUMA),
  the tune taxonomy hands off to `doca-debug`.
- `doca-structured-tools-contract/SKILL.md`
  — the bundle's detect → prefer → fall back → report contract.
  The Command appendix in the tasks companion honors it.
- `doca-setup/SKILL.md` — env preparation,
  install verification, hugepages, NUMA, and the *I have no
  install yet* path with the public NGC DOCA container.
- `doca-hardware-safety/SKILL.md` —
  the cross-cutting hardware-safety meta-policy this skill's
  `## Safety policy` overlays.
