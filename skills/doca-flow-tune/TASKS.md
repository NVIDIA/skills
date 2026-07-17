# DOCA Flow Tune — Tasks

**Where to start:** The verbs that carry real workflow content are
`## configure`, `## run`, `## modify`, `## test`, `## debug`, and
`## use`. `## install` and `## build` are routing stubs because the
binary is shipped pre-built; `## modify` is substantive for this
tool because the *output* of a tune session is a recommendation
that gets applied back to the surrounding `doca-flow` program via
the universal modify-a-sample workflow.

This file is loaded by [`SKILL.md`](SKILL.md) after
[`CAPABILITIES.md`](CAPABILITIES.md). It walks the agent through
the in-scope task verbs and the agent-side `## use` workflow, then
explicitly defers verbs that do not belong here.

For `doca_flow_tune`, the verbs that carry real workflow content
are `configure`, `run`, `modify`, `test`, `debug`, and `use`. The
other two (`install`, `build`) carry meaningful **routing stubs**
because the binary is shipped pre-built.

## install

`doca_flow_tune` is **shipped pre-built** with the DOCA install
when the Flow tune package profile is present. There is no
separate install workflow this skill owns.

Routing for nearby "install" questions:

- *"The binary isn't there — do I need to install something?"* →
  yes. Route to
  [`doca-setup ## install`](../../doca-setup/TASKS.md#configure)
  for the DOCA install with the appropriate package profile, or
  to [`doca-setup ## no-install`](../../doca-setup/TASKS.md#no-install)
  for the public NGC DOCA container path when the user has no
  install yet.
- *"I want a structured probe before installing"* → the
  detect → prefer → fall back → report contract per
  [`doca-structured-tools-contract`](../../doca-structured-tools-contract/SKILL.md)
  is the right surface; the structured helpers it documents
  (`doca-env --json` for version + libraries + drivers in one
  shot) are the cheapest pre-install reconnaissance.

The `## What this skill deliberately does not ship` block in
[`SKILL.md`](SKILL.md) forbids adding an install recipe here;
revisit that policy before changing this section.

## configure

`configure` for `doca_flow_tune` is *"decide the three-axis
configuration AND pick the operating mode AND populate the JSON
config file BEFORE invoking the binary"*. The three axes
(tuning axis × measurement × scope) are laid out in
[`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes);
the three modes (offline / online read-only / online
state-changing) are owned there as well. Skipping this verb is
the canonical Flow-tune failure mode.

Steps the agent should walk the user through, in order:

1. **Confirm DOCA is installed and the Flow library is present.**
   Run [`doca-setup ## test`](../../doca-setup/TASKS.md#test) and
   confirm `pkg-config --modversion doca-flow` resolves. If the
   user has no install yet, route to
   [`doca-setup ## no-install`](../../doca-setup/TASKS.md#no-install)
   before any tune discussion. The tool has nothing to do if
   `doca-flow` is not present.
2. **Confirm a `doca-flow` application is running (online modes)
   or a captured config is on hand (offline mode).** Per
   [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes)
   operating-modes table; offline mode consumes a captured JSON
   only, online mode requires the application.
3. **Decide tune vs perf.** Per the *tune-vs-perf* boundary in
   [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes);
   if the user's question is *"what is the baseline number"*,
   route to [`doca-flow-perf`](../doca-flow-perf/SKILL.md) or
   [`doca-flow-dpa-perf`](../doca-flow-dpa-perf/SKILL.md) instead
   of this skill. Tune does not produce a primary baseline number.
4. **Axis 1 — pick the tuning axis.** Per the three-axis table in
   [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes),
   commit explicitly to which class of change is being explored
   (rule placement, resource hints / table sizing, hardware-
   offload mode). Naming one and dropping the others silently
   narrows the session.
5. **Axis 2 — pick the measurement.** Rule-install rate vs lookup
   latency vs hardware-counter delta. The measurement must match
   the tuning axis; the agent should quote *why* it picked the
   measurement so the user can challenge the framing.
6. **Axis 3 — pick the scope.** Confirm the PCI address set and
   the `flow_port_id` set the session will look at. The shipped
   `flow_tune_cfg_public.json` template documents the
   `monitor.hardware.pci_addresses` and
   `monitor.software[*].flow_port_id` fields as the scope-naming
   knobs; confirm against the live application's port set.
7. **Pick the operating mode.** Offline / online read-only /
   online state-changing per
   [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes).
   The state-changing mode is high-stakes; the agent must surface
   the [`CAPABILITIES.md ## Safety policy`](CAPABILITIES.md#safety-policy)
   rules before recommending it.
8. **Populate the JSON config file.** Start from the shipped
   `flow_tune_cfg_public.json` template on the user's install (or
   `flow_tune_cfg_hardware_only.json` /
   `flow_tune_cfg_software_only.json` for the subset variants);
   set `outputs_directory`, the `network.server_uds` socket path,
   the `dumper` / `analyze` / `visualize` / `monitor` sections
   per the chosen scope and mode. The shipped templates are the
   schema source of truth on the user's installed version; do
   not invent field names from prose memory.
9. **Sanity check before any invocation.** Confirm with the user:
   what is the question (tune vs perf)? Which pipeline? Which
   tuning axis? Which measurement? Which mode? If any answer is
   unclear, stop and ask — do NOT run the tune binary against a
   fuzzy posture.

If steps 1-9 read clean, proceed to [`## run`](#run). If any step
surfaces a precondition failure, walk the error taxonomy in
[`CAPABILITIES.md ## Error taxonomy`](CAPABILITIES.md#error-taxonomy)
before retrying.

## build

`doca_flow_tune` is **shipped pre-built** as part of every DOCA
install that includes the Flow tune package. There is no source
tree the external user is expected to compile, no build flags,
no `meson` or `make` workflow for the tool itself.

Routing for nearby "build" questions:

- *"The binary isn't there — do I need to build it?"* → no.
  Route to [`## install`](#install). The fix is to install (or
  re-install) DOCA with the right package profile.
- *"I want to build my own DOCA Flow tuning analyzer."* → not a
  `doca_flow_tune` question. Route to
  [`doca-programming-guide ## build`](../../doca-programming-guide/TASKS.md#build)
  for the cross-library build pattern and to
  [`doca-flow TASKS.md ## build`](../../libs/doca-flow/TASKS.md#build)
  for the Flow-specific build overlay; the tune binary is the
  shipped harness, a bespoke analyzer is a different artifact.
- *"I want the trace-build flavor of `doca-flow` so the tune
  attach role works."* → the trace-flavor question belongs to
  [`doca-flow TASKS.md ## build`](../../libs/doca-flow/TASKS.md#build);
  this skill does not duplicate the Flow build recipe.

The `## What this skill deliberately does not ship` block in
[`SKILL.md`](SKILL.md) forbids shipping a build recipe for the
tune binary or wrappers around it.

## modify

`modify` for `doca_flow_tune` is substantive: the *output* of a
tune session is a recommendation, and applying the recommendation
is a `doca-flow` *program* change. The agent does not modify the
shipped tune binary; the agent modifies the surrounding
`doca-flow` application's code via the universal modify-a-sample
workflow.

The pattern:

1. **Read the analyze / visualize output as evidence, not as
   answer.** Per
   [`CAPABILITIES.md ## Observability`](CAPABILITIES.md#observability),
   the analyze JSON and visualize mermaid file are *recommendations
   under named assumptions*; the named assumptions (chosen tuning
   axis, captured before-state, three-axis decision) are the
   provenance of the recommendation.
2. **Translate the recommendation into a Flow-program change
   proposal.** Per
   [`doca-flow TASKS.md ## modify`](../../libs/doca-flow/TASKS.md#modify),
   identify which Flow API call in the program is the lever the
   recommendation moves (a pipe attribute, a resource-hint
   parameter, a steering-mode selection). If the recommendation
   does not correspond to a knob the program exposes, surface
   that as
   [`CAPABILITIES.md ## Error taxonomy`](CAPABILITIES.md#error-taxonomy)
   layer 5 (recommendation-unactionable) rather than guessing at
   a wider modification.
3. **Apply the minimum diff.** Walk
   [`doca-programming-guide TASKS.md ## modify`](../../doca-programming-guide/TASKS.md#modify)
   for the cross-library minimum-diff modification pattern; the
   shipped DOCA Flow samples on disk are the verified source of
   truth, the tune recommendation is the *direction* of the diff
   on top.
4. **Re-run the tune session against the post-change state.**
   The before / after pair is the only defensible artifact; a
   recommendation applied without a captured before snapshot and
   a captured after snapshot cannot be evaluated. The
   eval-loop overlay in [`## test`](#test) enforces this.

Routing for nearby "modify" questions:

- *"Can I patch the `doca_flow_tune` binary itself?"* → out of
  scope. The shipped binary is the contract; modifying it is a
  contributor question, not an external-consumer question.
- *"Can I edit the dumper CSV format?"* → no; the documented
  CSV format is the documented format. If the user wants a
  different post-processing surface, the right answer is *"read
  the shipped `hw_counters_csv_analyzer.py` and adapt the
  parsing for your downstream tool"*, but even the adapter is
  out of scope per
  [`SKILL.md ## What this skill deliberately does not ship`](SKILL.md#what-this-skill-deliberately-does-not-ship).

## run

The snapshot → analyze → visualize → propose loop. The full
invocation surface lives in the public DOCA Flow Tune guide;
this section names the *shape* of the flow, not verbatim
command lines (per
[`CAPABILITIES.md ## Safety policy`](CAPABILITIES.md#safety-policy)
*"do not invent flags"*).

1. **Confirm the binary, version, and JSON config are
   consistent.** Per [`## configure`](#configure) steps 1-8.
   Quote `pkg-config --modversion doca-flow` and the tune
   binary's `--version` against
   [`doca-version TASKS.md ## test`](../../doca-version/TASKS.md#test)
   so the four-way match is anchored before invocation.
2. **Smoke run — offline first.** When a captured pipeline-
   description JSON is on hand, run the offline analyze /
   visualize chain against it as the cheapest read-only smoke.
   Goal: *the tool can parse the config, render the analyze
   output, and render the visualize output*. The smoke is not
   a usable recommendation; it is a precondition check.
3. **Online read-only snapshot.** When a live application is
   available, attach via the documented IPC channel (per the
   `network.server_uds` field on the user's installed
   templates), run the dumper / monitor for the configured
   duration, and capture the CSV + JSON + mermaid outputs into
   the directory named by `outputs_directory`. This is the
   *online-mode read-only smoke* per
   [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes)
   operating-modes table.
4. **Read the analyze and visualize outputs.** Per
   [`CAPABILITIES.md ## Observability`](CAPABILITIES.md#observability),
   the analyze JSON is the structured recommendation surface
   and the visualize mermaid is the human-readable pipe layout.
   Quote what they said — do not paraphrase the recommendation
   away.
5. **Inspect the dumper CSV for soundness.** Per
   [`CAPABILITIES.md ## Error taxonomy`](CAPABILITIES.md#error-taxonomy)
   layer 4, an exit-0 run with numbers is not enough — verify
   warm-up actually happened, verify the captured counters
   reflect steady state, verify outliers / distribution are
   captured alongside any single number.
6. **STOP here unless the user has committed to applying a
   recommendation.** Steps 1-5 are the read-only surface; the
   high-stakes step is [`## modify`](#modify) (apply the
   recommendation back to the Flow program) and is gated on
   the smoke in [`## test`](#test).
7. **For the exact, current invocation surface — mode flag
   names, JSON config field names, dumper CSV columns,
   analyze JSON schema, visualize mermaid format — read
   `--help`** on the installed binary and the shipped
   `flow_tune_cfg*.json` templates plus the public DOCA Flow
   Tune guide via
   [`doca-public-knowledge-map ## DOCA tools`](../../doca-public-knowledge-map/SKILL.md#doca-tools).
   Do **not** invent any of these.

When recording the run for downstream consumers, write down: the
DOCA version, the host (host x86 / Arm, BlueField Arm, or NGC
container), the full JSON config file used, the exact command
line, the device target (PCI address set), and the full
unredacted dumper / analyze / visualize output. The downstream
[`## test`](#test) and [`## debug`](#debug) workflows depend on
those fields.

## test

`doca_flow_tune` is **a recommendation tool**, so its `## test`
verb is about *testing the recommendation* — i.e. confirming the
before / after pair is sound and reproducible — not unit-testing
the binary itself.

**`## test` is an iterative loop, not a one-shot pass.** Every
mutation — a Flow-program change derived from a recommendation, a
driver / firmware change under the application, a BlueField mode
flip, a DOCA reinstall — re-opens the smoke.

The smoke-before-bulk shape:

1. **Capture a pre-change snapshot.** Run the online read-only
   smoke per [`## run`](#run) steps 3-5; the resulting CSV +
   analyze JSON + visualize mermaid is the *before* leg.
2. **Confirm the four-way version match.** Quote the tune
   binary's `--version`, the application-side
   `pkg-config --modversion doca-flow`, and the host package
   version per
   [`doca-version TASKS.md ## test`](../../doca-version/TASKS.md#test).
   A measurement under a partial install is not a measurement of
   anything the operator can defend.
3. **Apply the recommendation via [`## modify`](#modify).** The
   minimum-diff change to the Flow program lands in the
   application's source; redeploy the application on its
   documented cadence.
4. **Capture a post-change snapshot.** Re-run step 1 against the
   redeployed application; the resulting outputs are the
   *after* leg.
5. **Diff the two legs.** Use the shipped `flow_json_diff.py` /
   `flow_mermaid_diff.py` helpers under the tool's `scripts/`
   directory for the structural and counter diff; quote the diff
   the script produced, do not paraphrase it.
6. **Only after steps 1-5 read clean** may the agent declare the
   tuning iteration converged on a real change. If the after-leg
   measurement does not move in the direction the recommendation
   predicted, the next move is [`## debug`](#debug) — not a
   repeat of the same recommendation.

Eval-loop overlay (rows apply to every tune session):

| Iteration trigger | What it looks like | What changes next iteration |
| --- | --- | --- |
| Smoke completed; analyze recommendation does not match the question | Could be the wrong tuning axis was selected, the wrong measurement was selected, or the scope is wrong | Re-walk axes 1-3 in [`## configure`](#configure) before re-running |
| Online attach fails on a known-good Flow app | Attach-failed layer per [`CAPABILITIES.md ## Error taxonomy`](CAPABILITIES.md#error-taxonomy) layer 2; not a measurement-soundness issue | Walk [`## debug`](#debug) layer 2; do not loop on the dumper expecting different output |
| Before / after diff is empty when a Flow-program change shipped | Either the change did not deploy, or the captured before-leg snapshot was already the post-change state | Re-confirm the Flow application's running version, re-capture the before leg from a known-pre-change state |
| Recommendation says *"increase table size"* but the Flow program hard-codes the value | Recommendation-unactionable layer 5; not a tune bug | Modify the Flow program per [`doca-flow TASKS.md ## modify`](../../libs/doca-flow/TASKS.md#modify) to expose the knob, then re-run the tune session |
| After-leg measurement is in the opposite direction from the prediction | Could be a real regression the recommendation introduced, or could be a different workload running on the device | Re-confirm workload stability; quote the diff; route through [`doca-debug ## debug`](../../doca-debug/SKILL.md) layer 7 if the workload is steady |

This skill does **not** ship a "test fixture" or pre-recorded
expected output. The expected output is install-, device-,
firmware-, application-, and DOCA-version-specific. See
[`SKILL.md ## What this skill deliberately does not ship`](SKILL.md#what-this-skill-deliberately-does-not-ship).

## debug

When `doca_flow_tune` fails to attach, produces empty output,
produces a recommendation that does not move the measurement, or
disagrees with the Flow application's own view, walk the
[`CAPABILITIES.md ## Error taxonomy`](CAPABILITIES.md#error-taxonomy)
layers in order:

1. **Config-syntax.** Invocation / JSON config does not parse.
   Confirm flag / field exists on the installed binary and the
   shipped `flow_tune_cfg*.json` templates; confirm units /
   values are in the documented form. Do not infer from generic
   CLI / JSON intuition.
2. **Attach-failed (online mode).** Confirm the live Flow
   application is up via
   [`doca-flow TASKS.md ## test`](../../libs/doca-flow/TASKS.md#test);
   confirm the `network.server_uds` path matches the
   application's actual setting; confirm filesystem permissions
   and namespace boundary. Re-attempt only after the application
   is on the snapshot.
3. **Pipe-not-found.** Confirm the pipe set against the Flow
   application's own view per
   [`doca-flow TASKS.md ## modify`](../../libs/doca-flow/TASKS.md#modify);
   if the pipe is genuinely not created, the next move is the
   Flow program, not the tune binary.
4. **Measurement-unsound.** Walk the three sub-layers per
   [`CAPABILITIES.md ## Error taxonomy`](CAPABILITIES.md#error-taxonomy)
   layer 4 — warm-up / steady-state, outliers / distribution,
   before / after pair captured — before quoting any
   recommendation.
5. **Recommendation-unactionable.** Surface the recommendation
   as a Flow-program-change proposal per [`## modify`](#modify);
   do not loop on tune trying to find a knob the program does
   not expose.
6. **Version.** Walk
   [`doca-version TASKS.md ## debug`](../../doca-version/TASKS.md#debug)
   end-to-end with the four-way match captured. Common
   tune-specific symptom is a tune binary from a different DOCA
   train than the Flow application's linked library.
7. **Cross-cutting.** Hand off to
   [`doca-debug ## debug`](../../doca-debug/SKILL.md) for the
   cross-cutting debug ladder and
   [`doca-setup ## debug`](../../doca-setup/TASKS.md#debug) for
   the env-side layers.

In every case: **quote what the tool reported.** Do not
paraphrase the dumper CSV summary, do not reorder mermaid nodes,
do not summarize a recommendation into prose. The tune binary is
in the loop precisely to break the agent out of the
inference-from-datasheet trap.

## use

`## use` is the agent-side workflow for *consuming*
`doca_flow_tune` output as evidence inside a larger conversation
with the user — i.e. the workflow when the agent is the second
reader of a tune session the user (or a previous agent) ran.

The pattern:

1. **Read the four-tuple first.** Confirm the captured tune
   session names the command line + JSON config + DOCA version +
   device + as-deployed environment per
   [`CAPABILITIES.md ## Safety policy`](CAPABILITIES.md#safety-policy)
   *"quote the four-tuple"* rule. A captured session that omits
   any leg is evidence that cannot be re-checked, and the agent
   should say so rather than draw inferences from it.
2. **Read the analyze / visualize outputs as scoped
   recommendations.** Per
   [`CAPABILITIES.md ## Observability`](CAPABILITIES.md#observability),
   the analyze JSON's recommendation is conditioned on the
   chosen three-axis decision; an agent quoting the
   recommendation without re-stating the axes for the user is
   handing them an answer to an unstated question.
3. **Cross-check against the Flow application's own view.** Per
   the *quote, do not paraphrase* rule, the agent compares the
   captured tune session against the live Flow application's
   own counter / inspector surface per
   [`doca-flow CAPABILITIES.md ## Observability`](../../libs/doca-flow/CAPABILITIES.md#observability)
   when both are available. Disagreement is signal — it routes
   to [`## debug`](#debug) layer 6 (version) first.
4. **Route to [`## modify`](#modify) only after the agent
   confirms the recommendation maps to a knob the Flow program
   exposes.** Otherwise the recommendation is *recommendation-
   unactionable* per
   [`CAPABILITIES.md ## Error taxonomy`](CAPABILITIES.md#error-taxonomy)
   layer 5; the right next step is a Flow-program proposal, not
   a tune re-run.
5. **Save the tune session alongside the perf baseline.** The
   tune recommendation is downstream of the perf baseline per
   [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes)
   tune-vs-perf table; quoting the recommendation without the
   baseline is half the picture and the
   [`doca-debug TASKS.md ## debug`](../../doca-debug/TASKS.md#debug)
   ladder cannot pick up where the agent left off without both.

## Deferred task verbs

The verbs below are not `doca_flow_tune` work and should be
routed out before the agent does any of them under this skill's
name.

- **measure a Flow pipeline baseline** ⇒
  [`doca-flow-perf`](../doca-flow-perf/SKILL.md) (host-side) or
  [`doca-flow-dpa-perf`](../doca-flow-dpa-perf/SKILL.md) (DPA-
  offloaded). Tune does not produce a primary baseline number.
- **write a doca-flow application** ⇒
  [`doca-flow`](../../libs/doca-flow/SKILL.md), layered on
  [`doca-programming-guide`](../../doca-programming-guide/SKILL.md).
  Tune observes pipelines the program created; it does not
  create them.
- **install DOCA** ⇒
  [`doca-setup ## install`](../../doca-setup/TASKS.md#configure)
  and [`## no-install`](../../doca-setup/TASKS.md#no-install).
- **library-internal pipe / counter / inspector deep dive** ⇒
  [`doca-flow`](../../libs/doca-flow/SKILL.md). Tune transports
  the data the Flow library exposes programmatically; the deeper
  per-pipe semantics belong to the library.
- **streaming telemetry export of Flow KPIs to a production
  consumer** ⇒ not a `doca_flow_tune` feature, and tune is not
  positioned as a production telemetry surface. The DOCA
  Telemetry Service (DTS) is the documented telemetry surface;
  routing belongs in
  [`doca-public-knowledge-map ## DOCA services`](../../doca-public-knowledge-map/SKILL.md#doca-services).

## Command appendix

`doca_flow_tune`-specific invocation classes the verbs above
reach for. Every row is a CLASS — the agent must not invent
flags, JSON field names, or recommendation identifiers beyond
`--help` on the installed binary, the shipped
`flow_tune_cfg*.json` templates, and the public DOCA Flow Tune
guide.

**Infra-aware preamble (every row below).** Per the bundle's
detect → prefer → fall back → report contract documented in
[`doca-structured-tools-contract ## The agent behavior contract`](../../doca-structured-tools-contract/SKILL.md#the-agent-behavior-contract),
the agent should:

1. Probe for the matching structured helper FIRST (`doca-env --json`
   for version + devices + libraries + drivers + hugepages;
   `doca-capability-snapshot` for per-device capability flags;
   `version-matrix.json` for *"available since"* lookups).
2. If the probe succeeds, the structured tool's output is the
   authoritative answer and the agent SHOULD NOT also run the
   manual command in the row below. Report *"using structured
   `<tool>`"*.
3. If the probe fails, fall back to the manual command in the
   row. Report *"falling back to manual chain"*.
4. The schemas the structured tools emit are defined in
   [`doca-structured-tools-contract ## Schemas`](../../doca-structured-tools-contract/SKILL.md#schemas);
   the version-handling semantics are owned by
   [`doca-version`](../../doca-version/SKILL.md).

| Purpose (class) | Invocation (shape) | Owning step | Reads as healthy when … |
| --- | --- | --- | --- |
| Discover the documented flag surface | `doca_flow_tune --help` plus the public DOCA Flow Tune guide for long-form documentation | [`## configure`](#configure) step 4-7; [`## debug`](#debug) layer 1 | Prints the documented mode-flag inventory; the agent uses `--help` as the only source of truth for mode / scope flag names. |
| Read the shipped JSON config templates | Read `flow_tune_cfg_public.json`, `flow_tune_cfg_hardware_only.json`, `flow_tune_cfg_software_only.json` on the user's install | [`## configure`](#configure) step 8 | The templates' field set is the schema source of truth for the JSON the operator writes; do not infer fields not present in the templates. |
| Confirm the Flow library version on the application side | `pkg-config --modversion doca-flow` on the side the Flow app runs | [`## configure`](#configure) step 1; [`## test`](#test) step 2; [`## debug`](#debug) layer 6 | Matches the version reported by the tune binary's `--version`; disagreement = partial install (route to [`doca-version TASKS.md ## debug`](../../doca-version/TASKS.md#debug) layer 2). |
| Run offline analyze / visualize against a captured config | The documented offline-mode invocation against a captured pipeline-description JSON | [`## run`](#run) step 2 | The analyze JSON and visualize mermaid render without error; the captured config is the only input. |
| Snapshot a live Flow pipeline (online read-only) | The documented online-mode invocation against a JSON config naming `network.server_uds` and the scope | [`## run`](#run) step 3 | The dumper CSV, analyze JSON, and visualize mermaid land in the `outputs_directory`; live Flow state is not changed. |
| Diff two captured tune sessions | `scripts/flow_json_diff.py` and `scripts/flow_mermaid_diff.py` from the shipped `scripts/` directory | [`## test`](#test) step 5 | The diff scripts produce a structural and counter diff that the agent quotes verbatim; never paraphrased. |
| Post-process the dumper CSV | `scripts/hw_counters_csv_analyzer.py` from the shipped `scripts/` directory | [`## run`](#run) step 5 | The analyzer reports per-counter statistics over the captured CSV; the agent quotes the analyzer's output. |
| Save a session snapshot for debug | Capture the full `outputs_directory` (CSV + JSON + mermaid + log files) plus the four-tuple of (cmd line, JSON config, DOCA version, device, env) | [`## test`](#test) step 1 + [`doca-debug TASKS.md ## debug`](../../doca-debug/TASKS.md#debug) | The saved bundle is consumed by the cross-cutting debug ladder as the Flow-side evidence pair. |

Three cross-cutting rules for this appendix:

- **Never invent a `doca_flow_tune` flag, mode name, JSON
  config field name, or recommendation identifier.** `--help` on
  the installed binary plus the public DOCA Flow Tune guide via
  [`doca-public-knowledge-map ## DOCA tools`](../../doca-public-knowledge-map/SKILL.md#doca-tools)
  plus the shipped `flow_tune_cfg*.json` templates on the user's
  install are the joint contract.
- **Smoke before bulk.** Every row presumes the offline smoke or
  the online read-only smoke succeeded first; a state-changing
  application of a recommendation without the smoke is the
  canonical dataplane-disruption failure mode.
- **Cross-link instead of duplicate.** Cross-cutting commands
  (`pkg-config --modversion`, `doca_caps --list-devs`, `dmesg`)
  live in [`doca-debug TASKS.md ## Command appendix`](../../doca-debug/TASKS.md);
  Flow-application-side build / port / pipe commands live in
  [`doca-flow TASKS.md ## Command appendix`](../../libs/doca-flow/TASKS.md);
  the perf-side measurement commands live in
  [`doca-flow-perf`](../doca-flow-perf/SKILL.md) and
  [`doca-flow-dpa-perf`](../doca-flow-dpa-perf/SKILL.md); this
  appendix names only `doca_flow_tune`-specific invocations on
  top.

## Cross-cutting

A few rules that apply across every verb in this file, restated
here so they are visible at the point of action and not buried
in [`SKILL.md`](SKILL.md):

- The **public DOCA Flow Tune guide** plus the installed
  `--help` plus the shipped `flow_tune_cfg*.json` templates are
  the joint source of truth. When they disagree (e.g. a field
  landed in a release this skill was not written against), the
  *installed* templates win for the user's actual session.
- `doca_flow_tune` *is* one binary with two internal roles.
  The agent must say which role a session uses (offline /
  online read-only / online state-changing) before invocation
  and must surface that there is no separate *"server package"*
  and *"client package"* to install.
- **Quote the four-tuple, not just the recommendation.** Command
  line + JSON config + DOCA version + device + as-deployed
  environment is the minimum unit a tune recommendation is
  meaningful in.
- The **state-changing application of a recommendation is
  high-stakes**; the agent must label every operation as
  read-only (offline / online read-only) or state-changing
  (online application of recommendation via Flow-program
  modify-a-sample) and gate every state-changing operation on
  a clean smoke per [`## test`](#test).
- This skill **assumes a healthy DOCA install** (or the public
  NGC DOCA container) and a working `doca-flow` application
  (or a captured configuration of one). If either is in doubt,
  route to
  [`doca-setup`](../../doca-setup/SKILL.md) and
  [`doca-flow`](../../libs/doca-flow/SKILL.md) before running
  anything else here. For baselines that the recommendations
  optimize on top of, load
  [`doca-flow-perf`](../doca-flow-perf/SKILL.md) and / or
  [`doca-flow-dpa-perf`](../doca-flow-dpa-perf/SKILL.md)
  alongside.
