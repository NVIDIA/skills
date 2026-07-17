# DOCA Flow Perf — Tasks

**Where to start:** The verbs that carry real workflow
content for `doca_flow_perf` are `## configure`, `## run`,
`## test`, and `## debug`. The other verbs (`## install`,
`## build`, `## modify`, `## use`) carry routing stubs or a
tightly-scoped agent-side workflow, because `doca_flow_perf`
is a shipped binary plus a shipped library of JSON policy
exemplars, not a source artifact the user compiles or
patches.

This file is loaded by [`SKILL.md`](SKILL.md) after
[`CAPABILITIES.md`](CAPABILITIES.md). It walks the agent
through the task verbs every artifact in this bundle exposes.

## install

`doca_flow_perf` is **shipped pre-built** with the DOCA
install when the Flow Perf component is present in the
install profile. There is no separate install workflow this
skill owns.

Routing for nearby "install" questions:

- *"The binary isn't on my system — do I need to install
  something?"* → yes. Route to
  [`doca-setup ## install`](../../doca-setup/TASKS.md#configure)
  or to [`doca-setup ## no-install`](../../doca-setup/TASKS.md#no-install).
  When the DOCA install profile excluded the perf tools,
  the binary will be absent even on a healthy DOCA install.
- *"The `configs/` JSON library is missing."* → same
  routing — same install component owns both.
- *"I want to install a newer version of flow-perf without
  upgrading DOCA."* → flow-perf is versioned WITH DOCA per
  [`CAPABILITIES.md ## Version compatibility`](CAPABILITIES.md#version-compatibility);
  there is no independent upgrade path. Route the
  cross-release decision through
  [`doca-version TASKS.md ## configure`](../../doca-version/TASKS.md#configure).

## configure

`configure` for `doca_flow_perf` is *"commit to the
three-axis decision AND pick the policy JSON AND pick the
backend AND capture the four-tuple BEFORE running the tool"*.
Skipping any of those four is the canonical failure mode.

Steps the agent should walk the user through, in order:

1. **Confirm DOCA is installed and Flow is healthy.** Run
   [`doca-setup ## test`](../../doca-setup/TASKS.md#test);
   then confirm the surrounding `doca-flow` library is
   healthy on the device per
   [`doca-flow TASKS.md ## test`](../../libs/doca-flow/TASKS.md#test).
   flow-perf is a microbenchmark of `doca-flow`; if the
   library is not healthy, the measurement is meaningless.
2. **Commit to the three-axis decision.** Per
   [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes):
   - *pipeline shape* — how many ports, how many pipes, what
     match / action shape (the workload's "skeleton");
   - *traffic class* — what fields vary across entries
     (5-tuple drop? 5-tuple hairpin? VXLAN decap? src-IP
     plus shared-counter?), and which fields are `fixed`
     vs. `increase`;
   - *measurement axis* — rule install rate, rule delete
     rate, or per-entry query rate.
3. **Pick the JSON policy.** The shipped `configs/`
   directory is the schema by example; the operator picks
   the closest canned policy AND tells the agent which
   one. If no canned policy matches, the operator copies
   the closest one and edits — the agent walks them
   through which fields are safe to change (entry count,
   value steps) versus which fields change the workload
   class entirely (match field structure, action list).
4. **Pick the backend.** DPDK or DOCA. Per
   [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes)
   (Backends): the JSON contract is shared, the
   measurement path is not. The operator MUST commit to
   one and report it; the agent must NOT silently pick a
   default.
5. **Pick the workload knobs.** Number of workers, queue
   size, burst size, number of iterations. These are
   first-class methodology variables; the agent must
   confirm them before any run and capture them in the
   four-tuple. The shipped exemplars provide default-ish
   values for the policy shape, not for a specific number
   of workers — that is always operator-specified.
6. **Pick the BlueField mode and confirm preconditions.**
   Per [`doca-version`](../../doca-version/SKILL.md), the
   device mode is load-bearing for which matchers / actions
   the underlying Flow library exposes. The agent
   cross-checks the JSON policy against the device-mode
   preconditions before run-time, not after a failed run.
7. **Capture the four-tuple.** Per
   [`CAPABILITIES.md ## Version compatibility`](CAPABILITIES.md#version-compatibility),
   every reported flow-perf number must come with: DOCA
   version, BlueField / ConnectX generation, firmware
   version, the exact JSON policy contents, the chosen
   backend, the worker / queue / burst / iteration config.
   The agent captures all of those BEFORE running, not
   after.
8. **Sanity check before any invocation.** Confirm with the
   user: which JSON policy? Which backend? How many workers
   / how many iterations? Is the device dedicated to this
   measurement (no production `doca-flow` application
   running)? OOB access available? If any answer is
   unclear, stop and ask.

Do not invent CLI flag spellings, JSON schema keys, or
default numeric workload sizes. The public DOCA Flow Perf
guide on `docs.nvidia.com`, the installed binary's `--help`,
and the shipped `configs/` JSONs are the joint source of
truth.

## build

`doca_flow_perf` is **shipped pre-built** as part of every
DOCA install that includes the Flow Perf component. There is
no source tree the external user is expected to compile, no
build flags, no `meson` or `make` workflow for the binary
itself.

Routing for nearby "build" questions:

- *"The binary isn't there — do I need to build it?"* → no.
  Route to [`## install`](#install).
- *"I want to add a custom backend to flow-perf."* → out of
  scope; this skill is for external operators consuming the
  shipped tool, not contributors building their own
  backends.
- *"Build my application that flow-perf measures."* → not
  a flow-perf task at all — flow-perf measures its own
  synthetic JSON-described pipeline. If the operator wants
  to measure a LIVE application instead, route to
  [`doca-flow-tune`](../doca-flow-tune/SKILL.md).

## modify

**Do not modify the shipped `doca_flow_perf` binary or the
shipped `configs/` JSON exemplars in place.** They are
references; modifying them in place breaks the version
overlay rule per
[`CAPABILITIES.md ## Version compatibility`](CAPABILITIES.md#version-compatibility)
and makes a published number un-comparable across
machines / installs.

What the agent *does* modify, every time, is the **operator's
copy of a chosen `configs/` JSON** when the canned exemplars
don't quite express the workload. The agent walks the user
through:

1. Copy the closest canned exemplar to a per-run working
   file (do NOT edit the shipped exemplar);
2. Edit only fields the agent can point to in the shipped
   exemplar — i.e. fields with a documented spelling. Common
   safe edits: `num_inserted_entries` (workload size),
   `val` (the starting value of an increasing field),
   `steps` (the step size). Common unsafe edits: adding new
   match fields not demonstrated by any shipped exemplar
   (the agent refuses to invent the JSON spelling).
3. Run the single-iteration smoke per [`## run`](#run) on
   the edited JSON FIRST before scaling up.
4. Capture the edited JSON contents in the four-tuple per
   [`## configure`](#configure) step 7.

Routing for nearby "modify" questions:

- *"Patch the binary to add a custom matcher."* → out of
  scope; that is contributor work, not external-consumer
  work.
- *"Modify the live `doca-flow` application I have
  deployed."* → out of scope here; route to
  [`doca-flow-tune`](../doca-flow-tune/SKILL.md) for the
  optimization workflow, or
  [`doca-flow TASKS.md ## modify`](../../libs/doca-flow/TASKS.md#modify)
  for the library-level modification workflow.
- *"Switch the workload from rule-install measurement to
  packet throughput measurement."* → out of scope of
  flow-perf entirely; flow-perf does NOT measure
  packet-side throughput per
  [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes)
  (What is NOT measured).

## run

The start → smoke → measure flow. The full invocation surface
lives in the public DOCA Flow Perf guide; this section names
the *shape* of the flow.

1. **Confirm preconditions.** Per
   [`## configure`](#configure) steps 1-7.
2. **Run ONE single-iteration smoke** on the chosen JSON
   policy with a SMALL `num_inserted_entries` (a few
   thousand, not millions) and a SMALL iteration count
   (e.g. 1). This is the **smoke before the bulk**; per
   [`CAPABILITIES.md ## Safety policy`](CAPABILITIES.md#safety-policy)
   it is non-negotiable. The agent must NOT skip it.
3. **Confirm the smoke ran cleanly.** Per
   [`## test`](#test) layer 1 — the tool printed an
   iteration summary, `num_failed == 0`, and the device
   has not entered a degraded state per the
   `doca-flow` library's own view.
4. **Scale up to the operator's chosen workload.** Increase
   `num_inserted_entries`, increase iteration count,
   increase worker count — but ONLY after the smoke is
   clean. Capture the full output per
   [`## test`](#test) (capture step).
5. **Repeat the chosen scaled run multiple times to
   establish variance.** A single Kops/sec number from a
   single run is NOT defensible — per
   [`CAPABILITIES.md ## Error taxonomy`](CAPABILITIES.md#error-taxonomy)
   layer 5, the agent expects the operator to repeat the
   measurement and report variance.

When recording the run for downstream consumers, write down:
the DOCA version, the host, the BlueField / ConnectX
generation, the firmware version, the BlueField mode, the
JSON policy (verbatim), the backend choice (DPDK or DOCA),
the worker / queue / burst / iteration configuration, the
raw per-iteration cycle output, and the derived Kops/sec.
Partial captures are not defensible; the downstream
[`## test`](#test) and [`## debug`](#debug) workflows depend
on all of those fields.

## test

The flow-perf `## test` is **the canonical
smoke-before-bulk loop for the measurement**. *"Test"* in
this skill means *"prove ONE small-scale iteration of the
chosen policy runs cleanly end-to-end before scaling up to
the operator's full workload, then iterate the full workload
enough times to establish variance"*, not
*"unit-test the tool"*.

**`## test` is an iterative loop, not a one-shot pass.**
Every mutation — a JSON policy edit, a backend switch, a
worker / queue / burst / iteration config change, a driver /
firmware change, a BlueField-mode change — re-opens the
smoke.

The smoke-before-bulk shape:

1. **Run the single-iteration smoke** per [`## run`](#run)
   steps 1-2 with small entries and small iteration count.
2. **Confirm the tool printed an iteration summary** and
   that the printed counters reflect a successful run
   (`num_failed == 0`, `num_pushed > 0`).
3. **Cross-check against the `doca-flow` library's view.**
   The surrounding doca-flow application's
   counter / inspector view per
   [`doca-flow CAPABILITIES.md ## Observability`](../../libs/doca-flow/CAPABILITIES.md#observability)
   should reflect the rules flow-perf installed (or
   none, if flow-perf cleaned up after itself per the
   chosen mode). Disagreement is a finding — walk
   [`## debug`](#debug) layer 3 first.
4. **Scale up to the operator's chosen workload** per
   [`## run`](#run) step 4.
5. **Repeat the scaled run** N times (operator-chosen,
   typically 3-10) and capture per-iteration output for
   every run. Compute variance.
6. **Decide if variance is defensible.** Per
   [`CAPABILITIES.md ## Error taxonomy`](CAPABILITIES.md#error-taxonomy)
   layer 5, if variance is too high the agent walks the
   user to stabilize (CPU pinning, interrupt routing,
   isolating sibling CPUs) and REPEATS the smoke.
7. **Capture the four-tuple alongside every reported
   number.** Per [`## configure`](#configure) step 7.

Eval-loop overlay (rows apply to every flow-perf
measurement, not just one):

| Step | Why this is a loop, not a step | Where the substance lives |
| --- | --- | --- |
| 1 → ## debug | The smoke run fails to even start; walk the tool-not-installed / JSON-malformed / pipeline-creation layers, then re-run step 1 | [`## debug`](#debug) layers 1-3 |
| 2 → ## debug | `num_failed > 0` during the smoke; walk the insertion-rate layer, then re-run step 1 with a smaller entry count or a fixed-pattern policy | [`## debug`](#debug) layer 4 |
| 3 → ## debug | Library view disagrees with the tool's view; walk the cross-cutting / version layer first | [`## debug`](#debug) layers 6-7 |
| 5 → step 6 | Per-iteration variance is high; the printed Kops/sec is a misleading average | [`## debug`](#debug) layer 5 |
| 6 → host stabilize → 1 | After CPU pinning / interrupt routing changes, re-run the smoke; the prior smoke is stale | [`doca-debug`](../../doca-debug/SKILL.md) |
| any → JSON edit → 1 | After ANY JSON edit, re-run the smoke; the prior smoke is stale | [`## modify`](#modify) |

The agent's rule: every state-changing action on the JSON,
the backend choice, the worker / queue / burst config, or
the host state re-opens the smoke. Saving a stale smoke is
exactly the failure mode this loop is here to prevent.

This skill does **not** ship a "test fixture" Kops/sec
expectation. The expected number is install-, version-,
firmware-, and platform-specific; pinning one would mislead
operators on a different platform / version.

## debug

When the user reports the tool failing to start, the tool
reporting `num_failed > 0`, the variance being too high, or
the number being implausibly different from a published
reference, walk the
[`CAPABILITIES.md ## Error taxonomy`](CAPABILITIES.md#error-taxonomy)
layers in order:

1. **Tool-not-installed.** Confirm the binary and the
   `configs/` JSON library are present; if not, route to
   [`doca-setup ## install`](../../doca-setup/TASKS.md#configure)
   and [`## no-install`](../../doca-setup/TASKS.md#no-install).
2. **JSON-policy-malformed.** Quote the tool's parser error
   verbatim; walk the user back to the closest shipped
   `configs/` exemplar and edit forward from there. The
   agent must NOT invent JSON keys to fix a parse error.
3. **Pipeline-creation-failed.** The JSON parsed but the
   underlying `doca-flow` library refused the pipe. Route
   the diagnosis to
   [`doca-flow TASKS.md ## debug`](../../libs/doca-flow/TASKS.md#debug);
   common flow-perf-specific causes: matcher / action
   unsupported on the device mode, table-size limit
   exceeded, BlueField mode wrong.
4. **Insertion-rate-failure.** `num_failed > 0` during the
   measurement. Reduce entry count, switch a
   `mode: increase` field to a deeper increment step (so
   the entries are unique), confirm the device's Flow
   tables can hold the requested workload. This is a
   methodology finding — report it explicitly.
5. **Iteration-variance-too-high.** Per
   [`CAPABILITIES.md ## Error taxonomy`](CAPABILITIES.md#error-taxonomy)
   layer 5: stabilize the host (pin worker threads to
   dedicated CPUs, isolate sibling logical CPUs, route
   interrupts away from worker CPUs), increase iteration
   count, then RE-RUN the smoke. Do NOT silently average a
   high-variance measurement.
6. **Version.** Walk
   [`doca-version TASKS.md ## debug`](../../doca-version/TASKS.md#debug)
   end-to-end. Common flow-perf-specific symptoms:
   firmware doesn't expose the matcher / action the JSON
   asks for; BlueField is in the wrong mode; the comparison
   target is from a different DOCA release.
7. **Cross-cutting.** Hand off to
   [`doca-debug ## debug`](../../doca-debug/SKILL.md) and
   [`doca-setup ## debug`](../../doca-setup/TASKS.md#debug)
   for the env-side layers (driver, firmware, BlueField
   mode, PCIe, hugepages, IOMMU).

In every case: **capture the raw per-iteration output, the
JSON policy contents, the backend choice, and the four-tuple
verbatim BEFORE retrying.** Paraphrasing the output is the
canonical lost-fidelity failure for this skill.

## use

`## use` is the agent-side workflow for *consuming* a
captured `doca_flow_perf` run as evidence.

1. **Read the raw per-iteration output, not just the
   printed summary.** A printed average Kops/sec hides
   variance the per-iteration cycles surface.
2. **Read the four-tuple alongside the number.** A
   flow-perf number without the four-tuple is not
   actionable; route to [`## debug`](#debug) layer 6 if any
   leg is missing.
3. **Cross-check the rule count.** `num_pushed` must equal
   `num_inserted_entries` for the run to be clean;
   `num_failed > 0` is a finding, not a result. Route to
   [`## debug`](#debug) layer 4.
4. **Compare ONLY against flow-perf numbers with the
   matching four-tuple.** Comparing against a
   different-backend / different-DOCA-version /
   different-firmware / different-policy number is not
   apples-to-apples; the agent says so explicitly.
5. **Route to [`doca-flow-tune`](../doca-flow-tune/SKILL.md)**
   only when the agent confirms the next step is to
   OPTIMIZE a live application, not measure further.
   flow-perf is a measurement, not an optimization
   recommendation generator.

## Deferred task verbs

The verbs below are not `doca_flow_perf` work and should be
routed out before the agent does any of them under this
skill's name.

- **install DOCA** ⇒
  [`doca-setup ## install`](../../doca-setup/TASKS.md#configure)
  and [`## no-install`](../../doca-setup/TASKS.md#no-install).
- **write or modify a doca-flow application** ⇒
  [`doca-flow`](../../libs/doca-flow/SKILL.md), layered on
  [`doca-programming-guide`](../../doca-programming-guide/SKILL.md).
  flow-perf measures its own synthetic pipeline; it does
  not produce or modify application code.
- **optimize a live doca-flow application** ⇒
  [`doca-flow-tune`](../doca-flow-tune/SKILL.md). The
  measurement-vs-optimization boundary is exactly the
  flow-perf-vs-flow-tune boundary.
- **measure the DPA-offloaded Flow path** ⇒
  [`doca-flow-dpa-perf`](../doca-flow-dpa-perf/SKILL.md).
  flow-perf measures the host or DPU-CPU path; flow-dpa-perf
  measures the DPA-offloaded path.
- **library-internal pipe / counter / inspector deep dive**
  ⇒ [`doca-flow`](../../libs/doca-flow/SKILL.md). The
  deeper per-pipe semantics belong to the library.
- **end-to-end dataplane packet throughput / latency
  measurement** ⇒ NOT a flow-perf task per
  [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes)
  (What is NOT measured). That is the application's
  responsibility, layered on
  [`doca-flow`](../../libs/doca-flow/SKILL.md).
- **firmware / driver upgrade** ⇒ route to
  [`doca-version`](../../doca-version/SKILL.md) +
  [`doca-setup`](../../doca-setup/SKILL.md). flow-perf has
  no opinion on the firmware upgrade path beyond requiring
  that the four-tuple capture identifies the firmware
  version used.

## Command appendix

`doca_flow_perf`-specific invocation classes the verbs above
reach for. Every row is a CLASS — the agent must not invent
flag spellings, JSON schema keys, or default workload sizes
beyond the shipped `configs/` exemplars and the public DOCA
Flow Perf guide.

**Infra-aware preamble (every row below).** Per the bundle's
detect → prefer → fall back → report contract documented in
[`doca-structured-tools-contract ## The agent behavior contract`](../../doca-structured-tools-contract/SKILL.md#the-agent-behavior-contract),
the agent should:

1. Probe for the matching structured helper FIRST
   (`doca-env --json`; `doca-capability-snapshot`;
   `version-matrix.json`).
2. If the probe succeeds, the structured tool's output is
   the authoritative answer.
3. If the probe fails, fall back to the manual command in
   the row.
4. The schemas the structured tools emit are defined in
   [`doca-structured-tools-contract ## Schemas`](../../doca-structured-tools-contract/SKILL.md#schemas);
   the version-handling semantics are owned by
   [`doca-version`](../../doca-version/SKILL.md).

| Purpose (class) | Invocation (shape) | Owning step | Reads as healthy when … |
| --- | --- | --- | --- |
| Discover the documented CLI surface | `doca_flow_perf --help` plus the public DOCA Flow Perf guide on `docs.nvidia.com` | [`## configure`](#configure) step 8 + [`## debug`](#debug) layer 1 | Prints the documented flag inventory; the agent uses this as the only source of truth for flag spellings. |
| Locate the shipped `configs/` JSON exemplars | Inspect the `configs/` subdirectory of the flow-perf install (the exact path is install-specific) | [`## configure`](#configure) step 3 | The directory exists; the canned policy that most closely matches the operator's workload is identified by name. |
| Confirm DOCA Flow library version | `pkg-config --modversion doca-flow` on the host where flow-perf runs | [`## configure`](#configure) step 1 + [`## debug`](#debug) layer 6 | Matches `doca_caps --version` and the firmware version per [`doca-version TASKS.md ## debug`](../../doca-version/TASKS.md#debug). |
| Confirm device capabilities (matchers / actions / table sizes) | `doca_caps`-class probes from the shipped DOCA install; capture verbatim | [`## configure`](#configure) step 6 + [`## debug`](#debug) layer 3 | The matchers and actions the JSON policy references are present in `doca_caps` output for the target device. |
| Run the single-iteration smoke (low entry count, single iteration) | `doca_flow_perf` with the chosen JSON policy at a small `num_inserted_entries` and iteration count (per the binary's `--help` for the exact flag spelling) | [`## run`](#run) step 2 + [`## test`](#test) step 1 | Exit 0; printed iteration summary present; `num_failed == 0`; cross-check against `doca-flow` library view matches. |
| Run the scaled measurement (operator's chosen workload size, N iterations) | `doca_flow_perf` with the chosen JSON policy at full workload size + N iterations (per the binary's `--help`) | [`## run`](#run) step 4 + [`## test`](#test) step 5 | Exit 0; per-iteration cycles captured; variance computable; `num_pushed == num_inserted_entries` per iteration. |
| Compute / report variance across iterations | Capture the per-iteration cycles for each of the N runs; compute mean + std-dev OFF-line. The agent does NOT invent a flow-perf-built-in variance flag. | [`## test`](#test) steps 5-6 | Std-dev across iterations is small enough that the reported number is defensible; if not, [`## debug`](#debug) layer 5. |
| Save a session snapshot for debug | Capture (a) the JSON policy verbatim, (b) the full command line, (c) the per-iteration cycles + counters verbatim, (d) the four-tuple (DOCA version, BlueField generation, firmware version, BlueField mode), (e) the backend choice (DPDK or DOCA) | [`## test`](#test) capture step + [`doca-debug TASKS.md ## debug`](../../doca-debug/TASKS.md#debug) | The saved bundle is consumed by the cross-cutting debug ladder. |

Three cross-cutting rules for this appendix:

- **Never invent a flag spelling, JSON schema key, or
  default workload size.** The shipped binary's `--help`,
  the shipped `configs/` exemplars, and the public DOCA
  Flow Perf guide are the joint contract; prose-derived
  spellings are the most common hallucination failure for
  this skill.
- **Single-iteration smoke before scaled measurement.** A
  scaled measurement that has not been preceded by a clean
  smoke on the same JSON is not defensible; the agent
  re-runs the smoke per [`## test`](#test) before issuing
  anything else.
- **Cross-link instead of duplicate.** Cross-cutting
  commands (`pkg-config --modversion`, `dmesg`, `lspci`,
  `ethtool`) live in
  [`doca-debug TASKS.md ## Command appendix`](../../doca-debug/TASKS.md);
  the `doca-flow` application build / port / pipe commands
  live in
  [`doca-flow TASKS.md ## Command appendix`](../../libs/doca-flow/TASKS.md);
  this appendix names only flow-perf-specific invocations
  on top.

## Cross-cutting

A few rules that apply across every verb in this file:

- The **public DOCA Flow Perf guide** + the installed
  binary's `--help` + the shipped `configs/` JSONs are the
  joint source of truth on the DOCA side; the
  `docs.nvidia.com` release notes are the joint source of
  truth for published reference numbers.
- The **JSON policy is the workload description.** Quote
  it verbatim alongside every reported number; do not
  paraphrase fields from memory.
- **Single-iteration smoke before bulk.** Per
  [`CAPABILITIES.md ## Safety policy`](CAPABILITIES.md#safety-policy)
  the rule is non-negotiable; flow-perf at scale can
  exhaust device Flow tables and disrupt co-tenant
  applications.
- **Report the backend.** A flow-perf number without a
  named backend (DPDK or DOCA) is not interpretable;
  every reported number must say which backend produced it.
- **Capture the four-tuple.** Every reported number comes
  with: DOCA version, BlueField / ConnectX generation,
  firmware version, JSON policy contents, backend choice,
  worker / queue / burst / iteration configuration.
- This skill **assumes a healthy DOCA install** (or the
  public NGC DOCA container) and a healthy underlying
  [`doca-flow`](../../libs/doca-flow/SKILL.md) library on
  the target device. If either is in doubt, route to
  [`doca-setup`](../../doca-setup/SKILL.md) and
  [`doca-flow`](../../libs/doca-flow/SKILL.md) before
  running anything else here.
