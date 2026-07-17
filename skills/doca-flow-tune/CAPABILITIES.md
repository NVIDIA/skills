# DOCA Flow Tune — Capabilities

**Where to start:** `doca_flow_tune` is a unified DOCA tool that
observes and recommends parameter changes on a live or captured
`doca-flow` pipeline. The pattern overview below names the
recurring `doca_flow_tune`-class questions. Pick the pattern
first, then drill into the H2 that owns the substance. For the
*how* of executing each pattern, jump to [TASKS.md](TASKS.md).
For the `doca-flow` API surface that created the pipeline this
tool tunes, see
[`doca-flow CAPABILITIES.md`](../../libs/doca-flow/CAPABILITIES.md).

This file is loaded by [`SKILL.md`](SKILL.md). It documents *what
`doca_flow_tune` is*, *what it observes and what it recommends on*,
*how its server role and client / consumer role decompose inside a
single binary*, *what versions it ships in*, *what its layered
error and observability surfaces look like*, and *the safety
posture* the tool's role at the edge of a live dataplane forces.

## Pattern overview

Every `doca_flow_tune` question this skill teaches resolves into
one of SIX patterns. The patterns are CLASSES — they apply
across every `doca-flow` application a tune session can attach
to, not one specific pipeline.

| `doca_flow_tune` pattern | Class shape | Where the substance lives |
| --- | --- | --- |
| 1. Decide tune vs perf vs other | flow-tune *optimizes*; [`doca-flow-perf`](../doca-flow-perf/SKILL.md) *measures*; the program-side debug surface lives in [`doca-flow`](../../libs/doca-flow/SKILL.md). Picking the wrong tool produces an answer to a different question. | [`## Capabilities and modes`](#capabilities-and-modes) tune-vs-perf bullet + [TASKS.md ## configure](TASKS.md#configure) |
| 2. Decide the role inside the artifact | `doca_flow_tune` is one binary with two internal roles: a server role that snapshots and exposes live pipe state via a local IPC channel, and a client / consumer role that analyzes / visualizes / recommends on top of either the live snapshot or a captured config. Pick which role the session uses *before* invocation. | [`## Capabilities and modes`](#capabilities-and-modes) one-binary-two-roles + [TASKS.md ## configure](TASKS.md#configure) |
| 3. Pick the tuning axis | rule placement vs resource hints / table sizing vs hardware-offload mode. Picking one is mandatory; mixing axes per session hides which axis moved the measurement. | [`## Capabilities and modes`](#capabilities-and-modes) three-axis table + [TASKS.md ## configure](TASKS.md#configure) |
| 4. Pick the measurement axis | rule-install rate vs lookup latency vs hardware-counter delta. The chosen measurement must match the tuning axis; a latency answer to a rule-install-rate question is the canonical apples-to-oranges failure. | [`## Capabilities and modes`](#capabilities-and-modes) measurement table + [TASKS.md ## configure](TASKS.md#configure) |
| 5. Snapshot → analyze → visualize | The cheapest read-only flow: snapshot live state (or a captured config), feed the analyze step, render the visualize step, then read the recommendation. No state on the live dataplane is changed by this flow. | [TASKS.md ## run](TASKS.md#run) + [`## Observability`](#observability) |
| 6. Diagnose unsound / attach-failed / empty output | Walk the layered error taxonomy in [`## Error taxonomy`](#error-taxonomy) instead of guessing at causes from a stack trace or an empty mermaid diagram. | [`## Error taxonomy`](#error-taxonomy) + [TASKS.md ## debug](TASKS.md#debug) |

Two cross-cutting rules that apply to *every* pattern above:

- **`doca_flow_tune` is one binary, two internal roles.** The
  historical *Flow Tune Server* and *Flow Tune Tool* (client) live
  inside this one artifact: there is one binary on disk, one
  package, one version, and the roles decompose inside the
  invocation surface — not across two separately-shipped CLIs.
  Recommendations that frame *"the tune server"* and *"the tune
  client"* as different artifacts are categorically wrong; they
  are different *modes* of the same artifact and the agent must
  surface that.
- **Tune optimizes; perf measures.** The most common up-front
  mistake is reaching for `doca_flow_tune` to *measure* a
  baseline number (use [`doca-flow-perf`](../doca-flow-perf/SKILL.md))
  or reaching for `doca-flow-perf` to *recommend* a parameter
  change (use this skill). An agent that does not say which side
  of that boundary the user's question lands on has silently
  conflated optimization with measurement.

## Capabilities and modes

`doca_flow_tune` is shipped as a **single CLI tool** in the DOCA
install, plus its companion JSON configuration templates (the
`flow_tune_cfg*.json` family on the user's install), companion
Python scripts (e.g. `flow_json_diff.py`, `flow_mermaid_diff.py`,
`hw_counters_csv_analyzer.py` per the shipped `scripts/`
directory), and the validation set the install ships. There is
no daemon process the operator runs as an independent service and
no library to link against for the tool itself — the entire
interaction model is *write a JSON config, invoke the binary,
read the printed and / or CSV / mermaid / JSON output, propose a
minimum-diff change back to the surrounding `doca-flow` program*.

### Tune vs perf — the boundary

The two Flow performance tools the bundle ships are deliberately
different surfaces. The agent must name which side a user's
question lands on before recommending an invocation.

| Surface | What it produces | When to reach for it |
| --- | --- | --- |
| [`doca-flow-perf`](../doca-flow-perf/SKILL.md) | A defensible, four-tuple-anchored baseline number (throughput / latency / rule-install rate) on a *static* pipeline configuration driven by the perf tool itself | The question is *"what does this Flow pipeline configuration actually deliver on this device"*; no live application is required |
| [`doca-flow-dpa-perf`](../doca-flow-dpa-perf/SKILL.md) | Same idea as `doca-flow-perf` but for the DPA-offloaded Flow path specifically | The question is *"what does this Flow pipeline deliver when offloaded to the DPA processor"*; the device must be DPA-capable |
| `doca_flow_tune` (this skill) | A snapshot of live (or captured) pipe state plus an *analysis-driven recommendation* for a parameter change | The question is *"given this measured baseline, how do I improve it"*; a `doca-flow` application is running (or its config is captured) |

The downstream rule: use `doca-flow-perf` to *measure*; use
`doca_flow_tune` to *optimize on top of the measurement*. An
optimization without a measured baseline is optimizing in the
dark; a measurement without a question to apply it to is a
benchmark for the sake of it.

### One binary, two internal roles

`doca_flow_tune` exposes two internal roles inside the one
artifact:

| Role | What it does | Where it lives in the binary |
| --- | --- | --- |
| **Server role** | Sits next to (or, on configurations where the documented attach API is linked, *inside*) a `doca-flow` application and snapshots its live pipe / port / counter / KPI state, exposing it over a documented local IPC channel (per the shipped `flow_tune_cfg_public.json` template, the channel is a Unix-domain socket whose path lives in the `network.server_uds` field of the JSON config). | The server-side dumper / monitor invocation modes of the binary |
| **Client / consumer role** | Reads the snapshot (or a previously captured one), runs the analyze / visualize / diff scripts on it, and produces the recommendation the operator applies back to the Flow program. | The analyze / visualize / diff invocation modes of the binary plus the companion `scripts/` Python helpers shipped with the tool |

The two roles share one binary and one JSON configuration file.
Picking which role a session engages is a *configuration*
decision (which sections of the JSON config are populated, which
mode flag is passed on the command line) — not a binary-choice
decision. Recommendations that frame *"install the server
package, then the client package"* are categorically wrong; the
DOCA install ships the unified tool and there is one binary on
disk.

### Three-axis configuration

Every concrete `doca_flow_tune` session commits to a point in
three independent axes. Skipping any axis produces a session that
inspects or tunes the wrong thing under unstated assumptions.

| Axis | What it picks | Why the agent must name it |
| --- | --- | --- |
| **1. Tuning axis** | The CLASS of change being explored — rule placement (where rules live, which steering tier), resource hints (cache / table sizing / aging), or hardware-offload mode (HWS vs SWS vs hybrid, where applicable). | A tuning session that mixes axes per op produces a measurement that cannot be attributed to one change; the agent must commit to exactly one tuning axis per session. |
| **2. Measurement** | The metric the recommendation is being judged against — rule-install rate, lookup latency, or hardware-counter delta. | A latency answer to a rule-install-rate question, or a counter-delta answer to a latency question, is the canonical apples-to-oranges failure. The measurement axis must match the tuning axis the operator named. |
| **3. Scope** | WHICH pipe / port / app the session targets. Per the `flow_tune_cfg_public.json` template's `monitor.hardware.pci_addresses` and `monitor.software[*].flow_port_id` fields, the configuration explicitly names the scope; the agent's job is to confirm the scope matches the pipeline the user actually cares about. | Tuning the wrong pipe answers a different question and can disrupt a dataplane path the user did not intend to touch. |

### Operating modes

Per the shipped JSON config templates and the live invocation
surface, `doca_flow_tune` exposes three operating modes the
agent should name explicitly:

| Mode | What it does | Read-only? |
| --- | --- | --- |
| **Offline** | Consumes a previously captured pipeline-description JSON file (per the `analyze.file_name` field) and runs the analyze / visualize chain over it; the live Flow application is not contacted. | Yes |
| **Online (read-only)** | Attaches to the live Flow application via the documented IPC channel (per `network.server_uds`), snapshots the live counters / KPIs, and feeds them into the dumper / monitor / analyze chain. | Yes; the dumper writes a CSV file, the visualize step renders a mermaid file, neither touches Flow state. |
| **Online (state-changing application of recommendation)** | After the operator has read a recommendation and committed to applying it, the recommendation is fed back into the `doca-flow` program (typically as a code change applied via the universal modify-a-sample workflow). | NO — this mode changes live Flow state via the surrounding application and is the high-stakes mode the [`## Safety policy`](#safety-policy) gates. |

The exact CLI mode flag names, the full JSON config schema, and
the documented pipeline-description file format live in the
public DOCA Flow Tune guide on `docs.nvidia.com` (reached via
[`doca-public-knowledge-map ## DOCA tools`](../../doca-public-knowledge-map/SKILL.md#doca-tools))
and in the installed `--help` and shipped `flow_tune_cfg*.json`
templates on the user's version. The skill deliberately does not
pin them — see
[`SKILL.md ## What this skill deliberately does not ship`](SKILL.md#what-this-skill-deliberately-does-not-ship).

## Version compatibility

For the canonical DOCA version-detection chain, the four-way
match rule, NGC container semantics, and the headers-win-over-docs
rule, see [`doca-version`](../../doca-version/SKILL.md). The body
lives there; this skill does not duplicate it.

**The `doca_flow_tune`-specific overlay** is:

- **The tool rides the `doca-flow` library version it observes.**
  The tune binary, the Flow library `*.so` the application links,
  and (when the attach role is used) the application's compile-
  time Flow headers must all come from the same DOCA install.
  Cross-train pairings (a tune binary from one DOCA train looking
  at a Flow application linked against a different train's `*.so`)
  are partial-install hazards per
  [`doca-version CAPABILITIES.md ## Version compatibility`](../../doca-version/CAPABILITIES.md#version-compatibility),
  and the agent must surface the mismatch rather than paper over
  it. The right answer for *"the tool runs but the visualize step
  produces wrong-shaped output"* is to confirm the four-way match
  first per
  [`doca-version TASKS.md ## test`](../../doca-version/TASKS.md#test).
- **JSON config / output stability is not contractually frozen.**
  The shipped `flow_tune_cfg_public.json` template's field
  inventory, the dumper CSV column order, the analyze JSON
  schema, and the visualize mermaid layout can shift across
  releases. Agents that need to consume tune output
  programmatically should prefer the structured-helper contract
  in
  [`doca-structured-tools-contract ## Schemas`](../../doca-structured-tools-contract/SKILL.md#schemas)
  when present and re-verify the textual layout against the
  user's installed templates when absent.
- **Per-platform support varies.** The documented BlueField
  generation and host-side support set for Flow tuning varies
  per release; do not copy a tune capability claim from one
  BlueField generation to another. Re-read the public matrix per
  [`doca-public-knowledge-map ## DOCA tools`](../../doca-public-knowledge-map/SKILL.md#doca-tools).
- **Where it runs.** The binary runs on the side the operator
  has DOCA installed (host x86 / Arm, BlueField Arm, or NGC
  container). The Flow application it observes can be on the
  same side or — when the documented attach role's IPC reaches
  across — on the colocated side. What matters is that the tool,
  the application's linked Flow library, and the install version
  agree per the four-way match above.

## Error taxonomy

`doca_flow_tune`'s error surface is broader than a pure
read-only CLI because the tool both *observes* live state and
*recommends* a change against it; each side has its own failure
modes. The error layers the agent should distinguish, in
escalating order:

1. **Config-syntax.** The invocation itself does not parse or
   the JSON config is malformed: unknown flag, malformed JSON,
   a `network.server_uds` path that the operator cannot create,
   a `monitor.hardware.pci_addresses` entry the device does not
   expose, conflicting flags. Cause: the operator wrote a flag
   or JSON field that does not exist on the installed version
   (often a field taken from prose / blog / older release).
   Routing: re-read `--help` on the installed binary and the
   shipped `flow_tune_cfg*.json` templates; do not guess.
2. **Attach-failed (online mode).** The invocation parses; the
   tool cannot reach the live Flow application's tune-server
   endpoint. Cause: the Flow application is not running, the
   application's tune-server role was never started, the
   configured `network.server_uds` path does not exist or is
   not writable from the application's process, the application
   is running but the file-system namespace the socket lives in
   is not visible from the tool's side (e.g. one is inside a
   container the other is not). Routing: confirm the Flow
   application is up via
   [`doca-flow TASKS.md ## test`](../../libs/doca-flow/TASKS.md#test);
   confirm the socket path matches the application's actual
   `network.server_uds` value; confirm the namespace boundary.
3. **Pipe-not-found.** The tool attached, but the pipe the
   operator named in scope is missing from the snapshot. Cause:
   the Flow application never created that pipe (route to
   [`doca-flow TASKS.md ## modify`](../../libs/doca-flow/TASKS.md#modify)),
   the pipe was destroyed by a previous program-side action, or
   the operator is using the wrong identifier. The right move
   is to confirm the pipe set against the Flow application's
   own view, not to re-run the tune session expecting different
   output.
4. **Measurement-unsound.** The session runs to completion and
   reports counters / KPIs, but the numbers are unsound and
   must not be quoted as-is. Three sub-layers:
    - *Warm-up / steady-state not reached.* The snapshot was
      taken during a transient state of the live Flow
      application; the dumper CSV captures cold-cache /
      cold-pipeline iterations and is below steady-state.
      Fix: let the application reach steady state before
      capturing.
    - *Outliers / distribution unreported.* A single
      hardware-counter delta hides a heavy tail; a single
      lookup-latency reading hides a 99.99-percentile spike.
      Fix: report the distribution alongside any single
      number per the *quote what the tool said* rule in
      [`## Safety policy`](#safety-policy).
    - *Before / after pair not captured.* The recommendation
      was applied without a before snapshot; the *delta*
      cannot be defended. Fix: re-run with a pre-change
      snapshot, then the change, then the post-change
      snapshot.
5. **Recommendation-unactionable.** The analyze step produced
   a recommendation, but it does not correspond to a knob the
   surrounding `doca-flow` program exposes (the recommendation
   targets a pipe attribute the program hard-codes, or the
   recommendation requires a Flow API the program does not
   call). Cause: the recommendation was generic across the
   tool's library of known patterns and the program is more
   constrained than the patterns assume. The right move is the
   minimum-diff modification workflow in
   [`TASKS.md ## modify`](TASKS.md#modify) — surface the
   recommendation as a Flow-program-change proposal, not as a
   knob the tune binary itself can flip.
6. **Version.** Cross-cutting partial-install / mixed-version
   layer per
   [`doca-version CAPABILITIES.md ## Version compatibility`](../../doca-version/CAPABILITIES.md#version-compatibility).
   Symptoms: tune binary version disagrees with the Flow
   application's linked Flow library version, the visualize
   step renders a degenerate diagram on a known-good pipeline
   layout, the analyze step produces JSON fields the public
   guide's documented schema does not list. Routing: walk
   [`doca-version TASKS.md ## debug`](../../doca-version/TASKS.md#debug)
   before any further investigation.
7. **Cross-cutting.** The cause is below DOCA — driver,
   firmware, BlueField mode, NUMA, hugepages. Symptoms that
   do not fit layers 1-6 (e.g. counter deltas that fall
   sharply only on one NUMA node, attach failures correlated
   with `mlx5_core` reload, recommendations that do not move
   the metric even after a clean before / after pair).
   Routing: hand off to
   [`doca-debug ## debug`](../../doca-debug/SKILL.md) and
   [`doca-setup ## debug`](../../doca-setup/TASKS.md#debug);
   the tune surface has reached its limit.

`doca_flow_tune` does not *itself* participate in the
cross-library `DOCA_ERROR_*` taxonomy in the way an application-
linked library does; the tool is a CLI that drives observation
and analysis. For the cross-library `DOCA_ERROR_*` family and
the program-side debug order, see
[`doca-programming-guide CAPABILITIES.md ## Error taxonomy`](../../doca-programming-guide/CAPABILITIES.md#error-taxonomy).

## Observability

`doca_flow_tune` is itself an **observability primitive** for
the rest of the Flow surface — it is *what other skills load to
see what a Flow pipeline is actually doing* without
re-instrumenting the Flow program. Specifically:

- **Dumper output (CSV).** Per the shipped
  `flow_tune_cfg_public.json` template's `dumper.file_name`
  field, the tool can write a CSV of the captured counters /
  KPIs. CSV is the documented machine-readable surface — the
  agent's rule for any baseline that will be re-read by a later
  run is *capture CSV, not just stdout*. The hardware-counters
  analyzer script `hw_counters_csv_analyzer.py` shipped under
  `scripts/` is the vendor-supplied post-processor.
- **Analyze output (JSON).** Per the `analyze.file_name` field,
  the tool emits a pipeline-description JSON file that captures
  the structural and counter view of the pipeline; this is the
  artifact the visualize step consumes and the artifact a
  follow-up tune session can replay in offline mode.
- **Visualize output (mermaid markdown).** Per the
  `visualize.file_name` field, the tool renders the pipeline
  description as a mermaid markdown diagram suitable for a
  pull-request review or a debug session log. The
  `flow_mermaid_diff.py` script shipped under `scripts/` is the
  vendor-supplied diff helper for two mermaid snapshots.
- **Monitor output.** Per the `monitor.screen_mode` field plus
  the `monitor.hardware` and `monitor.software` profile sets,
  the tool can drive a screen-mode live view of the configured
  hardware / software profiles (per-pipe operations rates,
  hardware counters by PCI device, etc.). Useful for a
  real-time check that a tuning change has taken effect; the
  CSV dumper is the right surface for a captured baseline.
- **Logs.** Per the `logging.developer_log` and
  `logging.operational_log` fields, the tool writes a developer-
  level and an operational-level log; the agent's rule is *quote
  what the log said* when diagnosing layer 2 (attach-failed) or
  layer 4 (measurement-unsound) per
  [`## Error taxonomy`](#error-taxonomy).

For the cross-cutting env-side observability primitives
(representor enumeration, `devlink dev show`, `mlxconfig`) see
[`doca-setup CAPABILITIES.md ## Observability`](../../doca-setup/CAPABILITIES.md#observability).
For the program-side observability surface (`DOCA_LOG_LEVEL`,
`--sdk-log-level`, the trace build flavor) see
[`doca-programming-guide CAPABILITIES.md ## Observability`](../../doca-programming-guide/CAPABILITIES.md#observability).
For the Flow library's own programmatic counter / inspector
surface see
[`doca-flow CAPABILITIES.md ## Observability`](../../libs/doca-flow/CAPABILITIES.md#observability).

## Safety policy

> **Overlay on the bundle-wide hardware-safety meta-policy.** The rules below are this skill's per-artifact overlay on the cross-cutting rules in [`doca-hardware-safety` CAPABILITIES.md ## Safety policy](../../doca-hardware-safety/CAPABILITIES.md#safety-policy) (specifically [### Per-artifact overlay pattern](../../doca-hardware-safety/CAPABILITIES.md#per-artifact-overlay-pattern)). When the two layers disagree, the stricter wins; when either layer says STOP, the agent stops.

`doca_flow_tune` is the **most dataplane-sensitive** tool this
bundle currently teaches an agent to drive directly: a recommended
parameter change *applied back to the surrounding Flow program*
lands in live Flow state on a running dataplane. The safety rules:

- **Read-only by default; state-changing only after a clean
  snapshot.** The offline mode and the online read-only mode do
  not touch live Flow state and can be re-run freely. Applying a
  recommendation back to the Flow program — even via a code change
  reviewed in a pull request — *does* change live Flow state on
  the next deploy, can disrupt in-flight traffic, and is the
  high-stakes class of operation the agent must call out as such
  before recommending it.
- **Smoke-before-bulk is mandatory.** Before any application of a
  recommendation, the agent runs the snapshot → analyze →
  visualize → confirm-before-change sequence in
  [`TASKS.md ## test`](TASKS.md#test). A recommendation applied
  without that sequence is a guess against possibly-stale
  evidence — exactly the failure mode this rule exists to
  prevent.
- **Quote the four-tuple, not just the recommendation.** A
  `doca_flow_tune` recommendation is only meaningful with the
  (command line + JSON config + DOCA version + device +
  as-deployed environment) tuple. Quoting *"tune said sizing X
  is better"* without the tuple is unfalsifiable.
- **Do not invent flags, JSON fields, or recommendation
  identifiers.** The documented surface is the surface; the
  public DOCA Flow Tune guide plus the installed `--help` and
  the shipped `flow_tune_cfg*.json` templates are the joint
  source of truth. Prose-derived field names are the most common
  hallucination failure for this skill; see the cross-cutting
  rule in
  [`TASKS.md ## Command appendix`](TASKS.md#command-appendix).
- **Never widen the attach role as a workaround.** When the tool
  cannot read what it expected, walk
  [`## Error taxonomy`](#error-taxonomy) — do not enable a
  state-changing access surface, do not bypass a namespace
  boundary, do not relax filesystem permissions on the
  `network.server_uds` socket so the symptom goes away.

## Public-source pointer

The single canonical public source for `doca_flow_tune` is the
**DOCA Flow Tune** page on `docs.nvidia.com`, reachable through
[`doca-public-knowledge-map ## DOCA tools`](../../doca-public-knowledge-map/SKILL.md#doca-tools).
Do not invent flag strings, JSON config field names,
recommendation identifiers, or output column names beyond what
that page documents — and re-verify against `--help` on the
user's installed binary and the shipped `flow_tune_cfg*.json`
templates, since granular-build support means the *available*
surface is install-specific within the *documented* surface. For
the `doca-flow` API behind the pipeline being tuned, the source
of truth is [`doca-flow`](../../libs/doca-flow/SKILL.md) plus the
public DOCA Flow guide reached the same way. For the matching
measurement tools, see
[`doca-flow-perf`](../doca-flow-perf/SKILL.md) and
[`doca-flow-dpa-perf`](../doca-flow-dpa-perf/SKILL.md).
