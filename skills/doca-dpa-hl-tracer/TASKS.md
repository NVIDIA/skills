# DOCA DPA High-Level Tracer — Tasks

**Where to start:** The verbs that carry real workflow
content are `## install` (route to host-side DOCA install +
DPA env), `## configure` (mode + JSON config + capture
window), `## run` (the capture → decode → render
pipeline), `## test` (iterative tuning of mode / window /
overhead), `## debug` (layered diagnosis), and `## use`
(hand the captured trace back to the
[`doca-dpa`](../../libs/doca-dpa/SKILL.md) debug loop). The
two routing-stub verbs (`build`, `modify`) are kept because
the agent's task-verb contract is uniform across the
bundle, and each carries a meaningful pointer to where the
user's question actually belongs.

This file is loaded by [`SKILL.md`](SKILL.md) after
[`CAPABILITIES.md`](CAPABILITIES.md). It walks the agent
through the documented invocations of `doca_dpa_hl_tracer`,
the capture → decode → render workflow, and the hand-off
back to the [`doca-dpa`](../../libs/doca-dpa/SKILL.md)
debug ladder.

## install

`doca_dpa_hl_tracer` is **shipped pre-built** as part of
every DOCA install that includes the DOCA DPA Tools
optional component, under `/opt/mellanox/doca/tools/`. The
operator-side install path:

1. **Confirm DOCA is installed with the DPA Tools
   component.** If the binary is missing, route to
   [`doca-setup ## install`](../../doca-setup/TASKS.md#configure)
   to install or repair the host-side DOCA package
   selection; confirm the version per
   [`doca-version TASKS.md ## configure`](../../doca-version/TASKS.md#configure).
2. **Confirm the DPACC compiler is installed and
   version-matched to the DOCA install.** Per the
   [`doca-dpa`](../../libs/doca-dpa/SKILL.md) overlay,
   DPA-side images are built by DPACC and the host-side
   `doca-dpa` library is paired with a matching DPACC
   version; the tracer decodes against ELFs produced by
   that DPACC.
3. **Confirm a BlueField with a DPA processor is
   physically present and visible to the host through
   DOCA.** Walk
   [`doca-dpa TASKS.md ## configure`](../../libs/doca-dpa/TASKS.md#configure)
   step 1 to enumerate the device and confirm DPA
   support. If the BlueField is not visible or the DPA
   is not exposed, tracing is meaningless.
4. **Confirm a DPA-side application image exists on
   disk.** The tracer decodes against the same ELF the
   host-side
   [`doca-dpa`](../../libs/doca-dpa/SKILL.md) lifecycle
   loaded into the `doca_dpa_app` context. The user must
   know where their DPA-side ELF lives (the build that
   DPACC produced).
5. **Confirm the operator has the privileges the public
   DOCA DPA Tools guide requires.** Cycle-accurate
   capture and high-priority threads may require
   elevated privileges; the public guide is the
   authoritative source for the exact permission set.

If the binary is not at the standard path, the fix is to
install / repair the host-side DOCA package selection that
includes DPA Tools, not to patch the file in place.

## configure

The tracer's configuration is the invocation flags + the
JSON config file. Steps the agent walks the user through,
in order:

1. **Gate — confirm DPA-side tracing is the right
   surface.** Walk the when-is-this-the-right-surface
   table in
   [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes).
   If the bug is host-side, the DPACC build, raw cycle
   profiling, or device visibility, route to the right
   skill first; do not configure the tracer.
2. **Pick capture mode.** `--mode TRACE` (full per-event,
   higher overhead) vs `--mode CRIT` (critical-events
   only, lower overhead). Default *"start here"* is
   `CRIT` unless the bug demands per-event detail. The
   sub-mode variants accepted on `--sub-mode` are
   authoritative on `--help`; the agent does not invent
   sub-mode strings. **`--device <mlx5_*>` is mandatory**
   (`dpa_hl_tracer_utils.cpp` calls
   `doca_argp_param_set_mandatory(dev_param)`); every
   invocation in `## run` and the Command appendix below
   ships it. An agent producing a bare
   `doca_dpa_hl_tracer --mode <X>` command will fail
   argp validation before the tool reaches the DPA.
3. **Size the JSON config.** Stage a config file with:
    - `limits_config.log_file_max_size_in_bytes` and
      `bin_file_max_size_in_bytes` — sized to the
      expected event rate × capture window.
    - `limits_config.file_size_limit_policy` — pick
      *stop-on-limit* (`0`) for *"don't lose old
      events"* or *truncate-and-continue* (`1`) for
      *"keep the most recent events"*. The two fail
      differently; pick explicitly.
    - `threads_config.receiver_thread.thread_priority` /
      `thread_core` — a real-time priority + a pinned
      core for the receiver thread so the capture is
      not at the mercy of the scheduler. The shipped
      example pins this to a high priority.
    - `threads_config.binary_writer_thread`,
      `file_writer_thread`, `printer_thread` —
      priorities + cores for the writers and the
      stdout printer; `-1` means default.
   Pin the chosen JSON config to disk; it is part of
   the captured artifact set per
   [`CAPABILITIES.md ## Observability`](CAPABILITIES.md#observability).
4. **Plan the capture window.** Decide explicitly when
   the tracer will start, when it will stop, and which
   workload phase needs to be inside the window per
   [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes)
   capture-window invariant. Events outside the window
   are silently lost.
5. **Snapshot the DPA-side ELF.** Pin the DPA-side ELF
   path and capture a SHA of the file. The decode step
   requires this exact ELF; a rebuild between capture
   and decode produces noise per the
   ELF-must-match-image rule.
6. **Confirm the host-side `doca-dpa` lifecycle has
   already brought the DPA workload up at least once.**
   Per
   [`doca-dpa TASKS.md ## run`](../../libs/doca-dpa/TASKS.md#run),
   the workload must be running for events to fire;
   the tracer is observation, not a workload driver.

For the canonical DOCA universal lifecycle on the
host-side workload this tracer observes, see
[`doca-programming-guide TASKS.md ## configure`](../../doca-programming-guide/TASKS.md#configure)
and [`doca-dpa TASKS.md ## configure`](../../libs/doca-dpa/TASKS.md#configure).
This skill is concerned with the *operator-side*
configuration of the tracer invocation.

## build

`doca_dpa_hl_tracer` is **shipped pre-built** as part of
every DOCA install that includes the DPA Tools optional
component (`/opt/mellanox/doca/tools/doca_dpa_hl_tracer`).
There is no source tree the external user is expected to
compile, no build flags, no `meson` or `make` workflow for
the tracer itself.

Routing for nearby "build" questions:

- *"The binary isn't there — do I need to build it?"* → no.
  Route to
  [`doca-setup ## install`](../../doca-setup/TASKS.md#configure).
  The fix is to install (or re-install) the DOCA package
  selection that includes the DPA Tools component.
- *"I want to build my own DPA-side application that
  emits trace events."* → not a tracer question. Route to
  [`doca-dpa TASKS.md ## build`](../../libs/doca-dpa/TASKS.md#build)
  for the cross-side build pattern + the public DPACC
  guide via
  [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md)
  for the DPACC-side build flags that enable the
  tracer-instrumentation hooks.
- *"I want to extend the tracer with new event classes."* →
  out of scope here; this skill is for external operators
  consuming the shipped tracer, not for contributors
  extending it.

The `## What this skill deliberately does not ship` block
in [`SKILL.md`](SKILL.md) explicitly forbids adding a build
recipe for the tracer; revisit that policy before changing
this section.

## modify

**Do not modify the shipped `doca_dpa_hl_tracer` binary.**
It is an NVIDIA-shipped CLI; there is no documented public
way to change its capture behaviour, event taxonomy, or
output format, and none should be invented.

What the agent *does* modify, every time, is the **tracer
invocation and the JSON config** — the `--mode`, the
`--sub-mode`, the config file's `limits_config` and
`threads_config` sections, the capture window. That is the
configuration loop in [`## configure`](#configure) above
and the iteration loop in [`## test`](#test) below; treat
*modify the invocation + JSON config, not the binary* as
the operating mode.

Routing for nearby "modify" questions:

- *"The trace overhead is too high — can I disable some
  events?"* → adjust `--mode` and `--sub-mode` per
  [`## configure`](#configure) step 2; raise the
  receiver-thread priority per step 3. If the event
  taxonomy itself is too coarse / too fine, the answer
  is *"this is the documented surface; for other
  granularity, use the raw cycle profiler"* — route via
  [`doca-public-knowledge-map ## DOCA tools`](../../doca-public-knowledge-map/SKILL.md#doca-tools).
- *"Can I change the trace output format?"* → no; the
  binary trace format is documented and the human-
  readable log is its decoded view. If the user needs a
  different post-processing format, the right answer is
  *"write a parser against the documented format on your
  installed version"* — and even that scripting is out
  of scope per
  [`SKILL.md ## What this skill deliberately does not ship`](SKILL.md#what-this-skill-deliberately-does-not-ship).
- *"I want a different measurement than the tracer
  reports."* → re-examine the
  when-is-this-the-right-surface gate in
  [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes);
  if the question is genuinely outside the DPA
  programming events surface (e.g. raw cycle counts,
  cache behaviour), this is not the right tool.

## run

The capture → decode → render pipeline — every tracing
session goes through it, no exceptions. The full
invocation surface lives in the public DOCA DPA Tools
page; this section names the *shape* of the flow, not
verbatim command lines (per
[`CAPABILITIES.md ## Safety policy`](CAPABILITIES.md#safety-policy)
*"do not invent flags"*).

1. **Confirm prerequisites.** Per [`## install`](#install)
   and [`## configure`](#configure): binary present,
   version paired with `doca-dpa` library + DPACC,
   BlueField visible and DPA exposed, DPA-side ELF
   identified, host-side `doca-dpa` lifecycle has
   brought the workload up.
2. **Capture.** Run
   `doca_dpa_hl_tracer --device <mlx5_*> --mode <TRACE|CRIT>
   --sub-mode <variant> --config-file <json> --output-file
   <bin-path> --log-file <log-path>`. The tracer attaches
   to the named device, opens the JSON config, and writes
   the binary trace + log file while the capture window
   is open. Re-confirm exact flag names against `--help`
   on the installed binary.
3. **Stop the capture explicitly.** Either let it run
   until the file-size limit closes it (per the
   `file_size_limit_policy`) or signal it to stop at the
   end of the chosen window. Note which way the capture
   ended — the two failure modes (stop-on-limit vs
   truncate-and-continue) look different in the trace.
4. **Decode against the matching ELF.** Run
   `doca_dpa_hl_tracer --input-file <bin-path>
   --parse-file <parsed-output> --elf-file <dpa-elf>`.
   The ELF must be the exact build the `doca_dpa_app`
   context loaded at capture time per the
   ELF-must-match-image rule in
   [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes).
   Decoding against a different ELF produces noise.
5. **Read the rendered output.** The human-readable log
   plus the parsed output are the agent's read-side
   surface. Compare event ordering against the DPA-side
   kernel source; quote events verbatim, do not
   paraphrase.
6. **Inspect the tool's own stderr.** Per
   [`CAPABILITIES.md ## Observability`](CAPABILITIES.md#observability),
   stderr carries DOCA log output the agent uses to
   distinguish install / device-binding / image-not-
   instrumented failures from real *"no events"*
   answers.

When recording the run for downstream consumers, write
down: the DOCA version, the DPACC compiler version, the
BlueField identity + firmware version, the DPA-side ELF
path + SHA, the JSON config used, the mode + sub-mode, the
capture wall-clock window, and the produced binary trace +
log files. The downstream [`## test`](#test) and
[`## debug`](#debug) workflows depend on those fields.

## test

`doca_dpa_hl_tracer` is **a diagnostic capture tool**, so
its `## test` verb is about *testing the trace itself* —
confirming the captured events are the right events,
captured at sound overhead, with a sound capture window —
not unit-testing the tracer binary.

**`## test` is an iterative loop, not a one-shot pass.**
A capture that completes is not the same as a capture
that produced defensible evidence; each iteration tightens
one axis of capture soundness (mode vs overhead, window
vs missed events, file-size limit vs truncation, ELF
match vs decode noise, DPA-side workload up vs idle) and
loops back to [`## run`](#run).

The eval-loop overlay (rows apply to every trace
session):

| Iteration trigger | What it looks like | What changes next iteration |
| --- | --- | --- |
| Decoded trace is empty | Could be install (binary failed to attach), device-binding, image not instrumented, capture window missed the workload, or workload was idle | Walk the error taxonomy in [`CAPABILITIES.md ## Error taxonomy`](CAPABILITIES.md#error-taxonomy) layers 1–4 in order; do not jump. |
| `TRACE`-mode capture shifted measured kernel timing materially | Overhead-saturated; per-event interval near the receiver-thread scheduling floor | Drop to `CRIT` mode; raise receiver-thread priority in the JSON config; shrink the capture window. |
| `bin_file` hit `bin_file_max_size_in_bytes` mid-run | Capture truncated by `file_size_limit_policy=0` (stop-on-limit) | Either raise the limit, or switch to `file_size_limit_policy=1` (truncate-and-continue) if losing the oldest events is acceptable. |
| Decoded events look like noise / wrong-symbol | ELF mismatch — the `--elf-file` is not the build that produced the trace | Re-confirm the ELF SHA matches the capture-time SHA per [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes) ELF-must-match-image rule; re-decode. |
| Same workload, same config, traces look different across runs | Workload non-determinism, mode/sub-mode drift, or capture window straddling different phases | Pin the workload's input, the capture window, and the JSON config; re-run; if still divergent, the workload's non-determinism is the real answer the user came for. |
| Event ordering disagrees with the kernel source | Could be real (a sync mis-issue, a comm-call ordering bug) or apparent (the renderer's ordering is not source order) | Re-read the public DOCA DPA Tools page's documented event-ordering rules; correlate captured events with the kernel's expected DPA programming events sequence. |
| Capture completes but the kernel-entry event is missing | Layer 3 (image-not-instrumented) of the error taxonomy | Confirm DPACC build flags enabled instrumentation; re-build the DPA image if not. |

The agent's rule: every change to the capture parameters
re-opens the loop. Re-running with a tweaked mode and
quoting the *previous* event-counts without re-checking
overhead / window / ELF match is exactly the failure mode
this loop replaces.

**Baseline-capture rule.** When the goal of the tracer
session is a baseline (vs an ad-hoc question), the
captured artifact must include the metadata tuple per
[`CAPABILITIES.md ## Observability`](CAPABILITIES.md#observability)
— (DOCA version, DPACC version, ELF path + SHA, mode +
sub-mode, JSON config, capture window, BlueField identity
+ firmware) — alongside the binary trace, the log file,
and the tool's stderr. Without all of these the baseline
cannot be regression-tested later.

Loop termination: stop iterating once two consecutive
runs do not change the picture — the answer is now *"this
is what the DPA-side workload does at the high-level
event layer on this image + this BlueField"*. Escalate
cross-version comparisons to
[`doca-version TASKS.md ## test`](../../doca-version/TASKS.md#test)
or
[`doca-dpa TASKS.md ## debug`](../../libs/doca-dpa/TASKS.md#debug)
with the captured artifacts as evidence.

This skill does NOT ship a "test fixture" or pre-recorded
expected output. The expected output is workload-, DPA-
image-, BlueField-, and DOCA-version-specific; pinning
one would mislead operators on a different setup.

## debug

When `doca_dpa_hl_tracer` fails to capture, the captured
trace is empty / unreadable, or the decoded events do not
match the kernel source, walk the layered error taxonomy
in
[`CAPABILITIES.md ## Error taxonomy`](CAPABILITIES.md#error-taxonomy)
in order. The shape of the diagnosis:

1. **Install.** Confirm the binary at
   `/opt/mellanox/doca/tools/doca_dpa_hl_tracer` exists,
   is executable, and its loader-dependent shared libs
   are present. Route to
   [`doca-setup ## install`](../../doca-setup/TASKS.md#configure)
   if not.
2. **Device-binding.** Confirm the BlueField the tracer
   is attaching to is visible to DOCA per
   [`doca-dpa TASKS.md ## configure`](../../libs/doca-dpa/TASKS.md#configure)
   step 1 and that the DPA processor is exposed. The
   tracer cannot attach if either is missing.
3. **Image-not-instrumented.** Confirm the DPA-side
   application is running (per
   [`doca-dpa TASKS.md ## run`](../../libs/doca-dpa/TASKS.md#run))
   and that the DPACC build flags enabled the
   tracer-instrumentation hooks per the public DPACC
   guide via
   [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md).
   An idle DPA produces an empty trace by construction.
4. **Capture-window.** Confirm the capture window
   actually overlapped the workload phase under
   investigation. Re-walk [`## configure`](#configure)
   step 4; widen or re-time the window.
5. **Decode.** Confirm the ELF passed at decode time is
   the same build that produced the trace. SHA the ELF
   at capture and decode time; mismatch means re-build
   or re-locate the right ELF.
6. **Overhead-saturated.** If `TRACE` mode produced a
   suspiciously uniform per-event interval, drop to
   `CRIT` and re-capture; if the workload's wall-clock
   shifted materially under tracing, raise the
   receiver-thread priority + pin its core in the JSON
   config.
7. **Version.** Walk
   [`doca-version TASKS.md ## debug`](../../doca-version/TASKS.md#debug)
   end-to-end; the tracer ↔ `doca-dpa` library ↔ DPACC
   triple is the common version mismatch failure.
8. **Cross-cutting.** Cause is below DOCA — driver,
   firmware, BlueField mode, IOMMU, NUMA. Hand off to
   [`doca-debug ## debug`](../../doca-debug/SKILL.md)
   and
   [`doca-setup ## debug`](../../doca-setup/TASKS.md#debug).

In every case: **quote what the tracer reported.** Do not
paraphrase event names, do not reorder fields, do not
*summarize* the event stream into a single number.

## use

The captured trace is **evidence** for a host-side DPA
debug session. The agent's hand-off:

1. **Pair the captured trace with the (DOCA version,
   DPACC version, ELF SHA, mode, JSON config, capture
   window, BlueField + firmware) tuple** per
   [`CAPABILITIES.md ## Observability`](CAPABILITIES.md#observability).
2. **Walk
   [`doca-dpa TASKS.md ## debug`](../../libs/doca-dpa/TASKS.md#debug)
   with the captured trace as the runtime evidence.**
   The host-side library skill owns the *"what to
   change next"* decision; the tracer owns the *"what
   happened on the DPA"* signal.
3. **Cross-reference with the cross-cutting debug
   ladder** in
   [`doca-debug ## debug`](../../doca-debug/SKILL.md) when
   the captured events point at a layer below DOCA
   (driver, firmware, NUMA).
4. **Retain the captured artifact set** for later
   regression hunts; a trace without the metadata
   tuple is unreplicable.

The agent's rule: the tracer captures; the
[`doca-dpa`](../../libs/doca-dpa/SKILL.md) library skill
acts. Conflating the two is the most common DPA-debug
first-touch error.

## Deferred task verbs

The verbs below are not `doca_dpa_hl_tracer` work and
should be routed out before the agent does any of them
under this skill's name.

- **DPA-side programming** (writing the DPA kernel,
  using `doca-dpa-comms`, using `doca-dpa-verbs`) →
  route via
  [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md)
  to the public DOCA DPA, DPACC, DPA-Comms, and
  DPA-Verbs guides + the shipped DPA samples. This
  tool traces those events; it does not redefine them.
- **Raw cycle profiling of the DPA processor** →
  different surface, different tool. Route via
  [`doca-public-knowledge-map ## DOCA tools`](../../doca-public-knowledge-map/SKILL.md#doca-tools).
- **Host-side `doca-dpa` API debugging** →
  [`doca-dpa TASKS.md ## debug`](../../libs/doca-dpa/TASKS.md#debug).
  The tracer's host-side error surface routes back here
  when the cause is host-side, not DPA-side.
- **Production observability of DPA workloads** →
  [`doca-telemetry`](../../libs/doca-telemetry/SKILL.md).
  The tracer is a diagnostic capture surface, not a
  continuous production observability surface.

## Command appendix

`doca_dpa_hl_tracer`-specific invocation classes the verbs
above reach for. Every row is a CLASS — the agent must
not invent flags beyond `--help` on the installed binary
and the public DOCA DPA Tools page.

**Infra-aware preamble (every row below).** Per the
bundle's detect → prefer → fall back → report contract
documented in
[`doca-structured-tools-contract ## The agent behavior contract`](../../doca-structured-tools-contract/SKILL.md#the-agent-behavior-contract),
the agent should:

1. Probe for the matching structured helper FIRST
   (`doca-env --json` for version + devices + DPA
   availability in one shot;
   `doca-capability-snapshot` for per-device DPA
   capability flags).
2. If the probe succeeds, the structured tool's output
   is the authoritative answer.
3. If the probe fails, fall back to the manual command
   in the row.
4. The schemas the structured tools emit are defined in
   [`doca-structured-tools-contract ## Schemas`](../../doca-structured-tools-contract/SKILL.md#schemas).

| Purpose (class) | Invocation (shape) | Owning step | Reads as healthy when … |
| --- | --- | --- | --- |
| Discover the documented flag surface | `doca_dpa_hl_tracer --help` + the public DOCA DPA Tools page | [`## configure`](#configure) step 2; [`## debug`](#debug) layer 1 | Prints the documented inventory of `--device`, `--mode`, `--sub-mode`, `--config-file`, `--input-file`, `--output-file`, `--parse-file`, `--elf-file`, `--log-file`. |
| Capture a `CRIT`-mode baseline | `doca_dpa_hl_tracer --device <mlx5_*> --mode CRIT --config-file <json> --output-file <bin>` | [`## run`](#run) step 2 (start-here mode) | Binary trace file grows during the capture window; stderr is quiet; on stop, the file size is bounded by `bin_file_max_size_in_bytes`. |
| Widen to a `TRACE`-mode capture for fine-grained perf | Same as above with `--mode TRACE` (and the relevant `--sub-mode`) | [`## test`](#test) iteration | Captured event stream contains the full per-event detail; overhead is acceptable for the answer the user needs. |
| Decode a captured trace against the matching ELF | `doca_dpa_hl_tracer --input-file <bin> --parse-file <parsed> --elf-file <dpa-elf>` | [`## run`](#run) step 4 | Parsed output names the documented DPA programming events; symbols resolve cleanly against the ELF. |
| Render a quick read of a captured trace | Inspect the human-readable `--log-file` alongside the binary trace | [`## run`](#run) step 5; [`## test`](#test) iteration | The log file shows decoded events with sensible timestamps. |

Three cross-cutting rules for this appendix:

- **Never invent a flag, event name, mode token, or
  JSON key.** `--help` on the installed binary and the
  public DOCA DPA Tools page are the joint contract.
- **Pair the trace with the (DOCA + DPACC + ELF SHA +
  mode + JSON config + window + BlueField + firmware)
  tuple.** Every row above presumes the tuple was
  captured.
- **Cross-link instead of duplicate.** Cross-cutting
  commands (`pkg-config --modversion doca-dpa`,
  `doca_caps --list-devs`, `dmesg`, `mlxconfig -d
  <bdf> q`) live in
  [`doca-debug ## debug`](../../doca-debug/SKILL.md) and
  [`doca-setup TASKS.md ## debug`](../../doca-setup/TASKS.md#debug);
  this appendix names only `doca_dpa_hl_tracer`-specific
  invocation classes.

## Cross-cutting

A few rules that apply across every verb in this file,
restated here so they are visible at the point of action
and not buried in [`SKILL.md`](SKILL.md):

- The **public DOCA DPA Tools page** plus the installed
  `--help` are the joint source of truth. When they
  disagree, the *installed* `--help` wins for the user's
  actual run.
- **Tracing is observation, not workload.** The DPA
  workload must already be up via
  [`doca-dpa`](../../libs/doca-dpa/SKILL.md); the tracer
  does not drive the kernel.
- **Capture is bounded.** Pick the file-size limit
  policy explicitly; the two failure modes
  (stop-on-limit vs truncate-and-continue) differ.
- **Quote the (DOCA + DPACC + ELF SHA + mode + JSON
  config + window + BlueField + firmware) tuple.** A
  trace artifact without the tuple is unreplicable.
- This skill **assumes a healthy DOCA install** with
  the DPA Tools component, a paired DPACC, a BlueField
  with a visible DPA, and the
  [`doca-dpa`](../../libs/doca-dpa/SKILL.md) host-side
  lifecycle already started. If any of those is in
  doubt, route to
  [`doca-setup`](../../doca-setup/SKILL.md) or
  [`doca-dpa`](../../libs/doca-dpa/SKILL.md) before
  running anything else here.
