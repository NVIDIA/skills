# DOCA DPA High-Level Tracer — Capabilities

**Where to start:** `doca_dpa_hl_tracer` is a host-side CLI
that captures, decodes, and renders DPA-side execution traces
at the *DPA programming events* layer rather than as raw
cycle counts. The pattern overview below names the recurring
`doca_dpa_hl_tracer`-class questions. Pick the pattern first,
then drill into the H2 that owns the substance. For the *how*
of executing each pattern, jump to [TASKS.md](TASKS.md).

This file is loaded by [`SKILL.md`](SKILL.md). It documents
*what the tracer captures*, *which DPA programming events it
exposes*, *what its capture-mode + JSON config surface looks
like*, *what the trace-overhead-vs-fidelity tradeoff is*,
*how it pairs with the [`doca-dpa`](../../libs/doca-dpa/SKILL.md)
library and the DPACC compiler*, *its layered error and
observability surfaces*, *and the safety posture* that
bounds when tracing is appropriate. For step-by-step
invocations and the capture → decode → render workflow, see
[`TASKS.md`](TASKS.md).

## Pattern overview

Every `doca_dpa_hl_tracer`-class question this skill teaches
resolves into one of SIX patterns. The patterns are CLASSES —
they apply across every DPA-side workload the tool can trace,
not just one.

| `doca_dpa_hl_tracer` pattern | Class shape | Where the substance lives |
| --- | --- | --- |
| 1. Confirm DPA-side tracing is the right surface | The bug is on the DPA side (kernel wrong-result, sync gap, wrong event ordering), the cycle profiler is too low-level, and host-side `doca-dpa` reports clean completions. If any of those does not hold, the tracer is not the right tool. | [`## Capabilities and modes`](#capabilities-and-modes) when-is-this-the-right-surface gate + [TASKS.md ## configure](TASKS.md#configure) step 1 |
| 2. Pick capture mode | `TRACE` (full per-event capture, high fidelity, high overhead) vs `CRIT` (critical-events-only, lower overhead, narrower visibility). The two modes are NOT interchangeable; quoting a `TRACE` capture as if it were `CRIT` or vice versa misrepresents what was observed. | [`## Capabilities and modes`](#capabilities-and-modes) mode-vs-overhead table + [TASKS.md ## configure](TASKS.md#configure) step 2 |
| 3. Size the capture window | Trace overhead grows with capture duration and event rate; file size limits in the JSON config bound the buffer. `file_size_limit_policy` selects *stop-on-limit* vs *truncate-and-continue*, each of which fails in a different way. | [`## Capabilities and modes`](#capabilities-and-modes) JSON-config table + [TASKS.md ## configure](TASKS.md#configure) step 3 |
| 4. Pair the trace with the DPA-side ELF | Decode requires the matching DPA-side ELF (the `doca_dpa_app`-class image DPACC produced). A mismatched ELF renders the trace as noise even if the capture itself is clean. | [`## Version compatibility`](#version-compatibility) ELF-must-match-image + [TASKS.md ## run](TASKS.md#run) decode step |
| 5. Diagnose an empty-or-broken trace | Walk the error taxonomy in [`## Error taxonomy`](#error-taxonomy) — install / device-binding / image-instrumented / capture-window / decode / overhead-saturated / version / cross-cutting — instead of guessing why the rendered output is empty. | [`## Error taxonomy`](#error-taxonomy) + [TASKS.md ## debug](TASKS.md#debug) |
| 6. Hand the trace back to the DPA debug loop | A captured trace is evidence for the host-side [`doca-dpa TASKS.md ## debug`](../../libs/doca-dpa/TASKS.md#debug) ladder. The tracer surfaces *what happened on the DPA*; the host-side library skill surfaces *what to change next*. | [TASKS.md ## use](TASKS.md#use) hand-off + [TASKS.md ## debug](TASKS.md#debug) |

Two cross-cutting rules that apply to *every* pattern above:

- **Tracing is observation, not workload.** The tracer
  captures what the DPA kernel does; it does not cause
  the workload to run. The DPA-side workload must be
  brought up by [`doca-dpa`](../../libs/doca-dpa/SKILL.md)
  first — start the host-side `doca_dpa` context, load
  the DPA application image, run at least one launch.
  A tracer started against an idle DPA produces an empty
  capture and the failure is by construction, not a bug.
- **Capture is bounded, not unbounded.** Trace overhead and
  trace file size both grow with capture duration. The
  agent's rule: pick the smallest defensible mode + window
  for the question being asked; widen only when the bug
  demands it.

## Capabilities and modes

`doca_dpa_hl_tracer` is shipped as a single CLI binary at
`/opt/mellanox/doca/tools/doca_dpa_hl_tracer` on every DOCA
install that includes the DOCA DPA Tools optional component.
It runs on the **host** and observes a DPA-side workload
already running on a BlueField the host can reach through
DOCA. There is no daemon component on the DPA side; the
DPA-side instrumentation is part of the image DPACC produced.

**When is DPA-side high-level tracing the right surface?**
A small gate the agent walks before configuring the tracer:

| Question | Right surface |
| --- | --- |
| The host-side `doca_dpa_*` call returns a `DOCA_ERROR_*` | Host-side debug; not this tool. Walk [`doca-dpa TASKS.md ## debug`](../../libs/doca-dpa/TASKS.md#debug) first. |
| The DPA-side kernel is producing wrong results, with clean host-side completions | DPA-side high-level tracing. This tool. |
| The DPA-side kernel is slower than expected at a granularity above per-instruction (kernel-entry to first-comm latency, RDMA-WR-issue to completion-drain gap, sync-point dwell) | DPA-side high-level tracing. This tool. |
| The DPA-side kernel is slow at per-instruction granularity (cache misses, branch behaviour, individual cycles) | Raw DPA cycle profiler, NOT this tool. Route via [`doca-public-knowledge-map ## DOCA tools`](../../doca-public-knowledge-map/SKILL.md#doca-tools) to the matching low-level surface. |
| The image DPACC produced is wrong / does not start | DPACC compiler diagnostics, not a runtime trace. Walk the public DPACC guide via [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md) first. |
| The DPA processor is not visible to the host at all | DPA env layer, not this tool. Walk [`doca-dpa TASKS.md ## configure`](../../libs/doca-dpa/TASKS.md#configure) step 1 + [`doca-setup ## debug`](../../doca-setup/TASKS.md#debug). |

**DPA programming event taxonomy.** The tracer exposes the
documented *DPA programming events* surface — i.e. events
the DPA programming model itself defines, not the raw
instruction stream. The exact event names + IDs are
versioned with the [`doca-dpa`](../../libs/doca-dpa/SKILL.md)
library and authoritative on `--help` of the installed
binary. The event *classes* the agent reasons about:

| Event class | What it tells the operator |
| --- | --- |
| Kernel entry / exit | A host-side launch produced an actual DPA-side function entry, and the function returned without faulting. Missing entry events when host-side launches reported success points at the host-launch → DPA-entry path. |
| Sync points | The DPA-side execution hit a documented sync primitive. Missing or out-of-order sync events versus the kernel's source order points at sync misuse. |
| Comm primitive calls (DPA-Comms surface) | The DPA kernel issued a `doca-dpa-comms`-class operation. Calls that issue but produce no completion locate where a comm path stalls. |
| RDMA WR submission (DPA-Verbs surface) | The DPA kernel issued an RDMA work request via `doca-dpa-verbs`. Pairing WR submission with the subsequent completion drain event localizes RDMA-side latency on the DPA side. |
| Completion drain | The DPA kernel observed a completion event. Used in pair with the corresponding issue event. |

The agent's rule for any *"my kernel is doing the wrong
thing"* report: name which event-class signal is
inconsistent with the kernel source, and quote the captured
events verbatim — do not paraphrase event names.

**Capture modes.** Two documented operating modes:

| Mode | What it captures | Overhead | When to pick |
| --- | --- | --- | --- |
| `TRACE` | Full per-event capture across every event class the DPA-side instrumentation emits. | Higher; can shift kernel timing by a measurable amount in tight loops. | When the bug is correctness (wrong-result, wrong ordering) or when the perf question is fine-grained (sub-microsecond gaps between events). |
| `CRIT` | Critical-events-only — a documented narrower subset (errors, faults, lifecycle transitions). | Lower; minimal kernel timing perturbation. | When the bug is *"did anything bad happen"* (crash, fault, error event) and full event detail is not needed. Also the default *"start here"* mode before widening to `TRACE`. |

The `--sub-mode` strings the binary accepts are `DBUF` or
`COMCH` — a transport selector for *how* trace data is
carried off the DPA, valid only for `TRACE` mode (not a
narrower TRACE / CRIT capture variant). The exact strings
remain authoritative on `--help` of the installed version;
the agent does not invent them.

**JSON config layout.** Capture parameters live in a JSON
config file (passed via `--config-file`). The documented
keys (re-confirm against the public DOCA DPA Tools page and
`--help` on the installed version):

| Section | Keys | What they control |
| --- | --- | --- |
| `limits_config` | `log_file_max_size_in_bytes`, `bin_file_max_size_in_bytes`, `file_size_limit_policy` | Bound the log file and binary trace file in bytes; choose between *stop-on-limit* (`file_size_limit_policy=0`) and *truncate-and-continue* (`file_size_limit_policy=1`). |
| `threads_config.receiver_thread` | `thread_priority`, `thread_core` | Priority + CPU core for the thread that drains events from the DPA side. Documented default in the shipped example config is real-time priority. |
| `threads_config.binary_writer_thread` | `thread_priority`, `thread_core` | Priority + core for the thread that writes the binary trace file. `-1` means default priority / no affinity. |
| `threads_config.file_writer_thread` | `thread_priority`, `thread_core` | Priority + core for the human-readable log file writer. |
| `threads_config.printer_thread` | `thread_priority`, `thread_core` | Priority + core for the stdout printer. |

The capture is bounded by the file-size pair: with
`file_size_limit_policy=0` (stop-on-limit) the capture
ends at the smaller limit, with `file_size_limit_policy=1`
(truncate-and-continue) the older events are evicted and
the capture continues — the two policies fail in different
ways and the operator must choose explicitly.

**Capture-window invariant.** The tracer captures events
that fire during the window between `--output-file` write
start and capture end. Events that fire outside the window
are silently lost. The agent's rule for a sweep test: pre-
plan the window to span the relevant workload phase, not
just *"a few seconds"*.

**ELF-must-match-image rule.** Decode (`--parse-file`,
`--elf-file`) requires the same DPA-side ELF that
[`doca-dpa`](../../libs/doca-dpa/SKILL.md) loaded into the
`doca_dpa_app` context. A decode against a different ELF
build produces noise / wrong-symbol output even when the
binary trace is intact. The agent's rule: snapshot the ELF
path (and a SHA of the ELF) at capture time; pair them in
the captured artifact.

## Version compatibility

For the canonical DOCA version-detection chain, the four-way
match rule, NGC container semantics, and the
headers-win-over-docs rule, see
[`doca-version`](../../doca-version/SKILL.md). The body lives
there; this skill does not duplicate it.

**The `doca_dpa_hl_tracer`-specific overlay** is:

- **Tracer ↔ `doca-dpa` library ↔ DPACC must match.** The
  tracer is the operator-side companion to the host-side
  [`doca-dpa`](../../libs/doca-dpa/SKILL.md) library and
  the DPA-side image is produced by the DPACC compiler.
  All three must come from a matched DOCA release band
  per the
  [`doca-version CAPABILITIES.md ## Version compatibility`](../../doca-version/CAPABILITIES.md#version-compatibility)
  four-way rule, plus the DPA-specific *DOCA must match
  DPACC* overlay
  [`doca-dpa`](../../libs/doca-dpa/SKILL.md) carries.
- **The DPA-side image must be the same image the tracer
  decodes against.** The captured binary trace file
  references event IDs and offsets that resolve against
  the specific ELF DPACC produced. Decoding against a
  different ELF build (e.g. an older / newer build of the
  same source) yields wrong-symbol output silently. The
  agent attaches the ELF path + SHA to the captured
  artifact.
- **Capture-mode tokens and JSON keys are versioned.**
  `TRACE` / `CRIT` / `--sub-mode` strings and the JSON
  key set evolve release to release; re-verify against
  `--help` on the installed binary and the public DOCA
  DPA Tools page rather than against this skill's prose.
- **Public DOCA DPA Tools page is the source of truth for
  the tool's command-line surface.** When that page
  disagrees with this skill (e.g. a flag landed in a
  release this skill was not written against), the live
  guide wins.

## Error taxonomy

`doca_dpa_hl_tracer`'s error surface spans the host install,
the DPA / device binding, the DPA-side image's
instrumentation, the capture window, decode-against-ELF, and
the cross-cutting partial-install / mixed-version layer.
The agent distinguishes these layers in escalating order;
jumping layers wastes the user's time on the wrong fix.

1. **Install.** The binary is missing from
   `/opt/mellanox/doca/tools/`, the DOCA DPA Tools
   optional component is not installed, the binary is
   not executable, or its loader-dependent shared libs
   are missing. Routing: route to
   [`doca-setup ## install`](../../doca-setup/TASKS.md#configure)
   to install / repair the host-side DOCA package
   selection; confirm the installed version per
   [`doca-version TASKS.md ## configure`](../../doca-version/TASKS.md#configure).
2. **Device-binding.** The tracer cannot bind the
   BlueField device (PCIe address / IB name not found /
   not accessible). Cause: the BlueField is not visible
   to DOCA on this host, the device's DPA processor is
   not exposed, or the underlying driver stack
   (`mlx5_core`, IB stack) is not loaded. Routing: first
   confirm the device is on the DOCA capability snapshot
   per [`doca-dpa TASKS.md ## configure`](../../libs/doca-dpa/TASKS.md#configure)
   step 1 and
   [`doca-setup ## test`](../../doca-setup/TASKS.md#test);
   only re-attempt the tracer once the device is visible.
3. **Image-not-instrumented.** The tracer binds but the
   DPA-side image emits no events. Cause: the DPA-side
   workload is not running (no kernel launched), the
   DPA-side application image is not the documented
   instrumented build (DPACC build flags did not enable
   the tracer-instrumentation hooks), or the workload
   ran to completion before the tracer started capture.
   Routing: confirm via
   [`doca-dpa TASKS.md ## run`](../../libs/doca-dpa/TASKS.md#run)
   that the workload is up; re-read the public DPACC
   guide via
   [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md)
   for the documented instrumentation build options.
4. **Capture-window.** Capture started after the relevant
   events fired, ended before they fired, or saturated
   the file-size limit before completing. Cause: window
   too short, file-size limits too small for the chosen
   mode, `file_size_limit_policy=0` (stop-on-limit) truncated
   the capture. Routing: re-walk
   [TASKS.md ## configure](TASKS.md#configure) step 3
   (size the window + limits); re-run with adjusted
   bounds.
5. **Decode.** The binary trace file is intact but the
   parser produces noise / wrong-symbol output. Cause:
   the ELF passed to `--elf-file` is the wrong build,
   was rebuilt after capture, or is missing the
   debug-info sections the tracer's parser uses. Routing:
   confirm the ELF SHA at decode time matches the SHA
   captured at capture time (per
   [`## Capabilities and modes`](#capabilities-and-modes)
   ELF-must-match-image rule).
6. **Overhead-saturated.** The capture completed but the
   measured behaviour is dominated by tracer overhead, not
   the workload. Symptoms: per-event timing in the trace
   is suspiciously uniform; the workload's wall-clock
   latency went up by a noticeable factor compared to
   without the tracer; the per-event interval is at the
   noise floor of the receiver thread's scheduling
   quantum. Routing: drop from `TRACE` to `CRIT` mode;
   re-run with a smaller capture window; consider
   raising the receiver-thread priority in the JSON
   config.
7. **Version.** Cross-cutting partial-install /
   mixed-version per
   [`doca-version CAPABILITIES.md ## Version compatibility`](../../doca-version/CAPABILITIES.md#version-compatibility).
   Symptoms: tracer version disagrees with
   `pkg-config --modversion doca-dpa`; DPACC compiler
   version is not aligned with the DOCA install. Routing:
   walk
   [`doca-version TASKS.md ## debug`](../../doca-version/TASKS.md#debug)
   before any further investigation.
8. **Cross-cutting.** Cause is below DOCA — driver,
   firmware, BlueField mode, IOMMU, NUMA. Routing: hand
   off to [`doca-debug ## debug`](../../doca-debug/SKILL.md)
   and
   [`doca-setup ## debug`](../../doca-setup/TASKS.md#debug).

`doca_dpa_hl_tracer` does not itself participate in the
cross-library `DOCA_ERROR_*` taxonomy because it is a CLI
driving instrumentation, not a `doca_*` library call. The
cross-library taxonomy is owned by
[`doca-programming-guide CAPABILITIES.md ## Error taxonomy`](../../doca-programming-guide/CAPABILITIES.md#error-taxonomy).

## Observability

`doca_dpa_hl_tracer`'s observability surface is **the
captured trace itself**, plus the tool's own stderr /
log file. Specifically:

- **Binary trace file (`bin_file`).** The primary
  machine-readable surface. Capped by
  `bin_file_max_size_in_bytes` in the JSON config; the
  agent's rule for any baseline that will be re-read
  later is capture the `bin_file`, not just the
  human-readable rendering.
- **Human-readable log file (`log_file`).** The
  decoded / rendered view of the events. Capped by
  `log_file_max_size_in_bytes`. Useful for a quick read
  during the capture-decode loop; the `bin_file` is the
  durable artifact.
- **Tool's own stderr.** The tracer emits DOCA log
  output to stderr (the `DOCA_LOG_BACKEND` stack) — the
  agent captures it alongside the trace files; an empty
  trace with a meaningful stderr is the most common
  install / capture-window failure mode.
- **JSON config echoed at start.** The agent's rule:
  retain the JSON config used for each capture as part
  of the captured artifact set, so the (mode + JSON
  config + ELF) triple that produced the trace is
  reconstructable later.
- **Capture-time tuple.** The minimum metadata to
  attach to any captured trace artifact: (DOCA version,
  DPACC compiler version, ELF path + SHA, mode +
  sub-mode, JSON config, capture wall-clock window,
  BlueField identity + firmware version). Without this,
  two captures in a fleet drawer cannot be ranked or
  attributed.

For the cross-cutting env-side observability primitives
(PCIe scans, `devlink`, `mlxconfig`), see
[`doca-setup CAPABILITIES.md ## Observability`](../../doca-setup/CAPABILITIES.md#observability).
For the program-side observability surface (DOCA log
levels, `DOCA_LOG_LEVEL`, `--sdk-log-level`), see
[`doca-programming-guide CAPABILITIES.md ## Observability`](../../doca-programming-guide/CAPABILITIES.md#observability).

## Safety policy

> **Overlay on the bundle-wide hardware-safety meta-policy.** The rules below are this skill's per-artifact overlay on the cross-cutting rules in [`doca-hardware-safety` CAPABILITIES.md ## Safety policy](../../doca-hardware-safety/CAPABILITIES.md#safety-policy) (specifically [### Per-artifact overlay pattern](../../doca-hardware-safety/CAPABILITIES.md#per-artifact-overlay-pattern)). When the two layers disagree, the stricter wins; when either layer says STOP, the agent stops.

`doca_dpa_hl_tracer` is a **diagnostic observation tool** —
it does not modify firmware, change BlueField mode, or
patch DPA images. Its safety surface is instead about the
*honesty* of the captured signal and the *boundedness* of
the capture itself. The rules:

- **Tracing is not a production observability surface.**
  The tool is documented for diagnostic capture; it is
  not a substitute for production telemetry (route
  there via
  [`doca-telemetry`](../../libs/doca-telemetry/SKILL.md)).
  The agent does not recommend running the tracer
  continuously against a production workload.
- **The tracer perturbs what it observes.** `TRACE`
  mode in particular can shift kernel timing by a
  measurable factor in tight loops. Any conclusion
  drawn from a trace must account for the overhead-vs-
  fidelity tradeoff in
  [`## Capabilities and modes`](#capabilities-and-modes).
  Quoting a captured latency without noting the mode
  is misleading.
- **Capture is bounded; choose the failure mode
  explicitly.** `file_size_limit_policy=0`
  (stop-on-limit) silently truncates at the limit;
  `file_size_limit_policy=1` (truncate-and-continue)
  evicts older events. Both fail in different ways
  and the operator must pick.
- **Pair the trace with the ELF + config.** A trace
  artifact without the (ELF path + SHA, JSON config,
  mode) triple is unreplicable. The agent's rule for
  every captured trace: retain the triple alongside
  the binary trace and log files.
- **Captured traces can leak workload information.**
  Event arguments, RDMA WR fields, and comm-call
  payload pointers may show up in the trace; treat
  trace files as workload-sensitive artifacts and
  apply the operator's data-handling policy to them.
- **Do not invent event names, mode tokens, or JSON
  keys.** The documented surface lives on the public
  DOCA DPA Tools page and in `--help` on the installed
  binary; prose-derived names are the most common
  hallucination failure for this skill.

## Public-source pointer

The canonical public source for `doca_dpa_hl_tracer` is the
**DOCA DPA Tools** umbrella on `docs.nvidia.com`, reachable
through
[`doca-public-knowledge-map ## DOCA tools`](../../doca-public-knowledge-map/SKILL.md#doca-tools).
The companion programming-model surface lives in the public
DOCA DPA, DPACC, DPA-Comms, and DPA-Verbs guides on the
same site. Do not invent flags, event names, mode tokens,
or JSON keys beyond what those public sources document, and
re-verify against `--help` on the installed binary.
