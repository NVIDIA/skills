---
license: Apache-2.0
name: doca-dpa-hl-tracer
description: >
  Use this skill when the user runs doca_dpa_hl_tracer to
  capture/decode DPA-side traces at the programming-events
  layer (kernel entry/exit, sync points, comm primitive
  calls, RDMA WR submission, completion drain) — picking
  TRACE vs CRIT, tuning the JSON config (file-size limits
  + file_size_limit_policy, thread priorities/cores),
  decoding against the matching DPA-side ELF, or
  diagnosing empty/noisy captures. Trigger even when the
  user does not explicitly mention "DOCA DPA tracer" or
  "high-level tracer" — typical implicit phrasings include
  "DPA kernel returns wrong result but host completions
  look clean", "kernel-entry to first-comm latency is
  huge", "RDMA WR to drain gap on the DPA", "trace file
  truncated mid-run", "TRACE doubled my DPA latency", or
  "tracer wrote a file but parser shows zero events".
  Refuse and route elsewhere for writing DPA kernels,
  DPA-Comms/DPA-Verbs programming, raw per-cycle DPA
  profiling, host-side doca-dpa debugging, or production
  DPA telemetry — those belong to other skills.
metadata:
  kind: tool
compatibility: >
  Requires DOCA SDK installed at /opt/mellanox/doca on
  Linux (Ubuntu 22.04/24.04 or RHEL/SLES) with a BlueField
  device whose DPA processor is exposed to the host, plus
  the DOCA DPA Tools optional component (binary at
  /opt/mellanox/doca/tools/doca_dpa_hl_tracer). Requires a
  DPACC-built DPA-side ELF and a live doca-dpa-launched
  workload for events to fire.
---

# DOCA DPA High-Level Tracer

**Where to start:** This is a tool skill for invoking
`doca_dpa_hl_tracer` — the documented host-side CLI that
captures DPA-side execution traces in higher-level terms
(DPA programming events: kernel entry / exit, sync points,
comm primitive calls, RDMA WR submission, completions) rather
than raw cycle counts. Open [`TASKS.md`](TASKS.md) and start at
[`## configure`](TASKS.md#configure) for the
mode-vs-overhead decision and the JSON config layout, then
[`## run`](TASKS.md#run) for the
capture → decode → render pipeline. Open
[`CAPABILITIES.md`](CAPABILITIES.md) when the question is
*what does this tool actually trace*, *which DPA programming
events does it expose*, *what is the trace-overhead vs
fidelity tradeoff*, or *how does it slot into a DPA debug
loop alongside [`doca-dpa`](../../libs/doca-dpa/SKILL.md)
and [`doca-debug`](../../doca-debug/SKILL.md)*. If DPA is not
the right surface for the user's question (e.g. the bug is
host-side, the bug is in the DPACC-produced image, the user
wants raw cycle counts), the path-selection rule in
[`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes)
routes the agent before any capture is attempted.

## Example questions this skill answers well

The CLASSES of `doca_dpa_hl_tracer` questions this skill is
built to answer, each with one worked example. The class is
the load-bearing piece; the worked example is one instance.

- **"My DPA kernel is doing the wrong thing — where do I
  look?"** — worked example: *"my host-side
  `doca_dpa_kernel_launch_update_*` completes, but the
  kernel's reported result is wrong; no host-side
  `DOCA_ERROR_*`"*. Answered by the *when DPA-side
  high-level tracing is the right surface* gate in
  [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes)
  + the
  capture → decode → render flow in
  [`TASKS.md ## run`](TASKS.md#run) + the
  *which DPA programming events to focus on* rule in
  [`TASKS.md ## debug`](TASKS.md#debug).
- **"My DPA kernel is slow at a granularity that doesn't
  show up in cycle profiles — how do I see kernel-entry to
  first-comm-call latency?"** — worked example: *"my DPA
  kernel runs but the time between launch and the first
  RDMA WR submission is bigger than I expected"*. Answered
  by the event-taxonomy table in
  [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes)
  + the iterative loop in
  [`TASKS.md ## test`](TASKS.md#test) which treats trace
  overhead, mode (`TRACE` vs `CRIT`), and capture window
  as axes to tune.
- **"How do I capture a trace without burying the DPA in
  observation overhead?"** — worked example: *"`TRACE` mode
  is producing too much data and my measured DPA latency
  went up by 2x compared to without the tracer"*. Answered
  by the mode-vs-overhead tradeoff in
  [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes)
  + the `CRIT`-first guidance in
  [`TASKS.md ## configure`](TASKS.md#configure) (start
  with critical-events-only; widen to `TRACE` only when the
  bug demands per-event detail).
- **"My trace file got truncated mid-run — how should I
  configure the file-size limits?"** — worked example:
  *"binary trace file hit 5 GB and the capture stopped"*.
  Answered by the `log_file_max_size_in_bytes` /
  `bin_file_max_size_in_bytes` / `file_size_limit_policy`
  triple in
  [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes)
  + the JSON config layout in
  [`TASKS.md ## configure`](TASKS.md#configure).
- **"Is the tracer on my install, and is it paired with the
  matching `doca-dpa` library and DPACC compiler
  version?"** — worked example: *"is the tracer ABI on my
  install compatible with the DPA image my DPACC just
  produced?"*. Answered by the version-overlay in
  [`CAPABILITIES.md ## Version compatibility`](CAPABILITIES.md#version-compatibility),
  which redirects to the canonical
  [`doca-version`](../../doca-version/SKILL.md) chain and
  adds the *tracer ↔ `doca-dpa` library ↔ DPACC compiler*
  match rule.
- **"The capture file looks empty / decode failed — is the
  install broken, no events fired, or am I tracing the
  wrong thing?"** — worked example: *"`doca_dpa_hl_tracer`
  ran, wrote a file, but the parser shows zero events"*.
  Answered by the layered error taxonomy in
  [`CAPABILITIES.md ## Error taxonomy`](CAPABILITIES.md#error-taxonomy)
  (install / device-binding / DPA-image-instrumented /
  capture-window / decode-vs-elf / overhead-saturated /
  version / cross-cutting) + the layered walk in
  [`TASKS.md ## debug`](TASKS.md#debug).

## Audience

This skill serves **external developers, platform operators,
and AI agents who have already brought up a DPA-side workload
through [`doca-dpa`](../../libs/doca-dpa/SKILL.md) and now
need higher-level visibility into what the DPA kernel is
actually doing on the wire** — DPA programming events
ordering, sync gaps, comm-call latencies, RDMA-WR / completion
timing — without dropping all the way down to raw cycle
counters. Concretely:

- A DPA developer who can launch their kernel cleanly from
  the host side but whose kernel's *result* is wrong or
  whose *DPA-side performance* is below expectation, and
  who needs a DPA-side ground truth before triaging.
- A platform operator running a DPA-using workload (RDMA
  offload from accelerator, custom CC algorithm via
  `doca-pcc`) and needs to localize a regression to the
  DPA side without instrumenting the application.
- An AI agent producing a *DPA-side trace report* as
  evidence for the host-side
  [`doca-dpa TASKS.md ## debug`](../../libs/doca-dpa/TASKS.md#debug)
  ladder when the host side reports clean completions but
  the DPA-side behaviour is wrong.

It is **not** for users debugging the tracer binary itself,
**not** a substitute for the live public DOCA DPA Tools
guide, **not** the right place for users learning how to
write a DPA kernel (that audience belongs in
[`doca-dpa`](../../libs/doca-dpa/SKILL.md) plus the public
DOCA DPA / DPACC / DPA-Comms / DPA-Verbs guides), and **not**
the right place for raw per-instruction cycle profiling
(different surface, different tool — route via
[`doca-public-knowledge-map ## DOCA tools`](../../doca-public-knowledge-map/SKILL.md#doca-tools)).

The tracer is shipped as a **CLI binary** under
`/opt/mellanox/doca/tools/`, not a library you link against.
The skill uses the same `kind: tool` three-file shape as
the rest of the bundle so the agent's task-verb contract is
uniform across libraries, services, and tools.

## Language scope

`doca_dpa_hl_tracer` is a C++ host-side CLI. Its inputs are
its JSON config file, the DPA-side ELF (the
`doca_dpa_app`-class image produced by DPACC), and a running
DPA-side workload that the host-side `doca-dpa` lifecycle
already started. Its outputs are a binary trace file
(`bin_file`) and a human-readable log file (`log_file`).
The skill keeps the workflow guidance language-neutral —
the DPA-side workload it traces can be C compiled by DPACC
or any other DPA translation unit DPACC accepts — and
routes per-language questions to the public DPA / DPACC
guides via
[`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md).

## When to load this skill

Load this skill when the user is — or the agent needs to —
invoke `doca_dpa_hl_tracer` on a real host with DOCA
installed against a BlueField with a DPA processor visible to
the host, and the host-side
[`doca-dpa`](../../libs/doca-dpa/SKILL.md) lifecycle has
already brought a DPA workload up at least once. Concretely:

- Capturing a DPA-side trace to localize a DPA kernel's
  wrong-result or wrong-ordering behaviour when the
  host-side `doca-dpa` lifecycle reports clean completions.
- Capturing a DPA-side trace to localize a DPA-side
  performance gap (kernel-entry to first-comm latency,
  RDMA-WR-issue to completion gap, sync-point dwell time)
  at a granularity above raw cycle counts.
- Choosing between `TRACE` and `CRIT` capture modes based
  on the bug-vs-overhead tradeoff and the available
  capture window.
- Tuning the JSON config (thread priorities, core
  affinities, file size limits, file-size-limit policy) so
  the capture itself does not perturb the workload more
  than the bug it is investigating.
- Decoding a captured `bin_file` against the matching
  DPA-side ELF to render the human-readable event stream.
- Capturing a side-effect-bounded trace as prerequisite
  evidence for a host-side
  [`doca-dpa TASKS.md ## debug`](../../libs/doca-dpa/TASKS.md#debug)
  ladder step.

Do **not** load this skill for general DOCA orientation,
DPA-side programming model questions, raw cycle profiling,
or DOCA / DPACC install. For those, route to
[`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md),
[`doca-dpa`](../../libs/doca-dpa/SKILL.md), or
[`doca-setup`](../../doca-setup/SKILL.md).

## What this skill provides

This is a **thin loader**. Substantive material lives in two
companion files:

- `CAPABILITIES.md` — what `doca_dpa_hl_tracer` captures: the
  DPA programming event taxonomy (kernel entry / exit, sync
  points, comm primitive calls, RDMA WR submission and
  completion drain), the two documented capture modes
  (`TRACE` for full per-event, `CRIT` for critical-events
  only), the trace-overhead-vs-fidelity tradeoff, the
  config-file shape (receiver / binary-writer / file-writer
  / printer threads with priority + core affinity, file
  size limits, `file_size_limit_policy`), the
  capture-window + workload-must-be-running invariant, the
  ELF-must-match-image rule for decode, the
  version-availability overlay (tracer ↔
  [`doca-dpa`](../../libs/doca-dpa/SKILL.md) library ↔
  DPACC compiler), the layered error taxonomy
  (install / device-binding / image-instrumented /
  capture-window / decode / overhead-saturated / version /
  cross-cutting), the observability surface (binary trace
  file + log file + tool's own stderr), and the safety
  policy (capture is bounded; tracing is not a production
  observability surface).
- `TASKS.md` — step-by-step workflows for the in-scope
  task verbs: `install` (route to host-side DOCA install +
  DPA prerequisites), `configure` (mode + JSON config
  layout + capture window), `build` (route to install —
  the binary is shipped, the DPA-side application is
  user-built by DPACC), `modify` (refuse — do not patch
  the binary; modify the JSON config and the invocation
  instead), `run` (the capture flow with `--mode`,
  `--config-file`, `--output-file`), `test` (iterative
  loop tuning mode, window, and overhead), `debug` (walk
  the error taxonomy), `use` (consume the decoded trace
  in a `doca-dpa` debug session), plus a `Deferred task
  verbs` block.

The skill assumes a host where DOCA is already installed at
the standard location, a BlueField with a DPA processor is
present and visible to the host, the DPACC compiler is
installed at a version matched to the host-side DOCA, the
DPA-side application image (the ELF the tracer decodes
against) is on disk and matches what the
[`doca-dpa`](../../libs/doca-dpa/SKILL.md) lifecycle loaded,
and the operator has the privileges the public DOCA DPA
Tools guide requires.

## What this skill deliberately does not ship

This skill is **agent guidance**, not a samples or scripts
bundle. To keep the boundary clean, it deliberately does not
contain — and pull requests should not add:

- **Specific flag strings, event names, or mode tokens
  beyond what the public DOCA DPA Tools page and `--help`
  document.** The DPA programming events surface evolves
  release to release; `--help` on the installed binary is
  the authoritative inventory.
- **Pre-baked example traces or expected event timings.**
  Trace output is workload-, DPA-image-, BlueField-, and
  firmware-specific; a captured example pinned to one
  setup misleads operators elsewhere.
- **Wrappers, parsers, or rendering scripts** in any
  language that consume the binary trace format. The
  format is documented; users who want to script against
  it should read the live guide and write the parser
  against their installed version.
- **A specific tuning recommendation derived from a single
  trace.** A DPA-side perf decision (move a sync, batch a
  comm call, change a launch argument) is a workload
  question and the skill prescribes how to *capture and
  read* traces — it refuses to translate a captured gap
  into a kernel-rewrite recommendation without the user's
  own analysis.
- **A `samples/` or `reference/` subtree.** This is a thin
  loader for a shipped CLI; substantive material lives on
  the public page, in `--help`, and in
  [`doca-dpa`](../../libs/doca-dpa/SKILL.md).

## Loading order

1. Read this `SKILL.md` first to confirm the user's question
   is in scope (DPA-side high-level tracing, not DPA-side
   programming and not raw cycle profiling).
2. **For the event taxonomy, capture modes, overhead
   tradeoff, JSON config layout, version overlay, error
   taxonomy, observability, and safety policy, see
   [CAPABILITIES.md](CAPABILITIES.md).**
3. **For the documented invocations and the
   capture → decode → render workflow — `install`,
   `configure`, `build`, `modify`, `run`, `test`, `debug`,
   `use` — see [TASKS.md](TASKS.md).**

## Related skills

- [`doca-dpa`](../../libs/doca-dpa/SKILL.md) — the host-side
  DPA control library whose loaded application image the
  tracer captures. Pair them in every DPA debug session:
  `doca-dpa` brings the workload up; the tracer captures
  what the workload does at the DPA programming event
  layer. Conflating the library with the tracer is the
  most common DPA-debug first-touch error.
- [`doca-debug`](../../doca-debug/SKILL.md) — the
  cross-cutting debug ladder. The tracer slots in at the
  *runtime* layer as the DPA-side ground truth before any
  DPA-side perf or correctness conclusion is made.
- [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md)
  — routing to the public DOCA DPA Tools page on
  `docs.nvidia.com` and the rest of the public DOCA
  documentation set.
- [`doca-version`](../../doca-version/SKILL.md) — canonical
  DOCA version-handling rules. The `## Version
  compatibility` section in [`CAPABILITIES.md`](CAPABILITIES.md)
  is a concise overlay that redirects here for the body
  and adds the *tracer ↔ `doca-dpa` library ↔ DPACC
  compiler* matching rule.
- [`doca-setup`](../../doca-setup/SKILL.md) — env
  preparation, install verification, DPACC compiler
  install / verification, BlueField mode (the DPA
  processor must be exposed before any tracing is
  meaningful), and the *I have no install yet* path with
  the public NGC DOCA container.
- [`doca-structured-tools-contract`](../../doca-structured-tools-contract/SKILL.md)
  — the bundle's detect → prefer → fall back → report
  contract for structured helper tools. The command
  appendix in [`TASKS.md`](TASKS.md) honors this contract.
- [`doca-programming-guide`](../../doca-programming-guide/SKILL.md)
  — general DOCA programming patterns shared by every
  library / tool surface, including the cross-library
  `DOCA_ERROR_*` taxonomy this tool's host-side error
  layer overlays on top of when host-side `doca-dpa`
  calls fail in tandem.

The DPA-side companion libraries `doca-dpa-comms` (comm
primitives the DPA kernel itself calls) and
`doca-dpa-verbs` (RDMA verbs the DPA kernel itself calls)
are **different artifacts** that the tracer's *DPA
programming events* surface visibly names; for the
DPA-side programming model itself, route through
[`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md)
to the public DOCA DPA-Comms and DPA-Verbs guides and to
the shipped `/opt/mellanox/doca/samples/doca_dpa/` samples.
This tool *traces* their use; it does not redefine them.
