# DOCA Bench Extension — Capabilities

**Where to start:** The DOCA Bench Extension framework is
the plug-in surface that augments
[`doca-bench`](../doca-bench/SKILL.md). The pattern overview
below names the recurring extension-class questions. Pick
the pattern first, then drill into the H2 that owns the
substance. For the *how* of executing each pattern, jump to
[TASKS.md](TASKS.md). For the parent tool's built-in modes
and inventory, see
[`doca-bench CAPABILITIES.md`](../doca-bench/CAPABILITIES.md).

This file is loaded by [`SKILL.md`](SKILL.md). It documents
*what an extension can do that built-in `doca-bench` modes
cannot*, *what the API surface looks like in broad strokes
(the shipped `doca_bench_cuda` reference exemplar is the
schema by example)*, *how the build / registration /
discovery flow works*, *what versions it ships in
(including the `DOCA_EXPERIMENTAL`-stability overlay)*,
*what its layered error and observability surfaces look
like*, and *the safety posture* the tool's role as a
loadable plug-in into a benchmark that touches the
dataplane forces.

## Pattern overview

Five recurring patterns drive every
`doca-bench-extension`-class question, and the H2s below
own one each. Pick the pattern, then jump to the owning H2.

| Pattern | Recognise it when … | Owning H2 |
| --- | --- | --- |
| **Extension-vs-built-in decision** | "Does `doca-bench` measure my workload natively, or do I need an extension?" | [`## Capabilities and modes`](#capabilities-and-modes) (Extension-as-exit-ramp) |
| **API surface design** | "What does my extension's entry-point need to look like?" "How do I copy the `doca_bench_cuda` reference?" | [`## Capabilities and modes`](#capabilities-and-modes) (API surface shape) |
| **Build / discovery flow** | "How does `doca-bench` find and load my extension at runtime?" "What does the `meson.build` look like?" | [`## Capabilities and modes`](#capabilities-and-modes) (Build / registration / discovery) |
| **`DOCA_EXPERIMENTAL` stability** | "The API is marked experimental — what does that mean for cross-release stability?" | [`## Version compatibility`](#version-compatibility) |
| **Failure mode** | "I built the extension, but `doca-bench` cannot find / load / call it." | [`## Error taxonomy`](#error-taxonomy) + [`## Observability`](#observability) |

Two non-patterns the agent must NOT collapse into the
above:

- *"Which built-in `doca-bench` mode do I pick?"* — that is
  not an extension question. Route to
  [`doca-bench`](../doca-bench/SKILL.md). Extensions exist
  only when the built-in modes don't cover the workload.
- *"How do I program a custom DOCA Flow / Comch / RMAX
  application?"* — that is a library-level question for
  the underlying primitive's skill. Extensions plug
  workloads INTO `doca-bench`; they are not a substitute
  for using the primitive directly.

## Capabilities and modes

The DOCA Bench Extension framework has three orthogonal
decisions the operator commits to in
[`## configure`](TASKS.md#configure), and one
non-decision (the parent-tool relationship) the agent
re-asserts every time.

### Extension-as-exit-ramp from built-in modes

The framework exists for ONE reason: a workload class that
no built-in `doca-bench` mode measures. The agent's first
question, before any extension-shaped answer, is *which
built-in mode might apply?* Per
[`doca-bench CAPABILITIES.md`](../doca-bench/CAPABILITIES.md),
the parent tool ships its own inventory of built-in workload
modes; the operator must walk that inventory before
committing to an extension.

Genuine extension cases the agent should validate against:

- **GPU-side workloads** that need to drive DOCA GPUNetIO
  RX / TX queues from CUDA kernels (the case the shipped
  `doca_bench_cuda` reference exemplar covers — RX /
  TX / bidir kernels in CUDA, accounting via
  `doca_bench_cuda_kernel_stats`, lifetime gated by a
  `stop_flag`).
- **Custom hardware engine workloads** that none of the
  built-in modes drives (the operator must show the
  agent why the built-in mode inventory does not apply).
- **Aggregated / multi-engine workloads** that combine
  primitives in a way no built-in mode expresses.

False extension cases — the agent should push back when
the operator proposes an extension for:

- *"I want different units / output format"* — that is a
  reporting question, not a workload question;
- *"I want to compare two built-in modes head to head"* —
  that is the parent tool's job;
- *"I want to add CLI flags"* — extensions don't add flags
  to `doca-bench`; they expose a workload class the parent
  can invoke.

### API surface shape

The shipped `doca_bench_cuda/doca_bench_cuda.h` reference
exemplar on the user's DOCA install is the schema by
example. The agent treats it as the *shape* of any custom
extension's surface, not as a memorized inventory of
required functions. Recurring axes the reference surface
demonstrates:

- A small set of **`DOCA_EXPERIMENTAL`-marked C entry
  points** (the reference exposes `*_init`,
  `*_device_query`, `*_device_synchronize`, and per-workload
  `*_start_*_kernel` family functions for nop / eth-recv /
  eth-send / eth-bidir kernels).
- Per-workload **settings structs** that the parent passes
  through (the reference uses
  `doca_bench_cuda_kernel_settings` for common fields —
  block count, threads-per-block, a `stop_flag`, a `stats`
  pointer — plus per-workload extensions that carry RX /
  TX queue handles, buffer address / mkey / size, and
  per-call timing parameters).
- Per-workload **accounting structs** (the reference uses
  `doca_bench_cuda_kernel_stats` — `jobs_processed`,
  `bytes_processed` — so the parent has a uniform way to
  read measurement output).
- A **lifetime contract** — the kernel runs until the
  `stop_flag` is set by the parent; the entry point's
  return code follows the `doca_error_t` convention.

The agent must surface that these are *shapes from the
shipped reference*, not a documented stable API the agent
can replicate from memory. A custom extension's surface
should be adapted from the reference; the reference is the
ground truth on the user's install, NOT this skill.

### Build / registration / discovery flow

Per the shipped `/opt/mellanox/doca/tools/bench_extension/meson.build`
and `/opt/mellanox/doca/tools/bench_extension/doca_bench_cuda/`
subtree, an extension is:

- a **versioned shared library** (the reference builds
  `doca_bench_cuda_impl` as a `shared_library` with
  `version : doca_version` and `soversion : doca_so_version`
  — this is the version-overlay rule in
  [`## Version compatibility`](#version-compatibility)
  applied at build time);
- compiled with the Meson `tool_cpp_args` + GPU compile
  flags the reference uses (when the extension is GPU-side);
- installed (`install : true`) into the DOCA library path
  on the user's install.

The exact runtime *discovery* mechanism the parent
`doca-bench` uses to locate and load the extension's
shared library — search path, naming convention,
registration call sequence — lives in the public DOCA
Bench documentation on `docs.nvidia.com` and the
installed `doca-bench` binary's `--help` /
extension-related flags. The agent does NOT invent the
mechanism from memory. The shipped `meson.build` provides
the build-time half of the contract; the parent tool
provides the runtime half; the agent points the operator
at both.

### Parent-tool relationship (non-decision)

Every extension lifetime is bounded by a `doca-bench`
invocation. The extension:

- has no standalone CLI of its own;
- is loaded, called, and unloaded by the parent;
- measures via the parent's reporting machinery (the
  `doca_bench_cuda_kernel_stats`-style accounting the
  reference uses is consumed by the parent);
- inherits all of the parent's safety preconditions per
  [`doca-bench CAPABILITIES.md ## Safety policy`](../doca-bench/CAPABILITIES.md#safety-policy).

The agent re-asserts this on every extension answer:
extensions plug INTO `doca-bench`; they do not replace it.

## Version compatibility

`doca-bench-extension` is **shipped with the DOCA release**
and is versioned *with the surrounding DOCA install*, not
on a separate cadence. Specifically:

- the shipped `doca_bench_cuda_impl` reference shared
  library carries `version : doca_version` and
  `soversion : doca_so_version` per the in-tree
  `meson.build` — the extension's ABI is the DOCA
  release's ABI;
- the extension's API surface is marked
  **`DOCA_EXPERIMENTAL`**. That carries a real cross-release
  stability cost the agent must surface:
  - the surface MAY change across DOCA releases without a
    deprecation window;
  - a custom extension built against one DOCA release's
    headers MUST be rebuilt against the new release's
    headers when DOCA is upgraded;
  - "rebuild on DOCA upgrade" is the **rule**, not the
    exception, for any custom extension.

This skill **does NOT** maintain its own version-handling
rules in parallel with
[`doca-version`](../../doca-version/SKILL.md). The agent
treats `doca-version` as the source of truth for:

- the four-way DOCA install match (host package, kernel
  module, firmware, parent `doca-bench`'s version);
- the BlueField-mode / device-mode questions;
- the cross-release behaviour-change questions.

The extension-specific overlay on top of that:

- **The `DOCA_EXPERIMENTAL` rebuild rule.** Every DOCA
  upgrade re-opens the question: does my custom
  extension still build, link, and load? The agent treats
  this as a smoke per [`TASKS.md ## test`](TASKS.md#test)
  on every DOCA upgrade, not as a "do it once" task.
- **`soversion` match.** The custom extension's
  `soversion` must match the running DOCA `so_version`
  for the dynamic linker / parent loader to accept it.
  Mismatches surface as load failures in
  [`## Error taxonomy`](#error-taxonomy) layer 3.
- **Toolchain version.** When the extension is GPU-side
  (CUDA), the CUDA toolkit version that built the
  extension matters; route the CUDA toolchain version
  questions to NVIDIA's public CUDA documentation on
  `docs.nvidia.com`, not to this skill.

Concretely the agent applies this rule in
[`TASKS.md ## configure`](TASKS.md#configure) (the
DOCA-version preconditions step) and in
[`TASKS.md ## debug`](TASKS.md#debug) (layer 6 — version).

## Error taxonomy

`doca-bench-extension` failures fall into seven layers. The
[TASKS.md `## debug`](TASKS.md#debug) verb walks them in
order; this section names them so the agent can route fast.

1. **Built-in-mode-would-have-sufficed.** The first layer
   is upstream of any extension build — the operator
   never needed an extension. The agent's first check
   when an extension is misbehaving is whether the
   workload class could have been expressed by a built-in
   `doca-bench` mode. Route back to
   [`doca-bench TASKS.md ## configure`](../doca-bench/TASKS.md#configure).
2. **Build-failed.** The extension's `meson.build` /
   compiler / linker reports an error. Common causes:
   wrong DOCA headers, wrong CUDA toolkit version,
   missing GPU build flags. Route the toolchain side to
   the public CUDA documentation; route the DOCA-side to
   [`doca-version TASKS.md ## debug`](../../doca-version/TASKS.md#debug)
   for the header-version question.
3. **Load-failed.** The library built, but the parent
   `doca-bench` cannot load it at runtime. Common causes:
   `soversion` mismatch with the running DOCA, library
   not on the dynamic linker's search path,
   architecture / ABI mismatch, missing CUDA runtime when
   the extension is GPU-side. Route the `ldd` /
   `LD_LIBRARY_PATH` / dynamic-linker side to
   [`doca-debug ## debug`](../../doca-debug/SKILL.md).
4. **Registration-mismatch.** The library loaded, but the
   parent cannot find the extension's expected entry
   points / version handshake. The agent's response is to
   re-check the extension's entry-point signatures against
   the shipped reference (the `DOCA_EXPERIMENTAL`-marked
   symbols in `doca_bench_cuda.h` are the shape) and
   confirm the parent's documented expectation per the
   public DOCA Bench documentation.
5. **Runtime-call-failed.** The extension was loaded and
   registered, but a per-workload call returned
   `doca_error_t` other than `DOCA_SUCCESS` or the kernel
   crashed / hung. The agent walks the extension's own
   logging plus the per-workload settings the parent
   passed in; common GPU-side cause is the `stop_flag`
   never being set so the kernel runs forever.
6. **Version.** Walk
   [`doca-version TASKS.md ## debug`](../../doca-version/TASKS.md#debug)
   end-to-end. Common extension-specific symptoms: the
   extension was built against a different DOCA release's
   headers than the parent is from (the
   `DOCA_EXPERIMENTAL` rebuild rule from
   [`## Version compatibility`](#version-compatibility))
   was missed; the CUDA toolkit version is wrong; the
   firmware doesn't expose what the extension assumes.
7. **Cross-cutting.** Hand off to
   [`doca-debug ## debug`](../../doca-debug/SKILL.md) and
   [`doca-setup ## debug`](../../doca-setup/TASKS.md#debug)
   for the env-side layers (driver, firmware, CUDA driver,
   GPU PCIe state, dynamic linker).

The full procedural ladder for each layer lives in
[`TASKS.md ## debug`](TASKS.md#debug); this section names
them so the agent can route on the FIRST symptom.

## Observability

Three kinds of evidence the framework emits, and how the
agent is expected to read each:

- **the parent `doca-bench`'s own output**, including its
  extension-load logging and its consumption of the
  extension's accounting structs (the reference's
  `doca_bench_cuda_kernel_stats` — `jobs_processed`,
  `bytes_processed`). The parent is the canonical sink
  for the extension's measurement; the extension's
  internal logging is a debugging aid, NOT the
  authoritative output.
- **the extension's own per-workload logging** during the
  smoke (the agent treats stdout / stderr from the
  extension as raw signal; quoting verbatim is the rule).
  When the extension is GPU-side, also capture relevant
  CUDA driver logging from `dmesg`.
- **the host / device cross-cutting evidence** the
  preconditions require: DOCA version, BlueField /
  ConnectX generation, firmware version, CUDA toolkit
  version (when GPU-side), the extension's
  `version` / `soversion`, the `ldd` output for the
  built library. Partial captures are not actionable.

For the cross-cutting host / device observability surfaces
(dynamic linker, CUDA driver, PCIe, firmware), route to
[`doca-debug ## Observability`](../../doca-debug/SKILL.md)
and [`doca-setup TASKS.md ## test`](../../doca-setup/TASKS.md#test).

The skill explicitly does NOT add a streaming-telemetry
export of extension state — the extension is invoked by
the parent and reports through the parent's machinery.

## Safety policy

> **Hardware-safety meta-policy applies.** Every operator
> action below inherits the safety contract defined in
> [`doca-hardware-safety ## Safety policy`](../../doca-hardware-safety/CAPABILITIES.md#safety-policy):
> the pre-flight inventory (DOCA version, firmware,
> kernel module, BlueField mode, OOB access, dataplane
> co-tenancy), the OOB / blast-radius rules, the
> change-class classification, and the smoke-before-bulk
> discipline. This section names ONLY the
> `doca-bench-extension`-specific overlay on top of that
> meta-policy.

`doca-bench-extension` is an unusual surface: it asks the
operator to LOAD CUSTOM CODE into a benchmark that touches
the device. That has four operational consequences the
agent must surface:

- **The extension runs in the parent `doca-bench`'s
  address space.** A bug in the extension can crash, hang,
  or corrupt the parent's measurement. The
  `DOCA_EXPERIMENTAL` marking on the API means the parent
  does not promise the surface won't change, but it also
  means the operator has accepted the responsibility for
  the loaded code's correctness.
- **The shipped `doca_bench_cuda` reference is GPU-side,
  and GPU-side workloads can hang the device.** A CUDA
  kernel that never sets its `stop_flag` (the reference's
  `*_kernel_settings` field) runs forever; recovering
  requires resetting the GPU. The OOB / reset
  precondition from
  [`doca-hardware-safety ## Safety policy`](../../doca-hardware-safety/CAPABILITIES.md#safety-policy)
  is non-negotiable when the extension drives a GPU.
- **All of `doca-bench`'s safety constraints apply.** Per
  [`doca-bench CAPABILITIES.md ## Safety policy`](../doca-bench/CAPABILITIES.md#safety-policy),
  the parent tool's preconditions (firmware, BlueField
  mode, dataplane co-tenancy) inherit into the extension
  invocation. Extensions add MORE preconditions; they
  never remove any.
- **The smoke-before-bulk rule from
  [`doca-hardware-safety ## Safety policy`](../../doca-hardware-safety/CAPABILITIES.md#safety-policy)
  applies twice here.** First on the build: confirm the
  extension builds cleanly and the parent can load it
  with a no-op kernel (the reference's
  `doca_bench_cuda_start_nop_kernel` is the canonical
  smoke for this) before any I/O-side workload. Second
  on the workload: confirm one minimal invocation
  completes cleanly before scaling up. Skipping either
  smoke and going straight to a full GPUNetIO RX / TX
  workload is the canonical failure mode.

In addition:

- **OOB access.** When the extension drives a GPU, OOB
  reset access is required. When the extension touches
  BlueField Flow / Comch state, the BlueField OOB
  preconditions from the parent's safety policy apply.
- **No mutation of the parent `doca-bench` binary.**
  Extensions add code at load time; they do not patch
  the parent.
- **Capture before retry.** Per
  [`doca-hardware-safety ## Error taxonomy`](../../doca-hardware-safety/CAPABILITIES.md#error-taxonomy):
  if an extension load or invocation fails, capture the
  parent's logs, the extension's logs, `ldd` output,
  and the DOCA + toolchain version stack BEFORE
  retrying. Retrying without capture is the canonical
  lost-signal failure.
- **Stop-flag discipline.** Per the reference's
  `stop_flag` field, every long-running extension kernel
  must respect a parent-set stop signal. The agent
  refuses to recommend a kernel design that has no
  bounded termination.

The full procedural application of the safety overlay
(when to abort, when to escalate, what to capture) lives
in [`TASKS.md ## debug`](TASKS.md#debug) and
[`TASKS.md ## test`](TASKS.md#test) plus the
[`doca-debug`](../../doca-debug/SKILL.md) cross-cutting
debug ladder. This section names the rules that constrain
those verbs for the bench-extension framework
specifically.
