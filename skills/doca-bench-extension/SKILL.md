---
license: Apache-2.0
name: doca-bench-extension
description: >
  Use this skill when the operator is authoring, building, loading,
  or debugging a custom doca-bench plug-in — a versioned shared
  library with DOCA_EXPERIMENTAL-marked C entry points that
  doca-bench loads to measure a workload class its built-in modes
  do not cover, with doca_bench_cuda as the shipped reference
  exemplar. Trigger even when the user does not say
  "doca-bench-extension" or "doca_bench_cuda" — typical implicit
  phrasings include "no built-in doca-bench mode fits my workload",
  "how do I benchmark a CUDA GPUNetIO RX/TX kernel", "doca-bench
  cannot find or load my custom .so", "extension exported symbols
  do not match what the parent expects", "soversion mismatch after
  a DOCA upgrade", or "my GPU kernel hangs because stop_flag was
  never set". Refuse and route elsewhere for questions about which
  built-in doca-bench mode to pick, DOCA GPUNetIO programming
  semantics, CUDA toolkit installation, or contributor work on
  in-tree extensions — those belong to other skills.
metadata:
  kind: tool
compatibility: >
  Requires DOCA SDK installed at /opt/mellanox/doca on Linux
  (Ubuntu 22.04/24.04 or RHEL/SLES) with a BlueField DPU or
  ConnectX NIC. Source tree: `/opt/mellanox/doca/tools/bench_extension/`
  (underscored, NOT kebab-case); the built shared library
  `libdoca_bench_cuda_impl.so` lands in the platform libdir on a
  binary install. Also needs `pkg-config doca-common` and, for
  the GPU-side reference exemplar (DOCA GPUNetIO RX/TX kernels),
  an NVIDIA GPU + matching CUDA toolkit.
---

# DOCA Bench Extension

**Where to start:** This is a tool skill for the **extension /
plug-in framework** that augments
[`doca-bench`](../doca-bench/SKILL.md) — NOT a workload-shape
skill on its own. Open [`TASKS.md`](TASKS.md) and start at
[`## configure`](TASKS.md#configure) to commit to the three-axis
decision (workload class is genuinely outside doca-bench's
built-in modes × extension API surface fits × parent-tool
co-load is acceptable), then [`## build`](TASKS.md#build) for
how a custom extension is compiled and laid out, then
[`## run`](TASKS.md#run) for how `doca-bench` discovers and
invokes the extension, then [`## test`](TASKS.md#test) for the
smoke-before-bulk loop the agent applies to every new
extension. Open [`CAPABILITIES.md`](CAPABILITIES.md) when the
question is *what an extension can do that built-in
`doca-bench` modes cannot*, *what the extension API surface
looks like in broad strokes (the `DOCA_EXPERIMENTAL` C entry
points the shipped reference exposes)*, *how the
build / registration / discovery flow works*, or *how the
extension's lifetime is bounded by the parent `doca-bench`
invocation*. If `doca-bench` itself is the question, route to
[`doca-bench`](../doca-bench/SKILL.md). If the question is
"which built-in `doca-bench` mode do I pick?", that is also
[`doca-bench`](../doca-bench/SKILL.md) — extensions are the
*exit ramp* for workloads built-in modes do not cover.

## Example questions this skill answers well

- *"My workload class is `<X>` — does `doca-bench` measure it
  natively, or do I need an extension?"* — the
  extension-vs-built-in decision question. The agent walks
  the user back to [`doca-bench`](../doca-bench/SKILL.md)'s
  built-in mode inventory FIRST and only routes to the
  extension framework when no built-in mode applies.
- *"I want to benchmark a CUDA / GPU-side workload that
  drives DOCA GPUNetIO RX and TX queues. Where do I start?
  Is there a reference extension I can copy?"* — the agent
  surfaces the shipped `doca_bench_cuda` extension under
  `/opt/mellanox/doca/tools/bench_extension/doca_bench_cuda/` as the
  reference exemplar and walks the operator through its
  API surface and build shape.
- *"How does `doca-bench` actually discover and load my
  custom extension at runtime? Is it a versioned shared
  library? What does my entry-point need to look like?"* —
  the build / registration / discovery flow question. The
  agent walks the Meson-built shared library shape, the
  versioning, and the parent-tool's runtime discovery path
  (which the agent does NOT invent from memory — the
  shipped extension's `meson.build` and the public DOCA Bench
  documentation on `docs.nvidia.com` are the source of
  truth).
- *"The API headers I have are marked `DOCA_EXPERIMENTAL`.
  What does that mean for my extension's stability across
  DOCA releases? Am I going to have to rebuild it every
  release?"* — the experimental-surface and version
  compatibility question.
- *"Once I build my extension, what is the cheapest possible
  smoke I can run before pointing my real workload at it?
  How do I know `doca-bench` actually loaded it, called
  into it, and that the call returned the data the parent
  tool expected?"* — the smoke-before-bulk question.
- *"My custom extension builds, but `doca-bench` says it
  cannot find / load / call it. Where do I look first?"* —
  the layered-debug question that distinguishes
  build-failures, load-failures, registration-mismatches,
  and runtime-call-failures.

## Audience

Experienced AI agents and platform / performance engineers
who already use [`doca-bench`](../doca-bench/SKILL.md) for
the built-in workload modes and now have a workload class
that the built-in modes do not cover. Readers are expected
to be comfortable with native build systems (Meson, in this
codebase), shared-library packaging on Linux, and the
`DOCA_EXPERIMENTAL` API stability contract. If the user
asks about GPU-side benchmarking via the shipped
`doca_bench_cuda` reference extension, the reader is also
expected to be familiar with DOCA GPUNetIO and CUDA toolchain
basics — those domains live in their own skills, not here.

This skill is NOT for:

- operators who can express their workload with one of
  `doca-bench`'s built-in modes — that is
  [`doca-bench`](../doca-bench/SKILL.md);
- operators who want to benchmark a different DOCA primitive
  (Flow, Comch, RMAX) via that primitive's own
  measurement tool — route to that tool;
- contributors authoring or modifying the in-tree extensions
  themselves (this skill is for external operators consuming
  the framework, not for internal DOCA contributors).

## Language scope

A `doca-bench` extension surfaces as:

1. A **versioned shared library** on Linux (`.so` with
   `soversion` matching the DOCA release), built via the
   `doca-bench-extension` Meson rules in the shipped
   `/opt/mellanox/doca/tools/bench_extension/meson.build` and the
   per-extension subdirectory (the reference exemplar is
   `doca_bench_cuda/`).
2. A small set of **`DOCA_EXPERIMENTAL`-marked C entry
   points** that the parent `doca-bench` invokes — i.e. the
   API surface declared in the extension's header file.
   The shipped `doca_bench_cuda/doca_bench_cuda.h` is the
   reference for what that surface shape looks like in
   practice (`*_init`, `*_device_query`,
   `*_device_synchronize`, and per-workload kernel-start
   entry points such as `*_start_nop_kernel`,
   `*_start_eth_recv_kernel`, `*_start_eth_send_kernel`,
   `*_start_eth_bidir_kernel`).
3. A set of **per-workload settings structs** that the
   parent passes through (e.g. the reference exemplar's
   `doca_bench_cuda_kernel_settings`,
   `doca_bench_cuda_eth_rx_kernel_settings`,
   `doca_bench_cuda_eth_tx_kernel_settings`,
   `doca_bench_cuda_eth_bidir_kernel_settings` carry block
   counts, threads-per-block, RX / TX queues, buffer
   address / mkey / size, a stop flag, and a stats
   pointer).

The skill itself is Markdown. The user's extension source
is whatever language the workload requires (C / C++ / CUDA
in the reference case). The agent does NOT prescribe a
language beyond what the shipped reference demonstrates.

## When to load this skill

Load `doca-bench-extension` when ANY of the following is
true:

- the user explicitly mentions `doca-bench-extension`, the
  `doca_bench_cuda` reference extension, the
  `doca_bench_cuda_impl` shared library, or any of the
  `DOCA_EXPERIMENTAL` extension entry points;
- the user has confirmed (via
  [`doca-bench TASKS.md ## configure`](../doca-bench/TASKS.md#configure))
  that none of `doca-bench`'s built-in workload modes
  measures the class they want, and an extension is the
  exit ramp;
- the user wants to copy / extend the shipped
  `doca_bench_cuda` reference into a custom GPU-side
  workload extension;
- the user is debugging why `doca-bench` cannot find / load
  / call a custom extension they built.

Co-load this skill with:

- [`doca-bench`](../doca-bench/SKILL.md) (the parent tool —
  ALWAYS co-loaded; extensions only have value as
  plug-ins into `doca-bench`);
- [`doca-version`](../../doca-version/SKILL.md) (the
  `DOCA_EXPERIMENTAL` surface is versioned with DOCA; the
  extension's `soversion` is the DOCA `soversion`; the
  four-way version match applies);
- [`doca-gpunetio`](../../libs/doca-gpunetio/SKILL.md) when
  the extension is GPU-side and uses GPUNetIO RX / TX
  queues like the reference exemplar (route the GPUNetIO
  semantics there, not here);
- [`doca-debug`](../../doca-debug/SKILL.md) and
  [`doca-setup`](../../doca-setup/SKILL.md) for the
  env-side debug ladder (driver, firmware, CUDA toolkit,
  dynamic linker).

Do NOT load this skill when the user's workload fits a
`doca-bench` built-in mode — extensions add cost (build
toolchain, version churn, the experimental-surface
contract); the built-in modes are always the first answer to
try.

## What this skill provides

Three companion files in this directory, each owning a
different question shape:

- [`SKILL.md`](SKILL.md) — this file. Audience, scope,
  loading order, related skills. Routes everything else.
- [`CAPABILITIES.md`](CAPABILITIES.md) — *what an extension
  can do that the built-in modes cannot*, *what the API
  surface looks like in broad strokes*, *how the
  build / registration / discovery flow works*, *what
  versions it ships in (including the
  `DOCA_EXPERIMENTAL`-stability overlay on top of
  `doca-version`)*, *the layered error taxonomy*,
  observability, and the safety policy overlay.
- [`TASKS.md`](TASKS.md) — the procedural verbs
  (`configure`, `build`, `run`, `test`, `debug`, etc.) plus
  a `doca-bench-extension`-specific command appendix and
  the agent-side `use` workflow that consumes the captured
  extension run.

The combined skill teaches an AI agent to drive the
*extension-author-and-wire-in class* of `doca-bench`
questions: confirm an extension is needed at all; locate
the shipped reference exemplar
(`/opt/mellanox/doca/tools/bench_extension/doca_bench_cuda/`); copy
its build + API surface shape; build a versioned shared
library that matches the DOCA release; smoke that the
parent `doca-bench` actually loads it; diagnose layered
failures when it does not.

## What this skill deliberately does not ship

- **Inventory of `doca-bench`'s built-in workload modes.**
  That belongs to [`doca-bench`](../doca-bench/SKILL.md).
  This skill is the exit ramp for what the built-in modes
  do not cover; it does not duplicate the parent's mode
  inventory.
- **Invented `DOCA_EXPERIMENTAL` entry-point names beyond
  what the shipped reference declares.** The shipped
  `doca_bench_cuda/doca_bench_cuda.h` on the user's
  install is the reference for what the surface shape
  looks like; the agent does not assert other extensions
  exist with specific signatures.
- **A canonical "right" extension layout.** The shipped
  `doca_bench_cuda` reference IS the canonical layout;
  rewriting it here would drift from the source of truth.
  The agent points the operator at the shipped tree and
  walks the operator through *adapting* it.
- **A documented runtime discovery mechanism the agent
  invents.** The exact mechanism `doca-bench` uses to
  locate and load extensions (search path, naming
  convention, registration call) lives in the public DOCA
  Bench documentation on `docs.nvidia.com` and the
  installed `doca-bench` binary. The agent points the
  operator there rather than asserting a mechanism from
  memory.
- **DOCA GPUNetIO programming details.** When the
  extension is GPU-side (as the reference exemplar is),
  the GPUNetIO RX / TX queue semantics live in
  [`doca-gpunetio`](../../libs/doca-gpunetio/SKILL.md);
  this skill cross-links rather than duplicates.
- **CUDA toolchain installation guidance.** Route to the
  public NVIDIA CUDA Toolkit documentation on
  `docs.nvidia.com`; this skill does not duplicate it.
- **Library-internal `doca-bench` invocation details
  unrelated to extensions.** The parent's CLI flags,
  pipeline shapes, and built-in workload classes belong
  to [`doca-bench`](../doca-bench/SKILL.md).

## Loading order

When a `doca-bench-extension` question arrives:

1. Confirm DOCA is installed AND `doca-bench` is reachable
   on the user's install — if not, route to
   [`doca-setup`](../../doca-setup/SKILL.md);
2. **Confirm none of `doca-bench`'s built-in modes covers
   the workload class** — if any of them does, route back
   to [`doca-bench TASKS.md ## configure`](../doca-bench/TASKS.md#configure)
   and stop. Extensions are the exit ramp, not the first
   answer;
3. Read [`CAPABILITIES.md`](CAPABILITIES.md) to commit to
   the three-axis decision and walk the reference
   exemplar's API surface shape;
4. Read [`TASKS.md`](TASKS.md) and walk
   `## configure → ## build → ## run → ## test → ## debug`
   in that order; do NOT start with `## run` without the
   build precondition step.

## Related skills

Cross-link conventions follow the bundle's relative path
contract from `tools/<X>/`:

- [`doca-bench`](../doca-bench/SKILL.md) — the parent tool.
  **ALWAYS co-loaded.** Extensions are plug-ins into
  `doca-bench`; they do not replace it, they do not have a
  standalone CLI, they do not measure anything without the
  parent invoking them. Every question on this skill
  presupposes the parent.
- [`doca-version`](../../doca-version/SKILL.md) — the
  `DOCA_EXPERIMENTAL` surface is versioned with DOCA; the
  extension's `soversion` matches the DOCA release per
  the shipped `meson.build`. The four-way version match
  applies; rebuilding the extension across DOCA upgrades
  is the rule, not the exception.
- [`doca-gpunetio`](../../libs/doca-gpunetio/SKILL.md) —
  when the extension is GPU-side and uses GPUNetIO RX /
  TX queues like the reference `doca_bench_cuda`. Route
  the GPUNetIO semantics there.
- [`doca-setup`](../../doca-setup/SKILL.md) — DOCA install
  posture (does `doca-bench` exist? does the
  `doca_bench_cuda_impl` reference library exist? is the
  CUDA toolchain installed when needed?).
- [`doca-debug`](../../doca-debug/SKILL.md) — the
  cross-cutting debug ladder for env-side issues (dynamic
  linker, library search path, CUDA driver / toolkit,
  firmware).
- [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md)
  — routing to the public DOCA Bench / DOCA GPUNetIO pages
  on `docs.nvidia.com` and the release notes for the
  documented extension lifecycle / discovery mechanism.
- [`doca-structured-tools-contract`](../../doca-structured-tools-contract/SKILL.md)
  — the agent's detect → prefer → fall back → report
  contract for the structured helpers
  (`doca-env --json`, `doca-capability-snapshot`,
  `version-matrix.json`) the build / load preconditions
  rely on.
- [`doca-hardware-safety`](../../doca-hardware-safety/SKILL.md)
  — the canonical hardware-safety meta-policy that
  [`CAPABILITIES.md ## Safety policy`](CAPABILITIES.md#safety-policy)
  overlays. Extensions are external code loaded into
  `doca-bench`; the safety implications of loading
  experimental code into a benchmark that touches the
  dataplane / device are real.

This skill assumes the user has built shared libraries on
Linux before and knows what a Meson build is. Background
material on those topics belongs in the toolchain docs, not
in this skill.
