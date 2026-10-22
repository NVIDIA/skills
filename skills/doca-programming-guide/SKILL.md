---
license: Apache-2.0
name: doca-programming-guide
description: >
  Use this skill when the user is writing their first DOCA app or asking
  a library-agnostic programming question — picking a shipped sample to
  copy and modify, wiring pkg-config + meson (or FFI from Rust / Go /
  Python against the public C ABI), walking the Core lifecycle (create →
  configure → start → connect → use → stop → destroy), or decoding a
  DOCA_ERROR_*. Trigger on implicit phrasing: "write my first DOCA
  program", "meson line for doca_rdma_*", "DOCA_ERROR_BAD_STATE on first
  call", "call DOCA from Rust", "built clean but nothing on the wire".
  Refuse install / hugepages / pkg-config (doca-setup), docs or version
  lookup (doca-public-knowledge-map), and library-internal API work
  (matching library skill).
metadata:
  author: NVIDIA <doca@nvidia.com>
  kind: library
  doca_version: "3.5"
  tags:
    - doca
    - programming
    - first-app
compatibility: >
  No DOCA install required to read this skill (it is an overlay
  loaded against any DOCA artifact skill); the validation steps
  within DO require a live DOCA install at /opt/mellanox/doca.
---

# DOCA programming guide

## Examples

- **"How do I write my first DOCA program?"** — copy a shipped sample
  and follow the modify workflow, then the canonical pkg-config + meson
  build. Docker / NGC can compile and link; runtime needs a real
  BlueField or ConnectX.
- **"What's the lifecycle every DOCA object follows?"** — create →
  configure → start → connect → use → stop → destroy. Configuration
  finishes before start.

## Instructions

Follow Loading order below. Open the tasks companion for how-to and
the capabilities companion for coverage. Do not open every companion
file.

**Where to start:** Read [`## Audience`](#audience) to confirm the user
is *consuming* DOCA, not *contributing* to it. Then jump to the H2
that matches the verb (`## modify` for first-app derivation,
`## build` for the canonical build pattern, `## test` for the test
loop, `## debug` for the program-class debug ladder). **Docker / NGC
is the build half, not the runtime half.** The NGC container can
compile and link. Flow pipe creation, RDMA QP ops, and any call
that needs a `doca_dev` on a port need a real BlueField or
ConnectX. Do not promise the app will work entirely inside docker.

## Example questions this skill answers well

These are the CLASSES of program-class questions the skill is built
to answer, each with one worked example. Library-specific overlays
(Flow / DMS / Caps / …) live in the matching library skill; this
skill answers the library-agnostic shape.

- **"How do I write my first DOCA program for &lt;any library&gt;?"** —
  worked example: *"I have docker; help me build my first DOCA Flow
  app."* Answered by the modify-a-shipped-sample workflow in
  ## modify plus the canonical build
  pattern in ## build. Say that docker
  / NGC builds and links, but runtime needs a real NIC or DPU.
- **"What's the right build line for any DOCA library?"** — worked
  example: *"How do I compile a program that calls `doca_rdma_*`?"*
  Answered by the `pkg-config doca-<library>` pattern in
  ## build (C/C++ Track 1) and the
  FFI/bindings pattern in Track 2.
- **"What's the lifecycle every DOCA object follows?"** — worked
  example: *"What's the right order of `doca_flow_pipe_*` calls in my
  program?"* Answered by the Core order create → configure → start →
  connect → use → stop → destroy in ## Capabilities and modes
  (`cfg-create` / `init` implement configure and finish before start).
- **"`DOCA_ERROR_*` came back — what does it mean and what do I do?"**
  — worked example: *"My code got `DOCA_ERROR_BAD_STATE`."* Answered
  by the cross-library `doca_error_get_descr()` rule in
  ## Error taxonomy
  + the program-class debug order in
  ## debug.
- **"My program built and started, but does nothing on the wire."** —
  worked example: *"My Flow program runs cleanly but no traffic is
  matched."* Answered by the validate-before-commit rule in
  ## Safety policy
  and the layered program-class debug ladder in
  ## debug.
- **"What does &lt;language&gt; consumer of DOCA look like (FFI /
  bindings)?"** — worked example: *"How do I call DOCA Comch from Rust
  without writing C?"* Answered by Track 2 of
  ## build (FFI against the public C
  ABI) and the language-neutral lifecycle in
  ## Capabilities and modes.
- **"How should I classify and build all the shipped DOCA samples and
  applications? What's the difference between a sample and an
  application?"** — worked example: *"I tried to build all DOCA apps
  and 20/159 failed — which ones are real regressions vs missing
  optional stacks?"* Answered by the sample-vs-application model and
  the category / dependency / skip-vs-fail taxonomy in
  ## sample-and-app-categorization,
  which separates *"the SDK is broken"* from *"the optional GPU /
  RMAX / MPI stack is not on this BlueField"*.

If the question is env-class (install / build env / hugepages /
devices), route to `doca-setup/SKILL.md`. If it is
library-specific (Flow pipe topology, RDMA QP setup, DMS service
deploy), layer the matching library skill on top.

## Audience

This skill serves **external developers building applications that
*consume* DOCA libraries** — i.e., users whose code calls one or more
`doca_<library>_*` symbols (directly in C/C++, or through FFI /
bindings from another language). It is *programming **with** DOCA*,
not *programming **of** DOCA*: it is *not* for NVIDIA developers
contributing to DOCA itself, and it does *not* assume access to the
DOCA source tree, internal NVIDIA tooling, or any non-public
information. The only inputs it ever points the agent at are the
ones any external user has: the public docs at
[`docs.nvidia.com/doca/sdk/`](https://docs.nvidia.com/doca/sdk/),
the public catalog at
[`catalog.ngc.nvidia.com`](https://catalog.ngc.nvidia.com/), the
public GitHub repos under
[`github.com/NVIDIA`](https://github.com/NVIDIA) /
[`github.com/NVIDIA-DOCA`](https://github.com/NVIDIA-DOCA), the
public developer forum, and the on-disk `/opt/mellanox/doca` tree
that the public DOCA install (or the public NGC DOCA container,
`nvcr.io/nvidia/doca/doca`) puts on the user's host. *Where to find*
and *how to install* questions are routed elsewhere — see *Related
skills* below.

**Language scope.** DOCA itself is a C library family; every shipped
sample in `/opt/mellanox/doca/samples/` and every shipped reference
application in `/opt/mellanox/doca/applications/` is C. C and C++
consumers are the canonical case for every prescriptive workflow in
this skill. Other-language consumers (Rust, Go, Python, …) consume the
same `*.so` libraries through FFI or language-specific bindings against
the public C ABI; the skill keeps the lifecycle, capability, error,
observability, and safety guidance language-neutral, and routes the
language-specific build / FFI work back to the consumer's own toolchain
without authoring wrappers.

## When to load this skill

Load this skill when the user has DOCA installed *and* the env-class
preconditions are already satisfied (i.e.,
`doca-setup/SKILL.md` has produced a clean install
where `pkg-config doca-<library>` resolves, hugepages are mounted, and
devices are visible), and is now asking a question about **how to
actually program against DOCA** in a library-agnostic way:

- Understanding what DOCA's pieces are (libraries, apps, services,
  tools) and which side of the wire they run on.
- The canonical `pkg-config` + meson build pattern any DOCA application
  follows, regardless of which library it consumes.
- The universal *derive a custom first application from a shipped
  sample* workflow that every library skill extends with
  library-specific overrides — moved here from `doca-setup` because it
  is a programming verb, not an env verb.
- The universal DOCA Core lifecycle (`create → configure (including
  permissions and task `set_conf`) → start → connect → use → stop →
  destroy`) and how it manifests across libraries. Configuration MUST
  finish before `doca_ctx_start()`. Connect is its own step after
  start; do not write `connect/use` as one token.
- The general `DOCA_ERROR_*` pattern, `doca_error_t`, and
  `doca_error_get_descr()` — the cross-library shape, not Flow- or
  RDMA-specific overlays.
- The validate-before-commit rule and the program-side safety policy
  that every library skill inherits.
- Programming patterns that apply to consumers in *any* language
  (C/C++ directly, FFI / bindings for Rust / Go / Python / …) — the
  skill keeps the patterns language-neutral and points the agent at
  the public C ABI as the authoritative surface.

Do **not** load this skill for:

- *"Is my install healthy? Why does `pkg-config` not find `doca-flow`?
  How do I mount hugepages? I don't have DOCA installed yet — can I
  use a container?"* — env-class questions belong in
  `doca-setup/SKILL.md`, which owns install
  verification, env preparation, env-class debugging, and the
  *no-install → NGC container fallback* path for any user on macOS,
  Windows, or Linux without DOCA.
- *"What is DOCA? Where is the Flow programming guide? Which package
  do I install?"* — routing / orientation questions belong in
  `doca-public-knowledge-map/SKILL.md`.
- *"How do I construct a Flow pipe / set up an RDMA queue / use Comch
  to send a message?"* — library-internal API questions belong in the
  matching library skill (e.g.
  `doca-flow/SKILL.md`).

## What this skill provides

This is a **thin loader**. The body keeps only the orientation needed
to pick the right next file. The substantive material lives in two
companion files:

- the capabilities companion — what every DOCA program looks like in the
  abstract: the shape of DOCA (host / DPU / switch, libraries / apps /
  services / tools, build flavor selection rationale), the universal
  program lifecycle, the unified version-compat rule that applies to
  any program linking DOCA, the cross-library `DOCA_ERROR_*` taxonomy,
  the program-side observability surface (DOCA logging, capability
  snapshots), and the program-side safety policy that every library
  skill inherits.
- the tasks companion — step-by-step workflows for the six in-scope programming
  verbs: `configure`, `build`, `modify`, `run`, `test`, `debug`. The
  `## modify` verb owns the universal *derive a custom first app from
  a sample* pattern that every DOCA library skill extends with
  library-specific overrides; the `## build` verb owns the canonical
  `pkg-config doca-<library>` build pattern in two language tracks
  (C/C++ direct, non-C via FFI).

This skill assumes `doca-setup/SKILL.md` has already
produced a clean install. It does **not** cover env preparation; that
is `doca-setup`'s job, including the *no-install → NGC container
(`nvcr.io/nvidia/doca/doca`)* fallback for users on macOS, Windows, or
Linux without DOCA.

## What this skill deliberately does not ship

This skill is **agent guidance**, not a samples or templates bundle.
It deliberately does not contain — and pull requests should not add:

- **Pre-written DOCA application source code, in any language.** This
  includes C / C++ files, Rust crates, Go packages, Python modules,
  and wrapper code for any other language. The DOCA API surface
  evolves between releases and code written from documentation prose
  cannot be verified without compiling / linking / FFI-loading it
  against the live library on a real install. The verified DOCA
  application source code is the shipped C samples on the user's
  installed system; the agent's job is to route the user to that file
  and prescribe a minimum-diff modification on it
  (## modify) — for C/C++ users — or to
  route non-C users to the public C ABI surface that their bindings
  will call (## build Track 2), *not* to
  author the wrapper.
- **Standalone build manifests** (`meson.build`, `CMakeLists.txt`,
  `Cargo.toml`, `setup.py`, `go.mod`, …) parked inside the skill. The
  agent constructs the build manifest *in the user's project
  directory* against the user's installed DOCA, where `pkg-config
  --modversion doca-<library>` is the source of truth.
- **A `samples/`, `bindings/`, or `reference/` subtree** of any kind.
  A mock or incomplete artifact in this skill's tree, even one
  labeled "reference", is misleading: users will read it as
  buildable.

## Loading order

1. Read this skill first to confirm the user's question is a
   programming-class question (not env, not routing, not library-API).
2. **For the DOCA shape, the universal lifecycle, the cross-library
   error taxonomy, the program-side observability surface, and the
   safety policy that every library skill inherits, see
   the capabilities companion.**
3. **For step-by-step programming workflows — `configure`, `build`,
   `modify`, `run`, `test`, `debug` — see the tasks companion.**

If the user is asking for a *first app* in a specific library, walk
through ## modify for the universal
copy-and-edit pattern, then hand off to the library skill (e.g.
`doca-flow/SKILL.md`) for the library-specific values
to swap.

## Related skills

- `doca-public-knowledge-map/SKILL.md`
  — public DOCA documentation routing and the on-disk layout of an
  installed DOCA package. This skill defers all *"where is X
  documented"*, *"where on disk is Y"*, and *"how do I check the
  installed version"* questions to the knowledge-map.
- `doca-setup/SKILL.md` — env preparation, install
  verification, env-class debugging, and the *I have no install yet*
  procedure with the NGC container (`nvcr.io/nvidia/doca/doca`) as
  the universal Stage-1 fallback for any user on macOS, Windows, or
  Linux without DOCA. This skill assumes `doca-setup`'s preconditions
  are already satisfied.
- `doca-flow/SKILL.md` — DOCA Flow on BlueField.
  Extends this skill's `## modify` (universal first-app derivation)
  with the Flow-specific list of fields to swap.
- `doca-container-deployment/SKILL.md`
  — shipping an application that already builds: container manifests,
  rollout, and service-readiness verification. This skill ends once the
  program compiles and runs locally; it does not cover deploying it.
