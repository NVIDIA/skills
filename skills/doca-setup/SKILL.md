---
license: Apache-2.0
name: doca-setup
description: >
  Use this skill when the user is dealing with the DOCA environment
  around their workload — verifying an install is healthy, preparing
  the build env (pkg-config, headers, LD_LIBRARY_PATH, hugepages,
  devlink, representors), debugging env-class failures, deciding
  container-vs-bare-metal deployment shape, or reaching a DOCA install
  from a host that doesn't have one yet via the NGC DOCA container
  Stage-1 fallback. Trigger even when the user does not explicitly
  mention "DOCA setup" — typical implicit phrasings include "I just
  got a BlueField, what now", "my code is built, how do I run it",
  "pkg-config can't find doca-flow", "no free 2048 kB hugepages",
  "representor X not found", "I'm on a Mac and want to learn DOCA".
  Refuse and route elsewhere for library API specifics (Flow pipes,
  RDMA queues), the modify-a-sample first-app workflow or DOCA_ERROR_*
  program-side debugging, and "where is X documented" knowledge-map
  questions — those belong to other skills.
metadata:
  kind: library
compatibility: >
  No DOCA install required to read this skill (it is an overlay loaded
  against any DOCA artifact skill); the validation steps within DO
  require a live DOCA install at /opt/mellanox/doca.
---

# DOCA setup

**Where to start:** If the user's question is *deployment-shaped*
(*"how do I deploy"*, *"how do I run my DOCA workload"*, *"I just got a
BlueField, what now"*, *"my code is built, what next"*), walk
[`TASKS.md ## recognize`](TASKS.md#recognize) **first**. It is the
bundle's front-door: it detects the system shape (host x86 / BlueField
Arm bare-metal / DPU-only / fresh laptop), asks the developer the
minimal set of questions needed to disambiguate, and routes to the
correct downstream skill — the container deployment path
([`doca-container-deployment`](../doca-container-deployment/SKILL.md)),
the bare-metal hardware deployment path
([`doca-bare-metal-deployment`](../doca-bare-metal-deployment/SKILL.md)),
or the no-hardware fallback
([`TASKS.md ## no-install`](TASKS.md#no-install)). The wrong failure
mode is to silently steer every developer onto containers because the
agent loaded that skill first; `## recognize` exists to prevent that.

If the user does not have DOCA installed yet, jump straight to
[`TASKS.md ## no-install`](TASKS.md#no-install) for the NGC container
path. Otherwise read [`## When to load this skill`](#when-to-load-this-skill)
to confirm the question is env-class, then route to the section below
that matches the user's intent.

## Example questions this skill answers well

The CLASSES this skill is built to handle, each with one worked example.
The skill must answer the *class*; the worked example is illustrative.

- **"I want to deploy a DOCA workload — what is the right path?"** —
  worked example: *"I just got a BlueField; my code is built; what
  now?"* Resolved by the front-door decision tree in
  [`TASKS.md ## recognize`](TASKS.md#recognize), which detects the
  system shape, asks the minimum residual question, and routes to
  either the container or the bare-metal deployment skill.
- **"Verify DOCA is installed and healthy."** — worked example:
  *"Is DOCA Flow actually available on this box?"* Resolved by the
  install-health snapshot in [`TASKS.md ## test`](TASKS.md#test) plus the
  version-detection rules in
  [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes).
- **"I do not have DOCA installed — what now?"** — worked example: *"I'm
  on macOS and want to learn DOCA before I get a BlueField."* Resolved
  by [`TASKS.md ## no-install`](TASKS.md#no-install) (NGC DOCA container
  as universal Stage-1).
- **"Prepare the build environment for any DOCA library."** — worked
  example: *"`pkg-config --cflags doca-flow` returns nothing — what's
  missing?"* Resolved by the build-prep workflow in
  [`TASKS.md ## configure`](TASKS.md#configure) and the build-class
  error taxonomy in
  [`CAPABILITIES.md ## Error taxonomy`](CAPABILITIES.md#error-taxonomy).
- **"Prepare the runtime preconditions on a real DPU box."** — worked
  example: *"hugepages / representors / devlink — what's the minimum set
  before I run my first DOCA Flow program?"* Resolved by
  [`TASKS.md ## configure`](TASKS.md#configure) and the runtime
  observability rules in
  [`CAPABILITIES.md ## Observability`](CAPABILITIES.md#observability).
- **"Diagnose an env-class failure (install / build / runtime)."** —
  worked example: *"My DOCA Flow program built fine but says
  `pkg-config` cannot find it at runtime."* Resolved by the layered
  env-class debug workflow in [`TASKS.md ## debug`](TASKS.md#debug).
- **"Change something about the environment safely."** — worked
  example: *"I want to switch eswitch mode from legacy to switchdev."*
  Resolved by the safety constraints in
  [`CAPABILITIES.md ## Safety policy`](CAPABILITIES.md#safety-policy).

If the question is library-API-shaped (Flow pipe construction, RDMA
queue setup, …) or program-shaped (how to build, modify a sample, debug
the program itself), route to
[`doca-programming-guide`](../doca-programming-guide/SKILL.md) or the
matching library skill instead — env-class only lives here.

## When to load this skill

Load this skill when the user is dealing with the **environment around DOCA** — installing it, verifying the install is healthy, preparing the build / runtime preconditions, debugging env-class failures, figuring out *how to reach an install* from a host that doesn't have one yet, or asking a **deployment-shaped question** that hasn't yet been routed (containers vs. bare-metal) — the front-door routing decision lives here. Concretely:

- The deployment-shape routing front door: *"I just got a BlueField, what now?"*, *"my code is built, how do I run it?"*, *"how do I deploy this?"* All of these load `## recognize` first so the agent does not silently push the user onto the wrong path.
- Verifying that the DOCA install is healthy and that the build environment can find it (`pkg-config`, headers, library paths).
- Preparing the runtime: hugepages, `devlink` device visibility, representor enumeration, kernel-module prerequisites.
- Diagnosing common setup-class failures: missing `*.pc` file, hugepages not mounted, representor not visible, header-vs-runtime version mismatch.
- The *I have no install yet* path: the user is on macOS, Windows, or a Linux box without DOCA, and needs to reach an environment where DOCA is actually installed. The canonical Stage-1 answer is the public **NGC DOCA container** at `nvcr.io/nvidia/doca/doca` (works on any OS that runs Docker; no NVIDIA hardware required for the build / read / learn loop). See [`TASKS.md ## no-install`](TASKS.md#no-install).

Do **not** load this skill for:

- *"What is DOCA?", "where is the developer guide?", "where is the install layout documented?"* — those are routing questions; use [`doca-public-knowledge-map`](../doca-public-knowledge-map/SKILL.md).
- *"How do I derive a custom first application from a sample?", "how do I structure a DOCA build?", "what does `DOCA_ERROR_BAD_STATE` mean?"* — those are **programming-class** questions and live in [`doca-programming-guide`](../doca-programming-guide/SKILL.md), which owns the universal `## modify` (first-app derivation), the canonical `## build` pattern, the universal lifecycle, and the cross-library `DOCA_ERROR_*` taxonomy.
- *Library-internal API questions* (Flow pipe construction, RDMA queue setup, etc.) — those belong in the matching library skill (e.g. [`doca-flow`](../libs/doca-flow/SKILL.md)). This skill stops at *"the install is healthy and the env is ready"*; it does not own program semantics.

## What this skill provides

This is a **thin loader**. The body keeps only the orientation needed to pick the right next file. The substantive env material lives in two companion files:

- `CAPABILITIES.md` — what the install / build / runtime *environment* surface looks like: install profiles (`doca-all`, `doca-ofed`, `doca-networking`), where the build flavors (release vs trace) live on disk and how to point `LD_LIBRARY_PATH` at them, the env-side version-detection rules, the env-class error taxonomy (`pkg-config` not finding `doca-flow`, hugepages not reserved, representors not visible), what a healthy install looks like under observation, and the safety constraints on environment changes (hugepages global, `mlxconfig` reset, eswitch mode change).
- `TASKS.md` — env workflows: `recognize` (the front-door system-shape detect + dev-Q decision tree that routes deployment-shaped questions to either container or bare-metal), `configure` (env prep), `test` (install health snapshot), `debug` (env-class layered diagnosis), and `no-install` (the *I have no install yet* procedure with the NGC container as Path 0). Three other anchors (`build`, `modify`, `run`) exist for lint compliance and route to [`doca-programming-guide`](../doca-programming-guide/SKILL.md), which owns those verbs after the env / program split.

This skill assumes nothing about whether DOCA is installed — the `## no-install` workflow exists precisely for the *fresh laptop* case.

## Loading order

1. Read this `SKILL.md` first to confirm the user's question is env-class, deployment-routing-class, not programming-class or knowledge-map-class.
2. **If the question is beginner-orientation-shaped (*"I am new to DOCA, guide me to my first app"* or similar), open the answer with the [`TASKS.md ## no-install` Stage 1 vs Stage 2 staged-roadmap table](TASKS.md#stage-1-vs-stage-2--open-every-i-am-new-to-doca-answer-with-the-staged-roadmap) BEFORE any command.** Beginners overwhelm easily when the first thing the agent emits is a stack of commands; the staged roadmap establishes *where the user is going* before *what to type next*. If the agent's first emitted block is a `docker pull` or `pkg-config` invocation rather than the two-row stage table, the answer is mis-shaped — back up.
3. **If the question is deployment-shaped (containers vs. bare-metal not yet decided), walk [TASKS.md ## recognize](TASKS.md#recognize) FIRST.** That is the front-door routing decision and must happen before any other env work.
4. **For install profiles, build-flavor disk locations, env-side version-detection, the env-class error taxonomy, and the env-side safety policy, see [CAPABILITIES.md](CAPABILITIES.md).**
5. **For env workflows — `configure`, `test`, `debug`, and the critical `no-install` (NGC container path) — see [TASKS.md](TASKS.md).** The `build`, `modify`, and `run` anchors in `TASKS.md` are stubs that route to [`doca-programming-guide`](../doca-programming-guide/SKILL.md); their substance lives there. The NGC tag selection rule (no guessing) lives in [`TASKS.md ## no-install` → *How to pick an NGC tag without guessing*](TASKS.md#how-to-pick-an-ngc-tag-without-guessing) and is mandatory before any `docker pull` recommendation.
6. Once the env is healthy, hand off to the deployment skill `## recognize` routed you to ([`doca-container-deployment`](../doca-container-deployment/SKILL.md) for the container path, [`doca-bare-metal-deployment`](../doca-bare-metal-deployment/SKILL.md) for the bare-metal hardware path) for the deployment runtime, then to [`doca-programming-guide`](../doca-programming-guide/SKILL.md) for the build pattern, the first-app workflow, and the program-side error / debug order, and finally to the matching library skill (e.g. [`doca-flow`](../libs/doca-flow/SKILL.md)) for library-specific API guidance.

Both companion files cross-link to each other and to `doca-public-knowledge-map` whenever the right answer is *"look it up in the public docs or the installed package layout"* rather than *"setup-specific guidance"*.

## Related skills

- [`doca-container-deployment`](../doca-container-deployment/SKILL.md) — the container deployment runtime (kubelet standalone + YAML pod-spec drop) for any DOCA service container on BlueField. `## recognize` here routes to this skill when the developer's workload + system shape land on the container path.
- [`doca-bare-metal-deployment`](../doca-bare-metal-deployment/SKILL.md) — the bare-metal hardware deployment runtime (host x86 OR BlueField Arm direct launch — systemd / tmux / direct invocation, hardware-resource binding, per-tenant isolation, restart discipline) for a DOCA-linked binary. `## recognize` here routes to this skill when the developer's workload + system shape land on the bare-metal path.
- [`doca-public-knowledge-map`](../doca-public-knowledge-map/SKILL.md) — public DOCA documentation routing and the on-disk layout of an installed DOCA package. This skill defers all *"where is X documented"*, *"where on disk is Y"*, and *"how do I check the installed version"* questions to the knowledge-map.
- [`doca-programming-guide`](../doca-programming-guide/SKILL.md) — general DOCA programming patterns once the env is healthy: the canonical `pkg-config doca-<library>` build pattern, the universal *derive a custom first app from a sample* workflow (with C / C++ + non-C tracks), the universal lifecycle, and the cross-library `DOCA_ERROR_*` taxonomy. Anything beyond *"is the install healthy and the env ready"* lives there.
- [`doca-flow`](../libs/doca-flow/SKILL.md) — DOCA Flow on BlueField. Builds on this skill for env preparation and on `doca-programming-guide` for the universal first-app derivation, then layers Flow-specific overrides on top.
