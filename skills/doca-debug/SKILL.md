---
license: Apache-2.0
name: doca-debug
description: >
  Use this skill when the user is debugging any DOCA symptom — a build
  that won't compile, a link step that can't resolve a doca_* symbol,
  a runtime call returning DOCA_ERROR_*, a silent service or tool, or
  a stack trace / valgrind / core dump — and needs the layered ladder
  (install → version → build → link → runtime → program → driver),
  verbosity controls (--sdk-log-level, DOCA_LOG_LEVEL, the
  doca-{lib}-trace flavor), container-debug constraints, or how to
  capture state for a Developer Forum post. Trigger even when the user
  does not say "DOCA debug" — implicit phrasings include "undefined
  reference to doca_*", "how do I get more logs", "packets aren't
  reaching the wire", "doca_caps returned nothing", or "hugepages
  empty in the container". Refuse and route elsewhere for
  library-specific debug (Flow pipe trace, RDMA QP, Comch stats),
  env-class pkg-config or hugepages symptoms, the DOCA_ERROR_*
  taxonomy and lifecycle interpretation, and performance or
  incident-response work — those belong to other skills.
metadata:
  kind: library
compatibility: >
  No DOCA install required to read this skill (it is an overlay loaded
  against any DOCA artifact skill); the validation steps within DO
  require a live DOCA install at /opt/mellanox/doca.
---

# DOCA debug

**Where to start:** This skill is the canonical layered-ladder
reference both [`doca-setup ## debug`](../doca-setup/TASKS.md#debug)
and [`doca-programming-guide ## debug`](../doca-programming-guide/TASKS.md#debug)
escalate to. If the symptom does not fit cleanly in env-class or
program-class, start at [`TASKS.md ## debug`](TASKS.md#debug) — the
full layered ladder lives there.

## Example questions this skill answers well

The CLASSES of debug questions this skill is built to answer, each
with one worked example.

- **"My DOCA build / link / runtime / program failed — which layer is
  it?"** — worked example: *"`ld` says `undefined reference to
  doca_flow_init` — which layer?"* Answered by the canonical layered
  ladder in [`TASKS.md ## debug`](TASKS.md#debug) (layer 4 = Link).
- **"How do I turn up verbosity for any DOCA library or tool?"** —
  worked example: *"My DOCA Flow program is silent — where do I get
  more log output?"* Answered by the trace-flavor / log-level surface
  in [`CAPABILITIES.md ## Observability`](CAPABILITIES.md#observability)
  + the run-with-verbosity workflow in
  [`TASKS.md ## run`](TASKS.md#run).
- **"How do I capture state for a forum question or bug report?"** —
  worked example: *"I'd like to file a forum question with reproducible
  context — what should I include?"* Answered by the
  capture-a-reproducible-state workflow in
  [`TASKS.md ## test`](TASKS.md#test).
- **"What does this DOCA tool's output mean / which tool answers
  this?"** — worked example: *"`doca_caps` returned nothing for RDMA
  — does that mean unsupported?"* Answered by the
  tool-vs-capability decision tree in
  [`CAPABILITIES.md ## Capabilities and modes`](CAPABILITIES.md#capabilities-and-modes)
  + cross-link to [doca-caps](../tools/doca-caps/SKILL.md).
- **"I'm debugging inside the NGC container — what's
  observable?"** — worked example: *"`hugepages` is empty inside the
  container; is that a real problem or a container thing?"*
  Answered by the container-specific debug constraints in
  [`CAPABILITIES.md ## Safety policy`](CAPABILITIES.md#safety-policy)
  + [`TASKS.md ## debug`](TASKS.md#debug) layer 5 (Runtime).
- **"Where do I ask for help with this?"** — worked example: *"Where
  is the customer-facing DOCA forum and what should the post
  contain?"* Answered by the Developer-Forum routing rule in
  [`TASKS.md ## debug`](TASKS.md#debug) plus
  [doca-public-knowledge-map](../doca-public-knowledge-map/SKILL.md).

If the symptom is purely env-class, route to
[`doca-setup ## debug`](../doca-setup/TASKS.md#debug); if purely
program-class, route to
[`doca-programming-guide ## debug`](../doca-programming-guide/TASKS.md#debug).
This skill is the cross-cutting layer both call into.

## When to load this skill

Load this skill when the user is debugging anything DOCA-related — a build that won't compile, a link step that can't resolve a `doca_*` symbol, a runtime call that returns `DOCA_ERROR_*`, a packet that does not appear on the wire, a service that won't start, or a tool that returns no useful output. Concretely:

- The user reports a symptom and needs to find the layer that caused it (install / version / build / link / runtime / program).
- The user asks "how do I get more logs?" or "how do I turn up the verbosity?" for any DOCA library or tool.
- The user wants to capture state for a forum question or an internal bug report (the bundle does not own the internal-bug-report channel; it routes to the public DOCA Developer Forum).
- The user is reading a stack trace, a `valgrind` output, or a core dump from a DOCA program and wants to know where to look first.
- The user is debugging *inside* the NGC DOCA container and needs to know what is and is not observable from inside it.

Do **not** load this skill for:

- *"What is `DOCA_ERROR_BAD_STATE`?", "what error codes does DOCA return?"* — that is the cross-library *error taxonomy*, owned by [`doca-programming-guide CAPABILITIES.md ## Error taxonomy`](../doca-programming-guide/CAPABILITIES.md#error-taxonomy). This skill consumes that taxonomy; it does not redefine it.
- *"My `pkg-config` cannot find `doca-flow`", "hugepages are not mounted", "my representor isn't visible"* — those are env-class symptoms, owned by [`doca-setup ## debug`](../doca-setup/TASKS.md#debug). This skill is the canonical pointer that env-class debug ladder redirects to once the symptom escalates beyond install / version / build prerequisites.
- *Library-specific debugging* (Flow pipe trace, RDMA queue-pair state, Comch channel statistics) — those live in the matching library skill (e.g. [`doca-flow ## debug`](../libs/doca-flow/TASKS.md#debug)). This skill provides the cross-cutting debug ladder; library skills layer their library-specific debug surface on top of it.

## What this skill provides

This is a **thin loader**. The body keeps only the orientation needed to pick the right next file. The substantive debug material lives in two companion files:

- [CAPABILITIES.md](CAPABILITIES.md) — what *kinds of debug surface* DOCA exposes: the layered debug model (install / version / build / link / runtime / program), the read-only-first stance, version-availability of debug tools (e.g. `doca_caps` since DOCA 2.6.0), the cross-library error taxonomy (cross-link only — owned by `doca-programming-guide`), the observability primitives DOCA emits (`stderr` logs, `--sdk-log-level`, the `doca-<lib>-trace` build flavor, library counters), and the safety constraints on debug actions (read-only first, don't mutate install tree mid-investigation).
- [TASKS.md](TASKS.md) — the actual debug workflows: `## configure` (set up env for high-verbosity debug), `## test` (capture a reproducible state), `## debug` (the canonical layered ladder, the universal entry point that every library `## debug` redirects to), and the *Where to ask for help* routing (NVIDIA DOCA Developer Forum). Three other anchors (`build`, `modify`, `run`) exist for lint compliance and route to [`doca-programming-guide`](../doca-programming-guide/SKILL.md), which owns those verbs after the env / program split.

## Loading order

1. Read this `SKILL.md` first to confirm the user's symptom is *cross-cutting* debug (not env-class only, not program-class only, not library-internal only).
2. **For the layered debug model, the read-only stance, the version-availability of debug tools, and the observability surface DOCA emits, see [CAPABILITIES.md](CAPABILITIES.md).**
3. **For the canonical layered debug ladder and the *capture-and-report* workflow, see [TASKS.md](TASKS.md).** The `build`, `modify`, and `run` anchors in `TASKS.md` are stubs that route to [`doca-programming-guide`](../doca-programming-guide/SKILL.md); their substance lives there.
4. If the user's symptom turns out to be env-class (install / build prerequisites), hand off to [`doca-setup ## debug`](../doca-setup/TASKS.md#debug). If program-class, hand off to [`doca-programming-guide ## debug`](../doca-programming-guide/TASKS.md#debug). If library-internal, hand off to the matching library skill's `## debug` (e.g. [`doca-flow ## debug`](../libs/doca-flow/TASKS.md#debug)).

The two companion files cross-link to each other and to [`doca-public-knowledge-map`](../doca-public-knowledge-map/SKILL.md) whenever the right answer is *"look it up in the public docs or the installed package layout"* rather than *"debug-specific guidance"*.

## Related skills

- [`doca-public-knowledge-map`](../doca-public-knowledge-map/SKILL.md) — public DOCA documentation routing and the on-disk layout of an installed DOCA package. This skill defers all *"where is X documented"*, *"where on disk is Y"*, and *"how do I check the installed version"* questions to the knowledge-map.
- [`doca-setup`](../doca-setup/SKILL.md) — env-class debug (install / build prerequisites): `pkg-config` failures, missing hugepages, representors not visible, header-vs-runtime version mismatches. `doca-setup ## debug` is the env-class layered ladder; this skill is the cross-cutting debug ladder both env and program ladders escalate to.
- [`doca-programming-guide`](../doca-programming-guide/SKILL.md) — program-class debug (lifecycle order, `DOCA_ERROR_*` interpretation, `doca_error_get_descr()` use, the validate-before-commit rule). `doca-programming-guide ## debug` is the program-class layered ladder; this skill picks up where it leaves off when the symptom involves cross-library tooling (`gdb`, `valgrind`, container introspection, core dumps).
- Library skills (e.g. [`doca-flow`](../libs/doca-flow/SKILL.md), [`doca-dms`](../services/doca-dms/SKILL.md), [`doca-caps`](../tools/doca-caps/SKILL.md)) — library-specific debug overlays. Each library's `## debug` builds on the cross-cutting ladder defined here, then adds its own counters, traces, and inspector tools.
