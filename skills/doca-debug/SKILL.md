---
license: Apache-2.0
name: doca-debug
description: >
  Use this skill when the user is debugging any DOCA symptom — a build
  that won't compile, a link step that can't resolve a doca_* symbol, a
  runtime call returning DOCA_ERROR_*, a silent service or tool, or a
  stack trace / valgrind / core dump — and needs the layered ladder
  (install → version → build → link → runtime → program → driver),
  verbosity controls (--sdk-log-level, DOCA_LOG_LEVEL, the
  doca-{lib}-trace flavor), container-debug constraints, or how to
  capture state for a Developer Forum post. Trigger even when the user
  does not say "DOCA debug" — implicit phrasings include "undefined
  reference to doca_*", "how do I get more logs", "packets aren't
  reaching the wire", "doca_caps returned nothing", or "hugepages empty
  in the container". Refuse and route elsewhere for library-specific
  debug (Flow pipe trace, RDMA QP, Comch stats), env-class pkg-config or
  hugepages symptoms, the DOCA_ERROR_* taxonomy and lifecycle
  interpretation, and performance or incident-response work — those
  belong to other skills.
metadata:
  author: NVIDIA <doca-feedback@nvidia.com>
  kind: library
  doca_version: "3.5"
  tags:
    - doca
    - debug
compatibility: >
  No DOCA install required to read this skill (it is an overlay loaded
  against any DOCA artifact skill); the validation steps within DO
  require a live DOCA install at /opt/mellanox/doca.
---

# DOCA debug

## Examples

- **"My DOCA build / link / runtime failed — which layer is
  it?"** — this is in scope. Walk the layered ladder; do not
  jump to library-internal debug first.
- **"How do I turn up verbosity for any DOCA library or
  tool?"** — use the log-level / trace-flavor surface, then
  capture state for a forum post.

## Instructions

An `undefined reference` is a **Link-layer symptom**, not proof of a
missing library or an application-code defect. Before recommending edits,
request these observations from the **same host/container and shell that
ran the failing build**:

- `pkg-config --modversion doca-common` and the selected library's version;
  for Flow, `pkg-config --modversion doca-flow`.
- `pkg-config --libs doca-<library>`; for Flow, `pkg-config --libs doca-flow`.
- The actual failing compiler/linker invocation from the build log and its
  full error output. `pkg-config` reports flags, not the command that failed.

**If these outputs are missing, request them and pause before prescribing a
repair.** Listing a check is not evidence that it passed. Do not append a
speculative fix, guessed `-L`/`-l` flags, predicted library filenames, or a
replacement compiler command to that capture request. Once evidence is
available, compare the original invocation with the complete observed
`pkg-config` output through the existing build system; do not hand-assemble
DOCA's version-dependent split libraries. Use `ldd` only if a relevant binary
was actually produced; a declaration in a header does not prove that a linked
shared object exports the symbol.

Follow `doca-version/TASKS.md#configure` for version coherence,
`doca-programming-guide/TASKS.md#debug` for the
environment-before-program investigation, and
`doca-public-knowledge-map/SKILL.md` for verified
install locations and public documentation. A bundle skill name is not an SDK
page slug: never construct a documentation URL from it. If no matching guide
has been verified, use the [DOCA SDK index](https://docs.nvidia.com/doca/sdk/index.html)
and say the specific page is not yet confirmed.
Then follow [TASKS.md](TASKS.md#debug) for the staged investigation; later
phases are conditional on observed results, not instructions to guess a fix.

**First reply when only a linker error is available (Flow example).**
Use this capture-only shape, adapting the library to the actual question:

> This is a link-time failure. An undefined-symbol message does not tell us
> the installed version or which build setting is wrong. From the same
> host/container and shell as the failing build, please paste these outputs:
>
> ```bash
> pkg-config --modversion doca-common
> pkg-config --modversion doca-flow
> pkg-config --libs doca-flow
> ```
>
> Also include the actual failing linker invocation, full error output and
> whether the build ran on the host or in a container. Do not change flags,
> packages or source yet. Phases 3–5 are pending this evidence; no repair has
> been selected. Once the observations are available, we can compare them,
> choose one justified correction and rerun the original build. The link
> check is successful only when that build exits cleanly without the missing
> symbols; it does not establish runtime or hardware correctness.

End that first reply at the evidence request and pending decision. Do not add
an example repair command after it, even with a disclaimer to verify it later.

**Where to start:** This skill is the canonical layered-ladder
reference both `doca-setup/SKILL.md`
and `doca-programming-guide/SKILL.md`
escalate to. If the symptom does not fit cleanly in env-class or
program-class, start at the tasks companion — the
full layered ladder lives there.

## Example questions this skill answers well

The CLASSES of debug questions this skill is built to answer, each
with one worked example.

- **"My DOCA build / link / runtime / program failed — which layer is
  it?"** — worked example: *"`ld` says `undefined reference to
  doca_flow_init` — which layer?"* Answered by the canonical layered
  ladder in the tasks companion (layer 4 = Link).
- **"How do I turn up verbosity for any DOCA library or tool?"** —
  worked example: *"My DOCA Flow program is silent — where do I get
  more log output?"* Answered by the trace-flavor / log-level surface
  in the capabilities companion
  + the run-with-verbosity workflow in
  the tasks companion.
- **"How do I capture state for a forum question or bug report?"** —
  worked example: *"I'd like to file a forum question with reproducible
  context — what should I include?"* Answered by the
  capture-a-reproducible-state workflow in
  the tasks companion.
- **"What does this DOCA tool's output mean / which tool answers
  this?"** — worked example: *"`doca_caps` returned nothing for RDMA
  — does that mean unsupported?"* Answered by the
  tool-vs-capability decision tree in
  the capabilities companion
  + cross-link to `doca-caps/SKILL.md`.
- **"I'm debugging inside the NGC container — what's
  observable?"** — worked example: *"`hugepages` is empty inside the
  container; is that a real problem or a container thing?"*
  Answered by the container-specific debug constraints in
  the capabilities companion
  + the tasks companion layer 5 (Runtime).
- **"Where do I ask for help with this?"** — worked example: *"Where
  is the customer-facing DOCA forum and what should the post
  contain?"* Answered by the Developer-Forum routing rule in
  the tasks companion plus
  `doca-public-knowledge-map/SKILL.md`.

If the symptom is purely env-class, route to
`doca-setup/SKILL.md`; if purely
program-class, route to
`doca-programming-guide/SKILL.md`.
This skill is the cross-cutting layer both call into.

## When to load this skill

Load this skill when the user is debugging anything DOCA-related — a build that won't compile, a link step that can't resolve a `doca_*` symbol, a runtime call that returns `DOCA_ERROR_*`, a packet that does not appear on the wire, a service that won't start, or a tool that returns no useful output. Concretely:

- The user reports a symptom and needs to find the layer that caused it (install / version / build / link / runtime / program).
- The user asks "how do I get more logs?" or "how do I turn up the verbosity?" for any DOCA library or tool.
- The user wants to capture state for a forum question or an internal bug report (the bundle does not own the internal-bug-report channel; it routes to the public DOCA Developer Forum).
- The user is reading a stack trace, a `valgrind` output, or a core dump from a DOCA program and wants to know where to look first.
- The user is debugging *inside* the NGC DOCA container and needs to know what is and is not observable from inside it.

Do **not** load this skill for:

- *"What is `DOCA_ERROR_BAD_STATE`?", "what error codes does DOCA return?"* — that is the cross-library *error taxonomy*, owned by `doca-programming-guide/SKILL.md`. This skill consumes that taxonomy; it does not redefine it.
- *"My `pkg-config` cannot find `doca-flow`", "hugepages are not mounted", "my representor isn't visible"* — those are env-class symptoms, owned by `doca-setup/SKILL.md`. This skill is the canonical pointer that env-class debug ladder redirects to once the symptom escalates beyond install / version / build prerequisites.
- *Library-specific debugging* (Flow pipe trace, RDMA queue-pair state, Comch channel statistics) — those live in the matching library skill (e.g. `doca-flow/SKILL.md`). This skill provides the cross-cutting debug ladder; library skills layer their library-specific debug surface on top of it.

## What this skill provides

This is a **thin loader**. The body keeps only the orientation needed to pick the right next file. The substantive debug material lives in two companion files:

- the capabilities companion — what *kinds of debug surface* DOCA exposes: the layered debug model (install / version / build / link / runtime / program), the read-only-first stance, version-availability of debug tools (e.g. `doca_caps` since DOCA 2.6.0), the cross-library error taxonomy (cross-link only — owned by `doca-programming-guide`), the observability primitives DOCA emits (`stderr` logs, `--sdk-log-level`, the `doca-<lib>-trace` build flavor, library counters), and the safety constraints on debug actions (read-only first, don't mutate install tree mid-investigation).
- the tasks companion — the actual debug workflows: `## configure` (set up env for high-verbosity debug), `## test` (capture a reproducible state), `## debug` (the canonical layered ladder, the universal entry point that every library `## debug` redirects to), and the *Where to ask for help* routing (NVIDIA DOCA Developer Forum). Three other anchors (`build`, `modify`, `run`) route to `doca-programming-guide/SKILL.md`, which owns those verbs after the env / program split.

## Loading order

1. Read this skill first to confirm the user's symptom is *cross-cutting* debug (not env-class only, not program-class only, not library-internal only).
2. **For the layered debug model, the read-only stance, the version-availability of debug tools, and the observability surface DOCA emits, see the capabilities companion.**
3. **For the canonical layered debug ladder and the *capture-and-report* workflow, see the tasks companion.** The `build`, `modify`, and `run` anchors in the tasks companion are stubs that route to `doca-programming-guide/SKILL.md`; their substance lives there.
4. If the user's symptom turns out to be env-class (install / build prerequisites), hand off to `doca-setup/SKILL.md`. If program-class, hand off to `doca-programming-guide/SKILL.md`. If library-internal, hand off to the matching library skill's `## debug` (e.g. `doca-flow/SKILL.md`).

The two companion files cross-link to each other and to `doca-public-knowledge-map/SKILL.md` whenever the right answer is *"look it up in the public docs or the installed package layout"* rather than *"debug-specific guidance"*.

## Related skills

- `doca-public-knowledge-map/SKILL.md` — public DOCA documentation routing and the on-disk layout of an installed DOCA package. This skill defers all *"where is X documented"*, *"where on disk is Y"*, and *"how do I check the installed version"* questions to the knowledge-map.
- `doca-setup/SKILL.md` — env-class debug (install / build prerequisites): `pkg-config` failures, missing hugepages, representors not visible, header-vs-runtime version mismatches. `doca-setup ## debug` is the env-class layered ladder; this skill is the cross-cutting debug ladder both env and program ladders escalate to.
- `doca-programming-guide/SKILL.md` — program-class debug (lifecycle order, `DOCA_ERROR_*` interpretation, `doca_error_get_descr()` use, the validate-before-commit rule). `doca-programming-guide ## debug` is the program-class layered ladder; this skill picks up where it leaves off when the symptom involves cross-library tooling (`gdb`, `valgrind`, container introspection, core dumps).
- Library skills (e.g. `doca-flow/SKILL.md`, `doca-dms/SKILL.md`, `doca-caps/SKILL.md`) — library-specific debug overlays. Each library's `## debug` builds on the cross-cutting ladder defined here, then adds its own counters, traces, and inspector tools.
