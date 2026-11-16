# Nsys Orchestrator

A problem-scoped profile-plan-port loop using flavor packs and shared
subskills.

Port only what the **active problem** exercises. Resolve it by matching
`progress.json:current_problem` to `config.json:problems[]`.

This orchestrator runs the round loop defined in the
[router](../../SKILL.md#the-porting-round-loop). It does **not**
redefine the loop — it supplies:

- **Phase 1 (Plan)** — from trace evidence
  ([planning-from-trace.md](references/planning-from-trace.md)).
- **Phase 5 (Assess)** — re-profile the problem, then summarize
  application performance (**how much of the timeline is GPU work** and
  **how much whitespace/idle gap remains**) for the user, who decides
  whether to run another optimization round.

## Before Round 0

Do this for the **active problem** (repeated per problem in sequence
mode; profiles are namespaced under `profiles/{problem_id}/` — see
profiling guide):

1. Run `scripts/profile-check.sh {to_lang}-{framework}` from the
   skill root. If CPU
   sampling prerequisites fail (see profiling guide), **stop** and
   help the user fix the environment — do **not** skip profiling
   and plan from a whole-tree read instead.
2. Confirm the active problem's `description` and `run_performance`
   are set.
3. Baseline must pass: **`build_correctness`**, **`run_correctness`**,
   and **one successful `run_performance`** for the active problem
   (see setup guide).
4. Do not start nsys until `run_performance` exits with the active
   entry's `expected_exit_code`.
5. Round 0 requires `nsys profile --sample=process-tree` to produce
   usable hotspot data. If sampling fails, stop and iterate with the
   user (`/proc/sys/kernel/perf_event_paranoid` ≤ 2, Docker/seccomp,
   `nsys status --environment`).
   If sampling succeeds but sample count is zero, the problem run
   is too short — scale the workload (user approval); see the
   profiling guide.

## Round Loop (nsys mapping)

The loop structure and continue-gate live in the
[router](../../SKILL.md#the-porting-round-loop). This is how each
canonical phase is realized for nsys. Read each round:

- [profiling-guide.md](references/profiling-guide.md)
- [planning-from-trace.md](references/planning-from-trace.md)
- [iteration-report-template.md](references/iteration-report-template.md)

| Canonical phase | nsys behavior |
|-----------------|------------------|
| 1 Plan | Update `porting_plan.json` from the trace plus judgment (planning-from-trace). |
| 2 Refactor | refactor-task subskill (this round's `refactor_tasks[]`). |
| 3 Device utils | translate-utility subskill (this round's `device_utilities[]`). |
| 4 Translate | translate-function subskill, per target. |
| 5 Assess | **Re-profile** the problem (GPU build when ready), then write the **performance assessment** — GPU-active % of the problem window and remaining timeline whitespace — and present the [iteration report](references/iteration-report-template.md). |

After each Phase 4 translate, re-run **`run_correctness`** for
numerical correctness. The Phase 5 re-profile builds with
`build_performance` and profiles `run_performance` under nsys (see
profiling-guide
[Performance Assessment](references/profiling-guide.md#performance-assessment-post-translate)).

The continue-gate (Phase 5 → ask "another round?") is the router's; the
iteration report is its presentation form. Record the round in both
`progress.json:rounds[]` (router) and `profile_rounds[]` (nsys profile
artifacts). In `sequence` mode, tag each round record with the active
`problem` id and store artifacts under
`.nsight-ai-port/profiles/{problem_id}/round_N/` so problems never share
a profile directory.

Resolve flavor: `flavors/{to_lang}/{framework}/`.

### NVTX

Add markers per `flavors/{from_lang}/nvtx.md` (and
`flavors/cpp/kokkos/nvtx.md` when `framework` is `kokkos`). Git
commit NVTX separately from GPU translation commits.

Two markers carry **measurement** semantics (the profiling window),
not a plan veto:

- `iter:` defines the profiling window. Emit exactly one `iter:` name,
  pushed once per iteration of the outermost repeating unit of work.
- Every `phase:` MUST be classed `phase:compute:` or `phase:host:`.
  `phase:host:` is a hint that work is unsuited to GPU (MPI, I/O, init) 
  for windowing and commentary.

If a GPU-build assessment reports `gpu_active_pct: 0.0`, suspect these
markers **before** you conclude the port did nothing.

### GPU build env

Use `USE_{FRAMEWORK}`, the upper-cased `config.json:framework` (e.g.
`USE_OPENACC`, `USE_OPENMP`, `USE_CUDA`, `USE_KOKKOS`). Never assume
a specific framework.

## Subskills

| Phase | Path |
|-------|------|
| 2 Refactor | [refactor-task](../../subskills/refactor-task/workflow.md) |
| 3 Device utils | [translate-utility](../../subskills/translate-utility/workflow.md) |
| 4 Translate | [translate-function](../../subskills/translate-function/workflow.md) |

## Wrap-up

When the router reaches Phase 6:

1. Set `progress.json`: `status: "completed"`.
2. Summarize profile rounds, artifact paths, and functions ported —
   per problem and cumulatively across the sequence.
3. Append a final `usage_log` snapshot and aggregate it into
   `session_summary.stats.usage`.
4. **Write `session_summary`** to `.nsight-ai-port/skill_feedback.json`
   (required — include **nsys_extraction** section; see
   [skill-feedback-log.md](../../references/skill-feedback-log.md)).
5. Git-commit `skill_feedback.json` if it changed.
6. Present the retrospective in chat (not JSON-only).
