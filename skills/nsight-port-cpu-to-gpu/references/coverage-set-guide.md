# Performance Coverage-Set Guide

Read this when the user asks **you** to generate the performance
workload(s) during the interview (Q3/Q4) — either a single
"typical use" workload or a **sequence** that covers the major
compute paths. If the user supplies their own scripts, copy them in
(setup guide Q4) and skip this guide.

The core idea: dynamic profiling (nsys) and per-round assessment can
only see the code a workload actually **executes**. Anything a
workload never runs is invisible and never gets ported. A single
problem rarely exercises a large code's full compute path, so a
sequence of problems — each hitting a distinct code path — lets the
port accumulate coverage one workload at a time.

---

## Single mode — one "typical use" workload

Goal: one workload that exercises the codebase's **dominant compute
path** as a representative user would run it.

1. Inspect the build system, `README`, example inputs, and any
   `examples/` or `benchmarks/` directory to find the most common
   invocation.
2. Prefer the smallest input that still spends the **majority of its
   wall time in the main compute kernel(s)** — not in setup/IO.
3. One program invocation, deterministic, ideally under ~10 minutes
   (see setup guide "Performance Scripts").
4. State plainly which compute path it covers and what it does **not**
   (e.g. "covers the 2-D solver only; 1-D and 3-D paths are not
   exercised"). This tells the user when they may later want to add
   problems (a subsequent run can append them — see the run-lifecycle
   section of the setup guide).

Present the drafted `build_performance.sh` / `run_performance.sh` for
approval before proceeding.

---

## Sequence mode — a coverage set

Goal: a **minimal ordered set** of workloads that together exercise
the major compute code paths. Fewer, well-chosen problems beat many
redundant ones — each new problem should light up compute code the
earlier ones did not.

### Step 1 — Enumerate the compute-path axes

Compute coverage is driven by **which code paths run**, not by problem
size. Find the axes that select distinct kernels/branches. Sources, in
order of reliability:

1. **The project's own regression / CI / test matrix.** These decks
   were designed to hit specific code paths — the best coverage map
   you have. Look at `ctest` lists, CI YAML, `tests/`, and example
   input decks.
2. **Command-line / input flags** that switch algorithms — any knob
   that selects a different kernel or branch (e.g. problem
   dimensionality, discretization or solver choice, boundary-condition
   handling, or optional physics/features the code can toggle). The
   relevant axes are whatever *this* codebase exposes.
3. **Static structure** — `grep` for large `if (mode == …)` /
   `select case` dispatch and for families of routines that differ
   only by a suffix or parameter (a common sign of one code path per
   variant).

Write the axes and the knob that selects each into a short table
(this becomes the rationale you present to the user).

### Step 2 — Pick a minimal covering set

- One problem per **distinct dominant compute path**, not per input
  size. Merge axes that always co-occur.
- Skip paths that are **GPU-only / new code** (nothing to port *from*)
  and host-only paths (MPI/IO); note them as out of scope.
- Aim for the fewest problems that cover every major compute family;
  call out any deliberately deferred path.

### Step 3 — Order the set (core-first)

Front-load the problems that exercise the **shared / core** compute
(the common routines most paths reuse), then the specialized ones.
Because each problem re-plans from the **current** (already
partially-ported) codebase, later problems mostly port only what is
**new** to them — the shared ops are already on the GPU and show as
done. Good ordering minimizes total work.

### Step 4 — Assign ids and author scripts

Give each problem a short stable `id` — a neutral slug that names the
path it covers (e.g. `case_a`, `case_b`, or a short descriptive tag
drawn from *this* codebase's own terminology). Author a
`build_performance_{id}.sh` /
`run_performance_{id}.sh` pair per problem under `.nsight-ai-port/`
(each exactly one program invocation — same rules as the single case).
Record each in `config.json:problems[]` with `id`, `description`,
script paths, and a `covers` note.

### Step 5 — Present for approval

Show the proposed set as a table: `id`, what it runs, and **which
compute path it covers**. Get explicit user approval before creating
the scripts and starting the round loop. Do not silently drop or add
problems after approval without asking.

---

## Correctness note (per problem)

The correctness gate (`run_correctness`, setup guide Q6) is separate
from the performance workloads and may already run many cases. When a
sequence exercises genuinely different code paths, prefer that
`run_correctness` carry a golden CPU reference for **each** problem's
code path — "each problem still matches its own CPU reference" is a
stronger gate than a single aggregate sanity check. This is a
recommendation, not a required restructuring of the correctness pair.
