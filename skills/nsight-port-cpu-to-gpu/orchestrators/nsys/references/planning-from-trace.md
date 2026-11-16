# Planning from Trace

After NVTX instrumentation and at least one instrumented profile, build
`porting_plan.json` from the distilled trace and your best judgment.
Read source to map symbols, classify callees, and fill gaps the trace
does not settle. Do not skip the profile and plan from a whole-tree
read instead.

Plan for the **active problem** only. In `sequence` mode, re-plan
against the current (already partially-ported) tree: functions already
in `progress.json:translated_functions` are done — exclude them and
target only what this problem newly makes hot.

> **Examples are pedagogical.** Sample paths and snippets illustrate
> workflow shape only. Always derive targets from **this** codebase's
> traces — never copy example names into `porting_plan.json`.

## Prerequisites

1. Round 0 CPU sample succeeded (see profiling-guide — **required**).
2. NVTX tags follow naming grammar in `flavors/{from_lang}/nvtx.md`.
3. Canonical stats export with agent-chosen **K** and
   `extraction_meta.txt` written.
4. **Rule:** every function you propose as a translation target must
   have a matching `function:<file>:<function>` NVTX range in the
   problem window. If not — add/refine tags and re-profile before
   planning.

## Inputs

Resolve `PROFILE_ROOT` as described in the profiling guide. Paths below
are relative to that active problem's profile root, except where marked
project-relative.

| File | Purpose |
|------|---------|
| `round_N/nvtx_sum.csv` | NVTX inclusive time (trimmed top K) |
| `round_N/cpu_hotspots.csv` | Round 0 CPU symbols (if used) |
| `round_N/extraction_meta.txt` | K, coverage, re-export notes |
| `round_N/gaps.txt` | Unmarked time |
| `round_N/summary.txt` | Human windowing summary |

## Symbol → Source Mapping

CPU samples and NVTX names often show **mangled** or partial symbols.
Do not guess from one hot row.

**At scale:** spawn one **subagent per unmapped hot symbol** (from top
K) with a narrow prompt:

- Read only files suggested by symbol prefix / module name
- Map symbol → `file:function` using `grep`, `nm`, or build maps
- Return `{symbol, file, function, confidence, notes}`

Merge subagent results into `trace_evidence.json` before drafting the
plan.

For `function:` NVTX ranges, mapping is direct — use the tag string.

Optional `block:` ranges inform **refactor** tasks (extract loop nest)
but translation targets still use `function:` entries.

## trace_evidence.json Schema

```json
{
  "profile_round": 1,
  "extraction_meta_path": "{PROFILE_ROOT}/round_1/extraction_meta.txt",
  "window": {
    "iter_marker": "iter:timestep",
    "selection": "median iter: range by duration",
    "iter_ranges_seen": 200,
    "notes": "One representative iteration — not the longest, which absorbs init"
  },
  "hot_nvtx": [
    {
      "name": "function:src/kernels.f90:kernel_a",
      "inclusive_pct": 42.1,
      "mapped_file": "src/kernels.f90",
      "mapped_function": "kernel_a"
    }
  ],
  "hot_cpu_symbols": [
    {
      "symbol": "kernel_a_",
      "inclusive_pct": 38.0,
      "mapped_file": "src/kernels.f90",
      "mapped_function": "kernel_a",
      "mapping_confidence": "high"
    }
  ],
  "gaps": [
    {"description": "12% unmarked in dynamics phase", "action": "add function: tags"}
  ],
  "excluded_phases": ["phase:host:mpi_halo", "phase:host:io"]
}
```

## What IS a Translation Target

A computational **kernel**: a function whose body contains
data-parallel loops over arrays / grid cells / particles with
significant arithmetic.

Select functions with:

- Loops over arrays with real computational work
- Data-parallel structure (independent iterations, or made
  independent via reductions)
- Enough arithmetic intensity to justify GPU launch overhead

**When choosing between a caller and a callee, pick the level
that exposes the MOST data parallelism, and translate only ONE
of them.** If A loops over `k` and calls B that loops over
`(j,i)`, select A (the function owning the full `k,j,i` nest) —
B becomes a device utility, not a second target.

### Suitability scoring (1–10)

| Band | Signals |
|------|---------|
| High (7–10) | nested loops, large counts, array ops, no I/O, no allocate, independent iters |
| Medium (4–6) | reductions (reduction clause), scalar temporaries (private), modest counts |
| Low (1–3) | heavy I/O in loops, complex branching, MPI in loops, recursion, tiny loops |

## What is NOT a Translation Target

- **Device routines** — any callee invoked inside a target's
  parallel loops (small helpers *and* nested compute callees), 
  especially when shared by multiple functions that may all be ported.
  These go in `device_utilities`, ported in the utility phase as
  device-callable routines per the flavor pack (`!$acc routine
  seq`, `__device__`, `KOKKOS_INLINE_FUNCTION`).
- I/O, file handling, display code
- Init / teardown / allocation routines
- Program entry points (unless they hold substantial loop
  compute)
- Pure dispatch / wrapper functions

## Loop Hierarchy — One Target Per Nest

When a call stack splits loop dimensions across functions, walk
**up** from hot callees to the highest function whose loops
cover the grid dimensions you care about:

```text
driver_k:  do k = 1, nk
             call column_work(k, ...)   ! DO j; DO i inside
           end do
```

Select `driver_k`; add `column_work` to `device_utilities`. If
outer dimensions live in a caller but compute lives in a
callee, add a **strided-block refactor task** (see flavor
`refactoring.md`) and still list only the caller as the target.

## Call Graph Awareness

Use this after nsys has named a hot function. Walk its
neighborhood in the source; do not rank the whole tree as the
plan source.

1. Trace from the selected function down through callees.
2. At each level ask: does this function's loop nest already
   capture the full grid parallelism? If yes, stop — this is
   the target.
3. Callees called only inside a target's loops →
   `device_utilities`.
4. Two **independent** functions (neither called inside the
   other's parallel loops) may both be targets; order by any
   non-parallel call dependency between them.

There is no blanket "bottom-up" ordering within one loop
hierarchy — pick one level, not every function with a loop.

## Callability Check (MANDATORY)

Before finalizing, verify every translation target is
**externally callable** from a standalone test program — a
non-negotiable prerequisite for per-function testing.

**Externally callable** means:

- Reachable via the language's normal import/include mechanism
  (`USE module` in Fortran, a header in C/C++) — not nested
  inside a program/parent scope.
- **Explicit arguments** — no reliance on host-scoped globals
  or parent-scope variables.
- Dependencies (types, constants, arrays) reachable from
  outside.

**Blocks callability:** internal/nested subroutines, reliance
on program- or parent-scope state, inaccessible private module
state, or heavyweight host types as primary arguments (Fortran
derived types, C++ classes/`std::vector`).

**If a target is not externally callable, you MUST add a
refactor task to fix it** (extract into a module/header with
flat-array arguments; see the flavor `refactoring.md` for
language-specific decoupling). Refactor tasks complete
**before** translation.

**If the required refactor is infeasible or disproportionately
large, STOP and explain to the user.** Do not silently fall
back to whole-program testing or in-place annotation without
per-function tests until the user explicitly accepts the
deviation.

## Collecting Device Utilities

After analyzing selected functions, **deduplicate**
device-routine callees across targets — a helper used by
several kernels appears **once** in `device_utilities`. For
each entry record the original file/module, the device name,
and `used_by` (the targets that call it).

## porting_plan.json

Write `.nsight-ai-port/porting_plan.json`. Order `functions[]` by
**inclusive time** in the problem window (highest first). Each
function entry adds a `trace_evidence` link:

```json
{
  "include_guard": "USE_OPENACC",
  "overall_comment": "Summary of porting strategy",
  "device_utilities": [
    {
      "original_file": "utils/kernel_utils.f90",
      "original_module": "kernel_utils_mod",
      "function": "periodic_index",
      "device_name": "periodic_index_device",
      "used_by": ["kernel_a", "kernel_b"],
      "notes": "Simple modular index, trivial rewrite"
    }
  ],
  "functions": [
    {
      "file": "src/kernels.f90",
      "function": "kernel_a",
      "order": 1,
      "suitability": 9,
      "depends_on": [],
      "description": "42% inclusive in iter:timestep window",
      "trace_evidence": {
        "nvtx": "function:src/kernels.f90:kernel_a",
        "inclusive_pct": 42.1
      },
      "accelerator_chunks": [
        {
          "description": "Main triple-nested loop",
          "target_construct": "DO k=1,nlevels / DO j / DO i"
        }
      ]
    }
  ],
  "refactor_tasks": [
    {
      "file": "src/dynamics/timestep.f90",
      "function": "timestep_loop",
      "description": "Extract compute kernel from host driver",
      "instructions": "Extract the RHS loop into a callable function with flat array args.",
      "new_functions": [
        { "name": "compute_rhs", "suitable_for_gpu": true }
      ]
    }
  ]
}
```

Notes:

- `functions[]` — ordered translation targets (one per subskill
  run).
- `device_utilities[]` — deduplicated device-callable callees.
- `refactor_tasks[]` — structural changes (extraction,
  decoupling, strided-block) done before translation; each
  lists any `new_functions` so they can be folded into
  `functions[]`.

## User Review Gate

Present the plan with trace percentages. Show:

1. Ordered targets with suitability scores and inclusive time
2. Device utilities (host → device name map)
3. Refactor tasks needed first
4. Estimated effort (count, complexity)
5. Concerns / risks

Ask: "Does this plan look right? Add, remove, or reorder
anything?" User approval is required before translate
subskills run. Apply feedback to `porting_plan.json`, then
**seed `progress.json`** by identity (`file` + `function`):

1. Keep entries still in the plan.
   - If `status` is `completed`, preserve `status`, `phases`,
     `test_dir`, `test_executable`, and `commit_hash`.
   - Otherwise reset to pending: all `phases.*: false`, empty
     `test_dir` / `test_executable` / `commit_hash`.
2. Append a pending object (all `phases.*: false`, empty
   `test_dir` / `test_executable` / `commit_hash`) for each plan
   target not yet in `functions[]`.
3. Remove entries that are no longer in the plan **and** are
   not `completed`. Leave completed-but-dropped rows (they
   remain in `translated_functions`).
4. Set `plan_complete: true`. A phase with no work this round
   counts as done: if `refactor_tasks` is empty, set
   `refactor_complete: true`; if `device_utilities` is empty, set
   `device_utils_complete: true`.

Do not append a second object for a function already listed.

Each round **re-plans** from current traces — do not skip
planning because an earlier plan was approved. If the user
later adds or removes targets, keep `progress.json:functions[]`
in sync — no orphan or missing entries.

## If Evidence Is Insufficient

- Increase K and re-export stats (no new profile)
- Add `function:` / `block:` tags and re-profile
- Widen problem window notes in `summary.txt`

If the trace is thin, improve the evidence (above). Reading source
to interpret it is fine; do not replace the profile with a
whole-tree plan.
