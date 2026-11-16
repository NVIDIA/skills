# Translation Guide — Inline Mode — Fortran + OpenMP offload

**Read this file only when `config.json:translation_mode` is
`"inline"`.** For the default `"sibling"` mode read
[translation-sibling.md](translation-sibling.md) instead — do not mix the two.

Flavor pack: `flavors/fortran/openmp/`. This guide covers **Phase 4**
(per-function translation) for inline mode. Phase 3 device utilities
live in [device-utils-inline.md](device-utils-inline.md). For directive
syntax and pitfalls use [patterns.md](patterns.md); for the
verification protocol use
[verification-guide.md](../../../references/verification-guide.md).

This is **OpenMP target offload**, not host multithreading. Use
`-mp=gpu` on nvfortran (never `-fopenmp`); gfortran uses `-fopenmp
-foffload=nvptx-none`.

## What inline mode is

The GPU version is produced by **editing the original function in
place** instead of writing a separate
`{parent}/openmp/{function}_openmp.f90`. This works because `!$omp`
directives are inert comments without `-mp=gpu`, so one source
serves both the CPU and GPU build paths. A renamed, frozen copy of
the original becomes the CPU reference for the standalone test.

### Inline vs sibling at a glance

| Concern | Sibling (translation-sibling.md) | Inline (this file) |
|---------|--------------------------|--------------------|
| Translated code | new `_openmp` file | original file, edited in place |
| CPU reference | untouched original (in tree) | renamed frozen copy in `openmp/` |
| Test compares | original vs `_openmp` | reference copy vs in-place original |
| Call-site `#ifdef` | selects `_openmp` | usually none (see Phase 4e) |
| Device utilities | renamed `_device` module | annotate originals in place |
| Exec marker | at call site | inside the in-place function |

## Directory layout (inline)

| Artifact | Path |
|----------|------|
| In-place GPU code | original `{parent}/{source}.f90` (edited) |
| CPU reference copy | `{parent}/openmp/{function}_reference.f90` |
| Standalone test | `{parent}/openmp/test/test_{function}/` |

The reference copy is a **test-only** artifact. It must NOT be
linked into the main project — its procedure keeps the original
logic and would collide at link time with the in-place version,
which retains the original name.

---

## Phase 4: Per-Function Translation

## Phase 4a: Build System Exploration

`.nsight-ai-port/build_notes.txt` should exist. If not, generate it
from [build-integration.md](build-integration.md).

Read it before scaffolding so this function's GPU build uses the
project guard and OpenMP GPU flags, not a guessed toolchain.

---

## Phase 4b: Scaffold (reference copy + test)

**Prerequisite — externally callable.** Confirm the target can be
called from a standalone program (via `USE module_name`). If it is
nested inside a `program ... contains` block or relies on
host-associated state, the Phase 2 refactoring was not completed —
STOP and explain the blocker to the user. Do NOT substitute
whole-program verification.

### Steps

1. **Read the function** — signature, `USE` dependencies, input /
   output arrays and shapes, module-level variables used.

2. **Snapshot the pristine original BEFORE any edits.** Copy the
   target function into `{parent}/openmp/{function}_reference.f90`,
   renaming the module to `{module}_reference_mod` and the procedure
   to `{function}_reference`. Take the body from untouched source
   (e.g. `git show HEAD:{file}`). Keep the signature, `use`
   dependencies, and logic identical — only the module / procedure
   names change. Add **no** directives.

   ```fortran
   ! src/physics/openmp/compute_kernel_reference.f90
   module solver_reference_mod
     implicit none
   contains
     subroutine compute_kernel_reference(input, output, nx, ny, nz)
       ! exact copy of the original body — no directives
     end subroutine compute_kernel_reference
   end module solver_reference_mod
   ```

3. **Write the test**
   `{parent}/openmp/test/test_{function}/test_{function}.F90`. Use
   **unit-scale** inputs (physics-appropriate, fast) — do NOT copy
   sizes from the active `config.json:problems[]` entry. Call
   `{function}_reference` for
   the baseline; the `#ifdef {INCLUDE_GUARD}` block (empty now, filled
   in 4d) will call the **in-place** `{function}`:

   ```fortran
   program test_compute_kernel
     use solver_reference_mod   ! frozen CPU baseline
     use solver_mod             ! in-place (GPU) version
     implicit none
     ! Test extents: pick them from the function's own real extents
     ! (rank, aspect ratio, halo width). The values below are only a
     ! placeholder for a 3-D structured field.
     integer, parameter :: nx=100, ny=100, nz=50
     real, allocatable :: input(:,:,:), output(:,:,:)

     allocate(input(nx,ny,nz), output(nx,ny,nz))
     call random_number(input)

     ! Baseline from the frozen reference copy:
     call compute_kernel_reference(input, output, nx, ny, nz)
     ! Smoke fingerprint only — NOT a correctness gate (a bare sum lets
     ! opposite-sign errors cancel; see references/verification-guide.md).
     print *, 'Output mean|v|:', sum(abs(output)) / size(output)

   #ifdef {INCLUDE_GUARD}
     ! GPU comparison block — EMPTY for now (filled in Phase 4d).
   #endif

     print *, 'TEST PASSED'
     deallocate(input, output)
   end program
   ```

4. **Hook the reference file + test into the build as a test-only
   target.** Do NOT add `{function}_reference.f90` to the main
   project sources. Reference in-tree sources via `-I`, no copies.
   The test must compile under both paths.

5. **Write the manifest:**
   ```json
   {"test_executable": "/abs/path/to/test_compute_kernel"}
   ```

6. **Verify** (`USE_OPENMP=1 bash .nsight-ai-port/build_correctness.sh`
   then run the test), **commit**, and update `progress.json` (`phases.scaffold`,
   `test_dir`, `test_executable`).

At this point the in-place function is still the unmodified
original, so reference vs in-place trivially match.

---

## Phase 4c: Accelerate (edit in place)

Edit the original function **in place**, one chunk at a time. See
[patterns.md](patterns.md) for directive guidance.

- **Directives go in bare** (`!$omp ...`) — inert on CPU, active
  under `-mp=gpu`. Do NOT `#ifdef`-guard bare directives. Keep every
  `target teams distribute parallel do` paired with its matching
  `!$omp end ...` directive.
- **Any non-directive change** (construct substitution, manual
  restructuring, `separate`-mode changes that alter host semantics)
  MUST be guarded so the CPU build keeps compiling and producing the
  baseline result:
  ```fortran
  #ifdef {INCLUDE_GUARD}
    ! GPU-adapted statements
  #else
    ! original statements (unchanged)
  #endif
  ```
- **Memory-mode data clauses** — read `config.json:memory_mode`.
  Details in [patterns.md](patterns.md):

  | Mode | Directive style |
  |------|-----------------|
  | `unified` / `managed` | Compute only — no `target data`, no `map(...)`; add `!$omp requires unified_shared_memory` |
  | `separate` | Hoisted `!$omp target data` / `target enter data` |
- **Device callees** rely on the in-place `!$omp declare target`
  added in Phase 3 — no separate device module.

Per chunk: edit → lint (non-`separate` modes — grep the original
file for forbidden `map`/`target data` clauses, confirm `requires
unified_shared_memory` where needed) → verify CPU build +
`USE_OPENMP=1` build + test → git commit.

### Derived-type arguments (inline)

The inlined top-level function **keeps its original signature**,
including derived-type dummy arguments. Do NOT rewrite the public
signature to flat arrays.

The "no derived types across the GPU boundary" restriction applies
only **inside** the `!$omp target` compute regions, not at the
function boundary. Before each region, dereference the derived-type
components you need into local plain-array pointers / scalars, and
reference only those flat locals inside the region:

```fortran
subroutine compute_kernel(state, h, is, ie, js, je, nk)
  type(grid_type), intent(in) :: state       ! signature unchanged
  real, intent(inout) :: h(is:ie, js:je, nk)
  real, pointer :: dx(:,:), dy(:,:)
  dx => state%dx                             ! flatten before region
  dy => state%dy
  !$omp target teams distribute parallel do collapse(2)
  do j = js, je
    do i = is, ie
      h(i,j,1) = h(i,j,1) * dx(i,j) / dy(i,j) ! flat locals only
    end do
  end do
  !$omp end target teams distribute parallel do
end subroutine
```

Rules:
- No derived-type variable may be referenced **inside** an `!$omp
  target` region, and none may be passed to an `!$omp declare
  target` device callee — pass the flat locals instead.
- `separate` mode: place the flat locals (not the derived type) in
  the hoisted `!$omp target data` clauses.
- `unified` / `managed` mode: flattening still avoids descriptor
  deep-copy faults even though memory is coherent.

**STOP and ask the user** only when a needed component genuinely
cannot be flattened to a plain array/scalar before the region —
e.g. an array of derived types, deeply nested allocatable
components, or a polymorphic type that must be touched inside the
loop. Do not restructure these silently or change the public
signature.

---

## Phase 4d: Verify Numerical Correctness

The comparison harness ships with the skill as compilable source.
Do not re-derive it. Copy it **once for the whole port** at the
source root — see
[verification-guide.md](../../../references/verification-guide.md#one-copy-per-port-at-the-source-root):

```bash
mkdir -p "$NSIGHT_AI_PORT_SOURCE_DIR/openmp/test"
cp assets/test-harness/compare_field.f90 \
   "$NSIGHT_AI_PORT_SOURCE_DIR/openmp/test/"
```

Module `nsight_port_compare`:

| Call | What it does |
|------|--------------|
| `tolerances_for(sample)` | kind-aware defaults, then `WMAD_TOL` / `ABS_TOL` / `REL_TOL` / `REL_FLOOR` / `MAX_REL_CHECK` from the environment |
| `compare_field(candidate, reference, tol [, weight])` | generic over `real32`/`real64` at ranks 1–3 |
| `report_and_gate(result, tol, label)` | prints `wmad`, `max_rel`, `n_exceed` of `n_total`, and the worst index; returns `.true.` on pass |

Fill the `#ifdef {INCLUDE_GUARD}` block to call the **in-place**
`{function}` and compare against `{function}_reference`. Allocate a
second output array for the in-place call:

```fortran
#ifdef {INCLUDE_GUARD}
  block
    use nsight_port_compare
    real, allocatable :: output_candidate(:,:,:)
    type(compare_tolerances) :: tol
    type(compare_result) :: res

    allocate(output_candidate, mold=output)
    call compute_kernel(input, output_candidate, nx, ny, nz)
    tol = tolerances_for(output(1,1,1))
    res = compare_field(output_candidate, output, tol)
    if (.not. report_and_gate(res, tol, 'compute_kernel')) stop 1
  end block
#endif
```

One `report_and_gate` per output field. `wmad` is the gate.
`n_exceed` and the worst index make an outlier failure reviewable.

An outlier-only failure (`wmad` passes, `n_exceed` small and
scattered through low-magnitude cells) may be algorithmic numerics.
Review it; you may then raise `REL_FLOOR` or set
`MAX_REL_CHECK=off`, recording the reasoning in `progress.json`.
See
[Reviewing an outlier failure](../../../references/verification-guide.md#reviewing-an-outlier-failure)
for what must never be waived.

`{function}_reference` is a verbatim frozen copy of the original.
It must never receive directives, and must never be regenerated
from the now-modified `{function}` — that would compare the GPU
against itself and pass with a difference of exactly 0.0.

**Verify:**
```bash
source .nsight-ai-port/env.sh && \
  USE_OPENMP=1 bash .nsight-ai-port/build_correctness.sh
compute-sanitizer ./test_compute_kernel
```
(If `compute-sanitizer` is missing, run the test directly.)

Must pass before integration. If it fails: indexing, data maps
(`separate`: missing `map(to:)`/`map(from:)`; `unified`/`managed`:
forbidden per-kernel `map` and missing `requires
unified_shared_memory`), reductions, or GPU rounding
(`ABS_TOL`/`REL_TOL`). Git commit.

---

## Phase 4e: Integrate

Because the in-place function serves both build paths, call sites
usually need **no** `#ifdef` dispatch. Integration is mainly:

1. **Exec marker.** Add the one-time marker **inside** the in-place
   function, guarded by the include guard. One marker per function:
   ```fortran
   #ifdef {INCLUDE_GUARD}
     logical, save :: marker_printed = .false.
     ...
     if (.not. marker_printed) then
       print *, 'NSIGHT_AI_PORT_FRAMEWORK_EXEC: compute_kernel'
       marker_printed = .true.
     end if
   #endif
   ```
   Under MPI, add the rank-0 condition described in
   [GPU Execution Marker](../../../references/verification-guide.md#gpu-execution-marker).
2. **Preprocessing + flags.** The edited `.f90` now contains
   `#ifdef`, so ensure `-Mpreprocess` (nvfortran) / `-cpp`
   (gfortran) is on its CPU compile line. On the GPU path add
   `-D{INCLUDE_GUARD}` and the offload / memory flags.
   nvfortran uses `-mp=gpu`, never `-fopenmp`:

   | `memory_mode` | nvfortran flags |
   |---------------|-----------------|
   | `separate` | `-mp=gpu -gpu=ccNN` |
   | `managed` | `-mp=gpu -gpu=ccNN,mem:managed` |
   | `unified` | `-mp=gpu -gpu=ccNN,mem:unified` (+ `!$omp requires unified_shared_memory`) |

   gfortran: `-fopenmp -foffload=nvptx-none`. Record the exact flags
   in `.nsight-ai-port/build_notes.txt`. Never pass offload flags or
   `-D{INCLUDE_GUARD}` to external libraries.
3. **No new main-project files** to wire — the reference file is
   test-only; the in-place file is already in the build.
4. Run the full **dual-path verification** (CPU build+run, GPU
   build+run with the marker check, sanitized standalone test) per
   [verification-guide.md](../../../references/verification-guide.md).

Commit and update `progress.json` (`phases.integrate`, `status`,
`commit_hash`, `translated_functions`, `last_working_commit`).

---

## Resuming After Interruption

Read `.nsight-ai-port/progress.json` (and `porting_plan.json`):

1. Find the first function with `status != "completed"`.
2. Check which phases are done for that function.
3. Resume from the next incomplete phase.
4. Confirm a clean CPU build first:
   ```bash
   source .nsight-ai-port/env.sh && \
     bash .nsight-ai-port/build_correctness.sh
   ```
