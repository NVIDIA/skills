# OpenMP Offload Patterns and Pitfalls — Fortran + OpenMP

Flavor pack: `flavors/fortran/openmp/`. Used in translate-function
phase 4c. Read `config.json:memory_mode` before adding `map` clauses.

This is **OpenMP target offload** (GPU), not host multithreading.
Every GPU construct begins with `!$omp target`. The directive model
mirrors OpenACC closely — `target teams distribute parallel do` is the
analog of `acc parallel loop`, and `map(...)` clauses are the analog of
`copyin`/`copyout`/`copy`.

## Core Directives

### Target Teams Distribute Parallel Do

The most common pattern — offload a loop nest to the GPU and
parallelize it across teams and threads:

```fortran
!$omp target teams distribute parallel do collapse(3)
do k = 1, nz
  do j = 1, ny
    do i = 1, nx
      a(i,j,k) = b(i,j,k) * c(i,j,k)
    end do
  end do
end do
!$omp end target teams distribute parallel do
```

- `target` — move execution to the device.
- `teams` — create a league of teams (like CUDA blocks / OpenACC
  gangs).
- `distribute` — spread loop iterations across teams.
- `parallel do` — run a team's iterations on its threads.
- `collapse(N)` merges N tightly nested loops for more parallelism —
  same meaning as OpenACC. Use on independent, tightly nested loops.
- Add `simd` (`... parallel do simd`) to request vectorization of the
  innermost loop; start without it and add only if it helps.

**Always write the matching `!$omp end target teams distribute
parallel do`** in the same edit (Fortran block form requires it).

### Splitting the combined construct

When teams need to share work other than a single loop, split the
construct:

```fortran
!$omp target teams
  !$omp distribute parallel do collapse(2)
  do j = 1, ny
    do i = 1, nx
      a(i,j) = b(i,j) + c(i,j)
    end do
  end do
  !$omp end distribute parallel do
!$omp end target teams
```

Prefer the combined `target teams distribute parallel do` form unless
you specifically need multiple distributed loops inside one `target
teams` region.

### Map Clauses — Memory-Mode Policy

**Read `config.json:memory_mode` before adding any `map` clause.** The
policy differs by mode; the wrong mode's pattern causes silent
performance loss (per-call map/present-table overhead) or unnecessary
host↔device copies. Note the OpenMP default: arrays referenced in a
`target` region are implicitly mapped `tofrom` unless told otherwise —
which means **doing nothing still copies** in separate mode.

#### `unified` or `managed` — no explicit data movement

When `memory_mode` is `unified` or `managed`:

- **Do NOT** add `!$omp target data` / `!$omp end target data`
- **Do NOT** add `!$omp target enter data` / `exit data`
- **Do NOT** add `map(to:)`, `map(from:)`, or `map(tofrom:)` on
  `target` constructs
- **Do NOT** add `!$omp target update`

Declare unified memory once, at module scope, so the implicit `tofrom`
mapping is suppressed:

```fortran
!$omp requires unified_shared_memory
```

Then write **compute directives only**:

```fortran
!$omp target teams distribute parallel do collapse(3)
do k = 1, nz
  do j = 1, ny
    do i = 1, nx
      c(i,j,k) = a(i,j,k) + b(i,j,k)
    end do
  end do
end do
!$omp end target teams distribute parallel do
```

**Why:** Under unified/managed memory, explicit `map` clauses trigger
present-table bookkeeping (and, in managed mode, real `cudaMemcpy` of
static/module arrays) on every kernel entry even when zero bytes need
to move. Per-call mapping around hot kernels (called thousands of times
per run) can cost seconds per module. `!$omp requires
unified_shared_memory` tells the compiler all host data is already
device-accessible, so implicit maps become no-ops.

**Anti-patterns (never in unified/managed mode):**

```fortran
! BAD — per-call data region on a hot kernel
!$omp target data map(to: uh, hin) map(tofrom: h)
!$omp target teams distribute parallel do
  ...
!$omp end target teams distribute parallel do
!$omp end target data

! BAD — explicit map(from:) of a prognostic field every call
!$omp target teams distribute parallel do map(from: h)
```

After Phase 4c, lint translated files when mode is not `separate`
(repeat for each `{parent}/openmp/*_openmp.f90` file):
```bash
grep -nE '!\$omp (target data|target enter data|target exit data|target update)|map *\(' \
  src/physics/openmp/compute_kernel_openmp.f90
```
Or search all sibling trees:
```bash
find . -path '*/openmp/*_openmp.f90' -print0 | \
  xargs -0 grep -nE '!\$omp (target data|target enter data|target exit data)|map *\('
```
Revise if matches appear. Ensure `!$omp requires
unified_shared_memory` is present in unified/managed-mode files.

#### `separate` — explicit data management

When `memory_mode` is `separate`, you must manage residency. **Hoist
data regions to the widest stable scope** — e.g. once around the
timestep loop or at allocation — not around every inner kernel called
inside a hot loop.

```fortran
! GOOD — hoisted outside the per-timestep kernel calls
!$omp target data map(to: grid_metrics) map(tofrom: state)
  do step = 1, nsteps
    call kernel_a(state, ...)   ! state already on device
    call kernel_b(state, ...)
  end do
!$omp end target data

subroutine kernel_a(state, ...)
  !$omp target teams distribute parallel do
  ! state is found in the device data environment automatically
  ...
  !$omp end target teams distribute parallel do
end subroutine
```

| Clause | Direction | When to use (separate mode) |
|--------|-----------|----------------------------|
| `map(to:x)` | CPU → GPU | Read-only on GPU |
| `map(from:x)` | GPU → CPU | Write-only on GPU |
| `map(tofrom:x)` | Both ways | Read-write on GPU (the implicit default) |
| `map(alloc:x)` | Allocate only | Temporary GPU scratch |
| `map(present,alloc:x)` | Must already be present | Nested regions / hoisted data |

Inside a kernel nested in a hoisted `target data` region, arrays are
found automatically in the device data environment — you do not need to
re-`map` them. Use `map(present,alloc: x)` only to assert presence.

**Never** put `map(to:)`/`map(tofrom:)` on a `target teams distribute
parallel do` that is called inside a hot loop — that is a per-iteration
transfer by construction.

### Column Algorithms (vertical sweeps)

For tridiagonal / column-wise algorithms where an outer loop over
`(j,i)` wraps a sequential inner `k` sweep:

```fortran
!$omp target teams distribute parallel do collapse(2) private(work)
do j = js, je
  do i = is, ie
    do k = 1, nk            ! plain inner loop — sequential per thread
      work(k) = ...
      ! tridiagonal solve using work(1:nk)
    end do
  end do
end do
!$omp end target teams distribute parallel do
```

- `collapse(2)` over `(j,i)` for GPU parallelism.
- The inner `k` loop has no directive — it runs sequentially within
  each thread (the analog of OpenACC `!$acc loop seq`).
- `private(work)` for per-column scratch; size the array to the maximum
  column depth (e.g. `work(nk_max)`).

### Reductions

```fortran
!$omp target teams distribute parallel do reduction(+:total) &
!$omp& reduction(max:maxv)
do i = 1, n
  total = total + a(i)
  maxv = max(maxv, a(i))
end do
!$omp end target teams distribute parallel do
```

Supported reduction operators: `+`, `*`, `max`, `min`, `.and.`,
`.or.`, `iand`, `ior`. Initialize reduction variables **before** the
region. Use `&` continuation for long directive lines.

### Private Variables

Scalar temporaries used inside a loop are usually privatized
automatically. If values are wrong or the compiler warns:

```fortran
!$omp target teams distribute parallel do private(temp, scratch)
do i = 1, n
  temp = a(i) * 2.0
  b(i) = temp + 1.0
end do
!$omp end target teams distribute parallel do
```

Arrays used as scratch inside the loop need explicit `private`.

### Declare Target (device routines)

Utility functions called from GPU kernels (index calculators, math
helpers, boundary functions) need `!$omp declare target` so they can
run on-device. This is the OpenMP analog of OpenACC `!$acc routine
seq`:

```fortran
function periodic_index_device(i, N) result(idx)
  !$omp declare target
  integer, intent(in) :: i, N
  integer :: idx
  idx = modulo(i, N)
end function
```

**Placement:** first line inside the routine body, after the
`function`/`subroutine` statement, before `implicit none` and
declarations. A module-level `!$omp declare target(name)` listing
form also exists; the inside-the-routine form is the most robust.

Where the annotated routine lives is Phase 3 — read exactly one
guide from [device-utils.md](device-utils.md) per
`translation_mode`. Do **not** make utility functions separate
porting targets.

## Ifdef Guard Pattern

**Where `#ifdef` guards go — and where they don't:**

Translated files in `{parent}/openmp/` are **pure GPU code** — they are
only compiled when GPU flags are active.

**Do NOT wrap `!$omp target` directives in `#ifdef {INCLUDE_GUARD}`
inside `{parent}/openmp/` files.** That guard is redundant (the file is
never built without `-mp=gpu`) and is a common mistake. Write
directives directly:

```fortran
! In src/physics/openmp/solver_openmp.f90 (GPU-only file):
!$omp target teams distribute parallel do collapse(2)
do j = 1, ny
  do i = 1, nx
    a(i,j) = b(i,j) + c(i,j)
  end do
end do
!$omp end target teams distribute parallel do
```

`#ifdef` guards belong in **two places only**:

1. **Call sites** (integration, Phase 4e) — switching between the
   original and translated function:
   ```fortran
   #ifdef USE_OPENMP
     call compute_kernel_openmp(args)
   #else
     call compute_kernel(args)
   #endif
   ```
2. **Test harnesses** (`.F90` files) — optionally running the GPU
   comparison block.

### Build-system exception (makedep)

Some build systems (e.g. FMS `makedep`) scan `#ifdef` at configure time
**without** `-DUSE_OPENMP`. Module-level `#ifdef` guards can break
dependency generation. If `.nsight-ai-port/build_notes.txt` documents this, keep
`!$omp` directives unconditional in source files compiled with
`-mp=gpu`, and restrict CPU/GPU dispatch to call sites only.

## Common Pitfalls

### 1. Unpaired directives

Every `!$omp target data` needs `!$omp end target data`. The block form
of `!$omp target teams distribute parallel do` needs its matching
`!$omp end target teams distribute parallel do`. **Always write both
the open and close directive in the same edit.**

### 2. Implicit tofrom mapping (the OpenMP default)

Arrays referenced inside a `target` region are implicitly mapped
`tofrom` if not otherwise specified. In `separate` mode that means a
naive kernel silently copies arrays in **and** out on every call. Hoist
a `target data` region, or in unified/managed mode add `!$omp requires
unified_shared_memory` so implicit maps become no-ops.

### 3. External library flags

**Never** pass `-mp=gpu`, `-gpu=...`, or `-DUSE_OPENMP` to external
library compilations. Offload flags must be scoped to the project's own
source code only.

Bad: `$(FC) -mp=gpu -c external_lib/solver.f90`
Good: `$(FC) -c external_lib/solver.f90` (no GPU flags)

### 4. Module compile order

Fortran modules must be compiled before files that `USE` them. When
adding `USE translated_mod` to a file, ensure the Makefile compiles the
translated module first. The translated file in `{parent}/openmp/`
defines a module the test program `USE`s — build order must reflect it.

### 5. Array indexing

Fortran arrays are 1-based by default. OpenMP preserves this. Do not
shift to 0-based indexing. When comparing CPU vs GPU results, use the
same index ranges.

### 6. Compiler-specific flags

nvfortran uses different flags than gfortran:

| gfortran | nvfortran | Purpose |
|----------|-----------|---------|
| `-ffree-form` | `-Mfree` | Free-format source |
| `-fpp` / `-cpp` | `-Mpreprocess` | Preprocessing |
| `-ffixed-line-length-none` | `-Mextend` | Long lines |
| `-fopenmp -foffload=nvptx-none` | `-mp=gpu` | **OpenMP GPU offload** |

**Never use `-fopenmp` with nvfortran for GPU offload — use
`-mp=gpu`.** `-fopenmp` on nvfortran enables host multithreading only,
not device offload. Use `findstring` in Makefiles for compiler
detection, not `ifeq ($(FC),nvfortran)`:

```makefile
ifneq (,$(findstring nvfortran,$(FC)))
  FFLAGS += -mp=gpu -gpu=cc80
endif
```

### 7. `default(none)` requires explicit data-sharing

If a `target` directive uses `default(none)`, every variable must be
listed in a data-sharing or `map` clause or the compile fails. This is
strict but catches accidental host-variable capture. Either list all
variables explicitly or omit `default(none)`.

### 8. Data already on GPU (separate mode)

When `memory_mode` is `separate` and a kernel is nested inside a
hoisted `target data` region, the arrays are found automatically in the
device data environment — do not re-`map` them. Use
`map(present,alloc: x)` only to assert presence. In `unified`/`managed`
mode, omit data regions entirely.

### 9. Derived types on the GPU boundary

**Do NOT reference derived types inside an `!$omp target` region.**
OpenMP must map the whole structure (including allocatable components
and pointers), causing mapping overhead and deep-copy faults.
Dereference the components you need into flat local pointers or scalars
*before* the region, and use only those inside it.

The restriction is on the **region**, not the signature, and that holds
in both `translation_mode`s: the translated routine keeps its original
arguments, derived-type dummies and all. Only the flat locals may be
referenced inside the region.

```fortran
real, pointer :: dx(:,:)
dx => grid%dx                  ! before the region
!$omp target teams distribute parallel do
! ... dx(i,j) only, never grid%dx ...
```

For the sibling-mode entry point and dispatcher, see "GPU entry points
and derived types" in
[translation-sibling.md](translation-sibling.md); for inline mode, see
"Derived-type arguments (inline)" in
[translation-inline.md](translation-inline.md). A Phase 2 decoupling
refactor is the remedy only when a component cannot be flattened at
all — see refactoring.md.

### 10. Allocatable arrays in modules (separate mode)

When `memory_mode` is `separate`, module-level allocatable arrays need
explicit mapping at hoisted scope:

```fortran
!$omp target enter data map(to: module_array)
! ... use module_array in kernels (found in device data env) ...
!$omp target exit data map(delete: module_array)
```

In `unified`/`managed` mode, do not add enter/exit data — residency is
handled by the memory system (with `!$omp requires
unified_shared_memory`).

### 11. Floating-point precision

GPU floating-point operations may execute in a different order than the
CPU, producing small numerical differences. This is expected. Use
tolerances when comparing:
- Single precision: abs_tol=1e-5, rel_tol=1e-4
- Double precision: abs_tol=1e-12, rel_tol=1e-10

### 12. Nested offload / declare target

A `target` region **cannot launch another `target` region**. If a
kernel calls a routine that itself contains compute, mark the callee
`!$omp declare target` (it runs as a device routine), do not give it
its own `target` region. Never translate both a caller and a callee
with independent `target` regions if the caller invokes the callee.

### 13. Teams / threads / SIMD tuning

For fine-tuning (usually not needed initially):

```fortran
!$omp target teams distribute parallel do simd &
!$omp& num_teams(256) thread_limit(128)
```

- `teams` ↔ CUDA blocks / OpenACC gangs
- `parallel do` threads ↔ OpenACC workers/vectors
- `simd` ↔ vector lanes

Start with plain `!$omp target teams distribute parallel do` and let
the compiler choose `num_teams` / `thread_limit`.
