# OpenACC Patterns and Pitfalls — Fortran + OpenACC

Flavor pack: `flavors/fortran/openacc/`. Used in translate-function
phase 4c. Read `config.json:memory_mode` before adding data clauses.

## Core Directives

### Parallel Loop

The most common pattern. Tells the compiler to parallelize a
loop on the GPU:

```fortran
!$acc parallel loop collapse(3)
do k = 1, nz
  do j = 1, ny
    do i = 1, nx
      a(i,j,k) = b(i,j,k) * c(i,j,k)
    end do
  end do
end do
```

- `collapse(N)` merges N nested loops into one parallel region
  for better GPU utilization. Use it on tightly nested loops
  with independent iterations.
- Omit `collapse` if inner loops have dependencies or varying
  trip counts.

### Mapping loop nests to gang / worker / vector

For a non-trivial loop nest, do **not** emit a bare `!$acc
parallel loop` on the outermost loop and let the compiler guess
the rest. Choose the mapping from the **parallelism each loop
exposes** so the GPU stays busy. The levels map roughly to
hardware:

| Level | Hardware | Wants |
|-------|----------|-------|
| `gang` | thread blocks across the grid | *many* (≥ SM count × several) to fill all SMs |
| `worker` | warps within a block | optional middle tier; multiplies block width |
| `vector` | SIMT lanes (a warp) | full warp (32); block total `worker`×`vector` ≈ 128–256 |

This is a **judgement per nest**, not a fixed positional rule.
First mark which loops are **independent** (parallelizable) and
which carry a **dependence** (recurrence / ordered sweep);
sequential loops stay `seq` and reductions use `reduction(...)`.
The real decision is **how to distribute gang / worker / vector
across the independent loops** so the device gets both enough
gangs (thousands of blocks) and a full vector width. (OpenACC
rule: outermost parallel loop = `gang`, innermost = `vector`,
`worker` may sit between, `seq` at any level.)

| Independent parallelism available | Mapping |
|-----------------------------------|---------|
| One wide loop (or siblings you can flatten) | **combined `gang vector`** on it — partitions across blocks *and* vectorizes. `collapse` short siblings first if one alone is too narrow. |
| Two+ nested loops; inner is long and **contiguous** | **split: `gang` on the outer, `vector` on the inner** — inner is stride-1 ⇒ coalesced lanes. |
| Two+ nested loops; inner is **short or runtime-variable** | a narrow vector underfills the block — **`gang worker` on the outer + small `vector` on the inner** so several short vectors fill each block; or `collapse` to lengthen the vector. |
| Outer parallel; inner carries a **dependence** | **combined `gang vector` on the outer** (one thread per outer element); the inner runs sequentially per thread. |

`collapse` is the lever for "not enough trips": collapsing gang
loops makes more blocks; collapsing vector loops makes a longer
vector. Let the compiler pick the gang/vector *counts* unless a
profile shows a small inner loop wasting lanes — then set a small
`vector_length` and add `worker`s (the classic short-inner fix).

**Anti-pattern: small parallel degree + large serial body.** If
the only parallel loop is short while a big chunk runs `seq`, most
of the GPU is idle. Expose more parallelism (collapse, reorder
loops, push the wide dimension to the parallel level) instead. If
that requires a structural change to the loop nest — loop
interchange / fission, scalar privatization, making bounds
rectangular for `collapse`, or changing data layout — it is a
**refactoring task**, not a directive choice. See "Refactoring to
Expose Parallelism" in refactoring.md, and prefer to do it in
Phase 2 *before* adding directives rather than forcing `!$acc`
onto a GPU-hostile structure.

#### Skewed extents (one huge dimension, one small)

A very common HPC shape: a large independent outer space (10^4–10^7
elements / points / cells / particles) and a small inner extent
(tens of levels / components / colors). Do **not** reflexively put
the small inner loop on `vector` — a 20-wide vector wastes most of
a 32-lane warp, and if the large dimension then runs effectively
serial the kernel is starved. Weigh two competing goals:

- **Parallel degree / occupancy:** the *large* dimension should
  carry the parallelism — e.g. `!$acc loop gang vector` over it
  (one lane per element, small inner loop sequential per thread),
  or `collapse` it with another independent loop.
- **Memory coalescing:** adjacent vector lanes should touch
  adjacent memory. If the small dimension is the *contiguous*
  (fastest-varying) one, putting it on `vector` coalesces but is
  narrow; putting the large strided dimension on `vector` is wide
  but strided. Choose the balance the data layout favors — and
  consider that changing the layout may be the real win.

#### Example — map each loop to its own structure

Different loops in the same routine usually need **different**
mappings, each in its **own** `!$acc parallel` region. Do not wrap
them all in one outer parallel loop with everything `seq` inside —
that serializes the loops that are actually parallel.

```fortran
! e ~ 10^5 (independent outer), c ~ 30 (small inner).

! Loop A — independent in both e and c: collapse to one large
! iteration space (max parallelism; contiguous c coalesces).
!$acc parallel loop collapse(2) default(present)
do e = 1, n
   do c = 1, m
      a(c,e) = b(c,e) * coef(c)
   end do
end do

! Loop B — c carries a recurrence (x(c) needs x(c-1)), so c cannot
! be parallel. Drive parallelism from e instead: one thread per
! column (gang vector over e), c sequential.
!$acc parallel loop gang vector default(present)
do e = 1, n
   !$acc loop seq
   do c = 2, m
      x(c,e) = (x(c,e) - aa(c,e)*x(c-1,e)) * alpha(c,e)
   end do
end do
```

Same `(c,e)` shape, **opposite** mapping — decided only by whether
`c` is independent (Loop A) or a recurrence (Loop B). If a
parallel loop needs per-element scratch, mark it `private` (sized
to the inner extent). If the codebase already has hand-tuned GPU
regions, match their granularity.

### Kernels (alternative to parallel loop)

```fortran
!$acc kernels
do k = 1, nz
  do j = 1, ny
    do i = 1, nx
      a(i,j,k) = b(i,j,k) + c(i,j,k)
    end do
  end do
end do
!$acc end kernels
```

`kernels` lets the compiler decide parallelization. Use when
you're unsure about data dependencies. Prefer `parallel loop`
when you know the loop is safe to parallelize.

### Data Clauses — Memory-Mode Policy

**Read `config.json:memory_mode` before adding any data
clause.** The policy differs by mode; applying the wrong
mode's pattern causes silent performance loss (per-call present-
table overhead) or unnecessary host↔device copies.

#### `unified` or `managed` — no explicit data movement

When `memory_mode` is `unified` or `managed`:

- **Do NOT** add `!$acc data` / `!$acc end data` regions
- **Do NOT** add `!$acc enter data` / `!$acc exit data`
- **Do NOT** add `copyin`, `copyout`, or `copy` on compute
  constructs (`!$acc parallel loop`, `!$acc kernels`)
- **Do NOT** add `!$acc update`

Use **compute directives only**:

```fortran
!$acc parallel loop collapse(3) present(a, b, c)
do k = 1, nz
  do j = 1, ny
    do i = 1, nx
      c(i,j,k) = a(i,j,k) + b(i,j,k)
    end do
  end do
end do
```

`present(...)` is optional under unified/managed — omit it
unless you are nesting inside an outer `separate`-mode data
region. Never use `present` as an excuse to add `copyin`.

**Why:** Under unified memory, explicit data clauses trigger
present-table bookkeeping on every kernel entry even when zero
bytes move. Per-call `!$acc data` around hot kernels (called
thousands of times per run) can cost seconds per module.
Under managed mode, `copyin` of static/module arrays still
issues real `cudaMemcpy` on every call.

**Anti-patterns (never in unified/managed mode):**

```fortran
! BAD — per-call data region on a hot kernel
subroutine hot_kernel(...)
  !$acc data copyin(uh, hin, IareaT) copy(h)
  !$acc parallel loop collapse(3)
    ...
  !$acc end data
end subroutine

! BAD — copyout of a prognostic field every call
!$acc parallel loop copy(h)
```

After Phase 4c, lint translated files when mode is not
`separate` (repeat for each `{parent}/openacc/*_openacc.f90`):
```bash
grep -nE '!$acc (data|enter data|exit data)|copy(in|out)?\(' \
  src/physics/openacc/compute_kernel_openacc.f90
```
Or search all sibling trees:
```bash
find . -path '*/openacc/*_openacc.f90' -print0 | \
  xargs -0 grep -nE '!$acc (data|enter data|exit data)|copy(in|out)?\('
```

#### `separate` — explicit data management

When `memory_mode` is `separate`, you must manage residency.
**Hoist data regions to the widest stable scope** — e.g. once
around the timestep loop or at allocation — not around every
inner kernel called inside a hot loop.

```fortran
! GOOD — hoisted outside the per-timestep kernel calls
!$acc data copyin(grid_metrics) copy(state)
  do step = 1, nsteps
    call kernel_a(state, ...)   ! uses present() inside
    call kernel_b(state, ...)
  end do
!$acc end data

subroutine kernel_a(state, ...)
  !$acc parallel loop present(state)
  ...
end subroutine
```

| Clause | Direction | When to use (separate mode) |
|--------|-----------|----------------------------|
| `copyin(x)` | CPU → GPU | Read-only on GPU |
| `copyout(x)` | GPU → CPU | Write-only on GPU |
| `copy(x)` | Both ways | Read-write on GPU |
| `create(x)` | Allocate only | Temporary GPU scratch |
| `present(x)` | Already on GPU | Nested regions / hoisted data |

**Never** put `copyin`/`copy` on a `!$acc parallel loop` that
is called inside a hot loop — that is a per-iteration transfer
by construction.

**Partial-array writes.** If a kernel writes only a sub-range of
an array (e.g. an interior/owned sub-block `i0:i1`, or skipping
boundary entries), do **not** `copyout` the whole array — the
unwritten device region (boundary/halo entries, uninitialised
slots) is copied back and clobbers valid host values on exit.
Bound the section actually written, or use `copy` when host values
outside the written range must survive:

```fortran
! BAD: whole-array copyout when only an interior block is written
!$acc data copyout(arr)

! GOOD: copy out only the written section ...
!$acc data copyout(arr(:, i0:i1))
! ... or preserve untouched host values with copy
!$acc data copy(arr)
```

### Column Algorithms (vertical sweeps)

For tridiagonal / column-wise algorithms where an outer loop
over `(j,i)` wraps a sequential inner `k` sweep:

```fortran
!$acc parallel loop collapse(2) private(work)
do j = js, je
  do i = is, ie
    !$acc loop seq
    do k = 1, nk
      work(k) = ...
      ! tridiagonal solve using work(1:nk)
    end do
  end do
end do
```

- `collapse(2)` over `(j,i)` for GPU parallelism
- `!$acc loop seq` on the inner `k` loop (inherently serial)
- `private(work)` for per-column scratch; size the array to
  the maximum column depth (e.g. `work(nk_max)`)

### Reductions

```fortran
!$acc parallel loop reduction(+:total) reduction(max:maxval)
do i = 1, n
  total = total + a(i)
  maxval = max(maxval, a(i))
end do
```

Supported reduction operators: `+`, `*`, `max`, `min`, `.and.`,
`.or.`, `iand`, `ior`.

Initialize reduction variables **before** the loop.

### Private Variables

Scalar temporaries used inside a loop are automatically private
in most cases. If the compiler warns or values are wrong:

```fortran
!$acc parallel loop private(temp, scratch)
do i = 1, n
  temp = a(i) * 2.0
  b(i) = temp + 1.0
end do
```

Arrays used as scratch inside a loop need explicit `private`:
```fortran
!$acc parallel loop private(work_array)
```

### Routine Directive

Utility functions called from GPU kernels (index calculators,
math helpers, boundary functions) need `!$acc routine seq` so
they can run on-device:

```fortran
function periodic_index_device(i, N) result(idx)
  !$acc routine seq
  integer, intent(in) :: i, N
  integer :: idx
  idx = modulo(i, N)
end function
```

`seq` = one GPU thread, sequentially. Other options: `gang`,
`worker`, `vector`. `seq` is the usual choice for helpers.

**Placement:** first line inside the routine body, after the
`function`/`subroutine` statement, before `implicit none` and
declarations. nvfortran rejects it before the function
statement.

Where the annotated routine lives is Phase 3 — read exactly
one guide from [device-utils.md](device-utils.md) per
`translation_mode`. Do **not** make utility functions separate
porting targets.

## Ifdef Guard Pattern

**Where `#ifdef` guards go — and where they don't:**

Translated files in `{parent}/openacc/` are **pure GPU code** —
they are only compiled when GPU flags are active.

**Do NOT wrap `!$acc` directives in `#ifdef {INCLUDE_GUARD}`
inside `{parent}/openacc/` files.** That guard is redundant
(the file is never built without `-acc`) and is a common
mistake. Write directives directly:

```fortran
! In src/physics/openacc/solver_openacc.f90 (GPU-only file):
!$acc parallel loop collapse(2)
do j = 1, ny
  do i = 1, nx
    a(i,j) = b(i,j) + c(i,j)
  end do
end do
```

**Bad (redundant guard inside `{parent}/openacc/`):**
```fortran
#ifdef USE_OPENACC
  !$acc parallel loop collapse(2)
#endif
```

`#ifdef` guards belong in **two places only**:

1. **Call sites** (integration, Phase 4e) — switching between
   the original and translated function:
   ```fortran
   #ifdef USE_OPENACC
     call compute_kernel_openacc(args)
   #else
     call compute_kernel(args)
   #endif
   ```

2. **Test harnesses** (`.F90` files) — optionally running the
   GPU comparison block:
   ```fortran
   #ifdef USE_OPENACC
     ! call translated version + compare
   #endif
   ```

### Build-system exception (makedep)

Some build systems (e.g. FMS `makedep`) scan `#ifdef` at
configure time **without** `-DUSE_OPENACC`. Module-level
`#ifdef` guards can break dependency generation. If
`.nsight-ai-port/build_notes.txt` documents this, keep `!$acc` directives
unconditional in source files compiled with `-acc`, and restrict
CPU/GPU dispatch to call sites only. Do not add `#ifdef` around
directives inside `{parent}/openacc/` GPU-only files — that
problem is separate from the makedep issue.

## Common Pitfalls

### 1. Unpaired directives

Every `!$acc data` needs an `!$acc end data`. Every `!$acc
parallel` needs an `!$acc end parallel` (or use the single-line
`!$acc parallel loop` form which is self-closing).

**Always write both the open and close directive in the same
edit.** Don't add `!$acc data` and plan to add `!$acc end data`
later.

### 2. External library flags

**Never** pass `-acc`, `-gpu`, or `-DUSE_OPENACC` to external
library compilations. OpenACC flags must be scoped to the
project's own source code only.

Bad: `$(FC) -acc -gpu=cc80 -c external_lib/solver.f90`
Good: `$(FC) -c external_lib/solver.f90` (no GPU flags)

### 3. Module compile order

Fortran modules must be compiled before the files that `USE`
them. When adding `USE translated_mod` to a file, ensure the
Makefile compiles the translated module first.

Watch for: the translated file in `{parent}/openacc/` defines a
module that the test program `USE`s — the build order must
reflect this dependency.

### 4. Array indexing

Fortran arrays are 1-based by default. OpenACC preserves this.
Do not shift to 0-based indexing. If comparing CPU vs GPU
results, make sure both use the same index ranges.

### 5. Compiler-specific flags

nvfortran uses different flags than gfortran:

| gfortran | nvfortran | Purpose |
|----------|-----------|---------|
| `-ffree-form` | `-Mfree` | Free-format source |
| `-fpp` | `-Mpreprocess` | Preprocessing |
| `-ffixed-line-length-none` | `-Mextend` | Long lines |
| `-O2` | `-O2` | Optimization (same) |

Use `findstring` in Makefiles for compiler detection, not
`ifeq ($(FC),nvfortran)`:

```makefile
ifneq (,$(findstring nvfortran,$(FC)))
  FFLAGS += -acc -gpu=cc80
endif
```

### 6. Data already on GPU (separate mode only)

When `memory_mode` is `separate` and a data region is nested
inside another, use `present`:

```fortran
!$acc data copy(a, b)
  call kernel1(a, b)   ! a, b are on GPU
  call kernel2(a, b)   ! still on GPU
!$acc end data

subroutine kernel1(a, b)
  !$acc parallel loop present(a, b)
  do i = 1, n
    ...
  end do
end subroutine
```

In `unified` or `managed` mode, omit outer data regions
entirely — see "Data Clauses — Memory-Mode Policy" above.

### 7. Derived types on the GPU boundary

**Do NOT reference derived types inside an `!$acc` compute region.**
OpenACC maps the whole structure (including allocatable components and
pointers), causing present-table overhead and deep-copy faults.
Dereference the components you need into flat local pointers or scalars
*before* the region, and use only those inside it.

The restriction is on the **region**, not the signature, and that holds
in both `translation_mode`s: the translated routine keeps its original
arguments, derived-type dummies and all. Only the flat locals may be
referenced inside the region.

```fortran
real, pointer :: lat(:,:)
lat => grid%lat                ! before the region
!$acc parallel loop
! ... lat(i,j) only, never grid%lat ...
```

For the sibling-mode entry point and dispatcher, see "GPU entry points
and derived types" in
[translation-sibling.md](translation-sibling.md); for inline mode, see
"Derived-type arguments (inline)" in
[translation-inline.md](translation-inline.md). A Phase 2 decoupling
refactor is the remedy only when a component cannot be flattened at
all — see refactoring.md.

Plan a decoupling refactor (Phase 2) when translation targets
take derived-type arguments.

**Inline mode (`translation_mode: inline`).** The rule above is
about a *separate* GPU entry point (sibling mode). In inline mode
the targeted function keeps its original derived-type signature —
do not decouple it. The same restriction still holds **inside** the
compute region: dereference the components into flat local
array-pointers/scalars before the `!$acc` region and reference only
those locals inside it. See "Derived-type arguments (inline)" in
`translation-inline.md`.

### 8. Allocatable arrays in modules (separate mode)

When `memory_mode` is `separate`, module-level allocatable
arrays need explicit data management at hoisted scope:

```fortran
!$acc enter data copyin(module_array)
! ... use module_array in kernels with present() ...
!$acc exit data delete(module_array)
```

In `unified` or `managed` mode, do not add enter/exit data
for these — residency is handled by the memory system.

### 9. Floating-point precision

GPU floating-point operations may execute in a different order
than CPU, producing small numerical differences. This is
expected. Use tolerances when comparing:
- Single precision: abs_tol=1e-5, rel_tol=1e-4
- Double precision: abs_tol=1e-12, rel_tol=1e-10

### 10. Nested parallelism

OpenACC kernels **cannot launch other kernels**. If function A
calls function B and both have `!$acc parallel` regions, you'll
get a runtime error. Solutions:
- Translate B first and mark it with `!$acc routine`
- Or restructure so A's parallel region doesn't call B
- Never translate both A and B with independent `!$acc parallel`
  regions if A calls B

### 11. Gang/Worker/Vector

- `gang`: thread blocks across the grid (need many to fill SMs)
- `worker`: warps within a block (optional middle tier)
- `vector`: SIMT lanes within a warp (block total ≈ 128–256)

For **simple dense, fully-independent** nests, a bare `!$acc
parallel loop [collapse(N)]` is fine — the compiler picks
reasonable defaults.

For **non-trivial nests** (skewed extents, gathers, recurrences,
irregular/indirect access), do not rely on defaults: choose the
mapping from each loop's parallelism and the data layout — see
"Mapping loop nests to gang / worker / vector" above. The most
common miss is mapping a small inner dimension to `vector` (or
leaving it implicit) so blocks are under-filled while the large
dimension goes to waste.
