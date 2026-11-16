# Refactoring Guide (Phase 2) — Fortran

Language docs: `flavors/fortran/`. Subskill `refactor-task` always
reads this file (Phase 2 edits **source** Fortran). When
`from_lang == to_lang`, also read the framework addendum
`flavors/fortran/{framework}/refactoring.md`.

## When to Refactor

Refactor when a function that should be GPU-accelerated is
tangled with code that cannot run on a GPU:

1. **Orchestration + compute mixed.** A function both manages
   workflow (file I/O, MPI, allocation) and contains the hot
   compute loop. Extract the compute into a new subroutine.

2. **Host parallel regions spanning multiple calls.** An
   `!$OMP PARALLEL` (or similar) region wraps multiple CALL
   statements. Extract the parallel region into a standalone
   subroutine so a GPU region/kernel can replace it. Framework
   addenda say how that region must look (OpenACC vs OpenMP
   `target`).

3. **Host code between parallel regions.** MPI communication,
   Fortran I/O, or runtime library calls appear between two
   parallelizable loop nests. Extract each loop nest into its
   own subroutine.

4. **Widely separated loop dimensions.** An outer loop in a
   caller invokes a callee whose loops cover different grid
   dimensions (e.g. caller loops `DO k`, callee loops
   `DO j; DO i`). Do NOT list both as translation targets.
   Instead, refactor the caller into strided blocks and pass
   block bounds to the callee. See "Strided block refactor"
   below.

5. **Program-scoped internal subroutines.** The translation
   targets are `contains` subroutines inside a `program` block
   and rely on host-associated variables (grid dimensions,
   state arrays, constants). They cannot be called from a
   standalone test program. Extract them into a module with
   explicit arguments.

6. **Un-flattenable derived-type components.** Translated
   routines keep their original signature in both modes and
   flatten components into local pointers before each region, so
   this refactor is **not** the normal path for derived-type
   arguments. Raise it only when a component the hot loop needs
   cannot be flattened at all — an array of derived types, deeply
   nested allocatable components, or a polymorphic type touched
   inside the loop. Then extract a GPU-callable entry point with
   flat array arguments. See "Decoupled GPU Entry Point" below.

7. **Insufficient exposed parallelism / GPU-hostile loop shape.**
   The compute is already in a clean, callable routine, but its
   loop *structure* maps poorly to the GPU: the only parallel
   dimension is small while a large amount of work is serial; a
   loop-carried scalar serializes an otherwise-parallel loop; the
   wide independent dimension is buried as the innermost (strided)
   loop; or independent loops have triangular / data-dependent
   bounds that block `collapse`. This is **not** a directive-
   placement problem — the loop nest needs a behavior-preserving
   structural change first. See "Refactoring to Expose
   Parallelism" below.

## Refactoring Rules

**DO:**
- Make structural changes only (move code, change signatures)
- Keep the same program behavior — outputs must not change
- Use standard Fortran (no GPU directives)
- Update all callers when changing function signatures
- Update `USE` / `INCLUDE` statements for moved code

**DO NOT:**
- Add any GPU directives or framework code
- Change algorithms or numerical behavior
- Introduce new dependencies
- Break the build at any point

## Per-Task Workflow

For each task in `porting_plan.json["refactor_tasks"]`:

1. **Read the instructions** from the plan item.
2. **Read the target function** and understand the code flow.
3. **Make the change.** Typical patterns:
   - Extract subroutine: cut lines, create new subroutine with
     explicit arguments, add `CALL` at the extraction point.
   - Update module: add new subroutine to the module's
     `CONTAINS` section or a new file + `USE` statement.
4. **Run build verification** (build script must pass).
5. **Run test verification** (run script must pass — behavior
   unchanged).
6. **Git commit:**
   ```bash
   git add -A
   git commit -m "refactor: extract {new_func} from {old_func}"
   ```
7. **Update the plan** if new functions were created:
   - **Orchestration + compute extraction** (cases 1–3): add
     the extracted compute function to `functions[]` if it is
     now the clean GPU target. The original orchestrator is
     NOT a translation target.
   - **Strided block refactor** (case 4): do NOT add the inner
     callee to `functions[]`. Add it to `device_utilities`
     instead. Only the caller remains in `functions[]`.
   - **Callability extraction** (case 5): add extracted
     functions to `functions[]` if they are GPU targets.
   - **Decoupling extraction** (case 6, rare): add the flat-array
     GPU entry point to `functions[]`. The derived-type wrapper
     is NOT a translation target — it dispatches at the call
     site during integration.

## Decoupled GPU Entry Point (Derived Types)

**Remedy, not default.** Translated routines normally keep their
original signature and flatten components into local pointers before
each region — see "GPU entry points and derived types" in the
active framework pack's `translation-sibling.md` (or "Derived-type
arguments (inline)" in `translation-inline.md`). Use this section
only for the case that
defeats flattening: a component the hot loop needs that has no
contiguous plain-array image, such as an array of derived types, a
deeply nested allocatable chain, or a polymorphic type referenced
inside the loop.

**Goal:** A host-callable routine with the original signature
(unchanged for CPU builds) plus a new flat-array GPU entry
point that the host wrapper calls under
`#ifdef {include_guard}`. The suffix is `_{framework}` and the guard
is `config.json:include_guard`. See the active pack's
`refactoring.md` addendum for the dispatch snippet.

### Steps

1. Identify every array/scalar the hot loop reads or writes.
2. Create `subroutine {name}_{framework}(...)` with explicit-
   bounds array dummies and integer bounds (`is, ie, js, je,
   nk`, etc.).
3. Move the loop body into the new subroutine (pure Fortran,
   no GPU directives yet — those come in Phase 4c).
4. The original subroutine body becomes a thin CPU path (or
   the call site dispatches directly in integration).
5. Standalone tests call `{name}_{framework}` with the same
   flat arguments, allocating test arrays directly.

### Example

**Before:**
```fortran
subroutine compute_interaction(grid, vgrid, h, u, v, du, dv)
  type(grid_type), intent(in) :: grid, vgrid
  real, intent(in) :: h(grid%is:grid%ie, grid%js:grid%je, :)
  ...
  do k = 1, grid%nk
    do j = grid%js, grid%je
      do i = grid%is, grid%ie
        du(i,j,k) = ...
      end do
    end do
  end do
end subroutine
```

**After refactor (still no GPU directives — Phase 4c adds those):**
```fortran
subroutine compute_interaction_gpu(dx, dy, h, u, v, du, dv, &
                                   is, ie, js, je, nk)
  integer, intent(in) :: is, ie, js, je, nk
  real, intent(in) :: dx(is:ie, js:je), dy(is:ie, js:je)
  real, intent(in) :: h(is:ie, js:je, nk), u(...), v(...)
  real, intent(out) :: du(is:ie, js:je, nk), dv(...)
  ! loop body unchanged except grid% bounds → is, ie, js, je, nk
  do k = 1, nk
    do j = js, je
      do i = is, ie
        du(i,j,k) = ...
      end do
    end do
  end do
end subroutine
```

**Integration dispatch (Phase 4e):** use the active pack's
`include_guard` and `_{framework}` suffix. See the framework
`refactoring.md` addendum for a copy-paste snippet. Passing
`grid%dx` passes by reference — no element copy. Pointer
association (`dx => grid%dx`) is equivalent.

The `_gpu` name above is a placeholder; the real suffix is
`_openacc` or `_openmp`.

## Strided Block Refactor (Separated Loop Dimensions)

Use when a translation target would iterate over fewer array
dimensions than its callers because outer loop variables live
in a caller higher up the call stack. The inner function will
not expose enough parallelism for efficient GPU execution if
ported alone.

**Goal:** One translation target owns all dimensions needed
for GPU parallelism, while preserving CPU behaviour unchanged.

### Steps

1. Add a `stride` parameter to the caller (default `1`).
2. Replace the outer loop with strided blocks and pass block
   bounds down to the callee.
3. The callee loops over `(start, end)` internally for each
   dimension that was previously owned by the caller.
4. Update the plan:
   - Caller stays in (or moves into) `functions[]`.
   - Callee moves to `device_utilities[]`, not `functions[]`.
5. Document in the refactor task: CPU call sites use the
   default `stride=1`; GPU translation may use a larger stride
   to fuse dimensions.

### Example

**Before:**
```fortran
subroutine driver(a3d, b3d, ni, nj, nk, dt)
  integer, intent(in) :: ni, nj, nk
  real, intent(inout) :: a3d(ni, nk, nj), b3d(ni, nk, nj)
  real, intent(in) :: dt
  integer :: i, j

  do j = 1, nj
    do i = 1, ni
      call column_rhs(a3d(i,:,j), b3d(i,:,j), nk, dt)
    end do
  end do
end subroutine
```

**After:**
```fortran
subroutine driver(a3d, b3d, ni, nj, nk, dt, stride)
  integer, intent(in) :: ni, nj, nk
  integer, intent(in), optional :: stride
  real, intent(inout) :: a3d(ni, nk, nj), b3d(ni, nk, nj)
  real, intent(in) :: dt
  integer :: i_outer, j_outer, i_start, i_end, j_start, j_end
  integer :: s

  s = 1
  if (present(stride)) s = max(1, stride)

  do j_outer = 1, nj, s
    j_start = j_outer
    j_end   = min(j_outer + s - 1, nj)
    do i_outer = 1, ni, s
      i_start = i_outer
      i_end   = min(i_outer + s - 1, ni)
      call column_rhs(a3d, b3d, ni, nj, nk, dt, &
                      i_start, i_end, j_start, j_end)
    end do
  end do
end subroutine

subroutine column_rhs(a3d, b3d, ni, nj, nk, dt, &
                      i_start, i_end, j_start, j_end)
  ! loops i_start:i_end, j_start:j_end, then k internally
  ...
end subroutine
```

With `stride=1`, behaviour is identical to the original. With
a larger stride, the GPU translator can parallelise over all
dimensions simultaneously inside `driver`.

Choose which dimensions to stride based on the specific code —
not every outer dimension needs striding. Focus on dimensions
that would provide meaningful additional GPU parallelism.

### CPU vs GPU stride

| Path | Stride | Effect |
|------|--------|--------|
| CPU (default) | 1 | Original loop structure preserved |
| GPU translation | > 1 | Fuses outer/inner dims for GPU mapping |

Existing call sites need no changes when `stride` is optional
with default `1`.

## Refactoring to Expose Parallelism (Loop-Level)

Use when a routine is already GPU-callable but its loop *nest*
does not expose enough independent parallelism for a good
GPU mapping (see the active framework pack's `patterns.md`).
These are **behavior-preserving**
transforms — same rules as any refactor: standard Fortran only, no
GPU directives, outputs unchanged, build + run verified before and
after.

| Transform | When | Effect |
|-----------|------|--------|
| **Loop interchange** | the large independent dimension is the innermost (strided) loop | move it outward so it can carry gangs / vector |
| **Loop fission (split)** | a nest mixes an independent part with a serial/dependent part | split so the independent part is a clean parallel kernel and only the recurrence stays `seq` |
| **Loop fusion (merge)** | many tiny independent loops over the same index, or one kernel launch each | merge into one loop to raise work per launch and cut launch / transfer overhead |
| **Scalar expansion / privatization** | a scalar written then read across iterations serializes the loop | promote to a per-iteration `private` (or a small per-element array) so iterations are independent |
| **Make bounds rectangular** | triangular / data-dependent inner bounds block `collapse` | pad or hoist the variable bound so independent loops can `collapse` into one wide iteration space |
| **Data-layout change** | the contiguous dimension is not the one you want on `vector` | transpose the array (or pack a local working copy) so vector lanes touch contiguous memory (coalesced) |

Prefer to identify these during **planning** so they become
explicit `refactor_tasks`, done and verified before Phase 4c adds
directives. If Phase 4c discovers a nest that cannot be mapped
well as written, **stop and raise a refactor task** rather than
forcing directives onto a GPU-hostile structure.

Scope note: a global layout change (transpose) touches every
access of that array across the code base — scope it carefully and
verify, or keep it local to the kernel via a copied / packed
working array. As with all refactors, change structure only —
never algorithms or numerical results.

## Example: Extract Compute Kernel

**Before:**
```fortran
subroutine timestep_loop(grid, dt, nsteps)
  use grid_types
  type(grid_type), intent(inout) :: grid
  real, intent(in) :: dt
  integer, intent(in) :: nsteps
  integer :: step, i, j, k

  do step = 1, nsteps
    ! MPI halo exchange (cannot run on GPU)
    call exchange_halos(grid)

    ! Compute the RHS (GPU candidate)
    do k = 1, grid%nz
      do j = 1, grid%ny
        do i = 1, grid%nx
          grid%rhs(i,j,k) = ...complex computation...
        end do
      end do
    end do

    ! File output (cannot run on GPU)
    if (mod(step, 100) == 0) call write_checkpoint(grid)
  end do
end subroutine
```

**After:**
```fortran
subroutine compute_rhs(grid)
  use grid_types
  type(grid_type), intent(inout) :: grid
  integer :: i, j, k

  do k = 1, grid%nz
    do j = 1, grid%ny
      do i = 1, grid%nx
        grid%rhs(i,j,k) = ...complex computation...
      end do
    end do
  end do
end subroutine

subroutine timestep_loop(grid, dt, nsteps)
  use grid_types
  type(grid_type), intent(inout) :: grid
  real, intent(in) :: dt
  integer, intent(in) :: nsteps
  integer :: step

  do step = 1, nsteps
    call exchange_halos(grid)
    call compute_rhs(grid)
    if (mod(step, 100) == 0) call write_checkpoint(grid)
  end do
end subroutine
```

Now `compute_rhs` is a clean GPU candidate with no I/O
or MPI.

## Example: Extract from Program Contains

**Before** (single-file program with internal subroutines):
```fortran
program solver
  implicit none
  integer :: nx, nz
  real, allocatable :: state(:,:,:)

  nx = 400; nz = 200
  allocate(state(nx+4, nz+4, 4))
  ! ... initialization, time loop ...
  call update_state(dt)
  ! ...

contains

  subroutine update_state(dt)
    real, intent(in) :: dt
    integer :: i, k
    ! Uses nx, nz, state from host scope
    do k = 1, nz
      do i = 1, nx
        state(i+2,k+2,1) = state(i+2,k+2,1) + dt * ...
      end do
    end do
  end subroutine

end program
```

**After** (module with explicit arguments):
```fortran
module solver_kernels_mod
  implicit none
contains

  subroutine update_state(state, nx, nz, dt)
    integer, intent(in) :: nx, nz
    real, intent(inout) :: state(nx+4, nz+4, 4)
    real, intent(in) :: dt
    integer :: i, k
    do k = 1, nz
      do i = 1, nx
        state(i+2,k+2,1) = state(i+2,k+2,1) + dt * ...
      end do
    end do
  end subroutine

end module

program solver
  use solver_kernels_mod
  implicit none
  integer :: nx, nz
  real, allocatable :: state(:,:,:)

  nx = 400; nz = 200
  allocate(state(nx+4, nz+4, 4))
  ! ...
  call update_state(state, nx, nz, dt)
  ! ...
end program
```

Now `update_state` can be called from a standalone test
program via `use solver_kernels_mod`. The key changes:
- Host-associated variables become explicit arguments
- The subroutine moves from `contains` to a module
- Array bounds are parameterized (not hard-coded from host)
- The main program adds `use solver_kernels_mod` and
  passes arguments explicitly at call sites

For programs with many shared variables, create a separate
state module:
```fortran
module solver_state_mod
  implicit none
  integer :: nx, nz
  real, allocatable :: state(:,:,:)
end module
```

Then kernels can `use solver_state_mod` for dimensions
while still taking arrays as explicit arguments (preferred for
testability), or the test program can initialize the module
state directly.

## Subagent Use

Each refactor task is independent — it has:
- **Input:** the refactoring instructions + current source
- **Success criterion:** build + run scripts pass
- **Output:** git commit with the structural change

This makes each task a natural subagent boundary.
