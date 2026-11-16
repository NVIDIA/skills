# Translation Guide (Phase 4 — Per-Function) — Sibling Mode — Fortran + OpenACC

Flavor pack: `flavors/fortran/openacc/`. The `translate-function`
subskill reads this file for **Phase 4** (per-function translation) in
**sibling** mode. Phase 3 device utilities live in
[device-utils-sibling.md](device-utils-sibling.md).

Translated code lives under `{parent}/openacc/`. GPU builds use
`USE_OPENACC=1`.

This is the **sibling** guide
(`config.json:translation_mode == "sibling"`, the default). For
inline mode the subskill reads `translation-inline.md` instead — do
not mix the two.

## Directory Layout

Each original source file gets a **sibling** `openacc/` directory
in the **same parent folder** as that source file — not a single
project-root `openacc/`. Nested codebases therefore have many
`openacc/` directories.

**Path rule:** for original `{parent}/{source}.f90` and target
function `{function}`:

| Artifact | Path |
|----------|------|
| Translated kernel | `{parent}/openacc/{function}_openacc.f90` |
| Standalone test | `{parent}/openacc/test/test_{function}/` |
| Device utilities | `{parent}/openacc/device_utils.f90` (where the host utility lives) |
| Device util test | `{parent}/openacc/test/test_device_utils/` |

Example: `src/physics/solver.f90` → `src/physics/openacc/…`

```text
project_root/
├── src/
│   ├── physics/
│   │   ├── solver.f90       ← original
│   │   └── openacc/         ← sibling to solver.f90
│   │       ├── compute_kernel_openacc.f90
│   │       └── test/
│   │           └── test_compute_kernel/
│   │               └── test_compute_kernel.F90
│   └── dynamics/
│       ├── kernel_a.f90
│       └── openacc/
│           ├── kernel_a_openacc.f90
│           └── test/test_kernel_a/
├── utils/
│   ├── kernel_utils.f90
│   └── openacc/
│       ├── device_utils.f90 ← Phase 3 (host utils here)
│       └── test/test_device_utils/
└── Makefile
```

Key points:
- **`openacc/` is per source directory**, not one global tree
- **One translated file per function** in that directory's
  `openacc/` — do NOT combine multiple functions into one file
- A kernel in `src/dynamics/openacc/` may
  `use utils_device_utils_mod` from
  `utils/openacc/device_utils.f90`; wire both dirs in the build
- Original source files are never modified (except call sites
  during integration)

---

## Phase 4: Per-Function Translation

## Phase 4a: Build System Exploration

`.nsight-ai-port/build_notes.txt` should exist. If not, generate it
from [build-integration.md](build-integration.md).

Read it before scaffolding so this function is compiled and linked
the same way as the rest of the GPU path (guard, OpenACC flags,
how `{parent}/openacc/` is picked up).

## Phase 4b: Scaffold Test Harness

**Subagent opportunity.** This has a clear input (function
signature + dependencies from the plan) and a clear success
criterion (the test builds and runs using the original code).

### Prerequisite: function must be externally callable

Before writing the test harness, confirm the target function
can be called from a standalone program (via `USE module_name`).
If the function is still inside a `program ... contains` block
or relies on host-associated state, the Phase 2 refactoring
was not completed — go back and do it first.

**If you discover at this point that the function cannot be
tested in isolation, STOP and explain the blocker to the user.
Do NOT substitute whole-program verification or skip the
per-function test.** Wait for the user to accept a deviation
before proceeding differently.

### Steps

1. **Read the function.** Understand:
   - Subroutine/function signature and arguments
   - `USE` module dependencies
   - Input/output arrays and their shapes
   - Any module-level variables used

2. **Create test directory** (sibling `openacc/` next to the
   original source file):
   ```bash
   # Example: target in src/physics/solver.f90
   mkdir -p src/physics/openacc/test/test_{function}/
   ```

3. **Write test program**
   `src/physics/openacc/test/test_{function}/test_{function}.F90`:

   ```fortran
   program test_compute_kernel
     use solver_mod  ! module containing the function
     implicit none

     ! Declare test arrays
     ! Test extents: pick them from the function's own real extents
     ! (rank, aspect ratio, halo width). The values below are only a
     ! placeholder for a 3-D structured field.
     integer, parameter :: nx=100, ny=100, nz=50
     ! One input, shared by both paths. Two outputs, named for their role
     ! in the comparison — the harness takes (candidate, reference).
     real, allocatable :: input(:,:,:), output_reference(:,:,:)

     ! Allocate and initialize with realistic values
     allocate(input(nx,ny,nz), output_reference(nx,ny,nz))
     call random_number(input)

     ! Call the ORIGINAL function — it produces the reference
     call compute_kernel(input, output_reference, nx, ny, nz)

     ! Smoke fingerprint only — NOT a correctness gate. Never gate on a
     ! bare sum: opposite-sign errors cancel and a near-zero mean makes
     ! the sum ill-conditioned (see references/verification-guide.md).
     print *, 'Output mean|v|:', sum(abs(output_reference)) / size(output_reference)
     print *, 'Output peak|v|:', maxval(abs(output_reference))

   #ifdef USE_OPENACC
     ! GPU comparison block — EMPTY for now.
     ! Phase 4d will fill this in.
   #endif

     print *, 'TEST PASSED'
     deallocate(input, output_reference)
   end program
   ```

   Rules for the test:
   - Use **realistic** input sizes and values for **unit** testing
     — physics-appropriate, fast to run. Do **not** copy grid sizes,
     timestep counts, or case parameters from the active
     `config.json:problems[]` entry
     (integration/profiling workload). Standalone tests must stay
     small and deterministic.
   - Call the **original** function (not the translated one yet)
   - The `#ifdef` block must be present but empty
   - Must exit 0 and print a recognizable success line
   - Must not copy source files — reference them in-tree

4. **Hook into the build system.** Add a test target:

   For **Makefile** (adjust paths to match the target's
   `{parent}/openacc/`):
   ```makefile
   # The comparison module is built once for the whole port and linked by
   # every test — one object, one .mod in the shared MODDIR.
   HARNESS_OBJ := $(OBJDIR)/openacc/test/compare_field.o

   # Example: target in src/physics/
   test_compute_kernel: $(OBJS) $(HARNESS_OBJ) \
       src/physics/openacc/test/test_compute_kernel/test_compute_kernel.o
   	$(FC) $(FFLAGS) -o $@ $^
   ```

   Reference source files via `-I` include paths, not copies.
   The test target must compile under the GPU build path (with
   `-acc` flags), even though the `#ifdef` block is empty.

5. **Create placeholder translated file** in the same
   `{parent}/openacc/` as the original source — one GPU-only
   file per function (do **not** copy the entire original file):
   ```fortran
   ! src/physics/openacc/compute_kernel_openacc.f90
   module compute_kernel_openacc_mod
     implicit none
   contains
     subroutine compute_kernel_openacc(input, output, nx, ny, nz)
       ! TODO: paste loop body from original in Phase 4c
     end subroutine compute_kernel_openacc
   end module compute_kernel_openacc_mod
   ```
   Name convention: `{parent}/openacc/{function}_openacc.f90`
   (or `{original_basename}_openacc.f90` when function name
   matches the source file name).

6. **Write manifest:**
   ```json
   {
     "test_executable": "/abs/path/to/test_compute_kernel"
   }
   ```

7. **Verify:**
   ```bash
   source .nsight-ai-port/env.sh && \
     USE_OPENACC=1 bash .nsight-ai-port/build_correctness.sh
   ./test_{function}
   ```
   Both must pass.

8. **Git commit:**
   ```bash
   git add -A && git commit -m \
     "scaffold: test harness for compute_kernel"
   ```

9. **Update progress.json:**
   Set this function's `phases.scaffold: true` and record
   `test_dir` and `test_executable`.

## Phase 4c: Add OpenACC Directives

See [patterns.md](patterns.md) for directive
guidance and common pitfalls.

Work on this function's dedicated file in `{parent}/openacc/`
(e.g. `src/physics/openacc/solver_openacc.f90`). Each
function has its own translated file — do **not** combine
functions or modify the original source file.

### Execution marker (mandatory)

Open `{function}_openacc` with a one-time marker, so Phase 4d can prove
the GPU path ran rather than a host fallback. It belongs in the
translated body, never at the call site — see
[GPU Execution Marker](../../../references/verification-guide.md#gpu-execution-marker)
for the rank-0 form under MPI.

```fortran
logical, save :: marker_printed = .false.
...
if (.not. marker_printed) then
   print *, 'NSIGHT_AI_PORT_FRAMEWORK_EXEC: {function}'
   marker_printed = .true.
end if
```

### Memory mode (mandatory)

Read `config.json:memory_mode` before adding directives.
Follow the data-clause policy in patterns.md:

| Mode | Directive style |
|------|-----------------|
| `unified` / `managed` | Compute only — no `!$acc data`, no `copyin`/`copyout`/`copy` |
| `separate` | Hoisted `!$acc data` / `enter data`; `present()` inside kernels |

### Loop mapping

One rule here is mandatory: for any non-trivial loop nest, do **not**
emit a bare `!$acc parallel loop` on the outer loop and leave the rest
to the compiler. *Which* mapping you choose instead is a judgement call —
the rest of this section is how to make it, not a checklist to satisfy.

Follow "Mapping loop nests to gang / worker / vector" in
patterns.md. Choose the mapping per nest from the
**parallelism each loop exposes** and the data layout:

- drive the bulk of parallelism from the largest independent
  dimension(s); ensure both enough gangs (blocks) and a full
  vector width;
- `collapse` independent loops when no single one is wide enough;
- `seq` only for true dependences (recurrences / ordered sweeps)
  or tiny fixed-count inner loops; `reduction(...)` for reductions;
- avoid the small-parallel-degree + large-serial anti-pattern.

There is **no fixed positional rule** (e.g. "outer = gang, inner =
vector"). For skewed extents (one huge dimension, one small) the
small inner loop is often a poor `vector` target — weigh occupancy
against memory coalescing and decide per nest. If the codebase
already has hand-tuned GPU regions, match their granularity. Note
the reasoning briefly in the port notes.

If a nest cannot be mapped well **as written** — the only parallel
dimension is too small, a loop-carried scalar serializes it, the
wide dimension is buried innermost, or bounds block `collapse` —
that is a structural problem, not a directive choice. Stop and
raise a Phase 2 refactor task (see "Refactoring to Expose
Parallelism" in refactoring.md) instead of forcing `!$acc` onto a
GPU-hostile loop. Ideally such restructuring is caught in planning
and done before this phase.

Even when a `separate`-mode `!$acc data` region is per-routine
(because residency cannot yet be hoisted), the compute inside it
must still use a deliberate mapping — a temporary data region is
not an excuse for low-quality loop mapping.

### Honest execution scope

The default is to offload every loop nest in the target routine. Leaving
one on the host is a legitimate choice — a nest with too little work to
pay for a launch, one that is genuinely serial, one whose data would
force a transfer that costs more than the kernel saves.

Judge that last cost across the routine's data flow, not the nest in
isolation. A nest that is no faster on the device — or even somewhat
slower — is still worth offloading when it keeps its arrays resident and
spares the nests on either side a device-to-host-to-device round trip.
The expensive case is a host nest in the middle of an offloaded
sequence, which drags the data back and forth twice. Under `managed` or
`unified` memory those transfers are implicit page migration rather than
explicit maps, but they are not free. What is not legitimate is leaving
a nest on the host silently.

So before marking a function done, write a short loop-coverage note in
`progress.json` for that function:

- loop nests offloaded (with line ranges)
- loop nests left on the host, and why

The execution marker is **one per function** and says only that the
translated body ran — it does not claim every nest inside it was
offloaded (see
[GPU Execution Marker](../../../references/verification-guide.md#gpu-execution-marker)).
The coverage note is what carries that detail, which is why it is
required rather than optional.

If the routine is too large to finish in one pass, extract or split it
(Phase 2 refactor) and keep `status: "pending"` until all planned nests
are offloaded and integrated.

### No `#ifdef` in translated files

Files in `{parent}/openacc/` are the pure GPU version — they
are only compiled when GPU flags are active. Write OpenACC
directives and device utility calls **directly**.

**Do NOT** wrap directives in `#ifdef {INCLUDE_GUARD}`:
```fortran
#ifdef {INCLUDE_GUARD}        ! WRONG inside {parent}/openacc/
  !$acc parallel loop
#endif
```

The `#ifdef` switching between original and translated code
belongs at the **call site** (Phase 4e integration), not
inside `{parent}/openacc/` files.

### GPU entry points and derived types

The translated routine takes the **same arguments as the routine it
replaces**, derived types included, so the dispatcher is a one-line
`#ifdef` swap and no flattening logic leaks into the host file.

Inside the routine, dereference the components you need into flat local
pointers or scalars *before* any compute region, and reference only
those locals inside it:

```fortran
subroutine compute_kernel_openacc(state, h, is, ie, js, je, nk)
  use grid_types_mod, only: grid_type
  integer, intent(in) :: is, ie, js, je, nk
  type(grid_type), intent(in) :: state        ! signature unchanged
  real, intent(inout) :: h(is:ie, js:je, nk)
  real, pointer :: dx(:,:), dy(:,:)
  dx => state%dx                              ! flatten before region
  dy => state%dy
  !$acc parallel loop collapse(2)
  do j = js, je
    do i = is, ie
      h(i,j,1) = h(i,j,1) * dx(i,j) / dy(i,j) ! flat locals only
    end do
  end do
end subroutine compute_kernel_openacc
```

**Host dispatcher (integration call site):**
```fortran
#ifdef USE_OPENACC
  call compute_kernel_openacc(state, h, is, ie, js, je, nk)
#else
  call compute_kernel(state, h, is, ie, js, je, nk)
#endif
```

The argument lists are identical, so the swap cannot silently reorder
or drop an argument.

Rules:
- No derived-type variable may be referenced **inside** an `!$acc`
  compute region, or passed to an `!$acc routine` callee — pass the
  flat locals instead.
- `separate` mode: place the flat locals, not the derived type, in the
  hoisted `!$acc data` clauses.
- `unified` / `managed` mode: flattening still avoids descriptor
  deep-copy faults even though memory is coherent.
- The translated module `use`s the module that declares the type. The
  host file already `use`s the translated module, so if the type is
  declared in that same host module the two close a `use` cycle —
  extract the type into its own module first (Phase 2).

**Non-contiguous components.** Pointer association preserves a stride
without building a temporary, but a non-contiguous target still has no
contiguous device image: `dtarray(:)%comp` and strided slices cannot be
mapped cleanly. Repack once outside the hoisted region, or change the
layout in a Phase 2 refactor.

**STOP and ask the user** when a needed component genuinely cannot be
flattened before the region — an array of derived types, deeply nested
allocatable components, or a polymorphic type touched inside the loop.
A decoupling refactor (Phase 2, see [refactoring.md](refactoring.md))
is the remedy for those cases, not the default path.

### Chunking strategy

For functions with multiple loop nests, work in chunks. Each
chunk should be a logically complete unit:
- One loop nest or a group of related loop nests
- Data clauses consistent with `memory_mode`
- Complete `!$acc` directive pairs in `separate` mode only
  (never leave an `!$acc data` without its `!$acc end data`)

### Per-chunk workflow

1. **Identify the target construct** from the plan's
   `accelerator_chunks` list.

2. **Use device utilities.** If the loop body calls any
   utility functions that were prepared in Phase 3, add a `USE` of
   that parent’s device-utils module (see
   [device-utils-sibling.md](device-utils-sibling.md) for the
   `{parent_id}_device_utils_mod` name) at the top of the
   translated file and call the device version directly:
   ```fortran
   use src_physics_device_utils_mod
     ...
     im = periodic_index_device(i - 1, N)
   ```
   No `#ifdef` needed — this file is GPU-only.

   If the function calls utilities not in `device_utilities`
   (missed during planning), add them to the appropriate
   `{parent}/openacc/device_utils.f90` now and re-run that
   directory's device-utils test before continuing.

3. **Add directives** per memory mode.

   **`unified` or `managed`** (most common on coherent HW):
   ```fortran
   !$acc parallel loop collapse(3)
   do k = 1, nz
     do j = 1, ny
       do i = 1, nx
         output(i,j,k) = compute(input(i,j,k))
       end do
     end do
   end do
   ```

   **`separate`** (discrete GPU memory):
   ```fortran
   ! Data region belongs at hoisted scope (timestep loop),
   ! NOT here if this kernel is called thousands of times.
   !$acc parallel loop collapse(3) present(input, output)
   do k = 1, nz
     ...
   end do
   ```

4. **Lint** (when `memory_mode` is not `separate`):
   ```bash
   grep -nE '!$acc (data|enter data|exit data)|copy(in|out)?\(' \
     src/physics/openacc/compute_kernel_openacc.f90
   ```
   Repeat for each `{parent}/openacc/*_openacc.f90` file.
   Revise if any matches appear.

5. **Verify after each chunk:**
   ```bash
   source .nsight-ai-port/env.sh && \
     USE_OPENACC=1 bash .nsight-ai-port/build_correctness.sh
   ./test_{function}
   ```
   Must pass before moving to the next chunk.

6. **On failure** (see [Fault Tolerance](../../../SKILL.md#fault-tolerance)):
   - Read the build error
   - Common issues: redundant `#ifdef` in openacc/, forbidden
     data clauses for memory mode, wrong `private`/`reduction`
     clauses, module compile order
   - If stuck, ask the user for help

7. **Git commit after each successful chunk:**
   ```bash
   git commit -am "openacc: add directives to compute_kernel (chunk N/M)"
   ```

## Phase 4d: Verify Numerical Correctness

The comparison harness ships with the skill as compilable source. Do not
re-derive it inline. Copy it **once for the whole port**, at the
source root rather than beside each original — see
[verification-guide.md](../../../references/verification-guide.md#one-copy-per-port-at-the-source-root):

```bash
mkdir -p "$NSIGHT_AI_PORT_SOURCE_DIR/openacc/test"
cp assets/test-harness/compare_field.f90 \
   "$NSIGHT_AI_PORT_SOURCE_DIR/openacc/test/"
```

It defines module `nsight_port_compare`, whose contract is:

| Call | What it does |
|------|--------------|
| `tolerances_for(sample)` | kind-aware defaults, then `WMAD_TOL` / `ABS_TOL` / `REL_TOL` / `REL_FLOOR` / `MAX_REL_CHECK` from the environment |
| `compare_field(candidate, reference, tol [, weight])` | generic over `real32`/`real64` at ranks 1–3; `weight` is the cell volume or area on a non-uniform mesh, omitted on a uniform one |
| `report_and_gate(result, tol, label)` | prints `wmad`, `max_rel`, `n_exceed` of `n_total`, and the worst element's subscripts; returns `.true.` on pass |

Fill in the empty `#ifdef {INCLUDE_GUARD}` block in the test with the call:

```fortran
#ifdef {INCLUDE_GUARD}
  block
    use nsight_port_compare
    real, allocatable :: output_candidate(:,:,:)
    type(compare_tolerances) :: tol
    type(compare_result) :: res

    allocate(output_candidate, mold=output_reference)

    ! Identical inputs to both paths: `input` is passed unchanged, so the
    ! original call above must not have modified it. If it does, keep a
    ! pristine copy before that call and pass the copy here.
    call compute_kernel_openacc(input, output_candidate, nx, ny, nz)

    tol = tolerances_for(output_reference(1,1,1))
    res = compare_field(output_candidate, output_reference, tol)
    if (.not. report_and_gate(res, tol, 'compute_kernel')) stop 1
  end block
#endif
```

Compare **every** output the function produces, not just the first — one
`report_and_gate` call per output field, each with its own label.

`wmad` (normalized weighted mean absolute difference) is the gate.
`n_exceed` and the worst subscripts are printed so an outlier failure is
reviewable rather than just a number to loosen.

An outlier-only failure (`wmad` passes, `n_exceed` small and scattered
through low-magnitude cells) may be algorithmic numerics rather than a
bug. You are expected to review it, and may then raise `REL_FLOOR` or set
`MAX_REL_CHECK=off`, recording the reasoning in `progress.json`. See
[Reviewing an outlier failure](../../../references/verification-guide.md#reviewing-an-outlier-failure)
for what must never be waived — non-finite values, exceedance counts that
grow with problem or thread count, high-magnitude cells, and structured
(contiguous / strided / boundary) patterns.

**Verify:**
```bash
source .nsight-ai-port/env.sh && \
  USE_OPENACC=1 bash .nsight-ai-port/build_correctness.sh
compute-sanitizer ./test_compute_kernel
```
(If `compute-sanitizer` is not available, run the test
directly.)

Must pass. If numerical differences exceed tolerance:
  - Check Fortran array indexing (1-based vs 0-based)
  - If `memory_mode` is `separate`: check data clause
    completeness at hoisted scope (missing `copyin`/`copyout`)
  - If `memory_mode` is `unified` or `managed`: check for
    forbidden per-kernel `copyin`/`copyout` (remove them)
  - Check reduction variables
  - Check if GPU floating-point order differs (use relaxed
    tolerances if needed via `ABS_TOL`/`REL_TOL` env vars)

**This phase must pass before integration.**

Git commit:
```bash
git commit -am "verify: numerical correctness for compute_kernel"
```

## Phase 4e: Integrate

See [verification-guide.md](../../../references/verification-guide.md) for the full
dual-path verification protocol and [build-integration.md](build-integration.md)
for build-system best practices.

### Preprocessing: every Fortran compiler that touches the guarded source

Call-site `#ifdef` blocks live in **original** `.f90` files. Do
**not** rename sources to `.F90` project-wide. Enable preprocessing on the
compile line instead — for **every configuration that compiles that file**,
not just the one building the GPU path:

- **nvfortran:** add `-Mpreprocess` to `FFLAGS` / per-target rules
- **gfortran:** `-cpp` (or `-fpp`, as the project prefers)
- **ifx / ifort:** `-fpp`

The GPU path already has preprocessing enabled alongside `-acc`, so the
configurations that break are the *other* ones. The case that is easy to
miss is a project keeping a **pure-Fortran CPU baseline built by
gfortran** while the accelerated path uses nvfortran: that baseline
compiles the same guarded file and needs `-cpp`, even though it never
defines the guard. The symptom is not a wrong answer — it stops building
at all, the moment dispatch lands.

Settle this while exploring the build, not after editing the source: any
project with more than one Fortran configuration will hit it. Then verify
by building **every** configuration, not only the accelerated one.

### Call-site dispatch

At the call site in the original code (not the translated file).
The translated routine takes the original argument list, so both
branches pass the same actuals:

```fortran
#ifdef {INCLUDE_GUARD}
  use compute_kernel_openacc_mod  ! translated module
#endif

  ! ... existing code ...

#ifdef {INCLUDE_GUARD}
  call compute_kernel_openacc(grid, h, is, ie, js, je, nk)
#else
  call compute_kernel(grid, h, is, ie, js, je, nk)
#endif
```

### Build system updates

1. **Compiler flags:** Add `-D{INCLUDE_GUARD}` (from
   `config.json:include_guard`) to the preprocessor flags for
   project sources.
   **Never** pass `-acc` or `-DUSE_OPENACC` to external library
   compilations.

2. **GPU flags:** Add `-acc -gpu=ccNN` plus the memory flag
   from `config.json:memory_mode` (see setup guide):

   | `memory_mode` | Add to `-gpu=` |
   |---------------|----------------|
   | `separate` | `ccNN` only |
   | `managed` | `ccNN,mem:managed` |
   | `unified` | `ccNN,mem:unified` |

   Record the exact flags in `.nsight-ai-port/build_notes.txt`. If the project
   uses configure-time dependency scanning (makedep, autotools),
   changing OpenACC flags may require wiping the build/configure
   cache before rebuilding.

3. **Archive commands:** If the Makefile uses `ar` with explicit
   file lists or glob patterns, ensure **every**
   `{parent}/openacc/*.o` is included. Globs like `*.o` do not
   match nested `src/*/openacc/*.o`.

4. **Separate trees, so cleaning is unnecessary:** give the CPU and GPU
   configurations their own object directories
   (`$NSIGHT_AI_PORT_BUILD_DIR/cpu` and `/gpu`). Then never clean
   unconditionally — a blanket `make clean` destroys incremental builds,
   while sharing one tree can link a part-CPU/part-GPU binary. See
   [never clean unconditionally](../../../references/verification-guide.md#never-clean-unconditionally)
   for the few cases that do warrant a full rebuild.

### Dual-path verification

All five checks must pass:

```bash
# 1. CPU build
source .nsight-ai-port/env.sh && bash .nsight-ai-port/build_correctness.sh

# 2. CPU run
source .nsight-ai-port/env.sh && bash .nsight-ai-port/run_correctness.sh

# 3. GPU build
source .nsight-ai-port/env.sh && \
  USE_OPENACC=1 bash .nsight-ai-port/build_correctness.sh

# 4. GPU run — check for execution marker in output
source .nsight-ai-port/env.sh && \
  USE_OPENACC=1 bash .nsight-ai-port/run_correctness.sh

# 5. Standalone test (sanitized)
compute-sanitizer ./test_{function}
```

Step 4: check stdout for `NSIGHT_AI_PORT_FRAMEWORK_EXEC:
compute_kernel`. If missing, the GPU path did not execute —
investigate (see verification guide for causes).

### Commit and update progress

```bash
git commit -am "integrate: compute_kernel with OpenACC dispatch"
```

Update `progress.json`:
- Set function's `phases.integrate: true`
- Set function's `status: "completed"`
- Set function's `commit_hash` to `git rev-parse HEAD`
- Add `"file:function"` to `translated_functions` array
- Update `last_working_commit` to the same SHA

## Resuming After Interruption

If a session is interrupted, check `.nsight-ai-port/progress.json`
to find where you left off:

1. Read `progress.json` and `porting_plan.json`.
2. Find the first function with `status != "completed"`.
3. Check which phases are done for that function.
4. Resume from the next incomplete phase.
5. Verify the codebase is in a consistent state:
   ```bash
   source .nsight-ai-port/env.sh && bash .nsight-ai-port/build_correctness.sh
   ```
