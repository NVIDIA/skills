# Translation Guide (Phase 4 — Per-Function) — C/C++ + Kokkos

Flavor pack: `flavors/cpp/kokkos/`. The `translate-function` subskill
reads this file for **Phase 4** (per-function translation). Phase 3
device utilities live in [device-utils.md](device-utils.md).

Translated code lives under `{parent}/kokkos/`. GPU builds use
`USE_KOKKOS=1` and define `-DUSE_KOKKOS`. Target **Kokkos 4 + C++17** (see patterns.md).

This pack accelerates **C++** (Kokkos). The steps below are written
for a same-language port (`from_lang == to_lang`): no
language-translation step and no cross-language shim — only the
accelerator rewrite (add Kokkos), verification, and integration.

**Cross-language (`from_lang != to_lang`):** Stage A already
produced C++ and this pack is Stage B. Read the Handoff in
`transitions/{from_lang}-to-{to_lang}.md` first; where it conflicts
with this file, the Handoff wins.

## Directory Layout

Each original source file gets a **sibling** `kokkos/` directory in
the **same parent folder** as that source file — not a single
project-root `kokkos/`. Nested codebases therefore have many
`kokkos/` directories.

**Path rule:** for original `{parent}/{source}.cpp` and target
function `{function}`:

| Artifact | Path |
|----------|------|
| Translated kernel (impl) | `{parent}/kokkos/{function}_kokkos.cpp` |
| Translated kernel (header) | `{parent}/kokkos/{function}_kokkos.hpp` |
| Standalone test | `{parent}/kokkos/test/test_{function}/` |
| Device utilities | `{parent}/kokkos/device_utils.hpp` (where the host utility lives) |
| Device util test | `{parent}/kokkos/test/test_device_utils/` |

Example: `src/physics/solver.cpp` → `src/physics/kokkos/…`

```text
project_root/
├── src/
│   ├── physics/
│   │   ├── solver.cpp       ← original
│   │   └── kokkos/          ← sibling to solver.cpp
│   │       ├── compute_kernel_kokkos.cpp
│   │       ├── compute_kernel_kokkos.hpp
│   │       └── test/
│   │           └── test_compute_kernel/
│   │               └── test_compute_kernel.cpp
│   └── dynamics/
│       ├── kernel_a.cpp
│       └── kokkos/
│           ├── kernel_a_kokkos.cpp
│           ├── kernel_a_kokkos.hpp
│           └── test/test_kernel_a/
├── utils/
│   ├── kernel_utils.cpp
│   └── kokkos/
│       ├── device_utils.hpp ← Phase 3 (host utils here)
│       └── test/test_device_utils/
└── CMakeLists.txt
```

Key points:
- **`kokkos/` is per source directory**, not one global tree
- **One translated function per `{function}_kokkos.{cpp,hpp}` pair**
  in that directory's `kokkos/` — do NOT combine multiple functions
  into one file
- A kernel in `src/dynamics/kokkos/` may `#include` a header from
  `utils/kokkos/device_utils.hpp`; wire both dirs in the build
- The header declares the translated function's signature so the
  original `.cpp` can call it after `#include`-ing it under the guard
- Original source files are never modified (except call sites during
  integration)

---

## Phase 4: Per-Function Translation

## Phase 4a: Build System Exploration

`.nsight-ai-port/build_notes.txt` should exist. If not, generate it
from [build-integration.md](build-integration.md).

Read it before scaffolding so this function is compiled and linked
the same way as the rest of the GPU path (guard, Kokkos install,
how `{parent}/kokkos/` is picked up).

## Phase 4b: Scaffold Test Harness

**Subagent opportunity.** Clear input (function signature +
dependencies from the plan) and a clear success criterion (the test
builds and runs using the original code).

### Prerequisite: function must be externally callable

Before writing the test harness, confirm the target function can be
called from a standalone program (declared in a header, with linkage
the test can reach). If the function is a `static` free function, an
unexposed class private method, or relies on file-local state, the
Phase 2 refactoring was not completed — go back and do it first.

**If you discover at this point that the function cannot be tested in
isolation, STOP and explain the blocker to the user. Do NOT substitute
whole-program verification or skip the per-function test.** Wait for
the user to accept a deviation before proceeding differently.

### Prerequisite: pick a guard-free CPU reference symbol

The CPU symbol the test compares against must **still be CPU** in a
`USE_KOKKOS=1` build. Use the original `{function}`: integration puts
the guard at its **call site**, so the body itself stays CPU.

Confirm the reference function's body contains **no** `#ifdef
USE_KOKKOS`. If integration would have to put one inside it, split the
body out into a `_host` symbol first — otherwise the test compares the
GPU against itself, passes with a max difference of exactly 0.0, and
verifies nothing.

### Steps

1. **Read the function.** Understand:
   - Signature and arguments (pointers, sizes, references, by-value)
   - `#include` dependencies (which headers declare types it needs)
   - Input/output arrays and their shapes / memory layout
   - Any global or member state used

2. **Create test directory** (sibling `kokkos/` next to the original
   source file):
   ```bash
   # Example: target in src/physics/solver.cpp
   mkdir -p src/physics/kokkos/test/test_{function}/
   ```

3. **Write test program** — in the **original language** (see
   [Test `main` must be in the original language](../../../references/verification-guide.md#test-main-must-be-in-the-original-language)),
   which here is C++:
   `src/physics/kokkos/test/test_{function}/test_{function}.cpp`

   ```cpp
   #include <Kokkos_Core.hpp>
   #include <cstdio>
   #include <cmath>
   #include <vector>
   #include "solver.hpp"          // declares the ORIGINAL function

   int main(int argc, char* argv[]) {
     Kokkos::ScopeGuard guard(argc, argv);

     // Test extents: pick them from the function's own real extents
     // (rank, aspect ratio, halo width). The values below are only a
     // placeholder for a 3-D structured field.
     const int nx = 100, ny = 100, nz = 50;
     // One input, shared by both paths. Two outputs, named for their role
     // in the comparison — the harness takes (candidate, reference).
     std::vector<double> input(nx*ny*nz);
     std::vector<double> output_reference(nx*ny*nz);
     for (size_t n = 0; n < input.size(); ++n)
       input[n] = /* realistic value */;

     // Call the guard-free CPU reference symbol chosen above
     compute_kernel(input.data(), output_reference.data(), nx, ny, nz);

     // Smoke fingerprint only — NOT a correctness gate. Never gate on a
     // bare sum: opposite-sign errors cancel and a near-zero mean makes
     // the sum ill-conditioned (see references/verification-guide.md).
     double mean_abs = 0.0, peak = 0.0;
     for (double v : output_reference) { mean_abs += std::fabs(v);
                               peak = std::fmax(peak, std::fabs(v)); }
     mean_abs /= (double)output_reference.size();
     std::printf("Output mean|v| %g  peak|v| %g\n", mean_abs, peak);

   #ifdef USE_KOKKOS
     // GPU comparison block — EMPTY for now.
     // Phase 4d will fill this in.
   #endif

     std::printf("TEST PASSED\n");
     return 0;
   }
   ```

   Rules for the test:
   - Use **realistic** input sizes and values for **unit** testing —
     physics-appropriate, fast to run. Do **not** copy grid sizes,
     timestep counts, or case parameters from the active
     `config.json:problems[]` entry
     (integration/profiling workload). Standalone tests must stay
     small and deterministic.
   - Call the **original** function (not the translated one yet).
   - The `#ifdef USE_KOKKOS` block must be present but empty.
   - Must exit 0 and print a recognizable success line.
   - Must not copy source files — reference them in-tree via include
     paths.
   - Wrap everything in a `Kokkos::ScopeGuard` so the test can launch
     kernels in Phase 4d.

4. **Hook into the build system.** Add a test target.

   For **CMake** (the common Kokkos case):
   ```cmake
   # In src/physics/kokkos/test/test_compute_kernel/CMakeLists.txt
   add_executable(test_compute_kernel test_compute_kernel.cpp
       ${CMAKE_SOURCE_DIR}/src/physics/solver.cpp)
   target_include_directories(test_compute_kernel PRIVATE
       ${CMAKE_SOURCE_DIR}/src/physics
       ${CMAKE_SOURCE_DIR}/src/physics/kokkos
       ${CMAKE_SOURCE_DIR}/kokkos/test)  # header-only comparison harness
   target_link_libraries(test_compute_kernel PRIVATE Kokkos::kokkos)
   target_compile_features(test_compute_kernel PRIVATE cxx_std_17)
   ```
   Add `add_subdirectory(test/test_compute_kernel)` so a normal
   `cmake --build .` builds it.

   For **Make**: add a target that compiles the test with
   `nvcc_wrapper`, the Kokkos include/lib, `-std=c++17
   --extended-lambda --expt-relaxed-constexpr`, and `-DUSE_KOKKOS`,
   referencing the original `.cpp` in-tree (do NOT copy it).

   The test target must compile under the GPU build path (with
   `-DUSE_KOKKOS` and Kokkos flags) even though the `#ifdef` block is
   empty. Pass `-DUSE_KOKKOS` to the test main AND to every in-tree
   dependency that carries the matching `#ifdef`.

5. **Create placeholder translated files** in the same
   `{parent}/kokkos/` as the original source — one impl + header pair
   per function (do **not** copy the entire original file):
   ```cpp
   // src/physics/kokkos/compute_kernel_kokkos.hpp
   #pragma once
   void compute_kernel_kokkos(const double* input, double* output,
                              int nx, int ny, int nz);
   ```
   ```cpp
   // src/physics/kokkos/compute_kernel_kokkos.cpp
   #include "compute_kernel_kokkos.hpp"
   #include <Kokkos_Core.hpp>
   void compute_kernel_kokkos(const double* input, double* output,
                              int nx, int ny, int nz) {
     // TODO: Kokkos implementation added in Phase 4c
   }
   ```
   Name convention: `{parent}/kokkos/{function}_kokkos.{cpp,hpp}`.

6. **Write manifest:**
   ```json
   {
     "test_executable": "/abs/path/to/test_compute_kernel"
   }
   ```

7. **Verify:**
   ```bash
   source .nsight-ai-port/env.sh && \
     USE_KOKKOS=1 bash .nsight-ai-port/build_correctness.sh
   ./test_{function}
   ```
   Both must pass.

8. **Git commit:**
   ```bash
   git add -A && git commit -m \
     "scaffold: test harness for compute_kernel"
   ```

9. **Update progress.json:** Set this function's `phases.scaffold:
   true` and record `test_dir` and `test_executable`.

## Phase 4c: Add Kokkos Constructs

See [patterns.md](patterns.md) for kernel patterns, Views, memory
spaces, and common pitfalls.

Work on this function's dedicated files in `{parent}/kokkos/` (e.g.
`src/physics/kokkos/solver_kokkos.cpp`). Each function has its own
translated file pair — do **not** combine functions or modify the
original source file.

### Execution marker (mandatory)

Open `{function}_kokkos` with a one-time marker, so Phase 4d can prove
the GPU path ran rather than a host fallback. It belongs in the
translated body, never at the call site — see
[GPU Execution Marker](../../../references/verification-guide.md#gpu-execution-marker)
for the rank-0 form under MPI.

```cpp
static bool marker_printed = false;
if (!marker_printed) {
  std::printf("NSIGHT_AI_PORT_FRAMEWORK_EXEC: {function}\n");
  marker_printed = true;
}
```

### Memory mode (mandatory)

Read `config.json:memory_mode` before choosing View types. Follow the
View / memory-mode policy in patterns.md:

| Mode | View strategy |
|------|---------------|
| `managed` | `Kokkos::SharedSpace` Views (or unmanaged over managed ptr) — no `deep_copy` |
| `unified` | Unmanaged Views over host pointers — no `deep_copy` |
| `separate` | Device Views + host mirror + **hoisted** `deep_copy` |

In `managed`/`unified` mode there must be **zero** `deep_copy` calls
in the translated kernel.

### No `#ifdef` in translated files

Files in `{parent}/kokkos/` are the pure GPU version — they are only
compiled when `USE_KOKKOS` is defined. Write Kokkos constructs
**directly**. Do **not** wrap them in `#ifdef USE_KOKKOS` — that guard
belongs at the **call site** (Phase 4e integration), not inside
`{parent}/kokkos/` files.

### Decoupled GPU entry points

Translated functions must take **flat arguments only**: raw pointers
(or `const` pointers) + integer sizes, not heavyweight host
container/class references that are not device-safe. This keeps the
GPU boundary clean and lets the wrapper build Views internally.

**Pattern:**
1. Kernel in `{parent}/kokkos/{function}_kokkos.cpp` takes scalars +
   raw array pointers + extents.
2. Internally it wraps the pointers in Views appropriate to
   `memory_mode` (SharedSpace / unmanaged / device + mirror).
3. Host call site (integration) passes `.data()` / raw pointers.

```cpp
void compute_interaction_kokkos(const double* dx, const double* dy,
                                const double* h, double* du,
                                int is, int ie, int js, int je, int nk) {
  using UV = Kokkos::View<const double**, Kokkos::LayoutRight,
      Kokkos::MemoryTraits<Kokkos::Unmanaged>>;   // unified mode
  const int ni = ie - is, nj = je - js;
  UV dxv(dx, ni, nj), dyv(dy, ni, nj);
  // ... build output Views, launch parallel_for ...
}
```

If a target takes a non-device-safe class/struct, add a decoupling
refactor task (Phase 2) before translating (see refactoring.md).

### Chunking strategy

For functions with multiple kernels, work in chunks. Each chunk should
be a logically complete unit:
- One kernel or a group of related kernels
- View setup consistent with `memory_mode`
- Complete RAII / `fence()` placement (never leave a pending async
  result unread before the host needs it)

### Per-chunk workflow

1. **Identify the target construct** from the plan's
   `accelerator_chunks` list.

2. **Use device utilities.** If a kernel calls helpers prepared in
   Phase 3, `#include "device_utils.hpp"` and call the device version
   directly — no `#ifdef` needed (GPU-only file):
   ```cpp
   #include "device_utils.hpp"
   ...
   im = periodic_index_device(i - 1, N);
   ```
   If the function calls utilities not in `device_utilities` (missed
   during planning), add them to the appropriate
   `{parent}/kokkos/device_utils.hpp` now and re-run that directory's
   device-utils test before continuing.

3. **Add Kokkos constructs** per memory mode.

   **`managed`** (Kokkos 4 SharedSpace):
   ```cpp
   // Unmanaged: these Views wrap caller-owned raw pointers and must not
   // take ownership. LayoutRight matches C row-major indexing.
   Kokkos::View<double***, Kokkos::LayoutRight, Kokkos::SharedSpace,
                Kokkos::MemoryTraits<Kokkos::Unmanaged>>
       in(input, nz, ny, nx), out(output, nz, ny, nx);
   using MD = Kokkos::MDRangePolicy<Kokkos::Rank<3>>;
   Kokkos::parallel_for("compute", MD({0,0,0},{nz,ny,nx}),
       KOKKOS_LAMBDA(const int k,const int j,const int i){
         out(k,j,i) = compute(in(k,j,i));
       });
   Kokkos::fence();
   ```

   **`unified`**: same kernel, but Views are
   `Kokkos::MemoryTraits<Kokkos::Unmanaged>` over the host pointers.

   **`separate`**: allocate device Views, `deep_copy` inputs in /
   outputs out **once** around the kernel(s), never per call on a hot
   path.

4. **Lint** (when `memory_mode` is not `separate`): there must be no
   `deep_copy` and no device-vs-host mirror copies in the translated
   file:
   ```bash
   grep -nE 'deep_copy|create_mirror' \
     src/physics/kokkos/compute_kernel_kokkos.cpp
   ```
   Or search all sibling trees:
   ```bash
   find . -path '*/kokkos/*_kokkos.cpp' -print0 | \
     xargs -0 grep -nE 'deep_copy|create_mirror'
   ```
   Revise if any matches appear in managed/unified mode.

5. **Verify after each chunk:**
   ```bash
   source .nsight-ai-port/env.sh && \
     USE_KOKKOS=1 bash .nsight-ai-port/build_correctness.sh
   ./test_{function}
   ```
   Must pass before moving to the next chunk.

6. **On failure** (see [Fault Tolerance](../../../SKILL.md#fault-tolerance)):
   - Read the build error.
   - Common issues: redundant `#ifdef` in `kokkos/`, forbidden
     `deep_copy` for memory mode, missing `--extended-lambda`,
     capturing `this`/class members in a lambda, View access with
     `[]` instead of `()`, wrong layout when wrapping raw pointers,
     missing `Kokkos::fence()`.
   - If stuck, ask the user for help.

7. **Git commit after each successful chunk:**
   ```bash
   git commit -am "kokkos: add constructs to compute_kernel (chunk N/M)"
   ```

## Phase 4d: Verify Numerical Correctness

The comparison harness ships with the skill as compilable source. Do not
re-derive it inline. Copy the C++ header **once for the whole port**, at
the source root rather than beside each original — see
[verification-guide.md](../../../references/verification-guide.md#one-copy-per-port-at-the-source-root):

```bash
mkdir -p "$NSIGHT_AI_PORT_SOURCE_DIR/kokkos/test"
cp assets/test-harness/compare_field.hpp \
   "$NSIGHT_AI_PORT_SOURCE_DIR/kokkos/test/"
```

It defines namespace `nsight_ai_port`, whose contract is:

| Call | What it does |
|------|--------------|
| `tolerances_for<T>()` | element-type defaults, then `WMAD_TOL` / `ABS_TOL` / `REL_TOL` / `REL_FLOOR` / `MAX_REL_CHECK` from the environment |
| `compare_field(candidate, reference, n, tol [, weights])` | templated on the element type; `weights` is the cell volume or area on a non-uniform mesh, `nullptr` on a uniform one |
| `report_and_gate(result, tol, label)` | prints `wmad`, `max_rel`, `n_exceed` of `n_total`, and the worst flat index; returns `true` on pass |

Fill in the empty `#ifdef USE_KOKKOS` block in the test with the call:

```cpp
#ifdef USE_KOKKOS
  #include "compare_field.hpp"
  #include "compute_kernel_kokkos.hpp"   // (at top of file)
  // ...
  // Identical inputs to both paths: `input` is passed unchanged, so the
  // reference call above must not have modified it. If it does, keep a
  // pristine copy before that call and pass the copy here.
  std::vector<double> output_candidate(output_reference.size(), 0.0);
  compute_kernel_kokkos(input.data(), output_candidate.data(), nx, ny, nz);

  // `output_reference` came from the guard-free CPU reference.
  const auto tol = nsight_ai_port::tolerances_for<double>();
  if (!nsight_ai_port::report_and_gate(
          nsight_ai_port::compare_field(output_candidate.data(),
                                      output_reference.data(),
                                      output_reference.size(), tol),
          tol, "compute_kernel")) {
    return 1;
  }
#endif
```

Compare **every** output the function produces, not just the first — one
`report_and_gate` call per output field, each with its own label.

When the candidate lives in a Kokkos `View`, mirror it to the host and
pass the mirror's `data()`; `compare_field` takes plain host pointers.

`wmad` is the gate. `n_exceed` counts elements failing the element-wise
test (an element passes if EITHER abs diff <= `abs_tol` OR rel diff <=
`rel_tol`); it and the worst index are printed so an outlier failure is
reviewable. A non-finite value anywhere is reported separately and is
never waivable.

An outlier-only failure (`wmad` passes, `n_exceed` small and scattered in
low-magnitude cells) may be algorithmic numerics rather than a bug. You
are expected to review it and may raise `REL_FLOOR` or set
`MAX_REL_CHECK=off`, recording the reasoning in `progress.json`. Read
[Reviewing an outlier failure](../../../references/verification-guide.md#reviewing-an-outlier-failure)
for what must never be waived.

**Verify:**
```bash
source .nsight-ai-port/env.sh && \
  USE_KOKKOS=1 bash .nsight-ai-port/build_correctness.sh
compute-sanitizer ./test_compute_kernel
```
(If `compute-sanitizer` is not available, run the test directly.)

The TRANSLATED path MUST actually execute on the device — at least one
GPU kernel must launch. If `compute-sanitizer` reports no kernels
launched / "terminated before first instrumented API call", the GPU
path was compiled out or never called — that is a **build/wiring bug**
(usually the test built without `-DUSE_KOKKOS`, so the `#ifdef` block
was preprocessed away), not a numerical one. Fix the wiring before
touching the math.

Must pass. If numerical differences exceed tolerance:
  - Check View layout (LayoutLeft vs LayoutRight) when wrapping raw
    pointers — a transpose shows up as large, structured diffs.
  - If `memory_mode` is `separate`: check `deep_copy` completeness and
    direction at hoisted scope (missing copy-in/out).
  - If `memory_mode` is `unified`/`managed`: check for a stray
    `deep_copy` or wrong View memory space.
  - Check reduction initial values and reducer choice.
  - Missing `Kokkos::fence()` before the host reads results.
  - GPU floating-point order differs (loosen tolerances via
    `ABS_TOL`/`REL_TOL` for genuine FP noise only).

**This phase must pass before integration.**

Git commit:
```bash
git commit -am "verify: numerical correctness for compute_kernel"
```

## Phase 4e: Integrate

See [verification-guide.md](../../../references/verification-guide.md) for
the full dual-path verification protocol and
[build-integration.md](build-integration.md) for build-system best
practices.

### Same-language: no shim needed

Source and target are both C++, so no cross-language shim is required.
Place the translated `{function}_kokkos.{cpp,hpp}` in
`{parent}/kokkos/` and `#include` the header at the call site under
the guard.

### Call-site dispatch

At the call site in the original code (not the translated file):

```cpp
#ifdef USE_KOKKOS
#include "kokkos/compute_kernel_kokkos.hpp"
#endif

  // ... existing code ...

#ifdef USE_KOKKOS
  compute_kernel_kokkos(input, output, nx, ny, nz);
#else
  compute_kernel(input, output, nx, ny, nz);
#endif
```

### Kokkos initialize/finalize

The translated kernels assume Kokkos is initialized. Ensure the host
application calls `Kokkos::initialize(argc, argv)` once at program
start and `Kokkos::finalize()` before exit (guarded by
`#ifdef USE_KOKKOS`). Prefer a single `Kokkos::ScopeGuard` at
`main()` scope. Do **not** initialize/finalize inside the hot
function — that is extremely expensive and incorrect across repeated
calls.

### Build system updates

1. **Preprocessor define:** Add `-DUSE_KOKKOS` (from
   `config.json:include_guard`) to the project sources that carry the
   guard. For CMake, a `-DUSE_KOKKOS=ON` option does NOT become a C++
   define automatically:
   ```cmake
   option(USE_KOKKOS "Enable Kokkos GPU acceleration" OFF)
   if(USE_KOKKOS)
     add_compile_definitions(USE_KOKKOS)
   endif()
   ```
   Or per-target: `target_compile_definitions(<target> PRIVATE
   USE_KOKKOS)`. Apply the SAME `-DUSE_KOKKOS` to the project, in-tree
   dependencies, AND the standalone test.

2. **Kokkos linkage and flags:** Compile and link the new
   `{parent}/kokkos/*.cpp` with the Kokkos package and C++17. For
   CMake use `find_package(Kokkos REQUIRED)` and link
   `Kokkos::kokkos`. For the `nvcc_wrapper` path add
   `-std=c++17 --extended-lambda --expt-relaxed-constexpr`. The
   compute capability is chosen by the Kokkos build (`Kokkos_ARCH_*`)
   — detect the actual GPU rather than copying a hardcoded arch (see
   build-integration.md).
   **Never** apply Kokkos/CUDA device flags to external/third-party
   libraries.

3. **Archive/link of new subdir:** When adding files in new
   `{parent}/kokkos/` subdirectories, ensure **every** `*.o` (or
   CMake target object) is included in the library/link step. Globs
   like `src/*.cpp` do not match nested `src/*/kokkos/*.cpp`.

4. **Separate trees, so cleaning is unnecessary:** give the CPU and GPU
   configurations their own object directories
   (`$NSIGHT_AI_PORT_BUILD_DIR/cpu` and `/gpu`). Then never clean
   unconditionally — a blanket `make clean` destroys incremental builds,
   while sharing one tree can link a part-CPU/part-GPU binary. See
   [never clean unconditionally](../../../references/verification-guide.md#never-clean-unconditionally)
   for the few cases that do warrant a full rebuild.

### Dual-path verification

All steps must pass:

```bash
# 1. CPU build (no USE_KOKKOS)
source .nsight-ai-port/env.sh && bash .nsight-ai-port/build_correctness.sh

# 2. CPU run
source .nsight-ai-port/env.sh && bash .nsight-ai-port/run_correctness.sh

# 3. GPU build
source .nsight-ai-port/env.sh && \
  USE_KOKKOS=1 bash .nsight-ai-port/build_correctness.sh

# 4. GPU run — check for execution marker in output
source .nsight-ai-port/env.sh && \
  USE_KOKKOS=1 bash .nsight-ai-port/run_correctness.sh

# 5. Standalone test (sanitized)
compute-sanitizer ./test_{function}
```

Step 4: check stdout for `NSIGHT_AI_PORT_FRAMEWORK_EXEC:
compute_kernel`. If missing, the GPU path did not execute —
investigate (see verification guide for causes; for Kokkos the usual
cause is `-DUSE_KOKKOS` not reaching the file, or Kokkos not
initialized).

### Commit and update progress

```bash
git commit -am "integrate: compute_kernel with Kokkos dispatch"
```

Update `progress.json`:
- Set function's `phases.integrate: true`
- Set function's `status: "completed"`
- Set function's `commit_hash` to `git rev-parse HEAD`
- Add `"file:function"` to `translated_functions` array
- Update `last_working_commit` to the same SHA

## Resuming After Interruption

If a session is interrupted, check `.nsight-ai-port/progress.json` to
find where you left off:

1. Read `progress.json` and `porting_plan.json`.
2. Find the first function with `status != "completed"`.
3. Check which phases are done for that function.
4. Resume from the next incomplete phase.
5. Verify the codebase is in a consistent state:
   ```bash
   source .nsight-ai-port/env.sh && bash .nsight-ai-port/build_correctness.sh
   ```
