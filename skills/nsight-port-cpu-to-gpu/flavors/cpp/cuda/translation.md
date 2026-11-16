# Translation Guide (Phase 4 — Per-Function) — C/C++ + CUDA

Flavor pack: `flavors/cpp/cuda/`. The `translate-function` subskill
reads this file for **Phase 4** (per-function translation). Phase 3
device utilities live in [device-utils.md](device-utils.md).

Translated code lives under `{parent}/cuda/`. GPU builds use
`USE_CUDA=1` and define `-DUSE_CUDA`. See patterns.md for the host/device compilation split,
kernel syntax, and memory-mode policy.

This pack accelerates **C++** (CUDA C++ kernels). The steps below
are written for a same-language port (`from_lang == to_lang`).

**Cross-language (`from_lang != to_lang`):** Stage A already
produced C++ and this pack is Stage B. Read the Handoff in
`transitions/{from_lang}-to-{to_lang}.md` first; where it conflicts
with this file, the Handoff wins.

There IS a **host/device compilation boundary**: kernels live in
`.cu` files (compiled by `nvcc`), while the original `.cpp` call
site only ever sees a **plain C++ header**. Honor that split in
every phase.

## Directory Layout

Each original source file gets a **sibling** `cuda/` directory in the
**same parent folder** as that source file — not a single project-root
`cuda/`. Nested codebases therefore have many `cuda/` directories.

**Path rule:** for original `{parent}/{source}.cpp` and target
function `{function}`:

| Artifact | Path |
|----------|------|
| Translated kernels + wrapper | `{parent}/cuda/{function}_cuda.cu` |
| Public declaration (plain C++) | `{parent}/cuda/{function}_cuda.hpp` |
| Standalone test | `{parent}/cuda/test/test_{function}/` |
| Device utilities | `{parent}/cuda/device_utils.cuh` (where the host utility lives) |
| Error-check macro | `{parent}/cuda/cuda_check.cuh` (shared) |
| Device util test | `{parent}/cuda/test/test_device_utils/` |

Example: `src/physics/solver.cpp` → `src/physics/cuda/…`

```text
project_root/
├── src/
│   ├── physics/
│   │   ├── solver.cpp                  ← original
│   │   └── cuda/                       ← sibling to solver.cpp
│   │       ├── compute_kernel_cuda.cu  ← kernels + wrapper (nvcc)
│   │       ├── compute_kernel_cuda.hpp ← plain C++ decl (host)
│   │       ├── cuda_check.cuh
│   │       └── test/
│   │           └── test_compute_kernel/
│   │               └── test_compute_kernel.cpp
│   └── dynamics/
│       ├── kernel_a.cpp
│       └── cuda/
│           ├── kernel_a_cuda.cu
│           ├── kernel_a_cuda.hpp
│           └── test/test_kernel_a/
├── utils/
│   ├── kernel_utils.cpp
│   └── cuda/
│       ├── device_utils.cuh            ← Phase 3 (host utils here)
│       └── test/test_device_utils/
└── CMakeLists.txt
```

Key points:
- **`cuda/` is per source directory**, not one global tree.
- **One translated function per `{function}_cuda.{cu,hpp}` pair** —
  do NOT combine multiple functions into one file.
- The `.hpp` is plain C++ (host-compilable); the `.cu` holds all CUDA
  syntax. The original `.cpp` `#include`s only the `.hpp` under the
  guard.
- A kernel in `src/dynamics/cuda/` may `#include` a `.cuh` from
  `utils/cuda/`; wire both dirs in the build.
- Original source files are never modified (except call sites during
  integration).

---

## Phase 4: Per-Function Translation

## Phase 4a: Build System Exploration

`.nsight-ai-port/build_notes.txt` should exist. If not, generate it
from [build-integration.md](build-integration.md).

Read it before scaffolding so this function is compiled and linked
the same way as the rest of the GPU path (guard, `nvcc`, arch,
how `{parent}/cuda/` is picked up).

## Phase 4b: Scaffold Test Harness

**Subagent opportunity.** Clear input (signature + dependencies) and a
clear success criterion (the test builds and runs using the original
code).

### Prerequisite: function must be externally callable

Confirm the target can be called from a standalone program (declared
in a header, reachable linkage). If it is a `static` free function, an
unexposed class private method, or relies on file-local state, the
Phase 2 refactoring was not completed — go back and do it first.

**If you discover at this point that the function cannot be tested in
isolation, STOP and explain the blocker to the user. Do NOT substitute
whole-program verification or skip the per-function test.** Wait for
the user to accept a deviation before proceeding differently.

### Prerequisite: pick a guard-free CPU reference symbol

The test compares the GPU against a CPU implementation, so that CPU
symbol must **still be CPU** in a `USE_CUDA=1` build. Use the original
`{function}`: integration puts the guard at its **call site**, so the
body itself stays CPU.

Then confirm the invariant directly: the reference function's body must
contain **no** `#ifdef USE_CUDA`. If integration would have to put a
guard inside it, split the body out into a `_host` symbol first —
otherwise the test compares the GPU against itself, passes with a max
difference of exactly 0.0, and verifies nothing.

### Steps

1. **Read the function.** Understand its signature, `#include`
   dependencies, input/output arrays + shapes/layout, and any
   global/member state.

2. **Create test directory:**
   ```bash
   # Example: target in src/physics/solver.cpp
   mkdir -p src/physics/cuda/test/test_{function}/
   ```

3. **Write test program** — in the **original language** (see
   [Test `main` must be in the original language](../../../references/verification-guide.md#test-main-must-be-in-the-original-language)),
   which here is C++:
   `src/physics/cuda/test/test_{function}/test_{function}.cpp`

   ```cpp
   #include <cstdio>
   #include <cmath>
   #include <vector>
   #include "solver.hpp"          // declares the ORIGINAL function

   int main() {
     // Test extents: pick them from the function's own real extents
     // (rank, aspect ratio, halo width). The values below are only a
     // placeholder for a 3-D structured field.
     const int nx = 100, ny = 100, nz = 50;
     // One input, shared by both paths. Two outputs, named for their role
     // in the comparison — the harness takes (candidate, reference).
     std::vector<double> input((size_t)nx*ny*nz);
     std::vector<double> output_reference((size_t)nx*ny*nz);
     for (size_t n = 0; n < input.size(); ++n)
       input[n] = /* realistic value */;

     // Call the ORIGINAL function — it produces the reference
     compute_kernel(input.data(), output_reference.data(), nx, ny, nz);

     // Smoke fingerprint only — NOT a correctness gate. Never gate on a
     // bare sum: opposite-sign errors cancel and a near-zero mean makes
     // the sum ill-conditioned (see references/verification-guide.md).
     double mean_abs = 0.0, peak = 0.0;
     for (double v : output_reference) { mean_abs += std::fabs(v);
                               peak = std::fmax(peak, std::fabs(v)); }
     mean_abs /= (double)output_reference.size();
     std::printf("Output mean|v| %g  peak|v| %g\n", mean_abs, peak);

   #ifdef USE_CUDA
     // GPU comparison block — EMPTY for now. Phase 4d fills it in.
   #endif

     std::printf("TEST PASSED\n");
     return 0;
   }
   ```

   Rules for the test:
   - Use **realistic** input sizes/values for **unit** testing — fast,
     deterministic. Do **not** copy grid sizes, timestep counts, or
     case parameters from the active `config.json:problems[]` entry.
   - Call the **guard-free CPU reference symbol** chosen above (not the
     translated one yet, and never a dispatcher).
   - The `#ifdef USE_CUDA` block must be present but empty.
   - Must exit 0 and print a recognizable success line.
   - Must not copy source files — reference them in-tree via include
     paths.
   - When the GPU block is filled in (Phase 4d), this file `#include`s
     the translated `.hpp` (plain C++), so it can be a `.cpp`. The
     test executable links against the nvcc-compiled `.cu`.

4. **Hook into the build system.** Add a test target that compiles the
   original `.cpp` (in-tree) and the test driver, links the CUDA
   runtime, and is built when the normal build command runs.

   For **CMake** (the common CUDA case):
   ```cmake
   # In .../test/test_compute_kernel/CMakeLists.txt
   add_executable(test_compute_kernel
       test_compute_kernel.cpp
       ${CMAKE_SOURCE_DIR}/src/physics/solver.cpp)
   target_include_directories(test_compute_kernel PRIVATE
       ${CMAKE_SOURCE_DIR}/src/physics
       ${CMAKE_SOURCE_DIR}/src/physics/cuda
       ${CMAKE_SOURCE_DIR}/cuda/test)   # header-only comparison harness
   target_compile_features(test_compute_kernel PRIVATE cxx_std_17)
   # Phase 4d adds: link the translated .cu target + CUDA::cudart,
   # and target_compile_definitions(... PRIVATE USE_CUDA)
   ```
   Add `add_subdirectory(test/test_compute_kernel)` so a normal
   `cmake --build .` builds it.

   For **Make**: add a target compiling the test driver + original
   `.cpp` with the host compiler, the `.cu` with `nvcc`
   (`-std=c++17`, detected `-gencode`), `-DUSE_CUDA` on every unit
   that carries the guard, and linking `-lcudart`. Reference the
   original `.cpp` in-tree (do NOT copy it).

   The test target must compile under the GPU build path (with
   `-DUSE_CUDA`) even though the `#ifdef` block is empty.

5. **Create placeholder translated files** in the same
   `{parent}/cuda/` as the original — a plain header + a `.cu`:
   ```cpp
   // src/physics/cuda/compute_kernel_cuda.hpp  (PLAIN C++ — no CUDA syntax)
   #pragma once
   void compute_kernel_cuda(const double* input, double* output,
                            int nx, int ny, int nz);
   ```
   ```cpp
   // src/physics/cuda/compute_kernel_cuda.cu   (nvcc)
   #include "compute_kernel_cuda.hpp"
   #include <cuda_runtime.h>
   void compute_kernel_cuda(const double* input, double* output,
                            int nx, int ny, int nz) {
     // TODO: kernels + launch added in Phase 4c
   }
   ```
   Name convention: `{parent}/cuda/{function}_cuda.{cu,hpp}`.

6. **Write manifest:**
   ```json
   {"test_executable": "/abs/path/to/test_compute_kernel"}
   ```

7. **Verify:**
   ```bash
   source .nsight-ai-port/env.sh && \
     USE_CUDA=1 bash .nsight-ai-port/build_correctness.sh
   ./test_{function}
   ```
   Both must pass.

8. **Git commit:**
   ```bash
   git add -A && git commit -m "scaffold: test harness for compute_kernel"
   ```

9. **Update progress.json:** Set this function's `phases.scaffold:
   true` and record `test_dir` and `test_executable`.

## Phase 4c: Add CUDA Kernels

See [patterns.md](patterns.md) for kernel syntax, indexing, error
checking, memory-mode policy, and common pitfalls.

Work on this function's dedicated files in `{parent}/cuda/` (e.g.
`src/physics/cuda/solver_cuda.cu`). Each function has its own
translated file pair — do **not** combine functions or modify the
original source file.

### Execution marker (mandatory)

Open `{function}_cuda` with a one-time marker, so Phase 4d can prove
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

Read `config.json:memory_mode` before allocating memory. Follow the
memory-mode policy in patterns.md:

| Mode | Allocation | Copies |
|------|-----------|--------|
| `managed` | `cudaMallocManaged` (or caller's managed ptr) | none — `cudaDeviceSynchronize` only |
| `unified` | none (use host ptr) | none |
| `separate` | `cudaMalloc` device buffers | `cudaMemcpy`, **hoisted** only |

In `managed`/`unified` mode there must be **zero** `cudaMemcpy` /
`cudaMalloc` calls in the translated kernel.

### No `#ifdef` in translated files

Files in `{parent}/cuda/` are pure GPU code compiled by nvcc only on
the GPU path. Write CUDA constructs **directly**. Do **not** wrap them
in `#ifdef USE_CUDA` — that guard belongs at the **call site** (Phase
4e), not inside `{parent}/cuda/` files.

### Public header stays plain C++

Keep `{function}_cuda.hpp` free of CUDA syntax (no `__global__`,
`<<<>>>`, `dim3`, `cudaError_t`). It is included by the host-compiled
original `.cpp`. All CUDA syntax lives in the `.cu` (and `.cuh`).

### Decoupled GPU entry points

The translated wrapper should take **flat arguments only**: raw
pointers + integer extents (and lower bounds if non-zero), not
device-unsafe host class/container references. The wrapper allocates /
wraps device memory internally per `memory_mode`; the host call site
passes `.data()` / raw pointers.

```cpp
// compute_kernel_cuda.hpp  (plain C++)
void compute_kernel_cuda(const double* input, double* output,
                         int nx, int ny, int nz);
```

If a target takes a non-device-safe class/struct, add a decoupling
refactor task (Phase 2) before translating (see refactoring.md).

### Per-chunk workflow

1. **Identify the target construct** from the plan's
   `accelerator_chunks` list (one kernel or a group of related
   kernels per chunk).

2. **Use device utilities.** If a kernel calls helpers from Phase 3,
   `#include "device_utils.cuh"` and call the device version directly —
   no `#ifdef` needed. If a needed helper is missing, add it to the
   appropriate `{parent}/cuda/device_utils.cuh` and re-run that
   directory's device-utils test before continuing.

3. **Write the kernel(s) + launch + wrapper** per memory mode, with
   bounds checks and `CUDA_CHECK(cudaGetLastError())` after each
   launch (see patterns.md).

4. **Lint** (when `memory_mode` is not `separate`): there must be no
   `cudaMalloc`/`cudaMemcpy` in the translated file:
   ```bash
   grep -nE 'cudaMalloc[^M]|cudaMemcpy' \
     src/physics/cuda/compute_kernel_cuda.cu
   ```
   Or all sibling trees:
   ```bash
   find . -path '*/cuda/*_cuda.cu' -print0 | \
     xargs -0 grep -nE 'cudaMalloc[^M]|cudaMemcpy'
   ```
   Revise if matches appear in managed/unified mode. (`cudaMallocManaged`
   is allowed in managed mode — the pattern above excludes it.)

5. **Verify after each chunk:**
   ```bash
   source .nsight-ai-port/env.sh && \
     USE_CUDA=1 bash .nsight-ai-port/build_correctness.sh
   ./test_{function}
   ```
   Must pass before the next chunk.

6. **On failure** (see [Fault Tolerance](../../../SKILL.md#fault-tolerance)):
   - Read the build/run error.
   - Common issues: CUDA syntax leaked into the plain `.hpp`, missing
     bounds check, missing launch error check, wrong `cudaMemcpy`
     direction, host pointer passed to a kernel in `separate` mode,
     `cudaMemcpy` present in managed/unified mode, 32-bit index
     overflow, wrong `-gencode` arch, missing `cudaDeviceSynchronize`
     before host read.
   - If stuck, ask the user for help.

7. **Git commit after each successful chunk:**
   ```bash
   git commit -am "cuda: add kernels to compute_kernel (chunk N/M)"
   ```

## Phase 4d: Verify Numerical Correctness

The comparison harness ships with the skill as compilable source. Do not
re-derive it inline. Copy the C++ header **once for the whole port**, at
the source root rather than beside each original — see
[verification-guide.md](../../../references/verification-guide.md#one-copy-per-port-at-the-source-root):

```bash
mkdir -p "$NSIGHT_AI_PORT_SOURCE_DIR/cuda/test"
cp assets/test-harness/compare_field.hpp \
   "$NSIGHT_AI_PORT_SOURCE_DIR/cuda/test/"
```

It defines namespace `nsight_ai_port`, whose contract is:

| Call | What it does |
|------|--------------|
| `tolerances_for<T>()` | element-type defaults, then `WMAD_TOL` / `ABS_TOL` / `REL_TOL` / `REL_FLOOR` / `MAX_REL_CHECK` from the environment |
| `compare_field(candidate, reference, n, tol [, weights])` | templated on the element type; `weights` is the cell volume or area on a non-uniform mesh, `nullptr` on a uniform one |
| `report_and_gate(result, tol, label)` | prints `wmad`, `max_rel`, `n_exceed` of `n_total`, and the worst flat index; returns `true` on pass |

Fill in the empty `#ifdef USE_CUDA` block in the test with the call:

```cpp
#ifdef USE_CUDA
  #include "compare_field.hpp"
  #include "cuda/compute_kernel_cuda.hpp"   // plain C++ header
  // ...
  // Identical inputs to both paths: `input` is passed unchanged, so the
  // reference call above must not have modified it. If it does, keep a
  // pristine copy before that call and pass the copy here.
  std::vector<double> output_candidate(output_reference.size(), 0.0);
  compute_kernel_cuda(input.data(), output_candidate.data(), nx, ny, nz);

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
  USE_CUDA=1 bash .nsight-ai-port/build_correctness.sh
compute-sanitizer ./test_compute_kernel
```
(If `compute-sanitizer` is not available, run the test directly.)

The TRANSLATED path MUST actually execute on the device — at least one
kernel must launch. If `compute-sanitizer` reports no kernels launched
/ "terminated before first instrumented API call", the GPU path was
compiled out or never called — that is a **build/wiring bug** (usually
the test built without `-DUSE_CUDA`, so the `#ifdef` block was
preprocessed away, or the `.cu` was not linked), not a numerical one.
Fix the wiring before touching the math.

Must pass. If numerical differences exceed tolerance:
  - Check index flattening / row-major mapping (a transpose shows up
    as large, structured diffs).
  - If `memory_mode` is `separate`: check `cudaMemcpy` completeness and
    direction (missing copy-in/out, wrong `kind`).
  - If `memory_mode` is `managed`/`unified`: check for a stray
    `cudaMemcpy`, or a missing `cudaDeviceSynchronize` before the host
    reads results.
  - Check reduction init values / atomic correctness.
  - Confirm `-use_fast_math` is NOT enabled during verification.
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

### Same-language: no cross-language shim, but mind the host header

Source and target are both C++, so no cross-language shim is required.
Place the translated `{function}_cuda.{cu,hpp}` in `{parent}/cuda/`
and `#include` the **plain** header at the call site under the guard.
The original `.cpp` is compiled by the host compiler — it must never
see CUDA syntax, only the `.hpp` declaration.

### Call-site dispatch

At the call site in the original code (not the translated file):

```cpp
#ifdef USE_CUDA
#include "cuda/compute_kernel_cuda.hpp"
#endif

  // ... existing code ...

#ifdef USE_CUDA
  compute_kernel_cuda(input, output, nx, ny, nz);
#else
  compute_kernel(input, output, nx, ny, nz);
#endif
```

### Build system updates

1. **Preprocessor define:** Add `-DUSE_CUDA` (from
   `config.json:include_guard`) to the project sources that carry the
   guard. For CMake, a `-DUSE_CUDA=ON` option does NOT become a C++
   define automatically:
   ```cmake
   option(USE_CUDA "Enable CUDA GPU acceleration" OFF)
   if(USE_CUDA)
     enable_language(CUDA)
     add_compile_definitions(USE_CUDA)   # so #ifdef USE_CUDA works
   endif()
   ```
   Or per-target: `target_compile_definitions(<target> PRIVATE
   USE_CUDA)`. Apply the SAME `-DUSE_CUDA` to the project, in-tree
   dependencies, AND the standalone test.

2. **Compile `.cu` with nvcc + link the CUDA runtime:** Add the
   translated `{parent}/cuda/*.cu` to the build, compiled by `nvcc`
   (or treated as CUDA-language sources by CMake) with `-std=c++17`
   and the detected `-gencode` arch. Link `cudart`
   (`-lcudart` / `CUDA::cudart`). Detect the compute capability rather
   than hardcoding it (see build-integration.md).
   **Never** apply CUDA flags / `nvcc` to external/third-party
   libraries.

3. **Archive/link of new subdir:** Ensure every `{parent}/cuda/*.o`
   (or CMake target object) is included in the library/link step.
   Globs like `src/*.cpp` do not match nested `src/*/cuda/*.cu`.

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
# 1. CPU build (no USE_CUDA)
source .nsight-ai-port/env.sh && bash .nsight-ai-port/build_correctness.sh

# 2. CPU run
source .nsight-ai-port/env.sh && bash .nsight-ai-port/run_correctness.sh

# 3. GPU build
source .nsight-ai-port/env.sh && \
  USE_CUDA=1 bash .nsight-ai-port/build_correctness.sh

# 4. GPU run — check for execution marker in output
source .nsight-ai-port/env.sh && \
  USE_CUDA=1 bash .nsight-ai-port/run_correctness.sh

# 5. Standalone test (sanitized)
compute-sanitizer ./test_{function}
```

Step 4: check stdout for `NSIGHT_AI_PORT_FRAMEWORK_EXEC:
compute_kernel`. If missing, the GPU path did not execute —
investigate (see verification guide; for CUDA the usual causes are
`-DUSE_CUDA` not reaching the file, or the `.cu` not linked).

### Commit and update progress

```bash
git commit -am "integrate: compute_kernel with CUDA dispatch"
```

Update `progress.json`:
- Set function's `phases.integrate: true`
- Set function's `status: "completed"`
- Set function's `commit_hash` to `git rev-parse HEAD`
- Add `"file:function"` to `translated_functions` array
- Update `last_working_commit` to the same SHA

## Resuming After Interruption

If a session is interrupted, check `.nsight-ai-port/progress.json`:

1. Read `progress.json` and `porting_plan.json`.
2. Find the first function with `status != "completed"`.
3. Check which phases are done for that function.
4. Resume from the next incomplete phase.
5. Verify the codebase is in a consistent state:
   ```bash
   source .nsight-ai-port/env.sh && bash .nsight-ai-port/build_correctness.sh
   ```
