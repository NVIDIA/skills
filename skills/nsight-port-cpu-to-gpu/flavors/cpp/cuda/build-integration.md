# Build Integration — Best Practices (C/C++ + CUDA)

General guidance for embedding a CUDA GPU path into **existing** build
systems. Adapt to CMake, Make, or scripts — do not assume one layout.

## Principles

1. **Dual path** — CPU build unchanged; CUDA path behind the include
   guard (`USE_CUDA`).
2. **Scope accelerator flags** — apply `nvcc`, CUDA include paths,
   `-gencode`, and `-DUSE_CUDA` only to **project** sources, not
   vendored/third-party trees.
3. **Host/device split** — `.cu` files are compiled by `nvcc`; the
   plain `.hpp` they expose is compiled by the host compiler. Keep
   CUDA syntax out of host-compiled translation units.
4. **The build system owns compiler choice** — pin compilers in the
   Makefile/CMakeLists with `:=`, and let exactly one environment
   variable (`USE_CUDA`) select CPU vs GPU.
5. **Per-configuration object trees** — CPU and GPU objects go to
   separate directories, so incremental builds stay correct and
   unconditional cleaning is never needed.
6. **New subdirs** — translated `.cu` sources live in new
   `{parent}/cuda/` directories; update archive/link/object rules
   (globs often miss subdirs and the `.cu` extension).

## Build Notes to Capture (Phase 4a)

Record in `.nsight-ai-port/build_notes.txt`:

```text
INCLUDE_GUARD: USE_CUDA (or project-specific symbol if one exists)
NVCC: <path to nvcc>, version=<from nvcc --version>
CUDA_HOME: <toolkit root>, libdir=<lib64 with libcudart.so>
HOST_CXX: <host compiler used for .cpp>
CXX_STD: c++17
GPU_ARCH: <compute_cap from nvidia-smi, e.g. 90 → sm_90>  (DETECT)
BUILD_SYSTEM: cmake | make, how .cu objects are compiled/linked, externals handling
AMBIENT_CC / AMBIENT_CXX / AMBIENT_FC / AMBIENT_NVCC:
  <values inherited from the shell, or "unset">
COMPILER_OWNER: makefile (`:=`) | cmake | wrapper script
BUILD_TREES: <cpu tree>, <gpu tree>   (must be different paths)
```

Capture the **ambient** compiler variables even when the Makefile pins
its own with `:=`. If a build later behaves differently on another
machine, the inherited environment is the first thing to compare, and it
is unrecoverable after the fact.

### Detecting the GPU architecture

Do NOT hardcode `sm_70`/`sm_80`. Detect and generate the matching
`-gencode`:

```bash
GENCODE=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null \
          | head -1 | tr -d '.')
: "${GENCODE:?could not detect compute capability — set GENCODE=NN}"
NVCCFLAGS="-O2 -std=c++17 \
  -gencode arch=compute_${GENCODE},code=sm_${GENCODE} \
  -gencode arch=compute_${GENCODE},code=compute_${GENCODE}"
uname -m   # x86_64 vs aarch64 (Grace Hopper → sm_90)
```

## CMake Patterns (preferred)

CMake has first-class CUDA support — prefer it over hand-rolled nvcc
rules.

```cmake
option(USE_CUDA "Enable CUDA GPU acceleration" OFF)

if(USE_CUDA)
  enable_language(CUDA)                 # treat .cu as CUDA sources
  find_package(CUDAToolkit REQUIRED)    # CUDA::cudart target
  add_compile_definitions(USE_CUDA)     # so #ifdef USE_CUDA works
  # Detect arch (CMake >= 3.24):
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()

if(USE_CUDA)
  target_sources(myproj PRIVATE src/physics/cuda/compute_kernel_cuda.cu)
  target_include_directories(myproj PRIVATE src/physics/cuda)
  target_link_libraries(myproj PRIVATE CUDA::cudart)
  set_target_properties(myproj PROPERTIES CUDA_STANDARD 17)
endif()
```

Notes:
- `-DUSE_CUDA=ON` (the CMake option) does NOT define the C++ macro —
  `add_compile_definitions(USE_CUDA)` (or `target_compile_definitions`)
  is required, or `#ifdef USE_CUDA` is silently skipped.
- `CMAKE_CUDA_ARCHITECTURES native` (3.24+) auto-detects; otherwise set
  it from the detected compute capability.
- Prefer `cmake --build .` without pinning a single target — the
  scaffold phase adds test targets.
- When the source structure changes, delete `CMakeCache.txt` before
  reconfiguring.
- Link `CUDA::cudart` only on targets that need it; never on external
  libraries.

## Make Patterns

### One environment switch; the Makefile owns everything else

The intended UX is a **single** environment variable — `USE_CUDA` — that
selects CPU or GPU. Compiler paths, flags, and object locations belong
**in the Makefile**, where they are version-controlled and reviewable,
not reassembled on the command line on every invocation.

That makes the assignment operator load-bearing:

```makefile
# The ONE variable that must come from the environment. `?=` on purpose.
USE_CUDA ?= 0

# Everything else is owned by the Makefile, so use `:=`, never `?=`.
# `?=` yields to any ambient CC/CXX/NVCC left in the shell, silently
# building with a different compiler than the one you think you are
# testing.
ifeq ($(USE_CUDA),1)
  CONFIG    := gpu
  CXX       := nvc++
  NVCC      := nvcc
  CUDA_HOME := /usr/local/cuda
  CXXFLAGS  := -O2 -std=c++17 -DUSE_CUDA
  LDLIBS    := -L$(CUDA_HOME)/lib64 -lcudart
  # Detected on the GPU path only. A CPU build must still succeed on a host
  # where nvidia-smi is unavailable.
  GENCODE   := $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader \
                 2>/dev/null | head -1 | tr -d '.')
  ifeq ($(GENCODE),)
    $(error Could not detect compute capability; pass GENCODE=NN explicitly)
  endif
  NVCCFLAGS := -O2 -std=c++17 \
    -gencode arch=compute_$(GENCODE),code=sm_$(GENCODE) \
    -gencode arch=compute_$(GENCODE),code=compute_$(GENCODE) -DUSE_CUDA
else
  CONFIG    := cpu
  CXX       := g++
  CXXFLAGS  := -O2 -std=c++17
  LDLIBS    :=
endif
```

Precedence, as measured with GNU Make 4.x:

| Assignment in Makefile | vs ambient env var | vs `make VAR=…` | vs `make -e` |
|------------------------|--------------------|-----------------|--------------|
| `VAR ?= x` | **env wins** | cmdline wins | env wins |
| `VAR := x` | **Makefile wins** | cmdline wins | env wins |
| `override VAR := x` | Makefile wins | **Makefile wins** | **Makefile wins** |

`:=` is therefore the right default: it defends against a stale `CXX` in
the environment while still letting a human say `make CXX=clang++`. Use
`override :=` only when the project's own wrapper scripts pass compilers
on the make command line and the Makefile must win anyway. Never invoke
`make -e` — it promotes every ambient variable above the Makefile.

Prefer writing these decisions into the Makefile over passing a long
prefix of variables per command. A rule that lives in the source is
applied consistently; one that depends on remembering the right
decorators every time is not.

### Per-configuration object trees

Key the object tree on the switch, so configurations never collide:

```makefile
BUILD_ROOT ?= $(NSIGHT_AI_PORT_BUILD_DIR)
OBJDIR     := $(BUILD_ROOT)/$(CONFIG)/obj
BIN        := $(BUILD_ROOT)/$(CONFIG)/myproj

OBJS := $(patsubst %.cpp,$(OBJDIR)/%.o,$(SRCS)) \
        $(patsubst %.cu,$(OBJDIR)/%.o,$(CU_SRCS))

$(OBJDIR)/%.o: %.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(OBJDIR)/%.o: %.cu
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@
```

This is what makes `make clean` genuinely unnecessary rather than merely
discouraged. CPU and GPU objects never share a path, so alternating
`bash build_correctness.sh` and `USE_CUDA=1 bash build_correctness.sh`
cannot produce a half-CPU/half-GPU binary, and each configuration stays
incremental.

### Never clean *unconditionally*

The rule is never clean unconditionally — not "never clean". A blanket
`make clean` in a build script destroys incremental builds and makes
every round slow. But a build that quietly reuses objects compiled with
different flags is worse: it yields a binary matching neither
configuration, and the symptoms (identical CPU/GPU results, no kernels in
the profile) look like translation bugs.

Per-configuration trees remove nearly every reason to clean. When a full
rebuild really is needed it is for one of these, and you should say which:

- A semantics-changing `#define` reached only some translation units.
- Stale Fortran `.mod` files — module interface changes are a notorious
  blind spot in `make` dependency tracking.
- The compiler itself changed.
- CMake: the source structure changed and `CMakeCache.txt` still holds
  the old file list.

Rebuild only the affected configuration. For CMake, use a fresh
configuration (`cmake --fresh -S "$SOURCE_ROOT" -B "$BUILD_ROOT/gpu"`);
otherwise use the build system's configuration-scoped rebuild target. Validate
both resolved paths first, never use a broad glob, and record why the rebuild
was required.

### Verify the build did what you asked

Exit code 0 does not mean the build did what you intended. After the
first build of each configuration, confirm rather than assume:

```bash
# Which compiler actually ran, and did the guard reach it?
make -n 2>&1 | grep -E 'nvcc|nvc\+\+|g\+\+' | head
make -n 2>&1 | grep -c -- -DUSE_CUDA          # expect > 0 on the GPU path

# Did objects land in the configuration-specific tree?
find "$NSIGHT_AI_PORT_BUILD_DIR" -name '*.o' -newermt '-5 minutes' | head

# Is the CUDA runtime actually linked?
ldd <binary> | grep -i cudart
```

These are cheap, and they catch the case where a stale environment
variable or an unwired subdirectory kept the guard from reaching the
compiler — which otherwise surfaces much later as "the GPU path gives
identical results" or "`gpu_active_pct` is 0".

### Other Make notes

- Add `$(wildcard src/*/cuda/*.o)` to the archive/link explicitly —
  `src/*.o` does not match nested `cuda/` objects, and a `%.cpp` rule
  will not pick up `.cu` files.
- Link `-lcudart` (and `-lstdc++` when the final link is driven by a
  C/Fortran compiler).
- Export variables to sub-makes when the tree uses recursive Make.
- When you must branch on the compiler in use, match with `findstring`
  rather than a brittle exact-match `ifeq`.

## Host/Device Boundary Wiring

- The original `.cpp` includes only the **plain** `{function}_cuda.hpp`
  — never a `.cu`/`.cuh`. Confirm the host compiler never sees CUDA
  syntax (a "`__global__` undeclared" / "expected `;`" error from the
  host compiler means CUDA leaked into a host header).
- The wrapper in the `.cu` can be plain C++ linkage (same language).
  Use `extern "C"` only if the caller is C or Fortran.
- Mixing the final link: if the program is linked by the host compiler,
  add `-lcudart`; if linked by `nvcc`, it adds the runtime
  automatically.

## Correctness and performance build scripts

- Apply this wiring to **both** build scripts. The performance build may
  differ from the correctness build (e.g. grid size baked in at compile
  time), so keep them as separate files even if identical today.
- Source `.nsight-ai-port/env.sh` for static paths (`CUDA_HOME`, `NVCC`,
  library dirs).
- Read `USE_CUDA` from the environment (passed directly as
  `USE_CUDA=1 bash .nsight-ai-port/build_correctness.sh`, NOT
  via `source env.sh`) and translate it into the build-system toggle:
  ```bash
  CMAKE_DEFINES=""
  if [ "${USE_CUDA:-0}" = "1" ]; then
    CMAKE_DEFINES="-DUSE_CUDA=ON"
  fi
  cmake $CMAKE_DEFINES "$NSIGHT_AI_PORT_SOURCE_DIR"
  cmake --build .
  ```
- Give each configuration its own build directory —
  `$NSIGHT_AI_PORT_BUILD_DIR/cpu` and `$NSIGHT_AI_PORT_BUILD_DIR/gpu` — and
  pass it to CMake with `-B`. Two CMake configurations must never share a
  binary dir; the second one silently reuses the first one's cache.
- BYPASS EARLY-EXIT CACHING: if the script skips compilation when the
  binary already exists, force the FIRST GPU build but allow subsequent
  GPU builds to be incremental.
- Do not add `make clean` (set `AUTOCLEAN=false` if present). With
  separate trees per configuration it buys nothing and costs a full
  rebuild every round. See
  [Never clean *unconditionally*](#never-clean-unconditionally) for the
  cases that do warrant a rebuild.
- Build externals first WITHOUT CUDA flags / `nvcc`, then the core WITH
  them.
- Do NOT enable `-use_fast_math` for verification builds (it widens FP
  error and can fail the numerical compare).
- Document required env vars in `.nsight-ai-port/build_notes.txt`.

## Managed / Unified Memory

- `memory_mode == "managed"` uses `cudaMallocManaged` (UVM) — no
  special build flag beyond linking `cudart`. Works on any CUDA GPU
  because the allocator, not the hardware, provides coherence.
- With a **Fortran caller** in `managed` mode, `-DUSE_CUDA` must also
  reach the Fortran compiler (`-Mpreprocess` or a `.F90` extension), or
  the managed-allocation shim is compiled out and the device silently
  receives host memory. See
  [patterns.md](patterns.md#fortran-caller-the-allocation-shim-is-not-optional).

<a id="fortran-managed-memory-guard-gate"></a>
For that Fortran/managed combination, the following build gate is mandatory
before any correctness or performance test:

1. Enable verbose build output and capture in
   `.nsight-ai-port/build_notes.txt` the exact Fortran compile command for the
   allocation shim and every call site guarded by `USE_CUDA`.
2. Confirm each of those Fortran commands contains `-DUSE_CUDA` and enables
   preprocessing through a `.F90` input, `-Mpreprocess`, or the selected
   compiler's equivalent. A CMake `-DUSE_CUDA=ON` option, or a matching C++,
   CUDA, or linker command, is not evidence that the Fortran guard is active.
3. Inspect the preprocessed Fortran output produced with those same compiler
   settings. The GPU output must retain the `call alloc_managed_1d(...)`
   branch and must not retain its host `allocate(...)` fallback. Record the
   source path, command, and observed branch in `build_notes.txt`.

If any command or preprocessed branch cannot be proved, fail the GPU build and
stop before running tests; numerical agreement cannot waive this gate.

- `memory_mode == "unified"` (passing plain host pointers to kernels)
  requires a coherent platform (HMM / Grace Hopper). If the target is
  not coherent, record it and fall back to `managed` or `separate`
  rather than failing silently.
- `memory_mode == "separate"` needs only `cudart`.

## Verification Hooks

- CPU path: default build + run (existing behavior), no `USE_CUDA`.
- GPU path: same scripts with `USE_CUDA=1` + standalone tests under the
  `{parent}/cuda/test/` dirs created by the translate subskill.
- `compute-sanitizer ./test_{function}` for memory-error and
  no-kernel-launch detection.

## NVTX / Profiling (Nsys Orchestrator)

- Add NVTX ranges around the problem compute path (see
  `flavors/cpp/nvtx.md`).
  Prefer `libnvtx3interop` (CUDA 13+); discover the library path — do
  not hardcode SDK paths.
- NVTX build should work on the **CPU** path first where possible.

## Anti-Patterns

- Leaking CUDA syntax (`__global__`, `<<<>>>`, `dim3`, `cudaError_t`)
  into a header included by host-compiled `.cpp`.
- Passing `nvcc` / CUDA include paths / `-DUSE_CUDA` to external
  library builds.
- Hardcoding `sm_70`/`sm_80` or x86-only arch on aarch64 (Grace
  Hopper → `sm_90`).
- Defining the CMake option `USE_CUDA=ON` but forgetting
  `add_compile_definitions(USE_CUDA)` (the `#ifdef` is then silently
  skipped).
- `%.cpp`-only build rules that never compile the new `.cu` files.
- Adding `make clean` / wiping objects on every build.
- `?=` for compiler variables, which lets a stale ambient `CXX`/`NVCC`
  silently take over the build.
- CPU and GPU builds sharing one object tree or one CMake binary dir.
- Rebuilding correctness by prefixing a long list of variables on each
  command instead of writing the decision into the build files.
- Reporting a build as successful on exit code alone, without confirming
  the guard reached the compiler.
- `-use_fast_math` during numerical verification.
- Nested wrapper scripts the agent cannot easily debug.
