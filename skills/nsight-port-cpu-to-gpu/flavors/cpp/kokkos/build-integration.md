# Build Integration — Best Practices (C/C++ + Kokkos)

General guidance for embedding a Kokkos GPU path into **existing**
build systems. Adapt to CMake, Make, or scripts — do not assume one
layout. Target **Kokkos 4 + C++17**.

## Principles

1. **Dual path** — CPU build unchanged; Kokkos path behind the include
   guard (`USE_KOKKOS`).
2. **Scope accelerator flags** — apply Kokkos include paths,
   `nvcc_wrapper`/CUDA device flags, and `-DUSE_KOKKOS` only to
   **project** sources, not vendored/third-party trees.
3. **C++17 everywhere** — the project, in-tree dependencies, and the
   test must all compile at `-std=c++17` (Kokkos 4). Do not mix
   standards across translation units that share Kokkos headers.
4. **The build system owns compiler choice** — pin compilers in the
   Makefile/CMakeLists with `:=`, and let exactly one environment
   variable (`USE_KOKKOS`) select CPU vs GPU.
5. **Per-configuration object trees** — CPU and GPU objects go to
   separate directories, so incremental builds stay correct and
   unconditional cleaning is never needed.
6. **New subdirs** — translated sources live in new `{parent}/kokkos/`
   directories; update archive/link/object rules (globs often miss
   subdirs).

## Build Notes to Capture (Phase 4a)

Record in `.nsight-ai-port/build_notes.txt`:

```text
INCLUDE_GUARD: USE_KOKKOS (or project-specific symbol if one exists)
KOKKOS: home=<KOKKOS_HOME or CMake package>, version=<4.x>
COMPILER: nvcc_wrapper | nvc++ (which, and how selected)
CXX_STD: c++17
LAMBDA_FLAGS: --extended-lambda --expt-relaxed-constexpr (nvcc path)
GPU_ARCH: <Kokkos_ARCH_* / detected compute_cap from nvidia-smi>
SHARED_SPACE: available? (needed for memory_mode=managed)
BUILD_SYSTEM: cmake | make, object/archive rules, externals handling
```

### Detecting the Kokkos install and GPU arch

```bash
# Kokkos via CMake package (preferred):
#   find_package(Kokkos REQUIRED) — set Kokkos_ROOT / Kokkos_DIR
# Kokkos via raw paths: locate nvcc_wrapper to infer KOKKOS_HOME
command -v nvcc_wrapper && KOKKOS_HOME=$(dirname $(dirname $(command -v nvcc_wrapper)))

# Detect actual GPU compute capability — do NOT hardcode:
nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1
uname -m   # x86_64 vs aarch64 (Grace Hopper)
```

Map the detected compute capability to the Kokkos arch flag
(`Kokkos_ARCH_HOPPER90`, `Kokkos_ARCH_AMPERE80`, …) when configuring
Kokkos, or rely on an existing Kokkos install's baked-in arch. Never
copy a hardcoded `cc70/cc80` from sample makefiles onto an aarch64
machine.

## CMake Patterns (preferred for Kokkos)

```cmake
option(USE_KOKKOS "Enable Kokkos GPU acceleration" OFF)

if(USE_KOKKOS)
  find_package(Kokkos REQUIRED)        # 4.x
  add_compile_definitions(USE_KOKKOS)  # so #ifdef USE_KOKKOS works
endif()

# Translated sources, compiled only on the GPU path:
if(USE_KOKKOS)
  target_sources(myproj PRIVATE
      src/physics/kokkos/compute_kernel_kokkos.cpp)
  target_include_directories(myproj PRIVATE src/physics/kokkos)
  target_link_libraries(myproj PRIVATE Kokkos::kokkos)
  target_compile_features(myproj PRIVATE cxx_std_17)
endif()
```

Notes:
- `-DUSE_KOKKOS=ON` (the CMake option) does NOT define the C++ macro —
  you MUST `add_compile_definitions(USE_KOKKOS)` (or
  `target_compile_definitions`) explicitly.
- Prefer `cmake --build .` without pinning a single target — the
  scaffold phase adds test targets.
- When the source structure changes, delete `CMakeCache.txt` before
  reconfiguring.
- Link `Kokkos::kokkos` only on targets that use it; do not propagate
  it to external libraries.

## Make Patterns

`USE_KOKKOS` is the **single** environment switch; the Makefile owns
compiler paths and flags. Pin them with `:=` so a stale ambient `CXX`
cannot redirect the build, and give each configuration its own object
tree. The precedence table, the per-configuration tree rule, and the
"never clean unconditionally" policy are in
[verification-guide.md](../../../references/verification-guide.md#the-build-system-owns-compiler-choice)
— read them once; they apply to every flavor.

```makefile
# The ONE variable that must come from the environment. `?=` on purpose.
USE_KOKKOS ?= 0

# Owned by the Makefile — `:=`, never `?=`.
KOKKOS_HOME := /opt/kokkos
KOKKOS_INC  := -I$(KOKKOS_HOME)/include
KOKKOS_LIB  := -L$(KOKKOS_HOME)/lib -lkokkoscore -lkokkoscontainers \
               -Wl,-rpath,$(KOKKOS_HOME)/lib

ifeq ($(USE_KOKKOS),1)
  CONFIG   := gpu
  CXX      := $(KOKKOS_HOME)/bin/nvcc_wrapper
  CXXFLAGS := -O2 -std=c++17 --extended-lambda \
              --expt-relaxed-constexpr $(KOKKOS_INC) -DUSE_KOKKOS
  LDLIBS   := $(KOKKOS_LIB) -lcudart -lcuda
else
  CONFIG   := cpu
  CXX      := g++
  CXXFLAGS := -O2 -std=c++17
  LDLIBS   :=
endif

OBJDIR := $(NSIGHT_AI_PORT_BUILD_DIR)/$(CONFIG)/obj

$(OBJDIR)/%.o: %.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@
```

- Add `$(wildcard src/*/kokkos/*.o)` to the archive/link explicitly —
  `src/*.o` does not match nested `kokkos/` objects.
- Export variables to sub-makes when the tree uses recursive Make.
- Link the CUDA runtime (`-lcudart -lcuda`) alongside Kokkos when the
  install needs it (see the smoke-test Makefile pattern).
- When you must branch on the compiler in use, match with `findstring`
  rather than a brittle exact-match `ifeq`:
  ```makefile
  ifneq (,$(findstring nvcc_wrapper,$(CXX)))
    CXXFLAGS += --extended-lambda --expt-relaxed-constexpr
  endif
  ```
- After the first build of each configuration, confirm the compiler that
  actually ran and that `-DUSE_KOKKOS` reached it (`make -n | grep`).
  Exit code 0 is not proof.

## Correctness and performance build scripts

- Apply this wiring to **both** build scripts. The performance build may
  differ from the correctness build (e.g. grid size baked in at compile
  time), so keep them as separate files even if identical today.
- Source `.nsight-ai-port/env.sh` for static paths (`KOKKOS_HOME`,
  `CUDA_HOME`, `CXX=nvcc_wrapper`, library dirs).
- Read `USE_KOKKOS` from the environment (passed directly as
  `USE_KOKKOS=1 bash .nsight-ai-port/build_correctness.sh`, NOT
  via `source env.sh`) and translate it into the build-system toggle:
  ```bash
  CMAKE_DEFINES=""
  if [ "${USE_KOKKOS:-0}" = "1" ]; then
    CMAKE_DEFINES="-DUSE_KOKKOS=ON -DKokkos_ROOT=$KOKKOS_HOME"
  fi
  cmake $CMAKE_DEFINES -DCMAKE_CXX_COMPILER="$CXX" \
        -DCMAKE_CXX_STANDARD=17 "$NSIGHT_AI_PORT_SOURCE_DIR"
  cmake --build .
  ```
- Give each configuration its own build directory —
  `$NSIGHT_AI_PORT_BUILD_DIR/cpu` and `$NSIGHT_AI_PORT_BUILD_DIR/gpu` — and
  pass it to CMake with `-B`. Two CMake configurations must never share a
  binary dir; the second one silently reuses the first one's cache.
- BYPASS EARLY-EXIT CACHING: if the script skips compilation when the
  binary already exists, force the FIRST GPU build but allow
  subsequent GPU builds to be incremental.
- Do not add `make clean` (set `AUTOCLEAN=false` if present). With
  separate trees per configuration it buys nothing and costs a full
  rebuild every round — see
  [never clean unconditionally](../../../references/verification-guide.md#never-clean-unconditionally).
- Build externals first WITHOUT Kokkos/CUDA flags, then the core WITH
  them.
- Document required env vars in `.nsight-ai-port/build_notes.txt`.

## SharedSpace / Managed Memory (Kokkos 4)

`memory_mode == "managed"` requires a Kokkos 4 install with CUDA + UVM
support so `Kokkos::SharedSpace` is available. Confirm at setup:
- The Kokkos build has `Kokkos_ENABLE_CUDA=ON` and a UVM-capable
  allocator.
- A tiny probe (`Kokkos::View<int*, Kokkos::SharedSpace> v("v", 1);`)
  compiles and runs.

If `SharedSpace` is unavailable, record it in `.nsight-ai-port/build_notes.txt` and
fall back to `memory_mode=separate` (device View + mirror + hoisted
`deep_copy`) rather than forcing managed memory. Do NOT silently
switch modes — surface the deviation to the user.

## Verification Hooks

- CPU path: default build + run (existing behavior), no `USE_KOKKOS`.
- GPU path: same scripts with `USE_KOKKOS=1` + standalone tests under
  the `{parent}/kokkos/test/` dirs created by the translate subskill.

## NVTX / Profiling (Nsys Orchestrator)

- Kernel labels go to NVTX when the Kokkos-Tools NVTX connector
  is loaded — no manual NVTX wiring for those ranges. Follow this
  pack's [nvtx.md](nvtx.md) (naming grammar in
  `flavors/cpp/nvtx.md`).
- For `iter:` / `phase:` application ranges use
  `Kokkos::Profiling::pushRegion` / `popRegion` (or the RAII
  helper in `nvtx.md`). Do **not** link NVTX directly on that
  path — `Kokkos::Profiling` routes through the connector.
- Direct NVTX (`nvtx3` / `-lnvtx3interop`) is optional fallback
  only, if ranges must work without Kokkos Tools. See `nvtx.md`.
- Profiling build should work on the **CPU** path first where
  possible.

## Anti-Patterns

- Passing Kokkos include paths / CUDA device flags to external library
  builds.
- Compiling some translation units at `-std=c++17` and others at a
  different standard while sharing Kokkos headers.
- Using C++20-only syntax (Kokkos 4 targets C++17; Kokkos 5 needs
  C++20 — not targeted here).
- Hardcoding `cc70/cc80` or x86-only arch on aarch64 (Grace Hopper).
- Adding `make clean` / wiping objects on every build.
- `?=` for compiler variables, which lets a stale ambient `CXX` silently
  take over the build.
- CPU and GPU builds sharing one object tree or one CMake binary dir.
- Rebuilding correctness by prefixing a long list of variables on each
  command instead of writing the decision into the build files.
- Reporting a build as successful on exit code alone, without confirming
  the guard reached the compiler.
- Defining the CMake option `USE_KOKKOS=ON` but forgetting
  `add_compile_definitions(USE_KOKKOS)` (the `#ifdef` is then silently
  skipped).
- Nested wrapper scripts the agent cannot easily debug.
