# Build Integration — Best Practices (Fortran + OpenMP offload)

General guidance for embedding GPU paths into **existing** build
systems. Adapt to CMake, Make, or scripts — do not assume one layout.

This is **OpenMP target offload**. The enabling flag is `-mp=gpu`
(nvfortran) — **never `-fopenmp`**, which on nvfortran only enables host
multithreading.

## Principles

1. **Dual path** — CPU build unchanged; GPU path behind include guard
   (e.g. `USE_OPENMP`).
2. **Scope offload flags** — apply `-mp=gpu`, `-gpu=…`, `-DUSE_OPENMP`
   only to **project** sources, not vendored/third-party trees.
3. **Same compiler family** — CPU and GPU Fortran should use the same
   vendor (e.g. nvfortran for both) when the project already uses it.
4. **The build system owns compiler choice** — pin compilers in the
   Makefile/CMakeLists with `:=`, and let exactly one environment
   variable (`USE_OPENMP`) select CPU vs GPU.
5. **Per-configuration object trees** — CPU and GPU objects (and `.mod`
   files) go to separate directories, so incremental builds stay correct
   and unconditional cleaning is never needed.
6. **New subdirs** — translated sources live in new `{parent}/openmp/`
   directories; update archive/link rules (globs often miss subdirs).

## Compiler Flags

### nvfortran (recommended)

| Purpose | Flag |
|---------|------|
| OpenMP GPU offload | `-mp=gpu` |
| Target architecture | `-gpu=ccNN` (e.g. `cc80`, `cc90`) |
| Managed memory | `-gpu=ccNN,mem:managed` |
| Unified memory | `-gpu=ccNN,mem:unified` |
| Free-format | `-Mfree` |
| Preprocess | `-Mpreprocess` |
| Long lines | `-Mextend` |

**Never** use `-fopenmp` with nvfortran for offload — it enables host
multithreading only.

### gfortran (fallback)

| Purpose | Flag |
|---------|------|
| OpenMP + GPU offload | `-fopenmp -foffload=nvptx-none` |
| Preprocess | `-cpp` |

gfortran offload maturity lags nvfortran for HPC codes; prefer
nvfortran when available.

## Build notes to capture (`.nsight-ai-port/build_notes.txt`)

- Exact GPU compiler + flags per `memory_mode`
- The include guard symbol (default `USE_OPENMP`)
- How each `{parent}/openmp/` subdir is compiled and archived
- Whether the project uses configure-time `#ifdef` scanning (makedep,
  autotools) that may break on module-level guards
- Whether `!$omp requires unified_shared_memory` is needed (unified /
  managed modes)
- The **ambient** `FC`/`CC` values inherited from the shell (or "unset"),
  and who owns compiler choice (Makefile `:=` / CMake / wrapper script)
- The CPU and GPU build tree paths — they must differ

## Makefile Patterns

`USE_OPENMP` is the **single** environment switch; the Makefile owns
compiler paths and flags. Pin them with `:=` so a stale ambient `FC`
cannot redirect the build, and give each configuration its own object and
module tree. The precedence table, the per-configuration tree rule, and
the "never clean unconditionally" policy are in
[verification-guide.md](../../../references/verification-guide.md#the-build-system-owns-compiler-choice)
— read them once; they apply to every flavor.

```makefile
# The ONE variable that must come from the environment. `?=` on purpose.
USE_OPENMP ?= 0

# Owned by the Makefile — `:=`, never `?=`. Same vendor on both paths.
FC     := nvfortran

ifeq ($(USE_OPENMP),1)
  CONFIG := gpu
  # Detected on the GPU path only. A CPU build must still succeed on a host
  # where nvidia-smi is unavailable.
  GPU_CC := $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader \
              2>/dev/null | head -1 | tr -d '.')
  ifeq ($(GPU_CC),)
    $(error Could not detect compute capability; pass GPU_CC=NN explicitly)
  endif
  FFLAGS := -O2 -Mpreprocess -mp=gpu -gpu=cc$(GPU_CC) -DUSE_OPENMP
else
  CONFIG := cpu
  FFLAGS := -O2 -Mpreprocess
endif

# Separate object AND module dirs: stale .mod files are the classic
# reason a Fortran incremental build goes wrong.
OBJDIR := $(NSIGHT_AI_PORT_BUILD_DIR)/$(CONFIG)/obj
MODDIR := $(NSIGHT_AI_PORT_BUILD_DIR)/$(CONFIG)/mod

$(OBJDIR)/%.o: %.f90
	@mkdir -p $(dir $@) $(MODDIR)
	$(FC) $(FFLAGS) -module $(MODDIR) -I$(MODDIR) -c $< -o $@
```

- Append `,mem:managed` / `,mem:unified` to `-gpu=` per
  `config.json:memory_mode` (see [Memory mode](#memory-mode) below).
- `-Mpreprocess` (or `-cpp`) stays on **both** paths: the CPU build must
  still process the `#ifdef` at the dispatch site.
- Export variables to sub-makes when the tree uses recursive Make.
- When you must branch on the compiler in use, match with `findstring`
  rather than a brittle exact-match `ifeq ($(FC),…)`:
  ```makefile
  ifneq (,$(findstring nvfortran,$(FC)))
    FFLAGS += -mp=gpu -gpu=cc$(GPU_CC)
  endif
  ```
- After the first build of each configuration, confirm which compiler ran
  and that `-DUSE_OPENMP` reached it (`make -n | grep`). Exit code 0 is
  not proof.

## CMake Patterns

- Use `find_package(OpenMP)` and link `OpenMP::OpenMP_Fortran`, but add
  the **offload** flags explicitly — `find_package` alone enables host
  OpenMP, not device offload:
  ```cmake
  target_compile_options(myproj PRIVATE
    $<$<COMPILE_LANGUAGE:Fortran>:-mp=gpu;-gpu=cc80>)
  target_link_options(myproj PRIVATE -mp=gpu -gpu=cc80)
  target_compile_definitions(myproj PRIVATE USE_OPENMP)
  ```
- Apply options to **project** targets only, never to imported/vendored
  targets.
- Prefer `cmake --build .` without pinning a single target — agents may
  add test targets.

## Correctness and performance build scripts

- Apply this wiring to **both** build scripts. The performance build may
  differ from the correctness build (e.g. grid size baked in at compile
  time), so keep them as separate files even if identical today.
- Source `.nsight-ai-port/env.sh` for static paths (SDK root, library
  dirs). Compiler *choice* belongs in the Makefile, not in `env.sh`.
- Support `USE_OPENMP=1` (or project guard) as the GPU toggle — this is
  the one switch the environment supplies.
- Use separate build directories per configuration
  (`$NSIGHT_AI_PORT_BUILD_DIR/cpu`, `$NSIGHT_AI_PORT_BUILD_DIR/gpu`); with
  CMake pass them via `-B`, never sharing one binary dir.
- Do not add `make clean`. With separate trees it buys nothing and costs a
  full rebuild every round — see
  [never clean unconditionally](../../../references/verification-guide.md#never-clean-unconditionally).
- Document required env vars, and the **ambient** `FC`/`CC` values you
  found, in `.nsight-ai-port/build_notes.txt`.

## Memory mode

- **separate**: `-mp=gpu -gpu=ccNN`. Data residency managed by hoisted
  `!$omp target data` regions in source.
- **managed**: `-mp=gpu -gpu=ccNN,mem:managed` + `!$omp requires
  unified_shared_memory` in offloaded modules.
- **unified**: `-mp=gpu -gpu=ccNN,mem:unified` + `!$omp requires
  unified_shared_memory`. Assumes coherent memory (e.g. GH200).

When `!$omp requires unified_shared_memory` is present, every
translation unit that touches device data should compile with the
matching memory flag, or the requires directive will conflict.

## Verification Hooks

- CPU path: default build + run (existing behavior).
- GPU path: same scripts with guard enabled + standalone tests under
  test dirs created by translate subskill.

## NVTX (Nsys Orchestrator)

- Add NVTX wrapper module to the link line; discover library path
  (see `flavors/fortran/nvtx.md`) — do not hardcode SDK paths.
- NVTX build should work on **CPU** path first (no offload required).

## Anti-Patterns

- Using `-fopenmp` with nvfortran for GPU offload (use `-mp=gpu`).
- Passing `-mp=gpu` / `-gpu=…` to external library builds.
- Renaming all `.f90` → `.F90` project-wide — prefer `-Mpreprocess`
  on CPU flags where `#ifdef` appears in sources (see `translation.md`).
- Mixing memory-mode flags (`mem:managed`) with explicit `map`
  clauses — pick one strategy per `memory_mode`.
- `?=` for compiler variables, which lets a stale ambient `FC` silently
  take over the build.
- CPU and GPU builds sharing one object/`.mod` tree or one CMake binary
  dir.
- Rebuilding correctness by prefixing a long list of variables on each
  command instead of writing the decision into the build files.
- Reporting a build as successful on exit code alone, without confirming
  the guard reached the compiler.
- Nested wrapper scripts the agent cannot easily debug.
