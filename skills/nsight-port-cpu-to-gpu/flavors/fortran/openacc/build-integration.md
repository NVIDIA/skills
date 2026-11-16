# Build Integration — Best Practices (Fortran + OpenACC)

General guidance for embedding GPU paths into **existing** build
systems. Adapt to CMake, Make, or scripts — do not assume one layout.

## Principles

1. **Dual path** — CPU build unchanged; GPU path behind include guard
   (e.g. `USE_OPENACC`).
2. **Scope accelerator flags** — apply `-acc`, `-gpu=…`, `-DUSE_OPENACC`
   only to **project** sources, not vendored/third-party trees.
3. **Same compiler family** — CPU and GPU Fortran should use the same
   vendor (e.g. nvfortran for both) when the project already uses it.
4. **The build system owns compiler choice** — pin compilers in the
   Makefile/CMakeLists with `:=`, and let exactly one environment
   variable (`USE_OPENACC`) select CPU vs GPU.
5. **Per-configuration object trees** — CPU and GPU objects (and
   `.mod` files) go to separate directories, so incremental builds stay
   correct and unconditional cleaning is never needed.
6. **New subdirs** — if translated sources live in new directories,
   update archive/link rules (globs often miss subdirs).

## Makefile Patterns

`USE_OPENACC` is the **single** environment switch; the Makefile owns
compiler paths and flags. Pin them with `:=` so a stale ambient `FC`
cannot redirect the build, and give each configuration its own object and
module tree. The precedence table, the per-configuration tree rule, and
the "never clean unconditionally" policy are in
[verification-guide.md](../../../references/verification-guide.md#the-build-system-owns-compiler-choice)
— read them once; they apply to every flavor.

```makefile
# The ONE variable that must come from the environment. `?=` on purpose.
USE_OPENACC ?= 0

# Owned by the Makefile — `:=`, never `?=`. Same vendor on both paths.
FC     := nvfortran

ifeq ($(USE_OPENACC),1)
  CONFIG := gpu
  # Detected on the GPU path only. A CPU build must still succeed on a host
  # where nvidia-smi is unavailable.
  GPU_CC := $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader \
              2>/dev/null | head -1 | tr -d '.')
  ifeq ($(GPU_CC),)
    $(error Could not detect compute capability; pass GPU_CC=NN explicitly)
  endif
  FFLAGS := -O2 -Mpreprocess -acc -gpu=cc$(GPU_CC) -DUSE_OPENACC
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

- `-Mpreprocess` stays on **both** paths: the CPU build must still
  process the `#ifdef` at the dispatch site.
- Export variables to sub-makes when the tree uses recursive Make.
- When you must branch on the compiler in use, match with `findstring`
  rather than a brittle exact-match `ifeq ($(FC),…)`.
- After the first build of each configuration, confirm which compiler ran
  and that `-DUSE_OPENACC` reached it (`make -n | grep`). Exit code 0 is
  not proof.

## CMake Patterns

- `target_compile_definitions` / `target_compile_options` on project
  targets only.
- Prefer `cmake --build .` without pinning a single target — agents may
  add test targets.

## Correctness and performance build scripts

- Apply this wiring to **both** build scripts. The performance build may
  differ from the correctness build (e.g. grid size baked in at compile
  time), so keep them as separate files even if identical today.
- Source `.nsight-ai-port/env.sh` for static paths (SDK root, library
  dirs). Compiler *choice* belongs in the Makefile, not in `env.sh`.
- Support `USE_OPENACC=1` (or project guard) as the GPU toggle — this is
  the one switch the environment supplies.
- Use separate build directories per configuration
  (`$NSIGHT_AI_PORT_BUILD_DIR/cpu`, `$NSIGHT_AI_PORT_BUILD_DIR/gpu`); with
  CMake pass them via `-B`, never sharing one binary dir.
- Do not add `make clean`. With separate trees it buys nothing and costs a
  full rebuild every round — see
  [never clean unconditionally](../../../references/verification-guide.md#never-clean-unconditionally).
- Document required env vars, and the **ambient** `FC`/`CC` values you
  found, in `.nsight-ai-port/build_notes.txt`.

## Verification Hooks

- CPU path: default build + run (existing behavior).
- GPU path: same scripts with guard enabled + standalone tests under
  test dirs created by translate subskill.

## NVTX (Nsys Orchestrator)

- Add NVTX wrapper module to the link line; discover library path
  (see `flavors/fortran/nvtx.md`) — do not hardcode SDK paths.
- NVTX build should work on **CPU** path first (no OpenACC required).

## Anti-Patterns

- Passing `-acc` to external library builds.
- Renaming all `.f90` → `.F90` project-wide — prefer `-Mpreprocess`
  on CPU flags where `#ifdef` appears in sources (see `translation.md`).
- `?=` for compiler variables, which lets a stale ambient `FC` silently
  take over the build.
- CPU and GPU builds sharing one object/`.mod` tree or one CMake binary
  dir.
- Rebuilding correctness by prefixing a long list of variables on each
  command instead of writing the decision into the build files.
- Reporting a build as successful on exit code alone, without confirming
  the guard reached the compiler.
- Nested wrapper scripts the agent cannot easily debug.
