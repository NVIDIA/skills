# NVTX Instrumentation — C/C++

Language docs: `flavors/cpp/`. Nsys Round 0 uses this for marker
placement on **C++ that actually runs** (`config.json:from_lang`).
Header-only `nvtx3` is the default how. If `framework` is `kokkos`,
also read [kokkos/nvtx.md](kokkos/nvtx.md) for Kokkos Tools /
`Kokkos::Profiling`.

CUDA kernel names appear in the nsys timeline automatically once
kernels exist; NVTX ranges add the named host-phase / function
structure needed to plan a port.

## Two sources of timeline structure

1. **Kernel names (automatic).** Every `__global__` kernel shows up in
   nsys under its symbol name — give kernels descriptive names so the
   timeline is readable (`compute_flux_kernel`, not `k1`).
2. **NVTX ranges (you add these).** Wrap host phases and functions in
   NVTX ranges using the naming grammar below so planning can map hot
   ranges to translation targets.

## NVTX in C++ — header-only `nvtx3` (preferred)

Modern CUDA ships the header-only **NVTX v3 C++ API**. It needs no
extra link library — just an include — and gives RAII scoped ranges:

```cpp
#include <nvtx3/nvtx3.hpp>

void timestep() {
  nvtx3::scoped_range r{"phase:compute:dynamics"};   // pop happens at scope exit
  // ... work ...
}
```

For a one-off named range use the macro form:
```cpp
NVTX3_FUNC_RANGE();                  // range named after the function
```

Prefer the header-only C++ API. Only fall back to the C API + explicit
link library if `nvtx3/nvtx3.hpp` is unavailable.

### Manual push/pop (C API fallback)

```cpp
#include <nvtx3/nvToolsExt.h>
nvtxRangePushA("iter:timestep");
// ... work ...
nvtxRangePop();
```

Always pair push/pop (treat like `{ }`). A thin RAII helper keeps them
balanced:

```cpp
struct NvtxRange {
  explicit NvtxRange(const char* name) { nvtxRangePushA(name); }
  ~NvtxRange() { nvtxRangePop(); }
};
#define ACC_RANGE(name) NvtxRange acc_range__##__LINE__(name)
```

### Linking (only if NOT using the header-only API)

CUDA 13+ / recent HPC SDK builds often **omit** `libnvToolsExt`.
Prefer **`libnvtx3interop`**; keep `-lnvToolsExt` as a legacy
fallback. Discover the library directory at setup time and **always
append `|| true`** to `-print-file-name` (it may exit non-zero under
`set -e`):

```bash
CXX_COMP="${CXX:-g++}"
NVTX_LINK=-lnvtx3interop
NVTX_LIB=$("$CXX_COMP" -print-file-name=libnvtx3interop.so 2>/dev/null || true)
if [ ! -f "$NVTX_LIB" ]; then
  NVCC_PATH=$(command -v nvcc || true)
  SDK_ROOT="${NVCC_PATH%/bin/nvcc}"
  NVTX_LIB=$(find "$SDK_ROOT" -name libnvtx3interop.so -type f 2>/dev/null | head -1)
fi
if [ ! -f "$NVTX_LIB" ]; then
  echo "libnvtx3interop.so not found; try -lnvToolsExt" >&2
  exit 1
fi
NVTX_LIBDIR="${NVTX_LIB%/*}"
# Link line: -L"$NVTX_LIBDIR" $NVTX_LINK -Wl,-rpath,"$NVTX_LIBDIR"
```

Record `$NVTX_LINK` and `$NVTX_LIBDIR` in
`.nsight-ai-port/build_notes.txt`. With the header-only `nvtx3` API you
usually need none of this. **Never** hardcode machine-specific
absolute SDK paths in a performance build script.

Run `scripts/profile-check.sh {to_lang}-{framework}` during setup —
it prints NVTX link hints. NVTX ranges work on the **CPU** path too,
so add them before any GPU translation.

## Mandatory Naming Grammar

Use **only** these prefixes for NVTX range names (and as the basis for
kernel names where practical). Names are stable before any porting plan
exists — do not use `port:` (that implied post-plan tagging).

| Prefix | Use | Example |
|--------|-----|---------|
| `iter:` | The repeating unit of work — **the** measurement window | `iter:timestep` |
| `phase:compute:` | Major section that looks like compute | `phase:compute:dynamics` |
| `phase:host:` | Likely host-only (MPI, I/O, init) — a hint, not a plan veto | `phase:host:mpi_halo`, `phase:host:io` |
| `function:` | A specific source function (`file:function`) | `function:src/kernels.cpp:kernel_a` |
| `block:` | Optional sub-region inside a function (e.g. one kernel) | `block:kernel_a_ij_loop` |

Rules:

1. **`function:`** — wrap every function you suspect is hot in an NVTX
   range using `file:function` matching source layout (same strings
   `porting_plan.json` will use later).
2. **`block:`** — optional; use when a function has several kernels and
   you want a per-kernel breakdown. Nest inside the matching
   `function:` range.
3. **`iter:`** — one push/pop per iteration of the outermost repeating
   unit of work. This is the range the profiler uses as its measurement
   window, so emit exactly **one** `iter:` name per program. If the
   program has no repeating unit (a single-shot pipeline), omit `iter:`
   entirely — the window then falls back to the `phase:compute:` span.
4. **`phase:compute:` vs `phase:host:`** — every phase MUST declare its
   class. `host` typically means MPI, I/O,
   allocation, init, teardown. The class still drives the default
   profiling window, so an unclassified or misclassified phase can
   corrupt the measurement: a host phase treated as compute can
   swallow the entire window and report 0% GPU activity.

Example:

```cpp
for (int iter = 0; iter < niter; ++iter) {
  nvtx3::scoped_range it{"iter:timestep"};
  {
    nvtx3::scoped_range ph{"phase:compute:dynamics"};
    kernel_a(...);
  }
  {
    nvtx3::scoped_range ph{"phase:host:halo_exchange"};
    exchange_halos(...);        // excluded from the window automatically
  }
}
```

Inside `kernel_a`:

```cpp
void kernel_a(...) {
  nvtx3::scoped_range fn{"function:src/kernels.cpp:kernel_a"};
  // launch CUDA kernel(s) ...
}
```

Before **planning from an nsys trace**, every translation target you
propose must have a matching `function:<file>:<function>` range in the
profile window (or add/refine ranges and re-profile first).

## Git Hygiene

Commit NVTX instrumentation separately from GPU translation:

```bash
git commit -m "nvtx: mark problem compute path for profiling"
```

Do not strip all markers until this problem's porting effort is done.
NVTX ranges cost effectively nothing when no profiler is attached, so
they are safe to keep through the porting effort.
