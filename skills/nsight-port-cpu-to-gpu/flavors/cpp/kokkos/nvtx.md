# NVTX / Profiling — Kokkos addendum

Read [../nvtx.md](../nvtx.md) first for the **naming grammar**. This
file is the Kokkos **how**: Tools connector + `Kokkos::Profiling`.
Use it when `config.json:framework` is `kokkos` (same-language C++
ports, or Stage B after a transition into C++).

Prefer this file's how over `nvtx3.hpp` once Kokkos is on the
compile line — `Kokkos::Profiling` routes to whichever tool is
loaded and needs no extra NVTX link flags.

## Layer 1: Kernel labels (automatic via Kokkos Tools)

Every `parallel_for` / `parallel_reduce` / `parallel_scan` takes a
**label** as its first argument. When the Kokkos-Tools NVTX connector
is loaded, those labels appear as NVTX ranges in the nsys timeline
with **no extra source ranges** — so always give kernels meaningful
labels using the grammar in `flavors/cpp/nvtx.md`:

```cpp
Kokkos::parallel_for("function:src/kernels.cpp:kernel_a",
    policy, KOKKOS_LAMBDA(...){ ... });
```

Load the connector in each performance run script under `USE_KOKKOS`:

```bash
# Point Kokkos at the NVTX connector shipped with Kokkos Tools.
# Discover it — do NOT hardcode an absolute path.
KP_NVTX=$(find "$KOKKOS_TOOLS_HOME" "$KOKKOS_HOME" -name 'kp_nvtx_connector.so' 2>/dev/null | head -1)
if [ -n "$KP_NVTX" ]; then
  export KOKKOS_TOOLS_LIBS="$KP_NVTX"   # newer Kokkos
  export KOKKOS_PROFILE_LIBRARY="$KP_NVTX"  # older Kokkos alias
fi
```

Record the discovered path in `.nsight-ai-port/build_notes.txt`. If the
connector is not installed, kernel labels still appear in nsys via
the CUDA kernel names, but named NVTX ranges will be missing.

## Layer 2: Application-phase ranges (`Kokkos::Profiling`)

For coarser markers around host phases (timestep loop, MPI, I/O) that
are **not** Kokkos kernels, use `Kokkos::Profiling::pushRegion` /
`popRegion`. These route to whichever profiling tool is loaded (the
NVTX connector), so you do **not** need to link `nvToolsExt` directly:

```cpp
#include <Kokkos_Core.hpp>

Kokkos::Profiling::pushRegion("phase:compute:dynamics");
// ... host orchestration / kernel launches ...
Kokkos::Profiling::popRegion();
```

A thin RAII helper keeps push/pop balanced (always paired, like a
`{ }` scope):

```cpp
struct ProfRange {
  explicit ProfRange(const char* name) { Kokkos::Profiling::pushRegion(name); }
  ~ProfRange() { Kokkos::Profiling::popRegion(); }
};
#define ACC_RANGE(name) ProfRange acc_range__##__LINE__(name)
```

```cpp
{
  ACC_RANGE("iter:timestep");
  ...
}   // popRegion happens automatically at scope exit
```

### Direct NVTX (fallback)

Only if you need NVTX independent of Kokkos Tools, follow the
link-library fallback in [../nvtx.md](../nvtx.md). Prefer Layer 2
over direct NVTX.

Run `scripts/profile-check.sh {to_lang}-{framework}` during setup —
it prints NVTX/connector hints. Never hardcode machine-specific
absolute SDK paths in a performance build script.

## Grammar reminders (Kokkos-specific use)

1. **`function:`** — use this as the **kernel label** for every
   function you suspect is hot. Because Kokkos forwards labels to
   NVTX, this tags hot kernels for free.
2. **`iter:` / `phase:`** — wrap with `Kokkos::Profiling` regions
   (Layer 2), not `nvtx3::scoped_range`, when Kokkos is linked.

Example:

```cpp
for (int iter = 0; iter < niter; ++iter) {
  ACC_RANGE("iter:timestep");
  {
    ACC_RANGE("phase:compute:dynamics");
    kernel_a(...);   // its parallel_for is labeled function:...
  }
  {
    ACC_RANGE("phase:host:halo_exchange");
    exchange_halos(...);
  }
}
```

Inside `kernel_a`:

```cpp
void kernel_a(...) {
  Kokkos::parallel_for("function:src/kernels.cpp:kernel_a",
      Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{ny,nx}),
      KOKKOS_LAMBDA(const int j, const int i) { /* ... */ });
}
```

## Git Hygiene

```bash
git commit -m "nvtx: label compute path for profiling"
```

Kernel labels (Layer 1) are good to keep permanently — they cost
nothing at runtime when no tool is loaded.
