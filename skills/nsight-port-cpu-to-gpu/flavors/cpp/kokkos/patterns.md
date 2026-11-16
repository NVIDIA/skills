# Kokkos Patterns and Pitfalls — C/C++ + Kokkos

Flavor pack: `flavors/cpp/kokkos/`. Used in translate-function
phase 4c. Read `config.json:memory_mode` before choosing the View
memory space and deciding whether `deep_copy` is needed.

## Target: Kokkos 4 + C++17

Target **Kokkos 4.x compiled to the C++17 standard** (`-std=c++17`).
Kokkos 4 gives us `Kokkos::SharedSpace` (managed/UVM memory) without
the C++20 requirement that Kokkos 5 imposes. Do **not** use C++20-only
syntax (concepts, `consteval`, designated-initializer-heavy code,
`<ranges>`); it will break the Kokkos 4 build on the CUDA backend.

| Concern | Decision |
|---------|----------|
| Language standard | `-std=c++17` everywhere (project + tests) |
| Managed memory | `Kokkos::SharedSpace` (Kokkos 4, no C++20) |
| Compiler | `nvcc_wrapper` (CUDA backend) or `nvc++` |
| Lambda flags | `--extended-lambda --expt-relaxed-constexpr` (nvcc path) |

## Core Concepts

Kokkos has three things you must reason about for every kernel:

1. **Execution space** — *where* code runs (`Kokkos::Cuda`,
   `Kokkos::OpenMP`, `Kokkos::Serial`). The default execution space
   is set by the build; on a CUDA build it is `Kokkos::Cuda`.
2. **Memory space** — *where* data lives (`Kokkos::CudaSpace`,
   `Kokkos::HostSpace`, `Kokkos::SharedSpace` for UVM). A `View`'s
   space is part of its type.
3. **Execution policy** — *how many* iterations and their shape
   (`RangePolicy`, `MDRangePolicy`, `TeamPolicy`).

`memory_mode` (from `config.json`) decides which memory space your
Views use and whether you copy data explicitly — see "Views and
Memory-Mode Policy" below.

## Initialization Lifecycle

Kokkos must be initialized before any View or kernel and finalized
after. The translated **kernel files do not** call
`initialize`/`finalize` — the standalone test program and the host
application own the lifecycle. In a test program use a scope guard:

```cpp
int main(int argc, char* argv[]) {
  Kokkos::ScopeGuard guard(argc, argv);   // init + finalize (RAII)
  // ... all View allocation and kernel launches here ...
  return failures;   // guard finalizes as it goes out of scope
}
```

In an existing application, initialization is added once at the
program entry point during integration (Phase 4e) — never inside a
hot function.

## Core Kernel Patterns

### parallel_for — 1D range

The most common pattern. Replaces a single `for` loop:

```cpp
// Original:
for (int i = 0; i < n; ++i) c[i] = a[i] + b[i];

// Kokkos:
Kokkos::parallel_for(
    "add", n, KOKKOS_LAMBDA(const int i) {
      c(i) = a(i) + b(i);
    });
```

- The first argument is a **label** (used in profiling — give it the
  kernel's purpose, e.g. `"compute_flux_ij"`).
- `KOKKOS_LAMBDA` expands to `[=] __host__ __device__` — it captures
  by value. Views are captured by value cheaply (reference-counted
  handles); raw scalars are copied.
- The index type should match the loop (`const int`, `const size_t`).

### parallel_for — multi-dimensional (MDRangePolicy)

Replaces tightly nested loops. This is the Kokkos analog of OpenACC
`collapse(N)`:

```cpp
// Original triple loop (row-major C indexing):
for (int k = 0; k < nz; ++k)
  for (int j = 0; j < ny; ++j)
    for (int i = 0; i < nx; ++i)
      a(k,j,i) = b(k,j,i) * c(k,j,i);

// Kokkos MDRangePolicy (rank 3):
using MDPolicy = Kokkos::MDRangePolicy<Kokkos::Rank<3>>;
Kokkos::parallel_for(
    "scale", MDPolicy({0,0,0}, {nz,ny,nx}),
    KOKKOS_LAMBDA(const int k, const int j, const int i) {
      a(k,j,i) = b(k,j,i) * c(k,j,i);
    });
```

- `Rank<N>` must match the loop nest depth.
- Bounds are `{lower...}, {upper...}` (upper is exclusive).
- Order the rank dimensions so the **fastest-varying index maps to
  the last argument** for the View's layout (see View Layout pitfall).

### parallel_reduce

```cpp
double total = 0.0;
Kokkos::parallel_reduce(
    "sum", n,
    KOKKOS_LAMBDA(const int i, double& local) {
      local += a(i);
    },
    total);
Kokkos::fence();   // total is valid after the reduce returns
```

For `max`/`min` and other operators use a **reducer**:

```cpp
double maxval;
Kokkos::parallel_reduce(
    "max", n,
    KOKKOS_LAMBDA(const int i, double& lmax) {
      lmax = Kokkos::max(lmax, a(i));
    },
    Kokkos::Max<double>(maxval));
```

Available reducers: `Kokkos::Sum`, `Prod`, `Max`, `Min`, `MinMax`,
`BAnd`, `BOr`, `LAnd`, `LOr`, `MaxLoc`, `MinLoc`. Multiple reductions
in one kernel are supported by passing several reducer arguments.

The scalar reduction result is on the host after the call — no manual
copy needed — but treat it as valid only after the kernel completes
(a `parallel_reduce` is effectively synchronizing for the result).
This host-blocking behavior is implementation-defined in the Kokkos
spec rather than guaranteed, so an explicit `Kokkos::fence()` after
the reduce is always acceptable (and redundant) if you want to be
unambiguous.

### parallel_scan (prefix sums)

```cpp
Kokkos::parallel_scan(
    "prefix", n,
    KOKKOS_LAMBDA(const int i, double& partial, const bool is_final) {
      partial += a(i);
      if (is_final) prefix(i) = partial;
    });
```

### Column / vertical sweeps (sequential inner loop)

For column algorithms where an outer `(j,i)` loop wraps a sequential
`k` sweep (tridiagonal solves, etc.), keep the `k` loop **serial
inside** the lambda and parallelize only the outer dimensions:

```cpp
using MDPolicy = Kokkos::MDRangePolicy<Kokkos::Rank<2>>;
Kokkos::parallel_for(
    "column_solve", MDPolicy({js,is}, {je,ie}),
    KOKKOS_LAMBDA(const int j, const int i) {
      double work[NK_MAX];          // per-thread private scratch
      for (int k = 0; k < nk; ++k) {
        work[k] = /* ... */;        // sequential dependence in k
      }
      // tridiagonal solve over work[0..nk-1]
    });
```

- Per-column scratch declared **inside** the lambda is automatically
  private to each thread. Size it to the maximum column depth.
- For large per-team scratch, use `TeamPolicy` scratch memory (below).

## Hierarchical Parallelism (TeamPolicy)

The Kokkos analog of OpenACC `gang` / `worker` / `vector`. Use when
a kernel needs two levels of parallelism or shared scratch per team.

```cpp
using team_policy = Kokkos::TeamPolicy<>;
using member_type = team_policy::member_type;

Kokkos::parallel_for(
    "teams", team_policy(num_teams, Kokkos::AUTO),
    KOKKOS_LAMBDA(const member_type& team) {
      const int j = team.league_rank();          // team index
      Kokkos::parallel_for(
          Kokkos::TeamThreadRange(team, nx),
          [&](const int i) {
            a(j,i) = b(j,i) + c(j,i);             // thread level
          });
    });
```

| OpenACC | Kokkos |
|---------|--------|
| `gang`  | league (team) — `team.league_rank()` |
| `worker`| `TeamThreadRange` |
| `vector`| `ThreadVectorRange` |

Start with flat `RangePolicy` / `MDRangePolicy`. Only reach for
`TeamPolicy` when you need team scratch or nested parallelism — it is
harder to get right.

### Team scratch memory

```cpp
int scratch_bytes = Kokkos::View<double*,
    Kokkos::DefaultExecutionSpace::scratch_memory_space>::
    shmem_size(nx);
Kokkos::parallel_for(
    "scratch", team_policy(num_teams, Kokkos::AUTO)
        .set_scratch_size(0, Kokkos::PerTeam(scratch_bytes)),
    KOKKOS_LAMBDA(const member_type& team) {
      Kokkos::View<double*,
          Kokkos::DefaultExecutionSpace::scratch_memory_space>
          tmp(team.team_scratch(0), nx);
      // ... use tmp as fast per-team shared memory ...
    });
```

## Views and Memory-Mode Policy

**Read `config.json:memory_mode` before choosing a View type.** A
`View`'s memory space is part of its C++ type, so the mode decides
how you declare every array and whether `deep_copy` appears at all.

### View basics

```cpp
Kokkos::View<double*>      v1d("v1d", n);          // 1D
Kokkos::View<double**>     v2d("v2d", nrows, ncols);// 2D
Kokkos::View<double***>    v3d("v3d", nz, ny, nx); // 3D
// Access with PARENTHESES, not brackets:
v3d(k,j,i) = 1.0;
```

Views are reference-counted: copying a View copies a handle, not the
data. Capturing a View by value in a lambda is the correct, cheap way
to use it on the device.

### `managed` — Kokkos::SharedSpace (UVM, the Kokkos 4 default path)

When `memory_mode == "managed"`, allocate Views in
`Kokkos::SharedSpace`. The same pointer is valid on host and device;
**no `deep_copy` is needed**.

```cpp
Kokkos::View<double***, Kokkos::SharedSpace> a("a", nz, ny, nx);
// initialize on host directly:
a(0,0,0) = 1.0;
// use on device with no copy:
Kokkos::parallel_for("k", MDPolicy({0,0,0},{nz,ny,nx}),
    KOKKOS_LAMBDA(const int k,const int j,const int i){
      a(k,j,i) *= 2.0;
    });
Kokkos::fence();
// read back on host directly — still no copy:
double x = a(0,0,0);
```

When wrapping a **raw managed pointer** handed in from the caller
(e.g. allocated by a managed-memory wrapper), build a non-owning
unmanaged View over it:

```cpp
Kokkos::View<double***, Kokkos::SharedSpace,
    Kokkos::MemoryTraits<Kokkos::Unmanaged>>
    a(raw_ptr, nz, ny, nx);
```

Requires Kokkos configured with `SharedSpace` support
(`Kokkos_ENABLE_CUDA` + the default UVM-capable allocator). See
build-integration.md.

### `unified` — hardware-coherent memory (e.g. GH200)

When `memory_mode == "unified"`, standard host allocations are
already device-accessible. Wrap the caller's raw host pointers in
**non-owning unmanaged Views**; no `deep_copy`, no SharedSpace:

```cpp
Kokkos::View<double***, Kokkos::MemoryTraits<Kokkos::Unmanaged>>
    a(host_ptr, nz, ny, nx);
Kokkos::parallel_for(...);   // runs on device, reads host_ptr directly
Kokkos::fence();
```

### `separate` — explicit device memory

When `memory_mode == "separate"`, data lives in device memory
(`CudaSpace`) and you manage host↔device transfers explicitly with
mirror views and `deep_copy`. **Hoist allocations and copies to the
widest stable scope** — never `deep_copy` inside a hot loop.

```cpp
// Device View (default space on a CUDA build):
Kokkos::View<double***> d_a("d_a", nz, ny, nx);
// Host mirror:
auto h_a = Kokkos::create_mirror_view(d_a);

// Fill on host, copy IN once (hoisted, outside the timestep loop):
/* fill h_a ... */
Kokkos::deep_copy(d_a, h_a);

for (int step = 0; step < nsteps; ++step) {
  Kokkos::parallel_for("k", MDPolicy({0,0,0},{nz,ny,nx}),
      KOKKOS_LAMBDA(const int k,const int j,const int i){
        d_a(k,j,i) = /* ... */;
      });
}
Kokkos::fence();

// Copy OUT once after the loop:
Kokkos::deep_copy(h_a, d_a);
```

| Mode | View memory space | deep_copy? |
|------|-------------------|-----------|
| `managed` | `Kokkos::SharedSpace` (or unmanaged over managed ptr) | No |
| `unified` | default + `Unmanaged` over host ptr | No |
| `separate` | `CudaSpace` device View + host mirror | Yes (hoisted only) |

**Never** put a `deep_copy` on a per-call hot path. In `managed` /
`unified` modes there should be **zero** `deep_copy` calls in the
translated kernel — if you find yourself adding one, you picked the
wrong View type.

## Device Functions (KOKKOS_INLINE_FUNCTION)

Helper functions called from inside a kernel (index math, boundary
helpers, small math) must be callable on the device. Mark them
`KOKKOS_INLINE_FUNCTION` (expands to `__host__ __device__ inline`).
This is the Kokkos analog of OpenACC `!$acc routine seq`.

These helpers are **not** annotated inline in scattered call sites.
They are collected during planning and placed in the per-parent
header `{parent}/kokkos/device_utils.hpp` during Phase 3:

```cpp
// src/utils/kokkos/device_utils.hpp
#pragma once
#include <Kokkos_Core.hpp>

KOKKOS_INLINE_FUNCTION
int periodic_index_device(int i, int N) {
  return ((i % N) + N) % N;
}
```

Translated kernel files `#include "device_utils.hpp"` and call the
device version directly — no `#ifdef` guards inside the GPU-only file:

```cpp
im = periodic_index_device(i - 1, N);
```

Rules:
- `KOKKOS_INLINE_FUNCTION` goes **before** the return type.
- Keep device helpers header-only (`inline`) so every translation
  unit that includes them gets the definition on the device.
- Signatures must match the original host helper exactly (only the
  name gets a `_device` suffix).
- Do **not** make utility functions separate porting targets — they
  are handled collectively in Phase 3.

## Ifdef Guard Pattern

**Where `#ifdef` guards go — and where they don't:**

Translated files in `{parent}/kokkos/` are **pure GPU code** — they
are only compiled when `USE_KOKKOS` is defined. Write Kokkos
constructs directly; do **not** wrap them in
`#ifdef {INCLUDE_GUARD}` inside the translated file:

```cpp
// In src/physics/kokkos/solver_kokkos.cpp (GPU-only file):
Kokkos::parallel_for("rad", MDPolicy({0,0},{ny,nx}),
    KOKKOS_LAMBDA(const int j, const int i) {
      a(j,i) = b(j,i) + c(j,i);
    });
```

**Bad (redundant guard inside `{parent}/kokkos/`):**
```cpp
#ifdef USE_KOKKOS                         // WRONG here
  Kokkos::parallel_for(...);
#endif
```

`#ifdef USE_KOKKOS` guards belong in **two places only**:

1. **Call sites** (integration, Phase 4e) — switching between the
   original and translated function:
   ```cpp
   #ifdef USE_KOKKOS
     compute_kernel_kokkos(args);
   #else
     compute_kernel(args);
   #endif
   ```
2. **Test harnesses** — optionally running the GPU comparison block:
   ```cpp
   #ifdef USE_KOKKOS
     // call translated version + compare
   #endif
   ```

Unlike Fortran, C/C++ sources are **always preprocessed**, so there
is no `.f90` vs `.F90` / `-cpp` concern. The remaining requirement is
that the build pass `-DUSE_KOKKOS` to **every** translation unit that
carries the guard: the project source, in-tree dependencies, AND the
standalone test (see build-integration.md and translation.md).

## Common Pitfalls

### 1. View access uses `()` not `[]`

`v(i,j,k)`, never `v[i][j][k]`. Multi-dimensional `operator[]` does
not exist on Views.

### 2. External library flags

Do **not** compile third-party/vendored libraries with the Kokkos
include path, `nvcc_wrapper`, or CUDA device flags unless they are
genuinely Kokkos code. Scope Kokkos/CUDA flags to the project's own
translated sources. External libraries keep their original compiler
and flags.

### 3. View layout (LayoutLeft vs LayoutRight) — column-major trap

The default layout depends on the memory space: **`CudaSpace` Views
default to `LayoutLeft` (column-major)**, while `HostSpace` defaults
to `LayoutRight` (row-major). This matters in two cases:

- **Wrapping a raw pointer** in an unmanaged View: the raw C array is
  row-major, so you MUST specify `Kokkos::LayoutRight` explicitly or
  the indices will be transposed:
  ```cpp
  Kokkos::View<double***, Kokkos::LayoutRight,
      Kokkos::MemoryTraits<Kokkos::Unmanaged>>
      a(host_ptr, nz, ny, nx);   // matches a[k*ny*nx + j*nx + i]
  ```
- **Performance**: for a freshly allocated device View, prefer the
  default `LayoutLeft` and order MDRange so the **first** index is the
  fastest-varying for coalesced access. When matching existing
  row-major data, keep `LayoutRight` for correctness first.

Correctness before performance: when in doubt, pin the layout to
match the original array's memory order.

### 4. Array indexing

C/C++ is 0-based on both host and device. If the original source is
already C/C++ (this flavor), indexing is unchanged. (Only relevant if
porting from a 1-based language.)

### 5. Compiler and lambda flags

Use `nvcc_wrapper` (or `nvc++`) as the C++ compiler for Kokkos+CUDA.
With the `nvcc_wrapper` path you MUST pass:

```text
-std=c++17 --extended-lambda --expt-relaxed-constexpr
```

`--extended-lambda` allows `__device__` lambdas (KOKKOS_LAMBDA);
without it the kernel lambda fails to compile. Detect the compiler
and add these only for the Kokkos build path, not the CPU path.

### 6. Capturing `this` / class members in a lambda

`KOKKOS_LAMBDA` captures by value `[=]`. Inside a class method this
captures the **`this` pointer**, not the members — dereferencing it
on the device is illegal and crashes. Fixes:
- Copy needed members into **local variables** before the lambda and
  capture those locals:
  ```cpp
  auto a = this->a_;     // a_ is a View member
  const int n = this->n_;
  Kokkos::parallel_for("k", n, KOKKOS_LAMBDA(const int i){
      a(i) += n;          // captures local View handle + scalar
  });
  ```
- Or use `KOKKOS_CLASS_LAMBDA` only if the whole object is device-safe
  (rare). Prefer local copies.

### 7. Calling host-only functions in device code

Inside a `KOKKOS_LAMBDA` or `KOKKOS_INLINE_FUNCTION` you may NOT call:
- host-only library functions (most of `<cmath>` is OK via
  `Kokkos::` math, but general `std::` containers/streams are not),
- `printf` is allowed; `std::cout`/`std::vector`/`new`/`throw` are not,
- functions not marked `__host__ __device__`.

Use `Kokkos::` math functions (`Kokkos::sqrt`, `Kokkos::exp`,
`Kokkos::abs`, `Kokkos::max`) for portability across backends.

### 8. Missing `Kokkos::fence()`

Kernel launches are asynchronous. Before reading results on the host
(after a `parallel_for` that wrote a host-visible View, or before
timing), call `Kokkos::fence()`. `parallel_reduce` into a scalar is
synchronizing for that scalar in practice (though implementation-
defined — an explicit fence after a reduce is always safe), but a
following host read of a View written by a separate `parallel_for`
still needs a fence.

### 9. Floating-point precision

GPU floating-point may execute in a different order than the CPU,
producing small numerical differences. This is expected — use
tolerances when comparing:
- Single precision: abs_tol=1e-5, rel_tol=1e-4
- Double precision: abs_tol=1e-12, rel_tol=1e-10

The standalone test reads `ABS_TOL` / `REL_TOL` from the environment;
loosen by 10x at a time for genuine FP-ordering differences. Do NOT
change the translated math for FP-ordering noise.

### 10. Nested kernel launches

You cannot launch a `parallel_for` from inside another
`parallel_for`'s lambda on the device. If a kernel needs a second
level of parallelism, use `TeamPolicy` with `TeamThreadRange` /
`ThreadVectorRange` instead. Helper routines called from a kernel must
be `KOKKOS_INLINE_FUNCTION`, not new kernels.

### 11. deep_copy direction and mismatched spaces

`Kokkos::deep_copy(dst, src)` copies `src → dst`. Mixing it up
silently corrupts data. In `managed`/`unified` modes you should have
**no** `deep_copy` at all; if one appears, you likely used the wrong
View memory space.

### 12. SharedSpace availability (Kokkos 4)

`Kokkos::SharedSpace` requires Kokkos 4 built with CUDA and UVM
support. If the build reports `SharedSpace` is not a member of
`Kokkos`, the Kokkos install lacks UVM support — record this in
`.nsight-ai-port/build_notes.txt` and fall back to `separate` mode (device View +
mirror + hoisted `deep_copy`) rather than forcing managed memory.

### 13. View default initialization cost

`Kokkos::View<T*>("x", n)` value-initializes (zeros) by default. If
you immediately overwrite every element, use
`Kokkos::view_alloc(Kokkos::WithoutInitializing, "x")` to skip it.
Correctness first — only optimize this once results match.
