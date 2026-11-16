# CUDA Patterns and Pitfalls — C/C++ + CUDA

Flavor pack: `flavors/cpp/cuda/`. Used in translate-function phase 4c.
Read `config.json:memory_mode` before choosing how device memory is
allocated and whether `cudaMemcpy` appears at all.

## Compilation Model (read this first)

CUDA C++ has a **host/device compilation split** that shapes every
file you write:

- **`.cu` files** are compiled by `nvcc` (or `nvc++ -cuda`). Only
  `.cu`/`.cuh` may contain CUDA syntax: `__global__`, `__device__`,
  `<<<...>>>` launches, `threadIdx`, `cudaMalloc`, `cudaError_t`, etc.
- **The public header** (`.hpp`/`.h`) that the original `.cpp` call
  site `#include`s must be **plain C++** — a normal function
  declaration with NO CUDA syntax. It is compiled by the project's
  host compiler, which does not understand `<<<>>>` or `cudaError_t`.

So the translated unit is always **two files**:

```text
{function}_cuda.hpp   ← plain C++ declaration (host-compilable)
{function}_cuda.cu    ← kernels + host wrapper (nvcc-compiled)
```

The wrapper in the `.cu` is an ordinary C++ (or `extern "C"`) function
the host code can call; it hides all CUDA internals.

## Kernel Basics

### A `__global__` kernel + launch

```cpp
// in {function}_cuda.cu
__global__ void add_kernel(const double* a, const double* b,
                           double* c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) c[i] = a[i] + b[i];          // ALWAYS bounds-check
}

void add_cuda(const double* a, const double* b, double* c, int n) {
  int threads = 256;
  int blocks  = (n + threads - 1) / threads;
  add_kernel<<<blocks, threads>>>(a, b, c, n);
  CUDA_CHECK(cudaGetLastError());          // catch launch errors
  CUDA_CHECK(cudaDeviceSynchronize());     // wait + catch run errors
}
```

Key rules:
- Compute the global index from `blockIdx`, `blockDim`, `threadIdx`.
- **Always** guard with `if (i < n)` — the last block over-runs `n`.
- Choose a block size that is a multiple of 32 (warp size); 128/256
  are good defaults.
- After every launch, check `cudaGetLastError()` (launch config /
  compile) and synchronize-or-check later for async runtime errors.

### Multi-dimensional indexing

Map nested loops onto a 2D/3D grid. Match the **fastest-varying loop
index to `x`** so consecutive threads touch consecutive memory
(coalesced access for row-major C arrays):

```cpp
__global__ void scale_kernel(const double* in, double* out,
                             int nx, int ny, int nz) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;  // fastest (cols)
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int k = blockIdx.z * blockDim.z + threadIdx.z;
  if (i < nx && j < ny && k < nz) {
    size_t idx = (size_t)k * ny * nx + (size_t)j * nx + i;  // row-major
    out[idx] = in[idx] * 2.0;
  }
}

// launch:
dim3 block(32, 8, 1);
dim3 grid((nx + block.x - 1) / block.x,
          (ny + block.y - 1) / block.y,
          (nz + block.z - 1) / block.z);
scale_kernel<<<grid, block>>>(in, out, nx, ny, nz);
```

Use `size_t` for flattened indices to avoid 32-bit overflow on large
arrays.

### Grid-stride loops (preferred for portability)

A grid-stride loop lets one launch cover any problem size and any
grid size — robust and the recommended idiom:

```cpp
__global__ void saxpy(int n, double a, const double* x, double* y) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x;
       i < n;
       i += blockDim.x * gridDim.x) {
    y[i] = a * x[i] + y[i];
  }
}
```

## Device Functions (`__device__`)

Helper functions called from inside a kernel (index math, math
helpers, boundary functions) must be `__device__` (or
`__host__ __device__` if the host also calls them — useful for the
device-utils test). This is the CUDA analog of OpenACC
`!$acc routine seq` / Kokkos `KOKKOS_INLINE_FUNCTION`.

Collect these during planning and place them in the per-parent
header `{parent}/cuda/device_utils.cuh` during Phase 3:

```cpp
// src/utils/cuda/device_utils.cuh
#pragma once

__host__ __device__ inline
int periodic_index_device(int i, int N) {
  return ((i % N) + N) % N;
}
```

Translated kernel files `#include "device_utils.cuh"` and call the
device version directly — no `#ifdef` guards inside the GPU-only `.cu`:

```cpp
im = periodic_index_device(i - 1, N);
```

Rules:
- `__host__ __device__ inline` (host-callable too) makes the
  device-utils test able to compare host vs device versions.
- Keep them header-only (`inline`) so every including `.cu` gets the
  device definition.
- Signatures must match the original host helper exactly (only the
  name gets a `_device` suffix).
- Use device math intrinsics where helpful (`__fmul_rn`, `rsqrt`) but
  prefer plain `sqrt`/`exp` first for correctness.
- Do NOT make utility functions separate porting targets — Phase 3
  handles them collectively.

## Error Checking

Wrap every CUDA runtime call. Put the macro in a shared header (e.g.
`{parent}/cuda/cuda_check.cuh`) and include it from the `.cu` files:

```cpp
#define CUDA_CHECK(call)                                              \
  do {                                                               \
    cudaError_t err__ = (call);                                      \
    if (err__ != cudaSuccess) {                                      \
      std::fprintf(stderr, "CUDA error %s at %s:%d: %s\n",          \
                   #call, __FILE__, __LINE__,                        \
                   cudaGetErrorString(err__));                       \
      std::abort();                                                  \
    }                                                               \
  } while (0)
```

After a kernel launch, check **both**:
```cpp
kernel<<<g,b>>>(...);
CUDA_CHECK(cudaGetLastError());        // launch/config error (sync)
CUDA_CHECK(cudaDeviceSynchronize());   // execution error (async)
```

Silent kernel failures are the #1 cause of "GPU ran but results are
wrong/zero" — never skip the launch error check.

## Memory-Mode Policy

**Read `config.json:memory_mode` before allocating anything.** The
mode decides whether you call `cudaMalloc`/`cudaMemcpy` at all.

### `managed` — `cudaMallocManaged` (UVM)

**The contract:** `managed` is *software-level* memory management. Every
allocation a kernel touches must come from the managed allocator, and
the CUDA runtime then migrates pages between host and device on demand.
In exchange you write no `cudaMemcpy`. This works on **any** CUDA GPU,
discrete included, precisely because the allocator — not the hardware —
supplies the coherence. Do not confuse it with `unified` mode, which
changes no allocations and relies on hardware coherence.

The obligation that follows: an allocation the runtime does not know
about is plain host memory, and on a discrete GPU a kernel dereferencing
it faults or reads garbage. So **whoever owns the array must allocate it
as managed**, even if that owner is not the code you are translating.

Allocate with `cudaMallocManaged`; the same pointer is valid on host
and device. **No `cudaMemcpy`** — just synchronize before the host
reads results:

```cpp
double* a;
CUDA_CHECK(cudaMallocManaged(&a, n * sizeof(double)));
// init on host directly:
for (int i = 0; i < n; ++i) a[i] = ...;
kernel<<<blocks, threads>>>(a, n);
CUDA_CHECK(cudaGetLastError());
CUDA_CHECK(cudaDeviceSynchronize());   // required before host reads
double x = a[0];                       // valid, no copy
CUDA_CHECK(cudaFree(a));
```

When the **caller already passes managed pointers** (e.g. from a
managed-memory wrapper), the kernel/wrapper uses them directly — do
NOT allocate device copies or `cudaMemcpy`.

#### Fortran caller: the allocation shim is not optional

When the original code is Fortran (`from_lang: fortran`) the Fortran side
still owns the arrays, and a plain `allocate(a(n))` yields host memory.
Passing it to a kernel in `managed` mode is a silent correctness bug on
any discrete GPU. Give Fortran a managed allocator and bind the result
back to a Fortran pointer with `c_f_pointer`:

```fortran
! src/cuda_managed.F90  (.F90 so the guard is preprocessed)
module cuda_managed
  use iso_c_binding
  implicit none
  integer(c_int), parameter :: cudaMemAttachGlobal = 1

  interface
    integer(c_int) function cudaMallocManaged(ptr, bytes, flags) &
        bind(C, name="cudaMallocManaged")
      import :: c_ptr, c_size_t, c_int
      type(c_ptr),       intent(out) :: ptr
      integer(c_size_t), value       :: bytes
      integer(c_int),    value       :: flags
    end function
    integer(c_int) function cuda_free(ptr) bind(C, name="cudaFree")
      import :: c_ptr, c_int
      type(c_ptr), value :: ptr
    end function
  end interface

contains

  ! Drop-in replacement for `allocate(a(n))` on kernel-visible arrays.
  subroutine alloc_managed_1d(a, n)
    real(c_double), pointer, intent(out) :: a(:)
    integer,                 intent(in)  :: n
    type(c_ptr)    :: cptr
    integer(c_int) :: stat
    stat = cudaMallocManaged(cptr, int(n, c_size_t) * c_sizeof(1.0_c_double), &
                             cudaMemAttachGlobal)
    if (stat /= 0) error stop "cudaMallocManaged failed"
    call c_f_pointer(cptr, a, [n])
  end subroutine
end module
```

Call sites become guarded, so the CPU build keeps ordinary allocations:

```fortran
real(c_double), pointer :: a(:)      ! pointer, NOT allocatable
#ifdef USE_CUDA
call alloc_managed_1d(a, n)
#else
allocate(a(n))
#endif
```

Three ways this shim gets silently defeated — check all three:

- **`-DUSE_CUDA` never reaches the Fortran compiler.** The managed branch
  compiles out, the code still runs, and the device sees host memory. The
  guard must be on the Fortran flags too (`-Mpreprocess` / `.F90`).
- **The array is declared `allocatable`.** `c_f_pointer` targets need
  `pointer`; leaving it `allocatable` forces a second, unmanaged buffer.
- **The pointer is copied into a temporary at the `bind(C)` boundary.**
  nvfortran materializes contiguous host temporaries for array dummies,
  so the kernel operates on the temp rather than the managed buffer. Pass
  the address by value (`type(c_ptr), value` + `c_loc`) or receive it as
  an assumed-size `u(*)` dummy.

A managed-mode Fortran build that produces correct results *only* because
every array happens to be small enough to migrate is not correct code. Before
any test, satisfy the mandatory
[Fortran managed-memory guard gate](build-integration.md#fortran-managed-memory-guard-gate):
capture the actual Fortran compile commands and inspect their preprocessed
output to prove that `-DUSE_CUDA` selected the shim branch. Result agreement
alone cannot waive this gate.

### `unified` — hardware-coherent memory (e.g. GH200)

Standard host allocations are device-accessible. Pass the caller's
raw host pointers straight to the kernel; **no `cudaMalloc`, no
`cudaMemcpy`**:

```cpp
void compute_cuda(double* host_ptr, int n) {
  kernel<<<blocks, threads>>>(host_ptr, n);   // host_ptr usable on device
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
}
```

Unlike `managed`, **no** allocation changes anywhere — including on a
Fortran caller's side, so there is no shim to write. That is also why
`unified` is not a portability fallback: the same code faults on a
discrete GPU without coherence. Requires a coherent platform (HMM /
Grace Hopper); record the assumption in `.nsight-ai-port/build_notes.txt`, and if you
cannot confirm coherence, fall back to `managed` rather than hoping.

### `separate` — explicit device memory

Allocate device buffers, copy in/out **once at the widest stable
scope**, never inside a hot loop:

```cpp
void compute_cuda(const double* h_in, double* h_out, int n) {
  size_t sz = (size_t)n * sizeof(double);
  double *d_in = nullptr, *d_out = nullptr;
  CUDA_CHECK(cudaMalloc(&d_in,  sz));
  CUDA_CHECK(cudaMalloc(&d_out, sz));
  CUDA_CHECK(cudaMemcpy(d_in, h_in, sz, cudaMemcpyHostToDevice)); // in

  kernel<<<blocks, threads>>>(d_in, d_out, n);
  CUDA_CHECK(cudaGetLastError());

  CUDA_CHECK(cudaMemcpy(h_out, d_out, sz, cudaMemcpyDeviceToHost)); // out
  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaFree(d_out));
}
```

For functions called thousands of times per run, hoist the
`cudaMalloc`/`cudaMemcpy`-in and the copy-out **outside** the call
loop (persistent device buffers) — see translation.md decoupled entry
points.

| Mode | Allocation | Host↔device copies |
|------|-----------|--------------------|
| `managed` | `cudaMallocManaged` (or caller's managed ptr) | None — just `cudaDeviceSynchronize` |
| `unified` | none (use host ptr) | None |
| `separate` | `cudaMalloc` device buffers | `cudaMemcpy`, hoisted only |

In `managed`/`unified` modes there must be **zero** `cudaMemcpy` /
`cudaMalloc` calls in the translated kernel. If you find yourself
adding one, you picked the wrong mode.

## Reductions

A naive `out[0] += x` across threads is a race. Options, simplest
first:

### Atomics (simplest, correct)

```cpp
__global__ void sum_kernel(const double* a, double* result, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) atomicAdd(result, a[i]);   // result is a single device double
}
```
Initialize `*result = 0` (host on managed, or `cudaMemset`) before the
launch. `atomicAdd` for `double` requires SM 6.0+.

### Library reduction (preferred for large reductions)

Use CUB / Thrust rather than hand-rolling block reductions:
```cpp
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>
double total = thrust::reduce(thrust::device, d_a, d_a + n, 0.0);
```

Only hand-write block/warp-shuffle reductions when CUB/Thrust is
unavailable. Correctness first.

## Shared Memory and Streams

- **Shared memory** for block-local scratch / tiling:
  ```cpp
  __global__ void k(...) {
    __shared__ double tile[256];
    tile[threadIdx.x] = ...;
    __syncthreads();              // barrier before reading neighbors
  }
  ```
  Dynamic shared memory: `extern __shared__ double s[];` with the size
  as the 3rd launch parameter `<<<g, b, bytes>>>`.
- **Streams** for overlap (advanced; not needed initially). Default
  stream (0) is fine for first translation. Add streams only after
  correctness is established.

## Ifdef Guard Pattern

**Where `#ifdef USE_CUDA` guards go — and where they don't:**

Translated files in `{parent}/cuda/` are **pure GPU code** compiled by
nvcc only when the GPU path is built. Write CUDA constructs directly;
do **not** wrap them in `#ifdef USE_CUDA` inside the `.cu`/`.cuh`:

```cpp
// In src/physics/cuda/solver_cuda.cu (GPU-only file):
rad_kernel<<<grid, block>>>(a, b, c, nx, ny);   // direct, no #ifdef
```

`#ifdef USE_CUDA` guards belong in **two places only**:

1. **Call sites** (integration, Phase 4e) — switching between original
   and translated function, including the `#include` of the plain
   translated header:
   ```cpp
   #ifdef USE_CUDA
   #include "cuda/compute_kernel_cuda.hpp"
   #endif
   ...
   #ifdef USE_CUDA
     compute_kernel_cuda(args);
   #else
     compute_kernel(args);
   #endif
   ```
2. **Test harnesses** — the GPU comparison block.

C/C++ sources are always preprocessed, so there is no `.f90` vs
`.F90` concern. The requirement is that the build pass `-DUSE_CUDA` to
**every** translation unit that carries the guard (project source,
in-tree dependencies, and the standalone test — see
build-integration.md).

## Common Pitfalls

### 1. Missing bounds check

Every index-based kernel needs `if (i < n)` (and `j < ny`, …). The
last block launches extra threads that would read/write out of bounds.

### 2. CUDA syntax in a host-compiled header

The public `{function}_cuda.hpp` is included by the original `.cpp`
(host compiler). It must contain **only** a plain C++ declaration — no
`__global__`, `<<<>>>`, `dim3`, or `cudaError_t`. Put all CUDA syntax
in the `.cu`/`.cuh`.

### 3. Forgetting the launch error check

`kernel<<<>>>()` returns void. A bad launch config or device fault is
silent unless you call `cudaGetLastError()` (sync errors) and
`cudaDeviceSynchronize()` / a later sync call (async errors).

### 4. Row-major vs coalescing

C arrays are row-major: `a[k*ny*nx + j*nx + i]`. Map the `i`
(fastest) index to `threadIdx.x` so consecutive threads read
consecutive addresses. A transposed mapping is correct but slow;
correctness first, then fix coalescing.

### 5. 32-bit index overflow

`int idx = k*ny*nx + ...` overflows for large arrays. Use `size_t`
for flattened indices and extents that can exceed 2^31.

### 6. Host pointer used on device (separate mode)

In `separate` mode you must pass **device** pointers (`d_*`) to the
kernel, not host pointers. Passing a host pointer to a kernel is an
illegal access. (In `managed`/`unified` modes the pointer is valid on
both sides — that's the whole point.)

### 7. `cudaMemcpy` direction

`cudaMemcpy(dst, src, n, kind)` — `kind` is
`cudaMemcpyHostToDevice` (in) or `cudaMemcpyDeviceToHost` (out).
Getting the direction or argument order wrong silently corrupts data.

### 8. Missing synchronization before host read

Kernel launches are asynchronous. Before the host reads results in
`managed`/`unified` mode, call `cudaDeviceSynchronize()`. In
`separate` mode the `cudaMemcpy`-out is synchronizing, so an explicit
sync is usually unnecessary.

### 9. Calling host-only code in a kernel

Inside `__global__`/`__device__` code you may use `printf` and CUDA
math, but NOT `std::vector`, `std::cout`, `new`/`delete` of host
objects, exceptions, or functions not marked `__device__`. Mark shared
helpers `__host__ __device__`.

### 10. Per-call allocation on hot paths (separate mode)

`cudaMalloc`/`cudaFree` and `cudaMemcpy` are expensive. For functions
called many times, hoist allocation and the copy-in/out outside the
call loop (persistent buffers) rather than allocating every call.

### 11. Wrong `-gencode` / architecture

The compiled SM arch must match the actual GPU. Do NOT hardcode
`sm_70`/`sm_80` — detect the compute capability with `nvidia-smi
--query-gpu=compute_cap` and generate the matching `-gencode` (see
build-integration.md). Hardcoded x86-era arch fails on Grace Hopper
(`sm_90`, aarch64).

### 12. External library flags

Do NOT compile third-party/vendored code with `nvcc`, CUDA include
paths, or `-DUSE_CUDA` unless it is genuinely CUDA code. Scope CUDA
flags to the project's own translated `.cu` sources.

### 13. Floating-point precision

GPU floating-point may execute in a different order than the CPU,
producing small numerical differences. This is expected — use
tolerances when comparing:
- Single precision: abs_tol=1e-5, rel_tol=1e-4
- Double precision: abs_tol=1e-12, rel_tol=1e-10

The standalone test reads `ABS_TOL` / `REL_TOL` from the environment;
loosen by 10x at a time for genuine FP-ordering noise only. Do NOT
change the translated math for FP-ordering differences. Note `-use_fast_math`
widens error — avoid it during verification.
