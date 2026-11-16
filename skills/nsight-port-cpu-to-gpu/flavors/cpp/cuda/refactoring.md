# Refactoring addendum — C/C++ + CUDA

Read [../refactoring.md](../refactoring.md) first. This file is only
the CUDA-specific facts. `refactor-task` opens it when
`from_lang == to_lang` and `framework` is `cuda`.

## Extract so a CUDA kernel can replace it

Host threading regions (`#pragma omp parallel`, `std::thread`) that
wrap several calls should become a standalone function so a **CUDA
kernel** can replace that function. Same for loop nests split by
host MPI/I/O: each nest becomes a clean kernel candidate.

## No CUDA in Phase 2

Standard C++ only. Do not add `__global__`, `<<<>>>`, or CUDA
headers. Kernels come in Phase 4c.

Extracted GPU-target signatures must stay expressible in a **plain
C++ header** (flat pointers + scalars) so the future `.cu` wrapper
can be declared without CUDA syntax leaking into host TUs.

## Decoupled GPU entry point

You cannot pass a host `std::vector` / class with virtual methods
into a `__global__` kernel. The new symbol is
`{function}_cuda` and dispatches under `config.json:include_guard`
(default `USE_CUDA`):

```cpp
#ifdef USE_CUDA
  compute_interaction_cuda(grid.dx.data(), state.h.data(), du.data(),
                           grid.is, grid.ie, grid.js, grid.je, grid.nk);
#else
  compute_interaction(grid, state, du);
#endif
```

The `.cu` wrapper handles device memory per `memory_mode` (see
[patterns.md](patterns.md)). The `compute_interaction_cuda`
declaration lives in a plain C++ header.

## Class-method extraction

The free function's header must stay plain C++ so the Phase 4c
`.cu` wrapper can be declared without CUDA syntax in `Solver`'s
translation unit.
