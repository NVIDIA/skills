# Refactoring addendum — C/C++ + Kokkos

Read [../refactoring.md](../refactoring.md) first. This file is only
the Kokkos-specific facts. `refactor-task` opens it when
`from_lang == to_lang` and `framework` is `kokkos`.

## Extract so a Kokkos kernel can replace it

Host threading regions (`#pragma omp parallel`, `std::thread`) that
wrap several calls should become a standalone function so a
**Kokkos kernel** can replace that function.

## No Kokkos in Phase 2

Standard C++ only (`-std=c++17`, no C++20). Do not add Kokkos,
`parallel_for`, or CUDA. Constructs come in Phase 4c.

Capturing a host object (or `this`) in a `KOKKOS_LAMBDA` is illegal
on the device — that is why class methods become free functions
with flat pointer arguments (see the language guide).

## Decoupled GPU entry point

The new symbol is `{function}_kokkos` and dispatches
under `config.json:include_guard` (default `USE_KOKKOS`):

```cpp
#ifdef USE_KOKKOS
  compute_interaction_kokkos(grid.dx.data(), state.h.data(), du.data(),
                             grid.is, grid.ie, grid.js, grid.je, grid.nk);
#else
  compute_interaction(grid, state, du);
#endif
```

The kernel wraps these pointers in Views internally per
`memory_mode` (see [patterns.md](patterns.md)).
