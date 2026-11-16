# Refactoring addendum — Fortran + OpenMP offload

Read [../refactoring.md](../refactoring.md) first. This file is only
the OpenMP-offload-specific facts. `refactor-task` opens it when
`from_lang == to_lang` and `framework` is `openmp`.

## Host OpenMP vs `target`

An existing host-multithreading `!$omp parallel` region (CPU
threads, **not** offload) that wraps multiple CALL statements
should be extracted into a standalone subroutine so it can become
a single `!$omp target` offload region.

**Do not nest `target` regions inside leftover host `parallel`
regions.**

## No offload in Phase 2

Structural Fortran only. Do not add `!$omp target` (or any GPU
syntax). Directives come in Phase 4c.

## Decoupled GPU entry point

When flattening is impossible, the new symbol uses
`{function}_openmp` and the host wrapper dispatches
under `config.json:include_guard` (default `USE_OPENMP`):

```fortran
#ifdef USE_OPENMP
  call compute_interaction_openmp(grid%dx, grid%dy, h, u, v, du, dv, &
       grid%is, grid%ie, grid%js, grid%je, grid%nk)
#else
  call compute_interaction(grid, vgrid, h, u, v, du, dv)
#endif
```

See "GPU entry points and derived types" in
[translation-sibling.md](translation-sibling.md) (or "Derived-type
arguments (inline)" in [translation-inline.md](translation-inline.md)).

## Stride on the GPU path

A GPU translation with `stride > 1` fuses outer/inner dims for
**OpenMP** teams/distribute mapping. See [patterns.md](patterns.md).
