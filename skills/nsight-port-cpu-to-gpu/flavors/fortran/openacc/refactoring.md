# Refactoring addendum — Fortran + OpenACC

Read [../refactoring.md](../refactoring.md) first. This file is only
the OpenACC-specific facts. `refactor-task` opens it when
`from_lang == to_lang` and `framework` is `openacc`.

## Host OpenMP regions

An `!$OMP PARALLEL` region that wraps multiple CALL statements
should be extracted into a standalone subroutine so **OpenACC** can
replace it. Do not leave the compute inside a host OpenMP parallel
region that Phase 4c would have to nest `!$acc` inside.

## No OpenACC in Phase 2

Structural Fortran only. Do not add `!$acc` (or any GPU syntax).
Directives come in Phase 4c.

## Decoupled GPU entry point

When flattening is impossible, the new symbol uses
`{function}_openacc` and the host wrapper dispatches
under `config.json:include_guard` (default `USE_OPENACC`):

```fortran
#ifdef USE_OPENACC
  call compute_interaction_openacc(grid%dx, grid%dy, h, u, v, du, dv, &
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
**OpenACC** gang/vector mapping. See "Mapping loop nests to gang /
worker / vector" in [patterns.md](patterns.md).
