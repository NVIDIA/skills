# Device Utilities (Phase 3) — Inline Mode — Fortran + OpenMP offload

**Read this file only when `config.json:translation_mode` is
`"inline"`.** For the default `"sibling"` mode read
[device-utils-sibling.md](device-utils-sibling.md). The router
[device-utils.md](device-utils.md) selects between them.

Per-function translation (Phase 4) lives in
[translation-inline.md](translation-inline.md).

**Skip if** `porting_plan.json` has no `device_utilities`.

Do **not** create the separate `device_utils.f90` module used in
sibling mode. Instead, annotate each device-callee in its
**original** file with `!$omp declare target` as the first line
inside the routine body (before `implicit none` and declarations):

```fortran
function periodic_index(i, N) result(idx)
  !$omp declare target
  integer, intent(in) :: i, N
  integer :: idx
  idx = modulo(i, N)
end function periodic_index
```

`!$omp declare target` is an inert comment on the CPU path. The
reference copy calls these same annotated originals, so CPU results
are unaffected. Re-run the project build to confirm the annotated
callees compile on both paths, then commit.
