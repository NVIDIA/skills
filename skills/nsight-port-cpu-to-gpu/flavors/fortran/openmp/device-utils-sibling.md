# Device Utilities (Phase 3) — Sibling Mode — Fortran + OpenMP offload

**Read this file only when `config.json:translation_mode` is
`"sibling"`** (the default). For `"inline"` mode read
[device-utils-inline.md](device-utils-inline.md). The router
[device-utils.md](device-utils.md) selects between them.

Per-function translation (Phase 4) lives in
[translation-sibling.md](translation-sibling.md); directory layout and
the `{parent}/openmp/` path rules are documented there.

**Skip if** `porting_plan.json` has no `device_utilities`.

## Purpose

Functions called from inside a translation target's `target`
regions must run on-device. This includes:

- **Small helpers** — index calculations, math helpers,
  boundary functions
- **Nested compute callees** — subroutines with their own
  loops (e.g. a `(j,i)` column kernel) that are only reached
  from inside a parent target's loop nest. These are device
  routines, not separate offload regions.

Do not annotate the originals in-place. Instead, create device 
versions in one module per parent directory, test them against 
the originals, and let kernels `USE` that parent’s module.

## Step 1: Read the plan

Read `device_utilities` from `porting_plan.json`. Each entry
has:
- `original_file`, `original_module`: where the host version
  lives
- `function`: original function name
- `device_name`: name for the device version (convention:
  `{function}_device`)
- `used_by`: which translation targets call this function

## Step 2: Write `{parent}/openmp/device_utils.f90`

Group utilities by the **parent directory of `original_file`**
from the plan. For each distinct parent directory, create
`{parent}/openmp/device_utils.f90` containing every device
routine whose host version lives under `{parent}/`:

Create one module per parent. Name it
`{parent_id}_device_utils_mod`. Build `{parent_id}` from
`{parent}` relative to the source root:

1. Map every character that is not `A–Z`, `a–z`, or `0–9` to
   `_` (so `/`, `-`, `.` all become `_`).
2. Collapse repeated `_` and strip leading/trailing `_`.
3. If the result is empty or starts with a digit, prefix `p`.
4. If two parents in this plan produce the same string, append
   `_` plus the first 8 hex digits of SHA-256 of the original
   relative path (UTF-8) to **each** colliding `{parent_id}`:
   ```bash
   python3 -c "import hashlib; print(
     hashlib.sha256(b'REL_PATH').hexdigest()[:8])"
   ```
5. Truncate `{parent_id}` from the **right** so the full module
   name is at most 63 characters. If truncation makes two parents
   share a name, go to step 4. Never cut `_device_utils_mod` or a
   hash suffix.

Examples: `src/physics` → `src_physics_device_utils_mod`;
`src/physics-core` → `src_physics_core_device_utils_mod`.
Do **not** use a fixed `device_utils_mod` — Fortran `.mod`
files usually share one module directory, so two parents
would collide.

```fortran
module src_physics_device_utils_mod
  use iso_fortran_env, only: dp => real64
  implicit none
contains

  function periodic_index_device(i, N) result(idx)
    !$omp declare target
    integer, intent(in) :: i, N
    integer :: idx
    idx = modulo(i, N)
  end function periodic_index_device

  ! ... more device utilities as needed ...

end module src_physics_device_utils_mod
```

Rules:
- Each function gets `!$omp declare target` **inside** the
  routine body (first line after the `function`/`subroutine`
  statement, before `implicit none` and declarations).
  Do NOT place it before the function statement.
- The `!$omp declare target` is NOT guarded by `#ifdef` — this
  file is only compiled in the GPU build path.
- Function signatures must match the originals exactly (same
  arguments, same types, same intent). Only the name changes.
- If the original function uses module-level constants or
  types, `USE` those modules at the top of this parent’s
  device-utils module.

## Step 3: Write the test

Create `{parent}/openmp/test/test_device_utils/test_device_utils.F90`
(one test program per `{parent}/openmp/` that has utilities):

```fortran
program test_device_utils
  use kernel_utils_mod     ! host versions
  use src_physics_device_utils_mod  ! this parent’s device versions
  implicit none
  integer :: i, N, host_result, device_result
  integer :: failures

  failures = 0
  N = 256

  ! Test periodic_index vs periodic_index_device
  do i = -10, N + 10
    host_result = periodic_index(i, N)
    device_result = periodic_index_device(i, N)
    if (host_result /= device_result) then
      print *, 'MISMATCH periodic_index: i=', i, &
               ' host=', host_result, &
               ' device=', device_result
      failures = failures + 1
    end if
  end do

  ! ... repeat for each device utility ...

  if (failures > 0) then
    print *, 'DEVICE UTILS TEST FAILED:', failures, &
             'mismatches'
    stop 1
  end if

  print *, 'DEVICE UTILS TEST PASSED'
end program
```

The test runs on the CPU — `!$omp declare target` functions are
valid Fortran even without `-mp=gpu`. This validates the rewrite
is faithful before any kernel uses it.

For floating-point utilities, use the same tolerances as kernel
tests (double: 1e-12 abs / 1e-10 rel; single: 1e-5 / 1e-4).

## Step 4: Hook into the build system

Add build rules for each `{parent}/openmp/device_utils.f90` and
its test. Ensure each `device_utils.o` is compiled before kernel
files in the same or dependent `openmp/` trees that `USE` it.

## Step 5: Verify

```bash
source .nsight-ai-port/env.sh && \
  USE_OPENMP=1 bash .nsight-ai-port/build_correctness.sh
./test_device_utils
```

Both must pass.

## Step 6: Commit and update progress

```bash
git add -A
git commit -m "device utils: shared device functions + tests"
```

**Subagent opportunity.** This phase has a clear input (list
of functions from the plan) and a clear success criterion
(test passes). Good candidate for subagent delegation.
