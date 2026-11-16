# Device Utilities (Phase 3) — C/C++ + Kokkos

Flavor pack: `flavors/cpp/kokkos/`. The `translate-utility` subskill
reads this file for **Phase 3**. Per-function translation (Phase 4)
lives in [translation.md](translation.md); directory layout and the
`{parent}/kokkos/` path rules are documented there.

**Skip if** `porting_plan.json` has no `device_utilities`.

## Purpose

Functions called from inside a translation target's kernels must run
on-device. This includes:

- **Small helpers** — index calculations, math helpers, boundary
  functions
- **Nested compute callees** — functions with their own loops (e.g. a
  `(j,i)` column kernel) that are only reached from inside a parent
  target's parallel region. These become `KOKKOS_INLINE_FUNCTION`
  device functions, not separate kernel launches.

Instead of annotating the originals in-place (which mixes concerns and
causes duplication when multiple kernels share the same routine),
create device versions in **one header per parent directory**, test
them against the originals, and let kernels include that parent’s
header.

## Step 1: Read the plan

Read `device_utilities` from `porting_plan.json`. Each entry has:
- `original_file`, `original_module`: where the host version lives
- `function`: original function name
- `device_name`: name for the device version (convention:
  `{function}_device`)
- `used_by`: which translation targets call this function

## Step 2: Write `{parent}/kokkos/device_utils.hpp`

Group utilities by the **parent directory of `original_file`** from
the plan. For each distinct parent directory, create
`{parent}/kokkos/device_utils.hpp` containing every device routine
whose host version lives under `{parent}/`. Make it a header-only
module of `KOKKOS_INLINE_FUNCTION` free functions:

```cpp
// src/utils/kokkos/device_utils.hpp
#pragma once
#include <Kokkos_Core.hpp>

KOKKOS_INLINE_FUNCTION
int periodic_index_device(int i, int N) {
  return ((i % N) + N) % N;
}

// ... more device utilities as needed ...
```

Rules:
- `KOKKOS_INLINE_FUNCTION` goes **before** the return type
  (`__host__ __device__ inline`).
- The directive is NOT guarded by `#ifdef` — this header is only
  included by GPU-only files / the device-utils test built with
  `-DUSE_KOKKOS`.
- Function signatures must match the originals exactly (same
  arguments, same types). Only the name changes (`_device` suffix).
- Keep them header-only (`inline`) so every including translation unit
  gets the device definition.
- If the original uses constants/types from another header, `#include`
  those at the top of `device_utils.hpp`.

## Step 3: Write the test

Create
`{parent}/kokkos/test/test_device_utils/test_device_utils.cpp` (one
test program per `{parent}/kokkos/` that has utilities):

```cpp
#include <Kokkos_Core.hpp>
#include <cstdio>
#include "kernel_utils.hpp"      // host versions
#include "device_utils.hpp"      // device versions

int main(int argc, char* argv[]) {
  Kokkos::ScopeGuard guard(argc, argv);
  int failures = 0;
  const int N = 256;

  // Test periodic_index vs periodic_index_device (host-callable too)
  for (int i = -10; i < N + 10; ++i) {
    int host_result = periodic_index(i, N);
    int device_result = periodic_index_device(i, N);
    if (host_result != device_result) {
      std::printf("MISMATCH periodic_index: i=%d host=%d device=%d\n",
                  i, host_result, device_result);
      ++failures;
    }
  }
  // ... repeat for each device utility ...

  if (failures > 0) {
    std::printf("DEVICE UTILS TEST FAILED: %d mismatches\n", failures);
    return 1;
  }
  std::printf("DEVICE UTILS TEST PASSED\n");
  return 0;
}
```

`KOKKOS_INLINE_FUNCTION` functions are `__host__ __device__`, so the
host can call them directly — this validates the rewrite is faithful
before any kernel uses it. For a stronger check you may also launch a
`parallel_for` that calls the device version into a View and compares.

For floating-point utilities, use the same tolerances as kernel tests
(double: 1e-12 abs / 1e-10 rel; single: 1e-5 / 1e-4).

## Step 4: Hook into the build system

Add build rules for the device-utils test and ensure
`{parent}/kokkos/` is on the include path of any kernel files that
`#include "device_utils.hpp"`. Because device utilities are
header-only, there is no separate `.o` to order — just include paths.

## Step 5: Verify

```bash
source .nsight-ai-port/env.sh && \
  USE_KOKKOS=1 bash .nsight-ai-port/build_correctness.sh
./test_device_utils
```

Both must pass.

## Step 6: Commit and update progress

```bash
git add -A
git commit -m "device utils: shared device functions + tests"
```

**Subagent opportunity.** This phase has a clear input (list of
functions from the plan) and a clear success criterion (test passes).
Good candidate for subagent delegation.
