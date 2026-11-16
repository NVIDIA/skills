# Device Utilities (Phase 3) — C/C++ + CUDA

Flavor pack: `flavors/cpp/cuda/`. The `translate-utility` subskill
reads this file for **Phase 3**. Per-function translation (Phase 4)
lives in [translation.md](translation.md); directory layout and the
`{parent}/cuda/` path rules are documented there.

**Skip if** `porting_plan.json` has no `device_utilities`.

## Purpose

Functions called from inside a translation target's kernels must run
on-device. This includes:

- **Small helpers** — index calculations, math helpers, boundary
  functions.
- **Nested compute callees** — functions with their own loops only
  reached from inside a parent target's kernel. These become
  `__device__` functions, not separate kernel launches.

Instead of annotating originals in-place, create device versions in
**one header per parent directory**, test them against the originals,
and let kernels include that parent’s header.

## Step 1: Read the plan

Read `device_utilities` from `porting_plan.json`. Each entry has
`original_file`, `original_module`, `function`, `device_name`
(convention `{function}_device`), and `used_by`.

## Step 2: Write `{parent}/cuda/device_utils.cuh`

Group utilities by the **parent directory of `original_file`**. For
each distinct parent directory create `{parent}/cuda/device_utils.cuh`
containing every device routine whose host version lives under
`{parent}/`, as `__host__ __device__ inline` free functions:

```cpp
// src/utils/cuda/device_utils.cuh
#pragma once

__host__ __device__ inline
int periodic_index_device(int i, int N) {
  return ((i % N) + N) % N;
}

// ... more device utilities as needed ...
```

Rules:
- `__host__ __device__ inline` makes them callable from both the test
  (host) and kernels (device) so the device-utils test can compare.
- NOT guarded by `#ifdef` — the header is only included by GPU-only
  `.cu` files / the device-utils test built with `-DUSE_CUDA`.
- Signatures must match the originals exactly (only the name changes).
- Keep header-only (`inline`) so every including `.cu` gets the
  device definition.

## Step 3: Write the test

Create
`{parent}/cuda/test/test_device_utils/test_device_utils.cpp` (one per
`{parent}/cuda/` that has utilities). Because the helpers are
`__host__ __device__`, the host can call them directly:

```cpp
#include <cstdio>
#include "kernel_utils.hpp"        // host versions
#include "device_utils.cuh"        // device versions (host-callable)

int main() {
  int failures = 0;
  const int N = 256;
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

This file includes a `.cuh`, so build it with `nvcc` (or rename to
`.cu`). For a stronger check you may also launch a kernel that calls
the device version into a device buffer and compares. For
floating-point utilities use kernel-test tolerances (double: 1e-12 abs
/ 1e-10 rel; single: 1e-5 / 1e-4).

## Step 4: Hook into the build system

Add a build rule for the device-utils test (compiled by `nvcc`) and
ensure `{parent}/cuda/` is on the include path of kernel files that
`#include "device_utils.cuh"`. Header-only utilities need no separate
`.o` — just include paths.

## Step 5: Verify

```bash
source .nsight-ai-port/env.sh && \
  USE_CUDA=1 bash .nsight-ai-port/build_correctness.sh
./test_device_utils
```

Both must pass.

## Step 6: Commit and update progress

```bash
git add -A
git commit -m "device utils: shared device functions + tests"
```

**Subagent opportunity.** Clear input (list of functions) and clear
success criterion (test passes).
