---
name: cuopt-installation-api-c
version: "26.04.00"
description: Install cuOpt for C — conda, locate lib/headers, verification. Use when the user is installing or verifying the C API. Standalone; no common skill.
---

# cuOpt Installation — C API (user)

Install cuOpt to *use* it from C. Standalone skill (no separate common).

## System requirements

- **GPU**: NVIDIA Compute Capability ≥ 7.0 (Volta+). CUDA 12.x or 13.x.
- **Driver**: Compatible NVIDIA driver. Python and C are separate installables.

## conda (C / libcuopt)

```bash
conda install -c rapidsai -c conda-forge -c nvidia cuopt
# libcuopt is provided by the same channel; Python and C are separate packages.
```

## Verify C API

```bash
find $CONDA_PREFIX -name "cuopt_c.h"
find $CONDA_PREFIX -name "libcuopt.so"
```

## Examples

- [verification_examples.md](resources/verification_examples.md) — C API verification
