<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Standalone Runtime Setup

Use this reference when the user chooses standalone libraries instead of Kit or
when no Kit candidate is available.

## Statuses

- `ready-standalone`: standalone Usd Optimize and usd-validation-nvidia paths are
  selected and verified.
- `needs-runtime-choice`: setup cannot continue without the user choosing Kit,
  standalone, or installation.
- `blocked_missing_usd_optimize`: the user requested Usd Optimize but no
  supported SO runtime can be selected or installed.

## Usd Optimize Prompt

When standalone Usd Optimize is missing, ask before invoking
`install-usd-optimize-standalone`. The prompt must include:

- Python 3.12 hard requirement.
- Approximate download size (~350-380 MB for the prebuilt standalone package).
- Intended install location.
- Requirement for a published prebuilt Usd Optimize release package
  (asset name + download: `references/upstreams/usd-optimize.md`)
  archive path, direct archive URL, or extracted package root when no package
  root is already available.
- Usd Optimize validators auto-register into OAV via `@register_rule` decorators when
  both packages share the same Python environment — no manual enabling needed.
- Limitation that render-time profiling needs Kit.

Offer:

1. Proceed with standalone Usd Optimize install.
2. Install Kit instead.
3. Stop and produce diagnosis-only output from available evidence.

If the user proceeds and Python 3.12 is missing, install or select Python 3.12
first, then invoke `install-usd-optimize-standalone`.

## Expected Standalone Layout

Usd Optimize standalone uses:

```text
<USD_OPTIMIZE_ROOT>/.agents/operations/INDEX.md
<USD_OPTIMIZE_ROOT>/python
<USD_OPTIMIZE_ROOT>/usdpy
<USD_OPTIMIZE_ROOT>/lib
<USD_OPTIMIZE_ROOT>/extraLibs
```

Invoke `install-usd-optimize-standalone` when `USD_OPTIMIZE_ROOT`, `USD_OPTIMIZE_ROOT`,
or `WU_SO_PACKAGE_DIR` is missing or does not point at an extracted package with
the sentinel paths above. Do not clone the Usd Optimize source repository to
satisfy standalone setup.

For standalone Omni usd-validation-nvidia, invoke `install-usd-validation-nvidia-standalone`
when `omni_asset_validate` is missing. Install into the same venv that Scene
Optimizer uses — Usd Optimize validators auto-register via `@register_rule` when both
packages are importable.

Do not use the Usd Optimize package's bundled `validator-venv` as the
default usd-validation-nvidia runtime — it may lack `numpy` and is slower on large
stages.

## Handoff

After standalone setup, return to:

- `omniverse-usd-performance-tuning` for broad performance requests.
- `usd-validation-runner` for validation.
- `usd-optimize-run-operations` only after Usd Optimize operation availability is
  verified and recorded in `<output_path>/setup-preflight.json`.
