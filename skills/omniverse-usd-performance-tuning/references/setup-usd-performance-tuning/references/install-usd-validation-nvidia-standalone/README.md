# Install usd-validation-nvidia Standalone

<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

## When to Use

Use when standalone Omni usd-validation-nvidia is needed outside Kit. This installs
into the **same Python 3.12 environment** that Usd Optimize uses. The SO
validator rules auto-register via `@register_rule` decorators when both
packages share a Python environment — no manual enabling required.

## Instructions

1. Confirm Python 3.12 is available and the target environment is identified.
2. Install `omniverse-asset-validator` and `numpy` into the environment.
3. Ensure `pxr` (USD Python) is importable; if it is not already provided by SO's
   `usdpy/`, install `usd-core` so a validator-only standalone venv still gets `pxr`.
4. Verify the imports and CLI work.

## Output Format

Return a concise status naming the environment path, Python executable,
`omni_asset_validate` version, and `numpy` version.

## Purpose

Install the base Omni usd-validation-nvidia runtime into a standalone Python 3.12
environment. When Usd Optimize is also on `PYTHONPATH` in this environment,
`import omni.scene.optimizer.validators` triggers `@register_rule` decorators
that register 25 SO performance validator rules into OAV automatically.

## Prerequisites

- Python 3.12 is available.
- Network access to a package index that provides `usd-validation-nvidia`
  (the renamed usd-validation-nvidia package; the old `omniverse-asset-validator`
  name is frozen at 1.18.0 on PyPI — new fixes ship only to the new name).
- The SO standalone package is already extracted (via `install-usd-optimize-standalone`)
  or will be set up afterward — order does not matter as long as both are
  importable in the same environment at runtime.

## Install

Use the **same venv** that Usd Optimize will use. If `install-usd-optimize-standalone`
already created a venv, reuse it. Otherwise create one:

Linux:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install usd-validation-nvidia numpy
# Guarantee pxr (USD Python): if SO's usdpy/ is not already on PYTHONPATH,
# install usd-core. This makes `import pxr` succeed in a validator-only venv.
python -c "import pxr" 2>/dev/null || python -m pip install usd-core
```

Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install usd-validation-nvidia numpy
# Guarantee pxr (USD Python): install usd-core only if SO does not already provide it.
python -c "import pxr" 2>$null; if ($LASTEXITCODE -ne 0) { python -m pip install usd-core }
```

> **Note:** `omniverse-asset-validator` does not declare `pxr` as a pip
> dependency. The SO standalone package provides `pxr` via its `usdpy/`
> directory on `PYTHONPATH`; when SO is present, do not double-install. The rule
> is **ensure `pxr` is importable** — if it is not (e.g. a validator-only
> standalone venv with no SO yet), `pip install usd-core` provides it. After this
> step `import pxr` must succeed.

## Verify

```bash
python -c "import omni.asset_validator; print('OAV', omni.asset_validator.__version__)"
python -c "import numpy; print('numpy', numpy.__version__)"
python -c "import pxr; print('pxr OK')"
omni_asset_validate --version
```

## SO Validator Auto-Registration

Once both OAV and the Usd Optimize package are importable in the same environment:

```bash
python -c "
import omni.scene.optimizer.validators
from omni.asset_validator import CategoryRuleRegistry
registry = CategoryRuleRegistry()
perf = [c for c in registry.categories if 'Performance' in c]
print(f'Usd Optimize validator categories registered: {perf}')
print(f'Total rules: {len(list(registry.rules))}')
"
```

Expected: `Usd:Performance` and `Omni:Geometry` categories appear with ~25
additional rules. No `register_all()` call is needed for rule discovery: the
validator registration decorators handle registration at import time. Category
names confirm discovery only; `usd-validation-runner` selects validators by
canonical concept and resolves them to rule classes by identity (via
`scripts/usd_validation_executor.py`) before calling `enable_rule()`.

## Output

Report these values so downstream references use the same environment:

- environment path
- Python executable path
- `omni_asset_validate` executable path and version
- `numpy` version

Then return to `setup-usd-performance-tuning` or `usd-validation-runner`.

## Troubleshooting

- If `pxr` import fails: ensure SO's `activate.sh` has been sourced (provides
  `usdpy/` on PYTHONPATH), or install `usd-core` via pip.
- If `omni_asset_validate` is not found on PATH, use the venv-local executable.
- If package resolution fails, use the user's organization-approved pip
  configuration rather than adding an unapproved index URL.
