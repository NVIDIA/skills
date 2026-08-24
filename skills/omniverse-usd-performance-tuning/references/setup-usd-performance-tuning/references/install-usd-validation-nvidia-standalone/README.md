# Install usd-validation-nvidia Standalone

## When to Use

Use when standalone Omni usd-validation-nvidia is needed outside Kit. This installs
into the **same Python 3.12 environment** that Usd Optimize uses. Sharing an
environment is what makes the SO validator rules *available*; it does not register
them. Registration takes an explicit
`usd_optimize.validators.register_all()` — see
[so-validator-auto-registration.md](../so-validator-auto-registration.md).

## Instructions

1. Confirm Python 3.12 is available and the target environment is identified.
2. Install `usd-validation-nvidia` and `numpy` into the environment.
3. Ensure `pxr` (USD Python) is importable; if it is not already provided by SO's
   `usdpy/`, install `usd-core` so a validator-only standalone venv still gets `pxr`.
4. Verify the imports and CLI work.

## Output Format

See [§ Output](#output) below for the value list to report.

## Purpose

Install the base Omni usd-validation-nvidia runtime into a standalone Python 3.12
environment. When Usd Optimize is also on `PYTHONPATH` in this environment,
calling `usd_optimize.validators.register_all()` registers its 25 SO performance
validator rules into OAV. Importing the package does not do it on its own — see
[so-validator-auto-registration.md](../so-validator-auto-registration.md).

## Prerequisites

- Python 3.12 is available.
- Network access to a package index that provides `usd-validation-nvidia`.
  The distribution was renamed; the previous name is frozen at 1.18.0 on PyPI
  and new fixes ship only to `usd-validation-nvidia`, so install by the new
  name and query versions by it too.
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

> **Note:** `usd-validation-nvidia` does not declare `pxr` as a pip
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
python -c "import importlib.metadata as md; print('dist', md.version('usd-validation-nvidia'))"
nvidia_usd_validate --version
```

`nvidia_usd_validate` is the only console script the `usd-validation-nvidia`
distribution installs. The version lookup takes the distribution name
(`usd-validation-nvidia`), which is a different string from the console script —
`importlib.metadata.version('nvidia_usd_validate')` raises
`PackageNotFoundError`.

## SO Validator Registration

See [so-validator-auto-registration.md](../so-validator-auto-registration.md) for
the shared rule (explicit `register_all()` is required — importing the package
registers nothing, and the extracted release zip has no distribution metadata for
entry-point discovery to read; category names confirm discovery only, not
validation scope).

Once both OAV and the Usd Optimize package are importable in the same
environment, this command lists the registered categories and rule count:

```bash
python -c "
import usd_optimize.validators as V
from omni.asset_validator import CategoryRuleRegistry
print(f'Rules before register_all(): {len(list(CategoryRuleRegistry().rules))}')
V.register_all()
registry = CategoryRuleRegistry()
perf = [c for c in registry.categories if 'Performance' in c]
print(f'Usd Optimize validator categories registered: {perf}')
print(f'Total rules: {len(list(registry.rules))}')
"
```

Expected on usd-validation-nvidia 1.20.0 + usd-optimize 1.0.4: 40 rules before,
`Usd:Performance` and `Omni:Geometry` categories present after, and 65 rules
total — the 25 SO rules. If the count does not move, `register_all()` was not
reached; a bare import will not register anything.

## Output

Report these values so downstream references use the same environment:

- environment path
- Python executable path
- `nvidia_usd_validate` executable path
- `usd-validation-nvidia` version, read from distribution metadata
- `numpy` version

Then return to `setup-usd-performance-tuning` or `usd-validation-runner`.

## Troubleshooting

- If `pxr` import fails: ensure SO's `activate.sh` has been sourced (provides
  `usdpy/` on PYTHONPATH), or install `usd-core` via pip.
- If `nvidia_usd_validate` is not found on PATH, use the venv-local executable at
  `<venv>/bin/nvidia_usd_validate` (`<venv>\Scripts\nvidia_usd_validate.exe` on
  Windows). An activated venv is the usual reason PATH lookup succeeds; a
  non-activated one still has the executable at that path.
- If package resolution fails, use the user's organization-approved pip
  configuration rather than adding an unapproved index URL.
