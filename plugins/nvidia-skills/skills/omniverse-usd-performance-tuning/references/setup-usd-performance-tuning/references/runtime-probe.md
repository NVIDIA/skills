<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Runtime Probe Contract

Use this reference for setup Step 1.6 and Step 3. The probe is the only
authoritative check for Kit, Usd Optimize, usd-validation-nvidia, and operation
availability.

## Probe Outputs

The probe emits one JSON object on stdout. Free-form logs go to stderr and are
captured on disk.

Before importing `omni.asset_validator` or `omni.asset_validator.core`,
configure Python logging so plugin startup messages cannot corrupt stdout:

```python
import logging
import sys

logging.basicConfig(stream=sys.stderr, force=True)
```

If INFO-level plugin logs are needed for troubleshooting, set
`level=logging.INFO` in the same call; keep stdout reserved for the JSON object.

Required blocks:

- `kit`: chosen application, version, build, path, launcher.
- `usdOptimize`: extension/package name, version, operation count,
  `operationsAvailable`, and source.
- `assetValidator`: package/extension name, version, and source.
- `runtime_context`: mirror of the user-facing values consumed by
  `runtime-context-header.md`.

`operationsAvailable` must come from the live runtime and must be sorted. Do
not hand-copy operation keys from a snapshot.

Note: `probe-snapshot.schema.json` (flat fixture, snake_case `operations_available`) is a curation reference for version comparison — it is a different artifact from `setup-preflight.json` (nested runtime config, camelCase `usdOptimize.operationsAvailable`) which is the agent's runtime output consumed by downstream phases.

## Launchers

Use the launcher selected during Kit discovery:

- Classic Windows Kit: `<kit>\python.bat`
- Classic Linux Kit: `<kit>/python.sh` or `<kit>/python`
- Windows Kit venv: `<venv>\Scripts\python.exe`
- Linux Kit venv: `<venv>/bin/python`

Set `OMNI_KIT_ACCEPT_EULA=yes`. Start Kit with `--no-window`,
`--enable omni.scene.optimizer.core`, and
(validator runs from the pip package, not a Kit extension).

## Import Modes

The validator always imports standalone (`omni.asset_validator` from the usd-validation-nvidia pip package); the Kit validator extension is not a supported path.

| Mode | Usd Optimize import | AV import | AV version |
|---|---|---|---|
| Standalone | `omni.scene.optimizer.core` | `omni.asset_validator` | `importlib.metadata.version("omniverse-asset-validator")` |

Usd Optimize uses the same import in both modes. usd-validation-nvidia is the
asymmetric case.

## Version Sources

Prefer these sources in order:

- **Usd Optimize (standalone):** use this fallback chain — stop at the first
  that returns a non-empty, non-`0.0.0` value:
  1. `omni.scene.optimizer.core.__version__` (may not exist on prebuilts).
  2. `omni.scene.optimizer.impl.core.SOPluginVersion()` →
     `"{major}.{minor}.{rev}"`. If all three are `0`, treat as unstamped.
  3. `$USD_OPTIMIZE_ROOT/CHANGELOG.md` — read the first `## <version>`
     heading (e.g. `## 110.0.5 — 2026-06-01`). Report as
     `"0.0.0+changelog:<heading>"` to signal the binding is unstamped but the
     package is identifiable.
  4. If all fail, report `"unknown"` with an `errors` entry.
- **usd-validation-nvidia (standalone):** `importlib.metadata.version("omniverse-asset-validator")`.
- **Kit application:** `omni.kit.app.get_app().get_app_version()`.
- **Usd Optimize (Kit):** extension manager package version for
  `omni.scene.optimizer.core`.
- **usd-validation-nvidia (Kit):** extension manager package version for
  `omni.asset_validator.core`.

For supported Usd Optimize operation keys, use this fallback chain:

```python
# Preferred:
from omni.scene.optimizer.core import SceneOptimizerCore
inst = SceneOptimizerCore.getInstance()
ops = inst.getOperations()  # returns iterable of operation names

# Fallback for lower-level binding-only builds:
omni.scene.optimizer.core.bindings._omni_scene_optimizer_core \
    .acquire_interface().json_parser().get_supported_operations()
```

## Success Criteria

Expect at least 40 Usd Optimize operations and a successful
`omni.asset_validator.core` import for Kit-mode validation.

If either probe fails, ask for another path or fall back to Kit.
Do not pre-check extension directories as a substitute for this probe.

## Log Discipline

Follow
`skills/omniverse-usd-performance-tuning/references/runtime-artifact-token-budget.md`.
Keep full stdout/stderr files on disk. If troubleshooting is needed, inspect
structured stdout first, then show at most the last 80 stderr lines or targeted
`ERROR|WARN|exception|failed` matches.
