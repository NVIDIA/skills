# Runtime Probe Contract

Use this reference for setup Step 1.6 and Step 3. The probe is the only
authoritative check for Kit, Usd Optimize, usd-validation-nvidia, and operation
availability.

## Probe Outputs

The probe emits one JSON object on stdout. Free-form logs go to stderr and are
captured on disk.

Before importing `omni.asset_validator`, configure Python logging so plugin
startup messages cannot corrupt stdout:

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
| Standalone | `usd_optimize.core` | `omni.asset_validator` | `importlib.metadata.version("usd-validation-nvidia")` |

`omni.scene.optimizer.core` still resolves on 1.0.4 and 1.1.0 as a deprecated
alias of `usd_optimize.core`, but importing it emits a `DeprecationWarning`.
Use it only as a fallback for runtimes that predate the rename. The Kit
extension id remains `omni.scene.optimizer.core` — that is an extension name,
not a Python import path.

## Version Sources

Prefer these sources in order:

- **Usd Optimize (standalone):** use this fallback chain — stop at the first
  that returns a non-empty, non-`0.0.0` value. Wrap each step in its own
  `try`/`except`: a step that raises must fall through to the next one, not
  abort the probe.
  1. `usd_optimize.core.__version__` (absent on both the 1.0.4 and 1.1.0
     prebuilts).
  2. `$USD_OPTIMIZE_ROOT/CHANGELOG.md` — read the first `## <version>`
     heading (e.g. `## [1.1.0] - 2026-07-13`). Report as
     `"0.0.0+changelog:<heading>"` to signal the binding is unstamped but the
     package is identifiable. **This is the step that does the work in
     practice** — it is the only one that identifies either released package,
     so treat it as load-bearing rather than as a last-ditch fallback.
  3. If all fail, report `"unknown"` with an `errors` entry.

  No binding-level source reports the package version on either release, so do
  not add a step that asks the bindings for one.
- **usd-validation-nvidia (standalone):** `importlib.metadata.version("usd-validation-nvidia")`.
- **Kit application:** `omni.kit.app.get_app().get_app_version()`.
- **Usd Optimize (Kit):** extension manager package version for
  `omni.scene.optimizer.core`.

For supported Usd Optimize operation keys, use this fallback chain:

```python
# Preferred:
from usd_optimize.core import UsdOptimizeCore
inst = UsdOptimizeCore.getInstance()
ops = inst.getOperations()  # returns iterable of operation names

# Fallback for runtimes that predate the rename (emits DeprecationWarning):
from omni.scene.optimizer.core import SceneOptimizerCore
ops = SceneOptimizerCore.getInstance().getOperations()
```

There is no lower-level binding fallback: `usd_optimize.core.bindings` does not
exist on either release, so `acquire_interface().json_parser()` is not a usable
path.

## Success Criteria

Expect at least 40 Usd Optimize operations (both 1.0.4 and 1.1.0 report 47) and
a successful `omni.asset_validator` import.

If either probe fails, stop with `needs-runtime-choice` and ask for a different
standalone package path or a pip-installable environment. Do not route SO/AV work
to Kit. Do not pre-check extension directories as a substitute for this probe.

## Log Discipline

Follow
`skills/omniverse-usd-performance-tuning/references/runtime-artifact-token-budget.md`.
Keep full stdout/stderr files on disk. If troubleshooting is needed, inspect
structured stdout first, then show at most the last 80 stderr lines or targeted
`ERROR|WARN|exception|failed` matches.
