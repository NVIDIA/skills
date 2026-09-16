# Install Usd Optimize Standalone

## When to Use

Use when SO core operations or packaged Usd Optimize validator rules are needed outside Kit.

## Instructions

See `references/_shared/standard-instructions.md`.

## Output Format

See `references/_shared/standard-output-format.md`.

## Purpose

PyPI wheel isn't released yet; this reference consumes a prebuilt
`usd_optimize_...release.zip` package from GitHub Releases. Do not clone the Scene
Optimizer source repo, run `repo.sh`, or depend on repo helper wrappers for
standalone runtime setup. Sizes depend on the USD flavor: the 1.1.0
`usd_25.11` Linux x86_64 asset is ~174 MiB (~650 MB extracted) and the 1.0.4
Linux x86_64 asset is ~331 MiB (~920 MB extracted). Download + extract takes
~1-2 min on a fast connection. EULA env var **not** needed (no Kit).

Use this reference for standalone Usd Optimize core operations and the
packaged `omni.scene.optimizer.validators` rules when a Kit runtime is
unavailable or not desired. For validator execution, pair this package with a
project-managed `usd-validation-nvidia` environment that can import the
same SO package. Kit remains useful when automatic extension registration or
render-time profiling is needed.

This install reference does not define operation invocation. Keep operation
execution examples in `usd-optimize-run-operations/references/invocation.md` so agents
have one source of truth.

## Prerequisites

> **Python 3.12 is a HARD requirement.** The drop ships `cp312`-only wheels.
> There is no `abi3`, no `cp310`/`cp311`/`cp313` fallback, and no source
> build path here. Installing under any other Python will appear to succeed
> until the first `import usd_optimize.core`, which fails with a
> cryptic ABI error. Verify `python3.12 --version` **before** downloading
> the zip.

```bash
python3.12 --version            # required — package is cp312-only, no fallback
command -v unzip                # preferred extractor on Linux (Windows: Expand-Archive)
```

If either is missing, install before continuing
(`apt-get install python3.12 unzip` on Debian/Ubuntu; on systems without a
3.12 package, `uv python install 3.12` is also fine but see the
*uv-managed Python* note in Step 4).

## Step 2 — Pick Archive or Extracted Root by Platform

Use a user-provided package archive path, direct archive URL, or extracted
package root when supplied. Do not clone the source repository.
If an extracted package root is supplied and it has the sentinel paths listed
under Package Version, set `USD_OPTIMIZE_ROOT` and `USD_OPTIMIZE_ROOT` to that
root and skip the download/extract steps.

Prebuilt packages are published as **GitHub release assets** on
[NVIDIA-Omniverse/usd-optimize](https://github.com/NVIDIA-Omniverse/usd-optimize/releases)
(Linux x86_64, Linux aarch64, Windows x86_64). The latest release is **1.1.0**:

```bash
gh release list -R NVIDIA-Omniverse/usd-optimize          # confirm the latest tag
gh release download v1.1.0 -R NVIDIA-Omniverse/usd-optimize \
  -p 'usd_optimize_usd_25.11_py_3.12@*manylinux_2_35_x86_64.release.zip'
# or browse: https://github.com/NVIDIA-Omniverse/usd-optimize/releases
```

Pin the USD flavor in the pattern. 1.1.0 publishes both `usd_25.05` and
`usd_25.11` builds per platform, so a bare `-p '*manylinux*x86_64*'` matches two
assets and downloads ~1.1 GB. Use `usd_25.11` unless a caller asks for 25.05.

Auto-pick the platform by `uname -s`/`-m`. Without `gh`, use the asset's browser
URL from the releases page (no URL-encoding gymnastics needed).

## Step 3 — Pick install location

Ask the user to choose:

- **Per-user (default):** `~/scene-optimizer/` — shared across
  projects, downloaded once. Same literal on Linux/Windows shells.
- **Project-local:** `$(pwd)/packages/scene-optimizer/` — isolated to
  this CWD.

## Step 4 — Download, extract, configure

Use this step only for a direct archive path or URL.

```bash
export SO_PACKAGE=<direct archive path or URL>
export USD_OPTIMIZE_ROOT=<chosen path>
mkdir -p "$USD_OPTIMIZE_ROOT"
case "$SO_PACKAGE" in
  http://*|https://*) curl -L "$SO_PACKAGE" -o "$USD_OPTIMIZE_ROOT/usd_optimize_package.zip" ;;
  *) cp "$SO_PACKAGE" "$USD_OPTIMIZE_ROOT/usd_optimize_package.zip" ;;
esac
cd "$USD_OPTIMIZE_ROOT"
python3.12 - <<'PY'
import zipfile

archive = "usd_optimize_package.zip"
if not zipfile.is_zipfile(archive):
    raise SystemExit(
        f"{archive} is not a zip archive; set SO_PACKAGE to a direct .zip "
        "archive path or URL and retry"
    )
PY
unzip -q usd_optimize_package.zip

cat > "$USD_OPTIMIZE_ROOT/activate.sh" <<'EOF'
export USD_OPTIMIZE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export USD_OPTIMIZE_ROOT="$USD_OPTIMIZE_ROOT"
export PYTHONPATH="$USD_OPTIMIZE_ROOT/python:$USD_OPTIMIZE_ROOT/usdpy:$PYTHONPATH"
export LD_LIBRARY_PATH="$USD_OPTIMIZE_ROOT/lib:$USD_OPTIMIZE_ROOT/extraLibs:$LD_LIBRARY_PATH"

# uv-managed Python 3.12 ships libpython3.12.so.1.0 outside the system
# loader path. Prepend the chosen interpreter's lib dir so SO's C++
# extensions can dlopen it. No-op when the interpreter is a system Python.
_so_pylib="$(python3.12 -c 'import sys, os; print(os.path.join(sys.base_prefix, "lib"))' 2>/dev/null)"
if [ -n "$_so_pylib" ] && [ -d "$_so_pylib" ]; then
    export LD_LIBRARY_PATH="$_so_pylib:$LD_LIBRARY_PATH"
fi
unset _so_pylib
EOF
source "$USD_OPTIMIZE_ROOT/activate.sh"
```

Env vars are **session-scoped**. Re-source `$USD_OPTIMIZE_ROOT/activate.sh` in
any new shell.

> **uv-managed Python 3.12.** When `python3.12` was installed via
> `uv python install 3.12`, `libpython3.12.so.1.0` lives under
> `~/.local/share/uv/python/cpython-3.12.*/lib/` and is **not** on the
> default loader path. Without the snippet above, the first Usd Optimize import fails
> with `ImportError: libpython3.12.so.1.0: cannot open shared object
> file`. The `_so_pylib` block in `activate.sh` derives the right
> directory from `sys.base_prefix` so it works for both uv-managed and
> system Pythons.

On Windows: write `activate.bat` instead, using
`set USD_OPTIMIZE_ROOT=%USD_OPTIMIZE_ROOT%` and
`set PATH=%USD_OPTIMIZE_ROOT%\lib;%USD_OPTIMIZE_ROOT%\extraLibs;%PATH%` (no `LD_LIBRARY_PATH`).
Windows resolves `python312.dll` through the launcher that started the
process, so the uv-managed-Python caveat above does not apply.

## Step 5 — Verify

```bash
python3.12 - <<'PY'
def operation_count():
    try:
        from usd_optimize.core import UsdOptimizeCore

        return "UsdOptimizeCore.getInstance", len(UsdOptimizeCore.getInstance().getOperations())
    except Exception:
        pass

    from omni.scene.optimizer.core import SceneOptimizerCore

    return "SceneOptimizerCore.getInstance", len(SceneOptimizerCore.getInstance().getOperations())

surface, count = operation_count()
print(f"{surface}: {count} operations")
PY
```

Expect >= 40; both 1.0.4 and 1.1.0 report **47**. Prefer the `usd_optimize`
namespace: `omni.scene.optimizer` still imports on both versions but emits a
`DeprecationWarning`, and `SceneOptimizerCore` warns in favor of
`UsdOptimizeCore`. This verifies import and operation registry only. Operation
invocation is defined by `usd-optimize-run-operations/references/invocation.md`;
do not infer mutation call shapes from this install probe.

## Limitations

The standalone package supports analysis-mode operations — set
`ExecutionContext.analysisMode = 1` to get per-operation findings without the
full validator engine.

The drop may include a bundled `validator-venv/`. Do not use it as the default
runtime — it may lack `numpy` and is slower on large stages. Use a
project-managed venv with `install-usd-validation-nvidia-standalone` instead.

## SO Validator Registration

See [so-validator-auto-registration.md](../so-validator-auto-registration.md) for
the shared rule on category names confirming discovery only, not validation scope.

**Registration is not automatic.** Measured on both 1.0.4 and 1.1.0: importing
`usd_optimize.validators` registers **zero** rules. The `@register_rule`
decorators are applied inside `register_all()`, not at module import, and the
extracted release zip ships no distribution metadata (no `*.dist-info` /
`entry_points.txt`), so entry-point discovery cannot find the plugin either.
Call it explicitly:

```python
import usd_optimize.validators as sov

sov.register_all()   # returns the 25 rule classes it registered
```

That takes the registry from 40 rules to 65: 19 added under `Usd:Performance`
and 6 under `Omni:Geometry`. Identical on 1.0.4 and 1.1.0.

To verify the install can run a scoped concept after `usd-validation-runner`
has scoped the plan:

```python
from usd_validation_executor import load_registry, validate_concepts

registry = load_registry()
issues = validate_concepts(
    "path/to/asset.usd",
    ["primvar_indexability"],     # canonical concept from the scope note
    registry=registry,
)
```

The executor builds the engine with `init_rules=False` and enables only the
resolved rule class.

The standalone import is `from omni.asset_validator import ValidationEngine`
(no `.core`). The `.core` submodule only exists inside a running Kit session.

## Package Version

Current expected package family (Kit 110.1 parity):

```
usd_optimize_usd_25.11_py_3.12 (version 1.1.0, <platform>.release.zip)
```

1.0.4 remains the minimum supported runtime. Both are verified; the layouts
differ only in where per-operation docs live.

Expected layout after unpack (1.1.0):

```
$USD_OPTIMIZE_ROOT/
├── .agents/     # Agent skills (run-operations, run-validators, ...)
├── bin/         # Native usdOptimize CLI (1.1.x only)
├── docs/        # Install notes + operations/<key>.rst per-op guides (1.1.x)
├── python/      # Python modules (usd_optimize.*, omni.scene.optimizer.* shim)
├── usdpy/       # USD Python bindings (pxr.*)
├── lib/         # Core shared libraries
└── extraLibs/   # Additional dependencies
```

On 1.0.4 there is no `bin/`, per-op guides live at `.agents/operations/<key>.md`,
and `docs/` holds install notes only. `.agents/` is present on both.

Sentinel check (all runtime dirs plus agent docs must exist for a valid install):

```bash
for sub in .agents python lib extraLibs usdpy; do
    [[ -d "$USD_OPTIMIZE_ROOT/$sub" ]] || echo "MISSING: $sub"
done
# Per-operation docs: docs/operations.rst (1.1.x packages) or .agents/operations/INDEX.md (1.0.x). Valid if either exists.
[[ -f "$USD_OPTIMIZE_ROOT/docs/operations.rst" || -f "$USD_OPTIMIZE_ROOT/.agents/operations/INDEX.md" ]] || echo "MISSING: per-operation docs (docs/operations.rst or .agents/operations/INDEX.md)"
```

Verified 2026-08-04: this check passes unchanged on extracted 1.0.4 and 1.1.0.

## Environment for Docker/CI

Set `WU_SO_PACKAGE_DIR` to point tools at the local backend:

```bash
export WU_SO_PACKAGE_DIR="$USD_OPTIMIZE_ROOT"
export USD_OPTIMIZE_ROOT="$USD_OPTIMIZE_ROOT"
```

If absent, downstream tools may fall back to NVCF cloud backend or fail.

## Troubleshooting

- If `omni.scene.optimizer.core` cannot be imported, confirm Python 3.12 is
  running and `$USD_OPTIMIZE_ROOT/activate.sh` has been sourced in the current shell.
- `ImportError: libpython3.12.so.1.0: cannot open shared object file` →
  the active `python3.12` is uv-managed (or otherwise installed outside
  the system loader path) and `$USD_OPTIMIZE_ROOT/activate.sh` was not re-sourced
  after a fresh shell or after the `uv` install. The activate script
  prepends `$(python3.12 -c 'import sys, os; print(os.path.join(sys.base_prefix, "lib"))')`
  to `LD_LIBRARY_PATH`; re-source it. If the import still fails, run
  `python3.12 -c 'import sys; print(sys.base_prefix)'` manually and
  confirm a `lib/libpython3.12.so.1.0` exists under that prefix.
- If library loading fails on Linux, verify `$USD_OPTIMIZE_ROOT/lib` and
  `$USD_OPTIMIZE_ROOT/extraLibs` are present in `LD_LIBRARY_PATH`.
- If the install looks incomplete, run the sentinel check above and redownload
  when any required directory is missing.
- If downstream tools use a cloud backend or fail to find the package, set
  `WU_SO_PACKAGE_DIR="$USD_OPTIMIZE_ROOT"` in the same environment.
