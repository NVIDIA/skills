# Install the standalone runtime

Use only when [setup](setup.md) identifies a missing capability. Reuse installation
authorization already provided. Otherwise name the required package, destination and expected
download footprint before requesting that installation decision. Do not restart installation
when an existing environment passes the required imports and capability checks.

Prefer a supplied package root/archive. For a download, use an official prebuilt
`usd_optimize_...release.zip` from
[GitHub Releases](https://github.com/NVIDIA-Omniverse/usd-optimize/releases), with the exact
platform, USD flavor and release pinned. [Upstream guidance](upstreams/usd-optimize.md)
records the verified layouts and package resolution. Do not clone or build upstream source
just to obtain operation docs. Kit installation is outside this standalone path.

The verified 1.3.1 package requires Python 3.12. Check interpreter ABI and platform
before downloading. Extract to a dedicated runtime directory, reuse a project-managed venv,
and set `USD_OPTIMIZE_ROOT` to the extracted package root containing `python`, `usdpy`,
`lib` and `extraLibs`. Export it to the workflow and worker processes; resumable execution
requires it. Add the package's `python` and `usdpy` directories to `PYTHONPATH`. On Linux add `lib`
and `extraLibs` to `LD_LIBRARY_PATH`; on Windows add them to the process DLL search path.
For uv-managed Python, include the interpreter's library directory when `libpython3.12`
cannot be loaded. Keep environment changes scoped to the selected runtime.

Install `usd-validation-nvidia` and `numpy` into the same interpreter environment when
validation is required. First check `nvidia_usd_validate --version` and the actual imports;
do not reinstall a working validator. Use the optimizer's matching `pxr` bindings. A
validator-only environment needs a compatible `usd-core` when `pxr` is absent; do not add
a conflicting USD wheel to an optimizer environment. Avoid the bundled validator venv as a
default without checking its dependencies.

Verify `pxr`, optimizer native imports, the live operation inventory, validator imports and
explicit `usd_optimize.validators.register_all()`. Standalone validation imports
`ValidationEngine` from `omni.asset_validator`, without Kit's `.core` suffix.
Capture package/version evidence and loader failures in `setup-preflight.json`, then return
to setup. An install command returning success does not prove the native runtime loads.
