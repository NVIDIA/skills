## Process environment

Use `scripts/launch_environment.py` for setup commands and to enter the supervisor.
If uv is missing, first complete the [uv bootstrap](setup.md#3-install-the-locked-environment-and-skill-managed-supplements)
directly from the host shell; the launcher requires an existing uv executable.
The supervisor uses that same builder for preflight, training, evaluation,
checkpoint reads and artifact verification. A new `.venv` alone does not isolate
Python imports or native libraries from the parent shell.

Prepare the session plan before setup; the launcher only needs `training_root`
and optional `environment`. The package root must already exist. After the EULA
receipt is confirmed, define this shell helper from that root:

```bash
# SKILL and PLAN are absolute paths; use the same plan throughout setup/preflight.
training_cmd() {
  /usr/bin/python3 "$SKILL/scripts/launch_environment.py" --plan "$PLAN" -- "$@"
}
training_cmd uv sync --locked --group sim --no-install-package onnxruntime --dry-run
training_cmd .venv/bin/python "$SKILL/scripts/supervisor.py" preflight --plan "$PLAN"
```

The helper checks the EULA receipt without creating a campaign.
For an installation-only request, stop after preflight;
report that simulator scene execution and the training smoke remain untested.

### Selection and explicit exceptions

Optional plan configuration:

```json
"environment": {
  "uv_executable": "/absolute/path/to/uv",
  "tool_directories": [],
  "variables": {
    "UV_CACHE_DIR": "/absolute/path/to/private-cache/uv",
    "UV_PYTHON_INSTALL_DIR": "/absolute/path/to/private-python",
    "XDG_CACHE_HOME": "/absolute/path/to/private-cache",
    "XDG_CONFIG_HOME": "/absolute/path/to/private-config"
  }
}
```

Select the intended uv executable explicitly for isolated installations. Otherwise
it is discovered once during plan normalization. Child PATH contains the selected
`.venv/bin`, uv directory, optional explicit tool directories and standard system
directories; the rest of the parent PATH is discarded.

`LD_LIBRARY_PATH`, `LD_PRELOAD`, `PYTHONPATH`, CUDA toolkit paths and Isaac Sim asset
overrides are **not inherited**. If a documented runtime requirement needs one,
set it explicitly in `environment.variables` and record why. Library/import paths
must be absolute and nonempty; empty path-list entries would search the current
directory. For example, an aarch64 preload must name the specific required library,
not copy the shell's entire preload setting. Do not use exceptions to import an old
Isaac Lab/Sim checkout. Arbitrary credentials, activation variables and a replacement
PATH cannot be supplied through this field.

Approved user/locale, display, GPU selection, cache and CPU-thread settings are
snapshotted in the normalized session plan. Explicit variables override that
snapshot. `--print` displays the effective environment; preflight and run records
also retain it. For a requested fresh install, use newly created private cache,
configuration and managed-Python directories; ordinary reuse does not require
redownloading the environment.

### Launch boundary

Sanitize **before starting each subprocess**. Unsetting a variable inside an
already-running simulator cannot undo loaded imports or native libraries.
tmux/systemd launchers reset their server environment before starting supervisor
Python; detached workers receive the same snapshot. Restart with a new session
after changing these settings or the helper; do not rewrite an active session.
