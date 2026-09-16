# Host Prerequisites

Read this reference before workflow preflight and whenever `deft_python.sh`
cannot select an interpreter.

The host-side helpers require Python 3.11 with `numpy`, `pandas`, `pyarrow`,
`PyYAML`, and `huggingface_hub`. A Python used to provision the workspace must
also provide `venv` and `ensurepip`; a globally installed `pip` executable is
not required. Resolve `DEFT_SKILL_ROOT` to the installed
`tao-run-deft-cr-its-mining` skill directory, then export the workspace and
select one dependency-complete interpreter:

```bash
export WORKSPACE_DIR="$WORKSPACE"
DEFT_PYTHON="$("$DEFT_SKILL_ROOT/scripts/deft_python.sh")"
```

The selector prefers `<deft_workspace>/.venv`, checks every required import,
and never installs packages. If it fails, show its exact commands and obtain
user approval before provisioning the workspace-local environment:

```bash
BOOTSTRAP_PYTHON="$(command -v python3.11 || command -v python3)"
"$BOOTSTRAP_PYTHON" -c 'import sys; assert sys.version_info >= (3, 11)'
"$BOOTSTRAP_PYTHON" -m venv "<deft_workspace>/.venv"
VENV_PYTHON="<deft_workspace>/.venv/bin/python"
"$VENV_PYTHON" -m pip --version >/dev/null 2>&1 || "$VENV_PYTHON" -m ensurepip --upgrade
"$VENV_PYTHON" -m pip install numpy pandas pyarrow pyyaml huggingface_hub
```

If venv creation or the `ensurepip` fallback fails, stop and tell the user that
the host Python lacks venv bootstrap support. Ask how they want to provide a
Python 3.11 installation with `venv`/`ensurepip`; do not install an OS package,
run `get-pip.py`, or use elevated privileges without explicit user direction.

Rerun the selector after installation. Use `DEFT_PYTHON` for every bundled
Python helper. Do not install these packages into an arbitrary system Python.
The selected platform must also provide its native CLI, GPU access, status and
log operations, and writable shared storage.
