#!/usr/bin/env bash
set -euo pipefail
cueq_stage=initialize
cueq_report_failure() {
  cueq_exit=$?
  if [ "$cueq_exit" -ne 0 ]; then
    printf 'CUEQ_SETUP_FAILED stage=%s exit_code=%s\n' "$cueq_stage" "$cueq_exit" >&2
  fi
}
trap cueq_report_failure EXIT

cueq_root="${CUEQ_EVAL_ROOT:-/workspace}"
cueq_env="$cueq_root/.venv"
cueq_publish=false
if [ ! -e "$cueq_env" ] && [ ! -L "$cueq_env" ]; then
  cueq_stage=select_interpreter
  cueq_host_python=$(command -v python3)
  cueq_stage=create_environment
  cueq_env=$(mktemp -d "$cueq_root/.cueq-env.XXXXXX")
  cueq_publish=true
  printf 'Creating CPU environment: %s (host Python: %s)\n' "$cueq_env" "$cueq_host_python"
  "$cueq_host_python" -m venv --system-site-packages --without-pip "$cueq_env"
  # Let pip select the pinned CPU wheel from PyTorch's alternate index.
  # Keep HTTPS verification and resolve dependencies on the normal index below.
  cueq_stage=install_cpu_torch
  printf 'Installing CPU PyTorch into %s\n' "$cueq_env"
  "$cueq_host_python" -m pip --python "$cueq_env" install \
    --cert "${PIP_CERT:-/etc/ssl/certs/ca-certificates.crt}" \
    --disable-pip-version-check --no-input --only-binary=:all: --no-compile \
    --progress-bar off --no-deps 'torch==2.14.1+cpu' \
    --index-url https://download-r2.pytorch.org/whl/cpu
  # Resolve Torch's dependencies on the normal index. Older host pip
  # rejects some dependency metadata on the PyTorch index.
  cueq_stage=install_dependencies
  printf 'Installing Torch dependencies and cuEquivariance into %s\n' "$cueq_env"
  "$cueq_host_python" -m pip --python "$cueq_env" install \
    --cert "${PIP_CERT:-/etc/ssl/certs/ca-certificates.crt}" \
    --disable-pip-version-check --no-input --only-binary=:all: --no-compile \
    --progress-bar off \
    'torch==2.14.1+cpu' 'cuequivariance==0.13.0' 'cuequivariance-torch==0.13.0'
fi
cueq_stage=smoke_check
printf 'Checking CPU environment: %s\n' "$cueq_env"
"$cueq_env/bin/python" -u - <<'PY'
import importlib.metadata as metadata
import sys

print("Smoke check runtime:", sys.executable, sys.version.split()[0], flush=True)
import torch
import cuequivariance_torch as cuet

for package, expected in {
    "cuequivariance": "0.13.0", "cuequivariance-torch": "0.13.0",
    "torch": "2.14.1+cpu",
}.items():
    assert metadata.version(package) == expected, (package, metadata.version(package))
assert torch.version.cuda is None, torch.__version__
layer = cuet.SphericalHarmonics([0, 1], normalize=False, method="naive")
x = torch.ones((1, 3), dtype=torch.float64, requires_grad=True)
y = layer(x)
assert y.shape == (1, 4) and bool(torch.isfinite(y).all())
y.square().sum().backward()
assert x.grad is not None and bool(torch.isfinite(x.grad).all())
print("CPU environment ready:", sys.executable, sys.version.split()[0], torch.__version__)
PY
# A failed install or smoke check must not leave .venv pointing at a
# partial environment. Preserve it for diagnosis and retry in a fresh path.
if [ "$cueq_publish" = true ]; then
  cueq_stage=publish_environment
  ln -sT "$cueq_env" "$cueq_root/.venv"
fi
