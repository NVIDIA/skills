#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Installs the CPU toolkit stack for the skill evaluation sandbox, in both
# arms (with the skill and the baseline). It is for evaluation only; users
# install nvalchemi-toolkit from PyPI as the documentation describes.
set -euo pipefail
nva_stage=initialize
nva_report_failure() {
  nva_exit=$?
  if [ "$nva_exit" -ne 0 ]; then
    printf 'NVALCHEMI_SETUP_FAILED stage=%s exit_code=%s\n' "$nva_stage" "$nva_exit" >&2
  fi
}
trap nva_report_failure EXIT

nva_root="${NVALCHEMI_EVAL_ROOT:-/workspace}"
nva_env="$nva_root/.venv"
nva_publish=false
if [ ! -e "$nva_env" ] && [ ! -L "$nva_env" ]; then
  nva_stage=select_interpreter
  nva_host_python=$(command -v python3)
  nva_stage=create_environment
  nva_env=$(mktemp -d "$nva_root/.nvalchemi-env.XXXXXX")
  nva_publish=true
  printf 'Creating CPU environment: %s (host Python: %s)\n' "$nva_env" "$nva_host_python"
  "$nva_host_python" -m venv --system-site-packages --without-pip "$nva_env"
  # The CPU torch wheel comes alone from the PyTorch index, so the resolver
  # never takes the CUDA wheel from PyPI.
  nva_stage=install_cpu_torch
  printf 'Installing CPU PyTorch into %s\n' "$nva_env"
  "$nva_host_python" -m pip --python "$nva_env" install \
    --cert "${PIP_CERT:-/etc/ssl/certs/ca-certificates.crt}" \
    --disable-pip-version-check --no-input --only-binary=:all: --no-compile \
    --progress-bar off --no-deps 'torch==2.13.0+cpu' \
    --index-url https://download-r2.pytorch.org/whl/cpu
  # Everything else resolves on PyPI. antlr4-python3-runtime 4.9.3 (via
  # physicsnemo and hydra-core) ships only as a pure-Python sdist, so it is
  # the one exception to wheels only.
  nva_stage=install_toolkit
  printf 'Installing nvalchemi-toolkit and its dependencies into %s\n' "$nva_env"
  "$nva_host_python" -m pip --python "$nva_env" install \
    --cert "${PIP_CERT:-/etc/ssl/certs/ca-certificates.crt}" \
    --disable-pip-version-check --no-input --only-binary=:all: --no-compile \
    --progress-bar off --no-binary=antlr4-python3-runtime \
    'torch==2.13.0+cpu' 'nvalchemi-toolkit==0.3.0' \
    'nvalchemi-toolkit-ops==0.5.0' 'ase==3.27.0'
fi
nva_stage=smoke_check
printf 'Checking CPU environment: %s\n' "$nva_env"
"$nva_env/bin/python" -u - <<'PY'
import sys

print("Smoke check runtime:", sys.executable, sys.version.split()[0], flush=True)
import torch
import nvalchemi
from nvalchemi.data import AtomicData, Batch
from nvalchemi.models.lj import LennardJonesModelWrapper

assert nvalchemi.__version__ == "0.3.0", nvalchemi.__version__
assert torch.version.cuda is None, torch.__version__
data = AtomicData(
    positions=torch.tensor([[0.0, 0.0, 0.0], [3.8, 0.0, 0.0]]),
    atomic_numbers=torch.tensor([18, 18]),
)
# Two atoms, each the other's only neighbor; padding uses n_atoms.
data.add_node_property("neighbor_matrix", torch.tensor([[1, 2], [0, 2]], dtype=torch.int32))
data.add_node_property("num_neighbors", torch.tensor([1, 1], dtype=torch.int32))
batch = Batch.from_data_list([data])
model = LennardJonesModelWrapper(epsilon=0.0104, sigma=3.40, cutoff=8.5)
energy = model(batch)["energy"]
assert energy.shape[0] == 1 and bool(torch.isfinite(energy).all())
print("nvalchemi environment ready:", sys.executable, sys.version.split()[0],
      "torch", torch.__version__, "energy", float(energy.sum()))
PY
# Publish .venv only after the install and the smoke check pass. A failed
# environment stays at its own path for diagnosis; a retry uses a fresh one.
if [ "$nva_publish" = true ]; then
  nva_stage=publish_environment
  ln -sT "$nva_env" "$nva_root/.venv"
fi
