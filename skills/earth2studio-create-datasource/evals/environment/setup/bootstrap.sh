#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
#
# Harbor pre_agent_setup / healthcheck: uv sync when a repo checkout is present.
# Default ACES runs stage only the skill; pass --copy-repo to populate /workspace/repo.

set -euo pipefail

REPO_ROOT="${EARTH2STUDIO_ROOT:-/workspace/repo}"

if [[ ! -f "${REPO_ROOT}/pyproject.toml" ]]; then
    echo "e2s-eval-bootstrap: no repo at ${REPO_ROOT}; skipping uv sync (skill-only eval mode)" >&2
    exit 0
fi

cd "${REPO_ROOT}"

export UV_LINK_MODE=copy
export UV_PYTHON=3.13

uv venv --python 3.13
uv sync --group dev --extra data
uv run pre-commit install --install-hooks

cat >/etc/profile.d/e2s-eval.sh <<EOF
export EARTH2STUDIO_ROOT=${REPO_ROOT}
export PATH="${REPO_ROOT}/.venv/bin:\${PATH}"
cd ${REPO_ROOT}
EOF

echo "e2s-eval-bootstrap: ready at ${REPO_ROOT} ($(uv run python --version))"
