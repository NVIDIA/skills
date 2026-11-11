# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Side-effect contract for help on vss-build-vision-ai executables."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest


SCRIPTS_DIR = Path(__file__).resolve().parents[1]
SCRIPTS = sorted(
    path
    for path in SCRIPTS_DIR.iterdir()
    if path.is_file() and path.suffix in {".py", ".sh"}
)
EXTERNAL_CLIENTS = (
    "curl",
    "docker",
    "git",
    "grpcurl",
    "kubectl",
    "nc",
    "ncat",
    "ngc",
    "ping",
    "ssh",
    "wget",
)


def snapshot(root: Path) -> dict[str, tuple[int, bytes | None]]:
    """Return enough sandbox state to detect created, removed, or changed files."""
    state: dict[str, tuple[int, bytes | None]] = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        mode = path.stat().st_mode
        state[relative] = (mode, path.read_bytes() if path.is_file() else None)
    return state


@pytest.mark.parametrize("flag", ["-h", "--help"])
@pytest.mark.parametrize("script", SCRIPTS, ids=lambda path: path.name)
def test_help_is_credential_independent_and_side_effect_free(
    tmp_path: Path, script: Path, flag: str
) -> None:
    bin_dir = tmp_path / "bin"
    guard_dir = tmp_path / "python-guard"
    home_dir = tmp_path / "home"
    temp_dir = tmp_path / "tmp"
    work_dir = tmp_path / "work"
    run_dir = tmp_path / "scripts"
    for directory in (bin_dir, guard_dir, home_dir, temp_dir, work_dir, run_dir):
        directory.mkdir()

    marker = tmp_path / "external-client-called"
    wrapper = (
        '#!/bin/sh\nprintf "%s\\n" "${0##*/}" >> "$HELP_EXTERNAL_MARKER"\nexit 99\n'
    )
    for command in EXTERNAL_CLIENTS:
        executable = bin_dir / command
        executable.write_text(wrapper, encoding="utf-8")
        executable.chmod(0o755)

    # Python automatically imports sitecustomize. Blocking socket connections here
    # covers direct urllib/http.client/socket probes that do not use a CLI client.
    (guard_dir / "sitecustomize.py").write_text(
        """\
import socket


def _blocked(*_args, **_kwargs):
    raise RuntimeError("network access is forbidden while rendering help")


socket.create_connection = _blocked
socket.socket.connect = _blocked
socket.socket.connect_ex = _blocked
""",
        encoding="utf-8",
    )

    runnable = run_dir / script.name
    shutil.copy2(script, runnable)

    # Use a deliberately minimal environment: help must not require any credential.
    env = {
        "HOME": str(home_dir),
        "HELP_EXTERNAL_MARKER": str(marker),
        "LANG": "C.UTF-8",
        "PATH": f"{bin_dir}:/usr/bin:/bin",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": str(guard_dir),
        "TMPDIR": str(temp_dir),
    }
    command = (
        ["bash", str(runnable)]
        if script.suffix == ".sh"
        else [sys.executable, str(runnable)]
    )
    before = snapshot(tmp_path)
    result = subprocess.run(
        [*command, flag],
        capture_output=True,
        cwd=work_dir,
        text=True,
        timeout=10,
        env=env,
        check=False,
    )
    after = snapshot(tmp_path)

    assert result.returncode == 0, result.stderr
    assert "usage" in f"{result.stdout}\n{result.stderr}".lower()
    assert not marker.exists(), (
        f"{script.name} called {marker.read_text(encoding='utf-8').strip()} while rendering {flag}"
    )
    assert after == before, f"{script.name} changed files while rendering {flag}"
