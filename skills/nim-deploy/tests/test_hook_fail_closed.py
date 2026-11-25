# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The documented hook command must fail closed: only a clean guard exit 0 lets a call run.

Claude Code blocks a PreToolUse hook only on exit 2; any other failure (missing python or file,
import error, crash before main) lets the command run. `|| exit 2` closes that gap.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path

import pytest
from _testkit import SCRIPTS

SKILL = SCRIPTS.parent
DOCS = [
    SKILL / "references" / "hook.md",
]
GUARD_DIR = Path(".agents") / "skills" / "nim-deploy" / "scripts"


def documented_command(doc: Path) -> str:
    """Read the `command` of the JSON settings snippet in a doc."""
    block = re.search(r"```json\n(.*?)\n```", doc.read_text(), re.S)
    assert block, f"no JSON snippet in {doc}"
    hook = json.loads(block.group(1))["hooks"]["PreToolUse"][0]["hooks"][0]
    assert set(hook) == {"type", "command"}
    command: str = hook["command"]
    return command.replace("NAME", "nim-deploy")


def hook_payload(command: str, tool: str = "Bash") -> str:
    return json.dumps({"tool_name": tool, "tool_input": {"command": command}, "cwd": "/tmp"})


def run_documented(command: str, project: Path, stdin: str) -> subprocess.CompletedProcess[str]:
    env = {**os.environ, "CLAUDE_PROJECT_DIR": str(project)}
    return subprocess.run(
        ["sh", "-c", command],
        input=stdin,
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
        check=False,
    )


@pytest.fixture
def real_project(tmp_path: Path) -> Path:
    """A project folder whose .agents/skills/nim-deploy is the real skill."""
    link = tmp_path / ".agents" / "skills" / "nim-deploy"
    link.parent.mkdir(parents=True)
    link.symlink_to(SKILL)
    return tmp_path


@pytest.mark.parametrize("doc", DOCS, ids=lambda doc: doc.name)
def test_docs_do_not_use_onfailure_and_fail_closed(doc: Path) -> None:
    text = doc.read_text()
    assert "onFailure" not in text.replace("There is no `onFailure` hook option", "")
    assert "|| exit 2" in documented_command(doc)


def test_hook_doc_table_rows_fail_closed() -> None:
    rows = [r for r in DOCS[0].read_text().splitlines() if r.startswith("| ") and "python3" in r]
    assert len(rows) == 4
    assert all(r"\|\| exit 2" in r for r in rows)


@pytest.mark.parametrize("doc", DOCS, ids=lambda doc: doc.name)
def test_documented_command_allows_and_denies(doc: Path, real_project: Path) -> None:
    command = documented_command(doc)
    allowed = run_documented(command, real_project, hook_payload("nim list"))
    assert allowed.returncode == 0, allowed.stderr
    denied = run_documented(command, real_project, hook_payload("nim stop --all"))
    assert denied.returncode == 2
    assert "Fix:" in denied.stderr


@pytest.mark.parametrize("doc", DOCS, ids=lambda doc: doc.name)
def test_documented_command_fails_closed_when_the_guard_is_missing(
    doc: Path, tmp_path: Path
) -> None:
    result = run_documented(documented_command(doc), tmp_path, hook_payload("nim list"))
    assert result.returncode == 2  # python exits 2 for a missing file; 127 would also map to 2


def test_documented_command_fails_closed_when_the_guard_crashes_at_import(tmp_path: Path) -> None:
    crash = tmp_path / GUARD_DIR / "guard.py"
    crash.parent.mkdir(parents=True)
    crash.write_text("raise RuntimeError('boom before main')\n")
    result = run_documented(documented_command(DOCS[0]), tmp_path, hook_payload("nim list"))
    assert result.returncode == 2
    assert "boom before main" in result.stderr


def test_documented_command_fails_closed_when_the_guard_exits_one(tmp_path: Path) -> None:
    odd = tmp_path / GUARD_DIR / "guard.py"
    odd.parent.mkdir(parents=True)
    odd.write_text("import sys\nsys.exit(1)\n")
    result = run_documented(documented_command(DOCS[0]), tmp_path, hook_payload("nim list"))
    assert result.returncode == 2


def test_the_guard_itself_handles_bad_input_and_other_tools(real_project: Path) -> None:
    command = documented_command(DOCS[0])
    assert run_documented(command, real_project, "").returncode == 2
    assert run_documented(command, real_project, "{broken").returncode == 2
    assert run_documented(command, real_project, "[]").returncode == 2
    assert run_documented(command, real_project, hook_payload("rm -rf /", "Read")).returncode == 0


def test_documented_command_fails_closed_when_python_is_missing(real_project: Path) -> None:
    command = documented_command(DOCS[0]).replace("python3", "no-such-python-binary")
    result = run_documented(command, real_project, hook_payload("nim list"))
    assert result.returncode == 2  # the shell alone would give 127, which does not block
