# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The skills work from any install folder: project .agents/skills, .claude/skills,
~/.claude/skills, a folder with a space in its name, and the nim-deploy-cli symlink. The agent
stays in an unrelated project folder; every printed command must be absolute, shell-quoted and
runnable from there."""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest
from _testkit import OFFLINE_TOOL, SCRIPTS

IMAGE = "nvcr.io/nim/meta/llama-3.1-8b-instruct:2.1.4"
LOCATIONS = [
    ".agents/skills",
    ".claude/skills",
    "home/.claude/skills",
    "my skills/with space",
]


def install(root: Path, location: str, skill: str) -> Path:
    """Copy the scripts to root/location/nim-deploy; nim-deploy-cli links to them like the repo."""
    base = root / location
    scripts = base / "nim-deploy" / "scripts"
    shutil.copytree(SCRIPTS, scripts, ignore=shutil.ignore_patterns("__pycache__"))
    if skill == "nim-deploy-cli":
        (base / "nim-deploy-cli").mkdir()
        (base / "nim-deploy-cli" / "scripts").symlink_to("../nim-deploy/scripts")
    return base / skill / "scripts"


@pytest.fixture
def project(tmp_path: Path) -> Path:
    folder = tmp_path / "project dir"
    (folder / ".git").mkdir(parents=True)
    (folder / "src").mkdir()
    return folder


@pytest.fixture
def env(tmp_path: Path) -> dict[str, str]:
    offline = tmp_path / "offline-bin"
    offline.mkdir()
    for name in ("docker", "skopeo"):
        (offline / name).write_text(OFFLINE_TOOL)
        (offline / name).chmod(0o755)
    clean = {
        key: value
        for key, value in os.environ.items()
        if key not in ("NIM_DEPLOY_HOME", "CLAUDE_PROJECT_DIR")
    }
    return {**clean, "PATH": f"{offline}{os.pathsep}{clean['PATH']}"}


def shell(command: str, cwd: Path, env: dict[str, str]) -> tuple[int, dict[str, Any]]:
    """Run COMMAND exactly as printed, through bash, so a quoting mistake fails here."""
    proc = subprocess.run(
        ["bash", "-c", command],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        timeout=60,
        check=False,
    )
    assert proc.stdout.strip(), proc.stderr
    return proc.returncode, json.loads(proc.stdout.strip().splitlines()[-1])


def first_command(scripts: Path, *words: str) -> str:
    return " ".join(["python3", shlex.quote(str(scripts / "nimdeploy.py")), *words])


def assert_absolute_quoted_script(command: str, scripts: Path) -> None:
    resolved = scripts.resolve()
    argv = shlex.split(command)
    assert argv[0] == "python3"
    assert Path(argv[1]).is_absolute()
    assert Path(argv[1]).parent == resolved
    assert Path(argv[1]).is_file()
    assert command.startswith(f"python3 {shlex.quote(argv[1])}")


@pytest.mark.parametrize("skill", ["nim-deploy", "nim-deploy-cli"])
@pytest.mark.parametrize("location", LOCATIONS)
def test_the_whole_flow_runs_from_an_unrelated_project_folder(
    tmp_path: Path, project: Path, env: dict[str, str], location: str, skill: str
) -> None:
    scripts = install(tmp_path / "install", location, skill)
    cwd = project / "src"

    code, new = shell(
        first_command(scripts, "new", "--target", "docker", "--set", f"image={IMAGE}"), cwd, env
    )
    assert code == 0, new
    run = new["facts"]["run_id"]
    assert_absolute_quoted_script(new["facts"]["next_command"], scripts)

    code, rendered = shell(new["facts"]["next_command"], cwd, env)
    assert code == 0, rendered
    code, checked = shell(
        first_command(
            scripts, "docs-check", run, "--url", "https://docs.nvidia.com/nim/x.html",
            "--doc-version", "2.1.4", "--note", "port=8000",
        ),
        cwd,
        env,
    )  # fmt: skip
    assert code == 0, checked
    code, status = shell(first_command(scripts, "status", run), cwd, env)
    assert code == 0, status
    assert_absolute_quoted_script(status["facts"]["next_command"], scripts)
    code, validated = shell(status["facts"]["next_command"] + " --offline", cwd, env)
    assert code == 0, validated
    code, summary = shell(first_command(scripts, "approval-summary", run), cwd, env)
    assert code == 0, summary
    assert_absolute_quoted_script(summary["facts"]["approve_command"], scripts)
    code, approved = shell(summary["facts"]["approve_command"], cwd, env)
    assert code == 0, approved
    apply_command = approved["facts"]["next_command"]

    store = project / ".nim-deploy"
    assert (store / "runs" / run / "state.json").is_file()
    assert not list((tmp_path / "install").rglob(".nim-deploy"))

    guard = scripts / "guard.py"
    code, decision = shell(
        f"python3 {shlex.quote(str(guard))} --check {shlex.quote(apply_command)}", cwd, env
    )
    assert (code, decision["state"]) == (0, "allow"), decision
    code, denied = shell(
        f"python3 {shlex.quote(str(guard))} --check 'kubectl delete namespace prod'", cwd, env
    )
    assert code == 2 and denied["ok"] is False


@pytest.mark.parametrize("location", LOCATIONS)
def test_an_error_fix_names_the_quoted_absolute_script(
    tmp_path: Path, project: Path, env: dict[str, str], location: str
) -> None:
    scripts = install(tmp_path / "install", location, "nim-deploy")
    code, payload = shell(first_command(scripts, "status", "no-such-run"), project, env)
    assert code == 2
    fix = payload["errors"][0]["fix"]
    assert f"python3 {shlex.quote(str(scripts.resolve() / 'nimdeploy.py'))} list" in fix


def test_error_text_names_scripts_by_absolute_quoted_path() -> None:
    import nd_common as nd

    path = shlex.quote(str(SCRIPTS / "nimdeploy.py"))
    fix = nd.error("x", "m", "then nimdeploy.py set RUN a=b and 'verify.py triage'")["fix"]
    assert fix.startswith(f"then python3 {path} set RUN a=b")
    assert f"'python3 {shlex.quote(str(SCRIPTS / 'verify.py'))} triage'" in fix
    assert nd.error("x", "m", fix)["fix"] == fix  # already absolute: left alone
    assert nd.error("x", "m", "no script here")["message"] == "m"


def test_a_guard_denial_names_the_runnable_script(
    tmp_path: Path, project: Path, env: dict[str, str]
) -> None:
    scripts = install(tmp_path / "install", "my skills/with space", "nim-deploy")
    proc = subprocess.run(
        ["python3", str(scripts / "guard.py"), "--check", "helm install x y -n ns --set a=b"],
        cwd=project, env=env, capture_output=True, text=True, check=False,
    )  # fmt: skip
    assert proc.returncode == 2
    assert shlex.quote(str(scripts / "nimdeploy.py")) in proc.stderr


def test_the_run_store_follows_the_project_not_the_install(
    tmp_path: Path, project: Path, env: dict[str, str]
) -> None:
    scripts = install(tmp_path / "install", "home/.claude/skills", "nim-deploy")
    env = {**env, "CLAUDE_PROJECT_DIR": str(tmp_path / "other project")}
    (tmp_path / "other project").mkdir()
    code, new = shell(
        first_command(scripts, "new", "--target", "docker", "--set", f"image={IMAGE}"), project, env
    )
    assert code == 0, new
    assert (tmp_path / "other project" / ".nim-deploy" / "runs" / new["facts"]["run_id"]).is_dir()
