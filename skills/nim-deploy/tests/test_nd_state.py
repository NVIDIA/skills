# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for scripts/nd_state.py: the facade in nd_common exposes the same objects."""

from __future__ import annotations

from pathlib import Path

import nd_common as nd
import nd_state
import pytest
from _testkit import make_run, run_cli, runs_home  # noqa: F401


def test_nd_common_re_exports_the_state_helpers() -> None:
    for name in (
        "emit",
        "error",
        "transition",
        "load_state",
        "save_state",
        "sha256_file",
        "looks_secret",
        "EDGES",
    ):
        assert getattr(nd, name) is getattr(nd_state, name)


@pytest.mark.usefixtures("runs_home")
def test_state_round_trips() -> None:
    state = make_run({"image": "i"}, target="helm")
    assert nd_state.load_state("run1") == state
    assert nd_state.input_value(state, "image") == "i"
    assert nd_state.run_inputs(state) == {"image": "i", "run_id": "run1"}


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NIM_DEPLOY_HOME", raising=False)
    monkeypatch.delenv("CLAUDE_PROJECT_DIR", raising=False)


@pytest.mark.usefixtures("clean_env")
def test_the_run_store_is_the_nearest_git_root_of_the_current_folder(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    (tmp_path / "project" / ".git").mkdir(parents=True)
    (tmp_path / "project" / "a" / "b").mkdir(parents=True)
    monkeypatch.chdir(tmp_path / "project" / "a" / "b")
    assert nd_state.default_runs_root() == tmp_path.resolve() / "project" / ".nim-deploy"


@pytest.mark.usefixtures("clean_env")
def test_an_existing_run_store_marks_the_project_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    (tmp_path / "work" / ".nim-deploy").mkdir(parents=True)
    (tmp_path / "work" / "sub").mkdir()
    monkeypatch.chdir(tmp_path / "work" / "sub")
    assert nd_state.default_runs_root() == tmp_path.resolve() / "work" / ".nim-deploy"


@pytest.mark.usefixtures("clean_env")
def test_without_a_project_marker_the_run_store_is_in_the_current_folder(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    assert nd_state.default_runs_root() == tmp_path.resolve() / ".nim-deploy"


def test_claude_project_dir_wins_over_the_current_folder(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("NIM_DEPLOY_HOME", raising=False)
    monkeypatch.setenv("CLAUDE_PROJECT_DIR", str(tmp_path / "proj"))
    monkeypatch.chdir(tmp_path)
    assert nd_state.default_runs_root() == tmp_path / "proj" / ".nim-deploy"


def test_nim_deploy_home_wins_over_everything(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("CLAUDE_PROJECT_DIR", str(tmp_path / "proj"))
    monkeypatch.setenv("NIM_DEPLOY_HOME", str(tmp_path / "home"))
    assert nd_state.runs_root() == tmp_path / "home"


@pytest.mark.usefixtures("clean_env")
def test_a_current_folder_inside_the_skill_is_ignored(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    skill = Path(nd_state.__file__).resolve().parent.parent
    monkeypatch.chdir(skill / "scripts")
    root = nd_state.default_runs_root()
    assert skill not in (root, *root.parents)


def test_library_modules_print_a_contract_object_when_run() -> None:
    code, payload = run_cli("nd_state.py")
    assert (code, payload["state"]) == (0, "library")
