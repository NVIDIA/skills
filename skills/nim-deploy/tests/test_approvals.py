# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for scripts/approvals.py: approved-run lookup and the single-command checks."""

from __future__ import annotations

import os
from datetime import timedelta
from pathlib import Path

import approvals
import nd_common as nd
import pytest
from _testkit import K8S_VALUES, make_run, run_cli, runs_home  # noqa: F401

pytestmark = pytest.mark.usefixtures("runs_home")


@pytest.mark.parametrize(
    "text",
    [
        "kubectl get pods",
        "NO_COLOR=1 kubectl get pods",
        "kubectl get pods -o 'jsonpath={.items}'",
        'echo "a b"',
    ],
)
def test_single_command_accepts_one_plain_command(text: str) -> None:
    argv, problem = approvals.single_command(text)
    assert problem is None and argv


@pytest.mark.parametrize(
    ("text", "why"),
    [
        ("a && b", "several commands"),
        ("a | b", "several commands"),
        ("a; b", "several commands"),
        ("a > f", "redirection"),
        ("a\nb", "newline"),
        ("a # c", "comment"),
        ("a *", "glob"),
        ("echo $HOME", "$-expansion"),
        ("echo `x`", "backtick"),
        ("echo 'x", "unterminated"),
        ("FOO=1 kubectl get x", "only NO_COLOR and TERM"),
        ("", "empty"),
    ],
)
def test_single_command_rejects_everything_else(text: str, why: str) -> None:
    argv, problem = approvals.single_command(text)
    assert argv == [] and problem is not None and why in problem


def test_take_options_splits_known_flags_and_flags_unknown_ones() -> None:
    names = {"-n": "namespace", "--context": "context"}
    found, words, problem = approvals.take_options(["x", "-n", "ns", "--context=c"], names)
    assert (found, words, problem) == ({"namespace": ["ns"], "context": ["c"]}, ["x"], None)
    assert approvals.take_options(["--all"], names)[2] == "'--all' is not part of the approved form"
    assert approvals.take_options(["-n"], names)[2] == "-n needs a value"
    assert approvals.take_options(["--server-side"], names, {"--server-side"})[2] is None


def test_helm_extras_validates_the_timeout() -> None:
    assert approvals.helm_extras(["a", "--wait", "--timeout", "10m", "--dry-run=server"]) == (
        ["a"],
        None,
    )
    assert approvals.helm_extras(["--timeout=5m"]) == ([], None)
    assert "not a duration" in str(approvals.helm_extras(["--timeout", "soon"])[1])


def test_active_approvals_need_an_active_state_and_an_unexpired_record() -> None:
    state = make_run(K8S_VALUES, state="apply")
    assert approvals.active_approvals(Path.cwd(), nd.utc_now()) == []
    now = nd.utc_now()
    state["approvals"].append(
        {
            "target": "nim-operator",
            "expires_at": (now + timedelta(minutes=5)).isoformat(),
            "artifacts": {},
        }
    )
    nd.save_state(state)
    (found,) = approvals.active_approvals(Path.cwd(), now)
    assert found.target == "nim-operator" and found.state["run_id"] == "run1"
    assert approvals.active_approvals(Path.cwd(), now + timedelta(minutes=6)) == []
    state["state"] = "approve"
    nd.save_state(state)
    assert approvals.active_approvals(Path.cwd(), now) == []


def test_a_bad_expiry_means_no_approval() -> None:
    assert approvals.approval_expiry({"expires_at": "soon"}) is None
    assert approvals.approval_expiry({}) is None
    assert approvals.approval_expiry({"expires_at": "2030-01-01T00:00:00"}) is not None


def test_runs_root_comes_from_the_environment_or_the_nearest_folder(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    assert approvals.runs_roots(tmp_path) == [Path(os.environ["NIM_DEPLOY_HOME"])]
    monkeypatch.delenv("NIM_DEPLOY_HOME")
    (tmp_path / ".nim-deploy").mkdir()
    (tmp_path / "sub").mkdir()
    assert approvals.runs_roots(tmp_path / "sub") == [tmp_path / ".nim-deploy"]


def test_cli_lists_approved_runs() -> None:
    env = {**os.environ, "NIM_DEPLOY_HOME": str(nd.runs_root())}
    code, payload = run_cli("approvals.py", env=env)
    assert (code, payload["next"], payload["facts"]["approved_runs"]) == (0, "ask_user", [])
