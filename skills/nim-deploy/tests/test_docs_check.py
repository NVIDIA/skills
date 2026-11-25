# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the docs gate: no approval summary without a fresh docs-check of the official docs."""

from __future__ import annotations

import os
from datetime import timedelta
from typing import Any

import nd_common as nd
import nextstep
import pytest
from _testkit import (
    DOCKER_VALUES,
    FAKE_NVAPI,
    K8S_VALUES,
    docs_entry,
    make_run,
    run_cli,
    with_docs,
)
from _testkit import runs_home as runs_home

pytestmark = pytest.mark.usefixtures("runs_home")

DOC = "https://docs.nvidia.com/nim/large-language-models/latest/get-started/quickstart.html"
IMAGE = "nvcr.io/nim/meta/llama-3.1-8b-instruct:1.8"


def env() -> dict[str, str]:
    return {**os.environ, "NIM_DEPLOY_HOME": str(nd.runs_root())}


def validate_run(target: str = "nim-operator", values: dict[str, Any] | None = None) -> str:
    """A run in the validate state with no docs check yet (as the render scripts leave it)."""
    state = make_run(values or K8S_VALUES, target=target, state="render")
    renderer = {
        "docker": ["render_docker.py"],
        "helm": ["render_k8s.py", "--kind", "helm"],
        "nim-operator": ["render_k8s.py", "--kind", "nim-operator"],
    }[target]
    code, _ = run_cli(renderer[0], "--run", str(state["run_id"]), *renderer[1:], env=env())
    assert code == 0
    return str(state["run_id"])


def docs_check(run: str, *args: str) -> tuple[int, dict[str, Any]]:
    return run_cli("nimdeploy.py", "docs-check", run, *args, env=env())


ONLINE = ("--url", DOC, "--doc-version", "2.1.4", "--note", "port=8000; env=NGC_API_KEY")


def test_validate_refuses_until_the_docs_were_checked() -> None:
    run = validate_run()
    code, payload = run_cli("validate.py", "--run", run, "--offline", env=env())
    assert (code, payload["next"], payload["state"]) == (3, "docs_check", "validate")
    error = payload["errors"][0]
    assert error["code"] == "docs_check"
    assert "docs-check" in error["fix"] and "docs-sources.md" in error["fix"]
    assert "docs-check" in payload["facts"]["next_command"]
    assert docs_check(run, *ONLINE)[0] == 0
    code, payload = run_cli("validate.py", "--run", run, "--offline", env=env())
    assert (code, payload["state"]) == (0, "approve")


def test_status_points_at_the_docs_check_in_the_validate_state() -> None:
    run = validate_run()
    _, payload = run_cli("nimdeploy.py", "status", run, env=env())
    assert payload["next"] == "docs_check"
    assert "docs-check" in payload["facts"]["next_command"]
    assert "docs-sources.md" in payload["facts"]["next_command"]


def test_an_online_check_records_urls_version_note_and_time() -> None:
    run = validate_run()
    code, payload = docs_check(run, *ONLINE, "--url", DOC.replace("quickstart", "architecture"))
    assert code == 0 and payload["facts"]["docs_check"]["mode"] == "online"
    check = nd.load_state(run)["docs_check"]
    assert check["mode"] == "online" and len(check["urls"]) == 2
    assert (check["doc_version"], check["note"]) == ("2.1.4", "port=8000; env=NGC_API_KEY")
    assert nd.utc_now() - nextstep_time(check) < timedelta(seconds=30)


def nextstep_time(check: dict[str, Any]) -> Any:
    from datetime import datetime

    return datetime.fromisoformat(check["fetched_at"])


@pytest.mark.parametrize(
    "args",
    [
        (),
        ("--url", DOC),
        ("--url", DOC, "--doc-version", "2.1.4"),
        ("--url", "not-a-url", "--doc-version", "2.1.4", "--note", "x"),
        ("--doc-version", "2.1.4", "--note", "x"),
        ("--offline",),
        ("--offline", "--reason", "x", "--url", DOC),
        ("--url", DOC, "--doc-version", "2.1.4", "--note", f"key {FAKE_NVAPI}"),
    ],
)
def test_incomplete_checks_are_refused(args: tuple[str, ...]) -> None:
    run = validate_run()
    code, payload = docs_check(run, *args)
    assert code in (2, 5) and not payload["ok"]
    assert "docs_check" not in nd.load_state(run)


def test_offline_needs_a_reason_and_is_shown_prominently() -> None:
    run = validate_run()
    code, payload = docs_check(
        run, "--offline", "--reason", "the user said docs.nvidia.com is unreachable here"
    )
    assert code == 0
    assert nd.load_state(run)["docs_check"]["mode"] == "offline"
    code, payload = run_cli("validate.py", "--run", run, "--offline", env=env())
    assert code == 0
    facts = payload["facts"]
    assert facts["docs_warning"].startswith("DOCS NOT CHECKED (offline): the user said")
    assert "documented default (offline)" in facts["run_basis"]
    assert facts["docs"]["mode"] == "offline"
    approve = run_cli("nimdeploy.py", "approve", run, "--token", facts["token"], env=env())
    assert approve[0] == 0  # the token flow is unchanged


def test_the_summary_shows_urls_version_time_and_the_basis() -> None:
    run = validate_run()
    docs_check(run, "--url", DOC, "--doc-version", "1.8.x", "--note", "port=8000; env=NGC_API_KEY")
    facts = run_cli("validate.py", "--run", run, "--offline", env=env())[1]["facts"]
    assert facts["docs"]["urls"] == [DOC] and facts["docs"]["doc_version"] == "1.8.x"
    assert facts["docs"]["fetched_at"] and facts["docs_warning"] == ""
    assert "NIM docs 1.8.x read" in facts["run_basis"] and DOC in facts["run_basis"]
    assert "port=8000" in facts["run_basis"]


def test_a_stale_check_must_be_redone_before_the_summary_and_the_approval() -> None:
    run = validate_run()
    docs_check(run, *ONLINE)
    facts = run_cli("validate.py", "--run", run, "--offline", env=env())[1]["facts"]
    state = nd.load_state(run)
    state["docs_check"]["fetched_at"] = (nd.utc_now() - timedelta(minutes=61)).isoformat()
    nd.save_state(state)
    for argv in (("approval-summary", run), ("approve", run, "--token", facts["token"])):
        code, payload = run_cli("nimdeploy.py", *argv, env=env())
        assert (code, payload["errors"][0]["code"], payload["next"]) == (
            3,
            "docs_stale",
            "ask_user",
        )
    code, payload = run_cli("nimdeploy.py", "status", run, env=env())
    assert payload["next"] == "docs_check"
    docs_check(run, *ONLINE)
    assert run_cli("nimdeploy.py", "approval-summary", run, env=env())[0] == 0


def test_an_approval_without_a_check_is_refused() -> None:
    state = make_run(K8S_VALUES, state="approve")
    code, payload = run_cli("nimdeploy.py", "approval-summary", state["run_id"], env=env())
    assert (code, payload["errors"][0]["code"]) == (3, "docs_check")
    code, payload = run_cli(
        "nimdeploy.py", "approve", state["run_id"], "--token", "abcd1234", env=env()
    )
    assert (code, payload["errors"][0]["code"]) == (3, "docs_check")


def test_changing_the_image_clears_the_check_but_other_inputs_do_not() -> None:
    run = validate_run("docker", DOCKER_VALUES)
    docs_check(run, *ONLINE)
    run_cli("nimdeploy.py", "set", run, "port=8100", env=env())
    assert "docs_check" in nd.load_state(run)
    other = "image=nvcr.io/nim/other/model:1.0"
    run_cli("nimdeploy.py", "set", run, other, env=env())  # a different image
    assert "docs_check" not in nd.load_state(run)
    docs_check(run, *ONLINE)
    run_cli("nimdeploy.py", "set", run, other, env=env())  # the same image again
    assert "docs_check" in nd.load_state(run)


def test_a_docs_check_cannot_be_changed_after_approval() -> None:
    run = with_docs(make_run(K8S_VALUES, state="apply"))["run_id"]
    code, payload = docs_check(run, *ONLINE)
    assert (code, payload["errors"][0]["code"]) == (5, "run_approved")


@pytest.mark.parametrize(
    ("image", "docs", "expect"),
    [
        ("nvcr.io/nim/x:2.1.4", "2.1.4", "matches"),
        ("nvcr.io/nim/x:2.1.4", "2.1.x", "matches"),
        ("nvcr.io/nim/x:2.1.4-rc.20261007", "2.1.x", "pre-release"),
        ("nvcr.io/nim/x:2.2.0", "2.1.4", "image newer than docs"),
        ("nvcr.io/nim/x:2.1.5", "2.1.4", "image newer than docs"),
        ("nvcr.io/nim/x:2.0.9", "2.1.4", "image older than docs"),
        ("nvcr.io/nim/x:1.8", "2.1.4", "image older than docs"),
        ("nvcr.io/nim/x:latest", "2.1.4", "no version"),
        ("nvcr.io/nim/x:2.1.4", "latest", "not numeric"),
    ],
)
def test_version_check(image: str, docs: str, expect: str) -> None:
    assert expect in nextstep.version_check(image, docs)


def test_a_run_without_an_image_is_not_compared() -> None:
    assert nextstep.version_check("", "2026-09-03").startswith("not compared")


def test_a_mismatch_is_flagged_in_the_summary() -> None:
    run = validate_run("docker", {**DOCKER_VALUES, "image": "nvcr.io/nim/x:2.2.0"})
    docs_check(run, *ONLINE)
    facts = run_cli("validate.py", "--run", run, "--offline", env=env())[1]["facts"]
    assert facts["docs_warning"].startswith("image newer than docs")
    assert "2.2.0" in facts["docs"]["version_check"]


def test_every_target_is_gated() -> None:
    for target, values in (
        ("helm", K8S_VALUES),
        ("nim-operator", K8S_VALUES),
        ("docker", DOCKER_VALUES),
    ):
        run = validate_run(target, values)
        code, payload = run_cli("validate.py", "--run", run, "--offline", env=env())
        assert (code, payload["next"]) == (3, "docs_check"), target


def test_a_fresh_entry_helper_matches_the_gate() -> None:
    state = make_run(K8S_VALUES)
    assert nd.docs_problem(state) is not None
    with_docs(state)
    assert nd.docs_problem(state) is None
    with_docs(state, fetched_at=(nd.utc_now() - timedelta(hours=2)).isoformat())
    problem = nd.docs_problem(state)
    assert problem is not None and problem[0] == "docs_stale"
    assert docs_entry(mode="offline")["mode"] == "offline"
