# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for scripts/runners: the runner seam works end to end with a fake runner, and adding a
runner needs no change to SKILL.md or references/."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import guard
import nd_common as nd
import pytest
import runners
from _testkit import (
    DOCKER_VALUES,
    SCRIPTS,
    assert_contract,
    docs_cli,
    make_run,
    run_cli,
    with_docs,
)
from _testkit import runs_home as runs_home

pytestmark = pytest.mark.usefixtures("runs_home")

SKILL = SCRIPTS.parent
FAKE_ENV = {"NIM_DEPLOY_RUNNERS_EXTRA": "fake=_fake_runner"}


def skill_text_hashes() -> dict[str, str]:
    files = [SKILL / "SKILL.md", *sorted((SKILL / "references").glob("*.md"))]
    return {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}


@pytest.fixture
def fake_runner(monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    for key, value in FAKE_ENV.items():
        monkeypatch.setenv(key, value)
    return {**os.environ, **FAKE_ENV, "NIM_DEPLOY_HOME": str(nd.runs_root())}


def test_registry_default_and_loading(monkeypatch: pytest.MonkeyPatch) -> None:
    assert runners.RUNNERS == {"docker": "runners.docker", "nimcli": "runners.nimcli"}
    assert runners.DEFAULT_RUNNER == "docker" or os.environ.get("NIM_DEPLOY_RUNNER")
    assert runners.load().__name__ == "runners.docker"
    assert runners.load("docker").required_inputs() == ("image",)
    with pytest.raises(ValueError, match="unknown runner 'fake'"):
        runners.load("fake")
    monkeypatch.setenv("NIM_DEPLOY_RUNNERS_EXTRA", "fake=_fake_runner, other=x.y")
    assert runners.registry()["fake"] == "_fake_runner" and runners.registry()["other"] == "x.y"
    assert runners.load("fake").required_inputs() == ("model", "gpus", "port")


def test_the_default_runner_can_come_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NIM_DEPLOY_RUNNERS_EXTRA", "fake=_fake_runner")
    monkeypatch.setenv("NIM_DEPLOY_RUNNER", "fake")
    assert runners.default_runner() == "fake"
    assert runners.load().__name__ == "_fake_runner"
    assert runners.runner_of({"inputs": {}}).__name__ == "_fake_runner"
    docker_run: dict[str, Any] = {"inputs": {"runner": {"value": "docker"}}}
    assert runners.runner_of(docker_run).__name__ == "runners.docker"


def test_the_effective_runner_is_saved_with_the_run(
    fake_runner: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("NIM_DEPLOY_RUNNER", "fake")
    code, payload = run_cli(
        "nimdeploy.py", "new", "--target", "docker", "--model", "acme/tiny",
        "--set", "gpus=1", "--set", "port=9000", env={**fake_runner, "NIM_DEPLOY_RUNNER": "fake"},
    )  # fmt: skip
    assert code == 0
    run = payload["facts"]["run_id"]
    assert nd.load_state(run)["runner"] == "fake"
    monkeypatch.delenv("NIM_DEPLOY_RUNNER")  # the variable is gone when the run is reopened
    assert runners.runner_of(nd.load_state(run)).__name__ == "_fake_runner"
    assert nd.approval_bindings(nd.load_state(run))["runner"] == "fake"
    code, payload = run_cli("nimdeploy.py", "set", run, "runner=docker", env=fake_runner)
    assert code == 0 and nd.load_state(run)["runner"] == "docker"


def test_every_runner_module_provides_the_contract() -> None:
    names = (
        "required_inputs",
        "render",
        "apply_command",
        "rollback_command",
        "guard_forms",
        "ready_url",
        "container_name",
    )
    for name in ("docker", "fake"):
        os.environ["NIM_DEPLOY_RUNNERS_EXTRA"] = "fake=_fake_runner"
        module = runners.load(name)
        assert all(callable(getattr(module, attr, None)) for attr in names), name
    del os.environ["NIM_DEPLOY_RUNNERS_EXTRA"]


def drive_to_apply(env: dict[str, str]) -> tuple[str, dict[str, Any]]:
    """new > render > validate > approve through the real scripts, with the fake runner."""
    code, payload = run_cli(
        "nimdeploy.py", "new", "--target", "docker", "--model", "acme/tiny",
        "--set", "runner=fake", "--set", "gpus=1", "--set", "port=9000", env=env,
    )  # fmt: skip
    assert (code, payload["state"]) == (0, "render")  # the fake runner's required inputs only
    run = payload["facts"]["run_id"]
    assert payload["facts"]["next_command"].endswith(f"render_docker.py --run {run}")
    code, payload = run_cli("render_docker.py", "--run", run, env=env)
    assert (code, payload["state"], payload["facts"]["container"]) == (0, "validate", "fake-tiny")
    assert (
        nd.run_dir(run) / "fake-plan.txt"
    ).read_text() == "up acme/tiny on 1 gpu(s), port 9000\n"
    assert not (nd.run_dir(run) / "docker-run.sh").exists()
    docs_cli(run, env)
    code, payload = run_cli("validate.py", "--run", run, env=env)
    assert (code, payload["state"]) == (0, "approve")
    summary = payload["facts"]
    assert summary["apply_command"] == "fakecli up acme/tiny --port 9000"
    assert summary["objects"] == [{"kind": "container", "name": "fake-tiny"}]
    code, payload = run_cli("nimdeploy.py", "approve", run, "--token", summary["token"], env=env)
    assert (code, payload["state"], payload["next"]) == (0, "apply", "apply")
    return run, payload


def test_full_flow_with_a_fake_runner_without_touching_the_skill_text(
    fake_runner: dict[str, str],
) -> None:
    before = skill_text_hashes()
    run, payload = drive_to_apply(fake_runner)
    assert payload["facts"]["next_command"] == "fakecli up acme/tiny --port 9000"
    assert payload["facts"]["then_command"].endswith(f"advance {run} --to wait_ready")
    assert nd.load_state(run)["approvals"][-1]["runner"] == "fake"
    _, payload = run_cli("nimdeploy.py", "advance", run, "--to", "wait_ready", env=fake_runner)
    assert payload["facts"]["next_command"].endswith(
        f"verify.py ready --url http://localhost:9000 --run {run}"
    )
    assert skill_text_hashes() == before
    text = "\n".join(path.read_text() for path in map(Path, before))
    assert "fakecli" not in text
    skill = (SKILL / "SKILL.md").read_text()
    for command in (
        "docker run",
        "docker rm",
        "docker stop",
        "nim run",
        "nim stop",
        "docker-run.sh",
    ):
        assert command not in skill, command  # SKILL.md names no runner command
    assert "facts.next_command" in skill


def test_guard_allows_exactly_the_fake_runners_forms(
    fake_runner: dict[str, str], tmp_path: Path
) -> None:
    run, _ = drive_to_apply(fake_runner)
    now = nd.utc_now()

    def decide(command: str) -> str:
        return guard.evaluate(command, tmp_path, now)[0]

    assert decide("fakecli up acme/tiny --port 9000") == "allow"
    assert decide("fakecli down acme/tiny") == "allow"
    for denied in (
        "fakecli up acme/tiny --port 9001",
        "fakecli up other/model --port 9000",
        "fakecli down other/model",
        "fakecli rm --all",
        "fakecli up acme/tiny --port 9000 && echo done",
        "NGC_API_KEY=x fakecli up acme/tiny --port 9000",
        f"bash {nd.run_dir(run)}/docker-run.sh",
        "docker rm -f fake-tiny",
        "docker stop fake-tiny",
    ):
        assert decide(denied) == "deny", denied
    reason = guard.evaluate("fakecli up other --port 1", tmp_path, now)[2]
    assert "fakecli up acme/tiny --port 9000" in reason


def test_guard_denies_the_fake_form_after_a_plan_file_changes(
    fake_runner: dict[str, str], tmp_path: Path
) -> None:
    run, _ = drive_to_apply(fake_runner)
    (nd.run_dir(run) / "fake-plan.txt").write_text("edited\n")
    decision, _, reason = guard.evaluate("fakecli up acme/tiny --port 9000", tmp_path, nd.utc_now())
    assert decision == "deny" and "changed after approval" in reason


def test_without_an_approval_the_fake_program_is_not_gated(
    fake_runner: dict[str, str], tmp_path: Path
) -> None:
    assert guard.evaluate("fakecli up acme/tiny --port 9000", tmp_path, nd.utc_now())[0] == "allow"


def test_the_docker_runner_forms_are_allowed_for_an_approved_docker_run(
    fake_runner: dict[str, str], tmp_path: Path
) -> None:
    state = make_run(DOCKER_VALUES, target="docker", state="render", run_id="dock")
    folder = nd.run_dir("dock")
    paths = runners.load("docker").render(nd.run_inputs(state), folder)
    nd.transition(state, "validate", [str(p) for p in paths])
    nd.transition(state, "approve")
    with_docs(state)
    summary = run_cli("nimdeploy.py", "approval-summary", "dock", env=fake_runner)[1]["facts"]
    run_cli("nimdeploy.py", "approve", "dock", "--token", summary["token"], env=fake_runner)
    now = nd.utc_now()
    assert guard.evaluate(f"bash {folder}/docker-run.sh", tmp_path, now)[0] == "allow"
    assert guard.evaluate("docker rm -f nim-llama", tmp_path, now)[0] == "allow"
    assert guard.evaluate("docker rm -f other", tmp_path, now)[0] == "deny"


def test_a_bad_runner_name_is_refused_at_the_command_line(fake_runner: dict[str, str]) -> None:
    code, payload = run_cli(
        "nimdeploy.py",
        "new",
        "--target",
        "docker",
        "--model",
        "m",
        "--set",
        "runner=nope",
        env=fake_runner,
    )
    assert (code, payload["errors"][0]["code"]) == (2, "bad_runner")
    assert_contract(payload)
    assert json.dumps(payload).count("fake") >= 1  # the fix lists the registered runners
