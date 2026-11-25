# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for scripts/validate.py: drift detection, structure checks and live checks (fake tools)."""

from __future__ import annotations

import json
import os
import shutil
import socket
from pathlib import Path
from typing import Any

import nd_common as nd
import pytest
import render_k8s
import runners
import validate
from _testkit import (
    DOCKER_VALUES,
    FAKE_NVAPI,
    K8S_VALUES,
    assert_contract,
    fake_tool,
    make_run,
    run_cli,
    with_docs,
)
from _testkit import runs_home as runs_home

pytestmark = pytest.mark.usefixtures("runs_home")


def prepare(target: str, values: dict[str, Any]) -> dict[str, Any]:
    """A run in the validate state with its artifacts rendered, as the render scripts leave it."""
    state = make_run(values, target=target, state="render")
    inputs = nd.run_inputs(state)
    folder = nd.run_dir("run1")
    files: list[Path] = []
    if target == "nim-operator":
        (path := folder / "nimservice.yaml").write_text(render_k8s.operator_text(inputs))
        files = [path]
    elif target == "helm":
        helm_values, command = render_k8s.render_helm(inputs)
        (folder / "values.yaml").write_text(nd.to_yaml(helm_values))
        (folder / "helm-cmd.txt").write_text(command)
        files = [folder / "values.yaml", folder / "helm-cmd.txt"]
    else:
        files = runners.load(inputs.get("runner")).render(inputs, folder)
    nd.transition(state, "validate", [str(path) for path in files])
    return with_docs(state)


def run_validate(capsys: pytest.CaptureFixture[str], *extra: str) -> tuple[int, dict[str, Any]]:
    code = 0
    try:
        validate.main(["--run", "run1", *extra])
    except SystemExit as exit_info:
        code = int(exit_info.code or 0)
    payload = json.loads(capsys.readouterr().out)
    assert_contract(payload)
    return code, payload


@pytest.mark.parametrize(
    ("target", "values"),
    [
        ("nim-operator", K8S_VALUES),
        ("helm", K8S_VALUES),
        ("docker", DOCKER_VALUES),
    ],
)
def test_offline_pass_moves_to_approve_with_a_summary(
    capsys: pytest.CaptureFixture[str], target: str, values: dict[str, Any]
) -> None:
    prepare(target, values)
    code, payload = run_validate(capsys, "--offline")
    assert (code, payload["state"], payload["next"]) == (0, "approve", "ask_user")
    assert payload["facts"]["token"] and payload["facts"]["apply_command"]
    expected = {"live": "skipped"} if target == "docker" else {"live": "skipped", "placement": "none"}
    assert payload["facts"]["checks"] == expected


def test_a_hand_edited_file_is_drift_and_goes_back_to_render(
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepare("nim-operator", K8S_VALUES)
    path = nd.run_dir("run1") / "nimservice.yaml"
    path.write_text(path.read_text().replace("replicas: 1", "replicas: 9", 1))
    code, payload = run_validate(capsys, "--offline")
    assert (code, payload["state"], payload["next"]) == (3, "render", "render")
    assert payload["errors"][0]["code"] == "file_drift"
    assert "Never edit rendered files" in payload["errors"][0]["fix"]


def test_missing_artifact_is_reported(capsys: pytest.CaptureFixture[str]) -> None:
    prepare("docker", DOCKER_VALUES)
    (nd.run_dir("run1") / "docker-run.sh").unlink()
    _, payload = run_validate(capsys, "--offline")
    assert payload["errors"][0]["code"] == "artifact_missing"


def test_input_errors_are_reported_not_raised(capsys: pytest.CaptureFixture[str]) -> None:
    state = prepare("docker", DOCKER_VALUES)
    state["inputs"]["port"]["value"] = "99999"
    nd.save_state(state)
    _, payload = run_validate(capsys, "--offline")
    assert payload["errors"][0]["code"] == "render_input"


def test_a_run_from_results_without_a_pinned_profile_fails_validation(
    capsys: pytest.CaptureFixture[str],
) -> None:
    state = prepare("docker", {**DOCKER_VALUES, "model_profile": "f" * 64})
    state["handoff"] = {
        "files": [],
        "profile": {"id": "f" * 64},
        "tuning": "not tuned",
        "values": {},
        "warnings": [],
    }
    del state["inputs"]["model_profile"]
    nd.save_state(state)
    _, payload = run_validate(capsys, "--offline")
    error = payload["errors"][0]
    assert error["code"] == "profile_not_pinned"
    assert "Never drop model_profile" in error["fix"]
    assert "token" not in payload["facts"]


def test_validate_outside_its_state_is_a_precondition_error(
    capsys: pytest.CaptureFixture[str],
) -> None:
    make_run(K8S_VALUES, state="render")
    code, payload = run_validate(capsys, "--offline")
    assert (code, payload["errors"][0]["code"]) == (3, "wrong_state")


def test_the_fourth_render_visit_stops_the_loop(capsys: pytest.CaptureFixture[str]) -> None:
    state = prepare("nim-operator", K8S_VALUES)
    state["attempts"]["render"] = nd.MAX_VISITS
    nd.save_state(state)
    (nd.run_dir("run1") / "nimservice.yaml").write_text("edited: true\n")
    code, payload = run_validate(capsys, "--offline")
    assert (code, payload["next"]) == (3, "ask_user")
    assert [e["code"] for e in payload["errors"]] == ["file_drift", "too_many_attempts"]
    assert payload["state"] == "validate"


def test_check_helm_catches_set_and_scope_mismatch() -> None:
    state = make_run(K8S_VALUES, target="helm")
    good = (
        "helm upgrade --install llama nim/nim-llm --kube-context prod-a -n nim-prod -f values.yaml"
    )
    assert validate.check_helm(good, state) == []
    bad = "helm upgrade llama nim/nim-llm --set a=1 -n other -f values.yaml"
    assert {e["code"] for e in validate.check_helm(bad, state)} == {
        "namespace_mismatch",
        "context_mismatch",
        "helm_set_flag",
    }


def test_a_secret_added_to_the_docker_script_is_drift(capsys: pytest.CaptureFixture[str]) -> None:
    prepare("docker", DOCKER_VALUES)
    path = nd.run_dir("run1") / "docker-run.sh"
    path.write_text(path.read_text().replace("-e NGC_API_KEY ", f"-e NGC_API_KEY={FAKE_NVAPI} "))
    _, payload = run_validate(capsys, "--offline")
    assert payload["errors"][0]["code"] == "file_drift" and "nvapi" not in json.dumps(payload)


# --- live checks with fake tools ---------------------------------------------------------------


def use_fake_tools(monkeypatch: pytest.MonkeyPatch, folder: Path, **scripts: str) -> None:
    for name, body in scripts.items():
        fake_tool(folder, name, body)
    monkeypatch.setenv("PATH", f"{folder}{os.pathsep}{os.environ['PATH']}")


@pytest.mark.parametrize(
    ("script", "code", "next_"),
    [
        (
            'echo "no matches for kind NIMService in version v1beta1" >&2; exit 1',
            "crd_missing",
            "ask_user",
        ),
        (
            'echo "Error: context \\"kind-a\\" does not exist" >&2; exit 1',
            "context_missing",
            "ask_user",
        ),
        (
            'echo "error: You must be logged in: Unauthorized" >&2; exit 1',
            "cluster_auth",
            "ask_user",
        ),
        ('echo "admission webhook denied the request" >&2; exit 1', "dry_run_rejected", "render"),
    ],
)
def test_server_dry_run_failures(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    script: str,
    code: str,
    next_: str,
) -> None:
    prepare("nim-operator", K8S_VALUES)
    use_fake_tools(monkeypatch, tmp_path / "bin", kubectl=script)
    status, payload = run_validate(capsys)
    assert (status, payload["errors"][0]["code"], payload["next"]) == (3, code, next_)


def test_server_dry_run_passes_with_the_run_scope(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    prepare("nim-operator", K8S_VALUES)
    use_fake_tools(monkeypatch, tmp_path / "bin", kubectl=f'echo "$@" > {tmp_path}/args; exit 0')
    code, payload = run_validate(capsys)
    assert (code, payload["facts"]["checks"]["live"]) == (0, "ok")
    args = (tmp_path / "args").read_text().split()
    assert (
        "--dry-run=server" in args
        and args[args.index("--context") + 1] == "prod-a"
        and args[args.index("-n") + 1] == "nim-prod"
    )


def test_unreachable_cluster_skips_the_live_check(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    prepare("nim-operator", K8S_VALUES)
    use_fake_tools(
        monkeypatch,
        tmp_path / "bin",
        kubectl='echo "Unable to connect to the server: dial tcp" >&2; exit 1',
    )
    code, payload = run_validate(capsys)
    assert (code, payload["facts"]["checks"]["live"]) == (0, "skipped")


def test_helm_template_failure(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    prepare("helm", K8S_VALUES)
    use_fake_tools(monkeypatch, tmp_path / "bin", helm='echo "Error: chart not found" >&2; exit 1')
    _, payload = run_validate(capsys)
    assert payload["errors"][0]["code"] == "helm_template_failed" and payload["next"] == "render"


DOCKER_OK = 'if [ "$1" = info ]; then echo \'{"Runtimes":{"nvidia":{}}}\'; fi; exit 0'


def test_docker_preflight_passes(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    prepare("docker", {**DOCKER_VALUES, "port": str(free_port())})
    use_fake_tools(monkeypatch, tmp_path / "bin", docker=DOCKER_OK)
    code, payload = run_validate(capsys)
    assert (code, payload["facts"]["checks"]["live"]) == (0, "ok")


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("0.0.0.0", 0))
        return int(sock.getsockname()[1])


def test_docker_preflight_user_errors(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    with socket.socket() as busy:
        busy.bind(("0.0.0.0", 0))
        port = busy.getsockname()[1]
        prepare("docker", {**DOCKER_VALUES, "port": str(port)})
        script = (
            'if [ "$1" = info ]; then echo "{}"; '
            'elif [ "$1" = ps ]; then echo nim-llama; fi; exit 0'
        )
        use_fake_tools(monkeypatch, tmp_path / "bin", docker=script)
        code, payload = run_validate(capsys)
    assert code == 3 and payload["next"] == "ask_user"
    assert {e["code"] for e in payload["errors"]} == {
        "port_in_use",
        "no_nvidia_runtime",
        "name_in_use",
    }


def test_missing_docker_is_a_user_error(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    prepare("docker", {**DOCKER_VALUES, "port": str(free_port())})
    monkeypatch.setattr(shutil, "which", lambda name: None)
    _, payload = run_validate(capsys)
    assert [e["code"] for e in payload["errors"]] == ["docker_missing"] and payload[
        "next"
    ] == "ask_user"


def test_cli_contract(tmp_path: Path) -> None:
    env = {**os.environ, "NIM_DEPLOY_HOME": str(tmp_path / "h")}
    code, payload = run_cli("validate.py", "--run", "missing", env=env)
    assert (code, payload["errors"][0]["code"]) == (2, "run_not_found")
