# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for scripts/nimdeploy.py: the run state machine, approval token and resume point."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import nd_common as nd
import nextstep
import nimdeploy
import pytest
from _testkit import (
    DOCKER_VALUES,
    FAKE_HF,
    FAKE_NVAPI,
    K8S_VALUES,
    assert_contract,
    docs_cli,
    make_run,
    run_cli,
    with_docs,
)
from _testkit import runs_home as runs_home

pytestmark = pytest.mark.usefixtures("runs_home")


def call(capsys: pytest.CaptureFixture[str], *argv: str) -> tuple[int, dict[str, Any]]:
    """Run nimdeploy.main in-process; return (exit code, the JSON object it printed)."""
    code = 0
    try:
        nimdeploy.main(list(argv))
    except SystemExit as exit_info:
        code = int(exit_info.code or 0)
    payload = json.loads(capsys.readouterr().out)
    assert_contract(payload)
    return code, payload


def sets(values: dict[str, Any]) -> list[str]:
    return [
        item
        for key, value in values.items()
        if key != "model"
        for item in ("--set", f"{key}={value}")
    ]


def new_operator(capsys: pytest.CaptureFixture[str]) -> str:
    _, payload = call(capsys, "new", "--target", "nim-operator", *sets(K8S_VALUES))
    return str(payload["facts"]["run_id"])


def test_new_with_missing_inputs_waits_in_plan(capsys: pytest.CaptureFixture[str]) -> None:
    code, payload = call(
        capsys, "new", "--target", "docker", "--model", "meta/llama-3.1-8b-instruct"
    )
    assert code == 0
    assert (payload["state"], payload["next"]) == ("plan", "ask_user")
    assert payload["facts"]["missing"] == ["image"]  # name, port, gpus default
    assert "set " in payload["facts"]["next_command"]


def test_new_with_all_inputs_goes_to_render(capsys: pytest.CaptureFixture[str]) -> None:
    run = new_operator(capsys)
    _, payload = call(capsys, "status", run)
    assert payload["state"] == "render"
    assert payload["facts"]["next_command"].endswith(
        f"render_k8s.py --run {run} --kind nim-operator"
    )


def test_set_completes_inputs_and_moves_to_render(capsys: pytest.CaptureFixture[str]) -> None:
    _, payload = call(capsys, "new", "--target", "docker", "--model", "m")
    run = payload["facts"]["run_id"]
    code, payload = call(
        capsys, "set", run, "image=nvcr.io/nim/x:1", "gpus=all", "port=8000", "name=n"
    )
    assert (code, payload["state"]) == (0, "render")


def test_set_refuses_secret_values_with_exit_5(capsys: pytest.CaptureFixture[str]) -> None:
    run = new_operator(capsys)
    code, payload = call(capsys, "set", run, f"ngc_api_key={FAKE_NVAPI}")
    assert code == nd.EXIT_REFUSED
    assert payload["errors"][0]["code"] == "secret_input"
    assert "nvapi" not in json.dumps(nd.load_state(run))


def test_new_refuses_a_secret_in_set(capsys: pytest.CaptureFixture[str]) -> None:
    code, payload = call(
        capsys, "new", "--target", "docker", "--model", "m", "--set", f"hf_token={FAKE_HF}"
    )
    assert (code, payload["errors"][0]["code"]) == (nd.EXIT_REFUSED, "secret_input")
    assert list((nd.runs_root() / "runs").glob("*")) == []


def test_set_rejects_malformed_pairs(capsys: pytest.CaptureFixture[str]) -> None:
    run = new_operator(capsys)
    code, payload = call(capsys, "set", run, "Not-Valid=1")
    assert (code, payload["errors"][0]["code"]) == (nd.EXIT_INPUT, "bad_input")


def test_unknown_run_is_an_input_error(capsys: pytest.CaptureFixture[str]) -> None:
    code, payload = call(capsys, "status", "nope")
    assert (code, payload["errors"][0]["code"]) == (nd.EXIT_INPUT, "run_not_found")


def test_advance_rejects_illegal_edges_and_is_a_noop_for_the_current_state(
    capsys: pytest.CaptureFixture[str],
) -> None:
    run = new_operator(capsys)
    code, payload = call(capsys, "advance", run, "--to", "apply")
    assert (code, payload["errors"][0]["code"]) == (nd.EXIT_INPUT, "illegal_transition")
    code, payload = call(capsys, "advance", run, "--to", "render")
    assert (code, payload["state"]) == (0, "render")


def test_three_visits_then_ask_the_user(capsys: pytest.CaptureFixture[str]) -> None:
    run = new_operator(capsys)
    for _ in range(nd.MAX_VISITS - 1):  # render was visited once by `new`
        call(capsys, "advance", run, "--to", "validate")
        call(capsys, "advance", run, "--to", "render")
    call(capsys, "advance", run, "--to", "validate")
    code, payload = call(capsys, "advance", run, "--to", "render")
    assert (code, payload["errors"][0]["code"]) == (nd.EXIT_PRECONDITION, "too_many_attempts")
    assert payload["next"] == "ask_user"


def render_and_validate(run: str) -> dict[str, Any]:
    """Render and validate a nim-operator run offline; return validate's facts."""
    env = {**os.environ, "NIM_DEPLOY_HOME": str(nd.runs_root())}
    code, _ = run_cli("render_k8s.py", "--run", run, "--kind", "nim-operator", env=env)
    assert code == 0
    docs_cli(run, env)
    code, payload = run_cli("validate.py", "--run", run, "--offline", env=env)
    assert code == 0
    return payload


def test_full_approval_flow(capsys: pytest.CaptureFixture[str]) -> None:
    run = new_operator(capsys)
    summary = render_and_validate(run)["facts"]
    assert summary["token"] and summary["apply_command"].startswith("kubectl apply -f ")
    assert "--context prod-a -n nim-prod" in summary["apply_command"]

    code, payload = call(capsys, "approve", run, "--token", "00000000")
    assert (code, payload["errors"][0]["code"]) == (nd.EXIT_REFUSED, "token_mismatch")
    assert nd.load_state(run)["state"] == "approve"


    code, payload = call(capsys, "approve", run, "--token", summary["token"])
    assert (code, payload["state"], payload["next"]) == (0, "apply", "apply")
    record = nd.load_state(run)["approvals"][-1]
    assert record["context"] == "prod-a" and record["namespace"] == "nim-prod"
    assert list(record["artifacts"]) == [str(nd.run_dir(run) / "nimservice.yaml")]
    assert payload["facts"]["then_command"].endswith(f"advance {run} --to wait_ready")

    code, payload = call(capsys, "set", run, "namespace=other")
    assert (code, payload["errors"][0]["code"]) == (nd.EXIT_REFUSED, "run_approved")


def test_review_unverified_labels_need_an_explicit_recorded_confirmation(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    run = new_operator(capsys)
    token = render_and_validate(run)["facts"]["token"]
    state = nd.load_state(run)
    state["placement_check"] = {"status": "unverified"}
    nd.save_state(state)
    _, summary = call(capsys, "approval-summary", run)
    assert "--confirm-placement" not in summary["facts"]["approve_command"]
    checked = {"status": "unverified", "selector": "pool=a"}
    monkeypatch.setattr(nimdeploy.placement, "check", lambda _state: (checked, []))
    code, payload = call(capsys, "approve", run, "--token", token)
    assert code == nd.EXIT_PRECONDITION
    assert payload["errors"][0]["code"] == "placement_confirmation_required"
    code, payload = call(
        capsys, "approve", run, "--token", token, "--confirm-placement"
    )
    assert (code, payload["state"]) == (0, "apply")
    confirmation = nd.load_state(run)["approvals"][-1]["placement_confirmation"]
    assert confirmation["selector"] == "pool=a" and confirmation["token"] == token


def test_approve_refuses_a_placement_check_weaker_than_the_approved_summary(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    run = new_operator(capsys)
    token = render_and_validate(run)["facts"]["token"]
    state = nd.load_state(run)
    state["placement_check"] = {"status": "verified", "nodes_matched": ["n1"]}
    nd.save_state(state)
    now = {"status": "unverified", "reason": "nodes is forbidden"}
    monkeypatch.setattr(nimdeploy.placement, "check", lambda _state: (now, []))
    code, payload = call(capsys, "approve", run, "--token", token)
    assert (code, payload["errors"][0]["code"]) == (nd.EXIT_PRECONDITION, "placement_weaker")
    assert "advance" in payload["errors"][0]["fix"]
    kept = nd.load_state(run)
    assert kept["state"] == "approve" and kept["placement_check"]["status"] == "verified"
    now = {"status": "verified", "nodes_matched": ["n1"]}
    code, payload = call(capsys, "approve", run, "--token", token)
    assert (code, payload["state"]) == (0, "apply")


@pytest.mark.parametrize(
    ("now", "error"),
    [
        ("unverified", "placement_confirmation_required"),
        ("skipped", "placement_unchecked"),
    ],
)
def test_review_approval_requires_shown_confirmation_or_a_live_check(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    now: str,
    error: str,
) -> None:
    run = new_operator(capsys)
    token = render_and_validate(run)["facts"]["token"]
    state = nd.load_state(run)
    state["placement_check"] = {"status": "skipped"}
    nd.save_state(state)
    monkeypatch.setattr(
        nimdeploy.placement, "check", lambda _state: ({"status": now}, [])
    )
    code, payload = call(capsys, "approve", run, "--token", token)
    assert (code, payload["errors"][0]["code"]) == (nd.EXIT_PRECONDITION, error)
    assert nd.load_state(run)["state"] == "approve"


def test_token_changes_when_the_approved_file_changes(capsys: pytest.CaptureFixture[str]) -> None:
    run = new_operator(capsys)
    token = render_and_validate(run)["facts"]["token"]
    (nd.run_dir(run) / "nimservice.yaml").write_text("edited: true\n")
    code, payload = call(capsys, "approve", run, "--token", token)
    assert (code, payload["errors"][0]["code"]) == (nd.EXIT_REFUSED, "token_mismatch")


def test_approve_outside_the_approve_state_is_a_precondition_error(
    capsys: pytest.CaptureFixture[str],
) -> None:
    run = new_operator(capsys)
    code, payload = call(capsys, "approve", run, "--token", "abcd1234")
    assert (code, payload["errors"][0]["code"]) == (nd.EXIT_PRECONDITION, "wrong_state")


def test_record_finishes_the_run_and_appends_a_history_line(
    capsys: pytest.CaptureFixture[str],
) -> None:
    make_run(K8S_VALUES, state="verify")
    code, payload = call(capsys, "record", "run1", "--result", "ok")
    assert (code, payload["state"]) == (0, "done")
    line = json.loads((nd.runs_root() / "runs.jsonl").read_text())
    assert (line["run_id"], line["result"]) == ("run1", "ok")


def test_record_is_illegal_early(capsys: pytest.CaptureFixture[str]) -> None:
    make_run(K8S_VALUES, state="render")
    code, payload = call(capsys, "record", "run1", "--result", "ok")
    assert (code, payload["errors"][0]["code"]) == (nd.EXIT_INPUT, "illegal_transition")


def test_list_shows_runs(capsys: pytest.CaptureFixture[str]) -> None:
    _, empty = call(capsys, "list")
    assert empty["facts"]["runs"] == []
    run = new_operator(capsys)
    _, payload = call(capsys, "list")
    assert payload["facts"]["runs"][0]["run_id"] == run


@pytest.mark.parametrize(
    ("target", "values", "state", "apply_prefix", "rollback_prefix"),
    [
        ("nim-operator", K8S_VALUES, "apply", "kubectl apply -f ", "kubectl delete -f "),
        ("docker", DOCKER_VALUES, "apply", "bash ", "docker rm -f nim-llama"),
    ],
)
def test_apply_and_rollback_commands(
    target: str, values: dict[str, Any], state: str, apply_prefix: str, rollback_prefix: str
) -> None:
    record = make_run(values, target=target, state=state)
    assert nextstep.apply_command(record).startswith(apply_prefix)
    assert nextstep.rollback_command(record).startswith(rollback_prefix)


def test_helm_commands_carry_context_and_namespace() -> None:
    record = make_run(K8S_VALUES, target="helm", state="apply")
    folder = nd.run_dir("run1")
    (folder / "values.yaml").write_text("a: 1\n")
    head = "helm upgrade --install llama nim/nim-llm --kube-context prod-a -n nim-prod -f"
    (folder / "helm-cmd.txt").write_text(f"{head} values.yaml\n")
    assert nextstep.apply_command(record) == f"{head} {folder}/values.yaml"
    assert (
        nextstep.rollback_command(record)
        == "helm uninstall llama -n nim-prod --kube-context prod-a"
    )


def test_status_gives_the_resume_command_for_every_state() -> None:
    expected = {
        "validate": "validate.py --run run1",
        "approve": "approval-summary run1",
        "wait_ready": "verify.py ready --url http://localhost:8000 --run run1",
        "verify": "verify.py smoke --url http://localhost:8000 --model meta/llama-3.1-8b-instruct",
        "record": "record run1 --result ok",
        "diagnose": "verify.py triage --k8s nim-prod/POD --context prod-a --run run1",
    }
    for state, text in expected.items():
        record = with_docs(make_run(K8S_VALUES, state=state))
        assert text in nextstep.next_step(record)[1], state


def test_status_in_wait_ready_gives_a_port_forward_for_kubernetes() -> None:
    record = make_run(K8S_VALUES, state="wait_ready")
    forward = nextstep.status_facts(record)["port_forward"]
    assert forward == "kubectl port-forward svc/llama 8000:8000 --context prod-a -n nim-prod"
    assert "port_forward" not in nextstep.status_facts(
        make_run(DOCKER_VALUES, "docker", "wait_ready")
    )


def test_an_unknown_runner_is_refused(capsys: pytest.CaptureFixture[str]) -> None:
    code, payload = call(
        capsys, "new", "--target", "docker", "--model", "m", "--set", "runner=nope"
    )
    assert (code, payload["errors"][0]["code"]) == (nd.EXIT_INPUT, "bad_runner")
    assert "docker" in payload["errors"][0]["fix"]


def test_the_runner_input_is_frozen_into_the_approval(capsys: pytest.CaptureFixture[str]) -> None:
    code, payload = call(
        capsys, "new", "--target", "docker", "--model", "m", "--set", "runner=docker"
    )
    assert code == 0 and payload["facts"]["inputs"]["runner"] == "docker"
    state = make_run(DOCKER_VALUES, target="docker", state="approve")
    assert nd.approval_bindings(state)["runner"] == "docker"
    assert nd.approval_bindings(make_run(K8S_VALUES))["runner"] is None


def test_cli_contract_and_exit_codes(tmp_path: Path) -> None:
    env = {**os.environ, "NIM_DEPLOY_HOME": str(tmp_path / "home")}
    code, payload = run_cli("nimdeploy.py", "new", "--target", "docker", "--model", "m", env=env)
    assert (code, payload["state"]) == (0, "plan")
    code, payload = run_cli("nimdeploy.py", "status", "nope", env=env)
    assert (code, payload["errors"][0]["code"]) == (2, "run_not_found")
    code, payload = run_cli("nimdeploy.py", "new", "--target", "bogus", "--model", "m", env=env)
    assert (code, payload["errors"][0]["code"]) == (2, "bad_input")


def test_an_unsupported_target_is_an_input_error_that_lists_the_supported_ones(
    tmp_path: Path,
) -> None:
    env = {**os.environ, "NIM_DEPLOY_HOME": str(tmp_path / "home")}
    code, payload = run_cli("nimdeploy.py", "new", "--target", "podman", "--model", "m", env=env)
    assert (code, payload["errors"][0]["code"]) == (2, "bad_input")
    message = payload["errors"][0]["message"]
    assert "invalid choice: 'podman'" in message
    assert all(target in message for target in nd.TARGETS)
    assert not (tmp_path / "home").exists()  # no run was created
