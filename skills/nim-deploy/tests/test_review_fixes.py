# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for the MR 19 review threads: env inputs, rollback after done, stdin shells."""

from __future__ import annotations

import os
from datetime import timedelta
from pathlib import Path
from typing import Any

import guard
import nd_common as nd
import pytest
from _testkit import DOCKER_VALUES, K8S_VALUES, docs_cli, make_run, run_cli, with_docs
from _testkit import runs_home as runs_home

pytestmark = pytest.mark.usefixtures("runs_home")

IMAGE = "nvcr.io/nim/meta/llama-3.1-8b-instruct:1.8"


def env() -> dict[str, str]:
    return {**os.environ, "NIM_DEPLOY_HOME": str(nd.runs_root())}


def set_inputs(run: str, *pairs: str) -> tuple[int, dict[str, Any]]:
    return run_cli("nimdeploy.py", "set", run, *pairs, env=env())


def new_docker() -> str:
    _, payload = run_cli(
        "nimdeploy.py", "new", "--target", "docker", "--set", f"image={IMAGE}", env=env()
    )
    return str(payload["facts"]["run_id"])


# --- thread 1: extra env can be set ---


def test_env_values_are_set_one_variable_at_a_time_and_merge() -> None:
    run = new_docker()
    code, payload = set_inputs(run, "env.NIM_MAX_MODEL_LEN=8192")
    assert code == 0 and payload["facts"]["inputs"]["env"] == {"NIM_MAX_MODEL_LEN": "8192"}
    code, payload = set_inputs(run, "env.NIM_LOG_LEVEL=DEBUG", "env.NIM_MAX_MODEL_LEN=4096")
    assert payload["facts"]["inputs"]["env"] == {
        "NIM_LOG_LEVEL": "DEBUG",
        "NIM_MAX_MODEL_LEN": "4096",
    }
    assert nd.load_state(run)["inputs"]["env"]["source"] == "user"


def test_the_env_mapping_reaches_the_docker_script_and_the_helm_values() -> None:
    run = new_docker()
    set_inputs(run, "env.NIM_MAX_MODEL_LEN=8192")
    assert run_cli("render_docker.py", "--run", run, env=env())[0] == 0
    assert "-e NIM_MAX_MODEL_LEN=8192" in (nd.run_dir(run) / "docker-run.sh").read_text()
    flags = [i for k, v in K8S_VALUES.items() if k != "model" for i in ("--set", f"{k}={v}")]
    _, payload = run_cli("nimdeploy.py", "new", "--target", "helm", *flags, env=env())
    helm = payload["facts"]["run_id"]
    set_inputs(helm, "env.NIM_MAX_MODEL_LEN=8192")
    assert run_cli("render_k8s.py", "--run", helm, "--kind", "helm", env=env())[0] == 0
    assert "NIM_MAX_MODEL_LEN" in (nd.run_dir(helm) / "values.yaml").read_text()


@pytest.mark.parametrize("pair", ["env.HF_TOKEN=abc", "env.NGC_API_KEY=abc", "env.SOME_PASSWORD=x"])
def test_secret_named_env_values_are_still_refused(pair: str) -> None:
    run = new_docker()
    code, payload = set_inputs(run, pair)
    assert (code, payload["errors"][0]["code"]) == (nd.EXIT_REFUSED, "secret_input")
    assert "env" not in nd.load_state(run)["inputs"]


@pytest.mark.parametrize("pair", ["env.=1", "env.1X=1", "env.a-b=1", "env.A B=1"])
def test_bad_env_names_are_input_errors(pair: str) -> None:
    code, payload = set_inputs(new_docker(), pair)
    assert (code, payload["errors"][0]["code"]) == (2, "bad_input")


def test_a_secret_shaped_env_value_is_refused(tmp_path: Path) -> None:
    from _testkit import FAKE_NVAPI

    code, payload = set_inputs(new_docker(), f"env.NOTE={FAKE_NVAPI}")
    assert (code, payload["errors"][0]["code"]) == (nd.EXIT_REFUSED, "secret_input")


# --- thread 2: a finished run can be rolled back ---


def done_run() -> dict[str, Any]:
    state = with_docs(make_run(DOCKER_VALUES, target="docker", state="verify"))
    folder = nd.run_dir("run1")
    (folder / "docker-run.sh").write_text("#!/bin/sh\n")
    state["artifacts"][str(folder / "docker-run.sh")] = nd.sha256_file(folder / "docker-run.sh")
    expires = (nd.utc_now() + timedelta(minutes=30)).isoformat()
    state["approvals"].append(
        {**nd.approval_bindings(state), "artifacts": state["artifacts"], "expires_at": expires}
    )
    nd.save_state(state)
    assert (
        run_cli("nimdeploy.py", "record", "run1", "--result", "ok", env=env())[1]["state"] == "done"
    )
    return nd.load_state("run1")


def test_a_done_run_can_go_to_diagnose_and_rollback_and_be_recorded_again() -> None:
    done_run()
    code, payload = run_cli("nimdeploy.py", "advance", "run1", "--to", "diagnose", env=env())
    assert (code, payload["state"]) == (0, "diagnose")
    code, payload = run_cli("nimdeploy.py", "advance", "run1", "--to", "rollback", env=env())
    assert (code, payload["state"], payload["next"]) == (0, "rollback", "rollback")
    assert payload["facts"]["next_command"] == "docker rm -f nim-llama"
    code, payload = run_cli(
        "nimdeploy.py", "record", "run1", "--result", "ok", "--reason", "rolled back", env=env()
    )
    assert (code, payload["state"]) == (0, "done")


def test_the_visit_limit_still_applies_after_done() -> None:
    done_run()
    state = nd.load_state("run1")
    state["attempts"]["diagnose"] = nd.MAX_VISITS
    nd.save_state(state)
    code, payload = run_cli("nimdeploy.py", "advance", "run1", "--to", "diagnose", env=env())
    assert (code, payload["errors"][0]["code"]) == (3, "too_many_attempts")


def test_rollback_needs_a_live_approval_and_the_user_can_renew_it(tmp_path: Path) -> None:
    state = done_run()
    run_cli("nimdeploy.py", "advance", "run1", "--to", "diagnose", env=env())
    run_cli("nimdeploy.py", "advance", "run1", "--to", "rollback", env=env())
    now = nd.utc_now()
    assert guard.evaluate("docker rm -f nim-llama", tmp_path, now)[0] == "allow"  # live approval
    late = now + timedelta(hours=3)
    assert guard.evaluate("docker rm -f nim-llama", tmp_path, late)[0] == "deny"  # expired
    # renewing: the summary names the rollback command and the token approves only that
    state = nd.load_state("run1")
    state["approvals"][-1]["expires_at"] = (now - timedelta(minutes=1)).isoformat()
    nd.save_state(state)
    code, payload = run_cli("nimdeploy.py", "approval-summary", "run1", env=env())
    assert code == 0 and payload["facts"]["apply_command"] == "docker rm -f nim-llama"
    assert "rollback" in payload["facts"]["instruction"].lower()
    code, payload = run_cli(
        "nimdeploy.py", "approve", "run1", "--token", payload["facts"]["token"], env=env()
    )
    assert (code, payload["state"]) == (0, "rollback")
    assert guard.evaluate("docker rm -f nim-llama", tmp_path, nd.utc_now())[0] == "allow"
    assert docs_cli is not None


# --- thread 3: a shell that reads its command from stdin ---


@pytest.mark.parametrize(
    "command",
    [
        "echo 'nim stop --all' | bash",
        "printf 'nim diag --fix\\n' | sh -s",
        "bash <<< 'nim stop --all'",
        "cat <<'EOF' | bash\nnim rm --all\nEOF",
        "echo 'nim uninstall' | zsh",
        "echo 'docker system prune -af' | sh",
        "python3 - <<< \"import os; os.system('nim stop --all')\"",
    ],
)
def test_a_shell_reading_a_denied_command_from_stdin_is_denied(command: str) -> None:
    assert guard.evaluate(command, Path.cwd(), nd.utc_now())[0] == "deny", command


@pytest.mark.parametrize(
    "command",
    ["echo hello | bash", "echo 'nim list' | sh", "bash <<< 'echo hi'", "printf 'ls\\n' | sh -s"],
)
def test_harmless_stdin_scripts_still_run(command: str) -> None:
    assert guard.evaluate(command, Path.cwd(), nd.utc_now())[0] == "allow", command


# --- thread 4: read-only nim commands stay allowed while approved ---


def approved_nimcli() -> None:
    _, payload = run_cli(
        "nimdeploy.py", "new", "--target", "docker", "--model", "meta/llama-3.1-8b-instruct",
        "--set", "runner=nimcli", "--set", "port=8000", env=env(),
    )  # fmt: skip
    run = payload["facts"]["run_id"]
    assert run_cli("render_docker.py", "--run", run, env=env())[0] == 0
    docs_cli(run, env())
    token = run_cli("validate.py", "--run", run, "--offline", env=env())[1]["facts"]["token"]
    assert run_cli("nimdeploy.py", "approve", run, "--token", token, env=env())[0] == 0


@pytest.mark.parametrize(
    "command",
    ["nim list", "nim list --running", "nim info", "nim diag", "nim search llama", "nim --version"],
)
def test_read_only_nim_commands_pass_while_a_nimcli_run_is_approved(command: str) -> None:
    approved_nimcli()
    assert guard.evaluate(command, Path.cwd(), nd.utc_now())[0] == "allow", command


@pytest.mark.parametrize(
    "command",
    [
        "nim stop other/model",
        "nim stop --all",
        "nim diag --fix",
        "nim run meta/llama-3.1-8b-instruct",
        "nim rm x",
    ],
)
def test_changing_nim_commands_are_still_denied_while_approved(command: str) -> None:
    approved_nimcli()
    assert guard.evaluate(command, Path.cwd(), nd.utc_now())[0] == "deny", command


# --- thread: an input changed after render must not leave stale artifacts ---


def drive_to_approve() -> tuple[str, str]:
    run = new_docker()
    assert run_cli("render_docker.py", "--run", run, env=env())[0] == 0
    docs_cli(run, env())
    code, payload = run_cli("validate.py", "--run", run, "--offline", env=env())
    assert (code, payload["state"]) == (0, "approve")
    return run, str(payload["facts"]["token"])


def test_changing_the_name_in_approve_returns_the_run_to_render() -> None:
    run, old_token = drive_to_approve()
    script = nd.run_dir(run) / "docker-run.sh"
    assert "--name llama-3.1-8b-instruct" in script.read_text()
    code, payload = set_inputs(run, "name=renamed")
    assert (code, payload["state"], payload["next"]) == (0, "render", "render")
    state = nd.load_state(run)
    assert state["state"] == "render" and state["artifacts"] == {}
    assert run_cli("render_docker.py", "--run", run, env=env())[0] == 0
    assert "--name renamed" in script.read_text()
    code, payload = run_cli("validate.py", "--run", run, "--offline", env=env())
    assert (code, payload["state"]) == (0, "approve")
    assert payload["facts"]["token"] != old_token
    assert payload["facts"]["name"] == "renamed"


def test_changing_the_port_after_render_returns_the_run_to_render() -> None:
    run = new_docker()
    assert run_cli("render_docker.py", "--run", run, env=env())[0] == 0
    assert nd.load_state(run)["state"] == "validate"
    code, payload = set_inputs(run, "port=9123")
    assert (code, payload["state"]) == (0, "render")
    assert nd.load_state(run)["artifacts"] == {}


def test_setting_the_same_value_again_keeps_the_validated_run() -> None:
    run, _ = drive_to_approve()
    code, payload = set_inputs(run, "name=llama-3.1-8b-instruct")
    assert (code, payload["state"]) == (0, "approve")
    assert nd.load_state(run)["artifacts"]


def test_a_changed_kubernetes_input_after_validate_returns_to_render() -> None:
    flags = [i for k, v in K8S_VALUES.items() if k != "model" for i in ("--set", f"{k}={v}")]
    _, payload = run_cli("nimdeploy.py", "new", "--target", "helm", *flags, env=env())
    run = payload["facts"]["run_id"]
    assert run_cli("render_k8s.py", "--run", run, "--kind", "helm", env=env())[0] == 0
    code, payload = set_inputs(run, "namespace=other-ns")
    assert (code, payload["state"]) == (0, "render")
    assert nd.load_state(run)["artifacts"] == {}
