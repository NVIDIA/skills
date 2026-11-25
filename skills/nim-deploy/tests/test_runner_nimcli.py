# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for scripts/runners/nimcli.py: the nim CLI runner, its guard forms and its denies."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import guard
import nd_common as nd
import nextstep
import pytest
import runners
import runners.docker as docker_runner
import runners.nimcli as nimcli
from _testkit import FAKE_NVAPI, GOLDEN, docs_cli, fake_tool, make_run, run_cli, with_docs
from _testkit import runs_home as runs_home

pytestmark = pytest.mark.usefixtures("runs_home")

MODEL = "meta/llama-3.1-8b-instruct"
BASE: dict[str, Any] = {"run_id": "golden", "model": MODEL, "port": 8000}
CASES: dict[str, dict[str, Any]] = {
    "basic": {},
    "full": {
        "port": 8100,
        "gpus": "0,1",
        "model_profile": "vllm-bf16-tp2",
        "env": {"NIM_MAX_MODEL_LEN": "8192"},
    },
    "model-free": {
        "model": "nvcr.io/nim/nvidia/model-free-nim:2.1.4",
        "model_path": "hf://Qwen/Qwen3-0.6B",
    },
}


@pytest.mark.parametrize("case", CASES)
def test_render_matches_golden(case: str) -> None:
    text = nimcli.render_text({**BASE, **CASES[case]})
    path = GOLDEN / f"nim-run-{case}.sh"
    if os.environ.get("UPDATE_GOLDEN"):
        path.write_text(text)
    assert text == path.read_text()
    subprocess.run(["bash", "-n"], input=text, text=True, check=True)
    assert text.rstrip().endswith("</dev/null")  # nim never waits on a prompt
    assert "NGC_API_KEY" not in text.splitlines()[-1]


def test_both_runners_implement_the_same_contract() -> None:
    names = ["required_inputs", "render", "apply_command", "rollback_command", "guard_forms"]
    names += ["ready_url", "container_name", "deny_rules", "preflight"]
    for name in names:
        assert callable(getattr(nimcli, name)) and callable(getattr(docker_runner, name)), name
    assert runners.RUNNERS["nimcli"] == "runners.nimcli"
    assert docker_runner.deny_rules() == []


@pytest.mark.parametrize(
    ("model", "name"),
    [
        (MODEL, "nim-llama-3.1-8b-instruct"),
        ("nvcr.io/nim/meta/llama-3.1-8b-instruct:1.8", "nim-llama-3.1-8b-instruct"),
        ("cnd/nim/Meta/Llama-X:latest", "nim-llama-x"),
        ("cnd-enterprise/deepseek-ai/deepseek-v4-flash", "nim-deepseek-v4-flash"),
        ("public/qwen/qwen3.5-35b", "nim-qwen3.5-35b"),
        ("localhost:5000/nim/a/b@sha256:abc", "nim-b"),
    ],
)
def test_container_name_follows_nim(model: str, name: str) -> None:
    assert nimcli.container_name({"model": model}) == name


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"model": "a b"}, "model must be"),
        ({"port": 70000}, "port must be between"),
        ({"gpus": "two"}, "gpus must be all or device ids"),
        ({"env": {"NGC_API_KEY": "x"}}, "set by nim itself"),
        ({"env": {"NIM_MODEL_PROFILE": "x"}}, "set by nim itself"),
        ({"env": {"HF_TOKEN": "x"}}, "credential"),
        ({"env": {"OK": FAKE_NVAPI}}, "credential"),
        ({"env": ["x"]}, "mapping"),
    ],
)
def test_bad_inputs_are_rejected(change: dict[str, Any], message: str) -> None:
    with pytest.raises(nd.RenderError, match=message):
        nimcli.render_text({**BASE, **change})


def test_served_model_name_is_rendered() -> None:
    text = nimcli.render_text({**BASE, "served_model_name": "qwen7b"})
    assert "NIM_SERVED_MODEL_NAME=qwen7b" in text


def approved_nimcli_run() -> str:
    env = {**os.environ, "NIM_DEPLOY_HOME": str(nd.runs_root())}
    _, payload = run_cli(
        "nimdeploy.py", "new", "--target", "docker", "--model", MODEL,
        "--set", "runner=nimcli", "--set", "port=8000", env=env,
    )  # fmt: skip
    run = str(payload["facts"]["run_id"])
    assert payload["state"] == "render"  # model and port are all nimcli needs
    code, payload = run_cli("render_docker.py", "--run", run, env=env)
    assert (code, payload["facts"]["container"]) == (0, "nim-llama-3.1-8b-instruct")
    docs_cli(run, env)
    _, payload = run_cli("validate.py", "--run", run, "--offline", env=env)
    code, payload = run_cli(
        "nimdeploy.py", "approve", run, "--token", payload["facts"]["token"], env=env
    )
    assert code == 0
    return run


def test_next_commands_and_guard_forms(tmp_path: Path) -> None:
    run = approved_nimcli_run()
    state = nd.load_state(run)
    script = nd.run_dir(run) / "nim-run.sh"
    assert nextstep.apply_command(state) == f"bash {script}"
    assert nextstep.rollback_command(state) == f"nim stop {MODEL}"
    assert nextstep.endpoint_url(state) == "http://127.0.0.1:8000"
    now = nd.utc_now()

    def decide(command: str) -> str:
        return guard.evaluate(command, tmp_path, now)[0]

    assert decide(f"bash {script}") == "allow"
    assert decide(f"nim stop {MODEL}") == "allow"
    for denied in (
        "nim stop other/model",
        "nim stop --all",
        f"nim run {MODEL} -p 8000 -d",
        f"bash {script} && nim list",
        "docker rm -f nim-llama-3.1-8b-instruct",
        f"bash {nd.run_dir(run)}/docker-run.sh",
    ):
        assert decide(denied) == "deny", denied
    script.write_text(script.read_text() + "# edited\n")
    assert decide(f"bash {script}") == "deny"


@pytest.mark.parametrize(
    ("command", "rule"),
    [
        ("nim diag --fix", "nim-diag-fix"),
        ("nim stop --all", "nim-all"),
        ("nim rm --all", "nim-all"),
        ("nim stop --all=true", "nim-all"),
        ("nim rm --all=1", "nim-all"),
        ("nim stop --all=", "nim-all"),
        ("nim uninstall --purge", "nim-uninstall"),
        (f"nim config set ngc-api-key {FAKE_NVAPI}", "secret-on-argv"),
        ("nim config set nvidia-api-key abc", "nim-config-key"),
        ("nim config registry login ngc -u me -p pw", "nim-registry-password"),
        ("nim run meta/llama-3.1-8b-instruct", "nim-raw-mutation"),
        ("nim pull meta/llama-3.1-8b-instruct", "nim-raw-mutation"),
        ("nim launch claude-code", "nim-raw-mutation"),
        ("nim update", "nim-raw-mutation"),
    ],
)
def test_nim_deny_rules(command: str, rule: str) -> None:
    decision, rule_id, reason = guard.evaluate(command, Path.cwd(), nd.utc_now())
    assert (decision, rule_id) == ("deny", rule) and "Fix:" in reason


@pytest.mark.parametrize(
    "command", ["nim diag", "nim list", "nim list --running", "nim search llama", "nim info"]
)
def test_read_only_nim_commands_pass(command: str) -> None:
    assert guard.evaluate(command, Path.cwd(), nd.utc_now())[0] == "allow"


def nimcli_validate_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, ps_line: str
) -> tuple[int, dict[str, Any]]:
    """Validate (live) a rendered nimcli run with fake nim and docker on PATH."""
    state = make_run(
        {"runner": "nimcli", "model": MODEL, "port": str(free_port())}, target="docker"
    )
    paths = nimcli.render(nd.run_inputs(state), nd.run_dir("run1"))
    nd.transition(state, "validate", [str(p) for p in paths])
    fake_tool(tmp_path / "bin", "nim", "exit 0")
    fake_tool(tmp_path / "bin", "docker", f'[ "$1" = ps ] && echo "{ps_line}"; exit 0')
    monkeypatch.setenv("PATH", f"{tmp_path / 'bin'}{os.pathsep}{os.environ['PATH']}")
    env = {**os.environ, "NIM_DEPLOY_HOME": str(nd.runs_root())}
    with_docs(nd.load_state("run1"))
    return run_cli("validate.py", "--run", "run1", env=env)


def free_port() -> int:
    import socket

    with socket.socket() as sock:
        sock.bind(("0.0.0.0", 0))
        return int(sock.getsockname()[1])


@pytest.mark.parametrize(
    "ps_line",
    [
        "some/other:1||",  # not managed by nim
        # another model with the same leaf
        "nvcr.io/nim/nvidia/llama-3.1-8b-instruct:1|true|nvidia/llama-3.1-8b-instruct",
    ],
)
def test_a_foreign_container_under_the_nim_name_is_refused(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, ps_line: str
) -> None:
    code, payload = nimcli_validate_run(monkeypatch, tmp_path, ps_line)
    assert (code, payload["next"]) == (nd.EXIT_REFUSED, "ask_user")
    assert [e["code"] for e in payload["errors"]] == ["name_collision"]


def test_the_same_model_container_is_not_a_collision(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    line = f"nvcr.io/nim/meta/llama-3.1-8b-instruct:1.8|true|{MODEL}"
    code, payload = nimcli_validate_run(monkeypatch, tmp_path, line)
    assert (code, payload["state"]) == (0, "approve")


def test_missing_nim_is_a_user_error(monkeypatch: pytest.MonkeyPatch) -> None:
    state = make_run(
        {"runner": "nimcli", "model": MODEL, "port": str(free_port())}, target="docker"
    )
    monkeypatch.setattr(shutil, "which", lambda name: None)
    status, errors = nimcli.preflight(state, offline=False)
    assert status == "failed" and [e["code"] for e in errors] == ["nim_missing"]
    assert nimcli.preflight(state, offline=True) == ("skipped", [])


@pytest.mark.parametrize(
    ("command", "rule"),
    [
        ("docker stop $(docker ps -q)", "docker-blanket-stop"),
        ("docker rm -f $(docker ps -aq)", "docker-blanket-stop"),
        ("docker kill $(docker ps -q)", "docker-blanket-stop"),
        ("docker ps -q | xargs docker stop", "docker-blanket-stop"),
        ("docker ps -aq | xargs -r docker rm -f", "docker-blanket-stop"),
        ("nim stop --all", "nim-all"),
        ("nim rm --all", "nim-all"),
        ("nim uninstall", "nim-uninstall"),
    ],
)
def test_blanket_stop_forms_say_list_and_ask_not_run_it_yourself(command: str, rule: str) -> None:
    decision, rule_id, reason = guard.evaluate(command, Path.cwd(), nd.utc_now())
    assert (decision, rule_id) == ("deny", rule)
    assert "run them themselves" not in reason and "run it themselves" not in reason
    assert "nim list" in reason and "ask the user which" in reason


@pytest.mark.parametrize("command", ["docker ps", "docker ps -aq", "docker ps -q", "nim list"])
def test_listing_stays_allowed_beside_the_blanket_rules(command: str) -> None:
    assert guard.evaluate(command, Path.cwd(), nd.utc_now())[0] == "allow"


@pytest.mark.parametrize(
    "command",
    ["nim stop nim-llama", "nim stop other/model", "nim rm some-name", "nim rm other/model -f"],
)
def test_stop_or_rm_of_any_name_is_denied_without_an_approved_run(command: str) -> None:
    decision, rule_id, reason = guard.evaluate(command, Path.cwd(), nd.utc_now())
    assert (decision, rule_id) == ("deny", "nim-stop-rm")
    assert "no run has an unexpired approval" in reason


def test_stop_or_rm_is_limited_to_the_approved_run_model(tmp_path: Path) -> None:
    approved_nimcli_run()
    now = nd.utc_now()

    def decide(command: str) -> str:
        return guard.evaluate(command, tmp_path, now)[0]

    assert decide(f"nim stop {MODEL}") == "allow"  # the printed rollback of this run
    for denied in (
        "nim stop nim-llama",
        "nim stop other/model",
        f"nim rm {MODEL}",
        f"nim stop {MODEL} && nim stop other/model",
    ):
        assert decide(denied) == "deny", denied
    for allowed in ("nim list", "nim list --running", "nim info", "nim diag", "docker ps"):
        assert decide(allowed) == "allow", allowed


@pytest.mark.parametrize("command", ["nim list", "nim list --running", "nim info", "nim diag"])
def test_read_only_nim_commands_stay_allowed_without_a_run(command: str) -> None:
    assert guard.evaluate(command, Path.cwd(), nd.utc_now())[0] == "allow"
