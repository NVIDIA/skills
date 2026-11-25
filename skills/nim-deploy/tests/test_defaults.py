# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for docker defaults: only `image` is required, name/port/gpus default and are shown."""

from __future__ import annotations

import json
import os
import socket
from pathlib import Path
from typing import Any

import nd_common as nd
import nimdeploy
import pytest
import runners
import runners.docker as docker_runner
import runners.imageinfo as imageinfo
import validate
from _testkit import docs_cli, fake_tool, run_cli, with_docs
from _testkit import runs_home as runs_home

pytestmark = pytest.mark.usefixtures("runs_home")

STAR = "nvcr.io/nim/meta/llama-3.1-8b-instruct:2.1.4"
MODEL_FREE = "nvcr.io/nim/nvidia/model-free-nim:2.1.4"


def env() -> dict[str, str]:
    return {**os.environ, "NIM_DEPLOY_HOME": str(nd.runs_root())}


def new(*pairs: str, target: str = "docker") -> tuple[int, dict[str, Any]]:
    flags = [item for pair in pairs for item in ("--set", pair)]
    return run_cli("nimdeploy.py", "new", "--target", target, *flags, env=env())


def sources(run: str) -> dict[str, str]:
    inputs = nd.load_state(run)["inputs"]
    return {key: entry["source"] for key, entry in inputs.items()}


def to_approve(run: str) -> dict[str, Any]:
    assert run_cli("render_docker.py", "--run", run, env=env())[0] == 0
    docs_cli(run, env())
    code, payload = run_cli("validate.py", "--run", run, "--offline", env=env())
    assert code == 0
    facts: dict[str, Any] = payload["facts"]
    return facts


def test_a_regular_image_needs_only_the_image_and_gets_defaults() -> None:
    code, payload = new(f"image={STAR}")
    assert (code, payload["state"], payload["facts"]["missing"]) == (0, "render", [])
    inputs, run = payload["facts"]["inputs"], payload["facts"]["run_id"]
    assert (inputs["name"], inputs["port"], inputs["gpus"]) == (
        "llama-3.1-8b-instruct",
        "8000",
        "all",
    )
    assert "model" not in inputs
    assert sources(run) == {
        "image": "user",
        "name": "default",
        "port": "default",
        "gpus": "default",
        "container_port": "documented-default",  # the tests hide docker and skopeo: no metadata
        "image_ports": "image-metadata",
    }
    assert payload["facts"]["defaulted"] == ["gpus", "name", "port"]
    assert "--model None" not in payload["facts"]["next_command"]


def test_the_run_id_is_named_after_the_image_when_there_is_no_model() -> None:
    _, payload = new(f"image={STAR}")
    assert payload["facts"]["run_id"].endswith("-docker-llama-3-1-8b-instruct")


def test_user_given_values_win_over_defaults() -> None:
    _, payload = new(f"image={STAR}", "port=8100", "name=mine", "gpus=0,1")
    run = payload["facts"]["run_id"]
    user = {k for k, v in sources(run).items() if v == "user"}
    assert user == {"image", "port", "name", "gpus"}
    assert payload["facts"]["inputs"]["port"] == "8100" and payload["facts"]["defaulted"] == []


def test_set_overrides_a_default_and_a_new_image_refreshes_a_default_name() -> None:
    _, payload = new(f"image={STAR}")
    run = payload["facts"]["run_id"]
    code, payload = run_cli("nimdeploy.py", "set", run, "port=9000", env=env())
    assert (code, sources(run)["port"], sources(run)["name"]) == (0, "user", "default")
    run_cli(
        "nimdeploy.py", "set", run, "image=nvcr.io/nim/meta/llama-3.1-8b-instruct:1.8", env=env()
    )
    assert nd.load_state(run)["inputs"]["name"]["value"] == "llama-3.1-8b-instruct"
    run_cli("nimdeploy.py", "set", run, "name=keep", env=env())
    run_cli("nimdeploy.py", "set", run, f"image={STAR}", env=env())
    assert nd.load_state(run)["inputs"]["name"]["value"] == "keep"


def test_without_an_image_the_run_waits_for_it() -> None:
    code, payload = new()
    assert code == 0 and payload["state"] == "plan" and payload["next"] == "ask_user"
    assert payload["facts"]["missing"] == ["image"]
    assert payload["facts"]["inputs"] == {"gpus": "all"}  # no name or port without an image


@pytest.mark.parametrize(
    ("image", "name"),
    [
        (STAR, "llama-3.1-8b-instruct"),
        ("nvcr.io/nim/meta/llama-3.1-8b-instruct:1.8", "llama-3.1-8b-instruct"),
        ("localhost:5000/team/My_Model@sha256:" + "a" * 64, "my_model"),
        ("registry.example.com:443/x/y/Weird+Name!:tag", "weird-name"),
        ("plain:latest", "plain"),
        ("nvcr.io/nim/-/_:1", "nim"),
    ],
)
def test_default_name_is_the_sanitized_image_leaf(image: str, name: str) -> None:
    assert docker_runner.default_name(image) == name
    assert docker_runner.default_inputs({"image": image})["name"]["value"] == name


def test_a_model_free_image_asks_for_the_model_source() -> None:
    code, payload = new(f"image={MODEL_FREE}")
    assert (code, payload["state"], payload["next"]) == (0, "plan", "ask_user")
    assert payload["facts"]["missing"] == ["model_path"]
    run = payload["facts"]["run_id"]
    code, payload = run_cli(
        "nimdeploy.py", "set", run, "model_path=hf://Qwen/Qwen3-0.6B", env=env()
    )
    assert (payload["state"], payload["facts"]["missing"]) == ("render", [])


def test_the_summary_shows_each_default_and_the_approval_keeps_the_final_values() -> None:
    _, payload = new(f"image={STAR}", "port=8100")
    run = payload["facts"]["run_id"]
    summary = to_approve(run)
    assert summary["defaults"] == {"gpus": "all", "name": "llama-3.1-8b-instruct"}
    assert (summary["name"], summary["port"], summary["gpus"]) == (
        "llama-3.1-8b-instruct",
        "8100",
        "all",
    )
    assert "default" in summary["instruction"]
    code, _ = run_cli("nimdeploy.py", "approve", run, "--token", summary["token"], env=env())
    assert code == 0
    record = nd.load_state(run)["approvals"][-1]
    assert (record["name"], record["port"], record["gpus"]) == (
        "llama-3.1-8b-instruct",
        "8100",
        "all",
    )
    assert record["runner"] == "docker"


def test_validate_uses_the_defaulted_name_for_the_collision_check(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    run = new(f"image={STAR}", "port=" + str(free_port()))[1]["facts"]["run_id"]
    assert run_cli("render_docker.py", "--run", run, env=env())[0] == 0
    log = tmp_path / "calls"
    fake_tool(
        tmp_path / "bin", "docker", f'echo "$@" >> {log}; [ "$1" = info ] && echo nvidia; exit 0'
    )
    monkeypatch.setenv("PATH", f"{tmp_path / 'bin'}{os.pathsep}{os.environ['PATH']}")
    docs_cli(run, env())
    code, _ = run_cli("validate.py", "--run", run, env=env())
    assert code == 0 and "name=^/llama-3.1-8b-instruct$" in log.read_text()


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("0.0.0.0", 0))
        return int(sock.getsockname()[1])


def new_in_process(*pairs: str) -> str:
    flags = [item for pair in pairs for item in ("--set", pair)]
    with pytest.raises(SystemExit):
        nimdeploy.main(["new", "--target", "docker", *flags])
    (path,) = sorted((nd.runs_root() / "runs").glob("*/state.json"))
    return path.parent.name


def test_a_busy_default_port_ends_in_ask_user(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    with socket.socket() as busy:
        busy.bind(("0.0.0.0", 0))
        monkeypatch.setattr(imageinfo, "DOCUMENTED_PORT", str(busy.getsockname()[1]))
        run = new_in_process(f"image={STAR}")
        capsys.readouterr()
        assert nd.load_state(run)["inputs"]["port"]["source"] == "default"
        assert run_cli("render_docker.py", "--run", run, env=env())[0] == 0
        with_docs(nd.load_state(run))
        with pytest.raises(SystemExit) as exit_info:
            validate.main(["--run", run])
    payload = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert exit_info.value.code == nd.EXIT_PRECONDITION and payload["next"] == "ask_user"
    assert "port_in_use" in [error["code"] for error in payload["errors"]]
    assert "port=NNNN" in payload["errors"][0]["fix"]


def test_the_nim_cli_runner_keeps_its_required_inputs_and_has_no_defaults() -> None:
    nimcli = runners.load("nimcli")
    assert nimcli.required_inputs() == ("model", "port")
    code, payload = new("runner=nimcli")
    assert code == 0 and payload["facts"]["missing"] == ["model", "port"]
    assert payload["facts"]["defaulted"] == []
    assert not hasattr(nimcli, "default_inputs")


def test_kubernetes_required_inputs_are_unchanged() -> None:
    assert nd.REQUIRED["helm"] == ("image", "namespace", "context", "name", "ngc_secret", "gpus")
    assert nd.REQUIRED["nim-operator"] == nd.REQUIRED["helm"]
    code, payload = new(f"image={STAR}", target="helm")
    assert code == 0 and payload["facts"]["missing"] == [
        "namespace",
        "context",
        "name",
        "ngc_secret",
        "gpus",
    ]
    assert payload["facts"]["defaulted"] == []
