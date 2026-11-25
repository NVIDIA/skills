# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for scripts/render_docker.py and the docker runner: golden scripts, secrets by name."""

from __future__ import annotations

import os
import stat
import subprocess
import types
from typing import Any

import nd_common as nd
import pytest
import runners.docker as docker_runner
from _testkit import FAKE_NVAPI, GOLDEN, make_run, run_cli
from _testkit import runs_home as runs_home

pytestmark = pytest.mark.usefixtures("runs_home")

BASE: dict[str, Any] = {
    "run_id": "golden",
    "image": "nvcr.io/nim/meta/llama-3.1-8b-instruct:latest",
    "gpus": "all",
    "port": 8000,
    "container_port": 8000,
    "name": "nim-llama-31-8b",
}
CASES = {
    "all-gpus": {},
    "devices-profile": {
        "gpus": "0,1",
        "port": 8100,
        "model_profile": "vllm-bf16-tp2",
        "cache_dir": "/data/nim cache",
        "shm_size": "32g",
    },
    "model-free": {
        "image": "nvcr.io/nim/nvidia/llm-nim:2.0.0",
        "gpus": 1,
        "pass_hf_token": True,
        "model_path": "hf://Qwen/Qwen3-0.6B",
        "served_model_name": "qwen3",
        "run_as_host_user": False,
        "restart": "no",
    },
}


@pytest.mark.parametrize("case", CASES)
def test_render_matches_golden(case: str) -> None:
    text = docker_runner.render_text({**BASE, **CASES[case]})
    path = GOLDEN / f"docker-{case}.sh"
    if os.environ.get("UPDATE_GOLDEN"):
        path.write_text(text)
    assert text == path.read_text()


def test_script_is_valid_bash() -> None:
    text = docker_runner.render_text({**BASE, "gpus": "0,1", "cache_dir": "/data/nim cache"})
    subprocess.run(["bash", "-n"], input=text, text=True, check=True)


def test_nim_container_facts_are_in_the_script() -> None:
    text = docker_runner.render_text(BASE)
    assert "-e NGC_API_KEY \\" in text  # by name only
    assert "NGC_API_KEY=" not in text.replace(': "${NGC_API_KEY:?', "")
    assert "--shm-size=16g" in text
    assert '-u "$(id -u):0"' in text
    assert "/opt/nim/.cache" in text and "-p 8000:8000" in text


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"image": "bad image"}, "image must be a container image"),
        ({"name": "-x"}, "name must be a docker container name"),
        ({"port": 70000}, "port must be between"),
        ({"port": "eighty"}, "port must be a whole number"),
        ({"gpus": "two"}, "gpus device ids must be numbers"),
        ({"gpus": 0}, "at least 1"),
        ({"restart": "sometimes"}, "restart must be one of"),
        ({"shm_size": "big"}, "shm_size must look like"),
        ({"cache_dir": "relative/dir"}, "cache_dir must be an absolute path"),
        ({"env": {"NGC_API_KEY": "x"}}, "credentials are passed by name only"),
        ({"env": {"OK": FAKE_NVAPI}}, "looks like a credential"),
        ({"env": ["a"]}, "env must be a mapping"),
    ],
)
def test_bad_inputs_are_rejected(change: dict[str, Any], message: str) -> None:
    with pytest.raises(nd.RenderError, match=message):
        docker_runner.render_text({**BASE, **change})


def test_runner_contract_for_the_docker_runner() -> None:
    assert docker_runner.required_inputs() == ("image",)
    assert docker_runner.ready_url({"port": 8100}) == "http://localhost:8100"
    assert docker_runner.container_name({"name": "x"}) == "x"
    state = make_run(
        {**{k: v for k, v in BASE.items() if k != "run_id"}}, target="docker", state="apply"
    )
    folder = nd.run_dir("run1")
    assert docker_runner.apply_command(state) == ["bash", str(folder / "docker-run.sh")]
    assert docker_runner.rollback_command(state) == ["docker", "rm", "-f", "nim-llama-31-8b"]

    approval = types.SimpleNamespace(record={"name": "nim-llama-31-8b"}, folder=nd.run_dir("run1"))
    forms = docker_runner.guard_forms(approval)
    assert (
        docker_runner.apply_command(state) in forms
        and docker_runner.rollback_command(state) in forms
    )
    assert ["docker", "stop", "nim-llama-31-8b"] in forms and len(forms) == 4


def test_cli_writes_an_executable_script_and_advances() -> None:
    make_run({k: v for k, v in BASE.items() if k != "run_id"} | {"model": "m"}, target="docker")
    env = {**os.environ, "NIM_DEPLOY_HOME": str(nd.runs_root())}
    code, payload = run_cli("render_docker.py", "--run", "run1", env=env)
    assert (code, payload["state"], payload["facts"]["container"]) == (
        0,
        "validate",
        "nim-llama-31-8b",
    )
    assert payload["facts"]["url"] == "http://localhost:8000"
    script = nd.runs_root() / "runs" / "run1" / "docker-run.sh"
    assert stat.S_IMODE(script.stat().st_mode) == 0o700
    assert payload["facts"]["sha256"] == {str(script): nd.sha256_file(script)}


def test_cli_bad_input_exits_2() -> None:
    make_run({**{k: v for k, v in BASE.items() if k != "run_id"}, "port": "99999"}, target="docker")
    env = {**os.environ, "NIM_DEPLOY_HOME": str(nd.runs_root())}
    code, payload = run_cli("render_docker.py", "--run", "run1", env=env)
    assert (code, payload["errors"][0]["code"]) == (2, "docker_input")


def test_cli_reports_missing_inputs() -> None:
    make_run({}, target="docker")
    env = {**os.environ, "NIM_DEPLOY_HOME": str(nd.runs_root())}
    code, payload = run_cli("render_docker.py", "--run", "run1", env=env)
    assert (code, payload["facts"]["missing"]) == (3, ["image"])


def test_extra_args_follow_the_image_as_container_arguments() -> None:
    inputs = {
        "image": "nvcr.io/nim/meta/llama-3.1-8b-instruct:2.1.4",
        "name": "nim-llama",
        "port": "8000",
        "container_port": "8000",
        "gpus": "all",
        "extra_args": "--max-num-seqs 64 --kv-cache-dtype 'fp8'",
    }
    text = docker_runner.render_text(inputs)
    assert text.endswith(
        "  nvcr.io/nim/meta/llama-3.1-8b-instruct:2.1.4 \\\n  --max-num-seqs \\\n  64 \\\n"
        "  --kv-cache-dtype \\\n  fp8\n"
    )
    listed = docker_runner.render_text({**inputs, "extra_args": ["--max-num-seqs", "64"]})
    assert listed.endswith("2.1.4 \\\n  --max-num-seqs \\\n  64\n")
    with pytest.raises(nd.RenderError, match="credential"):
        docker_runner.render_text({**inputs, "extra_args": "--api-key x"})
