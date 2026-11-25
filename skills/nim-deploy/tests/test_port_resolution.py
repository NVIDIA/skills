# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the container port: user value, image metadata, env override, several ports, fallback.

Image metadata comes from fake docker and skopeo on PATH; nothing here touches a registry."""

from __future__ import annotations

import json
import os
import socket
from pathlib import Path
from typing import Any

import nd_common as nd
import pytest
import runners.docker as docker_runner
import verify
from _testkit import assert_contract, docs_cli, fake_tool, make_run, run_cli, with_docs
from _testkit import runs_home as runs_home

pytestmark = pytest.mark.usefixtures("runs_home")

IMAGE = "nvcr.io/nim/meta/llama-3.1-8b-instruct:2.1.4"


def env() -> dict[str, str]:
    return {**os.environ, "NIM_DEPLOY_HOME": str(nd.runs_root())}


def config(ports: list[str], env_vars: list[str] | None = None) -> dict[str, Any]:
    return {
        "ExposedPorts": {port: {} for port in ports},
        "Env": ["PATH=/bin", "NGC_API_KEY=secret-must-not-be-stored", *(env_vars or [])],
    }


class Fakes:
    """Fake docker and skopeo that log every call to `calls`."""

    def __init__(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self.bin, self.calls = tmp_path / "fakes", tmp_path / "calls"
        self.calls.write_text("")
        self.monkeypatch = monkeypatch

    def install(
        self, *, local: Any = None, buildx: Any = None, skopeo: Any = None, port: str = ""
    ) -> None:
        """Each source answers with the given JSON, or fails when it is None."""
        for name, payload in (("local", local), ("buildx", buildx), ("skopeo", skopeo)):
            (self.bin).mkdir(exist_ok=True)
            (self.bin / f"{name}.json").write_text(json.dumps(payload))
        log = f'echo "$0 $@" >> {self.calls}\n'
        docker = log + (
            f'case "$1 $2" in\n'
            f'  "image inspect") {self.answer("local", "config")};;\n'
            f'  "buildx imagetools") {self.answer("buildx", "")};;\n'
            f'  "port "*) printf "%s" "{port}";;\n'
            f"esac\nexit 0"
        )
        fake_tool(self.bin, "docker", docker)
        fake_tool(self.bin, "skopeo", log + self.answer("skopeo", "") + "\nexit 0")
        self.monkeypatch.setenv("PATH", f"{self.bin}{os.pathsep}{os.environ['PATH']}")
        docker_runner.inspect_image.cache_clear()

    def answer(self, name: str, _shape: str) -> str:
        path = self.bin / f"{name}.json"
        denied = "unauthorized: requested access to the resource is denied"
        return f'if [ "$(cat {path})" = null ]; then echo "{denied}" >&2; exit 1; fi; cat {path}'

    def lines(self) -> list[str]:
        return self.calls.read_text().splitlines()


@pytest.fixture
def fakes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Fakes:
    return Fakes(tmp_path, monkeypatch)


def new_run(*pairs: str) -> dict[str, Any]:
    flags = [item for pair in pairs for item in ("--set", pair)]
    code, payload = run_cli("nimdeploy.py", "new", "--target", "docker", *flags, env=env())
    assert code == 0
    facts: dict[str, Any] = payload["facts"]
    return facts


def entry(run: str, key: str) -> dict[str, str]:
    entries: dict[str, dict[str, str]] = nd.load_state(run)["inputs"]
    return entries[key]


def test_a_user_given_container_port_wins(fakes: Fakes) -> None:
    fakes.install(skopeo={"config": config(["8000/tcp"])})
    facts = new_run(f"image={IMAGE}", "container_port=9000")
    assert facts["inputs"]["container_port"] == "9000" and facts["inputs"]["port"] == "9000"
    assert entry(facts["run_id"], "container_port")["source"] == "user"


def test_one_exposed_port_in_the_image_metadata(fakes: Fakes) -> None:
    fakes.install(skopeo={"config": config(["9000/tcp"])})
    facts = new_run(f"image={IMAGE}")
    port = entry(facts["run_id"], "container_port")
    assert (port["value"], port["source"]) == ("9000", "image-metadata")
    assert "9000/tcp" in port["note"] and "skopeo inspect" in port["note"]
    assert facts["inputs"]["port"] == "9000"  # the host port follows the container port
    assert facts["missing"] == []


def test_an_env_override_in_the_image_wins_over_the_exposed_port(fakes: Fakes) -> None:
    fakes.install(skopeo={"config": config(["8000/tcp"], ["NIM_HTTP_API_PORT=9100"])})
    port = entry(new_run(f"image={IMAGE}")["run_id"], "container_port")
    assert (port["value"], port["source"]) == ("9100", "image-metadata")
    assert "NIM_HTTP_API_PORT=9100" in port["note"]


def test_nim_server_port_is_honoured_too(fakes: Fakes) -> None:
    fakes.install(skopeo={"config": config([], ["NIM_SERVER_PORT=8010"])})
    assert entry(new_run(f"image={IMAGE}")["run_id"], "container_port")["value"] == "8010"


def test_several_exposed_ports_are_a_question(fakes: Fakes) -> None:
    fakes.install(skopeo={"config": config(["8000/tcp", "8001/tcp", "53/udp"])})
    facts = new_run(f"image={IMAGE}")
    assert facts["missing"] == ["container_port"] and facts["state"] == "plan"
    assert "container_port" not in facts["inputs"] and "port" not in facts["inputs"]
    assert entry(facts["run_id"], "image_ports")["value"] == "8000,8001"  # udp is ignored
    code, payload = run_cli(
        "nimdeploy.py", "set", facts["run_id"], "container_port=8001", env=env()
    )
    assert (code, payload["state"], payload["facts"]["inputs"]["port"]) == (0, "render", "8001")
    assert entry(facts["run_id"], "container_port")["source"] == "user"


def test_unavailable_metadata_falls_back_to_the_documented_port_and_says_so(fakes: Fakes) -> None:
    fakes.install()  # every source fails
    facts = new_run(f"image={IMAGE}")
    port = entry(facts["run_id"], "container_port")
    assert (port["value"], port["source"]) == ("8000", "documented-default")
    assert "not verified for this image" in port["note"] and "metadata unavailable" in port["note"]
    assert "registry login needed?" in port["note"]  # the skopeo failure said access denied


def test_the_local_image_is_read_first_and_the_registry_is_not_asked(fakes: Fakes) -> None:
    fakes.install(local=config(["7000/tcp"]), skopeo={"config": config(["9999/tcp"])})
    port = entry(new_run(f"image={IMAGE}")["run_id"], "container_port")
    assert (port["value"], "docker image inspect" in port["note"]) == ("7000", True)
    assert not any("skopeo" in call for call in fakes.lines())


def test_the_registry_platform_map_from_buildx_is_read(fakes: Fakes) -> None:
    fakes.install(
        buildx={
            "linux/amd64": {"config": config(["8000/tcp"])},
            "linux/arm64": {"config": config(["8000/tcp"])},
        }
    )
    port = entry(new_run(f"image={IMAGE}")["run_id"], "container_port")
    assert port["value"] == "8000" and "docker buildx imagetools inspect" in port["note"]


def test_lookups_are_read_only_and_store_no_secrets(fakes: Fakes) -> None:
    fakes.install(skopeo={"config": config(["8000/tcp"])})
    facts = new_run(f"image={IMAGE}")
    commands = [line.split(" ", 1)[1] for line in fakes.lines()]
    assert commands == [
        f"image inspect --format {{{{json .Config}}}} {IMAGE}",
        f"buildx imagetools inspect --format {{{{json .Image}}}} {IMAGE}",
        f"inspect --config docker://{IMAGE}",
    ]
    assert not any(" pull" in line or "run" in line.split() for line in fakes.lines())
    state_text = (nd.run_dir(facts["run_id"]) / "state.json").read_text()
    assert "secret-must-not-be-stored" not in state_text


def test_an_unchanged_image_is_not_looked_up_again(fakes: Fakes) -> None:
    fakes.install(skopeo={"config": config(["8000/tcp"])})
    facts = new_run(f"image={IMAGE}")
    looked_up = len(fakes.lines())
    run_cli("nimdeploy.py", "set", facts["run_id"], "name=mine", env=env())
    assert len(fakes.lines()) == looked_up
    run_cli("nimdeploy.py", "set", facts["run_id"], "image=nvcr.io/nim/other/model:1.0", env=env())
    assert len(fakes.lines()) > looked_up


def test_docs_supplied_ports_are_kept_and_labelled(fakes: Fakes) -> None:
    fakes.install(skopeo={"config": config(["8000/tcp"])})
    run = new_run(f"image={IMAGE}")["run_id"]
    code, _ = run_cli(
        "nimdeploy.py", "set", run, "container_port=9300", "--source", "docs", env=env()
    )
    assert code == 0 and entry(run, "container_port")["source"] == "docs"
    assert nd.load_state(run)["inputs"]["port"]["value"] == "9300"


def test_the_summary_shows_the_port_and_where_it_came_from(fakes: Fakes) -> None:
    fakes.install(skopeo={"config": config(["9000/tcp"])})
    run = new_run(f"image={IMAGE}")["run_id"]
    assert run_cli("render_docker.py", "--run", run, env=env())[0] == 0
    docs_cli(run, env())
    _, payload = run_cli("validate.py", "--run", run, "--offline", env=env())
    facts = payload["facts"]
    assert facts["container_port"] == "9000"
    assert facts["port_source"].startswith("image-metadata: the image exposes 9000/tcp")
    assert "-p 9000:9000" in (nd.run_dir(run) / "docker-run.sh").read_text()


def test_the_summary_flags_the_documented_default_as_unverified(fakes: Fakes) -> None:
    fakes.install()
    run = new_run(f"image={IMAGE}")["run_id"]
    assert run_cli("render_docker.py", "--run", run, env=env())[0] == 0
    docs_cli(run, env())
    facts = run_cli("validate.py", "--run", run, "--offline", env=env())[1]["facts"]
    assert facts["port_source"].startswith("documented-default:")
    assert "not verified for this image" in facts["port_source"]


# --- the real mapping after the start --------------------------------------------------------


def docker_run(port: int, container_port: int = 8000, image_ports: str = "8000") -> dict[str, Any]:
    values = {
        "image": IMAGE,
        "name": "llama-3.1-8b-instruct",
        "port": str(port),
        "container_port": str(container_port),
        "image_ports": image_ports,
        "gpus": "all",
    }
    return with_docs(make_run(values, target="docker", state="wait_ready"))


def ready(capsys: pytest.CaptureFixture[str], url: str) -> tuple[int, dict[str, Any]]:
    code = 0
    try:
        verify.main(
            ["ready", "--url", url, "--run", "run1", "--timeout", "0.2", "--interval", "0.05"]
        )
    except SystemExit as exit_info:
        code = int(exit_info.code or 0)
    payload = json.loads(capsys.readouterr().out)
    assert_contract(payload)
    return code, payload


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def test_a_wrong_mapping_shows_up_as_a_clear_port_message(
    fakes: Fakes, capsys: pytest.CaptureFixture[str]
) -> None:
    host = free_port()
    fakes.install(port=f"9000/tcp -> 0.0.0.0:{host}")
    docker_run(host, container_port=8000, image_ports="9000")
    code, payload = ready(capsys, f"http://127.0.0.1:{host}")
    assert (code, payload["state"], payload["next"]) == (4, "diagnose", "diagnose")
    error = payload["errors"][0]
    assert error["code"] == "port_mismatch"
    assert error["message"].startswith(f"image exposes 9000, you mapped 8000->{host}")
    assert "container_port=N" in error["fix"]


def test_a_matching_mapping_goes_on_to_probe_the_endpoint(
    fakes: Fakes, capsys: pytest.CaptureFixture[str]
) -> None:
    host = free_port()
    fakes.install(port=f"8000/tcp -> 0.0.0.0:{host}\n8000/tcp -> [::]:{host}")
    docker_run(host)
    code, payload = ready(
        capsys, f"http://127.0.0.1:{host}"
    )  # nothing listens: not ready, no mismatch
    assert code == 4 and payload["errors"][0]["code"] == "not_ready"
    assert "maps container port 8000 to host port" in payload["errors"][0]["fix"]


def test_parse_mapping() -> None:
    text = "8000/tcp -> 0.0.0.0:8100\n8000/tcp -> [::]:8100\n9000/udp -> 0.0.0.0:1\n"
    assert verify.parse_mapping(text) == {"8000": "8100"}
