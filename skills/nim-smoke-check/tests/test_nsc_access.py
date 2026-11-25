# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for scripts/nsc_access.py with fake kubectl and docker: targets, ports, RBAC, privacy."""

from __future__ import annotations

import ast
import contextlib
import json
import os
import queue
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterator
from dataclasses import asdict
from pathlib import Path
from typing import IO, Any

import nsc_access as nsa
import nsc_common as nsc
import nsc_testkit as kit
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
CONTEXT, NAMESPACE = "lab", "nim"
SCOPE = ["--context", CONTEXT, "-n", NAMESPACE]
REPO = "nvcr.io/nim/meta/llama-3.1-8b-instruct"
IMAGE = f"{REPO}:2.1.5"
MODEL = "meta/llama-3.1-8b-instruct"
TOOL_FLAGS = "--enable-auto-tool-choice --tool-call-parser llama3_json"
SERVE = kit.PortForwardReply(serve={"model": MODEL})  # the fake forward runs its own fake NIM
HELM_LABELS = {"app.kubernetes.io/name": nsa.HELM_CHART, "app.kubernetes.io/instance": "my-nim"}
HELM_QUERY = f"app.kubernetes.io/name={nsa.HELM_CHART},app.kubernetes.io/instance=my-nim"
HELM_PODS = f"app.kubernetes.io/instance=my-nim,app.kubernetes.io/name={nsa.HELM_CHART}"

MakeTools = Callable[..., kit.FakeTools]


@pytest.fixture
def fake(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> MakeTools:
    """Write fake kubectl and docker (kubectl fixtures, docker fixtures) and put only them on
    PATH, with no DOCKER_HOST. Each call starts an empty call log."""
    monkeypatch.delenv("DOCKER_HOST", raising=False)

    def make(kubectl: dict[str, Any] | None = None,
             docker: dict[str, Any] | None = None) -> kit.FakeTools:
        tools = kit.make_fake_tools(tmp_path, kubectl, docker)
        tools.log.write_text("", encoding="utf-8")
        monkeypatch.setenv("PATH", str(tools.bin))
        return tools

    return make


def models(url: str) -> str:
    """The first served id, read through `url` without a proxy."""
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(url + "/v1/models", timeout=10) as reply:
        return json.loads(reply.read())["data"][0]["id"]


def gone(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    return False


def forward_pids(tools: kit.FakeTools) -> list[int]:
    return [call["pid"] for call in tools.calls("kubectl") if call["verb"] == "port-forward"]


def write_program(folder: Path, name: str, body: str) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    path.write_text(f"#!{sys.executable} -I\n{body}\n", encoding="utf-8")
    path.chmod(0o755)
    return path


def nim_pod(name: str = "llama-0", **options: Any) -> dict[str, Any]:
    options.setdefault("labels", {"app": "llama"})
    options.setdefault("image", IMAGE)
    return kit.k8s_pod(name, NAMESPACE, **options)


def operator(*pods: dict[str, Any], forward: Any = SERVE, **objects: Any) -> dict[str, Any]:
    """kubectl fixtures of a NIM Operator NIMService llama: Service llama selects app=llama."""
    pods = pods or (nim_pod(),)
    fixtures = {
        "auth can-i create pods": kit.can_i(objects.pop("can", True)),
        "get nimservice llama": objects.pop(
            "nimservice", kit.k8s_nimservice("llama", NAMESPACE, REPO, "2.1.5")
        ),
        "get service llama": objects.pop(
            "service", kit.k8s_service("llama", NAMESPACE, selector={"app": "llama"})
        ),
        "get pods -l app=llama": kit.k8s_list(*pods),
    }
    for pod in pods:
        fixtures[f"port-forward pod/{pod['metadata']['name']}"] = forward
    fixtures.update(objects.pop("extra", {}))
    assert not objects, objects
    return fixtures


@contextlib.contextmanager
def reach(target: str) -> Iterator[tuple[nsa.Access, nsa.PortForward]]:
    access, forward = nsa.from_k8s(target, CONTEXT, NAMESPACE)
    try:
        yield access, forward
    finally:
        forward.stop()


def raises(code: str, exit_code: int, call: Callable[[], Any]) -> nsc.ScriptError:
    with pytest.raises(nsc.ScriptError) as caught:
        call()
    assert (caught.value.code, caught.value.exit_code) == (code, exit_code), caught.value.message
    return caught.value


# --- programs: PATH and the allow-list -----------------------------------------------------------


@pytest.mark.parametrize("entries", [["", "BIN"], [".", "BIN"], ["project", "BIN"], ["./", "BIN"]])
def test_tool_takes_programs_only_from_absolute_path_folders(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, entries: list[str]
) -> None:
    project = tmp_path / "project"
    marker = tmp_path / "project-tool-ran"
    for folder in (project, project / "project"):
        for name in nsa.PROGRAMS:
            write_program(folder, name, f"open({str(marker)!r}, 'w').close()")
    tools = kit.make_fake_tools(tmp_path, docker_fixtures={
        "container inspect llama": kit.docker_container("llama", IMAGE),
    })
    monkeypatch.chdir(project)
    monkeypatch.delenv("DOCKER_HOST", raising=False)
    path = os.pathsep.join(str(tools.bin) if entry == "BIN" else entry for entry in entries)
    monkeypatch.setenv("PATH", path)
    assert nsa.tool("kubectl") == str(tools.bin / "kubectl")
    assert nsa.from_docker("llama").url == "http://127.0.0.1:8000"
    monkeypatch.setenv("PATH", os.pathsep.join(e for e in entries if e != "BIN"))
    raises("tool_missing", 3, lambda: nsa.tool("docker"))
    assert not marker.exists()


def test_a_missing_tool_is_a_precondition_and_only_two_programs_exist(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "empty").mkdir()
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    error = raises("tool_missing", 3, lambda: nsa.tool("kubectl"))
    assert "--url" in error.fix
    raises("tool_missing", 3, lambda: nsa.from_k8s("pod/llama-0", CONTEXT, NAMESPACE))
    with pytest.raises(ValueError):
        nsa.tool("helm")


REFUSED_KUBECTL = [
    ["delete", "pod", "x", *SCOPE],
    ["apply", "-f", "x.yaml", *SCOPE],
    ["exec", "pod/x", *SCOPE, "--", "sh"],
    ["run", "curl", "--image=curlimages/curl", *SCOPE],
    ["debug", "pod/x", *SCOPE],
    ["scale", "deployment/x", "--replicas=0", *SCOPE],
    ["patch", "nimservice", "x", "-p", "{}", *SCOPE],
    ["edit", "nimservice", "x", *SCOPE],
    ["cp", "x:/a", "b", *SCOPE],
    ["attach", "x", *SCOPE],
    ["logs", "x", *SCOPE],
    ["describe", "pod", "x", *SCOPE],
    ["proxy", *SCOPE],
    ["config", "use-context", "prod"],
    ["config", "view"],
    ["config", "get-contexts"],
    ["get", "secret", "ngc-api-secret", "-o", "json", *SCOPE],
    ["get", "secrets,pods", *SCOPE],
    ["get", "configmaps", *SCOPE],
    ["get", "--raw", "/api/v1/namespaces/nim/secrets", *SCOPE],
    ["get", "pods", "-w", *SCOPE],
    ["get", "pods", "-A", *SCOPE],
    ["get", "pods", "-o", "yaml", *SCOPE],
    ["get", "pods", "-l", "app in (x)", *SCOPE],
    ["get", "pods", "-o", "json"],
    ["get", "pods", "--context", CONTEXT],
    ["get", "pods", "--as", "admin", *SCOPE],
    ["get", "pods", "--kubeconfig", "/tmp/other", *SCOPE],
    ["get", "pods", "--context", "-n", NAMESPACE],
    ["get", "pods", "-n", NAMESPACE, "--namespace", "other", "--context", CONTEXT],
    ["auth", "can-i", "*", "*", *SCOPE],
    ["auth", "can-i", "create", "pods", "--subresource=exec", *SCOPE],
    ["auth", "reconcile", "-f", "x.yaml", *SCOPE],
    ["port-forward", "svc/llama", ":8000", *SCOPE],
    ["port-forward", "pod/llama-0", "8000", *SCOPE],
    ["port-forward", "pod/llama-0", ":8000", ":8001", *SCOPE],
    ["port-forward", "--address", "0.0.0.0", "pod/llama-0", ":8000", *SCOPE],
    [],
]
REFUSED_DOCKER = [
    ["run", "-d", IMAGE],
    ["start", "llama"],
    ["stop", "llama"],
    ["restart", "llama"],
    ["rm", "-f", "llama"],
    ["exec", "llama", "sh"],
    ["pull", IMAGE],
    ["inspect", "llama"],
    ["ps"],
    ["logs", "llama"],
    ["update", "--restart=no", "llama"],
    ["context", "use", "remote"],
    ["container", "rm", "llama"],
    ["image", "rm", IMAGE],
    ["-H", "tcp://10.0.0.5:2375", "container", "inspect", "llama"],
    ["--context", "remote", "container", "inspect", "llama"],
    ["container", "inspect", "llama", "other"],
    ["container", "inspect", "-x"],
    ["port", "llama", "8000/tcp", "x"],
]


@pytest.mark.parametrize(
    ("program", "words"),
    [("kubectl", words) for words in REFUSED_KUBECTL]
    + [("docker", words) for words in REFUSED_DOCKER],
)
def test_run_tool_refuses_anything_off_the_list_before_it_runs(
    fake: MakeTools, program: str, words: list[str]
) -> None:
    tools = fake({"*": "anything"}, {"*": "anything"})
    error = raises(
        "refused_command", 5, lambda: nsa.run_tool([nsa.tool(program), *words], timeout=10)
    )
    assert "read-only" in error.message
    assert tools.calls() == []  # the fake, which exits 9 on a mutation, never even started


@pytest.mark.parametrize(
    "argv",
    [
        ["kubectl", "get", "pods", *SCOPE],
        ["/usr/bin/helm", "status", "x"],
        ["sh", "-c", "kubectl get pods"],
        "kubectl get pods",
        ["KUBECTL", "get", "pods", *SCOPE],
    ],
)
def test_run_tool_refuses_programs_that_are_not_absolute_kubectl_or_docker(
    fake: MakeTools, argv: Any
) -> None:
    tools = fake({"*": "anything"})
    raises("refused_command", 5, lambda: nsa.run_tool(argv, timeout=10))
    assert tools.calls() == []


def test_run_tool_refuses_a_kubectl_that_is_not_the_one_on_path(
    fake: MakeTools, tmp_path: Path
) -> None:
    fake({"*": "anything"})
    marker = tmp_path / "other-ran"
    other = write_program(tmp_path / "elsewhere", "kubectl", f"open({str(marker)!r}, 'w').close()")
    raises("refused_command", 5, lambda: nsa.run_tool([str(other), "get", "pods", *SCOPE],
                                                       timeout=10))
    assert not marker.exists()


def test_run_tool_refuses_control_characters_and_empty_words(fake: MakeTools) -> None:
    tools = fake({"*": "anything"})
    kubectl = nsa.tool("kubectl")
    for argv in ([kubectl, "get", "pods", "--context", "lab\nx", "-n", NAMESPACE],
                 [kubectl, "get", "pods", "--context", "", "-n", NAMESPACE],
                 [kubectl, "get", "", *SCOPE]):
        raises("refused_command", 5, lambda argv=argv: nsa.run_tool(argv, timeout=10))
    assert tools.calls() == []


def test_run_tool_runs_allowed_commands_literally_with_no_stdin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record = tmp_path / "record.json"
    marker = tmp_path / "shell-ran"
    body = (
        "import json, os, sys\n"
        "st, null = os.fstat(0), os.stat(os.devnull)\n"
        f"json.dump({{'argv': sys.argv[1:], 'devnull': (st.st_ino, st.st_dev) == "
        f"(null.st_ino, null.st_dev)}}, open({str(record)!r}, 'w'))\n"
        "print('{}')"
    )
    bin_dir = tmp_path / "bin"
    write_program(bin_dir, "kubectl", body)
    monkeypatch.setenv("PATH", str(bin_dir))
    context = f"lab;touch${{IFS}}{marker}"
    done = nsa.run_tool(
        [nsa.tool("kubectl"), "get", "pods", "-l", "app=llama", "-o", "json",
         "--context", context, "-n", NAMESPACE],
        timeout=30,
    )
    assert done.returncode == 0 and done.stdout.strip() == "{}"
    seen = json.loads(record.read_text())
    assert seen["argv"][-3] == context and seen["devnull"] is True
    assert not marker.exists()


def test_run_tool_reports_a_timeout_as_exit_124(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_program(tmp_path / "bin", "docker", "import time\ntime.sleep(20)")
    monkeypatch.setenv("PATH", str(tmp_path / "bin"))
    started = time.monotonic()
    done = nsa.run_tool([nsa.tool("docker"), "port", "llama"], timeout=0.5)
    assert done.returncode == nsa.TIMED_OUT and "did not finish within 0.5 s" in done.stderr
    assert time.monotonic() - started < 10


def test_run_tool_background_writes_only_to_files(fake: MakeTools, tmp_path: Path) -> None:
    fake({"*": "anything"})
    argv = [nsa.tool("kubectl"), "port-forward", "pod/llama-0", ":8000", *SCOPE]
    with open(tmp_path / "out", "wb") as out:
        for stdout, stderr in ((None, out), (out, None), (subprocess.PIPE, out)):
            with pytest.raises(ValueError):
                nsa.run_tool(argv, timeout=5, background=True, stdout=stdout, stderr=stderr)


def test_why_refused_accepts_exactly_the_commands_the_module_runs() -> None:
    kubectl, docker = "/opt/bin/kubectl", "/opt/bin/docker"
    for words in (
        ["auth", "can-i", "create", "pods", "--subresource=portforward", *SCOPE],
        ["get", "nimservice", "llama", "-o", "json", *SCOPE],
        ["get", "service", "llama", "-o", "json", *SCOPE],
        ["get", "services", "-l", HELM_QUERY, "-o", "json", *SCOPE],
        ["get", "pods", "-l", "app=llama,nim-llm-role=leader", "-o", "json", *SCOPE],
        ["get", "pod", "llama-0", "-o", "json", *SCOPE],
        ["port-forward", "pod/llama-0", ":8000", *SCOPE],
        ["port-forward", "pod/llama-0", "41234:8000", *SCOPE],
        ["config", "get-contexts", "-o", "name"],
        ["--namespace", NAMESPACE, "--context=lab", "get", "pods", "--output=json"],
    ):
        assert nsa.why_refused([kubectl, *words]) is None, words
    for words in (
        ["container", "inspect", "llama"],
        ["image", "inspect", "--format", "{{json .Config.Env}}", IMAGE],
        ["port", "llama"],
        ["port", "llama", "8000/tcp"],
        ["context", "inspect", "--format", "{{.Endpoints.docker.Host}}"],
    ):
        assert nsa.why_refused([docker, *words]) is None, words


# --- from_url -------------------------------------------------------------------------------------


def test_from_url_marks_the_hosted_catalog_and_keeps_other_urls_direct() -> None:
    hosted = nsa.from_url("https://integrate.api.nvidia.com/v1/")
    assert (hosted.place, hosted.url, hosted.how) == (
        "hosted", "https://integrate.api.nvidia.com", "direct")
    assert hosted.health_url == hosted.url and "/v1/health" in hosted.notes[0]
    assert hosted.config_known is False and hosted.facts()["passthrough"] == "unknown"
    plain = nsa.from_url("http://localhost:8000/v1", "http://localhost:8001")
    assert (plain.place, plain.url, plain.health_url) == (
        "url", "http://localhost:8000", "http://localhost:8001")
    assert plain.facts()["args"] is None and plain.port_forward is None
    password = kit.fake_token()  # built at run time: no credential literal in the repo
    error = raises("bad_input", 2, lambda: nsa.from_url(
        f"http://user:{password}@gateway.example.com"))
    assert password not in error.message + error.fix
    raises("bad_input", 2, lambda: nsa.from_url("ftp://gateway.example.com"))


# --- Docker ---------------------------------------------------------------------------------------


def container(**options: Any) -> list[dict[str, Any]]:
    return kit.docker_container("llama", IMAGE, **options)


def docker_fixtures(record: Any, **extra: Any) -> dict[str, Any]:
    return {"container inspect llama": record, "context inspect": kit.docker_context(), **extra}


def test_from_docker_uses_127_0_0_1_for_a_port_published_on_all_addresses(
    fake: MakeTools,
) -> None:
    record = container(host_port=8005, env={"NIM_PASSTHROUGH_ARGS": "--max-num-seqs 8"})
    record[0]["Config"]["Cmd"] = ["--max-model-len", "8192"]
    tools = fake(docker=docker_fixtures(record))
    access = nsa.from_docker("llama")
    assert (access.place, access.url, access.health_url, access.how) == (
        "docker", "http://127.0.0.1:8005", "http://127.0.0.1:8005", "published-port")
    assert access.image == IMAGE and access.image_id == record[0]["Image"]
    assert access.effective_passthrough == "--max-num-seqs 8"
    assert access.effective_args == ["--max-model-len", "8192"]
    assert access.config_known and access.notes == [] and access.port_forward is None
    assert access.facts()["passthrough"] == "set" and access.facts()["args"] == 2
    assert [call["argv"][:2] for call in tools.calls("docker")] == [
        ["context", "inspect"], ["container", "inspect"]]
    tools.assert_read_only()


@pytest.mark.parametrize(
    ("bindings", "url", "note"),
    [
        ([{"HostIp": "::", "HostPort": "8006"}], "http://[::1]:8006", None),
        ([{"HostIp": "127.0.0.1", "HostPort": "8007"}], "http://127.0.0.1:8007", "Docker host"),
        ([{"HostIp": "::1", "HostPort": "8007"}], "http://[::1]:8007", "Docker host"),
        ([{"HostIp": "192.168.1.100", "HostPort": "8008"}], "http://192.168.1.100:8008",
         "localhost does not answer"),
        ([{"HostIp": "", "HostPort": "8009"}], "http://127.0.0.1:8009", None),
    ],
)
def test_from_docker_picks_the_address_the_port_is_published_on(
    fake: MakeTools, bindings: list[dict[str, str]], url: str, note: str | None
) -> None:
    record = container()
    record[0]["NetworkSettings"]["Ports"]["8000/tcp"] = bindings
    fake(docker=docker_fixtures(record))
    access = nsa.from_docker("llama")
    assert access.url == url
    if note is None:
        assert access.notes == []
    else:
        assert len(access.notes) == 1 and note in access.notes[0]


def test_from_docker_with_host_networking_uses_the_server_port(fake: MakeTools) -> None:
    record = container(network="host", env={"NIM_SERVER_PORT": "9000",
                                            "NIM_HEALTH_PORT": "9001"})
    fake(docker=docker_fixtures(record))
    access = nsa.from_docker("llama")
    assert (access.url, access.health_url, access.how) == (
        "http://127.0.0.1:9000", "http://127.0.0.1:9001", "direct")
    assert "Host networking" in access.notes[0]


def test_from_docker_follows_the_health_port_and_tls(fake: MakeTools) -> None:
    record = container(host_port=8010, env={"NIM_HEALTH_PORT": "8001", "NIM_SSL_MODE": "mtls"})
    record[0]["NetworkSettings"]["Ports"]["8001/tcp"] = [{"HostIp": "0.0.0.0", "HostPort": "9011"}]
    fake(docker=docker_fixtures(record))
    access = nsa.from_docker("llama")
    assert (access.url, access.health_url) == ("https://127.0.0.1:8010", "https://127.0.0.1:9011")
    assert "--ca-file" in access.notes[0] and "--client-cert" in access.notes[0]
    del record[0]["NetworkSettings"]["Ports"]["8001/tcp"]
    fake(docker=docker_fixtures(record))
    hidden = nsa.from_docker("llama")
    assert hidden.health_url is None
    assert any("NIM_HEALTH_PORT=8001" in note for note in hidden.notes)


def test_from_docker_reads_docker_port_when_the_record_has_no_port_table(
    fake: MakeTools,
) -> None:
    record = container()
    del record[0]["NetworkSettings"]
    tools = fake(docker=docker_fixtures(record, **{"port llama": kit.docker_port_text(
        host_port=8012)}))
    assert nsa.from_docker("llama").url == "http://127.0.0.1:8012"
    assert ["port", "llama"] in tools.argvs("docker")


def test_from_docker_a_failed_docker_port_is_docker_failed_not_unpublished(
    fake: MakeTools,
) -> None:
    """With no port table in the record, `docker port NAME` is read. docker failing there (a
    daemon that stopped answering, a timeout) is docker_failed (exit 4), never
    port_not_published, whose fix changes the deployment; an answer that lists no port is
    port_not_published."""
    record = container()
    del record[0]["NetworkSettings"]
    down = kit.ToolReply(stderr="Cannot connect to the Docker daemon at unix:///var/run/docker"
                         ".sock. Is the docker daemon running?\n", exit=1)
    fake(docker=docker_fixtures(record, **{"port llama": down}))
    error = raises("docker_failed", 4, lambda: nsa.from_docker("llama"))
    assert "published ports" in error.message and "Start the Docker daemon" in error.fix
    fake(docker=docker_fixtures(record, **{"port llama": ""}))
    raises("port_not_published", 3, lambda: nsa.from_docker("llama"))


def test_parse_port_lines_and_pick_binding() -> None:
    text = (
        "8000/tcp -> 0.0.0.0:8005\n8000/tcp -> [::]:8005\n8001/tcp -> 127.0.0.1:9001\n"
        "9000/udp -> 0.0.0.0:9000\nnot a port line\n8002/tcp -> [::1]:9002\n"
    )
    assert nsa.parse_port_lines(text) == {
        8000: [("0.0.0.0", 8005), ("::", 8005)], 8001: [("127.0.0.1", 9001)],
        8002: [("::1", 9002)],
    }
    assert nsa.pick_binding([("::", 8005), ("0.0.0.0", 8005)]) == ("127.0.0.1", 8005, None)
    assert nsa.pick_binding([("::", 8005)]) == ("[::1]", 8005, None)
    host, port, note = nsa.pick_binding([("10.0.0.7", 9000), ("127.0.0.1", 9000)])
    assert (host, port) == ("127.0.0.1", 9000) and "Docker host alone" in (note or "")
    assert nsa.pick_binding([("fd00::5", 9000)])[0] == "[fd00::5]"
    for bad in ([], [("not-an-ip", 1)]):
        with pytest.raises(ValueError):
            nsa.pick_binding(bad)


def test_from_docker_never_keeps_any_other_environment_value(
    fake: MakeTools, capsys: pytest.CaptureFixture[str]
) -> None:
    secrets = {"NGC_API_KEY": kit.fake_key(), "HF_TOKEN": kit.fake_token(),
               "DB_PASSWORD": "pw-" + "7" * 12, "PLAIN_SETTING": "value-" + "x" * 10}
    record = container(env={**secrets, "NIM_PASSTHROUGH_ARGS": TOOL_FLAGS,
                            "NIM_SERVER_PORT": "8000"})
    fake(docker=docker_fixtures(record))
    access = nsa.from_docker("llama")
    assert access.effective_passthrough == TOOL_FLAGS
    shown = json.dumps(asdict(access)) + json.dumps(access.facts())
    captured = capsys.readouterr()
    for value in secrets.values():
        assert value not in shown and value not in captured.out + captured.err
    for name in secrets:
        assert name not in shown


def test_from_docker_refuses_a_remote_daemon(
    fake: MakeTools, monkeypatch: pytest.MonkeyPatch
) -> None:
    tools = fake(docker=docker_fixtures(container()))
    monkeypatch.setenv("DOCKER_HOST", "tcp://10.0.0.5:2376")
    error = raises("docker_remote", 3, lambda: nsa.from_docker("llama"))
    assert "--url" in error.fix and "10.0.0.5" in error.fix
    assert tools.calls() == []
    monkeypatch.setenv("DOCKER_HOST", "unix:///var/run/docker.sock")
    tools = fake(docker={"context inspect": kit.docker_context("ssh://admin@gpu-host"),
                         "container inspect llama": container()})
    error = raises("docker_remote", 3, lambda: nsa.from_docker("llama"))
    assert "ssh://gpu-host" in error.message and "admin@" not in error.message + error.fix
    assert [call["argv"][:2] for call in tools.calls()] == [["context", "inspect"]]


def test_from_docker_errors(fake: MakeTools) -> None:
    missing = kit.ToolReply(stderr="Error: No such container: llama\n", exit=1)
    fake(docker=docker_fixtures(missing))
    raises("target_not_found", 3, lambda: nsa.from_docker("llama"))
    fake(docker=docker_fixtures(container(running=False)))
    assert "exited" in raises("container_not_running", 3, lambda: nsa.from_docker("llama")).message
    record = container()
    record[0]["NetworkSettings"]["Ports"]["8000/tcp"] = None
    fake(docker=docker_fixtures(record))
    raises("port_not_published", 3, lambda: nsa.from_docker("llama"))
    down = kit.ToolReply(stderr="Cannot connect to the Docker daemon at unix:///var/run/docker"
                         ".sock. Is the docker daemon running?\n", exit=1)
    fake(docker=docker_fixtures(down))
    assert "Start the Docker daemon" in raises(
        "docker_failed", 4, lambda: nsa.from_docker("llama")).fix
    leaky = "[{" + json.dumps("NGC_API_KEY=" + kit.fake_key())  # not JSON; never echoed
    fake(docker=docker_fixtures(leaky))
    error = raises("docker_failed", 4, lambda: nsa.from_docker("llama"))
    assert kit.fake_key() not in error.message + error.fix


def test_from_docker_a_missing_docker_socket_is_a_daemon_error_not_a_missing_container(
    fake: MakeTools,
) -> None:
    """A missing socket says "connect: no such file or directory": not "No such container"."""
    socket_gone = kit.ToolReply(
        stderr="failed to connect to the docker API at unix:///var/run/docker.sock; check if "
               "the path is correct and if the daemon is running: dial unix "
               "/var/run/docker.sock: connect: no such file or directory\n",
        exit=1,
    )
    fake(docker=docker_fixtures(socket_gone))
    error = raises("docker_failed", 4, lambda: nsa.from_docker("llama"))
    assert "no such file or directory" in error.message and "Start the Docker daemon" in error.fix
    other = kit.ToolReply(stderr="Error response from daemon: No such object: llama\n", exit=1)
    fake(docker=docker_fixtures(other))
    raises("docker_failed", 4, lambda: nsa.from_docker("llama"))


def test_from_docker_a_restarting_container_is_not_running_not_unpublished(
    fake: MakeTools,
) -> None:
    """access-3: while its restart policy waits to start an exited NIM again, Docker keeps
    Running true and empties the published ports (HostConfig keeps the -p mapping). That is a
    NIM that keeps exiting, for nim-deploy to diagnose, never "publish the port"."""
    record = container()
    record[0]["State"] = {"Status": "restarting", "Running": True, "Restarting": True,
                          "Paused": False, "ExitCode": 1}
    record[0]["RestartCount"] = 4
    for ports in ({}, None):
        record[0]["NetworkSettings"]["Ports"] = ports
        fake(docker=docker_fixtures(record))
        error = raises("container_not_running", 3, lambda: nsa.from_docker("llama"))
        assert "restarting" in error.message and "4 restarts" in error.message
        assert "nim-deploy" in error.fix and "-p" not in error.fix and "publish" not in error.fix


def test_from_docker_a_paused_or_dead_container_is_not_running(fake: MakeTools) -> None:
    """access-3: a paused container answers no request (every check would wait out its
    timeout); only State.Status "running" is reachable."""
    paused = container()
    paused[0]["State"] = {"Status": "paused", "Running": True, "Paused": True,
                          "Restarting": False}
    fake(docker=docker_fixtures(paused))
    error = raises("container_not_running", 3, lambda: nsa.from_docker("llama"))
    assert "paused" in error.message and "unpause" in error.fix
    for state in ({"Status": "dead"}, {"Status": "created", "Running": False},
                  {"Paused": True}, {"Restarting": True}):
        record = container()
        record[0]["State"] = state
        fake(docker=docker_fixtures(record))
        raises("container_not_running", 3, lambda: nsa.from_docker("llama"))
    record = container()
    record[0]["State"] = {"Running": True}  # an inspect record with no Status
    fake(docker=docker_fixtures(record))
    assert nsa.from_docker("llama").url == "http://127.0.0.1:8000"


def test_running_image_names_the_image_by_id_or_digest_never_by_tag() -> None:
    """access-1: tier B reads the default of the image the container or pod runs. A tag such
    as :latest can name another image after a pull, so only an ID or a digest is used."""
    digest = "sha256:" + "c" * 64
    url = "http://127.0.0.1:8000"
    docker = nsa.Access("docker", url, url, "published-port", image=IMAGE, image_id=digest)
    assert nsa.running_image(docker) == digest
    pod = nsa.Access("kubernetes", url, url, "port-forward", image=IMAGE, image_id=None)
    for image_id, expected in (
        (f"{REPO}@{digest}", f"{REPO}@{digest}"),
        (f"docker-pullable://{REPO}@{digest}", f"{REPO}@{digest}"),
        (digest, digest),
        (IMAGE, None),
        (None, None),
        ("sha256:abc", None),
        (f"{REPO}@sha256:" + "C" * 64, None),
    ):
        pod.image_id = image_id
        assert nsa.running_image(pod) == expected, image_id
    for image_id in (None, IMAGE, f"{REPO}@{digest}"):
        docker.image_id = image_id
        assert nsa.running_image(docker) == (image_id if image_id == f"{REPO}@{digest}" else None)
    assert nsa.running_image(nsa.from_url(url)) is None
    assert all(nsa.IMAGE_REF.fullmatch(ref) for ref in (digest, f"{REPO}@{digest}"))


@pytest.mark.parametrize("name", ["-rm", "a b", "", "x;rm", "../x"])
def test_from_docker_refuses_a_bad_name_without_running_docker(
    fake: MakeTools, name: str
) -> None:
    tools = fake(docker={"*": "anything"})
    raises("bad_input", 2, lambda: nsa.from_docker(name))
    assert tools.calls() == []


def test_image_default_passthrough(fake: MakeTools, tmp_path: Path,
                                   monkeypatch: pytest.MonkeyPatch) -> None:
    tools = fake(docker={"image inspect": kit.docker_image(
        IMAGE, env={"PATH": "/usr/bin", "NIM_PASSTHROUGH_ARGS": TOOL_FLAGS,
                    "NGC_API_KEY": kit.fake_key()})})
    assert nsa.image_default_passthrough(IMAGE) == TOOL_FLAGS
    assert tools.argvs("docker")[-1] == ["image", "inspect", "--format",
                                          "{{json .Config.Env}}", IMAGE]
    fake(docker={"image inspect": kit.docker_image(IMAGE, env={"PATH": "/usr/bin"})})
    assert nsa.image_default_passthrough(IMAGE) == ""
    image = kit.docker_image(IMAGE)
    image[0]["Config"]["Env"] = None
    fake(docker={"image inspect": image})
    assert nsa.image_default_passthrough(IMAGE) == ""
    fake(docker={"image inspect": kit.ToolReply(stderr="Error: No such image\n", exit=1)})
    assert nsa.image_default_passthrough(IMAGE) is None
    tools = fake(docker={"*": "anything"})
    assert nsa.image_default_passthrough("-x") is None and tools.calls() == []
    assert nsa.image_default_passthrough(None) is None
    (tmp_path / "empty").mkdir()
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    assert nsa.image_default_passthrough(IMAGE) is None


def test_image_default_passthrough_is_unknown_when_docker_prints_nothing(
    fake: MakeTools,
) -> None:
    """docker exiting 0 with no output says nothing about the image: unknown (None), never ""
    (the image sets none). A JSON null, an image with no environment, is still ""."""
    for blank in ("", "\n  \n"):
        fake(docker={"image inspect": kit.ToolReply(stdout=blank)})
        assert nsa.image_default_passthrough(IMAGE) is None, repr(blank)
    fake(docker={"image inspect": kit.ToolReply(stdout="null\n")})
    assert nsa.image_default_passthrough(IMAGE) == ""


# --- Kubernetes -----------------------------------------------------------------------------------


def test_from_k8s_nimservice_forwards_to_a_ready_pod(fake: MakeTools) -> None:
    pod = nim_pod(args=["--max-num-seqs", "8"],
                  env={"NIM_PASSTHROUGH_ARGS": "--max-num-seqs 8"})
    with kit.FakeNim() as nim:
        tools = fake(operator(pod, forward=kit.port_forward_to(nim)))
        with reach("nimservice/llama") as (access, forward):
            assert (access.place, access.how) == ("kubernetes", "port-forward")
            assert access.url == forward.url == f"http://127.0.0.1:{forward.local_port}"
            assert access.health_url == access.url
            assert models(access.url) == MODEL
            assert access.image == IMAGE and access.config_known
            assert access.effective_args == ["--max-num-seqs", "8"]
            assert access.effective_passthrough == "--max-num-seqs 8"
            assert access.port_forward == forward.facts()
            assert access.port_forward["command"] == (
                "kubectl port-forward pod/llama-0 :8000 --context lab -n nim")
            assert access.port_forward["pod"] == "llama-0" and forward.restarts == 0
            assert forward.alive()
        assert not forward.alive()
    words = [call["argv"][:3] for call in tools.calls("kubectl")]
    assert words == [["auth", "can-i", "create"], ["get", "nimservice", "llama"],
                     ["get", "service", "llama"], ["get", "pods", "-l"],
                     ["port-forward", "pod/llama-0", ":8000"]]
    tools.assert_scoped(CONTEXT, NAMESPACE)
    tools.assert_read_only()
    assert all(gone(pid) for pid in forward_pids(tools))


def test_from_k8s_maps_a_named_service_port_to_the_container_port(fake: MakeTools) -> None:
    pod = nim_pod(port=8123)
    tools = fake(operator(pod))
    with reach("nimservice/llama") as (access, forward):
        assert forward.remote_port == 8123 and access.notes == []
    assert tools.argvs("kubectl")[-1][:3] == ["port-forward", "pod/llama-0", ":8123"]


def test_from_k8s_reads_the_container_that_declares_a_numeric_target_port(
    fake: MakeTools,
) -> None:
    """A Service whose targetPort is a number: the one container that declares that TCP port
    serves the NIM, also behind a sidecar listed first, so its TLS mode, image and args are
    read, not the sidecar's. The lookup by port name stays as it was."""
    pod = nim_pod(port=8443, port_name="https", env={"NIM_SSL_MODE": "TLS"},
                  args=["--max-num-seqs", "8"])
    sidecar = {"name": "proxy", "image": "registry.example.com/proxy:1.0",
               "ports": [{"name": "metrics", "containerPort": 9090, "protocol": "TCP"}]}
    pod["spec"]["containers"].insert(0, sidecar)
    service = kit.k8s_service("llama", NAMESPACE, selector={"app": "llama"})
    service["spec"]["ports"] = [{"name": "https", "port": 443, "targetPort": 8443,
                                 "protocol": "TCP"}]
    fake(operator(pod, service=service))
    with reach("nimservice/llama") as (access, forward):
        assert forward.remote_port == 8443
        assert access.url == f"https://127.0.0.1:{forward.local_port}"
        assert access.image == IMAGE and access.effective_args == ["--max-num-seqs", "8"]
    nim = pod["spec"]["containers"][1]
    assert nsa._nim_container(pod, 8443) is nim and nsa._nim_container(pod, "https") is nim
    assert nsa._nim_container(pod, 9090) is sidecar
    nim["ports"].append({"name": "admin", "containerPort": 9090, "protocol": "TCP"})
    assert nsa._nim_container(pod, 9090) is sidecar  # two declare it: the first, as before


@pytest.mark.parametrize(
    ("value", "port"),
    [("8000", 8000), (" 8443 ", 8443), (8080, 8080), ("\u00b2", None), ("\u0663", None),
     ("65536", None), ("0", None), ("123456", None), (True, None), ("", None)],
)
def test_port_number_takes_ascii_digits_only(value: Any, port: int | None) -> None:
    """str.isdigit() is True for "\u00b2" and "\u0663", which int() refuses: None, no crash."""
    assert nsa._port_number(value) == port


def test_a_service_target_port_of_unicode_digits_is_a_name_not_a_crash() -> None:
    service = kit.k8s_service("llama", NAMESPACE, selector={"app": "llama"})
    service["spec"]["ports"][0]["targetPort"] = "\u00b2"
    assert nsa._service_port(service, ("api",))[0] == "\u00b2"
    service["spec"]["ports"][0]["targetPort"] = "8123"
    assert nsa._service_port(service, ("api",))[0] == 8123


def test_service_port_takes_tcp_ports_only() -> None:
    """kubectl port-forward carries TCP only: a UDP or SCTP Service port is never the API port,
    by name or as the first port; a port with no protocol is TCP."""
    service = kit.k8s_service("custom", NAMESPACE, selector={"app": "custom"})
    service["spec"]["ports"] = [
        {"name": "statsd", "port": 9125, "targetPort": 9125, "protocol": "UDP"},
        {"name": "inference", "port": 8000, "targetPort": 8000},
    ]
    port, note = nsa._service_port(service, (nsa.OPERATOR_PORT, nsa.HELM_PORT))
    assert port == 8000 and "its first TCP port, inference" in (note or "")
    service["spec"]["ports"] = [
        {"name": nsa.OPERATOR_PORT, "port": 8000, "targetPort": 8000, "protocol": "SCTP"},
        {"name": "web", "port": 80, "targetPort": 8080, "protocol": "TCP"},
    ]
    assert nsa._service_port(service, (nsa.OPERATOR_PORT,))[0] == 8080
    service["spec"]["ports"] = [{"name": "statsd", "port": 9125, "protocol": "UDP"}]
    error = raises("target_not_found", 3, lambda: nsa._service_port(service, ("api",)))
    assert "no TCP port" in error.message


def test_from_k8s_helm_finds_the_service_by_labels_and_skips_the_headless_one(
    fake: MakeTools,
) -> None:
    headless = kit.k8s_service("my-nim-nim-llm-sts", NAMESPACE, selector={"app": "wrong"},
                               port_name=nsa.HELM_PORT, labels=HELM_LABELS, cluster_ip="None")
    service = kit.k8s_service("my-nim-nim-llm", NAMESPACE, selector=HELM_LABELS,
                              port_name=nsa.HELM_PORT, labels=HELM_LABELS)
    pod = nim_pod("my-nim-nim-llm-0", labels=HELM_LABELS, port_name=nsa.HELM_PORT, port=8080)
    tools = fake({
        "auth can-i create pods": kit.can_i(True),
        f"get services -l {HELM_QUERY}": kit.k8s_list(headless, service),
        f"get pods -l {HELM_PODS}": kit.k8s_list(pod),
        "port-forward pod/my-nim-nim-llm-0": SERVE,
    })
    with reach("helm/my-nim") as (access, forward):
        assert forward.pod == "my-nim-nim-llm-0" and forward.remote_port == 8080
        assert models(access.url) == MODEL
    calls = tools.argvs("kubectl")
    assert calls[1][:4] == ["get", "services", "-l", HELM_QUERY]
    assert calls[-1][:3] == ["port-forward", "pod/my-nim-nim-llm-0", ":8080"]
    tools.assert_scoped(CONTEXT, NAMESPACE)
    tools.assert_read_only()


def test_from_k8s_helm_with_only_a_headless_service_is_not_found(fake: MakeTools) -> None:
    headless = kit.k8s_service("my-nim-nim-llm-sts", NAMESPACE, selector=HELM_LABELS,
                               port_name=nsa.HELM_PORT, cluster_ip="None")
    fake({"auth can-i create pods": kit.can_i(True),
          f"get services -l {HELM_QUERY}": kit.k8s_list(headless)})
    error = raises("target_not_found", 3, lambda: nsa.from_k8s("helm/my-nim", CONTEXT, NAMESPACE))
    assert "headless" in error.message and "app.kubernetes.io/instance=my-nim" in error.fix


def test_from_k8s_svc_prefers_api_or_http_openai_else_the_first_port(fake: MakeTools) -> None:
    service = kit.k8s_service("custom", NAMESPACE, selector={"app": "custom"})
    service["spec"]["ports"] = [
        {"name": "metrics", "port": 9400, "targetPort": 9400},
        {"name": nsa.HELM_PORT, "port": 80, "targetPort": nsa.HELM_PORT},
    ]
    pod = nim_pod("custom-0", labels={"app": "custom"}, port_name=nsa.HELM_PORT, port=8000)
    fixtures = {"auth can-i create pods": kit.can_i(True), "get service custom": service,
                "get pods -l app=custom": kit.k8s_list(pod), "port-forward pod/custom-0": SERVE}
    fake(fixtures)
    with reach("svc/custom") as (access, forward):
        assert forward.remote_port == 8000 and access.notes == []
    service["spec"]["ports"] = [{"port": 9000, "targetPort": 9000}]
    fake(fixtures)
    with reach("svc/custom") as (access, forward):
        assert forward.remote_port == 9000
        assert "no port named api or http-openai" in access.notes[0]


def test_from_k8s_pod_target_and_its_readiness(fake: MakeTools) -> None:
    fixtures = {"auth can-i create pods": kit.can_i(True),
                "get pod llama-0": nim_pod(), "port-forward pod/llama-0": SERVE}
    tools = fake(fixtures)
    with reach("pod/llama-0") as (access, forward):
        assert forward.remote_port == 8000 and models(access.url) == MODEL
    fixtures["get pod llama-0"] = nim_pod(ready=False)
    tools = fake(fixtures)
    error = raises("pod_not_ready", 3, lambda: nsa.from_k8s("pod/llama-0", CONTEXT, NAMESPACE))
    assert "Running, not Ready" in error.message and "nim-deploy" in error.fix
    assert "port-forward" not in [call["verb"] for call in tools.calls()]


def test_from_k8s_prefers_the_multi_node_leader(fake: MakeTools) -> None:
    worker = nim_pod("a-worker", labels={"app": "llama", nsa.LEADER_LABEL: "worker"})
    leader = nim_pod("z-leader", labels={"app": "llama", nsa.LEADER_LABEL: "leader"})
    starting = nim_pod("b-starting", phase="Pending", ready=False)
    tools = fake(operator(worker, starting, leader))
    with reach("nimservice/llama") as (_, forward):
        assert forward.pod == "z-leader"
    assert tools.argvs("kubectl")[-1][1] == "pod/z-leader"


def test_from_k8s_needs_a_running_and_ready_pod(fake: MakeTools) -> None:
    pending = nim_pod("llama-0", phase="Pending", ready=False)
    unready = nim_pod("llama-1", ready=False)
    leaving = nim_pod("llama-2")
    leaving["metadata"]["deletionTimestamp"] = "2026-11-11T10:00:00Z"
    tools = fake(operator(pending, unready, leaving))
    error = raises("pod_not_ready", 3, lambda: nsa.from_k8s("nimservice/llama", CONTEXT,
                                                            NAMESPACE))
    assert "llama-0: Pending; llama-1: Running, not Ready; llama-2: terminating" in error.message
    assert "port-forward" not in [call["verb"] for call in tools.calls()]
    fake({**operator(), "get pods -l app=llama": kit.k8s_list()})
    raises("pod_not_ready", 3, lambda: nsa.from_k8s("nimservice/llama", CONTEXT, NAMESPACE))


def test_from_k8s_waits_for_a_ready_pod_with_wait_ready(fake: MakeTools) -> None:
    """access-6: --wait-ready waits for a pod that is still loading its model (Running, not
    Ready; or not created yet), reading the pods again with the same read-only get, and starts
    the forward once one is Ready. Without a wait, pod_not_ready at once, as before."""
    loading, ready = nim_pod(ready=False), nim_pod()
    pods = kit.in_turn(kit.k8s_list(), kit.k8s_list(loading), kit.k8s_list(loading),
                       kit.k8s_list(ready))
    tools = fake({**operator(), "get pods -l app=llama": pods})
    access, forward = nsa.from_k8s("nimservice/llama", CONTEXT, NAMESPACE, wait=20, poll=0.05)
    try:
        assert forward.alive() and forward.pod == "llama-0" and models(access.url) == MODEL
    finally:
        forward.stop()
    verbs = [call["argv"][:2] for call in tools.calls("kubectl")]
    assert verbs.count(["get", "pods"]) == 4 and verbs[-1] == ["port-forward", "pod/llama-0"]
    tools.assert_read_only()
    tools.assert_scoped(CONTEXT, NAMESPACE)

    tools = fake({**operator(), "get pods -l app=llama": kit.k8s_list(loading)})
    error = raises("pod_not_ready", 3, lambda: nsa.from_k8s("nimservice/llama", CONTEXT,
                                                            NAMESPACE))
    assert [call["verb"] for call in tools.calls("kubectl")].count("get") == 3  # one read
    assert "--wait-ready" in error.fix
    tools = fake({**operator(), "get pods -l app=llama": kit.k8s_list(loading)})
    started = time.monotonic()
    error = raises("pod_not_ready", 3, lambda: nsa.from_k8s(
        "nimservice/llama", CONTEXT, NAMESPACE, wait=0.4, poll=0.05))
    assert time.monotonic() - started >= 0.4
    assert "Running, not Ready" in error.message and "after waiting 0.4 s" in error.message
    assert "port-forward" not in [call["verb"] for call in tools.calls()]
    # how many reads fit in the wait is test_wait_ready_reads_until_the_deadline_on_a_fake_clock's


class FakeClock:
    """time.monotonic and time.sleep for nsc_access: the clock moves only when the code sleeps or
    a stubbed read says it took time, so a slow machine changes no count."""

    def __init__(self) -> None:
        self.now = 0.0

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


def test_wait_ready_reads_until_the_deadline_on_a_fake_clock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """--wait-ready reads the target every `poll` seconds, the last time at the deadline, then
    stops with pod_not_ready that says how long it waited; with no wait, one read. Stubbed
    reads on a fake clock: no kubectl starts, so the count depends on no machine's speed."""
    clock = FakeClock()
    monkeypatch.setattr(nsa, "time", clock)
    reads: list[float] = []

    def loading(kube: Any, name: str) -> nsa._Found:
        reads.append(clock.now)
        clock.now += 0.5  # each read takes half a second
        raise nsc.ScriptError("pod_not_ready", "no pod of the NIMService llama is Running and "
                              "Ready (llama-0: Running, not Ready)", nsa.NOT_READY_FIX,
                              nsc.EXIT_PRECONDITION)

    monkeypatch.setitem(nsa.RESOLVERS, "nimservice", loading)
    kube = nsa._Kube("/opt/bin/kubectl", CONTEXT, NAMESPACE)
    error = raises("pod_not_ready", 3, lambda: nsa._ready_target(kube, "nimservice", "llama",
                                                                 10, 2))
    assert reads == [0.0, 2.5, 5.0, 7.5, 10.0]
    assert error.message.endswith("after waiting 10 s (--wait-ready)")
    reads.clear()
    error = raises("pod_not_ready", 3, lambda: nsa._ready_target(kube, "nimservice", "llama",
                                                                 0, 2))
    assert len(reads) == 1 and "after waiting" not in error.message


def test_from_k8s_waits_for_a_pod_target_too(fake: MakeTools) -> None:
    tools = fake({"auth can-i create pods": kit.can_i(True),
                  "get pod llama-0": kit.in_turn(nim_pod(ready=False), nim_pod()),
                  "port-forward pod/llama-0": SERVE})
    access, forward = nsa.from_k8s("pod/llama-0", CONTEXT, NAMESPACE, wait=10, poll=0.05)
    try:
        assert models(access.url) == MODEL
    finally:
        forward.stop()
    assert [call["argv"][:2] for call in tools.calls("kubectl")].count(["get", "pod"]) == 2


POD_NOT_FOUND = kit.ToolReply(stderr='Error from server (NotFound): pods "llama-0" not found\n',
                              exit=1)


def test_from_k8s_waits_for_a_pod_target_that_does_not_exist_yet(fake: MakeTools) -> None:
    """A StatefulSet pod that is being created again keeps its name: with --wait-ready,
    pod/NAME that kubectl does not find is read again until it is Ready. With no wait it is
    target_not_found at once, and at the deadline target_not_found says how long it waited."""
    tools = fake({"auth can-i create pods": kit.can_i(True),
                  "get pod llama-0": kit.in_turn(POD_NOT_FOUND, nim_pod(ready=False), nim_pod()),
                  "port-forward pod/llama-0": SERVE})
    access, forward = nsa.from_k8s("pod/llama-0", CONTEXT, NAMESPACE, wait=20, poll=0.05)
    try:
        assert models(access.url) == MODEL
    finally:
        forward.stop()
    assert [call["argv"][:2] for call in tools.calls("kubectl")].count(["get", "pod"]) == 3
    fixtures = {"auth can-i create pods": kit.can_i(True), "get pod llama-0": POD_NOT_FOUND}
    tools = fake(fixtures)
    error = raises("target_not_found", 3, lambda: nsa.from_k8s("pod/llama-0", CONTEXT,
                                                               NAMESPACE))
    assert "after waiting" not in error.message
    assert [call["verb"] for call in tools.calls("kubectl")].count("get") == 1
    fake(fixtures)
    error = raises("target_not_found", 3, lambda: nsa.from_k8s(
        "pod/llama-0", CONTEXT, NAMESPACE, wait=0.3, poll=0.05))
    assert "after waiting 0.3 s" in error.message and "Check the name" in error.fix


def test_from_k8s_wait_ready_does_not_wait_out_other_errors(fake: MakeTools) -> None:
    """Only a pod that is not Ready yet is waited for: a missing NIMService or a refused role
    stops at once, whatever --wait-ready says."""
    missing = kit.ToolReply(
        stderr='Error from server (NotFound): nimservices.apps.nvidia.com "llama" not found\n',
        exit=1)
    tools = fake(operator(nimservice=missing))
    started = time.monotonic()
    raises("target_not_found", 3, lambda: nsa.from_k8s("nimservice/llama", CONTEXT, NAMESPACE,
                                                       wait=30, poll=0.05))
    assert time.monotonic() - started < 10
    assert [call["verb"] for call in tools.calls("kubectl")].count("get") == 1
    forbidden = kit.ToolReply(stderr='Error from server (Forbidden): pods "llama-0" is '
                              'forbidden: User "dev" cannot get resource "pods"\n', exit=1)
    tools = fake({"auth can-i create pods": kit.can_i(True), "get pod llama-0": forbidden})
    raises("rbac_missing", 3, lambda: nsa.from_k8s("pod/llama-0", CONTEXT, NAMESPACE, wait=30,
                                                   poll=0.05))
    assert [call["verb"] for call in tools.calls("kubectl")].count("get") == 1


def test_from_k8s_stops_at_the_rbac_check_when_port_forward_is_denied(fake: MakeTools) -> None:
    tools = fake(operator(can=False))
    error = raises("rbac_missing", 3, lambda: nsa.from_k8s("nimservice/llama", CONTEXT,
                                                           NAMESPACE))
    assert "pods/portforward" in error.fix and "view does not" in error.fix
    assert [call["argv"][:2] for call in tools.calls()] == [["auth", "can-i"]]


def test_rbac_missing_asks_to_confirm_the_namespace_first(fake: MakeTools) -> None:
    """access-7: can-i also says no for a namespace that does not exist (a role bound only in
    nim, and a typo nmi), so rbac_missing first asks to confirm the context and namespace with
    the user, before any request for a role."""
    fake({
        "auth can-i create pods --subresource=portforward --context lab -n nim": kit.can_i(True),
        "auth can-i create pods --subresource=portforward --context lab -n nmi": kit.can_i(False),
    })
    error = raises("rbac_missing", 3, lambda: nsa.from_k8s("nimservice/llama", CONTEXT, "nmi"))
    assert "namespace nmi" in error.message and "does not exist" in error.message
    assert error.fix.startswith("First confirm the context and the namespace with the user")
    assert error.fix.index("namespace with the user") < error.fix.index("cluster admin")
    forbidden = kit.ToolReply(stderr='Error from server (Forbidden): pods "llama-0" is '
                              'forbidden: User "dev" cannot get resource "pods" in API group ""'
                              ' in the namespace "nmi"\n', exit=1)
    fake({"auth can-i create pods": kit.can_i(True), "get pod llama-0": forbidden})
    error = raises("rbac_missing", 3, lambda: nsa.from_k8s("pod/llama-0", CONTEXT, "nmi"))
    assert error.fix.startswith("First confirm the context and the namespace")


def test_kubectl_json_nested_too_deeply_is_kubectl_failed(fake: MakeTools) -> None:
    """docs-1: an object nested past Python's recursion limit is not JSON to the script:
    kubectl_failed, never a traceback."""
    deep = "[" * 100_000 + "]" * 100_000
    fake(operator(nimservice=kit.ToolReply(stdout=deep)))
    raises("kubectl_failed", 4, lambda: nsa.from_k8s("nimservice/llama", CONTEXT, NAMESPACE))


@pytest.mark.parametrize(
    ("stderr", "code", "exit_code", "fix"),
    [
        ('Error from server (Forbidden): nimservices.apps.nvidia.com "llama" is forbidden: User '
         '"dev" cannot get resource "nimservices" in API group "apps.nvidia.com"', "rbac_missing",
         3, "svc/NAME"),
        ('Error from server (NotFound): nimservices.apps.nvidia.com "llama" not found',
         "target_not_found", 3, "namespace"),
        ('error: the server doesn\'t have a resource type "nimservice"', "target_not_found", 3,
         "NIM Operator"),
        ("Unable to connect to the server: dial tcp 10.0.0.1:6443: i/o timeout",
         "kubectl_failed", 4, "login"),
    ],
)
def test_from_k8s_kubectl_errors(
    fake: MakeTools, stderr: str, code: str, exit_code: int, fix: str
) -> None:
    fake(operator(nimservice=kit.ToolReply(stderr=stderr + "\n", exit=1)))
    error = raises(code, exit_code, lambda: nsa.from_k8s("nimservice/llama", CONTEXT, NAMESPACE))
    assert fix in error.fix


PLUGIN_HINT = (
    "\n\nIt looks like you are trying to use a client-go credential plugin that is not "
    "installed.\n\nTo learn more about this feature, consult the documentation available at:"
    "\n      https://kubernetes.io/docs/reference/access-authn-authz/authentication/"
    "#client-go-credential-plugins\n"
)


@pytest.mark.parametrize(
    ("stderr", "shown", "fix"),
    [
        ("Unable to connect to the server: getting credentials: exec: executable "
         "gke-gcloud-auth-plugin not found" + PLUGIN_HINT + "\nInstall gke-gcloud-auth-plugin "
         "for use with kubectl by following https://cloud.google.com/blog/products/"
         "containers-kubernetes/kubectl-auth-changes-in-gke\n", "gke-gcloud-auth-plugin",
         "credential plugin"),
        ("Unable to connect to the server: getting credentials: exec: executable aws not found"
         + PLUGIN_HINT, "executable aws not found", "credential plugin"),
        ("Unable to connect to the server: getting credentials: exec: executable kubelogin not "
         "found" + PLUGIN_HINT, "kubelogin", "credential plugin"),
        ("Unable to connect to the server: getting credentials: exec: executable aws failed "
         "with exit code 255\n", "exit code 255", "logs in"),
        ("Unable to connect to the server: Forbidden\n", "Forbidden", "reaches the cluster"),
    ],
)
def test_from_k8s_a_missing_login_or_credential_plugin_is_kubectl_failed(
    fake: MakeTools, stderr: str, shown: str, fix: str
) -> None:
    """access-5: client-go's "getting credentials: exec: executable aws not found" holds "not
    found", and a proxy's refusal "Forbidden": neither comes from the cluster, which was never
    asked. kubectl_failed (exit 4) with the line that names the plugin, never target_not_found
    ("check the name") or rbac_missing."""
    for where in ("can-i", "nimservice"):
        if where == "can-i":
            tools = fake({"auth can-i create pods": kit.ToolReply(stderr=stderr, exit=1)})
        else:
            tools = fake(operator(nimservice=kit.ToolReply(stderr=stderr, exit=1)))
        error = raises("kubectl_failed", 4, lambda: nsa.from_k8s("nimservice/llama", CONTEXT,
                                                                 NAMESPACE))
        assert shown in error.message and fix in error.fix, (where, error.message, error.fix)
        assert "Check the name" not in error.fix
        tools.assert_read_only()


def test_port_forward_start_without_credentials_is_kubectl_failed(fake: MakeTools) -> None:
    """access-5: the same credential failure at the port-forward is not pod_not_ready."""
    stderr = ("Unable to connect to the server: getting credentials: exec: executable aws not "
              "found" + PLUGIN_HINT)
    fake(forward_to(kit.ToolReply(stderr=stderr, exit=1)))
    forward = nsa.PortForward(nsa.tool("kubectl"), CONTEXT, NAMESPACE, "llama-0", 8000)
    try:
        error = raises("kubectl_failed", 4, forward.start)
    finally:
        forward.stop()
    assert "executable aws not found" in error.message and "credential plugin" in error.fix
    fake(forward_to(kit.ToolReply(stderr='Error from server (NotFound): pods "llama-0" not '
                                         "found\n", exit=1)))
    forward = nsa.PortForward(nsa.tool("kubectl"), CONTEXT, NAMESPACE, "llama-0", 8000)
    try:
        raises("pod_not_ready", 3, forward.start)
    finally:
        forward.stop()


def test_from_k8s_names_the_contexts_when_the_context_is_missing(fake: MakeTools) -> None:
    tools = fake({
        "auth can-i create pods": kit.ToolReply(stderr='error: context "nope" does not exist\n',
                                                exit=1),
        "config get-contexts": "lab\nprod\n",
    })
    error = raises("context_not_found", 3, lambda: nsa.from_k8s("pod/x", "nope", NAMESPACE))
    assert "lab, prod" in error.fix and "never pick one yourself" in error.fix
    assert tools.argvs("kubectl")[-1] == ["config", "get-contexts", "-o", "name"]
    tools.assert_read_only()


def test_from_k8s_value_from_passthrough_is_unknown_and_secrets_are_never_kept(
    fake: MakeTools, capsys: pytest.CaptureFixture[str]
) -> None:
    pod = nim_pod(env={"HF_TOKEN": kit.fake_token(), "LITERAL_KEY": kit.fake_key()})
    container = pod["spec"]["containers"][0]
    container["env"].append({"name": "NIM_PASSTHROUGH_ARGS", "valueFrom": {
        "configMapKeyRef": {"name": "nim-args", "key": "args"}}})
    fake(operator(pod))
    with reach("nimservice/llama") as (access, _):
        assert access.effective_passthrough == nsa.UNKNOWN
        assert access.facts()["passthrough"] == "unknown"
        assert any("valueFrom" in note for note in access.notes)
        shown = json.dumps(asdict(access)) + json.dumps(access.facts())
    captured = capsys.readouterr()
    for value in (kit.fake_token(), kit.fake_key()):
        assert value not in shown + captured.out + captured.err
    container["env"] = [{"name": "NIM_SERVER_PORT", "value": "8000"}]
    container["envFrom"] = [{"configMapRef": {"name": "nim-env"}}]
    fake(operator(pod))
    with reach("nimservice/llama") as (access, _):
        # envFrom may set NIM_PASSTHROUGH_ARGS and so replace the image default: unknown, never
        # "not set", which would let tier B take the image default for the deployment's value
        assert access.effective_passthrough == nsa.UNKNOWN
        assert access.facts()["passthrough"] == "unknown"
        note = next(note for note in access.notes if "envFrom" in note)
        assert "NIM_SSL_MODE" in note and "NIM_PASSTHROUGH_ARGS" in note
        assert "NIM_SERVER_PORT" not in note  # set in env, which wins over envFrom
    container["env"].append({"name": "NIM_PASSTHROUGH_ARGS", "value": TOOL_FLAGS})
    fake(operator(pod))
    with reach("nimservice/llama") as (access, _):
        assert access.effective_passthrough == TOOL_FLAGS


def from_config_map(name: str) -> dict[str, Any]:
    """An env entry of a pod that takes `name` from a ConfigMap (valueFrom)."""
    return {"name": name,
            "valueFrom": {"configMapKeyRef": {"name": "nim-settings", "key": name.lower()}}}


def test_from_k8s_stops_when_the_tls_mode_comes_from_a_config_map(
    fake: MakeTools, scrubber: nsc.Scrubber
) -> None:
    """NIM_SSL_MODE from a ConfigMap or a Secret (valueFrom) may be TLS, and an http URL would
    then fail every check for the wrong reason. The script never reads either, so it stops
    before the port-forward and asks for --url (setting_not_read, exit 3). The fix names the
    flags an MTLS NIM needs too, and the masked output keeps them all."""
    pod = nim_pod()
    pod["spec"]["containers"][0]["env"].append(from_config_map("NIM_SSL_MODE"))
    tools = fake(operator(pod))
    error = raises("setting_not_read", 3, lambda: nsa.from_k8s("nimservice/llama", CONTEXT,
                                                               NAMESPACE))
    assert "NIM_SSL_MODE" in error.message and "pod llama-0" in error.message
    assert "http or https" in error.message and "--url" in error.fix
    assert all(flag in error.fix for flag in ("--ca-file", "--client-cert", "--client-key"))
    assert scrubber.scrub(error.fix) == error.fix
    assert "port-forward" not in [call["verb"] for call in tools.calls()]
    tools.assert_read_only()


def test_from_k8s_server_and_health_port_from_a_config_map(fake: MakeTools) -> None:
    """NIM_SERVER_PORT from valueFrom matters only when the container declares no port: then
    setting_not_read; a declared port is used as before. NIM_HEALTH_PORT from valueFrom gets a
    note, and health stays on the forwarded port."""
    pod = nim_pod()
    container = pod["spec"]["containers"][0]
    container["env"] += [from_config_map("NIM_SERVER_PORT"), from_config_map("NIM_HEALTH_PORT")]
    fake(operator(pod))
    with reach("nimservice/llama") as (access, forward):
        assert forward.remote_port == 8000 and access.health_url == access.url
        assert any("NIM_HEALTH_PORT comes from a ConfigMap" in note for note in access.notes)
    del container["ports"]
    tools = fake({"auth can-i create pods": kit.can_i(True), "get pod llama-0": pod,
                  "port-forward pod/llama-0": SERVE})
    error = raises("setting_not_read", 3, lambda: nsa.from_k8s("pod/llama-0", CONTEXT,
                                                               NAMESPACE))
    assert "NIM_SERVER_PORT" in error.message and "declares no port" in error.message
    assert "port-forward" not in [call["verb"] for call in tools.calls()]


def test_from_k8s_kserve_finds_the_predictor_pod_by_its_label(fake: MakeTools) -> None:
    service = kit.k8s_nimservice("llama", NAMESPACE, REPO, "2.1.5")
    service["spec"]["inferencePlatform"] = "kserve"
    pod = nim_pod("llama-predictor-0", labels={nsa.KSERVE_LABEL: "llama"})
    pod["spec"]["containers"][0]["ports"] = [{"containerPort": 8000, "protocol": "TCP"}]
    tools = fake({
        "auth can-i create pods": kit.can_i(True),
        "get nimservice llama": service,
        f"get pods -l {nsa.KSERVE_LABEL}=llama": kit.k8s_list(pod),
        "port-forward pod/llama-predictor-0": SERVE,
    })
    with reach("nimservice/llama") as (access, forward):
        assert forward.remote_port == 8000
        assert any("KServe" in note for note in access.notes)
    assert ["get", "service", "llama"] not in [argv[:3] for argv in tools.argvs("kubectl")]


def test_from_k8s_health_port_tls_and_image_drift(fake: MakeTools) -> None:
    pod = nim_pod(image=f"{REPO}:2.1.6", env={"NIM_HEALTH_PORT": "8001", "NIM_SSL_MODE": "TLS"})
    fake(operator(pod))
    with reach("nimservice/llama") as (access, forward):
        assert access.url == f"https://127.0.0.1:{forward.local_port}"
        assert access.health_url is None and access.image == f"{REPO}:2.1.6"
        notes = " ".join(access.notes)
        assert "NIM_HEALTH_PORT=8001" in notes and "NIM_SSL_MODE=TLS" in notes
        assert f"asks for {IMAGE}" in notes


@pytest.mark.parametrize(
    ("target", "context", "namespace"),
    [
        ("deployment/llama", CONTEXT, NAMESPACE),
        ("nimservice/", CONTEXT, NAMESPACE),
        ("nimservice/Llama", CONTEXT, NAMESPACE),
        ("llama", CONTEXT, NAMESPACE),
        ("helm/" + "a" * 60, CONTEXT, NAMESPACE),
        ("pod/llama-0", "-x", NAMESPACE),
        ("pod/llama-0", "my context", NAMESPACE),
        ("pod/llama-0", CONTEXT, "Bad_NS"),
        ("pod/llama-0", CONTEXT, ""),
    ],
)
def test_from_k8s_refuses_bad_input_before_running_kubectl(
    fake: MakeTools, target: str, context: str, namespace: str
) -> None:
    tools = fake({"*": "anything"})
    raises("bad_input", 2, lambda: nsa.from_k8s(target, context, namespace))
    assert tools.calls() == []


def test_parse_target_forms() -> None:
    assert nsa.parse_target("nimservice/llama") == ("nimservice", "llama")
    assert nsa.parse_target("service/llama") == ("svc", "llama")
    assert nsa.parse_target(" Helm/my-nim ") == ("helm", "my-nim")
    assert nsa.parse_target("pod/llama-0") == ("pod", "llama-0")


# --- the port-forward -----------------------------------------------------------------------------


def forward_to(target: Any = SERVE, **fixtures: Any) -> dict[str, Any]:
    return {"port-forward pod/llama-0": target, **fixtures}


def test_port_forward_start_alive_stop_leaves_no_process(fake: MakeTools) -> None:
    with kit.FakeNim() as nim:
        tools = fake(forward_to(kit.port_forward_to(nim)))
        forward = nsa.PortForward(nsa.tool("kubectl"), CONTEXT, NAMESPACE, "llama-0", 8000)
        assert forward.url is None and forward.argv[3] == ":8000"
        try:
            port = forward.start()
            assert forward.alive() and forward.start() == port  # a second start reuses it
            assert forward._proc is not None and forward._proc.stdout is None  # a file, no pipe
            folder = forward._folder
            assert folder is not None and folder.stat().st_mode & 0o077 == 0
            assert models(forward.url or "") == MODEL
        finally:
            forward.stop()
        forward.stop()  # twice is fine
    assert not forward.alive() and not folder.exists()
    assert [call["argv"] for call in tools.calls()] == [
        ["port-forward", "pod/llama-0", ":8000", *SCOPE]]
    assert all(gone(pid) for pid in forward_pids(tools))


def test_port_forward_restarts_once_and_then_reports_lost(fake: MakeTools) -> None:
    with kit.FakeNim() as nim:
        tools = fake(forward_to(kit.port_forward_to(nim, lost_after=1, lose_starts=2)))
        forward = nsa.PortForward(nsa.tool("kubectl"), CONTEXT, NAMESPACE, "llama-0", 8000)
        try:
            forward.start()
            with pytest.raises((urllib.error.URLError, ConnectionError, OSError)):
                models(forward.url or "")
            wait_for(lambda: not forward.alive())
            forward.ensure()
            assert forward.restarts == 1 and forward.alive()
            with pytest.raises((urllib.error.URLError, ConnectionError, OSError)):
                models(forward.url or "")
            wait_for(lambda: not forward.alive())
            error = raises("port_forward_lost", 4, forward.ensure)
            assert "lost connection to pod" in error.message and "nim-deploy" in error.fix
        finally:
            forward.stop()
    # 2 runs, or 3 when the old local port was not free yet and the restart moved
    assert len(forward_pids(tools)) >= 2 and all(gone(pid) for pid in forward_pids(tools))


def test_settle_waits_for_kubectl_to_exit_after_a_lost_connection(fake: MakeTools) -> None:
    """access-2: kubectl closes the client's connection when the forward breaks and exits a
    moment later; a check right after the failure finds it running. settle() waits for the
    exit, so ensure() restarts the forward instead of handing back its dead port."""
    with kit.FakeNim() as nim:
        fake(forward_to(kit.port_forward_to(nim, lost_after=1)))
        forward = nsa.PortForward(nsa.tool("kubectl"), CONTEXT, NAMESPACE, "llama-0", 8000)
        try:
            forward.start()
            with pytest.raises((urllib.error.URLError, ConnectionError, OSError)):
                models(forward.url or "")
            assert forward.settle() is True and not forward.alive()
            forward.ensure()
            assert forward.restarts == 1 and models(forward.url or "") == MODEL
            started = time.monotonic()
            assert forward.settle(timeout=0.3) is False  # a forward that runs: no wait past it
            assert 0.25 <= time.monotonic() - started < 5 and forward.alive()
        finally:
            forward.stop()


LISTENER_GONE = """
import socket, sys, time
from pathlib import Path
counter = Path(__file__).resolve().with_name("starts")
starts = int(counter.read_text()) + 1 if counter.exists() else 1
counter.write_text(str(starts))
spec = next(word for word in sys.argv[1:] if word.endswith(":8000"))
local = int(spec.split(":")[0] or 0)
listener = socket.socket()
listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
listener.bind(("127.0.0.1", local))
listener.listen(8)
print(f"Forwarding from 127.0.0.1:{listener.getsockname()[1]} -> 8000", flush=True)
if starts == 1:
    listener.accept()[0].close()  # the start's connect, then the forward breaks: the
    listener.close()               # listener is gone while kubectl has not exited yet
    time.sleep(60)
while True:
    listener.accept()[0].close()
"""


def test_ensure_restarts_a_forward_whose_kubectl_runs_but_no_longer_listens(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """access-2: alive() alone is not enough. kubectl that still runs but refuses connections
    on its local port is ending: ensure() gives it SETTLE_WAIT seconds, then restarts it."""
    write_program(tmp_path / "bin", "kubectl", LISTENER_GONE)
    monkeypatch.setenv("PATH", str(tmp_path / "bin"))
    monkeypatch.setattr(nsa, "SETTLE_WAIT", 0.3)
    forward = nsa.PortForward(nsa.tool("kubectl"), CONTEXT, NAMESPACE, "llama-0", 8000)
    try:
        first = forward.start()
        wait_for(lambda: not nsa._connects("127.0.0.1", first))
        assert forward.alive() and forward.restarts == 0
        port = forward.ensure()
        assert forward.restarts == 1 and forward.alive()
        assert nsa._connects("127.0.0.1", port)
    finally:
        forward.stop()
    assert (tmp_path / "bin" / "starts").read_text() == "2"


SLOW_THEN_READY = """
import socket, sys
from pathlib import Path
counter = Path(__file__).resolve().with_name("starts")
starts = int(counter.read_text()) + 1 if counter.exists() else 1
counter.write_text(str(starts))
if starts == 1:
    sys.exit("error: unable to forward port because pod is not running. Current status=Pending")
listener = socket.socket()
listener.bind(("127.0.0.1", 0))
listener.listen(8)
print(f"Forwarding from 127.0.0.1:{listener.getsockname()[1]} -> 8000", flush=True)
while True:
    listener.accept()[0].close()
"""


def test_ensure_after_a_failed_first_start_starts_and_keeps_the_restart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A first start that failed (a pod still loading) never listened: ensure() starts it, it
    does not use up the one restart or log that the forward "stopped; restarting"."""
    write_program(tmp_path / "bin", "kubectl", SLOW_THEN_READY)
    monkeypatch.setenv("PATH", str(tmp_path / "bin"))
    forward = nsa.PortForward(nsa.tool("kubectl"), CONTEXT, NAMESPACE, "llama-0", 8000)
    try:
        raises("pod_not_ready", 3, forward.start)
        port = forward.ensure()
        assert forward.restarts == 0 and forward.alive() and nsa._connects("127.0.0.1", port)
        assert forward.facts()["restarts"] == 0
    finally:
        forward.stop()
    assert "restarting" not in capsys.readouterr().err
    assert (tmp_path / "bin" / "starts").read_text() == "2"


def test_a_restart_keeps_the_access_on_the_new_url(fake: MakeTools) -> None:
    with kit.FakeNim() as nim:
        fake(operator(forward=kit.port_forward_to(nim, lost_after=1)))
        with reach("nimservice/llama") as (access, forward):
            with pytest.raises((urllib.error.URLError, ConnectionError, OSError)):
                models(access.url)
            wait_for(lambda: not forward.alive())
            forward.ensure()
            assert access.url == forward.url == access.health_url
            assert access.port_forward is not None and access.port_forward["restarts"] == 1
            for _ in range(3):
                assert models(access.url) == MODEL


def test_a_restart_moves_to_a_new_local_port_when_the_old_one_is_taken(
    fake: MakeTools,
) -> None:
    tools = fake(forward_to())
    forward = nsa.PortForward(nsa.tool("kubectl"), CONTEXT, NAMESPACE, "llama-0", 8000)
    blocker = socket.socket()
    try:
        old = forward.start()
        forward._halt()  # kubectl dies; its port is free again
        blocker.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        blocker.bind(("127.0.0.1", old))
        blocker.listen(1)
        new = forward.ensure()
        assert new != old and forward.restarts == 1 and models(forward.url or "") == MODEL
        assert forward.facts()["command"].startswith("kubectl port-forward pod/llama-0 :8000")
    finally:
        blocker.close()
        forward.stop()
    tried = [call["argv"][2] for call in tools.calls()]
    assert tried == [":8000", f"{old}:8000", ":8000"]


@pytest.mark.parametrize(
    ("stderr", "code", "exit_code"),
    [
        ('Error from server (Forbidden): pods "llama-0" is forbidden: User "dev" cannot create '
         'resource "pods/portforward"', "rbac_missing", 3),
        ("error: unable to forward port because pod is not running. Current status=Pending",
         "pod_not_ready", 3),
        ("error: error upgrading connection: unable to upgrade connection", "port_forward_failed",
         4),
    ],
)
def test_port_forward_start_errors(
    fake: MakeTools, stderr: str, code: str, exit_code: int
) -> None:
    fake(forward_to(kit.ToolReply(stderr=stderr + "\n", exit=1)))
    forward = nsa.PortForward(nsa.tool("kubectl"), CONTEXT, NAMESPACE, "llama-0", 8000)
    try:
        error = raises(code, exit_code, forward.start)
    finally:
        forward.stop()
    assert stderr.split(":")[0].lower() in error.message.lower()
    assert not forward.alive()


def test_port_forward_that_never_listens_is_killed_at_the_deadline(fake: MakeTools) -> None:
    tools = fake(forward_to(kit.PortForwardReply(serve={"model": MODEL}, delay=20)))
    forward = nsa.PortForward(nsa.tool("kubectl"), CONTEXT, NAMESPACE, "llama-0", 8000)
    started = time.monotonic()
    try:
        error = raises("port_forward_failed", 4, lambda: forward.start(timeout=0.5))
    finally:
        forward.stop()
    assert "within 0.5 s" in error.message and time.monotonic() - started < 15
    assert all(gone(pid) for pid in forward_pids(tools))


def test_port_forward_context_manager_and_exit_cleanup(fake: MakeTools) -> None:
    tools = fake(forward_to())
    kubectl = nsa.tool("kubectl")
    with pytest.raises(RuntimeError), nsa.PortForward(kubectl, CONTEXT, NAMESPACE, "llama-0",
                                                     8000) as forward:
        assert forward.alive()
        raise RuntimeError("a check crashed")
    assert not forward.alive()
    left = nsa.PortForward(kubectl, CONTEXT, NAMESPACE, "llama-0", 8000)
    left.start()
    nsa._stop_all()  # what atexit runs
    assert not left.alive()
    assert all(gone(pid) for pid in forward_pids(tools))


def test_port_forward_rejects_bad_arguments() -> None:
    for pod, port in (("Bad_Pod", 8000), ("llama-0", 0), ("llama-0", 70000),
                      ("llama-0", "8000"), ("-x", 8000)):
        with pytest.raises(ValueError):
            nsa.PortForward("/opt/bin/kubectl", CONTEXT, NAMESPACE, pod, port)


CHILD = """
import sys, time
sys.path.insert(0, {scripts!r})
import nsc_access as nsa
forward = nsa.PortForward(nsa.tool("kubectl"), "lab", "nim", "llama-0", 8000)
forward.start()
print(forward._proc.pid, flush=True)
while True:
    time.sleep(0.05)
"""


def read_line(stream: IO[str], seconds: float) -> str:
    """One line from a child's pipe, or "" when none comes within `seconds`: a child that
    stalls cannot hold the test. The reading thread ends when the child's pipe closes."""
    lines: queue.Queue[str] = queue.Queue()
    threading.Thread(target=lambda: lines.put(stream.readline()), daemon=True).start()
    try:
        return lines.get(timeout=seconds)
    except queue.Empty:
        return ""


@pytest.mark.skipif(os.name != "posix", reason="POSIX signals")
def test_sigterm_ends_the_script_and_stops_the_forward(fake: MakeTools) -> None:
    tools = fake(forward_to())
    child = subprocess.Popen(
        [sys.executable, "-I", "-B", "-c", CHILD.format(scripts=str(SCRIPTS))],
        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        env=kit.child_env({"PATH": str(tools.bin)}), text=True,
    )
    pid = 0
    try:
        assert child.stdout is not None
        line = read_line(child.stdout, 30)  # its own deadline, apart from the SIGTERM one
        assert line.strip(), "the child printed no port-forward PID within 30 s"
        pid = int(line)
        assert not gone(pid)
        child.send_signal(signal.SIGTERM)
        assert child.wait(timeout=20) == 128 + signal.SIGTERM
        wait_for(lambda: gone(pid))
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=10)  # reap it
        for left in {pid, *forward_pids(tools)} - {0}:  # a kubectl the child never named too
            if not gone(left):
                os.kill(left, signal.SIGKILL)
        if child.stdout is not None:
            child.stdout.close()


def wait_for(condition: Callable[[], bool], seconds: float = 10) -> None:
    deadline = time.monotonic() + seconds
    while not condition():
        assert time.monotonic() < deadline, "timed out waiting"
        time.sleep(0.02)


# --- messages of kubectl and docker ---------------------------------------------------------------


@pytest.fixture
def scrubber(monkeypatch: pytest.MonkeyPatch) -> nsc.Scrubber:
    """A fresh SCRUBBER, so that a value one test registers masks nothing in another."""
    fresh = nsc.Scrubber()
    monkeypatch.setattr(nsc, "SCRUBBER", fresh)
    return fresh


def test_tool_messages_are_masked_before_they_are_cut(
    fake: MakeTools, scrubber: nsc.Scrubber, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A key that kubectl, docker or a credential plugin prints across the MAX_SHOWN cut: once
    cut, neither the registered key nor a known token shape matches its first part, so masking
    the cut text later (in the log or the JSON) would keep that part. The whole line is masked
    first, then cut."""
    scrubber.add(kit.fake_token())
    for secret in (kit.fake_token(), kit.fake_key()):
        part = secret[:9]  # what a cut at MAX_SHOWN keeps of it in each line below

        def line(lead: str, end: str = " ", secret: str = secret) -> str:
            return lead + "x" * (nsa.MAX_SHOWN - 9 - len(lead) - len(end)) + end + secret

        cluster = nsa._no_cluster(line("Unable to connect to the server: "), "read the pod")
        assert cluster is not None
        down = kit.ToolReply(stderr=line("Error response from daemon: ") + "\n", exit=1)
        fake(docker=docker_fixtures(down))
        shown = [
            nsa._first_line(line("error: ") + " more"),
            nsa._last_error("info\n" + line("error: ") + " more\n"),
            cluster[0],
            raises("docker_failed", 4, lambda: nsa.from_docker("llama")).message,
        ]
        monkeypatch.setenv("DOCKER_HOST", line("tcp://10.0.0.5:2376/", "/"))
        shown.append(raises("docker_remote", 3, lambda: nsa.from_docker("llama")).message)
        monkeypatch.delenv("DOCKER_HOST")
        for text in shown:
            assert part not in text and nsc.MASK in text, text


# --- the module itself ----------------------------------------------------------------------------

MODULE = SCRIPTS / "nsc_access.py"


def _enclosing(tree: ast.AST) -> dict[ast.AST, ast.AST]:
    parents: dict[ast.AST, ast.AST] = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node
    return parents


def test_nsc_access_starts_programs_only_inside_run_tool_and_never_with_a_shell() -> None:
    tree = ast.parse(MODULE.read_text(encoding="utf-8"))
    parents = _enclosing(tree)
    starters = {"run", "Popen", "call", "check_call", "check_output", "getoutput",
                "getstatusoutput"}
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert node.module not in ("subprocess", "os", "pty"), ast.dump(node)
        if isinstance(node, ast.keyword) and node.arg == "shell":
            assert isinstance(node.value, ast.Constant) and node.value.value is False
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)):
            continue
        module, name = node.func.value.id, node.func.attr
        if (module == "subprocess" and name in starters) or (
            module == "os" and (name in ("system", "popen") or name.startswith(("exec", "spawn",
                                                                                 "posix_spawn")))
        ):
            owner = parents.get(node)
            while owner is not None and not isinstance(owner, ast.FunctionDef):
                owner = parents.get(owner)
            found.append((name, getattr(owner, "name", None)))
    assert sorted(found) == [("Popen", "run_tool"), ("run", "run_tool")]


def test_nsc_access_reads_only_path_and_docker_host_from_the_environment() -> None:
    tree = ast.parse(MODULE.read_text(encoding="utf-8"))
    reads, uses = set(), 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr in ("environ", "getenv", "environb"):
            uses += 1
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "get" and isinstance(node.func.value, ast.Attribute)
                and node.func.value.attr == "environ"):
            reads.add(node.args[0].value)  # type: ignore[attr-defined]
    assert reads == {"PATH", "DOCKER_HOST"} and uses == 2


def test_nsc_access_runs_as_a_library_notice() -> None:
    code, payload, _ = kit.run_cli("nsc_access.py")
    assert code == 0 and payload["state"] == "library"
    code, payload, _ = kit.run_cli("nsc_access.py", "--bogus")
    assert code == 2 and payload["errors"][0]["code"] == "bad_input"
    code, text, _ = kit.run_cli("nsc_access.py", "--help", raw=True)
    assert code == 0 and "import it" in text
