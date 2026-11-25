# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Endpoint checks must use the same network and credentials as the pod client."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import threading
import urllib.error
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import nb_common  # noqa: E402
import nimbench  # noqa: E402
from test_nb_perf import export_for  # noqa: E402

ENDPOINT = "http://my-nim.nim:8000/v1"
MODEL = "qwen/qwen3.5-122b-a10b"
CLIENT = [
    "--client",
    "kubernetes",
    "--namespace",
    "nim",
    "--pod",
    "bench-client",
]
POD_COMMAND_INDEX = 6
PYTHON_ARGS_INDEX = 7
PYTHON_CODE_INDEX = 8
PROBE_URL_INDEX = 9


def invoke(argv: list[str], capsys: pytest.CaptureFixture[str]) -> tuple[int, dict[str, Any]]:
    with pytest.raises(SystemExit) as stop:
        nimbench.main(argv)
    return int(stop.value.code or 0), json.loads(capsys.readouterr().out)


@pytest.fixture
def pod_network(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    probes: list[str] = []

    def no_route(*_a: Any, **_k: Any) -> Any:
        raise urllib.error.URLError("no host route to the cluster")

    def kubectl(argv: list[str], **_kwargs: Any) -> Any:
        if argv[:2] == ["kubectl", "get"]:  # the client pod exists
            return subprocess.CompletedProcess(argv, 0, "Running", "")
        assert argv[:POD_COMMAND_INDEX] == [
            "kubectl",
            "exec",
            "-n",
            "nim",
            "bench-client",
            "--",
        ]
        if argv[POD_COMMAND_INDEX:PYTHON_CODE_INDEX] == ["python3", "-c"]:
            url = argv[PROBE_URL_INDEX]
            probes.append(url)
            name = "judge" if "judge" in url else MODEL
            return subprocess.CompletedProcess(argv, 0, json.dumps({"data": [{"id": name}]}), "")
        if argv[POD_COMMAND_INDEX:] == ["aiperf", "--version"]:
            return subprocess.CompletedProcess(argv, 0, "AIPerf 0.13.0", "")
        if argv[POD_COMMAND_INDEX] == "cat":
            return subprocess.CompletedProcess(argv, 0, json.dumps(export_for(40)), "")
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(nb_common.urllib.request, "urlopen", no_route)
    monkeypatch.setattr(nb_common, "run_command", kubectl)
    monkeypatch.setattr(shutil, "which", lambda name: f"/usr/bin/{name}")
    return probes


def test_cluster_dns_dry_run_and_approved_run_record_both_endpoints(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], pod_network: list[str]
) -> None:
    args = [
        "perf",
        "--endpoint",
        ENDPOINT,
        "--model",
        MODEL,
        "--tokenizer",
        "Qwen/Qwen3.5-122B-A10B",
        *CLIENT,
        "--loads",
        "1",
        "--out",
        str(tmp_path),
        "--baseline",
        "http://baseline.nim:8000/v1",
    ]
    code, planned = invoke([*args, "--dry-run"], capsys)
    assert code == 0 and planned["ok"]
    assert pod_network == [
        ENDPOINT + "/models",
        "http://baseline.nim:8000/v1/models",
    ]
    code, done = invoke([*args, "--approved", planned["facts"]["plan_id"]], capsys)
    assert code == 0 and done["ok"]
    assert len(pod_network) == 6  # planning, preconditions and manifests, for both endpoints
    for folder in (tmp_path, tmp_path / "baseline"):
        assert json.loads((folder / "manifest.json").read_text())["endpoint_models"] == [MODEL]


def test_check_can_use_cluster_dns(
    capsys: pytest.CaptureFixture[str], pod_network: list[str]
) -> None:
    code, checked = invoke(["check", "--endpoint", ENDPOINT, "--model", MODEL, *CLIENT], capsys)
    assert code == 0 and checked["facts"]["served_models"] == [MODEL]
    assert pod_network and set(pod_network) == {ENDPOINT + "/models"}


def test_reported_dry_run_with_cluster_only_endpoint(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], pod_network: list[str]
) -> None:
    # The reported command, with the required output folder supplied.
    code, planned = invoke(
        [
            "perf",
            "--endpoint",
            "http://my-nim.nim:8000/v1",
            "--model",
            "qwen/qwen3.5-122b-a10b",
            "--tokenizer",
            "Qwen/Qwen3.5-122B-A10B",
            "--client",
            "kubernetes",
            "--namespace",
            "nim",
            "--pod",
            "bench-client",
            "--dry-run",
            "--out",
            str(tmp_path),
        ],
        capsys,
    )
    assert code == 0 and planned["ok"] and planned["facts"]["plan_id"]
    assert pod_network == [ENDPOINT + "/models"]


def test_pod_probe_reads_credentials_inside_pod(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    received: list[tuple[str, str | None]] = []

    class Models(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            received.append((self.path, self.headers.get("Authorization")))
            self.send_response(200)
            self.end_headers()
            self.wfile.write(json.dumps({"data": [{"id": MODEL}]}).encode())

        def log_message(self, *_args: Any) -> None:
            pass

    def exec_in_pod(argv: list[str], timeout: float, **_kwargs: Any) -> Any:
        assert "pod-only-credential" not in " ".join(argv)
        return subprocess.run(
            [sys.executable, *argv[PYTHON_ARGS_INDEX:]],
            timeout=timeout,
            capture_output=True,
            text=True,
            env={"POD_KEY": "pod-only-credential"},
        )

    monkeypatch.setenv("POD_KEY", "different-host-credential")
    monkeypatch.setattr(nb_common, "run_command", exec_in_pod)
    with HTTPServer(("127.0.0.1", 0), Models) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            assert nb_common.check_endpoint(
                f"http://127.0.0.1:{server.server_port}",
                MODEL,
                "POD_KEY",
                pod=("ns", "client"),
            ) == ([MODEL], [])
        finally:
            server.shutdown()
            thread.join()
    assert received == [("/v1/models", "Bearer pod-only-credential")]


def test_judge_precondition_uses_the_selected_pod(pod_network: list[str]) -> None:
    args = nimbench.build_parser().parse_args(
        [
            "perf",
            "--endpoint",
            ENDPOINT,
            "--model",
            MODEL,
            "--tokenizer",
            "tok",
            "--out",
            "unused",
            *CLIENT,
        ]
    )
    errors = nimbench.precondition_errors(
        args,
        None,
        False,
        {
            "judge": {"endpoint": "http://judge.nim:8000/v1", "model": "judge"},
        },
    )
    assert errors == []
    assert pod_network == [
        ENDPOINT + "/models",
        "http://judge.nim:8000/v1/models",
    ]


@pytest.mark.parametrize("failure", ["exec", "timeout", "json", "model"])
def test_pod_probe_failures_are_structured_errors(
    monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    def failed(argv: list[str], **_kwargs: Any) -> Any:
        if failure == "timeout":
            raise subprocess.TimeoutExpired(argv, 30)
        return subprocess.CompletedProcess(
            argv,
            27 if failure == "exec" else 0,
            '{"data": [{"id": "other"}]}' if failure == "model" else "invalid JSON",
            "private error",
        )

    monkeypatch.setattr(nb_common, "run_command", failed)
    _names, errors = nb_common.check_endpoint(
        ENDPOINT.removesuffix("/v1"), MODEL, pod=("nim", "bench-client")
    )
    assert errors[0]["code"] == {
        "model": "model_not_served",
        "exec": "client_missing",
    }.get(failure, "endpoint_unreachable")
    assert "private error" not in json.dumps(errors)
    if failure == "exec":
        assert "exit code 27" in errors[0]["message"] and "kubectl exec" in errors[0]["message"]


@pytest.mark.parametrize("command", ["check", "perf"])
@pytest.mark.parametrize("pod_flags", [[], ["--namespace", "ns"], ["--pod", "client"]])
def test_kubernetes_commands_require_namespace_and_pod(
    command: str, pod_flags: list[str], capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_probe(*_args: Any, **_kwargs: Any) -> Any:
        pytest.fail("invalid arguments must be rejected before probing")

    monkeypatch.setattr(nb_common, "check_endpoint", unexpected_probe)
    args = [command, "--endpoint", ENDPOINT, "--model", MODEL, "--client", "kubernetes"]
    if command == "perf":
        args += ["--tokenizer", "tok", "--out", "unused", "--dry-run"]
    code, result = invoke([*args, *pod_flags], capsys)
    assert code == 2 and result["state"] == "bad_arguments"
    assert "--namespace and --pod" in result["errors"][0]["message"]


@pytest.mark.parametrize("pod", [
    ("--context=other", "client"), ("ns", "--help"), ("", "client"), ("ns", ""),
])
def test_pod_probe_rejects_empty_or_option_like_names(
    pod: tuple[str, str], monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_exec(*_args: Any, **_kwargs: Any) -> Any:
        pytest.fail("invalid pod arguments must not reach kubectl")

    monkeypatch.setattr(nb_common, "run_command", unexpected_exec)
    with pytest.raises(ValueError, match="namespace and pod"):
        nb_common.get_json(ENDPOINT + "/models", pod=pod)


def pod_failure(stdout: str = "", code: int = 0, stderr: str = "") -> Any:
    def run(argv: list[str], **_kwargs: Any) -> Any:
        return subprocess.CompletedProcess(argv, code, stdout, stderr)

    return run


@pytest.mark.parametrize(
    ("reason", "shown"),
    [
        ("URLError: [Errno -2] Name or service not known", "Name or service not known"),
        ("HTTPError: Unauthorized", "HTTPError: Unauthorized"),
    ],
)
def test_a_failed_request_in_the_pod_says_why(
    monkeypatch: pytest.MonkeyPatch, reason: str, shown: str
) -> None:
    monkeypatch.setattr(
        nb_common, "run_command", pod_failure(reason, nb_common.POD_REQUEST_FAILED)
    )
    _names, errors = nb_common.check_endpoint(
        "http://nosuch.nim:8000", MODEL, pod=("nim", "client")
    )
    assert errors[0]["code"] == "endpoint_unreachable"
    assert shown in errors[0]["message"] and "pod nim/client" in errors[0]["message"]


def test_the_in_pod_probe_exits_with_the_fixed_code_and_prints_the_reason() -> None:
    captured: list[list[str]] = []

    def run(argv: list[str], **_kwargs: Any) -> Any:
        captured.append(argv)
        return subprocess.run(  # the pod's python3, here this interpreter, to a closed port
            [sys.executable, *argv[PYTHON_ARGS_INDEX:]], capture_output=True, text=True, timeout=30
        )

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(nb_common, "run_command", run)
        _names, errors = nb_common.check_endpoint(
            "http://127.0.0.1:9", MODEL, pod=("nim", "client")
        )
    assert errors[0]["code"] == "endpoint_unreachable"
    assert "URLError" in errors[0]["message"] and "Traceback" not in errors[0]["message"]


def test_a_missing_python_in_the_pod_is_not_reported_as_a_bad_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(nb_common, "run_command", pod_failure("", 126, "no python3"))
    _names, errors = nb_common.check_endpoint(
        "http://svc.nim:8000", MODEL, pod=("nim", "client")
    )
    assert errors[0]["code"] == "client_missing" and "python3" in errors[0]["fix"]
    assert "no python3" not in json.dumps(errors)


def test_check_reports_the_pods_aiperf_not_this_machines(
    capsys: pytest.CaptureFixture[str], pod_network: list[str]
) -> None:
    code, checked = invoke(["check", "--endpoint", ENDPOINT, "--model", MODEL, *CLIENT], capsys)
    assert code == 0
    assert checked["facts"]["aiperf"] is True
    assert checked["facts"]["aiperf_in"] == "pod nim/bench-client"


def pod_commands(
    get: tuple[int, str, str] = (0, "Running", ""), exec_: tuple[int, str] = (0, "")
) -> Any:
    """kubectl get pod answers GET = (exit code, stdout, stderr); aiperf --version EXEC_ = (exit
    code, stderr). Anything else (a probe) must not happen."""

    def run(argv: list[str], **_kwargs: Any) -> Any:
        if argv[:3] == ["kubectl", "get", "pod"]:
            return subprocess.CompletedProcess(argv, get[0], get[1], get[2])
        if argv[-2:] == ["aiperf", "--version"]:
            return subprocess.CompletedProcess(argv, exec_[0], "", exec_[1])
        pytest.fail("the endpoint must not be probed through a pod that cannot run aiperf")

    return run


@pytest.mark.parametrize(
    ("commands", "code", "words"),
    [
        (pod_commands(get=(1, "", 'Error: pods "x" not found')), "client_pod_not_found", "not found"),
        (pod_commands(get=(1, "", "forbidden: cannot get")), "client_missing", "cannot be read"),
        (pod_commands(get=(0, "Pending", "")), "client_pod_not_running", "Pending, not Running"),
        (pod_commands(get=(0, "Succeeded", "")), "client_pod_not_running", "Succeeded"),
        (
            pod_commands(exec_=(126, 'exec: "aiperf": executable file not found in $PATH')),
            "client_missing",
            "aiperf is not installed",
        ),
        (
            pod_commands(exec_=(1, 'pods "x" is forbidden: cannot create resource "pods/exec"')),
            "client_missing",
            "not allowed to run commands",
        ),
        (pod_commands(exec_=(2, "boom: secret detail")), "client_missing", "did not run"),
    ],
)
def test_check_says_what_is_wrong_with_the_client_pod(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    commands: Any,
    code: str,
    words: str,
) -> None:
    monkeypatch.setattr(nb_common, "run_command", commands)
    monkeypatch.setattr(shutil, "which", lambda name: f"/usr/bin/{name}")
    exit_code, checked = invoke(
        ["check", "--endpoint", ENDPOINT, "--model", MODEL, *CLIENT], capsys
    )
    assert exit_code == 3 and not checked["ok"]
    assert checked["errors"][0]["code"] == code
    message = checked["errors"][0]["message"]
    assert words in message and "nim/bench-client" in message
    assert "secret detail" not in json.dumps(checked) and "cannot create resource" not in message
    assert checked["facts"]["aiperf"] is False and checked["facts"]["served_models"] == []


def test_a_key_the_pod_rejects_is_never_printed(monkeypatch: pytest.MonkeyPatch) -> None:
    """http.client quotes an invalid header value, the key, in its ValueError."""

    def exec_in_pod(argv: list[str], timeout: float, **_kwargs: Any) -> Any:
        return subprocess.run(
            [sys.executable, *argv[PYTHON_ARGS_INDEX:]],
            timeout=timeout,
            capture_output=True,
            text=True,
            env={"POD_KEY": "opaque:pod:credential\n"},
        )

    monkeypatch.setattr(nb_common, "run_command", exec_in_pod)
    _names, errors = nb_common.check_endpoint(
        "http://127.0.0.1:9", MODEL, "POD_KEY", pod=("ns", "client")
    )
    assert errors[0]["code"] == "endpoint_unreachable"
    assert "ValueError" in errors[0]["message"]
    assert "opaque" not in json.dumps(errors)


def serve(handler: type[BaseHTTPRequestHandler]) -> tuple[HTTPServer, threading.Thread]:
    server = HTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


@pytest.mark.parametrize("through_pod", [False, True])
def test_the_key_is_not_sent_to_a_redirect_target(
    monkeypatch: pytest.MonkeyPatch, through_pod: bool
) -> None:
    seen: dict[str, str | None] = {}

    class Target(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            seen["target"] = self.headers.get("Authorization")
            self.send_response(200)
            self.end_headers()
            self.wfile.write(json.dumps({"data": [{"id": MODEL}]}).encode())

        def log_message(self, *_args: Any) -> None:
            pass

    target, target_thread = serve(Target)

    class Redirect(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            seen["first"] = self.headers.get("Authorization")
            self.send_response(302)  # another origin: a different port
            self.send_header("Location", f"http://127.0.0.1:{target.server_port}/v1/models")
            self.end_headers()

        def log_message(self, *_args: Any) -> None:
            pass

    first, first_thread = serve(Redirect)

    def exec_in_pod(argv: list[str], timeout: float, **_kwargs: Any) -> Any:
        return subprocess.run(
            [sys.executable, *argv[PYTHON_ARGS_INDEX:]],
            timeout=timeout,
            capture_output=True,
            text=True,
            env={"KEY": "the-key"},
        )

    monkeypatch.setenv("KEY", "the-key")
    if through_pod:
        monkeypatch.setattr(nb_common, "run_command", exec_in_pod)
    try:
        names = nb_common.served_models(
            f"http://127.0.0.1:{first.server_port}",
            "KEY",
            pod=("ns", "client") if through_pod else None,
        )
    finally:
        for server, thread in ((first, first_thread), (target, target_thread)):
            server.shutdown()
            thread.join()
    assert names == [MODEL]
    assert seen == {"first": "Bearer the-key", "target": None}
