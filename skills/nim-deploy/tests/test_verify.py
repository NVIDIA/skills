# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for scripts/verify.py against a local fake NIM endpoint, plus log triage."""

from __future__ import annotations

import json
import os
import shutil
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from secrets import token_hex
from typing import Any

import nd_common as nd
import pytest
import verify
from _testkit import (
    DOCKER_VALUES,
    K8S_VALUES,
    assert_contract,
    fake_tool,
    make_run,
    run_cli,
)
from _testkit import runs_home as runs_home

pytestmark = pytest.mark.usefixtures("runs_home")

MODELS = {"data": [{"id": "meta/llama-3.1-8b-instruct"}]}
CHAT = {"choices": [{"finish_reason": "length", "message": {"content": "OK"}}]}


@contextmanager
def endpoint(routes: dict[str, Any]) -> Iterator[str]:
    """A local server; each path answers (status, JSON body) in order, then repeats the last."""
    calls: dict[str, int] = {}

    class Handler(BaseHTTPRequestHandler):
        def respond(self) -> None:
            self.rfile.read(int(self.headers.get("Content-Length") or 0))
            answers = routes.get(self.path)
            if answers is None:
                status, body = 404, {"error": "not found"}
            else:
                index = calls.get(self.path, 0)
                calls[self.path] = index + 1
                status, body = answers[min(index, len(answers) - 1)]
            data = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        do_GET = do_POST = respond

        def log_message(self, *args: Any) -> None:
            return

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()


def run(capsys: pytest.CaptureFixture[str], *argv: str) -> tuple[int, dict[str, Any]]:
    code = 0
    try:
        verify.main(list(argv))
    except SystemExit as exit_info:
        code = int(exit_info.code or 0)
    payload = json.loads(capsys.readouterr().out)
    assert_contract(payload)
    return code, payload


@pytest.mark.parametrize("kind", ["file", "ftp-with-userinfo"])
def test_get_json_checks_only_http_and_https(monkeypatch: pytest.MonkeyPatch, kind: str) -> None:
    password = token_hex(8)  # generated, so the source holds no credential-like value
    url = {"file": "file:///etc/hostname", "ftp-with-userinfo": f"ftp://u:{password}@x/v1/models"}[
        kind
    ]

    def never(*_a: Any, **_k: Any) -> Any:
        raise AssertionError("urlopen must not be called for a non-http(s) URL")

    monkeypatch.setattr(verify.urllib.request, "urlopen", never)
    status, body, detail = verify.get_json(url)
    assert (status, body) == (0, None)
    assert "only http and https" in detail
    assert password not in detail and url not in detail


@pytest.mark.parametrize("host", ["[::1", "exa\uff0fmple.com"])  # unclosed bracket, U+FF0F
def test_get_json_reports_a_malformed_url_without_its_credentials(host: str) -> None:
    password = token_hex(8)  # generated, so the source holds no credential-like value
    status, body, detail = verify.get_json(f"http://user:{password}@{host}/v1/models")
    assert (status, body) == (0, None)
    assert detail == "endpoint URL could not be parsed"
    assert password not in detail


def test_ready_waits_through_503_then_reads_models(capsys: pytest.CaptureFixture[str]) -> None:
    routes = {
        "/v1/health/ready": [(503, {}), (503, {}), (200, {"status": "ready"})],
        "/v1/models": [(200, MODELS)],
    }
    with endpoint(routes) as url:
        code, payload = run(capsys, "ready", "--url", url, "--interval", "0.01", "--timeout", "5")
    assert code == 0 and payload["next"] == "verify"
    assert payload["facts"]["polls"] == 3
    assert payload["facts"]["models"] == ["meta/llama-3.1-8b-instruct"]
    assert payload["facts"]["health_path"] == "/v1/health/ready"


def test_ready_falls_back_to_health_on_404(capsys: pytest.CaptureFixture[str]) -> None:
    routes = {"/health": [(200, {"status": "healthy"})], "/v1/models": [(200, MODELS)]}
    with endpoint(routes) as url:
        code, payload = run(capsys, "ready", "--url", url, "--interval", "0.01", "--timeout", "5")
    assert (code, payload["facts"]["health_path"]) == (0, "/health")


def test_ready_with_an_explicit_health_path(capsys: pytest.CaptureFixture[str]) -> None:
    with endpoint({"/custom": [(200, {})], "/v1/models": [(200, MODELS)]}) as url:
        args = ("ready", "--url", url, "--health-path", "custom", "--interval", "0.01")
        code, payload = run(capsys, *args)
    assert (code, payload["facts"]["health_path"]) == (0, "/custom")


def test_ready_needs_models_not_just_health(capsys: pytest.CaptureFixture[str]) -> None:
    routes = {"/v1/health/ready": [(200, {})], "/v1/models": [(200, {"data": []})]}
    with endpoint(routes) as url:
        args = ("ready", "--url", url, "--interval", "0.01", "--timeout", "0.1")
        code, payload = run(capsys, *args)
    assert code == 4 and payload["facts"]["ready"] is False
    assert "/v1/models" in payload["facts"]["last"]


def test_ready_timeout_moves_the_run_to_diagnose(capsys: pytest.CaptureFixture[str]) -> None:
    make_run(K8S_VALUES, state="apply")
    with endpoint({"/v1/health/ready": [(503, {})]}) as url:
        args = ("ready", "--url", url, "--interval", "0.01", "--timeout", "0.1", "--run", "run1")
        code, payload = run(capsys, *args)
    assert (code, payload["state"], payload["next"]) == (4, "diagnose", "diagnose")
    assert payload["errors"][0]["code"] == "not_ready"
    assert "triage" in payload["facts"]["next_command"]
    assert nd.load_state("run1")["state"] == "diagnose"


def test_ready_success_advances_wait_ready_then_verify(capsys: pytest.CaptureFixture[str]) -> None:
    make_run(K8S_VALUES, state="apply")
    routes = {"/v1/health/ready": [(200, {})], "/v1/models": [(200, MODELS)]}
    with endpoint(routes) as url:
        code, payload = run(capsys, "ready", "--url", url, "--run", "run1")
    assert (code, payload["state"]) == (0, "verify")
    assert nd.load_state("run1")["verify"]["ready"]["ready"] is True


def test_ready_unreachable_endpoint_times_out(capsys: pytest.CaptureFixture[str]) -> None:
    args = ("ready", "--url", "http://127.0.0.1:9", "--interval", "0.01", "--timeout", "0.05")
    code, payload = run(capsys, *args)
    assert code == 4 and payload["errors"][0]["code"] == "not_ready"


def test_smoke_passes_and_points_at_record(capsys: pytest.CaptureFixture[str]) -> None:
    make_run({**K8S_VALUES, "model": "meta/llama-3.1-8b-instruct"}, state="verify")
    routes = {"/v1/models": [(200, MODELS)], "/v1/chat/completions": [(200, CHAT)]}
    with endpoint(routes) as url:
        code, payload = run(capsys, "smoke", "--url", url, "--run", "run1")
    assert (code, payload["next"]) == (0, "record")
    assert payload["facts"]["model"] == "meta/llama-3.1-8b-instruct"
    assert "record run1 --result ok" in payload["facts"]["next_command"]


def test_smoke_uses_the_single_served_id_when_the_name_is_not_served(
    capsys: pytest.CaptureFixture[str],
) -> None:
    routes = {"/v1/models": [(200, MODELS)], "/v1/chat/completions": [(200, CHAT)]}
    with endpoint(routes) as url:
        code, payload = run(capsys, "smoke", "--url", url, "--model", "Other/Name")
    assert (code, payload["facts"]["model"]) == (0, "meta/llama-3.1-8b-instruct")


def test_smoke_with_several_served_ids_asks_the_user(capsys: pytest.CaptureFixture[str]) -> None:
    models = {"data": [{"id": "a"}, {"id": "b"}]}
    with endpoint({"/v1/models": [(200, models)]}) as url:
        code, payload = run(capsys, "smoke", "--url", url, "--model", "c")
    assert (code, payload["next"], payload["errors"][0]["code"]) == (
        3,
        "ask_user",
        "model_ambiguous",
    )
    assert payload["facts"]["served_models"] == ["a", "b"]


def test_smoke_failure_moves_to_diagnose(capsys: pytest.CaptureFixture[str]) -> None:
    make_run({**K8S_VALUES, "model": "meta/llama-3.1-8b-instruct"}, state="verify")
    routes = {"/v1/models": [(200, MODELS)], "/v1/chat/completions": [(500, {"error": "boom"})]}
    with endpoint(routes) as url:
        code, payload = run(capsys, "smoke", "--url", url, "--run", "run1")
    assert (code, payload["state"], payload["errors"][0]["code"]) == (4, "diagnose", "smoke_failed")


# --- triage ---------------------------------------------------------------------------------

SAMPLES = {
    "cuda-oom": "torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB",
    "ngc-401": "requests.exceptions.HTTPError: 401 Unauthorized for url https://api.ngc.nvidia.com",
    "no-gpu": 'docker: could not select device driver "" with capabilities: [[gpu]]',
    "profile-not-found": "ERROR: No compatible profile found for this system",
    "port-in-use": "docker: Error: Bind for 0.0.0.0:8000 failed: port is already allocated",
    "hf-token": "huggingface_hub.errors.GatedRepoError: Cannot access gated repo",
    "image-pull-backoff": "Back-off pulling image: ImagePullBackOff",
    "insufficient-gpu": "0/3 nodes are available: 3 Insufficient nvidia.com/gpu.",
    "shm-too-small": "RuntimeError: unable to allocate shared memory (shm)  Bus error",
}


def test_every_triage_rule_has_a_sample_that_matches_it(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rules = json.loads(verify.TRIAGE_RULES.read_text())
    assert {rule["id"] for rule in rules} == set(SAMPLES)
    for rule_id, line in SAMPLES.items():
        log = tmp_path / f"{rule_id}.log"
        log.write_text(f"INFO starting\n{line}\nINFO done\n")
        code, payload = run(capsys, "triage", "--log", str(log))
        assert code == 0 and payload["next"] == "ask_user"
        assert rule_id in [m["id"] for m in payload["facts"]["matches"]], rule_id
        match = next(m for m in payload["facts"]["matches"] if m["id"] == rule_id)
        assert match["fix"] and match["meaning"]


def test_triage_without_a_match_shows_the_log_tail(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    log = tmp_path / "x.log"
    log.write_text("\n".join(f"line {i}" for i in range(50)))
    _, payload = run(capsys, "triage", "--log", str(log))
    assert payload["facts"]["matches"] == [] and payload["facts"]["tail"][-1] == "line 49"


def test_triage_moves_a_run_to_diagnose(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    make_run(DOCKER_VALUES, target="docker", state="wait_ready")
    log = tmp_path / "x.log"
    log.write_text(SAMPLES["cuda-oom"])
    _, payload = run(capsys, "triage", "--log", str(log), "--run", "run1")
    assert payload["state"] == "diagnose"
    assert nd.load_state("run1")["verify"]["triage"] == {"matches": ["cuda-oom"]}


def put_on_path(monkeypatch: pytest.MonkeyPatch, folder: Path) -> None:
    monkeypatch.setenv("PATH", f"{folder}{os.pathsep}{os.environ['PATH']}")


def test_triage_docker_reads_container_logs(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake_tool(tmp_path / "bin", "docker", 'echo "$@" >&2; echo "CUDA out of memory"')
    put_on_path(monkeypatch, tmp_path / "bin")
    _, payload = run(capsys, "triage", "--docker", "nim-llama")
    assert [m["id"] for m in payload["facts"]["matches"]] == ["cuda-oom"]


def test_triage_k8s_carries_context_and_namespace(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    log = tmp_path / "calls"
    fake_tool(tmp_path / "bin", "kubectl", f'echo "$@" >> {log}; echo ImagePullBackOff')
    put_on_path(monkeypatch, tmp_path / "bin")
    _, payload = run(capsys, "triage", "--k8s", "ns1/pod-a", "--context", "ctx1")
    assert "image-pull-backoff" in [m["id"] for m in payload["facts"]["matches"]]
    calls = log.read_text().splitlines()
    assert len(calls) == 2 and all("--context ctx1 -n ns1" in call for call in calls)


def test_triage_reports_a_failed_docker_logs_instead_of_a_clean_triage(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake_tool(tmp_path / "bin", "docker", 'echo "Error: No such container: $3" >&2; exit 1')
    put_on_path(monkeypatch, tmp_path / "bin")
    code, payload = run(capsys, "triage", "--docker", "gone")
    assert (code, payload["ok"], payload["errors"][0]["code"]) == (4, False, "logs_unavailable")
    assert "No such container" in payload["errors"][0]["message"]


def test_triage_k8s_fails_only_when_no_command_collected_anything(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake_tool(tmp_path / "bin", "kubectl", 'echo "Error from server (NotFound)" >&2; exit 1')
    put_on_path(monkeypatch, tmp_path / "bin")
    code, payload = run(capsys, "triage", "--k8s", "ns1/pod-a", "--context", "ctx1")
    assert (code, payload["errors"][0]["code"]) == (4, "logs_unavailable")
    body = 'if [ "$1" = logs ]; then echo "logs refused" >&2; exit 1; fi; echo ImagePullBackOff'
    fake_tool(tmp_path / "bin", "kubectl", body)
    code, payload = run(capsys, "triage", "--k8s", "ns1/pod-a", "--context", "ctx1")
    assert code == 0
    assert [m["id"] for m in payload["facts"]["matches"]] == ["image-pull-backoff"]


def test_triage_k8s_requires_a_context(capsys: pytest.CaptureFixture[str]) -> None:
    code, payload = run(capsys, "triage", "--k8s", "ns1/pod-a")
    assert (code, payload["errors"][0]["code"]) == (2, "bad_input")


def test_triage_reports_missing_tools_and_files(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(shutil, "which", lambda name: None)
    code, payload = run(capsys, "triage", "--docker", "x")
    assert (code, payload["errors"][0]["code"]) == (4, "logs_unavailable")
    code, payload = run(capsys, "triage", "--log", "/nonexistent/file.log")
    assert (code, payload["errors"][0]["code"]) == (4, "logs_unavailable")


def test_cli_contract_for_usage_errors() -> None:
    code, payload = run_cli("verify.py", "smoke")
    assert (code, payload["errors"][0]["code"]) == (2, "bad_input")


# --- the pinned profile (GET /v1/metadata) -----------------------------------------------------

PINNED = "092ed4213624e774d24cdaf84e3b6222839bab2008a21d3c214ab46626366f90"


def pinned_run() -> None:
    state = make_run({**K8S_VALUES, "model_profile": PINNED}, state="apply")
    state["handoff"] = {"profile": {"id": PINNED, "name": "vllm-fp8-tp1-pp1"}}
    nd.save_state(state)


def ready_with_metadata(
    capsys: pytest.CaptureFixture[str], metadata: list[tuple[int, Any]] | None
) -> tuple[int, dict[str, Any]]:
    routes: dict[str, Any] = {"/v1/health/ready": [(200, {})], "/v1/models": [(200, MODELS)]}
    if metadata is not None:
        routes["/v1/metadata"] = metadata
    with endpoint(routes) as url:
        return run(capsys, "ready", "--url", url, "--run", "run1")


@pytest.mark.parametrize(
    "body",
    [
        {"modelInfo": [{"shortName": "llama"}], "activeProfile": {"id": PINNED}},
        {"selected_profile": PINNED, "profiles": [{"profile_id": "other"}]},
        {"assetInfo": {"profile_name": "vllm-fp8-tp1-pp1"}},
    ],
)
def test_ready_confirms_the_pinned_profile(
    capsys: pytest.CaptureFixture[str], body: dict[str, Any]
) -> None:
    pinned_run()
    code, payload = ready_with_metadata(capsys, [(200, body)])
    assert (code, payload["state"]) == (0, "verify")
    assert payload["facts"]["profile_check"]["status"] == "confirmed"
    assert payload["facts"]["warnings"] == []


def test_ready_fails_when_the_server_runs_another_profile(
    capsys: pytest.CaptureFixture[str],
) -> None:
    pinned_run()
    code, payload = ready_with_metadata(capsys, [(200, {"profile": {"id": "f" * 64}})])
    assert (code, payload["state"], payload["next"]) == (4, "diagnose", "diagnose")
    error = payload["errors"][0]
    assert error["code"] == "profile_mismatch"
    assert PINNED in error["message"] and "f" * 64 in error["message"]
    assert "Never drop model_profile" in error["fix"]
    assert nd.load_state("run1")["state"] == "diagnose"


@pytest.mark.parametrize("list_key", ["profiles", "available_profiles", "supportedProfiles"])
def test_ready_does_not_confirm_from_the_list_of_available_profiles(
    capsys: pytest.CaptureFixture[str], list_key: str
) -> None:
    pinned_run()
    other = "e" * 64
    body = {"activeProfile": other, list_key: [{"profile_id": PINNED}]}
    code, payload = ready_with_metadata(capsys, [(200, body)])
    check = payload["facts"]["profile_check"]
    assert (code, check["status"], check["found"]) == (4, "mismatch", [other])
    assert payload["errors"][0]["code"] == "profile_mismatch"


def test_ready_does_not_confirm_by_name_when_another_id_is_reported(
    capsys: pytest.CaptureFixture[str],
) -> None:
    pinned_run()
    body = {"profile": {"id": "f" * 64, "name": "vllm-fp8-tp1-pp1"}}  # same coarse name, other id
    code, payload = ready_with_metadata(capsys, [(200, body)])
    assert (code, payload["state"]) == (4, "diagnose")
    assert payload["errors"][0]["code"] == "profile_mismatch"


def test_ready_confirms_by_name_only_when_no_id_is_reported(
    capsys: pytest.CaptureFixture[str],
) -> None:
    pinned_run()
    code, payload = ready_with_metadata(capsys, [(200, {"profile_name": "vllm-fp8-tp1-pp1"})])
    check = payload["facts"]["profile_check"]
    assert (code, check["status"]) == (0, "confirmed")
    assert "by name only" in check["detail"]


@pytest.mark.parametrize("handoff_name", [None, "vllm-bf16-tp1-pp1"])
def test_a_name_only_report_without_a_matching_name_is_not_confirmed(
    capsys: pytest.CaptureFixture[str], handoff_name: str | None
) -> None:
    pinned_run()
    state = nd.load_state("run1")
    if handoff_name is None:
        del state["handoff"]  # model_profile typed in with `set`, no handoff
    else:
        state["handoff"]["profile"]["name"] = handoff_name
    nd.save_state(state)
    code, payload = ready_with_metadata(capsys, [(200, {"profile_name": "vllm-fp8-tp1-pp1"})])
    check = payload["facts"]["profile_check"]
    assert (code, payload["state"], check["status"]) == (0, "verify", "not_confirmed")
    assert "cannot match the pinned id" in check["detail"]


@pytest.mark.parametrize(
    ("metadata", "detail"),
    [
        (None, "GET /v1/metadata -> 404"),
        ([(200, {"modelInfo": [{"shortName": "llama"}], "profiles": [PINNED]})], "no profile"),
    ],
)
def test_an_unconfirmed_profile_is_a_visible_warning(
    capsys: pytest.CaptureFixture[str], metadata: Any, detail: str
) -> None:
    pinned_run()
    code, payload = ready_with_metadata(capsys, metadata)
    assert (code, payload["state"]) == (0, "verify")
    check = payload["facts"]["profile_check"]
    assert check["status"] == "not_confirmed" and detail in check["detail"]
    assert payload["facts"]["warnings"][0].startswith("PROFILE NOT CONFIRMED")
    routes = {"/v1/models": [(200, MODELS)], "/v1/chat/completions": [(200, CHAT)]}
    with endpoint(routes) as url:  # the smoke result repeats it for the final report
        code, payload = run(capsys, "smoke", "--url", url, "--run", "run1")
    assert code == 0 and payload["facts"]["warnings"][0].startswith("PROFILE NOT CONFIRMED")


def test_a_run_without_a_pinned_profile_skips_the_check(
    capsys: pytest.CaptureFixture[str],
) -> None:
    make_run(K8S_VALUES, state="apply")
    code, payload = ready_with_metadata(capsys, None)
    assert code == 0 and "profile_check" not in payload["facts"]
