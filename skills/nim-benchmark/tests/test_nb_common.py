# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for scripts/nb_common.py: plan ids, secrets, endpoints and the output contract."""

from __future__ import annotations

import json
import subprocess
import sys
import time
import urllib.error
from pathlib import Path
from secrets import token_hex
from typing import Any

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import nb_common  # noqa: E402


def test_plan_id_is_stable_and_changes_with_any_field() -> None:
    plan = {"mode": "perf", "loads": [8, 32], "endpoint": "http://a:8000"}
    assert nb_common.plan_id(plan) == nb_common.plan_id(dict(reversed(list(plan.items()))))
    assert nb_common.plan_id(plan) != nb_common.plan_id({**plan, "loads": [8, 33]})
    assert len(nb_common.plan_id(plan)) == 12


def test_scrubber_masks_token_shapes_and_registered_values() -> None:
    scrubber = nb_common.Scrubber()
    scrubber.add("a-very-private-value")
    text = "token hf_abcdefghijklmnopqrstuvwx and a-very-private-value and --api-key abcdef123456"
    out = scrubber.scrub(text)
    assert "hf_abcdefghijklmnopqrstuvwx" not in out
    assert "a-very-private-value" not in out
    assert "abcdef123456" not in out
    assert out.count(nb_common.MASK) == 3


def test_scrubber_reads_credential_named_variables(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MY_SECRET_THING", "supersecretvalue1")
    assert "supersecretvalue1" not in nb_common.Scrubber().scrub("x supersecretvalue1 y")


def test_scrubber_masks_the_value_of_a_named_key_variable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NIM_EP_BEARER", "bearer-value-not-matched-by-name")
    scrubber = nb_common.Scrubber()
    assert "bearer-value" in scrubber.scrub("bearer-value-not-matched-by-name")
    scrubber.add_env("NIM_EP_BEARER")
    scrubber.add_env("NOT_SET_ANYWHERE")
    scrubber.add_env(None)
    assert scrubber.scrub("x bearer-value-not-matched-by-name y") == f"x {nb_common.MASK} y"


def test_get_json_registers_the_named_key_with_the_scrubber(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NIM_EP_BEARER", "bearer-from-get-json-1")
    monkeypatch.setattr(nb_common, "SCRUBBER", nb_common.Scrubber())

    def refuse(*_a: Any, **_k: Any) -> Any:
        raise urllib.error.URLError("no route")

    monkeypatch.setattr(nb_common.urllib.request, "urlopen", refuse)
    with pytest.raises(urllib.error.URLError):
        nb_common.get_json("http://x/v1/models", "NIM_EP_BEARER")
    assert "bearer-from-get-json-1" not in nb_common.SCRUBBER.scrub("bearer-from-get-json-1")


@pytest.mark.parametrize("pod", [None, ("ns", "pod-0")])
@pytest.mark.parametrize("kind", ["file", "ftp-with-userinfo", "no-scheme"])
def test_get_json_reads_only_http_and_https(
    monkeypatch: pytest.MonkeyPatch, kind: str, pod: tuple[str, str] | None
) -> None:
    password = token_hex(8)  # generated, so the source holds no credential-like value
    url = {
        "file": "file:///etc/hostname",
        "ftp-with-userinfo": f"ftp://user:{password}@x/v1/models",
        "no-scheme": "x/v1/models",
    }[kind]

    def never(*_a: Any, **_k: Any) -> Any:
        raise AssertionError("nothing may run for a non-http(s) URL, in a pod or locally")

    monkeypatch.setattr(nb_common.urllib.request, "urlopen", never)
    monkeypatch.setattr(nb_common, "run_command", never)
    with pytest.raises(ValueError, match="only http and https") as raised:
        nb_common.get_json(url, pod=pod)
    assert password not in str(raised.value) and url not in str(raised.value)


URL_FORMS = {  # each puts the generated password {pw} in the userinfo of a different URL form
    "plain": "http://user:{pw}@host:8000",
    "uppercase-scheme": "HTTP://user:{pw}@host:8000",
    "space-in-userinfo": "http://user:{pw} tail@host:8000",
    "raw-at-in-password": "http://user:{pw}@x@host:8000",
    "leading-space": "  http://user:{pw}@host:8000",
    "other-scheme": "ftp://user:{pw}@host:8000",
    "scheme-relative": "//user:{pw}@host:8000",
    "no-scheme": "user:{pw}@host:8000",
}


@pytest.mark.parametrize("form", URL_FORMS)
def test_strip_userinfo_drops_the_credentials_of_every_url_form(form: str) -> None:
    password = token_hex(8)
    stripped = nb_common.strip_userinfo(URL_FORMS[form].format(pw=password))
    assert password not in stripped and stripped.endswith("host:8000")
    assert nb_common.strip_userinfo("http://host:8000/a@b") == "http://host:8000/a@b"


@pytest.mark.parametrize("pod", [None, ("ns", "pod-0")])
@pytest.mark.parametrize("form", URL_FORMS)
def test_check_endpoint_diagnostics_never_carry_url_credentials(
    monkeypatch: pytest.MonkeyPatch, pod: tuple[str, str] | None, form: str
) -> None:
    password = token_hex(8)  # generated, so the source holds no credential-like value
    base = URL_FORMS[form].format(pw=password)

    def never(*_a: Any, **_k: Any) -> Any:
        raise AssertionError("nothing may run for a URL that carries credentials")

    monkeypatch.setattr(nb_common.urllib.request, "urlopen", never)
    monkeypatch.setattr(nb_common, "run_command", never)
    _names, found = nb_common.check_endpoint(base, "m", pod=pod)  # get_json refuses the URL
    assert found and password not in json.dumps(found) and "host:8000" in json.dumps(found)

    def pod_failure(*_a: Any, **_k: Any) -> Any:
        raise nb_common.PodProbeError("connection refused")

    monkeypatch.setattr(nb_common, "served_models", pod_failure)
    _names, found = nb_common.check_endpoint(base, "m", pod=("ns", "pod-0"))
    assert found and password not in json.dumps(found) and "host:8000" in json.dumps(found)


@pytest.mark.parametrize("pod", [None, ("ns", "pod-0")])
@pytest.mark.parametrize("host", ["[::1", "exa\uff0fmple.com"])  # unclosed bracket, U+FF0F
def test_get_json_reports_an_unparsable_url_without_its_credentials(
    monkeypatch: pytest.MonkeyPatch, host: str, pod: tuple[str, str] | None
) -> None:
    password = token_hex(8)  # generated, so the source holds no credential-like value

    def never(*_a: Any, **_k: Any) -> Any:
        raise AssertionError("nothing may run for a URL that cannot be parsed")

    monkeypatch.setattr(nb_common.urllib.request, "urlopen", never)
    monkeypatch.setattr(nb_common, "run_command", never)
    with pytest.raises(ValueError, match="could not be parsed") as raised:
        nb_common.get_json(f"http://user:{password}@{host}/v1/models", pod=pod)
    assert str(raised.value) == "endpoint URL could not be parsed"
    assert password not in str(raised.value) and raised.value.__suppress_context__


@pytest.mark.parametrize("pod", [None, ("ns", "pod-0")])
def test_get_json_refuses_credentials_in_the_url(
    monkeypatch: pytest.MonkeyPatch, pod: tuple[str, str] | None
) -> None:
    password = token_hex(8)  # generated, so the source holds no credential-like value

    def never(*_a: Any, **_k: Any) -> Any:
        raise AssertionError("nothing may run for a URL that carries credentials")

    monkeypatch.setattr(nb_common.urllib.request, "urlopen", never)
    monkeypatch.setattr(nb_common, "run_command", never)
    with pytest.raises(ValueError, match="must not contain credentials") as raised:
        nb_common.get_json(f"http://user:{password}@host:8000/v1/models", pod=pod)
    assert password not in str(raised.value)


def is_running(pid: int) -> bool:
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return False
    return stat.rsplit(")", 1)[1].split()[0] != "Z"


@pytest.mark.skipif(not Path("/proc").is_dir(), reason="needs /proc")
def test_a_timeout_kills_what_the_program_started(tmp_path: Path) -> None:
    pid_file = tmp_path / "pid"
    script = f"sleep 60 & echo $! > {pid_file}; wait"
    with pytest.raises(subprocess.TimeoutExpired):
        nb_common.run_command(["sh", "-c", script], timeout=1)
    pid = int(pid_file.read_text())
    deadline = time.monotonic() + 5
    while is_running(pid) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not is_running(pid)


def test_run_command_returns_the_output_and_the_exit_code() -> None:
    done = nb_common.run_command(["sh", "-c", "echo out; echo err >&2; exit 3"], timeout=30)
    assert (done.returncode, done.stdout, done.stderr) == (3, "out\n", "err\n")


@pytest.mark.parametrize("returncode", [0, 3])
@pytest.mark.parametrize(
    ("output", "expected"),
    [
        (b"progress \xe2", "progress \ufffd"),
        (b"progress \xe2\x94", "progress \ufffd"),
        (b"progress \xff", "progress \ufffd"),
        (b"progress \xe2\x94\x80", "progress \u2500"),
    ],
    ids=["truncated-one-byte", "truncated-two-bytes", "invalid-byte", "valid-utf8"],
)
def test_run_command_decodes_console_output_without_losing_exit_code(
    output: bytes, expected: str, returncode: int
) -> None:
    script = (
        f"import os; os.write(1, {output!r}); os.write(2, {output!r}); "
        f"raise SystemExit({returncode})"
    )
    done = nb_common.run_command([sys.executable, "-c", script], timeout=30)
    assert (done.returncode, done.stdout, done.stderr) == (returncode, expected, expected)


def test_write_json_scrubs_before_writing(tmp_path: Path) -> None:
    nb_common.SCRUBBER.add("write-me-not-1234")
    nb_common.write_json(tmp_path / "a" / "f.json", {"k": "write-me-not-1234"})
    assert "write-me-not-1234" not in (tmp_path / "a" / "f.json").read_text()


def test_split_endpoint_gives_both_forms_and_drops_userinfo() -> None:
    assert nb_common.split_endpoint("http://u:p@host:8000/v1/") == (
        "http://host:8000",
        "http://host:8000/v1",
    )
    assert nb_common.split_endpoint("https://host") == ("https://host", "https://host/v1")


def test_check_endpoint_reports_unreachable_and_missing_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def refuse(*_a: Any, **_k: Any) -> Any:
        raise urllib.error.URLError("no route")

    monkeypatch.setattr(nb_common, "get_json", refuse)
    names, errors = nb_common.check_endpoint("http://x", "m")
    assert names == [] and errors[0]["code"] == "endpoint_unreachable"

    monkeypatch.setattr(nb_common, "get_json", lambda *_a, **_k: {"data": [{"id": "other"}]})
    names, errors = nb_common.check_endpoint("http://x", "m")
    assert names == ["other"] and errors[0]["code"] == "model_not_served"
    assert "other" in errors[0]["message"]

    monkeypatch.setattr(nb_common, "get_json", lambda *_a, **_k: {"data": [{"id": "m"}]})
    assert nb_common.check_endpoint("http://x", "m") == (["m"], [])


def test_script_prints_one_contract_object() -> None:
    proc = subprocess.run(
        [sys.executable, str(SCRIPTS / "nb_common.py"), "--version"],
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        timeout=60,
        check=False,
    )
    assert proc.returncode == 0
    assert set(json.loads(proc.stdout)) == {"ok", "state", "next", "facts", "errors"}


def test_bad_flag_exits_2_with_a_contract_object() -> None:
    proc = subprocess.run(
        [sys.executable, str(SCRIPTS / "nb_common.py"), "--nope"],
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        timeout=60,
        check=False,
    )
    assert proc.returncode == 2
    assert json.loads(proc.stdout)["errors"][0]["code"] == "bad_input"


@pytest.mark.parametrize("version", ["0.12.0", "0.13.0"])
def test_client_pod_requires_the_pinned_aiperf_version(
    monkeypatch: pytest.MonkeyPatch, version: str
) -> None:
    """A running tool with a different version cannot satisfy the benchmark plan."""

    def command(argv: list[str], **kwargs: Any) -> Any:
        output = "Running" if argv[:3] == ["kubectl", "get", "pod"] else f"AIPerf {version}"
        return subprocess.CompletedProcess(argv, 0, output, "")

    monkeypatch.setattr(nb_common, "run_command", command)
    ready, errors = nb_common.check_client_pod("default", "bench-client")
    assert ready == (version == "0.13.0")
    assert bool(errors) == (version != "0.13.0")
