# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for scripts/nsc_common.py: contract, keys, URLs, TLS, requests and private files."""

from __future__ import annotations

import contextlib
import errno
import json
import os
import re
import socket
import ssl
import subprocess
import sys
import threading
import time
import urllib.request
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import nsc_common as nsc
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
CONTRACT_KEYS = {"ok", "state", "next", "facts", "errors"}
PROXY_VARIABLES = (
    "http_proxy",
    "HTTP_PROXY",
    "https_proxy",
    "HTTPS_PROXY",
    "all_proxy",
    "ALL_PROXY",
    "no_proxy",
    "NO_PROXY",
)

# Fake credentials, built at run time so that no secret-looking literal sits in the repository.
NVAPI_KEY = "nvapi" + "-" + "q" * 30
HF_KEY = "hf" + "_" + "a" * 30
GLPAT_KEY = "glpat" + "-" + "b" * 20
GHP_KEY = "ghp" + "_" + "c" * 36
SK_KEY = "sk" + "-" + "d" * 30
AKIA_KEY = "AKIA" + "E" * 16
RANDOM_KEY = "Zx9" * 8  # a key with no well-known shape
GATEWAY_KEY = "gw" + "-" + "Zx9" * 8  # the same, as a gateway token with a prefix
# Keys whose short prefix ends in "_", so they also have the shape of a variable name.
PREFIXED_KEYS = (
    "gsk" + "_" + "Ab3" * 17 + "c",  # a 52-character body, as Groq keys have
    "r8" + "_" + "Qw7" * 12 + "e",  # 37 characters, as Replicate tokens have
    "sk" + "_live_" + "Zx9" * 8,  # a live secret key with two underscores
    "gw" + "_prod_" + "Kp4" * 13 + "a",  # a gateway's prefix and a 40-character body
)
PASSWORD = "pw" + "-" + "Qz7" * 6
# A password with a quote, a backslash and a letter that is not ASCII: JSON writes each escaped.
ESCAPED_PASSWORD = "pw" + '"' + "\\" + "ö" + "Qz7" * 4
# A value of no known shape, given to a credential flag in the scrubber tests.
FLAG_SECRET = "Qz7" + "-plain" * 2

# --- helpers ------------------------------------------------------------------------------------


def assert_contract(payload: dict[str, Any]) -> None:
    """The object follows the output contract {ok, state, next, facts, errors}."""
    assert set(payload) == CONTRACT_KEYS
    assert isinstance(payload["ok"], bool)
    assert isinstance(payload["state"], str) and payload["state"]
    assert isinstance(payload["next"], str) and payload["next"]
    assert isinstance(payload["facts"], dict)
    for item in payload["errors"]:
        assert set(item) == {"code", "message", "fix"}
        assert item["code"].replace("_", "").isalnum() and item["code"].islower()
        assert item["message"] and item["fix"]


def one_object(out: str) -> dict[str, Any]:
    lines = [line for line in out.splitlines() if line.strip()]
    assert len(lines) == 1, out
    payload = json.loads(lines[0])
    assert_contract(payload)
    return payload


def run_module(*args: str) -> subprocess.CompletedProcess[str]:
    """Run scripts/nsc_common.py as SKILL.md starts the scripts (python3 -I)."""
    return subprocess.run(
        [sys.executable, "-I", "-B", str(SCRIPTS / "nsc_common.py"), *args],
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        timeout=60,
        check=False,
    )


def closed_url() -> str:
    """An http URL on 127.0.0.1 where nothing listens."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    return f"http://127.0.0.1:{port}"


@dataclass
class Local:
    """A running local server: its base URL, the requests it got and an event set at the end."""

    url: str
    requests: list[dict[str, Any]]
    stop: threading.Event


Respond = Callable[[BaseHTTPRequestHandler, Local], None]


class QuietServer(ThreadingHTTPServer):
    daemon_threads = True

    def handle_error(self, request: Any, client_address: Any) -> None:
        return  # a client that left early (the tests cut streams on purpose)


@contextlib.contextmanager
def serve(respond: Respond, *, http11: bool = False) -> Iterator[Local]:
    """Serve every GET and POST on 127.0.0.1 with `respond`, recording each request."""
    requests: list[dict[str, Any]] = []
    stop = threading.Event()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1" if http11 else "HTTP/1.0"

        def handle_any(self) -> None:
            length = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(length) if length else b""
            headers = {name.lower(): value for name, value in self.headers.items()}
            requests.append(
                {"method": self.command, "path": self.path, "headers": headers, "body": body}
            )
            respond(self, local)

        do_GET = do_POST = handle_any

        def log_message(self, format: str, *args: Any) -> None:
            return

    server = QuietServer(("127.0.0.1", 0), Handler)
    local = Local(f"http://127.0.0.1:{server.server_address[1]}", requests, stop)
    thread = threading.Thread(target=server.serve_forever, args=(0.05,), daemon=True)
    thread.start()
    try:
        yield local
    finally:
        stop.set()
        server.shutdown()
        server.server_close()


def answer(status: int, body: Any = None, headers: dict[str, str] | None = None) -> Respond:
    def respond(handler: BaseHTTPRequestHandler, local: Local) -> None:
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        handler.send_response(status)
        handler.send_header("Content-Type", "application/json")
        for name, value in (headers or {}).items():
            handler.send_header(name, value)
        handler.send_header("Content-Length", str(len(data)))
        handler.end_headers()
        handler.wfile.write(data)

    return respond


def redirect(code: int, location: str) -> Respond:
    def respond(handler: BaseHTTPRequestHandler, local: Local) -> None:
        handler.send_response(code)
        handler.send_header("Location", location)
        handler.send_header("Content-Length", "0")
        handler.end_headers()

    return respond


def events(parts: list[tuple[float, bytes]], *, chunked: bool = False, hold: float = 0) -> Respond:
    """An event stream: each (delay, bytes) part after its delay; then `hold` seconds with the
    connection open; with `chunked`, Transfer-Encoding: chunked (serve with http11=True)."""

    def respond(handler: BaseHTTPRequestHandler, local: Local) -> None:
        handler.send_response(200)
        handler.send_header("Content-Type", "text/event-stream")
        if chunked:
            handler.send_header("Transfer-Encoding", "chunked")
        handler.end_headers()
        for delay, data in parts:
            if delay and local.stop.wait(delay):
                return
            handler.wfile.write(b"%x\r\n%s\r\n" % (len(data), data) if chunked else data)
        if hold and local.stop.wait(hold):
            return
        if chunked:
            handler.wfile.write(b"0\r\n\r\n")

    return respond


@contextlib.contextmanager
def plain_bytes_server(data: bytes) -> Iterator[str]:
    """A TCP server on 127.0.0.1 that reads what the client sends first, answers `data` and
    closes; yields host:port. A TLS client gets bytes that are not TLS."""
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(4)
    listener.settimeout(0.1)
    stop = threading.Event()

    def loop() -> None:
        while not stop.is_set():
            try:
                conn, _ = listener.accept()
            except OSError:
                continue
            with conn, contextlib.suppress(OSError):
                conn.settimeout(2)
                conn.recv(65536)
                conn.sendall(data)
                conn.shutdown(socket.SHUT_WR)
                conn.recv(65536)

    thread = threading.Thread(target=loop, daemon=True)
    thread.start()
    try:
        yield f"127.0.0.1:{listener.getsockname()[1]}"
    finally:
        stop.set()
        thread.join(2)
        listener.close()


@pytest.fixture
def scrubber(monkeypatch: pytest.MonkeyPatch) -> nsc.Scrubber:
    """A fresh SCRUBBER, so that a value one test registers masks nothing in another."""
    fresh = nsc.Scrubber()
    monkeypatch.setattr(nsc, "SCRUBBER", fresh)
    return fresh


@pytest.fixture
def no_proxy_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in PROXY_VARIABLES:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def local_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """A connection to any host but this machine fails at once, with no DNS lookup."""
    real = socket.getaddrinfo

    def guarded(host: Any, *args: Any, **kwargs: Any) -> Any:
        if host not in ("127.0.0.1", "localhost", "::1"):
            raise socket.gaierror(socket.EAI_NONAME, f"the test looks up no host ({host})")
        return real(host, *args, **kwargs)

    monkeypatch.setattr(socket, "getaddrinfo", guarded)


# --- the output contract ------------------------------------------------------------------------


def test_the_exit_codes_and_the_user_agent() -> None:
    codes = (
        nsc.EXIT_OK,
        nsc.EXIT_INPUT,
        nsc.EXIT_PRECONDITION,
        nsc.EXIT_EXTERNAL,
        nsc.EXIT_REFUSED,
    )
    assert codes == (0, 2, 3, 4, 5)
    assert nsc.USER_AGENT == "nim-smoke-check/0.1"


def test_error_and_script_error_have_the_contract_shape() -> None:
    assert nsc.error("no_models", "m", "f") == {"code": "no_models", "message": "m", "fix": "f"}
    plain = nsc.ScriptError("bad_input", "the message", "the fix")
    assert (plain.exit_code, plain.facts, str(plain)) == (2, {}, "the message")
    assert plain.as_error() == nsc.error("bad_input", "the message", "the fix")
    asked = nsc.ScriptError("model_ambiguous", "m", "f", nsc.EXIT_PRECONDITION, {"served": ["a"]})
    assert (asked.code, asked.exit_code, asked.facts) == ("model_ambiguous", 3, {"served": ["a"]})


@pytest.mark.parametrize(
    ("ok", "code", "expected"), [(True, None, 0), (False, None, 2), (False, 3, 3), (False, 4, 4)]
)
def test_emit_prints_one_contract_object_and_exits(
    capsys: pytest.CaptureFixture[str], ok: bool, code: int | None, expected: int
) -> None:
    errors = [] if ok else [nsc.error("checks_failed", "1 check failed", "Show the report")]
    with pytest.raises(SystemExit) as stop:
        nsc.emit(ok, "passed" if ok else "failed", "report", {"n": 1}, errors, code)
    assert stop.value.code == expected
    payload = one_object(capsys.readouterr().out)
    assert (payload["ok"], payload["facts"], payload["errors"]) == (ok, {"n": 1}, errors)


def test_emit_writes_what_json_cannot_hold_as_text(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        nsc.emit(True, "planned", "run", {"out": Path("/x/y"), "when": datetime(2026, 1, 2)})
    assert one_object(capsys.readouterr().out)["facts"] == {
        "out": "/x/y",
        "when": "2026-01-02 00:00:00",
    }


def test_emit_masks_keys_in_facts_and_errors(
    capsys: pytest.CaptureFixture[str], scrubber: nsc.Scrubber
) -> None:
    scrubber.add(GATEWAY_KEY)
    facts = {"seen": f"Authorization: Bearer {NVAPI_KEY}", "raw": GATEWAY_KEY, HF_KEY: 1}
    errors = [nsc.error("http_error", f"the gateway said {GATEWAY_KEY}", f"drop {SK_KEY}")]
    with pytest.raises(SystemExit) as stop:
        nsc.emit(False, "failed", "fix_errors", facts, errors, nsc.EXIT_EXTERNAL)
    assert stop.value.code == 4
    out = capsys.readouterr().out
    for secret in (NVAPI_KEY, GATEWAY_KEY, HF_KEY, SK_KEY):
        assert secret not in out
    assert one_object(out)["facts"]["raw"] == "***"


def test_log_writes_masked_text_to_stderr(
    capsys: pytest.CaptureFixture[str], scrubber: nsc.Scrubber
) -> None:
    nsc.log(f"sending with Bearer {NVAPI_KEY}")
    captured = capsys.readouterr()
    assert captured.out == ""
    assert NVAPI_KEY not in captured.err and "sending with Bearer ***" in captured.err


def test_a_bad_flag_is_a_json_bad_input_error(capsys: pytest.CaptureFixture[str]) -> None:
    parser = nsc.JsonArgumentParser(prog="nimsmoke.py")
    parser.add_argument("--concurrency", type=int)
    with pytest.raises(SystemExit) as stop:
        parser.parse_args(["--concurrency", "many"])
    assert stop.value.code == 2
    payload = one_object(capsys.readouterr().out)
    assert (payload["ok"], payload["state"], payload["next"]) == (False, "input", "ask_user")
    assert payload["errors"][0]["code"] == "bad_input"
    assert "nimsmoke.py --help" in payload["errors"][0]["fix"]


def test_a_subcommand_reports_a_bad_flag_the_same_way(capsys: pytest.CaptureFixture[str]) -> None:
    parser = nsc.JsonArgumentParser(prog="nimsmoke.py")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("run").add_argument("--timeout", type=float)
    with pytest.raises(SystemExit) as stop:
        parser.parse_args(["run", "--timeout", "soon"])
    assert stop.value.code == 2
    assert one_object(capsys.readouterr().out)["errors"][0]["code"] == "bad_input"


def test_a_flag_counts_only_when_written_in_full(
    capsys: pytest.CaptureFixture[str], scrubber: nsc.Scrubber
) -> None:
    """argparse takes a unique prefix of a flag by default, so --api-key=KEY would quietly be
    --api-key-env KEY. The parser and its subparsers take a flag only as written in full."""
    parser = nsc.JsonArgumentParser(prog="nimsmoke.py")
    parser.add_argument("--api-key-env", dest="top_env")
    run = parser.add_subparsers(dest="command").add_parser("run")
    run.add_argument("--api-key-env")
    run.add_argument("--concurrency", type=int)
    assert parser.allow_abbrev is False and run.allow_abbrev is False
    for argv in ([f"--api-key={RANDOM_KEY}"], ["--api-key", RANDOM_KEY],
                 ["run", f"--api-key={RANDOM_KEY}"], ["run", "--conc", "3"]):
        with pytest.raises(SystemExit) as stop:
            parser.parse_args(argv)
        assert stop.value.code == 2, argv
        payload = one_object(capsys.readouterr().out)
        assert payload["errors"][0]["code"] == "bad_input"
        # a stray key in the command's place is an invalid choice, not shown either
        assert re.search(r"unrecognized arguments|invalid choice: \(a value that looks like a "
                         r"key", payload["errors"][0]["message"]), payload
        assert RANDOM_KEY not in json.dumps(payload)
    assert parser.parse_args(["run", "--api-key-env", "NAME"]).api_key_env == "NAME"
    assert nsc.JsonArgumentParser(allow_abbrev=True).allow_abbrev is False


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["--api", "VALUE"], "unrecognized arguments: --api (values not shown)"),
        (["--api=VALUE", "-x"], "unrecognized arguments: --api -x (values not shown)"),
        (["VALUE"], "unrecognized arguments: (values not shown)"),
        (["--bogus"], "unrecognized arguments: --bogus"),
        (["rnu"], "invalid choice: 'rnu'"),  # the rest of the text differs by Python version
        (["Zx9VALUE"], "invalid choice: (a value that looks like a key, not shown)"),
    ],
)
def test_an_unknown_flag_is_named_without_its_value(
    capsys: pytest.CaptureFixture[str], argv: list[str], message: str
) -> None:
    """A value given to an unknown flag of no credential name (--api KEY) is not echoed."""
    value = "Qz" + "plain" * 3 + "7"  # of no known shape: the scrubber alone would not mask it
    parser = nsc.JsonArgumentParser(prog="nimsmoke.py")
    if argv[0] in ("rnu", "Zx9VALUE"):
        parser.add_subparsers(dest="command").add_parser("run")
    with pytest.raises(SystemExit):
        parser.parse_args([word.replace("VALUE", value) for word in argv])
    found = one_object(capsys.readouterr().out)["errors"][0]["message"]
    assert (message in found) if "choice" in message else (found == message), found
    assert value not in found


def test_an_unknown_flag_that_carries_a_key_is_not_echoed(
    capsys: pytest.CaptureFixture[str], scrubber: nsc.Scrubber
) -> None:
    parser = nsc.JsonArgumentParser(prog="nimsmoke.py")
    for argv in (["--api-key", NVAPI_KEY], ["--token", RANDOM_KEY], [f"--hf-token={RANDOM_KEY}"]):
        with pytest.raises(SystemExit) as stop:
            parser.parse_args(argv)
        assert stop.value.code == 2
        out = capsys.readouterr().out
        assert "unrecognized arguments" in out
        assert NVAPI_KEY not in out and RANDOM_KEY not in out


def test_the_module_run_directly_prints_a_library_notice() -> None:
    proc = run_module()
    assert proc.returncode == 0, proc.stderr
    payload = one_object(proc.stdout)
    assert (payload["ok"], payload["state"]) == (True, "library")


def test_the_module_reports_a_bad_flag_as_json_and_exits_2() -> None:
    proc = run_module("--bogus")
    assert proc.returncode == 2
    assert one_object(proc.stdout)["errors"][0]["code"] == "bad_input"


def test_the_module_help_exits_0() -> None:
    proc = run_module("--help")
    assert proc.returncode == 0
    assert "usage" in proc.stdout and "import it" in proc.stdout


# --- keys -----------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name", ["NVIDIA_API_KEY", "MY_GATEWAY_TOKEN", "_X", "a", "api_key_2", "A" * 64,
             "SMOKE_TOKEN_2024", "NGC_API_KEY_V2", "LLAMA31_70B_GATEWAY_TOKEN"]
)
def test_a_variable_name_is_accepted(name: str) -> None:
    assert nsc.check_env_name(name) == name


@pytest.mark.parametrize(
    "value",
    [
        NVAPI_KEY,
        HF_KEY,
        GHP_KEY,
        AKIA_KEY,
        SK_KEY,
        GLPAT_KEY,
        RANDOM_KEY,
        "Bearer " + RANDOM_KEY,
        "my key",
        "1ABC",
        "A" * 65,
        "NVIDIA_API_KEY\n",
        "NAME-WITH-DASHES",
        "",
        *PREFIXED_KEYS,
    ],
)
def test_a_key_given_as_the_variable_name_is_refused_and_never_echoed(
    capsys: pytest.CaptureFixture[str], scrubber: nsc.Scrubber, value: str
) -> None:
    with pytest.raises(nsc.ScriptError) as info:
        nsc.check_env_name(value)
    refused = info.value
    assert (refused.code, refused.exit_code) == ("literal_secret", nsc.EXIT_INPUT)
    with pytest.raises(SystemExit) as stop:
        nsc.emit(False, "input", "ask_user", errors=[refused.as_error()], code=refused.exit_code)
    assert stop.value.code == 2
    out = capsys.readouterr().out
    assert one_object(out)["errors"][0]["code"] == "literal_secret"
    if len(value) >= 16:  # a short value such as "1ABC" proves nothing by its absence
        assert value not in out
        assert value not in refused.message + refused.fix + str(refused)


def test_the_refusal_names_the_flag_it_is_about() -> None:
    with pytest.raises(nsc.ScriptError) as info:
        nsc.check_env_name(NVAPI_KEY, flag="--judge-key-env")
    assert "--judge-key-env" in info.value.message and NVAPI_KEY not in info.value.message


def test_read_key_reads_the_named_variable_and_masks_its_value(
    monkeypatch: pytest.MonkeyPatch, scrubber: nsc.Scrubber
) -> None:
    monkeypatch.setenv("NSC_TEST_GATEWAY_TOKEN", f" {GATEWAY_KEY}\n")
    assert nsc.read_key("NSC_TEST_GATEWAY_TOKEN") == GATEWAY_KEY
    assert (
        scrubber.scrub(f"sent {GATEWAY_KEY} and {json.dumps(GATEWAY_KEY)}") == 'sent *** and "***"'
    )


def test_read_key_gives_none_for_an_unset_or_empty_variable(
    monkeypatch: pytest.MonkeyPatch, scrubber: nsc.Scrubber
) -> None:
    monkeypatch.delenv("NSC_TEST_UNSET", raising=False)
    monkeypatch.setenv("NSC_TEST_EMPTY", " \n")
    assert nsc.read_key("NSC_TEST_UNSET") is None
    assert nsc.read_key("NSC_TEST_EMPTY") is None
    assert scrubber.values == frozenset()


def test_read_key_refuses_a_value_a_header_cannot_carry(
    monkeypatch: pytest.MonkeyPatch, scrubber: nsc.Scrubber
) -> None:
    monkeypatch.setenv("NSC_TEST_TWO_LINES", f"{GATEWAY_KEY}\nsecond-line-{RANDOM_KEY}")
    with pytest.raises(nsc.ScriptError) as info:
        nsc.read_key("NSC_TEST_TWO_LINES")
    refused = info.value
    assert (refused.code, refused.exit_code) == ("bad_input", 2)
    assert GATEWAY_KEY not in refused.message + refused.fix
    assert GATEWAY_KEY not in scrubber.scrub(f"x {GATEWAY_KEY}\nsecond-line-{RANDOM_KEY}")


def test_read_key_refuses_a_key_too_short_to_mask(
    monkeypatch: pytest.MonkeyPatch, scrubber: nsc.Scrubber
) -> None:
    """The scrubber skips values shorter than MIN_SECRET (they would mask common words), so a
    key that short would show wherever an endpoint echoed it: bad_input, before it is sent.
    A key of MIN_SECRET characters is read and masked."""
    short = RANDOM_KEY[: nsc.MIN_SECRET - 1]
    monkeypatch.setenv("NSC_TEST_SHORT", f" {short}\n")
    with pytest.raises(nsc.ScriptError) as info:
        nsc.read_key("NSC_TEST_SHORT")
    refused = info.value
    assert (refused.code, refused.exit_code) == ("bad_input", nsc.EXIT_INPUT)
    assert f"shorter than {nsc.MIN_SECRET} characters" in refused.message
    assert short not in refused.message + refused.fix + str(refused)
    enough = RANDOM_KEY[: nsc.MIN_SECRET]
    monkeypatch.setenv("NSC_TEST_SHORT", enough)
    assert nsc.read_key("NSC_TEST_SHORT") == enough
    assert scrubber.scrub(f"the gateway said {enough}.") == "the gateway said ***."


def test_read_key_refuses_a_key_in_place_of_the_name(scrubber: nsc.Scrubber) -> None:
    with pytest.raises(nsc.ScriptError) as info:
        nsc.read_key(NVAPI_KEY)
    assert info.value.code == "literal_secret"


# --- the scrubber -------------------------------------------------------------------------------


@pytest.mark.parametrize("secret", [NVAPI_KEY, HF_KEY, GLPAT_KEY, GHP_KEY, SK_KEY, AKIA_KEY])
def test_the_scrubber_masks_well_known_token_shapes(secret: str) -> None:
    assert nsc.Scrubber().scrub(f"got {secret} here") == "got *** here"


@pytest.mark.parametrize("prefix", ["Authorization: Bearer ", "authorization: bearer ", "BEARER\t"])
def test_the_scrubber_masks_bearer_tokens(prefix: str) -> None:
    token = "eyJhbGciOi." + "J" * 20 + ".sig+/="
    assert nsc.Scrubber().scrub(prefix + token) == prefix + "***"


def test_the_scrubber_masks_registered_values_also_when_json_escaped() -> None:
    scrubber = nsc.Scrubber()
    value = ESCAPED_PASSWORD
    assert json.dumps(value)[1:-1] != value and json.dumps(value, ensure_ascii=False)[1:-1] != value
    scrubber.add(value)
    assert scrubber.scrub(f"x {value} y") == "x *** y"
    assert scrubber.scrub(json.dumps({"k": value})) == '{"k": "***"}'
    assert scrubber.scrub(json.dumps({"k": value}, ensure_ascii=False)) == '{"k": "***"}'
    text = scrubber.dumps({"k": value, value: [value]}, indent=2)
    assert json.loads(text) == {"k": "***", "***": ["***"]}


def test_the_scrubber_ignores_values_too_short_to_be_keys() -> None:
    scrubber = nsc.Scrubber()
    scrubber.add("abc12")
    assert scrubber.scrub("abc12 stays") == "abc12 stays"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("run --api-key VALUE --model m", "run --api-key *** --model m"),
        ("--hf-token=VALUE", "--hf-token=***"),
        ('--password "VALUE two"', "--password ***"),
        ("--client-secret 'VALUE'", "--client-secret ***"),
        ("--max-num-batched-tokens 8192", "--max-num-batched-tokens 8192"),
        ("--tokenizer meta/llama", "--tokenizer meta/llama"),
        ("--api-key-env NVIDIA_API_KEY", "--api-key-env NVIDIA_API_KEY"),
        ("--author someone", "--author someone"),
        ("a re-key value", "a re-key value"),
        ("--tool-call-parser llama3_json", "--tool-call-parser llama3_json"),
    ],
)
def test_the_scrubber_masks_the_values_of_credential_flags(text: str, expected: str) -> None:
    """VALUE stands for FLAG_SECRET, built at run time."""
    found = nsc.Scrubber().scrub(text.replace("VALUE", FLAG_SECRET))
    assert found == expected and FLAG_SECRET not in found


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # a header, an assignment, a JSON or Python field, a query string
        ("x-api-key: VALUE", "x-api-key: ***"),
        ("X-Api-Key:VALUE", "X-Api-Key:***"),
        ("api_key=VALUE", "api_key=***"),
        ("export NGC_API_KEY=VALUE", "export NGC_API_KEY=***"),
        ('{"api_key": "VALUE"}', '{"api_key": "***"}'),
        ("{'password': 'VALUE two'}", "{'password': '***'}"),
        ("GET /v1/x?access_token=VALUE&y=1", "GET /v1/x?access_token=***&y=1"),
        ("HF_TOKEN=VALUE client_secret: VALUE", "HF_TOKEN=*** client_secret: ***"),
        ("Authorization: Basic VALUE", "Authorization: Basic ***"),
        ("Authorization: Bearer VALUE", "Authorization: Bearer ***"),
        ('{"Authorization": "Bearer ***"}', '{"Authorization": "Bearer ***"}'),
        # a credential flag and its value as two items of a list
        ('["--api-key", "VALUE"]', '["--api-key", "***"]'),
        ("['serve', '--hf-token', 'VALUE']", "['serve', '--hf-token', '***']"),
        # not credential names, or a variable name rather than a value
        ("key=VALUE token: VALUE", "key=VALUE token: VALUE"),
        ('{"key": "enable_thinking", "auth": "bearer from $NAME"}',
         '{"key": "enable_thinking", "auth": "bearer from $NAME"}'),
        ("max_tokens=512 eos_token: </s> author: me", "max_tokens=512 eos_token: </s> author: me"),
        ("getting credentials: exec: executable aws", "getting credentials: exec: executable aws"),
        ("api_key_env=NVIDIA_API_KEY", "api_key_env=NVIDIA_API_KEY"),
        ('["--api-key-env", "NVIDIA_API_KEY"]', '["--api-key-env", "NVIDIA_API_KEY"]'),
        ('["--api-key-env", "--url"]', '["--api-key-env", "--url"]'),
        ("--api-key-env NGC_API_KEY: never a key", "--api-key-env NGC_API_KEY: never a key"),
        ('{"password": null}', '{"password": null}'),
    ],
)
def test_the_scrubber_masks_credential_names_and_flag_items(text: str, expected: str) -> None:
    """A value of no known shape after a credential name (a header, an assignment, a field) or
    after a credential flag in a list is masked; bare "key", "auth" and "token", other names
    that hold no credential, and a variable's name are kept."""
    value = "Qz" + "plain" * 3  # of no known shape, built at run time
    assert nsc.Scrubber().scrub(text.replace("VALUE", value)) == expected.replace("VALUE", value)


def test_scrub_obj_masks_credential_fields_and_the_item_after_a_credential_flag() -> None:
    value = "Qz" + "plain" * 3
    obj = {
        "headers": {"X-Api-Key": value, "Authorization": f"Basic {value}", "Accept": "*/*"},
        "env": {"NGC_API_KEY": value, "PASSWORD": 1234, "api_key_env": "NVIDIA_API_KEY"},
        "argv": ["serve", "--api-key", value, "--token", "--port", "8000", "--model", value],
        "auth": "bearer from $NAME",
        "switch": {"key": "enable_thinking", "value": True},
    }
    expected = {
        "headers": {"X-Api-Key": "***", "Authorization": "Basic ***", "Accept": "*/*"},
        "env": {"NGC_API_KEY": "***", "PASSWORD": "***", "api_key_env": "NVIDIA_API_KEY"},
        "argv": ["serve", "--api-key", "***", "--token", "--port", "8000", "--model", value],
        "auth": "bearer from $NAME",
        "switch": {"key": "enable_thinking", "value": True},
    }
    scrubber = nsc.Scrubber()
    assert scrubber.scrub_obj(obj) == expected
    assert json.loads(scrubber.dumps(obj)) == expected
    assert value in scrubber.dumps(obj)  # the --model value is no credential
    text = scrubber.dumps({"message": json.dumps(obj["argv"]) + f" x-api-key: {value}"})
    assert text.count(value) == 1 and '\\"--api-key\\", \\"***\\"' in text


def test_the_scrubber_stays_linear_on_long_runs_of_credential_words() -> None:
    """The name patterns look ahead for a credential word once per name, so a body that
    repeats one without a break (an echo, a stream) is masked in linear time."""
    for text in ("api_key_" * 125_000, "secretsecret-" * 80_000, "x-api-key: " * 90_000):
        started = time.monotonic()
        nsc.Scrubber().scrub(text)
        assert time.monotonic() - started < 10  # the quadratic form took minutes


def test_a_key_given_to_an_env_flag_is_still_masked() -> None:
    assert nsc.Scrubber().scrub(f"--api-key-env {RANDOM_KEY}") == "--api-key-env ***"
    for key in PREFIXED_KEYS:
        assert nsc.Scrubber().scrub(f"--api-key-env {key}") == "--api-key-env ***"
        for end in (".", ";", ":", ","):
            masked = nsc.Scrubber().scrub(f"--api-key-env {key}{end} x")
            assert masked.startswith("--api-key-env ***") and key[:8] not in masked, masked


@pytest.mark.parametrize("end", [".", ";", ":", ",", "!", "?", ")"])
def test_a_variable_name_before_punctuation_is_kept(end: str) -> None:
    """docs-10: the name of a variable that ends a sentence ("pass --api-key-env NAME; never
    ...") stays readable, so the literal_secret fix and the unauthorized message keep it."""
    text = f"pass --api-key-env NVIDIA_API_KEY{end} never pass the key itself"
    assert nsc.Scrubber().scrub(text) == text
    assert nsc.Scrubber().scrub(f"--api-key-env=NAME{end}") == f"--api-key-env=NAME{end}"
    with pytest.raises(nsc.ScriptError) as info:
        nsc.check_env_name("not a name")
    assert "--api-key-env NVIDIA_API_KEY;" in nsc.Scrubber().scrub(info.value.fix)


def test_clip_masks_before_it_cuts(scrubber: nsc.Scrubber) -> None:
    """A cut that falls inside a registered key keeps none of it: the key is masked first."""
    scrubber.add(GATEWAY_KEY)
    text = "x" * 10 + " echoed  " + GATEWAY_KEY + " and more"
    assert nsc.clip(text, 30) == "x" * 10 + " echoed *** and more"[:20]
    assert all(GATEWAY_KEY[:n] not in nsc.clip(text, limit) for limit in range(10, 40)
               for n in range(nsc.MIN_SECRET, len(GATEWAY_KEY) + 1))


def test_scrub_obj_masks_every_string_of_a_nested_value() -> None:
    scrubber = nsc.Scrubber()
    scrubber.add(GATEWAY_KEY)
    value = {
        "a": [GATEWAY_KEY, {"b": (NVAPI_KEY, 3)}],
        GATEWAY_KEY: True,
        "n": 7,
        "p": Path("/x") / GATEWAY_KEY,
    }
    assert scrubber.scrub_obj(value) == {
        "a": ["***", {"b": ["***", 3]}],
        "***": True,
        "n": 7,
        "p": "/x/***",
    }


def nested(depth: int, leaf: Any = "leaf") -> Any:
    """A list nested `depth` levels deep around `leaf`, built without recursion."""
    value = leaf
    for _ in range(depth):
        value = [value]
    return value


def depth_of(value: Any) -> int:
    """The nesting depth along the first item of each list or object."""
    depth = 0
    while isinstance(value, (list, dict)):
        depth += 1
        items = list(value.values()) if isinstance(value, dict) else value
        if not items:
            break
        value = items[0]
    return depth


def test_scrub_obj_and_dumps_keep_no_value_nested_past_the_walk_limit() -> None:
    """security-5: the masker recurses once a level, so a value nested past MAX_WALK_DEPTH is
    replaced by NESTED instead of ending the script with RecursionError; the JSON stays valid
    and a key at a kept level is still masked."""
    scrubber = nsc.Scrubber()
    scrubber.add(GATEWAY_KEY)
    value = {"facts": nested(5000, GATEWAY_KEY), "near": nested(10, GATEWAY_KEY)}
    out = scrubber.scrub_obj(value)
    assert depth_of(out["facts"]) == nsc.MAX_WALK_DEPTH - 1
    assert nsc.NESTED in json.dumps(out)
    assert out["near"] == nested(10, "***")
    text = scrubber.dumps(value)
    assert json.loads(text)["near"] == nested(10, "***") and GATEWAY_KEY not in text
    edge = nested(nsc.MAX_WALK_DEPTH - 1, GATEWAY_KEY)
    assert scrubber.scrub_obj(edge) == nested(nsc.MAX_WALK_DEPTH - 1, "***")  # nothing cut


@pytest.mark.parametrize("depth", [nsc.MAX_JSON_DEPTH + 1, 950, 1000, 3000, 100_000])
def test_json_nested_too_deeply_is_not_json(depth: int) -> None:
    """security-5: json.loads raises RecursionError (not ValueError) on deep nesting, and the
    walkers that mask and cut a body recurse once a level: a server's JSON nested deeper than
    MAX_JSON_DEPTH is not JSON (parse_json: ValueError; json_body: None), never a crash."""
    for text in ("[" * depth + "]" * depth, '{"a":' * depth + "1" + "}" * depth):
        with pytest.raises(ValueError):
            nsc.parse_json(text)
        result = nsc.HttpResult(200, {"content-type": "application/json"}, text.encode())
        assert result.json_body() is None


def strict_json(text: str) -> Any:
    """`text` read as the JSON standard defines it: NaN, Infinity and -Infinity are refused."""

    def refuse(name: str) -> None:
        raise ValueError(f"{name} is not JSON")

    return json.loads(text, parse_constant=refuse)


@pytest.mark.parametrize(
    "body",
    ['{"usage": {"prompt_tokens": NaN}}', "[Infinity]", '{"logprob": -Infinity}',
     '{"n": 1e999}', "[-1E400]"],
)
def test_nan_or_infinity_in_a_server_answer_is_not_json(body: str) -> None:
    """json.loads takes NaN, Infinity, -Infinity and turns 1e999 into an infinity; written back,
    they made output that JSON readers refuse. A server's answer that holds one is not JSON."""
    with pytest.raises(ValueError):
        nsc.parse_json(body)
    result = nsc.HttpResult(200, {"content-type": "application/json"}, body.encode())
    assert result.json_body() is None
    assert nsc.parse_json('{"a": 1.5e300, "b": -0.0, "c": 2}') == {"a": 1.5e300, "b": -0.0, "c": 2}


def test_the_output_has_no_nan_or_infinity(capsys: pytest.CaptureFixture[str]) -> None:
    """A number JSON has no form for, whatever its source, is written as text: stdout and the
    files stay JSON that any reader takes."""
    value = {"nan": float("nan"), "big": [float("inf"), -float("inf")], "ok": 1.5}
    expected = {"nan": "nan", "big": ["inf", "-inf"], "ok": 1.5}
    assert strict_json(nsc.Scrubber().dumps(value)) == expected
    with pytest.raises(SystemExit):
        nsc.emit(True, "passed", "report", value)
    assert strict_json(capsys.readouterr().out)["facts"] == expected


def test_parse_json_takes_json_up_to_the_depth_limit() -> None:
    limit = nsc.MAX_JSON_DEPTH
    assert depth_of(nsc.parse_json("[" * limit + "]" * limit)) == limit
    assert nsc.parse_json('{"a": [1, {"b": null}]}') == {"a": [1, {"b": None}]}
    assert nsc.parse_json("3") == 3 and nsc.parse_json('"x"') == "x"
    assert nsc.parse_json(b'{"a": []}') == {"a": []}
    assert nsc.parse_json("[[[1]]]", limit=3) == [[[1]]]
    with pytest.raises(ValueError):
        nsc.parse_json("[[[1]]]", limit=2)
    for bad in ("", "{", "nan-ish", '{"a": }'):
        with pytest.raises(ValueError):
            nsc.parse_json(bad)


# --- base URLs ----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("url", "base"),
    [
        ("http://127.0.0.1:8000", "http://127.0.0.1:8000"),
        ("http://127.0.0.1:8000/", "http://127.0.0.1:8000"),
        ("http://127.0.0.1:8000/v1", "http://127.0.0.1:8000"),
        ("http://127.0.0.1:8000/v1/", "http://127.0.0.1:8000"),
        ("  http://127.0.0.1:8000/v1  ", "http://127.0.0.1:8000"),
        ("HTTPS://integrate.api.nvidia.com/v1", "https://integrate.api.nvidia.com"),
        ("https://gw.example.com/nim/llama/v1", "https://gw.example.com/nim/llama"),
        ("https://gw.example.com/nim/llama", "https://gw.example.com/nim/llama"),
        ("http://host:8000/v1/v1", "http://host:8000/v1"),
        ("http://host:8000//", "http://host:8000"),
        ("http://[::1]:8000/v1", "http://[::1]:8000"),
        ("http://[fe80::1%25eth0]:8000/v1", "http://[fe80::1%25eth0]:8000"),  # an IPv6 zone
        ("http://localhost:8000/V1", "http://localhost:8000/V1"),
    ],
)
def test_normalize_base_gives_the_base_the_scripts_append_v1_to(url: str, base: str) -> None:
    assert nsc.normalize_base(url) == base


@pytest.mark.parametrize(
    "url",
    [
        "",
        "   ",
        "ftp://host/v1",
        "file:///etc/hostname",
        "javascript:alert(1)",
        "host:8000",
        "127.0.0.1:8000",
        "//host:8000",
        "http://",
        "http:///v1",
        "http://host:port/",
        "http://host:99999/",
        "http://[::1/",
        "http://host:8000/v1?x=1",
        "http://host:8000/#top",
        "http://ho st:8000",
        "http://host\n:8000",
        # urllib decodes the host: these would reach the names 127.0.0.1.gw.example.com,
        # gw.example.com. and ::1.gw.example.com; only an IPv6 zone keeps its %
        "http://127.0.0.1%2Egw.example.com:8000",
        "http://gw.example.com%2E/v1",
        "http://[::1%2Egw.example.com]:8000",
    ],
)
def test_normalize_base_refuses_what_is_not_a_base_url(url: str) -> None:
    with pytest.raises(nsc.ScriptError) as info:
        nsc.normalize_base(url)
    assert (info.value.code, info.value.exit_code) == ("bad_input", nsc.EXIT_INPUT)


@pytest.mark.parametrize(
    "url",
    [
        f"http://user:{PASSWORD}@127.0.0.1:8000/v1",
        f"https://{PASSWORD}@gw.example.com/v1",
        f"http://user:{PASSWORD}@[::1]:8000",
        f"ftp://user:{PASSWORD}@host/",
        f"user:{PASSWORD}@host:8000",
    ],
)
def test_normalize_base_refuses_a_password_and_never_echoes_it(url: str) -> None:
    with pytest.raises(nsc.ScriptError) as info:
        nsc.normalize_base(url)
    assert info.value.code == "bad_input"
    assert PASSWORD not in info.value.message + info.value.fix + str(info.value)


@pytest.mark.parametrize(
    ("url", "loopback"),
    [
        ("http://127.0.0.1:8000", True),
        ("http://127.9.9.9", True),
        ("http://localhost:8000/v1", True),
        ("http://LOCALHOST", True),
        ("https://localhost.:8443", True),
        ("http://nim.localhost:8000", True),
        ("http://[::1]:8000", True),
        ("http://[::ffff:127.0.0.1]:8000", True),
        ("http://0.0.0.0:8000", True),
        ("http://10.0.0.5:8000", False),
        ("http://192.168.1.10", False),
        ("https://integrate.api.nvidia.com", False),
        ("http://localhost.example.com", False),
        ("http://[::2]", False),
        ("http://[fe80::1]:8000", False),
        ("not a url", False),
        ("", False),
        ("http://[::1", False),
        # legacy IPv4 forms the connection reads as 127.0.0.1 (inet_aton), with no DNS lookup
        ("http://127.1:8000", True),
        ("http://2130706433", True),
        ("http://0x7f000001:8000/v1", True),
        ("http://0X7F.1", True),
        ("http://127.000.000.001", True),
        ("http://0177.0.0.1", True),
        ("http://0:8000", True),  # 0.0.0.0
        ("http://10.1:8000", False),  # 10.0.0.1
        ("http://3232235777", False),  # 192.168.1.1
        ("http://127.0.0.08", False),  # not an address: 08 is no octal number
        ("http://127.1.example.com", False),
        ("http://0x7f.nim.example", False),
        # the host as the connection reads it: urllib decodes it, and only IPv6 has a zone
        ("http://127.0.0.1%2Egw.example.com:8000", False),
        ("http://0.0.0.0%2Egw.example.com", False),
        ("http://127.0.0.1%25lo:8000", False),
        ("http://[::1%25lo]:8000", True),
        ("http://[fe80::1%25eth0]:8000", False),
    ],
)
def test_is_loopback(url: str, loopback: bool) -> None:
    assert nsc.is_loopback(url) is loopback


def test_a_short_loopback_address_never_goes_through_the_proxy(
    monkeypatch: pytest.MonkeyPatch, no_proxy_settings: None
) -> None:
    """security-8: 127.1 and the other legacy forms reach this machine (the resolver reads them
    as numbers), so a request to one, and its key, never goes to HTTP_PROXY, which would reach
    its own loopback instead. Only numeric hosts are resolved here, never a name."""
    real = socket.getaddrinfo

    def numeric_only(host: Any, *args: Any, **kwargs: Any) -> Any:
        if not nsc.LEGACY_IPV4.fullmatch(str(host).lower()):
            raise socket.gaierror(socket.EAI_NONAME, f"the test looks up no name ({host})")
        return real(host, *args, **kwargs)

    monkeypatch.setattr(socket, "getaddrinfo", numeric_only)
    auth = {"Authorization": f"Bearer {GATEWAY_KEY}"}
    with serve(answer(200, {"data": []})) as proxy, serve(answer(200, {"data": []})) as nim:
        for name in ("http_proxy", "HTTP_PROXY"):
            monkeypatch.setenv(name, proxy.url)
        port = nim.url.rsplit(":", 1)[1]
        for host in ("127.1", "2130706433", "0x7f000001", "127.000.000.001"):
            result = nsc.request_json("GET", f"http://{host}:{port}/v1/models", headers=auth,
                                      timeout=5)
            assert (result.status, result.error) == (200, None), (host, result.error)
    assert proxy.requests == []
    assert [request["headers"]["authorization"] for request in nim.requests] == [
        f"Bearer {GATEWAY_KEY}"] * 4


def test_strip_userinfo() -> None:
    assert nsc.strip_userinfo(f"http://u:{PASSWORD}@h:1/x") == "http://h:1/x"
    assert nsc.strip_userinfo("https://h/x@y") == "https://h/x@y"


def test_same_origin_compares_scheme_host_and_port() -> None:
    base = "http://Gateway.example.com/prefix"
    assert nsc.origin(base) == ("http", "gateway.example.com", 80)
    assert nsc.origin("https://[::1]/x") == ("https", "::1", 443)
    for same in ("http://gateway.example.com:80/v1/health/ready", "HTTP://GATEWAY.example.com/x"):
        assert nsc.same_origin(same, base), same
    for other in ("https://gateway.example.com/prefix", "http://gateway.example.com:8001/prefix",
                  "http://10.0.0.7/prefix", "http://gateway.example.com.evil.invalid/prefix",
                  "http://gateway.example.com:bad/", "ftp://gateway.example.com/", "not a url"):
        assert not nsc.same_origin(other, base), other
    assert not nsc.same_origin("not a url", "not a url")


def test_a_host_with_a_final_dot_is_another_origin() -> None:
    """gw.example.com. is looked up as written; gw.example.com may get the resolver's search
    domains first (ndots), so the two names can reach different servers. The host is compared
    as written, and the key that goes to one never goes to the other."""
    assert nsc.origin("HTTP://GATEWAY.example.com./x") == ("http", "gateway.example.com.", 80)
    for dotted, plain in (("http://gw.example.com./v1/health/live", "http://gw.example.com"),
                          ("https://gw.example.com.:8443/x", "https://gw.example.com:8443")):
        assert not nsc.same_origin(dotted, plain) and not nsc.same_origin(plain, dotted)
        assert nsc.same_origin(dotted, dotted.upper())  # case still does not count


# --- TLS ----------------------------------------------------------------------------------------


def system_ca_file() -> str:
    """A CA bundle of this machine to pass as --ca-file (only read, never changed)."""
    paths = ssl.get_default_verify_paths()
    for candidate in (
        paths.cafile,
        paths.openssl_cafile,
        "/etc/ssl/certs/ca-certificates.crt",  # Debian, Ubuntu
        "/etc/pki/tls/certs/ca-bundle.crt",  # RHEL, Fedora
        "/etc/ssl/cert.pem",  # macOS, Alpine
    ):
        if candidate and os.path.isfile(candidate):
            return candidate
    pytest.skip("no CA bundle file on this machine")


def test_tls_verification_is_on_by_default() -> None:
    for tls in (nsc.Tls(), nsc.DEFAULT_TLS):
        context = tls.context()
        assert context.verify_mode == ssl.CERT_REQUIRED and context.check_hostname
    tls = nsc.Tls()
    assert tls.context() is tls.context()  # built once


def test_a_ca_file_keeps_verification_on() -> None:
    context = nsc.Tls(ca_file=system_ca_file()).context()
    assert context.verify_mode == ssl.CERT_REQUIRED and context.check_hostname
    assert context.cert_store_stats()["x509_ca"] > 0


def test_tls_files_that_cannot_be_loaded_are_bad_input(tmp_path: Path) -> None:
    not_pem = tmp_path / "not.pem"
    not_pem.write_text("not a certificate\n", encoding="utf-8")
    missing = str(tmp_path / "missing.pem")
    for tls in (
        nsc.Tls(ca_file=missing),
        nsc.Tls(ca_file=str(not_pem)),
        nsc.Tls(client_key=str(not_pem)),
        nsc.Tls(client_cert=missing),
        nsc.Tls(client_cert=str(not_pem), client_key=str(not_pem)),
    ):
        with pytest.raises(nsc.ScriptError) as info:
            tls.context()
        assert (info.value.code, info.value.exit_code) == ("bad_input", nsc.EXIT_INPUT), tls


def test_a_client_key_is_loaded_with_no_passphrase_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    """OpenSSL prompts on the terminal for the passphrase of an encrypted key unless a callback
    gives one: the callback gives an empty one, so loading such a key fails instead."""
    seen: list[tuple[Any, ...]] = []

    def load_cert_chain(
        self: ssl.SSLContext, certfile: Any, keyfile: Any = None, password: Any = None
    ) -> None:
        seen.append((certfile, keyfile, password))

    monkeypatch.setattr(ssl.SSLContext, "load_cert_chain", load_cert_chain)
    context = nsc.Tls(client_cert="~/client.pem", client_key="~/client.key").context()
    ((cert, key, password),) = seen
    assert (cert, key) == (os.path.expanduser("~/client.pem"), os.path.expanduser("~/client.key"))
    assert callable(password) and password() == b""
    assert context.verify_mode == ssl.CERT_REQUIRED


def test_open_url_always_hands_urllib_a_verifying_context(monkeypatch: pytest.MonkeyPatch) -> None:
    """The context that makes each TLS handshake verifies the chain and the host name, with the
    system CA store and with --ca-file (here the server does not speak TLS, so it fails)."""
    contexts: list[ssl.SSLContext] = []
    real = ssl.SSLContext.wrap_socket

    def recording(self: ssl.SSLContext, *args: Any, **kwargs: Any) -> ssl.SSLSocket:
        contexts.append(self)
        return real(self, *args, **kwargs)

    monkeypatch.setattr(ssl.SSLContext, "wrap_socket", recording)
    with plain_bytes_server(b"HTTP/1.0 400 Bad Request\r\n\r\n") as address:
        for tls in (None, nsc.Tls(ca_file=system_ca_file())):
            result = nsc.request_json("GET", f"https://{address}/v1/models", timeout=5, tls=tls)
            assert (result.status, result.error_code) == (0, "tls_error"), result.error
    assert len(contexts) == 2
    for context in contexts:
        assert context.verify_mode == ssl.CERT_REQUIRED and context.check_hostname


def test_https_to_a_server_that_does_not_speak_tls_is_a_tls_error() -> None:
    with plain_bytes_server(b"HTTP/1.0 400 Bad Request\r\n\r\n") as address:
        result = nsc.request_json("GET", f"https://{address}/v1/models", timeout=5)
    assert (result.status, result.error_code) == (0, "tls_error"), result.error


# --- requests -----------------------------------------------------------------------------------


def test_request_json_sends_the_contract_headers(no_proxy_settings: None) -> None:
    models = {"object": "list", "data": [{"id": "m"}]}
    with serve(answer(200, models)) as srv:
        got = nsc.request_json(
            "GET",
            f"{srv.url}/v1/models",
            headers={
                "X-Request-Id": "nsc-r1-models-1",
                "Authorization": f"Bearer {GATEWAY_KEY}",
                "X-Skipped": None,
            },
            timeout=5,
        )
        posted = nsc.request_json(
            "POST",
            f"{srv.url}/v1/chat/completions",
            {"model": "m", "messages": []},
            headers={"X-Request-Id": "nsc-r1-chat-1"},
            timeout=5,
        )
    get, post = srv.requests
    assert (get["method"], get["path"], get["body"]) == ("GET", "/v1/models", b"")
    assert get["headers"]["user-agent"] == "nim-smoke-check/0.1"
    assert get["headers"]["accept"] == "application/json"
    assert get["headers"]["authorization"] == f"Bearer {GATEWAY_KEY}"
    assert get["headers"]["x-request-id"] == "nsc-r1-models-1"
    assert "content-type" not in get["headers"] and "x-skipped" not in get["headers"]
    assert (post["method"], post["headers"]["content-type"]) == ("POST", "application/json")
    assert json.loads(post["body"]) == {"model": "m", "messages": []}
    assert "authorization" not in post["headers"]
    assert (got.status, got.error, got.json_body(), got.content_type) == (
        200,
        None,
        models,
        "application/json",
    )
    assert got.request_id == "nsc-r1-models-1" and posted.request_id == "nsc-r1-chat-1"
    assert got.ttfb_ms is not None and 0 <= got.ttfb_ms <= got.elapsed_ms


def test_the_request_id_the_server_sends_back_wins() -> None:
    with serve(answer(200, {}, {"X-Request-Id": "server-7"})) as srv:
        result = nsc.request_json(
            "GET", f"{srv.url}/v1/models", headers={"X-Request-Id": "nsc-r1-models-1"}, timeout=5
        )
    assert result.request_id == "server-7"


def test_a_body_is_sent_as_json_unless_it_is_bytes() -> None:
    with serve(answer(200, {})) as srv:
        for body in ("text", ["a", 1], b'{"raw": true}'):
            nsc.request_json("POST", f"{srv.url}/v1/chat/completions", body, timeout=5)
        nsc.request_stream(f"{srv.url}/v1/chat/completions", None, timeout=5)
    bodies = [request["body"] for request in srv.requests]
    assert bodies == [b'"text"', b'["a", 1]', b'{"raw": true}', b""]
    assert "content-type" not in srv.requests[3]["headers"]  # no body, no Content-Type


@pytest.mark.parametrize("status", [400, 404, 422, 500, 503])
def test_an_error_answer_keeps_its_status_and_body(status: int) -> None:
    body = {"error": {"message": "nope", "type": "BadRequestError", "param": None, "code": status}}
    with serve(answer(status, body)) as srv:
        result = nsc.request_json(
            "POST", f"{srv.url}/v1/chat/completions", {"model": "m"}, timeout=5
        )
    assert (result.status, result.error, result.json_body()) == (status, None, body)


def test_a_body_that_is_not_json_has_no_json_body() -> None:
    with serve(answer(200, b"<html>proxy login</html>")) as srv:
        result = nsc.request_json("GET", f"{srv.url}/v1/models", timeout=5)
    assert (result.status, result.json_body()) == (200, None)
    assert result.body == b"<html>proxy login</html>"


def test_a_closed_port_is_no_answer() -> None:
    result = nsc.request_json("GET", f"{closed_url()}/v1/models", timeout=5)
    assert (result.status, result.error_code, result.body, result.ttfb_ms) == (
        0,
        "connection_failed",
        b"",
        None,
    )
    assert result.headers == {}


def test_a_server_that_never_answers_is_a_timeout() -> None:
    def silent(handler: BaseHTTPRequestHandler, local: Local) -> None:
        local.stop.wait(10)

    with serve(silent) as srv:
        started = time.monotonic()
        result = nsc.request_json("GET", f"{srv.url}/v1/health/ready", timeout=0.5)
        stream = nsc.request_stream(f"{srv.url}/v1/chat/completions", {}, timeout=0.5)
        assert time.monotonic() - started < 4
    assert (result.status, result.error) == (0, "timeout")
    assert (stream.status, stream.error) == (0, "timeout")


def test_a_json_answer_that_does_not_end_in_time_is_a_timeout() -> None:
    def drip(handler: BaseHTTPRequestHandler, local: Local) -> None:
        handler.send_response(200)
        handler.send_header("Content-Type", "application/json")
        handler.send_header("Content-Length", "1000")
        handler.end_headers()
        while not local.stop.wait(0.1):
            handler.wfile.write(b" ")

    with serve(drip) as srv:
        result = nsc.request_json("GET", f"{srv.url}/v1/models", timeout=1.0)
    assert (result.status, result.error) == (200, "timeout")
    # not before the deadline; well before the 100 s the drip of 1,000 bytes would take
    assert 900 <= result.elapsed_ms < 10_000


@pytest.mark.parametrize("part", ["headers", "chunk size"])
def test_an_answer_that_keeps_arriving_ends_at_the_deadline(part: str) -> None:
    """A socket's timeout counts for one read. Headers, or the size line of a chunk, that come a
    byte every 0.05 s never let it fire: the request took as long as the server kept sending
    (here 5 s, then the answer ends). Each read now gets only the time left until the one
    deadline of the request, so it ends then."""
    if part == "headers":
        head, drip = b"HTTP/1.1 200 OK\r\nX-Slow: ", b"a"
        end = b"\r\nContent-Type: text/event-stream\r\nContent-Length: 2\r\n\r\n{}"
    else:
        head = (b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n"
                b"Transfer-Encoding: chunked\r\n\r\n")
        drip, end = b"0", b"2\r\n{}\r\n0\r\n\r\n"

    def slow(handler: BaseHTTPRequestHandler, local: Local) -> None:
        handler.wfile.write(head)
        for _ in range(100):
            if local.stop.wait(0.05):
                return
            handler.wfile.write(drip)
        handler.wfile.write(end)

    with serve(slow) as srv:
        result = nsc.request_stream(f"{srv.url}/v1/chat/completions", {}, timeout=1.0)
    # no answer when the headers never ended; else the answer, cut at the deadline
    assert (result.status, result.error, result.body) == (
        0 if part == "headers" else 200, "timeout", b"")
    assert 900 <= result.elapsed_ms < 4000


class NoAnswer:
    """A socket whose connection gets no answer: connect waits for the socket's timeout, then
    fails, as a host that drops the packets does."""

    def __init__(self, *args: Any) -> None:
        self.timeout: float | None = None

    def settimeout(self, timeout: float | None) -> None:
        self.timeout = timeout

    def connect(self, where: Any) -> None:
        time.sleep(self.timeout or 0)
        raise TimeoutError("timed out")

    def close(self) -> None:
        pass


class NoIpv6(socket.socket):
    """A socket that cannot be made for IPv6, as on a machine with IPv6 turned off."""

    def __init__(self, family: int = -1, *args: Any, **kwargs: Any) -> None:
        if family == socket.AF_INET6:
            raise OSError(errno.EAFNOSUPPORT, "Address family not supported by protocol")
        super().__init__(family, *args, **kwargs)


def tcp(family: int, *where: Any) -> tuple[Any, ...]:
    """One address as socket.getaddrinfo gives it."""
    return (family, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", where)


def test_a_host_name_with_several_addresses_ends_at_the_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """socket.create_connection gives each address of a host name the whole timeout, so a name
    with two addresses that did not answer took twice the request's timeout. Each address now
    gets only the time left until the one deadline of the request."""
    found = [tcp(socket.AF_INET, "192.0.2.1", 8000), tcp(socket.AF_INET, "192.0.2.2", 8000)]
    monkeypatch.setattr(socket, "getaddrinfo", lambda *args, **kwargs: found)
    monkeypatch.setattr(socket, "socket", NoAnswer)  # no packet leaves this machine
    result = nsc.request_json("GET", "http://nim.test:8000/v1/models", timeout=1.0,
                              loopback=True)
    assert (result.status, result.error) == (0, "timeout")
    assert 900 <= result.elapsed_ms < 1500  # about 2000 when each address got the whole second


def test_an_address_that_fails_is_skipped_for_the_next_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """As in socket.create_connection, an address whose socket cannot be made (IPv6 turned off)
    or that refuses the connection is skipped, and the next address of the host answers."""
    dead = int(closed_url().rsplit(":", 1)[1])
    with serve(answer(200, {"data": []})) as srv:
        live = int(srv.url.rsplit(":", 1)[1])
        found = [tcp(socket.AF_INET6, "::1", live, 0, 0), tcp(socket.AF_INET, "127.0.0.1", dead),
                 tcp(socket.AF_INET, "127.0.0.1", live)]
        monkeypatch.setattr(socket, "getaddrinfo", lambda *args, **kwargs: found)
        monkeypatch.setattr(socket, "socket", NoIpv6)
        result = nsc.request_json("GET", f"http://nim.test:{live}/v1/models", timeout=5,
                                  loopback=True)
        monkeypatch.undo()
    assert (result.status, result.error, len(srv.requests)) == (200, None, 1)


@pytest.mark.parametrize("url", ["file:///etc/hostname", "ftp://127.0.0.1/x", "data:,hello"])
def test_only_http_and_https_urls_are_opened(url: str) -> None:
    result = nsc.request_json("GET", url, timeout=5)
    assert (result.status, result.error_code, result.body) == (0, "bad_request", b"")
    with pytest.raises(ValueError):
        nsc.open_url(urllib.request.Request(url), timeout=5)


def test_only_get_and_post_are_sent_and_a_get_has_no_body() -> None:
    url = f"{closed_url()}/v1/models"
    for method in ("PUT", "DELETE", "PATCH", "HEAD", "get"):
        with pytest.raises(ValueError):
            nsc.request_json(method, url, timeout=1)
    with pytest.raises(ValueError):
        nsc.request_json("GET", url, {"a": 1}, timeout=1)
    for timeout in (0, -1, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            nsc.request_json("GET", url, timeout=timeout)


@pytest.mark.parametrize("code", [301, 302, 303, 307, 308])
def test_a_redirect_is_never_followed(code: int) -> None:
    """The redirect is reported, and the host it names gets no request, so no key reaches it."""
    with serve(answer(200, {"stolen": True})) as elsewhere:
        target = f"{elsewhere.url}/v1/chat/completions"
        with serve(redirect(code, target)) as first:
            auth = {"Authorization": f"Bearer {GATEWAY_KEY}"}
            result = nsc.request_json(
                "POST", f"{first.url}/v1/chat/completions", {"model": "m"}, headers=auth, timeout=5
            )
            stream = nsc.request_stream(
                f"{first.url}/v1/chat/completions", {"stream": True}, headers=auth, timeout=5
            )
            got = nsc.request_json("GET", f"{first.url}/v1/models", headers=auth, timeout=5)
    for item in (result, stream, got):
        assert (item.status, item.error, item.error_code) == (code, "redirect_refused", item.error)
        assert item.headers["location"] == target
    assert len(first.requests) == 3
    assert first.requests[0]["headers"]["authorization"] == f"Bearer {GATEWAY_KEY}"
    assert elsewhere.requests == []


def test_a_redirect_to_another_scheme_is_refused_too() -> None:
    with serve(redirect(302, "file:///etc/hostname")) as first:
        result = nsc.request_json("GET", f"{first.url}/v1/models", timeout=5)
    assert (result.status, result.error) == (302, "redirect_refused")
    assert result.body == b""


def test_a_loopback_url_ignores_the_proxy_settings(
    monkeypatch: pytest.MonkeyPatch, no_proxy_settings: None
) -> None:
    dead = closed_url()
    for name in ("http_proxy", "HTTP_PROXY", "https_proxy", "HTTPS_PROXY"):
        monkeypatch.setenv(name, dead)
    with serve(answer(200, {"data": []})) as srv:
        port = srv.url.rsplit(":", 1)[1]
        for base, loopback in (
            (srv.url, True),
            (srv.url, False),
            (f"http://localhost:{port}", False),
        ):
            result = nsc.request_json("GET", f"{base}/v1/models", timeout=5, loopback=loopback)
            assert (result.status, result.error) == (200, None), (base, loopback, result.error)
    assert [request["path"] for request in srv.requests] == ["/v1/models"] * 3


def test_a_remote_url_keeps_the_proxy_settings(
    monkeypatch: pytest.MonkeyPatch, no_proxy_settings: None, local_only: None
) -> None:
    """The control of the test above: with the same settings, a URL whose host is not this
    machine goes to the proxy (here a local server; the host is never looked up)."""
    with serve(answer(200, {"data": []})) as proxy:
        for name in ("http_proxy", "HTTP_PROXY"):
            monkeypatch.setenv(name, proxy.url)
        result = nsc.request_json("GET", "http://nim.invalid:8000/v1/models", timeout=5)
    assert (result.status, result.error) == (200, None)
    assert [request["path"] for request in proxy.requests] == ["http://nim.invalid:8000/v1/models"]


# --- event streams ------------------------------------------------------------------------------

EVENT_1 = b'data: {"object": "chat.completion.chunk", "n": 1}\n\n'
EVENT_2 = b'data: {"object": "chat.completion.chunk", "n": 2}\n\n'
DONE = b"data: [DONE]\n\n"


@pytest.mark.parametrize("chunked", [False, True])
def test_a_stream_is_read_as_it_arrives(chunked: bool) -> None:
    parts = [(0.4, EVENT_1), (0.4, EVENT_2), (0.0, DONE)]
    with serve(events(parts, chunked=chunked), http11=chunked) as srv:
        result = nsc.request_stream(
            f"{srv.url}/v1/chat/completions",
            {"model": "m", "stream": True},
            headers={"X-Request-Id": "nsc-r1-stream-1"},
            timeout=10,
        )
    assert (result.status, result.error, result.content_type) == (200, None, "text/event-stream")
    assert result.body == EVENT_1 + EVENT_2 + DONE
    assert result.ttfb_ms is not None
    # the first event (sent at 0.4 s), not the end (0.4 s later); the gap is kept loose
    assert 350 <= result.ttfb_ms <= result.elapsed_ms - 100
    request = srv.requests[0]
    assert request["headers"]["accept"] == "text/event-stream"
    assert request["headers"]["content-type"] == "application/json"
    assert json.loads(request["body"]) == {"model": "m", "stream": True}
    assert result.request_id == "nsc-r1-stream-1"


def test_a_stream_stops_at_max_bytes() -> None:
    event = b'data: {"x": "' + b"a" * 1000 + b'"}\n\n'
    parts = [(0.0, event)] * 50 + [(5.0, DONE)]
    with serve(events(parts, chunked=True), http11=True) as srv:
        result = nsc.request_stream(
            f"{srv.url}/v1/chat/completions", {}, timeout=10, max_bytes=10_000
        )
    assert (result.status, result.error) == (200, "body_too_large")
    assert result.body == (event * 50)[:10_000]
    assert result.elapsed_ms < 3000  # it did not wait for the rest


@pytest.mark.parametrize(
    "parts",
    [
        [(0.0, EVENT_1 + DONE)],
        [(0.0, EVENT_1 + b"data: [DO"), (0.2, b"NE]\n\n")],
        [(0.0, EVENT_1.replace(b"\n", b"\r\n") + b"data:[DONE]\r\n\r\n")],
    ],
)
def test_a_stream_ends_at_done_even_if_the_connection_stays_open(
    parts: list[tuple[float, bytes]],
) -> None:
    with serve(events(parts, chunked=True, hold=10), http11=True) as srv:
        result = nsc.request_stream(f"{srv.url}/v1/chat/completions", {}, timeout=10)
    assert (result.status, result.error) == (200, None)
    assert result.body == b"".join(data for _, data in parts)
    assert result.elapsed_ms < 3000


def test_a_stream_of_keep_alive_comments_ends_at_the_deadline() -> None:
    parts = [(0.1, b": keep-alive\n\n")] * 200
    with serve(events(parts, chunked=True), http11=True) as srv:
        result = nsc.request_stream(f"{srv.url}/v1/chat/completions", {}, timeout=1.0)
    assert (result.status, result.error) == (200, "timeout")
    assert result.body.startswith(b": keep-alive\n\n")
    # not before the deadline; well before the 20 s the 200 comments would take
    assert 900 <= result.elapsed_ms < 10_000


def test_a_stream_that_goes_quiet_ends_at_the_deadline() -> None:
    """Each read gets only the time left, so a quiet server cannot add a whole timeout: a read
    that got the whole 3 s after the event at 2 s would end at 5 s or later. The 1.5 s between
    the deadline and the bound leave room for a busy machine."""
    with serve(events([(2.0, EVENT_1), (30.0, EVENT_2)])) as srv:
        result = nsc.request_stream(f"{srv.url}/v1/chat/completions", {}, timeout=3.0)
    assert (result.status, result.error, result.body) == (200, "timeout", EVENT_1)
    assert 2900 <= result.elapsed_ms < 4500


def test_a_stream_cut_in_the_middle_of_a_chunk_is_a_lost_connection() -> None:
    def cut(handler: BaseHTTPRequestHandler, local: Local) -> None:
        handler.send_response(200)
        handler.send_header("Content-Type", "text/event-stream")
        handler.send_header("Transfer-Encoding", "chunked")
        handler.end_headers()
        handler.wfile.write(b"64\r\n" + EVENT_1[:10])  # promises 100 bytes, sends 10

    with serve(cut, http11=True) as srv:
        result = nsc.request_stream(f"{srv.url}/v1/chat/completions", {}, timeout=5)
    assert (result.status, result.error_code, result.body) == (200, "connection_lost", EVENT_1[:10])


def test_a_body_shorter_than_its_content_length_is_a_lost_connection() -> None:
    """http.client's read1 gives b"" with no error when the server closes before the
    Content-Length it declared, so 7 of 20 bytes, here whole-looking JSON, passed for the whole
    answer. That is connection_lost. With no Content-Length the close ends the body, and
    [DONE] ends a stream on purpose though more bytes were declared."""

    def answer_of(kind: str, length: str | None, data: bytes, hold: float = 0) -> Respond:
        def respond(handler: BaseHTTPRequestHandler, local: Local) -> None:
            handler.send_response(200)
            handler.send_header("Content-Type", kind)
            if length is not None:
                handler.send_header("Content-Length", length)
            handler.end_headers()
            handler.wfile.write(data)  # then the server closes the connection (HTTP/1.0)
            local.stop.wait(hold)

        return respond

    cut_short = answer_of("application/json", "20", b'{"a":1}')
    with serve(cut_short) as srv:
        results = [nsc.request_json("GET", f"{srv.url}/v1/models", timeout=5),
                   nsc.request_stream(f"{srv.url}/v1/chat/completions", {}, timeout=5)]
    for result in results:
        assert (result.status, result.error_code, result.body) == (
            200, "connection_lost", b'{"a":1}'), result.error
        assert result.json_body() == {"a": 1}
    with serve(answer_of("application/json", None, b'{"a":1}')) as srv:
        whole = nsc.request_json("GET", f"{srv.url}/v1/models", timeout=5)
    assert (whole.status, whole.error, whole.body) == (200, None, b'{"a":1}')
    with serve(answer_of("text/event-stream", "1000", EVENT_1 + DONE, hold=10)) as srv:
        done = nsc.request_stream(f"{srv.url}/v1/chat/completions", {}, timeout=10)
    assert (done.status, done.error, done.body) == (200, None, EVENT_1 + DONE)
    assert done.elapsed_ms < 3000


def test_an_answer_that_ignores_stream_is_read_whole() -> None:
    completion = {"object": "chat.completion", "choices": [{"message": {"content": "hi"}}]}
    with serve(answer(200, completion)) as srv:
        result = nsc.request_stream(f"{srv.url}/v1/chat/completions", {"stream": True}, timeout=5)
    assert (result.status, result.error, result.content_type) == (200, None, "application/json")
    assert result.json_body() == completion


# --- private folders and files ------------------------------------------------------------------

posix_only = pytest.mark.skipif(not hasattr(os, "getuid"), reason="POSIX owners only")


def mode(path: Path) -> int:
    return path.stat().st_mode & 0o7777


def test_ensure_private_dir_creates_every_missing_folder_0700(tmp_path: Path) -> None:
    made = nsc.ensure_private_dir(tmp_path / "a" / "b" / "c")
    assert made == (tmp_path / "a" / "b" / "c").resolve()
    for folder in (tmp_path / "a", tmp_path / "a" / "b", tmp_path / "a" / "b" / "c"):
        assert mode(folder) == 0o700
    existing = tmp_path / "mine"
    existing.mkdir(mode=0o755)
    existing.chmod(0o755)
    assert nsc.ensure_private_dir(existing) == existing.resolve()
    assert mode(existing) == 0o755  # a folder of the user's own is left as it is


def test_ensure_private_dir_refuses_a_path_through_a_file(tmp_path: Path) -> None:
    (tmp_path / "file").write_text("x", encoding="utf-8")
    for target in (tmp_path / "file", tmp_path / "file" / "run"):
        with pytest.raises(nsc.ScriptError) as info:
            nsc.ensure_private_dir(target)
        assert (info.value.code, info.value.exit_code) == ("bad_input", 2)


@posix_only
def test_a_folder_other_users_can_write_is_refused(tmp_path: Path) -> None:
    shared = tmp_path / "shared"
    shared.mkdir()
    shared.chmod(0o777)
    for target in (shared, shared / "run1"):
        with pytest.raises(nsc.ScriptError) as info:
            nsc.ensure_private_dir(target)
        assert (info.value.code, info.value.exit_code) == ("out_not_private", nsc.EXIT_INPUT)
        assert f"{shared.resolve()} can be written by other users" in info.value.message
    assert not (shared / "run1").exists()  # nothing is created in a shared folder
    for write in (nsc.write_json_file, nsc.write_text_file):
        with pytest.raises(nsc.ScriptError):
            write(shared / "results.out", {} if write is nsc.write_json_file else "x")
    assert list(shared.iterdir()) == []


@posix_only
def test_a_sticky_folder_is_allowed(tmp_path: Path) -> None:
    sticky = tmp_path / "sticky"
    sticky.mkdir()
    sticky.chmod(0o1777)
    assert nsc.ensure_private_dir(sticky / "run") == (sticky / "run").resolve()


@posix_only
def test_a_symlink_of_another_user_on_the_way_is_refused(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Another user's symbolic link in a sticky folder such as /tmp can point --out at a folder
    of the user's own, and the files would go there; the old check followed it and looked only
    at the folder it found. Each entry of the path as written is read with lstat: a link must
    belong to the user or root. (The link is the user's own, reported as another user's.)"""
    target = tmp_path / "mine"
    target.mkdir(mode=0o700)
    link = tmp_path / "out"
    link.symlink_to(target)
    real_lstat = os.lstat

    def lstat(path: Any, *args: Any, **kwargs: Any) -> os.stat_result:
        info = real_lstat(path, *args, **kwargs)
        if os.fspath(path) != str(link):
            return info
        return os.stat_result((*info[:4], os.getuid() + 4242, *info[5:10]))

    monkeypatch.setattr(os, "lstat", lstat)
    for check, folder in ((nsc.check_private, link), (nsc.ensure_private_dir, link),
                          (nsc.ensure_private_dir, link / "run")):
        with pytest.raises(nsc.ScriptError) as info:
            check(folder)
        assert (info.value.code, info.value.exit_code) == ("out_not_private", nsc.EXIT_INPUT)
        assert f"{link} is a symbolic link of another user" in info.value.message
    with pytest.raises(nsc.ScriptError):
        nsc.write_json_file(link / "results.json", {"a": 1})
    assert list(target.iterdir()) == []
    monkeypatch.setattr(os, "lstat", real_lstat)
    assert nsc.ensure_private_dir(link / "run") == (target / "run").resolve()


@posix_only
def test_a_symlink_on_the_way_counts_the_folder_that_holds_it(tmp_path: Path) -> None:
    """Whoever can write the folder that holds a link can replace the link: a link of the
    user's own in a folder other users can write, with no sticky bit, is refused; in a sticky
    folder (such as /tmp) it is followed."""
    target = tmp_path / "mine"
    target.mkdir(mode=0o700)
    shared, sticky = tmp_path / "shared", tmp_path / "sticky"
    for folder, mode_bits in ((shared, 0o777), (sticky, 0o1777)):
        folder.mkdir()
        folder.chmod(mode_bits)
        (folder / "out").symlink_to(target)
    with pytest.raises(nsc.ScriptError) as info:
        nsc.ensure_private_dir(shared / "out")
    assert info.value.code == "out_not_private"
    assert f"{shared.resolve()} can be written by other users" in info.value.message
    assert nsc.ensure_private_dir(sticky / "out") == target.resolve()
    assert list(target.iterdir()) == []


@posix_only
def test_the_folder_written_in_is_the_one_checked(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A link on the way can change after the check: the file goes to the folder the last check
    resolved and passed, never to the one the link points to by the time of the write."""
    first = tmp_path / "first"
    first.mkdir(mode=0o700)
    link = tmp_path / "out"
    link.symlink_to(first)
    checked: list[Path] = []
    real = nsc.check_private

    def check_then_move(path: Path) -> Path:
        found = real(path)
        checked.append(Path(path).resolve())  # the folder this check saw
        moved = tmp_path / f"moved-{len(checked)}"
        moved.mkdir(mode=0o700)
        link.unlink()
        link.symlink_to(moved)  # another process moves the link right after the check
        return found

    monkeypatch.setattr(nsc, "check_private", check_then_move)
    nsc.write_json_file(link / "results.json", {"a": 1})
    assert link.resolve() != checked[-1]
    assert [path.name for path in checked[-1].iterdir()] == ["results.json"]
    assert list(link.resolve().iterdir()) == []


def fake_accounts(
    monkeypatch: pytest.MonkeyPatch,
    group: str,
    members: tuple[str, ...] = (),
    others: tuple[Any, ...] = (),
) -> None:
    """The account database as private_group reads it: the user "me" (this process's uid and
    gid), the group of that gid named `group` with `members`, and the `others` accounts."""
    import grp
    import pwd

    me = SimpleNamespace(pw_name="me", pw_uid=os.getuid(), pw_gid=os.getgid())
    monkeypatch.setattr(pwd, "getpwuid", lambda uid: me)
    monkeypatch.setattr(pwd, "getpwall", lambda: [me, *others])
    monkeypatch.setattr(grp, "getgrgid",
                        lambda gid: SimpleNamespace(gr_name=group, gr_mem=list(members)))


@pytest.fixture
def group_folder(tmp_path: Path) -> Path:
    """A folder of the user's primary group that its group can write (0770, no sticky bit)."""
    folder = tmp_path / "group"
    folder.mkdir()
    folder.chmod(0o770)
    if folder.stat().st_gid != os.getgid():
        pytest.skip("the folder did not get the user's primary group (a setgid parent)")
    return folder


@posix_only
def test_the_users_private_group_may_write(
    monkeypatch: pytest.MonkeyPatch, group_folder: Path
) -> None:
    """With user private groups (umask 002) a user's folders are 0775 in a group of the user
    alone: that group write lets no one else in."""
    fake_accounts(monkeypatch, "me")
    assert nsc.private_group(os.getgid())
    nsc.write_json_file(group_folder / "results.json", {"a": 1})
    assert json.loads((group_folder / "results.json").read_text("utf-8")) == {"a": 1}


@posix_only
@pytest.mark.parametrize(
    "accounts",
    [
        {"group": "users"},  # the primary group that every account shares
        {"group": "me", "members": ("me", "alice")},  # a second member
        {"group": "me", "others": (SimpleNamespace(pw_name="bob", pw_uid=4242, pw_gid=-1),)},
        {"group": None},  # a group the system cannot name
    ],
)
def test_a_folder_a_shared_group_can_write_is_refused(
    monkeypatch: pytest.MonkeyPatch, group_folder: Path, accounts: dict[str, Any]
) -> None:
    """A group write is refused unless the group is the user's private group: on a machine
    where every account's primary group is `users`, any of them could replace the files."""
    others = tuple(
        SimpleNamespace(**{**vars(account), "pw_gid": os.getgid()})
        for account in accounts.get("others", ())
    )
    if accounts["group"] is None:
        import grp

        def unknown(gid: int) -> Any:
            raise KeyError(gid)

        fake_accounts(monkeypatch, "me")
        monkeypatch.setattr(grp, "getgrgid", unknown)
    else:
        fake_accounts(monkeypatch, accounts["group"], accounts.get("members", ()), others)
    assert not nsc.private_group(os.getgid())
    with pytest.raises(nsc.ScriptError) as info:
        nsc.write_json_file(group_folder / "results.json", {"a": 1})
    assert info.value.code == "out_not_private"
    assert f"{group_folder.resolve()} can be written by other users" in info.value.message
    assert list(group_folder.iterdir()) == []


@posix_only
def test_a_folder_of_another_user_is_refused(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    owner = os.stat(tmp_path).st_uid
    if owner == 0:
        pytest.skip("a folder of root is trusted; run as a user")
    monkeypatch.setattr(os, "getuid", lambda: owner + 4242)
    with pytest.raises(nsc.ScriptError) as info:
        nsc.ensure_private_dir(tmp_path / "run")
    assert info.value.code == "out_not_private"
    assert "belongs to another user" in info.value.message
    assert not (tmp_path / "run").exists()


def test_writes_are_atomic_0600_and_replace_a_planted_symlink(tmp_path: Path) -> None:
    folder = tmp_path / "out"
    folder.mkdir(mode=0o700)
    victim = tmp_path / "victim.txt"
    victim.write_text("precious\n", encoding="utf-8")
    (folder / "results.json").symlink_to(victim)
    nsc.write_json_file(folder / "results.json", {"a": 1, "text": "café"})
    nsc.write_text_file(folder / "report.md", "# Report\n")
    assert victim.read_text(encoding="utf-8") == "precious\n"
    assert not (folder / "results.json").is_symlink()
    assert json.loads((folder / "results.json").read_text(encoding="utf-8")) == {
        "a": 1,
        "text": "café",
    }
    assert (folder / "report.md").read_text(encoding="utf-8") == "# Report\n"
    assert mode(folder / "results.json") == 0o600 and mode(folder / "report.md") == 0o600
    assert sorted(path.name for path in folder.iterdir()) == ["report.md", "results.json"]


def test_a_write_creates_its_folders_private(tmp_path: Path) -> None:
    path = tmp_path / "run" / "evidence" / "chat-1.json"
    nsc.write_json_file(path, {"ok": True})
    assert mode(tmp_path / "run") == 0o700 and mode(tmp_path / "run" / "evidence") == 0o700
    assert mode(path) == 0o600


def test_a_failed_write_leaves_no_temporary_file(tmp_path: Path) -> None:
    folder = tmp_path / "out"
    folder.mkdir(mode=0o700)
    (folder / "results.json").mkdir()  # a folder where the file should go
    with pytest.raises(OSError):
        nsc.write_json_file(folder / "results.json", {"a": 1})
    assert [path.name for path in folder.iterdir()] == ["results.json"]


def test_written_files_are_masked(tmp_path: Path, scrubber: nsc.Scrubber) -> None:
    scrubber.add(GATEWAY_KEY)
    nsc.write_json_file(
        tmp_path / "out" / "e.json", {"sent": f"Bearer {GATEWAY_KEY}", "k": NVAPI_KEY}
    )
    nsc.write_text_file(tmp_path / "out" / "r.md", f"key {GATEWAY_KEY} and {HF_KEY}\n")
    for path in (tmp_path / "out" / "e.json", tmp_path / "out" / "r.md"):
        text = path.read_text(encoding="utf-8")
        assert GATEWAY_KEY not in text and NVAPI_KEY not in text and HF_KEY not in text
    assert json.loads((tmp_path / "out" / "e.json").read_text(encoding="utf-8")) == {
        "sent": "Bearer ***",
        "k": "***",
    }


@pytest.mark.parametrize(
    ("model", "slug"),
    [
        ("meta/llama-3.1-8b-instruct", "meta-llama-3.1-8b-instruct"),
        ("openai/gpt-oss-20b", "openai-gpt-oss-20b"),
        ("Acme/Custom LLM 7B", "acme-custom-llm-7b"),
        ("../../etc/hostname", "etc-hostname"),
        ("...", "nim"),
        ("", "nim"),
        (None, "nim"),
    ],
)
def test_model_slug(model: str | None, slug: str) -> None:
    assert nsc.model_slug(model) == slug


def test_the_default_out_dir_is_in_the_cache_named_by_utc_time_and_model(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    cache = tmp_path / ".cache" / "nim-smoke-check"
    utc = datetime(2026, 11, 11, 9, 30, 5, tzinfo=timezone.utc)
    india = datetime(2026, 11, 11, 15, 0, 5, tzinfo=timezone(timedelta(hours=5, minutes=30)))
    expected = cache / "20261111T093005Z-meta-llama-3.1-8b-instruct"
    assert nsc.default_out_dir("meta/llama-3.1-8b-instruct", utc) == expected
    assert nsc.default_out_dir("meta/llama-3.1-8b-instruct", india) == expected
    assert nsc.default_out_dir(None, utc.replace(tzinfo=None)) == cache / "20261111T093005Z-nim"
    assert nsc.default_out_dir("../../etc/x", utc).parent == cache
    assert not cache.exists()  # computed, not created
