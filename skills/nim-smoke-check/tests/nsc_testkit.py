#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test kit for nim-smoke-check: a fake NIM, fake kubectl and docker, run_cli and assert_contract.

Not a test file and not a skill script. Stdlib only, so the fake tools and the serve mode run
without pytest.

  FakeNim(**knobs)      a threaded HTTP server on 127.0.0.1 that answers like a vLLM-backed (or
                        SGLang-backed) NIM; see the Knobs class for every knob
  make_fake_tools(...)  fake `kubectl` and `docker` in tmp_path/bin that answer from fixtures,
                        log every argv and exit 9 on anything outside the read-only allow-list
  run_cli(script, ...)  runs a skill script as SKILL.md does (python -I -B) and parses its output
  assert_contract(obj)  the {ok, state, next, facts, errors} output contract

Standalone (the eval harness): python3 nsc_testkit.py serve --port N --model ID [knobs]
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import dataclasses
import hashlib
import json
import os
import re
import shlex
import signal
import socket
import subprocess
import sys
import threading
import time
import traceback
from collections.abc import Mapping
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

TESTS = Path(__file__).resolve().parent
SKILL = TESTS.parent
SCRIPTS = SKILL / "scripts"
FIXTURES = TESTS / "fixtures"
CONTRACT_KEYS = {"ok", "state", "next", "facts", "errors"}
ERROR_KEYS = {"code", "message", "fix"}
CODE = re.compile(r"[a-z][a-z0-9_]*")
CREATED = 1762819200  # a fixed `created` time, so recorded answers compare equal

# vLLM's exact 400 texts when no tool parser is configured (vllm/renderers/online_renderer.py)
AUTO_NEEDS_PARSER = (
    '"auto" tool choice requires --enable-auto-tool-choice and --tool-call-parser to be set'
)
FORCED_NEEDS_PARSER = "tool_choice={desc} requires --tool-call-parser to be set"
CONTROL_TOKENS = "<|channel|>final<|message|>"
IMAGE_TOKENS = 256  # prompt tokens one image adds on the fake VLM

# --- fake keys, contract, run_cli ----------------------------------------------------------------


def fake_key() -> str:
    """A key in the hosted-key shape, built at run time so no key-like literal sits in the repo."""
    return "nvapi" + "-" + "q" * 30


def fake_token() -> str:
    """A gateway token with no known key shape: only value registration can hide it."""
    return "smoke" + "gate" + "z" * 24


def assert_contract(obj: Any) -> None:
    """The object follows the output contract {ok, state, next, facts, errors}."""
    assert isinstance(obj, dict), f"not a JSON object: {obj!r}"
    assert set(obj) == CONTRACT_KEYS, f"keys {sorted(obj)} != {sorted(CONTRACT_KEYS)}"
    assert isinstance(obj["ok"], bool), obj["ok"]
    assert isinstance(obj["state"], str) and obj["state"], obj["state"]
    assert isinstance(obj["next"], str) and obj["next"], obj["next"]
    assert isinstance(obj["facts"], dict), obj["facts"]
    assert isinstance(obj["errors"], list), obj["errors"]
    for item in obj["errors"]:
        assert isinstance(item, dict) and set(item) == ERROR_KEYS, item
        assert isinstance(item["code"], str) and CODE.fullmatch(item["code"]), item["code"]
        assert isinstance(item["message"], str) and item["message"], item
        assert isinstance(item["fix"], str) and item["fix"], item


CREDENTIAL_NAME = re.compile(r"KEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL", re.IGNORECASE)


def child_env(env: Mapping[str, str] | None = None) -> dict[str, str]:
    """os.environ without credential-named variables (a developer's real key never reaches a
    test run), plus `env`."""
    out = {k: v for k, v in os.environ.items() if not CREDENTIAL_NAME.search(k)}
    out.update(env or {})
    return out


def script_path(script: str | Path) -> Path:
    """A bare name is a file in scripts/; anything with a folder is taken as given."""
    path = Path(script)
    return path if path.is_absolute() or len(path.parts) > 1 else SCRIPTS / path


def run_cli(
    script: str | Path,
    *args: str,
    env: Mapping[str, str] | None = None,
    timeout: float = 60,
    cwd: str | Path | None = None,
    raw: bool = False,
    strict_exit: bool = True,
) -> tuple[int, Any, str]:
    """Run [sys.executable, -I, -B, script, *args] with stdin DEVNULL, as SKILL.md starts it.

    Returns (exit code, the one JSON object on stdout, stderr). The object must follow the
    contract, and `ok` must agree with the exit code (strict_exit). `raw` returns stdout as text
    instead (for --help). The child gets child_env(env)."""
    argv = [sys.executable, "-I", "-B", str(script_path(script)), *map(str, args)]
    proc = subprocess.run(
        argv,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        env=child_env(env),
        cwd=cwd,
        timeout=timeout,
        check=False,
    )
    if raw:
        return proc.returncode, proc.stdout, proc.stderr
    text = proc.stdout.strip()
    try:
        obj = json.loads(text)
    except ValueError:
        raise AssertionError(
            f"stdout is not one JSON object (exit {proc.returncode}): {proc.stdout!r}; "
            f"stderr: {proc.stderr}"
        ) from None
    assert_contract(obj)
    if strict_exit:
        assert obj["ok"] == (proc.returncode == 0), (
            f"ok={obj['ok']} but exit {proc.returncode}; stderr: {proc.stderr}"
        )
    return proc.returncode, obj, proc.stderr


def files_containing(root: str | Path, needle: str) -> list[Path]:
    """Every file under `root` whose bytes contain `needle` (to prove a key never landed)."""
    data = needle.encode()
    found = []
    for path in sorted(Path(root).rglob("*")):
        if path.is_file() and not path.is_symlink() and data in path.read_bytes():
            found.append(path)
    return found


# --- the fake NIM: knobs and records -------------------------------------------------------------

CHOICES: dict[str, tuple[Any, ...]] = {
    "backend": ("vllm", "sglang"),
    "health": ("ok", "loading", "missing", "bare"),
    "tools": (
        "ok", "no_parser", "in_content", "prose", "double_encoded", "wrong_args",
        "arguments_object", "invalid_json", "schema_mismatch", "in_reasoning", "wrong_name",
        "none_empty", "none_markup",
    ),
    "reasoning": ("none", "kwargs", "system", "effort", "always"),
    "reasoning_field": (None, "reasoning", "reasoning_content", "inline", "inline_close_only"),
    "off_bug": (None, "answer_in_reasoning", "answer_in_reasoning_stream"),
    "vlm": (False, True, "image_ignored"),
    "json": ("ok", "ignored", "invalid", "truncated", "in_reasoning"),
    "stream": ("ok", "ignored", "no_done", "cut", "error_event", "keepalive", "no_usage", "empty"),
    "tool_calls_empty": (None, "omit", "list", "null"),
}


@dataclass
class Knobs:
    """How the fake NIM behaves. Every knob is a plain attribute: a test may change it while the
    server runs (nim.health = "ok").

    model, root, owned_by, extra_models, max_model_len: the /v1/models cards. root defaults to
        model; owned_by to the backend name; an extra model is an id or a card dict (a LoRA card
        sets "parent").
    backend: vllm (message.reasoning, {"error": {...}} bodies, 400 without a tool parser) or
        sglang (reasoning_content, {"object": "error", ...} bodies, the call as text with 200).
    health: ok | loading (ready 503) | missing (404 for /v1/health/* and /health) | bare (404
        for /v1/health/*, vLLM's own /health answers 200). ready_after: seconds after start
        when loading turns ok.
    tools: ok | no_parser (vLLM 400 texts) | in_content (call markup in content) | prose |
        double_encoded | wrong_args (valid schema, wrong values) | arguments_object (arguments
        not a string) | invalid_json | schema_mismatch (a required key missing) | in_reasoning
        (markup in the reasoning field) | wrong_name | none_empty and none_markup (for
        tool_choice "none"). A named or "required" call gets a valid call even for prose,
        in_content and in_reasoning, as guided decoding does. parallel_calls: one call per city
        named in the prompt when the request allows parallel calls.
    reasoning: none | kwargs (chat_template_kwargs[reasoning_key]) | system ("/think",
        "/no_think", "detailed thinking on/off" in a system message, or "/think" and "/no_think"
        at the end of the last user message) | effort (reasoning_effort; "none" is off) | always
        (ignores every switch). reasoning_default: on or off when no switch is sent.
    reasoning_field: reasoning | reasoning_content | inline (<think>X</think>Y) |
        inline_close_only (X</think>Y); None picks the backend's field.
    off_bug: answer_in_reasoning (reasoning switched off: the answer lands in the reasoning
        field and content is null) | answer_in_reasoning_stream (the same, streams only).
    vlm: False (an image is a 400) | True (each image adds 256 prompt tokens) | image_ignored.
    json: ok (valid for the schema: strings "NIM", integers 2) | ignored (prose) | invalid |
        truncated (cut, finish_reason length) | in_reasoning (the JSON in the reasoning field).
    stream: ok | ignored (plain JSON) | no_done | cut (connection dropped mid-stream) |
        error_event (data: {"error": ...} with HTTP 200) | keepalive (": keep-alive" comment
        lines) | no_usage | empty (no content chunk).
    require_key: 401 on every route without "Authorization: Bearer <key>".
    redirect_to: every route answers 302 to the same path under this base URL.
    fail_concurrent: 503 for a POST when more than N POSTs are in flight.
    delay / get_delay: seconds before each POST / GET answer; chunk_delay: between SSE chunks.
    answer: replaces every plain answer ("" gives an empty completion).
    control_tokens: content starts with "<|channel|>final<|message|>".
    reject_fields: request fields that give a 400 (a switch the model rejects).
    tool_calls_empty: how a message without calls shows tool_calls: omit (vLLM), list ([]) or
        null (SGLang); None picks the backend's form.
    routes: canned answers that win over the simulation: path -> (status, body[, content
        type]) or a list of them, answered in order and then repeating the last. A dict or list
        body is sent as JSON, labelled application/json unless a content type is given.
    version, metadata: the /v1/version release and the /v1/metadata body (1.x public fields).
    """

    model: str = "meta/llama-3.1-8b-instruct"
    root: str | None = None
    owned_by: str | None = None
    backend: str = "vllm"
    extra_models: tuple[Any, ...] = ()
    max_model_len: int = 131072
    health: str = "ok"
    ready_after: float | None = None
    tools: str = "ok"
    parallel_calls: bool = True
    reasoning: str = "none"
    reasoning_default: bool = True
    reasoning_key: str = "enable_thinking"
    reasoning_field: str | None = None
    off_bug: str | None = None
    vlm: bool | str = False
    json: str = "ok"
    stream: str = "ok"
    require_key: str | None = None
    redirect_to: str | None = None
    fail_concurrent: int | None = None
    delay: float = 0.0
    get_delay: float = 0.0
    chunk_delay: float = 0.0
    answer: str | None = None
    control_tokens: bool = False
    reject_fields: tuple[str, ...] = ()
    tool_calls_empty: str | None = None
    routes: dict[str, Any] = field(default_factory=dict)
    version: str = "2.1.5"
    metadata: dict[str, Any] | None = None

    def check_knobs(self) -> None:
        """Refuse a misspelt knob value, so a typo cannot pass a test by accident."""
        for name, allowed in CHOICES.items():
            value = getattr(self, name)
            if value not in allowed or isinstance(value, bool) != any(
                isinstance(a, bool) and a == value for a in allowed
            ):
                raise ValueError(f"{name}={value!r}: expected one of {allowed}")

    @property
    def field_name(self) -> str:
        """Where reasoning goes: a message field or inline in content."""
        if self.reasoning_field:
            return self.reasoning_field
        return "reasoning_content" if self.backend == "sglang" else "reasoning"

    @property
    def reasoning_attr(self) -> str:
        """The backend's reasoning field, also used by off_bug when reasoning is inline."""
        if self.field_name in ("reasoning", "reasoning_content"):
            return self.field_name
        return "reasoning_content" if self.backend == "sglang" else "reasoning"


@dataclass
class Seen:
    """One request the fake NIM received. headers has lower-case keys; body is the parsed JSON
    (None when there was none or it did not parse); raw is the body as sent."""

    method: str
    path: str
    query: str
    headers: dict[str, str]
    body: Any
    raw: bytes
    started: float
    ended: float | None = None
    status: int | None = None


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True
    request_queue_size = 64  # the default 5 can drop a SYN when 8 requests connect at once
    nim: FakeNim

    def handle_error(self, request: Any, client_address: Any) -> None:
        """A client that hangs up mid-answer is normal here; keep the trace for debugging."""
        self.nim.errors.append(traceback.format_exc())


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server: _Server
    seen: Seen | None = None

    def log_message(self, format: str, *args: Any) -> None:
        return

    def do_GET(self) -> None:
        self.server.nim.handle(self, "GET")

    def do_POST(self) -> None:
        self.server.nim.handle(self, "POST")

    def do_PUT(self) -> None:
        self.server.nim.handle(self, "PUT")

    def do_DELETE(self) -> None:
        self.server.nim.handle(self, "DELETE")

    def do_PATCH(self) -> None:
        self.server.nim.handle(self, "PATCH")

    def do_HEAD(self) -> None:
        self.server.nim.handle(self, "HEAD")


# --- the fake NIM: server, routes and records ----------------------------------------------------


class FakeNim(Knobs):
    """A fake NIM on 127.0.0.1 (port 0 picks a free one). Use it as a context manager, or call
    start() and stop(). It records every request (requests, posts()), the most POSTs in flight
    at once (peak), and any trace of its own failures (errors)."""

    def __init__(self, model: str = Knobs.model, *, port: int = 0, **knobs: Any) -> None:
        super().__init__(model=model, **knobs)
        self.check_knobs()
        self.port = port
        self.requests: list[Seen] = []
        self.errors: list[str] = []
        self.peak = 0
        self.in_flight = 0
        self.lock = threading.Lock()
        self.calls: dict[str, int] = {}
        self.counter = 0
        self.started_at = time.monotonic()
        self._server: _Server | None = None
        self._thread: threading.Thread | None = None

    # lifecycle

    def start(self) -> FakeNim:
        server = _Server(("127.0.0.1", self.port), _Handler)
        server.nim = self
        self._server = server
        self.port = server.server_address[1]
        self.started_at = time.monotonic()
        self._thread = threading.Thread(target=server.serve_forever, args=(0.05,),
                                        daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None

    def __enter__(self) -> FakeNim:
        return self.start()

    def __exit__(self, *exc: Any) -> None:
        self.stop()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    # records

    def posts(self, path: str | None = None) -> list[Seen]:
        with self.lock:
            return [s for s in self.requests if s.method == "POST" and path in (None, s.path)]

    def gets(self, path: str | None = None) -> list[Seen]:
        with self.lock:
            return [s for s in self.requests if s.method == "GET" and path in (None, s.path)]

    def sent_fields(self) -> set[str]:
        """Every top-level field of every POST body: proves the request allowlist."""
        return {key for s in self.posts() if isinstance(s.body, dict) for key in s.body}

    def reset(self) -> None:
        with self.lock:
            self.requests.clear()
            self.errors.clear()
            self.calls.clear()
            self.peak = 0

    def next_id(self, prefix: str) -> str:
        with self.lock:
            self.counter += 1
            return f"{prefix}{self.counter:032x}"

    # request handling

    def handle(self, handler: _Handler, method: str) -> None:
        length = int(handler.headers.get("Content-Length") or 0)
        raw = handler.rfile.read(length) if length > 0 else b""
        try:
            body = json.loads(raw) if raw else None
        except ValueError:
            body = None
        path, _, query = handler.path.partition("?")
        seen = Seen(
            method=method,
            path=path,
            query=query,
            headers={k.lower(): v for k, v in handler.headers.items()},
            body=body,
            raw=raw,
            started=time.monotonic(),
        )
        with self.lock:
            self.requests.append(seen)
        handler.seen = seen
        try:
            seen.status = self.dispatch(handler, seen)
        except Exception:  # a bug in the fake: answer 500 and keep the trace
            self.errors.append(traceback.format_exc())
            seen.status = 500
            with contextlib.suppress(Exception):
                self.send_json(handler, 500, self.error_body("fake NIM failed", 500))
        finally:
            if seen.ended is None:
                seen.ended = time.monotonic()

    def dispatch(self, handler: _Handler, seen: Seen) -> int:
        if self.redirect_to:
            location = self.redirect_to.rstrip("/") + handler.path
            return self.send_raw(handler, 302, b"", "text/plain", {"Location": location})
        if self.require_key is not None:
            sent = seen.headers.get("authorization")
            if sent is None:
                text = b"Header of type `authorization` was missing"
                return self.send_raw(handler, 401, text, "text/plain")
            if sent != f"Bearer {self.require_key}":
                body = self.error_body("Invalid API key", 401, "AuthenticationError")
                return self.send_json(handler, 401, body)
        if seen.path in self.routes:
            return self.canned(handler, seen)
        if seen.method == "GET":
            if self.get_delay:
                time.sleep(self.get_delay)
            return self.get(handler, seen)
        if seen.method == "POST" and seen.path == "/v1/chat/completions":
            return self.post_chat(handler, seen)
        if seen.method == "POST":
            return self.send_json(handler, 404, {"detail": "Not Found"})
        return self.send_json(handler, 405, {"detail": "Method Not Allowed"})

    def canned(self, handler: _Handler, seen: Seen) -> int:
        answer = self.routes[seen.path]
        if isinstance(answer, list):
            with self.lock:
                index = self.calls.get(seen.path, 0)
                self.calls[seen.path] = index + 1
            answer = answer[min(index, len(answer) - 1)]
        status, body, *rest = answer
        if isinstance(body, (dict, list)):  # JSON, with the given content type if any
            kind = rest[0] if rest else "application/json"
            return self.send_raw(handler, status, json.dumps(body).encode(), kind)
        text = str(body)
        kind = rest[0] if rest else ("text/html" if text.lstrip().startswith("<") else "text/plain")
        return self.send_raw(handler, status, text.encode(), kind)

    def health_state(self) -> str:
        waited = time.monotonic() - self.started_at
        if self.health == "loading" and self.ready_after is not None and (
            waited >= self.ready_after
        ):
            return "ok"
        return self.health

    def cards(self) -> list[dict[str, Any]]:
        owner = self.owned_by or self.backend
        cards = []
        for index, extra in enumerate((self.model, *self.extra_models)):
            card: dict[str, Any] = {"id": extra} if isinstance(extra, str) else dict(extra)
            card.setdefault("object", "model")
            card.setdefault("created", CREATED)
            card.setdefault("owned_by", owner)
            card.setdefault("root", (self.root or self.model) if index == 0 else card["id"])
            if self.backend == "vllm":
                card.setdefault("parent", None)
            card.setdefault("max_model_len", self.max_model_len)
            if self.backend == "vllm":
                card.setdefault("permission", [self.permission(card["id"])])
            cards.append(card)
        return cards

    @staticmethod
    def permission(model_id: str) -> dict[str, Any]:
        digest = hashlib.sha256(model_id.encode()).hexdigest()[:32]
        return {
            "id": f"modelperm-{digest}", "object": "model_permission", "created": CREATED,
            "allow_create_engine": False, "allow_sampling": True, "allow_logprobs": True,
            "allow_search_indices": False, "allow_view": True, "allow_fine_tuning": False,
            "organization": "*", "group": None, "is_blocking": False,
        }

    def served_ids(self) -> list[str]:
        return [card["id"] for card in self.cards()]

    def get(self, handler: _Handler, seen: Seen) -> int:
        state = self.health_state()
        path = seen.path
        if path.startswith("/v1/health/") and state in ("missing", "bare"):
            return self.send_json(handler, 404, {"detail": "Not Found"})
        if path == "/v1/health/live":
            return self.send_json(
                handler, 200, {"object": "health.response", "message": "live", "status": "live"}
            )
        if path == "/v1/health/ready":
            if state == "loading":
                body = self.error_body("Service is not ready", 503, "ServiceUnavailableError")
                return self.send_json(handler, 503, body)
            body = {"object": "health.response", "message": "ready", "status": "ready"}
            return self.send_json(handler, 200, body)
        if path == "/health":
            status = {"ok": 200, "bare": 200, "loading": 503, "missing": 404}[state]
            return self.send_raw(handler, status, b"", "application/json")
        if path == "/v1/models":
            return self.send_json(handler, 200, {"object": "list", "data": self.cards()})
        if path.startswith("/v1/models/"):
            wanted = path[len("/v1/models/"):]
            for card in self.cards():
                if card["id"] == wanted:
                    return self.send_json(handler, 200, card)
            body = self.error_body(f"The model `{wanted}` does not exist.", 404, "NotFoundError")
            return self.send_json(handler, 404, body)
        if path == "/v1/version":
            return self.send_json(handler, 200, {"release": self.version, "api": "1.0.0"})
        if path == "/v1/metadata":
            return self.send_json(handler, 200, self.metadata_body())
        if path == "/v1/metrics":
            return self.send_raw(handler, 200, self.metrics_text().encode(), "text/plain")
        return self.send_json(handler, 404, {"detail": "Not Found"})

    def metadata_body(self) -> dict[str, Any]:
        if self.metadata is not None:
            return self.metadata
        profile = hashlib.sha256(f"{self.model}:{self.version}".encode()).hexdigest()
        return {
            "version": self.version,
            "selectedModelProfileId": profile,
            "modelInfo": [{"modelUrl": f"ngc://nim/{self.model}:{self.version}",
                           "shortName": self.model}],
            "assetInfo": [],
            "licenseInfo": {},
            "repository_override": "",
        }

    def metrics_text(self) -> str:
        prefix = "sglang" if self.backend == "sglang" else "vllm"
        with self.lock:
            running = self.in_flight
        lines = []
        for name, value in (("num_requests_running", running), ("num_requests_waiting", 0)):
            lines += [
                f"# TYPE {prefix}:{name} gauge",
                f'{prefix}:{name}{{model_name="{self.model}"}} {float(value)}',
            ]
        return "\n".join(lines) + "\n"

    # answers on the wire

    def error_body(self, message: str, code: int, kind: str = "BadRequestError",
                   param: str | None = None) -> dict[str, Any]:
        if self.backend == "sglang":
            return {"object": "error", "message": message, "type": kind, "param": param,
                    "code": code}
        return {"error": {"message": message, "type": kind, "param": param, "code": code}}

    def send_json(self, handler: _Handler, status: int, body: Any) -> int:
        return self.send_raw(handler, status, json.dumps(body).encode(), "application/json")

    def send_raw(self, handler: _Handler, status: int, data: bytes, kind: str,
                 extra: Mapping[str, str] | None = None) -> int:
        handler.send_response(status)
        if kind:
            handler.send_header("Content-Type", kind)
        handler.send_header("Content-Length", str(len(data)))
        for name, value in (extra or {}).items():
            handler.send_header(name, value)
        self.echo_request_id(handler)
        handler.end_headers()
        self.mark_done(handler, status)
        if handler.command != "HEAD":
            handler.wfile.write(data)
        return status

    @staticmethod
    def mark_done(handler: _Handler, status: int) -> None:
        """Record the status and end time before the last byte leaves, so a test that reads
        the record right after its request returns sees them."""
        seen = getattr(handler, "seen", None)
        if seen is not None:
            seen.status, seen.ended = status, time.monotonic()

    @staticmethod
    def echo_request_id(handler: _Handler) -> None:
        """NIM echoes X-Request-Id on inference answers."""
        rid = handler.headers.get("X-Request-Id")
        if rid and handler.command == "POST":
            handler.send_header("X-Request-Id", rid)

    # the chat simulation

    def post_chat(self, handler: _Handler, seen: Seen) -> int:
        with self.lock:
            self.in_flight += 1
            self.peak = max(self.peak, self.in_flight)
            over = self.fail_concurrent is not None and self.in_flight > self.fail_concurrent
        try:
            if over:
                body = self.error_body("The request queue is full.", 503, "ServiceUnavailableError")
                return self.send_json(handler, 503, body)
            if self.delay:
                time.sleep(self.delay)
            reply = self.complete(seen)
            if reply.status != 200:
                return self.send_json(handler, reply.status, reply.error)
            if seen.body.get("stream") is True and self.stream != "ignored":
                return self.send_stream(handler, seen.body, reply)
            return self.send_json(handler, 200, self.completion(reply))
        finally:
            with self.lock:
                self.in_flight -= 1

    def fail(self, status: int, message: str, kind: str | None = None,
             param: str | None = None) -> Reply:
        kind = kind or ("NotFoundError" if status == 404 else "BadRequestError")
        return Reply(status=status, error=self.error_body(message, status, kind, param))

    def complete(self, seen: Seen) -> Reply:
        """The answer to one chat request, or the error vLLM (or SGLang) would give."""
        kind = seen.headers.get("content-type", "").split(";")[0].strip().lower()
        if kind != "application/json":
            return self.fail(400, "Unsupported Media Type: Only 'application/json' is allowed")
        body = seen.body
        if not isinstance(body, dict):
            return self.fail(400, "The request body is not a JSON object")
        messages = body.get("messages")
        if not isinstance(messages, list) or not messages:
            return self.fail(400, "messages: Field required")
        model = body.get("model")
        if model not in self.served_ids():
            return self.fail(404, f"The model `{model}` does not exist.", param="model")
        for name in self.reject_fields:
            if name in body:
                return self.fail(400, f"{name} is not supported for this model")
        images, bad = image_parts(messages)
        if bad:
            return self.fail(400, f"Invalid image: {bad}")
        if images and self.vlm is False:
            return self.fail(400, f"{self.root or self.model} is not a multimodal model")
        tools = body.get("tools")
        choice = body.get("tool_choice")
        if tools is not None and not tools:
            return self.fail(400, "`tools` must not be an empty array")
        if choice is not None and not tools:
            return self.fail(400, "When using `tool_choice`, `tools` must be set.")
        if tools and choice is None:
            choice = "auto"
        named = None
        if isinstance(choice, dict):
            named = (choice.get("function") or {}).get("name")
            if named not in tool_names(tools):
                return self.fail(400, "The tool specified in `tool_choice` does not match any "
                                      "of the specified `tools`")
        if tools and choice != "none" and self.tools == "no_parser" and self.backend == "vllm":
            if choice == "auto":
                return self.fail(400, AUTO_NEEDS_PARSER)
            desc = f'function "{named}"' if named else f'"{choice}"'
            return self.fail(400, FORCED_NEEDS_PARSER.format(desc=desc))
        return self.generate(body, messages, images, choice, named)

    def generate(self, body: dict[str, Any], messages: list[Any], images: int, choice: Any,
                 named: str | None) -> Reply:
        prompt = last_user_text(messages)
        on = self.reasoning_wanted(body, messages)
        reply = Reply(model=body["model"], where=self.field_name, attr=self.reasoning_attr,
                      completion_id=self.next_id("chatcmpl-"))
        reasoning = reasoning_text(prompt) if on else None
        content: str | None = None
        calls: list[dict[str, Any]] | None = None
        finish = "stop"
        fmt = body.get("response_format")
        if isinstance(fmt, dict) and fmt.get("type") in ("json_schema", "json_object"):
            content, finish, into_reasoning = self.json_answer(fmt)
            if into_reasoning:
                reasoning, content = content, None
        elif body.get("tools") and choice != "none":
            reasoning, content, calls, finish = self.tool_answer(body, prompt, choice, named,
                                                                 reasoning)
        else:
            content = self.plain_answer(prompt, images)
            if body.get("tools") and self.tools == "none_empty":
                content = ""
            elif body.get("tools") and self.tools == "none_markup":
                content = f"{self.markup(body['tools'][0], prompt)} {content}"
        switchable = self.reasoning in ("kwargs", "system", "effort")
        if switchable and not on and self.off_bug and content:
            if self.off_bug == "answer_in_reasoning":
                reasoning, content, reply.where = content, None, self.reasoning_attr
            else:
                reply.stream_misroute = True
        if self.control_tokens and content:
            content = CONTROL_TOKENS + content
        budget = body.get("max_completion_tokens") or body.get("max_tokens") or 4096
        reply.reasoning, reply.content, reply.calls, reply.finish = clip(
            reasoning, content, calls, finish, int(budget)
        )
        if reply.calls:
            prefix = "call_" if self.backend == "sglang" else "chatcmpl-tool-"
            for call in reply.calls:
                call["id"] = self.next_id(prefix)
        reply.prompt_tokens = self.prompt_tokens(body, images)
        reply.completion_tokens = (
            words(reply.reasoning) + words(reply.content) + call_cost(reply.calls)
        )
        return reply

    def reasoning_wanted(self, body: dict[str, Any], messages: list[Any]) -> bool:
        if self.reasoning in ("none", "always"):
            return self.reasoning == "always"
        if self.reasoning == "kwargs":
            kwargs = body.get("chat_template_kwargs")
            if isinstance(kwargs, dict) and self.reasoning_key in kwargs:
                return bool(kwargs[self.reasoning_key])
            return self.reasoning_default
        if self.reasoning == "effort":
            effort = body.get("reasoning_effort")
            return self.reasoning_default if effort is None else effort != "none"
        for message in messages:
            if isinstance(message, dict) and message.get("role") == "system":
                text = message_text(message).lower()
                if "/no_think" in text or "detailed thinking off" in text:
                    return False
                if "/think" in text or "detailed thinking on" in text:
                    return True
        tail = last_user_text(messages).rstrip().lower()
        if tail.endswith("/no_think"):
            return False
        if tail.endswith("/think"):
            return True
        return self.reasoning_default

    def plain_answer(self, prompt: str, images: int) -> str:
        if self.answer is not None:
            return self.answer
        low = prompt.lower()
        if images:
            return "Red" if self.vlm is True else "I do not see an image."
        if "image" in low:
            return "I do not see an image."
        for key, text in ANSWERS:
            if key in low:
                return text
        return "OK"

    def json_answer(self, fmt: dict[str, Any]) -> tuple[str, str, bool]:
        """(content, finish_reason, into the reasoning field)."""
        spec = fmt.get("json_schema") if fmt.get("type") == "json_schema" else None
        schema = spec.get("schema") if isinstance(spec, dict) else None
        good = json.dumps(sample(schema) if isinstance(schema, dict) else {"name": "NIM",
                                                                            "version": 2})
        if self.json == "ignored":
            return "Sure. The name is NIM and the version is 2.", "stop", False
        if self.json == "invalid":
            broken = good[:-1] + "," + good[-1] if good[-1] in "}]" else "{" + good
            return broken, "stop", False
        if self.json == "truncated":
            return good[: max(1, len(good) // 2)], "length", False
        return good, "stop", self.json == "in_reasoning"

    def markup(self, tool: dict[str, Any], prompt: str) -> str:
        function = tool.get("function") or {}
        args = tool_args(function.get("parameters"), prompt, cities_in(prompt)[0])
        call = {"name": function.get("name", ""), "arguments": args}
        return f"<tool_call>\n{json.dumps(call)}\n</tool_call>"

    def tool_answer(self, body: dict[str, Any], prompt: str, choice: Any, named: str | None,
                    reasoning: str | None) -> tuple[str | None, str | None, Any, str]:
        """(reasoning, content, calls, finish_reason) for a request with tools."""
        tools = [t for t in body["tools"] if isinstance(t, dict)]
        tool = next((t for t in tools if (t.get("function") or {}).get("name") == named),
                    tools[0])
        function = tool.get("function") or {}
        mode = self.tools
        if mode == "no_parser":  # SGLang without a parser: the call comes back as text
            mode = "in_content"
        if (choice == "required" or named) and mode in ("prose", "in_content", "in_reasoning"):
            mode = "ok"  # guided decoding forces a valid call
        if mode == "in_content":
            return reasoning, self.markup(tool, prompt), None, "stop"
        if mode == "prose":
            return reasoning, "I cannot look that up right now.", None, "stop"
        if mode == "in_reasoning":
            text = f"{reasoning} {self.markup(tool, prompt)}" if reasoning else self.markup(
                tool, prompt)
            return text, None, None, "stop"
        cities = cities_in(prompt)
        parallel = (self.parallel_calls and named is None
                    and body.get("parallel_tool_calls", True) is not False)
        params = function.get("parameters") or {}
        calls = []
        for city in cities if parallel else cities[:1]:
            args = tool_args(params, prompt, city, wrong=mode == "wrong_args")
            if mode == "schema_mismatch":
                required = params.get("required") or list(args)
                args.pop(required[-1], None)
            text = json.dumps(args)
            arguments: Any = {
                "double_encoded": json.dumps(text),
                "arguments_object": args,
                "invalid_json": text[: len(text) // 2],
            }.get(mode, text)
            name = "search_web" if mode == "wrong_name" else function.get("name", "")
            calls.append({"name": name, "arguments": arguments})
        return reasoning, None, calls, "stop" if named else "tool_calls"

    def prompt_tokens(self, body: dict[str, Any], images: int) -> int:
        count = 3
        for message in body["messages"]:
            count += 4 + words(message_text(message))
        if body.get("tools"):
            count += words(json.dumps(body["tools"]))
        return count + (IMAGE_TOKENS * images if self.vlm is True else 0)

    # answers: plain JSON and server-sent events

    def usage(self, reply: Reply) -> dict[str, Any]:
        out: dict[str, Any] = {
            "prompt_tokens": reply.prompt_tokens,
            "total_tokens": reply.prompt_tokens + reply.completion_tokens,
            "completion_tokens": reply.completion_tokens,
            "prompt_tokens_details": None,
        }
        if self.backend == "sglang":
            out["reasoning_tokens"] = 0
        return out

    def message(self, reply: Reply) -> dict[str, Any]:
        msg: dict[str, Any] = {"role": "assistant", "content": None}
        if self.backend == "vllm":
            msg.update(refusal=None, annotations=None, audio=None, function_call=None)
        content = reply.content
        inline = reply.where in ("inline", "inline_close_only")
        if reply.reasoning is not None and inline:
            opening = "<think>" if reply.where == "inline" else ""
            cut = reply.finish == "length" and content is None
            content = f"{opening}{reply.reasoning}{'' if cut else '</think>'}{content or ''}"
        msg["content"] = content
        if reply.calls:
            msg["tool_calls"] = [self.wire_call(call, i) for i, call in enumerate(reply.calls)]
        else:
            empty = self.tool_calls_empty or ("null" if self.backend == "sglang" else "omit")
            if empty != "omit":
                msg["tool_calls"] = [] if empty == "list" else None
        field_name = reply.attr if inline else reply.where
        msg[field_name] = None if inline else reply.reasoning
        return msg

    def wire_call(self, call: dict[str, Any], index: int) -> dict[str, Any]:
        out: dict[str, Any] = {"id": call["id"], "type": "function",
                               "function": {"name": call["name"], "arguments": call["arguments"]}}
        if self.backend == "sglang":
            out["index"] = index
        return out

    def completion(self, reply: Reply) -> dict[str, Any]:
        choice: dict[str, Any] = {"index": 0, "message": self.message(reply), "logprobs": None,
                                  "finish_reason": reply.finish}
        body: dict[str, Any] = {"id": reply.completion_id, "object": "chat.completion",
                                "created": CREATED, "model": reply.model, "choices": [choice]}
        if self.backend == "sglang":
            choice["matched_stop"] = None
        else:
            choice.update(stop_reason=None, token_ids=None)
            body.update(service_tier=None, system_fingerprint=None)
        body["usage"] = self.usage(reply)
        if self.backend == "vllm":
            body.update(prompt_logprobs=None, prompt_token_ids=None, kv_transfer_params=None)
        return body

    def events(self, body: dict[str, Any], reply: Reply) -> list[str]:
        """The SSE events of one streamed answer, keep-alive comments included."""
        base = {"id": reply.completion_id, "object": "chat.completion.chunk",
                "created": CREATED, "model": reply.model}

        def chunk(delta: dict[str, Any], finish: str | None = None) -> str:
            choice = {"index": 0, "delta": delta, "logprobs": None, "finish_reason": finish}
            return "data: " + json.dumps({**base, "choices": [choice]}) + "\n\n"

        out = [chunk({"role": "assistant", "content": ""})]
        if self.stream != "empty":
            if reply.reasoning and reply.where in ("inline", "inline_close_only"):
                cut = reply.finish == "length" and reply.content is None
                text = ("<think>" if reply.where == "inline" else "") + reply.reasoning
                text += "" if cut else "</think>"
                out += [chunk({"content": piece}) for piece in pieces(text)]
            elif reply.reasoning:
                out += [chunk({reply.where: piece}) for piece in pieces(reply.reasoning)]
            content = reply.content
            for index, call in enumerate(reply.calls or []):
                first = {"index": index, "id": call["id"], "type": "function",
                         "function": {"name": call["name"], "arguments": ""}}
                args = call["arguments"]
                if not isinstance(args, str):
                    first["function"]["arguments"] = args
                out.append(chunk({"tool_calls": [first]}))
                for part in thirds(args) if isinstance(args, str) else []:
                    out.append(chunk({"tool_calls": [{"index": index,
                                                      "function": {"arguments": part}}]}))
            target = reply.attr if reply.stream_misroute else "content"
            out += [chunk({target: piece}) for piece in pieces(content or "")]
        if self.stream == "error_event":
            error = {"error": {"message": "fake engine failure mid-stream",
                               "type": "InternalServerError", "param": None, "code": 500}}
            out = out[:2] + ["data: " + json.dumps(error) + "\n\n"]
        else:
            out.append(chunk({}, reply.finish))
            options = body.get("stream_options")
            asked = isinstance(options, dict) and options.get("include_usage") is True
            if asked and self.stream != "no_usage":
                usage = {**base, "choices": [], "usage": self.usage(reply)}
                out.append("data: " + json.dumps(usage) + "\n\n")
        if self.stream not in ("no_done", "cut"):
            out.append("data: [DONE]\n\n")
        if self.stream == "keepalive":
            out = [line for event in out for line in (": keep-alive\n\n", event)]
        return out

    def send_stream(self, handler: _Handler, body: dict[str, Any], reply: Reply) -> int:
        handler.send_response(200)
        handler.send_header("Content-Type", "text/event-stream; charset=utf-8")
        handler.send_header("Cache-Control", "no-cache")
        handler.send_header("Transfer-Encoding", "chunked")
        self.echo_request_id(handler)
        handler.end_headers()
        events = self.events(body, reply)
        for index, event in enumerate(events):
            if self.stream == "cut" and index == max(1, len(events) // 2):
                self.mark_done(handler, 200)
                handler.wfile.flush()
                handler.close_connection = True
                with contextlib.suppress(OSError):
                    handler.connection.shutdown(socket.SHUT_RDWR)
                return 200
            data = event.encode()
            handler.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
            handler.wfile.flush()
            if self.chunk_delay:
                time.sleep(self.chunk_delay)
        self.mark_done(handler, 200)
        handler.wfile.write(b"0\r\n\r\n")
        handler.wfile.flush()
        return 200


# --- the fake NIM: pure helpers ------------------------------------------------------------------


@dataclass
class Reply:
    """One simulated answer. where: the reasoning field, or inline / inline_close_only; attr:
    the backend's reasoning field (where the off bugs put the answer)."""

    status: int = 200
    error: Any = None
    model: str = ""
    completion_id: str = ""
    where: str = "reasoning"
    attr: str = "reasoning"
    reasoning: str | None = None
    content: str | None = None
    calls: list[dict[str, Any]] | None = None
    finish: str = "stop"
    prompt_tokens: int = 0
    completion_tokens: int = 0
    stream_misroute: bool = False


ANSWERS = (
    ("single word ok", "OK"),
    ("capital of france", "Paris"),
    ("pens", "8"),
    ("2 + 2", "4"),
)
CITIES = ("Paris", "Tokyo", "London", "Berlin", "Santa Clara")


def message_text(message: Any) -> str:
    """The text of a message: a string content, or its text parts joined."""
    content = message.get("content") if isinstance(message, dict) else None
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            str(part.get("text", "")) for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        )
    return ""


def last_user_text(messages: list[Any]) -> str:
    for message in reversed(messages):
        if isinstance(message, dict) and message.get("role") == "user":
            return message_text(message)
    return ""


def tool_names(tools: Any) -> list[str]:
    return [
        str((tool.get("function") or {}).get("name"))
        for tool in tools or [] if isinstance(tool, dict)
    ]


def image_parts(messages: list[Any]) -> tuple[int, str | None]:
    """(how many image_url parts, the first problem with one of them or None)."""
    count = 0
    for message in messages:
        content = message.get("content") if isinstance(message, dict) else None
        for part in content if isinstance(content, list) else []:
            if not (isinstance(part, dict) and part.get("type") == "image_url"):
                continue
            count += 1
            image = part.get("image_url")
            url = image.get("url", "") if isinstance(image, dict) else ""
            if url.startswith(("http://", "https://")):
                continue
            match = re.fullmatch(r"data:(image/[\w.+-]+);base64,(.*)", str(url), re.DOTALL)
            if not match:
                return count, "not an http(s) or data:image URL"
            try:
                data = base64.b64decode(match.group(2), validate=True)
            except ValueError:
                return count, "the base64 data does not decode"
            if match.group(1) == "image/png" and not data.startswith(b"\x89PNG\r\n\x1a\n"):
                return count, "the data is not a PNG"
    return count, None


def reasoning_text(prompt: str) -> str:
    low = prompt.lower()
    if "pens" in low:
        return ("Four pens at 3 dollars each cost 12 dollars. She pays with 20 dollars, so the "
                "change is 20 minus 12, which is 8.")
    if "weather" in low:
        return "The user wants the current weather, so I should call the tool with the city."
    return "The user asks a short question. I will read it carefully and then answer."


def sample(schema: Any) -> Any:
    """A value valid for a simple JSON schema: strings "NIM", integers 2, the first enum."""
    if not isinstance(schema, dict):
        return "NIM"
    if schema.get("enum"):
        return schema["enum"][0]
    if "const" in schema:
        return schema["const"]
    kind = schema.get("type")
    if isinstance(kind, list):
        kind = next((k for k in kind if k != "null"), "null")
    if kind == "object" or "properties" in schema:
        props = schema.get("properties") or {}
        return {name: sample(sub) for name, sub in props.items()}
    if kind == "array":
        items = schema.get("items")
        return [sample(items)] if isinstance(items, dict) else []
    return {"integer": 2, "number": 2, "boolean": True, "null": None}.get(str(kind), "NIM")


def cities_in(prompt: str) -> list[str]:
    """The known cities a prompt names, in order; Paris when it names none."""
    low = prompt.lower()
    found = sorted((low.find(c.lower()), c) for c in CITIES if c.lower() in low)
    return [c for _, c in found] or ["Paris"]


def tool_args(params: Any, prompt: str, city: str, wrong: bool = False) -> dict[str, Any]:
    """Arguments for a tool's parameters: the city, the enum value the prompt names (else the
    first), a sample for the rest. `wrong` keeps the schema but changes the values."""
    props = params.get("properties") if isinstance(params, dict) else None
    low = prompt.lower()
    out: dict[str, Any] = {}
    for key, sub in (props or {}).items():
        sub = sub if isinstance(sub, dict) else {}
        enum = sub.get("enum")
        if enum:
            said = [v for v in enum if isinstance(v, str) and v.lower() in low]
            value = said[0] if said else enum[0]
            if wrong:
                value = next((v for v in enum if v != value), value)
        elif key in ("city", "location"):
            value = "London" if wrong else city
        else:
            value = sample(sub)
        out[key] = value
    return out


def words(text: str | None) -> int:
    """Tokens, as the fake counts them: whitespace-separated words."""
    return len(text.split()) if text else 0


def cut_words(text: str, count: int) -> str:
    return "".join(re.findall(r"\S+\s*", text)[:count]).rstrip()


def call_cost(calls: list[dict[str, Any]] | None) -> int:
    return sum(4 + words(json.dumps(c["arguments"])) for c in calls or [])


def clip(reasoning: str | None, content: str | None, calls: Any, finish: str,
         budget: int) -> tuple[str | None, str | None, Any, str]:
    """Spend max_tokens as an engine does: reasoning first, then the calls or the answer."""
    used = words(reasoning)
    if used > budget:
        return cut_words(reasoning or "", budget), None, None, "length"
    if calls and used + call_cost(calls) > budget:
        return reasoning, None, None, "length"
    if content and used + words(content) > budget:
        return reasoning, cut_words(content, budget - used) or None, calls, "length"
    return reasoning, content, calls, finish


def pieces(text: str) -> list[str]:
    """Stream deltas: one word each, whitespace kept, so the joined deltas equal the text."""
    return re.findall(r"\S+\s*|\s+", text)


def thirds(text: str) -> list[str]:
    """Tool-call arguments arrive in fragments."""
    size = max(1, -(-len(text) // 3))
    return [text[i:i + size] for i in range(0, len(text), size)]


# --- fake kubectl and docker: the allow-list, fixtures and output --------------------------------

# Flags that take the next word as their value; any other flag is a switch. Enough to find the
# command words, which come first among the words that are not flags.
VALUE_FLAGS = {
    "kubectl": {
        "--context", "-n", "--namespace", "--kubeconfig", "--cluster", "--user", "-s", "--server",
        "--request-timeout", "--as", "--as-group", "--token", "--certificate-authority",
        "--client-certificate", "--client-key", "--tls-server-name", "--cache-dir", "-v", "--v",
        "-o", "--output", "-l", "--selector", "-c", "--container", "--field-selector",
        "--pod-running-timeout", "--address", "--subresource", "-f", "--filename", "--sort-by",
    },
    "docker": {
        "--context", "-c", "--host", "-H", "--config", "--log-level", "-l", "--tlscacert",
        "--tlscert", "--tlskey", "--format", "-f", "--type",
    },
}
# The scope flags dropped before a fixture key is compared (exact keys may still name them).
SCOPE_FLAGS = {"kubectl": {"--context", "-n", "--namespace"}, "docker": {"--context", "-c"}}
DOCKER_ALLOWED = (("container", "inspect"), ("image", "inspect"), ("context", "inspect"))


@dataclass
class ToolReply:
    """What a fake tool prints for one fixture: stdout, stderr and the exit code."""

    stdout: str = ""
    stderr: str = ""
    exit: int = 0


@dataclass
class PortForwardReply:
    """A fake `kubectl port-forward`: it listens on 127.0.0.1 (the local port of the argv, or a
    free one), prints kubectl's "Forwarding from 127.0.0.1:N -> M" and "[::1]:N" lines (only
    127.0.0.1 is served) and "Handling connection for N" per connection, and relays each
    connection to `target` until SIGINT. With lost_after=K, the first `lose_starts` runs serve
    K connections, then drop the next one and exit 1 with "error: lost connection to pod" (a pod
    restart). `serve` (knobs) runs its own FakeNim instead of relaying."""

    target: str | None = None
    remote_port: int = 8000
    lost_after: int | None = None
    lose_starts: int = 1
    delay: float = 0.0
    serve: dict[str, Any] | None = None


def port_forward_to(target: Any = None, **options: Any) -> PortForwardReply:
    """A port-forward fixture that relays to a FakeNim, a base URL, "host:port" or a port."""
    if isinstance(target, FakeNim):
        target = f"127.0.0.1:{target.port}"
    elif isinstance(target, int):
        target = f"127.0.0.1:{target}"
    elif isinstance(target, str) and "://" in target:
        target = target.split("://", 1)[1].split("/", 1)[0]
    return PortForwardReply(target=target, **options)


def can_i(allowed: bool = True) -> ToolReply:
    """`kubectl auth can-i` prints yes (exit 0) or no (exit 1)."""
    return ToolReply("yes\n") if allowed else ToolReply("no\n", exit=1)


@dataclass
class InTurn:
    """Replies given in turn to the calls one fixture key answers; the last one then repeats
    (a pod that turns Ready on the third read, for example)."""

    replies: tuple[Any, ...]


def in_turn(*replies: Any) -> InTurn:
    """A fixture whose N-th matching call gets the N-th reply (any fixture form but another
    in_turn), and every later call the last one."""
    assert replies and not any(isinstance(reply, InTurn) for reply in replies), replies
    return InTurn(tuple(replies))


def plain_words(argv: list[str], tool: str) -> list[str]:
    """The words of an argv that are neither flags nor flag values."""
    out: list[str] = []
    skip = False
    for token in argv:
        if skip:
            skip = False
        elif token.startswith("-") and token != "-":
            skip = "=" not in token and token in VALUE_FLAGS[tool]
        else:
            out.append(token)
    return out


def output_format(argv: list[str]) -> str | None:
    """kubectl's -o / --output value, in any of its spellings."""
    for i, token in enumerate(argv):
        for flag in ("-o", "--output"):
            if token == flag and i + 1 < len(argv):
                return argv[i + 1]
            if token.startswith(flag + "="):
                return token[len(flag) + 1:]
        if token.startswith("-o") and not token.startswith("--") and len(token) > 2:
            return token[2:].lstrip("=")
    return None


def flag_values(argv: list[str], *flags: str) -> list[str]:
    """Every value a kubectl argv gives one of `flags`, in any spelling: --flag VALUE,
    --flag=VALUE and, for a one-letter flag, -fVALUE. The values of other flags are skipped,
    and "--" ends the flags."""
    short = tuple(flag for flag in flags if len(flag) == 2)
    out: list[str] = []
    skip = False
    for i, token in enumerate(argv):
        if skip:
            skip = False
            continue
        if token == "--":
            break
        name, eq, value = token.partition("=")
        if name in flags:
            if eq:
                out.append(value)
            elif i + 1 < len(argv):
                out.append(argv[i + 1])
                skip = True
        elif short and token.startswith(short) and len(token) > 2:
            out.append(token[2:])
        else:
            skip = not eq and token in VALUE_FLAGS["kubectl"]
    return out


def docker_format(argv: list[str]) -> str | None:
    for i, token in enumerate(argv):
        if token in ("--format", "-f") and i + 1 < len(argv):
            return argv[i + 1]
        if token.startswith("--format="):
            return token[len("--format="):]
    return None


def allowed(tool: str, argv: list[str]) -> tuple[bool, str]:
    """The read-only allow-list of spec section 2.3, kept apart from the skill's own copy so the
    fake is an independent check. kubectl get of Secrets, and get --raw, are refused too."""
    found = plain_words(argv, tool)
    if tool == "docker":
        if found[:1] == ["port"] or tuple(found[:2]) in DOCKER_ALLOWED:
            return True, " ".join(found[:2])
        return False, " ".join(found[:2]) or "no command"
    verb = found[0] if found else ""
    if verb == "get" and any(t == "--raw" or t.startswith("--raw=") for t in argv):
        return False, "get --raw"  # any API path, Secrets included
    if verb == "get":
        kinds = found[1] if len(found) > 1 else ""
        if any(k.split("/")[0].split(".")[0] in ("secret", "secrets") for k in kinds.split(",")):
            return False, "get secrets"
        return True, "get"
    if verb == "auth":
        return found[1:2] == ["can-i"], " ".join(found[:2])
    if verb == "port-forward":
        return True, verb
    if verb == "config":
        ok = found[1:2] == ["get-contexts"] and output_format(argv) == "name"
        return ok, " ".join(found[:2]) + ("" if ok else " (only -o name is allowed)")
    return False, verb or "no command"


def without_scope(tokens: list[str], tool: str) -> list[str]:
    out: list[str] = []
    skip = False
    for token in tokens:
        if skip:
            skip = False
        elif token in SCOPE_FLAGS[tool]:
            skip = True
        elif not any(token.startswith(f + "=") for f in SCOPE_FLAGS[tool]):
            out.append(token)
    return out


def subsequence(needle: list[str], hay: list[str]) -> bool:
    it = iter(hay)
    return all(any(token == other for other in it) for token in needle)


def find_fixture(entries: list[dict[str, Any]], argv: list[str], tool: str) -> dict | None:
    """Exact argv first, then (for a key that names no --context or -n) the argv without
    them, then the longest key whose words all appear in order (so "get nimservice llama"
    answers any -o spelling; a key that names a context only matches that context); "*"
    answers the rest."""
    bare = without_scope(argv, tool)
    for entry in entries:
        if entry["tokens"] == argv:
            return entry
    for entry in entries:
        if entry["tokens"] == entry["bare"] == bare:
            return entry
    best = None
    for entry in entries:
        scoped = entry["tokens"] != entry["bare"]
        hay = argv if scoped else bare
        longer = best is None or len(entry["tokens"]) > len(best["tokens"])
        if entry["key"] != "*" and longer and subsequence(entry["tokens"], hay):
            best = entry
    return best or next((e for e in entries if e["key"] == "*"), None)


def show_value(value: Any, go: bool = False) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "<no value>" if go else ""
    if isinstance(value, str):
        return value
    if isinstance(value, (dict, list)) and not go:
        return json.dumps(value, separators=(",", ":"))
    if isinstance(value, dict):
        return "map[" + " ".join(f"{k}:{show_value(v, go)}" for k, v in sorted(value.items())) + "]"
    if isinstance(value, list):
        return "[" + " ".join(show_value(v, go) for v in value) + "]"
    return str(value)


def walk(value: Any, path: str) -> list[Any]:
    """The values at a kubectl JSONPath path: .a.b, [N] and [*]."""
    values = [value]
    for name, index in re.findall(r"\.([^.\[\]]+)|\[(\*|-?\d+)\]", path):
        found = []
        for item in values:
            if name and isinstance(item, dict) and name in item:
                found.append(item[name])
            elif index == "*" and isinstance(item, list):
                found.extend(item)
            elif index and index != "*" and isinstance(item, list):
                with contextlib.suppress(IndexError):
                    found.append(item[int(index)])
        values = found
    return values


def render_kubectl(value: Any, argv: list[str]) -> str:
    fmt = output_format(argv)
    if fmt in (None, "json"):
        return json.dumps(value, indent=4) + "\n"
    if fmt == "name":
        items = value.get("items") if isinstance(value, dict) and "items" in value else [value]
        names = [f"{str(i.get('kind', 'object')).lower()}/{i.get('metadata', {}).get('name', '')}"
                 for i in items if isinstance(i, dict)]
        return "".join(name + "\n" for name in names)
    if fmt.startswith("jsonpath="):
        template = fmt[len("jsonpath="):].strip("'")
        return re.sub(
            r"\{([^{}]*)\}",
            lambda m: " ".join(show_value(v) for v in walk(value, m.group(1).strip())),
            template,
        )
    raise ValueError(f"-o {fmt} is not supported by the fake; use json, name or jsonpath")


def render_docker(value: Any, argv: list[str]) -> str:
    template = docker_format(argv)
    if template is None:
        return json.dumps(value, indent=4) + "\n"

    def action(match: re.Match[str], item: Any) -> str:
        expr = match.group(1).strip()
        as_json = expr.startswith("json ")
        path = expr[5:].strip() if as_json else expr
        if not path.startswith("."):
            raise ValueError(f"{{{{{expr}}}}} is not supported by the fake")
        found = walk(item, path) if path != "." else [item]
        result = found[0] if found else None
        if as_json:
            return json.dumps(result, separators=(",", ":"))
        return show_value(result, go=True)

    items = value if isinstance(value, list) else [value]
    lines = [re.sub(r"\{\{(.*?)\}\}", lambda m, it=item: action(m, it), template)
             for item in items]
    return "".join(line + "\n" for line in lines)


# --- fake kubectl and docker: the programs -------------------------------------------------------

SHIM = '''#!{python} -I
"""Fake {tool} for the nim-smoke-check tests (written by nsc_testkit.make_fake_tools)."""
import sys
sys.dont_write_bytecode = True
sys.path.insert(0, {tests!r})
import nsc_testkit
sys.exit(nsc_testkit.fake_tool_main({tool!r}, {fixtures!r}, {log!r}, sys.argv[1:]))
'''


def fake_tool_main(tool: str, fixtures: str, log: str, argv: list[str]) -> int:
    """The body of a fake kubectl or docker: log the argv, refuse (exit 9) anything outside the
    allow-list, else answer from the matching fixture (exit 1 when none matches)."""
    entries = json.loads(Path(fixtures).read_text(encoding="utf-8"))
    ok, verb = allowed(tool, argv)
    entry = find_fixture(entries, argv, tool) if ok else None
    record = {"tool": tool, "argv": argv, "allowed": ok, "verb": verb,
              "fixture": entry["key"] if entry else None, "pid": os.getpid()}
    with open(log, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")
    if not ok:
        sys.stderr.write(f"fake {tool}: refused, not read-only ({verb}): {shlex.join(argv)}\n")
        return 9
    if entry is None:
        sys.stderr.write(f"fake {tool}: no fixture for: {shlex.join(argv)}\n")
        return 1
    reply = entry["reply"]
    if "turns" in reply:
        counter = Path(log).with_name("turns-" + hashlib.sha256(
            f"{tool} {entry['key']}".encode()).hexdigest()[:12])
        turn = int(counter.read_text()) if counter.exists() else 0
        counter.write_text(str(turn + 1))
        reply = reply["turns"][min(turn, len(reply["turns"]) - 1)]
    if "port_forward" in reply:
        if verb != "port-forward":
            sys.stderr.write(f"fake {tool}: a port-forward fixture answered {verb}\n")
            return 1
        return port_forward(reply["port_forward"], argv, entry["key"], log)
    if "json" in reply:
        try:
            render = render_kubectl if tool == "kubectl" else render_docker
            sys.stdout.write(render(reply["json"], argv))
        except ValueError as exc:
            sys.stderr.write(f"fake {tool}: {exc}\n")
            return 1
    else:
        sys.stdout.write(reply["stdout"])
    sys.stderr.write(reply["stderr"])
    return int(reply["exit"])


def port_forward(spec: dict[str, Any], argv: list[str], key: str, log: str) -> int:
    """Listen, print kubectl's lines, relay until SIGINT (exit 0) or a lost connection (1)."""
    found = plain_words(argv, "kubectl")
    ports = [w for w in found[2:] if re.fullmatch(r"\d*:[\w-]+|\d+", w)]
    if not ports:
        sys.stderr.write("error: at least 1 PORT is required for port-forward\n")
        return 1
    local_text, _, remote_text = ports[0].rpartition(":")
    local = int(local_text) if local_text else (0 if ":" in ports[0] else int(remote_text))
    remote = spec.get("remote_port") or (int(remote_text) if remote_text.isdigit() else 8000)
    if spec.get("delay"):
        time.sleep(spec["delay"])
    counter = Path(log).with_name("pf-starts-" + hashlib.sha256(key.encode()).hexdigest()[:12])
    starts = int(counter.read_text()) + 1 if counter.exists() else 1
    counter.write_text(str(starts))
    losing = spec.get("lost_after") is not None and starts <= spec.get("lose_starts", 1)
    if not (spec.get("serve") or spec.get("target")):
        sys.stderr.write("fake kubectl: the port-forward fixture has no target\n")
        return 1
    nim = FakeNim(**spec["serve"]).start() if spec.get("serve") else None
    host, _, port_text = (f"127.0.0.1:{nim.port}" if nim else spec["target"]).rpartition(":")
    target = (host.strip("[]"), int(port_text))
    try:
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.bind(("127.0.0.1", local))
    except OSError as exc:
        sys.stderr.write(f"Unable to listen on port {local}: {exc}\n"
                         f"error: unable to listen on any of the requested ports: "
                         f"[{{{local} {remote}}}]\n")
        return 1
    listener.listen(16)
    listener.settimeout(0.2)
    port = listener.getsockname()[1]
    print(f"Forwarding from 127.0.0.1:{port} -> {remote}", flush=True)
    print(f"Forwarding from [::1]:{port} -> {remote}", flush=True)
    failed: list[str] = []
    handled = 0
    try:
        while not failed:
            try:
                conn, _ = listener.accept()
            except TimeoutError:
                continue
            handled += 1
            print(f"Handling connection for {port}", flush=True)
            if losing and handled > spec["lost_after"]:
                conn.close()
                failed.append("connection reset: the pod restarted")
                break
            threading.Thread(target=relay, args=(conn, target, failed), daemon=True).start()
        sys.stderr.write(f"E1111 portforward.go] an error occurred forwarding {port} -> "
                         f"{remote}: {failed[0]}\nerror: lost connection to pod\n")
        return 1
    except KeyboardInterrupt:
        return 0
    finally:
        listener.close()
        if nim:
            nim.stop()


def relay(conn: socket.socket, target: tuple[str, int], failed: list[str]) -> None:
    """Copy bytes both ways between a client and the target until both sides close."""
    try:
        upstream = socket.create_connection(target, timeout=5)
    except OSError as exc:
        failed.append(f"failed to connect to {target[0]}:{target[1]}: {exc}")
        conn.close()
        return
    upstream.settimeout(None)

    def pump(src: socket.socket, dst: socket.socket) -> None:
        try:
            while data := src.recv(65536):
                dst.sendall(data)
        except OSError:
            pass
        finally:
            with contextlib.suppress(OSError):
                dst.shutdown(socket.SHUT_WR)

    back = threading.Thread(target=pump, args=(upstream, conn), daemon=True)
    back.start()
    pump(conn, upstream)
    back.join(timeout=60)
    conn.close()
    upstream.close()


def wire_reply(value: Any) -> dict[str, Any]:
    if isinstance(value, InTurn):
        return {"turns": [wire_reply(reply) for reply in value.replies]}
    if isinstance(value, ToolReply):
        return {"stdout": value.stdout, "stderr": value.stderr, "exit": value.exit}
    if isinstance(value, PortForwardReply):
        return {"port_forward": dataclasses.asdict(value)}
    if isinstance(value, str):
        return {"stdout": value if not value or value.endswith("\n") else value + "\n",
                "stderr": "", "exit": 0}
    return {"json": value, "stderr": "", "exit": 0}


@dataclass
class FakeTools:
    """Fake kubectl and docker in `bin`. env() is the PATH for a run (only these two programs);
    calls() reads the log of every argv either program received."""

    bin: Path
    data: Path
    log: Path

    def env(self, extra: Mapping[str, str] | None = None) -> dict[str, str]:
        return {"PATH": str(self.bin), **(extra or {})}

    def calls(self, tool: str | None = None) -> list[dict[str, Any]]:
        lines = self.log.read_text(encoding="utf-8").splitlines() if self.log.exists() else []
        records = [json.loads(line) for line in lines if line.strip()]
        return [r for r in records if tool in (None, r["tool"])]

    def argvs(self, tool: str) -> list[list[str]]:
        return [r["argv"] for r in self.calls(tool)]

    def refused(self) -> list[dict[str, Any]]:
        return [r for r in self.calls() if not r["allowed"]]

    def assert_read_only(self) -> None:
        assert not self.refused(), f"commands outside the read-only allow-list: {self.refused()}"

    def assert_scoped(self, context: str, namespace: str) -> None:
        """Every kubectl call names --context CONTEXT and -n NAMESPACE, and no other context or
        namespace: kubectl uses the last value of a repeated flag (config get-contexts aside)."""
        for argv in self.argvs("kubectl"):
            if plain_words(argv, "kubectl")[:2] == ["config", "get-contexts"]:
                continue
            contexts = flag_values(argv, "--context")
            namespaces = flag_values(argv, "-n", "--namespace")
            assert context in contexts, f"no --context {context}: {argv}"
            assert set(contexts) == {context}, f"--context {context} and another context: {argv}"
            assert namespace in namespaces, f"no -n {namespace}: {argv}"
            assert set(namespaces) == {namespace}, (
                f"-n {namespace} and another namespace: {argv}"
            )


def make_fake_tools(
    tmp_path: Path,
    kubectl_fixtures: Mapping[str, Any] | None = None,
    docker_fixtures: Mapping[str, Any] | None = None,
) -> FakeTools:
    """Write fake `kubectl` and `docker` into tmp_path/bin and return FakeTools.

    A fixture maps an argv key (the words after the program name; a leading program name is
    allowed) to: a str (printed), a dict or list (printed as kubectl -o json / jsonpath / name,
    or as docker JSON / --format), a ToolReply (stdout, stderr, exit), port_forward_to(...), or
    in_turn(...) of those (one per call, the last repeating).
    Both programs always exist, so a test never reaches a real cluster or daemon. Calling again
    rewrites the fixtures."""
    assert not re.search(r"\s", sys.executable), "a shebang cannot hold a path with spaces"
    tools = FakeTools(bin=tmp_path / "bin", data=tmp_path / "fake-tools",
                      log=tmp_path / "fake-tools" / "calls.jsonl")
    tools.bin.mkdir(parents=True, exist_ok=True)
    tools.data.mkdir(parents=True, exist_ok=True)
    tools.log.touch()
    for counter in tools.data.glob("turns-*"):  # new fixtures: in_turn starts again
        counter.unlink()
    for tool, fixtures in (("kubectl", kubectl_fixtures), ("docker", docker_fixtures)):
        entries = []
        for key, value in (fixtures or {}).items():
            tokens = shlex.split(key)
            if tokens[:1] == [tool]:
                tokens = tokens[1:]
            entries.append({"key": key, "tokens": tokens, "bare": without_scope(tokens, tool),
                            "reply": wire_reply(value)})
        data = tools.data / f"{tool}.json"
        data.write_text(json.dumps(entries), encoding="utf-8")
        program = tools.bin / tool
        program.write_text(
            SHIM.format(python=sys.executable, tool=tool, tests=str(TESTS), fixtures=str(data),
                        log=str(tools.log)),
            encoding="utf-8",
        )
        program.chmod(0o755)
    return tools


# --- fixture builders: realistic docker and Kubernetes objects -----------------------------------

NGC_KEY_FROM_SECRET = {"name": "NGC_API_KEY",
                       "valueFrom": {"secretKeyRef": {"name": "ngc-api-secret",
                                                      "key": "NGC_API_KEY"}}}


def digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def docker_container(name: str, image: str, *, host_port: int = 8000, port: int = 8000,
                     host_ip: str = "0.0.0.0", network: str = "bridge",
                     env: Mapping[str, str] | None = None, running: bool = True) -> list[dict]:
    """`docker container inspect NAME` output (the fields the access helpers read). A daemon
    that publishes on 0.0.0.0 also publishes on [::]."""
    bindings = [{"HostIp": host_ip, "HostPort": str(host_port)}]
    if host_ip == "0.0.0.0":
        bindings.append({"HostIp": "::", "HostPort": str(host_port)})
    published = network != "host"
    return [{
        "Id": digest(name),
        "Name": f"/{name}",
        "Image": "sha256:" + digest(image),
        "State": {"Status": "running" if running else "exited", "Running": running},
        "Config": {"Image": image, "Env": [f"{k}={v}" for k, v in (env or {}).items()],
                   "ExposedPorts": {f"{port}/tcp": {}}},
        "HostConfig": {
            "NetworkMode": network,
            "PortBindings": {f"{port}/tcp": [{"HostIp": "" if host_ip == "0.0.0.0" else host_ip,
                                              "HostPort": str(host_port)}]} if published else {},
        },
        "NetworkSettings": {"Ports": {f"{port}/tcp": bindings} if published else {}},
    }]


def docker_port_text(*, port: int = 8000, host_port: int = 8000, host_ip: str = "0.0.0.0") -> str:
    """`docker port NAME` output."""
    lines = [f"{port}/tcp -> {host_ip}:{host_port}"]
    if host_ip == "0.0.0.0":
        lines.append(f"{port}/tcp -> [::]:{host_port}")
    return "\n".join(lines) + "\n"


def docker_image(image: str, *, env: Mapping[str, str] | None = None) -> list[dict]:
    """`docker image inspect IMAGE` output: the image's default environment."""
    return [{"Id": "sha256:" + digest(image), "RepoTags": [image], "RepoDigests": [],
             "Config": {"Env": [f"{k}={v}" for k, v in (env or {}).items()]}}]


def docker_context(host: str = "unix:///var/run/docker.sock", name: str = "default") -> list[dict]:
    """`docker context inspect` output; a tcp:// or ssh:// host is a remote daemon."""
    return [{"Name": name, "Metadata": {},
             "Endpoints": {"docker": {"Host": host, "SkipTLSVerify": False}},
             "TLSMaterial": {}, "Storage": {"MetadataPath": "<IN MEMORY>",
                                            "TLSPath": "<IN MEMORY>"}}]


def k8s_env(env: Mapping[str, str] | None) -> list[dict[str, Any]]:
    return [NGC_KEY_FROM_SECRET, *({"name": k, "value": v} for k, v in (env or {}).items())]


def k8s_nimservice(name: str, namespace: str, repository: str, tag: str, *,
                   args: list[str] | None = None, env: Mapping[str, str] | None = None,
                   port: int = 8000, state: str = "Ready") -> dict[str, Any]:
    """A NIM Operator NIMService (apps.nvidia.com/v1alpha1) as `kubectl get -o json` shows it."""
    spec: dict[str, Any] = {
        "image": {"repository": repository, "tag": tag, "pullPolicy": "IfNotPresent",
                  "pullSecrets": ["ngc-secret"]},
        "authSecret": "ngc-api-secret",
        "replicas": 1,
        "env": k8s_env(env),
        "expose": {"service": {"type": "ClusterIP", "port": port}},
    }
    if args is not None:
        spec["args"] = list(args)
    ready = state == "Ready"
    return {
        "apiVersion": "apps.nvidia.com/v1alpha1", "kind": "NIMService",
        "metadata": {"name": name, "namespace": namespace, "uid": digest(name)[:36]},
        "spec": spec,
        "status": {
            "state": state,
            "conditions": [{"type": "Ready", "status": "True" if ready else "False",
                            "reason": "Ready" if ready else "NotReady"}],
            "model": {"name": "", "clusterEndpoint": f"10.96.0.10:{port}",
                      "externalEndpoint": ""},
        },
    }


def k8s_service(name: str, namespace: str, *, selector: Mapping[str, str],
                port_name: str = "api", port: int = 8000, labels: Mapping[str, str] | None = None,
                cluster_ip: str = "10.96.0.10") -> dict[str, Any]:
    """A Service; cluster_ip "None" is a headless one."""
    return {
        "apiVersion": "v1", "kind": "Service",
        "metadata": {"name": name, "namespace": namespace, "labels": dict(labels or {})},
        "spec": {"type": "ClusterIP", "clusterIP": cluster_ip, "selector": dict(selector),
                 "ports": [{"name": port_name, "port": port, "protocol": "TCP",
                            "targetPort": port_name}]},
    }


def k8s_pod(name: str, namespace: str, *, labels: Mapping[str, str], image: str,
            ready: bool = True, phase: str = "Running", args: list[str] | None = None,
            env: Mapping[str, str] | None = None, port_name: str = "api",
            port: int = 8000) -> dict[str, Any]:
    """A NIM pod; its NGC key comes from a Secret, as the Operator injects it."""
    container: dict[str, Any] = {
        "name": "nim", "image": image, "env": k8s_env(env),
        "ports": [{"name": port_name, "containerPort": port, "protocol": "TCP"}],
    }
    if args is not None:
        container["args"] = list(args)
    return {
        "apiVersion": "v1", "kind": "Pod",
        "metadata": {"name": name, "namespace": namespace, "labels": dict(labels)},
        "spec": {"containers": [container]},
        "status": {
            "phase": phase,
            "conditions": [{"type": "Ready", "status": "True" if ready else "False"}],
            "containerStatuses": [{"name": "nim", "ready": ready, "restartCount": 0,
                                   "image": image, "started": phase == "Running"}],
        },
    }


def k8s_list(*items: dict[str, Any]) -> dict[str, Any]:
    return {"apiVersion": "v1", "kind": "List", "items": list(items),
            "metadata": {"resourceVersion": ""}}


# --- standalone: python3 nsc_testkit.py serve ... ------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nsc_testkit.py",
        description="Run the fake NIM on 127.0.0.1 until interrupted (for the eval harness).",
        epilog="Example: python3 nsc_testkit.py serve --port 8001 "
               "--model meta/llama-3.1-8b-instruct --tools no_parser",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    serve = sub.add_parser("serve", help="serve a fake NIM; prints {url, model, pid} once ready")
    serve.add_argument("--port", type=int, required=True, help="0 picks a free port")
    serve.add_argument("--model", required=True)
    for name in ("root", "owned-by", "reasoning-key", "redirect-to", "answer"):
        serve.add_argument(f"--{name}")
    for name, choices in CHOICES.items():
        if name not in ("vlm", "reasoning_field", "off_bug", "tool_calls_empty"):
            serve.add_argument(f"--{name.replace('_', '-')}", choices=choices)
    serve.add_argument("--reasoning-field", choices=CHOICES["reasoning_field"][1:])
    serve.add_argument("--off-bug", choices=CHOICES["off_bug"][1:])
    serve.add_argument("--tool-calls-empty", choices=CHOICES["tool_calls_empty"][1:])
    serve.add_argument("--vlm", nargs="?", const="yes", choices=("no", "yes", "image_ignored"))
    serve.add_argument("--reasoning-default", choices=("on", "off"))
    serve.add_argument("--no-parallel-calls", action="store_true")
    serve.add_argument("--control-tokens", action="store_true")
    for name in ("ready-after", "delay", "get-delay", "chunk-delay"):
        serve.add_argument(f"--{name}", type=float)
    for name in ("fail-concurrent", "max-model-len"):
        serve.add_argument(f"--{name}", type=int)
    serve.add_argument("--nim-version", dest="version")
    serve.add_argument("--extra-model", action="append", default=[])
    serve.add_argument("--reject-field", action="append", default=[])
    serve.add_argument("--require-key-env", metavar="NAME",
                       help="require Authorization: Bearer <value of $NAME>")
    return parser


def knobs_from(args: argparse.Namespace) -> dict[str, Any]:
    knobs: dict[str, Any] = {}
    for f in dataclasses.fields(Knobs):
        value = getattr(args, f.name, None)
        if value is not None and f.name not in ("model", "vlm", "reasoning_default"):
            knobs[f.name] = value
    knobs["extra_models"] = tuple(args.extra_model)
    knobs["reject_fields"] = tuple(args.reject_field)
    knobs["parallel_calls"] = not args.no_parallel_calls
    if args.vlm is not None:
        knobs["vlm"] = {"no": False, "yes": True}.get(args.vlm, args.vlm)
    if args.reasoning_default is not None:
        knobs["reasoning_default"] = args.reasoning_default == "on"
    if args.require_key_env:
        key = os.environ.get(args.require_key_env)
        if not key:
            raise SystemExit(f"nsc_testkit.py: ${args.require_key_env} is not set")
        knobs["require_key"] = key
    return knobs


def interrupt(signum: int, frame: Any) -> None:
    raise KeyboardInterrupt


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    nim = FakeNim(args.model, port=args.port, **knobs_from(args))
    try:
        nim.start()
    except OSError as exc:
        sys.stderr.write(f"nsc_testkit.py: cannot listen on 127.0.0.1:{args.port}: {exc}\n")
        return 1
    signal.signal(signal.SIGTERM, interrupt)
    print(json.dumps({"url": nim.url, "model": nim.model, "pid": os.getpid()}), flush=True)
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        return 0
    finally:
        nim.stop()


if __name__ == "__main__":
    raise SystemExit(main())
