# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for scripts/nsc_checks.py: the pure parsers, every check against FakeNim, the evidence."""

from __future__ import annotations

import ast
import base64
import contextlib
import functools
import hashlib
import http.server
import json
import re
import socket
import stat
import struct
import sys
import threading
import zlib
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

TESTS = Path(__file__).resolve().parent
for folder in (TESTS, TESTS.parent / "scripts"):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

import nsc_checks as nc  # noqa: E402
import nsc_common as nsc  # noqa: E402
import nsc_features as nf  # noqa: E402
import nsc_testkit as kit  # noqa: E402

MODEL = "meta/llama-3.1-8b-instruct"
CLOSED = "http://127.0.0.1:9"  # the discard port: nothing listens
DOC = "https://docs.nvidia.com/nim/large-language-models/2.1.5/reference/api-reference.html"
ISSUE = (
    "https://docs.nvidia.com/nim/large-language-models/2.1.5/about-nim-llm/release-notes.html"
    "#known-issues"
)
CHECKS_MD = TESTS.parent / "references" / "checks.md"
TABLE = nf.default_table()
HOLD = 1.0  # seconds the fake holds each answer when requests must overlap


# --- helpers -------------------------------------------------------------------------------------


@pytest.fixture
def ev(tmp_path: Path) -> nc.Evidence:
    return nc.Evidence(tmp_path / "run")


def endpoint(url: str, **fields: Any) -> nc.Endpoint:
    return nc.Endpoint(base=url, run_id="t", **fields)


def entry(check: str, request: dict[str, Any] | None = None, **fields: Any) -> dict[str, Any]:
    """A plan entry as nsc_features.decide writes it."""
    out = {"check": check, "decision": "test", "reason": "documented", "sources": [DOC],
           "conflicts": [], "request": dict(request or {}), "known_issues": [], "hint": None,
           "warnings": []}
    out.update(fields)
    return out


def expect(result: nc.Result, status: str, reason: str) -> nc.Result:
    assert (result.status, result.reason) == (status, reason), result.as_dict()
    assert len(result.message) <= nc.MESSAGE_LIMIT
    json.dumps(result.as_dict())  # JSON-ready
    return result


def completion(content: Any = "OK", *, reasoning: str | None = None, field: str = "reasoning",
               finish: Any = "stop", calls: Any = None, usage: Any = None,
               obj: str = "chat.completion", model: str = MODEL) -> dict[str, Any]:
    """A canned chat.completion body."""
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if reasoning is not None:
        message[field] = reasoning
    if calls is not None:
        message["tool_calls"] = calls
    out: dict[str, Any] = {"id": "chatcmpl-1", "object": obj, "created": 1, "model": model,
                           "choices": [{"index": 0, "message": message,
                                        "finish_reason": finish}]}
    if usage is not False:
        out["usage"] = usage or {"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13}
    return out


def tool_call(name: str = "get_weather", arguments: Any = '{"city": "Paris", "unit": "celsius"}',
              **extra: Any) -> dict[str, Any]:
    return {"id": "call_1", "type": "function",
            "function": {"name": name, "arguments": arguments}, **extra}


def chunk(delta: dict[str, Any] | None = None, finish: str | None = None,
          **extra: Any) -> dict[str, Any]:
    return {"id": "c1", "object": "chat.completion.chunk",
            "choices": [{"index": 0, "delta": delta or {}, "finish_reason": finish}], **extra}


def sse(*events: Any, done: bool = True) -> str:
    text = "".join(f"data: {json.dumps(event)}\n\n" for event in events)
    return text + ("data: [DONE]\n\n" if done else "")


def chat_route(*answers: Any) -> dict[str, Any]:
    """FakeNim routes: canned answers of the chat route, in order (the last repeats)."""
    return {"/v1/chat/completions": list(answers) if len(answers) > 1 else answers[0]}


def plan_for(family: str | None = None, *, served: str | None = None,
             checks: Any = nf.CHECKS, hosted: bool = False, generation: str = "2.1.5",
             backend: str = "vllm", signals: dict[str, Any] | None = None) -> dict[str, Any]:
    record, _ = nf.match(TABLE, family=family, served_id=served)
    return nf.decide(record, checks=checks, hosted=hosted, generation=generation,
                     backend=backend, signals=signals, table=TABLE)


def bodies(nim: kit.FakeNim) -> list[dict[str, Any]]:
    return [seen.body for seen in nim.posts("/v1/chat/completions")]


def evidence_json(ev: nc.Evidence, path: str) -> dict[str, Any]:
    return json.loads((ev.out_dir / path).read_text(encoding="utf-8"))


def read_request(conn: socket.socket) -> bool:
    """Read one HTTP request (the head, then a Content-Length body) from `conn`. False when the
    client closes the connection first: recv() then returns b"" at once, again and again."""
    data = b""
    while b"\r\n\r\n" not in data:
        more = conn.recv(65536)
        if not more:
            return False
        data += more
    head, _, rest = data.partition(b"\r\n\r\n")
    found = re.search(rb"(?i)content-length:\s*(\d+)", head)
    while found and len(rest) < int(found.group(1)):
        more = conn.recv(65536)
        if not more:
            return False
        rest += more
    return True


@contextlib.contextmanager
def raw_server(reply: bytes) -> Iterator[str]:
    """A server that reads one HTTP request per connection, writes `reply` and closes. A
    connection closed before its request is complete gets no reply."""
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    sock.listen(8)
    sock.settimeout(0.1)
    stop = threading.Event()

    def serve() -> None:
        while not stop.is_set():
            try:
                conn, _ = sock.accept()
            except OSError:
                continue
            with conn:
                conn.settimeout(2)
                with contextlib.suppress(OSError):
                    if read_request(conn):
                        conn.sendall(reply)

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{sock.getsockname()[1]}"
    finally:
        stop.set()
        thread.join(2)
        sock.close()


# --- parse_sse -----------------------------------------------------------------------------------


def test_parse_sse_reads_events_and_done() -> None:
    raw = sse(chunk({"content": "O"}), chunk({"content": "K"}, "stop")).encode()
    parsed = nc.parse_sse(raw)
    assert [e["choices"][0]["delta"].get("content") for e in parsed["events"]] == ["O", "K"]
    assert parsed["done"] is True
    assert parsed["error_event"] is None
    assert parsed["bad_lines"] == [] and parsed["comments"] == []


def test_parse_sse_skips_keepalive_comments() -> None:
    raw = ": keep-alive\n\n" + sse(chunk({"content": "OK"})) + ": ping\n\n"
    parsed = nc.parse_sse(raw)
    assert len(parsed["events"]) == 1
    assert parsed["comments"] == ["keep-alive", "ping"]
    assert parsed["done"] is True


def test_parse_sse_missing_done_and_a_cut_last_line() -> None:
    raw = sse(chunk({"content": "OK"}), done=False) + 'data: {"id": "c1", "obj'
    parsed = nc.parse_sse(raw)
    assert parsed["done"] is False
    assert len(parsed["events"]) == 1
    assert parsed["bad_lines"] == ['{"id": "c1", "obj']


@pytest.mark.parametrize("error", [
    {"error": {"message": "boom", "type": "InternalServerError", "code": 500}},
    {"object": "error", "message": "boom", "code": 500},
])
def test_parse_sse_error_events_vllm_and_sglang_shapes(error: dict[str, Any]) -> None:
    parsed = nc.parse_sse(sse(chunk({"content": "O"}), error))
    assert parsed["error_event"]["message"] == "boom"
    assert len(parsed["events"]) == 1  # the error is not a chunk
    assert parsed["done"] is True


def test_parse_sse_line_endings_fields_and_events_after_done() -> None:
    raw = (
        "event: message\r\nid: 1\r\nretry: 5\r\ndata:" + json.dumps(chunk({"content": "a"}))
        + "\r\n\r\ndata:[DONE]\r\n\r\n" + "data: " + json.dumps(chunk({"content": "late"}))
        + "\n\ngarbage line\n\n"
    )
    parsed = nc.parse_sse(raw)
    assert len(parsed["events"]) == 1
    assert parsed["done"] is True  # "data:[DONE]" without a space counts
    assert parsed["after_done"] == 1
    assert parsed["bad_lines"] == ["garbage line"]


def test_parse_sse_joins_multi_line_data_and_a_last_event_without_blank_line() -> None:
    raw = 'data: {"object": "chat.completion.chunk",\ndata: "choices": []}\n\ndata: [DONE]'
    parsed = nc.parse_sse(raw)
    assert parsed["events"] == [{"object": "chat.completion.chunk", "choices": []}]
    assert parsed["done"] is True


def test_parse_sse_does_not_split_on_unicode_line_separators() -> None:
    text = "a" + chr(0x2028) + "b" + chr(0x2029) + "c" + chr(0x85) + "d"
    raw = ('data: {"object": "chat.completion.chunk", "choices": [{"index": 0, "delta": '
           '{"content": "' + text + '"}}]}\n\n')
    parsed = nc.parse_sse(raw.encode("utf-8"))
    assert parsed["events"][0]["choices"][0]["delta"]["content"] == text
    assert parsed["bad_lines"] == []


# --- split_reasoning (traps R2, R3, R5, R10) -----------------------------------------------------


@pytest.mark.parametrize(("message", "expected"), [
    ({"content": "Paris", "reasoning": "I know it."}, ("I know it.", "Paris", "reasoning")),
    ({"content": "8", "reasoning_content": "4 x 3 = 12"}, ("4 x 3 = 12", "8", "reasoning_content")),
    ({"content": "8", "reasoning": "vllm", "reasoning_content": "old"},
     ("vllm", "8", "reasoning")),
    ({"content": "8", "reasoning": "", "reasoning_content": "sglang"},
     ("sglang", "8", "reasoning_content")),
    ({"content": "<think>4 x 3 = 12</think>8"}, ("4 x 3 = 12", "8", "inline")),
    ({"content": "4 x 3 = 12</think>\n\n8"}, ("4 x 3 = 12", "8", "inline_close_only")),
    ({"content": "<think>4 x 3 = 12, so"}, ("4 x 3 = 12, so", "", "unclosed")),
    ({"content": "<think></think>Paris"}, ("", "Paris", "none")),
    ({"content": "<think>\n\n</think>\n\nParis"}, ("", "Paris", "none")),
    ({"content": "</think>Paris"}, ("", "Paris", "none")),
    ({"content": None, "reasoning": "   \n"}, ("", "", "none")),
    ({"content": "Paris", "reasoning": None, "reasoning_content": None}, ("", "Paris", "none")),
    ({"content": "Let me think step by step. The answer is 8."},
     ("", "Let me think step by step. The answer is 8.", "none")),
    ({"content": [{"type": "text", "text": "<think>a b</think>"}, {"type": "text", "text": "8"}]},
     ("a b", "8", "inline")),
    ({"content": "<think>inline</think>Paris", "reasoning": "field"},
     ("field\ninline", "Paris", "reasoning")),
    ("not a message", ("", "", "none")),
])
def test_split_reasoning_reads_every_form(message: Any, expected: tuple[str, str, str]) -> None:
    assert nc.split_reasoning(message) == expected


# --- join_stream ---------------------------------------------------------------------------------


def test_join_stream_joins_content_reasoning_finish_and_usage() -> None:
    events = [
        chunk({"role": "assistant", "content": ""}),
        chunk({"reasoning": "think "}),
        chunk({"reasoning": "more"}),
        chunk({"content": "O"}),
        chunk({"content": "K"}, "stop"),
        {"object": "chat.completion.chunk", "choices": [],
         "usage": {"prompt_tokens": 3, "completion_tokens": 4}},
    ]
    joined = nc.join_stream(events)
    assert joined["content"] == "OK" and joined["reasoning"] == "think more"
    assert joined["reasoning_field"] == "reasoning"
    assert joined["finish_reason"] == "stop" and joined["usage"]["completion_tokens"] == 4
    assert joined["chunks"] == 6 and joined["text_chunks"] == 4 and joined["role"] == "assistant"
    message = nc.stream_message(joined)
    assert nc.split_reasoning(message) == ("think more", "OK", "reasoning")


def test_join_stream_takes_one_reasoning_field_per_delta() -> None:
    events = [chunk({"reasoning": "a", "reasoning_content": "a"}),
              chunk({"reasoning_content": "b"}), chunk({}, "stop")]
    joined = nc.join_stream(events)
    assert joined["reasoning"] == "ab"  # vLLM 0.11.1-0.14 fills both: never counted twice


def test_join_stream_assembles_tool_calls_by_index() -> None:
    events = [
        chunk({"tool_calls": [{"index": 1, "id": "b", "type": "function",
                               "function": {"name": "get_weather", "arguments": ""}}]}),
        chunk({"tool_calls": [{"index": 0, "id": "a", "type": "function",
                               "function": {"name": "get_weather", "arguments": '{"city"'}}]}),
        chunk({"tool_calls": [{"index": 1, "function": {"arguments": '{"city": "Tokyo"}'}}]}),
        chunk({"tool_calls": [{"index": 0, "id": "late",
                               "function": {"name": "x", "arguments": ': "Paris"}'}}]}),
        chunk({}, "tool_calls"),
    ]
    calls = nc.join_stream(events)["tool_calls"]
    assert [c["id"] for c in calls] == ["a", "b"]  # first id wins
    assert [c["function"]["name"] for c in calls] == ["get_weather", "get_weather"]
    assert calls[0]["function"]["arguments"] == '{"city": "Paris"}'
    assert calls[1]["function"]["arguments"] == '{"city": "Tokyo"}'


def test_join_stream_keeps_non_string_arguments_and_uses_position_without_index() -> None:
    events = [chunk({"tool_calls": [{"id": "x", "function": {"name": "get_weather",
                                                             "arguments": {"city": "Paris"}}}]})]
    calls = nc.join_stream(events)["tool_calls"]
    assert calls[0]["function"]["arguments"] == {"city": "Paris"}
    status, reason, _ = nc.validate_tool_call({"tool_calls": calls})
    assert (status, reason) == (nc.FAIL, "arguments_not_json_string")


def test_join_stream_ignores_other_choices_and_junk() -> None:
    events = ["junk", {"choices": "no"}, {"object": "chat.completion.chunk",
                                          "choices": [{"index": 1, "delta": {"content": "x"}}]},
              chunk({"content": "y"})]
    joined = nc.join_stream(events)
    assert joined["content"] == "y" and joined["chunks"] == 2


# --- validate_schema -----------------------------------------------------------------------------


@pytest.mark.parametrize(("value", "schema", "problems"), [
    ({"name": "NIM", "version": 2}, nc.JSON_SCHEMA, 0),
    ({"name": "NIM", "version": 2.0}, nc.JSON_SCHEMA, 0),
    ({"name": "NIM", "version": 2.5}, nc.JSON_SCHEMA, 1),
    ({"name": "NIM", "version": True}, nc.JSON_SCHEMA, 1),
    ({"name": "NIM"}, nc.JSON_SCHEMA, 1),
    ({"name": "NIM", "version": 2, "extra": 1}, nc.JSON_SCHEMA, 1),
    ([1], nc.JSON_SCHEMA, 1),
    ({"city": "Paris", "unit": "kelvin"}, nc.WEATHER_PARAMETERS, 1),
    ({"city": "Paris", "unit": "celsius", "extra": 1}, nc.WEATHER_PARAMETERS, 0),
    (True, {"enum": [1]}, 1),
    (1, {"enum": [1.0]}, 0),
    ("[1, 2]", {"type": "array"}, 1),
    ([1, "x"], {"type": "array", "items": {"type": "integer"}}, 1),
    (None, {"type": ["string", "null"]}, 0),
    ({"a": "x"}, {"type": "object", "additionalProperties": {"type": "integer"}}, 1),
    ({"a": {"b": 1}}, {"properties": {"a": {"properties": {"b": {"type": "string"}}}}}, 1),
    ("anything", "not a schema", 0),
    ("x", {"type": "unknown-type"}, 0),
])
def test_validate_schema(value: Any, schema: Any, problems: int) -> None:
    found = nc.validate_schema(value, schema)
    assert len(found) == problems, found
    assert all(item.startswith("$") for item in found)


# --- validate_tool_call (traps T5, T6, T7) -------------------------------------------------------


@pytest.mark.parametrize(("message", "reason"), [
    ({"content": None, "tool_calls": [tool_call()]}, "ok"),
    ({"content": "I cannot.", "tool_calls": []}, "no_tool_call"),
    ({"content": "I cannot.", "tool_calls": None}, "no_tool_call"),
    ({"content": "I cannot."}, "no_tool_call"),
    ({"content": '<TOOLCALL>[{"name": "get_weather"}]</TOOLCALL>'}, "tool_call_in_content"),
    ({"content": '[TOOL_CALLS] [{"name": "get_weather"}]'}, "tool_call_in_content"),
    ({"content": "<tool_call>\n<function=get_weather>"}, "tool_call_in_content"),
    ({"content": '<|python_tag|>{"name": "get_weather"}'}, "tool_call_in_content"),
    ({"content": 'Sure: {"name" : "get_weather", "parameters": {}}'}, "tool_call_in_content"),
    ({"content": None, "reasoning": 'I will call <tool_call>{"name": "get_weather"}'},
     "tool_call_in_reasoning"),
    ({"content": '<think>{"name": "get_weather"}</think>No call.'}, "tool_call_in_reasoning"),
    ({"tool_calls": [tool_call(name="search_web")]}, "wrong_function"),
    ({"tool_calls": [tool_call(type="tool")]}, "bad_response_shape"),
    ({"tool_calls": ["not a call"]}, "bad_response_shape"),
    ({"tool_calls": {"0": tool_call()}}, "bad_response_shape"),
    ({"tool_calls": [tool_call(arguments={"city": "Paris", "unit": "celsius"})]},
     "arguments_not_json_string"),
    ({"tool_calls": [tool_call(arguments='{"city": "Par')]}, "invalid_json_arguments"),
    ({"tool_calls": [tool_call(arguments=json.dumps('{"city": "Paris", "unit": "celsius"}'))]},
     "double_encoded_arguments"),
    ({"tool_calls": [tool_call(arguments="[1, 2]")]}, "arguments_not_object"),
    ({"tool_calls": [tool_call(arguments='{"city": "Paris"}')]}, "schema_mismatch"),
    ({"tool_calls": [tool_call(arguments='{"city": "Paris", "unit": "kelvin"}')]},
     "schema_mismatch"),
    ({"tool_calls": [tool_call(arguments='{"city": ["Paris"], "unit": "celsius"}')]},
     "schema_mismatch"),
    ({"tool_calls": [tool_call(arguments='{"city": "London", "unit": "celsius"}')]},
     "argument_values_wrong"),
    ({"tool_calls": [tool_call(arguments='{"city": "Paris", "unit": "fahrenheit"}')]},
     "argument_values_wrong"),
    ("not a message", "bad_response_shape"),
])
def test_validate_tool_call_order_and_reasons(message: Any, reason: str) -> None:
    status, found, details = nc.validate_tool_call(message, expect=nc.EXPECT_PARIS)
    assert found == reason, details
    assert status == (nc.PASS if reason == "ok" else nc.FAIL)


def test_validate_tool_call_warns_on_a_missing_id_and_extra_keys() -> None:
    call = tool_call(arguments='{"city": "Paris", "unit": "celsius", "days": 1}')
    call.pop("id")
    status, reason, details = nc.validate_tool_call({"tool_calls": [call]},
                                                    expect=nc.EXPECT_PARIS)
    assert (status, reason) == (nc.PASS, "ok")
    assert any("no id" in w for w in details["warnings"])
    assert any("days" in w for w in details["warnings"])


def test_validate_tool_call_expect_forms() -> None:
    message = {"tool_calls": [tool_call()]}
    assert nc.validate_tool_call(message, expect={"unit": "celsius"})[1] == "ok"
    assert nc.validate_tool_call(message, expect={"city": lambda v: v == "Paris"})[1] == "ok"
    assert nc.validate_tool_call(message, expect={"city": "paris"})[1] == "argument_values_wrong"
    assert nc.validate_tool_call(message)[1] == "ok"  # no expectation: schema only


# --- control tokens, PNG, tool markup, parser-missing texts --------------------------------------


def test_control_tokens_every_token_in_order() -> None:
    for token in nc.CONTROL_TOKENS:
        assert nc.control_tokens(f"a {token} b") == [token]
    assert nc.control_tokens("<|message|>x<|channel|>final") == ["<|message|>", "<|channel|>"]
    assert nc.control_tokens("Plain answer with <think> tags") == []
    assert nc.control_tokens(None) == []


def test_png_solid_is_a_valid_tiny_png() -> None:
    data = nc.png_solid()
    assert data.startswith(b"\x89PNG\r\n\x1a\n") and len(data) < 300
    position, chunks = 8, {}
    while position < len(data):
        (length,) = struct.unpack(">I", data[position:position + 4])
        kind = data[position + 4:position + 8]
        body = data[position + 8:position + 8 + length]
        (crc,) = struct.unpack(">I", data[position + 8 + length:position + 12 + length])
        assert crc == zlib.crc32(kind + body) & 0xFFFFFFFF
        chunks[kind] = body
        position += 12 + length
    assert list(chunks) == [b"IHDR", b"IDAT", b"IEND"]
    assert struct.unpack(">IIBBBBB", chunks[b"IHDR"]) == (64, 64, 8, 2, 0, 0, 0)
    rows = zlib.decompress(chunks[b"IDAT"])
    assert rows == (b"\x00" + b"\xff\x00\x00" * 64) * 64
    small = nc.png_solid(2, 3, (0, 0, 255))
    assert struct.unpack(">II", small[16:24]) == (2, 3)
    for bad in ((0, 1, (0, 0, 0)), (1, 1, (256, 0, 0)), (1, 1, (1, 2))):
        with pytest.raises(ValueError):
            nc.png_solid(*bad)
    assert nc.data_url(small).startswith("data:image/png;base64,iVBORw0KGgo")


def test_tool_markup_and_parser_missing_texts() -> None:
    assert nc.tool_markup("") is None
    assert nc.tool_markup('{"name": "other_tool"}') is None
    assert nc.tool_markup('x [TOOL_CALLS] y <tool_call>') == "[TOOL_CALLS]"
    assert nc.PARSER_MISSING.search(kit.AUTO_NEEDS_PARSER)
    assert nc.PARSER_MISSING.search(kit.FORCED_NEEDS_PARSER.format(desc='"required"'))
    assert nc.PARSER_MISSING.search(
        kit.FORCED_NEEDS_PARSER.format(desc='function "get_weather"'))
    assert nc.PARSER_MISSING.search(json.dumps({"message": kit.AUTO_NEEDS_PARSER}))
    assert not nc.PARSER_MISSING.search("tool_choice must be one of auto, none, required")


# --- requests: allow-list, bodies, timeouts (trap R12) -------------------------------------------


@pytest.mark.parametrize("name", sorted(nc.NEVER_SEND))
def test_never_sent_fields_are_refused(name: str) -> None:
    body = {"model": MODEL, "messages": [], name: 1}
    with pytest.raises(ValueError):
        nc.check_request_fields(body)


def test_allow_list_rules() -> None:
    assert nc.NEVER_SEND.isdisjoint(nc.REQUEST_FIELDS)
    nc.check_request_fields({key: None for key in nc.REQUEST_FIELDS - {
        "parallel_tool_calls", "chat_template_kwargs"}})
    with pytest.raises(ValueError):
        nc.check_request_fields({"chat_template_kwargs": {"chat_template": "x"}})
    with pytest.raises(ValueError):
        nc.check_request_fields({"parallel_tool_calls": False})
    with pytest.raises(ValueError):
        nc.chat_body(MODEL, "x", max_tokens=1, stop=["\n"])


def test_chat_body_sampling_switch_mode_and_stream() -> None:
    plain = nc.chat_body(MODEL, "hi", max_tokens=8)
    assert plain == {"model": MODEL, "messages": [{"role": "user", "content": "hi"}],
                     "max_tokens": 8, "temperature": 0}
    server_defaults = nc.chat_body(MODEL, "hi", max_tokens=8, sampling={})
    assert "temperature" not in server_defaults
    qwen = nc.chat_body(MODEL, "hi", max_tokens=8,
                        sampling={"temperature": 0.6, "top_p": 0.95, "top_k": 20, "seed": 1})
    assert (qwen["temperature"], qwen["top_p"], qwen["top_k"]) == (0.6, 0.95, 20)
    assert "seed" not in qwen
    mode = [{"method": "system", "value": "detailed thinking off"}]
    body = nc.chat_body(MODEL, "hi", max_tokens=8, mode=mode,
                        switch={"method": "effort", "value": "low"}, stream=True)
    assert body["messages"][0] == {"role": "system", "content": "detailed thinking off"}
    assert body["reasoning_effort"] == "low"
    assert body["stream"] is True and body["stream_options"] == {"include_usage": True}


def test_request_timeout_scales_with_max_tokens_and_is_capped() -> None:
    ep = endpoint(CLOSED)
    assert nc.request_timeout(ep, 64) == pytest.approx(63.2)
    assert nc.request_timeout(ep, 4096) == pytest.approx(264.8)
    assert nc.request_timeout(ep, 16384) == 600.0
    assert nc.request_timeout(ep) == nc.GET_TIMEOUT
    assert nc.request_timeout(endpoint(CLOSED, timeout_cap=5), 4096) == 5


@pytest.mark.parametrize("given", [300.0, 3600.0, 5.0])
def test_a_given_timeout_sets_every_chat_budget_up_or_down(given: float) -> None:
    """docs-2: --timeout (Endpoint.timeout) is the budget of every chat request, so a slow NIM
    can be given more time than the formula's 63.2 s for 64 tokens; a GET never gets more than
    GET_TIMEOUT."""
    ep = endpoint(CLOSED, timeout=given)
    for tokens in (0, 64, 1024, 16384):
        assert nc.request_timeout(ep, tokens) == given
    assert nc.request_timeout(ep) == min(nc.GET_TIMEOUT, given)


# --- evidence ------------------------------------------------------------------------------------


def test_evidence_masks_the_key_and_hashes_the_image(ev: nc.Evidence) -> None:
    key = kit.fake_key()
    token = kit.fake_token()
    with kit.FakeNim(vlm=True, require_key=key) as nim:
        ep = endpoint(nim.url, headers={"Authorization": f"Bearer {key}", "X-Token": token})
        result = expect(nc.check_image(ep, ev, nim.model, entry("image")), nc.PASS, "ok")
    assert kit.files_containing(ev.out_dir, key) == []
    assert kit.files_containing(ev.out_dir, token) == []
    png = nc.png_solid()
    assert kit.files_containing(ev.out_dir, base64.b64encode(png).decode()[:40]) == []
    record = evidence_json(ev, result.evidence[0])
    assert record["request"]["headers"]["Authorization"] == "Bearer ***"
    assert record["request"]["headers"]["X-Token"] == "***"
    image = record["request"]["body"]["messages"][0]["content"][1]["image_url"]["url"]
    assert image == {"omitted": "inline data", "media_type": "image/png",
                     "sha256": hashlib.sha256(png).hexdigest(), "bytes": len(png)}
    assert result.details["image_sha256"] == hashlib.sha256(png).hexdigest()


def test_evidence_record_fields_and_private_files(ev: nc.Evidence) -> None:
    with kit.FakeNim() as nim:
        result = expect(nc.check_stream(endpoint(nim.url), ev, nim.model, entry("stream")),
                        nc.PASS, "ok")
    record = evidence_json(ev, result.evidence[0])
    assert record["check"] == "stream" and record["attempt"] == 1
    assert record["status"] == 200 and record["content_type"] == "text/event-stream"
    assert record["request_id"] == "nsc-t-stream-1"
    assert record["request"]["headers"]["X-Request-Id"] == "nsc-t-stream-1"
    assert record["request"]["headers"]["Accept"] == "text/event-stream"
    assert record["latency_ms"] >= record["ttfb_ms"] >= 0
    assert record["finish_reason"] == "stop" and record["usage"]["completion_tokens"] >= 1
    assert record["response_raw"].endswith("data: [DONE]\n\n")
    assert record["parsed"]["role"] == "attempt" and record["parsed"]["answer"] == "OK"
    folder = ev.out_dir / "evidence"
    assert stat.S_IMODE(folder.stat().st_mode) == 0o700
    assert stat.S_IMODE((ev.out_dir / result.evidence[0]).stat().st_mode) == 0o600


def test_evidence_cuts_long_text_and_records_the_full_length(ev: nc.Evidence) -> None:
    long_answer = "word " * 1000  # 5,000 characters
    with kit.FakeNim(answer=long_answer) as nim:
        result = nc.check_chat(endpoint(nim.url), ev, nim.model, entry("chat"))
    record = evidence_json(ev, result.evidence[0])
    content = record["response"]["choices"][0]["message"]["content"]
    assert len(content) == nc.TEXT_LIMIT
    assert record["cut"]["$.response.choices[0].message.content"] == len(long_answer)
    assert len(record["parsed"]["answer"]) == nc.TEXT_LIMIT
    assert long_answer[:100] not in result.message
    assert any("not OK" in warning for warning in result.warnings)


def test_evidence_cuts_a_raw_stream_to_its_byte_limit(ev: nc.Evidence) -> None:
    with kit.FakeNim(answer="word " * 400) as nim:
        result = expect(nc.check_stream(endpoint(nim.url), ev, nim.model,
                                        entry("stream", {"max_tokens": 1024})), nc.PASS, "ok")
    record = evidence_json(ev, result.evidence[0])
    assert record["response_bytes"] > nc.RAW_LIMIT
    assert len(record["response_raw"].encode("utf-8")) == nc.RAW_LIMIT
    assert record["cut"]["$.response_raw"] == record["response_bytes"]


def deep(depth: int, leaf: Any = 1) -> Any:
    """A list nested `depth` levels deep around `leaf`, built without recursion."""
    value = leaf
    for _ in range(depth):
        value = [value]
    return value


@pytest.mark.parametrize("depth", [nsc.MAX_JSON_DEPTH + 1, 950, 3000, 100_000])
def test_json_nested_too_deeply_never_crashes_a_parser(depth: int) -> None:
    """security-5: an event, tool arguments or a JSON answer nested past nsc.MAX_JSON_DEPTH
    levels is "not JSON" for the checks (json.loads would raise RecursionError)."""
    text = "[" * depth + "]" * depth
    parsed = nc.parse_sse(f"data: {text}\n\ndata: [DONE]\n\n")
    assert parsed["events"] == [] and len(parsed["bad_lines"]) == 1 and parsed["done"]
    arguments = '{"city": ' + text + "}"
    status, reason, _ = nc.validate_tool_call(
        {"tool_calls": [tool_call(arguments=arguments)]}, expect=nc.EXPECT_PARIS)
    assert (status, reason) == (nc.FAIL, "invalid_json_arguments")
    assert nc._json_object(text) is None


def test_evidence_walkers_keep_no_value_nested_past_the_walk_limit() -> None:
    """security-5: the evidence's cut and image redaction recurse once a level: a value nested
    past nsc.MAX_WALK_DEPTH levels becomes nsc.NESTED, and the record is still written."""
    value = deep(5000, "data:image/png;base64,iVBORw0KGgo=")
    cuts: dict[str, int] = {}
    assert nsc.NESTED in json.dumps(nc._cut(value, "$.x", cuts)) and cuts == {}
    assert nsc.NESTED in json.dumps(nc.redact_images(value))
    near = nc.redact_images(deep(5, "data:image/png;base64,iVBORw0KGgo="))
    assert near[0][0][0][0][0]["omitted"] == "inline data"
    request = {"method": "POST", "url": "http://127.0.0.1:1/v1/chat/completions",
               "headers": {}, "body": {"messages": value}}
    res = nsc.HttpResult(200, {"content-type": "application/json"},
                         ("[" * 5000 + "1" + "]" * 5000).encode(), 3.0)
    record = nc.evidence_record("chat", 1, request, res, {"role": "attempt", "x": deep(5000)})
    text = nsc.SCRUBBER.dumps(record)
    assert nsc.NESTED in text and json.loads(text)["response_text"].startswith("[[[")
    assert "response" not in record  # JSON nested past MAX_JSON_DEPTH is kept as text


def test_evidence_record_is_pure_and_keeps_text_bodies() -> None:
    res = nsc.HttpResult(502, {"content-type": "text/html"}, b"<html>bad gateway</html>", 3.0)
    token = kit.fake_token()  # the URL password, and the header value without "Bearer "
    request = {"method": "POST", "url": f"http://u:{token}@127.0.0.1:1/v1/chat/completions",
               "headers": {"authorization": token, "Accept": "x"},
               "body": {"messages": [{"content": "data:image/png;base64,iVBORw0KGgo="}]}}
    record = nc.evidence_record("chat", 2, request, res, {"role": "retry"})
    assert record["request"]["url"] == "http://127.0.0.1:1/v1/chat/completions"
    assert record["request"]["headers"]["authorization"] == "***"
    assert token not in json.dumps(record)
    assert record["request"]["body"]["messages"][0]["content"]["media_type"] == "image/png"
    assert record["response_text"] == "<html>bad gateway</html>"
    assert record["attempt"] == 2 and record["parsed"] == {"role": "retry"}
    assert "ttfb_ms" not in record  # not a stream


# --- health --------------------------------------------------------------------------------------


def test_health_passes_on_live_and_ready(ev: nc.Evidence) -> None:
    with kit.FakeNim() as nim:
        result = expect(nc.check_health(endpoint(nim.url), ev, entry("health")), nc.PASS, "ok")
        assert [s.path for s in nim.gets()] == ["/v1/health/live", "/v1/health/ready"]
        assert all(s.body is None and s.raw == b"" for s in nim.gets())  # no body (Kimi issue)
    assert result.evidence == ["evidence/health-1.json", "evidence/health-2.json"]
    assert result.details == {"live": 200, "ready": 200, "fallback": None, "polls": 1}


def test_health_not_ready_while_loading(ev: nc.Evidence) -> None:
    with kit.FakeNim(health="loading") as nim:
        expect(nc.check_health(endpoint(nim.url), ev), nc.FAIL, "not_ready")


def test_health_falls_back_to_slash_health_on_404(ev: nc.Evidence) -> None:
    with kit.FakeNim(health="bare") as nim:
        result = expect(nc.check_health(endpoint(nim.url), ev), nc.PASS, "ok")
        assert nim.gets()[-1].path == "/health"
    assert result.details["fallback"] == 200 and result.warnings


def test_health_route_not_served_is_not_applicable(ev: nc.Evidence) -> None:
    with kit.FakeNim(health="missing") as nim:
        result = expect(nc.check_health(endpoint(nim.url), ev), nc.NOT_APPLICABLE,
                        "health_route_not_served")
    assert result.warnings and len(result.evidence) == 3


def test_health_hosted_is_not_applicable_and_sends_nothing(ev: nc.Evidence) -> None:
    with kit.FakeNim() as nim:
        expect(nc.check_health(endpoint(nim.url, hosted=True), ev), nc.NOT_APPLICABLE,
               "hosted_no_health_route")
        assert nim.requests == []


def test_health_not_live(ev: nc.Evidence) -> None:
    routes = {"/v1/health/live": (503, {"object": "health.response", "status": "error"})}
    with kit.FakeNim(routes=routes) as nim:
        expect(nc.check_health(endpoint(nim.url), ev), nc.FAIL, "not_live")


def test_health_behind_a_gateway_that_wants_a_key_is_unauthorized(ev: nc.Evidence) -> None:
    """docs-5: a 401 or 403 on /v1/health/live is the gateway asking for a key (unauthorized,
    with the export advice), not a NIM that is not live; a 429 is rate_limited."""
    key = kit.fake_key()
    with kit.FakeNim(require_key=key) as nim:
        result = expect(nc.check_health(endpoint(nim.url), ev), nc.FAIL, "unauthorized")
        assert "/v1/health/live" in result.message and "--api-key-env" in result.message
        result = expect(nc.check_health(endpoint(nim.url, headers={
            "Authorization": f"Bearer {key}x"}), ev), nc.FAIL, "unauthorized")
        assert "the key was refused" in result.message
    for status, reason in ((403, "unauthorized"), (429, "rate_limited")):
        with kit.FakeNim(routes={"/v1/health/live": (status, {"detail": "x"})}) as nim:
            expect(nc.check_health(endpoint(nim.url), ev), nc.FAIL, reason)


def test_health_timeout_and_unreachable(ev: nc.Evidence) -> None:
    with kit.FakeNim(get_delay=1.0) as nim:
        expect(nc.check_health(endpoint(nim.url, timeout_cap=0.2), ev), nc.FAIL,
               "health_timeout")
    expect(nc.check_health(endpoint(CLOSED), ev), nc.FAIL, "endpoint_unreachable")


def test_health_ready_other_statuses(ev: nc.Evidence) -> None:
    for status, reason in ((500, "not_ready"), (418, "http_error"), (401, "unauthorized"),
                           (429, "rate_limited")):
        with kit.FakeNim(routes={"/v1/health/ready": (status, {"detail": "x"})}) as nim:
            expect(nc.check_health(endpoint(nim.url), ev), nc.FAIL, reason)


def test_health_wait_ready_polls_until_ready(ev: nc.Evidence) -> None:
    with kit.FakeNim(health="loading", ready_after=0.4) as nim:
        result = expect(nc.check_health(endpoint(nim.url), ev, wait=5, poll=0.1), nc.PASS, "ok")
    assert result.details["polls"] >= 2 and result.attempts == result.details["polls"]
    assert len(result.evidence) == 2  # each poll overwrites the same two files
    with kit.FakeNim(health="loading") as nim:
        result = expect(nc.check_health(endpoint(nim.url), ev, wait=0.3, poll=0.1), nc.FAIL,
                        "not_ready")
    assert "probes" in result.message


def test_health_uses_the_health_base(ev: nc.Evidence) -> None:
    with kit.FakeNim() as api, kit.FakeNim() as health:
        ep = endpoint(api.url, health_base=health.url)
        expect(nc.check_health(ep, ev), nc.PASS, "ok")
        assert api.requests == [] and len(health.gets()) == 2


def test_the_key_goes_only_to_the_scheme_host_and_port_of_the_url(ev: nc.Evidence) -> None:
    """A health base on another port (or host, or scheme) gets no Authorization header; the
    URL's own routes, also under another path, get it."""
    key = kit.fake_key()
    with kit.FakeNim(require_key=key) as api, kit.FakeNim() as health:
        ep = endpoint(api.url, health_base=health.url, headers={"Authorization": f"Bearer {key}"})
        expect(nc.check_health(ep, ev), nc.PASS, "ok")
        assert len(health.gets()) == 2
        assert all("authorization" not in seen.headers for seen in health.requests)
        expect(nc.check_models(ep, None, ev)[0], nc.PASS, "ok")
        assert api.requests and all(
            seen.headers.get("authorization") == f"Bearer {key}" for seen in api.requests
        )
        same_host = endpoint(api.url, health_base=api.url + "/health-prefix",
                             headers={"Authorization": f"Bearer {key}"})
        assert same_host.sends_key_to(same_host.health_url("/v1/health/ready"))
    saved = [path.read_text("utf-8") for path in ev.folder.glob("*.json")]
    assert saved and not any(key in text for text in saved)
    ep = endpoint("http://gateway.example.com/prefix",
                  headers={"Authorization": f"Bearer {key}", "X-Other": "1"})
    for url in ("http://gateway.example.com/prefix/v1/models",
                "http://GATEWAY.example.com:80/other/v1/health/live"):
        assert ep.headers_for(url) == ep.headers and ep.sends_key_to(url), url
    for url in ("https://gateway.example.com/prefix/v1/health/live",
                "http://gateway.example.com:8001/v1/health/live", "http://10.0.0.7/v1/health/live",
                "http://gateway.example.com.evil.invalid/v1/models"):
        assert ep.headers_for(url) == {"X-Other": "1"} and not ep.sends_key_to(url), url


def test_a_health_url_that_wants_the_key_it_does_not_get_says_so(ev: nc.Evidence) -> None:
    key = kit.fake_key()
    with kit.FakeNim() as api, kit.FakeNim(require_key=key) as health:
        ep = endpoint(api.url, health_base=health.url, headers={"Authorization": f"Bearer {key}"})
        result = expect(nc.check_health(ep, ev), nc.FAIL, "unauthorized")
        assert all("authorization" not in seen.headers for seen in health.requests)
    assert "goes only to the scheme, host and port of --url" in result.message
    assert "the key was refused" not in result.message


@contextlib.contextmanager
def health_proxy() -> Iterator[tuple[str, list[str]]]:
    """An HTTP proxy on 127.0.0.1 standing in for the only route to a remote health host: it
    answers 200 for /v1/health/* of any host and records the absolute URLs it is asked for."""
    asked: list[str] = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            asked.append(self.path)
            ok = self.path.rsplit("/v1/health/", 1)[-1] in ("live", "ready")
            body = b"{}" if ok else b'{"detail": "Not Found"}'
            self.send_response(200 if ok else 404)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: Any) -> None:
            return

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, args=(0.05,), daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}", asked
    finally:
        server.shutdown()
        server.server_close()


def test_a_remote_health_url_keeps_the_proxy_settings_of_a_loopback_endpoint(
    ev: nc.Evidence, monkeypatch: pytest.MonkeyPatch
) -> None:
    """security-4: the endpoint's loopback flag is its base URL's. A --health-url on another
    host, here reachable only through HTTP_PROXY, uses the environment's proxy settings, as
    every non-loopback URL does (references/access.md), and passes; the loopback API URL still
    skips the proxy. No name is ever looked up."""
    real = socket.getaddrinfo

    def numeric_only(host: Any, *args: Any, **kwargs: Any) -> Any:
        if host not in ("127.0.0.1", "::1"):
            raise socket.gaierror(socket.EAI_NONAME, f"the test looks up no name ({host})")
        return real(host, *args, **kwargs)

    monkeypatch.setattr(socket, "getaddrinfo", numeric_only)
    for name in ("no_proxy", "NO_PROXY", "all_proxy", "ALL_PROXY"):
        monkeypatch.delenv(name, raising=False)
    remote = "http://nim-health.invalid:8001"
    with kit.FakeNim() as api, health_proxy() as (proxy, asked):
        for name in ("http_proxy", "HTTP_PROXY"):
            monkeypatch.setenv(name, proxy)
        ep = endpoint(api.url, health_base=remote, loopback=True)
        assert ep.loopback_for(ep.url("/v1/models")) is True
        assert ep.loopback_for(ep.health_url("/v1/health/ready")) is False
        expect(nc.check_health(ep, ev), nc.PASS, "ok")
        assert asked == [f"{remote}/v1/health/live", f"{remote}/v1/health/ready"]
        expect(nc.check_models(ep, None, ev)[0], nc.PASS, "ok")
        assert len(asked) == 2 and api.gets("/v1/models")  # the API went direct
    loopback_health = endpoint("http://gateway.invalid:8000", health_base=api.url)
    assert loopback_health.loopback_for(loopback_health.health_url("/v1/health/live")) is True


def test_health_port_not_reached_sends_nothing_and_wait_ready_stops(ev: nc.Evidence) -> None:
    """NIM_HEALTH_PORT moved the health routes to a port this run cannot reach: not applicable
    with its own reason (not "the endpoint serves no health route"); --wait-ready cannot wait
    there, so it stops with exit 3 instead of moving on silently."""
    note = ("NIM_HEALTH_PORT=8001: the health routes moved to that port, which the container "
            "does not publish")
    with kit.FakeNim(health="loading") as nim:
        result = expect(nc.check_health(endpoint(nim.url), ev, unreachable=note),
                        nc.NOT_APPLICABLE, "health_port_not_reached")
        with pytest.raises(nsc.ScriptError) as info:
            nc.check_health(endpoint(nim.url), ev, wait=5, poll=0.1, unreachable=note)
        assert nim.requests == []
    assert "NIM_HEALTH_PORT=8001" in result.message and result.evidence == []
    assert result.warnings == [nc.NA_MESSAGES["health_port_not_reached"]]
    assert (info.value.code, info.value.exit_code) == ("health_port_not_reached",
                                                       nsc.EXIT_PRECONDITION)
    assert "--health-url" in info.value.fix and "NIM_HEALTH_PORT=8001" in info.value.message


def test_health_hosted_says_wait_ready_was_not_applied(ev: nc.Evidence) -> None:
    with kit.FakeNim() as nim:
        result = expect(nc.check_health(endpoint(nim.url, hosted=True), ev, wait=5),
                        nc.NOT_APPLICABLE, "hosted_no_health_route")
        assert nim.requests == []
    assert any("--wait-ready was not applied" in warning for warning in result.warnings)


# --- models --------------------------------------------------------------------------------------


def test_models_chooses_the_single_served_model(ev: nc.Evidence) -> None:
    with kit.FakeNim() as nim:
        result, chosen, cards = nc.check_models(endpoint(nim.url), None, ev, entry("models"))
    expect(result, nc.PASS, "ok")
    assert chosen == MODEL and result.evidence == ["evidence/models-1.json"]
    assert cards[0] == {"id": MODEL, "root": MODEL, "parent": None, "owned_by": "vllm",
                        "max_model_len": 131072}


def test_models_requested_lora_and_renamed_ids(ev: nc.Evidence) -> None:
    lora = {"id": "my-adapter", "parent": MODEL, "root": "/loras/my-adapter"}
    with kit.FakeNim(extra_models=(lora,)) as nim:
        ep = endpoint(nim.url)
        assert nc.check_models(ep, None, ev)[1] == MODEL  # adapters are not base models
        assert nc.check_models(ep, "my-adapter", ev)[1] == "my-adapter"
        result, chosen, _ = nc.check_models(ep, "meta/llama-3.1-8b", ev)
    # the model the user names is never swapped for the one model served
    expect(result, nc.FAIL, "model_not_served")
    assert chosen is None
    assert result.message.startswith("meta/llama-3.1-8b is not served") and MODEL in result.message
    assert result.details["served"] == [MODEL, "my-adapter"]


@pytest.mark.parametrize("extra", [(), ("other/model",), ("other/model", "third/model")])
def test_models_a_mistyped_id_is_not_served_however_many_base_models(
    ev: nc.Evidence, extra: tuple[str, ...]
) -> None:
    with kit.FakeNim(extra_models=extra) as nim:
        result, chosen, _ = nc.check_models(endpoint(nim.url), "typo/model", ev)
    expect(result, nc.FAIL, "model_not_served")
    assert chosen is None and result.message.startswith("typo/model is not served")
    assert all(model in result.message for model in (MODEL, *extra))


def test_models_ambiguous_no_models_and_unreachable_are_errors(ev: nc.Evidence) -> None:
    with kit.FakeNim(extra_models=("other/model",)) as nim:
        ep = endpoint(nim.url)
        assert nc.check_models(ep, "other/model", ev)[1] == "other/model"
        with pytest.raises(nsc.ScriptError) as info:
            nc.check_models(ep, None, ev)
    assert info.value.code == "model_ambiguous" and info.value.exit_code == nsc.EXIT_PRECONDITION
    assert info.value.facts["served_models"] == [MODEL, "other/model"]
    empty = {"/v1/models": (200, {"object": "list", "data": []})}
    with kit.FakeNim(routes=empty) as nim, pytest.raises(nsc.ScriptError) as info:
        nc.check_models(endpoint(nim.url), MODEL, ev)
    assert info.value.code == "no_models" and info.value.exit_code == nsc.EXIT_PRECONDITION
    assert info.value.facts["result"]["reason"] == "no_models"
    with pytest.raises(nsc.ScriptError) as info:
        nc.check_models(endpoint(CLOSED), MODEL, ev)
    assert info.value.code == "endpoint_unreachable"
    assert info.value.exit_code == nsc.EXIT_EXTERNAL


@pytest.mark.parametrize(("route", "reason"), [
    ((500, {"detail": "boom"}), "http_error"),
    ((200, "<html>proxy login</html>"), "not_json"),
    ((200, {"object": "list"}), "bad_response_shape"),
    ((401, {"detail": "no"}), "unauthorized"),
])
def test_models_failures_keep_the_requested_id(ev: nc.Evidence, route: Any, reason: str) -> None:
    with kit.FakeNim(routes={"/v1/models": route}) as nim:
        result, chosen, _ = nc.check_models(endpoint(nim.url), MODEL, ev)
    expect(result, nc.FAIL, reason)
    assert chosen == MODEL


@pytest.mark.parametrize("odd_id", [7, {"name": MODEL}, [MODEL], True])
def test_models_a_card_whose_id_is_not_text_never_crashes(ev: nc.Evidence, odd_id: Any) -> None:
    """security-6: the OpenAI shape makes a model id a string. A card with another id is left
    out with a warning; when no card has a text id, models fails bad_response_shape and keeps
    the requested id (as for any unreadable list), with no StopIteration."""
    cards = {"/v1/models": (200, {"object": "list", "data": [{"id": odd_id, "parent": None}]})}
    with kit.FakeNim(routes=cards) as nim:
        ep = endpoint(nim.url)
        for requested in (None, "7", MODEL):
            result, chosen, found = nc.check_models(ep, requested, ev)
            expect(result, nc.FAIL, "bad_response_shape")
            assert (chosen, found) == (requested, [])
            assert "text id" in result.message
    mixed = {"/v1/models": (200, {"object": "list", "data": [
        {"id": odd_id, "parent": None}, {"id": MODEL, "parent": None, "root": MODEL}]})}
    with kit.FakeNim(routes=mixed) as nim:
        result, chosen, found = nc.check_models(endpoint(nim.url), None, ev)
    expect(result, nc.PASS, "ok")
    assert chosen == MODEL and [card["id"] for card in found] == [MODEL]
    assert result.details["card"]["id"] == MODEL
    assert any("not text" in warning for warning in result.warnings)


def test_models_redirect_is_refused(ev: nc.Evidence) -> None:
    with kit.FakeNim() as other, kit.FakeNim(redirect_to=other.url) as nim:
        result, chosen, _ = nc.check_models(endpoint(nim.url), None, ev)
        assert other.requests == []
    expect(result, nc.FAIL, "redirect_refused")
    assert chosen is None


def test_models_hosted_reads_the_single_model_route(ev: nc.Evidence) -> None:
    with kit.FakeNim() as nim:
        ep = endpoint(nim.url, hosted=True)
        result, chosen, cards = nc.check_models(ep, MODEL, ev)
        assert nim.gets()[-1].path == f"/v1/models/{MODEL}"
        expect(result, nc.PASS, "ok")
        assert chosen == MODEL and cards[0]["id"] == MODEL
        result, chosen, _ = nc.check_models(ep, "nobody/model", ev)
        expect(result, nc.FAIL, "model_not_served")
        assert chosen is None
        assert nc.check_models(ep, None, ev)[1] == MODEL  # no id: the list
    gone = {"type": "about:blank", "title": "Gone", "status": 410,
            "detail": "The model has reached its end of life"}
    with kit.FakeNim(routes={f"/v1/models/{MODEL}": (410, gone)}) as nim:
        result, chosen, _ = nc.check_models(endpoint(nim.url, hosted=True), MODEL, ev)
    expect(result, nc.FAIL, "model_end_of_life")
    assert chosen is None


# --- chat ----------------------------------------------------------------------------------------


def test_chat_passes_and_sends_the_documented_request(ev: nc.Evidence) -> None:
    with kit.FakeNim() as nim:
        result = expect(nc.check_chat(endpoint(nim.url), ev, nim.model,
                                      entry("chat", {"max_tokens": 64})), nc.PASS, "ok")
        (body,) = bodies(nim)
        (seen,) = nim.posts()
    assert body == {"model": MODEL, "max_tokens": 64, "temperature": 0,
                    "messages": [{"role": "user", "content": nc.CHAT_PROMPT}]}
    assert seen.headers["content-type"] == "application/json"
    assert seen.headers["accept"] == "application/json"
    assert seen.headers["x-request-id"] == "nsc-t-chat-1"
    assert result.sources == [DOC] and result.attempts == 1 and result.latency_ms is not None
    assert result.message.startswith("HTTP 200 with an answer")


def test_chat_reasoning_model_with_a_small_budget_warns(ev: nc.Evidence) -> None:
    """Trap R1: reasoning uses up a small budget; the text is in the reasoning field."""
    with kit.FakeNim("openai/gpt-oss-20b", reasoning="always") as nim:
        result = expect(nc.check_chat(endpoint(nim.url), ev, nim.model,
                                      entry("chat", {"max_tokens": 5})), nc.PASS, "ok")
    assert result.details["where"] == "reasoning"
    assert any("finish_reason is length" in warning for warning in result.warnings)
    # the message says what came back: no answer, only reasoning
    assert "reasoning only, no answer within 5 tokens" in result.message
    assert "with an answer" not in result.message


@pytest.mark.parametrize(("knobs", "reason"), [
    ({"control_tokens": True}, "control_tokens_in_content"),
    ({"answer": ""}, "empty_completion"),
    ({"routes": chat_route((200, completion(obj="text_completion")))}, "bad_response_shape"),
    ({"routes": chat_route((200, {"choices": []}))}, "bad_response_shape"),
    ({"routes": chat_route((200, completion(usage=False)))}, "bad_response_shape"),
    ({"routes": chat_route((200, completion(usage={"completion_tokens": 0})))},
     "empty_completion"),
    ({"routes": chat_route((200, "<html>ok</html>"))}, "not_json"),
    ({"routes": chat_route((500, {"error": {"message": "engine died"}}))}, "http_error"),
    ({"routes": chat_route((429, {"detail": "Too Many Requests"}))}, "rate_limited"),
    ({"routes": chat_route((410, {"title": "Gone", "detail": "end of life"}))},
     "model_end_of_life"),
    ({"require_key": kit.fake_token()}, "unauthorized"),
])
def test_chat_failures(ev: nc.Evidence, knobs: dict[str, Any], reason: str) -> None:
    with kit.FakeNim(**knobs) as nim:
        result = expect(nc.check_chat(endpoint(nim.url), ev, nim.model, entry("chat")),
                        nc.FAIL, reason)
        assert len(nim.posts()) == 1  # never retried, a 429 included
    assert result.evidence == ["evidence/chat-1.json"]


def test_chat_model_not_served(ev: nc.Evidence) -> None:
    with kit.FakeNim() as nim:
        result = expect(nc.check_chat(endpoint(nim.url), ev, "other/model"), nc.FAIL,
                        "model_not_served")
    assert "does not exist" in result.message


def test_unauthorized_says_whether_a_key_was_sent(ev: nc.Evidence) -> None:
    key = kit.fake_token()
    with kit.FakeNim(require_key=key) as nim:
        without = nc.check_chat(endpoint(nim.url), ev, nim.model)
        wrong = nc.check_chat(endpoint(nim.url, headers={"Authorization": f"Bearer {key}x"}),
                              ev, nim.model)
    assert "--api-key-env" in expect(without, nc.FAIL, "unauthorized").message
    assert "refused" in expect(wrong, nc.FAIL, "unauthorized").message
    assert kit.files_containing(ev.out_dir, key) == []  # the refused key is masked too


# --- transport failures (every check shares them) ------------------------------------------------


def test_transport_unreachable_timeout_tls_and_bad_request(ev: nc.Evidence) -> None:
    expect(nc.check_chat(endpoint(CLOSED), ev, MODEL), nc.FAIL, "endpoint_unreachable")
    with kit.FakeNim(delay=1.0) as nim:
        expect(nc.check_chat(endpoint(nim.url, timeout_cap=0.2), ev, MODEL), nc.FAIL, "timeout")
    with kit.FakeNim() as nim:
        https = nim.url.replace("http://", "https://")
        expect(nc.check_chat(endpoint(https), ev, MODEL), nc.FAIL, "tls_error")
        ep = endpoint(nim.url, headers={"X-Bad": "a\nb"})
        expect(nc.check_chat(ep, ev, MODEL), nc.FAIL, "request_failed")


def test_transport_redirect_is_refused_and_never_followed(ev: nc.Evidence) -> None:
    with kit.FakeNim() as other, kit.FakeNim(redirect_to=other.url) as nim:
        ep = endpoint(nim.url, headers={"Authorization": "Bearer " + kit.fake_key()})
        expect(nc.check_chat(ep, ev, MODEL), nc.FAIL, "redirect_refused")
        expect(nc.check_stream(ep, ev, MODEL), nc.FAIL, "redirect_refused")
        assert other.requests == []


def test_transport_connection_lost_on_a_short_body(ev: nc.Evidence) -> None:
    reply = (b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 500\r\n"
             b"Connection: close\r\n\r\n{\"id\": \"chatcmpl-1\", \"obj")
    with raw_server(reply) as url:
        expect(nc.check_chat(endpoint(url), ev, MODEL), nc.FAIL, "connection_lost")


def test_raw_server_drops_a_request_cut_short_and_answers_the_next() -> None:
    """A client that hangs up before its request is complete gets no reply, and the server goes
    on to the next connection. Before, the read loop took recv()'s b"" as no data yet and spun
    forever, so the next client got nothing."""
    reply = b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\nOK"
    with raw_server(reply) as url:
        address = ("127.0.0.1", int(url.rsplit(":", 1)[1]))
        for cut in (b"POST / HTTP/1.1\r\nHost: x",  # in the head
                    b"POST / HTTP/1.1\r\nContent-Length: 10\r\n\r\n{}"):  # in the body
            with socket.create_connection(address, timeout=5) as client:
                client.sendall(cut)
                client.shutdown(socket.SHUT_WR)
                assert client.recv(1024) == b""  # closed, with no reply
        with socket.create_connection(address, timeout=5) as client:
            client.sendall(b"GET / HTTP/1.1\r\nHost: x\r\n\r\n")
            answer = b"".join(iter(functools.partial(client.recv, 1024), b""))
    assert answer == reply


def test_transport_response_too_large(ev: nc.Evidence, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(nc.nsc, "request_json",
                        functools.partial(nsc.request_json, max_bytes=100))
    with kit.FakeNim() as nim:
        expect(nc.check_chat(endpoint(nim.url), ev, MODEL), nc.FAIL, "response_too_large")


# --- stream --------------------------------------------------------------------------------------


@pytest.mark.parametrize(("mode", "status", "reason"), [
    ("ok", nc.PASS, "ok"),
    ("keepalive", nc.PASS, "ok"),
    ("no_usage", nc.PASS, "ok"),
    ("ignored", nc.FAIL, "stream_ignored"),
    ("no_done", nc.FAIL, "stream_incomplete"),
    ("cut", nc.FAIL, "stream_incomplete"),
    ("error_event", nc.FAIL, "stream_error_event"),
    ("empty", nc.FAIL, "empty_stream"),
])
def test_stream_modes(ev: nc.Evidence, mode: str, status: str, reason: str) -> None:
    with kit.FakeNim(stream=mode) as nim:
        result = expect(nc.check_stream(endpoint(nim.url), ev, nim.model, entry("stream")),
                        status, reason)
        (body,) = bodies(nim)
    assert body["stream"] is True and body["stream_options"] == {"include_usage": True}
    if mode == "keepalive":
        assert result.details["keepalive_comments"] > 0
    if mode == "no_usage":
        assert any("usage" in warning for warning in result.warnings)


def test_stream_canned_shapes(ev: nc.Evidence) -> None:
    one = sse(chunk({"content": "OK"}, "stop"))
    no_finish = sse(chunk({"content": "O"}), chunk({"content": "K"}))
    for text, reason in ((one, "stream_ignored"), (no_finish, "bad_response_shape"),
                         ("data: [DONE]\n\n", "stream_ignored")):
        with kit.FakeNim(routes=chat_route((200, text, "text/event-stream"))) as nim:
            expect(nc.check_stream(endpoint(nim.url), ev, MODEL), nc.FAIL, reason)
    with kit.FakeNim(routes=chat_route((503, {"detail": "busy"}))) as nim:
        expect(nc.check_stream(endpoint(nim.url), ev, MODEL), nc.FAIL, "http_error")


def test_stream_that_times_out_or_drops_before_its_first_chunk(ev: nc.Evidence) -> None:
    """HTTP 200, text/event-stream and a keep-alive comment, then no chunk before the timeout
    (a slow reasoning prefill): a timeout, not an ignored stream. A drop there is
    connection_lost."""
    with kit.FakeNim(stream="keepalive", chunk_delay=3.0) as nim:
        result = expect(nc.check_stream(endpoint(nim.url, timeout_cap=1.0), ev, MODEL),
                        nc.FAIL, "timeout")
    assert result.details["chunks"] == 0 and result.details["keepalive_comments"] >= 1
    comment = b": keep-alive\n\n"
    reply = (b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n"
             b"Transfer-Encoding: chunked\r\n\r\n" + f"{len(comment):x}\r\n".encode()
             + comment + b"\r\n")
    with raw_server(reply) as url:
        result = expect(nc.check_stream(endpoint(url), ev, MODEL), nc.FAIL, "connection_lost")
    assert result.details["chunks"] == 0


def test_stream_that_stalls_is_incomplete(ev: nc.Evidence) -> None:
    with kit.FakeNim(chunk_delay=0.3) as nim:
        result = expect(nc.check_stream(endpoint(nim.url, timeout_cap=0.5), ev, MODEL), nc.FAIL,
                        "stream_incomplete")
    assert "timed out" in result.message


# --- tools: T-AUTO (traps T1-T9, T11, T13) -------------------------------------------------------


def test_tools_pass_and_request(ev: nc.Evidence) -> None:
    with kit.FakeNim() as nim:
        result = expect(nc.check_tools(endpoint(nim.url), ev, nim.model,
                                       entry("tools", {"max_tokens": 1024})), nc.PASS, "ok")
        (body,) = bodies(nim)
    assert body["tool_choice"] == "auto" and "parallel_tool_calls" not in body
    assert body["tools"] == [nc.WEATHER_TOOL] and body["temperature"] == 0
    assert body["messages"] == [{"role": "user", "content": nc.TOOL_PROMPT}]  # T13: one turn
    assert result.details["finish_reason"] == "tool_calls"
    assert result.details["arguments"] == {"city": '"Paris"', "unit": '"celsius"'}


@pytest.mark.parametrize(("mode", "reason"), [
    ("double_encoded", "double_encoded_arguments"),
    ("wrong_args", "argument_values_wrong"),
    ("arguments_object", "arguments_not_json_string"),
    ("invalid_json", "invalid_json_arguments"),
    ("schema_mismatch", "schema_mismatch"),
    ("wrong_name", "wrong_function"),
    ("in_reasoning", "tool_call_in_reasoning"),
])
def test_tools_validation_failures(ev: nc.Evidence, mode: str, reason: str) -> None:
    with kit.FakeNim(tools=mode) as nim:
        expect(nc.check_tools(endpoint(nim.url), ev, nim.model, entry("tools")), nc.FAIL, reason)
        assert len(nim.posts()) == 1  # only plain text is retried


def test_tools_markup_in_content_attaches_the_hint(ev: nc.Evidence) -> None:
    hint = "Documented server switch: --tool-call-parser x. Changing the deployment is nim-deploy's"
    with kit.FakeNim(tools="in_content") as nim:
        result = expect(nc.check_tools(endpoint(nim.url), ev, nim.model,
                                       entry("tools", hint=hint)), nc.FAIL,
                        "tool_call_in_content")
    assert result.hint == hint and "<tool_call>" in result.message


def test_tools_prose_retries_then_sends_the_named_diagnostic(ev: nc.Evidence) -> None:
    """Traps T1, T3 and T14: a named call that works never makes the check pass (guided
    decoding, which speculative decoding can change, only proves a parsable call)."""
    with kit.FakeNim(tools="prose") as nim:
        result = expect(nc.check_tools(endpoint(nim.url), ev, nim.model,
                                       entry("tools", {"named_diagnostic": True})), nc.FAIL,
                        "model_did_not_call_tool_with_auto")
        sent = bodies(nim)
    assert [b["tool_choice"] for b in sent] == [
        "auto", "auto", {"type": "function", "function": {"name": "get_weather"}}]
    assert result.attempts == 2 and len(result.evidence) == 3
    assert result.details["attempt_reasons"] == ["no_tool_call", "no_tool_call"]
    assert result.details["named_diagnostic"]["reason"] == "ok"


@pytest.mark.parametrize("empty", ["omit", "list", "null"])
def test_tools_empty_tool_calls_forms_are_no_call(ev: nc.Evidence, empty: str) -> None:
    """Trap T5: [], null and a missing tool_calls mean the same; gpt-oss sends no named call."""
    with kit.FakeNim(tools="prose", tool_calls_empty=empty) as nim:
        result = expect(nc.check_tools(endpoint(nim.url), ev, nim.model,
                                       entry("tools", {"named_diagnostic": False})), nc.FAIL,
                        "no_tool_call")
        assert len(nim.posts()) == 2
    assert "named" in result.message


def test_tools_named_diagnostic_without_a_call(ev: nc.Evidence) -> None:
    prose = (200, completion("I cannot look that up."))
    with kit.FakeNim(routes=chat_route(prose)) as nim:
        result = expect(nc.check_tools(endpoint(nim.url), ev, nim.model, entry("tools")),
                        nc.FAIL, "no_tool_call")
        assert len(nim.posts()) == 3
    assert "named diagnostic" in result.message


def test_tools_retry_can_pass(ev: nc.Evidence) -> None:
    answers = [(200, completion("Let me see.")), (200, completion(None, calls=[tool_call()],
                                                                   finish="tool_calls"))]
    with kit.FakeNim(routes=chat_route(*answers)) as nim:
        result = expect(nc.check_tools(endpoint(nim.url), ev, nim.model, entry("tools")),
                        nc.PASS, "ok")
    assert result.attempts == 2


def test_tools_on_sglang_say_the_parser_may_be_missing(ev: nc.Evidence) -> None:
    with kit.FakeNim(backend="sglang", tools="no_parser") as nim:
        result = expect(nc.check_tools(endpoint(nim.url, backend="sglang"), ev, nim.model,
                                       entry("tools", hint="the hint")), nc.FAIL,
                        "tool_call_in_content")
    assert "may be missing" in result.message and result.hint == "the hint"
    with kit.FakeNim(backend="sglang", tools="prose") as nim:
        result = expect(nc.check_tools(endpoint(nim.url, backend="sglang"), ev, nim.model,
                                       entry("tools", {"named_diagnostic": False},
                                             hint="the hint")), nc.FAIL, "no_tool_call")
    assert "SGLang" in result.message and result.hint == "the hint"


def test_tools_rate_limited_is_never_retried(ev: nc.Evidence) -> None:
    with kit.FakeNim(routes=chat_route((429, "slow down", "text/plain"))) as nim:
        expect(nc.check_tools(endpoint(nim.url), ev, nim.model, entry("tools")), nc.FAIL,
               "rate_limited")
        assert len(nim.posts()) == 1


# --- tools: a missing parser (D5, trap T4) -------------------------------------------------------


@pytest.mark.parametrize(("plan", "status", "reason"), [
    ({"status": "fail", "reason": "parser_not_enabled", "warning": None}, nc.FAIL,
     "parser_not_enabled"),
    ({"status": "fail", "reason": "parser_dropped_by_override", "warning": None}, nc.FAIL,
     "parser_dropped_by_override"),
    ({"status": "not_applicable", "reason": "not_enabled_in_deployment", "warning": None},
     nc.NOT_APPLICABLE, "not_enabled_in_deployment"),
])
def test_tools_parser_missing_follows_the_plan(ev: nc.Evidence, plan: dict[str, Any],
                                               status: str, reason: str) -> None:
    hint = "Documented server switch: x. Changing the deployment is nim-deploy's job."
    with kit.FakeNim(tools="no_parser") as nim:
        for check in (nc.check_tools, nc.check_tools_stream, nc.check_tools_parallel):
            name = check.__name__[len("check_"):]
            result = expect(check(endpoint(nim.url), ev, nim.model,
                                  entry(name, if_parser_missing=plan, hint=hint)), status, reason)
            assert result.hint == hint
        # tool_choice "none" needs no parser on vLLM: it still answers
        expect(nc.check_tools_none(endpoint(nim.url), ev, nim.model,
                                   entry("tools_none", if_parser_missing=plan)), nc.PASS, "ok")
    assert not result.warnings


def test_tools_parser_missing_without_plan_data(ev: nc.Evidence) -> None:
    with kit.FakeNim(tools="no_parser") as nim:
        result = expect(nc.check_tools(endpoint(nim.url), ev, nim.model, entry("tools")),
                        nc.NOT_APPLICABLE, "not_enabled_in_deployment")
        assert len(nim.posts()) == 1
    assert result.hint == nc.PARSER_HINT and result.warnings
    with kit.FakeNim(tools="no_parser") as nim:
        result = nc.check_tools(endpoint(nim.url, hosted=True), ev, nim.model, entry("tools"))
    assert result.hint is None  # no server switch on the hosted API


def test_tools_parser_missing_with_real_plans(ev: nc.Evidence) -> None:
    """D5 from the table: unknown default (warning), override, on by default, opt-in."""
    cases = [
        (plan_for(served=MODEL), nc.NOT_APPLICABLE, "not_enabled_in_deployment", True),
        (plan_for(served=MODEL, signals=nf.tier_b(
            "--enable-auto-tool-choice --tool-call-parser llama3_json", "--max-num-seqs 8")),
         nc.FAIL, "parser_dropped_by_override", False),
        (plan_for(served="nvidia/nemotron-3.5-super-vl-120b-a12b"), nc.FAIL,
         "parser_not_enabled", False),
        (plan_for(served="moonshotai/kimi-k3.1-pro"), nc.NOT_APPLICABLE,
         "not_enabled_in_deployment", False),
    ]
    with kit.FakeNim(tools="no_parser") as nim:
        for plan, status, reason, warned in cases:
            result = expect(nc.check_tools(endpoint(nim.url), ev, nim.model,
                                           nf.plan_entry(plan, "tools")), status, reason)
            assert "nim-deploy" in result.hint and "--tool-call-parser" in result.hint
            assert bool(result.warnings) == warned, result.warnings
            if reason == "parser_dropped_by_override":
                assert "NIM_PASSTHROUGH_ARGS" in result.hint
            if reason == "not_enabled_in_deployment":  # "opt-in" only where the record says so
                assert ("opt-in" in result.message) == (not warned), result.message
                assert ("is not known" in result.message) == warned, result.message


def test_tools_parser_missing_names_an_override_when_the_image_default_is_unknown(
        ev: nc.Evidence) -> None:
    """D5 with an unknown image default stays not applicable, but is never called opt-in; when
    the deployment sets NIM_PASSTHROUGH_ARGS with no parser flags, the warning names it as the
    likely cause (the image default could not be read, as on a NIM Operator cluster)."""
    plan = plan_for(served=MODEL, signals=nf.tier_b(None, "--max-num-seqs 8"))
    with kit.FakeNim(tools="no_parser") as nim:
        result = expect(nc.check_tools(endpoint(nim.url), ev, nim.model,
                                       nf.plan_entry(plan, "tools")), nc.NOT_APPLICABLE,
                        "not_enabled_in_deployment")
    assert "opt-in" not in result.message
    (warning,) = result.warnings
    assert "do not say whether" in warning and "NIM_PASSTHROUGH_ARGS" in warning
    assert "dropped" in warning and "NIM_PASSTHROUGH_ARGS" in result.hint


# --- tools: stream, parallel, none (trap T10) ----------------------------------------------------


def test_tools_stream_assembles_the_call(ev: nc.Evidence) -> None:
    with kit.FakeNim() as nim:
        result = expect(nc.check_tools_stream(endpoint(nim.url), ev, nim.model,
                                              entry("tools_stream")), nc.PASS, "ok")
        (body,) = bodies(nim)
    assert body["stream"] is True and body["tool_choice"] == "auto"
    assert result.details["finish_reason"] == "tool_calls"


def test_tools_stream_retries_once_only_when_the_plan_says(ev: nc.Evidence) -> None:
    error = sse(chunk({"content": ""}), {"error": {"message": "harmony parse error"}})
    good = sse(chunk({"tool_calls": [{"index": 0, "id": "c", "type": "function",
                                      "function": {"name": "get_weather", "arguments": ""}}]}),
               chunk({"tool_calls": [{"index": 0, "function": {
                   "arguments": '{"city": "Paris", "unit": "celsius"}'}}]}),
               chunk({}, "tool_calls"))
    answers = [(200, error, "text/event-stream"), (200, good, "text/event-stream")]
    with kit.FakeNim(routes=chat_route(*answers)) as nim:
        result = expect(nc.check_tools_stream(endpoint(nim.url), ev, MODEL,
                                              entry("tools_stream", {"retry": 1})), nc.PASS, "ok")
        assert len(nim.posts()) == 2
    assert result.attempts == 2
    with kit.FakeNim(routes=chat_route(*answers)) as nim:
        expect(nc.check_tools_stream(endpoint(nim.url), ev, MODEL, entry("tools_stream")),
               nc.FAIL, "stream_error_event")
        assert len(nim.posts()) == 1


@pytest.mark.parametrize(("knobs", "reason"), [
    ({"tools": "in_content"}, "tool_call_in_content"),
    ({"tools": "prose"}, "no_tool_call"),
    ({"tools": "double_encoded"}, "double_encoded_arguments"),
    ({"stream": "no_done"}, "stream_incomplete"),
])
def test_tools_stream_failures(ev: nc.Evidence, knobs: dict[str, Any], reason: str) -> None:
    with kit.FakeNim(**knobs) as nim:
        expect(nc.check_tools_stream(endpoint(nim.url), ev, nim.model, entry("tools_stream")),
               nc.FAIL, reason)
        assert len(nim.posts()) == 1  # no named diagnostic for the stream


def test_tools_parallel(ev: nc.Evidence) -> None:
    with kit.FakeNim() as nim:
        result = expect(nc.check_tools_parallel(endpoint(nim.url), ev, nim.model,
                                                entry("tools_parallel")), nc.PASS, "ok")
        (body,) = bodies(nim)
    assert body["parallel_tool_calls"] is True
    assert body["messages"][0]["content"] == nc.PARALLEL_PROMPT
    assert sorted(result.details["cities"]) == ["paris", "tokyo"]


@pytest.mark.parametrize("cities", [("Paris, France", "Tokyo, Japan"), ("Paris, FR", "Tokyo, JP"),
                                    ("Paris, France", "Tokyo"), ("paris", "TOKYO")])
def test_tools_parallel_accepts_a_city_with_its_country(ev: nc.Evidence,
                                                        cities: tuple[str, str]) -> None:
    """Each call matches /paris|tokyo/i as T-AUTO's does, so "Paris, France" counts as paris."""
    calls = [tool_call(arguments=json.dumps({"city": city, "unit": "celsius"}), id=f"call_{n}")
             for n, city in enumerate(cities)]
    answer = completion(None, calls=calls, finish="tool_calls")
    with kit.FakeNim(routes=chat_route((200, answer))) as nim:
        result = expect(nc.check_tools_parallel(endpoint(nim.url), ev, nim.model,
                                                entry("tools_parallel")), nc.PASS, "ok")
    assert result.details["cities"] == ["paris", "tokyo"]


@pytest.mark.parametrize(("knobs", "reason"), [
    ({"parallel_calls": False}, "too_few_calls"),
    ({"tools": "prose"}, "no_tool_call"),
    ({"tools": "wrong_args"}, "argument_values_wrong"),
    ({"tools": "schema_mismatch"}, "schema_mismatch"),
    ({"routes": chat_route((200, completion(None, calls=[tool_call(), tool_call()])))},
     "argument_values_wrong"),
    ({"routes": chat_route((500, {"detail": "x"}))}, "http_error"),
])
def test_tools_parallel_failures(ev: nc.Evidence, knobs: dict[str, Any], reason: str) -> None:
    with kit.FakeNim(**knobs) as nim:
        expect(nc.check_tools_parallel(endpoint(nim.url), ev, nim.model,
                                       entry("tools_parallel")), nc.FAIL, reason)


@pytest.mark.parametrize(("knobs", "status", "reason"), [
    ({}, nc.PASS, "ok"),
    ({"tools": "none_markup"}, nc.PASS, "ok"),
    ({"tools": "none_empty"}, nc.FAIL, "empty_answer"),
    ({"routes": chat_route((200, completion("4", calls=[tool_call()])))}, nc.FAIL,
     "tool_called_with_none"),
    ({"routes": chat_route((400, {"error": {"message": "bad"}}))}, nc.FAIL, "http_error"),
])
def test_tools_none(ev: nc.Evidence, knobs: dict[str, Any], status: str, reason: str) -> None:
    with kit.FakeNim(**knobs) as nim:
        result = expect(nc.check_tools_none(endpoint(nim.url), ev, nim.model,
                                            entry("tools_none")), status, reason)
        (body,) = bodies(nim)
    assert body["tool_choice"] == "none"
    assert body["messages"][0]["content"] == nc.NONE_PROMPT
    if knobs.get("tools") == "none_markup":
        assert result.warnings


def test_tools_stream_not_applicable_where_streaming_calls_are_unsupported(
        ev: nc.Evidence) -> None:
    """Trap T10: Super v1.5's parser has no streaming, so nothing is sent."""
    plan = plan_for(served="nvidia/llama-3.3-nemotron-super-49b-v1.5",
                    checks=nf.CHECKS)
    with kit.FakeNim() as nim:
        result = expect(nc.check_tools_stream(endpoint(nim.url), ev, nim.model,
                                              nf.plan_entry(plan, "tools_stream")),
                        nc.NOT_APPLICABLE, "not_documented_for_model")
        assert nim.requests == []
    assert result.evidence == [] and result.sources


def test_tools_not_applicable_with_pipeline_parallelism_on_1x(ev: nc.Evidence) -> None:
    """Trap T12: NIM 1.x does not support tool calls with pipeline parallelism above 1."""
    record, _ = nf.match(TABLE, served_id=MODEL)
    plan = nf.decide(record, checks=nf.CHECKS, generation="1.15.0", backend="vllm",
                     profile={"name": "vllm-bf16-tp4-pp2"}, table=TABLE)
    with kit.FakeNim() as nim:
        for check in ("tools", "tools_stream", "tools_none"):
            expect(nc.run_check(check, endpoint(nim.url), ev, nim.model,
                                nf.plan_entry(plan, check)), nc.NOT_APPLICABLE,
                   "pipeline_parallel_unsupported")
        assert nim.requests == []


# --- reasoning on (traps R1-R3, R5, R7, R8, R10, R11) --------------------------------------------


@pytest.mark.parametrize("field", ["reasoning", "reasoning_content", "inline",
                                   "inline_close_only"])
def test_reasoning_on_reads_every_place(ev: nc.Evidence, field: str) -> None:
    plan = plan_for(served="nvidia/nemotron-3-nano")
    hint = nf.plan_entry(plan, "reasoning_on")["hint"]
    with kit.FakeNim("nvidia/nemotron-3-nano", reasoning="kwargs", reasoning_field=field) as nim:
        result = expect(nc.check_reasoning_on(endpoint(nim.url), ev, nim.model,
                                              nf.plan_entry(plan, "reasoning_on")),
                        nc.PASS, "ok")
        (body,) = bodies(nim)
    assert body["chat_template_kwargs"] == {"enable_thinking": True}  # R7: always sent
    assert (body["temperature"], body["top_p"]) == (1.0, 1.0)  # Nemotron 3 Nano sampling
    assert body["max_tokens"] == 4096 and body["messages"][0]["content"] == nc.REASONING_PROMPT
    assert result.details["where"] == field and result.details["expected_answer_seen"]
    inline = field.startswith("inline")
    assert (result.hint == hint) is inline
    assert any("inline in content" in warning for warning in result.warnings) is inline
    assert any("record expects" in warning for warning in result.warnings) is (
        field in ("reasoning_content", "inline"))


def test_reasoning_on_qwen3_sampling_is_never_greedy(ev: nc.Evidence) -> None:
    """Trap R8."""
    plan = plan_for(served="qwen/qwen3-32b")
    with kit.FakeNim("qwen/qwen3-32b", reasoning="kwargs") as nim:
        expect(nc.check_reasoning_on(endpoint(nim.url), ev, nim.model,
                                     nf.plan_entry(plan, "reasoning_on")), nc.PASS, "ok")
        (body,) = bodies(nim)
    assert (body["temperature"], body["top_p"], body["top_k"]) == (0.6, 0.95, 20)


def test_reasoning_on_gpt_oss_low_effort(ev: nc.Evidence) -> None:
    plan = plan_for(served="openai/gpt-oss-20b")
    with kit.FakeNim("openai/gpt-oss-20b", reasoning="always") as nim:
        expect(nc.check_reasoning_on(endpoint(nim.url), ev, nim.model,
                                     nf.plan_entry(plan, "reasoning_on")), nc.PASS, "ok")
        (body,) = bodies(nim)
    assert body["reasoning_effort"] == "low" and "chat_template_kwargs" not in body


def test_reasoning_on_missing_after_one_retry(ev: nc.Evidence) -> None:
    on = entry("reasoning_on", {"switch": {"method": "system", "key": None, "value": "/think"},
                                "max_tokens": 4096})
    with kit.FakeNim(reasoning="none") as nim:
        result = expect(nc.check_reasoning_on(endpoint(nim.url), ev, nim.model, on), nc.FAIL,
                        "reasoning_missing")
        first, second = bodies(nim)
    assert first == second and result.attempts == 2
    assert result.evidence == ["evidence/reasoning_on-1.json", "evidence/reasoning_on-2.json"]


@pytest.mark.parametrize("answer", [
    completion("<think></think>8"),  # R5: an empty think block is no reasoning
    completion("Let me think step by step. The answer is 8.",  # R10: prose and usage
               usage={"prompt_tokens": 5, "completion_tokens": 50, "reasoning_tokens": 40}),
])
def test_reasoning_on_ignores_empty_blocks_prose_and_usage(ev: nc.Evidence,
                                                           answer: dict[str, Any]) -> None:
    with kit.FakeNim(routes=chat_route((200, answer))) as nim:
        expect(nc.check_reasoning_on(endpoint(nim.url), ev, MODEL, entry("reasoning_on")),
               nc.FAIL, "reasoning_missing")
        assert len(nim.posts()) == 2


def test_reasoning_on_not_finished_doubles_the_budget(ev: nc.Evidence) -> None:
    """Trap R1."""
    with kit.FakeNim(reasoning="always") as nim:
        result = expect(nc.check_reasoning_on(endpoint(nim.url), ev, nim.model,
                                              entry("reasoning_on", {"max_tokens": 5})),
                        nc.FAIL, "reasoning_not_finished")
        assert [b["max_tokens"] for b in bodies(nim)] == [5, 10]
    assert "after one retry" in result.message


def test_reasoning_on_retry_budget_is_capped(ev: nc.Evidence) -> None:
    cut = completion(None, reasoning="Four pens cost", finish="length")
    done = completion("8", reasoning="Four pens cost 12 dollars, so the change is 8.")
    with kit.FakeNim(routes=chat_route((200, cut), (200, done))) as nim:
        expect(nc.check_reasoning_on(endpoint(nim.url), ev, MODEL,
                                     entry("reasoning_on", {"max_tokens": 9000})), nc.PASS, "ok")
        assert [b["max_tokens"] for b in bodies(nim)] == [9000, 16384]


def test_reasoning_on_retry_never_lowers_a_budget_above_the_cap(ev: nc.Evidence) -> None:
    cut = completion(None, reasoning="Four pens cost", finish="length")
    done = completion("8", reasoning="Four pens cost 12 dollars, so the change is 8.")
    with kit.FakeNim(routes=chat_route((200, cut), (200, done))) as nim:
        expect(nc.check_reasoning_on(endpoint(nim.url), ev, MODEL,
                                     entry("reasoning_on", {"max_tokens": 20000})), nc.PASS, "ok")
        assert [b["max_tokens"] for b in bodies(nim)] == [20000, 20000]


@pytest.mark.parametrize(("answer", "reason"), [
    (completion(None, reasoning="Four pens cost 12 dollars."), "answer_missing"),
    (completion("8", reasoning="Four pens cost 12 dollars.", finish="abort"),
     "unexpected_finish_reason"),
    (completion("<|channel|>final<|message|>8", reasoning="Four pens cost 12 dollars."),
     "control_tokens_in_content"),
])
def test_reasoning_on_answer_failures(ev: nc.Evidence, answer: dict[str, Any],
                                      reason: str) -> None:
    with kit.FakeNim(routes=chat_route((200, answer))) as nim:
        expect(nc.check_reasoning_on(endpoint(nim.url), ev, MODEL, entry("reasoning_on")),
               nc.FAIL, reason)
        assert len(nim.posts()) == 1


def test_reasoning_on_switch_rejected_and_other_4xx(ev: nc.Evidence) -> None:
    """Trap R11: a 4xx on the documented switch; a 4xx without a switch is http_error."""
    on = entry("reasoning_on", {"switch": {"method": "kwargs", "key": "enable_thinking",
                                           "value": True}})
    with kit.FakeNim(reasoning="kwargs", reject_fields=("chat_template_kwargs",)) as nim:
        result = expect(nc.check_reasoning_on(endpoint(nim.url), ev, nim.model, on), nc.FAIL,
                        "switch_rejected")
        assert len(nim.posts()) == 1
    assert "chat_template_kwargs.enable_thinking" in result.message
    with kit.FakeNim(routes=chat_route((400, {"error": {"message": "bad top_k"}}))) as nim:
        expect(nc.check_reasoning_on(endpoint(nim.url), ev, MODEL, entry("reasoning_on")),
               nc.FAIL, "http_error")
    with kit.FakeNim(routes=chat_route((404, {"detail": "Not Found"}))) as nim:
        expect(nc.check_reasoning_on(endpoint(nim.url), ev, MODEL, on), nc.FAIL, "http_error")


# --- reasoning off (traps R4, R6, R9, R13) -------------------------------------------------------


def test_reasoning_off_pass_and_request(ev: nc.Evidence) -> None:
    plan = plan_for(served="nvidia/nemotron-3-nano")
    with kit.FakeNim("nvidia/nemotron-3-nano", reasoning="kwargs") as nim:
        expect(nc.check_reasoning_off(endpoint(nim.url), ev, nim.model,
                                      nf.plan_entry(plan, "reasoning_off")), nc.PASS, "ok")
        result = expect(nc.check_reasoning_off_stream(
            endpoint(nim.url), ev, nim.model, nf.plan_entry(plan, "reasoning_off_stream")),
            nc.PASS, "ok")
        plain, streamed = bodies(nim)
    assert plain["chat_template_kwargs"] == {"enable_thinking": False}
    assert plain["temperature"] == 0 and plain["max_tokens"] == 256
    assert plain["messages"][0]["content"] == nc.OFF_PROMPT and "stream" not in plain
    assert streamed["stream"] is True and result.id == "reasoning_off_stream"
    assert result.evidence == ["evidence/reasoning_off_stream-1.json"]


@pytest.mark.parametrize(("family", "served", "field", "value"), [
    ("deepseek-v3.1-v3.2", "deepseek-ai/deepseek-v3.1", "chat_template_kwargs",
     {"thinking": False}),
    ("mistral-small-4", "mistralai/mistral-small-4-119b-2603", "reasoning_effort", "none"),
])
def test_reasoning_off_sends_the_family_switch(ev: nc.Evidence, family: str, served: str,
                                               field: str, value: Any) -> None:
    """Trap R6: DeepSeek V3.x reads thinking, not enable_thinking; Mistral Small 4 gets no
    chat_template_kwargs (HTTP 400 there)."""
    plan = plan_for(family, served=served)
    knobs = ({"reasoning": "kwargs", "reasoning_key": "thinking", "reasoning_default": False}
             if field == "chat_template_kwargs" else
             {"reasoning": "effort", "reasoning_default": False,
              "reject_fields": ("chat_template_kwargs",)})
    with kit.FakeNim(served, **knobs) as nim:
        expect(nc.check_reasoning_off(endpoint(nim.url), ev, nim.model,
                                      nf.plan_entry(plan, "reasoning_off")), nc.PASS, "ok")
        expect(nc.check_reasoning_on(endpoint(nim.url), ev, nim.model,
                                     nf.plan_entry(plan, "reasoning_on")), nc.PASS, "ok")
        off = bodies(nim)[0]
    assert off[field] == value


def test_reasoning_off_super_v15_uses_no_think(ev: nc.Evidence) -> None:
    served = "nvidia/llama-3.3-nemotron-super-49b-v1.5"
    plan = plan_for(served=served)
    with kit.FakeNim(served, reasoning="system", reasoning_field="inline_close_only") as nim:
        expect(nc.check_reasoning_off(endpoint(nim.url), ev, nim.model,
                                      nf.plan_entry(plan, "reasoning_off")), nc.PASS, "ok")
        on = expect(nc.check_reasoning_on(endpoint(nim.url), ev, nim.model,
                                          nf.plan_entry(plan, "reasoning_on")), nc.PASS, "ok")
        off_body, on_body = bodies(nim)
    assert off_body["messages"][0] == {"role": "system", "content": "/no_think"}
    assert on_body["messages"][0]["role"] == "user"  # the on switch is the default
    assert on.details["where"] == "inline_close_only"  # R3


def test_reasoning_off_answer_in_reasoning_field(ev: nc.Evidence) -> None:
    """Trap R4, non-streaming and streaming."""
    off = entry("reasoning_off", {"switch": {"method": "kwargs", "key": "enable_thinking",
                                             "value": False}})
    with kit.FakeNim(reasoning="kwargs", off_bug="answer_in_reasoning") as nim:
        result = expect(nc.check_reasoning_off(endpoint(nim.url), ev, nim.model, off), nc.FAIL,
                        "answer_in_reasoning_field")
    assert "reasoning" in result.message
    with kit.FakeNim(reasoning="kwargs", off_bug="answer_in_reasoning_stream") as nim:
        expect(nc.check_reasoning_off(endpoint(nim.url), ev, nim.model, off), nc.PASS, "ok")
        result = expect(nc.check_reasoning_off(endpoint(nim.url), ev, nim.model, off,
                                               stream=True), nc.FAIL, "answer_in_reasoning_field")
    assert "streaming" in result.message
    assert result.id == "reasoning_off_stream"  # its own id and files, even from the off entry
    assert result.evidence == ["evidence/reasoning_off_stream-1.json"]


def test_reasoning_off_reasoning_present_and_the_nano_caveat(ev: nc.Evidence) -> None:
    """Trap R9: the message cites Llama-Nemotron Nano's thinking with thinking off."""
    nano = "nvidia/llama-3.1-nemotron-nano-8b-v1"
    off = entry("reasoning_off", {"switch": {"method": "system", "key": None,
                                             "value": "detailed thinking off"}})
    with kit.FakeNim(nano, reasoning="always", reasoning_field="inline") as nim:
        result = expect(nc.check_reasoning_off(endpoint(nim.url), ev, nim.model, off), nc.FAIL,
                        "reasoning_present")
        (body,) = bodies(nim)
    assert "Nano" in result.message and any(nc.NANO_CARD in w for w in result.warnings)
    assert body["messages"][0] == {"role": "system", "content": "detailed thinking off"}
    with kit.FakeNim(reasoning="always") as nim:
        result = expect(nc.check_reasoning_off(endpoint(nim.url), ev, nim.model, off), nc.FAIL,
                        "reasoning_present")
    assert "Nano" not in result.message


@pytest.mark.parametrize(("knobs", "reason"), [
    ({"reasoning": "kwargs", "reject_fields": ("chat_template_kwargs",)}, "switch_rejected"),
    ({"reasoning": "kwargs", "answer": ""}, "empty_answer"),
    ({"routes": chat_route((200, completion("Paris is the capital and also", finish="length")))},
     "unexpected_finish_reason"),
    ({"routes": chat_route((200, completion("<think></think>Paris")))}, "ok"),
    ({"stream": "no_done", "reasoning": "kwargs"}, "stream_incomplete"),
])
def test_reasoning_off_other_results(ev: nc.Evidence, knobs: dict[str, Any], reason: str) -> None:
    off = entry("reasoning_off", {"switch": {"method": "kwargs", "key": "enable_thinking",
                                             "value": False}})
    stream = knobs.get("stream") == "no_done"
    with kit.FakeNim(**knobs) as nim:
        expect(nc.check_reasoning_off(endpoint(nim.url), ev, nim.model, off, stream=stream),
               nc.PASS if reason == "ok" else nc.FAIL, reason)


# --- json schema ---------------------------------------------------------------------------------


@pytest.mark.parametrize(("mode", "reason"), [
    ("ok", "ok"),
    ("ignored", "invalid_json"),
    ("invalid", "invalid_json"),
    ("truncated", "truncated_json"),
    ("in_reasoning", "json_only_in_reasoning"),
])
def test_json_schema_modes(ev: nc.Evidence, mode: str, reason: str) -> None:
    with kit.FakeNim(json=mode) as nim:
        expect(nc.check_json_schema(endpoint(nim.url), ev, nim.model, entry("json_schema")),
               nc.PASS if reason == "ok" else nc.FAIL, reason)
        (body,) = bodies(nim)
    assert body["response_format"] == nc.RESPONSE_FORMAT
    assert body["response_format"]["json_schema"]["name"] == "smoke"
    assert not {"guided_json", "nvext", "structured_outputs"} & set(body)


def test_json_schema_prompt_gives_no_json_hint(ev: nc.Evidence) -> None:
    """A server that ignores response_format must not pass by following the prompt: the
    prompt names neither JSON nor a schema key, so only enforced decoding answers in JSON."""
    with kit.FakeNim(json="ok") as nim:
        expect(nc.check_json_schema(endpoint(nim.url), ev, nim.model, entry("json_schema")),
               nc.PASS, "ok")
        (body,) = bodies(nim)
    prompt = body["messages"][-1]["content"].lower()
    assert prompt == nc.JSON_PROMPT.lower()
    for word in ("json", "{", *nc.JSON_SCHEMA["properties"]):
        assert word not in prompt


@pytest.mark.parametrize(("content", "finish", "status", "reason"), [
    ('{"name": "NIM"}', "stop", nc.FAIL, "schema_mismatch"),
    ('{"name": "NIM", "version": 2, "extra": true}', "stop", nc.FAIL, "schema_mismatch"),
    ("[1, 2]", "stop", nc.FAIL, "schema_mismatch"),
    ("null", "stop", nc.FAIL, "schema_mismatch"),
    ("", "stop", nc.FAIL, "invalid_json"),
    ('```json\n{"name": "NIM", "version": 2}\n```', "stop", nc.FAIL, "invalid_json"),
    ('{"name": "NIM", "version": 2}', "abort", nc.FAIL, "unexpected_finish_reason"),
    ('{"name": "X", "version": 3}', "stop", nc.PASS, "ok"),
    ('<think>a JSON object</think>{"name": "NIM", "version": 2}', "stop", nc.PASS, "ok"),
])
def test_json_schema_answers(ev: nc.Evidence, content: str, finish: str, status: str,
                             reason: str) -> None:
    with kit.FakeNim(routes=chat_route((200, completion(content, finish=finish)))) as nim:
        result = expect(nc.check_json_schema(endpoint(nim.url), ev, MODEL), status, reason)
    if content.startswith('{"name": "X"'):
        assert result.warnings  # other values are only a warning


def test_json_schema_sends_the_quiet_switch_from_the_plan(ev: nc.Evidence) -> None:
    plan = plan_for(served="nvidia/nemotron-3-nano")
    with kit.FakeNim("nvidia/nemotron-3-nano", reasoning="kwargs") as nim:
        expect(nc.check_json_schema(endpoint(nim.url), ev, nim.model,
                                    nf.plan_entry(plan, "json_schema")), nc.PASS, "ok")
        (body,) = bodies(nim)
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert body["max_tokens"] == 256


def test_json_schema_and_image_keep_the_budget_when_reasoning_stays_on(ev: nc.Evidence) -> None:
    """GLM-5.3-Flash cannot turn reasoning off and reasons at max effort by default ("use a
    max_tokens of at least 8192"): json_schema and image send at least 8192 tokens and the
    documented reasoning_effort low, as reasoning_on does."""
    served = "zai-org/glm-5.3-flash"
    plan = plan_for(served=served, backend="sglang")
    with kit.FakeNim(served, backend="sglang", reasoning="effort", vlm=True) as nim:
        ep = endpoint(nim.url)
        expect(nc.check_json_schema(ep, ev, nim.model, nf.plan_entry(plan, "json_schema")),
               nc.PASS, "ok")
        expect(nc.check_image(ep, ev, nim.model, nf.plan_entry(plan, "image")), nc.PASS, "ok")
        json_body, image_body, twin = bodies(nim)
    for body in (json_body, image_body):
        assert (body["max_tokens"], body["reasoning_effort"]) == (8192, "low")
    assert twin["max_tokens"] == 1 and twin["reasoning_effort"] == "low"


# --- image ---------------------------------------------------------------------------------------


def test_image_pass_compares_the_text_only_twin(ev: nc.Evidence) -> None:
    with kit.FakeNim(vlm=True) as nim:
        result = expect(nc.check_image(endpoint(nim.url), ev, nim.model,
                                       entry("image", {"max_tokens": 32})), nc.PASS, "ok")
        with_image, twin = bodies(nim)
    parts = with_image["messages"][0]["content"]
    assert parts[0] == {"type": "text", "text": nc.IMAGE_PROMPT}
    assert parts[1]["image_url"]["url"] == nc.data_url(nc.png_solid())
    assert twin["messages"][0]["content"] == [parts[0]] and twin["max_tokens"] == 1
    assert with_image["max_tokens"] == 32
    details = result.details
    assert details["prompt_tokens_with_image"] - details["prompt_tokens_text_only"] == 256
    assert details["names_red"] is True and result.attempts == 1 and len(result.evidence) == 2


@pytest.mark.parametrize(("knobs", "reason"), [
    ({"vlm": "image_ignored"}, "image_ignored"),
    ({"vlm": True, "answer": ""}, "empty_answer"),
    ({"vlm": False}, "http_error"),
    ({"routes": chat_route((200, completion("Red")), (500, {"detail": "x"}))}, "http_error"),
    ({"routes": chat_route((200, completion("Red", usage=False)))}, "bad_response_shape"),
])
def test_image_failures(ev: nc.Evidence, knobs: dict[str, Any], reason: str) -> None:
    with kit.FakeNim(**knobs) as nim:
        result = expect(nc.check_image(endpoint(nim.url), ev, nim.model, entry("image")),
                        nc.FAIL, reason)
    if "routes" in knobs and reason == "http_error":
        assert "twin" in result.message


TEXT_ONLY_400 = {"error": {"message": "At most 0 image(s) may be provided in one prompt.",
                           "type": "BadRequestError", "param": "image", "code": 400}}


@pytest.mark.parametrize("profile", [None, {"name": "vllm-fp8-tp8-pp1"}])
def test_image_on_a_documented_text_only_profile_is_not_applicable(
        ev: nc.Evidence, profile: dict[str, Any] | None) -> None:
    """deepseek-v4.1-flash's default FP8 profiles on B200, GB200 and H200 take text only and
    answer an image with a documented 400. When the active profile's id is unknown (no
    /v1/metadata, the hosted API, a profile name only), that 400 is text_only_profile, not a
    fail; any other 400, or that 400 from a model with no text-only profile, is still one."""
    served = "deepseek-ai/deepseek-v4.1-flash"
    record, _ = nf.match(TABLE, served_id=served)
    plan = nf.decide(record, checks=nf.CHECKS, generation="2.1.5", backend="vllm",
                     profile=profile, table=TABLE)
    item = nf.plan_entry(plan, "image")
    assert item["decision"] == "test" and item["warnings"]
    with kit.FakeNim(served, routes=chat_route((400, TEXT_ONLY_400))) as nim:
        result = expect(nc.check_image(endpoint(nim.url), ev, nim.model, item),
                        nc.NOT_APPLICABLE, "text_only_profile")
        assert len(nim.posts()) == 1  # no text-only twin
    assert "At most 0 image(s) may be provided" in result.message
    assert result.sources == item["sources"] and result.evidence == ["evidence/image-1.json"]
    other = {"error": {"message": "the image is too large", "code": 400}}
    with kit.FakeNim(served, routes=chat_route((400, other))) as nim:
        expect(nc.check_image(endpoint(nim.url), ev, nim.model, item), nc.FAIL, "http_error")
    with kit.FakeNim(routes=chat_route((400, TEXT_ONLY_400))) as nim:
        expect(nc.check_image(endpoint(nim.url), ev, nim.model, entry("image")), nc.FAIL,
               "http_error")


# --- concurrency (trap R13) ----------------------------------------------------------------------


@pytest.mark.parametrize("n", [1, 4, 8])
def test_concurrency_sends_n_at_once_without_a_readiness_probe(ev: nc.Evidence, n: int) -> None:
    with kit.FakeNim(delay=HOLD) as nim:  # long enough that a busy runner still overlaps all n
        result = expect(nc.check_concurrency(endpoint(nim.url), ev, nim.model,
                                             entry("concurrency"), n=n), nc.PASS, "ok")
        assert nim.peak == n and len(nim.posts()) == n
        assert nim.gets() == []  # no health probe while the requests run
    assert len(result.evidence) == n and result.details["n"] == n
    assert result.evidence[0] == "evidence/concurrency-1.json"
    assert len(result.details["latencies_ms"]) == n and result.latency_ms == result.details[
        "wall_ms"]


def test_concurrency_default_n_and_the_off_switch(ev: nc.Evidence) -> None:
    plan = plan_for(served="nvidia/nemotron-3-nano")
    with kit.FakeNim("nvidia/nemotron-3-nano", reasoning="kwargs", delay=0.2) as nim:
        result = expect(nc.check_concurrency(endpoint(nim.url, port_forward=True), ev, nim.model,
                                             nf.plan_entry(plan, "concurrency")), nc.PASS, "ok")
        sent = bodies(nim)
    assert len(sent) == nc.DEFAULT_CONCURRENCY
    assert all(b["chat_template_kwargs"] == {"enable_thinking": False} for b in sent)
    assert all(b["max_tokens"] == 64 for b in sent)
    assert any("port-forward" in warning for warning in result.warnings)


@pytest.mark.parametrize(("knobs", "reason"), [
    ({"fail_concurrent": 2, "delay": 0.3}, "rejected_503"),
    ({"routes": chat_route((502, "<html>Bad Gateway</html>"))}, "gateway_error"),
    ({"routes": chat_route((504, "<html>Gateway Timeout</html>"))}, "gateway_error"),
    ({"routes": chat_route((429, "Too Many Requests", "text/plain"))}, "rate_limited"),
    ({"routes": chat_route((200, completion("")))}, "empty_completion"),
    ({"routes": chat_route((400, {"error": {"message": "bad"}}))}, "http_error"),
])
def test_concurrency_failures(ev: nc.Evidence, knobs: dict[str, Any], reason: str) -> None:
    with kit.FakeNim(**knobs) as nim:
        result = expect(nc.check_concurrency(endpoint(nim.url), ev, nim.model,
                                             entry("concurrency")), nc.FAIL, reason)
        assert len(nim.posts()) == 4  # never retried
    assert "of 4 failed" in result.message


# --- the answer only in the reasoning field (chat, stream, concurrency) ----------------------


def reasoning_only_stream(field: str, finish: str) -> str:
    """A canned stream whose text is all in the reasoning field."""
    return sse(chunk({"role": "assistant", field: "The user wants "}),
               chunk({field: "OK."}), chunk({}, finish),
               {"id": "c1", "object": "chat.completion.chunk", "choices": [],
                "usage": {"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13}})


@pytest.mark.parametrize("field", ["reasoning", "reasoning_content"])
def test_chat_stream_and_concurrency_fail_on_the_answer_in_the_reasoning_field(
    ev: nc.Evidence, field: str
) -> None:
    """A misconfigured reasoning parser routes the whole answer into the reasoning field:
    content null and finish_reason stop. Every client that reads content gets nothing."""
    body = completion(None, reasoning="OK", field=field)
    with kit.FakeNim(routes=chat_route((200, body))) as nim:
        chat = expect(nc.check_chat(endpoint(nim.url), ev, nim.model, entry("chat")),
                      nc.FAIL, "answer_missing")
        many = expect(nc.check_concurrency(endpoint(nim.url), ev, nim.model,
                                           entry("concurrency")), nc.FAIL, "answer_missing")
    assert "4 of 4 failed (answer_missing x4)" in many.message
    text = reasoning_only_stream(field, "stop")
    with kit.FakeNim(routes=chat_route((200, text, "text/event-stream"))) as nim:
        stream = expect(nc.check_stream(endpoint(nim.url), ev, nim.model, entry("stream")),
                        nc.FAIL, "answer_missing")
    for result in (chat, stream, many):
        assert f"{field} holds all the text" in result.message
        assert result.hint == nc.REASONING_FIELD_HINT
        assert "reasoning parser" in result.hint and "nim-deploy" in result.hint
    assert "(streamed)" in stream.message
    # the hosted API's server settings are not the user's: no deployment hint there
    with kit.FakeNim(routes=chat_route((200, body))) as nim:
        hosted = expect(nc.check_chat(endpoint(nim.url, hosted=True), ev, nim.model,
                                      entry("chat")), nc.FAIL, "answer_missing")
    assert hosted.hint is None


def test_chat_fails_on_inline_reasoning_with_no_answer_and_stop(ev: nc.Evidence) -> None:
    body = completion("<think>The user wants OK.</think>")
    with kit.FakeNim(routes=chat_route((200, body))) as nim:
        result = expect(nc.check_chat(endpoint(nim.url), ev, nim.model, entry("chat")),
                        nc.FAIL, "answer_missing")
    assert result.details["where"] == "inline" and result.hint is None
    assert "but no answer, finish_reason stop" in result.message


def test_chat_stream_and_concurrency_warn_when_the_budget_ran_out(ev: nc.Evidence) -> None:
    """finish_reason length: the budget ran out in the reasoning; a pass with a warning."""
    body = completion(None, reasoning="The user wants", finish="length")
    with kit.FakeNim(routes=chat_route((200, body))) as nim:
        chat = expect(nc.check_chat(endpoint(nim.url), ev, nim.model, entry("chat")),
                      nc.PASS, "ok")
        many = expect(nc.check_concurrency(endpoint(nim.url), ev, nim.model,
                                           entry("concurrency")), nc.PASS, "ok")
    text = reasoning_only_stream("reasoning", "length")
    with kit.FakeNim(routes=chat_route((200, text, "text/event-stream"))) as nim:
        stream = expect(nc.check_stream(endpoint(nim.url), ev, nim.model, entry("stream")),
                        nc.PASS, "ok")
    assert "reasoning only, no answer within" in chat.message
    for result in (chat, stream):
        assert any("finish_reason is length" in warning for warning in result.warnings)
    assert any("4 of 4 answers have reasoning but no content" in warning
               for warning in many.warnings)
    assert all(result.hint is None for result in (chat, stream, many))


@pytest.mark.parametrize(("n", "code"), [(9, "concurrency_too_high"),
                                         (32, "concurrency_too_high"), (0, "bad_input"),
                                         (-1, "bad_input")])
def test_concurrency_limits(ev: nc.Evidence, n: int, code: str) -> None:
    with pytest.raises(nsc.ScriptError) as info:
        nc.check_concurrency(endpoint(CLOSED), ev, MODEL, entry("concurrency"), n=n)
    assert info.value.code == code and info.value.exit_code == nsc.EXIT_INPUT
    if code == "concurrency_too_high":
        assert "nim-benchmark" in info.value.fix


def test_concurrency_on_the_hosted_api_stays_at_4(ev: nc.Evidence) -> None:
    """docs-12: on the hosted API (rate-limited per key) at most 4 requests go at once."""
    for n in (5, 8):
        with pytest.raises(nsc.ScriptError) as info:
            nc.check_concurrency(endpoint(CLOSED, hosted=True), ev, MODEL, entry("concurrency"),
                                 n=n)
        assert info.value.code == "concurrency_too_high" and "hosted API" in info.value.message
    with kit.FakeNim(delay=HOLD) as nim:  # an endpoint marked hosted, on this machine
        expect(nc.check_concurrency(endpoint(nim.url, hosted=True), ev, nim.model,
                                    entry("concurrency"), n=4), nc.PASS, "ok")
        assert nim.peak == 4


def test_concurrency_raises_a_request_threads_error_in_the_caller(
    ev: nc.Evidence, monkeypatch: pytest.MonkeyPatch
) -> None:
    """docs-1: an error inside one request thread reaches the caller as itself (for the run's
    internal_error), not as an AssertionError on a missing answer."""
    def broken(*args: Any, **kwargs: Any) -> nsc.HttpResult:
        raise RuntimeError("boom in a request thread")

    monkeypatch.setattr(nsc, "request_json", broken)
    with pytest.raises(RuntimeError, match="boom in a request thread"):
        nc.check_concurrency(endpoint(CLOSED), ev, MODEL, entry("concurrency"), n=2)


# --- plan entries, known issues, dispatch --------------------------------------------------------


def test_not_applicable_entries_send_nothing(ev: nc.Evidence) -> None:
    plan = plan_for(served="openai/gpt-oss-20b", checks=nf.CHECKS)
    with kit.FakeNim("openai/gpt-oss-20b") as nim:
        for item in plan["checks"]:
            if item["decision"] != "not_applicable" or item["check"] in ("health", "models"):
                continue
            result = nc.run_check(item["check"], endpoint(nim.url), ev, nim.model, item)
            expect(result, nc.NOT_APPLICABLE, item["reason"])
            assert result.sources == item["sources"] and result.evidence == []
            assert result.message == nc.NA_MESSAGES[item["reason"]]
        assert nim.requests == []
    off = nf.plan_entry(plan, "reasoning_off")
    assert nc.result_from_entry(off).reason == "always_on"


def test_known_issues_are_attached(ev: nc.Evidence) -> None:
    with kit.FakeNim(control_tokens=True) as nim:
        result = expect(nc.check_chat(endpoint(nim.url), ev, nim.model,
                                      entry("chat", known_issues=[ISSUE])), nc.FAIL,
                        "control_tokens_in_content")
    assert result.known_issues == [ISSUE] and "A known issue" in result.message
    assert "see the warnings" in result.message and "attached" not in result.message
    # a fail to reach the endpoint is not about the model: no pointer to its known issues
    with kit.FakeNim(require_key=kit.fake_key()) as nim:
        result = expect(nc.check_chat(endpoint(nim.url), ev, nim.model,
                                      entry("chat", known_issues=[ISSUE])), nc.FAIL,
                        "unauthorized")
    assert result.known_issues == [ISSUE] and "known issue" not in result.message
    assert not nc.points_to_known_issues(nc.FAIL, "unauthorized", [ISSUE])
    assert nc.points_to_known_issues(nc.FAIL, "control_tokens_in_content", [ISSUE])
    assert not nc.points_to_known_issues(nc.PASS, nc.OK, [ISSUE])
    plan = plan_for(served="openai/gpt-oss-20b")
    assert nf.plan_entry(plan, "chat")["known_issues"]
    with kit.FakeNim("openai/gpt-oss-20b", reasoning="always") as nim:
        result = expect(nc.check_chat(endpoint(nim.url), ev, nim.model,
                                      nf.plan_entry(plan, "chat")), nc.PASS, "ok")
    assert result.known_issues == nf.plan_entry(plan, "chat")["known_issues"]


def test_run_check_dispatch_and_unknown_ids(ev: nc.Evidence) -> None:
    assert set(nc.CHECK_FUNCTIONS) == set(nf.CHECKS) - {"health", "models"}
    with pytest.raises(ValueError):
        nc.run_check("health", endpoint(CLOSED), ev, MODEL)


def test_every_request_of_a_full_run_uses_only_allowed_fields(ev: nc.Evidence) -> None:
    """Traps R12, T2 and T13 over a full extended run: no forbidden field, no "required"
    tool_choice, parallel_tool_calls only true and only in tools_parallel, one user turn."""
    plan = plan_for(served="nvidia/nemotron-3-nano", checks=nf.CHECKS)
    with kit.FakeNim("nvidia/nemotron-3-nano", reasoning="kwargs") as nim:
        ep = endpoint(nim.url)
        for item in plan["checks"]:
            if item["check"] not in ("health", "models"):
                nc.run_check(item["check"], ep, ev, nim.model, item)
        seen = nim.posts()
    assert len(seen) >= 12
    assert nim.sent_fields() <= nc.REQUEST_FIELDS
    assert not nim.sent_fields() & nc.NEVER_SEND
    for request in seen:
        body = request.body
        assert body.get("tool_choice") != "required"
        assert "chat_template" not in (body.get("chat_template_kwargs") or {})
        if "parallel_tool_calls" in body:
            assert body["parallel_tool_calls"] is True
            assert body["messages"][-1]["content"] == nc.PARALLEL_PROMPT
        roles = [message["role"] for message in body["messages"]]
        assert roles.count("user") == 1 and not {"assistant", "tool"} & set(roles)
        assert re.fullmatch(r"nsc-t-[a-z_]+-\d+", request.headers["x-request-id"])


def test_tool_mode_switches_from_the_plan(ev: nc.Evidence) -> None:
    """Traps T8, T9 and T11: Llama-Nemotron calls tools with thinking off; Nemotron 3 with
    enable_thinking and force_nonempty_content and a 4096-token budget."""
    v1 = plan_for(served="nvidia/llama-3.3-nemotron-super-49b-v1")
    n3 = plan_for(served="nvidia/nemotron-3-nano")
    with kit.FakeNim("nvidia/llama-3.3-nemotron-super-49b-v1", reasoning="system") as nim:
        expect(nc.check_tools(endpoint(nim.url), ev, nim.model, nf.plan_entry(v1, "tools")),
               nc.PASS, "ok")
        (body,) = bodies(nim)
    assert body["messages"][0] == {"role": "system", "content": "detailed thinking off"}
    with kit.FakeNim("nvidia/nemotron-3-nano", reasoning="kwargs") as nim:
        expect(nc.check_tools(endpoint(nim.url), ev, nim.model, nf.plan_entry(n3, "tools")),
               nc.PASS, "ok")
        (body,) = bodies(nim)
    assert body["chat_template_kwargs"] == {"enable_thinking": True,
                                            "force_nonempty_content": True}
    assert body["max_tokens"] == 4096 and (body["temperature"], body["top_p"]) == (0.6, 0.95)


def test_library_main_prints_a_notice() -> None:
    code, payload, _ = kit.run_cli("nsc_checks.py")
    assert code == 0 and payload["state"] == "library"
    assert payload["facts"]["checks"][:2] == ["health", "models"]
    code, text, _ = kit.run_cli("nsc_checks.py", "--help", raw=True)
    assert code == 0 and "nim-smoke-check" in text
    code, payload, _ = kit.run_cli("nsc_checks.py", "--bogus")
    assert code == 2 and payload["errors"][0]["code"] == "bad_input"


# --- references/checks.md ------------------------------------------------------------------------


def _codes_in(text: str) -> set[str]:
    return set(re.findall(r"`([a-z][a-z0-9_]*)`", text))


def test_checks_md_lists_every_reason_code() -> None:
    text = CHECKS_MD.read_text(encoding="utf-8")
    codes = _codes_in(text)
    wanted = (set(nc.REASONS) | set(nf.NOT_APPLICABLE_REASONS) | set(nf.TEST_REASONS)
              | set(nf.PARSER_MISSING_REASONS)
              | {"model_ambiguous", "no_models", "endpoint_unreachable", "concurrency_too_high",
                 "first_request_timeout", "interrupted", "internal_error"})
    assert wanted - codes == set()
    assert [check for check in nf.CHECKS if not re.search(rf"\b{check}\b", text)] == []


def test_checks_md_cites_public_sources_only_and_follows_the_wording_rules() -> None:
    text = CHECKS_MD.read_text(encoding="utf-8")
    assert "SPDX-License-Identifier: CC-BY-4.0 AND Apache-2.0" in text.split("\n## ", 1)[0]
    urls = re.findall(r"https?://[^\s)`>|]+", text)
    assert len(urls) >= 30
    # the hosted endpoint itself (named, not cited), the kubectl docs and NVIDIA's NIM FAQ
    extra_public = ("integrate.api.nvidia.com", "kubernetes.io", "forums.developer.nvidia.com")
    for url in urls:
        host = re.sub(r"^https?://([^/]+).*", r"\1", url)
        assert url.startswith("https://"), url
        assert nf.check_source_url(url) is None or host in extra_public, url
    forbidden = ["NIM " + "CLI", "Lean" + "Tune", "Infer" + "Fit", "write" + "-skill",
                 "nim" + "cli-model-ops", *nf.PRIVATE_MARKERS]  # parts: no literal here
    for path in (CHECKS_MD, Path(nc.__file__), Path(__file__)):
        body = path.read_text(encoding="utf-8").lower()
        for word in forbidden:
            assert word.lower() not in body, (path.name, word)
        assert not re.search(r"\b[a-z]+-cli\b", body), path.name


def test_checks_md_routes_rule_lists_every_get_a_run_sends(tmp_path: Path) -> None:
    """docs-14: the routes rule names every GET a run sends, the /v1/version and /v1/metadata
    probes too (recorded, never judged), and the hosted /v1/models/ID."""
    with kit.FakeNim(MODEL) as nim:
        code, payload, _ = kit.run_cli("nimsmoke.py", "run", "--url", nim.url, "--checks",
                                       "health,models,chat", "--out", str(tmp_path / "out"))
        gets = {seen.path for seen in nim.requests if seen.method == "GET"}
    assert code == 0, payload
    assert {"/v1/version", "/v1/metadata", "/v1/models", "/v1/health/ready"} <= gets
    rule = next(line for line in CHECKS_MD.read_text(encoding="utf-8").splitlines()
                if line.startswith("| routes |"))
    listed = set(re.findall(r"`(/[^`]*)`", rule))
    assert gets - listed == set()
    assert {"/health", "/v1/models/ID", "/v1/chat/completions"} <= listed


def _strings(node: ast.AST) -> set[str]:
    return {n.value for n in ast.walk(node) if isinstance(n, ast.Constant)
            and isinstance(n.value, str)}


def _columns(names: list[str], rows: list[ast.expr]) -> dict[str, list[ast.expr]]:
    """The values each name takes in a literal table of rows."""
    columns: dict[str, list[ast.expr]] = {name: [] for name in names}
    for row in rows:
        if isinstance(row, ast.Call) and getattr(row.func, "attr", None) == "param":
            values = row.args
        elif len(names) > 1 and isinstance(row, (ast.Tuple, ast.List)):
            values = row.elts
        else:
            values = [row]
        for name, value in zip(names, values, strict=False):
            columns[name].append(value)
    return columns


def _tables(test: ast.FunctionDef) -> list[dict[str, list[ast.expr]]]:
    """The literal tables a test walks: its parametrize lists and its `for x, y in (...)` loops."""
    tables = []
    for decorator in test.decorator_list:
        if (isinstance(decorator, ast.Call) and getattr(decorator.func, "attr", None)
                == "parametrize" and len(decorator.args) == 2
                and isinstance(decorator.args[1], (ast.List, ast.Tuple))):
            with contextlib.suppress(ValueError):
                spec = ast.literal_eval(decorator.args[0])
                names = [n.strip() for n in spec.split(",")] if isinstance(spec, str) else spec
                tables.append(_columns(list(names), decorator.args[1].elts))
    for loop in ast.walk(test):
        if isinstance(loop, ast.For) and isinstance(loop.iter, (ast.List, ast.Tuple)):
            targets = loop.target.elts if isinstance(loop.target, ast.Tuple) else [loop.target]
            names = [t.id if isinstance(t, ast.Name) else "" for t in targets]
            tables.append(_columns(names, loop.iter.elts))
    return tables


def asserted_strings(source: str) -> set[str]:
    """The string literals the tests in `source` assert: the reason argument of expect(), a
    literal in an assert statement, and the values a parametrize column or a loop variable
    takes when the test passes its name to one of those. A docstring or a request input does
    not count."""
    found: set[str] = set()
    for test in ast.walk(ast.parse(source)):
        if not isinstance(test, ast.FunctionDef):
            continue
        checked: list[ast.AST] = []
        for node in ast.walk(test):
            if isinstance(node, ast.Assert):
                checked.append(node.test)
            elif isinstance(node, ast.Call) and getattr(node.func, "id", None) == "expect":
                checked += node.args[2:3] + [k.value for k in node.keywords if k.arg == "reason"]
        names = {n.id for part in checked for n in ast.walk(part) if isinstance(n, ast.Name)}
        for part in checked:
            found |= _strings(part)
        for table in _tables(test):
            for name, values in table.items():
                if name in names:
                    found |= {text for value in values for text in _strings(value)}
    return found


def test_every_reason_code_has_a_test() -> None:
    """Each reason a check reports is asserted by a test of this file (see asserted_strings)."""
    source = Path(__file__).read_text(encoding="utf-8")
    assert set(nc.REASONS) - asserted_strings(source) == set()


def test_reason_coverage_counts_only_asserted_reasons() -> None:
    """A reason named only in a docstring, a request input, or a parametrize column or loop
    variable the test never asserts is not covered: before, any string literal counted."""
    source = '''
@pytest.mark.parametrize(("body", "reason"), [({"type": "param_input"}, "param_reason")])
def test_one(body, reason):
    """Mentions docstring_only."""
    result = run(body)
    expect(result, "fail", reason)
    expect(run({"type": "request_input"}), "fail", "expect_reason")
    assert result.reason == "assert_reason"
    for status, why in ((403, "loop_reason"),):
        expect(run(status), "fail", why)


@pytest.mark.parametrize("reason", ["column_not_asserted"])
def test_two(reason):
    run(reason)
    for body in ({"type": "loop_input"},):
        run(body)
'''
    found = asserted_strings(source)
    assert {"param_reason", "expect_reason", "assert_reason", "loop_reason"} <= found
    assert found.isdisjoint({"docstring_only", "param_input", "request_input",
                             "column_not_asserted", "loop_input", "fail"})


def test_reason_table_and_messages_are_consistent() -> None:
    assert all(re.fullmatch(r"[a-z][a-z0-9_]*", code) for code in nc.REASONS)
    assert set(nc.NA_MESSAGES) >= set(nf.NOT_APPLICABLE_REASONS)
    assert nc.STATUSES == ("pass", "fail", "not_applicable")
