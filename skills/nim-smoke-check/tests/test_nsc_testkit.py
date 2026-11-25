# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for tests/nsc_testkit.py: every FakeNim knob, the fake tools, run_cli and serve mode."""

from __future__ import annotations

import ast
import base64
import contextlib
import http.client
import json
import os
import queue
import re
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nsc_testkit as kit

TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name, for example Paris"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["city", "unit"],
        },
    },
}
WEATHER = "What is the weather in Paris right now? Use celsius. Call the get_weather tool."
SCHEMA = {
    "type": "object",
    "properties": {"name": {"type": "string"}, "version": {"type": "integer"}},
    "required": ["name", "version"],
    "additionalProperties": False,
}
PNG = "data:image/png;base64," + base64.b64encode(b"\x89PNG\r\n\x1a\n" + bytes(24)).decode()


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args: Any) -> None:
        return None


OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect)


def call(url: str, path: str, body: Any = None, *, method: str | None = None,
         headers: dict[str, str] | None = None) -> tuple[int, dict[str, str], bytes]:
    """(status, lower-case headers, body bytes) of one request, errors included."""
    data = json.dumps(body).encode() if body is not None else None
    sent = {"Content-Type": "application/json"} if data is not None else {}
    request = urllib.request.Request(url + path, data=data, method=method,
                                     headers={**sent, **(headers or {})})
    try:
        with OPENER.open(request, timeout=20) as answer:
            return answer.status, {k.lower(): v for k, v in answer.headers.items()}, answer.read()
    except urllib.error.HTTPError as exc:
        return exc.code, {k.lower(): v for k, v in exc.headers.items()}, exc.read()


def chat(url: str, body: dict[str, Any], **kw: Any) -> tuple[int, Any]:
    status, _, raw = call(url, "/v1/chat/completions", body, **kw)
    return status, json.loads(raw) if raw else None


def ask(nim: kit.FakeNim, content: Any, **fields: Any) -> dict[str, Any]:
    return {"model": nim.model, "messages": [{"role": "user", "content": content}], **fields}


def message(nim: kit.FakeNim, content: Any, **fields: Any) -> tuple[dict[str, Any], dict]:
    status, body = chat(nim.url, ask(nim, content, **fields))
    assert status == 200, body
    return body["choices"][0]["message"], body


def sse(raw: bytes) -> dict[str, Any]:
    """events, comments and done of a server-sent event body."""
    events, comments, done = [], [], False
    for line in raw.decode().splitlines():
        if line.startswith(":"):
            comments.append(line)
        elif line.startswith("data: "):
            payload = line[len("data: "):]
            if payload == "[DONE]":
                done = True
            else:
                events.append(json.loads(payload))
    return {"events": events, "comments": comments, "done": done}


def stream(nim: kit.FakeNim, content: Any, **fields: Any) -> tuple[int, str, dict[str, Any]]:
    body = ask(nim, content, stream=True, stream_options={"include_usage": True}, **fields)
    status, headers, raw = call(nim.url, "/v1/chat/completions", body)
    return status, headers.get("content-type", ""), sse(raw)


def joined(events: list[dict[str, Any]], key: str) -> str:
    return "".join(
        choice["delta"].get(key) or "" for e in events for choice in e.get("choices", [])
    )


# --- models, health, version, metadata, metrics --------------------------------------------------


def test_models_cards_have_the_vllm_fields_and_lora_parents() -> None:
    lora = {"id": "llama-lora", "parent": "meta/llama-3.1-8b-instruct", "root": "/loras/a"}
    with kit.FakeNim(extra_models=(lora,), root="/models/llama") as nim:
        status, _, raw = call(nim.url, "/v1/models")
        cards = json.loads(raw)["data"]
    assert status == 200 and [c["id"] for c in cards] == [nim.model, "llama-lora"]
    base, adapter = cards
    assert set(base) == {"id", "object", "created", "owned_by", "root", "parent",
                         "max_model_len", "permission"}
    assert base["owned_by"] == "vllm" and base["root"] == "/models/llama"
    assert base["parent"] is None and adapter["parent"] == nim.model
    assert base["permission"][0]["object"] == "model_permission"


def test_sglang_cards_have_no_parent_and_name_the_backend() -> None:
    with kit.FakeNim(backend="sglang") as nim:
        card = json.loads(call(nim.url, "/v1/models")[2])["data"][0]
        wrong = {"model": "nope", "messages": [{"role": "user", "content": "x"}]}
        status, one = chat(nim.url, wrong)
    assert card["owned_by"] == "sglang" and "parent" not in card
    assert status == 404 and one["object"] == "error" and "does not exist" in one["message"]


@pytest.mark.parametrize(
    ("health", "live", "ready", "bare"),
    [("ok", 200, 200, 200), ("loading", 200, 503, 503), ("missing", 404, 404, 404),
     ("bare", 404, 404, 200)],
)
def test_health_states(health: str, live: int, ready: int, bare: int) -> None:
    with kit.FakeNim(health=health) as nim:
        assert call(nim.url, "/v1/health/live")[0] == live
        status, _, raw = call(nim.url, "/v1/health/ready")
        assert status == ready
        assert call(nim.url, "/health")[0] == bare
    if ready == 200:
        assert json.loads(raw) == {"object": "health.response", "message": "ready",
                                   "status": "ready"}


def test_loading_turns_ready_after_the_set_time() -> None:
    with kit.FakeNim(health="loading", ready_after=0.3) as nim:
        assert call(nim.url, "/v1/health/ready")[0] == 503
        time.sleep(0.4)
        assert call(nim.url, "/v1/health/ready")[0] == 200


def test_version_metadata_metrics_and_unknown_routes() -> None:
    with kit.FakeNim(version="2.1.6") as nim:
        assert json.loads(call(nim.url, "/v1/version")[2]) == {"release": "2.1.6", "api": "1.0.0"}
        meta = json.loads(call(nim.url, "/v1/metadata")[2])
        metrics = call(nim.url, "/v1/metrics")[2].decode()
        assert call(nim.url, "/v1/nothing")[0] == 404
        assert call(nim.url, "/v1/load_lora_adapter", {"x": 1})[0] == 404
        assert call(nim.url, "/v1/models", method="DELETE")[0] == 405
        assert [s.method for s in nim.requests][-1] == "DELETE"
    assert meta["modelInfo"][0]["shortName"] == nim.model and meta["selectedModelProfileId"]
    assert 'vllm:num_requests_running{model_name="meta/llama-3.1-8b-instruct"}' in metrics
    with kit.FakeNim(metadata={"anything": 1}, backend="sglang") as nim:
        assert json.loads(call(nim.url, "/v1/metadata")[2]) == {"anything": 1}
        assert "sglang:num_requests_waiting" in call(nim.url, "/v1/metrics")[2].decode()


def test_hosted_style_single_model_route() -> None:
    with kit.FakeNim() as nim:
        assert json.loads(call(nim.url, f"/v1/models/{nim.model}")[2])["id"] == nim.model
        assert call(nim.url, "/v1/models/nosuch/model")[0] == 404


# --- chat -----------------------------------------------------------------------------------------


def test_chat_has_the_vllm_shape_and_answers_the_smoke_prompts() -> None:
    with kit.FakeNim() as nim:
        msg, body = message(nim, "Reply with the single word OK.", max_tokens=64, temperature=0)
        assert message(nim, "What is the capital of France? Answer with one word.")[0][
            "content"] == "Paris"
        assert message(nim, "What is 2 + 2? Answer with the number only.")[0]["content"] == "4"
    assert body["object"] == "chat.completion" and body["model"] == nim.model
    assert msg == {"role": "assistant", "content": "OK", "refusal": None, "annotations": None,
                   "audio": None, "function_call": None, "reasoning": None}
    assert "tool_calls" not in msg  # vLLM 0.31 drops an empty list
    choice = body["choices"][0]
    assert choice["finish_reason"] == "stop" and choice["logprobs"] is None
    usage = body["usage"]
    assert usage["completion_tokens"] == 1 and usage["total_tokens"] == (
        usage["prompt_tokens"] + usage["completion_tokens"])


def test_chat_errors_have_the_vllm_texts() -> None:
    with kit.FakeNim() as nim:
        status, body = chat(nim.url, {"model": "other/x", "messages": [{"role": "user",
                                                                         "content": "hi"}]})
        assert status == 404
        assert body["error"] == {"message": "The model `other/x` does not exist.",
                                 "type": "NotFoundError", "param": "model", "code": 404}
        status, body = chat(nim.url, ask(nim, "hi"), headers={"Content-Type": "text/plain"})
        assert status == 400 and "application/json" in body["error"]["message"]
        status, body = chat(nim.url, ask(nim, "hi", tool_choice="auto"))
        assert status == 400 and "`tools` must be set" in body["error"]["message"]


def test_request_records_headers_body_and_request_id_echo() -> None:
    with kit.FakeNim() as nim:
        status, headers, _ = call(nim.url, "/v1/chat/completions", ask(nim, "hi"),
                                  headers={"X-Request-Id": "nsc-1-chat-1"})
        call(nim.url, "/v1/models")
        seen = nim.posts("/v1/chat/completions")[0]
    assert status == 200 and headers["x-request-id"] == "nsc-1-chat-1"
    assert seen.headers["x-request-id"] == "nsc-1-chat-1"
    assert seen.body == ask(nim, "hi") and json.loads(seen.raw) == seen.body
    assert seen.ended is not None and seen.ended >= seen.started and seen.status == 200
    assert len(nim.gets()) == 1 and nim.sent_fields() == {"model", "messages"}


def test_answer_control_tokens_and_rejected_fields() -> None:
    with kit.FakeNim(answer="") as nim:
        msg, body = message(nim, "Reply with the single word OK.")
        assert msg["content"] == "" and body["usage"]["completion_tokens"] == 0
    with kit.FakeNim(control_tokens=True) as nim:
        assert message(nim, "hi")[0]["content"].startswith("<|channel|>final<|message|>")
    with kit.FakeNim(reject_fields=("chat_template_kwargs",)) as nim:
        status, body = chat(nim.url, ask(nim, "hi", chat_template_kwargs={"x": 1}))
        assert status == 400 and "chat_template_kwargs" in body["error"]["message"]


def test_canned_routes_win_and_answer_in_order() -> None:
    routes = {"/v1/models": (200, "<html>proxy login</html>"),
              "/v1/chat/completions": [(502, "<html>Bad Gateway</html>"), (429, {"e": 1})]}
    with kit.FakeNim(routes=routes) as nim:
        status, headers, raw = call(nim.url, "/v1/models")
        assert status == 200 and headers["content-type"] == "text/html"
        assert raw.startswith(b"<html>")
        assert call(nim.url, "/v1/chat/completions", ask(nim, "a"))[0] == 502
        assert call(nim.url, "/v1/chat/completions", ask(nim, "b"))[0] == 429
        assert call(nim.url, "/v1/chat/completions", ask(nim, "c"))[0] == 429


def test_canned_json_bodies_keep_a_given_content_type() -> None:
    """A dict or list body is sent as JSON: application/json by default, or the content type the
    answer names (a gateway that labels JSON as text/plain)."""
    cards = {"object": "list", "data": []}
    routes = {"/v1/models": (200, cards, "text/plain"), "/v1/version": (200, ["x"])}
    with kit.FakeNim(routes=routes) as nim:
        status, headers, raw = call(nim.url, "/v1/models")
        _, default, listed = call(nim.url, "/v1/version")
    assert (status, headers["content-type"], json.loads(raw)) == (200, "text/plain", cards)
    assert (default["content-type"], json.loads(listed)) == ("application/json", ["x"])


def test_a_misspelt_knob_is_refused() -> None:
    with pytest.raises(ValueError, match="tools"):
        kit.FakeNim(tools="noparser")
    with pytest.raises(ValueError, match="vlm"):
        kit.FakeNim(vlm=1)


# --- streaming ------------------------------------------------------------------------------------


def test_stream_ok_has_chunks_a_finish_chunk_usage_and_done() -> None:
    with kit.FakeNim() as nim:
        status, kind, parsed = stream(nim, "Reply with the single word OK.")
    events = parsed["events"]
    assert status == 200 and kind.startswith("text/event-stream") and parsed["done"]
    assert all(e["object"] == "chat.completion.chunk" for e in events)
    assert events[0]["choices"][0]["delta"] == {"role": "assistant", "content": ""}
    assert joined(events, "content") == "OK"
    finishes = [c["finish_reason"] for e in events for c in e["choices"] if c["finish_reason"]]
    assert finishes == ["stop"]
    assert events[-1]["choices"] == [] and events[-1]["usage"]["completion_tokens"] == 1


def test_stream_usage_only_when_asked() -> None:
    with kit.FakeNim() as nim:
        status, _, raw = call(nim.url, "/v1/chat/completions", ask(nim, "hi", stream=True))
    assert status == 200 and all(e["choices"] for e in sse(raw)["events"])


@pytest.mark.parametrize("mode", ["keepalive", "no_usage", "empty", "no_done"])
def test_stream_modes(mode: str) -> None:
    with kit.FakeNim(stream=mode) as nim:
        status, _, parsed = stream(nim, "Reply with the single word OK.")
    assert status == 200
    assert bool(parsed["comments"]) == (mode == "keepalive")
    assert all(c.startswith(": ") for c in parsed["comments"])
    assert parsed["done"] == (mode != "no_done")
    has_usage = any(e["choices"] == [] for e in parsed["events"])
    assert has_usage == (mode != "no_usage")
    assert (joined(parsed["events"], "content") == "") == (mode == "empty")


def test_stream_error_event_keeps_http_200() -> None:
    with kit.FakeNim(stream="error_event") as nim:
        status, _, parsed = stream(nim, "Reply with the single word OK.")
    assert status == 200 and parsed["done"]
    assert parsed["events"][-1]["error"]["type"] == "InternalServerError"


def test_stream_ignored_answers_plain_json() -> None:
    with kit.FakeNim(stream="ignored") as nim:
        status, headers, raw = call(nim.url, "/v1/chat/completions", ask(nim, "hi", stream=True))
    assert status == 200 and headers["content-type"] == "application/json"
    assert json.loads(raw)["object"] == "chat.completion"


def test_stream_cut_drops_the_connection_before_done() -> None:
    with kit.FakeNim(stream="cut") as nim:
        connection = http.client.HTTPConnection("127.0.0.1", nim.port, timeout=10)
        connection.request("POST", "/v1/chat/completions",
                           json.dumps(ask(nim, "hi", stream=True)),
                           {"Content-Type": "application/json"})
        answer = connection.getresponse()
        with pytest.raises(http.client.IncompleteRead) as cut:
            answer.read()
        connection.close()
    assert answer.status == 200 and b"chat.completion.chunk" in cut.value.partial
    assert b"[DONE]" not in cut.value.partial


def test_chunk_delay_spreads_the_stream() -> None:
    with kit.FakeNim(chunk_delay=0.05) as nim:
        started = time.monotonic()
        stream(nim, "Reply with the single word OK.")
    assert time.monotonic() - started >= 0.15


# --- tools ----------------------------------------------------------------------------------------


def tool_message(nim: kit.FakeNim, choice: Any = "auto", prompt: str = WEATHER,
                 **fields: Any) -> tuple[int, Any]:
    return chat(nim.url, ask(nim, prompt, tools=[TOOL], tool_choice=choice, **fields))


def test_tools_ok_auto_named_and_parallel() -> None:
    with kit.FakeNim(tools="ok") as nim:
        status, body = tool_message(nim)
        msg = body["choices"][0]["message"]
        call_ = msg["tool_calls"][0]
        assert status == 200 and msg["content"] is None
        assert body["choices"][0]["finish_reason"] == "tool_calls"
        assert call_["type"] == "function" and call_["id"].startswith("chatcmpl-tool-")
        assert call_["function"]["name"] == "get_weather"
        assert json.loads(call_["function"]["arguments"]) == {"city": "Paris", "unit": "celsius"}
        named = {"type": "function", "function": {"name": "get_weather"}}
        assert tool_message(nim, named)[1]["choices"][0]["finish_reason"] == "stop"
        both = "Weather in Paris and Tokyo? Use celsius."
        calls = tool_message(nim, prompt=both, parallel_tool_calls=True)[1]["choices"][0][
            "message"]["tool_calls"]
        cities = [json.loads(c["function"]["arguments"])["city"] for c in calls]
        assert cities == ["Paris", "Tokyo"]
        none = tool_message(nim, "none", prompt="What is 2 + 2? Answer with the number only.")
        assert none[1]["choices"][0]["message"]["content"] == "4"
        assert "tool_calls" not in none[1]["choices"][0]["message"]
    with kit.FakeNim(tools="ok", parallel_calls=False) as nim:
        calls = tool_message(nim, prompt=both, parallel_tool_calls=True)[1]["choices"][0][
            "message"]["tool_calls"]
        assert len(calls) == 1


def test_tools_no_parser_gives_the_exact_vllm_400_texts() -> None:
    with kit.FakeNim(tools="no_parser") as nim:
        auto = tool_message(nim)
        required = tool_message(nim, "required")
        named = tool_message(nim, {"type": "function", "function": {"name": "get_weather"}})
        none = tool_message(nim, "none", prompt="What is 2 + 2?")
    assert auto[0] == 400 and auto[1]["error"]["message"] == (
        '"auto" tool choice requires --enable-auto-tool-choice and --tool-call-parser to be set')
    assert required[1]["error"]["message"] == (
        'tool_choice="required" requires --tool-call-parser to be set')
    assert named[1]["error"]["message"] == (
        'tool_choice=function "get_weather" requires --tool-call-parser to be set')
    assert auto[1]["error"]["type"] == "BadRequestError" and none[0] == 200


def test_tools_no_parser_on_sglang_returns_the_call_as_text() -> None:
    with kit.FakeNim(tools="no_parser", backend="sglang") as nim:
        status, body = tool_message(nim)
    msg = body["choices"][0]["message"]
    assert status == 200 and "<tool_call>" in msg["content"] and msg["tool_calls"] is None


@pytest.mark.parametrize(
    ("mode", "check"),
    [
        ("in_content", lambda m: '"name": "get_weather"' in m["content"] and not m.get(
            "tool_calls")),
        ("prose", lambda m: m["content"] and not m.get("tool_calls")),
        ("in_reasoning", lambda m: "<tool_call>" in m["reasoning"] and not m.get("tool_calls")),
        ("double_encoded", lambda m: isinstance(json.loads(json.loads(
            m["tool_calls"][0]["function"]["arguments"])), dict)),
        ("wrong_args", lambda m: json.loads(m["tool_calls"][0]["function"]["arguments"]) == {
            "city": "London", "unit": "fahrenheit"}),
        ("arguments_object", lambda m: m["tool_calls"][0]["function"]["arguments"] == {
            "city": "Paris", "unit": "celsius"}),
        ("schema_mismatch", lambda m: json.loads(m["tool_calls"][0]["function"]["arguments"])
         == {"city": "Paris"}),
        ("wrong_name", lambda m: m["tool_calls"][0]["function"]["name"] != "get_weather"),
    ],
)
def test_tool_failure_modes(mode: str, check: Any) -> None:
    with kit.FakeNim(tools=mode) as nim:
        status, body = tool_message(nim)
    assert status == 200 and check(body["choices"][0]["message"]), body


def test_tools_invalid_json_and_forced_calls_get_a_valid_call() -> None:
    with kit.FakeNim(tools="invalid_json") as nim:
        arguments = tool_message(nim)[1]["choices"][0]["message"]["tool_calls"][0]["function"][
            "arguments"]
    with pytest.raises(ValueError):
        json.loads(arguments)
    with kit.FakeNim(tools="prose") as nim:
        named = {"type": "function", "function": {"name": "get_weather"}}
        msg = tool_message(nim, named)[1]["choices"][0]["message"]
    assert json.loads(msg["tool_calls"][0]["function"]["arguments"])["city"] == "Paris"


def test_tool_choice_none_modes_and_empty_tool_calls_forms() -> None:
    prompt = "What is 2 + 2? Answer with the number only."
    with kit.FakeNim(tools="none_empty") as nim:
        assert tool_message(nim, "none", prompt=prompt)[1]["choices"][0]["message"][
            "content"] == ""
    with kit.FakeNim(tools="none_markup") as nim:
        content = tool_message(nim, "none", prompt=prompt)[1]["choices"][0]["message"]["content"]
        assert content.startswith("<tool_call>") and content.endswith("4")
    with kit.FakeNim(tool_calls_empty="list") as nim:
        assert message(nim, "hi")[0]["tool_calls"] == []
    with kit.FakeNim(tool_calls_empty="null") as nim:
        assert message(nim, "hi")[0]["tool_calls"] is None


def test_streamed_tool_calls_assemble_by_index() -> None:
    with kit.FakeNim(tools="ok") as nim:
        body = ask(nim, "Weather in Paris and Tokyo? Use celsius.", tools=[TOOL],
                   tool_choice="auto", stream=True)
        parsed = sse(call(nim.url, "/v1/chat/completions", body)[2])
    calls: dict[int, dict[str, str]] = {}
    for event in parsed["events"]:
        for choice in event["choices"]:
            for delta in choice["delta"].get("tool_calls") or []:
                slot = calls.setdefault(delta["index"], {"name": "", "arguments": ""})
                function = delta.get("function") or {}
                slot["name"] += function.get("name") or ""
                slot["arguments"] += function.get("arguments") or ""
    assert [json.loads(c["arguments"])["city"] for c in calls.values()] == ["Paris", "Tokyo"]
    fragments = [d for e in parsed["events"] for c in e["choices"]
                 for d in c["delta"].get("tool_calls") or []]
    assert len(fragments) > 2 and parsed["done"]


# --- reasoning ------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("knobs", "fields", "system", "on"),
    [
        ({"reasoning": "kwargs"}, {"chat_template_kwargs": {"enable_thinking": False}}, None,
         False),
        ({"reasoning": "kwargs"}, {"chat_template_kwargs": {"enable_thinking": True}}, None, True),
        ({"reasoning": "kwargs", "reasoning_default": False}, {}, None, False),
        ({"reasoning": "kwargs", "reasoning_key": "thinking"},
         {"chat_template_kwargs": {"thinking": False}}, None, False),
        ({"reasoning": "system"}, {}, "/no_think", False),
        ({"reasoning": "system"}, {}, "/think", True),
        ({"reasoning": "system", "reasoning_default": False}, {}, "detailed thinking on", True),
        ({"reasoning": "system"}, {}, "detailed thinking off", False),
        ({"reasoning": "effort"}, {"reasoning_effort": "none"}, None, False),
        ({"reasoning": "effort"}, {"reasoning_effort": "low"}, None, True),
        ({"reasoning": "always"}, {"chat_template_kwargs": {"enable_thinking": False}},
         "/no_think", True),
        ({"reasoning": "none"}, {"chat_template_kwargs": {"enable_thinking": True}}, None, False),
    ],
)
def test_reasoning_switches(knobs: dict, fields: dict, system: str | None, on: bool) -> None:
    with kit.FakeNim(**knobs) as nim:
        body = ask(nim, "What is the capital of France? Answer with one word.", **fields)
        if system:
            body["messages"].insert(0, {"role": "system", "content": system})
        msg = chat(nim.url, body)[1]["choices"][0]["message"]
    assert bool(msg["reasoning"]) == on and msg["content"] == "Paris"


def test_reasoning_user_suffix_switch() -> None:
    with kit.FakeNim(reasoning="system") as nim:
        off = message(nim, "What is the capital of France? /no_think")[0]
        on = message(nim, "What is the capital of France? /think")[0]
    assert off["reasoning"] is None and on["reasoning"]


@pytest.mark.parametrize(
    ("where", "check"),
    [
        ("reasoning_content", lambda m: m["reasoning_content"] and m["content"] == "8"),
        ("inline", lambda m: m["content"].startswith("<think>")
         and m["content"].endswith("</think>8")),
        ("inline_close_only", lambda m: not m["content"].startswith("<think>")
         and m["content"].endswith("</think>8")),
    ],
)
def test_reasoning_field_placement(where: str, check: Any) -> None:
    with kit.FakeNim(reasoning="always", reasoning_field=where) as nim:
        msg = message(nim, "A shop sells pens for 3 dollars each. How much change?")[0]
        _, _, parsed = stream(nim, "A shop sells pens for 3 dollars each. How much change?")
    assert check(msg), msg
    inline = where.startswith("inline")
    text = joined(parsed["events"], "content" if inline else where)
    assert text == (msg["content"] if inline else msg[where])


def test_sglang_reasoning_goes_to_reasoning_content() -> None:
    with kit.FakeNim(reasoning="always", backend="sglang") as nim:
        msg, body = message(nim, "hi")
        _, _, parsed = stream(nim, "hi")
    assert msg["reasoning_content"] and "reasoning" not in msg
    assert body["usage"]["reasoning_tokens"] == 0
    assert joined(parsed["events"], "reasoning_content") == msg["reasoning_content"]


def test_small_budget_cuts_reasoning_and_leaves_no_answer() -> None:
    with kit.FakeNim(reasoning="always") as nim:
        msg, body = message(nim, "Reply with the single word OK.", max_tokens=5)
    assert body["choices"][0]["finish_reason"] == "length"
    assert msg["content"] is None and len(msg["reasoning"].split()) == 5
    with kit.FakeNim(reasoning="always", reasoning_field="inline") as nim:
        msg = message(nim, "Reply with the single word OK.", max_tokens=5)[0]
    assert msg["content"].startswith("<think>") and "</think>" not in msg["content"]


def test_off_bug_puts_the_answer_in_the_reasoning_field() -> None:
    off = {"chat_template_kwargs": {"enable_thinking": False}}
    prompt = "What is the capital of France? Answer with one word."
    with kit.FakeNim(reasoning="kwargs", off_bug="answer_in_reasoning") as nim:
        msg = message(nim, prompt, **off)[0]
        _, _, parsed = stream(nim, prompt, **off)
        on = message(nim, prompt)[0]
    assert msg["content"] is None and msg["reasoning"] == "Paris"
    assert joined(parsed["events"], "reasoning") == "Paris"
    assert on["content"] == "Paris" and on["reasoning"] != "Paris"
    with kit.FakeNim(reasoning="kwargs", off_bug="answer_in_reasoning_stream") as nim:
        msg = message(nim, prompt, **off)[0]
        _, _, parsed = stream(nim, prompt, **off)
    assert msg["content"] == "Paris" and msg["reasoning"] is None
    assert joined(parsed["events"], "content") == ""
    assert joined(parsed["events"], "reasoning") == "Paris"


# --- image, JSON ----------------------------------------------------------------------------------


def image_request(nim: kit.FakeNim, with_image: bool) -> dict[str, Any]:
    parts: list[dict[str, Any]] = [
        {"type": "text", "text": "What is the main color of this image? Answer with one word."}]
    if with_image:
        parts.append({"type": "image_url", "image_url": {"url": PNG}})
    return ask(nim, parts, max_tokens=32 if with_image else 1)


@pytest.mark.parametrize(("vlm", "extra"), [(True, 256), ("image_ignored", 0)])
def test_vlm_prompt_tokens_grow_per_image(vlm: Any, extra: int) -> None:
    with kit.FakeNim(vlm=vlm) as nim:
        status, with_image = chat(nim.url, image_request(nim, True))
        twin = chat(nim.url, image_request(nim, False))[1]
    assert status == 200
    gap = with_image["usage"]["prompt_tokens"] - twin["usage"]["prompt_tokens"]
    assert gap == extra
    answer = with_image["choices"][0]["message"]["content"]
    assert (answer == "Red") == (vlm is True)
    assert twin["choices"][0]["finish_reason"] == "length"


def test_text_only_model_rejects_an_image_and_bad_images_are_400() -> None:
    with kit.FakeNim(root="/models/llama") as nim:
        status, body = chat(nim.url, image_request(nim, True))
    assert status == 400 and body["error"]["message"] == "/models/llama is not a multimodal model"
    with kit.FakeNim(vlm=True) as nim:
        bad = ask(nim, [{"type": "image_url", "image_url": {"url": "data:image/png;base64,!!"}}])
        assert chat(nim.url, bad)[0] == 400


def json_request(nim: kit.FakeNim, schema: dict[str, Any] = SCHEMA) -> dict[str, Any]:
    fmt = {"type": "json_schema", "json_schema": {"name": "smoke", "schema": schema}}
    return ask(nim, 'Return a JSON object with name set to "NIM" and version set to 2.',
               response_format=fmt, max_tokens=256, temperature=0)


def test_json_ok_follows_the_schema() -> None:
    other = {"type": "object", "properties": {"tags": {"type": "array", "items": {
        "type": "string"}}, "ok": {"type": "boolean"}, "kind": {"enum": ["a", "b"]}}}
    with kit.FakeNim(json="ok", reasoning="always") as nim:
        body = chat(nim.url, json_request(nim))[1]
        second = chat(nim.url, json_request(nim, other))[1]
    msg = body["choices"][0]["message"]
    assert json.loads(msg["content"]) == {"name": "NIM", "version": 2} and msg["reasoning"]
    assert json.loads(second["choices"][0]["message"]["content"]) == {
        "tags": ["NIM"], "ok": True, "kind": "a"}


@pytest.mark.parametrize("mode", ["ignored", "invalid", "truncated", "in_reasoning"])
def test_json_failure_modes(mode: str) -> None:
    with kit.FakeNim(json=mode) as nim:
        body = chat(nim.url, json_request(nim))[1]
    choice = body["choices"][0]
    content = choice["message"]["content"]
    assert choice["finish_reason"] == ("length" if mode == "truncated" else "stop")
    if mode == "in_reasoning":
        assert content is None and json.loads(choice["message"]["reasoning"])["name"] == "NIM"
    else:
        with pytest.raises(ValueError):
            json.loads(content)


# --- key, redirect, concurrency, delay ------------------------------------------------------------


def test_require_key_answers_401_without_the_right_bearer() -> None:
    key = kit.fake_token()
    with kit.FakeNim(require_key=key) as nim:
        missing = call(nim.url, "/v1/models")
        wrong = call(nim.url, "/v1/models", headers={"Authorization": f"Bearer {key}x"})
        right = call(nim.url, "/v1/models", headers={"Authorization": f"Bearer {key}"})
        posted = chat(nim.url, ask(nim, "hi"), headers={"Authorization": f"Bearer {key}"})
    assert missing[0] == 401 and missing[1]["content-type"] == "text/plain"
    assert b"authorization" in missing[2]
    assert wrong[0] == 401 and right[0] == 200 and posted[0] == 200


def test_redirect_to_answers_302_and_the_other_server_gets_nothing() -> None:
    with kit.FakeNim() as other, kit.FakeNim(redirect_to=other.url) as nim:
        status, headers, _ = call(nim.url, "/v1/models")
        post = call(nim.url, "/v1/chat/completions", ask(nim, "hi"))
    assert status == 302 and headers["location"] == other.url + "/v1/models"
    assert post[0] == 302 and other.requests == []


def test_fail_concurrent_rejects_with_503_and_records_the_peak() -> None:
    results: list[int] = []
    with kit.FakeNim(fail_concurrent=2, delay=0.4) as nim:
        threads = [threading.Thread(target=lambda: results.append(
            chat(nim.url, ask(nim, "Reply with the single word OK."))[0])) for _ in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        peak = nim.peak
    # two held and at least one rejected overlapped; a rejected one may leave before the last
    assert sorted(results) == [200, 200, 503, 503] and peak >= 3


def test_delay_holds_each_post_and_get_delay_each_get() -> None:
    with kit.FakeNim(delay=0.2, get_delay=0.2) as nim:
        started = time.monotonic()
        chat(nim.url, ask(nim, "hi"))
        call(nim.url, "/v1/health/ready")
        elapsed = time.monotonic() - started
    assert elapsed >= 0.4


# --- fake kubectl and docker ----------------------------------------------------------------------

SCOPE = ["--context", "lab", "-n", "nim"]
IMAGE = "nvcr.io/nim/meta/llama-3.1-8b-instruct:2.1.5"


def run_tool(tools: kit.FakeTools, tool: str, *argv: str) -> subprocess.CompletedProcess[str]:
    """Run a fake by name through PATH, as the access helpers find it."""
    return subprocess.run([tool, *argv], env=tools.env(), stdin=subprocess.DEVNULL,
                          capture_output=True, text=True, timeout=30, check=False)


def test_fake_tools_answer_allowed_commands_from_fixtures(tmp_path: Path) -> None:
    service = kit.k8s_nimservice("llama", "nim", "nvcr.io/nim/meta/llama-3.1-8b-instruct",
                                 "2.1.5", args=["--max-num-seqs", "8"])
    tools = kit.make_fake_tools(tmp_path, kubectl_fixtures={
        "get nimservice llama": service,
        "kubectl get nimservice llama -o json --context other -n nim": {"exact": True},
        "auth can-i create pods": kit.can_i(False),
        "*": kit.ToolReply(stderr="Error from server (NotFound): x not found\n", exit=1),
    })
    got = run_tool(tools, "kubectl", "get", "nimservice", "llama", "-o", "json", *SCOPE)
    assert got.returncode == 0 and json.loads(got.stdout) == service
    exact = run_tool(tools, "kubectl", "get", "nimservice", "llama", "-o", "json",
                     "--context", "other", "-n", "nim")
    assert json.loads(exact.stdout) == {"exact": True}
    can = run_tool(tools, "kubectl", "auth", "can-i", "create", "pods",
                   "--subresource=portforward", *SCOPE)
    assert can.returncode == 1 and can.stdout == "no\n"
    other = run_tool(tools, "kubectl", "get", "pods", "-l", "app=llama", "-o", "json", *SCOPE)
    assert other.returncode == 1 and "NotFound" in other.stderr
    nothing = run_tool(tools, "docker", "port", "llama")
    assert nothing.returncode == 1 and "no fixture" in nothing.stderr
    assert [c["fixture"] for c in tools.calls()] == [
        "get nimservice llama", "kubectl get nimservice llama -o json --context other -n nim",
        "auth can-i create pods", "*", None]
    tools.assert_read_only()
    with pytest.raises(AssertionError, match="--context lab"):
        tools.assert_scoped("lab", "nim")  # the exact-key call used --context other


def test_fake_tools_in_turn_answers_each_call_in_order(tmp_path: Path) -> None:
    """in_turn: the N-th matching call gets the N-th reply, later calls the last one; new
    fixtures start the turns again."""
    loading = kit.k8s_pod("llama-0", "nim", labels={"app": "llama"}, image=IMAGE, ready=False)
    ready = kit.k8s_pod("llama-0", "nim", labels={"app": "llama"}, image=IMAGE)
    fixtures = {"get pod llama-0": kit.in_turn(
        kit.ToolReply(stderr='Error from server (NotFound): pods "llama-0" not found\n', exit=1),
        loading, ready)}
    tools = kit.make_fake_tools(tmp_path, kubectl_fixtures=fixtures)
    argv = ("get", "pod", "llama-0", "-o", "json", *SCOPE)
    first, second, third, fourth = (run_tool(tools, "kubectl", *argv) for _ in range(4))
    assert first.returncode == 1 and "NotFound" in first.stderr
    assert json.loads(second.stdout) == loading
    assert json.loads(third.stdout) == ready and json.loads(fourth.stdout) == ready
    tools = kit.make_fake_tools(tmp_path, kubectl_fixtures=fixtures)
    assert run_tool(tools, "kubectl", *argv).returncode == 1, "new fixtures: the first turn again"


def test_assert_scoped_takes_every_spelling_of_the_scope(tmp_path: Path) -> None:
    tools = kit.make_fake_tools(tmp_path, kubectl_fixtures={"*": "x"})
    for scope in (SCOPE, ["--context=lab", "--namespace=nim"], ["--context", "lab", "-nnim"],
                  ["--context", "lab", "-n=nim", "--namespace", "nim"]):
        run_tool(tools, "kubectl", "get", "pods", "-l", "app=llama", *scope)
    run_tool(tools, "kubectl", "config", "get-contexts", "-o", "name")  # needs no scope
    tools.assert_scoped("lab", "nim")


@pytest.mark.parametrize("extra", [
    ["--context", "prod"], ["--context=prod"], ["-n", "default"], ["--namespace", "default"],
    ["--namespace=default"], ["-ndefault"], ["-n=default"],
])
def test_assert_scoped_refuses_a_second_context_or_namespace(
    tmp_path: Path, extra: list[str]
) -> None:
    """kubectl uses the last value of a repeated flag: a call that names the expected scope and
    then another one runs in the other one."""
    tools = kit.make_fake_tools(tmp_path, kubectl_fixtures={"*": "x"})
    run_tool(tools, "kubectl", "get", "pods", *SCOPE, *extra)
    with pytest.raises(AssertionError, match="and another"):
        tools.assert_scoped("lab", "nim")


@pytest.mark.parametrize(
    ("tool", "argv"),
    [
        ("kubectl", ["delete", "pod", "x", *SCOPE]),
        ("kubectl", ["apply", "-f", "x.yaml"]),
        ("kubectl", ["exec", "pod/x", "--", "sh"]),
        ("kubectl", ["run", "curl", "--image=curlimages/curl"]),
        ("kubectl", ["scale", "deploy/x", "--replicas=0", *SCOPE]),
        ("kubectl", ["patch", "nimservice", "x", "-p", "{}"]),
        ("kubectl", ["--context", "lab", "logs", "x"]),
        ("kubectl", ["describe", "pod", "x"]),
        ("kubectl", ["config", "use-context", "prod"]),
        ("kubectl", ["config", "get-contexts"]),
        ("kubectl", ["get", "secret", "ngc-api-secret", *SCOPE]),
        ("kubectl", ["get", "secrets,pods", *SCOPE]),
        ("kubectl", ["get", "--raw", "/api/v1/namespaces/nim/secrets", *SCOPE]),
        ("kubectl", ["auth", "reconcile", "-f", "x.yaml"]),
        ("kubectl", ["proxy"]),
        ("kubectl", []),
        ("docker", ["run", "-d", IMAGE]),
        ("docker", ["stop", "llama"]),
        ("docker", ["rm", "-f", "llama"]),
        ("docker", ["exec", "llama", "sh"]),
        ("docker", ["pull", IMAGE]),
        ("docker", ["inspect", "llama"]),
        ("docker", ["ps"]),
        ("docker", ["update", "--restart=no", "llama"]),
        ("docker", ["context", "use", "remote"]),
        ("docker", ["container", "rm", "llama"]),
    ],
)
def test_fake_tools_refuse_anything_outside_the_allow_list(
    tmp_path: Path, tool: str, argv: list[str]
) -> None:
    tools = kit.make_fake_tools(tmp_path, {"*": "anything"}, {"*": "anything"})
    done = run_tool(tools, tool, *argv)
    assert done.returncode == 9 and "refused" in done.stderr and done.stdout == ""
    assert tools.refused()[0]["argv"] == argv
    with pytest.raises(AssertionError, match="read-only"):
        tools.assert_read_only()


def test_kubectl_output_forms(tmp_path: Path) -> None:
    pod = kit.k8s_pod("llama-0", "nim", labels={"app": "llama"}, image=IMAGE,
                      env={"NIM_PASSTHROUGH_ARGS": "--max-num-seqs 8"})
    tools = kit.make_fake_tools(tmp_path, kubectl_fixtures={
        "get pods": kit.k8s_list(pod),
        "config get-contexts": "lab\nprod\n",
    })
    image = run_tool(tools, "kubectl", "get", "pods", *SCOPE,
                     "-o", "jsonpath={.items[0].spec.containers[0].image}")
    assert image.stdout == IMAGE
    names = run_tool(tools, "kubectl", "get", "pods", "-oname", *SCOPE)
    assert names.stdout == "pod/llama-0\n"
    ready = run_tool(tools, "kubectl", "get", "pods", *SCOPE,
                     "--output=jsonpath={.items[*].status.containerStatuses[0].ready}")
    assert ready.stdout == "true"
    assert run_tool(tools, "kubectl", "get", "pods", "-o", "yaml", *SCOPE).returncode == 1
    contexts = run_tool(tools, "kubectl", "config", "get-contexts", "-o", "name")
    assert contexts.returncode == 0 and contexts.stdout.split() == ["lab", "prod"]
    tools.assert_scoped("lab", "nim")
    env = pod["spec"]["containers"][0]["env"]
    assert env[0]["valueFrom"]["secretKeyRef"]["name"] == "ngc-api-secret"


def test_docker_inspect_formats_and_port_text(tmp_path: Path) -> None:
    container = kit.docker_container("llama", IMAGE, host_port=8005,
                                     env={"NIM_SERVER_PORT": "8000", "NGC_API_KEY": "x"})
    tools = kit.make_fake_tools(tmp_path, docker_fixtures={
        "container inspect llama": container,
        "image inspect": kit.docker_image(IMAGE, env={"NIM_PASSTHROUGH_ARGS": "--x"}),
        "context inspect": kit.docker_context("ssh://gpu-host"),
        "port llama": kit.docker_port_text(host_port=8005),
    })
    ports = run_tool(tools, "docker", "container", "inspect", "llama",
                     "--format", "{{json .NetworkSettings.Ports}}")
    assert json.loads(ports.stdout) == {"8000/tcp": [{"HostIp": "0.0.0.0", "HostPort": "8005"},
                                                     {"HostIp": "::", "HostPort": "8005"}]}
    mode = run_tool(tools, "docker", "container", "inspect", "--format={{.HostConfig.NetworkMode}}",
                    "llama")
    assert mode.stdout == "bridge\n"
    host = run_tool(tools, "docker", "context", "inspect", "-f", "{{.Endpoints.docker.Host}}")
    assert host.stdout == "ssh://gpu-host\n"
    whole = run_tool(tools, "docker", "image", "inspect", IMAGE)
    assert json.loads(whole.stdout)[0]["Config"]["Env"] == ["NIM_PASSTHROUGH_ARGS=--x"]
    assert run_tool(tools, "docker", "port", "llama").stdout.splitlines() == [
        "8000/tcp -> 0.0.0.0:8005", "8000/tcp -> [::]:8005"]
    bad = run_tool(tools, "docker", "container", "inspect", "llama", "--format",
                   "{{range .Mounts}}{{end}}")
    assert bad.returncode == 1 and "not supported" in bad.stderr
    tools.assert_read_only()


# --- the fake port-forward ------------------------------------------------------------------------


def start_forward(tools: kit.FakeTools, *argv: str) -> tuple[subprocess.Popen[bytes], int, Path]:
    """Start the fake `kubectl port-forward ...` as the access helper does; wait for its port."""
    out = tools.data / f"forward-{time.monotonic_ns()}.out"
    err = out.with_suffix(".err")
    with open(out, "wb") as stdout, open(err, "wb") as stderr:
        proc = subprocess.Popen([str(tools.bin / "kubectl"), "port-forward", *argv],
                                stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr)
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        for line in out.read_text().splitlines():
            if line.startswith("Forwarding from 127.0.0.1:"):
                return proc, int(line.split(":")[1].split()[0]), err
        if proc.poll() is not None:
            break
        time.sleep(0.02)
    proc.kill()
    raise AssertionError(f"no Forwarding line: {out.read_text()!r} {err.read_text()!r}")


def test_port_forward_relays_to_the_fake_nim_until_sigint(tmp_path: Path) -> None:
    with kit.FakeNim(reasoning="always") as nim:
        fixture = {"port-forward pod/llama-0": kit.port_forward_to(nim)}
        tools = kit.make_fake_tools(tmp_path, fixture)
        proc, port, _ = start_forward(tools, "pod/llama-0", ":api", *SCOPE)
        try:
            local = f"http://127.0.0.1:{port}"
            assert json.loads(call(local, "/v1/models")[2])["data"][0]["id"] == nim.model
            body = ask(nim, "Reply with the single word OK.", stream=True)
            parsed = sse(call(local, "/v1/chat/completions", body)[2])
            assert parsed["done"] and joined(parsed["events"], "content") == "OK"
        finally:
            proc.send_signal(signal.SIGINT)
            code = proc.wait(timeout=10)
        assert code == 0 and [s.path for s in nim.requests] == ["/v1/models",
                                                                 "/v1/chat/completions"]
    out = sorted(tools.data.glob("forward-*.out"))[0].read_text()
    assert f"Forwarding from [::1]:{port} -> 8000" in out
    assert out.count(f"Handling connection for {port}") == 2
    tools.assert_read_only()


def test_port_forward_lost_connection_then_a_restart_survives(tmp_path: Path) -> None:
    with kit.FakeNim() as nim:
        fixture = {"port-forward svc/llama": kit.port_forward_to(nim.url, lost_after=1)}
        tools = kit.make_fake_tools(tmp_path, fixture)
        proc, port, err = start_forward(tools, "svc/llama", ":api", *SCOPE)
        local = f"http://127.0.0.1:{port}"
        assert call(local, "/v1/models")[0] == 200
        with pytest.raises((urllib.error.URLError, ConnectionError, http.client.HTTPException)):
            call(local, "/v1/models")
        assert proc.wait(timeout=10) == 1
        assert "error: lost connection to pod" in err.read_text()
        again, port, _ = start_forward(tools, "svc/llama", ":api", *SCOPE)
        try:
            for _ in range(3):
                assert call(f"http://127.0.0.1:{port}", "/v1/health/ready")[0] == 200
            assert again.poll() is None
        finally:
            again.send_signal(signal.SIGINT)
            assert again.wait(timeout=10) == 0


def test_port_forward_failures(tmp_path: Path) -> None:
    busy = socket.socket()
    busy.bind(("127.0.0.1", 0))
    busy.listen(1)
    taken = busy.getsockname()[1]
    denied = kit.ToolReply(stderr='Error from server (Forbidden): pods "x" is forbidden: cannot '
                                  'create resource "pods/portforward"\n', exit=1)
    tools = kit.make_fake_tools(tmp_path, {
        "port-forward pod/x": denied,
        "port-forward pod/llama-0": kit.port_forward_to(1),
    })
    try:
        done = run_tool(tools, "kubectl", "port-forward", "pod/llama-0", f"{taken}:api", *SCOPE)
        assert done.returncode == 1 and "unable to listen on any of the requested ports" in (
            done.stderr)
    finally:
        busy.close()
    forbidden = run_tool(tools, "kubectl", "port-forward", "pod/x", ":api", *SCOPE)
    assert forbidden.returncode == 1 and "pods/portforward" in forbidden.stderr
    proc, port, err = start_forward(tools, "pod/llama-0", ":api", *SCOPE)  # nothing on port 1
    with pytest.raises((urllib.error.URLError, ConnectionError, http.client.HTTPException)):
        call(f"http://127.0.0.1:{port}", "/v1/models")
    assert proc.wait(timeout=10) == 1 and "lost connection to pod" in err.read_text()


def test_port_forward_can_serve_its_own_fake(tmp_path: Path) -> None:
    fixture = kit.PortForwardReply(serve={"model": "acme/custom-llm-7b"}, remote_port=8080)
    tools = kit.make_fake_tools(tmp_path, {"port-forward": fixture})
    proc, port, _ = start_forward(tools, "svc/custom", ":http-openai", *SCOPE)
    try:
        models = json.loads(call(f"http://127.0.0.1:{port}", "/v1/models")[2])
    finally:
        proc.send_signal(signal.SIGINT)
        proc.wait(timeout=10)
    assert models["data"][0]["id"] == "acme/custom-llm-7b"
    assert "-> 8080" in sorted(tools.data.glob("forward-*.out"))[0].read_text()


# --- run_cli, assert_contract, keys ---------------------------------------------------------------

PROBE = '''import json, os, sys
facts = {"isolated": sys.flags.isolated, "no_bytecode": sys.flags.dont_write_bytecode,
         "x": os.environ.get("NSC_X"), "key": os.environ.get("SOME_API_KEY"),
         "stdin": sys.stdin.read(), "argv": sys.argv[1:]}
code = int(os.environ.get("NSC_CODE", "0"))
print(json.dumps({"ok": code == 0, "state": "probe", "next": "report", "facts": facts,
                  "errors": [] if code == 0 else [{"code": "probe_failed", "message": "m",
                                                   "fix": "f"}]}))
sys.exit(code)
'''


def test_run_cli_runs_isolated_with_no_stdin_and_no_real_keys(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = tmp_path / "probe.py"
    script.write_text(PROBE)
    monkeypatch.setenv("SOME_API_KEY", kit.fake_token())
    code, obj, _ = kit.run_cli(script, "--flag", "v", env={"NSC_X": "1"})
    assert code == 0 and obj["facts"] == {"isolated": 1, "no_bytecode": True, "x": "1",
                                          "key": None, "stdin": "", "argv": ["--flag", "v"]}
    code, obj, _ = kit.run_cli(script, env={"NSC_CODE": "4"})
    assert code == 4 and obj["errors"][0]["code"] == "probe_failed"
    assert not (tmp_path / "__pycache__").exists()


def test_run_cli_refuses_bad_output(tmp_path: Path) -> None:
    two = tmp_path / "two.py"
    two.write_text('print("{}")\nprint("{}")\n')
    with pytest.raises(AssertionError, match="one JSON object"):
        kit.run_cli(two)
    assert kit.run_cli(two, raw=True)[1] == "{}\n{}\n"
    liar = tmp_path / "liar.py"
    liar.write_text('import json, sys\nprint(json.dumps({"ok": True, "state": "s", "next": "n", '
                    '"facts": {}, "errors": []}))\nsys.exit(4)\n')
    with pytest.raises(AssertionError, match="exit 4"):
        kit.run_cli(liar)
    assert kit.run_cli(liar, strict_exit=False)[0] == 4


@pytest.mark.parametrize(
    "change",
    [
        {"extra": 1},
        {"ok": "yes"},
        {"state": ""},
        {"facts": []},
        {"errors": [{"code": "Bad-Code", "message": "m", "fix": "f"}]},
        {"errors": [{"code": "bad", "message": "m", "fix": ""}]},
        {"errors": [{"code": "bad", "message": "m"}]},
    ],
)
def test_assert_contract_rejects_broken_objects(change: dict[str, Any]) -> None:
    good = {"ok": False, "state": "input", "next": "ask_user", "facts": {},
            "errors": [{"code": "bad_input", "message": "m", "fix": "f"}]}
    kit.assert_contract(good)
    with pytest.raises(AssertionError):
        kit.assert_contract({**good, **change})


def test_fake_keys_are_built_at_run_time_and_found_in_files(tmp_path: Path) -> None:
    source = Path(kit.__file__).read_text()
    assert kit.fake_key().startswith("nvapi-") and kit.fake_key() not in source
    assert kit.fake_token() not in source and "-" not in kit.fake_token()
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "evidence.json").write_text(f'{{"auth": "{kit.fake_token()}"}}')
    (tmp_path / "report.md").write_text("clean")
    assert kit.files_containing(tmp_path, kit.fake_token()) == [tmp_path / "a" / "evidence.json"]


def test_these_tests_hold_no_credential_literal() -> None:
    """Keys, tokens and URL passwords here are built at run time (fake_key, fake_token), so no
    secret-looking literal sits in the repository: no Authorization value or require_key is a
    string literal, and no URL holds a password."""
    password_url = re.compile(r"://[^/@\s:]+:[^/@\s]+@")
    found = []
    for path in (Path(kit.__file__), Path(__file__), kit.TESTS / "test_nsc_checks.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.keyword) and node.arg == "require_key":
                values = [node.value]
            elif isinstance(node, ast.Dict):
                values = [value for key, value in zip(node.keys, node.values, strict=True)
                          if isinstance(key, ast.Constant)
                          and str(key.value).lower() in ("authorization", "require_key")]
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                values = [node] if password_url.search(node.value) else []
            else:
                values = []
            found += [(path.name, value.lineno) for value in values
                      if isinstance(value, ast.Constant) and isinstance(value.value, str)]
    assert found == []


# --- serve mode -----------------------------------------------------------------------------------

START_TIMEOUT = 30.0  # seconds a server process gets to print its ready line


def start_server(argv: list[str], env: dict[str, str] | None = None,
                 timeout: float = START_TIMEOUT) -> tuple[subprocess.Popen[str], str]:
    """Start a server process and return it with the "url" of its ready line, the JSON object
    it prints once it listens. The wait is bounded: on no line in time, an exit before it or a
    line that is not that JSON, the process is killed and reaped, then the test fails."""
    proc = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True, env=kit.child_env(env))
    out, err = proc.stdout, proc.stderr
    assert out is not None and err is not None
    lines: queue.Queue[str] = queue.Queue()
    reader = threading.Thread(target=lambda: lines.put(out.readline()), daemon=True)
    reader.start()
    url: Any = None
    try:
        line = lines.get(timeout=timeout)
    except queue.Empty:
        problem = f"no ready line within {timeout:g} s"
    else:
        with contextlib.suppress(ValueError, TypeError, KeyError):
            url = json.loads(line)["url"]
        problem = f"not a ready line: {line!r}" if line else "it exited before its ready line"
    finally:
        if not isinstance(url, str):  # an interrupt too: never leave the child running
            proc.kill()
            proc.wait(timeout=10)
            reader.join(timeout=10)
    if isinstance(url, str):
        return proc, url
    with out, err:
        raise AssertionError(f"the server did not start: {problem}\n{err.read()}")


def serve(*args: str, env: dict[str, str] | None = None) -> tuple[subprocess.Popen[str], str]:
    return start_server(
        [sys.executable, "-I", "-B", kit.__file__, "serve", "--port", "0", *args], env)


def test_start_server_kills_and_reaps_a_child_that_does_not_get_ready(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The wait for the ready line is bounded. A child that prints nothing in time, prints
    another line or exits first is killed and reaped before the test fails, so no server is
    left running; its stderr is in the failure."""
    started: list[subprocess.Popen[str]] = []
    real_popen = subprocess.Popen

    def popen(*args: Any, **kwargs: Any) -> subprocess.Popen[str]:
        started.append(real_popen(*args, **kwargs))
        return started[-1]

    monkeypatch.setattr(subprocess, "Popen", popen)
    sleep = "import time; time.sleep(60)"
    for code, problem in ((sleep, "no ready line within 1 s"),
                          (f"print('starting', flush=True); {sleep}", "not a ready line: 'start"),
                          ("import sys; sys.exit('bad knob')", "exited before its ready line")):
        began = time.monotonic()
        with pytest.raises(AssertionError, match=problem) as info:
            start_server([sys.executable, "-I", "-c", code], timeout=1)
        assert time.monotonic() - began < 30, code
        assert started[-1].returncode is not None, code  # killed and reaped
    assert "bad knob" in str(info.value)


def test_serve_mode_answers_with_its_knobs_until_terminated() -> None:
    proc, url = serve("--model", "openai/gpt-oss-20b", "--reasoning", "always", "--tools",
                      "no_parser", "--vlm", "--reasoning-field", "reasoning_content")
    try:
        assert json.loads(call(url, "/v1/models")[2])["data"][0]["id"] == "openai/gpt-oss-20b"
        body = {"model": "openai/gpt-oss-20b",
                "messages": [{"role": "user", "content": WEATHER}], "tools": [TOOL]}
        assert chat(url, body)[0] == 400
        del body["tools"]
        msg = chat(url, body)[1]["choices"][0]["message"]
        assert msg["reasoning_content"] and msg["content"] == "OK"
    finally:
        proc.terminate()
        assert proc.wait(timeout=10) == 0


def test_serve_mode_key_from_a_variable_and_bad_knobs() -> None:
    name, key = "NSC_FAKE_GATE", kit.fake_token()
    proc, url = serve("--model", "m/x", "--require-key-env", name, env={name: key})
    try:
        assert call(url, "/v1/models")[0] == 401
        assert call(url, "/v1/models", headers={"Authorization": f"Bearer {key}"})[0] == 200
    finally:
        proc.send_signal(signal.SIGINT)
        assert proc.wait(timeout=10) == 0
    bad = subprocess.run([sys.executable, "-I", "-B", kit.__file__, "serve", "--port", "0",
                          "--model", "m/x", "--tools", "nope"],
                         stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=30,
                         check=False)
    assert bad.returncode == 2 and "--tools" in bad.stderr


# --- conftest: a temporary folder that is not private ---------------------------------------------


@pytest.mark.skipif(not hasattr(os, "getuid"), reason="POSIX owners only")
def test_a_basetemp_that_is_not_private_is_named_once(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import conftest

    assert conftest.private_temp_problem(tmp_path) is None
    shared = tmp_path / "shared"
    (shared / "bt").mkdir(parents=True)
    shared.chmod(0o777)
    problem = conftest.private_temp_problem(shared / "bt")
    assert problem and str(shared / "bt") in problem and "can be written by other users" in problem
    assert "out_not_private" in problem and "--basetemp" in problem
    shared.chmod(0o700)
    owner = tmp_path.stat().st_uid  # a sandbox shows /tmp as owned by "nobody"
    monkeypatch.setattr(os, "getuid", lambda: owner + 4242)
    assert "belongs to another user" in (conftest.private_temp_problem(tmp_path) or "")


@pytest.mark.skipif(not hasattr(os, "getuid"), reason="POSIX owners only")
def test_a_run_whose_basetemp_is_not_private_stops_with_one_message(tmp_path: Path) -> None:
    """Without the session check, every test that writes a run there fails on its own with
    out_not_private, which reads like many bugs; with it the run stops at once, exit 1."""
    shared = tmp_path / "shared"
    shared.mkdir()
    shared.chmod(0o777)
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-o", "addopts=",
         f"--basetemp={shared / 'bt'}",
         "test_nsc_security.py::test_out_files_are_private_and_a_planted_symlink_is_not_followed"],
        cwd=Path(__file__).resolve().parent, stdin=subprocess.DEVNULL, capture_output=True,
        text=True, timeout=120, check=False,
        env=kit.child_env({"PYTHONDONTWRITEBYTECODE": "1", "PYTEST_ADDOPTS": ""}),
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode == 1, output
    assert output.count("fails the private-folder check of the scripts") == 1, output
    assert "Rerun with --basetemp" in output and " passed" not in output
