#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The checks of nim-smoke-check and the pure parsers they share: `import nsc_checks as nc`.

  check_health        GET /v1/health/live and /v1/health/ready (GET /health when ready is 404)
  check_models        GET /v1/models (hosted: /v1/models/ID); chooses the model to test
  check_chat, check_stream, check_tools, check_tools_stream, check_tools_parallel,
  check_tools_none, check_reasoning_on, check_reasoning_off (stream=True: its streaming twin),
  check_json_schema, check_image (with its text-only twin), check_concurrency
                      POST /v1/chat/completions, with the request settings of the plan entry
                      (nsc_features.decide: switch, sampling, max_tokens, mode)

Each check returns a Result: status pass, fail or not_applicable; a reason code
(references/checks.md); a short message, never a full prompt or answer; the evidence files; the
sources of the expectation; a hint (the documented server switch, a change for nim-deploy);
warnings; and the known issues of the model. Evidence saves one private file per request,
OUT/evidence/<check>-<n>.json: the request (Authorization masked, image data replaced by its
SHA-256 and size), the status, content type, latency (and time to first byte for a stream), the
response (text cut to 2,000 characters with the full length recorded; a raw event stream cut to
20,000 bytes), finish_reason, usage and X-Request-Id.

The pure parsers (parse_sse, split_reasoning, join_stream, validate_schema, validate_tool_call,
control_tokens, png_solid) send nothing and read no file.

Read-only: GET on the health and model routes and POST on /v1/chat/completions with only the
allow-listed request fields (REQUEST_FIELDS); no other route. No redirect is followed. A check
retries once only where references/checks.md says so, and never after HTTP 429.
Stdlib only. Run directly, it prints a notice: it has no command of its own.
Exit codes: 0 ok, 2 bad input.
"""

from __future__ import annotations

import base64
import contextlib
import copy
import dataclasses
import hashlib
import json
import re
import struct
import sys
import threading
import time
import urllib.parse
import zlib
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nsc_common as nsc
import nsc_features as nf

PASS, FAIL, NOT_APPLICABLE = "pass", "fail", "not_applicable"
STATUSES = (PASS, FAIL, NOT_APPLICABLE)
OK = "ok"  # the reason of every pass

CHAT_PATH = "/v1/chat/completions"
TEXT_LIMIT = 2000  # characters of one text field kept in an evidence file
RAW_LIMIT = 20_000  # bytes of a raw event stream kept in an evidence file
MESSAGE_LIMIT = 300  # characters of a result message
SHORT = 60  # characters of a model value quoted in a message or details
GET_TIMEOUT = 30.0
TIMEOUT_CAP = 600.0
DEFAULT_CONCURRENCY = 4
MAX_CONCURRENCY = 8
HOSTED_CONCURRENCY = 4  # the hosted API's rate limits: the concurrency check stays at 4 there
# the budget of the reasoning retry: twice the first, at most this, never less than the first
RETRY_TOKEN_CAP = 16384
MIN_REASONING_CHARS = 8
HEALTH_POLL = 5.0  # seconds between readiness probes with --wait-ready

# --- what the checks send ------------------------------------------------------------------------

# The only fields a request carries. vLLM accepts unknown fields silently, so a wrong field gives a
# false result rather than an error. parallel_tool_calls is sent only as true, only by
# tools_parallel; chat_template_kwargs and reasoning_effort only from a documented switch.
REQUEST_FIELDS = frozenset({
    "model", "messages", "max_tokens", "temperature", "top_p", "top_k", "stream",
    "stream_options", "tools", "tool_choice", "parallel_tool_calls", "response_format",
    "chat_template_kwargs", "reasoning_effort",
})
# Never sent (references/checks.md says why): removed, ignored, or a 400 on some NIMs.
NEVER_SEND = frozenset({
    "nvext", "guided_json", "guided_regex", "guided_choice", "guided_grammar",
    "structured_outputs", "chat_template", "stop", "min_p", "logit_bias", "mm_processor_kwargs",
    "media_io_kwargs",
})

CHAT_PROMPT = "Reply with the single word OK."
TOOL_PROMPT = "What is the weather in Paris right now? Use celsius. Call the get_weather tool."
PARALLEL_PROMPT = (
    "Get the current weather for Paris and for Tokyo, both in celsius. Call get_weather once "
    "for each city."
)
NONE_PROMPT = "What is 2 + 2? Answer with the number only."
REASONING_PROMPT = (
    "A shop sells pens for 3 dollars each. Ana buys 4 pens and pays with a 20 dollar bill. How "
    "much change does she get? Think it through, then give only the number."
)
REASONING_ANSWER = "8"  # recorded, never a pass condition
OFF_PROMPT = "What is the capital of France? Answer with one word."
OFF_ANSWER = "paris"
# no JSON hint and no schema key: only enforced decoding (response_format) can answer in JSON
JSON_PROMPT = "Introduce the product NIM, release 2, in one short sentence."
IMAGE_PROMPT = "What is the main color of this image? Answer with one word."

TOOL_NAME = "get_weather"
WEATHER_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "city": {"type": "string", "description": "City name, for example Paris"},
        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
    },
    "required": ["city", "unit"],
}
WEATHER_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": TOOL_NAME,
        "description": "Get the current weather for a city.",
        "parameters": WEATHER_PARAMETERS,
    },
}
EXPECT_PARIS: dict[str, Any] = {"city": re.compile(r"paris", re.IGNORECASE), "unit": "celsius"}
EXPECT_PARALLEL: dict[str, Any] = {
    "city": re.compile(r"paris|tokyo", re.IGNORECASE), "unit": "celsius",
}
PARALLEL_CITIES = frozenset({"paris", "tokyo"})
JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"name": {"type": "string"}, "version": {"type": "integer"}},
    "required": ["name", "version"],
    "additionalProperties": False,
}
JSON_EXPECTED = {"name": "NIM", "version": 2}  # different values are only a warning
RESPONSE_FORMAT: dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {"name": "smoke", "schema": JSON_SCHEMA},
}

# --- what the checks look for --------------------------------------------------------------------

THINK_OPEN, THINK_CLOSE = "<think>", "</think>"
# Harmony and Kimi control tokens that leak into content without the right parser.
CONTROL_TOKENS = (
    "<|channel|>", "<|message|>", "<|start|>", "<|end|>", "<|constrain|>", "<|close|>",
    "<|open|>", "<|sep|>",
)
# Tool-call markup a model writes when the tool parser did not take the call out of the text.
TOOL_MARKUP = ("<TOOLCALL>", "[TOOL_CALLS]", "<tool_call>", "<|python_tag|>")
# vLLM's 400 texts when no tool parser is configured (vllm/renderers/online_renderer.py v0.31.0):
# '"auto" tool choice requires --enable-auto-tool-choice and --tool-call-parser to be set' and
# 'tool_choice=... requires --tool-call-parser to be set'.
PARSER_MISSING = re.compile(
    r"auto.{0,4}tool.choice requires|tool_choice=.{0,200}?requires --tool-call-parser",
    re.IGNORECASE | re.DOTALL,
)
# Llama-Nemotron Nano 8B v1 and 4B v1.1 "emergently" think on harder prompts with thinking off.
NANO_CAVEAT = re.compile(r"nemotron-nano-(?:4b|8b)", re.IGNORECASE)
NANO_CARD = "https://huggingface.co/nvidia/Llama-3.1-Nemotron-Nano-8B-v1"

PARSER_HINT = (
    "Documented server switch: start the NIM with the tool-call parser of this model, on vLLM "
    "--enable-auto-tool-choice --tool-call-parser <parser>, on SGLang --tool-call-parser "
    "<parser>, as container arguments. Changing the deployment is nim-deploy's job."
)
# answer_missing in chat, stream and concurrency: the text is all in a reasoning field.
REASONING_FIELD_HINT = (
    "Documented server switch: the reasoning parser puts the answer in the reasoning field, so a "
    "client that reads content gets none. Start the NIM with the reasoning parser of this model "
    "(on vLLM and SGLang --reasoning-parser <parser>), as container arguments. Changing the "
    "deployment is nim-deploy's job."
)
DEFAULT_PARSER_MISSING = {
    "status": NOT_APPLICABLE,
    "reason": "not_enabled_in_deployment",
    "warning": "The docs do not say whether this image turns the tool parser on by default; it "
               "is off in this deployment.",
}

# Sources of health and models when no plan entry is given (the table's engine sources).
DEFAULT_SOURCES = {
    "health": [
        "https://docs.nvidia.com/nim/large-language-models/2.1.5/get-started/quickstart.html"
        "#ready-endpoint",
        "https://docs.nvidia.com/nim/large-language-models/2.1.5/reference/api-reference.html"
        "#management-endpoints",
    ],
    "models": [
        "https://docs.nvidia.com/nim/large-language-models/2.1.5/get-started/quickstart.html"
        "#find-the-model-name",
    ],
}

# Every reason a check reports, with its meaning (references/checks.md has the same table).
REASONS: dict[str, str] = {
    OK: "the check passed",
    # any request
    "endpoint_unreachable": "no answer: the connection failed",
    "timeout": "no complete answer within the request timeout",
    "tls_error": "the TLS handshake or the certificate check failed",
    "request_failed": "the request could not be sent",
    "redirect_refused": "the endpoint answered with a redirect, which is never followed",
    "connection_lost": "the connection broke in the middle of the answer",
    "response_too_large": "the answer was larger than the read limit",
    "unauthorized": "HTTP 401 or 403: no key, or the key was refused",
    "rate_limited": "HTTP 429: the endpoint's rate limit; never retried",
    "http_error": "another HTTP status than 200",
    "not_json": "a 2xx answer that is not JSON",
    "bad_response_shape": "the JSON is not a chat completion as documented",
    "model_not_served": "the endpoint does not serve the model",
    "model_end_of_life": "the hosted model reached its end of life (HTTP 410)",
    "unexpected_finish_reason": "finish_reason is not stop where the check needs stop",
    # health
    "not_live": "/v1/health/live did not answer 200",
    "not_ready": "/v1/health/ready (or /health) did not answer 200",
    "health_timeout": "a health route did not answer in time",
    "health_route_not_served": "neither /v1/health/ready nor /health is served (n/a)",
    "health_port_not_reached": "NIM_HEALTH_PORT moved the health routes to a port this run "
                               "cannot reach (n/a)",
    "hosted_no_health_route": "the hosted API serves no health route (n/a)",
    # models (exit 3 errors, not results: no_models, model_ambiguous)
    "no_models": "/v1/models lists no model",
    # chat and concurrency
    "empty_completion": "no generated text, or zero completion tokens",
    "control_tokens_in_content": "raw control tokens such as <|channel|> in content",
    "rejected_503": "HTTP 503: the server's queue limit rejected a request",
    "gateway_error": "HTTP 502 or 504 from a proxy or gateway",
    # stream
    "stream_ignored": "the answer to stream: true is not an event stream of chunks",
    "stream_error_event": "an error event inside the stream, after HTTP 200",
    "stream_incomplete": "the stream ended before data: [DONE]",
    "empty_stream": "the stream carried no text",
    # tools
    "parser_not_enabled": "no tool parser here, but the image or the docs turn it on by default",
    "parser_dropped_by_override": "NIM_PASSTHROUGH_ARGS replaced the image default that set it",
    "not_enabled_in_deployment": "documented, but not enabled here: opt-in, or the image "
                                 "default is unknown (n/a)",
    "no_tool_call": "no tool call, after one retry (and the named diagnostic)",
    "model_did_not_call_tool_with_auto": "no call with auto, but a named call works",
    "tool_call_in_content": "call markup in content: the parser did not take it out",
    "tool_call_in_reasoning": "call markup inside the reasoning",
    "wrong_function": "the call names another function",
    "arguments_not_json_string": "function.arguments is not a JSON string",
    "invalid_json_arguments": "function.arguments does not parse as JSON",
    "double_encoded_arguments": "function.arguments is a JSON string inside a JSON string",
    "arguments_not_object": "function.arguments is JSON but not an object",
    "schema_mismatch": "the value does not match the schema",
    "argument_values_wrong": "the arguments do not match the prompt",
    "too_few_calls": "fewer than 2 calls with parallel_tool_calls: true",
    "tool_called_with_none": "a tool call with tool_choice: none",
    "empty_answer": "no answer text",
    # reasoning
    "switch_rejected": "a 4xx answer to the documented reasoning switch",
    "reasoning_missing": "no reasoning with reasoning on, after one retry",
    "reasoning_not_finished": "finish_reason length, after one retry with twice the budget",
    "answer_missing": "reasoning but no answer, with finish_reason stop",
    "reasoning_present": "reasoning with reasoning off",
    "answer_in_reasoning_field": "with reasoning off, the answer landed in the reasoning field",
    # json_schema
    "invalid_json": "the answer does not parse as JSON",
    "truncated_json": "finish_reason length: the JSON was cut",
    "json_only_in_reasoning": "the JSON is in the reasoning, not the answer",
    # image
    "image_ignored": "no more prompt tokens with the image than without it",
}

# Fail reasons about reaching the endpoint, not about how the model answered: a known issue of
# the model cannot explain them, so a fail with one of them does not point to the known issues.
ACCESS_REASONS = frozenset({
    "endpoint_unreachable", "timeout", "tls_error", "request_failed", "redirect_refused",
    "connection_lost", "response_too_large", "unauthorized", "rate_limited", "model_not_served",
    "model_end_of_life", "health_timeout",
})


def points_to_known_issues(status: str, reason: str, known_issues: Iterable[Any]) -> bool:
    """Whether a result's known issues may explain it: a fail on how the model answered (not
    an ACCESS_REASONS one) of a check whose plan lists known issues."""
    return status == FAIL and reason not in ACCESS_REASONS and any(
        isinstance(url, str) for url in known_issues or ()
    )


# What a not_applicable plan entry means (the reason codes of nsc_features.decide).
NA_MESSAGES: dict[str, str] = {
    "hosted_no_health_route": "The hosted API serves no health route.",
    "out_of_scope": "This model family is out of scope for the chat smoke check.",
    "no_capability_record": "No capability record matches this model, so this model-specific "
                            "check does not run. Ask the user for the page that documents it "
                            "and pass --documented FEATURE=URL; --family ID only when the served "
                            "id renames a model of that family.",
    "not_documented": "No source documents this feature for this model.",
    "not_documented_for_model": "A source says this model does not support it.",
    "not_a_reasoning_model": "The sources record this model as not a reasoning model.",
    "always_on": "The sources say this model always reasons: there is no off switch.",
    "no_documented_switch": "No source documents a switch for this direction.",
    "no_prefilled_think_block": "The template does not pre-fill the think block, so the "
                                "streaming twin adds nothing.",
    "reasoning_not_detectable": "This model's reasoning cannot be told apart from its answer.",
    "text_only_model": "The model takes text only.",
    "text_only_profile": "The active profile takes text only.",
    "information_only": "The sources disagree on parallel tool calls, so they are information "
                        "only.",
    "pipeline_parallel_unsupported": "NIM 1.x does not support this with pipeline parallelism "
                                     "above 1.",
    "not_enabled_in_deployment": "Documented, but not enabled in this deployment: opt-in for "
                                 "this model, or the docs do not say whether the image turns "
                                 "it on.",
    "health_route_not_served": "The endpoint serves no health route.",
    "health_port_not_reached": "The health routes moved to NIM_HEALTH_PORT, which this run "
                               "cannot reach; pass --health-url to check them.",
}


# --- results -------------------------------------------------------------------------------------


@dataclass
class Endpoint:
    """Where and how the checks send their requests.

    base: the base URL (nsc.normalize_base); the checks append /v1/...
    health_base: the base of the health routes (a NIM_HEALTH_PORT listener); None: base.
    headers: extra headers of the requests, such as Authorization (sent unredirected), which goes
    only to the scheme, host and port of base (headers_for).
    tls: the TLS settings (None: the system CA store); loopback: base is this machine, so its
    requests skip the proxy settings (a health_base elsewhere decides for itself: loopback_for).
    timeout_cap: the most seconds a chat request gets from the budget formula (request_timeout).
    timeout: the user's --timeout, when given: the seconds every chat request gets, up or down
    from the formula (a GET gets at most this); None: the formula.
    hosted: the hosted API: no health route, and the model route is /v1/models/ID.
    run_id: part of the X-Request-Id of every request, nsc-<run_id>-<check>-<n>.
    backend: "vllm" or "sglang" when known (only the wording of some messages).
    port_forward: the URL is a kubectl port-forward, which sends every request to one pod.
    """

    base: str
    health_base: str | None = None
    headers: dict[str, str] = field(default_factory=dict)
    tls: nsc.Tls | None = None
    loopback: bool = False
    timeout_cap: float = TIMEOUT_CAP
    hosted: bool = False
    run_id: str = "run"
    backend: str | None = None
    port_forward: bool = False
    timeout: float | None = None

    def url(self, path: str) -> str:
        return self.base.rstrip("/") + path

    def health_url(self, path: str) -> str:
        return (self.health_base or self.base).rstrip("/") + path

    def loopback_for(self, url: str) -> bool:
        """Whether a request to `url` skips the proxy settings: `loopback` for a URL on base;
        for one elsewhere (a --health-url on another host) its own host decides, so a remote
        health URL keeps the environment's proxy settings, as every other URL does."""
        base = self.base.rstrip("/")
        if url == base or url.startswith(base + "/"):
            return self.loopback
        return nsc.is_loopback(url)

    @property
    def sends_key(self) -> bool:
        return any(name.lower() == "authorization" for name in self.headers)

    def headers_for(self, url: str) -> dict[str, str]:
        """The extra headers of a request to `url`: all of `headers` when `url` has the scheme,
        host and port of base; elsewhere (a --health-url or NIM_HEALTH_PORT on another host or
        port, or over another scheme) the same without Authorization, so the key goes only to
        the URL the user gave."""
        if nsc.same_origin(url, self.base):
            return dict(self.headers)
        return {name: value for name, value in self.headers.items()
                if name.lower() != "authorization"}

    def sends_key_to(self, url: str) -> bool:
        """Whether a request to `url` carries the key (headers_for)."""
        return any(name.lower() == "authorization" for name in self.headers_for(url))


@dataclass
class Result:
    """The outcome of one check (see the module docstring); as_dict() is its JSON form."""

    id: str
    status: str
    reason: str
    message: str
    evidence: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    hint: str | None = None
    attempts: int = 0
    latency_ms: float | None = None
    warnings: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)
    known_issues: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def request_timeout(ep: Endpoint, max_tokens: int | None = None) -> float:
    """The seconds one request may take. A chat request: the user's --timeout when given
    (ep.timeout, which can raise the budget for a slow NIM as well as lower it), else
    60 s + max_tokens x 0.05 s, at most ep.timeout_cap. A GET: GET_TIMEOUT, or less when the
    cap or --timeout is lower."""
    if max_tokens is None:
        wanted = min(GET_TIMEOUT, float(ep.timeout_cap))
        if ep.timeout is not None:
            wanted = min(wanted, float(ep.timeout))
    elif ep.timeout is not None:
        wanted = float(ep.timeout)
    else:
        wanted = min(60.0 + max(int(max_tokens), 0) * 0.05, float(ep.timeout_cap))
    return max(wanted, 0.001)


# --- evidence ------------------------------------------------------------------------------------


class Evidence:
    """Saves one private JSON file per request, OUT/evidence/<check>-<n>.json (the folder 0700,
    the file 0600, written atomically and masked by nsc.SCRUBBER), and returns its path relative
    to OUT. See evidence_record for what a file holds."""

    def __init__(self, out_dir: str | Path) -> None:
        self.out_dir = Path(out_dir)
        self.folder = self.out_dir / "evidence"

    def save(
        self,
        check: str,
        attempt: int,
        request: Mapping[str, Any] | None,
        http_result: nsc.HttpResult | None,
        parsed: Mapping[str, Any] | None = None,
    ) -> str:
        name = f"{_file_part(check)}-{int(attempt)}.json"
        nsc.write_json_file(
            self.folder / name, evidence_record(check, attempt, request, http_result, parsed)
        )
        return f"evidence/{name}"


def _file_part(text: str) -> str:
    return re.sub(r"[^a-z0-9_]+", "_", str(text).lower()).strip("_") or "check"


def evidence_record(
    check: str,
    attempt: int,
    request: Mapping[str, Any] | None,
    http_result: nsc.HttpResult | None,
    parsed: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """What one evidence file holds: the request (method, url, headers with Authorization masked,
    body with inline image data replaced by its SHA-256 and size); the status, content type,
    latency, time to the first byte (streams), transport error and X-Request-Id; the response
    (JSON, text, or a raw event stream cut to RAW_LIMIT bytes); finish_reason and usage; and
    `parsed`, what the check read from it. Text longer than TEXT_LIMIT characters is cut; `cut`
    maps the path of each cut field to its full length (bytes for response_raw)."""
    cuts: dict[str, int] = {}
    record: dict[str, Any] = {
        "check": check,
        "attempt": int(attempt),
        "saved_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "request": _request_view(request, cuts),
    }
    body = (request or {}).get("body")
    if http_result is not None:
        res = http_result
        record.update(
            status=res.status,
            content_type=res.content_type or None,
            latency_ms=res.elapsed_ms,
            error=res.error,
            request_id=res.request_id,
            response_bytes=len(res.body),
        )
        if isinstance(body, Mapping) and body.get("stream") is True:
            record["ttfb_ms"] = res.ttfb_ms
        finish: Any = None
        usage: Any = None
        if res.content_type == "text/event-stream":
            # masked before the cut, so the cut cannot keep the first part of a key
            raw = nsc.SCRUBBER.scrub(res.body.decode("utf-8", "replace")).encode("utf-8")
            record["response_raw"] = raw[:RAW_LIMIT].decode("utf-8", "ignore")
            if len(raw) > RAW_LIMIT:
                cuts["$.response_raw"] = len(res.body)
            joined = join_stream(parse_sse(res.body)["events"])
            finish, usage = joined["finish_reason"], joined["usage"]
        else:
            data = res.json_body()
            if data is not None:
                record["response"] = _cut(data, "$.response", cuts)
                choice = _first_choice(data)
                finish = choice.get("finish_reason") if choice else None
                usage = data.get("usage") if isinstance(data, Mapping) else None
            elif res.body:
                record["response_text"] = _cut(
                    res.body.decode("utf-8", "replace"), "$.response_text", cuts
                )
        record["finish_reason"] = finish
        record["usage"] = _cut(usage, "$.usage", cuts)
    if parsed:
        record["parsed"] = _cut(dict(parsed), "$.parsed", cuts)
    record["cut"] = cuts
    return record


def _request_view(request: Mapping[str, Any] | None, cuts: dict[str, int]) -> dict[str, Any]:
    request = request or {}
    headers = {}
    for name, value in (request.get("headers") or {}).items():
        if value is None:
            continue
        if name.lower() in ("authorization", "proxy-authorization"):
            text = str(value)
            headers[name] = "Bearer ***" if text.lower().startswith("bearer") else "***"
        elif nsc.SECRET_WORD.search(name):
            headers[name] = "***"
        else:
            headers[name] = value
    return {
        "method": request.get("method"),
        "url": nsc.strip_userinfo(str(request.get("url") or "")),
        "headers": headers,
        "body": _cut(redact_images(request.get("body")), "$.request.body", cuts),
    }


def redact_images(value: Any, depth: int = 0) -> Any:
    """A copy of a request body with each inline data URL (data:...;base64,...) replaced by
    {"omitted": "inline data", "media_type", "sha256", "bytes"}; a list or object nested deeper
    than nsc.MAX_WALK_DEPTH levels becomes the text nsc.NESTED."""
    if isinstance(value, (Mapping, list, tuple)) and depth >= nsc.MAX_WALK_DEPTH:
        return nsc.NESTED
    if isinstance(value, str):
        match = re.match(r"data:([\w.+/-]*);base64,(.*)\Z", value, re.DOTALL)
        if not match:
            return value
        try:
            data = base64.b64decode(match.group(2), validate=False)
        except ValueError:
            data = match.group(2).encode()
        return {
            "omitted": "inline data",
            "media_type": match.group(1) or None,
            "sha256": hashlib.sha256(data).hexdigest(),
            "bytes": len(data),
        }
    if isinstance(value, Mapping):
        return {key: redact_images(item, depth + 1) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [redact_images(item, depth + 1) for item in value]
    return value


def _cut(value: Any, path: str, cuts: dict[str, int], depth: int = 0) -> Any:
    """A copy with every string longer than TEXT_LIMIT masked (nsc.SCRUBBER), then cut;
    cuts[path] = its full length. Masking first: a cut never keeps the first part of a key.
    A list or object nested deeper than nsc.MAX_WALK_DEPTH levels becomes the text nsc.NESTED."""
    if isinstance(value, (Mapping, list, tuple)) and depth >= nsc.MAX_WALK_DEPTH:
        return nsc.NESTED
    if isinstance(value, str):
        if len(value) <= TEXT_LIMIT:
            return value
        masked = nsc.SCRUBBER.scrub(value)
        if len(masked) <= TEXT_LIMIT:
            return masked
        cuts[path] = len(value)
        return masked[:TEXT_LIMIT]
    if isinstance(value, Mapping):
        return {key: _cut(item, f"{path}.{key}", cuts, depth + 1) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_cut(item, f"{path}[{index}]", cuts, depth + 1)
                for index, item in enumerate(value)]
    return value


# --- pure parsers --------------------------------------------------------------------------------


def parse_sse(raw: bytes | str) -> dict[str, Any]:
    """Read a server-sent event stream.

    Events are separated by blank lines; the data lines of one event are joined with "\\n";
    lines that start with ":" are comments (keep-alives); event:, id: and retry: lines are
    ignored. Returns {"events": the JSON objects, "comments", "done": a data: [DONE] arrived,
    "error_event": the first {"error": ...} (or {"object": "error"}) event or None,
    "bad_lines": data that is not a JSON object and lines that are not SSE, "after_done": events
    after [DONE]}. A line cut at the end of a broken stream lands in bad_lines."""
    text = raw.decode("utf-8", "replace") if isinstance(raw, (bytes, bytearray)) else str(raw)
    events: list[dict[str, Any]] = []
    comments: list[str] = []
    bad_lines: list[str] = []
    state: dict[str, Any] = {"done": False, "error": None, "after_done": 0}
    data_lines: list[str] = []

    def dispatch() -> None:
        if not data_lines:
            return
        data = "\n".join(data_lines)
        data_lines.clear()
        if state["done"]:
            state["after_done"] += 1
            return
        if data.strip() == "[DONE]":
            state["done"] = True
            return
        try:
            obj = nsc.parse_json(data)
        except ValueError:  # not JSON, or nested too deeply
            bad_lines.append(nsc.clip(data, 200))
            return
        if not isinstance(obj, dict):
            bad_lines.append(nsc.clip(data, 200))
            return
        if obj.get("error") is not None or obj.get("object") == "error":
            if state["error"] is None:
                state["error"] = obj["error"] if isinstance(obj.get("error"), dict) else obj
            return
        events.append(obj)

    for line in re.split(r"\r\n|\r|\n", text):  # not splitlines: it also splits on U+2028
        if line == "":
            dispatch()
            continue
        if line.startswith(":"):
            comments.append(nsc.clip(line[1:], 200))
            continue
        name, separator, value = line.partition(":")
        if not separator:
            bad_lines.append(nsc.clip(line, 200))
            continue
        if value.startswith(" "):
            value = value[1:]
        if name == "data":
            data_lines.append(value)
        elif name not in ("event", "id", "retry"):
            bad_lines.append(nsc.clip(line, 200))
    dispatch()
    return {
        "events": events,
        "comments": comments,
        "done": state["done"],
        "error_event": state["error"],
        "bad_lines": bad_lines,
        "after_done": state["after_done"],
    }


def message_text(content: Any) -> str:
    """A message content as text: a string, or the text parts of a list joined."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            str(part.get("text") or "")
            for part in content
            if isinstance(part, Mapping) and part.get("type") == "text"
        )
    return ""


def _split_inline(content: str) -> tuple[str, str, str]:
    close = content.find(THINK_CLOSE)
    opening = content.find(THINK_OPEN)
    if close >= 0:
        if 0 <= opening < close:
            reasoning = content[opening + len(THINK_OPEN):close]
            answer = content[:opening] + content[close + len(THINK_CLOSE):]
            where = "inline"
        else:  # the template wrote <think> into the prompt
            reasoning = content[:close]
            answer = content[close + len(THINK_CLOSE):]
            where = "inline_close_only"
    elif opening >= 0:  # cut off inside the think block
        reasoning = content[opening + len(THINK_OPEN):]
        answer = content[:opening]
        where = "unclosed"
    else:
        return "", content.strip(), "none"
    reasoning, answer = reasoning.strip(), answer.strip()
    return reasoning, answer, where if reasoning else "none"


def split_reasoning(message: Any) -> tuple[str, str, str]:
    """(reasoning, answer, where) of an assistant message.

    where: "reasoning" or "reasoning_content" (the message field; vLLM 0.18 and later fill
    reasoning, SGLang and older vLLM reasoning_content), "inline" (<think>X</think>Y in content),
    "inline_close_only" (X</think>Y: the template pre-filled <think>), "unclosed" (<think>X with
    no closing tag: cut off), or "none". None, "", white space and an empty <think></think>
    block all count as no reasoning. The answer is the content outside any think block."""
    if not isinstance(message, Mapping):
        return "", "", "none"
    inline, answer, where = _split_inline(message_text(message.get("content")))
    for key in ("reasoning", "reasoning_content"):
        value = message.get(key)
        if isinstance(value, str) and value.strip():
            reasoning = value.strip()
            if inline:
                reasoning = f"{reasoning}\n{inline}"
            return reasoning, answer, key
    return inline, answer, where


def join_stream(events: Iterable[Any]) -> dict[str, Any]:
    """Assemble the chunks of a streamed chat completion (choice 0).

    content: the joined delta.content; reasoning: the joined delta.reasoning, or
    delta.reasoning_content when a delta has no reasoning (never both from one delta);
    reasoning_field: the field it came in; tool_calls: assembled by index, with id, type and
    function.name from the first chunk that has them and function.arguments joined (arguments
    that are not a string are kept as sent); finish_reason: the last one sent; usage: the last
    usage object; chunks: the chat.completion.chunk objects; text_chunks: deltas with text."""
    content: list[str] = []
    reasoning: list[str] = []
    reasoning_field: str | None = None
    calls: dict[int, dict[str, Any]] = {}
    finish: Any = None
    usage: dict[str, Any] | None = None
    chunks = text_chunks = 0
    role = None
    for event in events:
        if not isinstance(event, Mapping):
            continue
        if event.get("object") == "chat.completion.chunk":
            chunks += 1
        if isinstance(event.get("usage"), Mapping):
            usage = dict(event["usage"])
        choices = event.get("choices")
        for choice in choices if isinstance(choices, list) else []:
            if not isinstance(choice, Mapping) or choice.get("index", 0) != 0:
                continue
            delta = choice.get("delta")
            if isinstance(delta, Mapping):
                role = role or delta.get("role")
                text = delta.get("content")
                if isinstance(text, str) and text:
                    content.append(text)
                    text_chunks += 1
                for key in ("reasoning", "reasoning_content"):
                    piece = delta.get(key)
                    if isinstance(piece, str) and piece:
                        reasoning.append(piece)
                        reasoning_field = reasoning_field or key
                        text_chunks += 1
                        break
                _add_call_deltas(calls, delta.get("tool_calls"))
            if choice.get("finish_reason"):
                finish = choice["finish_reason"]
    return {
        "content": "".join(content),
        "reasoning": "".join(reasoning),
        "reasoning_field": reasoning_field,
        "tool_calls": [calls[index] for index in sorted(calls)],
        "finish_reason": finish,
        "usage": usage,
        "chunks": chunks,
        "text_chunks": text_chunks,
        "role": role,
    }


def _add_call_deltas(calls: dict[int, dict[str, Any]], deltas: Any) -> None:
    for position, call in enumerate(deltas if isinstance(deltas, list) else []):
        if not isinstance(call, Mapping):
            continue
        index = call.get("index")
        if not isinstance(index, int) or isinstance(index, bool):
            index = position
        slot = calls.setdefault(
            index, {"id": None, "type": None, "function": {"name": None, "arguments": ""}}
        )
        if call.get("id") and not slot["id"]:
            slot["id"] = call["id"]
        if call.get("type") and not slot["type"]:
            slot["type"] = call["type"]
        function = call.get("function")
        if not isinstance(function, Mapping):
            continue
        if function.get("name") and not slot["function"]["name"]:
            slot["function"]["name"] = function["name"]
        arguments = function.get("arguments")
        if isinstance(arguments, str):
            if isinstance(slot["function"]["arguments"], str):
                slot["function"]["arguments"] += arguments
        elif arguments is not None:
            slot["function"]["arguments"] = arguments


def stream_message(joined: Mapping[str, Any]) -> dict[str, Any]:
    """The assistant message that a joined stream (join_stream) amounts to."""
    message: dict[str, Any] = {
        "role": joined.get("role") or "assistant",
        "content": joined.get("content") or None,
        "tool_calls": list(joined.get("tool_calls") or []),
    }
    if joined.get("reasoning"):
        message[joined.get("reasoning_field") or "reasoning"] = joined["reasoning"]
    return message


def _json_type(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return type(value).__name__


def _is_type(value: Any, kind: Any) -> bool:
    if kind == "object":
        return isinstance(value, dict)
    if kind == "array":
        return isinstance(value, list)
    if kind == "string":
        return isinstance(value, str)
    if kind == "boolean":
        return isinstance(value, bool)
    if kind == "null":
        return value is None
    if kind == "integer":
        return (isinstance(value, int) and not isinstance(value, bool)) or (
            isinstance(value, float) and value.is_integer()
        )
    if kind == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    return True  # a type this small validator does not know


def _same(left: Any, right: Any) -> bool:
    """JSON equality: true is not 1, and 1 equals 1.0."""
    if isinstance(left, bool) != isinstance(right, bool):
        return False
    return bool(left == right)


def short(value: Any, limit: int = SHORT) -> str:
    """A value quoted in a message: JSON, masked (nsc.SCRUBBER), then cut to `limit`
    characters."""
    try:
        text = json.dumps(value, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        text = str(value)
    text = nsc.SCRUBBER.scrub(text)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def validate_schema(value: Any, schema: Any, path: str = "$") -> list[str]:
    """The ways `value` breaks a small JSON schema (empty when it matches): type (a name or a
    list), properties, required, enum, additionalProperties (false, or a schema) and items."""
    if not isinstance(schema, Mapping):
        return []
    problems: list[str] = []
    enum = schema.get("enum")
    if isinstance(enum, list) and not any(_same(value, item) for item in enum):
        problems.append(f"{path}: {short(value)} is not one of {short(enum)}")
    kinds = schema.get("type")
    if kinds is not None:
        wanted = kinds if isinstance(kinds, list) else [kinds]
        if not any(_is_type(value, kind) for kind in wanted):
            problems.append(f"{path}: expected {'/'.join(map(str, wanted))}, got "
                            f"{_json_type(value)}")
            return problems
    if isinstance(value, dict):
        properties = schema.get("properties")
        properties = properties if isinstance(properties, Mapping) else {}
        for key in schema.get("required") or []:
            if key not in value:
                problems.append(f"{path}: the required key {key!r} is missing")
        for key, sub in properties.items():
            if key in value:
                problems += validate_schema(value[key], sub, f"{path}.{key}")
        extra = [key for key in value if key not in properties]
        additional = schema.get("additionalProperties", True)
        if additional is False and extra:
            problems.append(f"{path}: keys the schema does not allow: {short(extra)}")
        elif isinstance(additional, Mapping):
            for key in extra:
                problems += validate_schema(value[key], additional, f"{path}.{key}")
    if isinstance(value, list) and isinstance(schema.get("items"), Mapping):
        for index, item in enumerate(value):
            problems += validate_schema(item, schema["items"], f"{path}[{index}]")
    return problems


def tool_markup(text: str, name: str = TOOL_NAME) -> str | None:
    """The first tool-call marker in `text` (<TOOLCALL>, [TOOL_CALLS], <tool_call>,
    <|python_tag|>, or "name": "<name>"), or None."""
    if not text:
        return None
    found = [(text.find(marker), marker) for marker in TOOL_MARKUP if marker in text]
    named = re.search(r'"name"\s*:\s*"' + re.escape(name) + '"', text)
    if named:
        found.append((named.start(), named.group(0)))
    return min(found)[1] if found else None


def control_tokens(text: Any) -> list[str]:
    """The control tokens (CONTROL_TOKENS) in `text`, in the order they first appear."""
    if not isinstance(text, str) or not text:
        return []
    return [token for _, token in sorted(
        (text.find(token), token) for token in CONTROL_TOKENS if token in text
    )]


def validate_tool_call(
    message: Any,
    *,
    name: str = TOOL_NAME,
    schema: Mapping[str, Any] | None = None,
    expect: Mapping[str, Any] | None = None,
) -> tuple[str, str, dict[str, Any]]:
    """(status, reason, details) of the first tool call in an assistant message, checked in
    order: a call was made (no call: markup in the answer is tool_call_in_content, in the
    reasoning tool_call_in_reasoning, else no_tool_call; [], null and a missing tool_calls are
    the same); the function is `name` (and type "function" if given); arguments is a string;
    it parses (invalid_json_arguments) to an object (a string is double_encoded_arguments,
    anything else arguments_not_object); it matches `schema` (schema_mismatch; extra keys are a
    warning); the values match `expect` (key -> exact value, compiled pattern or predicate;
    argument_values_wrong). A missing id is a warning. Status is pass or fail."""
    schema = WEATHER_PARAMETERS if schema is None else schema
    details: dict[str, Any] = {"calls": 0, "warnings": []}
    if not isinstance(message, Mapping):
        return FAIL, "bad_response_shape", {**details, "problem": "no assistant message"}
    calls = message.get("tool_calls")
    if calls is not None and not isinstance(calls, list):
        return FAIL, "bad_response_shape", {**details, "problem": "tool_calls is not a list"}
    calls = calls or []
    details["calls"] = len(calls)
    if not calls:
        reasoning, answer, _ = split_reasoning(message)
        marker = tool_markup(answer, name)
        if marker:
            return FAIL, "tool_call_in_content", {**details, "markup": marker}
        marker = tool_markup(reasoning, name)
        if marker:
            return FAIL, "tool_call_in_reasoning", {**details, "markup": marker}
        return FAIL, "no_tool_call", details
    call = calls[0]
    if not isinstance(call, Mapping) or not isinstance(call.get("function"), Mapping):
        return FAIL, "bad_response_shape", {**details, "problem": "a call without a function"}
    function = call["function"]
    details["name"] = short(function.get("name"))
    if function.get("name") != name:
        return FAIL, "wrong_function", details
    if call.get("type") is not None and call.get("type") != "function":
        return FAIL, "bad_response_shape", {
            **details, "problem": f"type is {short(call.get('type'))}, not function"
        }
    if not call.get("id"):
        details["warnings"].append("The tool call has no id.")
    arguments = function.get("arguments")
    details["arguments_type"] = _json_type(arguments)
    if not isinstance(arguments, str):
        return FAIL, "arguments_not_json_string", details
    try:
        parsed = nsc.parse_json(arguments)
    except ValueError:  # not JSON, or nested too deeply
        return FAIL, "invalid_json_arguments", {**details, "arguments": short(arguments)}
    if isinstance(parsed, str):
        return FAIL, "double_encoded_arguments", {**details, "arguments": short(arguments)}
    if not isinstance(parsed, dict):
        return FAIL, "arguments_not_object", {**details, "arguments": short(parsed)}
    details["arguments"] = {key: short(value) for key, value in list(parsed.items())[:10]}
    problems = validate_schema(parsed, schema)
    if problems:
        return FAIL, "schema_mismatch", {**details, "problems": problems[:10]}
    known = schema.get("properties") if isinstance(schema.get("properties"), Mapping) else {}
    extra = sorted(key for key in parsed if key not in known)
    if extra:
        details["warnings"].append(f"Keys the tool does not define: {short(extra)}.")
    wrong = []
    for key, want in (expect or {}).items():
        value = parsed.get(key)
        if isinstance(want, re.Pattern):
            good = isinstance(value, str) and bool(want.search(value))
        elif callable(want):
            good = bool(want(value))
        else:
            good = _same(value, want)
        if not good:
            wrong.append(f"{key}={short(value, 40)}")
    if wrong:
        return FAIL, "argument_values_wrong", {**details, "wrong": wrong}
    return PASS, OK, details


def png_solid(width: int = 64, height: int = 64, rgb: tuple[int, int, int] = (255, 0, 0)) -> bytes:
    """A PNG of one solid color (8-bit RGB), built with zlib and struct: 64x64 red by default,
    about a hundred bytes."""
    if not (1 <= width <= 4096 and 1 <= height <= 4096):
        raise ValueError("width and height must be 1 to 4096 pixels")
    if len(rgb) != 3 or any(not isinstance(c, int) or not 0 <= c <= 255 for c in rgb):
        raise ValueError("rgb must be three integers from 0 to 255")

    def chunk(kind: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(kind + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", crc)

    row = b"\x00" + bytes(rgb) * width  # filter type 0, then the pixels
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(row * height, 9))
        + chunk(b"IEND", b"")
    )


def data_url(data: bytes, media_type: str = "image/png") -> str:
    return f"data:{media_type};base64," + base64.b64encode(data).decode("ascii")


# --- requests ------------------------------------------------------------------------------------


def check_request_fields(body: Mapping[str, Any]) -> None:
    """ValueError when a body holds a field outside REQUEST_FIELDS, chat_template inside
    chat_template_kwargs, or parallel_tool_calls other than true. A programming error: the
    checks build every body themselves."""
    unknown = sorted(set(body) - REQUEST_FIELDS)
    if unknown:
        raise ValueError(f"request fields outside the allow-list: {', '.join(unknown)}")
    kwargs = body.get("chat_template_kwargs")
    if kwargs is not None and (not isinstance(kwargs, Mapping) or "chat_template" in kwargs):
        raise ValueError("chat_template_kwargs must be an object without chat_template")
    if "parallel_tool_calls" in body and body["parallel_tool_calls"] is not True:
        raise ValueError("parallel_tool_calls is only ever sent as true")


def chat_body(
    model: str,
    content: Any,
    *,
    max_tokens: int,
    sampling: Mapping[str, Any] | None = None,
    switch: Mapping[str, Any] | None = None,
    mode: Iterable[Mapping[str, Any] | None] = (),
    stream: bool = False,
    **fields: Any,
) -> dict[str, Any]:
    """One chat request: a user message with `content`, max_tokens, the sampling (None:
    temperature 0; {} sends none, so the server's defaults apply), each switch of `mode`, then
    `switch` (nsc_features.apply_switch), stream with include_usage, and `fields`."""
    body: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": copy.deepcopy(content)}],
        "max_tokens": int(max_tokens),
    }
    for key, value in ({"temperature": 0} if sampling is None else dict(sampling)).items():
        if key in nf.SAMPLING_KEYS:
            body[key] = value
    for item in mode or ():
        nf.apply_switch(body, item)
    nf.apply_switch(body, switch)
    if stream:
        body["stream"] = True
        body["stream_options"] = {"include_usage": True}
    body.update(copy.deepcopy(fields))
    check_request_fields(body)
    return body


def _tools_body(model: str, prompt: str, request: Mapping[str, Any], **fields: Any) -> dict:
    return chat_body(
        model,
        prompt,
        max_tokens=int(request.get("max_tokens") or nf.TOOL_TOKENS),
        sampling=request.get("sampling"),
        mode=request.get("mode") or (),
        tools=[copy.deepcopy(WEATHER_TOOL)],
        **fields,
    )


# --- one check's requests ------------------------------------------------------------------------


@dataclass
class _Exchange:
    """One request and what was read from its answer."""

    number: int
    role: str
    res: nsc.HttpResult
    stream: bool
    data: Any = None
    sse: dict[str, Any] | None = None
    joined: dict[str, Any] | None = None
    message: dict[str, Any] | None = None
    finish_reason: Any = None
    usage: dict[str, Any] | None = None
    evidence: str | None = None


def _first_choice(data: Any) -> dict[str, Any] | None:
    if not isinstance(data, Mapping):
        return None
    choices = data.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], Mapping):
        return dict(choices[0])
    return None


def _read(res: nsc.HttpResult, number: int, role: str, stream: bool) -> _Exchange:
    ex = _Exchange(number, role, res, stream)
    if res.content_type == "text/event-stream":
        ex.sse = parse_sse(res.body)
        ex.joined = join_stream(ex.sse["events"])
        ex.message = stream_message(ex.joined)
        ex.finish_reason = ex.joined["finish_reason"]
        ex.usage = ex.joined["usage"]
        return ex
    ex.data = res.json_body()
    choice = _first_choice(ex.data)
    if choice is not None:
        message = choice.get("message")
        ex.message = dict(message) if isinstance(message, Mapping) else None
        ex.finish_reason = choice.get("finish_reason")
    if isinstance(ex.data, Mapping) and isinstance(ex.data.get("usage"), Mapping):
        ex.usage = dict(ex.data["usage"])
    return ex


def error_text(res: nsc.HttpResult) -> str:
    """The error message of an answer, one masked line of at most 200 characters:
    error.message (vLLM), message (SGLang), detail or title (problem+json), or the text body.
    It is masked before the cut, so the cut cannot keep the first part of an echoed key."""
    data = res.json_body()
    text = ""
    if isinstance(data, Mapping):
        error = data.get("error")
        if isinstance(error, Mapping):
            text = str(error.get("message") or "")
        elif isinstance(error, str):
            text = error
        text = text or str(data.get("message") or data.get("detail") or data.get("title") or "")
    if not text:
        text = res.body.decode("utf-8", "replace")
    return nsc.clip(text, 200)


def _transport(res: nsc.HttpResult) -> tuple[str, str] | None:
    """The reason of a request that got no usable answer at the transport level."""
    if res.status == 0:
        code = res.error_code
        detail = (res.error or "").partition(":")[2].strip()
        if code == nsc.ERR_TIMEOUT:
            return "timeout", "no answer within the request timeout"
        if code == nsc.ERR_TLS:
            return "tls_error", f"TLS failed: {detail}; pass --ca-file with the endpoint's CA"
        if code == nsc.ERR_REQUEST:
            return "request_failed", f"the request could not be sent: {detail}"
        return "endpoint_unreachable", f"no answer: {detail or res.error}"
    if 300 <= res.status < 400:
        return "redirect_refused", (
            f"HTTP {res.status}: a redirect, which is never followed (pass the final URL)"
        )
    return None


def _status_failure(
    res: nsc.HttpResult, ep: Endpoint, keyed: bool | None = None
) -> tuple[str, str]:
    """The reason of an HTTP status other than 200. `keyed`: whether this request carried the
    key (None: ep.sends_key, as every request to base does)."""
    detail = error_text(res)
    keyed = ep.sends_key if keyed is None else keyed
    if res.status in (401, 403):
        if keyed:
            why = "the key was refused"
        elif ep.sends_key:
            why = ("this URL wants a key, and the key goes only to the scheme, host and port of "
                   "--url: pass a --health-url there, or leave health out")
        else:
            why = "the endpoint wants a key: export it and pass --api-key-env NAME"
        return "unauthorized", f"HTTP {res.status}: {why}"
    if res.status == 404 and ("does not exist" in detail or b"NotFoundError" in res.body):
        return "model_not_served", f"HTTP 404: {detail}"
    if res.status == 410:
        return "model_end_of_life", f"HTTP 410: {detail}"
    if res.status == 429:
        return "rate_limited", (
            "HTTP 429: the endpoint's rate limit (the gateway's, not the model's); not retried"
        )
    return "http_error", f"HTTP {res.status}: {detail}"


def _answer_failure(ex: _Exchange, ep: Endpoint) -> tuple[str, str] | None:
    """The reason of an answer that is not a usable HTTP 200 JSON body, or None."""
    res = ex.res
    found = _transport(res)
    if found:
        return found
    if res.status != 200:
        return _status_failure(res, ep)
    found = _body_problem(res)
    if found:
        return found
    if ex.data is None:
        kind = res.content_type or "no content type"
        return "not_json", f"HTTP 200 with {kind}, not a JSON body"
    return None


def _body_problem(res: nsc.HttpResult) -> tuple[str, str] | None:
    """The reason of an answer whose body did not arrive whole, or None. A body shorter than
    its Content-Length counts as a lost connection: http.client's read1 returns the short body
    without an error when the server closes early."""
    if res.error_code == nsc.ERR_TOO_LARGE:
        return "response_too_large", "the answer is larger than the read limit"
    declared = res.headers.get("content-length", "").strip()
    short_body = res.error is None and declared.isdigit() and len(res.body) < int(declared)
    if res.error_code == nsc.ERR_LOST or short_body:
        return "connection_lost", "the connection broke in the middle of the answer"
    if res.error_code == nsc.ERR_TIMEOUT:
        return "timeout", "the answer did not end within the request timeout"
    return None


def _stream_failure(ex: _Exchange, ep: Endpoint) -> tuple[str, str] | None:
    """The reason of an answer to stream: true that is not a complete event stream, or None."""
    res = ex.res
    found = _transport(res)
    if found:
        return found
    if res.status != 200:
        return _status_failure(res, ep)
    if res.error_code == nsc.ERR_TOO_LARGE:
        return "response_too_large", "the stream is larger than the read limit"
    if res.content_type != "text/event-stream" or ex.sse is None:
        kind = res.content_type or "no content type"
        return "stream_ignored", f"HTTP 200 with {kind}: one body, not an event stream"
    if ex.sse["error_event"] is not None:
        error = ex.sse["error_event"]
        text = error.get("message") if isinstance(error, Mapping) else None
        return "stream_error_event", (
            f"an error event after HTTP 200: {nsc.clip(text or short(error), 150)}"
        )
    chunks = ex.joined["chunks"] if ex.joined else 0
    if chunks == 0 and res.error_code == nsc.ERR_TIMEOUT:
        return "timeout", "the stream started, but no chunk came within the request timeout"
    if chunks == 0 and res.error_code == nsc.ERR_LOST:
        return "connection_lost", "the connection broke after the headers, before the first chunk"
    if chunks == 0:
        return "stream_ignored", "no chat.completion.chunk objects in the stream"
    if not ex.sse["done"]:
        why = "timed out" if res.error_code == nsc.ERR_TIMEOUT else "ended"
        return "stream_incomplete", f"the stream {why} after {chunks} chunks, before [DONE]"
    if chunks < 2:
        return "stream_ignored", "the whole answer came in one chunk"
    return None


def _shape_problem(ex: _Exchange, strict: bool = False) -> str | None:
    data = ex.data
    if not isinstance(data, Mapping):
        return "the body is not a JSON object"
    if _first_choice(data) is None:
        return "no choices[0]"
    if ex.message is None:
        return "choices[0] has no message object"
    if strict and data.get("object") != "chat.completion":
        return f"object is {short(data.get('object'))}, not chat.completion"
    if strict and ex.message.get("role") != "assistant":
        return f"message.role is {short(ex.message.get('role'))}, not assistant"
    return None


def _sources(entry: Mapping[str, Any], check: str) -> list[str]:
    urls = [url for url in entry.get("sources") or [] if isinstance(url, str)]
    return urls or list(DEFAULT_SOURCES.get(check, []))


def _dedupe(items: Iterable[Any]) -> list[str]:
    return list(dict.fromkeys(str(item) for item in items if item))


def _cap(text: str, limit: int = MESSAGE_LIMIT) -> str:
    """One line, masked (nsc.SCRUBBER) before it is cut to `limit` characters."""
    text = nsc.SCRUBBER.scrub(" ".join(str(text).split()))
    return text if len(text) <= limit else text[: limit - 3] + "..."


class _Session:
    """The requests of one check: numbering, headers, evidence and the Result."""

    def __init__(
        self, ep: Endpoint, ev: Evidence | None, check: str, entry: Mapping[str, Any] | None
    ) -> None:
        self.ep = ep
        self.ev = ev
        self.entry: dict[str, Any] = dict(entry or {})
        self.check = str(self.entry.get("check") or check)
        self.request: dict[str, Any] = dict(self.entry.get("request") or {})
        self.evidence: list[str] = []
        self.warnings: list[str] = list(self.entry.get("warnings") or [])
        self.details: dict[str, Any] = {}
        self.attempts = 0
        self.count = 0
        self.lock = threading.Lock()

    @property
    def not_applicable(self) -> bool:
        return self.entry.get("decision") == NOT_APPLICABLE

    def number(self) -> int:
        with self.lock:
            self.count += 1
            return self.count

    def headers(self, number: int, url: str) -> dict[str, str]:
        """The headers of request `number` to `url`: the key only on base (Endpoint.headers_for)."""
        run = re.sub(r"[^A-Za-z0-9._-]+", "-", str(self.ep.run_id))[:64] or "run"
        return {**self.ep.headers_for(url), "X-Request-Id": f"nsc-{run}-{self.check}-{number}"}

    def save(
        self,
        number: int,
        request: Mapping[str, Any],
        res: nsc.HttpResult,
        parsed: Mapping[str, Any],
    ) -> str | None:
        if self.ev is None:
            return None
        path = self.ev.save(self.check, number, request, res, parsed)
        with self.lock:
            if path not in self.evidence:
                self.evidence.append(path)
        return path

    def post(
        self,
        body: dict[str, Any],
        *,
        role: str = "attempt",
        number: int | None = None,
        count: bool = True,
    ) -> _Exchange:
        """Send one chat request (an event stream when body["stream"]), keep its evidence;
        `count` adds it to the check's attempts."""
        check_request_fields(body)
        number = number or self.number()
        stream = body.get("stream") is True
        url = self.ep.url(CHAT_PATH)
        headers = self.headers(number, url)
        timeout = request_timeout(self.ep, body.get("max_tokens") or 0)
        loopback = self.ep.loopback_for(url)
        if stream:
            res = nsc.request_stream(url, body, headers=headers, timeout=timeout,
                                     tls=self.ep.tls, loopback=loopback)
        else:
            res = nsc.request_json("POST", url, body, headers=headers, timeout=timeout,
                                   tls=self.ep.tls, loopback=loopback)
        ex = _read(res, number, role, stream)
        sent = {
            "User-Agent": nsc.USER_AGENT,
            "Accept": "text/event-stream" if stream else "application/json",
            "Content-Type": "application/json",
            **headers,
        }
        request = {"method": "POST", "url": url, "headers": sent, "body": body}
        ex.evidence = self.save(number, request, res, self.parsed(ex))
        if count:
            with self.lock:
                self.attempts += 1
        return ex

    def get(self, url: str, *, role: str, number: int) -> nsc.HttpResult:
        """One GET (no body), kept as evidence under `number`."""
        headers = self.headers(number, url)
        res = nsc.request_json("GET", url, headers=headers, timeout=request_timeout(self.ep),
                               tls=self.ep.tls, loopback=self.ep.loopback_for(url))
        sent = {"User-Agent": nsc.USER_AGENT, "Accept": "application/json", **headers}
        self.save(number, {"method": "GET", "url": url, "headers": sent, "body": None}, res,
                  {"role": role})
        return res

    @staticmethod
    def parsed(ex: _Exchange) -> dict[str, Any]:
        info: dict[str, Any] = {"role": ex.role}
        if ex.message is not None:
            reasoning, answer, where = split_reasoning(ex.message)
            info.update(where=where, reasoning=reasoning, answer=answer,
                        tool_calls=ex.message.get("tool_calls") or [])
        if ex.sse is not None and ex.joined is not None:
            info["stream"] = {
                "chunks": ex.joined["chunks"],
                "text_chunks": ex.joined["text_chunks"],
                "done": ex.sse["done"],
                "error_event": ex.sse["error_event"],
                "comments": len(ex.sse["comments"]),
                "bad_lines": ex.sse["bad_lines"][:5],
            }
        return info

    def result(
        self,
        status: str,
        reason: str,
        message: str,
        *,
        ex: _Exchange | None = None,
        hint: str | None = None,
        details: Mapping[str, Any] | None = None,
        latency_ms: float | None = None,
        warnings: Iterable[str] = (),
    ) -> Result:
        issues = [url for url in self.entry.get("known_issues") or [] if isinstance(url, str)]
        text = _cap(message)
        if points_to_known_issues(status, reason, issues):  # listed with the run's warnings
            text = _cap(f"{text.rstrip('.')}. A known issue of this model may explain it: see "
                        "the warnings.")
        return Result(
            id=self.check,
            status=status,
            reason=reason,
            message=text,
            evidence=list(self.evidence),
            sources=_sources(self.entry, self.check),
            hint=hint,
            attempts=self.attempts,
            latency_ms=latency_ms if latency_ms is not None else (
                ex.res.elapsed_ms if ex is not None else None
            ),
            warnings=_dedupe([*self.warnings, *warnings]),
            details={**self.details, **(details or {})},
            known_issues=issues,
        )

    def not_applicable_result(self) -> Result:
        reason = str(self.entry.get("reason") or "not_documented")
        message = NA_MESSAGES.get(reason, f"Not applicable: {reason}.")
        return self.result(NOT_APPLICABLE, reason, message, hint=self.entry.get("hint"))


def result_from_entry(entry: Mapping[str, Any]) -> Result:
    """The not_applicable Result of a plan entry whose decision is not_applicable."""
    return _Session(Endpoint(base=""), None, str(entry.get("check") or "check"),
                    entry).not_applicable_result()


# --- health and models ---------------------------------------------------------------------------


def check_health(
    ep: Endpoint,
    ev: Evidence | None,
    plan_entry: Mapping[str, Any] | None = None,
    *,
    wait: float = 0.0,
    poll: float = HEALTH_POLL,
    unreachable: str | None = None,
) -> Result:
    """GET /v1/health/live, then /v1/health/ready, on health_base; GET /health when ready is
    404. Judged by status code only. Pass: both 200 (or /health 200 when the NIM routes are
    404). Fail: not_live, not_ready (503), health_timeout, or the transport reasons. Not
    applicable: hosted_no_health_route; health_route_not_served when ready and /health are both
    404; health_port_not_reached when `unreachable` says why the health routes cannot be
    reached from here (NIM_HEALTH_PORT moved them to a port that is not published or
    forwarded), with nothing sent. `wait` > 0 polls every `poll` seconds until ready or `wait`
    seconds pass (--wait-ready); by default it probes once. With `unreachable`, a `wait` > 0
    raises nsc.ScriptError health_port_not_reached (exit 3): the readiness it asks for cannot
    be read."""
    s = _Session(ep, ev, "health", plan_entry)
    if s.not_applicable:
        return s.not_applicable_result()
    if ep.hosted:
        ignored = ["--wait-ready was not applied: the hosted API has no health route."]
        return s.result(NOT_APPLICABLE, "hosted_no_health_route",
                        NA_MESSAGES["hosted_no_health_route"], warnings=ignored if wait > 0 else ())
    if unreachable:
        if wait > 0:
            raise nsc.ScriptError(
                "health_port_not_reached",
                f"--wait-ready {wait:g} polls the health routes, and they cannot be reached "
                f"from here: {unreachable}",
                "Ask the user for a --health-url that reaches the health port, or rerun without "
                "--wait-ready once the NIM is ready",
                nsc.EXIT_PRECONDITION,
            )
        return s.result(NOT_APPLICABLE, "health_port_not_reached", f"Not checked: {unreachable}",
                        warnings=[NA_MESSAGES["health_port_not_reached"]])
    deadline = time.monotonic() + max(float(wait), 0.0)
    polls = 0
    while True:
        polls += 1
        live = s.get(ep.health_url("/v1/health/live"), role="live", number=1)
        ready = s.get(ep.health_url("/v1/health/ready"), role="ready", number=2)
        fallback = None
        if ready.status == 404:
            fallback = s.get(ep.health_url("/health"), role="fallback /health", number=3)
        status, reason, message, warnings = _judge_health(live, ready, fallback, ep)
        left = deadline - time.monotonic()
        if status != FAIL or left <= 0:
            break
        time.sleep(max(min(float(poll), left), 0.01))
    s.attempts = polls
    details = {
        "live": live.status,
        "ready": ready.status,
        "fallback": fallback.status if fallback is not None else None,
        "polls": polls,
    }
    if polls > 1 and status == FAIL:
        message = f"{message} (after {polls} probes over {wait:g} s)"
    return s.result(status, reason, message, details=details, warnings=warnings,
                    latency_ms=ready.elapsed_ms)


def _judge_health(
    live: nsc.HttpResult,
    ready: nsc.HttpResult,
    fallback: nsc.HttpResult | None,
    ep: Endpoint,
) -> tuple[str, str, str, list[str]]:
    for res in (live, ready, fallback):
        if res is None:
            continue
        found = _transport(res)
        if found:
            if found[0] == "timeout":
                return FAIL, "health_timeout", "a health route did not answer in time", []
            return FAIL, found[0], found[1], []
    keyed = ep.sends_key_to(ep.health_url("/v1/health/live"))
    if live.status in (401, 403, 429):  # a gateway's answer, not the NIM's liveness
        reason, message = _status_failure(live, ep, keyed)
        return FAIL, reason, f"/v1/health/live: {message}", []
    if live.status not in (200, 404):
        return FAIL, "not_live", f"/v1/health/live answered HTTP {live.status}", []
    if ready.status == 200:
        if live.status == 404:
            return PASS, OK, "/v1/health/ready answered 200", [
                "/v1/health/live is not served (404)."
            ]
        return PASS, OK, "/v1/health/live and /v1/health/ready answered 200", []
    if ready.status == 404 and fallback is not None:
        if fallback.status == 200:
            return PASS, OK, "/health answered 200 (the NIM health routes answered 404)", [
                "The NIM health routes are not served; vLLM's own /health answered 200, which "
                "says the engine is alive, not that the model is ready."
            ]
        if fallback.status == 404:
            return NOT_APPLICABLE, "health_route_not_served", (
                "/v1/health/ready and /health answered 404; a gateway may expose only the "
                "inference routes"
            ), ["The endpoint serves no health route, so health was not checked."]
        if fallback.status == 503:
            return FAIL, "not_ready", "/health answered 503: the model is still loading", []
        reason, message = _status_failure(fallback, ep, keyed)
        return FAIL, reason, f"/health: {message}", []
    if ready.status == 503:
        return FAIL, "not_ready", (
            "/v1/health/ready answered 503: the model is still downloading or loading"
        ), []
    if ready.status in (401, 403, 429):
        reason, message = _status_failure(ready, ep, keyed)
        return FAIL, reason, f"/v1/health/ready: {message}", []
    if ready.status >= 500:
        return FAIL, "not_ready", f"/v1/health/ready answered HTTP {ready.status}", []
    return FAIL, "http_error", f"/v1/health/ready answered HTTP {ready.status}", []


def _card(item: Mapping[str, Any]) -> dict[str, Any]:
    return {key: item.get(key) for key in ("id", "root", "parent", "owned_by", "max_model_len")}


def check_models(
    ep: Endpoint,
    requested: str | None,
    ev: Evidence | None = None,
    plan_entry: Mapping[str, Any] | None = None,
) -> tuple[Result, str | None, list[dict[str, Any]]]:
    """GET /v1/models (hosted, with a requested id: GET /v1/models/ID, which needs no key) and
    choose the model the other checks test: the requested id when it is served; with no
    requested id, the one base model (a card whose parent is null; LoRA adapters name their
    base as parent). A requested id that is not served fails with model_not_served (the served
    ids in the message): another model is never tested in its place. A card whose id is not
    text is left out with a warning; when no card has a text id, bad_response_shape.

    Returns (result, chosen id or None, cards: id, root, parent, owned_by, max_model_len).
    chosen is None when nothing can be tested (model_not_served, model_end_of_life, or a failed
    /v1/models with no requested id). Raises nsc.ScriptError: endpoint_unreachable (exit 4) when
    /v1/models gets no answer; no_models (exit 3) when it lists none; model_ambiguous (exit 3)
    when no id is requested and the base models served are not exactly one."""
    s = _Session(ep, ev, "models", plan_entry)
    s.attempts = 1
    hosted_one = ep.hosted and bool(requested)
    if hosted_one:
        url = ep.url("/v1/models/" + urllib.parse.quote(str(requested), safe="/:@"))
    else:
        url = ep.url("/v1/models")
    res = s.get(url, role="models", number=1)
    found = _transport(res)
    if found and found[0] != "redirect_refused":
        raise nsc.ScriptError(
            "endpoint_unreachable",
            f"GET {url} got no answer: {found[1]}",
            "Check the URL and port with the user (for Kubernetes, the port-forward), or pass "
            "--ca-file for a private CA; then rerun",
            nsc.EXIT_EXTERNAL,
            {"result": s.result(FAIL, found[0], found[1]).as_dict()},
        )
    if found:
        return s.result(FAIL, found[0], found[1], latency_ms=res.elapsed_ms), requested, []
    if res.status != 200:
        reason, message = _status_failure(res, ep)
        if hosted_one and res.status == 404:
            reason, message = "model_not_served", f"the hosted API does not serve {requested}"
        if reason in ("model_not_served", "model_end_of_life"):
            return s.result(FAIL, reason, message, latency_ms=res.elapsed_ms), None, []
        return s.result(FAIL, reason, message, latency_ms=res.elapsed_ms), requested, []
    found = _body_problem(res)
    if found:
        return s.result(FAIL, *found, latency_ms=res.elapsed_ms), requested, []
    data = res.json_body()
    if data is None:
        message = f"HTTP 200 with {res.content_type or 'no content type'}, not JSON"
        return s.result(FAIL, "not_json", message, latency_ms=res.elapsed_ms), requested, []
    if hosted_one:
        cards = [_card(data)] if isinstance(data, Mapping) else []
        if not cards or cards[0]["id"] != requested:
            message = f"/v1/models/{requested} did not return that model's card"
            return s.result(FAIL, "bad_response_shape", message, details={"cards": cards},
                            latency_ms=res.elapsed_ms), requested, cards
        return s.result(PASS, OK, f"the hosted API serves {requested}",
                        details={"chosen": requested, "card": cards[0]},
                        latency_ms=res.elapsed_ms), requested, cards
    items = data.get("data") if isinstance(data, Mapping) else None
    if not isinstance(items, list):
        message = 'the body has no "data" list of models'
        return s.result(FAIL, "bad_response_shape", message,
                        latency_ms=res.elapsed_ms), requested, []
    cards = [_card(item) for item in items if isinstance(item, Mapping) and item.get("id")]
    # an id that is not text breaks the OpenAI shape: such a card can name no model to test
    odd = [card for card in cards if not isinstance(card["id"], str)]
    cards = [card for card in cards if isinstance(card["id"], str)]
    if odd and not cards:
        message = (f"no model card has a text id (the id of {len(odd)} card(s) is "
                   f"{_json_type(odd[0]['id'])}), so the served model cannot be named")
        return s.result(FAIL, "bad_response_shape", message, details={"odd_cards": len(odd)},
                        latency_ms=res.elapsed_ms), requested, []
    if odd:
        s.warnings.append(f"{len(odd)} model card(s) whose id is not text were left out.")
    ids = [card["id"] for card in cards]
    base = [card["id"] for card in cards if card.get("parent") in (None, "")]
    s.details = {"served": ids[:50], "base_models": base[:50]}
    if not ids:
        raise nsc.ScriptError(
            "no_models",
            f"GET {url} lists no model",
            "Ask the user whether the NIM finished loading (nim-deploy checks a deployment), "
            "then rerun",
            nsc.EXIT_PRECONDITION,
            {"result": s.result(FAIL, "no_models", "/v1/models lists no model").as_dict()},
        )
    if requested and requested in ids:
        chosen = requested
    elif requested:
        # the user named the model to test: another model is never tested in its place, and
        # a mistyped id fails the same way however many base models are served
        return s.result(FAIL, "model_not_served", f"{requested} is not served; the endpoint "
                        f"serves {', '.join(ids[:20])}", latency_ms=res.elapsed_ms), None, cards
    elif len(base) == 1:
        chosen = base[0]
    else:
        shown = ", ".join((base or ids)[:20])
        raise nsc.ScriptError(
            "model_ambiguous",
            f"{url} serves {len(base or ids)} models ({shown})",
            "Ask the user which served model to test, then rerun with --model ID for each id "
            "they pick; if they do not know, test none",
            nsc.EXIT_PRECONDITION,
            {"served_models": ids[:50], "base_models": base[:50]},
        )
    card = next((card for card in cards if card["id"] == chosen), None)
    return s.result(PASS, OK, f"{chosen} is served", details={"chosen": chosen, "card": card},
                    latency_ms=res.elapsed_ms), chosen, cards


# --- chat and stream -----------------------------------------------------------------------------


def _norm_word(text: str) -> str:
    return re.sub(r"[\s.!\"'`*]+", " ", text).strip().lower()


def _answer_missing(
    ep: Endpoint, reasoning: str, answer: str, where: str, finish: Any
) -> tuple[str, str | None] | None:
    """(message, hint) of answer_missing for a chat answer with reasoning, no answer and
    finish_reason stop, else None. With stop the model finished, so no budget cut the answer:
    a client that reads content gets nothing. finish_reason length (the budget ran out) is the
    caller's warning, not a fail."""
    if answer or not reasoning or finish != "stop":
        return None
    if where in ("reasoning", "reasoning_content"):
        return (f"content is empty with finish_reason stop and {where} holds all the text "
                f"({len(reasoning)} characters): the reasoning parser put the answer in the "
                "reasoning field"), None if ep.hosted else REASONING_FIELD_HINT
    return (f"reasoning ({len(reasoning)} characters, {where}) but no answer, finish_reason "
            "stop"), None


def check_chat(
    ep: Endpoint, ev: Evidence | None, model: str, plan_entry: Mapping[str, Any] | None = None
) -> Result:
    """One short chat with no reasoning switch, as a plain client sends it. Pass: HTTP 200, a
    chat.completion whose message role is assistant, text in content or in a reasoning field,
    usage.completion_tokens > 0, and no control tokens in content. Reasoning with no answer
    fails (answer_missing) with finish_reason stop and warns with length."""
    s = _Session(ep, ev, "chat", plan_entry)
    if s.not_applicable:
        return s.not_applicable_result()
    max_tokens = int(s.request.get("max_tokens") or nf.CHAT_TOKENS_REASONING)
    ex = s.post(chat_body(model, CHAT_PROMPT, max_tokens=max_tokens))
    found = _answer_failure(ex, ep)
    if found:
        return s.result(FAIL, *found, ex=ex)
    problem = _shape_problem(ex, strict=True)
    if problem:
        return s.result(FAIL, "bad_response_shape", problem, ex=ex)
    content = message_text(ex.message.get("content"))
    reasoning, answer, where = split_reasoning(ex.message)
    details = {"finish_reason": ex.finish_reason, "where": where, "answer_chars": len(answer),
               "reasoning_chars": len(reasoning)}
    tokens = control_tokens(content)
    if tokens:
        return s.result(FAIL, "control_tokens_in_content",
                        f"content holds control tokens {', '.join(tokens)}", ex=ex,
                        details=details)
    if not (reasoning or answer):
        return s.result(FAIL, "empty_completion", "HTTP 200 with no generated text", ex=ex,
                        details=details)
    completion = (ex.usage or {}).get("completion_tokens")
    details["completion_tokens"] = completion
    if not isinstance(completion, int) or isinstance(completion, bool):
        return s.result(FAIL, "bad_response_shape", "the answer has no usage.completion_tokens",
                        ex=ex, details=details)
    if completion <= 0:
        return s.result(FAIL, "empty_completion", "usage.completion_tokens is 0", ex=ex,
                        details=details)
    missing = _answer_missing(ep, reasoning, answer, where, ex.finish_reason)
    if missing:
        return s.result(FAIL, "answer_missing", missing[0], ex=ex, hint=missing[1],
                        details=details)
    warnings = []
    if not answer and ex.finish_reason == "length":
        warnings.append(f"content is empty and finish_reason is length: the reasoning used the "
                        f"whole budget of {max_tokens} tokens before an answer.")
    elif answer and _norm_word(answer) != "ok":
        warnings.append(f"The answer is not OK: {short(answer, 40)}.")
    if answer:
        what = "an answer"
    elif ex.finish_reason == "length":
        what = f"reasoning only, no answer within {max_tokens} tokens"
    else:
        what = "reasoning only, no answer"
    return s.result(PASS, OK, f"HTTP 200 with {what} ({completion} completion tokens, "
                    f"finish_reason {ex.finish_reason})", ex=ex, details=details,
                    warnings=warnings)


def check_stream(
    ep: Endpoint, ev: Evidence | None, model: str, plan_entry: Mapping[str, Any] | None = None
) -> Result:
    """The chat request with stream: true and stream_options.include_usage. Pass: an event
    stream of 2 or more chat.completion.chunk objects, text, a finish_reason and data: [DONE].
    A missing usage chunk is a warning. Reasoning with no answer fails (answer_missing) with
    finish_reason stop and warns with length."""
    s = _Session(ep, ev, "stream", plan_entry)
    if s.not_applicable:
        return s.not_applicable_result()
    max_tokens = int(s.request.get("max_tokens") or nf.CHAT_TOKENS_REASONING)
    ex = s.post(chat_body(model, CHAT_PROMPT, max_tokens=max_tokens, stream=True))
    found = _stream_failure(ex, ep)
    joined = ex.joined or {}
    details = {
        "chunks": joined.get("chunks"),
        "text_chunks": joined.get("text_chunks"),
        "ttfb_ms": ex.res.ttfb_ms,
        "total_ms": ex.res.elapsed_ms,
        "keepalive_comments": len((ex.sse or {}).get("comments", [])),
        "finish_reason": joined.get("finish_reason"),
    }
    if found:
        return s.result(FAIL, *found, ex=ex, details=details)
    if not (joined["content"].strip() or joined["reasoning"].strip()):
        return s.result(FAIL, "empty_stream", f"{joined['chunks']} chunks and [DONE], but no text",
                        ex=ex, details=details)
    if not joined["finish_reason"]:
        return s.result(FAIL, "bad_response_shape", "no chunk carries a finish_reason", ex=ex,
                        details=details)
    reasoning, answer, where = split_reasoning({
        "content": joined["content"], joined.get("reasoning_field") or "reasoning":
        joined["reasoning"]})
    missing = _answer_missing(ep, reasoning, answer, where, joined["finish_reason"])
    if missing:
        return s.result(FAIL, "answer_missing", f"{missing[0]} (streamed)", ex=ex,
                        hint=missing[1], details=details)
    warnings = []
    if not answer and joined["finish_reason"] == "length":
        warnings.append(f"content is empty and finish_reason is length: the reasoning used the "
                        f"whole budget of {max_tokens} tokens before an answer.")
    if joined["usage"] is None:
        warnings.append("No usage chunk although stream_options.include_usage was sent.")
    return s.result(PASS, OK, f"{joined['chunks']} chunks, first byte after "
                    f"{ex.res.ttfb_ms} ms, [DONE] after {ex.res.elapsed_ms} ms", ex=ex,
                    details=details, warnings=warnings)


# --- tools ---------------------------------------------------------------------------------------


def _parser_missing(ex: _Exchange) -> bool:
    if ex.res.status != 400:
        return False
    return bool(PARSER_MISSING.search(error_text(ex.res))
                or PARSER_MISSING.search(ex.res.body[:2000].decode("utf-8", "replace")))


def _parser_missing_result(s: _Session, ex: _Exchange) -> Result:
    """D5: what the plan says a missing tool parser means for this model and deployment. A
    not_enabled_in_deployment plan with a warning is the unknown default (the docs do not say
    whether the image turns the parser on); without one, the record says it is opt-in."""
    plan = dict(s.entry.get("if_parser_missing") or DEFAULT_PARSER_MISSING)
    status = plan.get("status") if plan.get("status") in (FAIL, NOT_APPLICABLE) else NOT_APPLICABLE
    reason = str(plan.get("reason") or "not_enabled_in_deployment")
    texts = {
        "parser_not_enabled": "no tool parser in this deployment (vLLM 400 \"tool choice "
                              "requires ...\"), but the image or its docs turn it on by default",
        "parser_dropped_by_override": "no tool parser in this deployment: its "
                                      "NIM_PASSTHROUGH_ARGS replaced the image's default, which "
                                      "turned the tool parser on",
        "not_enabled_in_deployment": "tool calling is documented for this model but opt-in, and "
                                     "this deployment has no tool parser (vLLM 400 \"tool "
                                     "choice requires ...\")",
    }
    if reason == "not_enabled_in_deployment" and plan.get("warning"):
        texts[reason] = ("tool calling is documented for this model, and this deployment has no "
                         "tool parser (vLLM 400 \"tool choice requires ...\"); whether this "
                         "image turns it on by default is not known (see the warning)")
    hint = s.entry.get("hint") or (None if s.ep.hosted else PARSER_HINT)
    warnings = [plan["warning"]] if plan.get("warning") else []
    return s.result(status, reason, texts.get(reason, error_text(ex.res)), ex=ex, hint=hint,
                    warnings=warnings, details={"error": error_text(ex.res)})


def _tool_verdict(
    s: _Session,
    ex: _Exchange,
    *,
    stream: bool = False,
    expect: Mapping[str, Any] | None = None,
) -> tuple[str, str, str, dict[str, Any]]:
    """(status, reason, message, details) of one tools exchange; reason "parser_missing" marks
    vLLM's 400 for a missing tool parser."""
    if _parser_missing(ex):
        return FAIL, "parser_missing", "", {}
    found = _stream_failure(ex, s.ep) if stream else _answer_failure(ex, s.ep)
    if found:
        return FAIL, found[0], found[1], {}
    if not stream:
        problem = _shape_problem(ex)
        if problem:
            return FAIL, "bad_response_shape", problem, {}
    status, reason, details = validate_tool_call(
        ex.message, name=TOOL_NAME, schema=WEATHER_PARAMETERS,
        expect=EXPECT_PARIS if expect is None else expect,
    )
    details["finish_reason"] = ex.finish_reason
    return status, reason, _tool_message(s, reason, details), details


def _tool_message(s: _Session, reason: str, details: Mapping[str, Any]) -> str:
    sglang = s.ep.backend == "sglang"
    texts = {
        OK: f"a valid {TOOL_NAME} call (finish_reason {details.get('finish_reason')})",
        "no_tool_call": "no tool call, only text"
        + ("; on SGLang the tool parser may be missing" if sglang else ""),
        "tool_call_in_content": f"no tool call, but call markup {details.get('markup')} in "
                                "content: the tool parser did not take the call out (it may be "
                                "missing or wrong for this model)",
        "tool_call_in_reasoning": f"no tool call, but call markup {details.get('markup')} in "
                                  "the reasoning: calls are parsed from content only",
        "wrong_function": f"the call names {details.get('name')}, not {TOOL_NAME}",
        "arguments_not_json_string": "function.arguments is a JSON "
                                     f"{details.get('arguments_type')}, not a string",
        "invalid_json_arguments": "function.arguments does not parse as JSON",
        "double_encoded_arguments": "function.arguments is a JSON string that holds JSON "
                                    "(double encoded)",
        "arguments_not_object": "function.arguments is JSON but not an object",
        "schema_mismatch": "the arguments do not match the tool schema: "
                           + nsc.clip("; ".join(details.get("problems") or []), 150),
        "argument_values_wrong": "the arguments do not match the prompt: "
                                 + ", ".join(details.get("wrong") or []),
        "bad_response_shape": f"the tool call is malformed: {details.get('problem')}",
    }
    return texts.get(reason, reason)


def check_tools(
    ep: Endpoint, ev: Evidence | None, model: str, plan_entry: Mapping[str, Any] | None = None
) -> Result:
    """T-AUTO: one get_weather tool, tool_choice "auto", the family's mode switches, sampling
    and budget. Validated by validate_tool_call. vLLM's 400 for a missing tool parser is judged
    by the plan entry's if_parser_missing (D5), with the documented switch as hint. Plain text
    is retried once; then, where named tool_choice is documented (request.named_diagnostic), one
    named call is sent as a diagnostic: a valid one gives model_did_not_call_tool_with_auto.
    A named call never makes the check pass."""
    s = _Session(ep, ev, "tools", plan_entry)
    if s.not_applicable:
        return s.not_applicable_result()
    body = _tools_body(model, TOOL_PROMPT, s.request, tool_choice="auto")
    hint = s.entry.get("hint")
    history = []
    ex = s.post(body)
    status, reason, message, details = _tool_verdict(s, ex)
    history.append(reason)
    if reason == "parser_missing":
        return _parser_missing_result(s, ex)
    if reason == "no_tool_call":
        ex = s.post(body, role="retry")
        status, reason, message, details = _tool_verdict(s, ex)
        history.append(reason)
        if reason == "parser_missing":
            return _parser_missing_result(s, ex)
    s.details["attempt_reasons"] = history
    if reason == "no_tool_call" and s.request.get("named_diagnostic", True):
        named = dict(body, tool_choice={"type": "function", "function": {"name": TOOL_NAME}})
        diagnostic = s.post(named, role="named_diagnostic", count=False)
        d_status, d_reason, _, _ = _tool_verdict(s, diagnostic)
        if d_reason == "parser_missing":  # named calls need the parser too (vLLM)
            d_reason = "http_error"
        s.details["named_diagnostic"] = {"reason": d_reason, "evidence": diagnostic.evidence}
        if d_status == PASS:
            return s.result(FAIL, "model_did_not_call_tool_with_auto",
                            "no call with tool_choice auto (twice), but a named call is valid: "
                            "guided decoding works, the default path does not", ex=ex,
                            details=details)
        message = f"{message}; the named diagnostic call gave {d_reason} too"
    elif reason == "no_tool_call":
        message = f"{message} (twice; this model documents no named tool_choice for a diagnostic)"
    attach = reason in ("tool_call_in_content", "tool_call_in_reasoning") or (
        reason == "no_tool_call" and ep.backend == "sglang"
    )
    return s.result(status, reason, message, ex=ex, details=details,
                    hint=hint if attach else None,
                    warnings=details.get("warnings") or ())


def check_tools_stream(
    ep: Endpoint, ev: Evidence | None, model: str, plan_entry: Mapping[str, Any] | None = None
) -> Result:
    """T-STREAM: the T-AUTO request with stream: true; delta.tool_calls are assembled by index
    (join_stream) and validated the same way. Retried once only when the plan entry says so
    (request.retry: gpt-oss, a known intermittent Harmony parser error)."""
    s = _Session(ep, ev, "tools_stream", plan_entry)
    if s.not_applicable:
        return s.not_applicable_result()
    body = _tools_body(model, TOOL_PROMPT, s.request, tool_choice="auto", stream=True)
    retries = min(int(s.request.get("retry") or 0), 1)
    no_retry = {"parser_missing", "rate_limited", "unauthorized", "model_not_served",
                "model_end_of_life", "redirect_refused", "endpoint_unreachable", "tls_error"}
    while True:
        ex = s.post(body, role="attempt" if s.attempts == 0 else "retry")
        status, reason, message, details = _tool_verdict(s, ex, stream=True)
        if reason == "parser_missing":
            return _parser_missing_result(s, ex)
        if status == PASS or retries <= 0 or reason in no_retry:
            break
        retries -= 1
    attach = reason in ("tool_call_in_content", "tool_call_in_reasoning")
    return s.result(status, reason, message, ex=ex, details=details,
                    hint=s.entry.get("hint") if attach else None,
                    warnings=details.get("warnings") or ())


def check_tools_parallel(
    ep: Endpoint, ev: Evidence | None, model: str, plan_entry: Mapping[str, Any] | None = None
) -> Result:
    """T-PARALLEL: the prompt for Paris and Tokyo with parallel_tool_calls: true. Pass: 2 or
    more calls, each valid, for the cities {paris, tokyo}."""
    s = _Session(ep, ev, "tools_parallel", plan_entry)
    if s.not_applicable:
        return s.not_applicable_result()
    body = _tools_body(model, PARALLEL_PROMPT, s.request, tool_choice="auto",
                       parallel_tool_calls=True)
    ex = s.post(body)
    if _parser_missing(ex):
        return _parser_missing_result(s, ex)
    found = _answer_failure(ex, ep)
    if found:
        return s.result(FAIL, *found, ex=ex)
    problem = _shape_problem(ex)
    if problem:
        return s.result(FAIL, "bad_response_shape", problem, ex=ex)
    calls = ex.message.get("tool_calls")
    calls = calls if isinstance(calls, list) else []
    details: dict[str, Any] = {"calls": len(calls), "finish_reason": ex.finish_reason}
    if not calls:
        status, reason, info = validate_tool_call(ex.message)
        return s.result(status, reason, _tool_message(s, reason, info), ex=ex,
                        details={**details, **info})
    if len(calls) < 2:
        return s.result(FAIL, "too_few_calls", "1 call with parallel_tool_calls true; Paris and "
                        "Tokyo need 2", ex=ex, details=details)
    cities = []
    warnings: list[str] = []
    for index, call in enumerate(calls):
        status, reason, info = validate_tool_call(
            {"tool_calls": [call]}, expect=EXPECT_PARALLEL
        )
        warnings += info.get("warnings") or []
        if status != PASS:
            text = f"call {index + 1} of {len(calls)}: {_tool_message(s, reason, info)}"
            return s.result(FAIL, reason, text, ex=ex, details={**details, **info},
                            warnings=warnings)
        # the same search the validation used, so "Paris, France" counts as paris
        city = str(nsc.parse_json(call["function"]["arguments"]).get("city", ""))
        found = EXPECT_PARALLEL["city"].search(city)
        cities.append(found.group(0).lower() if found else nsc.clip(city.lower(), 40))
    details["cities"] = cities
    if not set(cities) >= PARALLEL_CITIES:
        return s.result(FAIL, "argument_values_wrong", f"the calls are for {short(cities)}, not "
                        "Paris and Tokyo", ex=ex, details=details, warnings=warnings)
    return s.result(PASS, OK, f"{len(calls)} valid calls for Paris and Tokyo", ex=ex,
                    details=details, warnings=warnings)


def check_tools_none(
    ep: Endpoint, ev: Evidence | None, model: str, plan_entry: Mapping[str, Any] | None = None
) -> Result:
    """T-NONE: the tool with tool_choice "none" and a question that needs no tool. Pass: no
    tool_calls and an answer. Call markup in content is only a warning: the API guarantees only
    that tool_calls is empty."""
    s = _Session(ep, ev, "tools_none", plan_entry)
    if s.not_applicable:
        return s.not_applicable_result()
    body = _tools_body(model, NONE_PROMPT, s.request, tool_choice="none")
    ex = s.post(body)
    if _parser_missing(ex):
        return _parser_missing_result(s, ex)
    found = _answer_failure(ex, ep)
    if found:
        return s.result(FAIL, *found, ex=ex)
    problem = _shape_problem(ex)
    if problem:
        return s.result(FAIL, "bad_response_shape", problem, ex=ex)
    calls = ex.message.get("tool_calls")
    _, answer, where = split_reasoning(ex.message)
    details = {"finish_reason": ex.finish_reason, "where": where, "answer_chars": len(answer)}
    if isinstance(calls, list) and calls:
        return s.result(FAIL, "tool_called_with_none", f"{len(calls)} tool call(s) although "
                        "tool_choice is none", ex=ex, details=details)
    if not answer:
        why = " (finish_reason length)" if ex.finish_reason == "length" else ""
        return s.result(FAIL, "empty_answer", f"no answer with tool_choice none{why}", ex=ex,
                        details=details)
    warnings = []
    marker = tool_markup(answer)
    if marker:
        warnings.append(f"Tool-call markup {marker} in content with tool_choice none; only "
                        "tool_calls is guaranteed empty.")
    return s.result(PASS, OK, "no tool call and an answer", ex=ex, details=details,
                    warnings=warnings)


# --- reasoning -----------------------------------------------------------------------------------


def _switch_label(switch: Mapping[str, Any] | None) -> str:
    if not switch or switch.get("method") in (None, "default"):
        return "the model's default"
    method, key, value = switch.get("method"), switch.get("key"), switch.get("value")
    if method == "kwargs":
        return f"chat_template_kwargs.{key}={short(value, 30)}"
    if method == "effort":
        return f"reasoning_effort={value}"
    return f"{method} {short(value, 40)}"


def _reasoning_failure(
    ex: _Exchange, ep: Endpoint, switch: Mapping[str, Any] | None, stream: bool
) -> tuple[str, str] | None:
    """The transport or HTTP reason of a reasoning exchange; a 4xx on a sent switch is
    switch_rejected."""
    found = _stream_failure(ex, ep) if stream else _answer_failure(ex, ep)
    if not found:
        if not stream:
            problem = _shape_problem(ex)
            if problem:
                return "bad_response_shape", problem
        return None
    sent = bool(switch) and switch.get("method") not in (None, "default")
    if found[0] == "http_error" and sent and 400 <= ex.res.status < 500 and ex.res.status != 404:
        return "switch_rejected", (
            f"HTTP {ex.res.status} on the documented switch {_switch_label(switch)}: "
            f"{error_text(ex.res)}"
        )
    return found


def _on_verdict(
    ex: _Exchange, ep: Endpoint, switch: Mapping[str, Any] | None, max_tokens: int
) -> tuple[str, str, str, dict[str, Any], str | None]:
    """(status, reason, message, details, retry): retry "budget" (finish_reason length) or
    "same" (no reasoning) asks for the one retry."""
    found = _reasoning_failure(ex, ep, switch, stream=False)
    if found:
        return FAIL, found[0], found[1], {}, None
    content = message_text(ex.message.get("content"))
    reasoning, answer, where = split_reasoning(ex.message)
    details = {"where": where, "reasoning_chars": len(reasoning), "answer_chars": len(answer),
               "finish_reason": ex.finish_reason, "max_tokens": max_tokens,
               "expected_answer_seen": REASONING_ANSWER in answer}
    tokens = control_tokens(content)
    if tokens:
        return FAIL, "control_tokens_in_content", (
            f"content holds control tokens {', '.join(tokens)}"
        ), details, None
    if ex.finish_reason == "length":
        return FAIL, "reasoning_not_finished", (
            f"finish_reason length at max_tokens {max_tokens}"
            + (": no answer after the reasoning" if not answer else "")
        ), details, "budget"
    if len(reasoning) < MIN_REASONING_CHARS:
        return FAIL, "reasoning_missing", (
            f"no reasoning in message.reasoning, reasoning_content or a think block with "
            f"{_switch_label(switch)}"
        ), details, "same"
    if not answer:
        return FAIL, "answer_missing", (
            f"reasoning ({len(reasoning)} characters) but no answer, finish_reason "
            f"{ex.finish_reason}"
        ), details, None
    if ex.finish_reason != "stop":
        return FAIL, "unexpected_finish_reason", (
            f"finish_reason is {short(ex.finish_reason)}, not stop"
        ), details, None
    return PASS, OK, f"reasoning in {where} ({len(reasoning)} characters) and an answer", \
        details, None


def check_reasoning_on(
    ep: Endpoint, ev: Evidence | None, model: str, plan_entry: Mapping[str, Any] | None = None
) -> Result:
    """R-ON: the two-step arithmetic prompt with the family's on switch, sampling and budget
    (default 4096). Pass: 8 or more characters of reasoning (a field or a think block),
    finish_reason stop and an answer outside the reasoning. One retry: the same request when no
    reasoning came (reasoning_missing), or twice the budget, at most 16384 (never less than the
    first budget), when finish_reason was length (reasoning_not_finished)."""
    s = _Session(ep, ev, "reasoning_on", plan_entry)
    if s.not_applicable:
        return s.not_applicable_result()
    switch = s.request.get("switch")
    max_tokens = int(s.request.get("max_tokens") or nf.DEFAULT_REASONING_TOKENS)
    sampling = s.request.get("sampling")
    body = chat_body(model, REASONING_PROMPT, max_tokens=max_tokens,
                     sampling={} if sampling is None else sampling, switch=switch)
    ex = s.post(body)
    status, reason, message, details, retry = _on_verdict(ex, ep, switch, max_tokens)
    if retry:
        if retry == "budget":
            max_tokens = max(max_tokens, min(max_tokens * 2, RETRY_TOKEN_CAP))
            body = dict(body, max_tokens=max_tokens)
        ex = s.post(body, role="retry")
        status, reason, message, details, _ = _on_verdict(ex, ep, switch, max_tokens)
        if status == FAIL:
            message = f"{message} (after one retry)"
    warnings = []
    hint = None
    where = details.get("where")
    expected = list(s.request.get("expect_where") or [])
    if status == PASS and where in ("inline", "inline_close_only"):
        warnings.append("The reasoning is inline in content: clients that read "
                        "message.reasoning see nothing; a reasoning parser moves it to a field.")
        hint = s.entry.get("hint")
    if status == PASS and expected and where not in expected:
        warnings.append(f"The reasoning came in {where}; the record expects "
                        f"{', '.join(expected)}.")
    return s.result(status, reason, message, ex=ex, details=details, hint=hint,
                    warnings=warnings)


def _answer_like(text: str) -> bool:
    """A short text that reads as the answer to OFF_PROMPT, not as reasoning."""
    return THINK_CLOSE not in text and len(text.split()) <= 12 and OFF_ANSWER in text.lower()


def _off_verdict(
    ex: _Exchange, ep: Endpoint, switch: Mapping[str, Any] | None, stream: bool, model: str
) -> tuple[str, str, str, dict[str, Any], list[str]]:
    found = _reasoning_failure(ex, ep, switch, stream)
    if found:
        return FAIL, found[0], found[1], {}, []
    message = ex.message or {}
    reasoning, answer, where = split_reasoning(message)
    details = {"where": where, "reasoning_chars": len(reasoning), "answer_chars": len(answer),
               "finish_reason": ex.finish_reason,
               "expected_answer_seen": OFF_ANSWER in answer.lower()}
    content = message_text(message.get("content"))
    if not answer and reasoning and where in ("reasoning", "reasoning_content") and \
            _answer_like(reasoning):
        return FAIL, "answer_in_reasoning_field", (
            f"content is empty and {where} holds the answer: the reasoning parser routed the "
            f"answer into the reasoning field{' (streaming)' if stream else ''}"
        ), details, []
    if reasoning:
        warnings = []
        text = f"reasoning ({len(reasoning)} characters, in {where}) with {_switch_label(switch)}"
        if NANO_CAVEAT.search(model):
            text += "; Llama-Nemotron Nano can think on harder prompts even with thinking off"
            warnings.append(f"Model card on thinking with thinking off: {NANO_CARD}")
        return FAIL, "reasoning_present", text, details, warnings
    if not answer:
        return FAIL, "empty_answer", "no reasoning and no answer" + (
            " (content is null)" if not content else ""
        ), details, []
    if ex.finish_reason != "stop":
        why = (": a one-word answer used the whole budget, so the model may be thinking in "
               "content" if ex.finish_reason == "length" else "")
        return FAIL, "unexpected_finish_reason", (
            f"finish_reason is {short(ex.finish_reason)}, not stop{why}"
        ), details, []
    return PASS, OK, "no reasoning and an answer" + (" (streamed)" if stream else ""), \
        details, []


def check_reasoning_off(
    ep: Endpoint,
    ev: Evidence | None,
    model: str,
    plan_entry: Mapping[str, Any] | None = None,
    stream: bool = False,
) -> Result:
    """R-OFF: a trivial question with the family's off switch, greedy sampling (or the
    family's) and max_tokens 256. Pass: finish_reason stop, no reasoning (absent, empty, or an
    empty think block) and an answer in content. stream=True (or request.stream) is the
    streaming twin, reasoning_off_stream, for templates that pre-fill the think block."""
    stream = bool(stream or ((plan_entry or {}).get("request") or {}).get("stream"))
    s = _Session(ep, ev, "reasoning_off_stream" if stream else "reasoning_off", plan_entry)
    if stream:  # also when given the reasoning_off entry: its own id and evidence files
        s.check = "reasoning_off_stream"
    if s.not_applicable:
        return s.not_applicable_result()
    switch = s.request.get("switch")
    body = chat_body(model, OFF_PROMPT, max_tokens=int(s.request.get("max_tokens")
                                                       or nf.OFF_TOKENS),
                     sampling=s.request.get("sampling"), switch=switch, stream=stream)
    ex = s.post(body)
    status, reason, message, details, warnings = _off_verdict(ex, ep, switch, stream, model)
    return s.result(status, reason, message, ex=ex, details=details, warnings=warnings)


def check_reasoning_off_stream(
    ep: Endpoint, ev: Evidence | None, model: str, plan_entry: Mapping[str, Any] | None = None
) -> Result:
    """The streaming twin of check_reasoning_off."""
    return check_reasoning_off(ep, ev, model, plan_entry, stream=True)


# --- json schema, image and concurrency ----------------------------------------------------------


def _json_object(text: str) -> Any:
    try:
        return nsc.parse_json(text.strip())
    except ValueError:  # not JSON, or nested too deeply
        return None


def check_json_schema(
    ep: Endpoint, ev: Evidence | None, model: str, plan_entry: Mapping[str, Any] | None = None
) -> Result:
    """response_format json_schema with a small schema (name, version). Pass: HTTP 200,
    finish_reason stop, and the answer outside any reasoning parses to an object that matches
    the schema. Other values than "NIM" and 2 are only a warning."""
    s = _Session(ep, ev, "json_schema", plan_entry)
    if s.not_applicable:
        return s.not_applicable_result()
    max_tokens = int(s.request.get("max_tokens") or nf.JSON_TOKENS)
    body = chat_body(model, JSON_PROMPT, max_tokens=max_tokens, switch=s.request.get("switch"),
                     response_format=copy.deepcopy(RESPONSE_FORMAT))
    ex = s.post(body)
    found = _answer_failure(ex, ep)
    if found:
        return s.result(FAIL, *found, ex=ex)
    problem = _shape_problem(ex)
    if problem:
        return s.result(FAIL, "bad_response_shape", problem, ex=ex)
    reasoning, answer, where = split_reasoning(ex.message)
    details = {"finish_reason": ex.finish_reason, "where": where, "answer_chars": len(answer)}
    if ex.finish_reason == "length":
        return s.result(FAIL, "truncated_json", f"finish_reason length at max_tokens "
                        f"{max_tokens}: the JSON was cut", ex=ex, details=details)
    if not answer and isinstance(_json_object(reasoning), dict):
        return s.result(FAIL, "json_only_in_reasoning", f"content is empty and the JSON is in "
                        f"{where}", ex=ex, details=details)
    value = _json_object(answer) if answer else None
    if value is None and answer.strip() != "null":
        text = "the answer is empty" if not answer else "the answer does not parse as JSON"
        return s.result(FAIL, "invalid_json", text, ex=ex, details=details)
    problems = validate_schema(value, JSON_SCHEMA)
    if problems:
        details["problems"] = problems[:10]
        return s.result(FAIL, "schema_mismatch", "the JSON does not match the schema: "
                        + nsc.clip("; ".join(problems), 200), ex=ex, details=details)
    if ex.finish_reason != "stop":
        return s.result(FAIL, "unexpected_finish_reason", f"finish_reason is "
                        f"{short(ex.finish_reason)}, not stop", ex=ex, details=details)
    warnings = []
    if not all(_same(value.get(key), want) for key, want in JSON_EXPECTED.items()):
        warnings.append(f"The values differ from the prompt: {short(value, 80)}.")
    details["value"] = {key: short(item, 40) for key, item in value.items()}
    return s.result(PASS, OK, "the answer is JSON that matches the schema", ex=ex,
                    details=details, warnings=warnings)


def _text_only_reply(s: _Session, ex: _Exchange) -> Result | None:
    """The plan entry's if_text_only_reply: when the image request gets the 400 the docs give
    for this model's text-only profiles (for example "At most 0 image(s) may be provided in one
    prompt."), the active profile takes text only and the check is not applicable; any other
    answer is None (judged as usual)."""
    plan = s.entry.get("if_text_only_reply")
    if not isinstance(plan, Mapping) or ex.res.status != 400:
        return None
    reply = " ".join(str(plan.get("reply") or "").split()).lower()
    message = error_text(ex.res)
    body = " ".join(ex.res.body[:4000].decode("utf-8", "replace").split()).lower()
    if not reply or (reply not in message.lower() and reply not in body):
        return None
    return s.result(NOT_APPLICABLE, str(plan.get("reason") or "text_only_profile"),
                    f"the active profile takes text only: HTTP 400 \"{message}\", the answer "
                    "the docs give for this model's text-only profiles", ex=ex,
                    details={"error": message})


def check_image(
    ep: Endpoint, ev: Evidence | None, model: str, plan_entry: Mapping[str, Any] | None = None
) -> Result:
    """One text part and one image_url part with an inline 64x64 red PNG; then the text-only
    twin (the same text, no image, max_tokens 1). Pass: HTTP 200, an answer, and more
    usage.prompt_tokens with the image than without (equal counts: image_ignored). An answer
    that names red is only recorded. The 400 the docs give for this model's text-only profiles
    (the plan entry's if_text_only_reply) is not_applicable text_only_profile."""
    s = _Session(ep, ev, "image", plan_entry)
    if s.not_applicable:
        return s.not_applicable_result()
    png = png_solid()
    switch = s.request.get("switch")
    text_part = {"type": "text", "text": IMAGE_PROMPT}
    image_part = {"type": "image_url", "image_url": {"url": data_url(png)}}
    max_tokens = int(s.request.get("max_tokens") or nf.IMAGE_TOKENS)
    body = chat_body(model, [text_part, image_part], max_tokens=max_tokens, switch=switch)
    ex = s.post(body)
    s.details = {"image_bytes": len(png), "image_sha256": hashlib.sha256(png).hexdigest()}
    found = _answer_failure(ex, ep)
    if found:
        return _text_only_reply(s, ex) or s.result(FAIL, *found, ex=ex)
    problem = _shape_problem(ex)
    if problem:
        return s.result(FAIL, "bad_response_shape", problem, ex=ex)
    _, answer, where = split_reasoning(ex.message)
    details = {"finish_reason": ex.finish_reason, "where": where,
               "names_red": "red" in answer.lower()}
    if not answer:
        why = " (finish_reason length)" if ex.finish_reason == "length" else ""
        return s.result(FAIL, "empty_answer", f"no answer to the image question{why}", ex=ex,
                        details=details)
    twin_body = chat_body(model, [dict(text_part)], max_tokens=1, switch=switch)
    twin = s.post(twin_body, role="text_only_twin", count=False)
    found = _answer_failure(twin, ep)
    if found:
        return s.result(FAIL, found[0], f"the text-only twin: {found[1]}", ex=ex,
                        details=details)
    with_image = (ex.usage or {}).get("prompt_tokens")
    without = (twin.usage or {}).get("prompt_tokens")
    details.update(prompt_tokens_with_image=with_image, prompt_tokens_text_only=without)
    if not all(isinstance(n, int) and not isinstance(n, bool) for n in (with_image, without)):
        return s.result(FAIL, "bad_response_shape", "usage.prompt_tokens is missing, so the "
                        "image cannot be told from no image", ex=ex, details=details)
    if with_image <= without:
        return s.result(FAIL, "image_ignored", f"{with_image} prompt tokens with the image and "
                        f"{without} without: the image was dropped", ex=ex, details=details)
    return s.result(PASS, OK, f"an answer, and {with_image - without} more prompt tokens with "
                    "the image", ex=ex, details=details)


def check_concurrency(
    ep: Endpoint,
    ev: Evidence | None,
    model: str,
    plan_entry: Mapping[str, Any] | None = None,
    n: int | None = None,
) -> Result:
    """N copies of the chat request (max_tokens 64, the family's off switch when there is one)
    sent at the same moment from N threads; no readiness probe while they run. Pass: all N
    answer 200 with text. Fail: rejected_503 (a queue limit), gateway_error (502, 504),
    rate_limited (429, never retried), empty_completion, answer_missing (reasoning, no answer,
    finish_reason stop; with length a warning), http_error. N is `n`, else
    request.concurrency, else 4; at most 8, and at most 4 on the hosted API, whose rate limits
    apply (concurrency_too_high: a load test is nim-benchmark's job)."""
    s = _Session(ep, ev, "concurrency", plan_entry)
    if s.not_applicable:
        return s.not_applicable_result()
    given = n if n is not None else s.request.get("concurrency")
    count = int(given) if given is not None else DEFAULT_CONCURRENCY
    if count > MAX_CONCURRENCY or (ep.hosted and count > HOSTED_CONCURRENCY):
        raise concurrency_too_high(count, ep.hosted)
    if count < 1:
        raise nsc.ScriptError(
            "bad_input",
            f"--concurrency {count}: send at least 1 request",
            f"Use --concurrency 1 to {MAX_CONCURRENCY} (default {DEFAULT_CONCURRENCY})",
            nsc.EXIT_INPUT,
        )
    max_tokens = int(s.request.get("max_tokens") or nf.CONCURRENCY_TOKENS)
    body = chat_body(model, CHAT_PROMPT, max_tokens=max_tokens, switch=s.request.get("switch"))
    exchanges: list[_Exchange | None] = [None] * count
    crashed: list[BaseException] = []
    barrier = threading.Barrier(count)

    def send(index: int) -> None:
        with contextlib.suppress(threading.BrokenBarrierError):
            barrier.wait(timeout=30)
        try:
            exchanges[index] = s.post(body, role=f"request {index + 1} of {count}",
                                      number=index + 1, count=False)
        except Exception as exc:  # raised again below, in the caller's thread
            crashed.append(exc)

    started = time.monotonic()
    threads = [threading.Thread(target=send, args=(index,), daemon=True)
               for index in range(count)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if crashed:  # the error a request thread hit, not a bare "None" exchange further on
        raise crashed[0]
    wall = round((time.monotonic() - started) * 1000, 1)
    s.attempts = 1
    s.evidence.sort(key=lambda path: int(re.sub(r"\D", "", path.rsplit("-", 1)[-1]) or 0))
    reasons = []
    for ex in exchanges:
        assert ex is not None
        reasons.append(_concurrency_reason(ex, ep))
    details = {
        "n": count,
        "statuses": [ex.res.status for ex in exchanges if ex is not None],
        "latencies_ms": [ex.res.elapsed_ms for ex in exchanges if ex is not None],
        "wall_ms": wall,
        "reasons": [reason for reason, _ in reasons],
    }
    warnings = []
    if ep.port_forward:
        warnings.append("Through kubectl port-forward every request reaches the same pod; the "
                        "Service's load balancing was not tested.")
    cut = sum(1 for ex, (reason, _) in zip(exchanges, reasons) if ex is not None and not reason
              and ex.finish_reason == "length" and not split_reasoning(ex.message)[1])
    if cut:
        warnings.append(f"{cut} of {count} answers have reasoning but no content, finish_reason "
                        f"length: the reasoning used the whole budget of {max_tokens} tokens.")
    failed = [(index, reason, text) for index, (reason, text) in enumerate(reasons) if reason]
    if failed:
        index, reason, text = failed[0]
        counts = ", ".join(f"{name} x{sum(1 for _, r, _ in failed if r == name)}"
                           for name in dict.fromkeys(r for _, r, _ in failed))
        hint = None
        if reason == "answer_missing":
            first = exchanges[index]
            assert first is not None
            found = _answer_missing(ep, *split_reasoning(first.message), first.finish_reason)
            hint = found[1] if found else None
        return s.result(FAIL, reason, f"{len(failed)} of {count} failed ({counts}); request "
                        f"{index + 1}: {text}", details=details, latency_ms=wall, hint=hint,
                        warnings=warnings)
    return s.result(PASS, OK, f"{count} of {count} answered 200 with text in {wall} ms",
                    details=details, latency_ms=wall, warnings=warnings)


def concurrency_too_high(count: int, hosted: bool = False) -> nsc.ScriptError:
    """concurrency_too_high (exit 2): more than MAX_CONCURRENCY requests at once, or more than
    HOSTED_CONCURRENCY on the hosted API."""
    if hosted:
        return nsc.ScriptError(
            "concurrency_too_high",
            f"--concurrency {count}: on the hosted API the concurrency check sends at most "
            f"{HOSTED_CONCURRENCY} requests at once, as its rate limits apply to every key",
            f"Use --concurrency {HOSTED_CONCURRENCY} or less on the hosted API (or leave it out); "
            "a load or latency test is nim-benchmark's job",
            nsc.EXIT_INPUT,
        )
    return nsc.ScriptError(
        "concurrency_too_high",
        f"--concurrency {count}: a smoke check sends 1 to {MAX_CONCURRENCY} requests at once",
        f"Use --concurrency {MAX_CONCURRENCY} or less; a load or latency test is nim-benchmark's "
        "job",
        nsc.EXIT_INPUT,
    )


def _concurrency_reason(ex: _Exchange, ep: Endpoint) -> tuple[str | None, str]:
    status = ex.res.status
    if status == 503:
        return "rejected_503", f"HTTP 503: {error_text(ex.res)} (a queue limit is configured)"
    if status in (502, 504):
        return "gateway_error", f"HTTP {status} from a proxy or gateway"
    found = _answer_failure(ex, ep)
    if found:
        return found
    problem = _shape_problem(ex)
    if problem:
        return "bad_response_shape", problem
    reasoning, answer, where = split_reasoning(ex.message)
    if not (reasoning or answer):
        return "empty_completion", "HTTP 200 with no text"
    missing = _answer_missing(ep, reasoning, answer, where, ex.finish_reason)
    if missing:
        return "answer_missing", missing[0]
    return None, ""


# --- dispatch ------------------------------------------------------------------------------------

CHECK_FUNCTIONS: dict[str, Callable[..., Result]] = {
    "chat": check_chat,
    "stream": check_stream,
    "tools": check_tools,
    "tools_stream": check_tools_stream,
    "tools_parallel": check_tools_parallel,
    "tools_none": check_tools_none,
    "reasoning_on": check_reasoning_on,
    "reasoning_off": check_reasoning_off,
    "reasoning_off_stream": check_reasoning_off_stream,
    "json_schema": check_json_schema,
    "image": check_image,
    "concurrency": check_concurrency,
}


def run_check(
    check: str,
    ep: Endpoint,
    ev: Evidence | None,
    model: str,
    plan_entry: Mapping[str, Any] | None = None,
    **options: Any,
) -> Result:
    """Run one POST check by id (CHECK_FUNCTIONS) with its plan entry. health and models have
    their own functions (check_health, check_models); options go to the check (n for
    concurrency)."""
    function = CHECK_FUNCTIONS.get(check)
    if function is None:
        raise ValueError(f"no POST check {check!r}: use check_health or check_models")
    return function(ep, ev, model, plan_entry, **options)


def main(argv: list[str] | None = None) -> None:
    parser = nsc.JsonArgumentParser(
        description="The checks and response parsers of nim-smoke-check; import it, do not run "
        "it.",
        epilog="Example: python3 scripts/nsc_checks.py --help",
    )
    parser.parse_args(argv)
    nsc.emit(True, "library", "done", {"note": "import this module; it has no command of its own",
                                       "checks": ["health", "models", *CHECK_FUNCTIONS]})


if __name__ == "__main__":
    main()
