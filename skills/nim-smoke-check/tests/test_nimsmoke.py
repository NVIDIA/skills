# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end tests of nimsmoke.py run, features and report against fake NIMs and fake tools."""

from __future__ import annotations

import json
import os
import re
import secrets
import signal
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import nimsmoke
import nsc_access as nsa
import nsc_checks as nc
import nsc_common as nsc
import nsc_features as nf
import nsc_testkit as kit
import pytest

LLAMA = "meta/llama-3.1-8b-instruct"
HOLD = 1.0  # seconds the fake holds each answer when requests must overlap
IMAGE = f"nvcr.io/nim/{LLAMA}:2.1.5"
TOOL_FLAGS = "--enable-auto-tool-choice --tool-call-parser llama3_json"
QWEN = "qwen/qwen3.5-122b-a10b"
NEMOTRON = "nvidia/nemotron-3-nano"
GPT_OSS = "openai/gpt-oss-20b"
UNKNOWN = "acme/custom-llm-7b"
RESULT_KEYS = {"id", "status", "reason", "message", "evidence", "sources", "hint"}
RUN_FACTS = {"url", "model", "out", "report", "results", "counts", "target", "access",
             "warnings", "handoff"}
PROMPTS = (nc.CHAT_PROMPT, nc.TOOL_PROMPT, nc.REASONING_PROMPT, nc.JSON_PROMPT,
           nc.OFF_PROMPT, nc.IMAGE_PROMPT)


def smoke(nim: kit.FakeNim, out: Path, *extra: str, env: dict[str, str] | None = None,
          ) -> tuple[int, dict[str, Any], str]:
    return kit.run_cli("nimsmoke.py", "run", "--url", nim.url, "--out", str(out), *extra,
                       env=env)


def by_id(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {result["id"]: result for result in payload["facts"]["results"]}


def statuses(payload: dict[str, Any]) -> dict[str, tuple[str, str]]:
    return {key: (r["status"], r["reason"]) for key, r in by_id(payload).items()}


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


# --- run: one model family at a time -------------------------------------------------------------


def test_run_llama_passes_tools_and_skips_reasoning_and_image(tmp_path: Path) -> None:
    with kit.FakeNim(LLAMA) as nim:
        code, payload, _ = smoke(nim, tmp_path / "out")
    assert (code, payload["ok"], payload["state"], payload["next"]) == (0, True, "passed",
                                                                        "report")
    facts = payload["facts"]
    assert RUN_FACTS <= set(facts)
    found = statuses(payload)
    assert list(found) == [c for c in nf.DEFAULT_CHECKS]  # in CHECKS order
    for check in ("health", "models", "chat", "stream", "tools", "json_schema", "concurrency"):
        assert found[check] == ("pass", "ok"), (check, found[check])
    for check in ("reasoning_on", "reasoning_off", "reasoning_off_stream"):
        assert found[check] == ("not_applicable", "not_a_reasoning_model")
    assert found["image"] == ("not_applicable", "text_only_model")
    assert facts["target"]["record"] == "llama-3.1-instruct"
    assert facts["target"]["how"] == "served_id" and facts["target"]["version"] == "2.1.5"
    assert facts["target"]["backend"] == "vllm" and facts["handoff"] is None
    assert facts["access"]["place"] == "url" and facts["access"]["auth"] == "none"
    assert facts["counts"] == {"pass": 7, "fail": 0, "not_applicable": 4, "total": 11}
    assert all(set(r) == RESULT_KEYS for r in facts["results"])
    assert all(r["sources"] for r in facts["results"]), "every result cites a source"


def test_run_llama_without_a_tool_parser_follows_the_record_per_generation(
    tmp_path: Path,
) -> None:
    """D5: the 2.x llama image does not say it turns the parser on (n/a, with a warning); the
    1.x docs say the model's own image does (fail parser_not_enabled), and the hint goes to
    nim-deploy."""
    with kit.FakeNim(LLAMA, tools="no_parser") as nim:
        code, payload, _ = smoke(nim, tmp_path / "two")
    assert code == 0, payload
    tools = by_id(payload)["tools"]
    assert (tools["status"], tools["reason"]) == ("not_applicable", "not_enabled_in_deployment")
    assert "--tool-call-parser llama3_json" in tools["hint"] and "nim-deploy" in tools["hint"]
    assert payload["facts"]["handoff"] is None

    with kit.FakeNim(LLAMA, tools="no_parser", version="1.15.0") as nim:
        code, payload, _ = smoke(nim, tmp_path / "one", "--image", f"nvcr.io/nim/{LLAMA}:1.15.0")
    assert (code, payload["state"], payload["next"]) == (4, "failed", "report")
    assert [e["code"] for e in payload["errors"]] == ["checks_failed"]
    assert "tools (parser_not_enabled)" in payload["errors"][0]["message"]
    tools = by_id(payload)["tools"]
    assert (tools["status"], tools["reason"]) == ("fail", "parser_not_enabled")
    assert "NIM_ENABLE_AUTO_TOOL_CHOICE" in tools["hint"]
    handoff = payload["facts"]["handoff"]
    assert handoff["skill"] == "nim-deploy"
    assert handoff["changes"] == [{"check": "tools", "reason": "parser_not_enabled",
                                   "hint": tools["hint"]}]
    assert payload["facts"]["target"]["generation"] == "1.x"
    assert payload["facts"]["target"]["how"] == "image"


@pytest.mark.parametrize(
    ("model", "image"),
    [
        (LLAMA, None),  # a URL run that does not know the image
        (LLAMA, "nvcr.io/nim/nvidia/llm-nim:1.15.0"),  # the multi-LLM NIM: tool calling is opt-in
        ("meta-llama/Llama-3.1-8B-Instruct", "nvcr.io/nim/nvidia/llm-nim:1.15.0"),
    ],
)
def test_run_on_1x_fails_a_missing_parser_only_in_the_models_own_image(
    tmp_path: Path, model: str, image: str | None
) -> None:
    """table-1: NIM 1.x enables tool calling by itself only in the model's own (LLM-specific)
    container. With the image unknown, or the multi-LLM NIM serving the same model, a missing
    parser is not applicable with a warning, never parser_not_enabled with a nim-deploy
    hand-off; a URL run is told to pass --image."""
    extra = ["--image", image] if image else []
    with kit.FakeNim(model, tools="no_parser", version="1.15.0") as nim:
        code, payload, _ = smoke(nim, tmp_path / "out", "--checks", "health,models,tools", *extra)
    assert (code, payload["state"]) == (0, "passed"), payload
    tools = by_id(payload)["tools"]
    assert (tools["status"], tools["reason"]) == ("not_applicable", "not_enabled_in_deployment")
    assert "opt-in" not in tools["message"] and "NIM_ENABLE_AUTO_TOOL_CHOICE" in tools["hint"]
    assert payload["facts"]["handoff"] is None
    warnings = payload["facts"]["warnings"]
    assert any(text.startswith("tools: ") and "own NIM container" in text for text in warnings)
    assert any("--image" in text for text in warnings) == (image is None)
    assert payload["facts"]["target"]["record"] == "llama-3.1-instruct"


def test_run_on_the_dgx_spark_variant_reports_tool_calling_as_opt_in(tmp_path: Path) -> None:
    """table-2: the Nemotron Nano 9B v2 DGX Spark image (1.0.0-variant) turns tool calling on
    only with TOOL_CALL_PARSER=1 and lists guided decoding as not supported (1.14.0 variants
    page): a missing parser is not applicable with that switch as the hint and no nim-deploy
    hand-off, and json_schema sends no request."""
    model = "nvidia/nvidia-nemotron-nano-9b-v2"
    image = "nvcr.io/nim/nvidia/nvidia-nemotron-nano-9b-v2-dgx-spark:1.0.0-variant"
    with kit.FakeNim(model, tools="no_parser", version="1.0.0-variant") as nim:
        code, payload, _ = smoke(nim, tmp_path / "out", "--image", image,
                                 "--checks", "health,models,tools,json_schema")
        posts = nim.posts()
    assert (code, payload["state"]) == (0, "passed"), payload
    found = statuses(payload)
    assert found["tools"] == ("not_applicable", "not_enabled_in_deployment")
    assert found["json_schema"] == ("not_applicable", "not_documented_for_model")
    hint = by_id(payload)["tools"]["hint"]
    assert "TOOL_CALL_PARSER=1" in hint and "by themselves" not in hint
    assert payload["facts"]["handoff"] is None
    target = payload["facts"]["target"]
    assert (target["record"], target["how"], target["generation"]) == (
        "nemotron-nano-9b-v2", "image", "1.x")
    assert posts and not any("response_format" in (s.body or {}) for s in posts)


def test_run_gpt_oss_reasons_always_so_reasoning_off_is_not_applicable(tmp_path: Path) -> None:
    with kit.FakeNim(GPT_OSS, reasoning="always") as nim:
        code, payload, _ = smoke(nim, tmp_path / "out")
    assert code == 0, payload
    found = statuses(payload)
    assert found["reasoning_on"] == ("pass", "ok")
    assert found["reasoning_off"] == ("not_applicable", "always_on")
    assert found["tools"] == ("pass", "ok")
    assert payload["facts"]["target"]["record"] == "gpt-oss"
    # the documented on switch (reasoning_effort) was sent; never a field off the allowlist
    assert any(isinstance(s.body, dict) and s.body.get("reasoning_effort") == "low"
               for s in nim.posts())
    assert nim.sent_fields() <= nc.REQUEST_FIELDS


@pytest.mark.parametrize("model", [QWEN, NEMOTRON])
def test_run_a_toggle_model_passes_on_and_off(tmp_path: Path, model: str) -> None:
    with kit.FakeNim(model, reasoning="kwargs", vlm=model == QWEN) as nim:
        code, payload, _ = smoke(nim, tmp_path / "out")
    assert code == 0, payload
    found = statuses(payload)
    assert found["reasoning_on"] == ("pass", "ok")
    assert found["reasoning_off"] == ("pass", "ok")
    kwargs = [s.body.get("chat_template_kwargs") for s in nim.posts() if isinstance(s.body, dict)]
    assert {"enable_thinking": True} in kwargs and {"enable_thinking": False} in kwargs


def test_run_a_toggle_model_whose_answer_lands_in_the_reasoning_field_fails(
    tmp_path: Path,
) -> None:
    with kit.FakeNim(NEMOTRON, reasoning="kwargs", off_bug="answer_in_reasoning") as nim:
        code, payload, _ = smoke(nim, tmp_path / "out")
    assert (code, payload["state"]) == (4, "failed")
    assert statuses(payload)["reasoning_off"] == ("fail", "answer_in_reasoning_field")
    # concurrency sends the off switch too: its answers come back with content null
    assert statuses(payload)["concurrency"] == ("fail", "answer_missing")
    assert payload["errors"][0]["code"] == "checks_failed"


def test_run_a_vlm_passes_the_image_check_and_fails_when_the_image_is_ignored(
    tmp_path: Path,
) -> None:
    with kit.FakeNim(QWEN, reasoning="kwargs", vlm=True) as nim:
        code, payload, _ = smoke(nim, tmp_path / "ok")
    assert code == 0 and statuses(payload)["image"] == ("pass", "ok")
    with kit.FakeNim(QWEN, reasoning="kwargs", vlm="image_ignored") as nim:
        code, payload, _ = smoke(nim, tmp_path / "ignored")
    assert code == 4 and statuses(payload)["image"] == ("fail", "image_ignored")
    assert payload["facts"]["target"]["record"] == "qwen3.5-vlm"


def test_run_an_unknown_model_runs_the_core_checks_and_documented_features(
    tmp_path: Path,
) -> None:
    with kit.FakeNim(UNKNOWN) as nim:
        code, payload, _ = smoke(nim, tmp_path / "plain")
        assert code == 0, payload
        found = statuses(payload)
        for check in ("tools", "reasoning_on", "reasoning_off", "image"):
            assert found[check] == ("not_applicable", "no_capability_record")
        assert found["chat"] == ("pass", "ok") and found["json_schema"] == ("pass", "ok")
        assert payload["facts"]["target"]["record"] is None
        assert any("No capability record" in text for text in payload["facts"]["warnings"])
        assert not any(isinstance(s.body, dict) and s.body.get("tools") for s in nim.posts())

        url = "https://huggingface.co/acme/custom-llm-7b"
        code, payload, _ = smoke(nim, tmp_path / "documented", "--documented", f"tools={url}")
    assert code == 0, payload
    tools = by_id(payload)["tools"]
    assert (tools["status"], tools["reason"]) == ("pass", "ok") and url in tools["sources"]


def test_run_sends_the_key_from_the_named_variable(tmp_path: Path) -> None:
    key = kit.fake_key()
    with kit.FakeNim(LLAMA, require_key=key) as nim:
        code, payload, _ = smoke(nim, tmp_path / "out", "--api-key-env", "SMOKE_KEY",
                                 env={"SMOKE_KEY": key})
        assert code == 0, payload
        assert payload["facts"]["access"]["auth"] == "bearer from $SMOKE_KEY"
        assert all(s.headers.get("authorization") == f"Bearer {key}" for s in nim.requests)

        code, payload, _ = smoke(nim, tmp_path / "nokey", "--model", LLAMA)
    assert (code, payload["state"]) == (4, "failed")
    assert statuses(payload)["chat"] == ("fail", "unauthorized")
    # docs-5: the gateway's 401 on /v1/health/live too is unauthorized, with the export advice
    assert statuses(payload)["health"] == ("fail", "unauthorized")
    assert "--api-key-env" in by_id(payload)["health"]["message"]
    # a 401 is not about the model: no fail points to the model's known issues
    assert statuses(payload)["tools"] == ("fail", "unauthorized")
    assert not any("known issue" in text for text in payload["facts"]["warnings"])
    assert not any("known issue" in r["message"] for r in payload["facts"]["results"])
    assert "Known issue" not in (tmp_path / "nokey" / "report.md").read_text(encoding="utf-8")


def test_run_refuses_a_named_key_variable_that_is_not_set(tmp_path: Path) -> None:
    """Bad input before any request. The name is not repeated: a key given in its place can
    pass the name check, and the error would then print it."""
    with kit.FakeNim(LLAMA) as nim:
        code, payload, stderr = smoke(nim, tmp_path / "out", "--api-key-env", "SMOKE_UNSET_KEY")
        assert nim.requests == []
    assert (code, payload["errors"][0]["code"]) == (2, "bad_input")
    assert "--api-key-env" in payload["errors"][0]["message"]
    assert "SMOKE_UNSET_KEY" not in json.dumps(payload) + stderr


HOSTED = "https://integrate.api.nvidia.com/v1"


def test_a_hosted_run_without_a_key_is_refused_before_any_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
    no_network: None,
) -> None:
    """docs-5: the hosted API answers every chat request without a key with 401. With
    NVIDIA_API_KEY unset the run is bad input before any request, and the fix asks to export
    the key (never to leave the key out); a dry run, or health and models alone, need none."""
    monkeypatch.delenv(nimsmoke.HOSTED_KEY_ENV, raising=False)
    with pytest.raises(SystemExit) as info:
        nimsmoke.main(["run", "--url", HOSTED, "--model", GPT_OSS, "--out",
                       str(tmp_path / "out")])
    payload = json.loads(capsys.readouterr().out)
    assert (info.value.code, payload["state"], payload["next"]) == (2, "input", "ask_user")
    error = payload["errors"][0]
    assert error["code"] == "bad_input" and nimsmoke.HOSTED_KEY_ENV in error["message"]
    assert "export the key" in error["fix"] and "leave out" not in error["fix"]
    assert not (tmp_path / "out").exists()
    for extra, checks in ((["--dry-run"], nf.DEFAULT_CHECKS), ([], ("health", "models"))):
        args = nimsmoke.build_parser().parse_args(["run", "--url", HOSTED, *extra])
        assert nimsmoke.read_key(args, True, checks) == (None, "none")
    monkeypatch.setenv(nimsmoke.HOSTED_KEY_ENV, kit.fake_key())
    args = nimsmoke.build_parser().parse_args(["run", "--url", HOSTED])
    assert nimsmoke.read_key(args, True, nf.DEFAULT_CHECKS)[1] == "bearer from $NVIDIA_API_KEY"


def test_an_unset_key_variable_for_the_hosted_api_is_never_left_out(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
    no_network: None,
) -> None:
    """docs-5: for the hosted API the fix of an unset --api-key-env variable asks to export it;
    leaving the flag out (as for an endpoint without a key) would fail every chat check."""
    monkeypatch.delenv("SMOKE_UNSET_KEY", raising=False)
    with pytest.raises(SystemExit) as info:
        nimsmoke.main(["run", "--url", HOSTED, "--api-key-env", "SMOKE_UNSET_KEY", "--out",
                       str(tmp_path / "out")])
    payload = json.loads(capsys.readouterr().out)
    assert (info.value.code, payload["errors"][0]["code"]) == (2, "bad_input")
    assert "export the key" in payload["errors"][0]["fix"]
    assert "leave out" not in payload["errors"][0]["fix"]
    with kit.FakeNim(LLAMA) as nim:
        code, payload, _ = smoke(nim, tmp_path / "local", "--api-key-env", "SMOKE_UNSET_KEY")
        assert nim.requests == []
    assert code == 2 and "leave out --api-key-env" in payload["errors"][0]["fix"]


def test_hosted_concurrency_stays_at_4(tmp_path: Path, no_network: None) -> None:
    """docs-12: the hosted API's rate limits apply to the concurrency check, so more than 4 at
    once is concurrency_too_high there (exit 2, before any request); up to 8 elsewhere. Every
    run names an --out under tmp_path, so validate never checks the home folder's cache."""
    out = ["--out", str(tmp_path / "out")]
    for value, code in (("5", "concurrency_too_high"), ("8", "concurrency_too_high"),
                        ("4", None), ("1", None)):
        args = nimsmoke.build_parser().parse_args(["run", "--url", HOSTED, "--concurrency", value,
                                                   *out])
        if code is None:
            assert nimsmoke.validate(args)["hosted"] is True
            continue
        with pytest.raises(nsc.ScriptError) as info:
            nimsmoke.validate(args)
        assert (info.value.code, info.value.exit_code) == (code, nsc.EXIT_INPUT), value
        assert "hosted API" in info.value.message and "nim-benchmark" in info.value.fix
    args = nimsmoke.build_parser().parse_args(["run", "--url", "http://127.0.0.1:9",
                                               "--concurrency", "8", *out])
    assert nimsmoke.validate(args)["hosted"] is False


def test_from_url_marks_only_the_hosted_api_as_hosted() -> None:
    """The hosted API has no health route (health n/a); only its host counts, never a look-alike
    or a flag."""
    hosted = nsa.from_url("https://integrate.api.nvidia.com/v1")
    assert (hosted.place, hosted.url) == ("hosted", "https://integrate.api.nvidia.com")
    assert nsa.from_url("https://INTEGRATE.api.nvidia.com.").place == "hosted"
    for url in ("https://integrate.api.nvidia.com.evil.example", "http://127.0.0.1:8000",
                "https://gateway.example.com/integrate.api.nvidia.com"):
        assert nsa.from_url(url).place == "url", url


# --- run: the plan, the exit codes and the files -------------------------------------------------


def test_dry_run_sends_no_post_and_writes_nothing(tmp_path: Path) -> None:
    out = tmp_path / "out"
    with kit.FakeNim(QWEN, reasoning="kwargs", vlm=True) as nim:
        code, payload, _ = smoke(nim, out, "--dry-run")
        assert nim.posts() == []
        assert {s.path for s in nim.gets()} <= {"/v1/health/live", "/v1/health/ready",
                                                "/v1/models", "/v1/version", "/v1/metadata"}
    assert (code, payload["state"], payload["next"]) == (0, "planned", "run")
    plan = {entry["check"]: entry for entry in payload["facts"]["plan"]}
    assert list(plan) == list(nf.DEFAULT_CHECKS)
    assert plan["image"]["decision"] == "test" and plan["reasoning_off"]["decision"] == "test"
    assert plan["reasoning_off"]["request"]["switch"]["value"] is False
    assert payload["facts"]["out"] == str(out) and not out.exists()


def test_checks_skip_and_extended_pick_the_checks(tmp_path: Path) -> None:
    with kit.FakeNim(LLAMA) as nim:
        code, payload, _ = smoke(nim, tmp_path / "a", "--checks", "chat,tools", "--extended")
        assert code == 0, payload
        assert list(by_id(payload)) == ["chat", "tools", "tools_stream", "tools_parallel",
                                        "tools_none"]
        assert nim.gets("/v1/models"), "models always runs: it picks the model"
        assert not nim.gets("/v1/health/ready")
        code, payload, _ = smoke(nim, tmp_path / "b", "--skip", "concurrency,stream",
                                 "--concurrency", "2")
    assert code == 0 and "concurrency" not in by_id(payload) and "stream" not in by_id(payload)


def test_concurrency_sends_n_requests_at_once(tmp_path: Path) -> None:
    """All 6 are in flight together. The fake holds each answer HOLD seconds, so the 6 starts
    may spread over a busy runner's scheduling delays; more than 6 at once would be a bug."""
    with kit.FakeNim(LLAMA, delay=HOLD) as nim:
        code, payload, _ = smoke(nim, tmp_path / "out", "--checks", "concurrency",
                                 "--concurrency", "6")
    assert code == 0, payload
    assert nim.peak == 6
    assert len(by_id(payload)["concurrency"]["evidence"]) == 6


def test_a_model_that_is_not_served_fails_and_no_other_model_is_tested(tmp_path: Path) -> None:
    """--model names the model to test: when the endpoint serves another one, models fails with
    model_not_served and no POST check runs, rather than testing the other model."""
    with kit.FakeNim(LLAMA) as nim:
        code, payload, _ = smoke(nim, tmp_path / "out", "--model", "meta/llama-3.3-70b-instruct")
        assert nim.posts() == []
    assert (code, payload["state"], payload["next"]) == (4, "failed", "report")
    assert "models (model_not_served)" in payload["errors"][0]["message"]
    models = by_id(payload)["models"]
    assert (models["status"], models["reason"]) == ("fail", "model_not_served")
    assert LLAMA in models["message"] and payload["facts"]["model"] is None
    assert list(by_id(payload)) == ["health", "models"]
    assert any(text.startswith("No POST check ran") for text in payload["facts"]["warnings"])


def test_model_ambiguous_is_exit_3_and_names_the_served_models(tmp_path: Path) -> None:
    with kit.FakeNim(LLAMA, extra_models=("meta/llama-3.3-70b-instruct",)) as nim:
        code, payload, _ = smoke(nim, tmp_path / "out")
        assert nim.posts() == []
    assert (code, payload["state"], payload["next"]) == (3, "precondition", "ask_user")
    assert payload["errors"][0]["code"] == "model_ambiguous"
    # the fix tells the agent to test only the ids the user picks, and none when they do not know
    fix = payload["errors"][0]["fix"]
    assert "--model ID for each id they pick" in fix, fix
    assert fix.endswith("if they do not know, test none"), fix
    assert set(payload["facts"]["served_models"]) == {LLAMA, "meta/llama-3.3-70b-instruct"}
    assert not (tmp_path / "out").exists()


def test_a_closed_port_is_endpoint_unreachable_exit_4(tmp_path: Path) -> None:
    code, payload, _ = kit.run_cli("nimsmoke.py", "run", "--url",
                                   f"http://127.0.0.1:{free_port()}", "--out",
                                   str(tmp_path / "out"))
    assert (code, payload["state"], payload["next"]) == (4, "unreachable", "fix_errors")
    assert payload["errors"][0]["code"] == "endpoint_unreachable"


def test_a_slow_first_chat_request_stops_the_run_with_a_timeout_fix(tmp_path: Path) -> None:
    """docs-2: the NIM answered the GETs but not the first chat request in time. The run stops
    after that request, as before, but with first_request_timeout (state timeout), not
    endpoint_unreachable, and the fix offers a larger --timeout instead of the URL and port."""
    with kit.FakeNim(LLAMA, delay=3.0) as nim:
        code, payload, _ = smoke(nim, tmp_path / "out", "--timeout", "1")
        posts = len(nim.posts())
    assert (code, payload["state"], payload["next"]) == (4, "timeout", "fix_errors"), payload
    error = payload["errors"][0]
    assert error["code"] == "first_request_timeout"
    assert "reachable but slow" in error["fix"] and "--timeout 300" in error["fix"]
    assert "requests-exceed-the-client-timeout" in error["fix"]
    assert posts == 1, "the run stops after the first chat request"
    assert list(by_id(payload)) == ["health", "models", "chat"]
    assert by_id(payload)["chat"]["reason"] == "timeout"
    doc = json.loads((tmp_path / "out" / "results.json").read_text(encoding="utf-8"))
    assert doc["state"] == "timeout"


def test_a_given_timeout_raises_the_budget_of_every_chat_request(tmp_path: Path) -> None:
    """docs-2: --timeout can raise the per-request budget (it was only a cap, so a slow NIM
    could never be given more than 63.2 s for a 64-token chat); without it, the formula."""
    url = "http://127.0.0.1:9"
    for given, budget in (("300", 300.0), (None, 63.2)):
        argv = ["run", "--url", url, "--out", str(tmp_path / "out")]
        argv += ["--timeout", given] if given else []
        args = nimsmoke.build_parser().parse_args(argv)
        run = nimsmoke.Run(args)
        run.inputs = nimsmoke.validate(args)
        run.access = nsa.from_url(url)
        endpoint = run.endpoint(None)
        assert nc.request_timeout(endpoint, 64) == pytest.approx(budget), given
        assert nc.request_timeout(endpoint) == nc.GET_TIMEOUT


def test_no_answer_to_the_first_post_stops_the_run_as_endpoint_unreachable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A connection failure on the first chat request stops the run with endpoint_unreachable;
    its fix is the URL and the port, never the timeout."""
    calls: list[str] = []

    def no_answer(check: str, *args: Any, **options: Any) -> nc.Result:
        calls.append(check)
        return nc.Result(check, nc.FAIL, "endpoint_unreachable", "no answer: refused",
                         attempts=1)

    monkeypatch.setattr(nimsmoke.nc, "run_check", no_answer)
    with kit.FakeNim(LLAMA) as nim, pytest.raises(SystemExit) as info:
        nimsmoke.main(["run", "--url", nim.url, "--out", str(tmp_path / "out")])
    payload = json.loads(capsys.readouterr().out)
    assert (info.value.code, payload["state"], payload["next"]) == (4, "unreachable",
                                                                    "fix_errors")
    error = payload["errors"][0]
    assert error["code"] == "endpoint_unreachable" and "--timeout" not in error["fix"]
    assert calls == ["chat"], "the run stops after the first chat request"
    assert (tmp_path / "out" / "results.json").is_file()


@pytest.mark.parametrize(
    ("knobs", "reason"),
    [
        ({"health": "loading"}, "not_ready"),  # /v1/health/ready answers 503 all along
        ({"routes": {"/v1/health/live": (500, {"detail": "engine down"})}}, "not_live"),
    ],
)
def test_wait_ready_that_ends_with_the_nim_not_ready_runs_no_other_check(
    tmp_path: Path, knobs: dict[str, Any], reason: str
) -> None:
    """--wait-ready asks for a ready NIM before the checks. When the wait ends and the NIM
    still answers that it is not ready (or not live), no other check runs: each would fail on
    that, not on a feature. health keeps its result and evidence; the rest is in not_run."""
    out = tmp_path / "out"
    with kit.FakeNim(LLAMA, **knobs) as nim:
        code, payload, _ = smoke(nim, out, "--wait-ready", "1")
        assert nim.posts() == [] and nim.gets("/v1/models") == []
    assert (code, payload["state"], payload["next"]) == (4, "failed", "report"), payload
    assert payload["errors"][0]["code"] == "checks_failed"
    assert statuses(payload) == {"health": ("fail", reason)}
    doc = json.loads((out / "results.json").read_text(encoding="utf-8"))
    assert doc["not_run"] == [check for check in nf.DEFAULT_CHECKS if check != "health"]
    assert doc["results"][0]["evidence"]
    assert all((out / path).is_file() for path in doc["results"][0]["evidence"])
    assert any(text.startswith("--wait-ready 1 ended") and "did not run: models, chat" in text
               for text in payload["facts"]["warnings"])


@pytest.mark.parametrize(
    ("extra", "code_word"),
    [
        (["--concurrency", "9"], "concurrency_too_high"),
        (["--concurrency", "0"], "bad_input"),
        (["--checks", "chat,warp_drive"], "bad_input"),
        (["--skip", "nope"], "bad_input"),
        (["--documented", "tools=http://insecure.example.com/x"], "bad_input"),
        (["--documented", "speed=https://huggingface.co/x"], "bad_input"),
        (["--reasoning-on", "telepathy"], "bad_input"),
        (["--family", "no-such-family"], "unknown_family"),
        (["--timeout", "0"], "bad_input"),
        (["--context", "lab"], "bad_input"),
        (["--image", "bad image ref"], "bad_input"),
        (["--wait-ready", "5", "--skip", "health"], "bad_input"),
    ],
)
def test_bad_input_is_exit_2_before_any_request(
    tmp_path: Path, extra: list[str], code_word: str
) -> None:
    with kit.FakeNim(LLAMA) as nim:
        code, payload, _ = smoke(nim, tmp_path / "out", *extra)
        assert nim.requests == []
    assert (code, payload["state"], payload["next"]) == (2, "input", "ask_user")
    assert payload["errors"][0]["code"] == code_word
    if code_word == "concurrency_too_high":
        assert "nim-benchmark" in payload["errors"][0]["fix"]


PASSWORD_SLOT = "{password}"  # a random value replaces it at run time: no password in the repo


@pytest.mark.parametrize(
    "argv",
    [
        ["run"],
        ["run", "--url", "ftp://127.0.0.1:8000"],
        ["run", "--url", f"http://smokeuser:{PASSWORD_SLOT}@127.0.0.1:8000"],
        ["run", "--url", "http://127.0.0.1:8000", "--docker", "llama"],
        ["run", "--k8s", "nimservice/llama"],
        ["run", "--url", "http://127.0.0.1:8000", "--warp"],
        ["features"],
        ["nonsense"],
    ],
)
def test_bad_flags_are_bad_input(argv: list[str]) -> None:
    """Bad flags are bad_input; a URL with a password is refused without showing it."""
    password = secrets.token_hex(8)
    code, payload, stderr = kit.run_cli("nimsmoke.py",
                                        *(arg.replace(PASSWORD_SLOT, password) for arg in argv))
    assert (code, payload["errors"][0]["code"]) == (2, "bad_input")
    assert password not in json.dumps(payload) + stderr


def test_results_json_report_md_and_evidence(tmp_path: Path) -> None:
    out = tmp_path / "out"
    with kit.FakeNim(NEMOTRON, reasoning="kwargs", off_bug="answer_in_reasoning") as nim:
        code, payload, _ = smoke(nim, out)
    assert code == 4
    doc = json.loads((out / "results.json").read_text(encoding="utf-8"))
    assert {"schema", "state", "created", "run_id", "url", "model", "out", "report", "target",
            "access", "counts", "results", "warnings", "handoff", "table"} <= set(doc)
    assert doc["schema"] == "nim-smoke-check/results/v1" and doc["state"] == "failed"
    assert doc["model"] == NEMOTRON and doc["table"]["reviewed"] == doc["target"]["reviewed"]
    for result in doc["results"]:
        assert result["status"] in nc.STATUSES
        assert result["reason"] in nc.REASONS or result["reason"] in nc.NA_MESSAGES
        for path in result["evidence"]:
            assert (out / path).is_file(), path
    assert payload["facts"]["results"] == [
        {key: r[key] for key in RESULT_KEYS} for r in doc["results"]
    ]
    evidence = sorted(p.name for p in (out / "evidence").iterdir())
    assert "probe_version-1.json" in evidence and "chat-1.json" in evidence
    report = (out / "report.md").read_text(encoding="utf-8")
    assert report.startswith(f"# nim-smoke-check: {NEMOTRON} at {nim.url}")
    assert "| check | result | reason | evidence |" in report
    assert "## Not applicable: why" in report and "## Failed: what the docs say to change" in report
    assert "`reasoning_off` | fail | `answer_in_reasoning_field`" in report
    assert "## Sources" in report and "https://" in report
    text = report + json.dumps(payload)
    for prompt in PROMPTS:
        assert prompt not in text, prompt
    assert kit.reasoning_text(nc.REASONING_PROMPT) not in text
    assert "Four pens" not in text


def test_check_warnings_reach_stdout(tmp_path: Path) -> None:
    """facts.warnings carries every check's warnings as "<check>: <text>" (the stdout results
    keep only their summary keys), for run and report alike."""
    out = tmp_path / "out"
    with kit.FakeNim(LLAMA, health="bare") as nim:
        code, payload, _ = smoke(nim, out, "--checks", "health,chat")
    assert code == 0, payload
    assert all(set(r) == RESULT_KEYS for r in payload["facts"]["results"])
    found = [text for text in payload["facts"]["warnings"] if text.startswith("health: ")]
    assert found and "not that the model is ready" in found[0]
    doc = json.loads((out / "results.json").read_text(encoding="utf-8"))
    assert f"health: {doc['results'][0]['warnings'][0]}" in payload["facts"]["warnings"]
    code, again, _ = kit.run_cli("nimsmoke.py", "report", str(out))
    assert again["facts"]["warnings"] == payload["facts"]["warnings"]


def test_report_reprints_a_finished_run(tmp_path: Path) -> None:
    out = tmp_path / "out"
    with kit.FakeNim(LLAMA, tools="no_parser", version="1.15.0") as nim:
        code, run, _ = smoke(nim, out, "--image", f"nvcr.io/nim/{LLAMA}:1.15.0")
    assert code == 4
    code, again, _ = kit.run_cli("nimsmoke.py", "report", str(out))
    assert (code, again["ok"], again["state"], again["next"]) == (0, True, "failed", "done")
    for key in RUN_FACTS:
        assert again["facts"][key] == run["facts"][key], key
    assert again["facts"]["run_state"] == "failed"

    (tmp_path / "empty").mkdir()
    code, missing, _ = kit.run_cli("nimsmoke.py", "report", str(tmp_path / "empty"))
    assert (code, missing["errors"][0]["code"]) == (2, "bad_input")
    (tmp_path / "other").mkdir()
    (tmp_path / "other" / "results.json").write_text('{"schema": "x"}', encoding="utf-8")
    code, other, _ = kit.run_cli("nimsmoke.py", "report", str(tmp_path / "other"))
    assert (code, other["errors"][0]["code"]) == (2, "bad_input")


@pytest.mark.parametrize(
    "fields",
    [
        {"results": [None]},
        {"results": ["chat"]},
        {"results": [{"id": "chat", "status": "pass", "warnings": 5}]},
        {"results": [{"id": "chat", "status": "fail", "reason": "x", "known_issues": 5}]},
        {"results": [], "warnings": 5},
    ],
)
def test_report_refuses_results_that_are_not_check_results(
    tmp_path: Path, fields: dict[str, Any]
) -> None:
    """A results.json with this skill's schema whose results (or warnings) cannot be read as a
    run's is bad input (exit 2), never an internal error; a partial report with only some of
    the keys of a result still reads back."""
    folder = tmp_path / "run"
    folder.mkdir()
    doc = {"schema": nimsmoke.RESULTS_SCHEMA, "state": "interrupted", **fields}
    (folder / "results.json").write_text(json.dumps(doc), encoding="utf-8")
    code, payload, stderr = kit.run_cli("nimsmoke.py", "report", str(folder))
    assert (code, payload["errors"][0]["code"]) == (2, "bad_input"), payload
    assert "Traceback" not in stderr
    doc["results"], doc["warnings"] = [{"id": "health", "status": "pass"}], None
    (folder / "results.json").write_text(json.dumps(doc), encoding="utf-8")
    code, payload, _ = kit.run_cli("nimsmoke.py", "report", str(folder))
    assert (code, payload["state"]) == (0, "interrupted"), payload
    assert payload["facts"]["counts"]["pass"] == 1
    assert payload["facts"]["results"][0]["id"] == "health"


def test_run_refuses_an_out_folder_that_is_not_empty(tmp_path: Path) -> None:
    """security-7: a rerun with the same --out is refused by the input checks, before any
    request, not after every check has run (which would discard all the evidence); a dry run
    is refused the same way, as the run it previews would be."""
    out = tmp_path / "out"
    out.mkdir(mode=0o700)
    (out / "old.txt").write_text("x", encoding="utf-8")
    with kit.FakeNim(LLAMA) as nim:
        for extra in ([], ["--dry-run"]):
            code, payload, _ = smoke(nim, out, "--checks", "chat", *extra)
            assert (code, payload["state"]) == (2, "input"), extra
            assert payload["errors"][0]["code"] == "bad_input"
            assert "is not empty" in payload["errors"][0]["message"]
            assert nim.requests == [], "refused before any request"
    assert sorted(p.name for p in out.iterdir()) == ["old.txt"]
    afile = tmp_path / "a-file"
    afile.write_text("x", encoding="utf-8")
    with kit.FakeNim(LLAMA) as nim:
        for target in (afile, afile / "below"):
            code, payload, _ = smoke(nim, target, "--checks", "chat")
            assert (code, payload["errors"][0]["code"]) == (2, "bad_input"), target
            assert "is not a folder" in payload["errors"][0]["message"]
        assert nim.requests == []


@pytest.mark.skipif(not hasattr(os, "getuid"), reason="POSIX owners only")
def test_run_checks_the_default_out_folder_before_any_request(tmp_path: Path) -> None:
    """security-7: with no --out, the default folder is named after the model, known only
    later, so its parent ~/.cache/nim-smoke-check is checked first: a cache that other users
    can change is refused (out_not_private) before any request, not after the checks."""
    home = tmp_path / "home"
    cache = home / ".cache"
    cache.mkdir(parents=True)
    home.chmod(0o700)
    cache.chmod(0o777)
    with kit.FakeNim(LLAMA) as nim:
        for extra in ([], ["--dry-run"]):
            code, payload, _ = kit.run_cli("nimsmoke.py", "run", "--url", nim.url, "--checks",
                                           "chat", *extra, env={"HOME": str(home)})
            assert (code, payload["errors"][0]["code"]) == (2, "out_not_private"), extra
            assert nim.requests == [], "refused before any request"
    assert list(cache.iterdir()) == []


def test_the_default_out_folder_is_private_under_the_cache(tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir(mode=0o700)
    with kit.FakeNim(LLAMA) as nim:
        code, payload, _ = kit.run_cli("nimsmoke.py", "run", "--url", nim.url, "--checks",
                                       "chat", env={"HOME": str(home)})
    assert code == 0, payload
    out = Path(payload["facts"]["out"])
    assert out.parent == home / ".cache" / "nim-smoke-check"
    assert out.name.endswith("-meta-llama-3.1-8b-instruct")
    assert out.stat().st_mode & 0o777 == 0o700
    assert (out / "results.json").stat().st_mode & 0o777 == 0o600


def test_runs_of_one_model_in_the_same_second_get_their_own_default_folders(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The default folder name changes once a second: another run of the same model that
    started in the same second takes the next free name (-2, then -3) instead of failing to
    save its evidence after every check ran."""
    monkeypatch.setattr(nsc, "CACHE_FOLDER", str(tmp_path / "cache"))
    now = datetime(2026, 11, 12, 9, 30, 0, tzinfo=timezone.utc)
    first = nimsmoke.claim_default_out(LLAMA, now)
    (first / "results.json").write_text("{}", encoding="utf-8")
    second = nimsmoke.claim_default_out(LLAMA, now)
    third = nimsmoke.claim_default_out(LLAMA, now)
    assert first.name == "20261112T093000Z-meta-llama-3.1-8b-instruct"
    assert [second.name, third.name] == [f"{first.name}-2", f"{first.name}-3"]
    for folder in (first, second, third):
        assert folder.parent == (tmp_path / "cache").resolve()
        assert folder.stat().st_mode & 0o777 == 0o700
    assert list(second.iterdir()) == [] and list(third.iterdir()) == []


# --- run: Docker and Kubernetes ------------------------------------------------------------------


def test_run_on_docker_reports_a_parser_dropped_by_an_override(tmp_path: Path) -> None:
    """Tier B: the image sets the tool parser by default; the container's -e replaced it."""
    with kit.FakeNim(LLAMA, tools="no_parser") as nim:
        record = kit.docker_container("llama", IMAGE, host_port=nim.port,
                                      env={"NIM_PASSTHROUGH_ARGS": "--max-num-seqs 8"})
        tools = kit.make_fake_tools(tmp_path, docker_fixtures={
            "container inspect llama": record,
            "context inspect": kit.docker_context(),
            "image inspect": kit.docker_image(IMAGE, env={"NIM_PASSTHROUGH_ARGS": TOOL_FLAGS}),
        })
        code, payload, _ = kit.run_cli(
            "nimsmoke.py", "run", "--docker", "llama", "--out", str(tmp_path / "out"),
            env=tools.env({"DOCKER_HOST": ""}),
        )
    assert (code, payload["state"]) == (4, "failed"), payload
    tools.assert_read_only()
    tool = by_id(payload)["tools"]
    assert (tool["status"], tool["reason"]) == ("fail", "parser_dropped_by_override")
    assert "replaces the image's default" in tool["hint"]
    facts = payload["facts"]
    assert facts["target"]["how"] == "image" and facts["target"]["resource_id"] == f"nim/{LLAMA}"
    assert facts["target"]["tier_b"]["tools_dropped_by_override"] is True
    assert facts["access"]["place"] == "docker" and facts["access"]["passthrough"] == "set"
    assert "--max-num-seqs" not in json.dumps(facts)  # the raw value is never shown
    assert facts["handoff"]["skill"] == "nim-deploy"


@pytest.mark.parametrize(
    ("tag_env", "container_env"),
    [
        ({"NIM_PASSTHROUGH_ARGS": TOOL_FLAGS}, {}),
        ({"NIM_PASSTHROUGH_ARGS": "--max-model-len 8192 " + TOOL_FLAGS},
         {"NIM_PASSTHROUGH_ARGS": "--max-model-len 8192"}),
    ],
)
def test_run_on_docker_reads_the_default_of_the_image_the_container_runs(
    tmp_path: Path, tag_env: dict[str, str], container_env: dict[str, str]
) -> None:
    """access-1: after the container started, a pull moved its tag (:latest) to an image whose
    default turns the tool parser on. Tier B reads the image the container runs, by its ID, so
    no false parser_not_enabled or parser_dropped_by_override; the tag is never inspected."""
    tag = f"nvcr.io/nim/{LLAMA}:latest"
    running = "sha256:" + "a" * 64
    with kit.FakeNim(LLAMA, tools="no_parser") as nim:
        record = kit.docker_container("llama", tag, host_port=nim.port, env=container_env)
        record[0]["Image"] = running
        tools = kit.make_fake_tools(tmp_path, docker_fixtures={
            "container inspect llama": record,
            "context inspect": kit.docker_context(),
            f"image inspect {running}": kit.docker_image(tag, env=container_env),
            f"image inspect {tag}": kit.docker_image(tag, env=tag_env),
        })
        code, payload, _ = kit.run_cli(
            "nimsmoke.py", "run", "--docker", "llama", "--checks", "health,models,tools",
            "--out", str(tmp_path / "out"), env=tools.env({"DOCKER_HOST": ""}),
        )
    assert (code, payload["state"]) == (0, "passed"), payload
    tools.assert_read_only()
    tool = by_id(payload)["tools"]
    assert (tool["status"], tool["reason"]) == ("not_applicable", "not_enabled_in_deployment")
    tier_b = payload["facts"]["target"]["tier_b"]
    assert tier_b["image_default_known"] is True and tier_b["tools_on_by_default"] is None
    assert tier_b["tools_dropped_by_override"] is False and payload["facts"]["handoff"] is None
    inspected = [argv[-1] for argv in tools.argvs("docker") if argv[:2] == ["image", "inspect"]]
    assert inspected == [running]


@pytest.mark.parametrize("local_has_the_digest", [False, True])
def test_run_on_kubernetes_reads_a_local_image_default_only_by_the_pods_digest(
    tmp_path: Path, local_has_the_digest: bool
) -> None:
    """access-1: for --k8s the image default is read on this machine's daemon only when it holds
    the very image the pod runs (the pod's imageID digest). A local copy of the tag can be
    another image: then tier B makes no claim, rather than a false parser_dropped_by_override."""
    repo = f"nvcr.io/nim/{LLAMA}"
    digest = "sha256:" + "b" * 64
    pod = kit.k8s_pod("llama-0", "nim", labels={"app": "llama"}, image=f"{repo}:latest",
                      env={"NIM_PASSTHROUGH_ARGS": "--max-num-seqs 8"})
    pod["status"]["containerStatuses"][0]["imageID"] = f"docker-pullable://{repo}@{digest}"
    pinned = (kit.docker_image(f"{repo}:latest", env={"NIM_PASSTHROUGH_ARGS": TOOL_FLAGS})
              if local_has_the_digest
              else kit.ToolReply(stderr=f"Error: No such image: {repo}@{digest}\n", exit=1))
    with kit.FakeNim(LLAMA, tools="no_parser") as nim:
        tools = kit.make_fake_tools(tmp_path, kubectl_fixtures={
            "auth can-i create pods": kit.can_i(True),
            "get nimservice llama": kit.k8s_nimservice("llama", "nim", repo, "latest"),
            "get service llama": kit.k8s_service("llama", "nim", selector={"app": "llama"}),
            "get pods -l app=llama": kit.k8s_list(pod),
            "port-forward pod/llama-0": kit.port_forward_to(nim),
        }, docker_fixtures={
            f"image inspect {repo}@{digest}": pinned,
            f"image inspect {repo}:latest": kit.docker_image(
                f"{repo}:latest", env={"NIM_PASSTHROUGH_ARGS": TOOL_FLAGS}),
        })
        code, payload, _ = kit.run_cli(
            "nimsmoke.py", "run", "--k8s", "nimservice/llama", "--context", "lab",
            "--namespace", "nim", "--checks", "health,models,tools", "--out",
            str(tmp_path / "out"), env=tools.env({"DOCKER_HOST": ""}),
        )
    tools.assert_read_only()
    inspected = [argv[-1] for argv in tools.argvs("docker") if argv[:2] == ["image", "inspect"]]
    assert inspected == [f"{repo}@{digest}"]
    tool = by_id(payload)["tools"]
    tier_b = payload["facts"]["target"]["tier_b"]
    if local_has_the_digest:  # the very image: its default had the parser, the pod's value not
        assert (code, tool["status"], tool["reason"]) == (4, "fail", "parser_dropped_by_override")
        assert tier_b["tools_dropped_by_override"] is True
    else:
        assert (code, tool["status"], tool["reason"]) == (0, "not_applicable",
                                                          "not_enabled_in_deployment")
        assert tier_b["image_default_known"] is False and payload["facts"]["handoff"] is None


def test_run_on_docker_keeps_the_containers_parser_flags_when_the_image_cannot_be_read(
    tmp_path: Path,
) -> None:
    """Tier B: the container's NIM_PASSTHROUGH_ARGS turns on the reasoning parser of Kimi K3.1
    Pro (opt-in for this model), and the image default cannot be read. The NIM runs with the
    container's value either way, so reasoning_on runs; nothing is claimed about the image
    default or whether -e replaced it."""
    kimi = "moonshotai/kimi-k3.1-pro"
    image = f"nvcr.io/nim/{kimi}:2.1.5"
    with kit.FakeNim(kimi, reasoning="always") as nim:
        record = kit.docker_container("kimi", image, host_port=nim.port,
                                      env={"NIM_PASSTHROUGH_ARGS": "--reasoning-parser kimi_k3"})
        tools = kit.make_fake_tools(tmp_path, docker_fixtures={
            "container inspect kimi": record,
            "context inspect": kit.docker_context(),
            "image inspect": kit.ToolReply(stderr="Error: No such image\n", exit=1),
        })
        code, payload, _ = kit.run_cli(
            "nimsmoke.py", "run", "--docker", "kimi", "--checks", "health,models,reasoning_on",
            "--out", str(tmp_path / "out"), env=tools.env({"DOCKER_HOST": ""}),
        )
    assert (code, payload["state"]) == (0, "passed"), payload
    tools.assert_read_only()
    assert any(argv[:2] == ["image", "inspect"] for argv in tools.argvs("docker"))
    assert payload["facts"]["target"]["record"] == "kimi-k3.1-pro"
    assert statuses(payload)["reasoning_on"] == ("pass", "ok")
    tier_b = payload["facts"]["target"]["tier_b"]
    assert tier_b["image_default_known"] is False
    assert tier_b["set_in_deployment"] is False and tier_b["overridden"] is False


def test_tier_b_with_an_unread_passthrough_takes_only_the_arguments() -> None:
    """A pod's NIM_PASSTHROUGH_ARGS that comes from a ConfigMap or a Secret (valueFrom), or that
    envFrom may set, is unknown: tier B takes no image default and no deployment value, only the
    container arguments, and the warning names both ways the value is hidden."""
    access = nsa.Access("kubernetes", "http://127.0.0.1:8000", None, "port-forward",
                        effective_passthrough=nsa.UNKNOWN, config_known=True,
                        effective_args=["--reasoning-parser", "kimi_k3"])
    warnings: list[str] = []
    signals = nimsmoke.tier_b_signals(access, warnings)
    assert signals == nf.tier_b(None, None, access.effective_args)
    assert signals["reasoning_parser_in_deployment"] == "kimi_k3"
    assert signals["set_in_deployment"] is False and signals["image_default_known"] is False
    assert len(warnings) == 1
    assert "(valueFrom)" in warnings[0] and "(envFrom)" in warnings[0]


def test_run_on_docker_with_an_unpublished_health_port(tmp_path: Path) -> None:
    """NIM_HEALTH_PORT moved the health routes to a port the container does not publish: health
    is not applicable with its own reason and nothing is probed; with --wait-ready the run
    stops (exit 3) before any request, as it cannot wait there."""
    with kit.FakeNim(LLAMA, health="loading") as nim:
        record = kit.docker_container("llama", IMAGE, host_port=nim.port,
                                      env={"NIM_HEALTH_PORT": "8001"})
        tools = kit.make_fake_tools(tmp_path, docker_fixtures={
            "container inspect llama": record,
            "context inspect": kit.docker_context(),
            "image inspect": kit.docker_image(IMAGE),
        })
        env = tools.env({"DOCKER_HOST": ""})
        code, payload, _ = kit.run_cli(
            "nimsmoke.py", "run", "--docker", "llama", "--checks", "health,models,chat",
            "--out", str(tmp_path / "a"), env=env,
        )
        assert nim.gets("/v1/health/ready") == [] and nim.gets("/v1/health/live") == []
        assert code == 0, payload
        health = by_id(payload)["health"]
        assert (health["status"], health["reason"]) == ("not_applicable", "health_port_not_reached")
        assert "NIM_HEALTH_PORT=8001" in health["message"]
        report = (tmp_path / "a" / "report.md").read_text(encoding="utf-8")
        assert nc.NA_MESSAGES["health_port_not_reached"] in report
        nim.reset()
        code, payload, _ = kit.run_cli(
            "nimsmoke.py", "run", "--docker", "llama", "--wait-ready", "30", "--checks",
            "health,models,chat", "--out", str(tmp_path / "b"), env=env,
        )
        assert nim.requests == []
    tools.assert_read_only()
    assert (code, payload["state"], payload["next"]) == (3, "precondition", "ask_user")
    assert payload["errors"][0]["code"] == "health_port_not_reached"
    assert "--health-url" in payload["errors"][0]["fix"]
    assert not (tmp_path / "b").exists()


def test_run_on_kubernetes_owns_and_stops_its_port_forward(tmp_path: Path) -> None:
    pod = kit.k8s_pod("llama-0", "nim", labels={"app": "llama"}, image=IMAGE)
    with kit.FakeNim(LLAMA) as nim:
        tools = kit.make_fake_tools(tmp_path, kubectl_fixtures={
            "auth can-i create pods": kit.can_i(True),
            "get nimservice llama": kit.k8s_nimservice("llama", "nim",
                                                       f"nvcr.io/nim/{LLAMA}", "2.1.5"),
            "get service llama": kit.k8s_service("llama", "nim", selector={"app": "llama"}),
            "get pods -l app=llama": kit.k8s_list(pod),
            "port-forward pod/llama-0": kit.port_forward_to(nim),
        })
        code, payload, _ = kit.run_cli(
            "nimsmoke.py", "run", "--k8s", "nimservice/llama", "--context", "lab",
            "--namespace", "nim", "--checks", "chat,tools", "--out", str(tmp_path / "out"),
            env=tools.env({"DOCKER_HOST": ""}),
        )
        assert nim.posts(), "the checks reached the NIM through the forward"
    assert code == 0, payload
    tools.assert_read_only()
    tools.assert_scoped("lab", "nim")
    access = payload["facts"]["access"]
    assert (access["place"], access["how"]) == ("kubernetes", "port-forward")
    assert access["port_forward"]["pod"] == "llama-0"
    assert access["port_forward"]["command"].startswith("kubectl port-forward pod/llama-0 ")
    assert payload["facts"]["target"]["record"] == "llama-3.1-instruct"
    for call in tools.calls("kubectl"):
        if call["verb"] == "port-forward":
            with pytest.raises(ProcessLookupError):
                os.kill(call["pid"], 0)


K8S_CHECKS = "health,models,chat,stream,json_schema,concurrency"


def operator_fixtures(forward: Any, **extra: Any) -> dict[str, Any]:
    """kubectl fixtures of a NIMService llama in namespace nim, with its pod llama-0."""
    pod = kit.k8s_pod("llama-0", "nim", labels={"app": "llama"}, image=IMAGE)
    return {
        "auth can-i create pods": kit.can_i(True),
        "get nimservice llama": kit.k8s_nimservice("llama", "nim", f"nvcr.io/nim/{LLAMA}",
                                                   "2.1.5"),
        "get service llama": kit.k8s_service("llama", "nim", selector={"app": "llama"}),
        "get pods -l app=llama": kit.k8s_list(pod),
        "port-forward pod/llama-0": forward,
        **extra,
    }


def k8s_smoke(tools: kit.FakeTools, out: Path) -> tuple[int, dict[str, Any], str]:
    return kit.run_cli("nimsmoke.py", "run", "--k8s", "nimservice/llama", "--context", "lab",
                       "--namespace", "nim", "--checks", K8S_CHECKS, "--out", str(out),
                       env=tools.env({"DOCKER_HOST": ""}))


def test_a_forward_lost_during_a_check_is_restarted_before_the_next(tmp_path: Path) -> None:
    """access-2: kubectl closes the connection when the forward breaks and exits a moment later.
    The run waits for that exit, so the next check restarts the forward: only the check that
    was running fails, it says why, and every check after it passes."""
    with kit.FakeNim(LLAMA) as nim:
        # the first forward breaks on about the stream request (the connections before it:
        # the start's and each check's connect probe, health, models and the GET probes)
        tools = kit.make_fake_tools(tmp_path, operator_fixtures(
            kit.port_forward_to(nim, lost_after=11)))
        code, payload, _ = k8s_smoke(tools, tmp_path / "out")
    tools.assert_read_only()
    results = payload["facts"]["results"]
    failed = [(index, r) for index, r in enumerate(results) if r["status"] == "fail"]
    assert len(failed) == 1, results
    index, lost = failed[0]
    assert lost["reason"] in ("endpoint_unreachable", "connection_lost"), lost
    later = [r for r in results[index + 1:] if r["id"] in ("json_schema", "concurrency")]
    assert later and all(r["status"] == "pass" for r in later), results
    assert payload["facts"]["access"]["port_forward"]["restarts"] == 1
    assert (code, payload["errors"][0]["code"]) == (4, "checks_failed")
    assert any(text.startswith(f"{lost['id']}: The port-forward stopped during this check")
               for text in payload["facts"]["warnings"])
    assert len([c for c in tools.calls("kubectl") if c["verb"] == "port-forward"]) >= 2


def test_a_forward_lost_on_the_first_chat_request_is_restarted_before_the_next(
    tmp_path: Path,
) -> None:
    """access-2: the forward breaks on the first chat request and kubectl exits. That is the
    forward, not the NIM, so the run does not stop as on a first request with no answer: the
    next check restarts the forward, and every check after it passes."""
    with kit.FakeNim(LLAMA) as nim:
        # 9 connections come first (the start's and each check's connect probe, health, models
        # and the GET probes); the first forward drops the 10th, the chat request
        tools = kit.make_fake_tools(tmp_path, operator_fixtures(
            kit.port_forward_to(nim, lost_after=9)))
        code, payload, _ = k8s_smoke(tools, tmp_path / "out")
    tools.assert_read_only()
    found = statuses(payload)
    assert found["chat"] == ("fail", "endpoint_unreachable"), payload
    for check in ("stream", "json_schema", "concurrency"):
        assert found[check] == ("pass", "ok"), (check, found)
    assert (code, payload["state"], payload["errors"][0]["code"]) == (4, "failed",
                                                                      "checks_failed")
    assert payload["facts"]["access"]["port_forward"]["restarts"] == 1
    assert any(text.startswith("chat: The port-forward stopped during this check")
               for text in payload["facts"]["warnings"])


def test_a_forward_that_cannot_restart_keeps_the_results_that_ran(tmp_path: Path) -> None:
    """access-4: the forward is lost mid-run and the pod is gone, so the restart fails
    (pod_not_ready). The checks that ran keep their results.json, report.md and evidence, the
    rest are listed as not run, and the error points at them; report DIR reads them back."""
    out = tmp_path / "out"
    gone = kit.ToolReply(stderr='Error from server (NotFound): pods "llama-0" not found\n', exit=1)
    with kit.FakeNim(LLAMA) as nim:
        tools = kit.make_fake_tools(tmp_path, operator_fixtures(
            gone, **{"port-forward pod/llama-0 :8000": kit.port_forward_to(nim, lost_after=11)}))
        code, payload, _ = k8s_smoke(tools, out)
    tools.assert_read_only()
    assert (code, payload["state"], payload["next"]) == (3, "precondition", "ask_user")
    error = payload["errors"][0]
    assert error["code"] == "pod_not_ready" and str(out.resolve()) in error["message"]
    facts = payload["facts"]
    assert facts["out"] == str(out.resolve())
    assert facts["report"] == str(out.resolve() / "report.md")
    doc = json.loads((out / "results.json").read_text(encoding="utf-8"))
    ran = [r["id"] for r in doc["results"]]
    assert ran[:3] == ["health", "models", "chat"] and doc["state"] == "precondition"
    assert doc["not_run"] and not set(doc["not_run"]) & set(ran)
    assert set(ran) | set(doc["not_run"]) == set(K8S_CHECKS.split(","))
    assert any("did not run: " + ", ".join(doc["not_run"]) in text for text in facts["warnings"])
    for result in doc["results"]:
        for path in result["evidence"]:
            assert (out / path).is_file(), path
    assert (out / "report.md").is_file()
    code, again, _ = kit.run_cli("nimsmoke.py", "report", str(out))
    assert (code, again["facts"]["run_state"]) == (0, "precondition")


class StubForward:
    """A port-forward whose ensure() raises `error` from its `fail_at`-th call on."""

    def __init__(self, error: Exception, fail_at: int) -> None:
        self.error, self.fail_at, self.calls, self.stopped = error, fail_at, 0, False

    def ensure(self) -> int:
        self.calls += 1
        if self.calls >= self.fail_at:
            raise self.error
        return 1

    def settle(self, timeout: float | None = None) -> bool:
        return False

    def stop(self) -> None:
        self.stopped = True

    def facts(self) -> dict[str, Any]:
        return {"pod": "llama-0", "restarts": 1, "local_port": 1}


def test_a_forward_lost_twice_keeps_the_results_that_ran(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """access-4: port_forward_lost (the second loss) mid-run: exit 4 with that error, after
    results.json, report.md and the evidence of the checks that ran are written."""
    lost = nsc.ScriptError("port_forward_lost", "the port-forward to pod llama-0 stopped again",
                           nsa.LOST_FIX, nsc.EXIT_EXTERNAL)
    out = tmp_path / "out"
    with kit.FakeNim(LLAMA) as nim:
        stub = StubForward(lost, fail_at=4)  # health, models, chat; then stream's ensure fails
        monkeypatch.setattr(nimsmoke, "open_access",
                            lambda args: (nsa.from_url(nim.url), stub))
        with pytest.raises(SystemExit) as info:
            nimsmoke.main(["run", "--url", nim.url, "--checks", "health,models,chat,stream,"
                           "json_schema", "--out", str(out)])
        posts = len(nim.posts())
    payload = json.loads(capsys.readouterr().out)
    assert (info.value.code, payload["state"]) == (4, "unreachable") and stub.stopped
    assert payload["errors"][0]["code"] == "port_forward_lost" and posts == 1
    doc = json.loads((out / "results.json").read_text(encoding="utf-8"))
    assert [r["id"] for r in doc["results"]] == ["health", "models", "chat"]
    assert doc["not_run"] == ["stream", "json_schema"]
    assert (out / "evidence" / "chat-1.json").is_file() and (out / "report.md").is_file()


@pytest.mark.parametrize("kubectl_exited", [False, True])
def test_no_answer_to_the_first_post_through_a_forward_stops_while_kubectl_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
    kubectl_exited: bool,
) -> None:
    """No answer to the first chat request through the port-forward stops the run, as on a URL,
    while kubectl still runs: the NIM did not answer. When kubectl exited, the forward was
    lost: the next check restarts it (ensure) and the run goes on."""
    calls: list[str] = []

    def no_answer_first(check: str, *args: Any, **options: Any) -> nc.Result:
        calls.append(check)
        if len(calls) == 1:
            return nc.Result(check, nc.FAIL, "endpoint_unreachable", "no answer: reset",
                             attempts=1)
        return nc.Result(check, nc.PASS, nc.OK, "HTTP 200", attempts=1)

    with kit.FakeNim(LLAMA) as nim:
        stub = StubForward(AssertionError("the forward is never lost twice here"), fail_at=99)
        monkeypatch.setattr(stub, "settle", lambda timeout=None: kubectl_exited)
        monkeypatch.setattr(nimsmoke, "open_access",
                            lambda args: (nsa.from_url(nim.url), stub))
        monkeypatch.setattr(nimsmoke.nc, "run_check", no_answer_first)
        checks = "health,models,chat,stream,json_schema"
        with pytest.raises(SystemExit) as info:
            nimsmoke.main(["run", "--url", nim.url, "--checks", checks, "--out",
                           str(tmp_path / "out")])
    payload = json.loads(capsys.readouterr().out)
    if kubectl_exited:
        assert calls == ["chat", "stream", "json_schema"]
        assert stub.calls == 5, "an ensure before each check: health, models and three POSTs"
        assert (info.value.code, payload["state"], payload["errors"][0]["code"]) == (
            4, "failed", "checks_failed")
    else:
        assert calls == ["chat"], "the run stops after the first chat request"
        assert (info.value.code, payload["state"], payload["errors"][0]["code"]) == (
            4, "unreachable", "endpoint_unreachable")


@pytest.mark.parametrize(
    ("reason", "state", "code"),
    [
        ("endpoint_unreachable", "unreachable", "endpoint_unreachable"),
        ("timeout", "timeout", "first_request_timeout"),
    ],
)
def test_no_answer_to_the_first_post_after_a_forward_restart_stops_the_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
    reason: str, state: str, code: str,
) -> None:
    """The chat request is answered, then kubectl exits during stream. The next check restarts
    the forward and its POST counts as the first: no answer to it while kubectl runs stops the
    run there, so the checks after it do not each wait out their full budget."""
    calls: list[str] = []

    def chat_then_no_answer(check: str, *args: Any, **options: Any) -> nc.Result:
        calls.append(check)
        if check == "chat":
            return nc.Result(check, nc.PASS, nc.OK, "HTTP 200", attempts=1)
        return nc.Result(check, nc.FAIL, reason, "no answer", attempts=1)

    exited = iter([True])  # kubectl exits during stream; the restarted one keeps running
    with kit.FakeNim(LLAMA) as nim:
        stub = StubForward(AssertionError("the forward is never lost twice here"), fail_at=99)
        monkeypatch.setattr(stub, "settle", lambda timeout=None: next(exited, False))
        monkeypatch.setattr(nimsmoke, "open_access",
                            lambda args: (nsa.from_url(nim.url), stub))
        monkeypatch.setattr(nimsmoke.nc, "run_check", chat_then_no_answer)
        checks = "health,models,chat,stream,json_schema,concurrency"
        with pytest.raises(SystemExit) as info:
            nimsmoke.main(["run", "--url", nim.url, "--checks", checks, "--out",
                           str(tmp_path / "out")])
    payload = json.loads(capsys.readouterr().out)
    assert calls == ["chat", "stream", "json_schema"], "the first POST after the restart stops"
    assert stub.calls == 5, "an ensure before each check: health, models and three POSTs"
    assert (info.value.code, payload["state"], payload["errors"][0]["code"]) == (4, state, code)
    assert any(text.startswith("stream: The port-forward stopped during this check")
               for text in payload["facts"]["warnings"])


class RestartingForward(StubForward):
    """A port-forward that kubectl restarts on its `restart_at`-th ensure() call."""

    def __init__(self, restart_at: int) -> None:
        super().__init__(AssertionError("the forward is never lost twice here"), fail_at=99)
        self.restart_at, self.restarts = restart_at, 0

    def ensure(self) -> int:
        port = super().ensure()
        if self.calls == self.restart_at:
            self.restarts += 1
        return port


@pytest.mark.parametrize(
    ("reason", "state", "code"),
    [
        ("endpoint_unreachable", "unreachable", "endpoint_unreachable"),
        ("timeout", "timeout", "first_request_timeout"),
    ],
)
def test_no_answer_after_a_restart_between_checks_stops_the_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
    reason: str, state: str, code: str,
) -> None:
    """The chat request is answered, then kubectl exits between checks, so the next check's
    ensure() restarts the forward before it sends anything. That check's POST counts as the
    first: no answer to it while kubectl runs stops the run there, so the checks after it do
    not each wait out their full budget."""
    calls: list[str] = []

    def chat_then_no_answer(check: str, *args: Any, **options: Any) -> nc.Result:
        calls.append(check)
        if check == "chat":
            return nc.Result(check, nc.PASS, nc.OK, "HTTP 200", attempts=1)
        return nc.Result(check, nc.FAIL, reason, "no answer", attempts=1)

    with kit.FakeNim(LLAMA) as nim:
        stub = RestartingForward(restart_at=4)  # ensure() before health, models, chat, stream
        monkeypatch.setattr(nimsmoke, "open_access",
                            lambda args: (nsa.from_url(nim.url), stub))
        monkeypatch.setattr(nimsmoke.nc, "run_check", chat_then_no_answer)
        checks = "health,models,chat,stream,json_schema,concurrency"
        with pytest.raises(SystemExit) as info:
            nimsmoke.main(["run", "--url", nim.url, "--checks", checks, "--out",
                           str(tmp_path / "out")])
    payload = json.loads(capsys.readouterr().out)
    assert calls == ["chat", "stream"], "the first POST after the restart stops the run"
    assert (stub.restarts, stub.calls) == (1, 4)
    assert (info.value.code, payload["state"], payload["errors"][0]["code"]) == (4, state, code)


def test_wait_ready_on_kubernetes_waits_for_a_ready_pod(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """access-6: --wait-ready works on --k8s as on --docker: a pod still loading its model
    (Running, not Ready) is waited for, then the checks run (it stopped the run at once with
    pod_not_ready, the health wait never reached)."""
    monkeypatch.setattr(nsa, "POD_POLL", 0.05)
    loading = kit.k8s_pod("llama-0", "nim", labels={"app": "llama"}, image=IMAGE, ready=False)
    ready = kit.k8s_pod("llama-0", "nim", labels={"app": "llama"}, image=IMAGE)
    with kit.FakeNim(LLAMA) as nim:
        tools = kit.make_fake_tools(tmp_path, operator_fixtures(kit.port_forward_to(nim), **{
            "get pods -l app=llama": kit.in_turn(kit.k8s_list(loading), kit.k8s_list(loading),
                                                 kit.k8s_list(ready)),
        }))
        monkeypatch.setenv("PATH", str(tools.bin))
        monkeypatch.delenv("DOCKER_HOST", raising=False)
        with pytest.raises(SystemExit) as info:
            nimsmoke.main(["run", "--k8s", "nimservice/llama", "--context", "lab", "--namespace",
                           "nim", "--wait-ready", "30", "--checks", "health,models,chat",
                           "--out", str(tmp_path / "out")])
        assert nim.posts(), "the checks reached the NIM through the forward"
    payload = json.loads(capsys.readouterr().out)
    assert (info.value.code, payload["state"]) == (0, "passed"), payload
    assert statuses(payload)["health"] == ("pass", "ok")
    tools.assert_read_only()
    tools.assert_scoped("lab", "nim")
    reads = [argv[:2] for argv in tools.argvs("kubectl")].count(["get", "pods"])
    assert reads == 3


def test_an_unexpected_error_still_prints_one_object_and_keeps_the_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """docs-1: an error no check expected (a bug, or an answer no parser handles) ends the run
    with one JSON object (internal_error, exit 4), never a bare traceback; the checks that ran
    keep results.json, report.md and their evidence, and the rest are listed as not run."""
    real = nc.run_check

    def crash_on_stream(check: str, *args: Any, **options: Any) -> nc.Result:
        if check == "stream":
            raise RuntimeError("boom in a parser")
        return real(check, *args, **options)

    monkeypatch.setattr(nimsmoke.nc, "run_check", crash_on_stream)
    out = tmp_path / "out"
    with kit.FakeNim(LLAMA) as nim, pytest.raises(SystemExit) as info:
        nimsmoke.main(["run", "--url", nim.url, "--checks", "health,models,chat,stream,"
                       "json_schema", "--out", str(out)])
    printed = capsys.readouterr()
    payload = json.loads(printed.out)
    kit.assert_contract(payload)
    assert (info.value.code, payload["state"], payload["next"]) == (4, "internal_error",
                                                                    "ask_user")
    error = payload["errors"][0]
    assert error["code"] == "internal_error"
    assert "RuntimeError: boom in a parser" in error["message"]
    assert str(out.resolve()) in error["message"] and "bug" in error["fix"]
    assert "Traceback" in printed.err
    doc = json.loads((out / "results.json").read_text(encoding="utf-8"))
    assert doc["state"] == "internal_error"
    assert [r["id"] for r in doc["results"]] == ["health", "models", "chat"]
    assert doc["not_run"] == ["stream", "json_schema"]
    assert (out / "evidence" / "chat-1.json").is_file() and (out / "report.md").is_file()
    assert any("did not run: stream, json_schema" in text for text in payload["facts"]["warnings"])
    code, again, _ = kit.run_cli("nimsmoke.py", "report", str(out))
    assert (code, again["facts"]["run_state"]) == (0, "internal_error")


def test_evidence_that_a_failed_write_left_out_is_written_by_the_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Writing the evidence fails part way (a full disk, say) after the first files were
    written: the run stops on that error, and the retry that keeps the results writes the files
    that were not written yet, so every evidence file results.json names exists."""
    real = nsc.write_json_file
    failed: list[str] = []

    def full_disk_once(path: Path, obj: Any) -> None:
        if Path(path).name == "chat-1.json" and not failed:
            failed.append(str(path))
            raise OSError(28, "No space left on device")
        real(path, obj)

    monkeypatch.setattr(nsc, "write_json_file", full_disk_once)
    out = tmp_path / "out"
    with kit.FakeNim(LLAMA) as nim, pytest.raises(SystemExit) as info:
        nimsmoke.main(["run", "--url", nim.url, "--checks", "health,models,chat,stream",
                       "--out", str(out)])
    payload = json.loads(capsys.readouterr().out)
    assert failed, "the write of chat-1.json failed once"
    assert (info.value.code, payload["errors"][0]["code"]) == (4, "internal_error")
    assert str(out.resolve()) in payload["errors"][0]["message"]
    doc = json.loads((out / "results.json").read_text(encoding="utf-8"))
    named = [path for result in doc["results"] for path in result["evidence"]]
    assert "evidence/health-1.json" in named and "evidence/chat-1.json" in named
    assert [path for path in named if not (out / path).is_file()] == []


def test_features_and_report_turn_an_unexpected_error_into_one_object(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def broken(*args: Any, **kwargs: Any) -> Any:
        raise KeyError("lost")

    monkeypatch.setattr(nimsmoke.nf, "decide", broken)
    with pytest.raises(SystemExit) as info:
        nimsmoke.main(["features", "--model", LLAMA])
    payload = json.loads(capsys.readouterr().out)
    kit.assert_contract(payload)
    assert (info.value.code, payload["errors"][0]["code"]) == (4, "internal_error")
    # a results.json nested past the recursion limit is bad input, not a traceback
    (tmp_path / "deep").mkdir()
    (tmp_path / "deep" / "results.json").write_text("[" * 100_000 + "]" * 100_000,
                                                    encoding="utf-8")
    code, payload, stderr = kit.run_cli("nimsmoke.py", "report", str(tmp_path / "deep"))
    assert (code, payload["errors"][0]["code"]) == (2, "bad_input") and "Traceback" not in stderr


def test_a_failed_checks_known_issues_reach_stdout(tmp_path: Path) -> None:
    """docs-3: a fail message points to the model's known issues; stdout carries them in
    facts.warnings (the results keep their summary keys), for run and report alike."""
    out = tmp_path / "out"
    with kit.FakeNim(GPT_OSS, reasoning="always", control_tokens=True) as nim:
        code, payload, _ = smoke(nim, out, "--checks", "chat")
    assert (code, payload["state"]) == (4, "failed"), payload
    chat = by_id(payload)["chat"]
    assert (chat["status"], chat["reason"]) == ("fail", "control_tokens_in_content")
    assert set(chat) == RESULT_KEYS and "see the warnings" in chat["message"]
    doc = json.loads((out / "results.json").read_text(encoding="utf-8"))
    issues = doc["results"][0]["known_issues"]
    assert issues
    report = (out / "report.md").read_text(encoding="utf-8")
    for url in issues:
        assert f"chat: a known issue of this model may explain the fail: {url}" in (
            payload["facts"]["warnings"])
        assert f"  - Known issue: {url}" in report
    code, again, _ = kit.run_cli("nimsmoke.py", "report", str(out))
    assert again["facts"]["warnings"] == payload["facts"]["warnings"]


def test_report_md_says_why_a_parser_is_not_applicable_and_gives_the_hint(
    tmp_path: Path,
) -> None:
    """docs-4: not_enabled_in_deployment covers an opt-in parser and a default the docs do not
    state; report.md gives this run's own reason (never "opt-in" for the unknown default) and
    the documented switch."""
    out = tmp_path / "out"
    with kit.FakeNim(LLAMA, tools="no_parser") as nim:
        code, payload, _ = smoke(nim, out, "--checks", "models,tools")
    assert code == 0, payload
    report = (out / "report.md").read_text(encoding="utf-8")
    line = next(text for text in report.splitlines() if text.startswith("- `tools`"))
    assert "not_enabled_in_deployment" in line and "opt-in" not in line
    assert "is not known" in line and by_id(payload)["tools"]["message"].rstrip(".") in line
    assert "Hint: Documented server switch" in line and "--tool-call-parser llama3_json" in line


REPORT_HEADINGS = {"## Not applicable: why", "## Failed: what the docs say to change",
                   "## Sources", "## Warnings"}
UNESCAPED_PIPE = re.compile(r"(?<!\\)\|")


def assert_report_structure(report: str) -> list[str]:
    """report.md has its own title and headings only, a table row per check with four cells,
    and no control, line-separator or bidi character; returns its lines."""
    lines = report.splitlines()
    assert lines[0].startswith("# nim-smoke-check: ") and lines[1] == ""
    assert {line for line in lines[1:] if line.lstrip().startswith("#")} <= REPORT_HEADINGS
    assert not any(
        (ord(char) < 0x20 and char != "\n") or char in "\x7f\x85\u2028\u2029\u202e"
        for char in report
    )
    rows = [line for line in lines if line.startswith("| ") and not line.startswith("| check")]
    assert rows and all(len(UNESCAPED_PIPE.findall(row)) == 5 for row in rows), rows
    return lines


def test_report_md_escapes_a_model_id_from_the_endpoint(tmp_path: Path) -> None:
    """A served model id with a line break, a heading, a link, a table pipe, a code span, a
    terminal escape and line and bidi separators adds nothing to report.md: it stays one
    escaped line of text. results.json and stdout keep the id as the endpoint sent it."""
    odd = "x/m\n# Injected [click](http://e.invalid) | `tick`\u2028\u202e\x1b[31m <b>"
    out = tmp_path / "out"
    with kit.FakeNim(odd) as nim:
        code, payload, _ = smoke(nim, out, "--checks", "health,models,chat")
    assert code == 0, payload
    assert payload["facts"]["model"] == odd
    report = (out / "report.md").read_text(encoding="utf-8")
    lines = assert_report_structure(report)
    assert lines[0].startswith(
        "# nim-smoke-check: x/m # Injected \\[click\\](http://e.invalid) \\| \\`tick\\`"
    )
    assert not re.search(r"(?<!\\)\[click|(?<!\\)<b", report) and "\\<b\\>" in report


def test_report_markdown_escapes_every_value_it_writes() -> None:
    odd = "a\n## Fake | [x](http://e.invalid) <img src=x> `c`"
    result = {"status": "fail", "reason": "http_error", "message": odd, "hint": odd,
              "sources": [odd], "known_issues": [], "warnings": [odd]}
    doc = {
        "model": odd, "url": "http://127.0.0.1:1", "created": odd, "state": "failed",
        "counts": {"pass": 0, "fail": 1}, "target": {"version": odd, "backend": odd},
        "access": {"place": "url", "how": "direct"}, "warnings": [odd, "# a heading"],
        "results": [
            {**result, "id": "chat",
             "evidence": ["evidence/chat-1.json", "evidence/x](http://e.invalid)"]},
            {**result, "id": "odd|id", "status": "not_applicable", "reason": "x`y",
             "evidence": []},
        ],
    }
    report = nimsmoke.report_markdown(doc)
    assert_report_structure(report)
    assert "[evidence/chat-1.json](evidence/chat-1.json)" in report  # our own link stays
    assert not re.search(r"(?<!\\)\[x|(?<!\\)<img|\[evidence/x", report)  # no link, no HTML
    assert "- \\# a heading" in report and "odd\\|id" in report and "x\\`y" in report


@pytest.mark.skipif(os.name != "posix", reason="POSIX signals")
def test_ctrl_c_still_prints_one_object_and_keeps_the_evidence(tmp_path: Path) -> None:
    """docs-6: Ctrl-C (SIGINT) during a run: exit 130 with one JSON object (interrupted, ask
    the user), and the checks that ran keep results.json and their evidence."""
    out = tmp_path / "out"
    with kit.FakeNim(LLAMA, delay=8.0) as nim:
        proc = subprocess.Popen(
            [sys.executable, "-I", "-B", str(kit.script_path("nimsmoke.py")), "run", "--url",
             nim.url, "--out", str(out)],
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            env=kit.child_env(), text=True,
        )
        try:
            deadline = time.monotonic() + 30
            while not nim.posts() and time.monotonic() < deadline:
                time.sleep(0.05)
            assert nim.posts(), "the first chat request was sent"
            proc.send_signal(signal.SIGINT)
            stdout, _ = proc.communicate(timeout=30)
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.communicate()
    assert proc.returncode == nimsmoke.EXIT_INTERRUPTED
    lines = [line for line in stdout.splitlines() if line.strip()]
    assert len(lines) == 1
    payload = json.loads(lines[0])
    kit.assert_contract(payload)
    assert (payload["ok"], payload["state"], payload["next"]) == (False, "interrupted",
                                                                  "ask_user")
    assert payload["errors"][0]["code"] == "interrupted"
    doc = json.loads((out / "results.json").read_text(encoding="utf-8"))
    assert doc["state"] == "interrupted"
    assert [r["id"] for r in doc["results"]] == ["health", "models"]
    assert doc["not_run"][0] == "chat" and "concurrency" in doc["not_run"]
    assert (out / "evidence" / "health-1.json").is_file()
    assert (out / "evidence" / "models-1.json").is_file()


def test_family_on_a_model_it_does_not_match_is_warned_about(tmp_path: Path) -> None:
    """docs-7: --family is for a renamed served id. Applied to a model whose id, root and
    image none of the family's ids or patterns match, the run warns that its verdicts hold only
    for a renamed model of that family; no_capability_record asks for the docs first."""
    with kit.FakeNim(UNKNOWN) as nim:
        code, payload, _ = smoke(nim, tmp_path / "plain", "--checks", "models,tools")
        tools = by_id(payload)["tools"]
        assert tools["reason"] == "no_capability_record"
        assert "--documented FEATURE=URL" in tools["message"]
        assert "only when the served id renames" in tools["message"]
        code, payload, _ = smoke(nim, tmp_path / "family", "--checks", "models,tools",
                                 "--family", "llama-3.1-instruct")
    assert code == 0, payload
    assert payload["facts"]["target"]["how"] == "family"
    found = [text for text in payload["facts"]["warnings"] if text.startswith("--family")]
    assert len(found) == 1 and UNKNOWN in found[0] and "NIM_SERVED_MODEL_NAME" in found[0]
    with kit.FakeNim(LLAMA) as nim:
        code, payload, _ = smoke(nim, tmp_path / "same", "--checks", "models",
                                 "--family", "llama-3.1-instruct")
    assert not any(text.startswith("--family") for text in payload["facts"]["warnings"])
    code, payload, _ = kit.run_cli("nimsmoke.py", "features", "--family", "llama-3.1-instruct",
                                   "--model", UNKNOWN)
    assert payload["facts"]["warnings"][0].startswith("--family llama-3.1-instruct")


def test_run_on_kubernetes_without_port_forward_rights_is_rbac_missing(tmp_path: Path) -> None:
    tools = kit.make_fake_tools(tmp_path, kubectl_fixtures={
        "auth can-i create pods": kit.can_i(False),
    })
    code, payload, _ = kit.run_cli(
        "nimsmoke.py", "run", "--k8s", "nimservice/llama", "--context", "lab", "--namespace",
        "nim", "--out", str(tmp_path / "out"), env=tools.env(),
    )
    assert (code, payload["state"], payload["errors"][0]["code"]) == (3, "precondition",
                                                                      "rbac_missing")
    tools.assert_read_only()


# --- features and help ---------------------------------------------------------------------------


def test_features_prints_the_record_how_it_matched_and_the_plan() -> None:
    code, payload, _ = kit.run_cli("nimsmoke.py", "features", "--image",
                                   "nvcr.io/nim/openai/gpt-oss-20b:2.1.5")
    assert (code, payload["state"], payload["next"]) == (0, "matched", "done")
    facts = payload["facts"]
    assert (facts["record"], facts["how"], facts["generation"]) == ("gpt-oss", "image", "2.x")
    plan = {entry["check"]: entry for entry in facts["plan"]}
    assert list(plan) == list(nf.CHECKS)
    assert (plan["reasoning_off"]["decision"], plan["reasoning_off"]["reason"]) == (
        "not_applicable", "always_on")
    assert plan["reasoning_on"]["decision"] == "test" and plan["reasoning_on"]["sources"]
    assert facts["features"]["reasoning"]["default"] == "always"
    assert all(url.startswith("https://") for url in facts["features"]["tools"]["sources"])

    code, payload, _ = kit.run_cli("nimsmoke.py", "features", "--model", UNKNOWN)
    assert (code, payload["state"], payload["facts"]["record"]) == (0, "no_record", None)
    assert payload["facts"]["how"] == "none"
    code, payload, _ = kit.run_cli("nimsmoke.py", "features", "--family", "warp")
    assert (code, payload["errors"][0]["code"]) == (2, "unknown_family")


@pytest.mark.parametrize("command", ["", "run", "features", "report"])
def test_help_of_every_subcommand_has_an_example(command: str) -> None:
    argv = [command, "--help"] if command else ["--help"]
    code, text, _ = kit.run_cli("nimsmoke.py", *argv, raw=True)
    assert code == 0 and "Example: python3 scripts/nimsmoke.py" in text
    assert "usage:" in text
