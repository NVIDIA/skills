#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run a live expected conversation against NVIDIA chat completions."""

from __future__ import annotations

import argparse
import atexit
import importlib
import json
import os
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import yaml

from nva_generic_defaults import resolve_llm_config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nva-root", required=True, type=Path)
    parser.add_argument(
        "--expected-file",
        type=Path,
        help=(
            "Expected conversation YAML. Defaults to "
            "<nva-root>/src/examples/generic/ambient_agent_expected_conversation.yaml."
        ),
    )
    parser.add_argument(
        "--llm-key",
        help=(
            "LLM catalog key to use from the generic service catalogs. Defaults to "
            "examples_registry.yaml examples.generic-assistant.defaults.llm[0]."
        ),
    )
    parser.add_argument("--model", help="Explicit model ID override.")
    parser.add_argument("--base-url", help="Explicit OpenAI-compatible base URL override.")
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument(
        "--current-datetime",
        help="ISO datetime override for relative date tests. Defaults to current time.",
    )
    parser.add_argument(
        "--current-timezone",
        help="Timezone for current datetime context. Defaults to NVA_HEALTHCARE_AGENT_TIMEZONE or UTC.",
    )
    parser.add_argument(
        "--live-test-approved",
        action="store_true",
        help=(
            "Confirm the user approved sending expected conversation histories "
            "to the configured LLM endpoint."
        ),
    )
    args = parser.parse_args()

    expected_path = _expected_path(args)
    expected = yaml.safe_load(expected_path.read_text(encoding="utf-8"))
    if not isinstance(expected, dict):
        print(f"FAIL: expected conversation YAML must be a mapping: {expected_path}", file=sys.stderr)
        return 2
    if not args.live_test_approved:
        print(
            "FAIL: live expected-conversation validation sends healthcare conversation "
            "histories to the configured LLM endpoint. Ask the user for explicit "
            "approval, then rerun with --live-test-approved.",
            file=sys.stderr,
        )
        return 2

    generic_dir = args.nva_root / "src" / "examples" / "generic"
    api_key = _load_api_key()
    if not api_key or api_key == "nvapi-...":
        print("FAIL: NVIDIA_API_KEY is missing from the process environment.", file=sys.stderr)
        return 2

    _configure_expected_environment(args.nva_root, expected)
    sys.path.insert(0, str(args.nva_root / "src"))
    tool_functions = importlib.import_module("examples.generic.ambient_agent_tools").TOOL_RESULT_FUNCTIONS

    prompt_key = str(expected["prompt_key"])
    try:
        tool_names = _expected_tool_names(expected)
    except RuntimeError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 2

    try:
        llm_config = resolve_llm_config(
            args.nva_root,
            llm_key=args.llm_key,
            model=args.model,
            base_url=args.base_url,
        )
    except RuntimeError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 2

    print(
        "Using LLM: "
        f"{llm_config['name']} "
        f"(key={llm_config['key']}, model_id={llm_config['model_id']}, base_url={llm_config['base_url']})."
    )

    try:
        prompt, datetime_summary = _append_current_datetime_context(
            _load_prompt(generic_dir, prompt_key),
            current_datetime=args.current_datetime or expected.get("current_datetime"),
            timezone_name=args.current_timezone or expected.get("current_timezone"),
        )
    except RuntimeError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 2
    print(f"Using current datetime context: {datetime_summary}.")

    tools = _load_tools(generic_dir, tool_names)
    missing_tool_functions = [name for name in tool_names if name not in tool_functions]
    if missing_tool_functions:
        print(
            "FAIL: tool(s) are not defined in TOOL_RESULT_FUNCTIONS: "
            + ", ".join(missing_tool_functions),
            file=sys.stderr,
        )
        return 2
    selected_tool_functions = {name: tool_functions[name] for name in tool_names}

    if "test_conversations" in expected:
        return _run_history_tests(
            expected=expected,
            prompt=prompt,
            tools=tools,
            tool_names=tool_names,
            tool_result_functions=selected_tool_functions,
            llm_config=llm_config,
            api_key=api_key,
            timeout=args.timeout,
        )

    return _run_legacy_transcript_test(
        expected=expected,
        prompt=prompt,
        tools=tools,
        tool_names=tool_names,
        tool_result_functions=selected_tool_functions,
        llm_config=llm_config,
        api_key=api_key,
        timeout=args.timeout,
    )


def _expected_path(args: argparse.Namespace) -> Path:
    if args.expected_file:
        path = args.expected_file
    else:
        path = args.nva_root / "src" / "examples" / "generic" / "ambient_agent_expected_conversation.yaml"
    if not path.is_file():
        raise SystemExit(f"expected conversation file does not exist: {path}")
    return path


def _run_history_tests(
    *,
    expected: dict[str, Any],
    prompt: str,
    tools: list[dict[str, Any]],
    tool_names: list[str],
    tool_result_functions: dict[str, Any],
    llm_config: dict[str, Any],
    api_key: str,
    timeout: float,
) -> int:
    test_conversations = expected.get("test_conversations")
    if not isinstance(test_conversations, list) or not test_conversations:
        print("FAIL: test_conversations must be a non-empty list.", file=sys.stderr)
        return 2

    semantic_review_count = 0
    for index, test_case in enumerate(test_conversations, start=1):
        if not isinstance(test_case, dict):
            print(f"FAIL: test_conversations[{index}] must be a mapping.", file=sys.stderr)
            return 2
        name = str(test_case.get("name") or f"conversation_{index}")
        history = _normalize_history(test_case.get("history"), name)
        if history is None:
            return 2
        if "expected_tool_call" not in test_case:
            print(f"FAIL: {name}: expected_tool_call is required.", file=sys.stderr)
            return 2
        expected_tool_call = _parse_expected_bool(test_case["expected_tool_call"], f"{name}.expected_tool_call")
        if expected_tool_call is None:
            return 2
        expected_tool_name = None
        if expected_tool_call:
            expected_tool_name = _expected_tool_name_for_case(test_case, tool_names, name)
            if expected_tool_name is None:
                return 2

        expected_next_message_content = str(test_case.get("expected_next_message_content") or "").strip()
        if not expected_next_message_content:
            print(f"FAIL: {name}: expected_next_message_content is required.", file=sys.stderr)
            return 2

        messages = [{"role": "system", "content": prompt}, *history]
        try:
            assistant_text, turn_results, turn_arguments, turn_tool_names = _run_turn(
                base_url=str(llm_config["base_url"]).rstrip("/"),
                api_key=api_key,
                model=str(llm_config["model_id"]),
                extra_payload=dict(llm_config.get("extra_payload") or {}),
                messages=messages,
                tools=tools,
                tool_result_functions=tool_result_functions,
                timeout=timeout,
            )
        except urllib.error.URLError as exc:
            print(f"FAIL: network error while calling NVIDIA chat completions: {exc.reason}", file=sys.stderr)
            return 1
        except RuntimeError as exc:
            print(f"FAIL: {name}: {exc}", file=sys.stderr)
            return 1

        print(f"TEST: {name}")
        print(f"HISTORY_LAST_USER: {history[-1]['content']}")
        print(f"EXPECTED_NEXT_MESSAGE_CONTENT: {expected_next_message_content}")
        for result, called_tool_name in zip(turn_results, turn_tool_names):
            result_id = _result_id(result, _result_id_field(expected, test_case))
            print(f"TOOL {called_tool_name}: status={result.get('status')} id={result_id}")
        print(f"ASSISTANT: {assistant_text}")
        print("SEMANTIC_REVIEW_REQUIRED: judge whether ASSISTANT aligns with EXPECTED_NEXT_MESSAGE_CONTENT.\n")
        semantic_review_count += 1

        if expected_tool_call and len(turn_results) != 1:
            print(
                f"FAIL: {name}: expected exactly one tool call, got {len(turn_results)}.",
                file=sys.stderr,
            )
            return 1
        if not expected_tool_call and turn_results:
            print(
                f"FAIL: {name}: expected no tool call, got {len(turn_results)}.",
                file=sys.stderr,
            )
            return 1

        if expected_tool_call:
            final_result = turn_results[-1]
            final_arguments = turn_arguments[-1]
            final_tool_name = turn_tool_names[-1]
            if expected_tool_name and final_tool_name != expected_tool_name:
                print(
                    f"FAIL: {name}: expected tool {expected_tool_name!r}, got {final_tool_name!r}.",
                    file=sys.stderr,
                )
                return 1
            final_status = str(test_case.get("expected_final_status", expected.get("expected_final_status", "saved")))
            if final_result.get("status") != final_status:
                print(f"FAIL: {name}: expected final status {final_status!r}, got {final_result}", file=sys.stderr)
                return 1
            if not _validate_expected_arguments(expected, test_case, final_arguments, name):
                return 1
            result_id_field = _result_id_field(expected, test_case)
            if result_id_field and result_id_field not in final_result:
                print(
                    f"FAIL: {name}: expected result ID field {result_id_field!r}, "
                    f"got keys={sorted(final_result)}",
                    file=sys.stderr,
                )
                return 1

    print(
        f"PASS: scenario={expected.get('scenario')} deterministic checks passed "
        f"for {len(test_conversations)} conversation(s); semantic review required for "
        f"{semantic_review_count} assistant message(s)."
    )
    return 0


def _run_legacy_transcript_test(
    *,
    expected: dict[str, Any],
    prompt: str,
    tools: list[dict[str, Any]],
    tool_names: list[str],
    tool_result_functions: dict[str, Any],
    llm_config: dict[str, Any],
    api_key: str,
    timeout: float,
) -> int:
    if len(tool_names) != 1:
        print("FAIL: legacy transcript validation supports exactly one tool.", file=sys.stderr)
        return 2
    tool_name = tool_names[0]
    transcript = list(expected["transcript"])
    expected_per_turn = list(expected["expected_per_turn_tool_calls"])
    expected_final_status = str(expected.get("expected_final_status", "saved"))
    expected_result_id_field = expected.get("expected_result_id_field")
    if expected_result_id_field is not None:
        expected_result_id_field = str(expected_result_id_field)

    if not _validate_optional_mapping(expected.get("expected_tool_arguments"), "expected_tool_arguments"):
        return 2
    if not _validate_optional_mapping(
        expected.get("expected_tool_argument_substrings"),
        "expected_tool_argument_substrings",
    ):
        return 2

    messages: list[dict[str, Any]] = [{"role": "system", "content": prompt}]
    per_turn: list[int] = []
    final_result: dict[str, Any] | None = None
    final_arguments: dict[str, Any] | None = None

    for turn_index, user_text in enumerate(transcript):
        messages.append({"role": "user", "content": user_text})
        try:
            assistant_text, turn_results, turn_arguments, _turn_tool_names = _run_turn(
                base_url=str(llm_config["base_url"]).rstrip("/"),
                api_key=api_key,
                model=str(llm_config["model_id"]),
                extra_payload=dict(llm_config.get("extra_payload") or {}),
                messages=messages,
                tools=tools,
                tool_result_functions=tool_result_functions,
                timeout=timeout,
            )
        except urllib.error.URLError as exc:
            print(f"FAIL: network error while calling NVIDIA chat completions: {exc.reason}", file=sys.stderr)
            return 1
        except RuntimeError as exc:
            print(f"FAIL: {exc}", file=sys.stderr)
            return 1
        per_turn.append(len(turn_results))
        if turn_results:
            final_result = turn_results[-1]
            final_arguments = turn_arguments[-1]

        print(f"USER: {user_text}")
        for result in turn_results:
            print(
                f"TOOL {tool_name}: "
                f"status={result.get('status')} "
                f"id={_result_id(result, expected_result_id_field)}"
            )
        print(f"ASSISTANT: {assistant_text}\n")

        if len(per_turn) <= len(expected_per_turn) and per_turn[-1] != expected_per_turn[turn_index]:
            print(
                "FAIL: unexpected per-turn tool timing: "
                f"expected={expected_per_turn}, actual_so_far={per_turn}",
                file=sys.stderr,
            )
            return 1

    if per_turn != expected_per_turn:
        print(f"FAIL: expected per_turn={expected_per_turn}, actual={per_turn}", file=sys.stderr)
        return 1
    if not final_result or final_result.get("status") != expected_final_status:
        print(f"FAIL: expected final status {expected_final_status!r}, got {final_result}", file=sys.stderr)
        return 1
    if not _validate_expected_arguments(expected, {}, final_arguments or {}, "legacy_transcript"):
        return 1
    if expected_result_id_field and expected_result_id_field not in final_result:
        print(
            f"FAIL: expected result ID field {expected_result_id_field!r}, got keys={sorted(final_result)}",
            file=sys.stderr,
        )
        return 1

    final_id = _result_id(final_result, expected_result_id_field)
    print(f"PASS: scenario={expected.get('scenario')} per_turn={per_turn} final_id={final_id}")
    return 0


def _run_turn(
    *,
    base_url: str,
    api_key: str,
    model: str,
    extra_payload: dict[str, Any],
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    tool_result_functions: dict[str, Any],
    timeout: float,
) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    turn_results: list[dict[str, Any]] = []
    turn_arguments: list[dict[str, Any]] = []
    turn_tool_names: list[str] = []
    for _ in range(4):
        payload = {
            "model": model,
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto",
            "temperature": 0,
            "max_tokens": 512,
            "stream": False,
            **extra_payload,
        }
        response = _chat_completion(
            base_url=base_url,
            api_key=api_key,
            payload=payload,
            timeout=timeout,
        )
        message = response["choices"][0]["message"]
        tool_calls = message.get("tool_calls") or []
        if not tool_calls:
            assistant_text = str(message.get("content") or "").strip()
            messages.append({"role": "assistant", "content": assistant_text})
            return assistant_text, turn_results, turn_arguments, turn_tool_names

        messages.append({"role": "assistant", "content": message.get("content"), "tool_calls": tool_calls})
        for tool_call in tool_calls:
            function = tool_call.get("function") or {}
            tool_name = str(function.get("name") or "")
            if tool_name not in tool_result_functions:
                raise RuntimeError(f"Unexpected tool call: {tool_name}")
            arguments = json.loads(function.get("arguments") or "{}")
            result = tool_result_functions[tool_name](arguments)
            turn_results.append(result)
            turn_arguments.append(arguments)
            turn_tool_names.append(tool_name)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "name": tool_name,
                    "content": json.dumps(result, separators=(",", ":")),
                }
            )
    raise RuntimeError("Tool-call loop exceeded retry limit")


def _chat_completion(*, base_url: str, api_key: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    completion_url = _validated_completion_url(base_url)
    request = urllib.request.Request(
        completion_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:  # nosec B310 - URL validated above
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            if exc.code in {502, 503, 504} and attempt < 2:
                time.sleep(3 * (attempt + 1))
                continue
            raise RuntimeError(f"NVIDIA chat completion failed HTTP {exc.code}: {body[:500]}") from exc
    raise RuntimeError("NVIDIA chat completion failed after retries")


def _validated_completion_url(base_url: str) -> str:
    normalized = base_url.rstrip("/")
    parsed = urllib.parse.urlparse(normalized)
    if parsed.username or parsed.password:
        raise RuntimeError("LLM base URL must not contain embedded credentials")
    if parsed.scheme == "https" and parsed.hostname:
        return f"{normalized}/chat/completions"
    if parsed.scheme == "http" and parsed.hostname in {"127.0.0.1", "localhost", "::1"}:
        return f"{normalized}/chat/completions"
    raise RuntimeError("LLM base URL must use HTTPS, except for an explicit loopback HTTP endpoint")


def _load_prompt(generic_dir: Path, prompt_key: str) -> str:
    catalog = yaml.safe_load((generic_dir / "prompts.yaml").read_text(encoding="utf-8"))
    prompt = catalog.get(prompt_key, {})
    content = str(prompt.get("content") or "").strip()
    if not content:
        raise RuntimeError(f"Prompt {prompt_key!r} was not found")
    return content


def _append_current_datetime_context(
    system_message: str,
    *,
    current_datetime: Any = None,
    timezone_name: Any = None,
) -> tuple[str, str]:
    timezone_label, timezone = _resolve_timezone(timezone_name)
    if current_datetime:
        now = _parse_datetime(current_datetime, timezone)
    else:
        now = datetime.now(timezone)

    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone)
    else:
        now = now.astimezone(timezone)

    today = now.strftime("%A, %B %d, %Y").replace(" 0", " ")
    context = (
        "\n\nCurrent datetime context:\n"
        f"- Current timezone: {timezone_label}\n"
        f"- Current datetime: {now.isoformat()}\n"
        f"- Today is {today}\n"
        "Use this context to resolve relative healthcare dates "
        "such as today, tomorrow, next week, and next Monday. Preserve "
        "absolute dates the user provides."
    )
    return system_message.rstrip() + context, f"{now.isoformat()} ({timezone_label}; today={today})"


def _resolve_timezone(timezone_name: Any = None) -> tuple[str, ZoneInfo]:
    timezone_label = str(timezone_name or os.getenv("NVA_HEALTHCARE_AGENT_TIMEZONE") or "UTC").strip() or "UTC"
    try:
        return timezone_label, ZoneInfo(timezone_label)
    except Exception as exc:
        if timezone_name:
            raise RuntimeError(f"Invalid current timezone {timezone_label!r}.") from exc
        return "UTC", ZoneInfo("UTC")


def _parse_datetime(value: Any, timezone: ZoneInfo) -> datetime:
    raw_value = str(value).strip()
    if not raw_value:
        return datetime.now(timezone)
    try:
        parsed = datetime.fromisoformat(raw_value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise RuntimeError(f"Invalid current_datetime {raw_value!r}; use ISO datetime format.") from exc

    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone)
    return parsed.astimezone(timezone)


def _configure_expected_environment(nva_root: Path, expected: dict[str, Any]) -> None:
    database_start_date = str(expected.get("database_start_date") or "").strip()
    if database_start_date and "NVA_APPOINTMENT_DB_START_DATE" not in os.environ:
        os.environ["NVA_APPOINTMENT_DB_START_DATE"] = database_start_date

    if (
        str(expected.get("scenario") or "") == "appointment-making"
        and "NVA_APPOINTMENT_DB_PATH" not in os.environ
    ):
        temp_dir = tempfile.TemporaryDirectory(prefix="nva-appointment-expected-")
        atexit.register(temp_dir.cleanup)
        os.environ["NVA_APPOINTMENT_DB_PATH"] = str(Path(temp_dir.name) / "appointment_schedule.sqlite")


def _expected_tool_names(expected: dict[str, Any]) -> list[str]:
    if "tool_names" in expected:
        value = expected["tool_names"]
        if not isinstance(value, list) or not value:
            raise RuntimeError("tool_names must be a non-empty list when provided.")
        names = [str(name).strip() for name in value]
    elif "tool_name" in expected:
        names = [str(expected["tool_name"]).strip()]
    else:
        raise RuntimeError("expected conversation YAML must define tool_name or tool_names.")

    if any(not name for name in names):
        raise RuntimeError("tool names must be non-empty strings.")
    if len(names) != len(set(names)):
        raise RuntimeError("tool_names must not contain duplicates.")
    return names


def _expected_tool_name_for_case(
    test_case: dict[str, Any],
    tool_names: list[str],
    name: str,
) -> str | None:
    value = str(test_case.get("expected_tool_name") or "").strip()
    if value:
        if value not in tool_names:
            print(
                f"FAIL: {name}: expected_tool_name {value!r} is not in tool_names={tool_names!r}.",
                file=sys.stderr,
            )
            return None
        return value
    if len(tool_names) == 1:
        return tool_names[0]
    print(f"FAIL: {name}: expected_tool_name is required when tool_names has multiple tools.", file=sys.stderr)
    return None


def _load_tools(generic_dir: Path, tool_names: list[str]) -> list[dict[str, Any]]:
    return [_load_tool(generic_dir, tool_name) for tool_name in tool_names]


def _load_tool(generic_dir: Path, tool_name: str) -> dict[str, Any]:
    catalog = yaml.safe_load((generic_dir / "tools.yaml").read_text(encoding="utf-8"))
    tool = catalog.get(tool_name)
    if not isinstance(tool, dict):
        raise RuntimeError(f"Tool {tool_name!r} was not found")
    return tool


def _normalize_history(history: Any, name: str) -> list[dict[str, str]] | None:
    if not isinstance(history, list) or not history:
        print(f"FAIL: {name}: history must be a non-empty list.", file=sys.stderr)
        return None

    normalized: list[dict[str, str]] = []
    for index, message in enumerate(history, start=1):
        if not isinstance(message, dict):
            print(f"FAIL: {name}: history[{index}] must be a mapping.", file=sys.stderr)
            return None
        role = str(message.get("role") or "").strip()
        content = str(message.get("content") or "").strip()
        if role not in {"user", "assistant"}:
            print(f"FAIL: {name}: history[{index}].role must be user or assistant.", file=sys.stderr)
            return None
        if not content:
            print(f"FAIL: {name}: history[{index}].content is required.", file=sys.stderr)
            return None
        normalized.append({"role": role, "content": content})

    if normalized[-1]["role"] != "user":
        print(f"FAIL: {name}: history must end with the latest user message.", file=sys.stderr)
        return None
    return normalized


def _parse_expected_bool(value: Any, name: str) -> bool | None:
    if isinstance(value, bool):
        return value
    print(f"FAIL: {name} must be a YAML boolean true or false.", file=sys.stderr)
    return None


def _result_id_field(expected: dict[str, Any], test_case: dict[str, Any]) -> str | None:
    value = test_case.get("expected_result_id_field", expected.get("expected_result_id_field"))
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _validate_expected_arguments(
    expected: dict[str, Any],
    test_case: dict[str, Any],
    actual_arguments: dict[str, Any],
    name: str,
) -> bool:
    expected_tool_arguments = test_case.get("expected_tool_arguments", expected.get("expected_tool_arguments"))
    if not _validate_optional_mapping(expected_tool_arguments, f"{name}.expected_tool_arguments"):
        return False
    if expected_tool_arguments:
        argument_mismatches = _argument_mismatches(expected_tool_arguments, actual_arguments)
        if argument_mismatches:
            print(f"FAIL: {name}: tool argument mismatches: " + "; ".join(argument_mismatches), file=sys.stderr)
            return False

    expected_substrings = test_case.get(
        "expected_tool_argument_substrings",
        expected.get("expected_tool_argument_substrings"),
    )
    if not _validate_optional_mapping(expected_substrings, f"{name}.expected_tool_argument_substrings"):
        return False
    if expected_substrings:
        substring_mismatches = _argument_substring_mismatches(expected_substrings, actual_arguments)
        if substring_mismatches:
            print(
                f"FAIL: {name}: tool argument substring mismatches: " + "; ".join(substring_mismatches),
                file=sys.stderr,
            )
            return False
    return True


def _validate_optional_mapping(value: Any, name: str) -> bool:
    if value is not None and not isinstance(value, dict):
        print(f"FAIL: {name} must be a mapping when provided.", file=sys.stderr)
        return False
    return True


def _argument_mismatches(expected: dict[str, Any], actual: dict[str, Any]) -> list[str]:
    mismatches: list[str] = []
    for field, expected_value in expected.items():
        if field not in actual:
            mismatches.append(f"{field}: missing")
            continue
        actual_value = actual[field]
        if _normalize_value(actual_value) != _normalize_value(expected_value):
            mismatches.append(f"{field}: expected {expected_value!r}, got {actual_value!r}")
    return mismatches


def _argument_substring_mismatches(expected: dict[str, Any], actual: dict[str, Any]) -> list[str]:
    mismatches: list[str] = []
    for field, expected_terms in expected.items():
        if field not in actual:
            mismatches.append(f"{field}: missing")
            continue
        actual_text = _normalize_value(actual[field])
        terms = expected_terms if isinstance(expected_terms, list) else [expected_terms]
        missing_terms = [str(term) for term in terms if _normalize_value(term) not in actual_text]
        if missing_terms:
            mismatches.append(f"{field}: missing substrings {missing_terms!r}, got {actual[field]!r}")
    return mismatches


def _normalize_value(value: Any) -> str:
    text = " ".join(str(value).strip().lower().split())
    if text.startswith("between ") and " and " in text:
        text = text.removeprefix("between ").replace(" and ", " to ", 1)
    return text


def _result_id(result: dict[str, Any], expected_field: str | None = None) -> str:
    if expected_field:
        return str(result.get(expected_field) or "")
    for field in ("record_id", "id"):
        if result.get(field):
            return str(result[field])
    for field in sorted(result):
        if field.endswith("_id") and result.get(field):
            return str(result[field])
    return str(result.get("id") or "")


def _load_api_key() -> str:
    return os.getenv("NVIDIA_API_KEY", "")


if __name__ == "__main__":
    raise SystemExit(main())
