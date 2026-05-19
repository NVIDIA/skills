#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Validate Nemo Gym pivot-format JSONL rows.

The structural checks are dependency-free. If --gym-repo is provided, the script also validates
responses_create_params and expected_action against the local Gym Pydantic models for
single_step_tool_use_with_argument_comparison.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


class ValidationError(RuntimeError):
    pass


def iter_jsonl(path: Path):
    with path.open() as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                yield line_number, json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValidationError(f"line {line_number}: invalid JSON: {exc}") from exc


def get_path(row: dict[str, Any], dotted_path: str) -> Any:
    current: Any = row
    for part in dotted_path.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def extract_tool_name(tool: Any) -> str | None:
    if not isinstance(tool, dict):
        return None
    if isinstance(tool.get("name"), str):
        return tool["name"]
    function = tool.get("function")
    if isinstance(function, dict) and isinstance(function.get("name"), str):
        return function["name"]
    return None


def expected_calls(expected_action: dict[str, Any]) -> list[dict[str, Any]]:
    action_type = expected_action.get("type")
    if action_type == "function_call":
        return [expected_action]
    return []


def validate_function_call(call: Any, line_number: int) -> str:
    if not isinstance(call, dict):
        raise ValidationError(f"line {line_number}: expected function call must be an object")
    if call.get("type") != "function_call":
        raise ValidationError(f"line {line_number}: expected call has type {call.get('type')!r}")
    name = call.get("name")
    if not isinstance(name, str) or not name:
        raise ValidationError(f"line {line_number}: function_call is missing a non-empty name")
    arguments = call.get("arguments")
    if not isinstance(arguments, str):
        raise ValidationError(f"line {line_number}: function_call.arguments must be a JSON string")
    try:
        json.loads(arguments)
    except (json.JSONDecodeError, TypeError, UnicodeDecodeError) as exc:
        raise ValidationError(f"line {line_number}: function_call.arguments is not valid JSON: {exc}") from exc
    return name


def validate_expected_action(expected_action: Any, line_number: int) -> tuple[str, list[str]]:
    if not isinstance(expected_action, dict):
        raise ValidationError(f"line {line_number}: expected_action must be an object")

    action_type = expected_action.get("type")
    if action_type == "message":
        if not isinstance(expected_action.get("content"), str):
            raise ValidationError(f"line {line_number}: message expected_action.content must be a string")
        return action_type, []

    if action_type != "function_call":
        raise ValidationError(f"line {line_number}: unsupported expected_action type {action_type!r}")

    tool_names = [validate_function_call(call, line_number) for call in expected_calls(expected_action)]
    return action_type, tool_names


def validate_responses_create_params(params: Any, line_number: int) -> set[str]:
    if not isinstance(params, dict):
        raise ValidationError(f"line {line_number}: responses_create_params must be an object")
    if "metadata" in params and params["metadata"] is None:
        raise ValidationError(f"line {line_number}: responses_create_params.metadata must be omitted or non-null")
    if "tool_choice" in params and params["tool_choice"] != "auto":
        raise ValidationError(
            f"line {line_number}: responses_create_params.tool_choice should be 'auto', "
            f"found {params['tool_choice']!r}"
        )
    input_items = params.get("input")
    if not isinstance(input_items, list) or not input_items:
        raise ValidationError(f"line {line_number}: responses_create_params.input must be a non-empty list")
    tools = params.get("tools")
    if not isinstance(tools, list):
        raise ValidationError(f"line {line_number}: responses_create_params.tools must be a list")
    if "parallel_tool_calls" in params and not isinstance(params["parallel_tool_calls"], bool):
        raise ValidationError(f"line {line_number}: responses_create_params.parallel_tool_calls must be a boolean")
    return {name for tool in tools for name in [extract_tool_name(tool)] if name}


def validate_agent_ref(agent_ref: Any, line_number: int, expected_agent_ref: str | None) -> str:
    if not isinstance(agent_ref, dict):
        raise ValidationError(f"line {line_number}: agent_ref must be an object")
    if agent_ref.get("type") != "responses_api_agents":
        raise ValidationError(f"line {line_number}: agent_ref.type must be responses_api_agents")
    name = agent_ref.get("name")
    if not isinstance(name, str) or not name:
        raise ValidationError(f"line {line_number}: agent_ref.name must be a non-empty string")
    if expected_agent_ref is not None and name != expected_agent_ref:
        raise ValidationError(
            f"line {line_number}: agent_ref.name {name!r} does not match expected {expected_agent_ref!r}"
        )
    return name


def load_gym_validators(gym_repo: Path | None):
    if gym_repo is None:
        return None, None
    sys.path.insert(0, str(gym_repo))
    try:
        from pydantic import TypeAdapter

        from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
        from resources_servers.single_step_tool_use_with_argument_comparison.common.verification_utils import (
            ExpectedAction,
        )
    except Exception as exc:  # noqa: BLE001
        raise ValidationError(f"failed to import Gym validators from {gym_repo}: {exc}") from exc

    return NeMoGymResponseCreateParamsNonStreaming, TypeAdapter(ExpectedAction)


def validate_with_gym_models(
    response_model: Any,
    expected_action_adapter: Any,
    params: dict[str, Any],
    expected_action: dict[str, Any],
    line_number: int,
) -> None:
    if response_model is None or expected_action_adapter is None:
        return
    try:
        response_model(**params)
        expected_action_adapter.validate_python(expected_action)
    except Exception as exc:  # noqa: BLE001
        raise ValidationError(f"line {line_number}: Gym model validation failed: {exc}") from exc


def parse_any_field_groups(values: list[str]) -> list[list[str]]:
    groups = []
    for value in values:
        group = [item.strip() for item in value.split(",") if item.strip()]
        if group:
            groups.append(group)
    return groups


def validate_file(args: argparse.Namespace) -> Counter:
    response_model, expected_action_adapter = load_gym_validators(args.gym_repo)
    any_field_groups = parse_any_field_groups(args.require_any_field)
    metrics: Counter = Counter()

    for line_number, row in iter_jsonl(args.path):
        metrics["rows"] += 1
        if not isinstance(row, dict):
            raise ValidationError(f"line {line_number}: row must be an object")

        for required_key in ("responses_create_params", "expected_action", "agent_ref"):
            if required_key not in row:
                raise ValidationError(f"line {line_number}: missing required key {required_key!r}")

        for required_field in args.require_field:
            if get_path(row, required_field) is None:
                raise ValidationError(f"line {line_number}: missing required field {required_field!r}")

        for group in any_field_groups:
            if not any(get_path(row, candidate) is not None for candidate in group):
                raise ValidationError(
                    f"line {line_number}: none of required alternative fields are present: {', '.join(group)}"
                )

        params = row["responses_create_params"]
        expected_action = row["expected_action"]
        agent_name = validate_agent_ref(row["agent_ref"], line_number, args.agent_ref)
        tool_names = validate_responses_create_params(params, line_number)
        action_type, expected_tool_names = validate_expected_action(expected_action, line_number)
        validate_with_gym_models(response_model, expected_action_adapter, params, expected_action, line_number)

        if args.check_tool_names:
            missing_tools = sorted(set(expected_tool_names) - tool_names)
            if missing_tools:
                raise ValidationError(
                    f"line {line_number}: expected_action uses tools absent from responses_create_params.tools: "
                    f"{missing_tools}"
                )

        metrics["action_type", action_type] += 1
        metrics["agent_ref", agent_name] += 1
        metrics["input_items_total"] += len(params["input"])
        metrics["tools_total"] += len(params.get("tools") or [])
        for tool_name in expected_tool_names:
            metrics["expected_tool", tool_name] += 1

    if metrics["rows"] == 0:
        raise ValidationError(f"{args.path} contains no JSONL rows")
    return metrics


def pct(count: int, total: int) -> str:
    if total == 0:
        return "0.00%"
    return f"{count / total:.2%}"


def print_expected_action_split(metrics: Counter) -> None:
    total = metrics["rows"]
    function_call_count = metrics["action_type", "function_call"]
    message_count = metrics["action_type", "message"]
    print("expected_action_split:")
    print(f"  function_call: {function_call_count} ({pct(function_call_count, total)})")
    print(f"  message: {message_count} ({pct(message_count, total)})")


def print_metrics(metrics: Counter) -> None:
    print(f"rows: {metrics['rows']}")
    for prefix in ("agent_ref", "expected_tool"):
        rows = [
            (key[1], count)
            for key, count in metrics.items()
            if isinstance(key, tuple) and len(key) == 2 and key[0] == prefix
        ]
        if not rows:
            continue
        print(prefix + ":")
        for value, count in sorted(rows, key=lambda item: (-item[1], str(item[0]))):
            print(f"  {value}: {count}")
    print_expected_action_split(metrics)
    print(f"avg_input_items: {metrics['input_items_total'] / metrics['rows']:.2f}")
    print(f"avg_tools: {metrics['tools_total'] / metrics['rows']:.2f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", type=Path, required=True, help="Pivot JSONL path to validate")
    parser.add_argument("--agent-ref", help="Expected row-level agent_ref.name")
    parser.add_argument("--gym-repo", type=Path, help="Optional Gym repo root for Pydantic model validation")
    parser.add_argument(
        "--no-check-tool-names",
        dest="check_tool_names",
        action="store_false",
        help="Do not require expected tool names to appear in responses_create_params.tools",
    )
    parser.add_argument(
        "--require-field",
        action="append",
        default=[],
        help="Require a dotted metadata field path. May be repeated.",
    )
    parser.add_argument(
        "--require-any-field",
        action="append",
        default=[],
        help="Require at least one comma-separated dotted metadata field path.",
    )
    parser.set_defaults(check_tool_names=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        metrics = validate_file(args)
    except ValidationError as exc:
        print(f"validation failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    print_metrics(metrics)


if __name__ == "__main__":
    main()
