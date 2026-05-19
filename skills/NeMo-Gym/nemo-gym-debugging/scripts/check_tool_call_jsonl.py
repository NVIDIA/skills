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
"""Static sanity checks for Nemo Gym tool-call JSONL data."""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


SCHEMA_VALUE_EXEMPT_KEYS = {"enum", "const", "default", "examples"}


def iter_jsonl(path: Path):
    with path.open(encoding="utf-8") as f:
        for row_idx, line in enumerate(f):
            if not line.strip():
                continue
            try:
                yield row_idx, json.loads(line)
            except json.JSONDecodeError as exc:
                yield row_idx, {"__json_error__": f"{type(exc).__name__}: {exc}"}


def value_key(value: Any) -> str:
    if value is None:
        return "None"
    return str(value)


def add_issue(
    issues: list[dict[str, Any]],
    *,
    row_idx: int,
    severity: str,
    code: str,
    message: str,
) -> None:
    issues.append(
        {
            "row_idx": row_idx,
            "severity": severity,
            "code": code,
            "message": message,
        }
    )


def walk_tool_schema(
    value: Any,
    *,
    path: str = "$",
    key: str | None = None,
    in_properties_map: bool = False,
    check_vllm_compat: bool = True,
):
    if key in SCHEMA_VALUE_EXEMPT_KEYS:
        return

    if isinstance(value, bool):
        if check_vllm_compat:
            yield "boolean_schema_node", path, value
        return

    if isinstance(value, list):
        for i, item in enumerate(value):
            yield from walk_tool_schema(
                item,
                path=f"{path}/{i}",
                key=key,
                check_vllm_compat=check_vllm_compat,
            )
        return

    if not isinstance(value, dict):
        return

    if in_properties_map:
        for prop_name, prop_schema in value.items():
            prop_path = f"{path}/{prop_name}"
            if not isinstance(prop_schema, (dict, bool)):
                yield "invalid_property_schema", prop_path, prop_schema
                continue
            yield from walk_tool_schema(
                prop_schema,
                path=prop_path,
                key=prop_name,
                check_vllm_compat=check_vllm_compat,
            )
        return

    if check_vllm_compat and isinstance(value.get("format"), str):
        yield "format_annotation", f"{path}/format", value["format"]

    for child_key, child_value in value.items():
        yield from walk_tool_schema(
            child_value,
            path=f"{path}/{child_key}",
            key=child_key,
            in_properties_map=child_key == "properties" and isinstance(child_value, dict),
            check_vllm_compat=check_vllm_compat,
        )


def schema_contains_property_key(value: Any, property_key: str) -> bool:
    if isinstance(value, dict):
        properties = value.get("properties")
        if isinstance(properties, dict) and property_key in properties:
            return True
        return any(schema_contains_property_key(child, property_key) for child in value.values())
    if isinstance(value, list):
        return any(schema_contains_property_key(child, property_key) for child in value)
    return False


def check_row(
    row_idx: int,
    row: dict[str, Any],
    *,
    require_response_mode: bool,
    check_vllm_compat: bool,
):
    issues: list[dict[str, Any]] = []

    if "__json_error__" in row:
        add_issue(
            issues,
            row_idx=row_idx,
            severity="error",
            code="json_parse_error",
            message=row["__json_error__"],
        )
        return issues

    if require_response_mode and row.get("response_mode") != "tool_call":
        add_issue(
            issues,
            row_idx=row_idx,
            severity="error",
            code="bad_response_mode",
            message=f"Expected response_mode='tool_call', got {row.get('response_mode')!r}",
        )

    responses_create_params = row.get("responses_create_params")
    if not isinstance(responses_create_params, dict):
        add_issue(
            issues,
            row_idx=row_idx,
            severity="error",
            code="missing_responses_create_params",
            message="responses_create_params must be an object",
        )
        return issues

    tools = responses_create_params.get("tools")
    if not isinstance(tools, list) or not tools:
        add_issue(
            issues,
            row_idx=row_idx,
            severity="error",
            code="missing_tools",
            message="responses_create_params.tools must be a non-empty list",
        )
        return issues

    tool_by_name: dict[str, dict[str, Any]] = {}
    tool_names = []
    for tool_idx, tool in enumerate(tools):
        if not isinstance(tool, dict):
            add_issue(
                issues,
                row_idx=row_idx,
                severity="error",
                code="bad_tool",
                message=f"tools[{tool_idx}] must be an object",
            )
            continue

        name = tool.get("name")
        if not isinstance(name, str) or not name:
            add_issue(
                issues,
                row_idx=row_idx,
                severity="error",
                code="bad_tool_name",
                message=f"tools[{tool_idx}].name must be a non-empty string",
            )
            continue
        tool_names.append(name)
        if name in tool_by_name:
            add_issue(
                issues,
                row_idx=row_idx,
                severity="error",
                code="duplicate_tool_name",
                message=f"Duplicate tool name {name!r}",
            )
        tool_by_name[name] = tool

        if tool.get("type") != "function":
            add_issue(
                issues,
                row_idx=row_idx,
                severity="error",
                code="bad_tool_type",
                message=f"tools[{tool_idx}].type must be 'function', got {tool.get('type')!r}",
            )

        parameters = tool.get("parameters")
        if not isinstance(parameters, dict):
            add_issue(
                issues,
                row_idx=row_idx,
                severity="error",
                code="bad_tool_parameters",
                message=f"tools[{tool_idx}].parameters must be an object",
            )
            continue
        for code, path, value in walk_tool_schema(parameters, check_vllm_compat=check_vllm_compat):
            add_issue(
                issues,
                row_idx=row_idx,
                severity="error",
                code=code,
                message=f"tools[{tool_idx}].parameters{path[1:]} = {value!r}",
            )

    expected_tool_name = row.get("tool_name")
    target_tool = tool_by_name.get(expected_tool_name) if expected_tool_name else None
    if expected_tool_name and target_tool is None:
        add_issue(
            issues,
            row_idx=row_idx,
            severity="error",
            code="target_tool_missing",
            message=f"tool_name {expected_tool_name!r} not found in tools {tool_names}",
        )

    tool_payload_key = row.get("tool_payload_key")
    if tool_payload_key and target_tool is not None:
        parameters = target_tool.get("parameters")
        if isinstance(parameters, dict) and not schema_contains_property_key(parameters, str(tool_payload_key)):
            add_issue(
                issues,
                row_idx=row_idx,
                severity="error",
                code="payload_key_not_in_target_tool_schema",
                message=f"tool_payload_key {tool_payload_key!r} not found in target tool parameters",
            )

    schema_str = row.get("schema_str")
    if schema_str is not None:
        try:
            json.loads(schema_str)
        except json.JSONDecodeError as exc:
            add_issue(
                issues,
                row_idx=row_idx,
                severity="error",
                code="schema_str_parse_error",
                message=f"{type(exc).__name__}: {exc}",
            )

    num_tools = row.get("num_tools")
    if num_tools is not None and num_tools != len(tools):
        add_issue(
            issues,
            row_idx=row_idx,
            severity="error",
            code="num_tools_mismatch",
            message=f"num_tools={num_tools!r}, actual tools={len(tools)}",
        )

    return issues


def print_distribution(rows: list[dict[str, Any]], keys: list[str]) -> None:
    if not keys:
        return

    print("\nDistributions:")
    for key in keys:
        counter = Counter(value_key(row.get(key)) for row in rows if key in row)
        if not counter:
            print(f"  {key}: <absent>")
            continue
        print(f"  {key}:")
        for value, count in sorted(counter.items(), key=lambda item: (-item[1], item[0])):
            print(f"    {value}: {count}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("--max-errors", type=int, default=30)
    parser.add_argument("--require-response-mode", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--check-vllm-compat", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--summary-key",
        action="append",
        default=[],
        help="Print a distribution for this row-level key. Repeat to inspect multiple keys.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    rows: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    for row_idx, row in iter_jsonl(input_path):
        rows.append(row)
        issues.extend(
            check_row(
                row_idx,
                row,
                require_response_mode=args.require_response_mode,
                check_vllm_compat=args.check_vllm_compat,
            )
        )

    error_counts = Counter(issue["code"] for issue in issues if issue["severity"] == "error")
    print(f"Checked {len(rows)} rows from {input_path}")
    if error_counts:
        print("\nErrors:")
        for code, count in sorted(error_counts.items(), key=lambda item: (-item[1], item[0])):
            print(f"  {code}: {count}")
    else:
        print("\nErrors: 0")

    print_distribution(rows, args.summary_key)

    if issues:
        print(f"\nFirst {min(args.max_errors, len(issues))} issues:")
        for issue in issues[: args.max_errors]:
            print(f"  row={issue['row_idx']} {issue['severity']} {issue['code']}: {issue['message']}")
        sys.exit(1)


if __name__ == "__main__":
    main()
