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
"""Reference script: convert tool-use message chat trajectories into single-step pivot rows.

This is a copied example from a concrete dataset conversion. It may contain
dataset-specific paths, assumptions, and branch-specific behavior, and should
be adapted before reuse.

The output format targets the
``single_step_tool_use_with_argument_comparison`` Nemo Gym environment. Each
assistant tool turn becomes one row: the prefix becomes
``responses_create_params.input`` and the assistant tool action becomes
``expected_action``. Multi-tool assistant turns are skipped for compatibility
with the singular `expected_action` row contract.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any
from uuid import uuid4


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_IN_PATH = PROJECT_DIR / "data" / "messages.jsonl"
DEFAULT_OUT_PATH = PROJECT_DIR / "data" / "pivot_datasets" / "messages_pivot.jsonl"
DEFAULT_AGENT_REF = "tool_use_single_step_tool_use_with_argument_comparison_agent"
DEFAULT_GYM_REPO = Path("/lustre/fsw/portfolios/llmservice/users/jkyi/current/nemo/Gym-github")


def _content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                if "text" in block:
                    parts.append(str(block["text"]))
                elif "content" in block:
                    parts.append(str(block["content"]))
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts)
    return str(content)


def _json_arguments(arguments: Any) -> str:
    if isinstance(arguments, str):
        return arguments
    if arguments is None:
        return "{}"
    return json.dumps(arguments)


def _normalized_tool_call(tool_call: dict[str, Any]) -> dict[str, Any] | None:
    if not isinstance(tool_call, dict):
        return None

    function = tool_call.get("function")
    if not isinstance(function, dict):
        return None

    name = function.get("name")
    if not name:
        return None

    call_id = tool_call.get("id") or tool_call.get("call_id") or f"call_{uuid4().hex}"
    return {
        "id": call_id,
        "type": "function",
        "function": {
            "name": name,
            "arguments": _json_arguments(function.get("arguments")),
        },
    }


def _message_tool_calls(message: dict[str, Any]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for tool_call in message.get("tool_calls") or []:
        normalized_tool_call = _normalized_tool_call(tool_call)
        if normalized_tool_call is not None:
            normalized.append(normalized_tool_call)
    return normalized


def _normalize_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized_tools = []
    for tool in tools:
        if "function" in tool and isinstance(tool["function"], dict):
            function = deepcopy(tool["function"])
        else:
            function = {k: deepcopy(v) for k, v in tool.items() if k not in ("type", "strict")}

        normalized_tools.append(
            {
                "type": "function",
                "strict": bool(tool.get("strict", True)),
                **function,
            }
        )
    return normalized_tools


def _chat_messages_to_responses_input(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    responses_input: list[dict[str, Any]] = []
    for message in messages:
        role = message.get("role")
        if role in {"system", "user", "developer"}:
            responses_input.append(
                {
                    "role": role,
                    "content": _content_to_text(message.get("content")),
                    "type": "message",
                }
            )
        elif role == "tool":
            tool_call_id = message.get("tool_call_id") or message.get("call_id") or message.get("id")
            if not tool_call_id:
                raise ValueError(f"Tool message is missing a tool_call_id: {message}")
            responses_input.append(
                {
                    "type": "function_call_output",
                    "call_id": tool_call_id,
                    "output": _content_to_text(message.get("content")),
                }
            )
        elif role == "assistant":
            reasoning = message.get("reasoning_content") or message.get("reasoning")
            if isinstance(reasoning, str) and reasoning.strip():
                responses_input.append(
                    {
                        "id": f"rs_{uuid4().hex}",
                        "summary": [
                            {
                                "text": reasoning,
                                "type": "summary_text",
                            }
                        ],
                        "type": "reasoning",
                        "status": "completed",
                    }
                )

            content = message.get("content")
            tool_calls = _message_tool_calls(message)
            if content is not None and (content != "" or not tool_calls):
                responses_input.append(
                    {
                        "id": f"msg_{uuid4().hex}",
                        "content": [
                            {
                                "annotations": [],
                                "type": "output_text",
                                "text": _content_to_text(content),
                            }
                        ],
                        "role": "assistant",
                        "status": "completed",
                        "type": "message",
                    }
                )

            for tool_call in tool_calls:
                responses_input.append(
                    {
                        "arguments": tool_call["function"]["arguments"],
                        "call_id": tool_call["id"],
                        "name": tool_call["function"]["name"],
                        "type": "function_call",
                        "id": tool_call["id"],
                        "status": "completed",
                    }
                )
        else:
            raise ValueError(f"Unsupported message role: {role}")

    return responses_input


def _expected_action(message: dict[str, Any], allow_parallel: bool = False) -> dict[str, Any] | None:
    tool_calls = _message_tool_calls(message)
    if not tool_calls:
        return None

    expected_calls = []
    for tool_call in tool_calls:
        arguments = tool_call["function"]["arguments"]
        try:
            json.loads(arguments)
        except (json.JSONDecodeError, TypeError, UnicodeDecodeError):
            return None

        expected_calls.append(
            {
                "type": "function_call",
                "name": tool_call["function"]["name"],
                "arguments": arguments,
            }
        )

    if len(expected_calls) == 1:
        return expected_calls[0]

    return None


def _assistant_index_by_message(messages: list[dict[str, Any]]) -> dict[int, int]:
    indices: dict[int, int] = {}
    assistant_index = 0
    for message_index, message in enumerate(messages):
        if message.get("role") == "assistant":
            indices[message_index] = assistant_index
            assistant_index += 1
    return indices


def _meta_info(prefix_messages: list[dict[str, Any]]) -> dict[str, int]:
    turn = 0
    step = 0
    assistant_depth = 0
    for message in prefix_messages:
        role = message.get("role")
        if role == "user":
            turn += 1
            step = 0
        elif role == "assistant":
            step += 1
            assistant_depth += 1

    target_step = step + 1
    target_depth = assistant_depth + 1
    return {
        "turn": turn,
        "step": target_step,
        "assistant_depth": target_depth,
        "depth": target_depth,
    }


def _previous_step_reasoning(prefix_messages: list[dict[str, Any]]) -> tuple[int, int, bool]:
    previous_steps = 0
    previous_steps_with_reasoning = 0
    for message in reversed(prefix_messages):
        role = message.get("role")
        if role == "user":
            break
        if role != "assistant":
            continue
        previous_steps += 1
        if message.get("reasoning_content") or message.get("reasoning"):
            previous_steps_with_reasoning += 1
    return previous_steps, previous_steps_with_reasoning, previous_steps_with_reasoning > 0


def _trajectory_stats(messages: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "trajectory_num_messages": len(messages),
        "trajectory_num_assistant_actions": sum(1 for message in messages if message.get("role") == "assistant"),
        "trajectory_num_tool_messages": sum(1 for message in messages if message.get("role") == "tool"),
        "trajectory_num_tool_calls": sum(
            len(message.get("tool_calls") or []) for message in messages if message.get("role") == "assistant"
        ),
    }


def _make_responses_create_params(
    prefix_messages: list[dict[str, Any]], tools: list[dict[str, Any]]
) -> dict[str, Any]:
    return {
        "input": _chat_messages_to_responses_input(prefix_messages),
        "tools": _normalize_tools(tools),
        "parallel_tool_calls": True,
        "tool_choice": "auto",
    }


def _make_pivot_row(
    source_row: dict[str, Any],
    trajectory_id: int,
    message_index: int,
    agent_ref: dict[str, str],
    allow_parallel: bool = False,
) -> dict[str, Any] | None:
    messages = source_row["messages"]
    target_message = messages[message_index]
    expected_action = _expected_action(target_message, allow_parallel=allow_parallel)
    if expected_action is None:
        return None

    prefix_messages = messages[:message_index]
    assistant_indices = _assistant_index_by_message(messages)
    previous_steps, previous_steps_with_reasoning, previous_steps_have_reasoning = _previous_step_reasoning(
        prefix_messages
    )
    tool_calls = _message_tool_calls(target_message)
    target_content = _content_to_text(target_message.get("content"))
    target_reasoning = target_message.get("reasoning_content") or target_message.get("reasoning") or ""

    return {
        "trajectory_id": trajectory_id,
        "responses_create_params": _make_responses_create_params(prefix_messages, source_row.get("tools") or []),
        "expected_action": expected_action,
        "scenario": None,
        "meta_info": _meta_info(prefix_messages),
        "tool_use_info": {
            "uuid": source_row.get("uuid"),
            "low_effort": source_row.get("low_effort"),
            "metadata": deepcopy(source_row.get("metadata")),
            "source_message_index": message_index,
            "source_assistant_index": assistant_indices[message_index],
            "pivot_tool_call_id": tool_calls[0]["id"],
            "pivot_tool_name": tool_calls[0]["function"]["name"],
            "pivot_tool_call_ids": [tool_call["id"] for tool_call in tool_calls],
            "pivot_tool_names": [tool_call["function"]["name"] for tool_call in tool_calls],
            "pivot_tool_call_count": len(tool_calls),
            "pivot_has_content": bool(target_content),
            "pivot_content": target_content,
            "pivot_has_reasoning": bool(target_reasoning),
            "pivot_reasoning_content": target_reasoning,
            "previous_steps_in_turn": previous_steps,
            "previous_steps_in_turn_with_reasoning": previous_steps_with_reasoning,
            "previous_steps_in_turn_have_reasoning": previous_steps_have_reasoning,
            **_trajectory_stats(messages),
        },
        "agent_ref": agent_ref,
    }


def _make_skip_record(source_row: dict[str, Any], trajectory_id: int, message_index: int) -> dict[str, Any]:
    messages = source_row["messages"]
    target_message = messages[message_index]
    assistant_indices = _assistant_index_by_message(messages)
    tool_calls = _message_tool_calls(target_message)
    return {
        "trajectory_id": trajectory_id,
        "uuid": source_row.get("uuid"),
        "source_message_index": message_index,
        "source_assistant_index": assistant_indices[message_index],
        "reason": "multi_tool_target",
        "tool_call_count": len(tool_calls),
        "tool_names": [tool_call["function"]["name"] for tool_call in tool_calls],
        "tool_calls": tool_calls,
        "metadata": deepcopy(source_row.get("metadata")),
    }


def _iter_jsonl(path: Path):
    with path.open() as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            yield line_number, json.loads(line)


def _pct(count: int, total: int) -> str:
    if total == 0:
        return "0.00%"
    return f"{count / total:.2%}"


def _count_pct(count: int, total: int) -> str:
    return f"{count} / {total} ({_pct(count, total)})"


def _markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    rendered = [[str(cell) for cell in row] for row in rows]
    widths = [
        max(len(headers[idx]), *(len(row[idx]) for row in rendered)) if rendered else len(headers[idx])
        for idx in range(len(headers))
    ]

    def render(row: list[str]) -> str:
        return "| " + " | ".join(row[idx].ljust(widths[idx]) for idx in range(len(row))) + " |"

    return "\n".join(
        [
            render(headers),
            "| " + " | ".join("-" * width for width in widths) + " |",
            *(render(row) for row in rendered),
        ]
    )


def _top_counter_rows(metrics: Counter, prefix: str, total: int, limit: int = 25) -> list[list[Any]]:
    rows = [
        [value, count, _pct(count, total)]
        for key, count in metrics.items()
        if isinstance(key, tuple) and len(key) == 2 and key[0] == prefix
        for value in [key[1]]
    ]
    return sorted(rows, key=lambda row: (-int(row[1]), str(row[0])))[:limit]


def _write_metrics(
    metrics_path: Path, info_path: Path, metrics: Counter, elapsed: float, args: argparse.Namespace
) -> None:
    total_pivots = metrics["pivot_rows_written"]
    total_candidates = (
        total_pivots
        + metrics["skipped_multi_tool_targets"]
        + metrics["skipped_single_tool_targets"]
        + metrics["malformed_expected_actions"]
    )
    tool_distribution = {
        str(key[1]): count
        for key, count in metrics.items()
        if isinstance(key, tuple) and len(key) == 2 and key[0] == "expected_tool"
    }
    action_type_distribution = {
        str(key[1]): count
        for key, count in metrics.items()
        if isinstance(key, tuple) and len(key) == 2 and key[0] == "expected_action_type"
    }
    batch_call_count_distribution = {
        str(key[1]): count
        for key, count in metrics.items()
        if isinstance(key, tuple) and len(key) == 2 and key[0] == "batch_call_count"
    }
    batch_tool_names_distribution = {
        str(key[1]): count
        for key, count in metrics.items()
        if isinstance(key, tuple) and len(key) == 2 and key[0] == "batch_tool_names"
    }
    turn_step_distribution = {
        str(key[1]): count
        for key, count in metrics.items()
        if isinstance(key, tuple) and len(key) == 2 and key[0] == "turn_step"
    }
    assistant_depth_distribution = {
        str(key[1]): count
        for key, count in metrics.items()
        if isinstance(key, tuple) and len(key) == 2 and key[0] == "assistant_depth"
    }

    metrics_json = {
        "input": str(args.in_path),
        "output": str(args.out_path),
        "agent_ref": args.agent_ref,
        "dataset_mode": args.dataset_mode,
        "elapsed_seconds": elapsed,
        "counts": {str(key): value for key, value in metrics.items() if not isinstance(key, tuple)},
        "action_type_distribution": action_type_distribution,
        "tool_distribution": tool_distribution,
        "batch_call_count_distribution": batch_call_count_distribution,
        "batch_tool_names_distribution": batch_tool_names_distribution,
        "turn_step_distribution": turn_step_distribution,
        "assistant_depth_distribution": assistant_depth_distribution,
    }
    metrics_path.write_text(json.dumps(metrics_json, indent=2) + "\n")

    summary_rows = [
        ["input", args.in_path],
        ["output", args.out_path],
        ["skipped_multi_tool_output", args.skipped_multi_tool_path],
        ["config_output", args.config_out_path],
        ["agent_ref", args.agent_ref],
        ["dataset_mode", args.dataset_mode],
        ["source_trajectories_seen", metrics["source_trajectories_seen"]],
        ["assistant_turn_candidates", total_candidates],
        ["pivot_rows_written", metrics["pivot_rows_written"]],
        ["single_call_rows", _count_pct(metrics["expected_action_type", "function_call"], total_pivots)],
        ["skipped_single_tool_targets", _count_pct(metrics["skipped_single_tool_targets"], total_candidates)],
        ["skipped_multi_tool_targets", _count_pct(metrics["skipped_multi_tool_targets"], total_candidates)],
        ["malformed_expected_actions", _count_pct(metrics["malformed_expected_actions"], total_candidates)],
        ["pivots_with_target_content", _count_pct(metrics["pivots_with_target_content"], total_pivots)],
        ["pivots_with_target_reasoning", _count_pct(metrics["pivots_with_target_reasoning"], total_pivots)],
        ["finish_expected_actions", _count_pct(metrics["expected_tool", "finish"], total_pivots)],
        ["elapsed_seconds", f"{elapsed:.2f}"],
    ]

    sections = [
        "# tool-use Pivot Dataset Metrics",
        "## Summary\n\n" + _markdown_table(["metric", "value"], summary_rows),
        "## Expected Action Type Distribution\n\n"
        + _markdown_table(
            ["expected_action_type", "count", "pct_of_pivots"],
            _top_counter_rows(metrics, "expected_action_type", total_pivots),
        ),
        "## Expected Tool Distribution\n\n"
        + _markdown_table(
            ["tool", "count", "pct_of_pivots"], _top_counter_rows(metrics, "expected_tool", total_pivots)
        ),
        "## Turn Step Distribution\n\n"
        + _markdown_table(
            ["turn_step", "count", "pct_of_pivots"], _top_counter_rows(metrics, "turn_step", total_pivots)
        ),
        "## Assistant Depth Distribution\n\n"
        + _markdown_table(
            ["assistant_depth", "count", "pct_of_pivots"],
            _top_counter_rows(metrics, "assistant_depth", total_pivots),
        ),
    ]
    info_path.write_text("\n\n".join(sections) + "\n")


def _write_config(config_path: Path, out_path: Path, agent_ref: str) -> None:
    resource_server_name = agent_ref.replace("_agent", "_resources_server")
    config_text = f"""{resource_server_name}:
  resources_servers:
    single_step_tool_use_with_argument_comparison:
      entrypoint: app.py
      domain: agent
      tool_call_comparator_config:
        word_count_similarity_threshold: 0.0
      verified: false
      description: Generic message pivot dataset for single-step tool-use argument comparison.
{agent_ref}:
  responses_api_agents:
    tool_simulation_agent:
      entrypoint: app.py
      resources_server:
        type: resources_servers
        name: {resource_server_name}
      model_server:
        type: responses_api_models
        name: policy_model
      datasets:
      - name: train
        type: train
        jsonl_fpath: {out_path}
        num_repeats: 1
        license: TBD
"""
    config_path.write_text(config_text)


def _iter_expected_calls(expected_action: dict[str, Any]) -> list[dict[str, Any]]:
    return [expected_action]


def validate_output(path: Path, gym_repo: Path, agent_ref: str) -> Counter:
    if str(gym_repo) not in sys.path:
        sys.path.insert(0, str(gym_repo))

    from pydantic import TypeAdapter

    from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
    from resources_servers.single_step_tool_use_with_argument_comparison.common.verification_utils import (
        ExpectedAction,
    )

    expected_action_adapter = TypeAdapter(ExpectedAction)
    metrics: Counter = Counter()

    for line_number, row in _iter_jsonl(path):
        metrics["rows_validated"] += 1

        required_keys = {
            "trajectory_id",
            "responses_create_params",
            "expected_action",
            "scenario",
            "meta_info",
            "tool_use_info",
            "agent_ref",
        }
        missing_keys = required_keys - set(row)
        if missing_keys:
            raise ValueError(f"Line {line_number} is missing keys: {sorted(missing_keys)}")

        if row["agent_ref"].get("name") != agent_ref:
            raise ValueError(f"Line {line_number} has mismatched agent_ref: {row['agent_ref']}")

        responses_create_params = row["responses_create_params"]
        if responses_create_params.get("metadata") is None and "metadata" in responses_create_params:
            raise ValueError(f"Line {line_number} has responses_create_params.metadata=null")

        NeMoGymResponseCreateParamsNonStreaming(**responses_create_params)
        expected_action_adapter.validate_python(row["expected_action"])

        expected_action = row["expected_action"]
        if expected_action["type"] != "function_call":
            raise ValueError(f"Line {line_number} has unexpected action type: {expected_action['type']}")

        tools = responses_create_params.get("tools") or []
        if len(tools) != 4:
            raise ValueError(f"Line {line_number} should preserve 4 tools, found {len(tools)}")

        metrics["expected_action_type", expected_action["type"]] += 1
        expected_calls = _iter_expected_calls(expected_action)
        for expected_call in expected_calls:
            json.loads(expected_call["arguments"])
            metrics["expected_tool", expected_call["name"]] += 1

    return metrics


def convert(args: argparse.Namespace) -> Counter:
    start = time.time()
    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    args.config_out_path.parent.mkdir(parents=True, exist_ok=True)
    args.skipped_multi_tool_path.parent.mkdir(parents=True, exist_ok=True)

    agent_ref = {"type": "responses_api_agents", "name": args.agent_ref}
    metrics: Counter = Counter()

    with args.out_path.open("w") as out_f, args.skipped_multi_tool_path.open("w") as skipped_f:
        for trajectory_id, (_, source_row) in enumerate(_iter_jsonl(args.in_path)):
            metrics["source_trajectories_seen"] += 1
            messages = source_row.get("messages") or []
            if not isinstance(messages, list):
                raise ValueError(f"Source row {trajectory_id} has invalid messages")

            for message_index, message in enumerate(messages):
                if message.get("role") != "assistant":
                    continue

                tool_calls = _message_tool_calls(message)
                if args.dataset_mode == "single_only" and len(tool_calls) > 1:
                    skipped_f.write(json.dumps(_make_skip_record(source_row, trajectory_id, message_index)) + "\n")
                    metrics["skipped_multi_tool_targets"] += 1
                    metrics["skipped_multi_tool_count", len(tool_calls)] += 1
                    continue
                if len(tool_calls) == 0:
                    metrics["skipped_no_tool_targets"] += 1
                    continue

                pivot_row = _make_pivot_row(
                    source_row,
                    trajectory_id,
                    message_index,
                    agent_ref,
                    allow_parallel=False,
                )
                if pivot_row is None:
                    metrics["malformed_expected_actions"] += 1
                    continue

                out_f.write(json.dumps(pivot_row) + "\n")
                metrics["pivot_rows_written"] += 1
                expected_action = pivot_row["expected_action"]
                expected_calls = _iter_expected_calls(expected_action)
                metrics["expected_action_type", expected_action["type"]] += 1
                for expected_call in expected_calls:
                    metrics["expected_tool", expected_call["name"]] += 1
                metrics["turn_step", (pivot_row["meta_info"]["turn"], pivot_row["meta_info"]["step"])] += 1
                metrics["assistant_depth", pivot_row["meta_info"]["assistant_depth"]] += 1
                if pivot_row["tool_use_info"]["pivot_has_content"]:
                    metrics["pivots_with_target_content"] += 1
                if pivot_row["tool_use_info"]["pivot_has_reasoning"]:
                    metrics["pivots_with_target_reasoning"] += 1

                if 0 < args.limit <= metrics["pivot_rows_written"]:
                    break
            if 0 < args.limit <= metrics["pivot_rows_written"]:
                break

    _write_config(args.config_out_path, args.out_path, args.agent_ref)
    elapsed = time.time() - start
    _write_metrics(args.metrics_path, args.info_path, metrics, elapsed, args)
    return metrics


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in-path", type=Path, default=DEFAULT_IN_PATH)
    parser.add_argument("--out-path", type=Path, default=DEFAULT_OUT_PATH)
    parser.add_argument("--agent-ref", default=DEFAULT_AGENT_REF)
    parser.add_argument(
        "--dataset-mode",
        choices=["single_only"],
        default="single_only",
        help="Which assistant tool turns to export",
    )
    parser.add_argument("--config-out-path", type=Path)
    parser.add_argument("--metrics-path", type=Path)
    parser.add_argument("--info-path", type=Path)
    parser.add_argument("--skipped-multi-tool-path", type=Path)
    parser.add_argument("--limit", type=int, default=-1, help="Maximum pivot rows to write")
    parser.add_argument(
        "--validate", action="store_true", help="Validate output rows against Gym models after writing"
    )
    parser.add_argument("--validate-only", action="store_true", help="Validate --out-path without rewriting it")
    parser.add_argument("--gym-repo", type=Path, default=DEFAULT_GYM_REPO)
    args = parser.parse_args()

    args.config_out_path = args.config_out_path or args.out_path.with_name(
        "tool_use_single_step_tool_use_with_argument_comparison.yaml"
    )
    args.metrics_path = args.metrics_path or args.out_path.with_suffix(".metrics.json")
    args.info_path = args.info_path or args.out_path.with_suffix(".info.md")
    args.skipped_multi_tool_path = args.skipped_multi_tool_path or args.out_path.with_suffix(
        ".skipped_multi_tool.jsonl"
    )
    return args


def main() -> None:
    args = _parse_args()

    if args.validate_only:
        validation_metrics = validate_output(args.out_path, args.gym_repo, args.agent_ref)
        print(f"Validated rows: {validation_metrics['rows_validated']}")
        return

    metrics = convert(args)
    print(f"Wrote pivot rows: {args.out_path}")
    print(f"Wrote skipped multi-tool audit: {args.skipped_multi_tool_path}")
    print(f"Wrote metrics: {args.metrics_path}")
    print(f"Wrote report: {args.info_path}")
    print(f"Wrote config: {args.config_out_path}")
    print(f"Source trajectories: {metrics['source_trajectories_seen']}")
    print(f"Pivot rows: {metrics['pivot_rows_written']}")
    print(f"Single-call rows: {metrics['expected_action_type', 'function_call']}")
    print(f"Skipped multi-tool targets: {metrics['skipped_multi_tool_targets']}")

    if args.validate:
        validation_metrics = validate_output(args.out_path, args.gym_repo, args.agent_ref)
        print(f"Validated rows: {validation_metrics['rows_validated']}")


if __name__ == "__main__":
    main()
