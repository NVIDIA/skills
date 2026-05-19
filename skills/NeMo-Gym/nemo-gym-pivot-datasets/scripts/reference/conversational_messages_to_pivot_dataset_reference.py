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
"""Reference script: convert successful conversational trajectories into single-step pivot rows.

This is a copied example from a concrete dataset conversion. It may contain
dataset-specific assumptions and should be adapted before reuse.

The output format is for the
`single_step_tool_use_with_argument_comparison` environment. Each non-seed
assistant turn becomes one row: the prefix goes into `responses_create_params`
and the assistant turn itself becomes `expected_action`.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from copy import deepcopy
from pathlib import Path
from uuid import uuid4

from tqdm import tqdm


PIVOT_RESPONSES_CREATE_PARAM_KEYS = (
    "model",
    "parallel_tool_calls",
    "tool_choice",
    "tools",
)


def _content_to_text(content) -> str | None:
    if content is None:
        return None
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                if "text" in block:
                    parts.append(block["text"])
                elif "content" in block:
                    parts.append(str(block["content"]))
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts)
    return str(content)


def _raw_choice_message(message: dict) -> dict:
    raw_data = message.get("raw_data")
    if not isinstance(raw_data, dict):
        return {}
    choices = raw_data.get("choices") or []
    if not choices or not isinstance(choices[0], dict):
        return {}
    raw_message = choices[0].get("message") or {}
    return raw_message if isinstance(raw_message, dict) else {}


def _strip_think_tags(text: str | None) -> str:
    if not text:
        return ""
    stripped = text.strip()
    while stripped.startswith("<think>"):
        stripped = stripped[len("<think>") :].lstrip()
    while stripped.endswith("</think>"):
        stripped = stripped[: -len("</think>")].rstrip()
    return stripped


def _message_content(message: dict, role: str) -> str | None:
    raw_message = _raw_choice_message(message)
    if "content" in raw_message:
        content = _content_to_text(raw_message.get("content"))
    else:
        content = _content_to_text(message.get("content"))

    tool_calls = _message_tool_calls(message)
    if role == "assistant" and tool_calls and content == "":
        return None
    return content


def _message_reasoning(message: dict) -> str:
    raw_message = _raw_choice_message(message)
    for key in ("reasoning_content", "reasoning"):
        value = raw_message.get(key)
        if isinstance(value, str) and value.strip():
            return _strip_think_tags(value)
    return ""


def _json_arguments(arguments) -> str:
    if isinstance(arguments, str):
        return arguments
    if arguments is None:
        return "{}"
    return json.dumps(arguments)


def _normalize_tool_call(tool_call: dict) -> dict | None:
    if not isinstance(tool_call, dict):
        return None

    if isinstance(tool_call.get("function"), dict):
        function = tool_call["function"]
        name = function.get("name")
        arguments = function.get("arguments")
    else:
        name = tool_call.get("name")
        arguments = tool_call.get("arguments")

    if not name:
        return None

    call_id = tool_call.get("id") or tool_call.get("call_id") or f"call_{uuid4().hex}"
    return {
        "id": call_id,
        "type": "function",
        "function": {
            "name": name,
            "arguments": _json_arguments(arguments),
        },
    }


def _message_tool_calls(message: dict) -> list[dict]:
    raw_message = _raw_choice_message(message)
    raw_tool_calls = raw_message.get("tool_calls")
    if raw_tool_calls is None:
        raw_tool_calls = message.get("tool_calls") or []

    normalized = []
    for tool_call in raw_tool_calls or []:
        normalized_tool_call = _normalize_tool_call(tool_call)
        if normalized_tool_call is not None:
            normalized.append(normalized_tool_call)
    return normalized


def _responses_item_to_chat_message(item: dict) -> dict:
    role = item.get("role")
    if role not in {"system", "user", "assistant", "developer"}:
        raise ValueError(f"Cannot convert input item to chat message: {item}")
    return {
        "role": role,
        "content": _content_to_text(item.get("content")) or "",
    }


def _seed_chat_messages(responses_create_params: dict) -> list[dict]:
    input_items = responses_create_params.get("input") or []
    if len(input_items) < 2:
        raise ValueError("responses_create_params.input must contain system and initial assistant messages")
    return [
        _responses_item_to_chat_message(input_items[0]),
        _responses_item_to_chat_message(input_items[1]),
    ]


def _same_chat_message(result_message: dict, chat_message: dict) -> bool:
    role = result_message.get("role")
    content = _content_to_text(result_message.get("content")) or ""
    return role == chat_message.get("role") and content == (chat_message.get("content") or "")


def _result_message_to_chat(message: dict) -> dict | None:
    role = message.get("role")
    if role == "user":
        return {
            "role": "user",
            "content": _message_content(message, role) or "",
        }
    if role == "assistant":
        chat_message = {
            "role": "assistant",
            "content": _message_content(message, role),
        }
        reasoning = _message_reasoning(message)
        if reasoning:
            chat_message["reasoning_content"] = reasoning
        tool_calls = _message_tool_calls(message)
        if tool_calls:
            chat_message["tool_calls"] = tool_calls
        if chat_message["content"] is None and not tool_calls:
            chat_message["content"] = ""
        return chat_message
    if role == "tool":
        tool_call_id = message.get("tool_call_id") or message.get("call_id") or message.get("id")
        if not tool_call_id:
            return None
        return {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": _content_to_text(message.get("content")) or "",
        }
    return None


def _clean_result_messages(row: dict, seed_messages: list[dict], metrics: Counter) -> list[dict]:
    result_messages = row.get("result", {}).get("messages") or []
    start_idx = 0
    if result_messages and _same_chat_message(result_messages[0], seed_messages[1]):
        start_idx = 1
        metrics["skipped_seed_assistant_messages"] += 1

    cleaned = []
    dropped_user_tool_call_ids = set()
    for result_message in result_messages[start_idx:]:
        role = result_message.get("role")
        if role == "user" and result_message.get("tool_calls"):
            metrics["dropped_user_tool_call_messages"] += 1
            for tool_call in _message_tool_calls(result_message):
                dropped_user_tool_call_ids.add(tool_call["id"])
            continue

        tool_message_id = (
            result_message.get("tool_call_id") or result_message.get("call_id") or result_message.get("id")
        )
        if role == "tool" and (
            result_message.get("requestor") == "user" or tool_message_id in dropped_user_tool_call_ids
        ):
            metrics["dropped_user_tool_output_messages"] += 1
            dropped_user_tool_call_ids.discard(tool_message_id)
            continue

        chat_message = _result_message_to_chat(result_message)
        if chat_message is None:
            metrics["dropped_unconvertible_messages"] += 1
            continue
        cleaned.append(chat_message)
    return cleaned


def _chat_messages_to_responses_input(messages: list[dict]) -> list[dict]:
    responses_input = []
    for message in messages:
        role = message["role"]
        if role in {"system", "user", "developer"}:
            responses_input.append(
                {
                    "role": role,
                    "content": message.get("content") or "",
                    "type": "message",
                }
            )
        elif role == "tool":
            responses_input.append(
                {
                    "type": "function_call_output",
                    "call_id": message["tool_call_id"],
                    "output": message.get("content") or "",
                }
            )
        elif role == "assistant":
            reasoning = message.get("reasoning_content")
            tool_calls = message.get("tool_calls") or []
            content = message.get("content")
            if reasoning:
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
            if content is not None and (content != "" or not tool_calls):
                responses_input.append(
                    {
                        "id": f"msg_{uuid4().hex}",
                        "content": [
                            {
                                "annotations": [],
                                "type": "output_text",
                                "text": content,
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
            raise NotImplementedError(f"Unsupported chat role: {role}")
    return responses_input


def _expected_action(message: dict) -> dict | None:
    tool_calls = message.get("tool_calls") or []
    if tool_calls:
        tool_call = tool_calls[0]
        arguments = tool_call["function"]["arguments"]
        try:
            json.loads(arguments)
        except (json.JSONDecodeError, TypeError):
            return None
        return {
            "type": "function_call",
            "name": tool_call["function"]["name"],
            "arguments": arguments,
        }
    return {
        "type": "message",
        "content": message.get("content") or "",
    }


def _meta_info(prefix_messages: list[dict]) -> dict:
    turn = 0
    step = 0
    assistant_depth = 0
    for message in prefix_messages:
        if message["role"] == "user":
            turn += 1
            step = 0
        elif message["role"] == "assistant":
            step += 1
            assistant_depth += 1
    return {
        "turn": turn,
        "step": step,
        "assistant_depth": assistant_depth,
    }


def _task_id(row: dict) -> str:
    task = row.get("task") or {}
    result = row.get("result") or {}
    return str(task.get("id") or result.get("task_id") or "")


def _domain(row: dict) -> str:
    config = row.get("config") or {}
    return str(config.get("domain") or "unknown")


def _pivot_responses_create_params(source_rcp: dict, prefix_messages: list[dict]) -> dict:
    # Source rollouts materialize many optional Responses fields as null.
    # Pivot rows keep one fixed top-level shape and omit unset optional fields.
    responses_create_params = {"input": _chat_messages_to_responses_input(prefix_messages)}
    for key in PIVOT_RESPONSES_CREATE_PARAM_KEYS:
        value = source_rcp.get(key)
        if value is None:
            raise ValueError(f"responses_create_params.{key} must be set for pivot export")
        responses_create_params[key] = deepcopy(value)
    return responses_create_params


def _make_pivot_rows(row: dict, trajectory_id: int, agent_ref: dict, metrics: Counter) -> list[dict]:
    source_rcp = row["responses_create_params"]
    seed_messages = _seed_chat_messages(source_rcp)
    cleaned_messages = _clean_result_messages(row, seed_messages, metrics)

    trajectory_num_assistant_actions = sum(1 for message in cleaned_messages if message["role"] == "assistant")
    trajectory_num_messages = len(cleaned_messages)
    metrics["trajectory_num_assistant_actions", trajectory_num_assistant_actions] += 1
    metrics["trajectory_num_messages", trajectory_num_messages] += 1

    pivot_rows = []
    pivot_assistant_index = 0
    previous_steps_in_turn = 0
    previous_steps_in_turn_with_reasoning = 0
    for pivot_message_index, answer in enumerate(cleaned_messages):
        if answer["role"] == "user":
            previous_steps_in_turn = 0
            previous_steps_in_turn_with_reasoning = 0
            continue
        if answer["role"] != "assistant":
            continue

        pivot_has_reasoning = bool(answer.get("reasoning_content"))
        previous_steps_in_turn_have_reasoning = previous_steps_in_turn_with_reasoning > 0
        expected_action = _expected_action(answer)
        if expected_action is None:
            metrics["malformed_expected_actions"] += 1
            pivot_assistant_index += 1
            previous_steps_in_turn += 1
            if pivot_has_reasoning:
                previous_steps_in_turn_with_reasoning += 1
            continue

        prefix_messages = seed_messages + cleaned_messages[:pivot_message_index]
        responses_create_params = _pivot_responses_create_params(source_rcp, prefix_messages)

        meta_info = _meta_info(prefix_messages)

        pivot_rows.append(
            {
                "trajectory_id": trajectory_id,
                "responses_create_params": responses_create_params,
                "expected_action": expected_action,
                "scenario": None,
                "meta_info": meta_info,
                "conversation_info": {
                    "_ng_task_index": row.get("_ng_task_index"),
                    "_ng_rollout_index": row.get("_ng_rollout_index"),
                    "task_id": _task_id(row),
                    "domain": _domain(row),
                    "reward": row.get("reward"),
                    "pivot_assistant_index": pivot_assistant_index,
                    "trajectory_num_assistant_actions": trajectory_num_assistant_actions,
                    "pivot_message_index": pivot_message_index,
                    "trajectory_num_messages": trajectory_num_messages,
                    "pivot_has_reasoning": pivot_has_reasoning,
                    "previous_steps_in_turn": previous_steps_in_turn,
                    "previous_steps_in_turn_with_reasoning": previous_steps_in_turn_with_reasoning,
                    "previous_steps_in_turn_have_reasoning": previous_steps_in_turn_have_reasoning,
                },
                "agent_ref": agent_ref,
            }
        )
        pivot_assistant_index += 1
        previous_steps_in_turn += 1
        if pivot_has_reasoning:
            previous_steps_in_turn_with_reasoning += 1

    if pivot_rows:
        metrics["trajectories_with_pivots"] += 1
    return pivot_rows


def _markdown_table(headers: list[str], rows: list[list[object]]) -> str:
    rendered_rows = [[str(cell) for cell in row] for row in rows]
    if rendered_rows:
        widths = [max(len(headers[idx]), *(len(row[idx]) for row in rendered_rows)) for idx in range(len(headers))]
    else:
        widths = [len(header) for header in headers]

    def render(row: list[str]) -> str:
        return "| " + " | ".join(row[idx].ljust(widths[idx]) for idx in range(len(row))) + " |"

    lines = [
        render(headers),
        "| " + " | ".join("-" * width for width in widths) + " |",
    ]
    lines.extend(render(row) for row in rendered_rows)
    return "\n".join(lines)


def _counter_rows(metrics: Counter, prefix: str) -> list[list[object]]:
    rows = [[key[1], count] for key, count in metrics.items() if isinstance(key, tuple) and key and key[0] == prefix]
    return sorted(rows, key=lambda row: row[0])


def _pct(count: int, total: int) -> str:
    if total == 0:
        return "0.00%"
    return f"{count / total:.2%}"


def _count_pct(count: int, total: int) -> str:
    return f"{count} / {total} ({_pct(count, total)})"


def _counter_pct_rows(metrics: Counter, prefix: str, total: int) -> list[list[object]]:
    return [[value, count, _pct(count, total)] for value, count in _counter_rows(metrics, prefix)]


def _write_metrics_md(
    info_path: Path,
    in_path: Path,
    out_path: Path,
    metrics: Counter,
    elapsed: float,
    agent_ref_name: str,
) -> None:
    total_pivots = metrics["pivot_rows_written"]
    total_trajectories = metrics["trajectories_seen"]
    total_expected_action_candidates = total_pivots + metrics["malformed_expected_actions"]
    summary_rows = [
        ["input", in_path],
        ["output", out_path],
        ["agent_ref", agent_ref_name],
        ["trajectories_seen", metrics["trajectories_seen"]],
        [
            "trajectories_with_pivots",
            _count_pct(metrics["trajectories_with_pivots"], total_trajectories),
        ],
        ["pivot_rows_written", metrics["pivot_rows_written"]],
        [
            "tool_expected_actions",
            _count_pct(metrics["expected_action", "function_call"], total_pivots),
        ],
        [
            "message_expected_actions",
            _count_pct(metrics["expected_action", "message"], total_pivots),
        ],
        [
            "assistant_messages_with_reasoning",
            _count_pct(metrics["pivot_has_reasoning", True], total_pivots),
        ],
        [
            "assistant_messages_without_reasoning",
            _count_pct(metrics["pivot_has_reasoning", False], total_pivots),
        ],
        ["multistep_pivots", _count_pct(metrics["multistep_pivots"], total_pivots)],
        [
            "multistep_pivots_with_prior_step_reasoning",
            _count_pct(metrics["multistep_pivots_with_prior_step_reasoning"], total_pivots),
        ],
        ["dropped_user_tool_call_messages", metrics["dropped_user_tool_call_messages"]],
        ["dropped_user_tool_output_messages", metrics["dropped_user_tool_output_messages"]],
        [
            "malformed_expected_actions",
            _count_pct(metrics["malformed_expected_actions"], total_expected_action_candidates),
        ],
        ["elapsed_seconds", f"{elapsed:.2f}"],
    ]

    sections = [
        "# Conversational Messages Pivot Dataset Metrics",
        "## Summary\n\n" + _markdown_table(["metric", "value"], summary_rows),
        "## Turn Step Distribution\n\n"
        + _markdown_table(
            ["turn_step", "count", "pct_of_pivots"],
            _counter_pct_rows(metrics, "turn_step", total_pivots),
        ),
        "## Assistant Depth Distribution\n\n"
        + _markdown_table(
            ["assistant_depth", "count", "pct_of_pivots"],
            _counter_pct_rows(metrics, "assistant_depth", total_pivots),
        ),
        "## Reasoning Distribution\n\n"
        + _markdown_table(
            ["reasoning_metric", "count", "pct_of_pivots"],
            [
                [
                    "pivot_has_reasoning=true",
                    metrics["pivot_has_reasoning", True],
                    _pct(metrics["pivot_has_reasoning", True], total_pivots),
                ],
                [
                    "pivot_has_reasoning=false",
                    metrics["pivot_has_reasoning", False],
                    _pct(metrics["pivot_has_reasoning", False], total_pivots),
                ],
                ["multistep_pivots", metrics["multistep_pivots"], _pct(metrics["multistep_pivots"], total_pivots)],
                [
                    "multistep_pivots_with_prior_step_reasoning",
                    metrics["multistep_pivots_with_prior_step_reasoning"],
                    _pct(metrics["multistep_pivots_with_prior_step_reasoning"], total_pivots),
                ],
            ],
        ),
        "## Original Assistant Action Length\n\n"
        + _markdown_table(
            ["trajectory_num_assistant_actions", "trajectory_count", "pct_of_trajectories"],
            _counter_pct_rows(metrics, "trajectory_num_assistant_actions", total_trajectories),
        ),
        "## Clean Message Length\n\n"
        + _markdown_table(
            ["trajectory_num_messages", "trajectory_count", "pct_of_trajectories"],
            _counter_pct_rows(metrics, "trajectory_num_messages", total_trajectories),
        ),
    ]
    info_path.write_text("\n\n".join(sections) + "\n")


def convert_rollouts(args) -> Counter:
    start = time.time()
    in_path = Path(args.in_path)
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    info_path = out_path.with_suffix(".info.md")
    agent_ref = {"type": "responses_api_agents", "name": args.agent_ref}

    metrics = Counter()
    with in_path.open() as source, out_path.open("w") as target:
        for trajectory_id, line in enumerate(tqdm(source, desc="Converting rollouts", unit="traj")):
            if not line.strip():
                continue
            metrics["trajectories_seen"] += 1
            row = json.loads(line)
            pivot_rows = _make_pivot_rows(row, trajectory_id, agent_ref, metrics)
            for pivot_row in pivot_rows:
                target.write(json.dumps(pivot_row) + "\n")
                metrics["pivot_rows_written"] += 1
                metrics["expected_action", pivot_row["expected_action"]["type"]] += 1
                pivot_has_reasoning = pivot_row["conversation_info"]["pivot_has_reasoning"]
                metrics["pivot_has_reasoning", pivot_has_reasoning] += 1
                if pivot_row["conversation_info"]["previous_steps_in_turn"] > 0:
                    metrics["multistep_pivots"] += 1
                    if pivot_row["conversation_info"]["previous_steps_in_turn_have_reasoning"]:
                        metrics["multistep_pivots_with_prior_step_reasoning"] += 1
                metrics[
                    "turn_step",
                    (pivot_row["meta_info"]["turn"], pivot_row["meta_info"]["step"]),
                ] += 1
                metrics["assistant_depth", pivot_row["meta_info"]["assistant_depth"]] += 1
                metrics["pivot_message_index", pivot_row["conversation_info"]["pivot_message_index"]] += 1
                if 0 < args.limit <= metrics["pivot_rows_written"]:
                    elapsed = time.time() - start
                    _write_metrics_md(info_path, in_path, out_path, metrics, elapsed, args.agent_ref)
                    return metrics

    elapsed = time.time() - start
    _write_metrics_md(info_path, in_path, out_path, metrics, elapsed, args.agent_ref)
    return metrics


def main(args) -> None:
    metrics = convert_rollouts(args)
    print(f"Wrote pivot rows: {args.out_path}")
    print(f"Wrote metrics: {Path(args.out_path).with_suffix('.info.md')}")
    print(f"Trajectories: {metrics['trajectories_seen']}")
    print(f"Pivot rows: {metrics['pivot_rows_written']}")
    print(
        "Expected actions: "
        f"tool={metrics['expected_action', 'function_call']}, "
        f"message={metrics['expected_action', 'message']}"
    )
    print(f"Dropped user tool calls: {metrics['dropped_user_tool_call_messages']}")
    print(f"Dropped user tool outputs: {metrics['dropped_user_tool_output_messages']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in-path", required=True)
    parser.add_argument("--out-path", required=True)
    parser.add_argument(
        "--agent-ref",
        default="single_step_tool_use_with_argument_comparison_agent",
    )
    parser.add_argument("--limit", type=int, default=-1, help="Maximum pivot rows to write")
    main(parser.parse_args())
