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
"""Reference script: convert chat-style rollout trajectories into pivot-step training samples.

This is a copied example from a concrete dataset conversion. It may contain
dataset-specific imports, filters, and assumptions, and should be adapted
before reuse.

Smart filtering using error profiles from analyze_rollouts.py:
  - Reward filtering (default: only reward=1.0 trajectories)
  - Hallucinated tool calls (session termination, wrong framework, misspellings)
  - Tool error responses (per-tool regex classification)
  - Doom loops (consecutive same-tool streaks — only keep the last in each streak)

Output format matches single_step_tool_use_with_argument_comparison Gym env.
"""

import argparse
import json
import os
import re
from collections import Counter, defaultdict
from copy import deepcopy
from multiprocessing import Pool
from uuid import uuid4

from analyze_rollouts import (
    CONTENT_RETURNING_TOOLS,
    _classify_tool,
    _match_tool_response,
)
from tqdm import tqdm


# ---------------------------------------------------------------------------
# CC -> responses_create_params / expected_action
# Reference conversion pattern from a prior tool-use dataset.
# ---------------------------------------------------------------------------


def chat_completion_create_params_to_responses_create_params(chat_completions_create_params):
    ccp = chat_completions_create_params
    rcp = {}

    ccp_messages = ccp["messages"]
    rcp_messages = []
    for i in range(len(ccp_messages)):
        ccp_message = ccp_messages[i]
        role = ccp_message["role"]
        if role in ["system", "user"]:
            rcp_messages.append(deepcopy(ccp_message))
        elif role == "tool":
            tool_response = {"type": "function_call_output", "output": ccp_message.get("content")}
            if ccp_message.get("tool_call_id"):
                tool_response["call_id"] = ccp_message.get("tool_call_id", "")
            else:
                tool_response["call_id"] = f"rs_{uuid4().hex}"
            rcp_messages.append(tool_response)
        elif role == "assistant":
            if "reasoning_content" in ccp_message and ccp_message["reasoning_content"]:
                rcp_messages.append(
                    {
                        "id": f"rs_{uuid4().hex}",
                        "summary": [{"text": ccp_message["reasoning_content"], "type": "summary_text"}],
                        "type": "reasoning",
                        "status": "completed",
                    }
                )
            if (
                ccp_message.get("content")
                or ccp_message.get("content") == ""
                or ((not ccp_message.get("reasoning_content")) and (not ccp_message.get("tool_calls")))
            ):
                content = ccp_message["content"] if ccp_message.get("content") else ""
                rcp_messages.append(
                    {
                        "id": f"msg_{uuid4().hex}",
                        "content": [{"annotations": [], "type": "output_text", "text": content}],
                        "role": "assistant",
                        "status": "completed",
                        "type": "message",
                    }
                )
            if ccp_message.get("tool_calls"):
                for tc in ccp_message["tool_calls"]:
                    tool_call = {
                        "arguments": tc["function"]["arguments"],
                        "call_id": tc["id"],
                        "name": tc["function"]["name"],
                        "type": "function_call",
                        "id": tc["id"],
                        "status": "completed",
                    }
                    if isinstance(tool_call["arguments"], dict):
                        tool_call["arguments"] = json.dumps(tool_call["arguments"])
                    rcp_messages.append(tool_call)
        else:
            raise NotImplementedError

    # Handle both nested format ({"type": "function", "function": {...}})
    # and flat format ({"name": ..., "parameters": ...})
    def extract_tool_function(tool):
        if "function" in tool:
            func = tool["function"]
        else:
            # Tool is already in flat format
            func = tool
        # Remove 'type' and 'strict' to avoid conflicts, we'll set them explicitly
        return {k: v for k, v in func.items() if k not in ("type", "strict")}

    tools = [dict(type="function", strict=True, **extract_tool_function(tool)) for tool in ccp["tools"]]

    rcp = {
        "input": rcp_messages,
        "tools": tools,
        "parallel_tool_calls": True,
    }
    return rcp


def chat_completion_message_to_expected_action(chat_completion_message):
    """Returns None if arguments are malformed JSON (drops the sample)."""
    if chat_completion_message.get("tool_calls"):
        tc = chat_completion_message.get("tool_calls")[0]
        expected_action = {
            "type": "function_call",
            "name": tc["function"]["name"],
            "arguments": tc["function"]["arguments"],
        }
        if isinstance(tc["function"]["arguments"], dict):
            expected_action["arguments"] = json.dumps(tc["function"]["arguments"])
        try:
            json.loads(expected_action["arguments"])
        except (json.JSONDecodeError, TypeError):
            return None
    else:
        expected_action = {"type": "message", "content": chat_completion_message["content"]}
    return expected_action


# ---------------------------------------------------------------------------
# Per-pivot filtering
# ---------------------------------------------------------------------------


def is_hallucinated_tool(msg: dict) -> bool:
    """Check if the assistant message calls a non-valid tool."""
    for tc in msg.get("tool_calls") or []:
        name = tc.get("function", {}).get("name", "")
        if _classify_tool(name) != "valid":
            return True
    return False


def is_tool_error(messages: list[dict], assistant_idx: int, msg: dict) -> bool:
    """Check if any tool call in this assistant message got an error response."""
    for tc in msg.get("tool_calls") or []:
        tool_name = tc.get("function", {}).get("name", "")
        tc_id = tc.get("id", "")
        resp = _match_tool_response(messages, assistant_idx, tc_id)
        if resp is None:
            continue

        # Use the same detection logic as analyze_rollouts
        if tool_name in CONTENT_RETURNING_TOOLS:
            is_error = bool(re.match(r"^Error:", resp))
        else:
            resp_lower = resp.lower()
            is_error = any(kw in resp_lower for kw in ("error", "not found", "invalid", "denied", "failed"))

        if is_error:
            return True
    return False


def compute_doom_loop_mask(messages: list[dict], threshold: int) -> list[bool]:
    """For each assistant message, decide if it should be kept based on doom loop logic.

    In a consecutive same-tool streak of length >= threshold, only keep the last one.
    Returns a list parallel to messages: True = keep, False = skip.
    """
    n = len(messages)
    keep = [True] * n

    # Find assistant indices and their tool names
    assistant_info = []  # (msg_index, tool_name_or_None)
    for i, m in enumerate(messages):
        if m.get("role") == "assistant":
            tool_name = None
            if m.get("tool_calls"):
                tool_name = m["tool_calls"][0].get("function", {}).get("name", "")
            assistant_info.append((i, tool_name))

    # Walk through assistant turns finding streaks
    j = 0
    while j < len(assistant_info):
        idx, tool = assistant_info[j]
        if tool is None:
            j += 1
            continue

        # Find extent of streak
        streak_start = j
        while j + 1 < len(assistant_info) and assistant_info[j + 1][1] == tool:
            j += 1
        streak_end = j  # inclusive
        streak_len = streak_end - streak_start + 1

        if streak_len >= threshold:
            # Mark all but the last in the streak as skip
            for k in range(streak_start, streak_end):
                keep[assistant_info[k][0]] = False

        j += 1

    return keep


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def extract_and_filter(args_tuple):
    """Worker: extract pivots from one trajectory with all filters applied."""
    traj_idx, record, filter_cfg, agent_ref = args_tuple
    messages = record["messages"]
    tools = record.get("tools", [])

    skips = defaultdict(int)
    samples = []

    # Precompute doom loop mask
    doom_threshold = filter_cfg.get("doom_loop_threshold", 3)
    if doom_threshold > 0:
        doom_keep = compute_doom_loop_mask(messages, doom_threshold)
    else:
        doom_keep = [True] * len(messages)

    for i, msg in enumerate(messages):
        if msg.get("role") != "assistant":
            continue

        # Skip chat-only
        if filter_cfg.get("skip_chat") and not msg.get("tool_calls"):
            skips["chat"] += 1
            continue

        # Doom loop filter
        if not doom_keep[i]:
            skips["doom_loop"] += 1
            continue

        # Hallucinated tool filter
        if filter_cfg.get("filter_hallucinated") and msg.get("tool_calls") and is_hallucinated_tool(msg):
            skips["hallucinated_tool"] += 1
            continue

        # Tool error filter
        if filter_cfg.get("filter_tool_errors") and msg.get("tool_calls") and is_tool_error(messages, i, msg):
            skips["tool_error"] += 1
            continue

        samples.append(
            {
                "traj_idx": traj_idx,
                "messages": messages[:i],
                "answer": msg,
                "tools": tools,
                "agent_ref": agent_ref,
            }
        )

    return traj_idx, samples, dict(skips)


def _flatten_list_content(messages):
    """Ensure system/user content is a string, not a list of blocks.

    chat-style data has user content as [{"type": "text", "text": "..."}] but the
    Gym server's NeMoGymEasyInputMessage expects str or input_text blocks.
    Safest to flatten to string before conversion.
    """
    for m in messages:
        if m.get("role") in ("system", "user") and isinstance(m.get("content"), list):
            parts = []
            for block in m["content"]:
                if isinstance(block, dict):
                    parts.append(block.get("text", ""))
                elif isinstance(block, str):
                    parts.append(block)
            m["content"] = "\n".join(parts)


def process_sample(sample: dict) -> dict:
    messages = deepcopy(sample["messages"])
    answer = sample["answer"]
    tools = sample["tools"]

    _flatten_list_content(messages)

    depth = sum(1 for m in messages if m.get("role") == "assistant") + 1

    rcp = chat_completion_create_params_to_responses_create_params({"messages": messages, "tools": tools})
    expected_action = chat_completion_message_to_expected_action(answer)
    if expected_action is None:
        return None

    turn, step = 0, 0
    for m in messages:
        if m.get("role") == "user":
            turn += 1
            step = 0
        elif m.get("role") == "assistant":
            step += 1
    step += 1

    return {
        "trajectory_id": sample["traj_idx"],
        "info": {"turn": turn, "step": step, "depth": depth},
        "responses_create_params": rcp,
        "expected_action": expected_action,
        "agent_ref": sample["agent_ref"],
    }


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def compute_metrics(rows):
    turn_step = []
    depth_info = []
    num_tool = num_chat = 0
    for r in rows:
        turn_step.append((r["info"]["turn"], r["info"]["step"]))
        depth_info.append(r["info"]["depth"])
        if r["expected_action"]["type"] == "function_call":
            num_tool += 1
        else:
            num_chat += 1
    ts_cnt = Counter(turn_step)
    d_cnt = Counter(depth_info)
    total = len(rows)
    return {
        "total_samples": total,
        "turn_step_distribution": [{"turn": k[0], "step": k[1], "count": v} for k, v in sorted(ts_cnt.items())],
        "depth_distribution": {k: v for k, v in sorted(d_cnt.items())},
        "answer_distribution": {"tool": num_tool, "chat": num_chat},
    }


def format_metrics(metrics, label=""):
    lines = []
    if label:
        lines.append(f"=== {label} ===")
    total = metrics["total_samples"]
    if total == 0:
        lines.append("(no samples)")
        return "\n".join(lines)
    lines.append(f"\nTotal samples: {total}")
    lines.append("\nDistribution (turn, step)")
    for item in metrics["turn_step_distribution"]:
        pct = 100.0 * item["count"] / total
        lines.append(f"  (turn={item['turn']}, step={item['step']:<2}) : {item['count']:>6}  ({pct:6.2f}%)")
    lines.append("\nDistribution (depth)")
    for d, c in metrics["depth_distribution"].items():
        pct = 100.0 * c / total
        lines.append(f"  depth={d:<10} : {c:>6}  ({pct:6.2f}%)")
    lines.append("\nAction type")
    nt = metrics["answer_distribution"]["tool"]
    nc = metrics["answer_distribution"]["chat"]
    lines.append(f"  Tool : {nt:>6}  ({100.0 * nt / total:6.2f}%)")
    lines.append(f"  Chat : {nc:>6}  ({100.0 * nc / total:6.2f}%)")
    return "\n".join(lines)


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(args):
    print(f"Reading {args.in_path} ...")
    records = []
    with open(args.in_path) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    print(f"  {len(records)} trajectories")

    # Reward filter
    if args.min_reward is not None:
        before = len(records)
        records = [r for r in records if (r.get("metadata", {}).get("reward") or 0) >= args.min_reward]
        print(f"  Reward filter (>= {args.min_reward}): {before} -> {len(records)}")

    filter_cfg = {
        "skip_chat": args.skip_chat,
        "filter_hallucinated": not args.keep_hallucinated_tools,
        "filter_tool_errors": not args.keep_tool_errors,
        "doom_loop_threshold": 0 if args.keep_doom_loops else args.doom_loop_threshold,
    }

    # Extract pivots
    print("Extracting pivots ...")
    num_workers = args.num_workers if args.num_workers > 0 else os.cpu_count()
    agent_ref = {"type": "responses_api_agents", "name": args.agent_ref}
    inputs = [(idx, rec, filter_cfg, agent_ref) for idx, rec in enumerate(records)]

    with Pool(num_workers) as pool:
        results = list(
            tqdm(
                pool.imap(extract_and_filter, inputs, chunksize=max(1, len(records) // (num_workers * 4))),
                total=len(records),
                desc="Extracting",
            )
        )

    # Aggregate skips
    total_skips = defaultdict(int)
    for _, _, skips in results:
        for k, v in skips.items():
            total_skips[k] += v
    if total_skips:
        print(f"  Skips: {json.dumps(dict(total_skips))}")

    # Flatten and optionally assign splits
    results.sort(key=lambda x: x[0])
    samples = []
    traj_cnt = 0
    for _, traj_samples, _ in results:
        if not traj_samples:
            continue
        if args.val_size > 0:
            split = "val" if traj_cnt < args.val_size else "train"
        else:
            split = "train"
        traj_cnt += 1
        for s in traj_samples:
            s["_split"] = split
            samples.append(s)
            if 0 < args.limit <= len(samples):
                break
        if 0 < args.limit <= len(samples):
            break
    print(f"  {len(samples)} pivot samples from {traj_cnt} trajectories")

    # Process
    print("Processing ...")
    with Pool(num_workers) as pool:
        processed = list(
            tqdm(
                pool.imap(process_sample, samples, chunksize=max(1, len(samples) // (num_workers * 4))),
                total=len(samples),
                desc="Processing",
            )
        )

    dropped = sum(1 for r in processed if r is None)
    if dropped:
        print(f"  Dropped {dropped} samples with malformed arguments")

    train_rows, val_rows = [], []
    train_ids, val_ids = set(), set()
    for sample, row in zip(samples, processed):
        if row is None:
            continue
        if sample["_split"] == "val":
            val_rows.append(row)
            val_ids.add(row["trajectory_id"])
        else:
            train_rows.append(row)
            train_ids.add(row["trajectory_id"])

    print(f"\n  Trajectories: {len(train_ids) + len(val_ids)} (train={len(train_ids)}, val={len(val_ids)})")
    print(f"  Samples:      {len(train_rows) + len(val_rows)} (train={len(train_rows)}, val={len(val_rows)})")

    # Write
    base = args.out_path.rsplit(".", 1)[0]
    if args.val_size > 0:
        train_path = f"{base}-train.jsonl"
        val_path = f"{base}-val.jsonl"
        write_jsonl(train_path, train_rows)
        write_jsonl(val_path, val_rows)
        print(f"\n  Wrote {train_path}")
        print(f"  Wrote {val_path}")
    else:
        write_jsonl(args.out_path, train_rows)
        print(f"\n  Wrote {args.out_path}")

    debug_path = f"{base}-debug.json"
    with open(debug_path, "w") as f:
        json.dump(train_rows[:20], f, indent=2)
    print(f"  Wrote {debug_path}")

    # Metrics
    train_metrics = compute_metrics(train_rows)
    val_metrics = compute_metrics(val_rows)
    all_metrics = compute_metrics(train_rows + val_rows)

    metrics_json = {
        "config": {
            "input": args.in_path,
            "min_reward": args.min_reward,
            "doom_loop_threshold": args.doom_loop_threshold,
            "filter_hallucinated": not args.keep_hallucinated_tools,
            "filter_tool_errors": not args.keep_tool_errors,
            "skip_chat": args.skip_chat,
            "limit": args.limit,
            "val_size": args.val_size,
        },
        "skips": dict(total_skips),
        "trajectories": {"total": len(train_ids) + len(val_ids), "train": len(train_ids), "val": len(val_ids)},
        "train": train_metrics,
        "val": val_metrics,
        "all": all_metrics,
    }
    metrics_json_path = f"{base}-metrics.json"
    with open(metrics_json_path, "w") as f:
        json.dump(metrics_json, f, indent=2)

    report = "\n".join(
        [
            format_metrics(train_metrics, "TRAIN"),
            "",
            format_metrics(val_metrics, "VAL"),
            "",
            format_metrics(all_metrics, "ALL"),
        ]
    )
    print("\n" + report)

    metrics_txt_path = f"{base}.txt"
    with open(metrics_txt_path, "w") as f:
        f.write(report)
    print(f"\n  Wrote {metrics_txt_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-f", "--in-path", required=True)
    p.add_argument("-o", "--out-path", required=True)
    p.add_argument(
        "--min-reward",
        type=float,
        default=1.0,
        help="Only keep trajectories with metadata.reward >= this (default: 1.0)",
    )
    p.add_argument("--no-reward-filter", action="store_true", help="Disable reward filtering")
    p.add_argument(
        "--doom-loop-threshold", type=int, default=3, help="Min streak length to trigger doom-loop filter (default: 3)"
    )
    p.add_argument("--keep-doom-loops", action="store_true", help="Disable doom loop filtering")
    p.add_argument("--keep-hallucinated-tools", action="store_true", help="Keep pivots with hallucinated tools")
    p.add_argument("--keep-tool-errors", action="store_true", help="Keep pivots where tool returned an error")
    p.add_argument(
        "--agent-ref",
        default="generic_pivot_single_step_tool_use_with_argument_comparison_agent",
        help="Agent name for agent_ref field (default: generic_pivot_single_step_tool_use_with_argument_comparison_agent)",
    )
    p.add_argument("--skip-chat", action="store_true", help="Skip assistant turns with no tool calls")
    p.add_argument(
        "--val-size", type=int, default=0, help="Number of trajectories for val split (default: 0 = no split)"
    )
    p.add_argument("--limit", type=int, default=-1)
    p.add_argument("--num-workers", type=int, default=-1)
    args = p.parse_args()
    if args.no_reward_filter:
        args.min_reward = None
    main(args)
