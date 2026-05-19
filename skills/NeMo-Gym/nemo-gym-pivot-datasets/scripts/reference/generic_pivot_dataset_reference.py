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
"""Reference script: convert generic source rows into pivot rows.

This is a copied example from a concrete data pipeline. It may contain
dataset-specific imports, assumptions, and config defaults, and should be
adapted before reuse.
"""

import argparse
import os
import random
from collections import Counter
from functools import partial
from multiprocessing import Pool

from filter_training_data.utils import (
    chat_completion_create_params_to_responses_create_params,
    chat_completion_message_to_expected_action,
    count_unique_actions,
    read_jsonl,
    write_jsonl,
)
from tqdm import tqdm


def process_row(sample):
    trajectory_id = sample["trajectory_id"]
    problem = sample["problem"]
    answer = sample["expected_action"]
    tools = sample["tools"]
    responses = sample["responses"]
    rewards = sample["rewards"]
    reward_mean = sample["reward_mean"]
    reward_std = sample["reward_std"]
    reward_var = sample["reward_var"]

    for tool in tools:
        if tool["function"].get("parameters") is None:
            return None

    ccp = {"messages": problem, "tools": tools}

    num_unique_actions = count_unique_actions(responses)

    rewards = [int(x) for x in rewards]

    rcp = chat_completion_create_params_to_responses_create_params(ccp)

    expected_action = chat_completion_message_to_expected_action(answer)

    turn = 0
    step = 0
    assistant_depth = 0
    for message in problem:
        if message["role"] == "user":
            turn += 1
            step = 0
        if message["role"] == "assistant":
            step += 1
            assistant_depth += 1

    return {
        "trajectory_id": trajectory_id,
        "responses_create_params": rcp,
        "expected_action": expected_action,
        "scenario": None,
        "num_unique_actions": num_unique_actions,
        "meta_info": {"turn": turn, "step": step, "assistant_depth": assistant_depth},
        "source_model_info": {
            "rewards": rewards,
            "reward_mean": reward_mean,
            "reward_std": reward_std,
            "reward_var": reward_var,
        },
        "agent_ref": {"type": "responses_api_agents", "name": "single_step_tool_simulation_agent"},
    }


def process_sample(args, sample):
    processed_row = process_row(sample)

    if processed_row is None:
        return None

    if args.difficulty_threshold > 0.0:
        if processed_row["source_model_info"]["reward_mean"] > args.difficulty_threshold:
            return None

    return processed_row


def main(args):
    data = read_jsonl(args.in_path)
    print(f"Loaded data with len={len(data)} !")

    # n_processes = os.cpu_count()
    # with Pool(n_processes):

    expanded_rows = []
    for i, row in tqdm(enumerate(data)):
        for sample in row:
            sample["trajectory_id"] = i
        expanded_rows.extend(row)
    print(f"Expanded data to len={len(expanded_rows)} !")

    process_fn = partial(process_sample, args)
    processed_samples = []
    turn_step_info = []
    num_broken = 0
    num_chat = []
    num_transfer = []
    num_tool = []
    with Pool(os.cpu_count() - 1) as pool:
        for result in tqdm(pool.imap_unordered(process_fn, expanded_rows), total=len(expanded_rows)):
            if result is not None:
                processed_samples.append(result)

    # processed_samples = []
    # for row in tqdm(expanded_rows, desc='format'):
    #     processed_row = process_sample(row)
    #     if processed_row is None:
    #         num_broken += 1
    #         continue

    #     if args.difficulty_threshold > 0.0:
    #         if processed_row['source_model_info']['reward_mean'] > args.difficulty_threshold:
    #             continue

    #     processed_samples.append(processed_row)

    if args.sort == "variance":
        processed_samples = sorted(processed_samples, key=lambda x: x["source_model_info"]["reward_var"], reverse=True)
    elif args.sort == "difficulty":
        processed_samples = sorted(processed_samples, key=lambda x: x["source_model_info"]["reward_mean"])

    if args.limit > 0:
        processed_samples = processed_samples[: args.limit]

    print(f"Got dataset with len={len(processed_samples)} !")

    if args.shuffle != 0:
        random.seed(args.shuffle)
        random.shuffle(processed_samples)

    for processed_row in tqdm(processed_samples, desc="metrics"):
        turn_step_info.append(
            (processed_row["meta_info"]["turn"], processed_row["meta_info"]["step"])
        )  # , processed_row['meta_info']['assistant_depth']))
        if processed_row["expected_action"]["type"] == "function_call":
            if "transfer" in processed_row["expected_action"]["name"]:
                num_transfer.append(1)
                num_tool.append(0)
                num_chat.append(0)
            else:
                num_tool.append(1)
                num_transfer.append(0)
                num_chat.append(0)
        elif processed_row["expected_action"]["type"] == "message":
            num_chat.append(1)
            num_tool.append(0)
            num_transfer.append(0)

    write_jsonl(args.out_path, processed_samples)

    turn_step_info_cnt = Counter(turn_step_info)
    print(f"Num Broken = {num_broken}")
    print("Data Distribution (turn, step)")
    print(turn_step_info_cnt)
    print("Answer Distribution")
    print(f"Tool: {sum(num_tool)}, Transfer: {sum(num_transfer)}, Chat: {sum(num_chat)}")

    print("For first 64k rows:")
    turn_step_info_cnt = Counter(turn_step_info[:64000])
    print("Data Distribution (turn, step)")
    print(turn_step_info_cnt)
    print("Answer Distribution")
    print(f"Tool: {sum(num_tool[:64000])}, Transfer: {sum(num_transfer[:64000])}, Chat: {sum(num_chat[:64000])}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--in-path", type=str)
    parser.add_argument("-o", "--out-path", type=str)
    parser.add_argument("--sort", type=str, choices=["none", "difficulty", "variance"], default="none")
    parser.add_argument("--shuffle", type=int, default=0)
    parser.add_argument("--difficulty-threshold", type=float, default=0.0)
    parser.add_argument("--limit", type=int, default=-1)
    args = parser.parse_args()
    main(args)
