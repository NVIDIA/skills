#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run Cosmos-RL SFT on conversation-style video records on one GPU."""

import argparse
import json
import threading
import tomllib
from copy import deepcopy
from pathlib import Path

import torch.nn.functional as F
from torch.utils.data import Dataset

from cosmos_rl.dispatcher.data.packer.hf_vlm_data_packer import HFVLMDataPacker
from cosmos_rl.launcher.worker_entry import main as launch_worker
from qwen_vl_utils import vision_process
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLVisionPatchEmbed,
)


def _patch_embed_forward_gb300(self, hidden_states):
    """Replace non-overlapping Qwen3-VL Conv3D patches with exact linear math."""
    target_dtype = self.proj.weight.dtype
    hidden_states = hidden_states.view(
        -1,
        self.in_channels,
        self.temporal_patch_size,
        self.patch_size,
        self.patch_size,
    )
    flattened_inputs = hidden_states.to(dtype=target_dtype).flatten(1)
    flattened_weights = self.proj.weight.flatten(1)
    return F.linear(flattened_inputs, flattened_weights, self.proj.bias)


Qwen3VLVisionPatchEmbed.forward = _patch_embed_forward_gb300


class VideoConversationDataset(Dataset):
    """Map ShareGPT/LLaVA annotations to Qwen3-VL records on demand."""

    def __init__(
        self,
        annotation_path: str,
        media_root: str,
        system_prompt: str,
        nframes: int,
        max_samples: int = -1,
    ):
        self.annotation_path = Path(annotation_path)
        self.media_root = Path(media_root)
        with self.annotation_path.open(encoding="utf-8") as stream:
            self.records = json.load(stream)
        if max_samples > 0:
            self.records = self.records[:max_samples]
        self.system_prompt = system_prompt
        self.nframes = nframes

    def setup(self, config, *args, **kwargs):
        self.config = config

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        media_value = next(
            (record[field] for field in ("video", "video_id", "media", "media_path")
             if isinstance(record.get(field), str)),
            None,
        )
        if media_value is None:
            raise ValueError(f"record {index} has no string video media field")
        video_path = Path(media_value)
        if not video_path.is_absolute():
            video_path = self.media_root / video_path
        if not video_path.is_file():
            raise FileNotFoundError(video_path)

        if isinstance(record.get("messages"), list):
            messages = deepcopy(record["messages"])
            if self.system_prompt and not any(message.get("role") == "system" for message in messages):
                messages.insert(0, {"role": "system", "content": self.system_prompt})
            for message in messages:
                if message.get("role") != "user":
                    continue
                content = message.get("content", "")
                if not isinstance(content, list):
                    message["content"] = [
                        {"type": "video", "video": str(video_path), "nframes": self.nframes},
                        {"type": "text", "text": str(content)},
                    ]
                break
            return messages

        conversations = record["conversations"]
        question = conversations[0]["value"].replace("<video>", "").strip()
        answer = conversations[1]["value"].strip()
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.extend(
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video",
                            "video": str(video_path),
                            "nframes": self.nframes,
                        },
                        {"type": "text", "text": question},
                    ],
                },
                {"role": "assistant", "content": answer},
            ]
        )
        return messages


# Conversation datasets may repeat media across questions. Cache sampled tensors
# so later records do not decode the same complete video again. A lock prevents the
# torchvision/PyAV bridge from opening the same file concurrently.
_reader_lock = threading.Lock()
_video_cache = {}
_torchvision_reader = vision_process.VIDEO_READER_BACKENDS["torchvision"]


def _read_video_cached(element):
    key = (
        element.get("video"),
        element.get("video_start"),
        element.get("video_end"),
        element.get("nframes"),
        element.get("fps"),
        element.get("min_frames"),
        element.get("max_frames"),
    )
    with _reader_lock:
        if key not in _video_cache:
            _video_cache[key] = _torchvision_reader(element)
        return _video_cache[key]


vision_process.VIDEO_READER_BACKENDS["torchvision"] = _read_video_cached


def _dataset_from_section(raw_config, section_name):
    custom = raw_config["custom"]
    section = custom[section_name]
    return VideoConversationDataset(
        annotation_path=section["annotation_path"],
        media_root=section.get("media_root", section["media_path"]),
        system_prompt=custom.get("system_prompt", ""),
        nframes=int(custom.get("vision", {}).get("nframes", 8)),
        max_samples=int(section.get("max_samples", -1)),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_known_args()[0]
    with open(args.config, "rb") as stream:
        raw_config = tomllib.load(stream)

    def get_train_dataset(config):
        return _dataset_from_section(raw_config, "train_dataset")

    def get_val_dataset(config):
        return _dataset_from_section(raw_config, "val_dataset")

    launch_worker(
        dataset=get_train_dataset,
        val_dataset=get_val_dataset,
        data_packer=HFVLMDataPacker(),
        val_data_packer=HFVLMDataPacker(),
    )


if __name__ == "__main__":
    main()
