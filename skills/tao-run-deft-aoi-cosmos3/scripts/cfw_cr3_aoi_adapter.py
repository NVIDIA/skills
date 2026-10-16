#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Interim two-image ShareGPT experiment for the baked Framework runtime.

Mount this file read-only as
``/workspace/.venv/lib/python3.13/site-packages/cosmos_framework/configs/base/reasoner/experiment/tao_cr3_aoi.py``.
Remove it when the pinned image provides an equivalent native image-pair
adapter.  Model, optimizer, LoRA, FSDP, and checkpoint behavior remain native.
"""

from __future__ import annotations

import copy
import json
import os
import re
from pathlib import Path
from typing import Any

from hydra.core.config_store import ConfigStore
from torch.utils.data import Dataset

from cosmos_framework.configs.base.reasoner.experiment.dataflow_roles import (
    VLMCollator,
    VLMProcessor,
)
from cosmos_framework.configs.base.reasoner.experiment.llava_ov_vlm import (
    pre_exp012_llava_ov,
)
from cosmos_framework.data.generator.dataflow import (
    CosmosDataLoader,
    MapDistributor,
    PoolPackingBatcher,
)
from cosmos_framework.data.generator.processors import build_processor
from cosmos_framework.utils.lazy_config import LazyCall as L
from cosmos_framework.utils.reasoner.constant import IGNORE_INDEX


class CR3TwoImageShareGPTDataset(Dataset):
    """Validate and expose bare-OK/NG two-image ShareGPT JSON arrays."""

    def __init__(self, annotation_path: str, media_root: str) -> None:
        self.annotation_path = Path(annotation_path).expanduser().resolve(strict=True)
        self.media_root = Path(media_root).expanduser().resolve(strict=True)
        with self.annotation_path.open(encoding="utf-8") as stream:
            rows = json.load(stream)
        if not isinstance(rows, list) or not rows:
            raise ValueError("CR3 training annotations must be a non-empty JSON array")
        self.rows = [self._validated_row(index, row) for index, row in enumerate(rows)]

    def _validated_row(self, index: int, value: Any) -> dict[str, Any]:
        if not isinstance(value, dict):
            raise TypeError(f"CR3 record {index} must be an object")
        row = copy.deepcopy(value)
        images = row.get("images")
        turns = row.get("conversations")
        if not isinstance(images, list) or len(images) != 2 or not all(isinstance(item, str) for item in images):
            raise ValueError(f"CR3 record {index} must contain exactly two string image paths")
        if not isinstance(turns, list) or len(turns) < 2:
            raise ValueError(f"CR3 record {index} must contain at least two ShareGPT turns")
        if not all(
            isinstance(turn, dict)
            and turn.get("from") in {"human", "gpt"}
            and isinstance(turn.get("value"), str)
            for turn in turns
        ):
            raise ValueError(f"CR3 record {index} has an invalid ShareGPT turn")
        if turns[0]["from"] != "human":
            raise ValueError(f"CR3 record {index} must begin with a human turn")
        if turns[-1].get("from") != "gpt" or turns[-1].get("value") not in {"OK", "NG"}:
            raise ValueError(f"CR3 record {index} final assistant label must be exactly OK or NG")
        resolved_images = []
        for image in images:
            path = Path(image).expanduser()
            path = path if path.is_absolute() else self.media_root / path
            resolved_images.append(str(path.resolve(strict=True)))
        row["images"] = resolved_images
        return row

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.rows[index]


class CR3TwoImageVLMProcessor(VLMProcessor):
    """Turn both CR3 image paths into one native OpenAI-style user turn."""

    def _sharegpt_to_openai(self, item: dict[str, Any]) -> list[dict[str, Any]]:
        decoded_images = []
        for image_path in item["images"]:
            decoded = self._decode_image({"path": image_path})
            if decoded is None:
                raise ValueError(f"Framework could not decode CR3 image: {image_path}")
            decoded_images.append(decoded)

        messages: list[dict[str, Any]] = []
        images_inserted = False
        for turn in item["conversations"]:
            role = "user" if turn.get("from") == "human" else "assistant"
            text = re.sub(r"(\n)?</?image>(\n)?", "", str(turn.get("value", ""))).strip()
            if role == "user" and not images_inserted:
                content: Any = [
                    *({"type": "image", "image": image} for image in decoded_images),
                    {"type": "text", "text": text},
                ]
                images_inserted = True
            else:
                content = text
            messages.append({"role": role, "content": content})
        if not images_inserted:
            raise ValueError("CR3 conversation has no human turn for its image pair")
        return messages


tao_cr3_aoi = copy.deepcopy(pre_exp012_llava_ov)
tao_cr3_aoi.job.name = "tao_cr3_aoi"
tao_cr3_aoi.job.group = "deft_aoi"
tao_cr3_aoi.trainer.run_validation = False
tao_cr3_aoi.checkpoint.load_from_object_store.enabled = False
tao_cr3_aoi.checkpoint.load_from_object_store.credentials = ""
tao_cr3_aoi.checkpoint.load_from_object_store.bucket = ""
tao_cr3_aoi.checkpoint.save_to_object_store.enabled = False
tao_cr3_aoi.checkpoint.save_to_object_store.credentials = ""
tao_cr3_aoi.checkpoint.save_to_object_store.bucket = ""
tao_cr3_aoi.upload_reproducible_setup = False

tao_cr3_aoi.dataloader_train = L(CosmosDataLoader)(
    distributor=L(MapDistributor)(
        dataset=L(CR3TwoImageShareGPTDataset)(
            annotation_path="${oc.env:TAO_CR3_TRAIN_ANNOTATION}",
            media_root="${oc.env:TAO_CR3_MEDIA_ROOT}",
        ),
        shuffle=True,
        seed="${oc.env:TAO_CR3_SEED,42}",
        name="cr3_train",
    ),
    processor=L(CR3TwoImageVLMProcessor)(
        processor=L(build_processor)(
            tokenizer_type="${model.config.policy.backbone.model_name}",
            config_variant="hf",
        ),
        ignore_index=IGNORE_INDEX,
    ),
    batcher=L(PoolPackingBatcher)(
        max_tokens=40960,
        pool_size=16,
        max_batch_size=1,
        long_threshold=16000,
    ),
    collator=L(VLMCollator)(),
    num_workers=0,
)
tao_cr3_aoi.dataloader_val = None

ConfigStore.instance().store(
    group="experiment",
    package="_global_",
    name="tao_cr3_aoi",
    node=tao_cr3_aoi,
)
