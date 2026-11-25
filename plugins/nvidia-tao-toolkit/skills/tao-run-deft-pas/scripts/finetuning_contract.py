# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed, immutable fine-tuning contract for PAS CLIP runs."""

from __future__ import annotations

import dataclasses
from typing import Any


@dataclasses.dataclass(frozen=True)
class LoraAdapterParameters:
    """LoRA adapter shape applied identically to the vision and text towers."""

    rank: int = dataclasses.field(
        default=8,
        metadata={
            "valid_min": 1,
            "description": "Low-rank adapter dimension for each targeted projection.",
        },
    )
    alpha: int = dataclasses.field(
        default=16,
        metadata={
            "valid_min": 1,
            "description": "LoRA scaling numerator applied to the adapter update.",
        },
    )
    num_last_blocks: int = dataclasses.field(
        default=3,
        metadata={
            "valid_min": 1,
            "description": "Number of final transformer blocks adapted in each tower.",
        },
    )
    dropout: float = dataclasses.field(
        default=0.05,
        metadata={
            "valid_min": 0.0,
            "valid_max": 1.0,
            "description": "Dropout probability applied in each LoRA adapter.",
        },
    )
    target_modules: tuple[str, ...] = dataclasses.field(
        default=("q_proj", "k_proj", "v_proj", "out_proj"),
        metadata={
            "valid_options": "q_proj,k_proj,v_proj,out_proj",
            "description": "SigLIP2 attention projections receiving LoRA adapters.",
        },
    )

    def tao_tower(self) -> dict[str, Any]:
        """Render the exact TAO PEFT tower block."""
        return {
            "mode": "lora",
            "target_modules": list(self.target_modules),
            "num_last_blocks": self.num_last_blocks,
            "rank": self.rank,
            "alpha": self.alpha,
            "dropout": self.dropout,
        }


DEFAULT_LORA_PARAMETERS = LoraAdapterParameters()


def materialize_peft(method: str) -> dict[str, Any]:
    """Render the TAO PEFT block for one approved fine-tuning method."""
    if method == "sft":
        return {"enabled": False}
    if method != "lora":
        raise ValueError("fine-tuning method must be lora or sft")
    return {
        "enabled": True,
        "method": "lora",
        "train_logit_calibration": True,
        "vision": DEFAULT_LORA_PARAMETERS.tao_tower(),
        "text": DEFAULT_LORA_PARAMETERS.tao_tower(),
    }


def expected_lora_record(method: str) -> tuple[int | None, int | None]:
    """Return the approval/state fields implied by the selected method."""
    if method == "lora":
        return DEFAULT_LORA_PARAMETERS.rank, DEFAULT_LORA_PARAMETERS.alpha
    if method == "sft":
        return None, None
    raise ValueError("fine-tuning method must be lora or sft")


def validate_peft_record(
    peft: Any,
    *,
    method: str,
    lora_rank: Any,
    lora_alpha: Any,
) -> None:
    """Bind durable approval fields to both materialized TAO encoder towers."""
    expected_rank, expected_alpha = expected_lora_record(method)
    if lora_rank != expected_rank or lora_alpha != expected_alpha:
        raise ValueError(
            "LoRA rank/alpha record does not match the supported fine-tuning contract"
        )
    if method == "sft":
        if peft != {"enabled": False}:
            raise ValueError("SFT requires tao_spec.peft to disable PEFT exactly")
        return
    if not isinstance(peft, dict):
        raise ValueError("LoRA requires tao_spec.peft to be an object")
    if peft.get("enabled") is not True or peft.get("method") != "lora":
        raise ValueError("LoRA requires enabled lora PEFT in tao_spec")
    for tower_name in ("vision", "text"):
        tower = peft.get(tower_name)
        if not isinstance(tower, dict):
            raise ValueError(f"tao_spec.peft.{tower_name} must be an object")
        if tower.get("rank") != lora_rank or tower.get("alpha") != lora_alpha:
            raise ValueError(
                f"tao_spec.peft.{tower_name} rank/alpha disagrees with approval"
            )
