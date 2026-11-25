# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from peft import LoraConfig, TaskType
from transformers import AutoModelForCausalLM

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"


def create_model_and_peft_config():
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    config = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"], task_type=TaskType.CAUSAL_LM)
    return model, config
