# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from peft import LoraConfig, TaskType
from transformers import AutoModelForCausalLM

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"


def create_model_and_peft_config():
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    config = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"], task_type=TaskType.CAUSAL_LM)
    return model, config
