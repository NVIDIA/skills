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

from datasets import Dataset
from model import create_model_and_peft_config
from peft import PeftModel
from transformers import TrainerCallback
from trl import SFTConfig, SFTTrainer


class AdapterAuditCallback(TrainerCallback):
    pass


def build_trainer():
    model, peft_config = create_model_and_peft_config()
    train_data = Dataset.from_dict({"text": ["first example", "second example"]})
    args = SFTConfig(output_dir="outputs", max_steps=2, report_to=[])
    trainer = SFTTrainer(model=model, args=args, train_dataset=train_data, peft_config=peft_config)
    if not isinstance(trainer.model, PeftModel):
        raise RuntimeError("expected PEFT model")
    trainer.add_callback(AdapterAuditCallback())
    return trainer


trainer = build_trainer()
trainer.train()
