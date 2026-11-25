# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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
