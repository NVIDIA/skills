# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from datasets import Dataset
from model import create_model
from transformers import Trainer, TrainingArguments


def build_trainer():
    train_data = Dataset.from_dict({"input_ids": [[1, 2], [3, 4]], "labels": [0, 1]})
    args = TrainingArguments(output_dir="outputs", max_steps=2, report_to=[])
    return Trainer(model=create_model(), args=args, train_dataset=train_data)


trainer = build_trainer()
trainer.train()
