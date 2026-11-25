# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os

from datasets import Dataset
from model import create_model
from transformers import Trainer, TrainingArguments


def build_trainer(world_size, local_rank):
    train_data = Dataset.from_dict({"input_ids": [[1, 2], [3, 4]], "labels": [0, 1]})
    args = TrainingArguments(output_dir="outputs", max_steps=2, local_rank=local_rank, report_to=[])
    if world_size > 1:
        args.ddp_find_unused_parameters = False
    return Trainer(model=create_model(), args=args, train_dataset=train_data)


world_size = int(os.environ.get("WORLD_SIZE", "1"))
local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
trainer = build_trainer(world_size, local_rank)
trainer.train()
