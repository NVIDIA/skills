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
