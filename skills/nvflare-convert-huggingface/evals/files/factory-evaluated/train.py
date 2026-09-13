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

import numpy as np
from datasets import Dataset
from model import create_model
from transformers import Trainer, TrainerCallback, TrainingArguments


class LifecycleAuditCallback(TrainerCallback):
    def on_train_begin(self, args, state, control, **kwargs):
        print("training started")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {"accuracy": float((predictions == labels).mean())}


def build_trainer():
    train_data = Dataset.from_dict({"input_ids": [[1, 2], [3, 4]], "labels": [0, 1]})
    eval_data = Dataset.from_dict({"input_ids": [[5, 6], [7, 8]], "labels": [0, 1]})
    args = TrainingArguments(output_dir="outputs", max_steps=2, report_to=[])
    return Trainer(
        model=create_model(),
        args=args,
        train_dataset=train_data,
        eval_dataset=eval_data,
        compute_metrics=compute_metrics,
        callbacks=[LifecycleAuditCallback()],
    )


trainer = build_trainer()
trainer.evaluate()
trainer.train()
