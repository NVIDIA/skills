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
from datasets import load_dataset
from model import MODEL_NAME, create_model
from transformers import AutoTokenizer, Trainer, TrainingArguments


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {"accuracy": float((predictions == labels).mean())}


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    dataset = load_dataset("json", data_files={"train": "train.jsonl", "validation": "valid.jsonl"})
    tokenized = dataset.map(lambda row: tokenizer(row["text"], truncation=True), batched=True)
    model = create_model()
    args = TrainingArguments(
        output_dir="outputs",
        num_train_epochs=1,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        report_to=[],
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        compute_metrics=compute_metrics,
    )
    trainer.evaluate()
    trainer.train()


if __name__ == "__main__":
    main()
