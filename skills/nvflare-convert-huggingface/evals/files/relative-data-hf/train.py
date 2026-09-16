# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
from pathlib import Path

from datasets import load_dataset
from model import MODEL_NAME, create_model
from transformers import AutoTokenizer, Trainer, TrainingArguments

DEFAULT_DATASET_PATH = Path("datasets/sst2")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=Path, default=DEFAULT_DATASET_PATH)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    dataset = load_dataset(
        "json",
        data_files={
            "train": str(args.dataset_path / "train.jsonl"),
            "validation": str(args.dataset_path / "valid.jsonl"),
        },
    )
    tokenized = dataset.map(lambda row: tokenizer(row["text"], truncation=True), batched=True)
    trainer = Trainer(
        model=create_model(),
        args=TrainingArguments(output_dir="outputs", max_steps=2, report_to=[]),
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
    )
    trainer.evaluate()
    trainer.train()


if __name__ == "__main__":
    main()
