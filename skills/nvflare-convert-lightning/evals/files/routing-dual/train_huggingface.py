# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from transformers import Trainer, TrainingArguments


def main():
    trainer = Trainer(model=model, args=TrainingArguments(output_dir="outputs"))
    trainer.train()


main()
