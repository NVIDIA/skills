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

from datasets import load_dataset
from model import MODEL_NAME, create_model_and_peft_config
from transformers import AutoTokenizer
from trl import SFTConfig, SFTTrainer


def main():
    model, peft_config = create_model_and_peft_config()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_data = load_dataset("json", data_files="train.jsonl", split="train")
    eval_data = load_dataset("json", data_files="valid.jsonl", split="train")
    args = SFTConfig(
        output_dir="outputs",
        num_train_epochs=1,
        per_device_train_batch_size=1,
        report_to=[],
    )
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=train_data,
        eval_dataset=eval_data,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.evaluate()
    trainer.train()


if __name__ == "__main__":
    main()
