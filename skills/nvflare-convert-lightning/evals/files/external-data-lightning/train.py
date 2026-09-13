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

import argparse
import csv
from pathlib import Path

import pytorch_lightning as pl
import torch
from model import LitNet
from torch.utils.data import DataLoader, TensorDataset

DEFAULT_DATA_DIR = "/data/nvflare/lightning-tabular"


def load_csv(data_path):
    features = []
    labels = []
    with Path(data_path).open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            features.append([float(row[f"feature_{index}"]) for index in range(4)])
            labels.append(int(row["label"]))
    if not features:
        raise ValueError(f"no rows loaded from {data_path}")
    return TensorDataset(torch.tensor(features, dtype=torch.float32), torch.tensor(labels, dtype=torch.long))


class TabularDataModule(pl.LightningDataModule):
    def __init__(self, data_dir=DEFAULT_DATA_DIR, batch_size=4):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size

    def setup(self, stage=None):
        self.train_dataset = load_csv(self.data_dir / "train.csv")
        self.val_dataset = load_csv(self.data_dir / "val.csv")

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    model = LitNet()
    datamodule = TabularDataModule(data_dir=args.data_dir, batch_size=args.batch_size)
    trainer = pl.Trainer(max_epochs=1, accelerator="cpu", devices=1, logger=False)
    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
