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

import torch
from model import SimpleNetwork
from torch import nn
from torch.optim import SGD
from torch.utils.data import DataLoader, TensorDataset

DEFAULT_DATA_PATH = "/data/nvflare/tabular/train.csv"


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
    return torch.tensor(features, dtype=torch.float32), torch.tensor(labels, dtype=torch.long)


def make_loader(data_path, batch_size):
    features, labels = load_csv(data_path)
    return DataLoader(TensorDataset(features, labels), batch_size=batch_size, shuffle=True)


def train_one_epoch(model, loader, optimizer, loss_fn):
    model.train()
    for features, labels in loader:
        optimizer.zero_grad()
        output = model(features)
        loss = loss_fn(output, labels)
        loss.backward()
        optimizer.step()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default=DEFAULT_DATA_PATH)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    model = SimpleNetwork()
    optimizer = SGD(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()
    loader = make_loader(args.data_path, args.batch_size)
    train_one_epoch(model, loader, optimizer, loss_fn)
    torch.save(model.state_dict(), "model.pt")


if __name__ == "__main__":
    main()
