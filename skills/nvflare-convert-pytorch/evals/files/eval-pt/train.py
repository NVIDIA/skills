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

import torch
from model import SimpleNetwork
from torch import nn
from torch.optim import SGD


def train_one_epoch(model, loader, optimizer, loss_fn):
    model.train()
    for features, labels in loader:
        optimizer.zero_grad()
        output = model(features)
        loss = loss_fn(output, labels)
        loss.backward()
        optimizer.step()


def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for features, labels in loader:
            predictions = model(features).argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.numel()
    return correct / total


def main():
    model = SimpleNetwork(input_size=4, num_classes=2)
    optimizer = SGD(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()
    features = torch.randn(8, 4)
    labels = torch.randint(0, 2, (8,))
    train_loader = [(features, labels)]
    val_features = torch.randn(4, 4)
    val_labels = torch.randint(0, 2, (4,))
    val_loader = [(val_features, val_labels)]
    for _epoch in range(2):
        train_one_epoch(model, train_loader, optimizer, loss_fn)
        accuracy = evaluate(model, val_loader)
        print(f"val_accuracy={accuracy:.4f}")
    torch.save(model.state_dict(), "model.pt")


if __name__ == "__main__":
    main()
