# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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
