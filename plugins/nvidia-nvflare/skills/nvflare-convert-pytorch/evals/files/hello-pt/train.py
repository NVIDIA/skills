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


def main():
    model = SimpleNetwork()
    optimizer = SGD(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()
    features = torch.randn(8, 4)
    labels = torch.randint(0, 2, (8,))
    loader = [(features, labels)]
    train_one_epoch(model, loader, optimizer, loss_fn)
    torch.save(model.state_dict(), "model.pt")


if __name__ == "__main__":
    main()
