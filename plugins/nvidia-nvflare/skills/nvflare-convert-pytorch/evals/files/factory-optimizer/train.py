# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch


def build_optimizer(parameters):
    return torch.optim.SGD(parameters, lr=0.1)


model = torch.nn.Linear(2, 2)
features = torch.randn(8, 2)
labels = torch.randint(0, 2, (8,))
optimizer = build_optimizer(model.parameters())

for _ in range(2):
    optimizer.zero_grad()
    loss = torch.nn.functional.cross_entropy(model(features), labels)
    loss.backward()
    optimizer.step()
