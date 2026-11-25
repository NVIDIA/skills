# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
from model import DataParallelNetwork


def main():
    device = torch.device("cuda")
    model = torch.nn.DataParallel(DataParallelNetwork()).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    features = torch.randn(16, 4, device=device)
    labels = torch.randint(0, 2, (16,), device=device)

    optimizer.zero_grad()
    loss = torch.nn.functional.cross_entropy(model(features), labels)
    loss.backward()
    optimizer.step()


if __name__ == "__main__":
    main()
