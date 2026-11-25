# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytorch_lightning as pl
import torch
from model import LitNet
from torch.utils.data import DataLoader, TensorDataset


def make_loader():
    features = torch.randn(8, 4)
    labels = torch.randint(0, 2, (8,))
    return DataLoader(TensorDataset(features, labels), batch_size=4)


def main():
    model = LitNet()
    train_loader = make_loader()
    val_loader = make_loader()
    trainer = pl.Trainer(max_epochs=1, accelerator="cpu", devices=1, logger=False)
    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    main()
