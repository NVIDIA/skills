# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F


class LitNet(pl.LightningModule):
    def __init__(self, input_size=4, num_classes=2, lr=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.fc1 = nn.Linear(input_size, 8)
        self.fc2 = nn.Linear(8, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

    def training_step(self, batch, batch_idx):
        features, labels = batch
        if labels.numel() == 0:
            raise ValueError("empty training batch; check per-site data partitioning")
        loss = F.cross_entropy(self(features), labels)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        features, labels = batch
        if labels.numel() == 0:
            raise ValueError("empty validation batch; check per-site data partitioning")
        loss = F.cross_entropy(self(features), labels)
        self.log("val_loss", loss)

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.hparams.lr)
