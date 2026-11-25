# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F


class LitNet(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(4, 2)

    def forward(self, features):
        return self.layer(features)

    def training_step(self, batch, batch_idx):
        features, labels = batch
        if labels.numel() == 0:
            raise ValueError("empty training batch; check per-site data partitioning")
        return F.cross_entropy(self(features), labels)

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.01)
