# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import lightning as L
import torch
from torch.utils.data import DataLoader, TensorDataset


class Classifier(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(2, 2)

    def training_step(self, batch, batch_idx):
        features, labels = batch
        return torch.nn.functional.cross_entropy(self.layer(features), labels)

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.1)


def build_trainer():
    trainer = L.Trainer(max_epochs=1, logger=False, enable_checkpointing=False)
    return trainer


dataset = TensorDataset(torch.randn(8, 2), torch.randint(0, 2, (8,)))
trainer = build_trainer()
trainer.fit(Classifier(), DataLoader(dataset, batch_size=4))
