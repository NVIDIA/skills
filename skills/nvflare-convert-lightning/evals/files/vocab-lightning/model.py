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

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F


class LitTextCNN(pl.LightningModule):
    def __init__(self, vocab_size, embed_dim=16, num_classes=2, lr=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, tokens):
        embedded = self.embedding(tokens)
        pooled = embedded.mean(dim=1)
        return self.classifier(pooled)

    def training_step(self, batch, batch_idx):
        tokens, labels = batch
        if labels.numel() == 0:
            raise ValueError("empty training batch; check per-site data partitioning")
        loss = F.cross_entropy(self(tokens), labels)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        tokens, labels = batch
        if labels.numel() == 0:
            raise ValueError("empty validation batch; check per-site data partitioning")
        loss = F.cross_entropy(self(tokens), labels)
        self.log("val_loss", loss)

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.hparams.lr)
