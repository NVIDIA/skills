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
