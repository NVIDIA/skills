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
from model import LitNet
from torch.utils.data import DataLoader, TensorDataset


def main():
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    dataset = TensorDataset(torch.randn(8, 4), torch.randint(0, 2, (8,)))
    trainer = pl.Trainer(max_epochs=1, accelerator=accelerator, devices=1, logger=False)
    trainer.fit(LitNet(), DataLoader(dataset, batch_size=4))


if __name__ == "__main__":
    main()
