# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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
