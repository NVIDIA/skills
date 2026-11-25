# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import lightning as L


def main():
    trainer = L.Trainer(max_epochs=1)
    trainer.fit(model, train_dataloaders=loader)


main()
