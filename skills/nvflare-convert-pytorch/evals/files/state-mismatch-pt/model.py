# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch.nn as nn


class Classifier(nn.Module):
    def __init__(self, input_size=4, num_classes=2):
        super().__init__()
        self.classifier = nn.Linear(input_size, num_classes)

    def forward(self, features):
        return self.classifier(features)
