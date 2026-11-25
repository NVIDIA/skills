# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from model import Classifier


def build_model():
    return Classifier(input_size=4, num_classes=2)
