# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from transformers import AutoModelForSequenceClassification

MODEL_NAME = "distilbert/distilbert-base-uncased"


def create_model():
    return AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
