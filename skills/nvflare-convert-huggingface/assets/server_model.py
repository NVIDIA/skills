# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hugging Face server-model template preserving the Trainer model keyspace.

Adapt the import and ``load_model`` call to the source project's existing model
factory. Returning that model directly avoids adding a wrapper prefix to its
``state_dict`` keys.
"""

import torch

from model import load_model


class ServerModel:
    """Construct the same importable model used by the patched Trainer."""

    def __new__(cls, model_name_or_path: str, **model_kwargs):
        model = load_model(model_name_or_path=model_name_or_path, **model_kwargs)
        if not isinstance(model, torch.nn.Module):
            raise TypeError(f"load_model() must return torch.nn.Module, got {type(model).__name__}")
        return model
