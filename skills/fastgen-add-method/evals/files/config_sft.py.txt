# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import List, Optional
import attrs
from omegaconf import DictConfig

from fastgen.configs.instantiate import LazyCall as L
from fastgen.configs.config import (
    BaseModelConfig,
    BaseConfig,
)
from fastgen.methods import SFTModel
from fastgen.configs.callbacks import BASE_CALLBACKS
from fastgen.configs.callbacks import (
    GradClip_CALLBACK,
)


@attrs.define
class ModelConfig(BaseModelConfig):
    # condition drop probability for better CFG inference
    cond_dropout_prob: Optional[float] = None
    # list of condition keys that do not drop
    cond_keys_no_dropout: List[str] = attrs.field(factory=list)
    # SFT samplers are deterministic unless a network-specific config opts into SDE.
    student_sample_type: str = "ode"


@attrs.define
class Config(BaseConfig):
    model: ModelConfig = attrs.field(factory=ModelConfig)
    model_class: DictConfig = attrs.field(factory=lambda: L(SFTModel)(config=None))


def create_config():
    config = Config()
    config.trainer.callbacks = DictConfig(
        {
            **GradClip_CALLBACK,
            **BASE_CALLBACKS,
        }
    )

    config.dataloader_train.batch_size = 256
    config.model.net_scheduler.warm_up_steps = [0]

    return config
