# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from omegaconf import DictConfig
from fastgen.configs.methods.config_dmd2 import create_config as dmd2_create_config
from fastgen.configs.callbacks import EMA_CONST_CALLBACKS
from fastgen.configs.net import CKPT_ROOT_DIR


def create_config():
    config = dmd2_create_config()

    config.model.pretrained_model_path = f"{CKPT_ROOT_DIR}/cifar10/edm-cifar10-32x32-cond-vp.pth"
    # Polynomial draws within entries 2 and 998 of EDM's 1000-entry polynomial table (rho = 7).
    config.model.sample_t_cfg.min_t, config.model.sample_t_cfg.max_t = 0.00210147, 79.5638

    config.model.use_ema = ["ema_9999", "ema_99995", "ema_9996"]
    config.trainer.callbacks = DictConfig(
        {k: v for k, v in config.trainer.callbacks.items() if not k.startswith("ema")}
    )
    config.trainer.callbacks.update(EMA_CONST_CALLBACKS)

    config.trainer.tf32_conv = False
    config.trainer.fp16_reduced_precision_reduction = False
    config.trainer.max_iter = 100000
    config.trainer.batch_size_global = 2048

    return config
