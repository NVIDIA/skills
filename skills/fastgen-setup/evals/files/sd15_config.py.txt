# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy

import fastgen.configs.methods.config_sft as config_sft_default
from fastgen.configs.data import ImageLoaderConfig
from fastgen.configs.net import SD15Config
from fastgen.configs.config_utils import model_pixel_shape
from fastgen.networks.SD15.network import StableDiffusion15

"""Configs for SFT (Supervised Fine-Tuning) on Stable Diffusion v1.5."""


def create_config():
    config = config_sft_default.create_config()

    # SD1.5 latent shape: [C, H, W] = [4, 64, 64] for 512x512 images
    config.model.input_shape = [4, 64, 64]

    # Network config
    config.model.net = copy.deepcopy(SD15Config)

    # Optimizer config
    config.model.net_optimizer.lr = 1e-5

    # Timestep sampling config
    config.model.sample_t_cfg.time_dist_type = "uniform"
    config.model.sample_t_cfg.min_t = 0.001
    config.model.sample_t_cfg.max_t = 0.999

    # Sampling config for visualization
    config.model.guidance_cfg = StableDiffusion15.default_guidance_cfg()
    config.model.student_grid_cfg.steps = 50

    # Dataloader
    config.dataloader_train = copy.deepcopy(ImageLoaderConfig)
    height, width = model_pixel_shape(config.model)
    config.dataloader_train.input_res = (width, height)
    # Trainer
    config.trainer.batch_size_global = 256
    config.trainer.max_iter = 100000
    config.trainer.logging_iter = 500

    config.log_config.group = "sd15_sft"

    return config
