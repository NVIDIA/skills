# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy

from fastgen.configs.discriminator import Discriminator_SD15_Res512_Config
import fastgen.configs.methods.config_dmd2 as config_dmd2_default
from fastgen.configs.data import ImageLoaderConfig
from fastgen.configs.net import SD15Config
from fastgen.configs.config_utils import model_pixel_shape
from fastgen.networks.SD15.network import StableDiffusion15

""" Configs for the DMD2 model on Stable Diffusion v1.5. """


def create_config():
    config = config_dmd2_default.create_config()

    config.model.net_optimizer.lr = 1e-5
    config.model.discriminator_optimizer.lr = 1e-5
    config.model.fake_score_optimizer.lr = 1e-5
    config.model.fake_score_pred_type = "x0"

    config.model.input_shape = [4, 64, 64]
    config.model.discriminator = copy.deepcopy(Discriminator_SD15_Res512_Config)
    config.model.gan_loss_weight_gen = 1e-3
    config.model.guidance_cfg = StableDiffusion15.default_guidance_cfg()
    config.model.guidance_cfg.cfg.scale = 1.75
    config.model.student_update_freq = 10
    config.model.net = copy.deepcopy(SD15Config)

    config.model.sample_t_cfg.time_dist_type = "uniform"
    config.model.sample_t_cfg.min_t = 0.001
    config.model.sample_t_cfg.max_t = 0.999

    config.dataloader_train = copy.deepcopy(ImageLoaderConfig)
    height, width = model_pixel_shape(config.model)
    config.dataloader_train.input_res = (width, height)
    config.trainer.batch_size_global = 2048
    config.trainer.max_iter = 100000
    config.trainer.save_ckpt_iter = 2000
    config.log_config.group = "sd15_dmd2"
    return config
