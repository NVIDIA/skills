# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SFT config for LTX-2 (video + audio)."""

import copy

import fastgen.configs.methods.config_sft as config_sft_default
from fastgen.configs.data import VideoAudioLoaderConfig
from fastgen.configs.net import LTX2Config
from fastgen.networks.LTX2.network import LTX2
from fastgen.configs.config_utils import model_pixel_shape


def create_config():
    config = config_sft_default.create_config()

    config.trainer.logging_iter = 100
    config.trainer.save_ckpt_iter = 1000

    config.trainer.fsdp = True
    config.model.meta_init = True
    config.model.net_optimizer.lr = 2e-5
    config.model.student_grid_cfg.steps = 40
    config.model.precision = "bfloat16"
    config.model.precision_opt = "float32"

    config.model.net = copy.deepcopy(LTX2Config)
    config.model.guidance_cfg = LTX2.default_guidance_cfg(version="2.0")

    # 720p resolution (1280x704)
    config.model.input_shape = {"video": [128, 31, 22, 40], "audio": None}
    # 1080p resolution (1920x1088)
    # config.model.input_shape = {"video": [128, 31, 34, 60], "audio": None}

    # Enable VAE tiling for high resolutions to avoid 32-bit index overflow.
    num_frames, height, width = model_pixel_shape(config.model)
    if height > 720:
        config.model.net.enable_vae_tiling = True

    config.model.sample_t_cfg.time_dist_type = "uniform"
    config.model.sample_t_cfg.min_t = 0.001
    config.model.sample_t_cfg.max_t = 0.999

    config.dataloader_train = copy.deepcopy(VideoAudioLoaderConfig)
    config.dataloader_train.batch_size = 1
    # LTX-2's own negative prompt, as used at inference.
    config.dataloader_train.presets_map = {"neg_condition_raw": "neg_prompt_ltx2"}
    config.dataloader_train.img_size = (width, height)
    config.dataloader_train.sequence_length = num_frames
    # Resample onto the model's fps if the dataset is not already preprocessed at it.
    # config.dataloader_train.target_fps = config.model.net.fps

    config.log_config.group = "ltx2_sft"
    return config
