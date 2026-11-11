# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Authored WanT2V experiment to adapt into a bounded training run.

Composes the released WanT2V DMD2 recipe at e0a92619fc56daf4dfa081273c7729c045bf1dcc.
The data paths are supplied deployment paths; this task does not contain those datasets.
"""

import copy

from fastgen.callbacks.carry import CarryCallback
from fastgen.configs.callbacks import EMA_CALLBACK
from fastgen.configs.experiments.WanT2V.config_dmd2 import create_config as create_wan_config
from fastgen.configs.instantiate import LazyCall as L


def create_config():
    config = create_wan_config()
    config.model.use_ema = True
    config.model.student_update_freq = 4
    config.model.backward_simulation = "staggered"
    config.trainer.callbacks.update(copy.deepcopy(EMA_CALLBACK))
    config.trainer.callbacks.update({"carry": L(CarryCallback)(enable_validation=False)})
    config.trainer.callbacks.wandb.logging_iter = 100
    config.trainer.callbacks.wandb.sample_logging_iter = 100

    config.trainer.ddp = False
    config.trainer.fsdp = False
    config.trainer.resume = True
    config.trainer.max_iter = 10000
    config.trainer.batch_size_global = 16
    config.trainer.grad_accum_rounds = 1
    config.trainer.save_ckpt_iter = 1000
    config.trainer.validation_iter = 1000
    config.trainer.logging_iter = 1000

    config.dataloader_train.datatags = ["WDS:/data/wan/train"]
    config.dataloader_train.batch_size = 2
    config.dataloader_train.deterministic = True
    config.dataloader_train.strict = True
    config.dataloader_train.num_workers = 1
    config.dataloader_val = copy.deepcopy(config.dataloader_train)
    config.dataloader_val.datatags = ["WDS:/data/wan/heldout"]
    config.dataloader_val.train = False
    config.dataloader_val.sampler_start_idx = 0
    config.trainer.global_vars_val = [{"MAX_VAL_STEPS": 1}]

    config.log_config.project = "fastgen-example"
    config.log_config.group = "wan_dmd2_smoke"
    config.log_config.name = "completed-run"
    return config
