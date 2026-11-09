# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
from typing import Any, Dict, Optional

import attrs
from omegaconf import DictConfig

from fastgen.configs.instantiate import LazyCall as L
from fastgen.callbacks.grad_clip import GradClipCallback
from fastgen.configs.config import (
    BaseModelConfig,
    BaseConfig,
)
from fastgen.configs.opt import BaseOptimizerConfig, BaseSchedulerConfig
from fastgen.methods import DMD2Model
from fastgen.configs.discriminator import Discriminator_EDM_CIFAR10_Config
from fastgen.configs.callbacks import BASE_CALLBACKS
from fastgen.configs.callbacks import (
    GradClip_CALLBACK,
)


@attrs.define
class ModelConfig(BaseModelConfig):
    # optimizer and scheduler for the fake score net
    fake_score_optimizer: DictConfig = attrs.field(factory=lambda: copy.deepcopy(BaseOptimizerConfig))
    fake_score_scheduler: DictConfig = attrs.field(factory=lambda: copy.deepcopy(BaseSchedulerConfig))

    discriminator: DictConfig = attrs.field(factory=lambda: copy.deepcopy(Discriminator_EDM_CIFAR10_Config))
    # optimizer and scheduler for the discriminator
    discriminator_optimizer: DictConfig = attrs.field(factory=lambda: copy.deepcopy(BaseOptimizerConfig))
    discriminator_scheduler: DictConfig = attrs.field(factory=lambda: copy.deepcopy(BaseSchedulerConfig))

    # Changes to sample_t_cfg for the fake-score DSM term, e.g. {"time_dist_type": "logitnormal"}: the term then
    # draws its own noising time and noise. None reuses the adversarial ones, which always come from sample_t_cfg.
    fake_score_sample_t_overrides: Optional[Dict[str, Any]] = None

    # What the student generates in training: "x0", its clean estimate, or "step", its student_sample_type step to
    # the next student-grid time t_{i+1} (Phased DMD, see the dmd2 module docstring), which the losses renoise with
    # the forward transition from that time, so they match its distribution to the data's at t_{i+1}. "step" needs a
    # per-sample student step and, with VSD, fake_score_pred_type="eps" and a transition fake_score_weighting, which
    # must be "x0" without sample_t_in_student_bucket. It works with any student_sample_type; with "sde", which
    # renoises the predicted x0 with fresh noise, the VSD and GAN losses match DMD2's on that x0 in distribution.
    # Most methods with their own generators support only "x0".
    student_generation: str = "x0"

    # Draw each noising time (student, fake-score, and discriminator updates) within the student-grid
    # bucket [t_{i+1}, t_i] of its generator input at t_i: sample_t_cfg's distribution conditioned on the
    # bucket, clipped to the sampling range. Requires each sample's student step when t is drawn, which DMD2's
    # _training_inputs provides (inherited by LADD and f-distill); CausVid, Self-Forcing, and AnyFlow
    # have no per-sample step and reject it. Without it, step generations are noised from t_{i+1} up to
    # max_t, where several student steps reach each t and the fake score requires fake_score_weighting="x0".
    sample_t_in_student_bucket: bool = False

    # Student update frequency: iterations 1..freq-1 update the fake score/discriminator and iteration freq
    # updates the student, i.e. freq-1 auxiliary updates per student update (upstream DMD2's
    # dfake_gen_update_ratio also updates the fake score on generator steps).
    student_update_freq: int = 5

    # Weight for the VSD loss in the student update phase. Zero disables the
    # complete VSD/fake-score path, including its model and optimizer.
    vsd_loss_weight: float = 1.0

    # weight for the GAN loss in the student update phase
    gan_loss_weight_gen: float = 0.001

    # use the same t and noise to perturb the real data and fake data
    gan_use_same_t_noise: bool = False

    # perform dsm loss in a space specified by fake_score_pred_type (None means use original net_pred_type)
    fake_score_pred_type: str | None = None
    # Fake-score loss: "native" is denoising score matching in fake_score_pred_type's space and treats the
    # generations as clean data; the other values weight the unit-noise loss of the forward transition from each
    # generation's time (see ``common_loss.transition_score_weight``), need fake_score_pred_type="eps", and are required
    # for step generations: "variance_matched" matches the gradient noise of DMD2's x0 loss, "inverse_variance" its
    # loss floor, "conservative" is sigma_t^2 / alpha_t^2, "x0" is x0 regression against the transition target, and
    # "unweighted" is the plain unit-noise (eps) loss. On x0 generations, they are "native" with
    # fake_score_pred_type="x0" ("eps" for "unweighted").
    fake_score_weighting: str = "native"
    # Per-dimension variance of the clean generations in the Gaussian proxy behind the
    # "variance_matched" and "inverse_variance" weights.
    fake_score_data_variance: float = 1.0

    # APT-style R1 regularization weight (arXiv:2501.08316); 0 means no R1 reg.
    # Unlike APT, which perturbs the clean latent, the penalty here is computed on the noised
    # sample x_t and only the discriminator head is optimized (whereas the teacher is frozen).
    # APT states its sigma as a variance whereas gan_r1_reg_alpha is a standard
    # deviation, so matching the strength gan_r1_reg_weight * gan_r1_reg_alpha**2 gives
    #     gan_r1_reg_weight = lambda_apt * sigma_apt / gan_r1_reg_alpha**2
    # where APT uses lambda_apt=100 with sigma_apt=0.01 for images and sigma_apt=0.1 for videos.
    gan_r1_reg_weight: float = 0.0
    # Perturbation scale (standard deviation) for approximate R1, applied to the noised network
    # input x_t. The loss is unnormalized, so the effective strength scales as
    # gan_r1_reg_weight * gan_r1_reg_alpha**2: scaling alpha by c needs the weight scaled by 1/c**2.
    gan_r1_reg_alpha: float = 0.1

    # Student input source: None uses noised real data; "staggered"/"synced" carry one rollout
    # across iterations with CarryCallback (rank-offset vs. common phases); "default" carries
    # nothing and starts each iteration at a random step. Shapes come from input_shape. The GAN
    # terms need data["real"] and are skipped in batches without it (GAN-only training requires
    # it), so a prompt-only loader makes training data-free.
    backward_simulation: str | None = None


@attrs.define
class Config(BaseConfig):
    model: ModelConfig = attrs.field(factory=ModelConfig)
    model_class: DictConfig = attrs.field(factory=lambda: L(DMD2Model)(config=None))


def create_config():
    config = Config()
    config.trainer.callbacks = DictConfig(
        {
            **GradClip_CALLBACK,
            "grad_clip_fake_score": L(GradClipCallback)(grad_norm=None, model_key="fake_score"),
            "grad_clip_discriminator": L(GradClipCallback)(grad_norm=None, model_key="discriminator"),
            **BASE_CALLBACKS,
        }
    )

    config.dataloader_train.batch_size = 256
    config.model.discriminator_scheduler.warm_up_steps = [0]
    config.model.fake_score_scheduler.warm_up_steps = [0]
    config.model.net_scheduler.warm_up_steps = [0]
    config.model.sample_t_cfg.time_dist_type = "polynomial"

    return config
