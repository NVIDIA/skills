# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Dict, Any, TYPE_CHECKING, Callable

import torch

from fastgen.methods import FastGenModel
from fastgen.methods.common_loss import flow_matching_loss
from fastgen.networks.network_causal import CausalContext
from fastgen.utils.tensor import expand_like
from fastgen.utils import basic_utils
from fastgen.utils.tree_utils import (
    TensorOrDict,
    assert_same,
    per_modality,
    tmap,
    to_dict,
    randn_like,
    get_device,
    cast_like,
    get_batch_size,
)

if TYPE_CHECKING:
    from fastgen.networks.network import FastGenNetwork


class SFTModel(FastGenModel):
    """SFT model that handles both unimodal and multimodal data.

    When ``data["real"]`` is a plain tensor, operates in standard unimodal
    mode.  When it is a dict of tensors (e.g.
    ``{"video": tensor, "audio": tensor}``), each modality is noised
    independently (shared timestep ``t``) and loss is accumulated across
    modalities.
    """

    # Generation runs the network's own sampler, which each network narrows further.
    student_sample_types = ("ode", "sde")

    def build_model(self):
        super().build_model()
        self.load_student_weights_and_ema()

    def _to_x0_prediction(
        self,
        noisy_data: TensorOrDict,
        net_pred: TensorOrDict,
        t: torch.Tensor,
    ) -> TensorOrDict:
        """Convert the network's native prediction into a clean-data prediction."""
        return self.net.noise_scheduler.convert_model_output(
            noisy_data,
            net_pred,
            t,
            src_pred_type=self.net.net_pred_type,
            target_pred_type="x0",
        )

    def _mix_condition(self, condition, neg_condition):
        """
        Mix condition with neg_condition based on cond_dropout_prob.

        Supported shapes:
          * ``torch.Tensor`` -- a flat tensor of conditioning embeddings.
          * ``dict[str, V]`` -- where ``V`` is a tensor, ``None``, or
            (recursively) a tuple/list of those. This covers Flux2's
            image-edit dict contract, where ``text_embeds`` holds a
            ``(None, prompt_embeds_tensor)`` tuple.

        Args:
            condition: torch.Tensor or dict of (tensor | None | tuple/list of tensors).
            neg_condition: same type/structure as condition.

        Returns:
            The mixed condition and a ``[B]`` boolean mask, ``True`` for dropped samples,
            or ``None`` when dropout is disabled or no condition keys are droppable.
        """
        if self.config.cond_dropout_prob is None:
            return condition, None
        if neg_condition is None:
            raise ValueError(
                "cond_dropout_prob needs a negative condition to drop to; provide neg_condition (a negative "
                "prompt or the zero class vector) or set cond_dropout_prob=None."
            )

        # Networks with structured conditions the generic tensor mixing cannot express
        # (e.g. Cosmos3's variable-length token-ID lists) own their per-sample mixing.
        if hasattr(self.net, "mix_condition"):
            return self.net.mix_condition(condition, neg_condition, self.config.cond_dropout_prob)

        # Restrict to the (sub)structure we actually want to mix; the
        # ``cond_keys_no_dropout`` keys are passed through untouched.
        if isinstance(condition, torch.Tensor):
            droppable, droppable_neg = condition, neg_condition
        elif isinstance(condition, dict):
            keys_no_drop = set(self.config.cond_keys_no_dropout)
            if not keys_no_drop.issubset(condition.keys()):
                raise ValueError(f"keys_no_drop: {keys_no_drop} not in {condition.keys()}")
            droppable = {k: v for k, v in condition.items() if k not in keys_no_drop}
            droppable_neg = {k: neg_condition[k] for k in droppable}
        else:
            raise TypeError(f"Unsupported type: {type(condition)}")

        # Nothing to mix (all keys are in ``cond_keys_no_dropout``): skip
        # leaf-introspection on an empty substructure and return condition as-is.
        if isinstance(droppable, dict) and not droppable:
            return condition, None

        # Derive (batch_size, device) from the first tensor leaf in the
        # droppable substructure. dtype is intentionally omitted from
        # ``torch.rand`` below: condition leaves may be integer/bool, but the
        # mask is derived from a float comparison and is always boolean.
        batch_size = get_batch_size(droppable)
        device = get_device(droppable)

        sample_mask = torch.rand(batch_size, device=device) >= self.config.cond_dropout_prob

        def mix_leaf(v: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
            assert isinstance(n, torch.Tensor), f"negative conditioning leaf must be a tensor, got {type(n).__name__}"
            assert_same(
                shape=True,
                dtype=True,
                device=self.device,
                positive_condition=v,
                neg_condition=n,
            )
            assert (
                v.shape[0] == batch_size
            ), f"condition leaf batch ({v.shape[0]}) must match conditioning batch ({batch_size})"
            return torch.where(expand_like(sample_mask, v), v, n)

        mixed = tmap(
            mix_leaf,
            droppable,
            droppable_neg,
        )

        mixed = mixed if isinstance(condition, torch.Tensor) else condition | mixed
        return mixed, ~sample_mask

    def generator_fn(
        self,
        net: FastGenNetwork,
        noise: TensorOrDict,
        **kwargs,
    ) -> TensorOrDict:
        sample_kwargs = dict(kwargs)
        assert hasattr(net, "sample"), "net must have the sample() method"
        with basic_utils.inference_mode(
            net,
            precision_amp=self.precision_amp_infer,
            device_type=get_device(noise).type,
        ):
            x = net.sample(
                noise,
                sample_steps=net.noise_scheduler.default_student_steps,
                sample_type=self.config.student_sample_type,
                guidance_cfg=self.config.guidance_cfg,
                **sample_kwargs,
            )
        return cast_like(x, noise)

    def single_train_step(
        self, data: Dict[str, Any], iteration: int
    ) -> tuple[Dict[str, torch.Tensor], Dict[str, TensorOrDict | Callable]]:
        """
        Single training step for supervised finetuning (sft)

        Args:
            data (Dict[str, Any]): Data dict for the current iteration.
            iteration (int): Current training iteration

        Returns:
            loss_map (dict[str, torch.Tensor]): Dictionary containing the loss values
            outputs (dict[str, TensorOrDict | Callable]): Dictionary containing the network output

        """
        real_data = data["real"]
        condition = data.get("condition")
        neg_condition = data.get("neg_condition")
        batch_size = get_batch_size(real_data)

        # Perturb the data according to the underlying time sampling type
        t = self.net.noise_scheduler.sample_t(
            batch_size,
            **self.config.sample_t_cfg.sample_t_kwargs(),
            device=self.device,
        )
        eps = randn_like(real_data)

        # replace condition with neg_condition with a probability for training
        # but keep original condition for _get_outputs (sampling uses original condition)
        condition_train, condition_dropout_mask = self._mix_condition(condition, neg_condition)

        noisy_real_data = self.net.noise_scheduler.forward_process(real_data, eps, t)
        net_pred = self.net(noisy_real_data, t, condition=condition_train)

        # DSM loss per modality
        dsm_loss = per_modality(
            flow_matching_loss,
            net_pred,
            self.net.net_pred_type,
            real_data,
            eps,
            self.net.noise_scheduler.schedules,
            t,
        )
        loss_map = to_dict(dsm_loss, suffix="dsm_loss")
        total_loss = sum(loss_map.values())
        loss_map["total_loss"] = total_loss

        x0_pred = self._to_x0_prediction(noisy_real_data, net_pred, t)
        outputs = self._get_outputs(
            x0_pred,
            condition=condition,
            neg_condition=neg_condition,
        )
        loss_map = self.apply_reward(x0_pred, data, loss_map, condition_dropout_mask=condition_dropout_mask)
        return loss_map, outputs


class CausalSFTModel(SFTModel):
    def generator_fn(
        self,
        net: FastGenNetwork,
        noise: TensorOrDict,
        **kwargs,
    ) -> TensorOrDict:
        sample_kwargs = dict(kwargs)
        sample_kwargs["context_noise"] = self.config.context_noise
        return super().generator_fn(net=net, noise=noise, **sample_kwargs)

    def single_train_step(
        self, data: Dict[str, Any], iteration: int
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor | Callable]]:
        """
        Single training step for causal supervised finetuning (sft).

        Args:
            data (Dict[str, Any]): Data dict for the current iteration.
            iteration (int): Current training iteration

        Returns:
            loss_map (dict[str, torch.Tensor]): Dictionary containing the loss values
            outputs (dict[str, torch.Tensor]): Dictionary containing the network output

        """

        real_data = data["real"]
        condition = data.get("condition")
        neg_condition = data.get("neg_condition")
        batch_size = real_data.shape[0]

        # Add noise to real image data (for multistep generation)
        eps_inhom = torch.randn(batch_size, *self.input_shape, device=self.device, dtype=real_data.dtype)
        assert hasattr(
            self.net.noise_scheduler, "sample_t_inhom"
        ), "net.noise_scheduler does not have the sample_t_inhom() method"
        t_inhom = self.net.noise_scheduler.sample_t_inhom(
            batch_size,
            self.input_shape[1],
            self.net.chunk_size,
            **self.config.sample_t_cfg.sample_t_kwargs(),
            device=self.device,
        )
        t_inhom_expanded = t_inhom[:, None, :, None, None]  # shape [B, 1, T, 1, 1]
        noisy_real_data = self.net.noise_scheduler.forward_process(real_data, eps_inhom, t_inhom_expanded)

        # replace condition with neg_condition with a probability for training
        # but keep original condition for _get_outputs (sampling uses original condition)
        condition_train, condition_dropout_mask = self._mix_condition(condition, neg_condition)

        causal_ctx = CausalContext(
            cache_tag="pos",
            chunk_size=self.net.chunk_size,
            total_num_frames=self.input_shape[1],
            store_kv=False,
            is_ar=False,
        )

        net_pred = self.net(noisy_real_data, t_inhom, condition=condition_train, causal_ctx=causal_ctx)
        # Frame-aligned timesteps: ``expand_like`` only appends trailing dims, so the [B, T] grid would
        # broadcast against the channel axis of ``real_data`` / ``eps_inhom`` ([B, C, T, H, W]).
        loss = per_modality(
            flow_matching_loss,
            net_pred,
            self.net.net_pred_type,
            real_data,
            eps_inhom,
            self.net.noise_scheduler.schedules,
            t_inhom_expanded,
        )

        # Build output dictionaries
        loss_map = {
            "total_loss": loss,
            "dsm_loss": loss,
        }

        x0_pred = self._to_x0_prediction(noisy_real_data, net_pred, t_inhom_expanded)
        outputs = self._get_outputs(
            x0_pred,
            condition=condition,
            neg_condition=neg_condition,
        )
        loss_map = self.apply_reward(x0_pred, data, loss_map, condition_dropout_mask=condition_dropout_mask)
        return loss_map, outputs


class CausalTeacherForcingSFTModel(CausalSFTModel):
    def single_train_step(
        self, data: Dict[str, Any], iteration: int
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor | Callable]]:
        del iteration

        real_data = data["real"]
        condition = data.get("condition")
        neg_condition = data.get("neg_condition")
        batch_size = real_data.shape[0]

        eps_inhom = torch.randn(batch_size, *self.input_shape, device=self.device, dtype=real_data.dtype)
        t_inhom = self.net.noise_scheduler.sample_t_inhom(
            batch_size,
            self.input_shape[1],
            self.net.chunk_size,
            **self.config.sample_t_cfg.sample_t_kwargs(),
            device=self.device,
        )
        t_inhom_expanded = t_inhom[:, None, :, None, None]
        noisy_real_data = self.net.noise_scheduler.forward_process(real_data, eps_inhom, t_inhom_expanded)

        if self.config.teacher_forcing_context_match_train_t:
            # The context stream sees exactly the noisy frames the denoising
            # stream is trained on.
            clean_t = t_inhom
            clean_context = noisy_real_data
        elif self.config.sample_t_context_cfg.enabled:
            clean_t = self.net.noise_scheduler.sample_t_inhom(
                batch_size,
                self.input_shape[1],
                self.net.chunk_size,
                **self.config.sample_t_context_cfg.sample_t_kwargs(),
                device=self.device,
            )
            clean_context = self.net.noise_scheduler.forward_process(
                real_data,
                torch.randn_like(real_data),
                clean_t[:, None, :, None, None],
            )
        else:
            clean_t = torch.zeros_like(t_inhom)
            clean_context = real_data

        condition_train, condition_dropout_mask = self._mix_condition(condition, neg_condition)

        causal_ctx = CausalContext(
            cache_tag="pos",
            chunk_size=self.net.chunk_size,
            total_num_frames=self.input_shape[1],
            store_kv=False,
            is_ar=False,
            mask_kind="teacher_forcing",
        )

        x_in = torch.cat([clean_context, noisy_real_data], dim=2)
        t_in = torch.cat([clean_t, t_inhom], dim=1)
        net_pred = self.net(
            x_in,
            t_in,
            condition=condition_train,
            causal_ctx=causal_ctx,
        )
        net_pred = net_pred[:, :, real_data.shape[2] :]
        loss = flow_matching_loss(
            net_pred=net_pred,
            pred_type=self.net.net_pred_type,
            noise_scheduler=self.net.noise_scheduler,
            x0=real_data,
            eps=eps_inhom,
            t=t_inhom_expanded,  # frame-aligned [B, 1, T, 1, 1]; see CausalSFTModel
        )

        loss_map = {
            "total_loss": loss,
            "dsm_loss": loss,
        }
        x0_pred = self._to_x0_prediction(noisy_real_data, net_pred, t_inhom_expanded)
        outputs = self._get_outputs(
            x0_pred,
            condition=condition,
            neg_condition=neg_condition,
        )
        loss_map = self.apply_reward(x0_pred, data, loss_map, condition_dropout_mask=condition_dropout_mask)
        return loss_map, outputs
