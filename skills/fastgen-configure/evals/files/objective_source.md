# FastGen source fixture

Source revision: `e0a92619fc56daf4dfa081273c7729c045bf1dcc`. Excerpts below are input evidence, not an installed checkout.

## fastgen/methods/distribution_matching/dmd2.py:132

```text

    @property
    def use_vsd(self) -> bool:
        """Whether this configuration enables VSD and fake-score training."""
        return self.config.vsd_loss_weight > 0

    @property
    def use_gan(self) -> bool:
        """Whether this configuration enables adversarial training."""
        return self.config.gan_loss_weight_gen > 0

    def build_model(self):
        super().build_model()

        self.build_teacher()
```

## fastgen/methods/distribution_matching/dmd2.py:516

```text
    def _student_update_step(
        self,
        inputs: TrainingInputs,
        data: Dict[str, Any],
    ) -> tuple[Dict[str, torch.Tensor], Dict[str, TensorOrDict | Callable]]:
        """Update the student with the enabled VSD and generator GAN terms."""
        # The auxiliary update trains the discriminator only on real data, so only then does the generator face it.
        # `real` must be present or absent on every rank: the discriminator path and f-distill's histogram
        # reduce collectively.
        use_gan = self.use_gan and data.get("real") is not None
        if not self.use_vsd and not use_gan:
            raise ValueError("DMD2 student updates require VSD training, or GAN training with data['real'].")

```

## fastgen/methods/distribution_matching/dmd2.py:700

```text
        inputs: TrainingInputs,
        data: Dict[str, Any],
    ) -> tuple[Dict[str, torch.Tensor], Dict[str, TensorOrDict | Callable]]:
        """Update the enabled fake-score and discriminator models."""
        if not self.use_vsd and not self.use_gan:
            raise ValueError("DMD2 requires at least one of VSD or GAN training to be enabled.")

        condition = data.get("condition")
        neg_condition = data.get("neg_condition")
        real_data = data.get("real")
        if self.use_gan and not self.use_vsd and real_data is None:
            raise ValueError("GAN-only discriminator updates require data['real'].")

        t_gen_data = self._generation_time(inputs.student_step)
        with torch.no_grad():
            generation = self.gen_data_from_net(
                inputs.input_student,
                inputs.student_step,
                condition=condition,
                neg_condition=neg_condition,
                **inputs.generation_kwargs,
            )
            x_t = self.net.noise_scheduler.forward_process(generation.data, inputs.eps, inputs.t, t_start=t_gen_data)

        loss_map: Dict[str, torch.Tensor] = {}
        gan_loss_map: Dict[str, torch.Tensor] = {}
        total_loss = None
        if self.use_gan and real_data is not None:
            gan_loss_map, total_loss = self._compute_gan_discriminator_losses(
                x_t,
                inputs.t,
```

## [fastgen/configs/experiments/WanI2V/config_dmd2_14b.py:32](https://github.com/NVlabs/FastGen/blob/e0a92619fc56daf4dfa081273c7729c045bf1dcc/fastgen/configs/experiments/WanI2V/config_dmd2_14b.py#L32)

```python
    config.model.input_shape = [16, 21, 60, 104]
    config.model.discriminator = copy.deepcopy(Discriminator_Wan_14B_Config)
    config.model.discriminator.disc_type = "multiscale_down_mlp_large"
    config.model.discriminator.feature_indices = [21, 30, 39]
    config.model.gan_loss_weight_gen = 0.03
    config.model.gan_use_same_t_noise = True
    config.model.fake_score_pred_type = "x0"
    config.model.net = copy.deepcopy(Wan21_I2V_14B_480P_Config)
    config.model.guidance_cfg = WanI2V.default_guidance_cfg()
```

## [fastgen/networks/WanI2V/network.py:569](https://github.com/NVlabs/FastGen/blob/e0a92619fc56daf4dfa081273c7729c045bf1dcc/fastgen/networks/WanI2V/network.py#L569)

```python
        assert isinstance(condition, dict), "condition must be a dict"
        assert "text_embeds" in condition, "condition must contain 'text_embeds'"
        assert "first_frame_cond" in condition, "condition must contain 'first_frame_cond'"

        # In TI2V mode the same state-domain tensor is reused for both the input
        # and output first-frame replacements.
        first_frame_cond = condition["first_frame_cond"]
```
