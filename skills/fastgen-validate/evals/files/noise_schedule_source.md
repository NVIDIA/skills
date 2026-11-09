# FastGen source evidence

Source revision: `e0a92619fc56daf4dfa081273c7729c045bf1dcc` in [NVlabs/FastGen](https://github.com/NVlabs/FastGen/tree/e0a92619fc56daf4dfa081273c7729c045bf1dcc).
These are copied source excerpts, not execution results.

Import the production class; do not turn these methods into a replacement schedule. The existing tests illustrate its API. Compute the independent oracle directly from the mathematical formula, without calling implementation methods for sigma or its derivative.

## [fastgen/networks/noise_schedule.py:2123](https://github.com/NVlabs/FastGen/blob/e0a92619fc56daf4dfa081273c7729c045bf1dcc/fastgen/networks/noise_schedule.py#L2123)

```python
class ShiftedRFNoiseSchedule(BaseNoiseSchedule):
    """Shifted Rectified Flow noise schedule: x_t = (1 - sigma(t)) x_0 + sigma(t) noise.

    ``sigma(t)`` is the exponential ``time_shift`` of ``t`` over ``[0, max_t]``, which is ``t`` itself at
    ``shift = 1``, and the network sees ``num_train_timesteps * sigma(t)``. Convention: t=0 is data, t=1 is noise.
    """

    def __init__(
        self,
        min_t: float = 0.0,
        max_t: float = 0.999,
        num_train_timesteps: int = 1000,
        shift: float = 1.0,
        **kwargs,
    ):
        """
        Args:
            num_train_timesteps: Scale of the time the network sees, as in the diffusers flow-matching schedulers.
            shift: Exponential shift of this schedule's own time against the shared one, so that
                ``sigma(t) = time_shift(t, shift, max_t=max_t)`` rather than ``t``.  ``1.0`` is ``RFNoiseSchedule``.
        """
        super().__init__(min_t, max_t, **kwargs)
        self.num_train_timesteps = self._validate_num_steps(num_train_timesteps, "num_train_timesteps")
        if max_t > 1.0:
            raise ValueError(f"RF max_t must be at most 1.0, got {max_t}")
        if shift <= 0:
            raise ValueError(f"schedule shift must be positive, got {shift}")
        self.shift = float(shift)

    @property
    def is_unwarped(self) -> bool:
        """Whether ``shift = 1``, the identity warp, so this schedule's own time is the shared time."""
        return self.shift == 1.0

    def _warp(self, t: torch.Tensor) -> torch.Tensor:
        """This schedule's own time for the shared time *t*; the identity at ``shift = 1``."""
        return time_shift(t, self.shift, max_t=self.max_t)

    def _warp_prime(self, t: torch.Tensor) -> torch.Tensor:
        """``d/dt`` of `_warp` -- ``s M^2 / (s t + (M - t))^2``, and ``1`` at ``shift = 1``."""
        if self.is_unwarped:
            return torch.ones_like(t)
        return self.shift * self.max_t**2 / (self.shift * t + (self.max_t - t)) ** 2

    def _rescale_t(self, t: torch.Tensor) -> torch.Tensor:
        return self.num_train_timesteps * self._warp(t)

    def _alpha(self, t: torch.Tensor) -> torch.Tensor:
        return 1 - self._warp(t)

    def _sigma(self, t: torch.Tensor) -> torch.Tensor:
        return self._warp(t)

    def _alpha_prime(self, t: torch.Tensor) -> torch.Tensor:
        return -self._warp_prime(t)

    def _sigma_prime(self, t: torch.Tensor) -> torch.Tensor:
        return self._warp_prime(t)

    def _unwarp(self, value: torch.Tensor) -> torch.Tensor:
        """The shared time whose own time is *value*; the warps form a group, so ``f_(1/shift)``."""
        return time_shift(value, 1.0 / self.shift, max_t=self.max_t)
```

## [fastgen/networks/noise_schedule.py:2186](https://github.com/NVlabs/FastGen/blob/e0a92619fc56daf4dfa081273c7729c045bf1dcc/fastgen/networks/noise_schedule.py#L2186)

```python
    def step(
        self,
        xt: torch.Tensor,
        model_output: torch.Tensor,
        t: torch.Tensor,
        t_next: torch.Tensor,
        pred_type: str = "x0",
        sample_type: str | float = "ode",
        clean_end: bool = True,
    ) -> torch.Tensor:
        """The sampler step, where the DDIM step on a flow prediction is ``xt + (sigma(t_next) - sigma(t)) (eps - x0)``.

        That is the Euler step in sigma (and in t at ``shift = 1``), taken without converting to x0. As in the
        base step, with ``clean_end`` ``t_next = 0`` is the clean end and is accepted below ``min_t``; RF's own
        time 0 is clean (``sigma(0) = 0``), so the step lands on the x0 the flow implies either way.
        """
        check_sample_type(sample_type)
        if pred_type != "flow" or sample_type not in ("ode", 0):
            return super().step(
                xt, model_output, t, t_next, pred_type=pred_type, sample_type=sample_type, clean_end=clean_end
            )
        self._assert_t_valid(t)
        self._assert_t_valid(torch.where(t_next == 0, t, t_next) if clean_end else t_next, "t_next")
        original_dtype = xt.dtype
        xt, model_output = to_high_precision(xt, model_output)
        t, t_next = to_high_precision(t, t_next)
        # ``flow = dx/dt = sigma'(t) (eps - x0)``
        delta = expand_like((self._determinant(t, t_next) / self._warp_prime(t)).to(xt.dtype), xt)
        x_next = xt + delta * model_output
        return x_next.to(original_dtype)
```

## [fastgen/networks/noise_schedule.py:2277](https://github.com/NVlabs/FastGen/blob/e0a92619fc56daf4dfa081273c7729c045bf1dcc/fastgen/networks/noise_schedule.py#L2277)

```python
    def _flow_to_x0(
        self,
        xt: torch.Tensor,
        v: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the x0-prediction from the flow-prediction.

        Args:
            xt (torch.Tensor): Perturbed data at time t
            v (torch.Tensor): The flow-prediction.
            t (torch.Tensor): The time step in the forward process.

        Returns:
            torch.Tensor: The x0-prediction.

        Note:
            Calculation is performed in at least ``HIGH_PRECISION`` for better numerical accuracy.
        """
        # Convert to high precision for calculation
        xt, v = to_high_precision(xt, v)
        t = to_high_precision(t)

        # ``sigma / sigma'`` rather than ``t``.  ``flow`` is ``dx/dt``, which for rectified flow is
        # ``sigma'(t) (eps - x0)``, and ``x_t - x0`` is ``sigma (eps - x0)`` -- so the two differ by
        # exactly this ratio.  Both factors reduce to the plain ``sigma`` at ``shift = 1``.
        scale = expand_like((self._sigma(t) / self.non_zero_clamp(self._sigma_prime(t))).to(xt.dtype), xt)

        # Perform calculation
        x0 = xt - v.mul(scale)

        return x0
```

## [fastgen/networks/noise_schedule.py:2311](https://github.com/NVlabs/FastGen/blob/e0a92619fc56daf4dfa081273c7729c045bf1dcc/fastgen/networks/noise_schedule.py#L2311)

```python
    def _cond_velocity(
        self,
        x: torch.Tensor,
        eps: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """``dx/dt = sigma'(t) (eps - x0)``, the base formula in closed form.

        Kept as an override for the same reasons the other conversions are: the base builds this
        from ``alpha'(t) x0 + sigma'(t) eps``, two multiplies and an add, whereas rectified flow's
        ``alpha' = -sigma'`` collapses it to one multiply of a difference.  It also short-circuits
        exactly at ``shift = 1``, where ``sigma'`` is identically one and the target is ``eps - x0``
        untouched -- which is what keeps ``RFNoiseSchedule`` bit-identical.
        """
        # Convert to high precision for calculation
        x, eps = to_high_precision(x, eps)
        t = to_high_precision(t)

        # Perform calculation
        velocity = eps - x
        if not self.is_unwarped:
            velocity = velocity * expand_like(self._sigma_prime(t).to(x.dtype), velocity)

        return velocity
```

## [tests/test_noise_schedule.py:122](https://github.com/NVlabs/FastGen/blob/e0a92619fc56daf4dfa081273c7729c045bf1dcc/tests/test_noise_schedule.py#L122)

```python
def test_rf_flow_is_the_time_derivative_at_any_shift():
    """``flow`` means ``dx/dt`` for rectified flow too, which is ``sigma'(t) (eps - x0)``.

    The overrides exist to compute that in one multiply instead of the base's two-plus-an-add, not
    to hold a different convention -- so they have to agree with the base formula, and be the plain
    ``eps - x0`` exactly where ``sigma'`` is one.
    """
    x0, eps = torch.randn(4, 3, dtype=torch.float64), torch.randn(4, 3, dtype=torch.float64)
    t = torch.linspace(0.05, 0.95, 4, dtype=torch.float64)[:, None]

    for shift in (1.0, 3.0 / 12.0, 3.0, 12.0):
        s = get_noise_schedule("shifted_rf", shift=shift, min_t=0.0, max_t=0.999)
        xt = s.alpha(t) * x0 + s.sigma(t) * eps
        flow = s.cond_velocity(x0, eps, t)

        torch.testing.assert_close(flow, BaseNoiseSchedule._cond_velocity(s, x0, eps, t), msg=f"shift {shift}")
        torch.testing.assert_close(flow, s.sigma_prime(t) * (eps - x0), msg=f"shift {shift}")
        # The conversions have to invert that same convention.
        torch.testing.assert_close(s.flow_to_x0(xt, flow, t), x0, msg=f"flow_to_x0 at shift {shift}")
        torch.testing.assert_close(s.x0_to_flow(xt, x0, t), flow, msg=f"x0_to_flow at shift {shift}")

    # Unshifted, the scaling is skipped rather than multiplied by a rounded one.
    unshifted = get_noise_schedule("rf", min_t=0.0, max_t=0.999)
    assert torch.equal(unshifted.cond_velocity(x0, eps, t), eps - x0)
```

## [tests/test_noise_schedule.py:1671](https://github.com/NVlabs/FastGen/blob/e0a92619fc56daf4dfa081273c7729c045bf1dcc/tests/test_noise_schedule.py#L1671)

```python
@pytest.mark.parametrize("pred_type", ["x0", "eps", "flow"])
@pytest.mark.parametrize("make_schedule", _SMOOTH_SCHEDULES)
def test_ode_step_with_an_exact_prediction_lands_on_the_forward_process(make_schedule, pred_type):
    torch.manual_seed(0)
    schedule = make_schedule()
    xt, predictions, t, t_next = _exact_predictions(schedule)

    x_next = schedule.step(xt, predictions[pred_type], t, t_next, pred_type=pred_type)

    expected = schedule.forward_process(predictions["x0"], predictions["eps"], t_next)
    torch.testing.assert_close(x_next, expected, rtol=1e-7, atol=1e-7)
```
