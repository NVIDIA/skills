# FastGen source fixture

Source revision: `e0a92619fc56daf4dfa081273c7729c045bf1dcc`. The source below is input evidence, not an installed checkout.

## fastgen/methods/README.md

```text
## Choosing a distillation baseline

These are practical heuristics; validate quality and diversity for the chosen task. Starting with **8 NFE** is a
good rule of thumb.

- **[DMD2](distribution_matching/README.md#dmd2-distribution-matching-distillation-v2):** distribution matching for
  fast convergence and high visual quality even with 4 NFE or less. Watch for unstable dynamics or diversity loss; retain the
  best checkpoint and consider early stopping when quality regresses.
- **[PDD](trajectory_matching/README.md#pdd-parallel-decoding-distillation):** trajectory matching for teacher
  fidelity, diversity and more stable training, typically at 4–8 NFE.

**Data-free DMD2 and PDD:** use backward simulation instead of noising real data. Start with
`model.backward_simulation="staggered"` and
[`CarryCallback`](../callbacks/carry.py); respect the [rollout phase constraints](backward_simulation.py).
For DMD2, enable VSD and explicitly disable GAN (`model.gan_loss_weight_gen=0`). Both still require a teacher and task
conditioning, including source media or cached conditions for I2I/V2V.
```
