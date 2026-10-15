# Evaluating using-ovnewton

The seven cases in `evals.json` are the initial CI smoke set: five integration
requests and two negative routing cases. They ask for code or guidance, not
GPU rendering, asset downloads, training, or long simulations. Evaluate both
with and without the skill on Codex and Claude Code in identical environments.
Use one attempt per case for initial onboarding; measure the total runtime and
keep the signing run below the NVCARPS one-hour limit.

For the basic case, execute the returned Python against a supported ovnewton
installation and the packaged `scene_rigid_bodies.usda`. Grade runnable behavior,
ordinal ordering, solver ownership, and state swapping, not exact wording.
The existing-stage case intentionally uses a non-default sealed ordinal.
Check negative cases from skill-load traces, not the agent's claim that it did
not use a skill. With-skill positive cases should read this skill; baseline
cases receive the same public docs and runtime but no skill instructions.

In a CPU-capable product environment, also run:

```bash
python -m ovnewton.examples.example_ovnewton_basic --device cpu --no-render --num-frames 8
python -m ovnewton.examples.example_ovnewton_basic --solver vbd --device cpu --no-render --num-frames 8
```

Extended GPU rendering and multiple-robot runs belong to release validation,
outside the CI smoke dataset. Record harness and model versions, trajectories,
completion rates, tokens, wall time, and with/without-skill results. Generate
`BENCHMARK.md` from measured results; do not create a passing report before
evaluation. NVSkills CI generates the signed card and signature.

See the [dataset contract](https://docs.nvidia.com/skills/skillevaluator/eval-datasets)
and [live evaluation guide](https://docs.nvidia.com/skills/skillevaluator/tier3-live-evaluation).
