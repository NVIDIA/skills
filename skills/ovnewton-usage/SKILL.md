---
name: using-ovnewton
description: Connect the Newton physics engine to an Omniverse simulation or ovstage with ovnewton. Use when importing a populated ovstage into Newton, stepping its physics, synchronizing stage edits, or publishing simulation state for Omniverse rendering.
license: Apache-2.0
metadata:
  author: Jan Carius <jcarius@nvidia.com>
---

# Using ovnewton

Use ovnewton for the Newton ↔ ovstage bridge. The application owns model
finalization, the solver, controls, collision handling, and the simulation loop.
ovnewton reads a populated ovstage; Newton does not parse USD in this workflow.
Standalone Newton simulations and generic Omniverse authoring do not need this skill.

## Start from the supported examples

Install `ovnewton` in the application's environment. Let its package metadata
select compatible dependencies; do not replace the qualified ovstage dependency
with an arbitrary build.

```bash
pip install ovnewton
python -m ovnewton.examples.example_ovnewton_basic --device cpu --no-render --num-frames 8
```

For rendering or MuJoCo-Warp, install `"ovnewton[examples]"`. The basic example
accepts `--solver mujoco` or `--solver vbd`; VBD also supports CPU simulation.
Rendering needs a supported NVIDIA GPU and its own device selection.

Read the [getting-started guide](https://github.com/NVIDIA-Omniverse/ovnewton/blob/main/docs/site/getting-started.md)
and adapt the [basic example](https://github.com/NVIDIA-Omniverse/ovnewton/blob/main/ovnewton/examples/example_ovnewton_basic.py).
For multiple robots, use the [robot example](https://github.com/NVIDIA-Omniverse/ovnewton/blob/main/ovnewton/examples/example_ovnewton_robot.py).
Use documentation and examples corresponding to the installed ovnewton release;
repository main may describe a newer API.

## Construct and bind the simulation

1. Call `ovnewton.register_usd_schemas()` before the first USD schema read.
   Create an `ovstage.Stage()` and populate the application-supplied USD with
   `population.open_usd(stage, scene_path, ordinal=1, domains=PopulationDomain.ALL)`.
   Seal population with `stage.advance_write_floor(1).wait()` before importing.
   If the application already supplies a populated stage, reuse it and its sealed
   ordinal; do not reload the scene or assume its ordinal is 1.
2. Create an empty `newton.ModelBuilder()`. Configure defaults and register any
   solver-owned custom attributes before calling
   `ovnewton.add_ovstage(builder, stage, ordinal=population_ordinal)`.
   Compose all scene assets into one ovstage first: this entry point currently
   requires a builder without model entities and supports one stage per builder.
3. For MuJoCo, register `SolverMuJoCo.register_custom_attributes(builder)` before
   calling `ovnewton.add_ovstage()`. For rigid-body VBD, register
   `SolverVBD.register_custom_attributes(builder)` before `ovnewton.add_ovstage()`
   and call `builder.color()` after that import, before finalization.
   Use `SolverVBD(model, rigid_compliant_alm=True)` for compliant rigid-body
   contacts, following the example's solver settings and contact-material tuning.
4. Finalize with
   `builder.finalize(skip_validation_joints=import_result.has_orphan_joints)`,
   where `import_result` is the result of `add_ovstage`. Construct the solver and
   bind with `ovnewton.attach_ovstage(stage, model=model, ordinal=population_ordinal)`.
   `attach_ovstage(stage)` is the convenience path when no builder customization
   is needed. When binding an existing model, body and joint labels must match
   stage paths.

## Step and synchronize

Allocate two model states, controls, and contacts. Each substep clears forces,
computes contacts, calls the solver, and swaps the states. Start from the example's
collision pipeline and solver configuration rather than assuming every solver
has identical preparation requirements.

After stepping, publish the current state with
`binding.update_to_ovstage(state, ordinal=output_ordinal)`, then seal it with
`stage.advance_write_floor(output_ordinal).wait()`. Coordinate increasing write
ordinals with the application's other stage writers.

To consume application edits, seal them first and call
`binding.update_from_ovstage(state, control)` before stepping, using the same
control passed to the solver. This reads the latest sealed supported state and
drive-target fields. An explicit read ordinal is a consistency ceiling, not
historical playback. Selected Newton-native output can instead be read through
`binding.query()` and `binding.read()` without stage publication.

For rendering, publish and seal Newton output before the renderer consumes the
stage. Newton's simulation device and OVRTX's rendering device are independent;
follow the example's synchronization and device-selection options.

Check the [support guide](https://github.com/NVIDIA-Omniverse/ovnewton/blob/main/docs/site/support.md)
before promising support for authored physics. Unsupported physics may fail
attachment; do not silently replace it with a different physical model. See the
[concepts guide](https://github.com/NVIDIA-Omniverse/ovnewton/blob/main/docs/site/concepts.md) and
[output-read guide](https://github.com/NVIDIA-Omniverse/ovnewton/blob/main/docs/site/output-reads.md)
for synchronization, diagnostics, read layouts, and lifetimes.
