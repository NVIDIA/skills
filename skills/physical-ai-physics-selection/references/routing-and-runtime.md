# Integration and runtime setup

Use this reference after choosing the engine. The routes below assume Newton was selected; they do not imply that every URDF/MJCF/USD task should migrate to Newton. For engine comparisons, read [engine selection](engine-selection.md).

## Task selection

| Request | Preferred starting point | Reason |
| --- | --- | --- |
| Run a URDF/MJCF robot in a standalone loop | Direct Newton | No stage binding is required. |
| Import a USD file and inspect a trajectory | Direct Newton unless a shared stage is required | Newton has `ModelBuilder.add_usd()` and viewers. |
| Simulate physics in a populated ovstage | Ovnewton | The model is built from populated stage data. |
| Receive drive targets from stage edits | Ovnewton, after checking supported fields | The persistent binding consumes sealed updates. |
| Publish motion for an existing Omniverse consumer | Ovnewton | It maintains the stage-to-model identity mapping and state transfer. |
| Train in an established framework | The framework's supported runtime | Training, reset logic, and batched environments are application responsibilities. |

The bridge pipeline is **USD -> ovpopulation -> ovstage -> ovnewton -> Newton**. Newton does not parse USD in this path. An application-owned model can also be bound when its body and joint labels match stage paths.

Newton's direct import, a framework's Newton backend, and standalone ovnewton can expose different features. In an existing Isaac Sim or Isaac Lab project, use the release's own integration and tested dependencies. For example, [Isaac Sim 6.1](https://docs.isaacsim.omniverse.nvidia.com/6.1.0/physics/newton_physics.html) documents its Newton extensions, tensor interface, and backend-specific limitations. Do not infer standalone ovnewton's support from that integration or replace its packages without a migration plan.

## Inspect and establish the runtime

Use the intended application's Python interpreter with [the metadata probe](../scripts/probe_environment.py):

```console
python scripts/probe_environment.py
```

Resolve the helper path from the installed skill directory. It needs Python 3.10+ and only the standard library. It reads Python, OS, architecture, installed Newton/ovnewton-related versions, and ovnewton's declared requirements. It does not import native simulation packages, install dependencies, test a GPU, or inventory competing engines.

Record the selected release, actual interpreter, platform, physics/render devices, and application constraints. For a new standalone application, use an isolated environment on a supported host and the project's package-manager convention. For an existing application, retain its lockfile and supported runtime. Check release-specific requirements before native imports.

For standalone ovnewton, the public installation path inside that environment is:

```console
python -m pip install ovnewton
python -m pip check
python -m ovnewton.examples.example_ovnewton_basic --device cpu --no-render --num-frames 8
```

Pin the chosen version when reproducibility is required; the recorded CPU baseline uses ovnewton 0.2.0, Newton 1.6.0, and Warp 1.17.0. Install `"ovnewton[examples]"` only when the task needs its renderer or MuJoCo-Warp dependencies. Extras do not establish hardware readiness. A successful eight-frame example checks execution, not long-horizon contact accuracy or every solver.

For direct Newton, follow its [installation guide](https://newton-physics.github.io/newton/latest/guide/installation.html), then choose a small installed example matching the solver, task, and device. Check that example's help and bound the run. Do not install ovnewton merely to validate direct Newton. For framework-owned environments, use their documented backend check instead of standalone installation commands.

For CUDA or rendering, follow the matching release's examples and verify physics and rendering separately. CPU physics can coexist with GPU rendering; OVRTX still needs a supported RTX-capable GPU and driver. In the basic ovnewton example, `--headless` can still render, while `--no-render` disables rendering.

On an unsupported host, identify a concrete supported-host path and the checks to run there. Do not infer ovnewton Windows compatibility from direct Newton or a framework's Newton backend. Do not install a VM or change system drivers as an incidental setup step. Label commands that were not run.

## Public 0.2.0 baseline

This is a documentation baseline, not a promise about every dependency combination:

- Python 3.11 through 3.14.
- Linux is the documented ovnewton runtime; verify architecture-specific wheels for the chosen release. Newton's broader OS support does not extend the ovnewton package automatically.
- The public package declares `ovstage>=0.2.0.377349,<0.3`, Newton >=1.6.0, and Warp >=1.17.0. The examples extra declares OVRTX >=0.5.0.377615,<0.6.
- CPU non-rendering is supported. Rendering additionally requires a supported RTX-capable GPU and driver.
- Warp wheel versions have their own hardware/driver requirements. Do not infer GPU readiness from the presence of an NVIDIA device or the package name.

Read the selected release's requirements before reproducing these values. The exact qualified environment can be narrower than open dependency ranges. Save the resolved versions after a successful test.

## Diagnostic decisions

| Evidence | Next useful check |
| --- | --- |
| No compatible distribution | Python, OS, architecture, and selected package version |
| Resolver conflict | Existing application constraints and declared dependency ranges |
| Native library import fails | Wheel/platform compatibility and process library environment |
| CPU smoke succeeds, CUDA fails | Warp initialization, selected CUDA device, and driver compatibility |
| Simulation succeeds, rendering fails | OVRTX dependencies, RTX device, renderer initialization, and output configuration |
| Scene import fails | Applied schemas, supported physics, units, and the reported prim path |
| A stage consumer sees old data | Publication ordering, sealed write floor, and the consumer's synchronization |

Do not remove physical constraints to make an installation check pass. Use the packaged basic scene to separate runtime problems from the user's model problems.

## Runtime ownership

The application owns state allocation, controls, collision detection, solver stepping, and timing. MuJoCo and VBD need their solver attributes registered on the builder before `add_ovstage`; VBD needs builder coloring before finalization. Use the installed example as the starting point rather than translating parameter names between solvers.

For the documented ovnewton MuJoCo path, MuJoCo-Warp advances the model and Newton's collision pipeline supplies contacts (`use_mujoco_contacts=False`). Choosing this solver does not require authoring MuJoCo-specific USD schemas.

## Sources

- [Public 0.2.0 requirements snapshot](https://github.com/NVIDIA-Omniverse/ovnewton/blob/f5febec0718c19a23cb8c4cfd9cfdb9a2069515e/pyproject.toml)
- [Getting started](https://github.com/NVIDIA-Omniverse/ovnewton/blob/main/docs/site/getting-started.md)
- [Newton installation and Warp compatibility](https://github.com/newton-physics/newton/blob/main/docs/guide/installation.rst)
