# Workflow Ownership Map

Use this map after reading `./DESIGN.md`. Inspect the concrete implementation before editing; filenames are not API contracts.

## Authoring and runtime ownership

| Concern | Owner | Source |
|---|---|---|
| World assets, cameras, randomization, adapters, reset hooks | Scene | `./arena/i4h_arena/scenes/`, `./arena/i4h_arena/assets/`, `./arena/i4h_arena/envcfg/` |
| Scene capabilities and mode overrides | Scene manifest | `./arena/i4h_arena/scenes/manifest/<scene>.yaml` |
| Cross-process robot labels, calibration, teleop devices | Embodiment manifest | `./arena/i4h_arena/embodiments/manifest/<robot>.yaml` |
| Reusable behavior, typed ports, requirements | Task | `./tasks/<project>/` |
| Simulator-compatible exported RSL-RL actors | In-process policy Task | `./tasks/rsl_rl/` |
| Remote backend/model/observation/training contract | Remote task manifest | `./tasks/<project>/i4h_tasks/<project>/manifest/<task>.yaml` |
| Scene selection, run modes, graph composition, goal semantics | Workflow | `./workflows/i4h_workflows/<specialty>/<workflow>.py` |
| Standard run-mode vocabulary and shared builders | Workflows | `./workflows/i4h_workflow_modes/README.md` |
| Node scheduling, wiring, retries, timeouts | Engine | `./engine/` |
| Episode attempts, reset/step/render/record loop | Simulation runner | `./arena/i4h_arena/runner.py` |
| HDF5 contract and cross-process messages | Common | `./common/i4h_common/` |
| Mimic, annotation, conversion, visualization | Offline tools | `./tools/` |
| Vectorized online RL stepping, trainer mapping, and hyperparameters | RL training lifecycle | `./rl/` |

The public authoring surface is `engine/i4h_engine/interface.py` and `engine/i4h_engine/graph.py`. A workflow module exports exactly one `WORKFLOW = Workflow(...)`. The workflow layout and specialty catalog are documented in `./workflows/README.md`.

## Dependency rules

- Keep `common` independent of every other i4h layer.
- Keep the Engine independent of concrete workflows, tasks, Arena, and Isaac Sim.
- Keep in-process tasks independent of workflows and Arena.
- Keep workflows independent of Arena, Isaac Sim, and policy stacks.
- Keep GR00T/openpi backends and offline tools dependent only on `common` among i4h layers.
- Let only `SimulationRunner` call `env.step` during normal Workflow execution; an isolated RL trainer owns vectorized stepping during online training.

## Existing pattern families

- SO-ARM scissors: `scissor_pick_and_place` plus `soarm_scissors`; policy, N1.7 alternative, rule-based, teleop, replay, and idle.
- G1 locomanipulation: `locomanip_tray_pick_and_place` and `locomanip_push_cart`; N1.6 policy with a 23-wide teleop mode override.
- G1 trocar: `assemble_trocar`; dex-hand scene, N1.5 policy Task, and separate RLinf PPO post-training profile.
- Ultrasound reach: `ultrasound_probe_reach`; Franka probe Scene, RSL-RL PPO profile, exported TorchScript Task, policy, and idle modes.
- Ultrasound: `ultrasound_liver_scan`; Panda scene with openpi policy, rule-based, teleop, replay, and idle.
- Surgical: PSM, dual-PSM, and STAR scenes with rule-based, replay, and idle modes.

Confirm these families with `./run.sh list`; manifests and workflows may evolve.

## Validation order

```bash
cd "$(git rev-parse --show-toplevel)"
./run.sh show <workflow> --mode <mode>
./run.sh lint <workflow> --mode <mode>
./run.sh lint --all
```

Run focused tests in each owning component environment. For Scene or adapter changes, finish with a visible simulator run in every affected dynamic mode; use `--idle` only for stationary layout inspection.
