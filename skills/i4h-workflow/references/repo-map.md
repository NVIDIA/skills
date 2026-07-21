# Agentic Repo Map

Load this when authoring, forking, or baking agentic environments. Keep env YAMLs as the source of truth and inspect the concrete source files for the selected env before generating code.

## Source of Truth

- Env YAML: `workflows/agentic/config/environments/<env>.yaml` defines robot, cameras, policy stack, model source, task text, arena defaults, and dataset mapping.
- Arena env discovery: `workflows/agentic/arena/arena/environments/__init__.py` imports every `*_environment.py` module and matches each class `name` to a YAML file. A class without YAML, or YAML without class, fails registration.
- `workflows/agentic/arena/arena/registry.py` is only a compatibility shim over `arena.environments`.
- Run commands from the repo root and keep the `workflows/agentic/` prefix when writing files.

## Subproject Roles

| Path | Role |
|---|---|
| `workflows/agentic/config/environments/` | Env YAML source of truth. |
| `workflows/agentic/arena/arena/environments/` | Env classes, CLI args, `get_env`, teleop/replay/policy rollout behavior. |
| `workflows/agentic/arena/arena/assets/` | Scene assets, USD constants, cameras, table/props/destinations. |
| `workflows/agentic/arena/arena/tasks/` | Observations, events, success/termination logic, task cfgs. |
| `workflows/agentic/arena/arena/runtimes/` | Runtime policy IO, camera publishing, action/state bridging. |
| `workflows/agentic/arena/arena/embodiments/` | Robot embodiment configs for SO-ARM, G1, Franka-style arm. |
| `workflows/agentic/policy/` | Policy stack dispatch, inference daemon, train CLIs, stack registries. |
| `workflows/agentic/dataset/` | HDF5 to LeRobot conversion and modality metadata. |
| `workflows/agentic/annotator/` | VLM success annotation and HDF5 filtering. |
| `workflows/agentic/mimic/` | HDF5 trajectory expansion. |
| `workflows/agentic/common/` | Shared config, robot constants, messaging utilities. |

## Existing Env Map

| Env | YAML | Env class | Assets | Task | Runtime | Policy stack |
|---|---|---|---|---|---|---|
| `scissor_pick_and_place` | `config/environments/scissor_pick_and_place.yaml` | `arena/environments/scissor_pick_and_place_environment.py` | `arena/assets/scissor_pick_and_place.py` | `arena/tasks/scissor_pick_and_place.py` | `arena/runtimes/scissor_pick_and_place.py` | `gr00t_n15` default, N1.7 alternative |
| `locomanip_tray_pick_and_place` | `config/environments/locomanip_tray_pick_and_place.yaml` | `arena/environments/locomanip_tray_pick_and_place_environment.py` | `arena/assets/locomanip.py` | `arena/tasks/tray_pick_and_place.py` | `arena/runtimes/locomanip_tray_pick_and_place.py` | `gr00t_n16` |
| `locomanip_push_cart` | `config/environments/locomanip_push_cart.yaml` | `arena/environments/locomanip_push_cart_environment.py` | `arena/assets/locomanip.py` | `arena/tasks/push_cart.py` | `arena/runtimes/locomanip_push_cart.py` | `gr00t_n16` |
| `assemble_trocar` | `config/environments/assemble_trocar.yaml` | `arena/environments/assemble_trocar_environment.py` | `arena/assets/assemble_trocar.py` | `arena/tasks/assemble_trocar.py` | `arena/runtimes/assemble_trocar.py` | `gr00t_n15`, inference-only |
| `ultrasound_liver_scan` | `config/environments/ultrasound_liver_scan.yaml` | `arena/environments/ultrasound_liver_scan_environment.py` | `arena/assets/ultrasound_liver_scan.py` | `arena/tasks/ultrasound_liver_scan.py` | `arena/runtimes/ultrasound_liver_scan.py` | `openpi_pi0` |

Paths in the table are under `workflows/agentic/`.

## Pattern Families

- Scissor SO-ARM: inline `InteractiveSceneCfg` in `arena/assets/scissor_pick_and_place.py`, wrapped by `ConfigAsset` and returned by `make_scissor_pick_and_place_scene_assets()`. Robot/cameras come from `SoArm101Embodiment`; the robot is not listed in the scene asset names.
- G1 locomanip: env classes extend `HumanoidEnvironmentBase`, fetch registered assets via `self.asset_registry.get_asset_by_name(...)`, and use G1 WBC/head-camera policy IO from the locomanip stack.
- Assemble trocar: G1 plus dex hands, inference-only. Do not add training hooks unless the codebase grows supported training for it.
- Ultrasound: Franka-style arm and `openpi_pi0`; inspect its YAML/task/runtime before changing camera or dataset fields because it does not follow the G1 GR00T modality pattern.

## Load Order for Env Code Work

1. Selected env YAML in `workflows/agentic/config/environments/`.
2. Selected source env class in `arena/arena/environments/`.
3. Selected assets file and constants in `arena/arena/assets/`.
4. Selected task file in `arena/arena/tasks/`.
5. Selected runtime in `arena/arena/runtimes/`.
6. Matching policy stack files in `workflows/agentic/policy/<stack>/` and `workflows/agentic/policy/policy_routing.py` when policy dispatch changes.

Do not infer APIs from names alone. Read the closest existing implementation first, then fork and make minimal changes.

## Registration Checks

Use these before claiming a new or modified env is wired correctly:

```bash
workflows/agentic/policy/run.sh --list-envs
workflows/agentic/arena/run.sh --env <env> --dry-run
workflows/agentic/policy/run.sh --env <env> --dry-run
python -m py_compile <changed-python-files>
```
