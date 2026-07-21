# Env Authoring Patterns

Load this with `skills/i4h-workflow/references/repo-map.md` when creating or forking an env. The model should copy the closest working pattern and adjust it, not invent a new architecture.

## First Reads

Before editing, read the selected source env's YAML, env class, assets file, task file, runtime file, and relevant policy stack files. For hybrid envs, read both the scene-source env and the robot-owner env.

## Component Decisions

Resolve these before code generation:

- New env id and closest source env.
- Scene/asset source.
- Robot owner and embodiment class/base class.
- Policy stack and foundation model or checkpoint.
- Cameras that must be rendered, published, recorded, and trained.
- Success rule and reset randomization.
- Validation commands and whether bridge validation can launch on this host.

## Forking Rules

- New env class lives at `workflows/agentic/arena/arena/environments/<env>_environment.py` and must define `name = "<env>"`.
- New YAML lives at `workflows/agentic/config/environments/<env>.yaml`; class name and YAML id must match for automatic discovery.
- Fork the nearest assets/task/runtime style. Preserve inline-scene vs asset-registry construction.
- Use `train_module: null` or omit training only for inference-only envs.
- Keep task text in the key used by the stack: scissor-style YAMLs use `policy.task_description`; locomanip/openpi-style YAMLs commonly use `policy.language_instruction`.

## Pattern Selection

| Desired env | Start from | Keep this pattern |
|---|---|---|
| SO-ARM table manipulation | `scissor_pick_and_place` | Inline `InteractiveSceneCfg` assets, SO-ARM embodiment, scissor task/runtime style. |
| G1 loco-manipulation | `locomanip_tray_pick_and_place` or `locomanip_push_cart` | `HumanoidEnvironmentBase`, asset registry, WBC/head camera, `policy.locomanip.*`. |
| G1 with a scissor-style table scene | Scene from `scissor_pick_and_place`, robot from locomanip | Hybrid: env extends G1 owner base, but scene asset file keeps scissor inline `ConfigAsset` shape. |
| Ultrasound/openpi | `ultrasound_liver_scan` | openpi PI0 YAML/task/runtime conventions. |

## Hybrid Guardrails

- Robot integration comes from the robot owner. For G1 policies, use `HumanoidEnvironmentBase` and registry embodiment; a raw `ArticulationCfg` cannot run the WBC policy stack.
- Scene construction comes from the scene source. Do not convert scissor inline `InteractiveSceneCfg` to locomanip registry assets unless the user explicitly asks for that refactor.
- G1 head camera is provided by the embodiment; do not add duplicate head cameras to the forked scissor scene cfg.
- Keep a ground plane and match G1 base height to ground z. Verify robot/table footprint clearance in the bridge.

## Policy Stack Touchpoints

| Stack | YAML policy fields to verify | Typical modules |
|---|---|---|
| `gr00t_n15` | `model_repo`, `embodiment_tag`, `data_config`, `infer_module`, optional `train_module` | `policy.scissor_pick_and_place.*` or supported N1.5 task modules |
| `gr00t_n16` | `language_instruction`, `pov_cam_names_sim`, G1 joint config paths, `infer_module`, `train_module` | `policy.locomanip.infer.infer`, `policy.locomanip.train.train` |
| `gr00t_n17` | TRT-related scissor alternative fields | inspect `gr00t_n17` stack and scissor YAML before use |
| `openpi_pi0` | openpi repo/model fields, action horizon, camera/state conventions | `policy/openpi_pi0` |

## Camera and Dataset Wiring

When adding a camera that should be used downstream, update all relevant surfaces in one pass:

- Assets: add/render the camera sensor.
- Task observations: add the camera to the `policy` observation group so it is recorded.
- Env YAML `zenoh.camera_names`.
- Env YAML policy camera fields such as `pov_cam_names_sim` for GR00T stacks.
- Env YAML `dataset.camera_mappings` and any modality template/train modality config required by that stack.
- Re-record data. Old HDF5 recordings without the new camera cannot be made equivalent by YAML edits.

## Validation Sequence

Run static checks first, then bridge validation for geometry and cameras:

```bash
workflows/agentic/policy/run.sh --list-envs
workflows/agentic/arena/run.sh --env <env> --dry-run
workflows/agentic/policy/run.sh --env <env> --dry-run
python -m py_compile <changed-python-files>
```

For create tasks, bridge validation is part of the implementation unless Isaac Sim cannot launch on the host. Use the scene-edit workflow for bridge endpoint details.
