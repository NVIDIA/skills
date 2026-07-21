# G1 Surgical Tool Sort Recipe

Load this only when the user asks to create `g1_surgical_tool_sort` from `scissor_pick_and_place` using the Unitree G1, or when repairing that generated env. This recipe resolves the choices for the first quick-start create prompt; follow-up edits such as red cube, G1 repositioning, room camera, and bake are scene-edit work.

## Resolved Choices

Do not re-ask or rename these:

| Choice | Value |
|---|---|
| Env id | `g1_surgical_tool_sort` |
| Scene/assets source | `scissor_pick_and_place` inline `InteractiveSceneCfg` + `ConfigAsset` + `make_*_scene_assets()` pattern |
| Robot owner | `locomanip_tray_pick_and_place` Unitree G1 via `HumanoidEnvironmentBase`, registry embodiment, WBC, and head camera |
| Policy stack | `gr00t_n16` locomanip, reused as-is with base model `nvidia/GR00T-N1.6-3B` |

Create exactly the five env shell files from `create-contract.md`:

1. `workflows/agentic/arena/arena/assets/g1_surgical_tool_sort.py`
2. `workflows/agentic/arena/arena/tasks/g1_surgical_tool_sort.py`
3. `workflows/agentic/arena/arena/environments/g1_surgical_tool_sort_environment.py`
4. `workflows/agentic/arena/arena/runtimes/g1_surgical_tool_sort.py`
5. `workflows/agentic/config/environments/g1_surgical_tool_sort.yaml`

Do not create a policy package, README, docs, modality files, or shared edits for this initial create prompt. Do not edit `constants.py`, shared embodiments, `__init__.py`, or `core/humanoid_base.py`.

## Load First

Read the normal create references plus the source env files before writing code:

- `skills/i4h-workflow/references/repo-map.md`
- `skills/i4h-workflow-create/references/create-contract.md`
- `skills/i4h-workflow-create/references/env-authoring-patterns.md`
- `skills/i4h-workflow-create/references/hybrid-layout-rules.md`
- source YAML/assets/task/env/runtime for `scissor_pick_and_place` and `locomanip_tray_pick_and_place`

Fast-path source files:

- `workflows/agentic/config/environments/scissor_pick_and_place.yaml`
- `workflows/agentic/arena/arena/assets/scissor_pick_and_place.py`
- `workflows/agentic/arena/arena/tasks/scissor_pick_and_place.py`
- `workflows/agentic/arena/arena/environments/scissor_pick_and_place_environment.py`
- `workflows/agentic/config/environments/locomanip_tray_pick_and_place.yaml`
- `workflows/agentic/arena/arena/environments/locomanip_tray_pick_and_place_environment.py`

Fast path for this exact quick-start prompt: after loading this recipe and the listed references, read only the fast-path source files above before writing. Do not read the full locomanip runtime; the generated runtime is the exact re-export shown below. Do not inspect third-party `TaskBase`, `IsaacLabArenaEnvironment`, or `Scene` internals unless validation fails. Do not re-check all existing YAMLs except a short grep for used `health_port` and `bridge_port`. Once those reads are done, write the five files.

Write barrier: after the short port grep and source reads above, the next action must create or edit the five contract files. Do not continue architecture analysis, do not inspect framework internals, do not calculate USD bbox dimensions in the language model, and do not wait for a bridge run before writing. Use the first-write seed values in this recipe; bridge validation is where exact height mistakes are discovered and fixed.

Use `health_port: 8777` and `bridge_port: 8876` unless those ports are already present in `workflows/agentic/config/environments/*.yaml`; if they are present, choose the next unused consecutive values.

Fast-output values for the first write:

- Asset class: `G1SurgicalToolSortSceneCfg(InteractiveSceneCfg)`.
- Asset factory: `make_g1_surgical_tool_sort_scene_assets()` returns `ground`, `table`, `scissors`, `tweezers`, `tray_a`, `tray_b`, `dome_light`, and `directional_light`.
- Asset USDs: `SCISSOR_TABLE_USD`, `SCISSORS_USD`, `SURGICAL_TWEEZERS_USD`, and `SCISSOR_TRAY_USD`; no new constants.
- Initial compact layout: ground below the robot by about eight tenths of a meter; table centered in front of the robot on positive x with the scissor-table footprint scale; tools near the table centerline; trays farther apart on the y axis. Do not derive these from USD bounds before the first write.
- Env class: `G1SurgicalToolSortEnvironment(HumanoidEnvironmentBase)`, `name = "g1_surgical_tool_sort"`, no top-level Isaac imports, imports for assets/task/Scene/Pose/IsaacLabArenaEnvironment inside `get_env`.
- G1 pose for the initial create: stand behind the table on negative x with y/z at zero, identity rotation, and `apply_wbc_default_base_height(embodiment, base_height_m=0.8)`.
- Task class: `G1SurgicalToolSortTask(TaskBase)`, `episode_length_s=30.0`, `env_spacing=4.0`, success pairs exactly `(("scissors", "tray_a"), ("tweezers", "tray_b"))`.
- Runtime: exactly `from arena.runtimes.locomanip_tray_pick_and_place import LocomanipPolicyIO, run  # noqa: F401`.
- YAML: fork locomanip G1 YAML, use `gr00t_n16`, base model `nvidia/GR00T-N1.6-3B`, `model_revision: null`, head camera only, locomanip infer/train modules, and dataset state/action dims `43`.

First-write seed constants; use these before bridge validation instead of deriving geometry:

```text
ground_z = -0.8
base_height_m = 0.8
table_x = 0.45
table_y = 0.0
table_z = -0.55
table_scale_x = 0.7
table_scale_y = 0.7
table_scale_z = 0.547
tool_x = 0.45
scissors_y = 0.06
tweezers_y = -0.06
tool_z = -0.249
tray_x = 0.45
tray_a_y = 0.24
tray_b_y = -0.24
tray_z = -0.286
robot_x = -0.4
robot_y = 0.0
robot_z = 0.0
```

## Assets

Fork `arena/arena/assets/scissor_pick_and_place.py` and keep its inline scene shape. Do not add scene cameras; the G1 embodiment supplies the policy head camera. Add two tools and two destinations: scissors -> `tray_a`, tweezers -> `tray_b`.

Use only existing USD constants from `arena/assets/constants.py`. Both trays reuse the existing scissor tray USD and differ by `visual_material`; do not add new constants. Keep scissors at the scissor source scale, set tweezers to native unit scale, and keep the scissor table footprint scale rather than expanding to full unit scale.

Tools are dynamic `RigidObjectCfg` with `mass_props` and `collision_props`. Trays are kinematic destinations: `AssetBaseCfg(..., rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True))`. Use snake_case cfg kwargs only.

Use the compact table-forward layout from `hybrid-layout-rules.md`: robot stands at negative x, table and props are at positive x, and the table near edge must clear the G1 footprint. Pair the ground and WBC height: ground z is about `-0.8`, and `apply_wbc_default_base_height(..., base_height_m=0.8)` is positive. Lower the tabletop to about 0.3 m below the waist, then place tabletop props by bbox, not root origin. The scissors, tweezers, and red cube bbox bottoms should equal the tabletop z. The scissor tray USD origin sits above its visual bottom; lower each tray so its bbox bottom is within 0-2 mm of the tabletop and there is no visible air gap. Do not reuse the scissor source's positive tabletop z.

Separate `tray_a` and `tray_b` far enough that success regions cannot overlap. A placement in one tray must never count for the other tray.

## Task

Fork the scissor task shape but extend `TaskBase` directly. Success is object-position only: `_SORT_PAIRS = (("scissors", "tray_a"), ("tweezers", "tray_b"))`, `success(env)` returns a per-env bool tensor, every paired tool must be inside its paired tray, and swapped placements fail. Do not include robot joints, end-effector pose, arm-home checks, settle tracking, success buffers, or tolerance args on the success signature.

Implement a reusable position helper that reads `.data.root_pos_w` when present and otherwise calls the scene entity method `entity.get_world_poses()[0][:, :3]`. Do not import `get_world_poses`; it is not an importable helper.

Add a local tray reset helper above the config classes. It must accept `(env, env_ids, pose_range, velocity_range, asset_cfg)` and use `set_world_poses(...)` for the kinematic tray entity. There is no `mdp.reset_xform_root_pose_uniform`. Reset tools with `mdp.reset_root_state_uniform`.

Events are exactly `reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")`, one reset per tool, and one reset per tray. Include `success` and `time_out` terminations. Match the source task method surface: `get_scene_cfg`, `get_termination_cfg`, `get_events_cfg`, `get_mimic_env_cfg`, `get_metrics`, `get_viewer_cfg`, and `modify_env_cfg`.

## Environment

Fork `arena/arena/environments/locomanip_tray_pick_and_place_environment.py`. Keep module-level imports lightweight: only `argparse` and `HumanoidEnvironmentBase`. Move assets, task, IsaacLab, IsaacLab-Arena, `Scene`, and pose imports inside `get_env`, because env discovery runs before Isaac Sim provides `carb`.

Extend `HumanoidEnvironmentBase`, set `name = "g1_surgical_tool_sort"`, fetch the registry embodiment, call `apply_wbc_default_base_height(embodiment, base_height_m=<positive ground depth>)`, and set the initial pose at negative x with identity rotation `(1, 0, 0, 0)`. A reliable starting point is roughly 0.8-0.9 m behind the table center; do not stand the robot at the table x, on the +x side, or with 180 degree yaw.

Build `Scene(assets=make_g1_surgical_tool_sort_scene_assets())` and do not add the locomanip room/background asset. Use a loco-manip horizon around 30 s, pass `env_spacing=getattr(args, "env_spacing", 4.0)`, and return `IsaacLabArenaEnvironment(...)` with `teleop_device=self._resolve_teleop_device(...)`.

## Runtime

Re-export the existing locomanip runtime:

```python
from arena.runtimes.locomanip_tray_pick_and_place import LocomanipPolicyIO, run
```

Do not copy the runtime loop.

## YAML

Fork `config/environments/locomanip_tray_pick_and_place.yaml` and keep stack/action/control/dataset/modality keys consistent with locomanip. Change only env-specific values: `model_repo: nvidia/GR00T-N1.6-3B`, `model_revision: null`, globally unique `policy.health_port`, globally unique `arena.bridge_port`, `policy.language_instruction` with the tool-to-tray color mapping, `train.output_dir` containing `g1_surgical_tool_sort`, and `arena.description` for the surgical tool sort task.

## Recipe Self-Check

Before validation, re-read the five generated files and check these failure modes:

- No extra files or shared edits exist.
- The env id is `g1_surgical_tool_sort` in every file and YAML path.
- Env file has no top-level heavy imports and no locomanip room/background asset.
- Runtime file is a short re-export.
- YAML uses the base GR00T N1.6 model, `model_revision: null`, and a health port unused by any other env YAML.
- G1 base height is positive, ground z is its negative, table/props are at negative z near the lowered tabletop, table/props are in positive x, and the G1 starts behind the table at negative x.
- Tool bbox bottoms are at the tabletop within 0-2 mm; a 1-5 cm air gap is a failure even if the tool entity is valid.
- Tray bbox bottoms are at the tabletop within 0-2 mm; a 1-5 cm air gap is a failure even if the tray entity is valid.
- Tools are `RigidObjectCfg`; trays are kinematic `AssetBaseCfg`; trays are visibly separated.
- Task success reads only tool/tray positions, uses the fallback position helper, has no robot-pose dependencies, and swapped placements fail.
- Task events start with `reset_scene_to_default`, then reset every tool and tray; no `reset_success_state`.
- Every `func=` helper is defined above the cfg class that references it.

Useful grep gate:

```bash
ENV=g1_surgical_tool_sort; A=workflows/agentic/arena/arena
ASSET=$A/assets/$ENV.py; TASK=$A/tasks/$ENV.py; ENVF=$A/environments/${ENV}_environment.py; YAML=workflows/agentic/config/environments/$ENV.yaml; ok=1
grep -EnH "disableGravity|kinematicEnabled|rigidBodyEnabled" "$ASSET" "$TASK" && { echo "FAIL camelCase cfg kwarg"; ok=0; }
grep -EnH "mdp\.reset_xform_root_pose_uniform|import +get_world_poses|from isaaclab_arena\.assets\.object_base import" "$TASK" && { echo "FAIL invalid tray helper/reset"; ok=0; }
grep -EnH "body_pos_w|default_joint_pos|reset_success_state" "$TASK" && { echo "FAIL robot-coupled success/reset"; ok=0; }
grep -EnH "base_height_m *= *-" "$ENVF" && { echo "FAIL negative base height"; ok=0; }
grep -EnH "Rheo|PickNPlaceTray" "$YAML" && { echo "FAIL fine-tuned parent checkpoint still present"; ok=0; }
grep -EnH "pre_op|background_scene" "$ENVF" && { echo "FAIL locomanip room/background included"; ok=0; }
grep -EnH "^(from arena\.(assets|tasks)|import isaaclab\b|from isaaclab\b|from isaaclab_arena\.(environments\.isaaclab_arena_environment|scene|utils))" "$ENVF" && { echo "FAIL top-level heavy import"; ok=0; }
find workflows/agentic/arena/arena workflows/agentic/config/environments -name '*surgical_tool*' ! -name 'g1_surgical_tool_sort*' 2>/dev/null | grep -q . && { echo "FAIL misnamed env file"; ok=0; }
[ $ok = 1 ] && echo "grep gate clean; still run bridge validation"
```

## Validation

Run the create validation contract in `create-validation.md`: `py_compile`, `policy/run.sh --list-envs`, arena and policy dry-runs, then `arena/run.sh ensure-bridge --env g1_surgical_tool_sort --log <run>/logs/bridge.log`. Resolve `BRIDGE_URL="$(arena/run.sh bridge-url --env g1_surgical_tool_sort)"`, then inspect `/objects`, key object poses, `/cameras`, and a viewport capture through that URL. A vision-capable agent must inspect the image directly; a blind local agent may use `./local-agent/validate-env.sh g1_surgical_tool_sort`. Stop with `workflows/agentic/arena/stop.sh --env g1_surgical_tool_sort`.

Do not report the env ready from static checks alone. The bridge must reach ready, the robot must remain upright, the trays/tools must be visibly resting on the tabletop with no tray air gap, and the viewport must show a coherent task scene.
