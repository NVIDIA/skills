# Create Env Contract

Load this for every new `workflows/agentic` env. It defines the expected output shape and the checks that prove the env is ready for later scene-edit, dataset, and policy prompts.

## Inputs to Resolve

- Env id: lowercase snake case, used identically in YAML, class `name`, and file names.
- Source env: the closest existing env to fork.
- Scene/assets source: inline scene cfg or registry assets.
- Robot owner: embodiment/base class that owns the robot and action space.
- Policy stack and model/checkpoint.
- Cameras required at creation time.
- Objects, destinations, success rule, and reset randomization.

If the prompt fully matches a recipe, use the recipe's resolved choices instead of re-asking.

## Output Files

Create the minimal env shell:

| File | Requirement |
|---|---|
| `workflows/agentic/config/environments/<env>.yaml` | Source of truth for robot, policy, cameras, dataset, task text, train defaults. |
| `workflows/agentic/arena/arena/assets/<env>.py` | Scene assets; keep the source env's construction pattern. |
| `workflows/agentic/arena/arena/tasks/<env>.py` | Observations/events/terminations/success/reset logic. |
| `workflows/agentic/arena/arena/environments/<env>_environment.py` | Env class with `name = "<env>"` and `get_env`. |
| `workflows/agentic/arena/arena/runtimes/<env>.py` | Re-export an existing runtime when the stack already has one. |

Avoid shared edits during create. Do not edit `constants.py`, shared embodiments, `__init__.py`, policy packages, docs, or README files unless the requested env cannot work without that shared change.

## YAML Contract

Required YAML surfaces:

- `robot.type`
- `zenoh.camera_names`
- `policy.stack`
- `policy.health_port` unique across `config/environments/*.yaml`
- `arena.bridge_port` unique across `config/environments/*.yaml`
- `policy.model_repo` and `policy.model_revision`
- `policy.infer_module`
- `policy.train_module` or `null` for inference-only envs
- stack-appropriate task text: commonly `policy.language_instruction` for locomanip/openpi, `policy.task_description` for scissor-style stacks
- `arena.description`
- `arena.max_timesteps`
- `dataset.*` keys required by the chosen converter

If a camera should be used downstream, all of these must agree:

- sensor exists in the scene
- task `observations.policy` records it
- `zenoh.camera_names` publishes it
- policy camera mapping consumes it, e.g. GR00T `pov_cam_names_sim`
- `dataset.camera_mappings` exports it
- modality template/train modality config contain its video key

## Success Rule Contract

Use the simplest task-specific success rule that matches the prompt:

- Object sorting: each object must be inside its own paired destination; swapped placements fail.
- Placement tasks: object pose inside destination bounds.
- Scanning/reach tasks: use the existing source env's metric pattern.

For kinematic/static destination assets, do not assume `.data.root_pos_w`; use a helper that reads `.data.root_pos_w` when present and otherwise calls `entity.get_world_poses()`.

## Validation Contract

Run static checks:

```bash
python -m py_compile <changed-python-files>
workflows/agentic/policy/run.sh --list-envs
workflows/agentic/arena/run.sh --env <env> --dry-run
workflows/agentic/policy/run.sh --env <env> --dry-run
```

Then run a real bridge build with `workflows/agentic/arena/run.sh ensure-bridge --env <env> --log <run>/logs/bridge.log`. Probe `/objects`, `/object?name=...`, `/cameras`, and capture the viewport. A vision-capable agent must inspect the JPEG directly; a blind local agent uses `./local-agent/validate-env.sh <env>`.

Stop the bridge with `workflows/agentic/arena/stop.sh --env <env>` before reporting done.

## Done Means

- All five env files exist with correct names.
- The env appears in `policy/run.sh --list-envs`.
- Arena and policy dry-runs pass.
- The real bridge reaches ready.
- Expected objects/cameras are valid.
- Viewport capture shows a coherent scene.
- No extra shared files were modified unless explicitly justified.
