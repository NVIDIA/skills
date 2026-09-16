---
name: i4h-workflow-train-rl
description: Use when training, evaluating, or exporting Workflow policies with online RSL-RL or RLinf, including RL checkpoint and Workflow handoff.
license: Apache-2.0
metadata:
  author: "Isaac for Healthcare Team <isaac-for-healthcare-support@nvidia.com>"
  version: "0.8.0"
  tags:
    - isaac-for-healthcare
    - i4h
    - reinforcement-learning
    - rsl-rl
    - rlinf
---

# Train a Workflow Policy with RL

## Purpose

Resolve a maintained online-RL profile, verify its Scene/objective/model contracts, run the selected vectorized trainer, evaluate and export its artifact, and hand that artifact to normal Workflow policy validation.

## Requirements

- Run the Workflow setup skill first so its uv environments and pinned third-party checkouts are available.
- Use a CUDA-capable Isaac Lab/Arena runtime for the RSL-RL workflow.
- For Trocar, provide a local GR00T N1.5 3B base or SFT checkpoint and two visible local GPUs for the isolated controller and simulator runtimes. A compatible checkpoint is a complete local Hugging Face directory that the pinned GR00T N1.5/RLinf loader accepts without conversion; it must retain the N1.5 3B architecture and support the maintained three-camera plus 28-joint observation mapping and 28-D policy action head. Reject another model family, an exported inference-only Task artifact, or a checkpoint whose config changes those interfaces.

## Instructions

1. Resolve the checkout and supported profiles.
2. Confirm the Workflow, Scene, observations, actions, rewards, resets, termination, trainer, and runtime Task contracts.
3. Dry-run the exact requested configuration.
4. Preflight the selected trainer runtime and train in the foreground.
5. Evaluate simulator success, export the policy, and validate it through the normal Workflow runner.

## Resolve the checkout

```bash
export I4H_WORKFLOWS_REPO_URL="${I4H_WORKFLOWS_REPO_URL:-https://github.com/isaac-for-healthcare/i4h-workflows}"
I4H_REPO_DIR_NAME="${I4H_WORKFLOWS_REPO_URL%/}"
I4H_REPO_DIR_NAME="${I4H_REPO_DIR_NAME##*/}"
I4H_REPO_DIR_NAME="${I4H_REPO_DIR_NAME##*:}"
I4H_REPO_DIR_NAME="${I4H_REPO_DIR_NAME%.git}"
[ -n "$I4H_REPO_DIR_NAME" ] || { echo "Cannot derive a checkout name from I4H_WORKFLOWS_REPO_URL" >&2; exit 2; }
ROOT="${I4H_WORKFLOWS:-$(git rev-parse --show-toplevel 2>/dev/null)}"
if [ ! -d "$ROOT/workflows/i4h_workflows" ]; then
  ROOT="${I4H_WORKFLOWS:-$HOME/$I4H_REPO_DIR_NAME}"
  [ -d "$ROOT/workflows/i4h_workflows" ] || git clone "$I4H_WORKFLOWS_REPO_URL" "$ROOT"
fi
export I4H_WORKFLOWS="$ROOT"
cd "$ROOT"
```

Treat this resolver as part of the skill contract. `I4H_WORKFLOWS_REPO_URL` selects the clone source; `I4H_WORKFLOWS` selects or reuses a checkout. Never replace an existing checkout.

## Resolve the profile and contracts

```bash
./train.sh rl list
./train.sh rl show <workflow>
./run.sh show <workflow> --mode policy
```

Read the profile under `./rl/profiles/`, its referenced declarative trainer config under `./rl/config/`, Workflow, Scene manifest and implementation, Arena objective config, embodiment, and runtime policy Task manifest before a long run.

Select the maintained path from the profile:

| Workflow | Trainer | Starting artifact | Training observations/actions | Export and runtime |
| --- | --- | --- | --- | --- |
| `ultrasound_probe_reach` | RSL-RL PPO | None; train from scratch | 34-D joint/probe/target state → 6-D relative EE pose | TorchScript `policy.pt` → `rsl_rl/ultrasound_probe_reach` in-process Task |
| `assemble_trocar` | RLinf PPO actor/critic | Local GR00T N1.5 SFT/base checkpoint | Three cameras + 28 arm/hand joints → 28 policy actions padded to the 43-D Scene action | Native RLinf run bundle → GR00T inference export → existing remote `gr00t_n15/assemble_trocar` Task |

For `ultrasound_probe_reach`, require the target to be sampled from verified upper-torso surface points, the table and phantom to remain fixed, success to require both position and orientation tolerance for consecutive steps, and the exported Task observation order to match training exactly.

For `assemble_trocar`, require Unitree G1 with Dex3 hands, `front` and both wrist cameras, the current 87-value body state (29 positions + 29 velocities + 29 torques), 14 Dex3 joint positions, the 28-D GR00T arm/hand mapping, the 15-value body-action prefix, and the maintained `g1_trocar` reward/termination contract. Do not copy another Trocar environment into the training tree.

## Author a new RL-backed Workflow

1. Create and visibly validate the Workflow and Scene before adding training. Keep assets, embodiment, observations, actions, rewards, resets, termination, and success in their normal Arena and Workflow owners.
2. Add `./rl/profiles/<workflow>.yaml` using the schema below. The filename and `workflow` value must match.
3. Add one declarative YAML trainer config under `./rl/config/`; use `<workflow>_<algorithm>_<backend>.yaml` and point `trainer_config` to it as `../config/<file>.yaml`.
4. Reuse a generic backend under `./rl/i4h_rl/backends/`. Add `./rl/i4h_rl/adapters/<workflow>.py` only when the maintained Scene needs workflow-specific observation, action, registration, or evaluation conversion. Do not add workflow branches to `cli.py`, `sim_server.py`, or a package `__init__.py`.
5. Train and evaluate the native checkpoint, export it into a reusable in-process or remote policy Task, add that Task to the Workflow's `policy` TaskGraph, and validate simulator success through the normal Workflow runner. Compare the runtime observation/action values with training, not only their dimensions and ordering: preserve coordinate frames, quaternion convention, normalization, action scaling, previous-action state, and reset semantics.

```yaml
schema_version: 1
workflow: <workflow>
scene: <scene>
trainer: <rsl_rl-or-rlinf>
algorithm: ppo
adapter_module: i4h_rl.adapters.<workflow>
trainer_config: ../config/<workflow>_ppo_<backend>.yaml
train_task_id: <trainer-environment-id>
eval_task_id: <trainer-evaluation-environment-id>
task_description: <short instruction>
action_dof: <scene-action-width>
policy_action_dof: <policy-action-width>
state_dof: <state-observation-width>
cameras: []
default_num_envs: <positive-integer>
default_epochs: <positive-integer>
simulation:
  env_spacing: <positive-metres>
  presets: physx
  enable_cameras: false
```

Run `./train.sh rl show <workflow>` and a small `--dry-run` immediately. Profile loading must reject missing sources, unknown fields, unsupported backends, inconsistent dimensions, cameras absent from the Scene, mismatched RLinf task IDs/action mapping, and malformed RSL-RL config ownership before a simulator starts.

## Dry-run

RSL-RL from scratch:

```bash
./train.sh rl ultrasound_probe_reach \
  --num-envs 128 \
  --epochs 400 \
  --dry-run
```

RLinf foundation-policy post-training:

```bash
./train.sh rl assemble_trocar \
  --model-path /absolute/path/to/gr00t-sft-checkpoint \
  --num-envs 64 \
  --epochs 1000 \
  --dry-run
```

Inspect the resolved Scene, trainer, task IDs, observation/action dimensions, environment count, iteration/epoch count, config path, starting model when required, and explicit overrides. Keep user-requested resource values exact.

## Preflight and train

The lightweight `rl` uv project owns profile resolution and command orchestration. Its trainer configs are YAML and do not import Isaac Lab. RSL-RL backend integration runs in the Arena environment because its trainer and simulator dependencies are compatible. Set its runtime explicitly only when the prepared Arena environment is not appropriate:

```bash
export I4H_RL_PYTHON=/absolute/path/to/arena-rsl-runtime-python
```

Train the compact RSL-RL example:

```bash
./train.sh rl ultrasound_probe_reach \
  --num-envs 128 \
  --epochs 400
```

Resolve `TRAIN_RUN` from the timestamped `run dir:` path printed by the training command. Do not guess or select a run by recency.

Trocar uses two isolated processes on one host because GR00T N1.5/RLinf requires Python 3.11 while the current Isaac Sim/Arena runtime requires Python 3.12. The model controller defaults to `tasks/gr00t_n15/.venv/bin/python` on physical GPU 0, the simulator defaults to `arena/.venv/bin/python` on physical GPU 1, and observations/actions cross a local Unix-socket data bridge. Override those runtimes when needed:

```bash
export I4H_RL_PYTHON=/absolute/path/to/gr00t-rlinf-python
export I4H_RL_SIM_PYTHON=/absolute/path/to/isaac-sim-arena-python
```

Train the GR00T/RLinf profile only with a compatible local starting checkpoint, the pinned RLinf checkout, and two visible GPUs:

```bash
./train.sh rl assemble_trocar \
  --model-path /absolute/path/to/gr00t-sft-checkpoint \
  --num-envs 64 \
  --epochs 1000
```

Keep training in the foreground. Preserve the run directory and exact command on failure. Do not silently lower environment counts or epochs. The current RSL-RL launcher does not expose resume or distributed multi-GPU training; a second GPU is not used automatically. Trocar's two GPUs isolate the model and simulator processes rather than distributing one trainer across both GPUs.

## Evaluate, export, and hand off

Evaluate the RSL-RL checkpoint over independent randomized episodes:

```bash
TRAIN_RUN="$PWD/runs/ultrasound_probe_reach/YYYYMMDD_HHMMSS"

./train.sh rl ultrasound_probe_reach \
  --eval \
  --checkpoint "$TRAIN_RUN/model_final.pt" \
  --episodes 20
```

Export a simulator-compatible TorchScript actor, then validate the concrete Workflow Task:

```bash
./train.sh rl export ultrasound_probe_reach \
  --checkpoint "$TRAIN_RUN/model_final.pt" \
  --output-dir "$TRAIN_RUN/exported"

./run.sh ultrasound_probe_reach --policy \
  --checkpoint "$TRAIN_RUN/exported/policy.pt" \
  --episodes 20
```

Successful RLinf training writes `checkpoint.json` in the run directory. It points to the native FSDP checkpoint and records the starting GR00T model, so evaluation accepts the run bundle directly without repeating `--model-path`. A successful evaluation must write `evaluation.json` with non-empty TensorBoard metrics and at least one trajectory. Export also discovers the resolved RLinf training config stored in that run bundle:

```bash
TRAIN_RUN="$PWD/runs/assemble_trocar/YYYYMMDD_HHMMSS"

./train.sh rl assemble_trocar \
  --eval \
  --checkpoint "$TRAIN_RUN" \
  --video

./train.sh rl export assemble_trocar \
  --checkpoint "$TRAIN_RUN" \
  --output-dir "$TRAIN_RUN/exported"

./run.sh assemble_trocar --policy \
  --checkpoint "$TRAIN_RUN/exported" \
  --episodes 1
```

Treat the native trainer checkpoint as the training result and evaluate it before export. The GR00T inference export is runtime packaging for the remote Task, not a substitute for checkpoint evaluation. Require exit status 0, requested evaluation episodes when the trainer exposes them, non-empty metrics, expected checkpoint/export artifacts, and normal Workflow simulator success. Never claim success from loss curves or training exit alone.

## Troubleshooting

Report the first missing runtime dependency, checkpoint path, registration failure, observation/action mismatch, non-finite loss, CUDA memory error, distributed/Ray/FSDP error, environment construction failure, or missing success artifact. Preserve the failed run directory. Do not change the Scene objective or resource settings without user direction.

## Limitations

The maintained profiles are `ultrasound_probe_reach` with RSL-RL and `assemble_trocar` with RLinf. RSL-RL resume, RSL-RL evaluation video, and distributed multi-GPU launch are not yet exposed. Trocar currently requires two visible local GPUs and its Unix-socket simulator bridge is single-host; a distributed simulator/controller deployment is not exposed. This skill does not perform supervised LeRobot fine-tuning or make unsupported workflows trainable.

## Examples

- `Train the ultrasound probe reach policy with PPO and evaluate 20 episodes.` → dry-run the RSL-RL profile, train from scratch, evaluate randomized episodes, export TorchScript, and validate the Workflow Task.
- `RL post-train the Trocar policy from my local GR00T checkpoint.` → dry-run the RLinf profile, verify the G1/camera/action mapping, train, export a loadable GR00T checkpoint, and validate the remote policy Task.

## Completion gate

Report the Workflow and Scene, trainer/profile/config, starting checkpoint when applicable, observation/action/reward/reset/termination contract, requested and completed resources, run directory, evaluation metrics, checkpoint and export artifacts, exact Workflow validation command and success rate, exit status, and any remaining runtime or hardware limitation.
