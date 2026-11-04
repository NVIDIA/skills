# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""First-episode, fixed-distribution DisplayPort checkpoint evaluation."""

from __future__ import annotations

import argparse
from importlib.metadata import version
import json
from pathlib import Path
import sys
import time

from storage import append_record, evaluation_provenance, read_session

started_epoch = time.time()

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--task', default='IsaacTraining-DisplayPortInsertion-Rizon4s-TaskSpace-ROS-Inference')
parser.add_argument('--checkpoint', type=Path, required=True)
destination = parser.add_mutually_exclusive_group(required=True)
destination.add_argument('--output', type=Path, help='Standalone evaluation JSON (outside a campaign).')
destination.add_argument('--journal', type=Path, help='Campaign evaluations.jsonl.')
parser.add_argument('--evaluation-id')
parser.add_argument('--session', type=Path)
parser.add_argument('--run-id')
parser.add_argument('--seed', type=int, required=True)
parser.add_argument('--num_envs', type=int, default=256)
parser.add_argument('--max_steps', type=int, default=260)
parser.add_argument('--run-spec', type=Path, default=None,
                    help='Normalized training spec for environment and agent overrides.')
args = parser.parse_args()
if args.journal and not args.evaluation_id:
    parser.error('--journal requires --evaluation-id')
if bool(args.session) != bool(args.run_id):
    parser.error('--session and --run-id must be supplied together')
spec = None
if args.session is not None:
    plan, state = read_session(args.session)
    attempt = next((item for item in state['attempts'] if item['run_id'] == args.run_id), None)
    if attempt is None:
        parser.error(f'run ID {args.run_id!r} was not found in session {args.session}')
    evaluation_provenance(args.session, plan, attempt)
    spec = attempt['spec']
elif args.run_spec is not None:
    spec = json.loads(args.run_spec.read_text())

# resolve_task_config uses Hydra, which otherwise sees this evaluator's CLI flags.
sys.argv = [sys.argv[0]]

import gymnasium as gym
import torch

from isaaclab.app import launch_simulation
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.utils import resolve_task_config
from rsl_rl.runners import OnPolicyRunner
import isaaclab_training.tasks  # noqa: F401; registers task

TASK = args.task
AGENT = 'rsl_rl_cfg_entry_point'

env_cfg, agent_cfg = resolve_task_config(TASK, AGENT, play_mode=False)


def apply_dotted(root, path: str, value) -> None:
    parts = path.split('.')
    target = root
    for part in parts[:-1]:
        target = target[part] if isinstance(target, dict) else getattr(target, part)
    last = parts[-1]
    if isinstance(target, dict):
        if last not in target:
            raise KeyError(path)
        target[last] = value
    else:
        if not hasattr(target, last):
            raise AttributeError(path)
        setattr(target, last, value)


if spec is not None:
    for key, value in spec['overrides'].items():
        if key.startswith('env.'):
            apply_dotted(env_cfg, key[4:], value)
        elif key.startswith('agent.'):
            apply_dotted(agent_cfg, key[6:], value)
        else:
            raise ValueError(f'Unsupported training override in evaluation: {key}')
    agent_cfg.num_steps_per_env = spec['num_steps_per_env']
env_cfg.scene.num_envs = args.num_envs
env_cfg.seed = args.seed
agent_cfg.seed = args.seed
agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, version('rsl-rl-lib'))
if agent_cfg.class_name != 'OnPolicyRunner':
    raise ValueError(f'Unsupported evaluation runner: {agent_cfg.class_name}')
params = env_cfg.events.reset_plug_curriculum.params
params.update(at_goal_prob=0.0, at_goal_prob_final=0.0, anneal_start_iter=0,
              anneal_end_iter=1, num_steps_per_env=agent_cfg.num_steps_per_env,
              approach_depth_range=[0.02, 0.06])
env_cfg.observations.policy.enable_corruption = False

launch_args = {'physics': 'isaacsim_physx', 'require_kit': True, 'headless': True,
               'visualizer': None, 'visualizer_explicit': True}
with launch_simulation(env_cfg, launch_args):
    raw_env = gym.make(TASK, cfg=env_cfg)
    base = raw_env.unwrapped
    completed = torch.zeros(args.num_envs, dtype=torch.bool, device=base.device)
    success = torch.zeros_like(completed)
    error_m = torch.full((args.num_envs,), float('nan'), dtype=torch.float32, device=base.device)
    keypoint_m = torch.full_like(error_m, float('nan'))
    terminal_step = torch.full((args.num_envs,), -1, dtype=torch.int32, device=base.device)
    current_step = 0
    original_reset = base._reset_idx

    def capture_reset(env_ids):
        if isinstance(env_ids, slice):
            ids = torch.arange(args.num_envs, device=base.device)[env_ids]
        else:
            ids = torch.as_tensor(env_ids, device=base.device, dtype=torch.long).reshape(-1)
        first = ids[~completed[ids]]
        if first.numel():
            mask, pos, kp = base._compute_success()
            success[first] = mask[first]
            error_m[first] = pos[first]
            keypoint_m[first] = kp[first]
            terminal_step[first] = current_step
            completed[first] = True
        return original_reset(env_ids)

    env = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(str(args.checkpoint))
    policy = runner.get_inference_policy(device=base.device)
    observations = env.get_observations()
    # RslRlVecEnvWrapper resets the environment during construction. Start
    # capturing only after that initial reset and after loading the policy.
    base._reset_idx = capture_reset
    try:
        with torch.inference_mode():
            for current_step in range(1, args.max_steps + 1):
                observations, _, dones, _ = env.step(policy(observations))
                policy.reset(dones)
                if bool(completed.all()):
                    break
        mask = completed.cpu()
        if not bool(mask.all()):
            raise RuntimeError(f'Only {int(mask.sum())}/{args.num_envs} first episodes completed after {args.max_steps} steps')
        payload = {
            'task': TASK, 'checkpoint': str(args.checkpoint.resolve()),
            'seed': args.seed, 'num_envs': args.num_envs, 'steps': current_step,
            'near_goal_probability': 0.0, 'approach_depth_m': [0.02, 0.06],
            'socket_pose_randomization': 'resolved task with recorded overrides', 'observation_corruption': False,
            'overrides': spec['overrides'] if spec else {},
            'physics': 'isaacsim_physx',
            'terminal_success_count': int(success.sum().item()),
            'terminal_success_rate': float(success.float().mean().item()),
            'terminal_mate_error_m_mean': float(error_m.mean().item()),
            'terminal_keypoint_error_m_mean': float(keypoint_m.mean().item()),
            'per_env_success': success.cpu().int().tolist(),
            'per_env_mate_error_m': error_m.cpu().tolist(),
            'per_env_terminal_step': terminal_step.cpu().tolist(),
        }
        if args.journal:
            append_record(args.journal, {**payload, 'kind': 'result',
                                        'evaluation_id': args.evaluation_id,
                                        'seconds': time.time() - started_epoch})
        else:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            tmp = args.output.with_name(args.output.name + '.tmp')
            tmp.write_text(json.dumps(payload, indent=2) + '\n')
            tmp.replace(args.output)
        print('[EVAL_RESULT]', json.dumps({k: payload[k] for k in
              ('checkpoint', 'seed', 'num_envs', 'steps', 'terminal_success_count',
               'terminal_success_rate', 'terminal_mate_error_m_mean')}), flush=True)
    finally:
        env.close()
