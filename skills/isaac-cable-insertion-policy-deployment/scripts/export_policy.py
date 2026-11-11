#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run the packaged exporter with explicit reset indices for legacy task events.

IsaacLab 3.0 uses slice(None) for all-environment resets. The DisplayPort event
implementation requires tensor indices. Normalize only during this export process;
restore the upstream method afterward. No repository files or policy weights change.
"""
from isaaclab_training.cli import export_policy as upstream

original_export = upstream.export_rsl_rl_agent


def compatible_export(*args, **kwargs):
    # Upstream enters launch_simulation before invoking this function.
    import torch
    from isaaclab.managers import EventManager
    from isaaclab_tasks.contrib.automate import factory_control

    original_apply = EventManager.apply
    add_legacy_ik_name = not hasattr(factory_control, "_get_delta_dof_pos")
    if add_legacy_ik_name:
        factory_control._get_delta_dof_pos = factory_control.get_delta_dof_pos
        print("EXPORT_COMPATIBILITY: alias renamed get_delta_dof_pos for task reset", flush=True)

    def explicit_indices(self, mode, env_ids=None, *positional, **keywords):
        if mode == "reset" and (env_ids is None or isinstance(env_ids, slice)):
            selector = env_ids if isinstance(env_ids, slice) else slice(None)
            env_ids = torch.arange(self._env.num_envs, device=self._env.device,
                                   dtype=torch.int32)[selector]
        return original_apply(self, mode, env_ids, *positional, **keywords)

    EventManager.apply = explicit_indices
    print("EXPORT_COMPATIBILITY: normalize all-environment reset indices", flush=True)
    try:
        return original_export(*args, **kwargs)
    finally:
        EventManager.apply = original_apply
        if add_legacy_ik_name:
            del factory_control._get_delta_dof_pos


if __name__ == "__main__":
    upstream.export_rsl_rl_agent = compatible_export
    raise SystemExit(0 if upstream.main() else 1)
