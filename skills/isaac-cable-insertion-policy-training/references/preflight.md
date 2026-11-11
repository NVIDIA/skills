## Preflight from the standalone package root

If the manipulation repository, uv, `.venv`, simulator, or task package is missing,
follow [Setup](setup.md). The agent performs installation and bounded repair;
the supervisor only verifies prerequisites. Run all checks from the selected
`isaaclab_training/` directory using its `.venv/bin/python`.

Enter preflight through the shared [process environment](environment.md), using
the same plan as setup. Check the recorded launch environment for intentional
exceptions; parent-shell library/import paths must not appear implicitly.

Before starting or resuming, the supervisor checks:

1. **Standalone package:** `training_root/pyproject.toml` names `isaaclab-training`;
   its `.venv` is the active probe environment. Git metadata is resolved from the
   parent manipulation repository. Record its commit, ref, remote, and dirty status;
   a present origin must identify `NVIDIA-ISAAC-ROS/isaac_ros_manipulation`.
2. **Pinned Isaac Lab dependency:** the project and `uv.lock` identify the same
   official wheel-builder Git commit. Installed distribution provenance must match
   that commit and version; the import must resolve inside the standalone venv.
   A separately installed editable Isaac Lab checkout does not qualify.
3. **Training and simulator:** the editable training distribution and module resolve
   to this package root. Isaac Sim matches `uv.lock` and the accepted 6.1 scope.
   The requested task is registered. No simulator is imported before EULA consent.
4. **Installed dependencies:** compare installed requirements and lockfile versions
   using the [setup policy](setup.md). Record satisfied project overrides and the
   version-bound LEAPP GPU-provider exception separately. Missing dependencies,
   unexplained conflicts, and both ONNX providers together block launch; a lockfile
   dry run alone is insufficient.
5. **Runtime:** uv, GPU access, available disk space, TensorBoard, and—when requested—a
   real MoviePy/FFmpeg encode and ImageIO decode. A missing video dependency blocks
   launch; disabling video skips that requirement.

The agent additionally checks that disk space is sufficient, existing GPU workloads, log-directory writability,
and `uv pip check` (allowing only documented project overrides and the reviewed
LEAPP GPU-provider substitution). Read the selected
supplement file, compare its pins with installed versions, and verify ONNX Runtime
imports and available execution providers. Verify that setup left the training
project's source, `pyproject.toml`, and `uv.lock` unchanged. Read the selected
task/agent defaults, including rollout, updates, mini-batches, rewards, noise,
randomization, action scale, reset curriculum, and video cadence. Record chosen
settings. Use a multiple of the recurrent mini-batch count for `num_envs`.

Estimate runtime from matching smoke evidence. The full 256-environment campaign
can exceed a short budget considerably. Shortening rollouts requires matching
`env.events.reset_plug_curriculum.params.num_steps_per_env`; shortening training
requires an explicit anneal schedule if the default would outlast it. Do not claim
that aggressive curriculum compression yields a trained policy.

```bash
training_cmd .venv/bin/python "$SKILL/scripts/supervisor.py" preflight --plan /absolute/path/to/plan.json
# Isolated CPU probes for diagnosis:
training_cmd .venv/bin/python "$SKILL/scripts/runtime_checks.py" installation --root "$PWD"
training_cmd .venv/bin/python "$SKILL/scripts/runtime_checks.py" dependencies --root "$PWD"
training_cmd .venv/bin/python "$SKILL/scripts/runtime_checks.py" preflight --video
```

These checks create no campaign and install nothing. Use `uv run --no-sync` after
setup to preserve the sim group and supplemental video dependencies. After repair,
repeat preflight and a bounded task/video smoke test before the requested campaign.
