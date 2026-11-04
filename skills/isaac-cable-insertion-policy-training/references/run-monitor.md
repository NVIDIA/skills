## Run and monitor

The default **full** task-space command, executed from the selected standalone training package root, is:

```bash
training_cmd uv run --no-sync isaaclab train --rl_library rsl_rl \
  --task IsaacTraining-DisplayPortInsertion-Rizon4s-TaskSpace-ROS-Inference \
  --num_envs 256 \
  --viz kit --video --video_length 200 --video_interval 19200
```

Set a unique `--run_name` and explicit, logged overrides when a run departs from defaults. Use the task-space `-ROS-Inference` ID by default; preserve an explicit user-supplied task ID, including joint-space variants. Record the selected task because it determines the policy observation/action contract. With 128 rollout steps, `--video_interval 19200` means roughly one clip per 150 PPO iterations; scale the interval when changing rollout length. Do not pass `--viz none` with `--video`.

Use the committed [supervisor](supervisor.md) for durable launch, progress checks,
and restart recovery. Monitor its compact status, GPU memory, artifacts and budget;
give the user progress and ETA updates. Check process state in the same namespace
as the trainer: a sandboxed `ps` or `nvidia-smi` may not see a host job. Do not rely
on a chat turn remaining active or a status file appearing eventually.

If a process disappears without an exit marker, log the exit code as **unavailable**. Check for the expected final TensorBoard iteration count, a nonempty final checkpoint at the expected step, the trainer's final training-time line, and decodable videos. If all are present, record an artifact-verified completion and continue only from that checkpoint; do not invent exit code 0. If evidence is incomplete, preserve the partial run and diagnose or restart it under a distinct run name. If the trainer remains alive after its final log line, check its process state before deciding it is complete or stalled. Keep the continuation logic restartable so the same completed run is not launched twice after a supervisor restart.

Use [`scripts/verify_run.py`](../scripts/verify_run.py) from the standalone training package root for the artifact check. Pass the run directory, raw console log, expected final scalar/checkpoint step, iteration count, and video length. Exit 0 means the artifacts are complete; exit 2 lists missing evidence. This checker cannot determine the trainer process exit code, so retain the recorded process status separately.

If a run fails, log the failure first. Apply the documented fixes for known failures, then restart as a new attempted run with a new name and linked log entry:

- **PhysX collision stack overflow:** inspect the exact `PxGpuDynamicsMemoryConfig::collisionStackSize` error; increase `sim.physics.gpu_collision_stack_size` from the task default `2**30` toward `2**31`, checking GPU memory, then retry.
- **CUDA out of memory:** first reduce `--num_envs` (for example 256 to 128). If still needed, lower plug/socket `solver_position_iteration_count` only while recording the physics trade-off. Disable video only when necessary, and mark that the resulting run has no clips.
- **Video source unavailable:** use `HEADLESS=1` with `--viz kit --video`, verify the Kit visualizer, and pass the video dependency/encoding preflight before retrying.
- **No observed progress:** inspect the recorded evidence source and time, TensorBoard, checkpoints, and probe errors. A supervisor timeout means it could not observe advancing progress; it is not proof of a simulator crash or that training never started. Do not change task physics or reset IK settings to repair buffered logging.
- For other errors, fix an obvious, bounded issue such as a path typo or missing editable install, log it, and retry once. If diagnosis is uncertain or the fix materially changes the task, stop and ask the user with the error and preserved log. Preserve the experiment settings unless the logged fix requires a change.
