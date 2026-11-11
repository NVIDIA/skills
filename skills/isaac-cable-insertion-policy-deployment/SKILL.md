---
name: isaac-cable-insertion-policy-deployment
description: Export a trained Isaac Lab DisplayPort task-space checkpoint with LEAPP, run it through Isaac ROS Deploy and Isaac Sim 6.0 in separate Docker containers, verify training asset parity, and record insertion video and measured results. Use for post-training simulation deployment and testing of cable insertion policies.
license: CC-BY-4.0 AND Apache-2.0
metadata:
  author: "NVIDIA Isaac Team <info@nvidia.com>"
  tags:
    - isaac-lab
    - isaac-sim
    - isaac-ros
    - leapp
---

# Isaac Cable Insertion Policy Deployment

Turn a trusted DisplayPort training checkpoint into a fresh LEAPP export and a
recorded simulation trial. Use the ROS Node LEAPP runtime with Triton and
Isaac Sim **6.0.0** in separate Docker containers. Compatibility checking and
simulation use the same simulator container and image. Use NVIDIA images and packaged ROS dependencies.

The supported task is
`IsaacTraining-DisplayPortInsertion-Rizon4s-TaskSpace-ROS-Inference` with the stock
recurrent actor and operational-space controller. The harness validates its
supported observation/action/controller contract before export. Joint-space and
incompatible task variants require a deliberate implementation change.

## Inputs

Find these from user input or the training report; ask only for missing inputs:

- Absolute `isaaclab_training` package path from the manipulation repo,
  with its verified `.venv` and locked LEAPP export dependencies. Discover and reuse
  the training skill's recorded installation and environment plan; honor explicit paths.
- Explicit `.pt` checkpoint with the originating `params/env.yaml` and
  `params/agent.yaml` alongside it. Do not silently select the newest checkpoint.
- Docker with GPU access and locally available Isaac Sim 6.0.0 and Isaac ROS 5.0
  images. Use the official defaults in [setup](references/setup.md), or explicit
  compatible image references supplied by the user. The simulator container needs ffmpeg.

Read [setup](references/setup.md) before installing anything. Reuse explicit EULA
acceptance from the current conversation or a saved receipt covering the selected
simulator version. Training's 6.1 receipt covers export; the 6.0 deployment
image needs consent covering that scope.
If unavailable, ask for acceptance before running Isaac Sim; a timeout is not
consent. Pass `--accept-eula` only after acceptance. Never store credentials in logs.

## Procedure

1. Read [interfaces and assets](references/interfaces.md). Locate this skill's
   absolute directory; run all orchestration commands from the training
   package root. The companion `isaac-cable-insertion-policy-training` skill must be installed
   alongside this skill for shared environment and provenance checks. Do not modify
   Isaac Lab or the training project source to make export work.
2. Create a unique output path, by default
   `<skill>/logs/YYYY-MM-DD_HH-MM-SS_UTC__checkpoint-NNN`. An explicit writable
   output root is supported. Do not copy scripts or source diffs into logs or
   create provenance archives. The parent driver creates the directory itself.
3. Run the committed driver. Inspect its live logs and report progress. For a
   long tool call, poll at a modest interval; preserve its process until completion.
   An existing tmux installation may host the command, but is not required.
4. Require the driver's container preflight, **fresh** numerical export validation,
   and synchronized ROS trial to pass. [Setup](references/setup.md) defines the
   container checks; [interfaces](references/interfaces.md#policy-and-controller)
   defines the policy timing contract.
5. Inspect the video and `result.json`, `trace.json`, asset checks, ROS logs, and
   exporter validation. Report measured insertion success only when the complete
   attempt passes the saved task's success criterion. Keep failures visible.
6. On infrastructure errors, read [troubleshooting](references/troubleshooting.md).
   Resolve a clear packaging/path problem and rerun in a **new** session. Do not
   tune contact properties, move the socket, disable collisions, or change the
   checkpoint to manufacture a successful video. See
   [trial and retry options](references/setup.md#trial-and-retry-options) for
   bounded repeats of valid trials and handling invalid timing evidence.
7. Confirm owned processes have stopped. Return the report, export bundle, video
   and measured outcome. If insertion fails, say so and identify the next diagnostic.

Example (replace paths with discovered absolute inputs):

```bash
cd "$TRAINING_ROOT"
/usr/bin/python3 "$SKILL_ROOT/scripts/deploy.py" \
  --training-root "$TRAINING_ROOT" \
  --training-plan "$TRAINING_PLAN" \
  --checkpoint "$CHECKPOINT_PATH" \
  --sim-image nvcr.io/nvidia/isaac-sim:6.0.0 \
  --output "$SKILL_ROOT/logs/$SESSION_NAME" \
  --install-sim-runtime --install-runtime --accept-eula --domain-id 211
```

See [driver options](references/setup.md#driver-options) for prepared images,
Docker access, environment selection, export/preflight modes, and trial limits.
Choose a domain unused by other ROS applications; never share a physical robot's
domain. The skill launches simulation only and does not start a hardware driver.

## Interpretation

The clip shows **insertion from a prepared grasp**, not autonomous pickup or the
full documented manipulation mission. Setup places the plug before recording;
recorded motion uses contact dynamics and the unchanged exported policy. The
first second holds the starting pose, then the policy controls insertion.

See [current Docker validation](evals/validation.md) for the timing regression,
its fix and the measured verification results. One successful attempt is integration evidence,
not a policy success-rate estimate.
Asset parity checks compare against saved training URLs resolved at test time;
without training-time hashes they do not prove historical byte identity.

## Outputs and report

Keep one short `report.md` per invocation, plus machine-readable evidence:

| Path | Contents |
| --- | --- |
| `params/` | Saved `agent.yaml` and `env.yaml` |
| `export/` | Fresh ONNX, LEAPP YAML and initial recurrent state |
| `input.json`, `asset-manifest.json` | Checkpoint hash, source versions, training assets and supported contract |
| `commands.jsonl`, `*.log` | Exact commands, export validation and runtime diagnostics |
| `environment.json`, `export-runtime.json`, `runtime.json`, `*image.json`, `*packages.log` | Simulator build, image identity and installed package versions |
| `compatibility.log`, `sim-preflight.log`, `sim-container.log` | Simulator compatibility, encoder and container setup evidence |
| `attempt-NN/` | `video.mp4`, `result.json`, `trace.json`, policy/adapter logs, `policy-sync.jsonl` and asset verification |
| `assets-cache/` | Flattened asset layers fetched for this invocation; no source-repository copies |

Final response: outcome and error in millimeters; checkpoint and simulator build;
links to video/report/export; any unresolved issue or limitation.

## References

- [NVIDIA DisplayPort simulation tutorial](https://nvidia-isaac-ros.github.io/manipulation/reference_workflows/isaac_for_manipulation/tutorials/isaac_sim/tutorial_displayport_task_space_isaac_sim.html)
- [Training and export](https://nvidia-isaac-ros.github.io/manipulation/training/tasks/displayport_insertion.html)
- [Isaac ROS 5.0 setup](https://nvidia-isaac-ros.github.io/v/release-5.0/getting_started/index.html)
- [Dependency inventory](references/dependencies.md)
