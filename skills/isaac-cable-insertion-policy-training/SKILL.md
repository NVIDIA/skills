---
name: isaac-cable-insertion-policy-training
description: Set up the standalone IsaacLabTraining package from NVIDIA-ISAAC-ROS/isaac_ros_manipulation, then launch, monitor, resume, evaluate, and tune DisplayPort insertion policies without a separate Isaac Lab source installation. Use for setup and campaign planning, dependency repair, curriculum or PhysX troubleshooting, video evidence, and long GPU runs.
license: CC-BY-4.0 AND Apache-2.0
metadata:
  author: "NVIDIA Isaac Team <info@nvidia.com>"
  tags:
    - isaac-lab
    - displayport-insertion
    - reinforcement-learning
---

# Isaac Cable Insertion Policy Training

## Purpose

Launch, monitor, troubleshoot, and tune DisplayPort insertion policies from the
manipulation repository’s standalone `isaaclab_training` package. Isaac Lab is a
pinned dependency installed by uv; never require a separate Isaac Lab checkout.
Keep the training project's source, `pyproject.toml`, and `uv.lock` unchanged.
The skill manages supplemental packages and ONNX Runtime selection in `.venv`.
The default is task-space ROS-inference training with video and reproducible run logs.

For a planning-only request, read the relevant bundled references and describe
the proposed commands and checks without running setup, preflight, or training.
Use supplied hypothetical facts as inputs; do not claim they were verified.

## Inputs

The training repository path is optional. Honor a supplied location or explicit
fresh-install request; otherwise check this skill's setup/session records for a
previously installed standalone repository and try to reuse its verified `.venv`
as described in [Setup](references/setup.md). If no suitable installation exists,
choose an unused `~/isaac_ros_manipulation` destination.
The working directory and plan’s `training_root` are the
repository’s `isaaclab_training/` subdirectory, which owns `.venv`, `pyproject.toml`,
and `uv.lock`. Accept an optional Git ref, run count, time budget, task variant,
seed, and hyperparameter overrides. Default to `release-r1`,
task-space training, and three completed runs when the budget permits.

## Instructions

For independent validation of an existing policy, follow
[Independent validation with fresh seeds](references/supervisor.md#independent-validation-with-fresh-seeds).
Reuse the verified checkpoint and originating session; choose unused evaluation seeds
and report the independent result separately. This request does not require retraining
or a new installation.

Read inputs, EULA, and preflight before launching. Use setup for installation or
repair, logging/supervisor for campaign execution, and tuning between runs:

1. [Inputs and boundaries](references/inputs.md): task defaults, repository access, and scope.
2. [Omniverse EULA](references/eula.md): read the consent rules before setup; combine missing consent with the setup/location choice and record an explicit Yes before installing or launching Isaac Sim.
3. [Setup and repair](references/setup.md): clone or reuse the public manipulation repository, sync the sim group with the documented CPU ONNX exclusion, install skill-managed supplements and requested video support, and verify before training.
4. [Preflight](references/preflight.md): verify the locked Isaac Lab wheel, editable training package, installed dependency consistency, GPU, and task configuration.
5. [Logging](references/logging.md): create a new invocation log and record every attempted run.
6. [Run and monitor](references/run-monitor.md): launch durably, detect missing exit markers, check video, and handle errors.
7. [Tuning and final report](references/tuning.md): compare runs, tune in the guide's order, and report back after completion.
8. [Reusable supervisor](references/supervisor.md): create a JSON campaign plan, launch with optional `tmux`, resume safely, and queue the next hypothesis.

Run from the selected `isaaclab_training` package root through the shared
[launch environment](references/environment.md). Discovery, consent checks and
repository creation and initial uv installation are the setup exceptions. Use the committed supervisor and
evaluator; keep one report and the supervisor's records in the skill's ignored
`logs/` directory. [Logging](references/logging.md) defines the artifacts and
provenance to retain. Respect the user's time budget and report the final outcome.

## Training command

Use the [canonical task-space command](references/run-monitor.md) from the
prepared `isaaclab_training` directory. That reference covers task selection,
video cadence, and monitoring. Choose and record shorter rollout, curriculum,
and iteration settings when a time limit makes the full run impractical.

## Troubleshooting

Use [Run and monitor](references/run-monitor.md) for PhysX collision stack overflow,
CUDA out-of-memory, video-source errors, bounded retries, and escalation of unclear failures.
