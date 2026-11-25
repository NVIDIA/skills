<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Evaluate an existing task folder

Use this path for tasks already in Gym's supported Harbor task-folder format. Gym prepares their rows,
starts the `harbor` Environment Server, runs the selected agent session, and invokes each task's verifier.
Keep the user's task instructions, verifiers, resources and time limits as supplied. Converting another
task format or implementing an unsupported harness is a separate task.

## Install and choose the backend

Follow the environment setup in [local and API-endpoint runs](local-and-api.md): Python 3.13.14 or newer,
`uv>=0.13.4`, and an activated virtual environment in the user's project. Add the sandbox dependencies at
the same exact public preview revision:

```bash
python -m pip install \
  "nemo-gym[sandbox] @ git+https://github.com/NVIDIA-NeMo/Gym.git@a8eeecedb433b6c483077160a2cf9a06a09fab1b"
gym eval run --help
gym dataset validate --help
gym list agents
```

This Git-based pip installation needs no manual Gym checkout. Record the full revision with the result;
the positional task-target command is a preview capability, so do not substitute an older release merely
because its package version looks similar.

Set `TASK_DIR` to the user's existing task or dataset folder. A common single-step task has `task.toml`,
`instruction.md`, an image in `[environment].docker_image` or `environment/Dockerfile`, and `tests/test.sh`
or `tests/verifier.py`. Datasets may provide shared verifiers and multi-step tasks have step-specific files;
use their existing layout. A dataset's optional `dataset.toml` supplies its `[gym]` settings, including
`allowed_agents` and `[gym.defaults].num_repeats`.

Choose the provider for the task's actual features. `--sandbox docker` needs a running local Docker daemon
and access to the images. Build-mode Dockerfiles need a build tool; a remote provider also needs a registry
it can pull the built images from. OpenSandbox needs its configured service and credentials, and is required
for Compose tasks and separate-verifier network restrictions. Preserve task resource requirements when
choosing a backend. Harness startup can download or stage its runtime; see [supported agents](supported-agents.md).
The examples allow 1200 seconds for first server setup; this leaves the task's agent and verifier limits unchanged.

The Gym host must reach the upstream model endpoint. A harness running inside a task sandbox also reaches
Gym's Model Server on the Gym host. The generated config advertises an address for that model server;
the sandbox must be able to reach its allocated port. The upstream `--model-url` and the address a sandbox
uses to call Gym are different connections.

If the detected address is not assigned to this host, export `GYM_MODEL_HOST` with an assigned address
that the chosen sandbox can reach, using the narrowest reachable address on a trusted network. For
`inference_provider`, save this overlay as `task-network.yaml`;
for another selected model adapter, use its config key in place of `inference_provider`:

```yaml
use_absolute_ip: false
policy_model:
  responses_api_models:
    inference_provider:
      host: ${oc.env:GYM_MODEL_HOST}
```

When using this overlay, add `--config task-network.yaml` to both model smoke and full commands below.
This sets the bound and advertised Gym Model Server address; `--model-url` still points to the upstream
provider. The model smoke must demonstrate actual sandbox-to-Gym connectivity before a full run.

## Check reference solutions when available

```bash
gym dataset validate "$TASK_DIR" --sandbox docker +server_spinup_timeout_seconds=1200
```

This executes reference solutions and verifiers without calling a model. Read its per-task status and
`results/harbor/validation/<taskset>/rollouts_run_summary.json`. A task without a solution is unvalidated;
a passing reference solution establishes that the verifier accepts that solution. Keep this result separate
from the model evaluation. Do not replace task tests or solutions to obtain a pass.

An unloadable task stops the evaluation by default. Read the named reason before continuing.
`--skip-unloadable` deliberately evaluates a smaller population; use it only for a requested subset and
report every omitted task. `--only-tasks` and `--exclude-tasks` likewise select comma-separated task names
or globs, such as `--only-tasks 'hello,arithmetic-*'`. Name and record that selection with its result.

## Smoke, then run the dataset protocol

Select an existing harness and the model adapter using [supported agents](supported-agents.md) and
[local/API setup](local-and-api.md). Export `MODEL_TYPE`, `POLICY_MODEL`, `POLICY_BASE_URL` and `POLICY_API_KEY`
for that endpoint. Hermes needs a Chat Completions endpoint: use `MODEL_TYPE=inference_provider`, or
`vllm_model` for an actual vLLM server. Task preparation disables Hermes's `chat_template_kwargs_enabled` by
default for provider compatibility. When the supplied vLLM recipe needs those template settings, enable them under
the generated agent instance in the run config and verify the captured request preserves that recipe.
With `HARNESS=hermes_agent`, the concrete smoke is:

```bash
HARNESS=hermes_agent
gym eval run "$TASK_DIR" --agent "$HARNESS" --sandbox docker \
  --model-type "$MODEL_TYPE" --model "$POLICY_MODEL" \
  --model-url "$POLICY_BASE_URL" --model-api-key '${oc.env:POLICY_API_KEY}' \
  --limit 5 --num-repeats 1 --concurrency 1 \
  +harbor_output_root="$PWD/results/task-smoke" \
  +require_complete=true ++observability_enabled=true \
  +server_spinup_timeout_seconds=1200 \
  ++model_call_capture_dir="$PWD/results/task-smoke/model-calls"
```

`--agent` takes the full harness config name, optionally `/FLAVOR`; ambiguous aliases are refused.
Task targets start their own servers and refuse `--no-serve`. A local filesystem path that could be read as
a Hydra override should start with `./` or be absolute.

After the smoke's artifacts pass the [local result checks](local-results.md), run the intended population in
a new output root. Omit both smoke controls to retain the dataset's task population and repeat defaults:

```bash
gym eval run "$TASK_DIR" --agent "$HARNESS" --sandbox docker \
  --model-type "$MODEL_TYPE" --model "$POLICY_MODEL" \
  --model-url "$POLICY_BASE_URL" --model-api-key '${oc.env:POLICY_API_KEY}' \
  --concurrency 1 \
  +harbor_output_root="$PWD/results/task-full" \
  +require_complete=true ++observability_enabled=true \
  +server_spinup_timeout_seconds=1200 \
  ++model_call_capture_dir="$PWD/results/task-full/model-calls"
```

An explicit `--num-repeats` replaces the dataset's default. Record an intended increase; do not silently
reduce repeats for a full evaluation. `harbor_output_root` moves prepared rows, generated config, image
records and verifier artifacts together. `--output` alone moves the rollout output without moving every
prepared artifact. Keep a fresh root for a different harness or protocol; ordinary reruns can replace
existing outputs. A changed selection or task digest must use a new `+harbor_output_root` before any command,
including a refused resume: preparation can rewrite generated metadata before the resume guard runs.
For an interrupted unchanged run, retain the original working directory and saved absolute paths, then follow
the direct-run resume checks in [local/API setup](local-and-api.md).

## Read and report this run

The default output is `results/harbor/<taskset>/`; the examples use
`results/task-smoke/<taskset>/` and `results/task-full/<taskset>/`. Read `rollouts_run_summary.json` first.
It identifies the run's files, stale files, expected/result/failed/missing counts, skipped tasks, task
rewards and verifier log locations; its artifact paths are relative to the summary directory. Also read
coverage, aggregate metrics and quality evidence using [local result checks](local-results.md).

Use `agent_ref.name` in `rollouts_aggregate_metrics.json` to select the aggregate entry; task-folder result
rows can omit `agent_ref`, and the generated instance can differ from the harness name. Report the dataset identity and digests, Gym revision, harness,
model, repeats, selected task population, aggregate score key and denominator, omissions and grading
evidence. A five-task smoke supports a smoke result. A complete dataset claim needs the intended tasks,
repeats and the observed verification evidence. Keep the task folders unchanged while using saved results.
