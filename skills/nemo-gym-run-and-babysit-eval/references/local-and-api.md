<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Evaluate locally against a model endpoint

Use this path to run Gym locally against an existing model endpoint, including a self-hosted model. Gym itself needs
no GPU for this path. For tasks in existing folders, use [task folders](task-folders.md); for selecting another Gym
harness, use [supported agents](supported-agents.md). Submitted Slurm runs follow the [cluster procedure](../SKILL.md#0-no-config-yet-compose-it).

## Install outside a checkout

Use Python 3.13.14 or later, Git, and a writable working directory outside a Gym checkout. Install this exact public
commit as a preview; do not substitute a moving branch or invent a release version. Use a supported interpreter;
the example uses `python3.13`, whose reported patch version must be at least 3.13.14. This revision includes
the Docker upload fix needed for pip-installed harness runners; older previews can copy dangling host symlinks:

```bash
mkdir gym-evaluation
cd gym-evaluation
python3.13 --version
python3.13 -m venv .venv
. .venv/bin/activate
python -m pip install 'uv>=0.13.4'
python -m pip install \
  'nemo-gym @ git+https://github.com/NVIDIA-NeMo/Gym.git@a8eeecedb433b6c483077160a2cf9a06a09fab1b'
gym --version
gym list benchmarks
gym list benchmarks ifeval
```

Built-in components and their configs are installed with Gym. Keep the preview commit with the result. Read the
selected benchmark's installed config, manifest and preparation instructions; its dataset, judge, prompts, sampling,
output budget, time limits and repeat requirement define the protocol. Configure any additional judge, search or
sandbox dependencies it declares before running it.

## Select the model client

Collect the endpoint's base URL, served model name, API protocol and authentication. Export `POLICY_BASE_URL`,
`POLICY_MODEL` and `POLICY_API_KEY` into the process environment using the user's values; do not put a real key in these instructions.
Choose `MODEL_TYPE` from what the endpoint implements. Pass the API-key interpolation literally as shown below,
so the shell does not expand the credential into the command. For an unauthenticated local endpoint, export a
non-secret placeholder such as `POLICY_API_KEY=EMPTY`; the environment variable must still be defined:

| Upstream API | `MODEL_TYPE` |
| --- | --- |
| Responses API | `openai_model` |
| OpenAI-style Chat Completions | `inference_provider` |
| Existing vLLM server, including its vLLM APIs | `vllm_model` |

`openai_model` forwards Responses requests to `/responses`. `inference_provider` translates them to Chat
Completions. This is the simple-agent common path; a harness that sends Chat Completions still needs that upstream
API when using `openai_model`, so check [agent compatibility](supported-agents.md) before selecting a Responses-only
endpoint. For `vllm_model`, match `uses_reasoning_parser` to the served parser; its shipped default is true, so
an endpoint without a reasoning parser needs `++policy_model.responses_api_models.vllm_model.uses_reasoning_parser=false`
on validation and run commands. Read the model's template and parser settings as described in
[model configuration](add-model-eval-config.md). `inference_provider` also has `uses_reasoning_parser`, default false;
set `++policy_model.responses_api_models.inference_provider.uses_reasoning_parser` to match the endpoint's actual
reasoning format. In the smoke, inspect the captured answer and scoring input: reasoning must be represented as
reasoning rather than accidentally graded as final-answer text. Do not edit collected answers to repair parsing.
An existing endpoint does not need another inference deployment.

## Start a local checkpoint when needed

If the user supplies a checkpoint instead of an endpoint, run vLLM in a separate inference environment on their
GPU host. Use their chosen vLLM version and complete serving recipe, retaining parallelism, context length, dtype,
quantization, template and parser flags. Ask for missing deployment values; do not select a GPU layout for them.
The Bash command shape is:

```bash
# VLLM_SERVE_ARGS contains the agreed model-specific serving flags as a Bash array.
vllm serve "$CHECKPOINT" --served-model-name "$POLICY_MODEL" \
  --host "$VLLM_HOST" --port "$VLLM_PORT" "${VLLM_SERVE_ARGS[@]}"
```

Populate the agreed arguments before starting the process. Point `POLICY_BASE_URL` at that reachable `/v1` endpoint
and set `MODEL_TYPE=vllm_model`; then use the same Gym validation, smoke and full-run steps below. Gym runs in its
own environment and does not allocate cluster resources for this path. Preserve the supplied serving command when
the endpoint is already running.

## Prepare, validate and smoke

IFEval is a concrete built-in example with one canonical repeat and no separate judge endpoint. Its preparer writes
beside its installed module. This preview's preparation CLI otherwise compares that output with a
working-directory-relative path and can report a mismatch after writing the data. Run preparation in a subshell at
the installed asset root; subsequent validation and collection remain in the user's working directory. This keeps
the benchmark's data declaration and protocol unchanged:

```bash
GYM_ASSETS=$(python -c 'import nemo_gym; print(nemo_gym.PARENT_DIR)')
(cd "$GYM_ASSETS" && gym eval prepare --benchmark ifeval)
gym env validate --server-types responses_api_models,responses_api_agents \
  --benchmark ifeval --model-type "$MODEL_TYPE" \
  --model "$POLICY_MODEL" --model-url "$POLICY_BASE_URL" --model-api-key '${oc.env:POLICY_API_KEY}'

gym eval run --benchmark ifeval --split benchmark \
  --model-type "$MODEL_TYPE" --model "$POLICY_MODEL" \
  --model-url "$POLICY_BASE_URL" --model-api-key '${oc.env:POLICY_API_KEY}' \
  --limit 5 --num-repeats 1 --concurrency 8 \
  --output results/ifeval/smoke/artifacts/rollouts.jsonl \
  +require_complete=true ++observability_enabled=true \
  ++model_call_capture_dir="$PWD/results/ifeval/smoke/model-calls"
```

The environment above is user-owned and writable; do not overwrite a shared installation's data to work around a
path error. For another built-in benchmark, use its actual split and preparation instructions. If its
preparation fails, investigate that failure; do not ignore it or claim the benchmark was prepared. `gym eval prepare`
runs preprocessing; `gym eval run` collates prepared data, downloads declared sources where supported and manages
Gym's servers. The first server start installs isolated component dependencies and may take several minutes.
Example-only JSONL instead uses the documented `env start` plus `eval run --no-serve` path.

Inspect the smoke's representative answers, tool calls, coverage and health using [result checks](local-results.md).
A five-task result is a smoke, regardless of its exit code or score.

## Run the full evaluation

After the smoke passes, run every task with the benchmark's canonical repeats in a new directory. This complete
IFEval command removes both smoke overrides:

```bash
gym eval run --benchmark ifeval --split benchmark \
  --model-type "$MODEL_TYPE" --model "$POLICY_MODEL" \
  --model-url "$POLICY_BASE_URL" --model-api-key '${oc.env:POLICY_API_KEY}' \
  --concurrency 8 --output results/ifeval/full/artifacts/rollouts.jsonl \
  +require_complete=true ++observability_enabled=true \
  ++model_call_capture_dir="$PWD/results/ifeval/full/model-calls"
```

Use the user-approved provider concurrency; do not raise benchmark timeouts or output budgets to improve a score.
Keep each run's output and absolute capture directory separate. Collection completion, measured coverage, evaluation
health and the saved headline metric must agree before reporting the full result; follow [result checks](local-results.md).

For an interrupted direct run, retain its artifacts and repeat the same command and configuration with `--resume`
from the original working directory, or resolve every input, output and capture path to its original absolute path.
First confirm that the saved rollouts, materialized inputs and manifest exist there. This preview can start fresh
when `--resume` finds no saved artifacts. Record the saved batch-status completion count before dispatch, then
compare the resumed run's retained-completion count with that record; stop
if it starts a fresh collection instead of continuing the remaining work. Completed selected answers must remain saved.
A changed model, dataset, prompt or judge belongs in a separate run. Direct runs have no Slurm job ID: use their
`eval run --resume` route, not `eval resubmit --recover`.
