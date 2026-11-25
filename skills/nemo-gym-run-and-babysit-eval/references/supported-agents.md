<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Evaluate an agent Gym already supports

Select an existing harness and a compatible task pairing. Keep the benchmark's data, prompts, verifier,
sampling and repeat protocol fixed. Record the harness and its settings as part of the evaluated system.
This procedure uses existing adapters; adding a runtime is a separate integration task.

## Discover the installed agent and pairing

After the pinned installation in [local and API-endpoint runs](local-and-api.md), inspect the built-ins
from the user's project directory:

```bash
gym list agents
gym list agents hermes_agent --json
gym list benchmarks
gym list benchmarks "$BENCHMARK" --json
```

The agent inspection reports its installed path and config variants. Read that adapter's installed README
and selected config for runtime requirements, tools, supported model protocols and settings. The same
files are available in the
[pinned adapter sources](https://github.com/NVIDIA-NeMo/Gym/tree/a8eeecedb433b6c483077160a2cf9a06a09fab1b/responses_api_agents).
Discovery alone does not establish that every agent pairs with every benchmark.

The selectors have different roles:

| Run | Selection |
| --- | --- |
| Built-in benchmark with its supplied agent | `gym eval run --benchmark NAME`; retain the benchmark's configured pairing. |
| Compatible alternate benchmark harness | `--agent-type NAME[/FLAVOR]` selects a shipped composition/profile. Inspect the resulting pairing before running. |
| Existing task folder | `gym eval run "$TASK_DIR" --agent NAME[/FLAVOR] --sandbox PROVIDER`; Gym writes the agent/environment binding. |
| Already running Gym servers | `--no-serve --agent INSTANCE` selects an agent instance already defined by the same configuration used at startup. |

For task folders, use full names such as `hermes_agent`. The short name `hermes` can match both
`hermes_agent` and `hermes_sandboxed_agent` and is refused. A flavor selects a config file under the
harness's `configs/`, such as `mini_swe_agent/mini_swe_agent_sandboxed`.

## Existing task-folder harness examples

These are existing session adapters at the pinned revision. The protocol column describes the harness's
connection to Gym's Model Server. The selected model adapter and upstream endpoint must support that
connection; use the compatibility rules below with [local/API setup](local-and-api.md).

| Harness selector | Harness to Gym model protocol | Task requirements |
| --- | --- | --- |
| `hermes_agent` | Chat Completions | Borrows the task sandbox; Gym enables its terminal toolset for task folders. |
| `pi_agent` | Chat Completions | Runtime and task tools must work in the selected task image. |
| `codex_agent` | Streaming Responses | Use a compatible streaming/tool-calling model path and a supported Linux task image. |
| `opencode_agent` | Chat Completions | Preserve its pinned runtime and any required runtime files, tools and network access. |
| `mini_swe_agent/mini_swe_agent_sandboxed` | Responses | Its session entrypoint does not start Ray. Borrowed `SandboxAccess.user` must be null; check the task image's runtime requirements. |

| Harness requests and upstream endpoint | Model path |
| --- | --- |
| Chat harness and Chat Completions endpoint | Use `inference_provider`, or `vllm_model` for an actual vLLM backend. |
| Responses harness and Responses endpoint | Use `openai_model`. |
| Responses harness and Chat Completions endpoint | Use the `inference_provider` bridge; check the harness's streaming and tool-call requirements in the smoke. |
| Chat harness and a Responses-only endpoint | This combination needs a compatible Chat endpoint or a supported Responses harness. `openai_model` does not translate its Chat route into an upstream Responses call. |

Use the task-folder commands in [task folders](task-folders.md) with the selected `HARNESS`; that reference also
covers preserving Hermes's template settings for a vLLM deployment. Check the
dataset's `[gym].allowed_agents` against the implementation's full folder name. Task MCP servers, skills,
task users, GPU requirements and Compose services need matching harness/provider capabilities. Keep
pairing checks enabled; changing a task or bypassing a pairing refusal changes what was validated.

The provider must run the task's image. Docker needs its daemon; remote providers need their service,
credentials and any image registry access. Gym starts adapters in component environments and stages or
installs the harness runtime as required by that adapter. Some runtimes need downloads, an appropriate
Linux architecture or system tools. A working harness on the host does not establish that its task-image
runtime or its tools work. Inspect one smoke trajectory and its verifier result before scaling up.

## Built-in benchmark composition

Prefer a benchmark's existing harness-specific variant when one exists. A generic session config with
`resources_server: null`, such as default `hermes_agent`, does not automatically replace a JSONL
benchmark's agent. A swappable profile or an explicit environment-server binding is required; Gym rejects
a selected agent that no workload task would use. For a Harbor-backed benchmark, `--agent-type` selects
the alternate harness while retaining the benchmark's non-harness settings. Its dataset still controls
allowed agents.

For a compatible shipped `AGENT_CONFIG`, validate composition before the model smoke:

```bash
gym env validate --benchmark "$BENCHMARK" --agent-type "$AGENT_CONFIG" \
  --model-type "$MODEL_TYPE" --model "$POLICY_MODEL" \
  --model-url "$POLICY_BASE_URL" --model-api-key '${oc.env:POLICY_API_KEY}' \
  --server-types responses_api_agents,responses_api_models,resources_servers,environment_servers \
  --allow-job-time-values
```

Read invalid, unchecked and uncomposed findings. Validation composes config and checks available schemas;
it does not run a model, verify tool connectivity or prove that the requested pairing is qualified.
Continue with the preparation and direct-run commands in [local/API setup](local-and-api.md), adding the
same `--agent-type "$AGENT_CONFIG"` to the evaluation run. Do not use a runtime-only config as a shortcut
around a missing environment binding or required tools.

## Model and result wiring

The built-in model configs create a Gym model-server instance named `policy_model`. `--model-type` chooses
its adapter; `--model`, `--model-url` and `--model-api-key` set `policy_model_name`, `policy_base_url` and
`policy_api_key`. Session harnesses normally refer to it with:

```yaml
model_server:
  type: responses_api_models
  name: policy_model
model: ${policy_model_name}
```

Preserve a supplied experiment's explicit wiring. Setting only an upstream model URL does not bind an
otherwise unreferenced agent to tasks. For an already started experiment and its prepared JSONL, collection
uses the same complete config and an existing instance:

```bash
gym eval run --no-serve --config experiment.yaml --agent "$AGENT_INSTANCE" \
  --input "$TASKS_JSONL" --output "$ROLLOUTS" +require_complete=true
```

Here `AGENT_INSTANCE` is the configured instance, not an instruction to install or compose a harness.
`--agent-type` cannot be combined with `--no-serve`. Task-folder targets always start their own servers.
The existing servers must already use the intended config and capture settings. Collector-only overrides
do not add observability to running servers; report gaps in the saved evidence.

Read `agent_ref.name` in the saved aggregate. Task-folder rows can omit `agent_ref`; the generated run config
and summary record their taskset/environment routing and harness implementation. A taskset-specific agent
instance or a composed benchmark instance can differ from the harness folder name. Report the selected instance,
harness implementation/version, model and endpoint protocol,
tool settings, tasks/repeats and the aggregate metric. Use [local result checks](local-results.md) for a
direct run, and the original Slurm workflow for a submitted chain.
