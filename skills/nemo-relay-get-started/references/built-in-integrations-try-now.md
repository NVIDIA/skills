<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Built-In Integrations Try-Now Reference

Use this path when the application already uses a maintained NeMo Relay
integration. This is less invasive than manually wrapping language-level call
sites because the framework or harness already owns callbacks, scheduling,
tools, or provider calls.

## Select The Maintained Surface

Choose only the integration already present in the target:

| Existing Surface | Relay Attachment | Current Support Boundary |
|---|---|---|
| LangChain | Maintained middleware and callbacks | Observability, security, and optimization |
| LangGraph | Maintained callbacks around graph execution | Observability, security, and optimization |
| Deep Agents | Maintained callbacks and lifecycle marks | Observability, security, and optimization |
| OpenClaw | OpenClaw-managed Relay plugin and hooks | Observability, partial security, no optimization through public hooks |

If none of these surfaces is present, do not add one just to use Relay. Use the
language try-now path when the application owns its calls directly. Building a
new framework adapter is not a quick start.

## Install And Attach

Inspect the application manifest and existing framework setup first. Use
`nemo-relay-install` for the selected integration package and preserve the
project's package manager.

Then use only the matching maintained guide:

- [LangChain integration](https://docs.nvidia.com/nemo/relay/dev/supported-integrations/langchain)
- [LangGraph integration](https://docs.nvidia.com/nemo/relay/dev/supported-integrations/langgraph)
- [Deep Agents integration](https://docs.nvidia.com/nemo/relay/dev/supported-integrations/deepagents)
- [OpenClaw integration](https://docs.nvidia.com/nemo/relay/dev/supported-integrations/openclaw-plugin)

Before editing, identify the smallest documented attachment point in the
existing agent or workflow. Show the proposed middleware, callback, or plugin
change and obtain confirmation. Do not patch framework internals or replace the
framework's scheduler, retry behavior, provider routing, or object lifecycle.

## Run The Minimal Trial

Use one existing, non-sensitive workflow invocation that exercises the
integration's normal path. Prefer a read-only tool and a small model request.
Do not build a second demonstration application when the target already has a
safe runnable example or test.

Use the integration guide's own verify step. Confirm observable evidence for
the boundaries that integration supports:

- A root agent, graph, or run scope
- A tool lifecycle when the workflow invokes a tool
- An LLM lifecycle when the integration can observe the provider call
- Framework-specific marks such as graph nodes, skills, subagents, or
  human-in-the-loop events when applicable

For OpenClaw, do not claim managed execution rewrites or optimization from
public hook coverage. Report partial support as a capability boundary, not a
failed quick start.

## Define Success And Continue

Treat this path as successful when the original framework workflow still
returns its expected result and the maintained Relay integration emits the
documented lifecycle evidence. A successful framework response without Relay
events verifies the application, not Relay.

After the first proof:

- Preserve the maintained attachment boundary and recommend one goal-aligned
  plugin as the primary next step. If the trial used only temporary inspection,
  configure plugin-managed Observability first; otherwise choose Adaptive,
  NeMo Guardrails, PII Redaction, or Model Pricing based on the user's outcome.
- Use the matching maintained integration guide for broader coverage.
- Use `nemo-relay-instrument-calls` only for application-owned calls that remain
  outside the integration.

For the complete maintained surface, see
[Supported Integrations](https://docs.nvidia.com/nemo/relay/dev/supported-integrations/about).
