<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Maintained Integration Installation

Use this path only when the target already uses the named framework or agent
harness. Install the maintained integration package, verify it through that
surface's package or plugin manager, and defer wiring to its integration guide.

## OpenClaw

```bash
openclaw plugins install npm:nemo-relay-openclaw
openclaw gateway restart
```

Verify that OpenClaw reports the plugin installed. Do not configure its runtime
hooks from the install skill.

## Hermes

```bash
pip install nemo-relay
hermes plugins enable observability/nemo_relay
```

Verify that Hermes reports the plugin enabled. Defer observability and gateway
configuration to the maintained Hermes guidance.

## LangChain, LangGraph, Or Deep Agents

Install only the extras needed by the target. Preserve the project's existing
Python package manager. For example, when all three integrations are present in
a `uv`-managed project:

```bash
uv add "nemo-relay[langchain,langgraph,deepagents]"
```

Use the equivalent `poetry add`, `pdm add`, or environment-scoped
`python -m pip install` command when that manager already owns the project. For
a narrower application, remove extras that are not used. Verify the package
manager result, then defer callbacks and runtime attachment to the matching
maintained integration guide.

For the complete maintained surface, see
[Supported Integrations](https://docs.nvidia.com/nemo/relay/dev/supported-integrations/about).
