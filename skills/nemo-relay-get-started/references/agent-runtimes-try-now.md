<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Agent Runtime Trial Routing

Use the runtime that owns execution and read its current guide:

- [Hermes](https://docs.nvidia.com/nemo/relay/getting-started/quick-start/hermes)
- [OpenClaw](https://docs.nvidia.com/nemo/relay/getting-started/quick-start/openclaw)

Confirm the guide applies to the installed Relay and host combination. Derive
all commands, configuration fields, package requirements, capture controls,
and event checks from that source.

## Hermes

Distinguish native Relay readiness from configured event export. Follow the
native integration guide instead of installing a duplicate runtime, a former
integration plugin, or a CLI wrapper. For an installation-only question, stop
at readiness; for a quick start, continue through the guide's observable proof.

Use the guide's current diagnostics to distinguish relevant Relay failures
from unrelated optional-service warnings. Native availability alone is not
proof that events were captured.

## OpenClaw

Preserve the user's working provider and choose an isolated test agent and
gateway when the guide requires service changes. Do not restart or stop a
shared gateway for a trial.

Read the guide's current provider compatibility notes and configuration sample.
Apply a documented workaround only when it applies to the user's setup; do not
retain an older workaround after the guide replaces it. Package installation,
plugin registration, and runtime activation are distinct checks.

Follow the guide's output locations and gateway finalization before verification.
Inspect current integration capability limits before suggesting a next plugin.

For both runtimes, use [Guide-Based First-Value Checks](first-value.md).
