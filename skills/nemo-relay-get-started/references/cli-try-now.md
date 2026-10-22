<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# CLI Trial Routing

Read the current [quick-start index](https://docs.nvidia.com/nemo/relay/getting-started/quick-start)
and the selected host's guide:

- [Codex](https://docs.nvidia.com/nemo/relay/getting-started/quick-start/codex)
- [Claude Code](https://docs.nvidia.com/nemo/relay/getting-started/quick-start/claude-code)
- [pi](https://docs.nvidia.com/nemo/relay/getting-started/quick-start/pi)

Use the guide compatible with the installed stable Relay release. Derive host
requirements, authentication, provider protocols, extension setup, configuration,
launch options, and event checks from that guide. Do not keep copied commands,
prompts, or version pins in this reference.

Preserve the selected host. When several fit and the user delegates the choice,
choose one and explain why; otherwise resolve the choice once. Do not configure
or launch multiple hosts.

Distinguish a temporary wrapped process from persistent host setup. Some hosts
need an in-process extension even when the gateway session is temporary; check
the selected guide rather than assuming identical installation behavior.
A local CLI trial does not prove capture of remote work.

For Codex Desktop, a wrapped CLI process does not instrument an already-running
Desktop app. For persistent setup, use `nemo-relay-install` and its continuity
checks. For an existing conversation, read the current supported resume
instructions and fully quit Desktop before resuming that same local thread.
Do not modify private session storage as a workaround.

Follow [Guide-Based First-Value Checks](first-value.md), including the selected
guide's permissions, expected outputs, finalization, and fresh capture evidence.
