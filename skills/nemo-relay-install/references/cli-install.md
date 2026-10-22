<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# CLI Installation

Use this reference after establishing that the user needs the executable or a
coding-agent host integration.

## Read Current Instructions

- [Installation](https://docs.nvidia.com/nemo/relay/getting-started/installation)
- [Support matrix](https://docs.nvidia.com/nemo/relay/reference/support-matrix)
- [Latest stable release](https://github.com/NVIDIA/NeMo-Relay/releases/latest)
- [CLI usage](https://docs.nvidia.com/nemo/relay/latest/nemo-relay-cli/basic-usage)
- [Persistent host installation](https://docs.nvidia.com/nemo/relay/latest/nemo-relay-cli/plugin-installation)

Resolve the current stable release and supported platform before deriving the
command. Use its documented installer or package route and verify integrity
before execution. Build from source only when selected by the user or needed
for a confirmed platform gap; explain that choice.

For coding agents, read the chosen host's current integration instructions for
host prerequisites, authentication, supported protocols, and whether an
in-process extension is required. Do not promise that every host supports a
fully temporary wrapper. Distinguish persistent host changes from a temporary
gateway session.

Verify executable and host readiness using the current documented checks.
Do not change provider configuration, start a gateway, or run a model prompt
as basic executable verification. Hand the established target to get-started.

## Persistent Codex Desktop Continuity

Read the [Codex integration guide](https://docs.nvidia.com/nemo/relay/latest/nemo-relay-cli/codex)
and the [provider-filtering issue](https://github.com/openai/codex/issues/24648).
Check whether the issue still applies to the user's setup rather than asserting
that an infrequently updated skill proves a current bug.

When the selected release and setup change provider identity, explain that
existing Desktop conversations can appear hidden without deletion. Scope that
warning to affected setups, such as a legacy `nemo-relay-openai` alias; check
whether the selected installer preserves the built-in `openai` identity.
Recommend a temporary trial first when it satisfies the user's goal.

Before an explicitly selected persistent installation:

1. Preview the supported installer change and any relevant continuity risk.
2. Propose a recovery note in the workspace root using
   `assets/codex-desktop-recovery.md`; confirm its recovery commands against
   the current supported documentation.
3. Obtain explicit confirmation covering the note and global configuration.
4. Render `NEMO_RELAY_CODEX_DESKTOP_RECOVERY.md`, record the resolved Relay
   release, planned provider identity, change, and available thread ID, and
   verify that the file is readable.
5. Run the supported installer and its documented diagnostic check.
6. Give the user the recovery-file path before any requested Desktop restart.

For recovery or continuation, use supported uninstall and temporary-resume
instructions from the current guide. Fully quit Desktop before resuming the
same local conversation. Avoid selecting the most recent conversation
implicitly when switching providers.

Never inspect, copy, delete, edit, or rewrite Codex session files, private
application configuration, or SQLite state as a workaround. Supported Relay
commands may manage the provider and hooks they generate. Do not stage the
recovery note unless requested.
