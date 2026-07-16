<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# CLI Installation

Use this path for the `nemo-relay` executable, a temporary coding-agent run,
local gateway use, or explicit persistent host-plugin installation.

## Contents

- [Check Prerequisites](#check-prerequisites)
- [Install](#install)
- [Verify](#verify)
- [Persistent Codex Desktop Safety Gate](#persistent-codex-desktop-safety-gate)

## Check Prerequisites

Confirm these prerequisites before selecting an installation command:

- Confirm the operating system and architecture have a published CLI asset.
- Use Cargo when the user prefers a source build or needs an unsupported
  platform.
- For a transparent run, confirm the selected `codex`, `claude`, or `hermes`
  command is already on `PATH`.

## Install

Use the installer on a supported platform when a prebuilt release asset is
available. Resolve a specific stable version from the NeMo Relay GitHub
Releases page. Download that version's installer, review it, and run it as a
separate step. Do not execute a response directly from the repository's `main`
branch.

For a supported Unix-like shell:

```bash
RELAY_VERSION="<release-version>"
curl --fail --location --proto '=https' --tlsv1.2 \
  "https://raw.githubusercontent.com/NVIDIA/NeMo-Relay/${RELAY_VERSION}/install.sh" \
  --output nemo-relay-install.sh
less nemo-relay-install.sh
NEMO_RELAY_VERSION="${RELAY_VERSION}" sh nemo-relay-install.sh
```

For Windows PowerShell:

```powershell
$RelayVersion = "<release-version>"
$Installer = "nemo-relay-install.ps1"
Invoke-WebRequest `
  -Uri "https://raw.githubusercontent.com/NVIDIA/NeMo-Relay/$RelayVersion/install.ps1" `
  -OutFile $Installer
Get-Content -LiteralPath $Installer
$env:NEMO_RELAY_VERSION = $RelayVersion
& ".\$Installer"
```

The reviewed installer downloads the selected release binary and its published
SHA-256 checksum, verifies them before replacing an existing binary, and does
not invoke `sudo`. Remove the downloaded installer only after verification if
the user does not want to retain it for audit.

Use Cargo on an unsupported platform or when the user prefers to build from
source:

```bash
cargo install nemo-relay-cli
```

## Verify

Run:

```bash
nemo-relay --version
```

For transparent-run readiness, optionally preview the generated temporary hook
configuration, gateway environment, gateway URL, and final command:

```bash
nemo-relay run --agent <agent> --dry-run --print
```

After installation, hand a generic trial to `nemo-relay-get-started`. Its
default path launches the selected coding agent with `nemo-relay codex`,
`nemo-relay claude`, `nemo-relay hermes`, or `nemo-relay run -- <command>`.
The wrapper is temporary for that process.

Use persistent host-plugin installation only when the user explicitly wants
Claude Code or Codex to load Relay through the host plugin system. Validate that
path with `nemo-relay doctor --plugin <host>`.

## Persistent Codex Desktop Safety Gate

Apply this gate when the current client is Codex Desktop or the user wants
Relay to remain active in Codex Desktop after restart.

Persistent installation configures the `nemo-relay-openai` provider in global
Codex configuration. Codex Desktop currently filters visible threads by the
active provider, so existing `openai` threads, including the setup thread, may
appear missing after restart. This is a visibility bug, not session deletion.

Before changing persistent configuration:

> **WARNING:** Persistent Codex installation changes global application
> configuration and can make existing Desktop threads appear missing after
> restart. Complete the recovery-file steps and obtain explicit confirmation
> before making this change.

1. Explain the difference between temporary `nemo-relay codex` and persistent
   `nemo-relay install codex`. Recommend temporary mode for evaluation.
2. Preview the persistent change without writing it:

   ```bash
   nemo-relay install codex --dry-run
   ```

3. Propose creating `NEMO_RELAY_CODEX_DESKTOP_RECOVERY.md` in the current
   workspace root from `assets/codex-desktop-recovery.md`.
4. Ask for confirmation covering both the recovery file and persistent Codex
   configuration.
5. Render the file with the current date, workspace path, planned command, and
   thread ID when available. Verify it is readable before installing.
6. Run `nemo-relay install codex`, then verify with
   `nemo-relay doctor --plugin codex`.
7. Before asking the user to restart Desktop, give them the exact recovery-file
   path. Do not stage or commit the generated recovery file unless requested.

The recovery file must include both supported exits:

- Restore normal Desktop visibility by fully quitting Desktop, running
  `nemo-relay uninstall codex`, and restarting Desktop.
- Continue an existing conversation through temporary Relay wiring by fully
  quitting Desktop and running `nemo-relay codex -- resume --all` or
  `nemo-relay codex -- resume <thread-id>`.

Avoid `resume --last` when crossing providers. Never directly inspect, copy,
delete, edit, or rewrite Codex session storage, private application
configuration, or SQLite state as a migration workaround. Supported
`nemo-relay` commands may manage Relay-generated provider and hook
configuration.

Use these references for the supported installation and host-integration paths:

- [Installation](https://docs.nvidia.com/nemo/relay/getting-started/installation)
- [Transparent run](https://docs.nvidia.com/nemo/relay/dev/nemo-relay-cli/basic-usage#transparent-run)
- [Codex integration](https://docs.nvidia.com/nemo/relay/dev/nemo-relay-cli/codex)
- [Persistent plugin installation](https://docs.nvidia.com/nemo/relay/dev/nemo-relay-cli/plugin-installation)
- [Codex Desktop provider-filter bug](https://github.com/openai/codex/issues/24648)
