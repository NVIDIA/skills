<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# CLI Try-Now Reference

Use this reference only for the default coding-agent trial. Keep the first run
local, user-scoped, confirmation-gated, and limited to the built-in
Observability plugin.

## Contents

- [Preflight](#preflight)
- [Protect Codex Desktop History Visibility](#protect-codex-desktop-history-visibility)
- [Inspect Configuration Before Editing](#inspect-configuration-before-editing)
- [Configure The Agent And Observability](#configure-the-agent-and-observability)
- [Validate And Preview](#validate-and-preview)
- [Run A Safe Trial](#run-a-safe-trial)
- [Verify Both Outputs](#verify-both-outputs)
- [Choose The Next Plugin](#choose-the-next-plugin)
- [Troubleshoot The Smallest Failed Boundary](#troubleshoot-the-smallest-failed-boundary)

## Preflight

Verify Relay and discover available coding agents:

```bash
nemo-relay --version
nemo-relay agents --json
```

If `nemo-relay agents --json` is unavailable, check the selected command
directly:

```bash
command -v codex && codex --version
command -v claude && claude --version
```

Use Codex CLI 0.129.0 or newer. Confirm that the selected agent is already
authenticated before launching Relay. Never print tokens, API keys, or stored
authentication files.

## Protect Codex Desktop History Visibility

Keep this quick start temporary. `nemo-relay codex` injects Relay configuration
only into the wrapped CLI process and does not instrument the already-running
Codex Desktop app or rewrite its global configuration.

Do not run `nemo-relay install codex` from this try-now path. Persistent setup
changes the active provider used by Codex Desktop. Because of the current
[provider-filter bug](https://github.com/openai/codex/issues/24648), restarting
Desktop can make the current setup thread and older threads appear missing even
though they remain stored locally.

If the user wants to continue an existing Desktop conversation through the
temporary Relay wrapper, ask them to fully quit Desktop before launching:

```bash
nemo-relay codex -- resume --all
```

Use `nemo-relay codex -- resume <thread-id>` when the ID is known. Avoid
`resume --last` when crossing providers.

If the user explicitly requests persistent Codex Desktop integration, stop this
quick start and hand off to `nemo-relay-install`. That skill must warn the user
and create `NEMO_RELAY_CODEX_DESKTOP_RECOVERY.md` before changing global Codex
configuration.

## Inspect Configuration Before Editing

Resolve the supported user configuration directory from
`$XDG_CONFIG_HOME/nemo-relay`, falling back to `$HOME/.config/nemo-relay`.
Inspect these files when they exist:

```text
${XDG_CONFIG_HOME:-$HOME/.config}/nemo-relay/config.toml
${XDG_CONFIG_HOME:-$HOME/.config}/nemo-relay/plugins.toml
```

Repository-local `.nemo-relay/config.toml` and `.nemo-relay/plugins.toml`
files are unsupported as active Relay configuration. If they exist, identify
them for the user and explain that this quick start will not create, edit,
merge, or trust them. Local output directories such as `.nemo-relay/atof` and
`.nemo-relay/atif` are artifacts, not configuration layers, and may remain
valid when explicitly configured as output locations.

Account for higher-precedence system policy, show the proposed user-file
change, and obtain confirmation. Merge with an existing plugin document; do
not replace unrelated components.

## Configure The Agent And Observability

When an interactive TTY is available, use the built-in setup path:

```bash
nemo-relay config codex
nemo-relay config claude
```

Run only the command for the selected agent. Setup writes the XDG user
configuration. Continue to plugin configuration, enable the built-in
`observability` component, and enable both ATOF and ATIF local file output.

When an interactive plugin editor is unavailable, add or merge the following
component in the XDG user `plugins.toml` after confirmation. Resolve the user
configuration directory and replace `<relay-user-config-dir>` below with its
absolute path.

```toml
version = 1

[[components]]
kind = "observability"
enabled = true

[components.config]
version = 3

[components.config.atof]
enabled = true

[[components.config.atof.sinks]]
type = "file"
output_directory = "<relay-user-config-dir>/atof"
filename = "events.jsonl"
mode = "append"

[components.config.atif]
enabled = true
output_directory = "<relay-user-config-dir>/atif"
filename_template = "{session_id}.atif.json"
```

If a base config must be written non-interactively for Codex or Claude Code,
use only the selected agent block and preserve existing sections:

```toml
[agents.codex]
command = "codex"
```

or:

```toml
[agents.claude]
command = "claude"
```

After the confirmed plugin change, create the configured user output directories
so doctor can verify that they are writable:

```bash
relay_user_dir="${XDG_CONFIG_HOME:-$HOME/.config}/nemo-relay"
mkdir -p "$relay_user_dir/atof" "$relay_user_dir/atif"
```

## Validate And Preview

Run doctor for the selected agent:

```bash
nemo-relay doctor codex --json
nemo-relay doctor claude --json
```

Run only one command. Summarize failed checks and the remediation they report.
Then inspect the generated wrapper plan without launching the agent:

```bash
nemo-relay run --agent codex --dry-run --print
nemo-relay run --agent claude --dry-run --print
```

Confirm that the plan uses a loopback gateway, the intended agent command, and
supported user or explicit plugin configuration. The plan must not depend on
repository-local `.nemo-relay/config.toml` or `.nemo-relay/plugins.toml`. Show
this summary and obtain user confirmation before the live run.

## Run A Safe Trial

Use this smoke prompt:

> Use a shell tool to print exactly `relay-smoke-test`, then reply that the tool
> call completed. Do not inspect files, environment variables, processes,
> credentials, network resources, or system configuration.

Launch the selected transparent wrapper:

```bash
nemo-relay codex -- exec "Use a shell tool to print exactly relay-smoke-test, then reply that the tool call completed. Do not inspect files, environment variables, processes, credentials, network resources, or system configuration."
```

```bash
nemo-relay claude -- "Use a shell tool to print exactly relay-smoke-test, then reply that the tool call completed. Do not inspect files, environment variables, processes, credentials, network resources, or system configuration."
```

## Verify Both Outputs

Check that ATOF output exists and is non-empty:

```bash
relay_user_dir="${XDG_CONFIG_HOME:-$HOME/.config}/nemo-relay"
test -s "$relay_user_dir/atof/events.jsonl"
wc -l "$relay_user_dir/atof/events.jsonl"
```

Find non-empty ATIF trajectories:

```bash
relay_user_dir="${XDG_CONFIG_HOME:-$HOME/.config}/nemo-relay"
find "$relay_user_dir/atif" -type f -name '*.json' -size +0c -print
```

Parse only the minimum JSON needed to report:

- The root agent or turn scope
- One tool start/end lifecycle
- One LLM start/end lifecycle when gateway routing is active
- The parent-child relationship between the root and calls

Do not paste complete event records or trajectories. Codex writes an ATIF
snapshot after each completed turn. Claude Code normally writes the trajectory
when the session ends.

## Choose The Next Plugin

After both outputs verify the first Relay boundary, explain the progression:
the coding-agent session is already instrumented, and later behavior can change
through plugin configuration without reinstrumenting that boundary.

Ask which outcome matters next and recommend one built-in plugin: Adaptive for
optimization, NeMo Guardrails for policy, PII Redaction for sensitive payloads,
or Model Pricing for cost estimates. Use the plugin overview to show the
smallest next configuration. Do not enable multiple plugins or extend
instrumentation unless the user requests it or the current boundary is
insufficient.

## Troubleshoot The Smallest Failed Boundary

- **No ATOF or ATIF files**: run `nemo-relay doctor <agent> --json`; check
  supported plugin discovery, component activation, config precedence, and
  output-directory permissions. Do not treat repository-local
  `.nemo-relay/config.toml` or `.nemo-relay/plugins.toml` as active
  configuration.
- **ATOF exists but ATIF does not**: finish the turn and close or finalize the
  agent session before changing configuration.
- **Agent and tool events exist but LLM events do not**: confirm the launched
  agent's provider traffic is using the temporary gateway.
- **No hook events**: confirm the agent loaded or approved the generated hooks.
  Codex may require manual hook review.
- **The wrapper does not launch**: inspect `--dry-run --print`, the selected
  agent command, authentication readiness, and doctor output.

Do not switch to persistent host plugins, external OTLP systems, or broad
reinstallation while validating this local trial.

## Source Documentation

Use these sources when the trial needs more detail:

- [CLI overview](https://docs.nvidia.com/nemo/relay/dev/nemo-relay-cli/about)
- [CLI basic usage](https://docs.nvidia.com/nemo/relay/dev/nemo-relay-cli/basic-usage)
- [Quick Start](https://docs.nvidia.com/nemo/relay/dev/getting-started/quick-start)
- [Observability configuration](https://docs.nvidia.com/nemo/relay/dev/configure-plugins/observability/configuration)
- [Codex integration](https://docs.nvidia.com/nemo/relay/dev/nemo-relay-cli/codex)
- [Codex Desktop provider-filter bug](https://github.com/openai/codex/issues/24648)
- [Plugin selection](https://docs.nvidia.com/nemo/relay/dev/configure-plugins/about)
