---
name: nemo-relay-install
description: Use this skill when choosing or running NeMo Relay installation for the CLI, Python, Node.js, Rust, OpenClaw, Hermes, or maintained framework integrations before runtime configuration or quick-start setup.
license: Apache-2.0
metadata:
  author: NVIDIA Corporation and Affiliates
---

# Install NeMo Relay

Choose the package or executable from the user's desired outcome. Stop after
installation and basic availability checks. Do not configure runtime behavior,
write `plugins.toml`, create scopes, register middleware, or build a first app
example from this skill.
Keep installation separate from first-use instrumentation.

## Choose The Install Path

If the user asks to install NeMo Relay but does not identify a target or
desired outcome, ask one short clarifying question before giving commands:

> Which install path do you want: CLI for coding-agent/local gateway use,
> language package for a Python/Node.js/Rust app, or framework integration for
> LangChain, LangGraph, Deep Agents, OpenClaw, or Hermes?

Do not ask when the user already names a CLI, language, framework, harness,
source checkout, or target project file such as `pyproject.toml`, `package.json`,
or `Cargo.toml`.

Use this order from least to most application-specific:

1. **CLI** for a generic "try Relay" request, a temporary coding-agent run, the
   local gateway, or explicit persistent host-plugin setup. Read
   [CLI Installation](references/cli-install.md).
2. **Maintained integration** when the user already uses OpenClaw, Hermes,
   LangChain, LangGraph, or Deep Agents. Read
   [Maintained Integration Installation](references/maintained-integrations.md).
3. **Language package** when a Python, Node.js, or Rust application directly
   owns its tool or model call sites. Read
   [Language Package Installation](references/language-packages.md).
4. **Source checkout** only for contributors or unpublished changes. Follow the
   repository development guide instead of published package commands.

For "try Relay" requests, default to the CLI and temporary transparent run.
Do not make persistent host-plugin installation the default.

## Protect Codex Desktop Continuity

Before persistent Codex installation, determine whether the user is operating
from Codex Desktop. Persistent `nemo-relay install codex` changes the active
Codex provider and can make the current or older Desktop threads appear missing
after restart because of an upstream provider-filtering bug. The threads are
not deleted.

For Codex Desktop users:

1. Recommend temporary transparent run first.
2. If the user still wants persistent installation, read the Codex Desktop
   section in [CLI Installation](references/cli-install.md).
3. Preview the install and proposed recovery-note location.
4. Obtain confirmation before writing either the recovery note or global Codex
   configuration.
5. Render `assets/codex-desktop-recovery.md` as
   `NEMO_RELAY_CODEX_DESKTOP_RECOVERY.md` in the user's workspace root before
   running the persistent installer.
6. Do not restart Codex Desktop until the user has the recovery-file path.

Do not directly inspect, copy, delete, edit, or rewrite Codex session files,
private application configuration, or SQLite state to work around the
visibility bug. Supported `nemo-relay` install, uninstall, and doctor commands
may manage the Relay-generated provider and hook configuration.

## Install And Verify

1. Inspect the target manifest, environment, operating system, architecture,
   and existing installation before changing anything.
2. Load only the reference for the selected path.
3. Preserve the project's existing package manager and virtual environment.
4. Use the latest compatible release unless the user or project requires an
   exact version.
5. Show the exact install command before running a remote installer or changing
   a project manifest.
6. Run only the selected path's basic availability check.
7. Report what was installed, where it was installed, and the verification
   result. Then stop.

The primary documented language paths are Rust, Python, and Node.js. Treat Go
and raw FFI as source-first advanced surfaces, not normal package installs.
Do not treat a first scope, subscriber, gateway, plugin config, or LLM call as
installation verification.

## Use Doctor For CLI Readiness And Configuration Issues

Know about `nemo-relay doctor`, but use it in the right scope:

- Run `nemo-relay doctor` when the user installed the CLI and reports config,
  gateway, agent-readiness, plugin, exporter, or model-pricing issues.
- Use `nemo-relay doctor --json` when structured output will help an agent
  inspect checks programmatically.
- Use `nemo-relay doctor --plugin claude-code`, `nemo-relay doctor --plugin
  codex`, or `nemo-relay doctor --plugin all` only for persistent host-plugin
  installations.
- Do not require plugin doctor for transparent runs. Transparent-run setup does
  not require persistent host-plugin state.
- Do not use doctor as proof that a Python, Node.js, or Rust package dependency
  was installed. Verify those with the language package manager/import checks.

When doctor reports failures, summarize the failed checks and the specific
remediation it suggests. Do not loop back to reinstalling every package unless
the failed check points to a broken or missing install.

## Hand Off After Install

Choose the next workflow from the user's immediate outcome:

- Use `nemo-relay-get-started` for a first working scope, tool call, LLM call,
  or trial plugin setup.
- Use the NeMo Relay CLI documentation or host-specific setup for a local CLI
  host-plugin workflow, not application runtime setup.
- Use the matching plugin or instrumentation skill for runtime configuration,
  plugin files, observability, or adaptive behavior.

## Common Mistakes

Avoid these installation-scope mistakes:

- Using repository development setup when the user only needs a published
  package.
- Installing the CLI when the user needs an application binding, or installing a
  binding when the user only needs the local `nemo-relay` executable.
- Treating persistent host-plugin installation as the default way to try Relay,
  instead of starting with temporary transparent run.
- Pinning old versions unless the user or project explicitly requires that
  version.
- Continuing into `plugins.toml`, middleware registration, scopes, or quick-start
  examples before the install step has been verified.

## Public Docs To Reference

Use these public entry points to confirm current installation guidance:

- [Installation](https://docs.nvidia.com/nemo/relay/getting-started/installation)
- [Prerequisites](https://docs.nvidia.com/nemo/relay/getting-started/prerequisites)
- [CLI transparent run](https://docs.nvidia.com/nemo/relay/dev/nemo-relay-cli/basic-usage#transparent-run)
- [Configuration and setup handoff](https://docs.nvidia.com/nemo/relay/getting-started/configuration)
