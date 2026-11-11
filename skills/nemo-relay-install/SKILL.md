---
name: nemo-relay-install
description: Use this skill when choosing or running NeMo Relay installation for an agent runtime, application language, framework integration, or plugin bundle, or checking a runtime's native Relay readiness. Do not use it for first-run instrumentation or export configuration.
license: Apache-2.0
metadata:
  author: NVIDIA Corporation and Affiliates
---

# Install NeMo Relay

Resolve what the user needs, install the appropriate current stable package or
executable, verify availability, and stop. Installation does not include a
first application, plugin configuration, scopes, or a model-consuming trial.

## Disambiguate Before Installing

Establish the desired outcome and the boundary that owns execution before
selecting a Relay type. Inspect available project context, but do not assume
that a Python manifest means the user wants a Python binding rather than Relay
for their coding agent.

When the target is unclear, ask one concise question:

> Are you adding Relay to an existing agent runtime or framework, integrating
> it into your own application, or using its CLI or local gateway?

Resolve the named runtime, framework, language, or plugin only as far as needed
to select its installation instructions. Preserve an explicit choice or an
earlier handoff; do not repeat questions already answered. A generic "install
Relay" or "try Relay" request alone does not authorize choosing the CLI.

| Established target | Reference |
|---|---|
| CLI, coding-agent wrapper, or persistent host integration | [CLI Installation](references/cli-install.md) |
| Native agent integration, framework integration, or plugin bundle | [Maintained Integrations](references/maintained-integrations.md) |
| Application-owned language binding | [Language Packages](references/language-packages.md) |
| Explicit contributor or source-build request | Current repository development guide |

## Resolve Current Stable Requirements

Read the current [installation guide](https://docs.nvidia.com/nemo/relay/getting-started/installation)
for the selected target. Resolve prerequisites from the selected release's
installation sections, support matrix, and package metadata rather than a
remembered standalone prerequisites page. Confirm the latest published stable
release using [release metadata](https://api.github.com/repos/NVIDIA/NeMo-Relay/releases/latest),
[official releases](https://github.com/NVIDIA/NeMo-Relay/releases/latest), and
the relevant package registry. Check the target's actual platform assets and
package availability, not just a repository version or a pending PR.

- Default to the latest published stable release, excluding prereleases unless
  explicitly requested. Preserve explicit user or project version constraints.
- Derive operating-system, architecture, language, host, and dependency
  requirements from that release's documentation and metadata at execution time.
- Do not copy version pins or minimum-version tables into this skill. Retain a
  specific version note only for an established incompatibility, with its source.
- A quick-start example's pin does not override the stable default. If its
  requirements conflict with the selected release or project, explain the
  mismatch and resolve the choice before installing.
- Native or dependency-managed integrations may own their Relay dependency.
  Follow that supported relationship rather than installing a second runtime.
- Resolve independently released bundles separately and check compatibility.
  Their release number need not match Relay's.

Resolve the documentation version explicitly: an unversioned page may display
an older stable guide. When rendered documentation is inaccessible or lacks the
selected path, use the official repository at the matching release tag; use
its development branch only for an explicitly requested development workflow.
A cached release redirect alone does not establish the latest version.

If current sources cannot be checked, state what is unverified. Do not infer
the newest release or requirements from this skill, an old checkout, a generated
benchmark, or a source-built quick-start test.

## Install And Verify

Inspect the environment, existing installation, manifest, and authoritative
package manager. Load only the relevant reference and current target-specific
documentation. Preserve the project's manager and virtual environment.

Derive the install command from those sources, preview it, and proceed within
existing authorization. For remote installers, separate download, review, and
execution; use the resolved release's installer and integrity checks rather
than executing an unreviewed default-branch response.

Use only the documented availability check for the chosen target. Diagnose a
failed check at that boundary instead of reinstalling unrelated packages.
Report the resolved release, installation location, and verification evidence.
Do not configure export, launch a gateway, or make a model call as an install
check.

## Persistent Codex Desktop Continuity

Persistent Codex setup can change the provider used to display Desktop
conversations. Before that path, read the continuity section in
[CLI Installation](references/cli-install.md), check the referenced upstream
issue's applicability, and explain any relevant risk.

Retain the recovery-note and explicit confirmation gate before global changes.
Do not inspect, copy, delete, or rewrite private session storage or SQLite state
as a visibility workaround. Use supported Relay install, uninstall, and
diagnostic commands.

## Hand Off

For a first working example, hand off to `nemo-relay-get-started` with the
chosen target, resolved release, environment, and verified installation.
Continue from the appropriate guide step without repeating selection.

For durable export, instrumentation expansion, or plugin behavior, use the
matching workflow. Do not repeat installation when it is already complete.
