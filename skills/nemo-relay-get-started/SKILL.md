---
name: nemo-relay-get-started
description: Use this skill when first-time NeMo Relay users want help selecting and following an appropriate quick-start guide or verifying their first working integration. Do not use it for production rollout or broad integration development.
license: Apache-2.0
metadata:
  author: NVIDIA Corporation and Affiliates
---

# Get Started With NeMo Relay

Help the user choose and follow the maintained quick-start guide that fits
their existing runtime, framework, plugin, or application. The guide owns the
implementation; this skill owns selection, continuity, and evidence.

## Select The Guide

Read the current [quick-start index](https://docs.nvidia.com/nemo/relay/getting-started/quick-start).
Use the user's stated goal and existing environment to find the smallest
applicable path. Preserve a target established in conversation or installation.

When the boundary is unclear, ask one concise question before choosing.
Do not infer that every generic "try Relay" request means a coding-agent demo,
or introduce a framework just to demonstrate Relay.

Use these references only for target-specific routing and cautions:

| Established boundary | Reference |
|---|---|
| Local coding agent or CLI | [CLI Trial](references/cli-try-now.md) |
| Native agent runtime or managed plugin | [Agent Runtime Trial](references/agent-runtimes-try-now.md) |
| Maintained framework integration | [Framework Trial](references/built-in-integrations-try-now.md) |
| Switchyard routing | [Switchyard Trial](references/switchyard-try-now.md) |
| Application-owned language calls | [Language Trial](references/manual-language-try-now.md) |

## Follow The Current Guide

Read the selected guide before proposing commands or configuration, including
its prerequisites, sample files, run instructions, verification, and limitations.
Use documentation compatible with the installed stable release. The current
published guide is the normal source; use an explicitly requested pending PR
or other supplied revision as such, without treating it as published support.
Resolve the documentation version instead of assuming an unversioned URL is
current. When rendered documentation is inaccessible or omits the selected
path, use the official source at the matching release tag, or the development
checkout for an explicitly requested development preview. A failed HTTP request
alone does not prove that a guide was removed; check its index and source.

Keep commands and referenced sample files from the same guide revision.
If working from source, resolve its referenced files in that checkout.
Do not reconstruct the recipe from this skill, an older session, or a
benchmark. If the guide cannot be read or its sample files are unavailable,
state the gap rather than inventing instructions.

For missing dependencies, use `nemo-relay-install` with the already-selected
target. Its default is the latest published stable release, subject to explicit
user or project constraints. Preserve a successful install and continue at
the next appropriate guide step; do not repeat target selection.

A guide's example pin is not an instruction to downgrade or use an unpublished
release. If the guide and installed release are incompatible, resolve that
mismatch using current release documentation before proceeding.

## Execute And Verify

Use [Guide-Based First-Value Checks](references/first-value.md) while adapting
only what the user's existing environment requires.

Explain the guide's attachment boundary and proposed changes. Preserve the
project's manager, unrelated settings, and existing artifacts. Proceed within
existing authorization, honoring requests to preview or stop before a live run.

Follow the selected guide's configuration, execution, shutdown, and verification
steps. Derive file locations, schemas, commands, prompts, lifecycle expectations,
and output formats from that guide rather than imposing a shared recipe.
Run only the chosen path and report the evidence it actually establishes.

Successful application output alone is not proof that Relay captured the work.
Use fresh evidence and the guide's own success criteria. Do not demand model
credentials, a tool call, ATIF, or another exporter when that guide does not.

## After The First Proof

Stop the initial trial when the guide's success checks pass. Explain the
observed value and Relay's progression: attach once, then extend supported
behavior through plugin configuration.

Resolve plugin and integration documentation for the installed Relay release,
including any explicit version constraint, before suggesting one next plugin
for the user's goal. Use the [stable plugin overview](https://docs.nvidia.com/nemo/relay/latest/configure-plugins/about)
only after confirming compatibility with that release; otherwise use the
release-matched documentation or official source. Do not repeat Observability
setup if the guide already established it, or offer capabilities unsupported
by the selected integration.

Hand broader instrumentation to `nemo-relay-instrument-calls`, unsupported
framework work to `nemo-relay-integrate-upstream`, and additional export to
`nemo-relay-plugin-observability`. Persistent host setup belongs to
`nemo-relay-install`, including its applicable Codex Desktop continuity gate.
Production deployment and multiple new behaviors belong to follow-up work.
