<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Maintained Integration Installation

Read the current [installation overview](https://docs.nvidia.com/nemo/relay/getting-started/installation),
[quick-start index](https://docs.nvidia.com/nemo/relay/getting-started/quick-start),
and the selected [integration guide](https://docs.nvidia.com/nemo/relay/latest/supported-integrations/about).
Use only the relevant target's instructions.

## Distinguish The Packaging Boundary

| Target | What to establish from current documentation |
|---|---|
| OpenClaw | Its Relay plugin package, registered plugin ID, host/runtime compatibility, and package-registration check |
| Hermes | Native Relay readiness and its managed dependency; whether a host update is needed |
| LangChain, LangGraph, Deep Agents | Required framework extras and any provider-specific dependencies |
| Switchyard | Its current first-party plugin installation guide, supported Relay releases, and documented user/global install scope |

Do not copy current version numbers, extras combinations, archive names, or
installer recipes into this reference. Resolve the current stable package and
requirements at execution time. Preserve the existing Python package manager
and install provider dependencies only when the chosen provider needs them.

Hermes native support is distinct from local event export. Avoid installing a
duplicate Relay runtime or the former Hermes observability plugin when the
current native integration owns that dependency. Hand export configuration to
get-started rather than claiming native availability proves event capture.

For OpenClaw, review the package source and requested capabilities. Separate
package registration from loading new code into a running gateway; do not
restart a shared gateway as an install check.

For Switchyard, discover its current guide through the
[plugin overview](https://docs.nvidia.com/nemo/relay/latest/configure-plugins/about),
[discoverable-plugin guidance](https://docs.nvidia.com/nemo/relay/latest/configure-plugins/discoverable-plugins),
and the maintained [Relay Plugins source](https://github.com/NVIDIA/NeMo-Relay-Plugins/tree/main/plugins/switchyard-plugin).
It may be documented under plugin configuration rather than quick starts or
framework integrations. Use its documented manager and install scope; do not
assume manual archive extraction or a project-local manifest is required.
Resolve source documentation to the selected release before using its commands.

For native bundles, use the selected release's platform assets and published
integrity checks. Keep the manifest and library together. Check Relay/plugin
compatibility and platform requirements independently; do not assume matching
release numbers or universal native-library support.

Stop after the documented package, host-readiness, or bundle-availability
checks. Defer callback attachment, plugin enablement, attestation policy,
gateway lifecycle, credentials, and model calls to the selected quick start.
