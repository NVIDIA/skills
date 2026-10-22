<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Language Package Installation

Use this path when the user's application owns the calls and needs a Relay
binding. Establish that intent before inferring an install type from a manifest.

Read the current [installation guide](https://docs.nvidia.com/nemo/relay/getting-started/installation)
for the selected language, together with the selected release's package metadata
and linked support requirements. Resolve the latest published stable package and
derive its runtime requirements from current package metadata and release
documentation. Do not retain a package-version or language-minimum table here.

## Preserve The Authoritative Environment

- Python: inspect project instructions, `pyproject.toml`, manager-specific
  configuration, and lockfiles. Preserve the established manager and virtual
  environment rather than introducing another dependency workflow.
- Node.js: inspect `package.json`, especially `packageManager`, and the
  lockfiles. Preserve the declared manager and its dependency conventions.
- Rust: inspect `Cargo.toml`, the workspace, toolchain configuration, and
  lockfile. Add only the runtime crates needed for the selected outcome.

When manager declarations and lockfiles conflict, explain the conflict and
resolve which is authoritative before changing the project. Do not initialize
another manifest or lockfile in an existing project.

Use the selected manager's syntax for the resolved stable dependency, respecting
explicit project constraints. A newer repository version or quick-start pin is
not evidence that a package has been published.

Verify availability in the actual target environment: an import or module-load
check, or package-manager resolution as documented for that binding. Do not
accidentally verify a system interpreter instead of the project environment.

Stop before scopes, tool/model examples, plugin configuration, or adaptive
behavior. Pass the selected language, environment, and installed version to the
matching quick-start guide through `nemo-relay-get-started`.
