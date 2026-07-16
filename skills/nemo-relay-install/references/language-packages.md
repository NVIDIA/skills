<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Language Package Installation

Use this path when a Python, Node.js, or Rust application directly owns its
tool or model call sites. Preserve the project's package manager and install
only the package needed for the selected language.

## Python

Require Python 3.11 or newer. Inspect `pyproject.toml`, its build metadata, and
the project lockfiles before choosing a command. Use explicit project
instructions and manager-specific `pyproject.toml` configuration as the source
of truth. When those signals are absent, use a single manager-specific lockfile.
If metadata and lockfiles conflict, or more than one manager appears active,
show the conflict and ask which manager is authoritative before changing the
project.

After resolving any conflict, preserve the selected manager:

- Use `uv add nemo-relay` when `uv.lock` or existing project instructions select
  `uv`.
- Use `poetry add nemo-relay` when `poetry.lock` or `[tool.poetry]` selects
  Poetry.
- Use `pdm add nemo-relay` when `pdm.lock` or `[tool.pdm]` selects PDM.
- Use the project's documented dependency workflow when another manager owns
  `pyproject.toml`.

For a `uv`-managed project, run:

```bash
uv add nemo-relay
```

For an active `uv` environment without project metadata, run:

```bash
uv pip install nemo-relay
```

Use `python -m pip install nemo-relay` only when no project manager owns the
environment. If multiple or conflicting lockfiles exist, ask which manager is
authoritative instead of creating another lockfile. Verify in the target
environment:

```bash
python -c "import nemo_relay"
```

## Node.js

Require Node.js 24 or newer and an existing `package.json`. Inspect its
`packageManager` field and lockfiles before choosing a command. Treat
`packageManager` as authoritative. If it disagrees with an existing lockfile,
show the mismatch and ask before running an install that could rewrite the stale
lockfile. Without `packageManager`, use a single lockfile to select the manager;
when multiple lockfiles exist, ask which one is authoritative before changing
the project.

After resolving any conflict, use the selected manager:

- Use `npm install nemo-relay-node` for `package-lock.json` or
  `npm-shrinkwrap.json`.
- Use `pnpm add nemo-relay-node` for `pnpm-lock.yaml`.
- Use `yarn add nemo-relay-node` for `yarn.lock`.
- Use `bun add nemo-relay-node` for `bun.lock` or `bun.lockb`.

Use npm only as the fallback when the project declares no package manager and
has no existing lockfile:

```bash
npm install nemo-relay-node
```

Verify that the package and lockfile record `nemo-relay-node`.

## Rust

Require Rust 1.86 or newer and an existing `Cargo.toml`:

```bash
cargo add nemo-relay
```

Add `nemo-relay-adaptive` only when adaptive runtime primitives are already in
scope:

```bash
cargo add nemo-relay-adaptive
```

Verify that `Cargo.toml` includes the selected crate and the lockfile resolves
it. Do not configure adaptive behavior from the install skill.

Use these public references to confirm prerequisites and published packages:

- [Prerequisites](https://docs.nvidia.com/nemo/relay/getting-started/prerequisites)
- [Installation](https://docs.nvidia.com/nemo/relay/getting-started/installation)
