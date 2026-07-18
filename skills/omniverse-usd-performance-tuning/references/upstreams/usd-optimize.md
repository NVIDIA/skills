<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# usd-optimize / Usd Optimize Package Handoff

Usd Optimize operation mechanics are owned by upstream `usd-optimize` and
ship with the prebuilt Usd Optimize package. This package owns digital twin
workflow routing, runtime setup context, validation scope, output workspace
policy, batch orchestration, and reporting.

- Public repository: [https://github.com/NVIDIA-Omniverse/usd-optimize/](https://github.com/NVIDIA-Omniverse/usd-optimize/)
- Prebuilt packages: **GitHub Releases** on the repository above
  (`https://github.com/NVIDIA-Omniverse/usd-optimize/releases`). Each release
  carries Linux x86_64, Linux aarch64, and Windows x86_64 zips (~330-360 MB).
- Package pattern: `usd_optimize_usd_<usd>_py_<python>@<version>.<platform>.release.zip`
  (1.0.x semver; usd-optimize 1.0.x is the minimum supported runtime for
  this skill).
- Download example:
  `gh release download v1.0.4 -R NVIDIA-Omniverse/usd-optimize -p '*manylinux*x86_64*'`
  (or pick the asset from the releases page in a browser).
- Package operation guides: `.agents/operations/<operation>.md`
- Package operation runner skill: `.agents/skills/run-operations/SKILL.md`
- Package validator runner skill: `.agents/skills/run-validators/SKILL.md`
- Package validator interpretation skill: `.agents/skills/interpret-validators/SKILL.md`
- Package proxy skill: `.agents/skills/create-proxy/SKILL.md`
- Package install skill: `.agents/skills/prebuilt-package/SKILL.md`

## Operation Guide Resolution

For any operation key listed in `references/operations/operations.json`, derive
the upstream mechanics path instead of storing per-operation package details in
this repo:

- Package path template: `.agents/operations/<operation-key>.md`
- Upstream web URL template: `https://github.com/NVIDIA-Omniverse/usd-optimize/blob/main/.agents/operations/<operation-key>.md`
- Package operation index: `.agents/operations/INDEX.md`

Resolve local upstream guidance without cloning the source repo:

1. `$USD_OPTIMIZE_ROOT`

Each root above must contain `.agents/operations/INDEX.md` and the runtime
sentinels `python/`, `usdpy/`, `lib/`, and `extraLibs/` when it is also used
for standalone execution. The package may include `.claude` and `.codex`
compatibility aliases, but handoffs should use `.agents` paths.

If no package root exists, download and extract the published
`usd_optimize_...release.zip` package for the target platform from GitHub
Releases, or use the package archive path, release-asset URL, or extracted
package root supplied by the user. Package-internal paths (`.agents/...`,
`python/`, `usdpy/`, `lib/`, `extraLibs/`) were last verified against the
110.x packages; re-verify against the extracted 1.0.x package on first use. If web or raw GitHub fetch is available, the public
repository URL can be used for docs-only reads. Do not clone the source repo
just to read operation parameters, defaults, or implementation gotchas.

Use `references/operations/operations.json` — the single catalog carrying both
routing metadata and the nested `curation` block (generated `status` +
authored `wired_into`; `rationale` only on overrides) — for digitaltwin
routing, risk, confirmation, and recommendation
posture. Before invoking any operation, consume
`<output_path>/setup-preflight.json` and confirm the op appears in
`usdOptimize.operationsAvailable`.
