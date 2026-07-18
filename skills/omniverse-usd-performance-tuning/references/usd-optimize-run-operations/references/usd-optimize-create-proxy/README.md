<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# usd-optimize-create-proxy - Specialty Handoff

This local reference preserves the digitaltwin workflow milestone. Scene
Optimizer mechanics for this step are owned by upstream `usd-optimize`.

- Public repository: [https://github.com/NVIDIA-Omniverse/usd-optimize/](https://github.com/NVIDIA-Omniverse/usd-optimize/)
- Package path: `.agents/skills/create-proxy/SKILL.md`
- Upstream web URL: [https://github.com/NVIDIA-Omniverse/usd-optimize/blob/main/.agents/skills/create-proxy/SKILL.md](https://github.com/NVIDIA-Omniverse/usd-optimize/blob/main/.agents/skills/create-proxy/SKILL.md)

Resolve the upstream guide without cloning the source repo:

1. `$USD_OPTIMIZE_ROOT/.agents/skills/create-proxy/SKILL.md`
2. `$USD_OPTIMIZE_ROOT/.agents/skills/create-proxy/SKILL.md`

If no package root is available, download and extract the prebuilt Usd Optimize release package (current asset name + download: `references/upstreams/usd-optimize.md`) (direct
archive URLs are in `references/upstreams/usd-optimize.md`), or use the package
path/URL supplied by the user. If the user supplies an extracted
package root directly, resolve this same package path under that root. If
GitHub raw fetch is available, the web URL above is acceptable for docs-only
reads. Do not clone the source repo just to read upstream SO guidance.

## Local Responsibilities

- Treat proxy creation as a specialty user-request path, not part of the main optimization flow.
- Use local runtime setup, output workspace, edit-target planning, and approval policy before any end-to-end run.
- Route `pythonScript` usage through upstream `create-proxy` mechanics and local approval/review policy.
