<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Usd Optimize Validator Infrastructure - Upstream Handoff

This local reference preserves the digitaltwin workflow milestone. Scene
Optimizer mechanics for this step are owned by upstream `usd-optimize`.

- Public repository: [https://github.com/NVIDIA-Omniverse/usd-optimize/](https://github.com/NVIDIA-Omniverse/usd-optimize/)
- Package path: `.agents/skills/validators/SKILL.md`
- Upstream web URL: [https://github.com/NVIDIA-Omniverse/usd-optimize/blob/main/.agents/skills/validators/SKILL.md](https://github.com/NVIDIA-Omniverse/usd-optimize/blob/main/.agents/skills/validators/SKILL.md)

Resolve the upstream guide without cloning the source repo:

1. `$USD_OPTIMIZE_ROOT/.agents/skills/validators/SKILL.md`
2. `$USD_OPTIMIZE_ROOT/.agents/skills/validators/SKILL.md`

If no package root is available, download and extract the prebuilt Usd Optimize release package (current asset name + download: `references/upstreams/usd-optimize.md`) (direct
archive URLs are in `references/upstreams/usd-optimize.md`), or use the package
path/URL supplied by the user. If the user supplies an extracted
package root directly, resolve this same package path under that root. If
GitHub raw fetch is available, the web URL above is acceptable for docs-only
reads. Do not clone the source repo just to read upstream SO guidance.

## Local Responsibilities

- Local validation scope, phase-aware subsets, and expensive-check gates remain in `usd-validation-runner/README.md`.
- Setup/install references own runtime selection and `setup-preflight.json` writer behavior.
