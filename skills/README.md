<!-- SPDX-License-Identifier: Apache-2.0 AND CC-BY-4.0 -->
<!-- Copyright (c) 2026 NVIDIA Corporation. All rights reserved. -->

# Skills Catalog

This folder contains the NVIDIA skills mirrored from product repositories. Each product folder contains one or more skill directories, and each skill directory has a `SKILL.md` file at its root.

Install skills with the `skills` CLI from the repository root source:

```bash
# Browse available skills
npx skills add nvidia/skills --list

# Install interactively
npx skills add nvidia/skills

# Install one skill
npx skills add nvidia/skills --skill cuopt-numerical-optimization-api-python
```

See the repository [README](../README.md) for the full catalog, or [Advanced installation](../docs/advanced-install.md) for agent-specific and non-interactive installs.

Do not hand-edit generated product folders here. Product skills are maintained in their source repositories and mirrored into this catalog by automation.
