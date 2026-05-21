<!-- SPDX-License-Identifier: Apache-2.0 AND CC-BY-4.0 -->
<!-- Copyright (c) 2026 NVIDIA Corporation. All rights reserved. -->

# Documentation

Long-form documentation for the NVIDIA Agent Skills catalog. Pages are authored in MDX and published to <https://docs.nvidia.com/skills>.

## Layout

- `index.mdx` — landing page
- `advanced-install.mdx` — advanced skills CLI usage
- `agent-skill-trust-pipeline.mdx` — trust pipeline overview
- `scanning-agent-skills.mdx` — pre-install scanning guide
- `signing-agent-skills.mdx` — signature verification guide
- `skill-cards.mdx` — Skill Card authoring guide
- `release-checklist.mdx` — release process

Navigation order and slugs are defined in [`../fern/docs.yml`](../fern/docs.yml).

## Build Locally

You only need to build locally if you are editing the documentation. The site is published automatically from `main` via the Fern GitHub integration.

Prerequisites: Node.js 18+ and npm.

```bash
# Install the Fern CLI
npm install -g fern-api

# From the repo root, preview the site with live reload
fern docs dev

# Validate the docs configuration and links
fern check
```

`fern docs dev` serves the site at <http://localhost:3000> and reloads on changes to `.mdx` files or `fern/docs.yml`.

## Authoring Notes

- New pages must be added to the `navigation` section of [`../fern/docs.yml`](../fern/docs.yml); otherwise they will not appear in the site.
- Use relative links between MDX pages (`./skill-cards.mdx`) and absolute repo links (`/components.d/README.md`) for files outside `docs/`.
- The Fern version is pinned in [`../fern/fern.config.json`](../fern/fern.config.json); bump it intentionally rather than as a side effect.
