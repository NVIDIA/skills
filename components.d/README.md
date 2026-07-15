<!-- SPDX-License-Identifier: Apache-2.0 AND CC-BY-4.0 -->
<!-- Copyright (c) 2026 NVIDIA Corporation. All rights reserved. -->

# `components.d/` — Component catalog

One file per component, aggregated at workflow runtime. This layout exists so simultaneous onboarding PRs from different teams never touch the same file — eliminating merge conflicts at scale.

## Onboarding a new component

1. Create `components.d/<slug>.yml` where `<slug>` is your component name lowercased with spaces replaced by dashes (e.g., `Nemotron Voice Agent` → `nemotron-voice-agent.yml`).
2. Fill in the fields below.
3. Open a pull request — that's it. The sync workflow picks it up automatically; the README's Available Skills and Getting Help tables regenerate on the next sync.

## Required fields

| Field         | Type   | Description                                                                 |
|---------------|--------|-----------------------------------------------------------------------------|
| `name`        | string | Display name shown in the README (e.g., `CUDA-Q`, `Nemotron Voice Agent`).  |
| `repo`        | string | GitHub repository (`owner/repo`).                                           |
| `description` | string | One-line description for the README's Available Skills table.               |
| `skills`      | list   | Skill source locations — one entry per skill (see below).                   |

Each entry under `skills:` requires:

| Field         | Type   | Description                                                                                |
|---------------|--------|--------------------------------------------------------------------------------------------|
| `path`        | string | Directory in the source repo containing a single skill (a directory with `SKILL.md` at its root). |
| `catalog_dir` | string | Top-level directory name under `skills/` in this catalog (must be unique across components). |

## Optional fields

| Field                  | Default            | Description                                                |
|------------------------|--------------------|------------------------------------------------------------|
| `ref`                  | `main`             | Branch to sync from.                                       |
| `links.contributing`   | `CONTRIBUTING.md`  | Path to the contributing file in the source repo.          |
| `links.discussions`    | `true`             | Set to `false` if the repo has no Discussions tab.         |
| `links.security`       | `true`             | Set to `false` if the repo has no `SECURITY.md`.           |

## Example

```yaml
# components.d/your-product.yml
name: Your Product
repo: NVIDIA/your-product
description: One-line description of what the skills do.
skills:
  - path: skills/your-product-install/
    catalog_dir: your-product-install
  - path: skills/your-product-deploy/
    catalog_dir: your-product-deploy
```

Each entry points at one skill source directory and creates one top-level catalog directory under `skills/`. The catalog layout mirrors the source naming 1:1 for discoverability — each skill becomes its own `skills/<catalog_dir>/` entry in this catalog, rather than nesting under a per-product subdirectory.

## Bulk layout (deprecated)

> [!IMPORTANT]
> The bulk layout — where one entry's `path` points at a *parent* directory containing multiple skills, all landing under a single nested `catalog_dir` — is deprecated for new components. Use the flat layout above instead.
>
> Existing components on the bulk layout continue to work and will be migrated incrementally. The sync workflow handles both layouts identically (a directory-to-directory `rsync`); the difference is in the catalog shape produced.

The deprecated pattern looked like:

```yaml
# DEPRECATED — do not copy for new components
skills:
  - path: skills/
    catalog_dir: your-product
```

This produced nested paths like `skills/your-product/<skill-name>/` for every skill found under the source `skills/` directory. The flat layout above is preferred because:

- Each skill gets a discoverable top-level catalog directory, scannable alongside skills from other components.
- The `catalog_dir` per skill makes the catalog naming explicit at onboarding time rather than relying on whatever subdirectory names happened to ship in the source `skills/` tree.
- Source skills can be added, removed, or renamed without changing the catalog's top-level shape silently.

## How aggregation works

The sync workflow runs:

```bash
yq ea '[.] | {"components": .}' components.d/*.yml > /tmp/components.aggregated.yml
```

Then iterates the resulting `components` list. Files are read in alphabetical order; the README regenerator sorts the result by display name independently.

## Removal and orphan pruning

Removing a skill entry from a component file removes its `skills/<catalog_dir>/` copy from the catalog on the next sync — the sync's orphan-pruning step (`.github/scripts/prune-orphans.sh`) deletes any top-level `skills/` dir with no `components.d` registration. Dirs that intentionally exist without a registration (e.g. contributor-facing or manually curated skills) must be listed in `catalog-exceptions.yml` at the repository root, with a reason and owner. Pruning is skipped for the whole run if any component file fails to parse, and refuses to act (flagging for human triage instead) if more than 5 dirs would be removed at once.
