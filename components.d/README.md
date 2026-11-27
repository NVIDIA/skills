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
| `product_primary`      | Exact `name` match, then existing metadata or AI | Canonical `product.primary` value from the metadata schema. Applies to every skill in this registry. |
| `ref`                  | `main`             | Branch to sync from.                                       |
| `links.contributing`   | `CONTRIBUTING.md`  | Path to the contributing file in the source repo.          |
| `links.discussions`    | `true`             | Set to `false` if the repo has no Discussions tab.         |
| `links.security`       | `true`             | Set to `false` if the repo has no `SECURITY.md`.           |
| `links.primary`        | `true`             | Set to `false` when this component shares its `name` with another component, so the shared README row keeps the other component's help links. |

## Plugins (optional)

A component may also publish one or more **plugins** — a directory in the product repo that bundles skills with their evals, MCP servers, hooks and assets, plus a `plugin.json`, `plugin-card.md`, `plugin.oms.sig` and `BENCHMARK.md` whose verdict is PASS or PASS WITH WARNINGS (plugin-level eval data is optional). Plugins are mirrored byte-for-byte into `plugins/<catalog_dir>/` by a separate hourly workflow (`sync-plugins.yml`, PR branch `automated/sync-plugins`). A missing, drifted or mismatched `plugin.oms.sig` holds the plugin back, as for skills; any other gap (a missing file, a non-PASS verdict, a name mismatch, a bundled skill without its own `skill.oms.sig`, `skill-card.md` or evals, or an unbumped version) is synced and flagged for the reviewer in the sync PR. The catalog never generates or edits anything inside a mirrored plugin; `version` in `plugin.json` is owned by the product repo and must increase whenever the plugin's content changes.

| Field         | Type   | Description                                                                                     |
|---------------|--------|-------------------------------------------------------------------------------------------------|
| `plugins`     | list   | Plugin source locations — one entry per plugin.                                                 |
| `plugins[].path`        | string | Directory in the source repo containing the plugin (`plugin.json` at its root). Use `plugins/<name>/` as the source layout. The required NVSkills CI status check watches only `skills/` and `team-skills/`; `plugins/` changes are excluded. |
| `plugins[].catalog_dir` | string | Directory name under `plugins/` in this catalog and the plugin's install name (must equal `name` in `plugin.json`, unique across components). |

```yaml
plugins:
  - path: plugins/nvidia-your-product/   # plugins/<name>/ in the product repo
    catalog_dir: nvidia-your-product     # must equal `name` in plugin.json; not a plugins.d name
```

Skills bundled inside a plugin are independent copies of the standalone `skills/` entries; the sync does not deduplicate them.

## Declared product mapping

`product_primary` is separate from the display `name`. Use it when the display
name differs from the canonical product value. The generator checks the value
against [the metadata schema](../.github/scripts/marketplace/metadata.schema.json).
An invalid declaration fails the run before output files are written.

The declared value overrides existing metadata and AI amendments. Without this
key, the generator keeps the existing exact-name match and enrichment behavior.
The published field remains `product.primary`; the output structure does not change.

```yaml
name: Jetson BSP
product_primary: Jetson
```

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

## Catalog metadata for a registered plugin

A `plugins:` entry gives the plugin its taxonomy on build.nvidia.com, the same way a `skills:` entry does for a skill. `.github/scripts/marketplace/generate-skill-metadata.py` reads this file, finds the mirrored tree at `plugins/<catalog_dir>/`, identifies the skills bundled inside it, and rolls them up into a `plugins` entry in `metadata.json`: one Domain, the set of Libraries, and the Audience values held by at least half of them. Nothing is authored per plugin and no inference runs.

A bundled skill is identified by the `name` in its `SKILL.md` frontmatter, or failing that by its path resolved through this file's own `skills:` entries. **It is not identified by its directory name**, because `catalog_dir` renames a skill on the way into the catalog and the two names often differ — BioIR ships `skills/bench-perf-oss/` and the catalog publishes it as `skills/bioir-bench-perf-oss`.

Two consequences worth knowing:

- Changing what a plugin bundles changes `metadata.json`. The plugin sync does not regenerate it; the Generate Skill Metadata workflow does, on its own run once the sync PR merges to `main`, and opens a regeneration PR. A hand-edited PR that touches `plugins/` is checked before it merges, so it must re-run the generator and commit the result, or CI's byte-stability check fails.
- A plugin registered here but not yet present at `plugins/<catalog_dir>/` is reported and left out, not treated as an error. That is the normal state before the first successful sync, and also while the plugin sync holds a plugin back over signature integrity. Either way, a plugin already published keeps its previous entry rather than disappearing.

A plugin whose name is declared in `plugins.d/` is built by the catalog rather than mirrored, and is read from there instead. The match is on the `name:` field inside the file, not on the filename, because that is the name `build-plugins.py` builds `plugins/<name>/` from. This matches the plugin sync, which resolves the same set through `list-built-plugins.sh` and never mirrors, prunes or lists such a directory.

## How aggregation works

The sync workflow runs:

```bash
yq ea '[.] | {"components": .}' components.d/*.yml > /tmp/components.aggregated.yml
```

Then iterates the resulting `components` list. Files are read in alphabetical order; the README regenerator sorts the result by display name independently.

## Removal and orphan pruning

Removing a skill entry from a component file removes its `skills/<catalog_dir>/` copy from the catalog on the next sync — the sync's orphan-pruning step (`.github/scripts/prune-orphans.sh`) deletes any top-level `skills/` dir with no `components.d` registration. Dirs that intentionally exist without a registration (e.g. contributor-facing or manually curated skills) must be listed in `catalog-exceptions.yml` at the repository root, with a reason and owner. Pruning is skipped for the whole run if any component file fails to parse, and refuses to act (flagging for human triage instead) if more than 5 dirs would be removed at once.

Removing a `plugins:` entry likewise removes `plugins/<catalog_dir>/` on the next plugin sync (`.github/scripts/prune-orphan-plugins.sh`, same parse guard and a separate cap of 5). Plugins built from `plugins.d/` (currently `nvidia-skills`) are outside the plugin sync entirely; other unregistered plugin dirs go under `plugin_exceptions:` in `catalog-exceptions.yml`.
