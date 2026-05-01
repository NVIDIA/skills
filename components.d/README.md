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
| `skills`      | list   | Skill source locations (see below).                                         |

Each entry under `skills:` requires:

| Field         | Type   | Description                                                              |
|---------------|--------|--------------------------------------------------------------------------|
| `path`        | string | Directory in the source repo containing skills (e.g., `.agents/skills/`).|
| `catalog_dir` | string | Directory name under `skills/` in this catalog (must be unique).         |

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
  - path: .agents/skills/
    catalog_dir: your-product
```

For multi-skill components, add multiple entries under `skills:`:

```yaml
name: NeMo Evaluator
repo: NVIDIA-NeMo/Evaluator
description: LLM evaluation — launch evaluations, access MLflow results, and bring-your-own benchmarks.
skills:
  - path: packages/nemo-evaluator-launcher/.claude/skills/
    catalog_dir: NeMo-Evaluator-Launcher
  - path: packages/nemo-evaluator/.claude/skills/
    catalog_dir: NeMo-Evaluator
```

## How aggregation works

The sync workflow runs:

```bash
yq ea '[.] | {"components": .}' components.d/*.yml > /tmp/components.aggregated.yml
```

Then iterates the resulting `components` list. Files are read in alphabetical order; the README regenerator sorts the result by display name independently.
