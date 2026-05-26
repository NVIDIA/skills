# `plugins.d/` — plugin catalog source

Each `<name>.yml` here defines one plugin. The build script
[`.github/scripts/build-plugins.py`](../.github/scripts/build-plugins.py)
parses these files and (re)generates:

- `plugins/<name>/.claude-plugin/plugin.json`
- `plugins/<name>/.codex-plugin/plugin.json`
- `plugins/<name>/skills/<skill-basename>/` &nbsp;**symlinks** into the canonical `skills/` catalog
- `.claude-plugin/marketplace.json` (top-level Claude marketplace registry)
- `.agents/plugins/marketplace.json` (top-level Codex marketplace registry)

Files whose names start with `_` are treated as includes and are not
themselves built into plugins. `_defaults.yml` provides shared author /
license / capability defaults; per-plugin yaml fields override the defaults
(shallow merge).

## Source of truth

The `skills/` directory is the single source of truth — every SKILL.md
exists exactly once there. The plugin tree under `plugins/` is reconstructed
from these YAML files on every build, so adding/removing a curated skill
only requires editing the `include_skills:` list and re-running:

```sh
.github/scripts/build-plugins.sh
```

## `materialize_skills:` — copy vs symlink

Each plugin selects how `include_skills` are reproduced under
`plugins/<name>/skills/`:

| Mode | What's on disk | Use when |
|---|---|---|
| `copy` (default) | real files (rsync) | publishing to Codex / Anthropic; required for `codex plugin add` (Codex drops symlinks during install) |
| `symlink` | relative symlinks → `../../../skills/<Product>/<skill>` | shipping to Claude only or to `npx skills add` consumers; avoids duplication |

The default lives in [`_defaults.yml`](./_defaults.yml); override per
plugin by setting `materialize_skills: symlink` (or `copy`) in
`plugins.d/<name>.yml`.

## Adding a plugin

1. Create `plugins.d/<name>.yml` with at minimum:

   ```yaml
   name: <name>           # lowercase kebab-case, must match the file basename
   description: ...        # one-line summary
   display_name: ...
   short_description: ...
   long_description: ...
   category: Developer Tools
   include_skills:
     - skills/<Product>/<skill>/
   ```

2. Run `.github/scripts/build-plugins.sh`.
3. Commit the regenerated `plugins/<name>/` tree and the updated
   `marketplace.json` files alongside the new yaml.

## Curated (hand-maintained) plugins

A directory under `plugins/<name>/` that has its own
`.skills-manifest.yml` instead of a `plugins.d/<name>.yml` is treated as
a curated plugin: the build script only refreshes its `skills/` symlinks
and otherwise leaves `.claude-plugin/`, `.codex-plugin/`, `assets/`, and
the marketplace entries hand-edited.
