# `plugins.d/` — plugin catalog source

Each `<name>.yml` here defines one plugin. The build script
[`.github/scripts/build-plugins.py`](../.github/scripts/build-plugins.py)
parses these files and (re)generates:

- `plugins/<name>/.claude-plugin/plugin.json`
- `plugins/<name>/.codex-plugin/plugin.json`
- `plugins/<name>/.cursor-plugin/plugin.json`
- `plugins/<name>/skills/<skill-basename>/` &nbsp;**symlinks** into the canonical `skills/` catalog
- `.claude-plugin/marketplace.json` (top-level Claude marketplace registry)
- `.agents/plugins/marketplace.json` (top-level Codex marketplace registry)
- `.cursor-plugin/marketplace.json` (top-level Cursor marketplace registry)

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

## `skill_files:` — copy vs symlink

Each plugin selects what kind of files end up under
`plugins/<name>/skills/`:

| Mode | What's on disk | Use when |
|---|---|---|
| `copy` (default) | real files (rsync) | publishing to Codex / Anthropic; required for `codex plugin add` (Codex drops symlinks during install) |
| `symlink` | relative symlinks → `../../../skills/<Product>/<skill>` | shipping to Claude only or to `npx skills add` consumers; avoids duplication |

The default lives in [`_defaults.yml`](./_defaults.yml); override per
plugin by setting `skill_files: symlink` (or `copy`) in
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

## Renaming a plugin

The build script doesn't know a rename happened — it just sees a new
name and builds a fresh `plugins/<new-name>/` next to the old folder.
You have to delete the old one yourself, or it will sit in the repo
forever (CI won't flag it).

To rename `plugins.d/old.yml` → `plugins.d/new.yml`:

```sh
git mv plugins.d/old.yml plugins.d/new.yml
# edit the file and change `name: old` to `name: new`
git rm -r plugins/old
.github/scripts/build-plugins.sh
```

The rebuild regenerates both `marketplace.json` files for you; just
`git add` everything that changed (the renamed yaml, the new
`plugins/new/` tree, the deleted `plugins/old/`, and both
`marketplace.json` files) and commit it all together.

Heads up: the plugin name is what users type to install
(`claude plugin install <name>`, `codex plugin add <name>`). If the old
name has been published anywhere, renaming is a breaking change for
those users.

## Curated (hand-maintained) plugins

> Legacy fallback — no plugins currently use this mode. Every plugin in
> this repo is catalog-driven (defined by a `plugins.d/<name>.yml`).

The build script still supports a hand-maintained mode: a directory under
`plugins/<name>/` that has its own `.skills-manifest.yml` instead of a
`plugins.d/<name>.yml` is treated as curated — the build only refreshes
its `skills/` tree and otherwise leaves `.claude-plugin/`,
`.codex-plugin/`, `.cursor-plugin/`, `assets/`, and the marketplace
entries hand-edited.
