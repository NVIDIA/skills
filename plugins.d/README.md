# `plugins.d/` — plugin catalog source

Each `<name>.yml` here defines one plugin. The build script
[`.github/scripts/build-plugins.py`](../.github/scripts/build-plugins.py)
parses these files and (re)generates:

- `plugins/<name>/.claude-plugin/plugin.json`
- `plugins/<name>/.codex-plugin/plugin.json`
- `plugins/<name>/.cursor-plugin/plugin.json`
- `plugins/<name>/plugin.json` ([Agent Plugins 1.0.0](https://agent-plugins.org/schemas/1.0.0/plugin.schema.json) manifest, read by Agent Plugins compatible clients such as Hermes)
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
| `symlink` | relative symlinks → `../../../skills/<skill>` | shipping to Claude only or to `npx skills add` consumers; avoids duplication |

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
     - skills/<skill>/
   ```

   The `skills/` catalog is flat: a skill directory sits directly under
   `skills/`, never under a product folder. `skills/tao-setup/` is a valid
   path; `skills/TAO Toolkit/tao-setup/` is not, and will not resolve.

2. Run `.github/scripts/build-plugins.sh`.
3. Run `python3 .github/scripts/marketplace/generate-skill-metadata.py`.
   Editing `include_skills` changes the plugin's catalog metadata, and PR CI
   fails on drift if the regenerated file is not committed.
4. Commit the regenerated `plugins/<name>/` tree, the updated
   `marketplace.json` files, and `.github/scripts/marketplace/metadata.json`
   alongside the new yaml.

## Catalog metadata

`include_skills` is also what gives a plugin its taxonomy on
build.nvidia.com. `.github/scripts/marketplace/generate-skill-metadata.py`
reads this directory, joins each listed path against the enriched skill
entries in `metadata.json`, and rolls the result up into a `plugins` entry
there: one Domain, the set of Libraries, and the Audience values held by at
least half the bundled skills. Taxonomy is derived rather than authored by
default, and no inference runs.

Two consequences worth knowing:

- Editing `include_skills` changes `metadata.json`. Re-run the generator and
  commit the regenerated file, or CI's byte-stability check will fail.
- A plugin that bundles no catalog skill has nothing to roll up. It is left
  out of `metadata.json` with a warning, or kept from the previous
  `metadata.json` if it was published before. Declare its taxonomy by hand
  instead:

  ```yaml
  taxonomy:
    classification.category.primary: developer_tools
    product.primary:
      - DOCA
    audience: developer,platform_engineer
  ```

  A declared field always wins over the rolled-up one. When a plugin bundles
  skills you may declare a single field and let the rest roll up; when it
  bundles none there is nothing to roll up, so all three are required.
  Values are checked against `metadata.schema.json`, which is the single
  source of truth for the controlled vocabulary, and an unrecognised field
  name is rejected rather than ignored.

A plugin that fails any of these checks is left out of `metadata.json` with a
warning. It never fails the generator: one team's stale `include_skills` entry
must not block regeneration of the whole catalog.

### Known limitation: directory-form includes

`build-plugins.sh` accepts a grouping directory in `include_skills` and
expands it to the immediate children that hold a `SKILL.md`. The metadata
generator does **not**. It refuses such an entry and skips the plugin, with a
warning naming the file.

This is deliberate. The `skills/` catalog is flat today — every skill sits at
`skills/<name>/`, there are no grouping directories, and no plugin uses the
form — so the feature has no users here, and approximating its semantics
produced bugs twice. List each skill path explicitly.

If the catalog is ever regrouped by product, implement the expansion in
`generate-skill-metadata.py` to match `expand_skill_paths` exactly: a
directory holding its own `SKILL.md` resolves to that skill, otherwise to its
immediate children only, never to deeper descendants.

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
