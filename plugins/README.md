# `plugins/` — generated plugin bundles

Each subdirectory here is a self-contained plugin built from a source
definition in [`plugins.d/`](../plugins.d/) by
[`.github/scripts/build-plugins.py`](../.github/scripts/build-plugins.py).

A plugin folder typically contains:

- `.claude-plugin/plugin.json` — Claude plugin manifest
- `.codex-plugin/plugin.json` — Codex plugin manifest
- `.cursor-plugin/plugin.json` — Cursor plugin manifest
- `skills/` — the bundled skills (real files or symlinks back into the
  top-level [`skills/`](../skills/) catalog, depending on `skill_files:`
  in the source yaml)
- `assets/` — optional logo and other hand-maintained assets

## Don't edit these files directly

With the exception of curated plugins (see below), everything under
`plugins/<name>/` is **regenerated on every build**. To change a
plugin's metadata or skill list, edit `plugins.d/<name>.yml` and re-run:

```sh
.github/scripts/build-plugins.sh
```

See [`plugins.d/README.md`](../plugins.d/README.md) for the full
authoring workflow.

## Curated plugins

A plugin folder that contains its own `.skills-manifest.yml` (instead of
having a matching `plugins.d/<name>.yml`) is **curated** — the build
script only refreshes its `skills/` links and leaves the manifests,
assets, and marketplace entries hand-edited. No plugins currently use
this mode: [`nvidia-skills/`](./nvidia-skills/) is catalog-driven, built
from [`plugins.d/nvidia-skills.yml`](../plugins.d/nvidia-skills.yml), and
ships to the official Anthropic (Claude), OpenAI (Codex), and Cursor
plugin marketplaces.
