# NVIDIA Skills Plugin

This plugin ships one skill, `nvidia-skill-finder`. It checks NVIDIA's live skills catalog, recommends the best match for the task, and gives an install command and a useful first prompt. Shipping only the finder keeps the plugin small: every skill in the `skills/` catalog stays reachable without loading each `SKILL.md` into the agent's context on install.

## Availability

This plugin is available as a first-party plugin in the Claude Code, Codex, and Cursor marketplaces. Hermes support is coming soon.

You can also get it by adding the self-hosted `nvidia-official` marketplace from the `NVIDIA/skills` repo.

## What the finder runs

- **Catalog lookup.** The finder runs `npx skills add nvidia/skills --list` to read the live catalog, which needs Node.js and downloads the `skills` CLI from npm. If the CLI is unavailable, it reads the catalog from `github.com/NVIDIA/skills`, `build.nvidia.com/skills`, or `raw.githubusercontent.com` instead.
- **Installs only with approval.** It never installs without asking. When the user agrees, it runs `npx skills add nvidia/skills --skill <name> --global --yes`, which installs that skill from the `main` branch of `NVIDIA/skills` into the agent's global skills directory.
- **Wider search.** If nothing in the NVIDIA catalog matches, it may suggest `npx skills find <query>`, which searches skills outside NVIDIA's catalog.

The plugin itself contains no code, stores no credentials, and sends no telemetry. The `skills` CLI follows its own policies.

## How this folder is built

Everything here except `assets/` (the logo, and a 2:1 `banner.png` for the Hermes plugin catalog card) and this README is generated from `plugins.d/nvidia-skills.yml` by `.github/scripts/build-plugins.sh`:

- `skills/` — copied from the canonical `skills/` catalog
- `.claude-plugin/plugin.json`, `.codex-plugin/plugin.json`, `.cursor-plugin/plugin.json` — per-agent manifests
- `plugin.json` — portable [Agent Plugins 1.0.0](https://agent-plugins.org/schemas/1.0.0/plugin.schema.json) manifest, read by Agent Plugins compatible clients such as Hermes
