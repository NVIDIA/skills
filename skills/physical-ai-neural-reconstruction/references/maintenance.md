# Keeping This Router Up to Date

The upstream `nurec-index` skill (at
<https://github.com/NVIDIA/nurec-skills/blob/main/skills/nurec-index/SKILL.md>)
is hand-curated by the NRS team. When it adds or restructures
sibling skills:

1. Add a row to `Pick a skill` in `SKILL.md` for any new use case.
2. Add a row to `Sibling skills (upstream)` in `SKILL.md`.
3. If the new skill changes a multi-step pipeline, update
   `references/workflows.md` — keep the workflow letters aligned with
   the upstream `nurec-index` `references/workflows.md` (currently
   A–G) so cross-references stay usable.
4. Re-verify the upstream URLs, container names, and release pins
   still match each sibling's frontmatter `metadata:` block:
   - `ncore` — <https://github.com/NVIDIA/ncore>, release `2026.04`
   - `nre` — `nvcr.io/nvidia/nre/nre-ga` +
     `nvcr.io/nvidia/nre/nre-tools-ga`, NRE `release_26.04`
   - `asset-harvester` — <https://github.com/NVIDIA/asset-harvester>,
     `nvidia/asset-harvester` on Hugging Face
   - `nurec-fixer` — <https://github.com/NVIDIA/harmonizer>,
     `nvidia/Harmonizer`, base image
     `nvcr.io/nvidia/pytorch:25.10-py3`
   - `physical-ai-datasets` — `nvidia/PhysicalAI-*` on Hugging Face
5. If the upstream renames a sibling skill (e.g.
   `ncore-data-conversion` → `ncore`), a container channel (`nre` →
   `nre-ga`), or a model repo (`nvidia/DiffusionHarmonizer` →
   `nvidia/Harmonizer`), search this skill for the old name and
   update every occurrence — the picker table, workflow steps,
   sibling skills table, mix-ups, hard rules, and troubleshooting.
6. Check whether the upstream layout still roots skills at
   `skills/<name>/SKILL.md` (with `.agents/skills` as a symlink); if
   it moves, update `metadata.upstream` and
   `references/upstream-fetch.md`.

Treat the upstream `nurec-index` as authoritative; this skill mirrors
only the picker tables, the workflow ordering, and the upstream fetch
recipe.
