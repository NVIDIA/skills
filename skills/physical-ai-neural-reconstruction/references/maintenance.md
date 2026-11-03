# Keeping This Router Up to Date

This catalog router is hand-curated. When a sibling skill is added
or restructured in its product repo:

1. Add a row to `Pick a skill` in `SKILL.md` for any new use case.
2. Add a row to `Sibling skills (upstream)` in `SKILL.md`.
3. If the new skill changes a multi-step pipeline, update
   `references/workflows.md` — keep the workflow letters aligned with
   this router's `references/workflows.md` (currently A–G) so
   cross-references stay usable.
4. Re-verify the upstream URLs, container names, and release pins
   still match each sibling's frontmatter `metadata:` block:
   - `ncore-data-conversion` — <https://github.com/NVIDIA/ncore>,
     `skills/ncore-data-conversion/`
   - `nurec` — <https://github.com/NVIDIA/nurec>, `skills/nurec/`;
     containers `nvcr.io/nvidia/nre/nre-ga` +
     `nvcr.io/nvidia/nre/nre-tools-ga`, NRE 26.04 (image tags `26.04.01` / `26.04` / `latest`;
     the release name `release_26.04` is not an image tag)
   - `asset-harvester` — <https://github.com/NVIDIA/asset-harvester>,
     `nvidia/asset-harvester` on Hugging Face. The **skill itself**
     ships from that repo (`skills/asset-harvester/`).
   - `harmonizer` — <https://github.com/NVIDIA/harmonizer>,
     `skills/harmonizer/`, `nvidia/Harmonizer`, base image
     `nvcr.io/nvidia/pytorch:25.10-py3`
   - `physical-ai-datasets` — <https://github.com/NVIDIA/nurec>,
     `skills/physical-ai-datasets/`; `nvidia/PhysicalAI-*` on Hugging Face
5. If a sibling skill is renamed (e.g. `ncore` →
   `ncore-data-conversion`), a container channel (`nre` →
   `nre-ga`), or a model repo (`nvidia/DiffusionHarmonizer` →
   `nvidia/Harmonizer`), search this skill for the old name and
   update every occurrence — the picker table, workflow steps,
   sibling skills table, mix-ups, hard rules, and troubleshooting.
6. Check whether each sibling still lives at
   `skills/<name>/SKILL.md` in its product repo. Nested entrypoints
   (for example a future onboard skill under
   `skills/<name>/<name>/SKILL.md`) must be recorded as the
   `skill_path`, not assumed from the outer folder. If a layout
   moves, update `metadata.upstream.sibling_skills` and
   `references/upstream-fetch.md`.

This catalog router is authoritative **for the routing taxonomy and
workflow ordering**; it mirrors only the picker tables, the workflow
ordering, and the product-repo identities. It is not
authoritative for `asset-harvester`, which is maintained in
<https://github.com/NVIDIA/asset-harvester>.
