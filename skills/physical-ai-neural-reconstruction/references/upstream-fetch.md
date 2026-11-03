# Locating and Fetching Upstream Skills

Each sibling skill ships from its own product repo. Use the
**Skill source** column in `SKILL.md` (and the matching
`metadata.upstream.sibling_skills` entries) for the registered
`name`, `skill_repo`, and `skill_path`. Companion files
(`references/`, `scripts/`, `assets/`) live next to that sibling's
`SKILL.md`, not next to this router.

There is no shared product-repo namespace and no upstream index to
fetch. This catalog router owns its picker tables and workflow
letters.

## `asset-harvester` and `physical-ai-datasets` — canonical product source

Those two skills kept their names. A generic same-name local hit may
be the retired bundled copy. Use their product-repo homes:

- [`NVIDIA/asset-harvester`](https://github.com/NVIDIA/asset-harvester)
  under `skills/asset-harvester/`
- [`NVIDIA/nurec`](https://github.com/NVIDIA/nurec) under
  `skills/physical-ai-datasets/`

A canonical definition already acquired for this task can be reused.

## Where to look on the local disk

For `ncore-data-conversion`, `nurec`, and `harmonizer`, try in this
order. A matching installed definition can be read locally without a
mandatory network refresh. If it is a symlink, read the selected
definition and its own companion files. Do not search broad developer
workspaces.

1. `.agents/skills/<name>/SKILL.md` (Cursor, Codex, NemoClaw)
2. `.claude/skills/<name>/SKILL.md` (Claude Code)
3. `.cursor/skills/<name>/SKILL.md` (project-scoped)
4. `~/.cursor/skills/<name>/SKILL.md` (personal skills)

This local eligibility does **not** apply to `physical-ai-datasets`
or `asset-harvester` — see the previous section.

Do not read sibling skills from a `nurec-skills` checkout: those
copies are retired and may still exist. Use each skill's product-repo
home. In particular, `physical-ai-datasets` and `asset-harvester`
kept their names, so a generic same-name local hit may be the old
copy. Renamed skills may resolve locally under their current names;
never fall back to the retired `ncore`, `nre` or `nurec-fixer`
skill folders.

Never accept a definition known to originate from the retired bundle
merely because someone renamed its local directory.

## Ask before cloning

> Before cloning a product repo or fetching an update to obtain a
> missing skill, show the user the selected repo, the skill path and
> the exact retrieval action and local destination. Get explicit
> consent before that network fetch and filesystem write. If consent
> is declined, report which skill is unavailable; do not silently
> fetch or fall back to the retired copy.

This does not require a new approval to read an already usable local
renamed skill, and it does not redefine ordinary read-only
documentation browsing as installation. The router supplies no
executable retrieval command, default directory, branch-switching
sequence or cache-refresh policy. Do not reuse an old clone or
change a user's current branch as an implied part of lookup.

If the selected current definition is unavailable after local lookup,
follow this consent step before obtaining it from the mapped product
repo. Missing a renamed local skill never falls back to an old folder
or clone.

Read the selected definition before running any mutating command.
Open its own References section on demand.
