# Locating and Fetching Upstream Skills

The canonical NuRec router (named `nurec-index`) and its five sibling
skills live in `https://github.com/NVIDIA/nurec-skills` under
`skills/<name>/SKILL.md`. The repo also exposes `.agents/skills` as a
symlink onto `skills/`, so both paths resolve to the same tree. Refer
to a sibling skill by its `name:` (e.g. `nre`) — that name is portable
across agent runtimes that implement the `agentskills.io` standard.
The folder name always matches the skill `name:` (e.g. the `ncore`
skill lives at `skills/ncore/`).

## Where to look on the local disk (try in order)

1. `.agents/skills/<name>/SKILL.md` (Cursor, Codex, NemoClaw)
2. `.claude/skills/<name>/SKILL.md` (Claude Code)
3. `.cursor/skills/<name>/SKILL.md` (project-scoped)
4. `~/.cursor/skills/<name>/SKILL.md` (personal skills)
5. An existing `nurec-skills` clone under the shared upstream root.

## Ask before cloning

> **Do not auto-clone.** The step below performs a **network fetch**
> of an external GitHub repository **and writes to the local
> filesystem**. That can violate organizational network/security
> policy and exposes the user to supply-chain risk if the upstream is
> ever tampered with. Show the user the exact command and get explicit
> consent (e.g. "OK to `git clone
> https://github.com/NVIDIA/nurec-skills` into `<DIR>`?") **before**
> running it. Prefer a pinned tag or SHA (`--branch <tag-or-sha>`)
> over `HEAD`, prefer fetching only the needed `SKILL.md` when the
> layout allows it, and never silently default to `/tmp`.
>
> If the user declines, stop and report which sibling skill is
> missing. Do not fall back to silent network access.

## Clone or refresh the upstream

Use the shared upstream root unless the user has set a NuRec-specific
override:

```bash
UPSTREAM_ROOT="${NUREC_SKILLS_UPSTREAM_ROOT:-${PHYSICAL_AI_SKILL_HUB_UPSTREAM_ROOT:-$HOME/.physical-ai-skill-hub/upstreams}}"
mkdir -p "$UPSTREAM_ROOT"
if [ -d "$UPSTREAM_ROOT/nurec-skills/.git" ]; then
  git -C "$UPSTREAM_ROOT/nurec-skills" fetch --tags
  git -C "$UPSTREAM_ROOT/nurec-skills" checkout main
  git -C "$UPSTREAM_ROOT/nurec-skills" pull --ff-only
else
  # Only after the user has agreed to the clone.
  git clone --depth 1 https://github.com/NVIDIA/nurec-skills.git \
    "$UPSTREAM_ROOT/nurec-skills"
fi
test -f "$UPSTREAM_ROOT/nurec-skills/skills/nurec-index/SKILL.md"
```

Then read the upstream skill before running any mutating command:

```bash
# Upstream router (table of contents), name: nurec-index
cat "$UPSTREAM_ROOT/nurec-skills/skills/nurec-index/SKILL.md"

# Sibling skills (replace <folder> per the table in SKILL.md):
cat "$UPSTREAM_ROOT/nurec-skills/skills/<folder>/SKILL.md"
```

Skills that pin a specific upstream commit ship the actual file under
`skills/<folder>/_versions/<branch>/<commit>/SKILL.md` with a
top-level `<folder>/SKILL.md` symlink to the currently-selected
version. Follow the symlink; don't hand-pick a `_versions/` path
unless the user asked for a specific revision.

Companion files (`references/`, `scripts/`, `assets/`) live next to
**the sibling skill's** `SKILL.md`, not next to this router. Open the
sibling skill first and follow its References section.
