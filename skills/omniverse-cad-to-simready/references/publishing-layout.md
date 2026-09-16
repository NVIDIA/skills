# CAD to SimReady Publishing Layout Notes

Background on this skill's own file layout and frontmatter conventions. Read
this reference only when changing the skill's structure, frontmatter, or
publishing story; it is not needed to execute the workflow.

## Source of Truth and Compatibility Aliases

Use `skills/omniverse-cad-to-simready/` as the source of truth for this
product repo's skill. The `.agents/skills` symlink is a compatibility alias
for local agentskills.io-style discovery, and `.codex/skills` and
`.claude/skills` are agent-specific compatibility aliases. Edit the
`skills/` copy; the symlinked paths are not independent copies.

## Frontmatter Field Placement

Frontmatter keeps `version` and `tools` at top level for agentskills.io
runtime compatibility. NVCARPS discoverability fields live under `metadata`
(`metadata.author`, `metadata.tags`, `metadata.domain`,
`metadata.languages`) rather than at top level, so this skill satisfies both
the agentskills.io runtime contract and the repo's own catalog metadata
checks at once.

## Nested References Tree

The nested `references/` tree under this workflow is intentional. It keeps
one public catalog skill while retaining script-bearing atomic stage
references, upstream handoff notes, and router documentation (including this
file and `references/troubleshooting.md`) under the workflow. Do not flatten
those references or promote nested README references to sibling `SKILL.md`
files unless the repo's publishing model changes.
