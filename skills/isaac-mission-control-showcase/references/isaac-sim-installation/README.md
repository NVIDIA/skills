# Isaac Sim installation handoff

Parent-owned routing contract. This is **not** `isaac-sim-installation` and
does not document how to install Isaac Sim. That skill is owned by its own
repository and its runtime `SKILL.md` is the authority.

There is deliberately no adapter here. Installation is an interactive,
gated operation that the agent performs by following the upstream skill
directly; the showcase runner never invokes it.

## When to use

Only when host inspection finds no usable local Isaac Sim, or when the user
explicitly asks to install, upgrade, or downgrade one. Never as showcase
recovery.

## Showcase inputs and outputs

**Requires:** a local standalone installation; preselect that method.
**Produces:** an installed path, reported back. Isaac Sim is **not** launched.

## Integration constraints

- Stop the showcase workflow before handing off. Resume only when the user asks
  to continue.
- Never bypass the upstream's method-confirmation, compatibility, licensing, or
  execution-approval gates.
- Never modify, repair, relabel, or overwrite an existing installation as
  recovery for a showcase failure.

## Who owns the behavior

`isaac-sim-installation`, resolved at runtime from the pinned checkout.

## Finding it

The dependency manifest records its directory and runtime `SKILL.md` at
`skills.isaac-sim-installation.skill_dir` and `.skill_md`. Read that file
completely and follow it. Do not rely on any summary here.

## Blockers and return behavior

If the manifest does not report this component `ready`, stop and report the
blocker. Do not search the machine for an installer and do not vendor one.

## Retained by the parent

Runtime selection, version-compatibility prompting, the prohibition on
modifying an existing installation, and the decision to resume.
