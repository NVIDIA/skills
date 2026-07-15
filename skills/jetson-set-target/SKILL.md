---
name: jetson-set-target
description: >-
  Switch the active Jetson target-platform pointer to an existing
  profile YAML. Use before customize/build/flash to change target; not
  for authoring profiles — use jetson-init-target instead.
version: 0.0.1
license: "Apache-2.0"
metadata:
  data-classification: public
  author: "Jetson Team"
  tags:
    - target-platform
    - configuration
    - meta
  domain: meta
---

# Set Target Platform

## Overview

This skill is the **switcher** side of the target-platform contract in
[`../../context/target-platform-contract.md`](../../context/target-platform-contract.md). It only
edits `target-platform/active_target.yml`; it never authors a new
profile YAML. To author a new profile, use the sibling skill
`jetson-init-target`.

Single output: `target-platform/active_target.yml` updated to point at
the user's chosen existing profile.

## When to invoke

- The user asks to switch / change / activate a different target
  platform among profiles that already exist.
- A downstream skill refused with "no active target" **and**
  `target-platform/` already contains one or more profile YAMLs.
- The user names a target by filename or product, and that profile
  YAML is already on disk.

If `target-platform/` is missing, empty, or contains only
`active_target.yml`, stop and tell the user to run `jetson-init-target`
first.

## Procedure

### Survey the profiles directory

1. Read `target-platform/active_target.yml` (treat a missing file as
   `active: NA` for the purposes of this skill — the file will be
   created in the "Update the active pointer" step if needed, preserving the template header).
2. List `target-platform/*.yaml`, **excluding** `active_target.yml`
   itself.

| Situation | Action |
|---|---|
| No profile YAMLs exist | Refuse. Tell the user to run `jetson-init-target` to author one. |
| Exactly one profile, already active | Tell the user it is already active; no change needed. Stop. |
| Exactly one profile, not active | Confirm with the user, then jump to the "Update the active pointer" step with that profile selected. |
| Multiple profiles | Proceed to the "Show the list" step. |

### Show the list

Print the profiles in alphabetical order, numbered. Mark the current
active selection with `(active)` so the user knows what they're
switching from. For each profile, parse its YAML and show a one-line
summary so the user can identify it without opening files: the
`reference_devkit.name` (or `custom_carrier.name` if present) plus the
`flash_config`.

Sample shape:

```
Profiles in target-platform/:
   1) acme-vision-x1.yaml          — Acme Vision X1 (custom carrier on AGX Thor T5000) — acme-vision-x1.conf
   2) jetson-agx-orin-32gb.yaml    — Jetson AGX Orin 32GB                              — jetson-agx-orin-devkit.conf
   3) jetson-agx-thor-devkit.yaml  — Jetson AGX Thor T5000                             — jetson-agx-thor-devkit.conf  (active)
   4) jetson-orin-nano-8gb.yaml    — Jetson Orin Nano 8GB                              — jetson-orin-nano-devkit.conf
```

If a profile YAML fails to parse or is missing the expected fields,
show the bare filename with a `(unparseable — fix or remove)` marker
and skip to the next; do not refuse the whole skill over one bad file.

### Prompt for selection

Prompt:

> Pick a number, the bare filename, the product name (substring
> match), or `cancel`.

Resolve to a single profile filename. If the user picks the currently
active profile, tell them so and stop without rewriting the pointer.
If ambiguous, list candidates and ask again.

### Update the active pointer

Edit `target-platform/active_target.yml`:

- Set `active: <chosen-filename>.yaml`.
- **Preserve** the header comment block verbatim — only modify the
  `active:` line.
- If the file does not exist (it can be missing if the user hand-set up
  `target-platform/` without the bootstrap step), create it from
  [`../../references/active_target_template.yaml`](../../references/active_target_template.yaml)
  and then set `active:`.

### Confirm

Print a summary: the previous active value, the new active value, the
chosen profile's `reference_devkit.name` (and `custom_carrier.name` if
present), and a reminder that downstream skills (`jetson-customize-clocks`,
`jetson-customize-nvpmodel`, `jetson-customize-fan`,
`jetson-optimize-memory`) will now resolve to this target.

If a downstream skill triggered this run, tell the user to re-issue
their original request; do not silently re-trigger it.

## Prerequisites

- `target-platform/` exists and contains at least one `*.yaml` profile
  (excluding `active_target.yml`).
- `target-platform/active_target.yml` is present, or the template at
  `../../references/active_target_template.yaml` is available so this
  skill can recreate the pointer.

## Limitations

- Edits only the `active:` line in `active_target.yml`; never writes
  profile YAMLs.
- Refuses when no profile YAMLs exist — routes the user to
  `jetson-init-target` instead.
- Does not resolve fuzzy product names beyond simple substring match;
  ambiguous selections are re-prompted.

## Troubleshooting

- **"no profiles found" refusal** — the directory has no
  `*.yaml` other than the pointer; run `/jetson-init-target` first.
- **Pointer rewritten to a missing profile** — should not happen because
  the chosen filename is verified before writing; if seen, hand-edit
  `active_target.yml` back to a valid filename or rerun with the right
  selection.
- **Active pointer reads `NA`** — legitimate "no selection" state; pick
  any listed profile and the pointer will be set.

## Gotchas

- **No authoring here.** If the user names a target whose profile YAML
  is not on disk, refuse and route them to `jetson-init-target`. Do
  not silently fall through to authoring.
- **Don't edit `active_target.yml` to point at a missing profile.**
  Verify the chosen filename exists in `target-platform/` before
  writing.
- **Preserve `active_target.yml`'s comment block** when editing —
  only the `active:` line should change.
- **Exclude `active_target.yml` from the profile list.** It lives in
  the same directory but is the pointer, not a profile.
- **`active: NA` is the legitimate "no selection" state.** Treat it
  the same as a fresh pointer — don't refuse, just show the list and
  let the user pick.
- **One-profile shortcut.** If only one profile exists, do not skip
  user confirmation — still confirm before flipping the pointer.

## References

- [`../../context/target-platform-contract.md`](../../context/target-platform-contract.md) — the contract this skill writes to.
- [`../../references/active_target_template.yaml`](../../references/active_target_template.yaml) — schema starter for the active-target pointer (used to recreate it if the file is missing).
- [`../jetson-init-target/SKILL.md`](../jetson-init-target/SKILL.md) — sibling skill: author a new profile YAML.
