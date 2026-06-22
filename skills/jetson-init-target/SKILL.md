---
name: jetson-init-target
description: >-
  Author a new Jetson target-platform profile (reference_devkit +
  optional custom_carrier) and update the active pointer. Use to
  create a target; not for switching existing profiles.
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

# Initialize Target Platform

## Overview

This skill **authors** a new target-platform profile YAML — the
write-side of the contract in
[`../../context/target-platform-contract.md`](../../context/target-platform-contract.md). It uses
[`../../references/bsp-platforms-catalogue.md`](../../references/bsp-platforms-catalogue.md)
as the canonical product / flash-config catalogue and
[`../../references/platform_template.yaml`](../../references/platform_template.yaml)
as the field schema.

Flow is linear: pick a reference product + flash config, then
**optionally** add custom-carrier details. The two contract cases —
reference-only vs reference + custom carrier — are distinguished only
by whether the user provides custom-carrier details; there is no
up-front mode prompt.

**Scope is `reference_devkit:` and `custom_carrier:` only.** Sibling
skills own the other blocks: `bsp_image:` →
[`jetson-init-image`](../jetson-init-image/SKILL.md); `source:` →
[`jetson-init-source`](../jetson-init-source/SKILL.md); `documents:` →
[`jetson-link-docs`](../jetson-link-docs/SKILL.md).

Outputs: a new `target-platform/<name>.yaml` and (by default) an
updated `active_target.yml` pointing at it. To switch among
already-authored profiles, use sibling `jetson-set-target`.

## When to invoke

- The user asks to create / author / add a new target platform profile.
- A downstream skill refused with "no active target" **and**
  `target-platform/` contains no profile YAML files yet (otherwise
  prefer `jetson-set-target`).
- The user wants a target whose profile does not yet exist on disk.

## Procedure

### Quick-start prefill mapping

Follow the shared
[`quick_start_prefill` contract](../../context/bsp-customization-workflow.md#quick_start_prefill-contract).
This skill consumes:

- `quick_start_prefill.target.active_platform` as the selected catalogue
  row when it resolves to exactly one product.
- `quick_start_prefill.target.custom_carrier` as the custom-carrier
  intent.

If `active_platform` resolves to one catalogue row, use it directly. Do
not re-ask reference-product or flash-config questions; use the row's
default flash config unless the prefill names a catalogue variant. Ask
only when the prefill is missing, invalid, ambiguous, or names an unknown
variant.

### How the skill consumes the template

Load
[`../../references/platform_template.yaml`](../../references/platform_template.yaml)
once at skill startup and treat it as the schema for prompted blocks.
Each placeholder value carries one of three sentinel markers:

| Marker | Behavior |
|---|---|
| `<REQUIRED: description>` | Prompt the user. Value must be provided; `NA` is accepted with a warning that downstream skills may refuse. |
| `<OPTIONAL: description>` | Prompt the user. Value may be skipped (Enter or `NA`); skipped fields are **omitted** from the written profile (no empty key, no `NA` placeholder). |
| `<DERIVED: description>` | Filled programmatically from `bsp-platforms-catalogue.md` and the user's product / flash-config choice in the "Pick the reference product" and "Pick the flash config" steps. **Not prompted.** |

Use the marker description as the prompt text verbatim. Match markers
with the regex `^<(REQUIRED|OPTIONAL|DERIVED):\s*(.*)>$` after YAML
parsing strips the surrounding quotes.

Block-level inclusion (`custom_carrier`) is decided by the skill
flow, not by markers — adding/removing fields **inside** a block is
a template-only change.

### Survey current state

1. If `target-platform/` does not exist, create it and copy
   [`../../references/active_target_template.yaml`](../../references/active_target_template.yaml)
   to `target-platform/active_target.yml` (note `.yml` extension).
2. Read `target-platform/active_target.yml`.
3. List `target-platform/*.yaml` (excluding `active_target.yml`).

| Situation | Action |
|---|---|
| No profile YAMLs exist | Proceed — this is the bootstrap case. The new profile will be set as `active_target` automatically in the "Write" step (no activation prompt in the "Confirm" step). |
| Profiles exist, `active: NA` | Show the list of existing profiles and confirm the user wants to author a new one (rather than activate one of the existing profiles via `jetson-set-target`). |
| Profiles exist, `active:` names a present file | Show the active profile and the full list. Ask whether the user wants to (a) author a new profile alongside, or (b) switch to one of the existing profiles via `jetson-set-target`. If (b), stop and tell the user to run `jetson-set-target`. |
| `active:` names a missing file | Tell the user the pointer is broken; offer to author a new profile (and replace the pointer) or to fix the pointer manually. |

### Read the platforms catalogue

Read [`../../references/bsp-platforms-catalogue.md`](../../references/bsp-platforms-catalogue.md).
Parse the **"Product / Chip / SKU / Flash Config Mapping"** table — that
is the source of truth for product names, CVM/CVB SKUs, default flash
confs, and variant suffixes. Do **not** scan `Linux_for_Tegra/*.conf`
or parse conf filenames; the catalogue already collapses that.

### Pick the reference product (baseline)

Print the full catalogue list before asking. Group rows by chip family
and number every entry with stable indexes for this run. Build the list
only from `bsp-platforms-catalogue.md`; do not copy platform rows from
examples or other docs.

Build explicit shortcut choices from the printed list by taking the first
row in each chip-family group. Each shortcut label must include the
full-list index and product name, for example `Index 1 — Jetson AGX Orin
64GB`. Do not hard-code product names; derive shortcuts from the parsed
catalogue rows. These shortcuts are the only explicit choices for this
question. Manual index entry must use the tool's built-in `Type something`
input; do not add a third custom choice for manual entry.

Prompt text:

> Pick a platform shortcut, or choose Type something and type the numeric
> index from the full platform list printed above. If your hardware is a
> custom carrier with one of these Jetson modules, pick the closest
> matching reference devkit as the baseline. Type cancel to cancel this
> run.

Do not show chip-family choices such as `T234 / Orin family` or
`T264 / Thor family`. Do not show `Type index`, `Manual entry`, `Other
platform`, or any custom manual-entry choice. The only manual-entry path
is the tool-provided `Type something` row. Do not add `Other` as a choice
or paraphrase `Type something` as `Other` in prompt text. Do not accept
product-name substring matches in this step; typed input must be a numeric
index from the printed full list or `cancel`. `skip` is not allowed
because this skill cannot author a target profile without a reference
platform.

Resolve the shortcut or typed index to exactly one catalogue row. If the
typed index is out of range or invalid, show the valid index range and ask
again. If the user types `skip`, explain that `jetson-init-target`
requires a reference platform and ask again. If the user types `cancel`,
stop without writing a profile.

### Pick the flash config

From the catalogue row, the **Default flash conf** is option `1`; the
**Variants** column lists suffixes — option `2..N`. Reconstruct each
variant's full conf filename by appending the suffix to the default
conf base. If a row says `(same as <other>)` in the Variants column,
reuse the referenced row's variant list verbatim. Build this list only
from the selected catalogue row.

Prompt:

> Pick a number, or press Enter to accept the default (`1`).

If the chosen row has no variants (`—` in the catalogue), proceed to
the "Optional: add custom carrier details" step with the default conf.

**Edge cases:**

- **Raw-conf variants** (catalogue marks `(raw)`). For Orin Nano,
  `-a0-maxn (raw)` → `p3768-0000-p3767-0000-a0-maxn.conf`. Use the raw
  filename verbatim.
- **`(same as <other>)` placeholders.** Reuse the referenced row's
  variants verbatim.

### Optional: add custom carrier details

After the baseline + flash config are chosen, resolve whether the profile
should record a custom carrier on top. If
`quick_start_prefill.target.custom_carrier` is a valid concrete intent,
consume it directly:

- `add_custom_carrier` → record `custom_carrier:` and collect the missing
  carrier metadata below.
- `no_custom_carrier` → omit `custom_carrier:` and proceed to the
  "Confirm" step.
- `keep_existing_custom_carrier` is valid only when editing an existing
  active profile that already has `custom_carrier:`. Because this skill
  authors a new profile, treat it as ambiguous and ask the custom-carrier
  intent question below.
- `skip` or a missing prefill value → ask the custom-carrier intent
  question below.

When intent is not already concrete, ask the user whether they also want
to record a custom carrier on top using the `AskUserQuestions` UI. This
is a bounded choice; show `add` and `skip` only:

> Optionally add custom-carrier details for a customer-designed carrier
> using this Jetson module? (`add` / `skip`, default `skip`)

If the resolved intent is `skip`, proceed to the "Confirm" step with the
`custom_carrier:` block omitted.

If the resolved intent is `add`, collect every still-missing
`custom_carrier:` value through the `AskUserQuestions` UI. Do not collect
custom-carrier metadata with plain-text prompts, inline chat questions,
freeform summary confirmation, inferred defaults, or example values
presented as selectable options. Split the collection into multiple
`AskUserQuestions` forms if the UI question limit would otherwise be
exceeded.

For the first `AskUserQuestions` form, ask `custom_carrier.name`,
`custom_carrier.id`, `custom_carrier.sku`, and
`custom_carrier.revision` in template document order. For these four
fields, the only selectable path shown to the user must be the tool's
built-in `Type something` row. Do not add example values, "Other",
guessed board IDs, guessed SKUs, common revision strings, or any explicit
domain options. These are customer-provided metadata: preserve them
exactly as entered and do not require or validate any naming convention,
prefix, case, character set, digit count, or SKU format. Validate only
that required fields are non-empty unless the user explicitly enters
`NA`. `NA` is accepted per the contract for required fields; warn that
downstream skills may refuse if a missing field is required for their
edit.

After `custom_carrier.name` is known, collect
`custom_carrier.flash_config` through a second `AskUserQuestions` form if
it is still missing. Show the suggested default as the only explicit
choice and allow the built-in `Type something` row for a custom `.conf`
filename. Require a non-empty filename ending in `.conf`; do not derive
it from board ID, SKU, or any guessed convention.

Field-specific behavior (defaults / suggestions not expressible via
markers alone):

- `custom_carrier.flash_config` — suggest a default of kebab-cased
  `custom_carrier.name + ".conf"` (e.g. "Acme Vision X1" →
  `acme-vision-x1.conf`).
- `custom_carrier.name` — prompt: "Type the customer-facing custom
  carrier or product name exactly as you want it recorded. No format is
  required." Use `Type something` only.
- `custom_carrier.id` — prompt: "Type the custom carrier board ID or
  project ID exactly as you want it recorded. No format is required."
  Use `Type something` only.
- `custom_carrier.sku` — prompt: "Type the custom carrier SKU or variant
  identifier exactly as you want it recorded. No format is required."
  Use `Type something` only; quote in the written YAML so numeric-looking
  values and leading zeros survive.
- `custom_carrier.revision` — prompt: "Type the custom carrier revision
  or build identifier exactly as you want it recorded, or press Enter to
  omit it. No format is required." Use `Type something` only; omit when
  skipped.

If the user does not know a value, suggest reading
`/proc/device-tree/chosen/ids` on the booted target. If genuinely
unavailable, accept `NA` for required fields or omit optional fields.

The template intentionally omits `custom_carrier.module` and
`custom_carrier.chip_family` — module identity comes from
`reference_devkit.module` (the same physical Jetson module plugs into
both carriers). Don't add them back.

### Confirm

Show the proposed YAML. Shape depends on whether the user added
custom-carrier details in the "Optional: add custom carrier details"
step. **Example:** see
[`references/ui-samples.md`](references/ui-samples.md) for the
reference-only and reference + custom-carrier YAML shapes, plus the
note on which sibling skill appends which block.

Suggested filename:

- **Reference-only:** kebab-cased product name from the selected
  catalogue row, with `.yaml` appended. E.g. `Jetson AGX Thor T5000` →
  `jetson-agx-thor-t5000.yaml`; `Jetson AGX Orin 32GB` →
  `jetson-agx-orin-32gb.yaml`. Do not derive the filename from
  `flash_config`; multiple products can share one default conf.
- **With custom carrier:** kebab-cased `custom_carrier.name`. E.g.
  "Acme Vision X1" → `acme-vision-x1.yaml`.

Use the suggested filename without asking when it is unique and was
generated from the selected platform or custom-carrier name. Print the
filename and continue. Ask only on collision, empty generated name,
ambiguous input, or explicit custom-filename request. Then handle
activation:

- **Bootstrap case** (no profile YAMLs existed at the "Survey current state" step): **do not
  prompt.** The new profile is the first one, so it is auto-activated
  in the "Write" step. Tell the user this is happening so the auto-activation is
  not silent.
- **Profiles already exist**: ask whether to **activate** the new
  profile (default: yes). Decline only if the user is authoring
  multiple profiles to choose from later — they can activate any of
  them with `jetson-set-target`.

### Write (and optionally activate)

1. Write `target-platform/<filename>.yaml`. If the file already
   exists, ask before overwriting; never silently clobber.
2. Edit `target-platform/active_target.yml` to set
   `active: <filename>.yaml` when **either** of the following holds:
   - This was the bootstrap case from the "Survey current state" step (no profile YAMLs existed
     before this run) — activation is automatic, regardless of any
     prior user input.
   - Profiles already existed and the user accepted activation in
     the "Confirm" step.

   Preserve the header comment block; only modify the `active:` line.
   If profiles already existed and the user declined activation, leave
   `active_target.yml` untouched.

### Confirm

Print a summary: profile path, active pointer (or "unchanged" if
activation was declined), Tegra family (T234 / T264), whether
`custom_carrier:` was recorded, and a reminder that downstream skills
will resolve to whichever profile is active. If the user authored
multiple profiles without activating, point them at `jetson-set-target`
to pick one.

**Next steps** (Setup → Customize chain):

1. `/jetson-init-image` — extract the BSP and append `bsp_image:`.
2. `/jetson-init-source` — clone the shared repos (and optionally
   override `<workspace>/Source` if the user wants `source.root_path`
   written into the profile).
3. `/jetson-link-docs` *(optional)* — register paths to
   pre-downloaded BSP / schematic / pinmux documents in the profile's
   `documents:` block.

Then, before any `customize-*` skill (Customize):

4. `/jetson-generate-kb` — KB that customize-* skills consult for
   target-specific file locations.
5. `/jetson-derive-carrier` *(custom carriers only)* — materialize the
   custom carrier's flash conf + supporting DTSIs from the reference
   devkit. Required before per-knob customize-* on a custom carrier.

If a downstream skill triggered this run, tell the user to re-issue
their original request; do not silently re-trigger it.

## Prerequisites

- `bsp-platforms-catalogue.md` and `platform_template.yaml` reachable
  via the relative paths in the References section.
- Write access to `target-platform/` (and `target-platform/active_target.yml`
  if activation is requested).
- For "what module/carrier is this?" lookups: `/proc/device-tree/chosen/ids`
  on the booted device.

## Limitations

- Owns only `reference_devkit:`, `custom_carrier:`, and the
  `active_target.yml` pointer; refuses to write `bsp_image:`,
  `source:`, or `documents:`.
- Will not invent SKUs, module IDs, custom carrier metadata, or flash
  configs — records `NA` or aborts when the user does not know a
  required value.

## Troubleshooting

- **User asks to switch instead of author** — refuse and route to
  `jetson-set-target`; do not re-author an existing profile.
- **Catalogue row not found for the product the user named** —
  add the row to `bsp-platforms-catalogue.md` first; do not patch the
  product list inside this skill.
- **Required field stuck at `NA`** — downstream `customize-*` skills
  may refuse; capture the missing value (read
  `/proc/device-tree/chosen/ids`) and rerun.
- **`active_target.yml` lost its comment block after activation** — a
  non-round-tripping YAML writer was used; switch to `ruamel.yaml` and
  restore from the template.

See [`references/gotchas.md`](references/gotchas.md) for the full
pitfalls list (catalogue/template separation, NA handling, conf
collision rules, etc.).

## References

- [`../../references/bsp-platforms-catalogue.md`](../../references/bsp-platforms-catalogue.md) — product / flash conf catalogue (the source of truth for the "Pick the reference product" and "Pick the flash config" steps).
- [`../../context/target-platform-contract.md`](../../context/target-platform-contract.md) — the contract this skill writes to.
- [`../../references/platform_template.yaml`](../../references/platform_template.yaml) — schema starter for a target profile (the source of truth for the "Optional: add custom carrier details" step).
- [`../../references/active_target_template.yaml`](../../references/active_target_template.yaml) — schema starter for the active-target pointer.
- [`../jetson-set-target/SKILL.md`](../jetson-set-target/SKILL.md) — sibling skill: switch the active pointer among already-authored profiles.
- [`../jetson-init-image/SKILL.md`](../jetson-init-image/SKILL.md) — sibling skill: extract BSP image and author the `bsp_image:` block.
- [`../jetson-init-source/SKILL.md`](../jetson-init-source/SKILL.md) — sibling skill: clone shared repos and (optionally) author the `source:` block.
- [`../jetson-link-docs/SKILL.md`](../jetson-link-docs/SKILL.md) — sibling skill: author the `documents:` block by registering pre-downloaded files.
