---
name: jetson-link-docs
description: >-
  Bind pre-downloaded Jetson reference docs (developer guide, design
  guide, pinmux, schematics) into the active profile documents
  block. Use after staging docs on disk; not for downloading.
version: 0.0.1
license: "Apache-2.0"
metadata:
  data-classification: public
  author: "Jetson Team"
  tags:
    - target-platform
    - documents
    - setup
  domain: meta
---

# jetson-link-docs

## Overview

This skill **writes** the `documents:` block of the active Jetson /
IGX target-platform profile YAML so downstream skills
(`/jetson-generate-kb`, `/jetson-customize-pinmux`, camera / pcie /
uphy, etc.) can resolve doc paths by name. It walks the user through
every document slot in the profile schema, tries to auto-bind each
slot to a file under `<documents.root_path>/` via case-insensitive
glob matching, and writes the resulting paths back into the active
profile.

Scope is **registering pointers only** — this skill does **not** fetch
or download. The files must already exist on disk under
`<documents.root_path>/`.

## When to invoke

- After `/jetson-init-target` finishes and the user has documents
  on disk to register.
- The user wants to add, change, or remove document references on an
  existing profile.
- A downstream skill (e.g. `jetson-generate-kb`) reports "no documents
  recorded" and the user wants to fix that.

## Procedure

### Resolve the active target

Resolve the active profile + `<workspace>` per the contract in
[`../../context/target-platform-contract.md`](../../context/target-platform-contract.md).
Cache the loaded profile in memory — this skill mutates it in
the "Write the `documents:` block back to the profile" step.

### Load the document-slot schema

Load
[`../../references/platform_template.yaml`](../../references/platform_template.yaml).
Parse the `documents:` block. Each per-document field is marked
`<OPTIONAL: description>`. Use the marker description as prompt text
verbatim. Match markers with the regex
`^<(REQUIRED|OPTIONAL|DERIVED):\s*(.*)>$` after YAML parsing strips
surrounding quotes.

Skip `custom_carrier_schematic` and `custom_carrier_pinmux_xls`
entirely when the active profile has no `custom_carrier:` block —
both are meaningless without one. This filter applies through the "Scan and auto-match" and "Manual prompts for unmatched fields" steps.

### Resolve `documents.root_path`

Default: `<workspace>/Documents`. If the profile already records
`documents.root_path`, use it. Otherwise, if `<workspace>/Documents/`
exists, use it (the field is **omitted** from the written profile —
downstream skills fall back to the workspace default). If neither is
available, prompt the user for an absolute path, or accept Enter /
`cancel` to skip the auto-scan. A user-provided path that doesn't
exist is treated as skipped (warn, don't refuse — the field is
OPTIONAL); manual prompts in the "Manual prompts for unmatched fields" step still run.

### Resolve the product token

Read the **Product Token** column from
[`../../references/bsp-platforms-catalogue.md`](../../references/bsp-platforms-catalogue.md)
for the row matching `reference_devkit.name`. The token is a
case-insensitive glob fragment (e.g. `*orin*nano*`, `*agx*thor*`)
consumed by the fallback patterns in the "Scan and auto-match" step.

If `reference_devkit.name` has no row in the catalogue, log a warning
and proceed without a product-token fallback — the "Scan and auto-match" step still works
with strictly SKU-keyed matching.

For custom carriers, derive `<custom-token>` from `custom_carrier.name`
using this recipe: lowercase, replace each space with `*`, wrap in
`*` on both ends. E.g. "Acme Vision X1" → `*acme*vision*x1*`.

### Scan and auto-match

Skip this step entirely if `documents.root_path` did not resolve in
the "Resolve `documents.root_path`" step (no scan target → no auto-suggest; fall through to manual
prompts in the "Manual prompts for unmatched fields" step).

Scan the directory once (one level deep) and try to auto-match each
remaining `<OPTIONAL:…>` field using the case-insensitive globs below.
Use the lower-case `module.id` / `carrier.id` / `custom_carrier.id`
strings from the profile in the SKU column.

| Field | SKU glob (primary) | Product-token glob (fallback) |
|---|---|---|
| `bsp_developer_guide`         | `*developer*guide*.pdf`, `*BSP*guide*.pdf` | _(no fallback — pattern is product-agnostic)_ |
| `soc_tech_ref_manual`         | `*TRM*.pdf`, `*tech*ref*manual*.pdf` | _(no fallback — same)_ |
| `module_data_sheet`           | `*<module.id>*data*sheet*.pdf`, `*<module.id>*datasheet*.pdf` | `<token>data*sheet*.pdf`, `<token>datasheet*.pdf` |
| `module_design_guide`         | `*<module.id>*design*guide*.pdf`, `*<module.id>*PDG*.pdf` | `<token>design*guide*.pdf`, `<token>PDG*.pdf` |
| `module_thermal_design_guide` | `*<module.id>*thermal*.pdf` (covers "Thermal Design Guide" / "TDG") | `<token>thermal*.pdf` |
| `module_schematic`            | `*<module.id>*schem*.pdf` | `<token>schem*.pdf` |
| `carrier_board_spec`          | `*<carrier.id>*board*spec*.pdf`, `*<carrier.id>*spec*.pdf` | `<token>carrier*spec*.pdf` |
| `carrier_schematic`           | `*<carrier.id>*schem*.pdf` | `<token>carrier*schem*.pdf` |
| `custom_carrier_schematic`    | `*<custom_carrier.id>*schem*.pdf` (only if custom carrier) | `<custom-token>schem*.pdf` (only if custom carrier) |
| `ref_devkit_pinmux_xls`       | `*<carrier.id>*pinmux*.xls*` (matches `.xls`, `.xlsx`, `.xlsm`) | `<token>pinmux*.xls*` |
| `custom_carrier_pinmux_xls`   | `*<custom_carrier.id>*pinmux*.xls*` (only if custom carrier) | `<custom-token>pinmux*.xls*` (only if custom carrier) |

`<token>` is the catalogue-resolved product token; `<custom-token>`
is derived from `custom_carrier.name` per the "Resolve the product token" step. Tokens already
include leading/trailing `*`, so the table does not repeat them.

### Match policy per field

For each field that has auto-match results:

- Take the **union** of hits across the SKU glob and the product-
  token glob, then **deduplicate by absolute path** — a file matched
  by both globs counts once.
- **Exactly 1 unique hit** → show the path and prompt
  `use this? (yes/no, default yes)`. On `yes`, record it and **skip**
  the manual prompt for that field. On `no`, fall through to the
  manual prompt in the "Manual prompts for unmatched fields" step.
- **0 hits** → skip auto-suggest entirely for that field; fall
  through to the "Manual prompts for unmatched fields" step.
- **2+ unique hits** → present them as a numbered list in the "Manual prompts for unmatched fields" step so
  the user can pick by number rather than typing a path; include a
  `skip / NA` option. Never silently bind a multi-hit candidate.
- **Never silently bind** without user confirmation —
  wrong-schematic / wrong-pinmux bindings are real and costly.

If `documents.root_path` is folder-organised one level deeper than
flat (NVIDIA archives often are: `Schematics/`, `Design-Guides/`,
`Pinmux/`, etc.), the file globs may return zero hits even when the
right documents exist. v0.2 only scans one level deep — when 0 hits
is suspicious (`documents.root_path` exists but no fields auto-bound),
surface the limitation to the user and offer to fall through to
manual prompts.

### Manual prompts for unmatched fields

For every field that wasn't auto-bound (and wasn't filtered out in
the "Load the document-slot schema" step), prompt using the marker description from the "Load the document-slot schema" step as prompt
text, in document order. Accept Enter and `NA` interchangeably as
"skip this field". When the "Match policy per field" step produced 2+ candidate hits for a
field, present them as a numbered list with a `skip / NA` option
rather than asking for a free-text path.

Validate that user-provided paths exist on disk (warn if not, but do
not refuse — the user may be recording a planned path). URLs (values
starting with `http://`, `https://`, or `ftp://`) are accepted
verbatim and not validated.

### Write the `documents:` block back to the profile

Edit `target-platform/<active>.yaml` in place. Preserve all other
top-level blocks (`reference_devkit:`, `custom_carrier:`,
`bsp_image:`, `source:`) and their comments verbatim. Write only the
fields the user provided — omit skipped / `NA` fields entirely (no
`NA` placeholders, no empty keys).

Edge behavior: when every field was skipped (including
`documents.root_path`), drop the `documents:` block entirely from
the profile — never write `documents: {}` or a block of `NA` values.
When only `documents.root_path` was provided (no per-document
binding), record it alone — the path has value as a hint for future
re-runs. On re-run with an existing `documents:` block, merge:
existing bindings are preserved unless the user picks a new file or
`NA`; newly bound fields are added.

### Confirm

Print a summary:

- Profile path written.
- `documents.root_path` — resolved value (or "default — omitted").
- Auto-bound fields: count + per-field one-line list.
- Manually entered fields: count + list.
- Skipped fields: count.
- A reminder that `jetson-generate-kb` re-reads `documents.*` and
  should be re-run if a KB exists.

If a downstream skill triggered this run, tell the user to re-issue
their original request; do not silently re-trigger it.

## Gotchas

- **Active profile must exist.** This skill writes back to whichever
  profile is active. If no profile is active, refuse and route to
  `jetson-set-target` / `jetson-init-target`.
- **Product-token globs are intentionally broad.** A token like
  `*orin*nano*` matches both Orin-Nano-specific docs and combined
  Orin-NX/Nano docs (e.g. `Jetson-Orin-NX-Nano-Design-Guide_…`). That
  is usually correct for module-side docs (NVIDIA ships combined
  manuals), but verify on schematic / pinmux / spec fields where
  wrong-product binding is costly.
- **Update `bsp-platforms-catalogue.md` when adding new product
  rows.** The **Product Token** column is consumed by the "Resolve the product token" step; a
  missing token degrades the auto-scan to SKU-only matching (the
  skill warns and continues, but doc-rich `documents.root_path`
  scans will degrade silently from "5 auto-binds" to "fewer auto-
  binds").
- **Use a round-tripping YAML loader.** the "Write the `documents:` block back to the profile" step mutates an existing
  YAML file. Plain `yaml.safe_load` + `yaml.safe_dump` loses comments,
  block ordering, and quoting style — use `ruamel.yaml` or
  equivalent so hand-edited fields and comments survive.
- **Re-runnable.** Re-running merges new bindings; existing
  bindings are preserved unless the user explicitly changes them.
  Safe to invoke as part of a profile refresh.

## Prerequisites

- Active target profile resolved per
  `../../context/target-platform-contract.md`.
- Documents available either under the recorded `documents.root_path`,
  under default `<workspace>/Documents/`, or as user-provided paths /
  URLs during manual prompts. A missing root only disables auto-scan; it
  is not a hard prerequisite.
- `ruamel.yaml` or another round-tripping YAML writer for the profile
  edit step.

## Limitations

- Registers pointers only; never downloads, copies, or renames files.
- Per-document field set is fixed to the schema in
  `../../references/platform_template.yaml` — no ad-hoc keys.
- Glob matching is filename-only; bad filenames in
  `documents.root_path` will under-bind and require manual selection.

## Troubleshooting

- **`documents.root_path` missing** — auto-scan is skipped. Provide an
  absolute root path, enter individual document paths / URLs manually,
  or skip the fields you do not want to bind.
- **Multiple files match a single slot** — the skill stops and prompts;
  pick or rename the file. Example: two
  `Jetson-Linux-Developer-Guide*.pdf` files → keep the active version,
  rename the stale one.
- **Profile comments lost after write** — a non-round-tripping YAML
  writer was used; switch to `ruamel.yaml` and rerun against a fresh
  pristine copy.
- **Validation fails because a binding points outside
  `documents.root_path`** — `documents.*` are relative paths only; move
  the file under the root and retry.

## References

- [`../../context/target-platform-contract.md`](../../context/target-platform-contract.md) — target-platform contract; this skill consumes and mutates the active profile.
- [`../../references/bsp-platforms-catalogue.md`](../../references/bsp-platforms-catalogue.md) — source of the **Product Token** column for the "Resolve the product token" step.
- [`../../references/platform_template.yaml`](../../references/platform_template.yaml) — schema for the `documents:` block (source of truth for prompts and field list).
- [`../jetson-init-target/SKILL.md`](../jetson-init-target/SKILL.md) — sibling skill that authors target identity (`reference_devkit:`, optional `custom_carrier:`).
- [`../jetson-init-image/SKILL.md`](../jetson-init-image/SKILL.md) — sibling skill that authors `bsp_image:`.
- [`../jetson-init-source/SKILL.md`](../jetson-init-source/SKILL.md) — sibling skill: clones shared repos and handles `source.root_path` overrides.
- [`../jetson-generate-kb/SKILL.md`](../jetson-generate-kb/SKILL.md) — sibling skill: consumes the `documents:` block this skill writes.
