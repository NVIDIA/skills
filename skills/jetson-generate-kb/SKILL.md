---
name: jetson-generate-kb
description: >-
  Build a per-target knowledge-base markdown next to the active
  profile by walking the BSP root and source tree. Use after
  init-image / init-source; not for editing profile fields.
version: 0.0.1
license: "Apache-2.0"
metadata:
  data-classification: public
  author: "Jetson Team"
  tags:
    - target-platform
    - knowledge-base
    - documentation
    - meta
  domain: meta
---

# Generate Target Knowledge Base

## Overview

This skill produces a **per-profile** markdown reference at
`target-platform/<profile-stem>.md` (sibling to the profile YAML). It
bundles three things into one file so a future Claude session — or the
user — can see the shape of the active target without re-walking the
filesystem:

1. **BSP image layout** — top-level directories under
   `bsp_image.root_path`, presence of canonical subtrees (`rootfs/`,
   `bootloader/`, `source/`, …), and the nvpmodel variants matching
   the active module SKU.
2. **Source tree layout** — top-level subtrees under
   `source.root_path` (`kernel-jammy-src/`, `hardware/nvidia/`,
   `nvidia-oot/`, etc.) and devicetree files matching the chip family.
3. **Documents** — the `documents.*` references recorded in the
   profile, with local-path existence checks and one-line descriptions.

The KB is a **snapshot**, dated in its header. Re-run this skill
whenever the underlying data changes — it is intentionally
re-runnable and overwrites the previous KB on each run.

## When to invoke

- After `jetson-init-image` prepares the BSP for a freshly authored
  profile.
- After re-extracting a BSP archive or applying patches under
  `bsp_image.root_path`.
- After updating the source tree at `source.root_path`.
- After editing `bsp_image.*` or `documents.*` in the profile YAML.
- When a downstream skill asks "where is X in this BSP?" and you'd
  rather check the KB than re-walk the tree.

## Procedure

### Resolve the active target

Resolve the active profile per the contract in
[`../../context/target-platform-contract.md`](../../context/target-platform-contract.md);
cache it in memory — the rest of the skill consumes only this
profile. Record `<profile-stem>` (the bare filename minus `.yaml`)
as the KB output filename stem.

### Validate inputs

| Field | Required for KB? | If missing |
|---|---|---|
| `bsp_image.root_path` | **yes** | Refuse. A KB with no BSP root to scan is just a YAML restatement; tell the user to run `jetson-init-image` or hand-edit the profile. |
| `source.root_path` | no | Skip the source-tree section; note "source_root not recorded" in the KB. |
| `documents.*` | no | Render an empty Documents table with a "no documents recorded" note. |

If `bsp_image.root_path` is set but the directory does not exist on disk,
refuse with a clear message — do not fabricate a layout for a path
that isn't there.

### BSP discovery (under `bsp_image.root_path`)

Run **only** the following cheap operations — no recursive scans, no
file content reads beyond directory listings:

1. `ls -1` of `bsp_image.root_path` (one level deep). Record which
   directories are present.
2. For each canonical subtree below, mark present/absent:
   `rootfs/`, `bootloader/`, `kernel/`, `source/`, `tools/`,
   `nv_tegra/`.
3. Verify the active `flash_config` file exists at
   `<bsp_image.root_path>/<flash_config>`. Record its path or
   `(missing)`.
4. List `rootfs/etc/nvpmodel/` and filter to filenames matching
   `nvpmodel_<module.id>_<module.sku>*.conf`. Record each match.
   Use the lower-case module id (e.g. `p3767`) and the YAML-quoted
   sku string (e.g. `0001`).

### Source tree discovery (under `source.root_path`)

Skip this step entirely if `source.root_path` is `NA` or missing.
Otherwise, run only:

1. `ls -1` of `source.root_path` (one level deep).
2. For each canonical subtree below, mark present/absent:
   `kernel-jammy-src/`, `hardware/nvidia/`, `nvidia-oot/`, `nvgpu/`,
   `nvethernetrm/`, `nvdisplay/`, `hwpm/`, `kernel-devicetree/`.
3. If `kernel-devicetree/generic-dts/dts/` exists, list filenames
   matching `tegra<chip>*` where `<chip>` is the chip-family numeric
   prefix (see chip-family map below). Record up to 30 hits; if more,
   record the count and a "showing first 30" note.

#### Chip-family map (used in the "BSP discovery" and "Source tree discovery" steps)

Derive `<chip>` from `module.id`:

| `module.id` | Chip family | `<chip>` prefix |
|---|---|---|
| `p3701`, `p3767` | T234 — Orin | `234` |
| `p3834` | T264 — Thor | `264` |

If `module.id` is not in this table, record the chip as
`unknown (module.id=<value>)` and skip the chip-prefixed devicetree
filter.

### Documents pass

For each field in `documents.*` from the loaded profile:

1. Classify the value as **URL** (starts with `http://`, `https://`,
   or `ftp://`) or **local path** (anything else).
2. For URLs: record verbatim. **Do not fetch the URL** — KB
   generation must remain offline. (A future skill can promote to deep
   indexing.)
3. For local paths: check `os.path.exists`. Record the path; if
   missing on disk, append ` (missing)`.

The one-line description for each field comes from the marker in
[`../../references/platform_template.yaml`](../../references/platform_template.yaml)
— strip the `<OPTIONAL: …>` wrapper and use the inner text.

If the profile has no `documents:` block, render the section with a
single line: `_No documents recorded — run `jetson-link-docs`
or hand-edit the profile to add references._`

### Render and write the KB

Render the markdown using the structure below. Use today's date
(YYYY-MM-DD) in the header. Always overwrite any existing KB file at
the destination — do not prompt before overwriting; re-runs are the
intended use.

Destination: `target-platform/<profile-stem>.md`.

#### Rendered structure

```markdown
# Target knowledge base — <profile-stem>

> Generated <YYYY-MM-DD> from `<bsp_image.root_path>` (BSP version `<bsp_image.version>`).
> Re-run `jetson-generate-kb` after extracting a new BSP, applying
> patches, or editing profile fields. This file is a regenerated
> snapshot — do not hand-edit.

## Profile facts

- **Reference devkit:** `<reference_devkit.name>`
- **Module:** `<module.id>-<module.sku>` (`<chip family label>`)
- **Reference carrier:** `<carrier.id>-<carrier.sku>`
- **Custom carrier:** `<custom_carrier.name>` (`<custom_carrier.id>-<custom_carrier.sku>`)  _← omit this row if Case 1_
- **Active flash conf:** `<flash_config>`
- **BSP path:** `<bsp_image.root_path>`
- **BSP version:** `<bsp_image.version>`
- **Source root:** `<source.root_path>`  _← or "_not recorded_" if NA_

## BSP image layout

Top-level directories under `<bsp_image.root_path>`:

| Directory | Present | Purpose |
|---|---|---|
| `rootfs/`     | ✓ / ✗ | userspace rootfs (nvpmodel, nvfan, systemd units, etc.) |
| `bootloader/` | ✓ / ✗ | firmware blobs, BCT, MB1/MB2 dts |
| `kernel/`     | ✓ / ✗ | prebuilt kernel + modules |
| `source/`     | ✓ / ✗ | BSP source tree (kernel, OOT drivers, DT) |
| `tools/`      | ✓ / ✗ | flashing helpers, jetson-io, kernel_flash |
| `nv_tegra/`   | ✓ / ✗ | nvidia firmware tarballs, kernel-supplements |

Active flash conf `<flash_config>`: present at
`<bsp_image.root_path>/<flash_config>` _or_ `(missing — verify before flashing)`.

### nvpmodel files matching the active SKU

Filtered from `rootfs/etc/nvpmodel/` by `nvpmodel_<module.id>_<module.sku>*.conf`:

- `<each match, one per line>`

The active variant at boot is selected by `nvpower.sh` from
`/proc/device-tree/compatible` plus super / safety state — see
`jetson-customize-nvpmodel` for the resolution rules.

## Source tree layout

(omit this whole section if `source.root_path` is `NA`/missing)

Top-level subtrees under `<source.root_path>`:

| Subtree | Present | Purpose |
|---|---|---|
| `kernel-jammy-src/`   | ✓ / ✗ | mainline 5.x kernel sources |
| `hardware/nvidia/`    | ✓ / ✗ | NVIDIA platform DTs (per chip family) |
| `nvidia-oot/`         | ✓ / ✗ | NVIDIA out-of-tree kernel modules |
| `nvgpu/`              | ✓ / ✗ | GPU driver |
| `nvethernetrm/`       | ✓ / ✗ | ethernet driver |
| `nvdisplay/`          | ✓ / ✗ | display driver |
| `hwpm/`               | ✓ / ✗ | hardware performance monitor |
| `kernel-devicetree/`  | ✓ / ✗ | devicetree sources |

### Devicetree files for chip family `<chip>`

(omit if `kernel-devicetree/generic-dts/dts/` is absent)

Files matching `tegra<chip>*` under `kernel-devicetree/generic-dts/dts/`:

- `<each match, one per line — cap at 30, then a "first 30 of N" note>`

## Documents

| Field | Reference |
|---|---|
| Documents root folder                    | `<doc_root>`                    _or_ _not recorded_ |
| BSP / Jetson Linux developer guide       | `<bsp_developer_guide>`         _or_ _not recorded_ |
| Tegra SoC Technical Reference Manual     | `<soc_tech_ref_manual>`         _or_ _not recorded_ |
| Jetson module data sheet                 | `<module_data_sheet>`           _or_ _not recorded_ |
| Jetson module design guide (PDG)         | `<module_design_guide>`         _or_ _not recorded_ |
| Jetson module thermal design guide (TDG) | `<module_thermal_design_guide>` _or_ _not recorded_ |
| Jetson module schematic                  | `<module_schematic>`            _or_ _not recorded_ |
| Reference carrier board specification    | `<carrier_board_spec>`          _or_ _not recorded_ |
| Reference carrier schematic              | `<carrier_schematic>`           _or_ _not recorded_ |
| Custom carrier schematic                 | `<custom_carrier_schematic>`    _or_ _not recorded / N/A (no custom carrier)_ |
| Reference-devkit pinmux spreadsheet      | `<ref_devkit_pinmux_xls>`       _or_ _not recorded_ |
| Custom-carrier pinmux spreadsheet        | `<custom_carrier_pinmux_xls>`   _or_ _not recorded / N/A (no custom carrier)_ |

(Local paths are tagged ` (missing)` if absent on disk. URLs are
recorded verbatim and not fetched. If `doc_root` is set, also tag
` (missing)` on it if the directory itself is gone — that signals
auto-mapping in `jetson-link-docs` won't work on a re-run.)

## How to refresh this file

Re-run `jetson-generate-kb` whenever any of the following changes:

- the BSP at `<bsp_image.root_path>` is re-extracted, patched, or upgraded,
- the source tree at `<source.root_path>` changes,
- the active profile's `bsp_image.*` or `documents.*` fields are edited.

This file is overwritten on every run. Do not hand-edit it — edit the
source data (profile YAML or the BSP tree) and re-run instead.
```

### Confirm

Print a short summary:

- Output path: `target-platform/<profile-stem>.md`.
- BSP top-level directory count and which canonical subtrees were
  present / absent.
- nvpmodel match count for the active SKU.
- Source-tree section: rendered or skipped (and why).
- Documents: count of recorded fields, count of local paths flagged
  `(missing)`.

If a downstream skill triggered this run, tell the user to re-issue
their original request.

## Gotchas

- **The KB is a snapshot, not a live view.** The dated header is
  authoritative — if it doesn't match today, the BSP/source/docs may
  have changed underneath. Re-run on demand.
- **Always overwrites.** No prompt before clobbering the previous KB
  at `target-platform/<profile-stem>.md`. This is intentional —
  re-runnability is the whole point. Tell the user not to hand-edit
  the file; edit profile YAML or the BSP and regenerate.
- **Refuses on `bsp_image.root_path = NA`.** A profile with no BSP path
  produces a content-free KB. Do not write one — instead, point the
  user at the profile YAML to fill in.
- **No URL fetching.** Document URLs are recorded verbatim. Promoting
  to deep indexing (HTTP HEAD, PDF parsing) is out of scope for v0.1.
- **No recursive scans.** Discovery is one level deep per directory,
  with at most one targeted glob (nvpmodel + devicetree). Never walk
  the entire BSP — it's huge and slow.
- **Cap devicetree listings at 30 entries** to keep the KB readable.
  Show "first 30 of N" when truncating.
- **Don't auto-invoke from setup skills.** Setup may suggest running
  this skill in its summary, but the user opts in. Auto-running hides
  the I/O step and surprises users whose BSP/doc paths are incomplete.
- **Filename collision risk.** The KB sits at
  `target-platform/<stem>.md` next to `<stem>.yaml`. Don't accidentally
  read `.md` files in the profile-listing logic of
  `jetson-set-target` (it already filters to `*.yaml`, but check
  before adding new file types).
- **Chip-family map is short.** If a future module SKU is added that
  isn't in the map, the devicetree filter step is skipped — the KB
  will note `chip: unknown` rather than fabricate a `<chip>` prefix.
  Update this skill's chip-family table when a new chip lands.

## Prerequisites

- Active target profile resolved per
  `../../context/target-platform-contract.md`.
- `bsp_image:` recorded by `/jetson-init-image`; this is the only
  required on-disk tree. If `source.root_path` is missing, render the KB
  without the source-tree section.
- Optional: `/jetson-init-source` already resolved `source:` when the
  user wants source-tree discovery included.
- Optional but recommended: `/jetson-link-docs` already wrote the
  `documents:` block.

## Limitations

- Read-only against the BSP and source trees; never edits the profile
  YAML or rewrites source files.
- Devicetree-file enumeration depends on the chip-family table inside
  this skill; an unknown module SKU lands in the KB as `chip: unknown`
  rather than a fabricated prefix.
- Filename layout is fixed at `target-platform/<stem>.md` to stay next
  to the profile YAML; renaming the YAML invalidates the link.

## Troubleshooting

- **`bsp_image.root_path` not found** — re-run `/jetson-init-image` so
  the BSP is extracted and the path is recorded before regenerating
  the KB.
- **Source tree walk picks up wrong subtrees** — `source.root_path`
  override is stale; rerun `/jetson-init-source` or correct the
  profile field.
- **`documents:` block missing from the KB** — `/jetson-link-docs` was
  never run; the KB falls back to "no documents bound" rather than
  guessing paths.
- **Devicetree section short / empty** — chip-family table doesn't
  cover the active SoC; update the table and rerun.

## References

- [`../../context/target-platform-contract.md`](../../context/target-platform-contract.md) — read-order contract this skill follows.
- [`../../context/bsp-customization-workflow.md`](../../context/bsp-customization-workflow.md) — origin of the canonical BSP/source subtree list.
- [`../../references/platform_template.yaml`](../../references/platform_template.yaml) — source of the documents-field one-line descriptions.
- [`../jetson-init-target/SKILL.md`](../jetson-init-target/SKILL.md) — sibling skill that authors the active target identity.
- [`../jetson-init-image/SKILL.md`](../jetson-init-image/SKILL.md) — sibling skill that authors the BSP image metadata this skill scans.
- [`../jetson-set-target/SKILL.md`](../jetson-set-target/SKILL.md) — sibling skill that flips the active pointer this skill resolves.
