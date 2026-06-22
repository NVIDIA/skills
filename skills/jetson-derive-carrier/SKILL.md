---
name: jetson-derive-carrier
description: >-
  Bootstrap a custom carrier board by forking carrier files and
  scaffolding a DT overlay from the reference devkit. Use after
  jetson-init-source; not for module-level or kernel-DTB changes.
version: 0.0.1
license: "Apache-2.0"
metadata:
  data-classification: public
  author: "Jetson Team"
  tags:
    - target-platform
    - custom-carrier
    - bring-up
    - setup
  domain: meta
---

# jetson-derive-carrier

Customize first-run gate for any `customize-*` skill on a custom
carrier. Resolve the active target per
[`target-platform-contract.md`](../../context/target-platform-contract.md).
**Refuse** if no `custom_carrier:` block, or if
`<source.root_path>/Linux_for_Tegra/.git` is missing
(`jetson-init-source` first). Template file create/copy rows land in a
single overlay-tracker commit and copy **directly** to the
custom-carrier-renamed target path — reference-named pristine files
are never staged (see [Fork plan](#fork-plan) for the local rule, which
overrides the standard pristine + customization split from the
[workflow](../../context/bsp-customization-workflow.md#commit-batching-in-the-overlay-tracker));
conf-chain discovery follows
[per-board conf dispatch](../../context/bsp-customization-software-layers.md#per-board-conf-dispatch--update_flash_args_common).
Kernel base DTB is **not** forked — carrier deltas layer as a DT
overlay wired via `OVERLAY_DTB_FILE`.

**Identifiers** (from active profile): `<chip>` from
`reference_devkit.module.id` via
[catalogue](../../references/bsp-platforms-catalogue.md) (unknown →
warn, fallback `tegra234`); `<module-id>`/`<module-sku>` from
`reference_devkit.module`; `<carrier-id>`/`<carrier-sku>` from
`reference_devkit.carrier`; `<custom-id>`/`<custom-sku>`/
`<custom-flash-conf>` from `custom_carrier`. `<real-conf>` =
`readlink <bsp_image.root_path>/Linux_for_Tegra/<reference_devkit.flash_config>`
(NVIDIA convention `<devkit>.conf` → `<carrier>-<module>-a<rev>.conf`;
warn + treat top-level as real + skip symlink-wrapper row if not a
symlink).

`<custom-id>` is a custom carrier board token, not necessarily an
NVIDIA-style `pNNNN` ID. Use it verbatim in dash-form filenames and DT
compatible strings. When a target file family uses p-stripped numeric
tokens, derive `<custom-id-file-token>` as `NNNN` only if `<custom-id>`
matches `^p[0-9]{4}$`; otherwise use `<custom-id>` unchanged. Do not
reject custom carrier IDs merely because they do not start with `p`.

## Instructions

The fork plan below is the instruction set: one discovery pass, then a
row-by-row fork of the per-board fileset, gated by the commit-message
preview before each `git commit`.

### Fork plan

Every `git commit` produced by this skill — in the overlay tracker
(`Linux_for_Tegra/`) or the `hardware/` source repo — runs through
the
[commit message preview gate](../../context/bsp-customization-workflow.md#commit-message-preview-gate).
Surface the staged file list + proposed message to the operator
and require accept / edit / cancel before each `git commit`; on
**cancel** leave the index staged for manual resolution.

**Materialize renamed forks even when bytes do not change.** For every
accepted fork-plan row, create and track the custom-carrier target path;
do not skip a missing target because the fork is only a rename/copy or is
byte-identical to the reference. This includes BCT forks such as pinmux,
GPIO/GPIOINT, padvoltage/PMC, misc, and MB2 misc. Skip only when the
Acceptance column allows it, the source is an explicit warn-and-skip
miss, or the expected target path is already tracked; in the last case,
report it as already derived and do not create an empty commit.

**Template file create/copy rows squash into one overlay-tracker
commit, and reference-named pristine files are never staged.** All
rows that materialize a new (template-derived) file in the overlay
tracker — board flash-conf, flash-conf symlink, MB1 BCT (pinmux,
GPIO/GPIOINT, padvoltage/PMC, misc), MB2 BCT (misc), BPMP DTB (when
opted in), nvpmodel, nvfancontrol — copy **directly** from
`<bsp_image.root_path>/Linux_for_Tegra/<rel>/<reference-name>` to
`<source.root_path>/Linux_for_Tegra/<rel>/<custom-name>` with any
content edits applied before staging. The reference-named
(`<carrier-id>-<carrier-sku>`-keyed) filename is never staged or
committed in the overlay tracker — only the custom-carrier-renamed
file is tracked. This overrides the standard pristine + customization
split from
[commit batching](../../context/bsp-customization-workflow.md#commit-batching-in-the-overlay-tracker).
All such rows land in a **single** overlay-tracker commit that adds:
(i) the custom-carrier-named files at their target paths with content
already applied, (ii) the flash-conf symlink, and (iii) the targeted
flash-conf content rewrites inside the renamed flash-conf for
`PINMUX_CONFIG` / `GPIO_CONFIG` / `GPIOINT_CONFIG` / `PMC_CONFIG` /
`MISC_CONFIG` / `MB2_BCT` (plus `BPFDTB_FILE` when BPMP DTB is opted
in). Rows that edit existing upstream files instead of creating
templates — the `nvpower.sh` patch — and the overlay wire-up
(`OVERLAY_DTB_FILE+=` append to the just-created flash-conf fork,
treated as a temporally distinct phase per
[commit batching](../../context/bsp-customization-workflow.md#commit-batching-in-the-overlay-tracker))
remain separate commits per their own rows. The DT overlay skeleton
commits in `bsp_sources/hardware/`, not the overlay tracker, so it is
unaffected. The commit message preview gate fires once on the
squashed commit; warn-and-skipped rows do not contribute, and if every
row in the bundle is already tracked the commit is omitted entirely.

### Discovery (single batched pass)

Run **one** discovery pass; do not interleave with staging. **Do not
truncate listings** — every candidate filename in each scanned
directory must be visible to the matcher. `head`, `tail`, `| head -N`,
`| tail -N`, and any other row-limiting filter are out of bounds for
this step; use `ls -1` / `find` unclipped, or grep on the full output.
Truncating risks false warn-and-skip calls when the matching file
sits past the cutoff. Capture:

- Flash-conf vars: `DTB_FILE` / `TBCDTB_FILE` / `BPFDTB_FILE` /
  `PINMUX_CONFIG` / `PMC_CONFIG` / `GPIO_CONFIG` / `GPIOINT_CONFIG`.
  `DTB_FILE` / `TBCDTB_FILE` are captured for **reference only** so
  the overlay wire-up can derive `<dtb-stem>` for the overlay
  filename — they are never rewritten by this skill.
- BCT `.dts` at `bootloader/generic/BCT/`; `.dtsi` siblings live
  **one level up at `bootloader/`** — NOT alongside.
- nvpmodel: `nvpmodel_<module-id>_<module-sku>*.conf`. nvfancontrol:
  `nvfancontrol_<module-id>_<module-sku>_<carrier-id>_<carrier-sku>.conf`.
- `nvpower.sh` anchors for (a)–(d): `<carrier-id>` cvb branch,
  `<module-id>-<module-sku>` SKU `elif`, `tegra<chip>` nvpmodel
  cascade, `tegra<chip>` nvfancontrol cascade.

| File / category | Discovery | Acceptance | Fork rule |
|---|---|---|---|
| Board flash conf | `<real-conf>` | Always | Filename sub `<carrier-id>-<carrier-sku>` → `<custom-id>-<custom-sku>`. Content sub is **targeted, not blanket**: rewrite RHS only for vars whose file this skill forks — `PINMUX_CONFIG`, `PMC_CONFIG`, `GPIO_CONFIG` / `GPIOINT_CONFIG`, `MISC_CONFIG`, `MB2_BCT`. Other carrier-keyed vars (`DTB_FILE`, `TBCDTB_FILE`, `SCR_CONFIG`, `PMIC_CONFIG`, `DEVICEPROD_CONFIG`, `PROD_CONFIG`, `MINRATCHET_CONFIG`, `UPHY_CONFIG`, dynamic `OVERLAY_DTB_FILE+=`) MUST stay at reference values — their files aren't forked here and a blanket `sed` would point them at nonexistent files. `DTB_FILE` / `TBCDTB_FILE` specifically: base DTB is not forked; the overlay row appends a new `OVERLAY_DTB_FILE+=` line instead. `BPFDTB_FILE`: opt-in extra commit. Never touch `<chip>`, `<module-id>`, `xxxx`. |
| Flash-conf symlink | Unconditional | Always (skip if `<real-conf>` not a symlink) | New symlink `<custom-flash-conf>` → flash-conf fork |
| MB1 BCT pinmux | `PINMUX_CONFIG` + `#include` follow ¶ | Always | Rename rule † |
| MB1 BCT GPIO | `GPIO_CONFIG`/`GPIOINT_CONFIG` + `#include` follow ¶ | Always | Rename rule † |
| MB1 BCT padvoltage | `PMC_CONFIG` + `#include` follow ¶ | Always | Rename rule † |
| MB1 BCT misc | `MISC_CONFIG` + `#include` follow ¶ | Always | Rename rule † |
| MB2 BCT misc | `MB2_BCT` + `#include` follow ¶ | Always | Rename rule † |
| Kernel DTB + source DTS | (reference only — see header) | Never forked | Base DTB stays at reference. No pristine binary copy in the overlay tracker; no source DTS fork in `bsp_sources/hardware/`. Carrier deltas live entirely in the DT overlay (next row). |
| DT overlay skeleton (NEW) | Unconditional | Always | Create `<source.root_path>/hardware/nvidia/<chip>/nv-public/overlay/<chip>-<custom-id>-<custom-sku>+<module-id>-<module-sku>.dts` (skeleton ‡); commit in `hardware/`, push to origin |
| Per-dir Makefile registration | `<source.root_path>/bsp_sources/hardware/nvidia/<chip>/nv-public/overlay/Makefile` | Always | Append `dtbo-y += <chip>-<custom-id>-<custom-sku>+<module-id>-<module-sku>.dtbo` after the **last literal-named** `dtbo-y +=` entry (BEFORE the `$(addprefix $(makefile-path)/,$(dtbo-y))` prefix block — inserting after `dtbo-y += $(old-dtbo)` silently drops the .dtbo). Same position-sensitive idiom and snippet as the composite slot's Makefile patch in [`../jetson-build-source/references/composite-registration.md#makefile-patch-idempotent-position-sensitive`](../jetson-build-source/references/composite-registration.md#makefile-patch-idempotent-position-sensitive). Commit in `hardware/`. Without this row, `nvidia-dtbs` never produces the `.dtbo` referenced by the next row's `OVERLAY_DTB_FILE+=` line and `flash.sh` aborts mid-flash on the missing file. |
| Overlay wire-up | Unconditional | Always | Append `OVERLAY_DTB_FILE+=",<chip>-<custom-id>-<custom-sku>+<module-id>-<module-sku>.dtbo"` to flash-conf fork (extra commit) |
| nvpmodel config | `rootfs/etc/nvpmodel/nvpmodel_<module-id-num>_<module-sku>.conf` | Always | Filename: append `_<custom-id-file-token>_<custom-sku>` |
| nvfancontrol config | `rootfs/etc/nvpower/nvfancontrol/nvfancontrol_<module-id-num>_<module-sku>_<carrier-id-num>_<carrier-sku>.conf` | Always | Filename: substitute carrier portion (append `_<custom-id-file-token>_<custom-sku>` if source is module-keyed only) |
| `nvpower.sh` patch | `rootfs/etc/systemd/nvpower.sh` | Always | Pristine + single customization commit with 4 insertions: (a) cvb cascade `elif [[ "${machine}" =~ "<custom-id>" ]]; then cvb="<custom-id-file-token>"` before reference; (b) inside module-SKU branch set `machine="<module-id>-<module-sku>-<custom-id>-<custom-sku>"`; (c) nvpmodel cascade branch for composite machine key → `conf_file=` nvpmodel fork; (d) cvb-keyed nvfancontrol cascade branch → `conf_file=` nvfancontrol fork |
| BPMP DTB | prebuilt at `bootloader/generic/<BPFDTB_FILE>` (alt naming: no `p` prefix, `xxxx` SKU) | **Opt-in** y/N, default N — module-level, usually shared with reference | Fork binary: pristine + filename sub `-<carrier-id-num>-` → `-<custom-id-file-token>-` (content unchanged). Extra commit on flash-conf fork rewriting `BPFDTB_FILE` to the renamed binary. |

¶ **`#include` follow.** Scan each captured `.dts` for `#include
"..."` directives **anywhere** in the file (NVIDIA BSPs put them
inside BCT node bodies); carrier-keyed includes join the fork list.
Stop at one level.

† **Rename rule.** Carrier-keyed source (e.g.
`…-p3834-xxxx-p4071-0000.dts`) → filename sub
`<carrier-id>-<carrier-sku>` → `<custom-id>-<custom-sku>`; rewrite the
flash-conf variable (e.g. `PINMUX_CONFIG`) to the renamed filename in
the same customization commit. **Module portion is preserved
verbatim from pristine** — if pristine has `p3834-xxxx`, fork keeps
`p3834-xxxx`; if pristine has `p3834-0008`, fork keeps `p3834-0008`.
Never pin module-SKU on your own. Same rule for any other `xxxx`
wildcard in the carrier-SKU position when pristine uses it (e.g.
`p4071-xxxx` → `p1234-xxxx`). Module-keyed
source (e.g. `…-p3767-dp-a03.dtsi`) → append
`_<custom-id-file-token>_<custom-sku>` (underscore-form) or
`-<custom-id>-<custom-sku>` (dash-form) **plus** an extra commit on
flash-conf fork rewriting the variable (e.g. `PINMUX_CONFIG`) to the
suffixed name. Content never substituted by default; prompt if
content holds a literal carrier-id-sku self-reference.

‡ **Overlay skeleton.** DT plugin overlay (`/plugin/;`) with one
`fragment@0` at `target-path = "/"`; inside `__overlay__` set
`model = "<custom_carrier.name> carrier board"` and `compatible =
"nvidia,<custom-id>-<custom-sku>+<module-id>-<module-sku>",
"nvidia,<chip>"`. `nvpower.sh` machine narrowing reads `compatible`.

### Edit verification

Re-grep after every `sed` / patch. `sed` silently no-ops on miss;
exit code 0 proves nothing. Multi-line replacements: use Python
`str.replace(old, new)`, not multi-line `sed` (brittle to whitespace
drift, silent no-ops). Refuse to commit on verification miss.

**Summary.** Print forks per repo (count + commit SHAs), BPMP-DTB
decision, warn-and-skipped rows (mandatory source file missing on
this BSP — don't fail the run), overlay path written.

## Examples

Trigger phrases the operator might use:

```text
derive custom carrier
bootstrap custom carrier
fork carrier board from reference devkit
```

Minimal `custom_carrier:` block in the active profile that the skill
expects to find:

```yaml
custom_carrier:
  id: p1234            # any token; need not be NVIDIA pNNNN style
  sku: 0000
  flash_config: p1234.conf
  name: Acme Custom Carrier
```

## Purpose

Customize the per-board fileset for a new carrier board so downstream
`customize-*` skills, `nvpower.sh`, and the boot chain land on the
custom carrier instead of the reference devkit. Base DTB stays at the
reference; carrier deltas live in a DT overlay so a re-derive against
a new reference BSP only needs to re-run this skill.

## Prerequisites

- Active profile resolved per
  [`target-platform-contract.md`](../../context/target-platform-contract.md)
  with a `custom_carrier:` block (id, sku, flash-conf, friendly name).
- `source.root_path/Linux_for_Tegra/.git` initialized — run
  `/jetson-init-source` first; this skill refuses if the overlay
  tracker is missing.
- `bsp_image.root_path` populated by `/jetson-init-image` so the
  reference flash-conf symlink can be resolved.
- `reference_devkit:` populated (module + carrier) so the rename rule
  has carrier-id/sku tokens to substitute away from.

## Limitations

- Kernel base DTB and its source DTS are **not** forked — carrier
  deltas must layer through the DT overlay wired via
  `OVERLAY_DTB_FILE`.
- BPMP DTB fork is opt-in (default OFF). Most carrier swaps share the
  reference BPMP DTB; only opt in when board power/clock topology
  diverges.
- Flash-conf content substitution is **targeted**, not blanket. Vars
  whose files this skill does not fork (`DTB_FILE`, `TBCDTB_FILE`,
  `SCR_CONFIG`, `PMIC_CONFIG`, `DEVICEPROD_CONFIG`, `PROD_CONFIG`,
  `MINRATCHET_CONFIG`, `UPHY_CONFIG`, dynamic `OVERLAY_DTB_FILE+=`)
  stay at reference values — rewriting them would point at files that
  do not exist.
- Custom-carrier IDs need not follow NVIDIA `pNNNN` style; the
  p-stripped numeric token is derived only when the ID matches
  `^p[0-9]{4}$`.
- Does not customize module-level files. Use the relevant
  `jetson-customize-*` skill for pinmux, clocks, fan curves,
  nvpmodel, PCIe, UPHY, USB, MGBE, or camera deltas.

## Troubleshooting

- **Refuses with "no `custom_carrier:` block"** — declare the block in
  the active profile before re-running; the skill will not infer it.
- **Refuses with ".git missing in `Linux_for_Tegra/`"** — overlay
  tracker not initialized. Run `/jetson-init-source` first.
- **"`<real-conf>` not a symlink"** — NVIDIA convention is a
  `<devkit>.conf` symlink to `<carrier>-<module>-a<rev>.conf`; if the
  top-level is the real conf, the skill warns, treats it as real, and
  skips the symlink-wrapper row. Not an error.
- **`sed` reported success but the edit is missing** — `sed` silently
  no-ops on miss. The skill re-greps after every patch and refuses to
  commit on verification miss; for multi-line replacements switch to
  Python `str.replace`.
- **"chip: unknown" / fallback to `tegra234`** — module ID is not in
  the [catalogue](../../references/bsp-platforms-catalogue.md); update
  the catalogue rather than override in the profile.
- **Commit message preview prompt blocks the run** — expected gate per
  [commit-batching](../../context/bsp-customization-workflow.md#commit-batching-in-the-overlay-tracker);
  accept, edit, or cancel. On cancel the index is left staged for
  manual resolution.
- **A mandatory source file is missing on this BSP** — the row is
  recorded as warn-and-skipped in the summary; the rest of the run
  still completes.
