---
name: jetson-customize-uphy
description: Configure Jetson UPHY lane allocation (uphy0/uphy1-config) on Orin/Thor custom carriers. Do NOT use for pinmux or PCIe-only edits.
version: 0.0.2
license: "Apache-2.0"
metadata:
  data-classification: public
  author: "Jetson Team"
  tags:
    - bsp
    - phase-2
    - io
    - uphy
  domain: meta
---

# Customize UPHY lane allocation

## Purpose

Select a UPHY lane allocation on a Jetson custom carrier and edit the
carrier flash-conf fork's `ODMDATA="..."` to apply the chosen
`uphyX-config-N` token(s) (plus the `UPHY_CONFIG=""` clear required for
`uphy0-config-6`). Kernel-DT alignment per controller is **not** done
here — after the ODMDATA commit lands, this skill dispatches to the
per-controller skills (`/jetson-customize-pcie`,
`/jetson-customize-mgbe`, `/jetson-customize-usb`), each of which must
compare the chosen allocation against the **reference kernel DTB**
node-by-node and emit an overlay fragment only when the K-stock value
disagrees with the chosen allocation. Discovery is agentic: options,
lanes, and controllers come from the Adaptation Guide, carrier
schematic, and Module / SoC TRM at run time — never hard-coded. Every
user-visible step renders its data as a markdown table, and the final
summary includes a changes-summary table.

## Prerequisites

- Active target-platform profile with `reference_devkit:` and
  `custom_carrier:`.
- `<source.root_path>/Linux_for_Tegra/.git` initialized
  (`/jetson-init-source`).
- Forked carrier conf present (`/jetson-derive-carrier`).
- Reachable Adaptation Guide via `documents.adaptation_guide` ->
  `documents.bsp_developer_guide` -> web fetch -> Step-1 prompt
  fallback.
- **When `custom_carrier:` is present, both
  `documents.custom_carrier_schematic` AND
  `documents.custom_carrier_pinmux_xls` are REQUIRED.** The skill
  refuses to run if either is missing — routing decisions for the
  custom carrier cannot be guessed. Reference-devkit-only profiles
  (no `custom_carrier:` block) do not require these.

## Overview

UPHY (unified PHY) is the shared high-speed PHY pool on Tegra264 (Thor)
and Tegra234 (Orin). Lane allocation is selected by `ODMDATA` tokens
(`uphy0-config-N`, Thor also `uphy1-config-N`) parsed at flash time by
`tegraflash_impl_t264.py::tegraflash_update_bpmp_dtb()` and written
into `/uphy/uphy{0,1}-config` of the BPMP DTB.

Output is a **single atomic ODMDATA commit** in
`<source.root_path>/Linux_for_Tegra/` carrying the chosen
`uphyX-config-N` token(s), the `UPHY_CONFIG=""` clear (for
`uphy0-config-6`), AND every per-controller ODMDATA token derived
from the chosen allocation (`pcie@N_status=*`, `mgbeN-speed-*`, USB SS
per-port tokens). Sub-skills (`/jetson-customize-pcie`,
`/jetson-customize-mgbe`, `/jetson-customize-usb`) own only the
kernel-DT overlay fragments — they MUST NOT touch ODMDATA. All
commits follow the batched pristine + customization pattern in
`../../context/bsp-customization-workflow.md`. Upstream BSP at
`<bsp_image.root_path>/` is never edited.

## When to invoke

- User says "configure UPHY", "uphy lane allocation",
  "set uphy0-config-N", "change MGBE speed", or asks to remap
  PCIe / MGBE / USB3 / UFS on a custom carrier.
- A UPHY-fed controller doesn't enumerate after flash, OR cold boot
  dies in BL31 SError / `BPMP firmware is not ready`.
- A downstream skill reports FMON fault or BPMP-DTB lane mismatch.

## Procedure (summary)

Eight steps; full detail in `references/procedure.md`.

1. **Resolve target + docs.** Refuse without active profile, custom
   carrier, source-tree git, or forked carrier conf. Resolve Adaptation
   Guide / schematic / Module Design Guide / SoC TRM.
2. **Locate "Configure the UPHY Lane"** in the Adaptation Guide (PDF /
   HTML mirror / `WebFetch`). Cross-check Module Design Guide + SoC TRM.
3. **Cross-reference the carrier schematic.** Cite UPHY net names
   (`MGBE2_TX_P/N`, `PEX5_LN0+-`, etc.). Zero matching nets = unrouted.
4. **Enumerate matching UPHY options.** Surface every documented
   `uphy0-config-N` (and Thor `uphy1-config-N`) index.
5. **Ask the user which config (HARD GATE).** Print tables first, then
   `AskUserQuestion` — one per UPHY surface, plus carrier-routing
   confirmation if any allocated lane is unrouted. Persist answers to
   the JSON sidecar (`references/run-state-sidecar.md`).
6. **Edit carrier flash-conf fork (atomic ODMDATA commit).** This skill
   owns every ODMDATA token for the run. One `ODMDATA="..."` line, one
   commit, all tokens. Sub-skills MUST NOT touch ODMDATA.

   Decompile the BPMP DTB at
   `<bsp_image.root_path>/Linux_for_Tegra/bootloader/generic/<BPFDTB_FILE>`
   (`BPFDTB_FILE` from the carrier conf) to snapshot stock state, then
   emit tokens in this order:

   a. **UPHY surface tokens** — every chosen `uphyX-config-N` (Thor:
      both surfaces, even if one equals the guide default). Order
      `uphy0` then `uphy1`; separator `,`.
   b. **Per-controller tokens** — one per row whose plan-state differs
      from BPMP-stock. Match-rows get no token (redundant tokens can
      drop the whole line).
      - PCIe: `pcie@N_status=okay|disabled`.
      - MGBE: `mgbeN-speed-<rate>` on allocate, **`mgbeN-speed-del`**
        on disable. FMON arms on the controller's own clocks
        regardless of UPHY allocation — missing `del` ⇒ BL31 SError
        reboot loop. Single most common post-flash failure on Thor.
      - USB SS: per-port tokens when the SoC grammar exposes them.
   c. **`UPHY_CONFIG=""` clear** when `uphy0-config-6` is selected
      (BCT pinmux clear per Adaptation Guide).
7. **Build the per-controller allocation table and dispatch.** Derive
   one row per UPHY-fed controller (PCIe / MGBE / USB SS / UFS) with
   `{class, instance, allocated?, BPMP-stock, K-stock, routed?,
   Desired K state}`. This table drives both (a) Step 6's ODMDATA
   tokens and (b) the sub-skills' overlay fragments — build it
   before Step 6 commits.

   Then invoke `/jetson-customize-pcie`, `/jetson-customize-mgbe`, and
   `/jetson-customize-usb` for **kernel-DT overlay fragments only**
   (no ODMDATA edits — Step 6 owns the line). Each sub-skill re-reads
   K-stock from
   `<bsp_image.root_path>/Linux_for_Tegra/kernel/dtb/tegra<soc>-*-nv.dtb`
   and skips emission when K-stock matches Desired K state. UFS
   handling stays inline here (no UFS sub-skill).

   Invoke all three whenever their controller class is present on
   this SoC (e.g. skip MGBE on Orin). Ask the operator first; on
   `yes`, run the sub-skill inline.
8. **Summary + next-step chain.** Headline, breakdown, **choices table**
   (UPHY surface | chosen config | lane summary | UPHY_CONFIG-clear),
   **changes-summary table** (file | repo | commit SHA | one-line
   summary covering this skill's commit + every dispatched sub-skill's
   commit), then drive the downstream chain (more I/O? build & promote?
   flash? validate?) via sequential `AskUserQuestion` prompts per
   `references/procedure.md` Step 8. Never substitute a printed
   "Next step: …" line for the prompts.

## Limitations

- Only supports Tegra234 (Orin) and Tegra264 (Thor) UPHY surfaces.
- Does not edit pinmux, PCIe-only DT properties absent from BPMP DTB
  (`num-lanes`, `pcie-mode`), or upstream BSP files.
- Does not flash, build, or promote — chain into `/jetson-build-source`
  and downstream skills.
- Hard-coded option tables are forbidden; if no Adaptation Guide
  source resolves the skill refuses rather than guessing.
- Not table-driven across releases: every run re-reads the Guide for
  the active BSP version.

## Troubleshooting

- **Cold boot reboot loop / BL31 `plat_setup.c:726` / `BPMP firmware
  is not ready`** after `uphy0-config-6`: a later
  `^UPHY_CONFIG=` line in the carrier conf re-overrode the clear.
  Comment it (see `references/procedure.md` Step 6).
- **`wait-for-device failed` at flash, BPMP DTB unchanged**: an
  ODMDATA token had wrong shape (e.g. `mgbe0-speed-0`). One bad token
  drops the whole `ODMDATA="..."` line. Inspect grammar in
  `references/procedure.md`.
- **BL31 SError reboot loop after disabling an MGBE**: missing
  `mgbeN-speed-del`. FMON arms on the controller's own clocks
  regardless of UPHY allocation.
- **Newly-routed controller doesn't enumerate**: stock kernel DTB had
  `status="disabled"`. Overlay must emit `status="okay"` (matrix
  row 4 in `references/procedure.md`).
- **Pinmap delta=0 but board still misbehaves**: UPHY differential
  pairs are absent from pinmux `.xlsm`. Drive decisions off schematic
  net names, not the pinmap.
- **Duplicate `pcie@<addr>` fragments**: another skill
  (`jetson-customize-pcie`) already owns that node. Scope this skill
  to MGBE / UFS / USB3 SS / PCIe-status-only and cite the other
  overlay.

## References

- `references/procedure.md` — full eight-step procedure.
- `references/gotchas.md` — cross-cutting gotchas.
- `references/run-state-sidecar.md` — JSON sidecar schema + idempotency.
- `../../references/platform_template.yaml` — `documents:` schema.
- `../../context/bsp-customization-workflow.md` — overlay edit protocol.
- `../../references/bsp-customization-kernel-dtb.md` — composite-overlay
  filename / append protocol.
- `../jetson-derive-carrier/SKILL.md` — produces the conf this skill
  edits.
- `../jetson-init-source/SKILL.md` — produces the two git repos this
  skill commits into.
- `../jetson-generate-kb/SKILL.md` — KB consulted for chip family +
  file locations.
