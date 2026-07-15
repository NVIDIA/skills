---
name: jetson-customize-pinmux
description: >-
  Per-pin SFIO / direction / initial-state configurator for a Jetson
  Orin or Thor custom carrier from the pinmux XLSM. Do NOT use for
  kernel-DT overlay or ODMDATA edits.
version: 0.0.1
license: "Apache-2.0"
metadata:
  data-classification: public
  author: "Jetson Team"
  tags:
    - bsp
    - phase-2
    - io
    - pinmux
  domain: meta
---

# Customize pinmux (per-pin SFIO / direction / state)

## Overview

The Tegra pinmux spreadsheet (`.xlsm`) is the ground truth for every
CVM ball: SoC pin name, supported SFIOs, customer-selected function,
direction, and initial state. This skill parses that XLSM, runs a
per-pin Q1–Q6 interactive loop, and emits the three BCT DTSIs
(`pinmux`, `gpio`, `padvoltage`) in one shot into the overlay tracker
at `<source.root_path>/Linux_for_Tegra/bootloader/`.

Unlike sibling skills `jetson-customize-uphy` / `jetson-customize-pcie` /
`jetson-customize-camera`, **pinmux has no kernel-DT overlay surface and no
ODMDATA edit**. The XLSM is the source of truth; the three emitted
DTSIs land at flash time via the carrier conf's
`PINMUX_CONFIG=` / `GPIOINT_CONFIG=` / `PMC_CONFIG=` references (which
`/jetson-derive-carrier` set up).

**Pad classification (silicon-fixed):** only `BD*` and `BI*` pads have
configurable pull / drive / open-drain attributes. `LP5XA_*`,
`UPHYDS_*`, `DP_SINGLE_*`, `BDMIPI16X_*`, `BDUSB2_*`, `OSCI27_*` are
fixed-function and **skip Q4–Q6** (`configurable: no`).

The bundled `scripts/modify_pinmux.py` is the workhorse: it parses
the XLSM via `openpyxl>=3.1`, builds the per-carrier pinmap JSON,
captures pin edits into a session shim, and (on `generate`) writes
the three DTSIs.

## When to invoke

- The user says "configure pin", "set SFIO", "edit pinmux DTSI",
  "set pin direction", "set initial state", or asks to repurpose a
  CVM ball (e.g. flip a pin between GPIO and a peripheral function).
- A sibling skill (`jetson-customize-camera`, `jetson-customize-pcie`,
  `jetson-customize-usb`, `jetson-customize-mgbe`) reports an HSIO pin mismatch
  via `pin_verifier.py` and the user wants to fix it.
- The user pre-derived a custom carrier with `/jetson-derive-carrier`
  and now wants to author the pinmux from a freshly-edited `.xlsm`.

**Prerequisites:**

- Active profile selected (`target-platform/active_target.yml` →
  `<profile>.yaml` with `reference_devkit:` AND `custom_carrier:`).
- `<source.root_path>/Linux_for_Tegra/` exists as a git repo
  (`/jetson-init-source`).
- `/jetson-derive-carrier` has run — the three pinmux-side BCT DTSIs
  (`PINMUX_CONFIG`, `GPIOINT_CONFIG`, `PMC_CONFIG` references in the
  carrier conf) exist in the overlay tracker.
- A pinmux `.xlsm` is registered in the active profile at
  `documents.custom_carrier_pinmux_xls` (preferred when custom-carrier-
  specific) or `documents.ref_devkit_pinmux_xls` (fallback). The
  bundled `modify_pinmux.py` requires `openpyxl>=3.1`.

## Procedure

See [`references/procedure.md`](references/procedure.md) for the full
step-by-step procedure (Steps 1–8). Summary:

1. **Resolve active target + XLSM.** Validate active profile,
   `custom_carrier:`, overlay-tracker prerequisites; resolve the
   pinmux `.xlsm` path from
   `documents.custom_carrier_pinmux_xls` →
   `documents.ref_devkit_pinmux_xls` → single XLSM under
   `documents.root_path` → user prompt.
2. **Probe.** Run `modify_pinmux.py probe` to parse the XLSM into the
   per-skill scratch `<KB>/pinmap/<custom-carrier>.json` plus
   `session.json` shim.
3. **Lookup.** Resolve a free-form user query (CVM ball, Verilog
   name, signal, DT pin) via `modify_pinmux.py lookup`; surface
   supported SFIO list, defaults, and `configurable: yes/no`.
4. **Set-pin (HARD GATE — Q1–Q6 via `AskUserQuestion`).** Q1–Q3
   (`sfio` / `direction` / `initial_state`) always asked; Q4–Q6
   (`pull` / `drive_type` / `open_drain`) only when
   `configurable: yes`. `tristate` and `e_input` are derived from
   `direction`, never asked.
5. **Generate.** `modify_pinmux.py generate --out-dir
   <source.root_path>/Linux_for_Tegra/bootloader/` (root, **not**
   `bootloader/generic/BCT/` — derive-carrier `.dts` forks live there,
   do not colocate). Emits:
   `tegra<soc>-mb1-bct-{pinmux,gpio,padvoltage}-<carrier-key>.dtsi`.
   **`<carrier-key>` comes from the carrier conf's `PINMUX_CONFIG=`
   reference, NOT the kebab-cased carrier name.**
6. **Commit (single batched commit per workflow rule).** All three
   DTSIs are one logical edit → one customization commit. Run the
   commit-preview gate before each commit.
7. **Run-state sidecar + session shim.** Write the user-facing
   `<profile-stem>.jetson-customize-pinmux.json` sidecar and the
   transient `session.json` shim under
   `<workspace>/target-platform/`.
8. **Summary.** Emit the standard one-line + table summary.

## Gotchas

- **No kernel-DT overlay; no `OVERLAY_DTB_FILE` edit; no
  `render_conf.py` hand-off.** This skill ends at the three BCT
  DTSIs — the carrier conf already references them via
  `PINMUX_CONFIG=` / `GPIOINT_CONFIG=` / `PMC_CONFIG=` (set up by
  `/jetson-derive-carrier`).
- **Re-point the `.dts` wrapper's `#include` after `generate`.**
  `/jetson-derive-carrier` forks the `.dts` wrappers at
  `bootloader/generic/BCT/`, but their `#include` lines may still
  pull the upstream devkit `.dtsi` (e.g.
  `…-p3834-xxxx-p4071-0000.dtsi`). After `generate` writes the new
  `<CARRIER_KEY>.dtsi` to `bootloader/` root, edit each wrapper's
  `#include` to the new filename — by **bare basename**
  (`#include "tegra<soc>-mb1-bct-pinmux-<CARRIER_KEY>.dtsi"`), not
  `../../…` filesystem-relative. The BCT build's `cpp -I bootloader/`
  resolves bare basenames; that's the convention every other BCT
  include in the tree follows. Roll the wrapper edits into the same
  customization commit as the three DTSIs. See `references/procedure.md`
  Step 5 ("Sanity-check the carrier `.dts` wrapper").
- **Q4–Q6 gated on `configurable: yes`.** Asking pull / drive_type /
  open_drain on a fixed-function pad (`LP5XA_*`, `UPHYDS_*`,
  `BDMIPI16X_*`, etc.) is silently dropped by the script and confuses
  the user. `lookup` prints `configurable: yes/no` — always check it
  before prompting Q4–Q6.
- **`tristate` and `e_input` are derived, never asked.** `unused` →
  tristate=ENABLE; `input` / `bidirectional` → enable-input=ENABLE.
  Exposing them as separate prompts produces inconsistent DTSIs.
- **`sfio=gpio` requires a parseable `gpio=GPIOn_PD.NN` entry in the
  pinmap row's `sfio` list.** Pins without one are GPIO-incapable
  silicon; `set-pin` rejects the call. Surface the rejection — don't
  silently fall back to a non-GPIO SFIO.
- **Marker idempotency.** Every per-pin edit carries
  `// custom-bsp: pinmux` on the closing brace; gpio default-state
  entries carry the same marker as a trailing comment. Re-running
  `generate` must detect and update — never duplicate.
- **`modify_pinmux.py` is unchanged from the original framework** —
  it reads its own `session.json` shim under `--kb-dir`. The shim
  is regenerated each run from the active profile + the user-facing
  sidecar. Do not hand-edit the shim; it's transient.
- **Multiple pinmux DTSI variants per Thor module SKU.** Some carrier
  pins live in a different DTSI variant than the one the carrier
  conf references. `modify_pinmux.py commit` (legacy patch-in-place
  flow) tolerates missing per-pin blocks via
  `pinmux.warnings[]` rather than failing. Surface the warning;
  point at the alternate DTSI variant.
- **Don't touch the upstream BSP at `<bsp_image.root_path>`.** All
  edits land in `<source.root_path>/Linux_for_Tegra/bootloader/` under
  the pristine + customization commit pattern.

## Available Scripts

| Script | Purpose | Arguments |
|---|---|---|
| `scripts/modify_pinmux.py` | XLSM parser + per-pin DTSI generator. Invoked via `run_script()` from Steps 3-6 with the subcommand of the current phase. | `probe \| lookup \| set-pin \| apply \| generate \| commit [...]` (see `--help`) |
| `scripts/generate_dtsi.py` | Renders pinmux/GPIO/padvoltage DTSI fragments from the bundled session state. Called by `modify_pinmux.py generate`. | `--session <path> --out-dir <dir>` |

Invoke from the skill body as a subprocess via `run_script()`:

```bash
# run_script: probe the carrier pinmux XLSM and write a session state
scripts/modify_pinmux.py probe --xlsm carrier.xlsm --session .pinmux-session.json

# run_script: render DTSI fragments from the final session state
scripts/modify_pinmux.py generate --session .pinmux-session.json --out-dir bsp_sources/pinmux/
```

## References
- [`references/procedure.md`](references/procedure.md) — full Step 1–8
  procedure prose.
- [`questions.json`](questions.json) — Q1–Q6 prompt schema consumed
  by Step 4.
- [`../../references/platform_template.yaml`](../../references/platform_template.yaml) — `documents:` block (`ref_devkit_pinmux_xls`, `custom_carrier_pinmux_xls`).
- [`../../context/bsp-customization-workflow.md`](../../context/bsp-customization-workflow.md#workflow-invariants) — overlay edit protocol (single-commit per DTSI fork).
- [`../jetson-derive-carrier/SKILL.md`](../jetson-derive-carrier/SKILL.md) — must run first; produces the pinmux / gpio / padvoltage DTSI forks this skill edits, and rewrites the carrier conf's `PINMUX_CONFIG=` / `GPIOINT_CONFIG=` / `PMC_CONFIG=` lines to point at them.
- [`../jetson-init-source/SKILL.md`](../jetson-init-source/SKILL.md) — produces the overlay tracker this skill commits into.
- [`../jetson-link-docs/SKILL.md`](../jetson-link-docs/SKILL.md) — author the profile's `documents:` block, including the pinmux XLSM bindings.
