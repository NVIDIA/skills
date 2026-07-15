---
name: jetson-customize-pcie
description: >-
  Per-controller PCIe enable / disable / lanes / link-speed for a
  Jetson Thor or Orin custom carrier via ODMDATA + kernel-DT overlay.
  Do NOT use for UPHY lane allocation or endpoint-mode bring-up.
version: 0.0.1
license: "Apache-2.0"
metadata:
  data-classification: public
  author: "Jetson Team"
  tags:
    - bsp
    - phase-2
    - io
    - pcie
  domain: meta
---

# Customize PCIe (per-controller status / lanes / speed)

## Overview

PCIe on Tegra264 (Thor, `pcie@C0..C5`) and Tegra234 (Orin,
`pcie@C0..C10`) is split across multiple controllers that share the
UPHY lane pool with USB3 / MGBE / UFS. Each controller's runtime
behavior is determined by **two surfaces, both required**:

| Surface | Target | Authoritative for |
|---|---|---|
| ODMDATA `pcie@N_status=…` (+ `pcie@N_max-link-speed`, `pcie@N_pcie-mode`, `pcie@N_clk-scheme`, `pcie-cN-endpoint-enable`) | `/pcie/pcie@N` in BPMP DTB | UPHY lane power, refclk gating, controller-side power rails |
| Kernel-DT overlay on `&pcieN` | `/bus@0/pcie@<addr>` in kernel DTB | Kernel probe, lane width, link speed, RC/EP mode |

Skipping the kernel overlay on a disable lets the kernel probe a
powered-down PHY (link timeouts in dmesg). Skipping the ODMDATA token
on a disable leaves BPMP holding the PHY hot.

**Agentic, not table-driven** — no controller table, no
`questions.json`. Every controller, lane width, schematic-routed
receptacle, and authoritative DT node address is discovered at runtime
from the docs + DTB + carrier pinmap.

The output is a **kernel-DT overlay commit only**. Per-controller
`fragment@N` blocks are appended to the composite custom overlay
`.dts` per
[`../../references/bsp-customization-kernel-dtb.md`](../../references/bsp-customization-kernel-dtb.md)
and committed to the `bsp_sources/` hardware repo.
`/jetson-build-source` compiles the composite to `.dtbo` and owns its
Makefile + flash-conf registration.

**This skill MUST NOT edit `ODMDATA="..."`.** All ODMDATA tokens
(`pcie@N_status=…`, `pcie@N_max-link-speed`, `pcie@N_pcie-mode`,
`pcie@N_clk-scheme`, `pcie-cN-endpoint-enable`, plus the
`uphyX-config-N` surface tokens and `UPHY_CONFIG=""` clear) are
emitted by `/jetson-customize-uphy` in a single atomic commit on the
carrier flash-conf fork. The allocation table this skill consumes
from the UPHY sidecar already tells the operator which controllers
are `okay` / `disabled` / per-lane sized; this skill only translates
that table into kernel-DT overlay fragments and verifies that the
overlay agrees with the ODMDATA already committed by `customize-uphy`
(consistency check in Step 8 — disagreement is reported, not silently
fixed).

## When to invoke

- The user says "configure PCIe", "enable PCIe controller", "set PCIe
  num-lanes", "change PCIe link speed", or asks to flip a
  `pcie@N_status` token.
- A specific PCIe slot or M.2 receptacle doesn't enumerate after
  flash, OR the link trains at the wrong width / speed.
- `jetson-customize-uphy` ran and re-allocated lanes across PCIe controllers
  (e.g. switched from `uphy0-config-7` to `uphy0-config-6` enabling
  PCIe C3); the per-controller side now needs to be brought up.
- `jetson-customize-mgbe` reports the QSFP path is wired but the kernel
  doesn't probe its PCIe-side companion (rare; XFI configurations).

**Prerequisites:**

- Active profile with `reference_devkit:` + `custom_carrier:` blocks.
- `<source.root_path>/Linux_for_Tegra/.git` exists
  (`/jetson-init-source`).
- `/jetson-derive-carrier` has run — carrier flash-conf fork is in
  the overlay tracker.
- `/jetson-customize-uphy` has run — its JSON sidecar at
  `<workspace>/target-platform/<profile-stem>.jetson-customize-uphy.json`
  drives the per-controller `enable` decision.
- Source-of-truth docs registered or supplied at prompt: Adaptation
  Guide, Module Design Guide, SoC TRM.
- **When `custom_carrier:` is present, both
  `documents.custom_carrier_schematic` AND
  `documents.custom_carrier_pinmux_xls` are REQUIRED.** Refuse the run
  if either is missing — routing on a custom carrier cannot be guessed.
  Reference-devkit-only profiles skip this check.
- `dtc` on PATH.

## Procedure (summary)

Full step-by-step walkthrough lives in
[`references/procedure.md`](references/procedure.md). High-level flow:

1. Resolve active target + open source-of-truth documents (incl.
   `<carrier-pinmap>`, `<ref-dtb>`, `<uphy-state>`). Refuse if
   `<uphy-state>` is missing.
2. Diff PCIe topology — devkit vs custom carrier — by decompiling
   `<ref-dtb>` and grepping the schematic for `PEX<N>_*` net labels.
3. `AskUserQuestion` (multiSelect) — which controllers to customize.
4. Per-controller verification: pinmap + schematic + `pin_verifier.py`
   for `PE<N>_CLKREQ_L`, `PE<N>_RST_L`, optional `PE<N>_WAKE_L`.
5. Auto-derive per-controller plan (`enable` from `<uphy-state>`,
   `lanes` / `speed` from Adaptation Guide, `mode` hard-pinned to
   `"rc"`) → mandatory confirm-or-customize gate.
6. Append per-controller `fragment@N` blocks (marker
   `/* custom-bsp: pcie:pcie@<addr> */`) to the composite custom
   overlay `.dts` in `bsp_sources/`. Pre-flight `dtc` + `fdtoverlay`.
   Commit via the workflow's preview gate.
   **Do not edit `ODMDATA`** — `/jetson-customize-uphy` already emitted
   `pcie@N_status=…`, `pcie@N_max-link-speed`, `pcie@N_pcie-mode`,
   `pcie@N_clk-scheme`, and `pcie-cN-endpoint-enable` in its single
   atomic ODMDATA commit. This skill only translates the per-controller
   plan into kernel-DT overlay fragments.
7. (Step folded into Step 6 — overlay-only emission.)
8. Cross-check ODMDATA vs overlay consistency. On a contradictory
   row, **stop and ask the user how to recover the two commits.
   Never run `git reset --hard` autonomously.**
9. Write run-state JSON sidecar at
   `<workspace>/target-platform/<profile-stem>.jetson-customize-pcie.json`
   + summary, then drive the downstream next-step chain via sequential
   `AskUserQuestion` prompts per `references/procedure.md` Step 9.
   Never substitute a printed "Next step: …" line for the prompts.

## Limitations

- **Mode hard-pinned to RC.** Endpoint mode is only emitted when the
  operator passes `mode_override="ep"` in Step 5c.
- **`enable` is derived, not asked.** UPHY-allocated controllers are
  mandatorily `okay`; non-allocated are mandatorily `disabled`.
- **No upstream BSP edits.** Output lands in `Linux_for_Tegra/` +
  `bsp_sources/` only.
- **Pre-flight overlay merge is a sanity check**, not the production
  build. `/jetson-build-source` is authoritative.
- **Flash-conf overlay registration is out of scope.** Owned by
  `/jetson-build-source` Step 5.0a.

## Troubleshooting

- **`<uphy-state>` missing** → run `/jetson-customize-uphy` first.
- **Slot doesn't enumerate after flash** → check `dmesg | grep pcie`;
  re-verify ODMDATA `pcie@<N>_status=okay` and the overlay fragment
  agree (Step 8 table in
  [`references/procedure.md`](references/procedure.md)).
- **Link trains at wrong width** → confirm UPHY config in
  `<uphy-state>` allocates the expected lane count; the kernel
  fragment's `num-lanes` must match.
- **`compatible` mismatch** → fix the composite root, not the
  fragment. UEFI plugin-manager silently skips on mismatch.
- **Contradictory ODMDATA-vs-overlay row** → ask the user; do not
  auto-`git reset --hard`. See gotchas.
- **Common pitfalls** — see
  [`references/gotchas.md`](references/gotchas.md) (RC pinning, node-
  address sourcing, stock-disabled controllers, intra-file handoff
  with `jetson-customize-uphy`).

## References

- [`references/procedure.md`](references/procedure.md) — full nine-
  step procedure (topology diff, plan derivation, overlay append,
  ODMDATA cross-check, sidecar).
- [`references/gotchas.md`](references/gotchas.md) — failure modes
  + invariants (RC pinning, address sourcing, BPMP handoff).
- [`../../scripts/pin_verifier.py`](../../scripts/pin_verifier.py)
  — shared HSIO pin verifier (Step 4).
- [`../../references/platform_template.yaml`](../../references/platform_template.yaml)
  — `documents:` block consumed by Step 1.
- [`../../context/bsp-customization-workflow.md`](../../context/bsp-customization-workflow.md#workflow-invariants)
  — overlay edit protocol + commit message preview gate.
- [`../../references/bsp-customization-kernel-dtb.md`](../../references/bsp-customization-kernel-dtb.md)
  — composite overlay filename / skeleton / append protocol.
- [`../jetson-customize-uphy/SKILL.md`](../jetson-customize-uphy/SKILL.md)
  — sibling skill that owns UPHY lane allocation; its sidecar drives
  the per-controller `enable` decision.
- [`../jetson-customize-pinmux/SKILL.md`](../jetson-customize-pinmux/SKILL.md)
  — sibling skill invoked by Step 4 (with operator confirmation) to
  fix HSIO pin SFIO mismatches.
- [`../jetson-customize-mgbe/SKILL.md`](../jetson-customize-mgbe/SKILL.md)
  — sibling for MGBE controllers; shares the two-surface (ODMDATA
  + overlay) pattern.
- [`../jetson-derive-carrier/SKILL.md`](../jetson-derive-carrier/SKILL.md)
  — must run first; produces the carrier flash-conf fork edited in
  Step 6.
- [`../jetson-init-source/SKILL.md`](../jetson-init-source/SKILL.md)
  — produces the overlay tracker + bsp_sources repo this skill
  commits into.
