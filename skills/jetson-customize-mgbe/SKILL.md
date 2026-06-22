---
name: jetson-customize-mgbe
description: >-
  Enable Jetson Thor 25G/10G/1G MGBE QSFP via kernel-DT overlay.
  Do NOT use for UPHY lane allocation or ODMDATA edits.
version: 0.0.1
license: "Apache-2.0"
metadata:
  data-classification: public
  author: "Jetson Team"
  tags:
    - bsp
    - phase-2
    - io
    - mgbe
    - ethernet
  domain: meta
  permissions:
    file_read:
      - "{workspace}/target-platform/"
      - "{source.root_path}/"
      - "{bsp_image.root_path}/Linux_for_Tegra/"
      - "{documents.root_path}/"
    file_write:
      - "{workspace}/target-platform/"
      - "{source.root_path}/bsp_sources/hardware/nvidia/"
    shell:
      - "dtc"
      - "fdtoverlay"
      - "cpp"
      - "git"
---

# Customize MGBE / 25G QSFP

## Overview

Thor T264 exposes mgbe0..mgbe3. On a custom carrier, the 25G QSFP cage
(or 10G / 1G fiber path) is wired to one of them through SerDes — with
or without an external MDIO PHY in front of the cage. This skill
renders the kernel-DT overlay that pairs the BPMP allocation with
kernel-side `status="okay"` + PHY plumbing on `&mgbeN`.

**Out of scope:**
- UPHY lane allocation — owned by `/jetson-customize-uphy`. Refuse if
  the chosen `uphy1-config-N` doesn't allocate the target MGBE.
- All ODMDATA tokens (`mgbeN-speed-*`, sub-node `mgbeN_status=*`) —
  owned by `/jetson-customize-uphy` in its single atomic ODMDATA
  commit. This skill MUST NOT touch `ODMDATA=`.

Output is **one commit** to the composite custom overlay `.dts` in the
`bsp_sources/` hardware repo. `/jetson-build-source` compiles the
composite to `.dtbo` and owns its Makefile + flash-conf registration.

## When to invoke

- The user says "enable 25G", "configure QSFP", "set MGBE PHY mode",
  "wire MGBE to QSFP", or asks to bring up a 10G / 1G fiber path.
- Cold boot succeeds but `ip link show mgbe<N>` reports `state DOWN`
  or `NO-CARRIER` on the configured controller, OR the controller
  never appears at all.
- `jetson-customize-uphy` ran with `uphy1-config-8` (or another config
  allocating MGBE) and you now need to bring up the per-controller
  side.

**Prerequisites:**

- Active profile selected with `reference_devkit:` (Thor) +
  `custom_carrier:` blocks.
- `<source.root_path>/Linux_for_Tegra/.git` exists
  (`/jetson-init-source`).
- `/jetson-derive-carrier` has run — carrier flash-conf fork is in
  the overlay tracker.
- `/jetson-customize-uphy` chose a UPHY config that allocates the target
  MGBE controller's lanes (`uphy1-config-8` on Thor for MGBE0..3 25G).
- Source-of-truth docs registered or supplied at prompt:
  Adaptation Guide, Module Design Guide, SoC TRM.
- **When `custom_carrier:` is present, both
  `documents.custom_carrier_schematic` AND
  `documents.custom_carrier_pinmux_xls` are REQUIRED.** Refuse the run
  if either is missing — MGBE routing on a custom carrier cannot be
  guessed. Reference-devkit-only profiles skip this check.
- `dtc` on PATH.

## Procedure

See `references/procedure.md` for the full step-by-step procedure (Steps 1–8). Summary:

1. **Resolve active target + documents.** Validate active profile, custom_carrier, overlay tracker; locate the relevant Adaptation Guide chapter and pinmap.
2. **Per-controller question loop.** AskUserQuestion driven by `questions.json` (controller, phy_mode, attach kind, I²C bus/addr, reset GPIO, compatible_list).
3. **Derive max-speed from phy_mode.** Decompile the BPMP DTB to pick sub-node vs top-level token grammar; cite the inspection in notes.
4. **Verify HSIO pins + auto-fix.** Run `pin_verifier.py` for MDC/MDIO/RESET/INT; surface mismatches and route to `/jetson-customize-pinmux`.
5. **(no ODMDATA edits.)** MGBE ODMDATA tokens are emitted by `/jetson-customize-uphy`. Step 5 only records the BPMP DTB token-form inspection (sub-node vs top-level) in `notes[]` for audit.
6. **Append composite-overlay fragments.** Write one fragment per controller into the composite custom overlay `.dts`; obey the `/* custom-bsp: mgbe:mgbe... */` marker contract; run the cpp/dtc/fdtoverlay pre-flight.
7. **(Reserved.)** Sibling-skill ordering / cross-cutting validation.
8. **Run-state sidecar + summary + next-step chain.** Write `<profile-stem>.jetson-customize-mgbe.json` and emit the one-line + table summary, then drive the downstream chain via sequential `AskUserQuestion` prompts per `references/procedure.md` Step 8. Never substitute a printed "Next step: …" line for the prompts.

## Gotchas

- **Stock Thor BPMP DTB has no `/mgbe/mgbe@N` subtree** — only
  `mgbe<N>-speed` under `/uphy`. The `mgbe<N>_status=disabled`
  sub-node token is silently rejected on these releases; the whole
  ODMDATA line is then dropped at flash time. Always decompile BPMP
  DTB (Step 3) before emitting; use the top-level dashed form
  (`mgbe<N>-speed-del` to remove, `mgbe<N>-speed-25G` to set) when
  the sub-node isn't there. Same wrong-form failure surface as
  `jetson-customize-uphy`.
- **`mdio` child needs both `#address-cells = <1>` AND
  `#size-cells = <0>`** when `phy_attach_kind=="phy"`. Missing either
  → kernel rejects `phy@<addr>` reg property at probe; MGBE never
  comes up.
- **Overlay root `compatible` must intersect live DT compatible.**
  UEFI plugin-manager filters by compatible match. A mismatched
  overlay is silently skipped — flash succeeds, MGBE stays disabled,
  no error in dmesg. Always sanity-check against
  `/proc/device-tree/compatible` on a booted reference DUT.
- **`OVERLAY_DTB_FILE` ordering is `jetson-build-source`'s problem,
  not this skill's.** This skill never touches the carrier flash
  conf. The composite custom overlay is registered (by
  `/jetson-build-source` Step 5.0a) AFTER the platform
  `*-dynamic.dtbo`, which is the correct ordering. If you find
  yourself appending `OVERLAY_DTB_FILE+=` in this skill, you're
  duplicating ownership — stop, and let the build skill do it.
- **UPHY lane allocation is `jetson-customize-uphy`'s job.** If the chosen
  `uphy1-config-N` doesn't allocate lanes for the target MGBE
  controller, BL31 SError (`fmon_update_config: detected fault
  0x80`) on cold boot. Always run `/jetson-customize-uphy` first; cite the
  chosen `uphy1-config-N` in this skill's run-summary `notes[]`.
- **Don't disable a stock-okay controller via ODMDATA alone.** Same
  rule as `jetson-customize-uphy`: `mgbe<N>_status=disabled` for a
  controller that's already disabled in BPMP DTB is a no-op the
  parser may treat as ambiguous → drops the rest of the ODMDATA
  line. Disable via the kernel-DT overlay (`status="disabled"`) only.
- **Don't touch the upstream BSP at `<bsp_image.root_path>`.** All
  edits land in the overlay tracker / `bsp_sources` mono-repo under
  the pristine + customization commit pattern.
- **JSON sidecar is structured state, not authoritative.** Same
  caveat as `jetson-customize-uphy`: ODMDATA + overlay `.dts` + two git
  commits are the device-facing outputs; the sidecar is for tooling
  and idempotency only.

## References

- [`questions.json`](references/questions.json) — Q-1..Q-8 prompt schema
  consumed by Step 2.
- [`../../scripts/pin_verifier.py`](../../scripts/pin_verifier.py)
  — shared HSIO pin verifier (Step 4).
- [`../../references/platform_template.yaml`](../../references/platform_template.yaml)
  — `documents:` block consumed by Step 1.
- [`../../context/bsp-customization-workflow.md`](../../context/bsp-customization-workflow.md#commit-batching-in-the-overlay-tracker)
  — overlay edit protocol (batched pristine + customization commit).
- [`../jetson-customize-uphy/SKILL.md`](../jetson-customize-uphy/SKILL.md) — sibling
  skill that owns UPHY lane allocation. Must run before this skill to
  set `uphy1-config-N` for MGBE-allocated configurations.
- [`../jetson-customize-pinmux/SKILL.md`](../jetson-customize-pinmux/SKILL.md) —
  sibling skill invoked by Step 4 (with operator confirmation) to
  fix pin SFIO mismatches.
- [`../jetson-derive-carrier/SKILL.md`](../jetson-derive-carrier/SKILL.md)
  — must run first; produces the carrier flash-conf fork edited in
  Step 5 and the carrier base overlay this skill orders after.
- [`../jetson-init-source/SKILL.md`](../jetson-init-source/SKILL.md) —
  produces the overlay tracker + bsp_sources repo this skill commits
  into.
