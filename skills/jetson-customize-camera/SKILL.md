---
name: jetson-customize-camera
description: >-
  Enable MIPI/GMSL camera sensors on a Jetson Thor or Orin custom
  carrier by rendering a kernel-DT overlay from the in-tree sensor
  DTSI. Do NOT use for UPHY lane allocation or ODMDATA edits.
version: 0.0.1
license: "Apache-2.0"
metadata:
  data-classification: public
  author: "Jetson Team"
  tags:
    - bsp
    - phase-2
    - io
    - camera
    - csi
  domain: meta
---

# Customize camera (CSI / MIPI / GMSL sensor bring-up)

## Overview

Tegra264 (Thor) and Tegra234 (Orin) expose a single `tegra-capture-vi`
controller fronted by NVCSI and a fixed set of CSI ports. Camera
bring-up is:

1. **Sensor selection** — picked from the set NVIDIA ships in-tree
   `.dtsi` references for on the active platform.
2. **Carrier + module support check** — verified against the Camera
   Development Guide, Adaptation Guide §Camera, carrier schematic,
   Module TRM, and carrier pinmap.
3. **Wiring** — derived from the in-tree
   `tegra<soc>-camera-<sensor>*.dtsi` when one exists (**the DTSI IS
   the wiring source of truth**); captured per-sensor from the user
   when the sensor is custom.
4. **Kernel-DT overlay** — cpp-expand the in-tree DTSI, extract its
   `fragment@N` body, append into the composite custom overlay
   `.dts` for the active target (per
   [`../../references/bsp-customization-kernel-dtb.md`](../../references/bsp-customization-kernel-dtb.md)),
   verify the composite with `fdtoverlay`.
   `/jetson-build-source` compiles the composite and owns the
   carrier conf's `OVERLAY_DTB_FILE+=` registration.

**Agentic, not table-driven** — sensor list is built at runtime by
globbing in-tree per-sensor dtbos. No `_THOR_CAMERAS` dict, no
`questions.json`, no Python renderer in the question path.

**No ODMDATA edit** — cameras don't consume UPHY lanes (CSI is a
separate PHY pool). The skill emits only a kernel-DT overlay; the
ODMDATA line in the carrier conf is untouched by this skill.

The output is **one commit**:
- Camera `fragment@N` block (plus `jetson-header-name` on the
  composite root if not already present) appended to the composite
  custom overlay `.dts` per
  [`../../references/bsp-customization-kernel-dtb.md`](../../references/bsp-customization-kernel-dtb.md)
  → committed to the `bsp_sources/` hardware repo.
  `/jetson-build-source` compiles the composite to `.dtbo` and
  owns its Makefile + flash-conf registration.

## When to invoke

- The user says "enable camera", "configure CSI", "wire a Hawk /
  Owl / IMX sensor", "MIPI camera", "GMSL camera", or asks to bring
  up `tegra-capture-vi` / NVCSI on a custom carrier.
- Flash boots but `v4l2-ctl --list-devices` shows no
  `tegra-capture-vi` channels, OR sensor enumeration on a fresh
  daughter-card needs to be confirmed.
- A sensor was previously enabled and the user wants to add another
  (multi-sensor bring-up).

**Prerequisites:**

- Active profile with `reference_devkit:` + `custom_carrier:` blocks.
- `<source.root_path>/Linux_for_Tegra/.git` exists
  (`/jetson-init-source`).
- `/jetson-derive-carrier` has run — the carrier flash-conf fork is
  in the overlay tracker.
- `<source.root_path>/bsp_sources/hardware/nvidia/<chip-dir>/nv-public/overlay/`
  exists and contains the in-tree per-sensor `.dtsi` files (sourced
  by `/jetson-init-source`'s Branch A archive extract).
- `<source.root_path>/bsp_sources/kernel/kernel-noble/include/dt-bindings/`
  contains the macro headers cpp needs (`source_sync.sh` may need to
  run if Branch B was used — see Step 5a.i below).
- Source-of-truth docs registered or supplied at prompt:
  Camera Development Guide (in `bsp_developer_guide` mirror or
  separate path), Adaptation Guide §Camera, carrier schematic, SoC
  TRM, Module Design Guide.
- `dtc`, `cpp`, `fdtoverlay` on PATH.

## Procedure

Detailed step-by-step procedure (Steps 1–7, with all tables, code
blocks, and gates) lives in
[`references/procedure.md`](references/procedure.md). Summary:

1. **Step 1** — Resolve active target + open source-of-truth docs.
2. **Step 2** — Enumerate supported sensors by globbing in-tree
   per-platform camera dtbos; classify as DPHY-direct / GMSL /
   custom. Never invent sensors.
3. **Step 3 / 3a** — Cross-check carrier + module support against
   DTSI, Camera Development Guide, Adaptation Guide §Camera, SoC TRM,
   Module Design Guide, schematic, and carrier pinmap. Render the
   wiring table FIRST, then issue the confirm-or-customize gate.
4. **Step 4** (custom path only) — Batched per-sensor wiring
   questions auto-filled from the carrier pinmap.
5. **Step 5** — Append exactly ONE `/* custom-bsp: camera:<sensor> */`
   fragment to the composite custom overlay `.dts` (see
   [`../../references/bsp-customization-kernel-dtb.md`](../../references/bsp-customization-kernel-dtb.md)).
   Clone path cpp-expands the in-tree DTSI; custom path splices Step-4
   answers + mode tables in-place. Idempotently set
   `jetson-header-name` on the composite root. Verify with
   `dtc` + `fdtoverlay` (pre-compile single-fragment gate;
   post-compile deep-tree uniqueness gate). Commit via the
   workflow's commit-message preview gate.
6. **Step 6** — Verify ancillary CAM pin SFIOs (`cam_i2c_*`,
   `extperiph<m>_clk`, reset/PWDN/PWR_EN GPIOs) via
   `pin_verifier.py`; route mismatches to `/jetson-customize-pinmux`.
7. **Step 7** — Atomic-write run-state JSON sidecar at
   `<workspace>/target-platform/<profile-stem>.jetson-customize-camera.json`
   and emit the headline, then drive the downstream next-step chain via
   sequential `AskUserQuestion` prompts per `references/procedure.md`
   Step 7. **The chain is a documented workflow gate, not a clarifying
   question — auto-mode does NOT exempt it.** Never substitute a
   printed "Next step: …" line for the prompts.

## Gotchas

- **Dual-fragment trap.** Contribute exactly ONE camera-tagged
  `fragment@N` to the composite. A second one carrying status
  overrides triggers dtc deep-merge → duplicate sibling subtrees
  (e.g. two `tca9546@70`) → runtime first-match drops the dtsi-
  supplied deep tree → camera silently doesn't enumerate. Gate on
  this skill's marker only (Step 5c).
- **Composite root `compatible` is owned globally, not by this
  skill.** Don't widen from any in-tree per-sensor dtbo's
  `compatible` (devkit-SKU-gated). Fix the composite root if needed.
- **`jetson-header-name` from any in-tree per-sensor dtbo.** Fixed,
  carrier-agnostic; read once, paste onto the metadata root.
- **DO NOT also append the in-tree per-sensor dtbo to
  `OVERLAY_DTB_FILE`.** Registering both your rendered overlay AND
  the in-tree `tegra<soc>-p3971-camera-<sensor>-overlay.dtbo`
  produces a phantom subdev bind that bricks camera enumeration.
- **Stub overlay is a known footgun.** Committing
  `tegra-capture-vi { status="okay"; num-channels=<N>; }` with no
  ports / sensor / nvcsi body bricks the camera (`all channel init
  failed`). Splice the FULL sensor body via cpp + dtc.
- **Sensor mode tables must be spliced, never hand-authored.**
  `mode<N>`, `sensor_modes`, `pixel_phase` — copy verbatim from the
  closest in-tree DTSI.
- **`camera_common_regulator_get (null) ERR: -EINVAL`** = missing
  `avdd-reg` / `iovdd-reg` / `dvdd-reg` strings — splice the FULL
  sensor body; always-on rails fall back to dummy regulator.
- **External `&label` refs must exist in base DTB's `__symbols__`.**
  Use `target-path = "/tegra-capture-vi"` when the label is absent;
  `fdtoverlay` exits non-zero with `FDT_ERR_NOTFOUND` otherwise.
- **`cpp` failure on `dt-bindings/gpio/gpio.h: No such file`** =
  L4T source tree isn't staged. Re-run `/jetson-init-source` (Branch
  B's `source_sync.sh` fetches the headers). Never fabricate the
  macro expansion.
- **No ODMDATA edit, no flash-conf edit.** Camera doesn't consume
  UPHY lanes. The carrier conf's `ODMDATA="..."` is untouched.
  `OVERLAY_DTB_FILE+=` is owned by `/jetson-build-source` Step
  5.0a — this skill never touches the carrier flash conf.
- **Don't touch the upstream BSP at `<bsp_image.root_path>`.** All
  edits land in `<source.root_path>/Linux_for_Tegra/` (overlay
  tracker) and `<source.root_path>/bsp_sources/` (overlay `.dts`)
  under the pristine + customization commit pattern.

## References

- [`references/procedure.md`](references/procedure.md) — full
  step-by-step Steps 1–7 procedure (extracted from this SKILL.md).
- [`references/csi-dt-bindings.md`](references/csi-dt-bindings.md) —
  CSI / nvcsi / vi DT binding reference notes.
- [`references/overlay-template.md`](references/overlay-template.md) —
  guidance on the metadata-root + clone-body overlay shape.
- [`references/camera-overlay-templates/`](references/camera-overlay-templates/)
  — starter `.dts.tmpl` templates: `dphy-direct.dts.tmpl`,
  `gmsl-serdes.dts.tmpl`.
- [`../../scripts/pin_verifier.py`](../../scripts/pin_verifier.py)
  — shared HSIO pin verifier (Step 6).
- [`../../references/platform_template.yaml`](../../references/platform_template.yaml)
  — `documents:` block consumed by Step 1.
- [`../../context/bsp-customization-workflow.md`](../../context/bsp-customization-workflow.md#workflow-invariants)
  — overlay edit protocol.
- [`../jetson-customize-pinmux/SKILL.md`](../jetson-customize-pinmux/SKILL.md) —
  sibling skill auto-invoked by Step 6 to fix HSIO pin SFIO
  mismatches (CAM I²C, MCLK, reset GPIOs).
- [`../jetson-derive-carrier/SKILL.md`](../jetson-derive-carrier/SKILL.md)
  — must run first; produces the carrier base overlay (the
  `*-dynamic.dtbo`) this skill's composite stacks after.
- [`../jetson-init-source/SKILL.md`](../jetson-init-source/SKILL.md) —
  produces the overlay tracker + `bsp_sources` repo (with the
  `hardware/nvidia/<chip-dir>/` per-sensor DTSI tree) this skill
  reads and commits into.
