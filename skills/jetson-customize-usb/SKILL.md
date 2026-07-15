---
name: jetson-customize-usb
description: Enable/disable Jetson USB2/USB3 SS ports via kernel-DT overlay. Do NOT use for UPHY lane allocation or ODMDATA edits.
version: 0.0.1
license: "Apache-2.0"
metadata:
  data-classification: public
  author: "Jetson Team"
  tags:
    - bsp
    - phase-2
    - io
    - usb
  domain: meta
---

# Customize USB (per-port enable / disable / role)

## Purpose

Enable, disable, or change the role of USB2 / USB3 SS ports on a
Jetson Thor (Tegra264) or Orin (Tegra234) custom carrier. Captures
per-port wiring (role, max speed, VBUS-EN / OC GPIOs, CC1/CC2 GPIOs
for Type-C, USB3 SS UPHY lane), resolves the SS to USB2 companion
graph from the in-tree DTB, then renders a self-contained kernel-DT
overlay that flips every port action in three places in lockstep
(lane status, port status, host xHCI `phys` + `phy-names`).

UPHY lane allocation belongs to `jetson-customize-uphy`. No ODMDATA
edit. Output is one commit to the composite custom overlay `.dts` in
the `bsp_sources/` hardware repo.

## Prerequisites

- Active profile with `reference_devkit:` + `custom_carrier:` blocks.
- `<source.root_path>/Linux_for_Tegra/.git` exists
  (`/jetson-init-source`).
- `/jetson-derive-carrier` has run — carrier flash-conf fork in the
  overlay tracker.
- `/jetson-customize-uphy` has run when any enabled USB3 SS port
  needs a non-stock UPHY lane allocation. Its JSON sidecar at
  `<workspace>/target-platform/<profile-stem>.jetson-customize-uphy.json`
  is consulted for SS lane allocation.
- Source-of-truth docs: Adaptation Guide §"Port the Universal Serial
  Bus", Module Design Guide §USB, SoC TRM (xusb block).
- **When `custom_carrier:` is present, both
  `documents.custom_carrier_schematic` AND
  `documents.custom_carrier_pinmux_xls` are REQUIRED.** Refuse the run
  if either is missing — per-port routing (VBUS-EN / OC / CC GPIOs, SS
  lane wiring, hub fan-out) on a custom carrier cannot be guessed.
  Reference-devkit-only profiles skip this check.
- `dtc`, `fdtoverlay` on PATH.

## Overview

USB on Tegra spans three IP surfaces: the `xusb_padctl` block (USB2
OTG + USB3 SS PHYs), the `tegra-xusb` xHCI host controller, and an
optional `tegra-xudc` device controller attached to the single
OTG-capable USB2 port (`usb2-0`).

**A per-port flip MUST touch three kernel-DT places in lockstep.**
Anything less crashes the host xHCI probe and leaves `lsusb` empty
on every port (collateral damage to stock-okay ports):

| # | Place | Path | What it controls |
|---|---|---|---|
| 1 | **Lane (PHY provider)** | `xusb_padctl/pads/usb<2\|3>/lanes/usb<2\|3>-N` | SS / OTG PHY hardware-binding. `status="disabled"` then lane stops providing a PHY. |
| 2 | **Port (controller-binding)** | `xusb_padctl/ports/usb<2\|3>-N` | Per-port mode (host/device/otg), companion link, VBUS / OC / CC pin refs. `status="disabled"` then port removed from user-facing topology. |
| 3 | **Host xHCI phys-list** | `bus@0/usb@<addr>.phys` + `.phy-names` | Array of phandles + names the xHCI driver iterates. A ref to a disabled PHY returns `-ENODEV` and aborts the whole host probe. |

NVIDIA's stock-disabled `usb3-3` in the Thor base DTB is the canonical
pattern — all three places flipped in lockstep.

**Two extra rules ride on top of the three-place pattern:**

- **Rule A — lane + port pairing.** Lane (place 1) and matching port
  (place 2) MUST flip together.
- **Rule B — companion cascade.**
  `xusb_padctl/ports/usb3-N.nvidia,usb2-companion` references a USB2
  port phandle. Disabling that USB2 without cascading to its SS
  companion then `tegra-xusb: failed to enable PHYs: -19`.

**Agentic, not table-driven** — every port, controller, lane,
companion link, phandle, and `__symbols__` lookup is resolved at
runtime from docs + DTB + carrier pinmap + schematic.

## When to invoke

- The user says "enable USB", "disable USB hub", "configure USB3 SS",
  "set USB role", "wire VBUS-EN", "tegra-xusb / xudc / dr_mode", or
  asks to bring up / take down a USB controller on a custom carrier.
- A USB receptacle on the carrier doesn't enumerate after flash, OR
  collateral USB damage (`lsusb` empty after a previous
  jetson-customize-usb attempt) needs to be fixed.
- `jetson-customize-uphy` re-allocated UPHY lanes affecting USB3 SS
  ports and per-port DT now needs to follow.

## Procedure (summary)

The full step-by-step procedure lives in `references/procedure.md`.

1. **Step 1** — resolve active target + open source-of-truth docs.
2. **Step 2** — build the USB topology + companion graph from the
   in-tree DTB.
3. **Step 3** — `AskUserQuestion` for port(s) to enable / disable;
   surface companion cascade + on-carrier hub fan-out explicitly.
4. **Step 4** — per-port verify (module + carrier + UPHY lane) and
   capture wiring (VBUS-EN / OC / CC GPIOs via `pin_verifier.py`).
5. **Step 5** — render the kernel-DT overlay using the three-place
   pattern, append fragments (`usb:padctl`, `usb:xhci`, optional
   `usb:xudc`) to the composite custom overlay `.dts`, run
   `fdtoverlay` + the three post-merge invariants, commit to
   `bsp_sources/`.
6. **Step 6** — write run-state JSON sidecar (shape in
   `references/run-state-sidecar.md`), emit headline, then drive the
   downstream next-step chain via sequential `AskUserQuestion` prompts
   per `references/procedure.md` Step 6. Never substitute a printed
   "Next step: …" line for the prompts.

See `references/gotchas.md` for the load-bearing failure modes.

## Limitations

- Owns kernel-DT overlay only. ODMDATA does not expose a per-port
  USB `status` knob; do not edit it.
- Does NOT allocate UPHY lanes — `jetson-customize-uphy` owns that.
  Refuse to commit an SS-enable until uphy run-state shows the lane
  allocated.
- Does NOT directly patch the pinmux DTSI — routes SFIO mismatches
  to `/jetson-customize-pinmux set-pin`.
- Does NOT compile the `.dtbo` or register `OVERLAY_DTB_FILE+=` —
  `/jetson-build-source` owns build + flash-conf registration.
- Tegra platform invariant: only `usb2-0` is OTG-capable; `xudc`
  attaches there only. All other USB2 ports and all USB3 SS ports
  are host-only.

## Troubleshooting

- **Empty `lsusb` on every port, USB-eth at 192.168.55.1 still up:**
  host xHCI bailed; three-place lockstep was broken. Inspect merged
  DTB; verify post-merge invariants in `references/procedure.md`
  Step 5d.
- **`tegra-xusb: failed to enable PHYs: -19`:** companion cascade
  (Rule B) violated — a USB2 was disabled without its SS companion.
- **`no port found` or `Requested PHY is disabled`:** Rule A
  violated — lane status and port status are mismatched.
- **`FDT_ERR_NOTFOUND` from `fdtoverlay`:** a fragment used
  `target = <&label>` for a node whose label is not in
  `__symbols__` (typical for host xHCI / `tegra-xudc`). Switch to
  `target-path = "/bus@0/usb@<addr>"`.
- **dtc warning `phys_property: cell 0 is not a phandle reference`:**
  benign; expected when using raw integer phandles in the host phys
  override.
- **Port boots but VBUS never asserts:** `vbus-supply` references a
  regulator parent node that does not exist. Ensure the fixed
  regulator node is present before referencing it.
- **xHCI binds the wrong port at boot:** host `phys` element order
  was not preserved. Only elide disabled entries; never reorder
  kept ones.

## References

- `references/procedure.md` — full step-by-step procedure.
- `references/gotchas.md` — load-bearing failure modes.
- `references/run-state-sidecar.md` — run-state JSON shape.
- `references/usb-architecture.md` — Tegra USB IP architecture
  notes (`xusb_padctl`, `tegra-xusb`, `xudc`).
- `references/usb-dt-bindings.md` — USB DT binding cheatsheet
  (lane / port / phys-list shapes).
- `../../scripts/pin_verifier.py` — shared HSIO pin verifier.
- `../../references/platform_template.yaml` — `documents:` block
  consumed by Step 1.
- `../../context/bsp-customization-workflow.md` — overlay edit
  protocol.
- `../../references/bsp-customization-kernel-dtb.md` — composite
  overlay append protocol.
- `../jetson-customize-uphy/SKILL.md` — sibling skill that owns
  UPHY lane allocation.
- `../jetson-customize-pinmux/SKILL.md` — sibling skill for
  VBUS-EN / OC / CC SFIO fixes.
- `../jetson-derive-carrier/SKILL.md` — must run first.
- `../jetson-init-source/SKILL.md` — produces the overlay tracker
  + `bsp_sources` repo.
