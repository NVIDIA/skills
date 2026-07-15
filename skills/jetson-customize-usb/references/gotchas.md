# jetson-customize-usb — Gotchas

- **Three-place lockstep is mandatory.** Lane status + port status +
  host xHCI `phys`/`phy-names` MUST flip together. Asymmetric output
  fingerprint: empty `lsusb` with `xudc` registering and USB-eth
  reachable at 192.168.55.1 = host xHCI bailed.
- **Rule A — lane + port pairing.** Mismatch then `no port found` or
  `Requested PHY is disabled` then host xHCI dead.
- **Rule B — companion cascade.** Disabling a USB2 without its SS
  companion then `tegra-xusb: failed to enable PHYs: -19`. Surface
  explicitly via `AskUserQuestion` before rendering — never silent.
- **Host `phys` element ordering MUST be preserved** — only elide
  disabled entries; never reorder kept ones. xHCI iterates by index
  and binds the wrong port if order changes. `fdtoverlay` exit code
  does NOT catch reorders — Step 5d invariant (3) does.
- **On-carrier hub detection.** Schematic TOC `USB 3.x HUB` block then
  SS+USB2 pair into one hub IC; disabling the pair drops every
  downstream receptacle. Surface in Step 3 — never silent scope creep.
- **Tegra platform invariant — only `usb2-0` is OTG-capable.** Hard-
  pin host on all non-`usb2-0` ports; `xudc` only attaches to `usb2-0`.
- **`xudc` fragment is conditional.** Emit only when at least one
  port has mode in `{device, otg}`.
- **Host xHCI / `tegra-xudc` lack `__symbols__` labels.** Use
  `target-path = "/bus@0/usb@<addr>"`. `&tegra_xhci` / `&tegra_xudc`
  produce `FDT_ERR_NOTFOUND` at `fdtoverlay`.
- **Raw integer phandles in the host phys override.** Lane nodes have
  no `__symbols__` entries; dtc warns `phys_property: cell 0 is not
  a phandle reference` — benign.
- **`vbus-supply = <&regulator-fixed-…>` requires the regulator
  parent node to exist.** Synthesizing the ref without the parent then
  port boots but VBUS never asserts.
- **UPHY lane allocation is `jetson-customize-uphy`'s job.** Refuse
  to commit an SS-enable until the uphy run-state shows the lane
  allocated.
- **No ODMDATA edit, no upstream BSP edits.** Kernel-DT overlay alone
  owns enable/disable; all edits land in the overlay tracker +
  `bsp_sources` mono-repo under the pristine + customization pattern.
