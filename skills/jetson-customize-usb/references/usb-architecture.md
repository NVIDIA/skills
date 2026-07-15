# Tegra USB architecture (T234 Orin / T264 Thor)

## Block model

```
        +-----------------+        +--------------------+
        |   tegra_xusb    |        |       xudc         |
        |  (host xHCI)    |        |   (device gadget)  |
        +--------+--------+        +---------+----------+
                 |                           |
                 +---------- shared ---------+
                            xusb_padctl
                                 |
        +------------------------+-------------------------+
        |                        |                         |
   USB2 padctl              USB3 SS padctl            UPHY lane pool
   (USB2_DP/DN x N)         (per-port phy)            (T264: 8 lanes;
                                                       T234: 4 lanes)
```

## USB2 (high-speed)

* Up to **4 fixed USB2 OTG ports** on Thor (`usb2-0`..`usb2-3`).
* The differential pair (DP/DN) is a hardwired *HSIO* lane — **not** in
  customer pinmux. We detect it from carrier pinmap evidence nets named
  `USB<N>_DP` / `USB<N>_DN`.
* Per-port mode (`host` / `device` / `otg`) is selectable in DT.

## USB3 (SuperSpeed / SS)

* Thor exposes up to **4 USB3 SS instances** (`usb3-0`..`usb3-3`); each
  consumes one (gen1/gen2) or two (gen2x2) UPHY lanes from the shared
  pool.
* The SS pair is also HSIO; we detect it from carrier pinmap nets named
  `USB3_SS<N>_TX_[NP]` / `USB3_SS<N>_RX_[NP]`.

## UPHY lane assignment

UPHY lanes are a *shared resource* between USB3, PCIe, UFS, and XFI. The
carrier's choice is encoded in the **MB2 BCT misc DTS** as a cell-list
property (canonically `nvidia,usb_lane_map`). Each cell is the UPHY lane
index assigned to the SS instance at that array offset:

```dts
nvidia,usb_lane_map = <0x04 0x05 0xff 0xff>;
                        ssN  ss1  ss2  ss3
                        ↑    ↑
                        lane 4  lane 5
```

`0xff` means "this SS instance is unused" (so its UPHY lane stays free for
PCIe/UFS).

## Ancillary pins (MPIO)

These *are* in customer pinmux (mb1 BCT pinmux DTSI):

| Role | Direction | Typical electrical |
|---|---|---|
| `VBUS_EN` | output | rsvd0 / GPIO, no pull, drives load-switch enable |
| `OC` | input | rsvd0 / GPIO, pull-up (active-low fault) |
| `CC1` / `CC2` | input | rsvd0 / GPIO, no pull, tristate (Type-C only) |

## What flows where

| Customization | File patched | By |
|---|---|---|
| VBUS_EN / OC / CC pinmux | `tegra*-mb1-bct-pinmux-*-<carrier>.dts` (the included `.dtsi`) | `modify-usb pinmux` |
| UPHY lane map | `tegra*-mb2-bct-misc-*-<carrier>.dts` | `modify-usb uphy` |
| Per-port host/device mode + vbus-supply | `tegra*-overlay-<carrier>-usb.dts` (kernel-DT overlay) | `modify-usb apply` |
| Carrier `.conf` overlay list | `jetson-<plat>-devkit-<carrier>.conf` | `modify-usb commit` |

## Devkit-proven vs carrier-only

The plan stage classifies each detected port:

* **devkit-USB-proven** — same `USB<N>_DP/DN` (or `USB3_SS<N>_TX/RX`) net
  appears on the reference devkit pinmap. The host BSP shipping for that
  port is known to work on the reference, so the carrier just needs the
  ancillary pins and (for SS) the lane map.
* **carrier-only** — the SS lanes / USB2 pair are only routed on the
  carrier. Functional but unproven; the user is on the hook for sign-off.

## Bandwidth notes (Thor)

* USB3.2 Gen2x2 (20 Gbps) requires **two adjacent UPHY lanes** mapped to
  the same SS instance. The skill records this as a lane *pair* in
  `ss_lane_map` when `max_speed == "usb3.2"`.
* USB3.0/3.1 require **one** UPHY lane per SS instance.
