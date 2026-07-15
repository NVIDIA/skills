# Tegra USB DT bindings used by /jetson-modify-usb

The overlay produced by `modify-usb apply` targets three nodes by phandle. The skill never declares the parent node — it only patches per-port mode and synthesises the missing bits.

## `&xusb_padctl`

```dts
&xusb_padctl {
    status = "okay";
    pads {
        usb2-N { mode = "host" | "device" | "otg"; };
        usb3-N { nvidia,usb2-companion = <N>; maximum-speed = "..."; };
    };
    ports {
        ... same shape, role-only ...
    };
};
```

`maximum-speed` accepts `"usb2"`, `"usb3.0"`, `"usb3.1"`, `"usb3.2"`. On Thor (T264), `"usb3.2"` engages Gen2x2 — requires a paired UPHY lane assignment in MB2 BCT misc.

## `&xusb` (host xHCI)

```dts
&xusb {
    status = "okay";
    usb2-N {
        mode = "host";
        vbus-supply = <&vbus_usb2_N_reg>;   /* synthesised by /jetson-modify-usb */
        nvidia,oc-pin = <N>;
    };
    usb3-N {
        mode = "host";
        vbus-supply = <&vbus_usb3_N_reg>;
    };
};
```

The synthesised regulator is a `regulator-fixed` with `gpio = <... ACTIVE_HIGH>` driving the VBUS_EN pad. The skill emits these into `fragment@0` at the root.

## `&xudc` (device gadget)

Only emitted when at least one port has `mode = "device"` or `"otg"`:

```dts
&xudc {
    status = "okay";
    phys = <&tegra264_usb2_0>;
    phy-names = "usb2-0";
    usb2-0 { mode = "device"; };
};
```

## Properties NOT touched by the skill

The skill explicitly does **not** set any of:

* `nvidia,xusb-padctl-pinctrl-default` (managed by mb1 pinmux)
* `nvidia,boost_cpu_freq` / bandwidth-tuning props (devkit defaults)
* PHY calibration tables (SoC-internal)

If the user needs to override any of these, edit the overlay by hand after `commit` (or before, in `<KB>/staged/`) — the skill is idempotent and only rewrites the props it owns.
