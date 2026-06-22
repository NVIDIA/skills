# NVIDIA `tegra-capture-vi` / `tegra-camera-platform` binding cheat-sheet

Reference: `<L4T_DIR>/kernel/dtb/tegra264-overlay-cus_carr_k-camera.dts` (shipped) and `<L4T_DIR>/kernel/dtb/tegra264-p3971-camera-e3333-overlay.dtbo`.

## Overlay header

```
/dts-v1/;
/plugin/;

/ {
    overlay-name = "Carrier <name> CSI base";
    fragment@0 {
        target-path = "/";
        board_config { ids = "<carrier-id>"; sw-modules = "kernel"; };
        __overlay__ { … };
    };
};
```

`board_config.ids` is matched against the platform's board-id at boot — this is what the L4T loader uses to decide whether to apply the overlay. Use the same short carrier name `/jetson-custom-bsp` chose.

## `tegra-capture-vi`

```
tegra-capture-vi {
    status = "okay";
    num-channels = <N>;       /* total ports below */
    ports {
        status = "okay";
        #address-cells = <1>;
        #size-cells = <0>;
        port@<idx> {
            reg = <idx>;
            status = "okay";
            <label>: endpoint {
                port-index = <idx>;       /* MIPI brick index 0..7 = A..H */
                bus-width  = <2 | 4>;     /* 2 = independent brick, 4 = paired (A+B / C+D / E+F / G+H) */
                vc-id = <0>;              /* virtual-channel id; usually 0 */
                remote-endpoint = <&<sensor-side label>>;
            };
        };
    };
};
```

For paired 4-lane ports the `port@<idx>` uses the *primary* brick's index (the one that owns the clock) — eg the A+B paired port is `port@0` with `bus-width=4`, and there is no `port@1`.

## `tegra-camera-platform`

```
tegra-camera-platform {
    status = "okay";
    compatible = "nvidia, tegra-camera-platform";
    num_csi_lanes = <N>;          /* sum of bus-widths */
    max_lane_speed = <1500000>;
    min_bits_per_pixel = <10>;
    vi_peak_byte_per_pixel = <2>;
    vi_bw_margin_pct = <25>;
    isp_peak_byte_per_pixel = <5>;
    isp_bw_margin_pct = <25>;
    modules {
        status = "okay";
        module<i> {
            status = "okay";
            badge = "<carrier>_port_<idx>_<sensor>";
            position = "front" | "rear" | "topleft" | …;
            orientation = "0".."3";
            drivernode0 {
                status = "okay";
                pcl_id = "v4l2_sensor";
                /* sysfs-device-tree filled in by sensor overlay */
            };
        };
    };
};
```

`modules/module<i>` is one entry per logical port (not per lane). The sensor-specific overlay extends each `module<i>` with the actual `drivernode0.devname`, sensor mode tables, and the `i2c@…/sensor@…` binding.

## Reference label convention

The skill emits these labels on the VI side:

```
carrier_vi_in_a: endpoint { port-index = <0>; bus-width = <2>; … };
carrier_vi_in_b: endpoint { port-index = <1>; bus-width = <2>; … };
…
```

The companion sensor overlay must define matching outbound endpoints:

```
carrier_csi_out_a: endpoint { remote-endpoint = <&carrier_vi_in_a>; };
```

If you change the prefix (`carrier_`), change it on both sides.

## Common pitfalls

- **`num-channels` must equal the count of `port@N` entries** under `ports`, not the lane count. 4 paired-mode 4-lane ports = 4 channels.
- **`port-index` must equal the brick number** (A=0…H=7). Mismatch = MIPI cal works for the wrong PHY and the port comes up dark.
- **`bus-width` must be 2 or 4**, not 1 or 8. Tegra MIPI is 2-lane minimum, 4-lane maximum per port.
- **The base CSI overlay must come *before* the sensor overlay in `OVERLAY_DTB_FILE`** so VI ports come up first; otherwise the sensor overlay's `remote-endpoint` references resolve to phandle 0xFFFFFFFF and the binding fails silently at probe time.
