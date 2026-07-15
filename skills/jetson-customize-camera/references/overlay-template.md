# Camera overlay regeneration recipe (referenced by SKILL.md §6)

The committed `tegra<soc>-overlay-<carrier_short>-camera.dts` is
**regenerated** from the in-tree NVIDIA reference templates. The
templates are read-only inputs; the agent never `#include`s them
into the carrier overlay at compile time, never copies the in-tree
`__fixups__` placeholder pattern, and never ships the artefact with
unresolved macros from `dt-bindings/`.

## Source templates (read-only inputs)

For SoC `<soc>` ∈ {`t264` (Thor), `t23x` (Orin)}:

```
$L4T_DIR/source/hardware/nvidia/<soc>/nv-public/overlay/
  tegra<soc>-p3971-camera-<sensor>-overlay.dts          # connectivity + status="okay" overrides
  tegra<soc>-camera-<sensor>*.dtsi                      # sensor body (mode tables, regulators, labels)
  tegra<soc>-p3971-camera-modules.dtsi                  # disabled-by-default tegra-camera-platform skeleton
```

For Thor with the dual-IMX274 selection:

| Role | File |
|---|---|
| Overlay shape (status overrides + per-sensor connectivity) | `tegra264-p3971-camera-dual-imx274-overlay.dts` |
| Sensor body + endpoint labels (inline) | `tegra264-camera-imx274-dual.dtsi` |
| Disabled `tegra-camera-platform` skeleton | `tegra264-p3971-camera-modules.dtsi` |
| Other sensor bodies (so the regenerator can swap) | `tegra264-camera-{ar0234,e3331,e3333,imx185,imx390,p3762}-*.dtsi` |

The dtsi defines all the cross-fragment endpoint labels
(`liimx274_csi_in0/1`, `liimx274_csi_out0/1`,
`liimx274_imx274_out0/1`, `liimx274_vi_in0/1`) inline, so a
preprocessed-and-merged DTS resolves them via `__local_fixups__`.

## Recipe

1. **Stitch input.** Inject `#include "<sensor>.dtsi"` after the
   gpio binding include in the per-sensor overlay file. This is a
   build-time stitch, not a runtime dependency:

   ```bash
   awk '/^#include <dt-bindings\/gpio\/tegra<soc>-gpio.h>/{
           print; print "#include \"tegra<soc>-camera-<sensor>*.dtsi\""; next}
        {print}' \
       $SRC/tegra<soc>-p3971-camera-<sensor>-overlay.dts > /tmp/cam_render/input.dts
   ```

2. **Materialise macros via `cpp -E`.** Resolve every
   `dt-bindings` symbol to its literal value before any DTC pass:

   ```bash
   cpp -nostdinc -undef -x assembler-with-cpp \
       -I $SRC \
       -I $L4T_DIR/source/hardware/nvidia/<soc>/nv-public/include/kernel-<soc> \
       -I $L4T_DIR/source/kernel/kernel-noble/include \
       /tmp/cam_render/input.dts -o /tmp/cam_render/expanded.dts
   ```

   `expanded.dts` now contains literal clock IDs (`<&bpmp 278U>`),
   literal GPIO offsets (`<&gpio_main 0x12 0>`), and the full
   sensor body inlined. No more `TEGRA264_CLK_EXTPERIPH1`, no more
   `CAM0_RST_L`, no more `GPIO_ACTIVE_HIGH`.

3. **Transform for the carrier.** Programmatically rewrite the
   expansion:
   - Strip cpp `# N "file"` line markers and SPDX headers emitted
     by the preprocessor; collapse consecutive blank lines.
   - Replace `overlay-name = "Jetson Camera ..."` with
     `"<carrier_short> <sensor_id>"`.
   - Replace the in-tree devkit compatible string list with
     `compatible = ` derived from the base kernel DTB's
     `/compatible` (read with
     `dtc -I dtb -O dts $L4T_DIR/kernel/dtb/<DTB_FILE>` then
     `sed -n '/^\/ {/,/^};/p' | grep '^[[:space:]]*compatible'`).
   - Replace `board_config.ids = "LPRD-..."` with a carrier-specific
     stub the carrier's daughter-card EEPROM (or a software stub)
     reports.
   - Substitute per-sensor wiring answers from `session.camera`:
     `reset-gpios` (controller + offset), `mux.channel` (which
     `i2c@<ch>` block under `tca9546@70`), `port-index` (CSI port
     per sensor), `bus-width`, `vc-id`.
   - Prepend the mandatory header comment (carrier short, sensor
     selection, CSI port + lane assignments, MCLK + reset GPIO per
     sensor, the reference template paths, the citation list).

4. **Compile + verify.**

   ```bash
   dtc -@ -I dts -O dtb \
       -o $L4T_DIR/kernel/dtb/tegra<soc>-overlay-<carrier_short>-camera.dtbo \
       <carrier_short>-camera.dts
   fdtoverlay -i $L4T_DIR/kernel/dtb/<DTB_FILE> -o /tmp/m.dtb \
       $L4T_DIR/kernel/dtb/tegra<soc>-overlay-<carrier_short>-camera.dtbo
   strings /tmp/m.dtb | grep -E '<sensor>|tca9546|skip_mux_detect'
   ```

   `fdtoverlay` MUST exit zero. The grep MUST return ≥1 hit per
   expected substring. If `FDT_ERR_NOTFOUND`, an external `&label`
   reference doesn't resolve in the base DTB's `__symbols__`; fix
   it before flashing.

## Invariants in the regenerated DTS

| Rule | Why |
|---|---|
| Cross-fragment endpoint phandles use inline DTS labels (`label: endpoint { ... }`) referenced via `<&label>`. dtc -@ emits `__local_fixups__`; **no** manual `__fixups__` block. | Resolution is internal to the dtbo; immune to base-DTB schema drift. |
| External `&label` refs (`&bpmp`, `&gpio_main`/`gpio_aon`/`gpio_uphy`, `&nvcsi`, the chosen i2c controller's `i2c<N>` label) must each appear in the base DTB's `__symbols__`. | Without it, fdtoverlay rejects with `FDT_ERR_NOTFOUND` and the runtime applier silently drops the overlay. |
| Anchor with `target = <&label>` when a symbol exists, else `target-path = "/path"`. On Thor `/tegra-capture-vi` lives at root with no symbol — that fragment uses `target-path`. | Same. |
| `i2c-mux,deselect-on-exit` lives on each `i2c@<ch>` child of the mux, **not** on `tca9546@70`. `skip_mux_detect = "yes"` lives on `tca9546@70`. | Driver expectation; missing them gives `i2c bus regbase unavailable` from camera_common, sensor probe `-75`. |
| Endpoint pairing is symmetric and double-hop. `vi_in<M>` references the nvcsi *output* endpoint (`channel@N/port@1`), not the nvcsi sensor-input. | Inverting passes dtc + fdtoverlay but kills v4l2 graph link creation (`failed to create … link`). |
| `port-index` on sensor + matching `vi/port@M` endpoint = SoC CSI port number (Thor: 0/2/4/6 for ×4; 1/3/5/7 for ×2 splits). | Hardware constraint. |
| `num-channels` on `tegra-capture-vi` and `nvcsi` = active sensor count. Never inline the dynamic dtbo's `port@2..5` / `channel@2..5` skeleton. | Skeleton's placeholder peer phandles aren't carried over and crash `tegra-nvcsi: Failed to init csi channel`. |
| **Mandatory header comment** — same rule as modify-uphy / modify-pcie: carrier short, sensor selection, CSI port + lane assignments, MCLK + reset GPIO per sensor, the reference template paths the regenerator consumed, the citation list. | Audit trail for a reviewer reading the dtbo six months later. |
| The committed DTS contains zero `#include` directives and zero NVIDIA-header symbols. | A reviewer or rebuilder must be able to compile the artefact from a sanitised tree without staging the full L4T sources. |

## Per-carrier substitution table

| Change | Where in expanded.dts |
|---|---|
| Different I²C bus | `bus@0/i2c@<addr>` wrapper + `sysfs-device-tree` strings under `tegra-camera-platform/modules/*` |
| Sensor count ≠ 2 | `num-channels` on capture-vi + nvcsi; add/remove `i2c@<ch>` blocks under `tca9546@70` + matching `channel@N` + `port@M` + `module<N>` |
| Different sensor compatible | `compatible = "<vendor,part>"` on each sensor node; sensor body comes from the matching dtsi |
| GMSL multi-link on one CSI port | one fragment-set per sensor with unique `vc-id = <0..3>` on the matching endpoints |
| No tca9546 mux | drop `tca9546@70 { ... }`; sensors sit directly under the i2c controller; rewrite `sysfs-device-tree` strings |
| Reset GPIO on a different controller | `reset-gpios = <&gpio_aon …>` or `<&gpio_uphy …>` per the carrier pinmap; verify the symbol exists in `__symbols__` |
| Carrier `__symbols__` missing the i2c label | use `target-path = "/bus@0/i2c@<addr>"` instead of `target = <&i2cN>` |
