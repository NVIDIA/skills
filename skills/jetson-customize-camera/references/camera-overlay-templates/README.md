# Camera overlay templates

These are agent-consumable starting templates for the kernel-DT
overlays `/jetson-modify-camera` emits. The skill is **agentic** — there is
no Python renderer; the agent picks the closest template by camera
type, decompiles a matching in-tree reference overlay for the
sensor-specific binding details, and patches placeholders.

## How to pick a template

| Camera class | Template | When to use | In-tree exemplar (decompile for sensor specifics) |
|---|---|---|---|
| **DPHY-direct** (single sensor on a CSI port, no SerDes) | [`dphy-direct.dts.tmpl`](./dphy-direct.dts.tmpl) | imx274, imx185, imx390, e3331, e3333, eCAM130A, ar0234 (direct), and any future sensor whose datasheet says "MIPI CSI-2 D-PHY" with no serializer | `tegra264-p3971-camera-<sensor>-overlay.dtbo` |
| **GMSL serdes chain** (sensor → MAX9295 serializer → coax → MAX96712 deserializer → CSI) | [`gmsl-serdes.dts.tmpl`](./gmsl-serdes.dts.tmpl) | Hawk (AR0234 over GMSL), Owl (OX03A10 over GMSL), Leopard p3762 daughtercards, anything with MAX96712 / MAX9296 in the chain | `tegra264-p3971-camera-p3762-a00-<config>-overlay.dtbo` |

Other GMSL deserializers (MAX9296 standalone, etc.) follow the same
GMSL template — the deserializer compatible string and channel layout
change but the structural pattern (vi → nvcsi → tca9546 mux →
deserializer → serializer → sensor) is identical.

## Placeholder grammar

Every placeholder is `<<UPPER_SNAKE_CASE>>` so a simple
`sed -e "s|<<KEY>>|value|g"` (or the agent's `Edit` tool) substitutes
it. The skill's session.camera block names every placeholder it
sets — see SKILL.md §"Output contract".

## How the agent uses these templates

Per `/jetson-modify-camera` SKILL.md step 6:

1. Read the template DTS that matches the chosen camera class.
2. **Also** decompile the closest in-tree `tegra<soc>-p3971-camera-*-overlay.dtbo`
   for the user's exact sensor model — that's the canonical source for
   sensor mode-table entries (`mode_type`, `pixel_phase`, sensor-mode
   reg-list rows). Splice those into the `sensor@<addr>` subnode of
   the template.
3. Substitute the placeholders from `session.camera`.
4. Write `$L4T_DIR/kernel/dtb/tegra<soc>-overlay-<carrier_short>-camera.dts`
   and compile to `.dtbo`.
5. Append to `OVERLAY_DTB_FILE` in the carrier `.conf` after the
   platform `*-dynamic.dtbo`.

The templates intentionally cover the structure (VI/NVCSI port plumbing,
mux subnode shape, sensor-driver `compatible` slot). Sensor-specific
register tables come from the in-tree exemplar — they encode silicon
constraints that hand-coding would silently miss.
