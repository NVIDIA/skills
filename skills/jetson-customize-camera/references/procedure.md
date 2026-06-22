# jetson-customize-camera — Detailed Procedure

This file holds the full step-by-step procedure for the
`jetson-customize-camera` skill. The SKILL.md links here from its
`## Procedure` section. Follow these steps in order.

## Step 1 — Resolve active target + open source-of-truth documents

Same as `jetson-customize-uphy` Step 1. Additionally resolve:

| Var | Source |
|---|---|
| `<chip>` / `<soc>` / `<chip-dir>` | Same as `jetson-customize-uphy` |
| `<carrier-pinmap>` | `<workspace>/target-platform/<profile-stem>.jetson-customize-pinmux/pinmap/<custom-carrier>.json` *(produced by `jetson-customize-pinmux` `probe`)* |
| `<devkit-pinmap>` | Optional — pinmap probed against the reference devkit's `.xlsm` |
| `<ref-dtb>` | `<bsp_image.root_path>/Linux_for_Tegra/kernel/dtb/<DTB_FILE from carrier conf>` |
| `<dtsi-dir>` | `<source.root_path>/bsp_sources/hardware/nvidia/<chip-dir>/nv-public/overlay/` |
| `<dtbo-dir>` | `<bsp_image.root_path>/Linux_for_Tegra/kernel/dtb/` (reference dtbos for sensor enumeration) |

Camera Development Guide pages live under
`<documents.root_path>/<bsp_developer_guide>/SD/CameraDevelopment*.html`
(the mirror saved by `/quick-start`). Cite the absolute path in
`sources[]`.

## Step 2 — Enumerate supported sensors

Glob in-tree camera dtbos for the active platform:

```bash
ls <dtbo-dir>/tegra<soc>-p3971-camera-*.dtbo  # Thor
# or
ls <dtbo-dir>/tegra<soc>-p3737-camera-*.dtbo  # Orin AGX
```

For each filename, decompile and read `overlay-name`:

```bash
dtc -I dtb -O dts <dtbo-dir>/tegra<soc>-<ref>-camera-<sensor>-overlay.dtbo | head -30
```

Classify each candidate:

| Class | Topology | Filename pattern |
|---|---|---|
| **DPHY-direct** | sensor → CSI lanes | `tegra<soc>-<ref>-camera-<sensor>-overlay.dtbo` (no `*p3762*`) |
| **GMSL** | sensor → MAX9295 → coax → MAX96712 → CSI | `tegra<soc>-<ref>-camera-p3762-*-overlay.dtbo` |
| **Custom** | user-provided wiring | (selected as "Other / custom" in Step 3) |

Surface a `multiSelect: true` `AskUserQuestion` with the discovered
list + an "Other / custom sensor" option. **Never invent sensors** —
if the tree ships 10 dtbos, list those 10. Persist the user's
selection as the run-state `selected_sensors[]`.

## Step 3 — Verify carrier + module physically support each chosen sensor

Cross-reference in priority order:

| Layer | Source | What it tells you |
|---|---|---|
| 3.1 In-tree DTSI | `<dtsi-dir>/tegra<soc>-camera-<sensor>*.dtsi` | Existence = supported on reference carrier; **DTSI IS the wiring source of truth** (I²C bus, MCLK, reset GPIOs, mux, CSI port-index, lane width, compatible) |
| 3.2 Camera Development Guide | `<documents.root_path>/<bsp_developer_guide>/SD/CameraDevelopment.html` | Per-module CSI port count + lane map, MCLK clock IDs, GMSL serdes conventions |
| 3.3 Adaptation Guide §Camera | `<bsp_developer_guide>/HR/JetsonModuleAdaptationAndBringUp/Jetson<Platform>AdaptationBringUp.html` | Module-side CSI port count, lane map, MCLK rates |
| 3.4 SoC TRM | `documents.soc_tech_ref_manual` | NVCSI block diagram, CSI port-lane mapping |
| 3.5 Module Design Guide | `documents.module_design_guide` | Module-side CSI/MCLK/I²C pinout |
| 3.6 Carrier schematic | `documents.custom_carrier_schematic` | CAM connector pinout, I²C bus assignments, reset/PWDN/PWR_EN GPIOs, on-board footprints, GMSL serdes IC presence |
| 3.7 Carrier pinmap | `<carrier-pinmap>` | Non-zero CAM rows = carrier physically wires CAM |

Decision tree:

1. **In-tree DTSI exists** → DTSI is wiring source of truth. Render
   the wiring table (Step 3a) and surface the confirm-or-customize
   gate **before** asking any wiring question.
2. **No DTSI but carrier pinmap has CAM/CSI/MCLK rows** → sensor is
   NEW. Skip to Step 4.
3. **No DTSI AND no carrier CAM rows** → carrier doesn't wire CAM.
   Warn; ask whether to proceed with custom wiring or drop camera
   from the selection.

## Step 3a — Confirm-or-customize gate (when in-tree DTSI exists)

**Mandatory ordering: tables FIRST, then the question.** Without
seeing the wiring, "Proceed with in-tree DTSI wiring" is a leap of
faith.

Lead with one sentence naming the source DTSI, then render both
tables. Every column mandatory; `—` when N/A:

> NVIDIA ships a tested overlay for **`<sensor>`** at
> `<dtsi-dir>/tegra<soc>-camera-<sensor>*.dtsi`. The table below shows
> the wiring it encodes — review and confirm.

| sensor instance | I²C bus | sensor addr | I²C mux + channel | reset GPIO | PWDN GPIO | PWR_EN GPIO | MCLK source / rate | sensor compatible | CSI port | bus-width | vc-id |
|---|---|---|---|---|---|---|---|---|---|---|---|

| field | value |
|---|---|
| `jetson-header-name` (composite root, added by this skill) | (from in-tree dtbo's `jetson-header-name`) |
| `compatible` (composite root, owned globally) | (from base kernel DTB's root `compatible`) |
| `board_config.ids` | (stub `<carrier_short>-<sensor>-001`) |
| Composite `.dtbo` registered by `/jetson-build-source` | `tegra<soc>-<carrier-id-sku>+<module-id>-xxxx-custom.dtbo` |

Cite source DTSI path, base kernel DTB path, AND the in-tree per-
sensor dtbo path under the tables.

THEN issue:

```
Question: "Above is the in-tree overlay's wiring for <sensor>. Proceed as-is or customize?"
Header:   "Camera plan"
Options:
  - "Proceed with in-tree DTSI wiring (Recommended)"  → skip Step 4, render in Step 5
  - "Customize one or more wiring parameters"        → fall through to Step 4
```

Never skip this gate even when evidence pins the answer; never
issue the question without rendering both tables first.

## Step 4 — Per-sensor wiring questions (custom path only)

Fires when Step 3a "Customize" was picked OR the chosen sensor has no
in-tree DTSI. One batched `AskUserQuestion` per sensor; auto-fill
from carrier pinmap and skip unambiguous fields:

| # | Field | Auto-fill source | Notes |
|---|---|---|---|
| 1 | I²C bus | `<carrier-pinmap>` `CAM<n>_I2C_*` + Adaptation Guide §I²C node-address table | Thor: `i2c@810c6d0000` (≠ Orin's `i2c@31e0000`) |
| 2 | I²C address (hex) | user | Sensor-specific (IMX274 = 0x1a) |
| 3 | Reset GPIO | Pinmap `customer_usage = CAM<n>_RST*` | DT pin name |
| 4 | PWDN GPIO | Pinmap `CAM<n>_PWDN` | Optional |
| 5 | PWR_EN GPIO | Pinmap `*PWR_EN*` | Per-sensor LDO enable |
| 6 | MCLK | Adaptation Guide §Camera Module Configuration | Default `extperiph1` @ 24 MHz |
| 7 | Sensor type / compatible | Sensor datasheet | E.g. `sony,imx274` |
| 8 | EEPROM `board_config.ids` | Daughter-card EEPROM or stub | Default `<carrier_short>-<sensor>-001` |
| 9 | I²C mux | Carrier schematic | If yes, ask channel + mux addr (default `0x70`) |
| 10 | CSI port-index + bus-width | Schematic + Module TRM | Thor: 0/2/4/6 ×4-lane; 1/3/5/7 ×2-lane |
| 11 | Virtual channel id | GMSL only — 0..3 unique per CSI port | |

Persist under run-state `sensors[<i>]`.

## Step 5 — Append fragments to the composite custom overlay

The composite custom overlay `.dts` (filename / location / skeleton
/ append protocol) is documented in
`../../../references/bsp-customization-kernel-dtb.md`;
follow that doc exactly. Camera is special only because (a) the
fragment body comes from a cpp-expanded in-tree DTSI rather than
hand-authored DT, and (b) plugin-manager needs `jetson-header-name`
on the composite **root** (not inside a fragment).

**Marker for this skill's fragment:** `/* custom-bsp: camera:<sensor> */`.
On re-run, delete every fragment matching this skill's marker
pattern before appending the new one.

**This skill also ensures `jetson-header-name` is set on the
composite root** — idempotent property add. If the composite root
already has `jetson-header-name`, verify it matches; if it
mismatches, refuse and route to user (a target with mixed Jetson
headers is unusual).

### Step 5a — Clone path (in-tree DTSI exists)

#### 5a.i — cpp-expand the dtsi

Materialise every macro to a literal so the dtbo doesn't depend on
L4T `dt-bindings/` at boot:

```bash
SRC=<source.root_path>/bsp_sources/hardware/nvidia/<chip-dir>/nv-public/overlay
INC1=<source.root_path>/bsp_sources/hardware/nvidia/<chip-dir>/nv-public/include/kernel-<chip-dir>
INC2=<source.root_path>/bsp_sources/kernel/kernel-noble/include
TMP=/tmp/cam_render && rm -rf $TMP && mkdir -p $TMP

cat > $TMP/wrap.dts <<EOF
/dts-v1/;
/plugin/;
#include <dt-bindings/clock/tegra<soc>-clk.h>
#include <dt-bindings/gpio/tegra<soc>-gpio.h>
#include "tegra<soc>-camera-<sensor>*.dtsi"
EOF

cpp -nostdinc -undef -x assembler-with-cpp \
    -I $SRC -I $INC1 -I $INC2 \
    $TMP/wrap.dts -o $TMP/expanded.dts
```

**If `cpp` fails with `fatal error: dt-bindings/gpio/gpio.h: No such
file or directory`** — the L4T source tree isn't staged. Ask the
user to re-run `/jetson-init-source` (Branch B's `source_sync.sh`
fetches the headers). Never fabricate the macro expansion.

#### 5a.ii — Extract the fragment body + ensure composite root

The cpp-expanded file has shape `/ { fragment@0 { … }; };`. Three
steps:

1. **Clean the expansion**: `grep -v '^# [0-9]' $TMP/expanded.dts | grep -v '^// SPDX' > $TMP/clean.dts`.
2. **Read `jetson-header-name`** from any in-tree per-sensor dtbo for this platform via `dtc -I dtb -O dts ... | awk '/jetson-header-name/{...}'`. Idempotently splice onto the composite root as a sibling of `overlay-name` (if absent, insert via `sed -i '/overlay-name = /a\    jetson-header-name = "$HEADER";'`; if present and matching, no-op; if present and mismatching, refuse).
3. **Extract `fragment@0 { ... };` body** from `$TMP/clean.dts` (brace-balanced awk), renumber to the composite's next-free `fragment@<N>`, inject the marker `/* custom-bsp: camera:<sensor> */`, and append to the composite.

Result: **the composite now has one more `fragment@<N>`** (plus
`jetson-header-name` on its root if it wasn't there yet). Do NOT
add a second `fragment@<N> { __overlay__ { … } };` block carrying
status overrides or duplicate sensor/mux subtrees — that is the
**dual-fragment trap** (dtc deep-merge produces duplicate sibling
nodes; runtime first-match drops the dtsi-supplied deep tree;
camera silently doesn't enumerate).

**Compatible string is set on the composite root, not by this
skill.** Camera does not widen `compatible` per-fragment. Fix the
composite root if it's missing the live-DUT string.

#### 5a.iii — Substitute carrier-specific wiring (custom path only)

If Step 3a "Customize" was picked, splice user answers IN-PLACE
inside the existing `fragment@0/__overlay__`: `reset-gpios`,
`mux.channel`, `port-index`, `bus-width`, `vc-id`, and the I²C bus
parent for `tca9546@70`. Edit IN-PLACE. **Never** introduce a
second fragment for overrides (dual-fragment trap).

### Step 5b — Custom-sensor path (no in-tree DTSI)

1. Pick the closest-match in-tree DTSI (same class + lane width).
2. cpp-expand it as in 5a.i.
3. Substitute Step-4 answers IN-PLACE.
4. **Splice mode tables (`mode<N>`, `sensor_modes`) from the closest
   in-tree DTSI** — never hand-author; they encode pixel-phase,
   pixel-bit-depth, framerate, gain, exposure ranges the tegracam
   driver validates.
5. Prepend the carrier metadata root as in 5a.ii.

### Step 5c — Compile + verify the COMPOSITE

```bash
cpp -nostdinc -x assembler-with-cpp $COMPOSITE /tmp/composite.tmp.dts
dtc -@ -I dts -O dtb -o /tmp/composite.dtbo /tmp/composite.tmp.dts
fdtoverlay -i <bsp_image.root_path>/Linux_for_Tegra/kernel/dtb/<DTB_FILE> \
           -o /tmp/__merged.dtb /tmp/composite.dtbo
```

**Pre-compile gate** — exactly one camera-tagged fragment in the
composite per re-run:
`[ "$(grep -cE '/\* custom-bsp: camera:' "$COMPOSITE")" = "1" ]`.
(Other non-camera `fragment@N` blocks are allowed; the gate is on
camera-tagged ones only.)

**Post-compile gate** — deep-tree uniqueness inside the merged
overlay's active subtree. Decompile `/tmp/composite.dtbo` with `dtc
-I dtb -O dts`, strip the trailing `__local_fixups__` / metadata
blocks, then verify each sentinel matches AT MOST once in the
active subtree: `tca9546@70 {`, `tegra-capture-vi {`,
`tegra-camera-platform {`, `nvcsi@[0-9a-f]+ {`. >1 = duplicate-merge
trap; abort.

**`fdtoverlay` must exit 0** AND
`grep -aco <sensor> /tmp/__merged.dtb` ≥ 1. On `FDT_ERR_NOTFOUND`,
inspect the overlay's `__fixups__`: every label must exist in the
base DTB's `__symbols__`. Replace missing `&label` with
`target-path` or fix the label. `tegra_xusb` / `tegra_xudc` aren't
in `__symbols__` — use `target-path`.

### Step 5d — Commit the composite `.dts`

```bash
git -C <source.root_path>/bsp_sources add <composite-relative-path>
git -C <source.root_path>/bsp_sources commit -m "jetson-customize-camera: append <sensor> to <board-tag>-custom"
```

Apply the
[preview gate](../../../context/bsp-customization-workflow.md#commit-message-preview-gate)
before `git commit` — surface the proposed message and require
accept / edit / cancel from the operator; no auto-commit.

`/jetson-build-source` re-compiles the composite `.dtbo` at build
time and owns its Makefile + flash-conf registration.

## Step 6 — Pin verification + auto-invoke `/jetson-customize-pinmux`

For every sensor in the run-state, verify ancillary pin SFIOs and
route mismatches to `/jetson-customize-pinmux` (same contract as
`jetson-customize-mgbe` Step 4 / `jetson-customize-pcie` Step 4):

| Signal | SFIO | Direction | Initial state |
|---|---|---|---|
| `CAM<n>_I2C_SCL` | `cam_i2c_scl` | bidirectional | n/a |
| `CAM<n>_I2C_SDA` | `cam_i2c_sda` | bidirectional | n/a |
| `CAM<n>_MCLK` | `extperiph<m>_clk` | output | n/a |
| `CAM<n>_RST_L` | `gpio` | output | high |
| `CAM<n>_PWDN` *(when wired)* | `gpio` | output | high |
| `CAM<n>_PWR_EN` *(when wired)* | `gpio` | output | high |

Invoke `pin_verifier.py` (read-only). For mismatches, surface
`"I'm setting pin X to SFIO Y for <reason>"` and route to
`/jetson-customize-pinmux set-pin`. Skill does NOT directly patch the
pinmux DTSI.

## Step 7 — Run-state JSON sidecar + summary

**Sidecar** at
`<workspace>/target-platform/<profile-stem>.jetson-customize-camera.json`:

```json
{
  "generator": "jetson-customize-camera",
  "generator_version": "0.1.0",
  "generated_at": "<ISO-8601 UTC>",
  "active_profile": "<profile-stem>.yaml",
  "bsp_version": "<bsp_image.version>",
  "selected_sensors": ["dual-imx274"],
  "class": "dphy-direct | gmsl | custom",
  "connector": {"jetson_header_name": "Jetson AGX CSI Connector", "compatible_list": ["nvidia,<carrier>+<module>", "nvidia,tegra<soc>"], "eeprom_id": "<carrier>-<sensor>-001"},
  "sensors": [
    {"id": "imx274_a", "compatible": "sony,imx274", "i2c_bus_node": "i2c@810c6d0000", "i2c_addr_hex": "0x1a", "reset_gpio_pin": "gpio_main TEGRA264_MAIN_GPIO(V,2)", "mclk_clock_node": "extperiph1", "port_index": 0, "bus_width": 4, "vc_id": 0, "mux": {"compatible": "nxp,pca9546", "addr_hex": "0x70", "channel": 0}}
  ],
  "overlay": {"composite_dts_path": "...tegra<soc>-<carrier>+<module>-xxxx-custom.dts", "fragments_appended": ["camera:<sensor>"], "jetson_header_name_added": "<header-or-null>", "render_path": "clone"},
  "pin_verifier_mismatches": [], "warnings": [], "notes": [], "sources": [],
  "commit_shas": {"overlay_tracker": "<short SHA>", "bsp_sources": "<short SHA>"}
}
```

**Atomic write** + **idempotency**: same as `jetson-customize-uphy` Step 5.

**Headline:**

```
jetson-customize-camera: <sensor> enabled (CSI<a>[+CSI<b>] x<lanes> via tca9546@70 ch<X> on i2c@<addr>); fragment appended to composite tegra<soc>-<carrier-id-sku>+<module-id>-xxxx-custom.dts (dtc clean, fdtoverlay merge clean).
```

**Next step (interactive prompt chain — MANDATORY, not exempted by auto-mode):**

After the summary table is printed, drive the downstream chain via
sequential `AskUserQuestion` prompts. Never auto-invoke; each prompt
needs an explicit `yes` to proceed. On `no` (or any abort), print the
remaining manual run-chain and exit.

**This is a documented workflow gate, not a clarifying question.**
Auto-mode / quiet-mode policies cover clarifying questions only and
do NOT exempt this chain. Substituting a printed "Next step: …"
line for the `AskUserQuestion` prompts is a skill violation —
always emit the prompts.

If Step 6 surfaced SFIO mismatches, prepend prompt **(0)** first:

0. **Re-run `/jetson-customize-pinmux` to fix the surfaced SFIO
   mismatches?** — on `yes` invoke `/jetson-customize-pinmux`, then
   continue to (1).

Then in order:

1. **Customize any other I/O before build?** — offer
   `/jetson-customize-pinmux`, `/jetson-customize-uphy`,
   `/jetson-customize-pcie`, `/jetson-customize-mgbe`,
   `/jetson-customize-usb`, `/jetson-customize-clocks`,
   `/jetson-customize-fan`, `/jetson-customize-nvpmodel`,
   `/jetson-customize-memory`, and `no` (proceed). On any non-`no`
   pick, invoke the chosen sub-skill inline; when it returns, re-ask
   this same question (loop) until the user picks `no`. Then continue
   to (2).
2. **Build & promote?** — on `yes` invoke `/jetson-build-source`, then
   on success invoke `/jetson-promote-image`.
3. **Flash the board?** — only offer if (2) ran and succeeded; on
   `yes` invoke `/jetson-flash-image`. If (2) was skipped, pre-skip
   this prompt with a warning that the staged image isn't promoted.
4. **Validate on the DUT?** — only offer if (3) ran and succeeded; on
   `yes` invoke `/jetson-validate-image`.
