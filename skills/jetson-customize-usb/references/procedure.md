# jetson-customize-usb — Procedure

Full step-by-step procedure for the USB per-port enable/disable/role
customization skill. The skill's `SKILL.md` only carries the slim
summary; this file is the load-bearing detail.

## Step 1 — Resolve active target + open source-of-truth documents

Same as `jetson-customize-uphy` Step 1 + `jetson-customize-pcie` Step 1.
Additionally resolve:

| Var | Source |
|---|---|
| carrier-pinmap | `<workspace>/target-platform/<profile-stem>.jetson-customize-pinmux/pinmap/<custom-carrier>.json` |
| ref-dtb | `<bsp_image.root_path>/Linux_for_Tegra/kernel/dtb/<DTB_FILE from carrier conf>` |
| schematic-nets | optional — pre-parsed `<carrier>_nets.json` if produced by a prior schematic ingest; otherwise read the PDF directly |
| uphy-state | `<workspace>/target-platform/<profile-stem>.jetson-customize-uphy.json` — drives the UPHY-lane cross-check for SS ports (Step 4a) |

Adaptation Guide section: `<documents.root_path>/<bsp_developer_guide>/HR/JetsonModuleAdaptationAndBringUp/Jetson<Platform>AdaptationBringUp.html#port-the-universal-serial-bus`.

## Step 2 — Build the USB topology + companion graph from the in-tree DTB

```bash
DTB=<ref-dtb>
DTS=$(dtc -I dtb -O dts $DTB)
```

**Node-selection + indent-bounded extraction (REQUIRED).** Tegra USB
spans three IP blocks with distinct compatibles. Filter on those
strings — never on the bare `usb<2|3>-N` glob — to avoid picking up
sibling sub-nodes that happen to share a name fragment:

| Block | Compatible to match | What lives here |
|---|---|---|
| Pad controller | `nvidia,tegra<soc>-xusb-padctl` | lane + port `status`, `mode`, `usb2-companion` |
| Host xHCI | `nvidia,tegra<soc>-xusb` | host `phys`/`phy-names` array |
| Device xudc *(optional)* | `nvidia,tegra<soc>-xudc` | OTG device-mode attach (only `usb2-0`) |

When extracting any block body, bound the awk/sed range by matching
the **closing brace at the same indent as the opening line**. A naive
`/usb<2|3>-N {/,/^\t};/` range can leak across `lanes { … };`,
`ports { … };`, or nested `mdio { … };` children and report a status
from the wrong scope. Use the same indent-matched pattern as the PCIe
and MGBE procedures:

```bash
ln=$(grep -nE "^[[:space:]]+(usb[23]-[0-9])\b.*\{" /tmp/k.dts | head -1 | cut -d: -f1)
indent=$(sed -n "${ln}p" /tmp/k.dts | sed -E 's/[^[:space:]].*//')
awk -v start="$ln" -v ind="$indent" \
  'NR>=start { if (NR>start && $0==ind"};") exit; print }' /tmp/k.dts
```

Same trap that hit `/jetson-customize-pcie` (reading ECAM root instead
of `nvidia,tegra264-pcie`). Apply both rules whenever you decode a
Tegra IP block in the kernel DTB.

For each `usb2-0..N` / `usb3-0..M` the platform exposes, capture:

| Field | Where |
|---|---|
| Lane status | `…/lanes/usb<2|3>-N.status` |
| Port status | `xusb_padctl/ports/usb<2|3>-N.status` |
| Port mode | `xusb_padctl/ports/usb<2|3>-N.mode` (Tegra invariant: only `usb2-0` is OTG-capable) |
| SS to USB2 companion | `xusb_padctl/ports/usb3-N.nvidia,usb2-companion` — build `companion_graph` mapping `usb2-M` to set of `usb3-N` that reference it |
| Host phys-list | `usb@<host xhci addr>.phys` (ordered phandle array) + `phy-names` (matching strings) — full list including stock-disabled entries |

Build the comparison table (devkit baseline vs custom carrier)
**before** any question:

| Port | Stock mode | Stock status | Carrier wiring | Receptacle | Schematic page | Companion |
|---|---|---|---|---|---|---|

Wiring + receptacle columns come from the schematic TOC:

- `USB 3.x HUB` block — one Tegra USB2+USB3 pair feeds a hub IC
  fanning out to multiple downstream ports (typical Realtek
  RTS5400 / Genesys Logic / VIA).
- `USB 3.x Type-A Stacked` — on-board stacked Type-A receptacles
  (often hub-fanned).
- `USBC<N> Connector` — Type-C; CC1 / CC2 nets present.
- `USB Debug Connector` — micro-USB debug port (typically `usb2-0`).

Cite schematic page numbers. **Mandatory before any
`AskUserQuestion`** — diff-driven scoping prevents asking about
ports that don't exist or are stock-disabled.

## Step 3 — Ask which port(s) to enable / disable

`AskUserQuestion` with `multiSelect: true`. Build options from Step 2's
table — no static list. Each option encodes port + action (enable /
disable / mode change) per the Tegra platform invariant:

> **Tegra platform invariant** — only `usb2-0` is OTG-capable. The
> device controller (`xudc`) attaches there only. `usb2-1..N` are
> host-only. All `usb3-N` ports are host-only.
>
> Default `usb2-0` to `otg` when the carrier exposes Type-C (CC1/CC2
> nets in schematic), otherwise `host`.

**When the user picks "disable port X":**

1. **Resolve the companion graph** for the chosen ports. For every
   USB2 being disabled, look up `companion_graph[usb2-X]` — that SS
   set MUST also be disabled (Rule B).
2. **Detect on-carrier hub upstream.** Schematic TOC `USB 3.x HUB`
   block then SS+USB2 pair feeding it presents two USB-tier
   enumerations of the same physical hub IC. Disabling the upstream
   pair takes both enumerations and **every downstream receptacle**
   on the hub with it.
3. **Surface the cascade** via a single confirmation
   `AskUserQuestion` listing the SS companion ports + the on-carrier
   hub's downstream receptacles that become unreachable.

**When the user picks "enable port X"** (e.g. stock-disabled `usb3-3`
on Thor):

1. Read uphy-state's `uphy.config` + `uphy.config_uphy1`. Verify
   they allocate a UPHY lane to this SS port (cross-check Adaptation
   Guide §"Configure the UPHY Lane"). If not, warn + recommend
   `/jetson-customize-uphy`; **refuse to commit** until the lane is
   allocated.
2. Verify the carrier schematic actually wires SS RX±/TX± to a
   physical receptacle. If schematic ingest is silent, fall back to
   PDF inspection (matching net labels: `USB_SS<N>_RX±/TX±`).

Do not invent ports. If the in-tree DTB exposes only a subset, do
not offer others — the node literally does not exist.

## Step 4 — Per-port verify + capture wiring

For each port picked in Step 3, run a sub-loop.

**4a. Verify carrier + module physically support the port.**

1. **Module side** (Module Design Guide §USB): does the SKU expose
   USB2 D± / USB3 SS RX±/TX± on the right pins for this port? Cite
   the table.
2. **Carrier side** (schematic + pinmap): does the carrier route
   those differential pairs to a receptacle matching the user's
   role choice (host receptacle for `host`, OTG-capable for
   `otg`/`device`)? Cite schematic page.
3. **UPHY lane** (USB3 SS only): does uphy-state's `uphy.config*`
   allocate a UPHY lane to this SS port? Without one the SS PHY is
   powered down and xHCI times out on enumeration.

Record failures as `warnings[]` and recommend the user drop the
port or fix upstream.

**4b. Cross-check carrier pinmap + schematic for ancillary pins.**

For each USB host port needing a powered receptacle:

| Role | Evidence keys | Field |
|---|---|---|
| VBUS-EN | `VBUS<N>_EN`, `EN_USB_VBUS_<N>`, `USB<N>_VBUS_EN` | output GPIO |
| OC | `USB<N>_OC`, `OC_USB_<N>`, `USB_OC` | input GPIO |
| CC1 / CC2 (Type-C only) | `USBC<N>_CC1`, `USBC<N>_CC2` | input GPIOs |

Auto-invoke `pin_verifier.py` against these expected SFIOs (same
contract as `jetson-customize-pcie` Step 4 / `jetson-customize-mgbe`
Step 4). For each mismatch, surface `"I'm setting pin X to SFIO Y
for <reason>"` and route to `/jetson-customize-pinmux set-pin`.
Skill does NOT directly patch the pinmux DTSI.

**4c. Per-port customization questions (only when ambiguous).** One
batched `AskUserQuestion` per port. Skip questions whose answer is
unambiguous from evidence:

1. **Mode** — `host` / `device` / `otg`. Skip on `usb2-1..N` and all
   `usb3-N` (hard-pinned to `host` per platform invariant).
2. **Maximum speed** — silicon-derived: USB2 to `high-speed`; USB3 SS
   to `super-speed` / `super-speed-plus` / `super-speed-plus-x2`
   (Thor).
3. **VBUS-EN GPIO** — auto-fill from 4b; ask only when ambiguous or
   unrouted.
4. **OC GPIO** + (Type-C) **CC1 / CC2 GPIOs** — same.

Persist under run-state `ports[<id>]` + `cascade`.

## Step 5 — Render the kernel-DT overlay (the three-place pattern)

The overlay is load-bearing. Always emit when any port is in the
delta.

**5a. Probe the base DTB's `__symbols__`** to know which labels are
externally referenceable:

```bash
dtc -I dtb -O dts <ref-dtb> | sed -n '/__symbols__/,$p' \
  | grep -E '(xusb_padctl|tegra_xhci|tegra_xusb|tegra_xudc|gpio_main|gpio_aon|gpio_uphy|bpmp|i2c[0-9]+) ='
```

Typically present: `xusb_padctl`, `bpmp`, `gpio_main` / `gpio_aon` /
`gpio_uphy`.

Typically **NOT** in `__symbols__`: host xHCI node `usb@<addr>` and
`tegra-xudc` node. Those fragments MUST use
`target-path = "/bus@0/usb@<addr>"` instead of `target = <&label>`.

If a referenced label is missing, use `target-path`. **Never invent
a `__fixups__` entry** — `fdtoverlay` rejects with
`FDT_ERR_NOTFOUND` and runtime silently drops.

**5b. Resolve per-port phandles + host phys-list ordering.** For
every port being flipped, look up its lane phandle:

```bash
echo "$DTS" | grep -B6 'phandle = <0xPHANDLE>;'
```

Build a `port_id` to `lane_phandle` map — the overlay's host-phys-list
override consumes it.

**5c. Append fragments to the composite custom overlay.** The
single composite `.dts` and append protocol are documented in
`../../../references/bsp-customization-kernel-dtb.md`; the notes here cover
only what's USB-specific.

**Marker prefix for this skill's fragments:** `usb` with one of
three sub-keys per anchor (see table). On re-run, delete every
fragment matching this skill's marker pattern before appending
the new ones.

Fragments grouped by anchor (one per anchor, NOT one per port):

| Fragment marker | Target | Changes |
|---|---|---|
| `usb:padctl` | `target = <&xusb_padctl>` | Per-port `pads/.../lanes/usb<2|3>-N.status` AND `ports/usb<2|3>-N.status` AND `.mode` (when role flips) AND `.vbus-supply` / `oc-pin` (when ancillary pins set). |
| `usb:xhci` | `target-path = "/bus@0/usb@<host xhci addr>"` | Override `phys` + `phy-names` to drop disabled-port entries; **keep stock element order**, only elide disabled ones. |
| `usb:xudc` (only when mode flips to/from `device`/`otg`) | `target-path = "/bus@0/usb@<xudc addr>"` | Mirror the host's pruning rule for xudc's `phys`. |

Fragment shape:

```dts
fragment@A   { /* custom-bsp: usb:padctl */ target = <&xusb_padctl>; __overlay__ { … }; };
fragment@A+1 { /* custom-bsp: usb:xhci   */ target-path = "/bus@0/usb@<addr>"; __overlay__ { … }; };
fragment@A+2 { /* custom-bsp: usb:xudc   */ target-path = "/bus@0/usb@<xudc-addr>"; __overlay__ { … }; };
```

**Companion cascade:** for every USB2 being disabled, every SS port
in `companion_graph[usb2-N]` MUST also be disabled in all three
places. No exceptions, no silent skip.

**Phandle override style:** for the host xHCI fragment, use raw
integer phandles (`phys = <0x157>, <0x159>, …`) rather than
`&label` — lane nodes lack `__symbols__` entries, and overlay merge
does not renumber base-DTB phandles. dtc emits a benign warning
`phys_property: cell 0 is not a phandle reference`; kernel resolves
cells correctly per the `nvidia,tegra<soc>-xusb` binding.

**Mandatory provenance comment as the first line inside each
fragment's `__overlay__`** — cover: (a) per-port actions with mode
+ max-speed + ancillary-pin details; (b) companion cascade with
phandle; (c) on-carrier hub topology when relevant; (d) three-place
delta vs reference DTB; (e) UPHY-state cite for enabled SS ports;
(f) citations — Adaptation Guide §"Port the Universal Serial Bus",
MDG §USB, schematic pages, in-tree DTB path, phandle integers.

**5d. Verify before commit** (pre-flight `dtc` + `fdtoverlay`
sanity check, per the context doc Step 10 of the append protocol):

```bash
cpp -nostdinc -x assembler-with-cpp <composite-abs-path> /tmp/composite.tmp.dts
dtc -@ -I dts -O dtb -o /tmp/composite.dtbo /tmp/composite.tmp.dts
fdtoverlay -i <ref-dtb> -o /tmp/__merged.dtb /tmp/composite.dtbo
```

Three post-merge invariants the agent MUST verify with
`dtc -I dtb -O dts /tmp/__merged.dtb`:

1. For every disabled port `usb<2|3>-N`:
   `pads/usb<2|3>/lanes/usb<2|3>-N/status == "disabled"` AND
   `ports/usb<2|3>-N/status == "disabled"`.
2. The host xHCI's `phys` list does NOT contain the lane phandle of
   any disabled port; matching `phy-names` does NOT contain the port
   string.
3. Every still-enabled port (stock-okay AND not in the disable set)
   appears in the host's `phys` list at its **stock index** —
   preserved ordering.

If any invariant fails, refuse to commit. `fdtoverlay` exit code
is necessary but not sufficient — it accepts dtbos that violate
(3) silently.

**Ordering vs `jetson-customize-uphy` / `jetson-customize-pcie` is
intra-file:** this skill should run AFTER those, so its USB
fragments land later in the composite and win any target-path
collision. `OVERLAY_DTB_FILE+=` registration is owned by
`/jetson-build-source` Step 5.0a — this skill never touches the
carrier flash conf.

**5e. Commit the composite `.dts`** in the `bsp_sources/` mono-repo
(filename resolved per the context doc):

```bash
git -C <source.root_path>/bsp_sources add <composite-relative-path>
git -C <source.root_path>/bsp_sources commit -m "jetson-customize-usb: append <port actions> to <board-tag>-custom"
```

Apply the preview gate (see `../../context/bsp-customization-workflow.md`,
`commit-message-preview-gate`) before `git commit` — surface the
proposed message and require accept / edit / cancel from the
operator; no auto-commit.

## Step 6 — Summary + headline

**Headline:**

```
jetson-customize-usb: usb2-3 + usb3-2 disabled (lane + port + host-phys list, companion cascade); on-carrier hub on page 27 fans out to 4 Type-A receptacles — unreachable. Host phys 7 to 5 entries. Fragments appended to composite tegra-soc-carrier-id-sku-module-id-xxxx-custom.dts.
```

**Next step (interactive prompt chain):**

After the summary table is printed, drive the downstream chain via
sequential `AskUserQuestion` prompts. Each prompt needs an explicit
`yes`. On `no` (or any abort), print the remaining manual run-chain
and exit. Never substitute a printed "Next step: …" line for the
prompts.

If Step 4b surfaced VBUS-EN / OC / CC SFIO mismatches, prepend prompt
**(0)** first:

0. **Re-run `/jetson-customize-pinmux` to fix the surfaced SFIO
   mismatches?** — on `yes` invoke `/jetson-customize-pinmux`, then
   continue to (1).

Then in order:

1. **Customize any other I/O before build?** — offer
   `/jetson-customize-pinmux`, `/jetson-customize-uphy`,
   `/jetson-customize-pcie`, `/jetson-customize-mgbe`,
   `/jetson-customize-camera`, `/jetson-customize-clocks`,
   `/jetson-customize-fan`, `/jetson-customize-nvpmodel`,
   `/jetson-customize-memory`, and `no` (proceed). On any non-`no`
   pick, invoke the chosen sub-skill inline; when it returns, re-ask
   this same question (loop) until the user picks `no`. Then continue
   to (2).
2. **Build & promote?** — on `yes` invoke `/jetson-build-source`, then
   on success invoke `/jetson-promote-image`.
3. **Flash the board?** — only offer if (2) ran and succeeded; on
   `yes` invoke `/jetson-flash-image`.
4. **Validate on the DUT?** — only offer if (3) ran and succeeded; on
   `yes` invoke `/jetson-validate-image`.
