## Procedure

### Step 1 — Resolve active target + open source-of-truth documents

Same as `jetson-customize-uphy` Step 1 (refuse table + identifier resolution
+ document resolution from `documents:` block). The Adaptation Guide
chapter relevant here is **"Enable 25 Gigabit Ethernet on QSFP Port"**.

Additionally resolve:

| Var | Source |
|---|---|
| `<platform>` | `thor` (T264) — this skill is Thor-only at present |
| `<carrier-pinmap>` | `<workspace>/target-platform/<profile-stem>.jetson-customize-pinmux/pinmap/<custom-carrier>.json` *(produced by `jetson-customize-pinmux` `probe`; optional but required for Step 4 auto-fix)* |

### Step 2 — Per-controller question loop

Drive the loop with `AskUserQuestion` using the schema in
[`questions.json`](questions.json). Per-controller fields:

| ID | Form | Notes |
|---|---|---|
| `controller_selection` | multi-select `mgbe0..mgbe3` | One controller per QSFP cage; multi-select only if the carrier has multiple MGBE-wired cages. |
| `phy_mode` | `25g-r` / `10g-kr` / `1000base-x` | Default `25g-r` for QSFP 25G. |
| `phy_attach_kind` | `phy` (external MDIO PHY) / `direct` (SerDes-direct) | Determines whether the overlay emits `phy-handle` + `mdio { phy@<addr> }`. |
| `phy_i2c_bus` | One of `i2c@3160000` .. `i2c@c250000` | Only when `phy_attach_kind=="phy"`. Pick the I²C controller whose SDA/SCL nets reach the PHY refdes per schematic. |
| `phy_i2c_addr_hex` | 7-bit hex (`^0x[0-7][0-9a-fA-F]$`) | Only when `phy_attach_kind=="phy"`. From PHY strap pins / datasheet. |
| `phy_reset_gpio_offset` | decimal integer | Offset within the chosen bank; cross-check the carrier pinmap. |
| `phy_reset_gpio_bank` | `gpio_main` / `gpio_aon` | |
| `compatible_list` | multi-select strings | Root `compatible` for the overlay. MUST contain at least one string that appears in the live DUT's `/proc/device-tree/compatible`. Defaults from catalogue (Thor): `nvidia,<custom-id>-<custom-sku>+<module-id>-<module-sku>`, `nvidia,<chip>`. Reuse prior `jetson-customize-camera` / `jetson-customize-pcie` runs' compatible list when present. |

### Step 3 — Derive `nvidia,max-speed` from `phy_mode`

| `phy_mode` | `nvidia,max-speed` | ODMDATA speed token |
|---|---|---|
| `25g-r` | `25000` | `mgbe<N>_speed=25G` (sub-node form) or `mgbe<N>-speed-25G` (`/uphy` top-level form) |
| `10g-kr` | `10000` | `mgbe<N>_speed=10G` / `mgbe<N>-speed-10G` |
| `1000base-x` | `1000` | `mgbe<N>_speed=1G` / `mgbe<N>-speed-1G` |

**Choice between sub-node vs top-level token form** is dictated by
the live BPMP DTB schema — decompile and check before emitting:

```bash
BPMP_DTB="<bsp_image.root_path>/Linux_for_Tegra/bootloader/$(grep '^BPFDTB_FILE=' <carrier conf> | cut -d'"' -f2)"
dtc -I dtb -O dts "$BPMP_DTB" | sed -n '/^\t\(uphy\|mgbe\) /,/^\t};/p'
```

- If `/mgbe/mgbe@N` subtree exists in BPMP DTB → use sub-node form
  (`mgbe<N>_status=okay`, `mgbe<N>_speed=25G`).
- If only `/uphy/mgbe<N>-speed` exists (stock Thor r38.4) → use
  top-level dashed form (`mgbe<N>-speed-25G`); the `status` knob is
  **not available** — kernel-DT overlay alone owns enable/disable.

**Kernel-DT K-stock probe — node-selection + indent-bounded extraction
(REQUIRED).** Before emitting any kernel-DT overlay fragment, re-read
each target MGBE node's stock `status` from the reference kernel DTB.
Two parser traps to avoid:

1. **Filter by `compatible`, not by node name.** The kernel DTB
   contains both NVIDIA controller nodes
   (`compatible = "nvidia,tegra264-mgbe"`, one per controller, bearing
   the real `status`) and unrelated sibling Ethernet helpers in the
   same `ethernet@<addr>` namespace. Match on the compatible string,
   not on the bare `ethernet@` glob.

   ```bash
   grep -nE 'compatible.*nvidia,tegra[0-9]+-mgbe\b' /tmp/k.dts
   ```

2. **Bound the block by matching the closing brace at the same indent
   as the opening line.** A naive `awk '/ethernet@<addr> {/,/^\t};/'`
   range will trip on the first `};` of any nested child (mdio, PHY,
   ipv4 subnodes) and return the wrong `status` from the inside of a
   sub-node. Use:

   ```bash
   ln=$(grep -nE "^[[:space:]]+ethernet@$addr \{" /tmp/k.dts | head -1 | cut -d: -f1)
   indent=$(sed -n "${ln}p" /tmp/k.dts | sed -E 's/[^[:space:]].*//')
   awk -v start="$ln" -v ind="$indent" \
     'NR>=start { if (NR>start && $0==ind"};") exit; print }' /tmp/k.dts
   ```

This is the same fix that protects `/jetson-customize-pcie` from
reading `pci-host-ecam-generic` instead of `nvidia,tegra264-pcie`. Apply
both rules whenever you decode a Tegra IP block.

The two grammars are not interchangeable; same rule as
`jetson-customize-uphy` Step 6's "wrong forms" table. Cite the BPMP DTB
inspection in `notes[]`.

### Step 4 — Verify HSIO pins + auto-fix via `/jetson-customize-pinmux`

For every controller with `phy_attach_kind=="phy"`, verify ancillary
pin SFIOs and route mismatches to `/jetson-customize-pinmux`. Expected per
controller (where `<N>` is the controller index):

| Signal | SFIO | Direction | Initial state | Notes |
|---|---|---|---|---|
| `XFI<N>_MDC` | `mgbe<N>_mdc` | output | n/a | |
| `XFI<N>_MDIO` | `mgbe<N>_mdio` | bidirectional | n/a | open-drain enable |
| `XFI<N>_RESET` | `gpio` (output) | output | high | drives PHY reset; offset/bank captured in Step 2 |
| `XFI<N>_INT_N` *(when wired)* | `gpio` | input | n/a | |

Invoke the shared verifier:

```python
# Pseudo-flow — actual call from agent
from pin_verifier import verify_pins
result = verify_pins(
    kb_dir=<workspace>/target-platform/<profile-stem>.jetson-customize-mgbe,
    l4t_dir=<source.root_path>/Linux_for_Tegra,
    expected=[
        {"signal_or_pin": f"XFI{N}_MDC",    "sfio": f"mgbe{N}_mdc",  "direction": "output",        "initial_state": "n/a", "reason": f"MGBE{N} MDC"},
        {"signal_or_pin": f"XFI{N}_MDIO",   "sfio": f"mgbe{N}_mdio", "direction": "bidirectional", "initial_state": "n/a", "reason": f"MGBE{N} MDIO (open-drain)"},
        {"signal_or_pin": f"XFI{N}_RESET",  "sfio": "gpio",          "direction": "output",        "initial_state": "high", "reason": f"MGBE{N} PHY reset"},
    ],
    io_label="mgbe",
)
```

`pin_verifier.py` is located at
`<workspace>/.claude/scripts/pin_verifier.py`. It is
**read-only**; it reports `{ok: [...], mismatches: [...]}`. For each
mismatch, surface a `"I'm setting pin X to SFIO Y because <reason>"`
message and call `/jetson-customize-pinmux set-pin` (or have the user
re-run `/jetson-customize-pinmux` directly) to stage the fix. Do not
auto-fix without confirmation.

> **Integration note:** `pin_verifier.py` was written for the
> legacy `session.json` model and reads `session.carrier_pinmap` /
> `session.carrier_short`. A compatible shim under
> `<workspace>/target-platform/<profile-stem>.jetson-customize-mgbe/session.json`
> is generated each run (same pattern as `jetson-customize-pinmux` Step 7)
> so the verifier's interface keeps working unchanged.

### Step 5 — (no ODMDATA edits)

All MGBE ODMDATA tokens (`mgbeN-speed-<rate>`, `mgbeN-speed-del`,
sub-node `mgbeN_status=*` where the BPMP DTB exposes it) are emitted
by `/jetson-customize-uphy` in its single atomic ODMDATA commit.
This skill MUST NOT touch `ODMDATA=`. Cite the BPMP DTB token-form
inspection (sub-node vs top-level) in `notes[]` so the UPHY skill's
emission can be audited.

### Step 6 — Append fragments to the composite custom overlay

This skill writes per-controller fragments into the single composite
custom overlay `.dts` for the active target. Filename / location /
skeleton / append protocol are documented in
`../../../references/bsp-customization-kernel-dtb.md` —
follow that doc exactly. The notes below cover only what's
MGBE-specific.

**Marker for this skill's fragments:** `/* custom-bsp: mgbe:mgbe<N> */`
(one fragment per controller; sub-key is the controller name). On
re-run, delete every fragment matching this skill's marker pattern
before appending the new ones (see context doc Step 5 of the append
protocol).

**Fragment body — one per enabled controller:**

```dts
fragment@<N> { /* custom-bsp: mgbe:mgbe<id> */
    target = <&mgbe<id>>;
    __overlay__ {
        status = "okay";
        phy-mode = "<phy_mode>";
        nvidia,max-speed = <<max_speed_mbps>>;

        /* phy_attach_kind == "phy" only — omit block for "direct": */
        phy-handle = <&mgbe<id>_phy>;
        nvidia,phy-reset-gpios = <&gpio_main <pin> 0>;
        mdio {
            #address-cells = <1>;
            #size-cells = <0>;
            mgbe<id>_phy: phy@<addr> {
                reg = <<addr>>;
            };
        };
    };
};
```

**`#address-cells = <1>; #size-cells = <0>;` on the `mdio` child is
mandatory** when `phy_attach_kind=="phy"`. Without it the kernel
rejects the `phy@<addr>` reg property at probe.

For `phy_attach_kind=="direct"`, drop `phy-handle`,
`nvidia,phy-reset-gpios`, and the entire `mdio { ... }` subnode; the
SerDes auto-negotiates without an external PHY.

**Compatible string is set on the composite root, not per-fragment** —
see the context doc. This skill does not touch `compatible`; if the
composite root's `compatible` is missing your active-DUT string,
fix the composite root (which any previous run set), don't widen
inside the fragment.

**Mandatory marker comment metadata** (same audit-trail contract as
`jetson-customize-uphy` Step 7.5): the fragment marker `/* custom-bsp:
mgbe:mgbe<N> */` is enough for grep-based discovery, but include a
one-line provenance comment inside the fragment summarizing `(a)`
phy mode + speed + attach kind, `(b)` pin & reset GPIO, `(c)` source
citation (Adaptation Guide section, schematic page range, MDG table,
PHY datasheet). Place it as the first line inside `__overlay__`.

Commit the composite `.dts` in the `bsp_sources/` mono-repo
(filename resolved per the context doc):

```bash
git -C <source.root_path>/bsp_sources add <composite-relative-path>
git -C <source.root_path>/bsp_sources commit -m "jetson-customize-mgbe: append mgbe<N> @ <phy_mode> to <board-tag>-custom"
```

**Pre-flight sanity check** (also documented in the context doc
Step 10 of the append protocol):

```bash
cpp -nostdinc -x assembler-with-cpp <composite-abs-path> /tmp/composite.tmp.dts
dtc -@ -I dts -O dtb -o /tmp/composite.dtbo /tmp/composite.tmp.dts
fdtoverlay -i <bsp_image.root_path>/Linux_for_Tegra/kernel/dtb/<carrier-base>.dtb \
           -o /tmp/merged.dtb /tmp/composite.dtbo
```

Fail the commit (`git restore --staged <composite-path>` and exit)
if either step errors. `/jetson-build-source` re-compiles the
composite via `nvidia-dtbs`; this is a pre-flight sanity check only.

**Flash-conf and Makefile registration are NOT this skill's job** —
`jetson-build-source` Step 5.0a owns both (it idempotently adds
`dtbo-y += <composite>.dtbo` and the single `OVERLAY_DTB_FILE+=`
line on its next run). Do not patch the carrier flash conf here.

### Step 8 — Run-state JSON sidecar + summary

**Sidecar** at
`<workspace>/target-platform/<profile-stem>.jetson-customize-mgbe.json`:

```json
{
  "generator": "jetson-customize-mgbe",
  "generator_version": "0.1.0",
  "generated_at": "<ISO-8601 UTC>",
  "active_profile": "<profile-stem>.yaml",
  "bsp_version": "<bsp_image.version>",
  "controllers": [{
    "id": "mgbe0",
    "enable": true,
    "phy_mode": "25g-r",
    "phy_attach_kind": "phy",
    "phy_i2c_bus": "i2c@3160000",
    "phy_i2c_addr_hex": "0x56",
    "phy_reset_gpio_offset": 12,
    "phy_reset_gpio_bank": "gpio_main",
    "max_speed_mbps": 25000
  }],
  "compatible_list": ["nvidia,p8181-0001+p3834-0008", "nvidia,tegra264"],
  "bpmp_token_grammar": "subnode" | "top_level_uphy",
  "odmdata_tokens": ["mgbe0_status=okay", "mgbe0_speed=25G"],
  "overlay": {
    "composite_dts_path": "<source.root_path>/bsp_sources/hardware/nvidia/<chip-dir>/nv-public/<sub>/tegra<soc>-<carrier-id-sku>+<module-id>-xxxx-custom.dts",
    "composite_dtbo": "tegra<soc>-<carrier-id-sku>+<module-id>-xxxx-custom.dtbo",
    "fragments_appended": ["mgbe:mgbe<N>"],
    "skipped": false
  },
  "pin_verifier_mismatches": [],
  "warnings": [],
  "notes": [],
  "sources": ["..."],
  "commit_shas": {
    "overlay_tracker": "<short SHA>",
    "bsp_sources": "<short SHA>"
  }
}
```

Atomic write + idempotency contract: same as `jetson-customize-uphy` Step 5.

**Headline + breakdown** mirrors `jetson-customize-uphy` Step 8 — adapted
for MGBE (controller list, PHY mode/speed/attach, ODMDATA tokens,
overlay path, commit SHAs, BPMP DTB grammar choice, pin-verifier
mismatches surfaced).

**Next step (interactive prompt chain):**

After the summary table is printed, drive the downstream chain via
sequential `AskUserQuestion` prompts. Each prompt needs an explicit
`yes`. On `no` (or any abort), print the remaining manual run-chain
and exit. Never substitute a printed "Next step: …" line for the
prompts.

If Step 4 surfaced SFIO mismatches, prepend prompt **(0)** first:

0. **Re-run `/jetson-customize-pinmux` to fix the surfaced SFIO
   mismatches?** — on `yes` invoke `/jetson-customize-pinmux`, then
   continue to (1).

Then in order:

1. **Customize any other I/O before build?** — offer
   `/jetson-customize-pinmux`, `/jetson-customize-uphy`,
   `/jetson-customize-pcie`, `/jetson-customize-camera`,
   `/jetson-customize-usb`, `/jetson-customize-clocks`,
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

