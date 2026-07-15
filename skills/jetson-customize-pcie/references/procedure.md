# Procedure — jetson-customize-pcie

Companion reference for `SKILL.md`. Follow steps in order. Cross-skill
context lives in `bsp-customization-workflow.md` and
`bsp-customization-kernel-dtb.md` under `context/`.

## Step 1 — Resolve active target + open source-of-truth documents

Same as `jetson-customize-uphy` Step 1 + `jetson-customize-mgbe` Step 1 (refuse
table + identifier resolution + document resolution). Additionally
resolve:

| Var | Source |
|---|---|
| `<carrier-pinmap>` | `<workspace>/target-platform/<profile-stem>.jetson-customize-pinmux/pinmap/<custom-carrier>.json` *(produced by `jetson-customize-pinmux` `probe`)* |
| `<devkit-pinmap>` | Optional — `<workspace>/target-platform/<profile-stem>.jetson-customize-pinmux/pinmap/<reference_devkit-name>.json` if a separate devkit pinmap was probed |
| `<ref-dtb>` | `<bsp_image.root_path>/Linux_for_Tegra/kernel/dtb/tegra<soc>-<reference_devkit-carrier>-<reference_devkit-module>-nv.dtb` |
| `<uphy-state>` | `<workspace>/target-platform/<profile-stem>.jetson-customize-uphy.json` — drives the `enable` decision in Step 5 |

**Refuse** if `<uphy-state>` is missing — run `/jetson-customize-uphy` first.

## Step 2 — Diff PCIe topology — devkit vs custom carrier

Decompile the reference DTB and enumerate every PCIe controller:

```bash
dtc -I dtb -O dts <ref-dtb> > /tmp/k.dts
```

**Use the NVIDIA Tegra controller nodes, NOT the ECAM root nodes.** The
kernel DTB contains two unrelated families of `pcie@<addr>` nodes:

| Compatible | Role | Use for this skill? |
|---|---|---|
| `nvidia,tegra264-pcie` (Thor) / `nvidia,tegra234-pcie` (Orin) | NVIDIA Tegra PCIe controller (one per CID) — bears the real `status` this skill flips | **yes** |
| `pci-host-ecam-generic` | Post-enumeration PCI bus root (one per `linux,pci-domain`) — `status` here reflects bus presence, NOT controller enable state, and is typically `disabled` for all non-default domains | **no — ignore** |

Filter by `compatible`, not by node name:

```bash
grep -nE 'compatible.*nvidia,tegra[0-9]+-pcie\b' /tmp/k.dts
```

Then for each matched `pcie@<addr>` block (the NVIDIA controller, NOT the
ECAM root) extract: address, `linux,pci-domain` (→ controller `C<N>`
mapping), stock `status`, `num-lanes`, `max-link-speed`.

When extracting a node body, bound the awk/sed range by matching the
**closing brace at the same indent as the opening line** — a naive
`/pcie@<addr> \{/,/^\t};/` range will leak past sub-nodes and report the
wrong `status` from a nested child.

Build the diff table:

| Controller | Addr | Devkit wiring | Custom carrier wiring | Stock status | Schematic page |
|---|---|---|---|---|---|

Wiring sources:
- **Devkit** — `<devkit-pinmap>` (if present) + Adaptation Guide
  §"PCIe" for the reference devkit; cross-check `linux,pci-domain` in
  the DTB.
- **Custom carrier** — `<carrier-pinmap>` for `PEX<N>_LN_*`,
  `PEX<N>_CLKREQ_*`, `PEX<N>_RST_*` + grep the schematic for
  `PCIe x<N>` / `PEX_<controller>_*` net labels and the receptacle
  they land on (M.2 M/E, x16 slot, on-board NIC).

Mark controllers whose carrier wiring differs from devkit ("changed").
Surface the table to the user (with schematic page numbers cited)
**before** posing any question.

## Step 3 — Ask which controller(s) to customize

`AskUserQuestion` with `multiSelect: true`, options built from Step 2:

1. **Changed controllers** first, marked `(changed: <one-line delta>)`.
2. **Unchanged controllers** next, marked `(stock)`.
3. **All controllers** — shortcut option.

Never offer a controller the in-tree DTB does not expose.

## Step 4 — Per-controller verification: pinmap + schematic + pin-verifier

For each selected controller, run a per-controller sub-loop.

**4a. Cross-check carrier pinmap + schematic.** Required ancillary
pins per controller:

| Signal | SFIO | Direction | Initial state | Optional? |
|---|---|---|---|---|
| `PE<N>_CLKREQ_L` | `pe<N>_clkreq_l` | input | n/a | no |
| `PE<N>_RST_L` | `pe<N>_rst_l` | output | high | no |
| `PE<N>_WAKE_L` | `pe<N>_wake_l` | input | n/a | yes (when wired) |

For each pin: resolve via `<carrier-pinmap>` to CVM ball + net label,
verify the net actually lands on the receptacle in the schematic,
verify the cloned pinmux DTSI in the overlay tracker has the pin at
the right SFIO. An unrouted pin → record in `notes[]` and surface
**before** asking customization questions so the user can drop the
controller.

**4b. Auto-invoke `pin_verifier.py` for SFIO mismatches** (same
contract as `jetson-customize-mgbe` Step 4; `kb_dir=<workspace>/target-platform/<profile-stem>.jetson-customize-pcie`, `io_label="pcie"`, expected list from 4a). For each mismatch, surface
`"I'm setting pin <X> to SFIO <Y> for <reason>"` and route to
`/jetson-customize-pinmux set-pin`. Verifier is read-only; the
`session.json` shim under `--kb-dir` is regenerated each run.

## Step 5 — Auto-derive per-controller plan, then confirm-or-customize

**5a. Auto-derive the per-controller plan.** For each controller in
the user's selection from Step 3:

- **`enable`** — derive from `<uphy-state>`'s `uphy.config` +
  `uphy.config_uphy1` against the Adaptation Guide §"Configure the
  UPHY Lane" table:
  - **UPHY allocates lanes** → mandatorily enabled (`status=okay` on
    both surfaces). Offering "keep disabled" would create an
    inconsistent ODMDATA-vs-overlay state — **forbidden**.
  - **UPHY does NOT allocate** → mandatorily disabled (`status=disabled`
    on both surfaces). Enabling would point the kernel at a powered-
    down PHY → link timeouts.
  Surface the UPHY-driven decision in the question preamble for
  auditability.
- **`lanes`** — from the UPHY config's lane-width entry for this
  controller (the table in the Adaptation Guide §"Configure the
  UPHY Lane" lists per-config controller widths).
- **`speed`** — from Adaptation Guide §PCIe per-controller max
  (Gen5 on Thor RC ports; Gen4 on Orin Tegra234 RC ports).
- **`mode`** — **hard-pinned to `"rc"`, never asked.** Every emitted
  `controllers[*].mode` is `"rc"`, every ODMDATA token is
  `pcie@<N>_pcie-mode=1`, every overlay fragment targets the RC sub-
  node. EP only via explicit `mode_override="ep"` in Step 5c
  fallthrough.

**5b. Confirm-or-customize gate (mandatory).** Render a per-controller
table:

| Controller | Enable | Lanes | Speed | Mode | Derivation |
|---|---|---|---|---|---|

Then issue ONE `AskUserQuestion`:

```
Question: "PCIe plan derived from <uphy-state> + Adaptation Guide.
           Proceed as-is or customize?"
Header:   "PCIe plan"
Options:
  - "Proceed with auto-derived plan (Recommended)"     → skip 5c, go to Step 6
  - "Override lanes / speed for one or more controllers" → 5c
  - "Mark a controller for EP-mode override"           → 5c (mode_override branch)
```

Default is one click. **Do NOT skip this gate even when every value
is determinable** — silent commit of a non-trivial customization is a
known failure mode (consistent with `jetson-customize-uphy`'s explicit
choice point).

**5c. Per-controller customization (fallthrough only).** One batched
`AskUserQuestion` per controller, sub-questions for lanes / speed /
`mode_override` as needed. If schematic suggests x4 but Adaptation
Guide caps at x2 for this controller, surface in `guidance`; let
the user decide.

## Step 6 — (no ODMDATA edits)

All PCIe-related ODMDATA tokens (`pcie@N_status=*`,
`pcie@N_max-link-speed`, `pcie@N_pcie-mode`, `pcie@N_clk-scheme`,
`pcie-cN-endpoint-enable`) are emitted by `/jetson-customize-uphy`
in its single atomic ODMDATA commit. This skill MUST NOT touch
`ODMDATA=`. Step 8 cross-checks consistency; on disagreement, stop
and ask the operator.

## Step 7 — Append fragments to the composite custom overlay

This skill writes per-controller fragments into the single composite
custom overlay `.dts` for the active target. Filename / location /
skeleton / append protocol are documented in
`bsp-customization-kernel-dtb.md` — follow that doc exactly. The notes
below cover only what's PCIe-specific.

**Marker for this skill's fragments:** `/* custom-bsp: pcie:pcie@<addr> */`
(one fragment per controller in the delta; sub-key is the kernel-DT
node address). On re-run, delete every fragment matching this skill's
marker pattern before appending the new ones (context doc Step 5 of
the append protocol).

**Fragment body — one per controller in the delta:**

```dts
fragment@<N> { /* custom-bsp: pcie:pcie@a8b0000000 */
    target-path = "/bus@0/pcie@a8b0000000";   /* C1 — addr from in-tree DTB, not TRM */
    __overlay__ {
        status = "okay";
        num-lanes = <1>;
        max-link-speed = <4>;        /* Gen4 */
    };
};

fragment@<N+1> { /* custom-bsp: pcie:pcie@b8b0000000 */
    target-path = "/bus@0/pcie@b8b0000000";   /* C3 — disabled, no UPHY allocation */
    __overlay__ {
        status = "disabled";
    };
};
```

**Node addresses come from `dtc -I dtb -O dts <ref-dtb>`**, NOT the
SoC TRM. TRM lists register bases; kernel DT may use a different
`reg` layout (RP windows vs ECAM).

**Skip a fragment entirely if every property already matches stock.**
No-op overlays are noise. Always emit the fragment when `status`,
`num-lanes`, or `max-link-speed` differs from stock.

**For `mode_override == "ep"`**: target the endpoint sub-node
(`/bus@0/pcie_ep@<addr>` from the reference DTB), set
`compatible = "nvidia,tegra<soc>-pcie-ep"` inside `__overlay__`
(not on the composite root), add `nvidia,refclk-select-gpios` when
the carrier sources refclk externally. Cite Adaptation Guide §"PCIe
Endpoint Mode" in the provenance comment.

**Compatible string is set on the composite root, not per-fragment.**
If it's missing the active-DUT string, fix the composite root, not
the fragment. The EP-mode `compatible` above is a kernel-DT property
on the endpoint node, not a UEFI plugin-manager gate.

**Mandatory provenance comment as the first line inside
`__overlay__`** (audit trail): controller delta vs stock kernel-DT,
which `uphyX-config-N` justifies enable/disable, citation list
(Adaptation Guide §, schematic page, MDG table, DTB path).

Commit the composite `.dts` in the `bsp_sources/` mono-repo
(filename resolved per the context doc):

```bash
git -C <source.root_path>/bsp_sources add <composite-relative-path>
git -C <source.root_path>/bsp_sources commit -m "jetson-customize-pcie: append pcie@<N> ... to <board-tag>-custom"
```

**Pre-flight sanity check** (also documented in the context doc
Step 10 of the append protocol):

```bash
cpp -nostdinc -x assembler-with-cpp <composite-abs-path> /tmp/composite.tmp.dts
dtc -@ -I dts -O dtb -o /tmp/composite.dtbo /tmp/composite.tmp.dts
fdtoverlay -i <bsp_image.root_path>/Linux_for_Tegra/kernel/dtb/<carrier-base>.dtb \
           -o /tmp/merged.dtb /tmp/composite.dtbo
```

Fail the commit if either step errors. `/jetson-build-source`
re-compiles the composite via `nvidia-dtbs`; this is a pre-flight
sanity check only.

**Flash-conf and Makefile registration are NOT this skill's job** —
`jetson-build-source` Step 5.0a owns both. Do not patch the carrier
flash conf here.

## Step 8 — Cross-check ODMDATA vs overlay (dry-run guard)

Before declaring the run complete, verify per-controller consistency:

| ODMDATA `pcie@<N>_status=` | Overlay fragment | Verdict |
|---|---|---|
| `okay` | `status="okay"` | ok |
| `okay` | absent | ok if stock kernel-DT already enables this controller |
| `okay` | `status="disabled"` | **abort** — contradictory; flag and re-prompt Step 5 |
| `disabled` | `status="disabled"` | ok |
| `disabled` | absent | ok if stock kernel-DT already disables |
| `disabled` | `status="okay"` | **abort** — contradictory |
| absent | any | ok (skipped from this run) |

A contradictory row means the commit pair is inconsistent. **Stop and
ask the user how to recover** — surface the contradictory row, the
two offending commits (overlay tracker + bsp_sources), and let the
operator choose `git revert`, manual edit + amend, or re-running
Step 5 with corrected inputs. Never run `git reset --hard`
autonomously.

## Step 9 — Run-state JSON sidecar + summary

**Sidecar** at
`<workspace>/target-platform/<profile-stem>.jetson-customize-pcie.json`:

```json
{
  "generator": "jetson-customize-pcie",
  "generator_version": "0.1.0",
  "generated_at": "<ISO-8601 UTC>",
  "active_profile": "<profile-stem>.yaml",
  "bsp_version": "<bsp_image.version>",
  "topology_diff": [
    {"id": "pcieN", "addr": "pcie@<addr>", "domain": N, "stock_status": "okay|disabled", "devkit_receptacle": "...", "carrier_receptacle": "...", "changed": true}
  ],
  "controllers": [
    {"id": "pcie4", "addr": "pcie@c0b0000000", "domain": 4, "enable": true, "lanes": 8, "mode": "rc", "speed": "gen5",
     "uphy_derivation": "uphy1-config-0 allocates UPHY1 L0..L7 to pcie@C4 x8",
     "ancillary_pins": {"clkreq": "pex_l4_clkreq_n_pp4", "rst": "pex_l4_rst_n_pp5", "wake": null}}
  ],
  "compatible_list": ["nvidia,<carrier-id-sku>+<module-id-sku>", "nvidia,tegra<soc>"],
  "odmdata_tokens": ["pcie@4_status=okay", "pcie@4_max-link-speed=5", "pcie@4_pcie-mode=1"],
  "overlay": {
    "composite_dts_path": "...tegra<soc>-<carrier-id-sku>+<module-id>-xxxx-custom.dts",
    "fragments_appended": [{"marker": "pcie:pcie@c0b0000000", "delta": {"status": "okay", "num-lanes": 8, "max-link-speed": 5}}]
  },
  "pin_verifier_mismatches": [], "warnings": [], "notes": [], "sources": [],
  "commit_shas": {"overlay_tracker": "<short SHA>", "bsp_sources": "<short SHA>"}
}
```

Atomic write + idempotency: same as `jetson-customize-uphy` Step 5.

**Headline + breakdown** mirrors `jetson-customize-uphy` Step 8 — per-
controller delta, ODMDATA tokens, overlay path, commit SHAs,
mode_override flags, pin-verifier mismatches surfaced.

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
   `/jetson-customize-mgbe`, `/jetson-customize-camera`,
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
