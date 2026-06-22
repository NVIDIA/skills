---
name: jetson-customize-clocks
description: Use to lock/cap Jetson CPU/GPU/EMC clocks, toggle EMC/CPU DVFS, or change cpufreq governors by editing BPMP DTB and nvpower.sh pre-flash. Do NOT use for live tuning or nvpmodel edits.
version: 0.0.1
license: "Apache-2.0"
metadata:
  data-classification: public
  author: "Jetson Team"
  tags:
    - clocks
    - cpu
    - gpu
    - emc
    - dvfs
    - bwmgr
    - bpmp
    - nvpower
    - cpufreq
    - devfreq
  domain: clocks
---

# Customize Clocks

## Purpose

Customize CPU, GPU, and EMC clock behavior on a Jetson target by editing files under `Linux_for_Tegra/` before flashing the image. Two layers are in scope:

- The **BPMP DTB** at `Linux_for_Tegra/bootloader/<BPFDTB_FILE>` — per-clock `max-rate-custom` ceilings, plus the EMC DVFS gate (bwmgr + cactmon on all SoCs; osp-controller on T26x only).
- **nvpower.sh** at `Linux_for_Tegra/rootfs/etc/systemd/nvpower.sh` — cpufreq / devfreq governors and (optionally) per-device min / max / static rates written to sysfs at boot.

Common triggers: "lock CPU/GPU/EMC frequency", "pin GPU to Fmax", "pin EMC to MAXN", "disable/enable EMC DVFS", "disable/enable CPU DVFS", "set CPU/GPU max rate", "change cpufreq governor".

Out of scope: runtime clock tuning on a live target (no flash step), nvpmodel power-mode edits (use the sibling skill `/jetson-customize-nvpmodel`), and silicon-ceiling overrides (`max-rate-maxn` is read-only).

## Prerequisites

Resolve the active profile per
[`../../context/target-platform-contract.md`](../../context/target-platform-contract.md).
Refuse and route in these cases:

| Condition | Refuse with |
|---|---|
| No active profile, or `active: NA` | Route to `/jetson-set-target` or `/jetson-init-target`. |
| Profile lacks `bsp_image:` block | Route to `/jetson-init-image`. |
| `<bsp_image.root_path>/Linux_for_Tegra/` missing | Route to `/jetson-init-image`. |
| `<source.root_path>/Linux_for_Tegra/` missing or not a git repo | Route to `/jetson-init-source`. |

Resolve paths:

- `<bsp_image.root_path>` from `bsp_image.root_path:` if present, else `<workspace>/Image`.
- `<source.root_path>` from `source.root_path:` if present, else `<workspace>/Source`.

`<bsp_image.root_path>` is **read-only** for this skill; every write
(Operation 1's BPMP DTB and Operation 2's `nvpower.sh`) lands under
`<source.root_path>` (the overlay tracker). This is the workflow
invariant in
[`../../context/bsp-customization-workflow.md#workflow-invariants`](../../context/bsp-customization-workflow.md#workflow-invariants) —
hand-editing upstream silently destroys the diff trail and makes
`/jetson-promote-image` a noop.

## Instructions

1. Resolve the prerequisites above (active profile, BSP image extracted, source overlay tracker initialized).
2. Pick the operation from the table below.
3. Follow the linked procedure section — Operation 1 (BPMP DTB), Operation 2 (`nvpower.sh`), or the MAXN recipe for both.
4. Commit the edit inside the overlay tracker per each Operation's commit convention.
5. Deploy with `/jetson-promote-image` → `/jetson-flash-image`. The new BPMP DTB and `nvpower.sh` take effect on the next boot.

### Supported operations

| Operation | Where the edit lives | Procedure section |
|---|---|---|
| Lock a CPU / GPU clock to a specific rate | BPMP DTB `max-rate-custom` on the clock node + `nvpower.sh` governor `performance` | "Content edit: `max-rate-custom`" + "Pick the edit" |
| Lock EMC at its init rate (disable EMC DVFS) | BPMP DTB: `bwmgr.enabled = 0`, `cactmon.enabled = 0`, plus `/delete-node/ osp-controller` on T26x only | "Content edit: EMC DVFS disable / enable" |
| Re-enable EMC DVFS | BPMP DTB: `bwmgr.enabled = 1`, `cactmon.enabled = 1`, restore `osp-controller` on T26x | "Content edit: EMC DVFS disable / enable" |
| Pin everything to MAXN for stress runs | Combine the above + nvpmodel MAXN as boot default | see Recipe |
| Lower a clock's hard ceiling without locking | BPMP DTB `max-rate-custom` only | "Content edit: `max-rate-custom`" |
| Bound a device's rate without pinning | `nvpower.sh` min/max via sysfs | "Pick the edit" |

## Operation 1 — BPMP DTB edits

Follow the BPMP-DTB customization protocol in
[`../../references/bsp-customization-bpmp-dtb.md`](../../references/bsp-customization-bpmp-dtb.md).
The protocol owns the mechanics — pristine import on first touch,
`dtc` decompile, recompile, sanity-check, commit. This skill
supplies only the **clock-specific content** (which nodes and
properties to edit during the protocol's "Edit the DTS" step).

The edited `.dtb` lands in the `<source.root_path>/Linux_for_Tegra/`
overlay tracker. `/jetson-promote-image`'s channel A walks the
tracker and copies the file into `bsp_image`. **Do not edit
`<bsp_image.root_path>/Linux_for_Tegra/bootloader/<bpmp-dtb>`
directly** — that's the promote output, not an input.

### Resolve the SKU-correct BPMP DTB

Per the protocol's "Resolving the active BPMP DTB" section, read
`BPFDTB_FILE` from the active flash conf. For the common Thor /
single-SKU conf shapes this is the static `BPFDTB_FILE=...` line
in the per-board `.conf` and the value is authoritative as-is.

For **SKU-multiplexed conf shapes** (Orin AGX devkit conf chain
that selects a different BPMP DTB per `board_sku`/`board_FAB`
via `update_flash_args_common` — see
[`../../context/bsp-customization-software-layers.md#per-board-conf-dispatch--update_flash_args_common`](../../context/bsp-customization-software-layers.md#per-board-conf-dispatch--update_flash_args_common)),
walk the dispatch chain with `board_sku=<module.sku>` and
`board_FAB=<module.revision or empty>` from the active profile,
and read `BPFDTB_FILE` from the dispatch output — **not** from
the static line of the per-board `.conf`. Static and dispatched
values match for non-multiplexed confs; the dispatch is
mandatory only when the conf chain conditionally overrides
`BPFDTB_FILE`.

### List effective max rates (inspection)

Inspect both layers of the runtime ceiling — see [`references/clock-control-model.md#effective-runtime-ceiling`](references/clock-control-model.md#effective-runtime-ceiling) — before deciding on a `max-rate-custom` value.

Inspection cookbook (BPMP-side decompile + grep; nvpmodel-side awk over the boot default mode) is in [`references/bpmp-dtb-clock-edits.md#inspection-cookbook`](references/bpmp-dtb-clock-edits.md#inspection-cookbook).

For the nvpmodel layer see [`/jetson-customize-nvpmodel`](../jetson-customize-nvpmodel/SKILL.md).

This step does not mutate state — it's a precondition for sizing
the edit in the "Content edit: `max-rate-custom` on a named clock node" step.

### Content edit: `max-rate-custom` on a named clock node

During the "Edit the DTS" step of the protocol, modify the property
inside the named clock node — never `lateinit`. `max-rate-custom`
must be strictly below the clock's hard cap (`max-rate-maxn` if
defined, otherwise the live `max_rate` from a running target of
the same chip / SKU).

DTS edit form, semantics, and the nvpmodel ↔ BPMP clock-node
mapping live in [`references/bpmp-dtb-clock-edits.md`](references/bpmp-dtb-clock-edits.md).

Then hand control back to the protocol — its "Recompile", "Sanity-check
the recompiled blob", "Stage in the overlay tracker", and "Cleanup"
steps cover the rest.
Commit-message convention per the protocol:
`<BPMP_BASENAME>: jetson-customize-clocks — <clock-node> max-rate-custom = <value>`.

### Content edit: EMC DVFS disable / enable

Default behavior (EMC DVFS on) requires no edit. Disabling EMC
DVFS is a **multi-node edit** applied inside the same "Edit the DTS" step of
the protocol, **not** a `bwmgr` toggle:

| # | Edit | Scope |
|---|---|---|
| 1 | `bwmgr.enabled = <0x00>` | All SoCs, mandatory |
| 2 | `cactmon.enabled = <0x00>` | All SoCs, mandatory |
| 3 | `/delete-node/ osp-controller` | **T26x (Thor) mandatory** — T23x (Orin) has no such node, skip |

Detection: `dtc -I dtb -O dts <bpmp-dtb> | grep -c osp-controller`
— zero hits ⇒ T23x path. Full DTS snippets, the surviving-paths
failure modes, and the re-enable procedure are in
[`references/emc-dvfs-disable.md`](references/emc-dvfs-disable.md).

Apply the protocol's "Recompile" through "Cleanup" steps once the multi-node edit is in
place. Commit-message convention:
`<BPMP_BASENAME>: jetson-customize-clocks — EMC DVFS disable (bwmgr + cactmon[+ osp-controller])`.

Disabling raises idle power; intended for stress / performance
tests, not production rootfs.

### Re-run + idempotency

Per the protocol's "Re-runnability" section, re-running this
skill with the same target value produces a no-op commit. Re-
running with a different value rewrites the same property — `git
log -- $BPMP_REL` shows the per-run history. To return a clock
to its `max-rate-maxn` ceiling, edit the DTS to remove the
`max-rate-custom` line and recompile.

## Operation 2 — nvpower.sh edits

Edits `nvpower.sh`, which runs at boot via `nvpower.service` to set
cpufreq / devfreq governors and rates.

### The per-script file

The script this Operation edits has the relative path:

```
Linux_for_Tegra/rootfs/etc/systemd/nvpower.sh
```

It lives in **two** roots; the Operation walks both:

| Role | Location | Skill writes? |
|---|---|---|
| Detection + pristine source | `<bsp_image.root_path>/Linux_for_Tegra/rootfs/etc/systemd/` | no — read-only |
| Overlay edit target + git commit | `<source.root_path>/Linux_for_Tegra/rootfs/etc/systemd/` | yes |

Subsequent sub-steps refer to **the per-script file** to mean the overlay
copy under `<source.root_path>`. The `<bsp_image.root_path>` copy is read
once during the pristine-import step below, then never touched again.

### Overlay edit recipe (apply before editing nvpower.sh)

Follow the canonical
[Off-skill edits recipe](../../context/bsp-customization-workflow.md#off-skill-edits)
in the workflow doc — pristine import + customization commit pair, both
gated by the preview gate. `nvpower.sh` is a single file with no
propagation set; one pristine commit + one customization commit covers
the entire change.

Concrete substitutions for this skill:

- `<rel>/<file>` is `rootfs/etc/systemd/nvpower.sh`.
- Suggested pristine-import message:
  `import pristine: rootfs/etc/systemd/nvpower.sh`,
  body `Source: <bsp_image.root_path>/Linux_for_Tegra/ (BSP <bsp_image.version>)`.
- Suggested customization-commit header:
  `jetson-customize-clocks: nvpower.sh <summary>`,
  body lines like `set_cpufreq_governor: desired_cpufreq_gov "schedutil" -> "performance"`.

### Pick the edit

Function locations (`set_cpufreq_governor`, `set_devfreq_governor`), common-edit recipes (pin to Fmax, static rate, min/max bounds), and the `nvidia-l4t-init` package-upgrade caveat live in [`references/nvpower-sh-edits.md`](references/nvpower-sh-edits.md).

### Deploy

The customization commit in the overlay tracker does not reach the device
on its own. The Deploy chain:

1. **`/jetson-promote-image`** — copies every tracked file in the overlay
   into `<bsp_image.root_path>/Linux_for_Tegra/`. Diff-aware (skip
   byte-identical); uses `sudo cp -p` for `rootfs/*` destinations.
2. **`/jetson-flash-image`** — flashes the updated `bsp_image` to the
   device. `nvpower.service` runs the new script on the next boot.
3. (Alternate, no flash) Copy `<source.root_path>/Linux_for_Tegra/rootfs/etc/systemd/nvpower.sh`
   directly to the running target's `/etc/systemd/nvpower.sh`, then
   `sudo systemctl restart nvpower.service` (or reboot).

Editing `<source.root_path>/...` without committing — or editing
`<bsp_image.root_path>/...` directly — does nothing for `/jetson-promote-image`
and is silently lost on the next `/jetson-init-image` re-extract.

## Recipe — pin everything to MAXN for stress / performance runs

Combines Operations 1 + 2. Operation 1's BPMP edits all flow
through one round of the protocol (a single decompile / multi-node
edit / recompile / commit cycle — don't round-trip the protocol
twice for the same `.dtb`):

1. **BPMP DTB** (the "Content edit: `max-rate-custom` on a named clock node" step content): leave `max-rate-custom` unset on every CPU / GPU / EMC clock; remove existing `max-rate-custom` lines that lower the ceiling.
2. **BPMP DTB** (the "Content edit: EMC DVFS disable / enable" step content): pin EMC at its init rate — `bwmgr.enabled = 0`, `cactmon.enabled = 0`, plus `/delete-node/ osp-controller` on T26x (skip on T23x).
3. Apply both content edits inside one protocol "Edit the DTS" invocation, then run the remaining protocol steps (recompile, sanity-check, single customization commit covering both content edits).
4. **nvpower.sh** (Operation 2): set `desired_cpufreq_gov="performance"` and `desired_devfreq_gov="performance"` unconditionally; remove the GPU/nvjpg skip in `set_devfreq_governor`. Applies via Operation 2's overlay edit recipe (the "Overlay edit recipe (apply before editing nvpower.sh)" step) — a separate overlay-tracker pristine + customization commit pair on the rootfs script, distinct from the BPMP-DTB protocol's commit.
5. Set the boot-default nvpmodel mode to MAXN via [`/jetson-customize-nvpmodel`](../jetson-customize-nvpmodel/SKILL.md) — the per-clock nvpmodel cap clamps below `max-rate-maxn` regardless of BPMP DTB content.

Deploy [`/jetson-promote-image`](../jetson-promote-image/SKILL.md) → [`/jetson-flash-image`](../jetson-flash-image/SKILL.md) picks up the new BPMP DTB (via the overlay tracker) and the edited `nvpower.sh` (via the same overlay tracker) on the next flash.

## Limitations

- **Image-build-time only.** All edits land under `<source.root_path>/Linux_for_Tegra/` and reach the device only via `/jetson-promote-image` → `/jetson-flash-image`. Live-target tuning is out of scope.
- **`max-rate-custom` only lowers the ceiling.** It must be strictly below `max-rate-maxn`; raising the silicon cap is not supported.
- **Effective ceiling is two-layer.** The runtime ceiling is `min(BPMP cap, active-nvpmodel-mode cap)`. The nvpmodel cap is owned by `/jetson-customize-nvpmodel`; this skill does not edit it.
- **SoC-conditional EMC DVFS gate.** Disabling EMC DVFS requires editing different node sets on T23x (bwmgr + cactmon) vs T26x (bwmgr + cactmon + delete `osp-controller`). Mis-detection produces undefined behavior.
- **T23x GPU cap is multi-node.** The GPU clock is split across `nafll_gpusys` and every `nafll_gpcX`; the cap binds only when applied to all of them.
- **`nvpower.sh` is package-managed.** It ships in `nvidia-l4t-init`; package upgrades clobber in-place edits. Long-lived setups should prefer a systemd drop-in or sibling helper.
- **ODMDATA wins.** When an ODMDATA token covers a property, the token overrides direct BPMP DTS edits at flash time. Direct BPMP DTS edits are the fallback for properties no NVIDIA token reaches.
- **`max-rate-maxn` and `lateinit` are off-limits.** `max-rate-maxn` is the silicon ceiling (read-only). `lateinit` is for boot-time clock init, not ceiling overrides — never touch either.
- **BPMP DTB may be SKU-multiplexed.** On compound / dispatched flash confs (Orin AGX devkit chain), `BPFDTB_FILE` is selected by `board_sku` / `board_FAB` via `update_flash_args_common`. Reading the static `BPFDTB_FILE=` line is wrong when the chain conditionally overrides it; resolve via the dispatch instead.

## Troubleshooting

| Error | Cause | Solution |
|---|---|---|
| `max-rate-custom` set but clock still ramps to `max-rate-maxn` on T23x GPU | Only `nafll_gpusys` was capped; the `nafll_gpcX` partitions still run at `max-rate-maxn` and dominate the effective ceiling. | Apply the same `max-rate-custom` to `nafll_gpusys` **and** every `nafll_gpcX` node enumerated by `grep -nE '^\s*nafll_gpc[0-9]+\s*:' <decompiled.dts>`. |
| EMC DVFS disable appears to apply but EMC still scales on T26x | Only `bwmgr.enabled = <0x00>` was set; `osp-controller` survives and re-issues frequency changes via the QoS path. | Add edits #2 (`cactmon.enabled = <0x00>`) and #3 (`/delete-node/ osp-controller`) inside the same "Edit the DTS" step. Verify `osp-controller` via `dtc -I dtb -O dts <bpmp-dtb> \| grep -c osp-controller` → expect 0. |
| EMC DVFS disable rejected on T23x with "node not found" for `osp-controller` | T23x (Orin) BPMP DTBs do not contain `osp-controller`; edit #3 must be skipped on T23x. | Detect SoC family with the `grep -c osp-controller` step; only apply #3 when the count is ≥1. |
| BPMP refuses to load DTB after edit: `max-rate-custom >= max-rate-maxn` | `max-rate-custom` was set to or above the silicon ceiling. | Lower `max-rate-custom` strictly below `max-rate-maxn`. If `max-rate-maxn` is absent from the node, query the live cap on a running target: `cat /sys/kernel/debug/bpmp/debug/clk/<clock>/max_rate`. |
| `osp-controller` re-appears after `status = "disabled"` | `status = "disabled"` does not remove the node from the device tree; BPMP still walks it. | Replace with `/delete-node/ osp-controller;` — the node must not exist for BPMP to skip the path. |
| Edits to `nvpower.sh` lost after `apt upgrade` | `nvpower.sh` is owned by the `nvidia-l4t-init` deb and gets overwritten on upgrade. | For long-lived test setups, package edits into a systemd drop-in or a sibling helper file referenced by `nvpower.sh`, rather than editing `nvpower.sh` in place. |
| `/jetson-promote-image` is a no-op after editing the BPMP DTB | The edit was applied to `<bsp_image.root_path>/Linux_for_Tegra/`, which is `/jetson-promote-image`'s output — not its input. | Move the edit to `<source.root_path>/Linux_for_Tegra/bootloader/<BPFDTB_FILE>` (the overlay tracker) and commit through the BPMP-DTB protocol. |
| Cap appears to apply on first boot then resets after a power-mode change | The active nvpmodel mode's per-clock cap clamps below `max-rate-custom`. | Inspect both layers; if nvpmodel is binding, raise (or remove) the nvpmodel cap via `/jetson-customize-nvpmodel`. The BPMP cap alone is not the runtime ceiling. |

## References

- [`../../references/bsp-customization-bpmp-dtb.md`](../../references/bsp-customization-bpmp-dtb.md) — canonical BPMP-DTB customization protocol (pristine import, decompile, edit, recompile, sanity-check, commit). Operation 1 of this skill is a content-only consumer; the protocol owns the mechanics.
- [`references/clock-control-model.md`](references/clock-control-model.md) — layer stack, two-ceilings overview, effective-runtime-ceiling formula.
- [`references/bpmp-dtb-clock-edits.md`](references/bpmp-dtb-clock-edits.md) — two-ceilings semantics, DTS edit form, nvpmodel ↔ BPMP clock-node mapping, inspection cookbook.
- [`references/emc-dvfs-disable.md`](references/emc-dvfs-disable.md) — full SoC-conditional EMC DVFS disable procedure with DTS snippets, detection, re-enable.
- [`references/nvpower-sh-edits.md`](references/nvpower-sh-edits.md) — `nvpower.sh` function locations + common-edit recipes + package-upgrade caveat.
- [`/jetson-customize-nvpmodel`](../jetson-customize-nvpmodel/SKILL.md) — sibling skill: nvpmodel power modes. The active mode's per-clock cap clamps below the BPMP DTB cap.
