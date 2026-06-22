---
name: jetson-customize-nvpmodel
description: >-
  Use when you need to add, remove, edit, list, or change the boot
  default of an nvpmodel power mode on a Jetson/Tegra (Orin, Thor)
  target. Triggers: edit power mode, tune frequency caps.
version: 0.0.1
license: "Apache-2.0"
metadata:
  data-classification: public
  author: "Jetson Team"
  tags:
    - power
    - nvpmodel
    - profile
  domain: power
---

# Modify nvpmodel Power Mode (BSP-side)

## Purpose

Edit the per-board nvpmodel configuration so the device boots with
the desired power-mode set, CPU/GPU/EMC/TPC clamps, and default
mode. BSP-side only — all writes land in the overlay tracker, the
upstream `bsp_image` copy is read-only.

This skill handles BSP-side edits to the per-board nvpmodel configuration file: adding modes, removing modes, editing CPU/GPU/EMC/TPC clamps, and changing the boot default. Applies on Jetson / Tegra platforms (T234 Orin, T264 Thor).

## File format (canonical, per the BSP file header)

```
# 1. PARAM definitions — declare named knobs and their sysfs paths
< PARAM TYPE=FILE  NAME=<param_name> >
<arg_name> </absolute/sysfs/path>
...
< PARAM TYPE=CLOCK NAME=<param_name> >
FREQ_TABLE        </sysfs/.../available_frequencies>
MAX_FREQ          </sysfs/.../max_freq>
MIN_FREQ          </sysfs/.../min_freq>
FREQ_TABLE_KNEXT  </sysfs/.../available_frequencies>     # kernel-NEXT variant
MAX_FREQ_KNEXT    </sysfs/.../max_freq>
MIN_FREQ_KNEXT    </sysfs/.../min_freq>

# 2. POWER_MODEL definitions — one block per profile
< POWER_MODEL ID=<int> NAME=<string> >
PARAM_NAME ARG_NAME <value>
...

# 3. PM_CONFIG — mandatory; selects boot default
< PM_CONFIG DEFAULT=<id> >
```

Rules:

- For `TYPE=FILE`, `<value>` is a **string** (`0`, `1`, `on`, `auto`, …).
- For `TYPE=CLOCK`, `<value>` is an **integer** (Hz for clocks, raw integer for masks).
- `-1` for a CLOCK value means **INT_MAX** (no cap).
- The header `< … >` line must start at column 0 with a space after `<` and before `>`. Strict.
- Every `PARAM_NAME` referenced in a POWER_MODEL must already be declared above.
- **Frequency values must come from the kernel's `available_frequencies`** table for that clock — values not in the table get silently rounded.
- `CORE_0` cannot be offlined; at least one CPU core must remain online in every profile.
- **Copy PARAM names verbatim** from an existing POWER_MODEL block in the per-board file — chip families differ (T234 Orin uses `CPU_A78_<n>` for CPU clusters; T264 Thor uses a different convention). Don't invent.

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

`<bsp_image.root_path>` is **read-only** for this skill; every write lands
under `<source.root_path>` (the overlay tracker). This is the workflow
invariant in
[`../../context/bsp-customization-workflow.md#workflow-invariants`](../../context/bsp-customization-workflow.md#workflow-invariants) —
hand-editing upstream silently destroys the diff trail and makes
`/jetson-promote-image` a noop.

## The per-board file

The conf this skill edits has the relative path:

```
Linux_for_Tegra/rootfs/etc/nvpmodel/nvpmodel_<active-sku>.conf
```

It lives in **two** roots; the skill walks both:

| Role | Location | Skill writes? |
|---|---|---|
| Detection + pristine source | `<bsp_image.root_path>/Linux_for_Tegra/rootfs/etc/nvpmodel/` | no — read-only |
| Overlay edit target + git commit | `<source.root_path>/Linux_for_Tegra/rootfs/etc/nvpmodel/` | yes |

Subsequent sections refer to **the per-board file** to mean the overlay copy
under `<source.root_path>`. Operations 1–4 all read, edit, and save against
that overlay copy. The `<bsp_image.root_path>` copy is read once during the
"Resolving `<active-sku>`" detection step and once during the
[Overlay edit recipe](#overlay-edit-recipe-apply-before-any-operation)'s
pristine-import step, then never touched again.

### Resolving `<active-sku>` — which file to edit

The filename is **not** always `module.id + "_" + module.sku`. Variants exist:

- `nvpmodel_p3767_0000_super.conf` (super-mode SKU variant)
- `nvpmodel_igx_orin.conf`, `nvpmodel_igx_orin_safety.conf` (IGX, no SKU number)

At boot, `nvpower.sh` (at `Linux_for_Tegra/rootfs/etc/systemd/nvpower.sh`) reads the kernel DTB's root `compatible` string and maps it — plus super/safety state — to the nvpmodel filename. Replicate that mapping **BSP-side**, against `<bsp_image.root_path>`:

1. Resolve the SKU-correct kernel DTB under `<bsp_image.root_path>/Linux_for_Tegra/` and read its root `compatible` (detect kernel DTB from the active flash conf).
2. Read `<bsp_image.root_path>/Linux_for_Tegra/rootfs/etc/systemd/nvpower.sh` to find the compatible-string → conf mapping, and apply it to the compatible from the previous step (factoring in super/safety flags).
3. Verify the resolved `nvpmodel_<...>.conf` exists under `<bsp_image.root_path>/Linux_for_Tegra/rootfs/etc/nvpmodel/`.

Shortcut: filter `<bsp_image.root_path>/Linux_for_Tegra/rootfs/etc/nvpmodel/` to `nvpmodel_<module.id>_<module.sku>*.conf`; for super-mode flash configs pick the `_super` variant. Use only when unambiguous; otherwise fall back to the DTB method.

Don't blindly compose `nvpmodel_<id>_<sku>.conf` — verify the file actually exists.

### Propagation set — confs to keep in sync

The active-SKU file is rarely the only conf that should carry a customization. After editing it, **apply the same edit to every sibling in the propagation set** so the change survives regardless of which module / baseboard SKU is booted:

- **The reference platform's nvpmodel conf** — the upstream conf the active file was forked from (resolve via `reference_devkit` in the active profile, or `jetson-derive-carrier` fork ancestry). For a BSP that contains only the reference (no derived carriers), this is the same file as the active and the rule reduces to a no-op.
- **Every carrier-derived nvpmodel conf** — each `nvpmodel_*.conf` produced by `jetson-derive-carrier` for a custom carrier on top of the same module SKU.
- **`_super` siblings** when present (e.g. `nvpmodel_p3767_0000_super.conf`) — apply the structural edit (new POWER_MODEL block, removed block, `PM_CONFIG DEFAULT=` flip, NAME rename) **but preserve the super conf's higher MAX_FREQ / MIN_FREQ caps**. Those elevated caps are the whole reason the `_super` variant exists; a blanket content overwrite from the non-super file would silently flatten the super envelope.

**"Apply the same edit" ≠ blanket file copy.** Port the changed POWER_MODEL / PARAM lines into each sibling; preserve every other line. Blanket-copying is safe **only** when both confs were byte-identical before the edit. Otherwise re-validate per the Rules on each target (available-frequencies table for clock values, ID uniqueness, `PM_CONFIG DEFAULT=` references a present ID) — sibling confs may carry SKU-specific `available_frequencies` tables that don't accept the active file's frequency values.

## Overlay edit recipe (apply before any Operation)

Follow the canonical
[Off-skill edits recipe](../../context/bsp-customization-workflow.md#off-skill-edits)
in the workflow doc — pristine import + customization commit pair, both
gated by the preview gate. Apply once per run, covering every per-board
file the run touches (the active conf plus every sibling in the
[Propagation set](#propagation-set--confs-to-keep-in-sync)).

Concrete substitutions for this skill:

- `<rel>/<file>` is `rootfs/etc/nvpmodel/<conf>`.
- Suggested pristine-import message:
  `import pristine: <comma-separated rel paths of imported confs>`,
  body `Source: <bsp_image.root_path>/Linux_for_Tegra/ (BSP <bsp_image.version>)`.
- Suggested customization-commit header:
  `jetson-customize-nvpmodel: <summary>`,
  body lines like `nvpmodel_p3767_0001.conf: PM_CONFIG DEFAULT 2 -> 0 (MAXN)`.

## Instructions

Pick the operation that matches the user's intent and follow the
matching subsection. All write-side operations (1–4) must first
apply the [Overlay edit recipe](#overlay-edit-recipe-apply-before-any-operation).

- Operation 1 — Add a new power mode (drives the NVIDIA Power Estimator).
- Operation 2 — Remove a power mode.
- Operation 3 — Edit an existing power mode (clamps, core-online, NAME).
- Operation 4 — Change the boot default (`PM_CONFIG DEFAULT=`).
- Operation 5 — List defined power modes (read-only).

After any write-side operation, run the Deploy chain (`## Deploy`)
to land the change on the device.

## Examples

Add a new 30 W power mode from a Power Estimator export and pin it
as the boot default on a P3767-0001 target:

```
/jetson-customize-nvpmodel
> add a new POWER_MODEL from ~/Downloads/nvpmodel_30W.conf as ID 8 NAME "30W_CUSTOM"
> set PM_CONFIG DEFAULT to 8
```

Cap a Thor target to the MAXN power envelope by flipping the boot
default (no envelope tuning needed):

```
/jetson-customize-nvpmodel
> set PM_CONFIG DEFAULT to the MAXN profile's ID
```

List which power modes the active SKU currently defines and which
is the boot default:

```
/jetson-customize-nvpmodel
> list defined power modes
```

## Operation 1 — Add a new power mode

Apply the [Overlay edit recipe](#overlay-edit-recipe-apply-before-any-operation) first.

### Generate the profile via the Power Estimator

**Must not interpolate or extrapolate frequencies from existing profiles to estimate power.**

For any non-stock power envelope (custom budget, custom workload), use the **NVIDIA Power Estimator**: `https://jetson-tools.nvidia.com/powerestimator/`

- Pick the exact module SKU and JetPack release.
- Enter workload (CPU cores, GPU usage, EMC, codecs, camera, display).
- Estimate power budget.
- Download custom `nvpmodel.conf`.
- Share the downloaded `nvpmodel.conf` file path.

### Append the POWER_MODEL block

In the per-board file, insert the new block **after** the last `< PARAM … >` declaration (POWER_MODEL must reference declared PARAMs) and **before** the final `< PM_CONFIG ... >` line. The block structure is the **File format** shown above; frequency values come from the Power Estimator output (see "Generate the profile via the Power Estimator").

Check `ID` is unique; `NAME` should be uppercase, no whitespace. Each numeric `MAX_FREQ` / `MIN_FREQ` value must satisfy the `available_frequencies` rule (see Rules). If you intend this mode to be the boot default, follow Operation 4.

## Operation 2 — Remove a power mode

Apply the [Overlay edit recipe](#overlay-edit-recipe-apply-before-any-operation) first.

1. In the per-board file, delete the entire `< POWER_MODEL ID=<n> NAME=... >` block including all its parameter lines, up to (but not including) the next `< POWER_MODEL ... >` or `< PM_CONFIG ... >` marker.
2. If the deleted ID matches the current `DEFAULT=<id>` value in the trailing `< PM_CONFIG … >` line, point `DEFAULT=` at a remaining ID — otherwise nvpmodel will fail to apply a default at boot.
3. Search for hard-coded references in the rootfs scripts before declaring the change safe:
   ```bash
   grep -rn "nvpmodel -m" \
     Linux_for_Tegra/rootfs/etc \
     Linux_for_Tegra/rootfs/opt 2>/dev/null
   ```

ID gaps are legal — you don't have to renumber remaining modes.

## Operation 3 — Edit an existing power mode

Apply the [Overlay edit recipe](#overlay-edit-recipe-apply-before-any-operation) first.

1. In the per-board file, edit the parameter lines inside the block. Keep `<param>` and `<arg>` names exactly matching the PARAM declarations.
2. **If the edit changes the power envelope** (any CPU/GPU/EMC/PVA/DLA `MAX_FREQ` / `MIN_FREQ`, core-online count for a freq-bound mode, or TPC mask), re-run the **NVIDIA Power Estimator** (see "Generate the profile via the Power Estimator") to ground the new frequencies in a real per-component model. Do not interpolate from neighboring modes.
3. Validate clock values per the `available_frequencies` rule (see Rules).

Edits that only toggle core-online flags within an already-validated envelope, or only change `NAME=`, don't need the Power Estimator pass.

## Operation 4 — Change the boot default

Apply the [Overlay edit recipe](#overlay-edit-recipe-apply-before-any-operation) first.

Edit the `< PM_CONFIG DEFAULT=<id> >` line at the bottom of the per-board file.

`<id>` must reference an existing `< POWER_MODEL ID=<id> … >` block in the same file. Pointing at an undefined ID makes nvpmodel fail to apply at boot.

## Operation 5 — List defined power modes

This is a read-only operation; no overlay-tracker setup is needed. Run
against whichever copy you want to inspect (`<bsp_image.root_path>/...`
for the pristine state, `<source.root_path>/...` for the post-edit state):

```bash
grep -E '^< POWER_MODEL ' <per-board file>
grep -E '^< PM_CONFIG '   <per-board file>
```

The first command prints every `ID`/`NAME`; the second prints the current boot default. On a running target, `nvpmodel -p --verbose` (or `nvpmodel -q`) is authoritative.

## Limitations

- BSP-side scope only — this skill never invokes `nvpmodel -m` on a
  running target. Live mode switching requires reboot via Deploy,
  or the side-channel `scp + nvpmodel -m` flow described in `## Deploy`.
- Edits land in the overlay copy under `<source.root_path>` only;
  the `<bsp_image.root_path>` copy is read-only and is rewritten by
  `/jetson-promote-image`. Hand-editing `bsp_image` is silently lost
  on the next `/jetson-init-image` re-extract.
- Frequency values must come from the kernel's `available_frequencies`
  table for the relevant clock — values outside the table get
  silently rounded. This skill does not validate against a live
  target's table; trust the per-board file's existing values and
  Power Estimator output.
- Non-trivial envelope edits (any CPU/GPU/EMC/PVA/DLA `MAX_FREQ` /
  `MIN_FREQ` change) require the NVIDIA Power Estimator
  (`https://jetson-tools.nvidia.com/powerestimator/`) — interpolation
  or extrapolation between existing modes is not supported.
- Propagation across siblings is partial by design: only the changed
  `POWER_MODEL` / `PARAM` lines are ported, never a blanket file
  overwrite, since `_super` siblings carry elevated `MAX_FREQ`/`MIN_FREQ`
  caps that must stay intact.
- PARAM names are chip-family-specific (e.g. `CPU_A78_<n>` on T234,
  different on T264). Copy verbatim from existing POWER_MODEL blocks;
  invented names silently fail to apply.

## Troubleshooting

| Error | Cause | Solution |
|---|---|---|
| nvpmodel fails to apply default at boot | `PM_CONFIG DEFAULT=<id>` references a deleted or missing POWER_MODEL ID | Point `DEFAULT=` at an existing `< POWER_MODEL ID=<n> ... >` in the same file. |
| Frequency value silently doesn't take effect | Value not in the kernel's `available_frequencies` table for that clock | Replace with the nearest legal value from the running target's `available_frequencies` (or Power Estimator output). |
| `MAX_FREQ -1` does nothing | `-1` on a `TYPE=FILE` PARAM (only valid for `TYPE=CLOCK`) | Use `-1` only on CLOCK params; for FILE params, write the actual string the sysfs node expects. |
| Module dies / fails to boot after edit | `CORE_0` was offlined, or the mode dropped below the platform's minimum quiescent envelope | Keep `CORE_0` online; floor `MIN_FREQ` per the Power Estimator's lowest-power profile for the SKU. |
| `nvpmodel -m <id>` returns "ID not found" at runtime | Mode ID was removed or renumbered but a rootfs script still references the old ID | `grep -rn 'nvpmodel -m' rootfs/etc rootfs/opt` and update / remove the stale references. |
| Change vanished after `/jetson-init-image` re-extract | Edit landed in `<bsp_image.root_path>` instead of `<source.root_path>` overlay | Re-apply via the [Overlay edit recipe](#overlay-edit-recipe-apply-before-any-operation) so the change is committed in the overlay tracker. |
| `_super` sibling lost elevated caps after propagation | Blanket file copy clobbered `_super`'s higher `MAX_FREQ`/`MIN_FREQ` | Port only the structural change; preserve `_super`'s caps per [Propagation set](#propagation-set--confs-to-keep-in-sync). |
| Parser rejects new POWER_MODEL block | Header `< … >` not at column 0, missing space after `<` or before `>`, or PARAM not declared earlier | Restore strict header formatting; ensure every referenced `PARAM_NAME` appears in a `< PARAM ... >` block above. |

## Deploy

The customization commit in the overlay tracker does not reach the device
on its own. The Deploy chain:

1. **`/jetson-promote-image`** — copies every tracked file in the overlay
   into `<bsp_image.root_path>/Linux_for_Tegra/`. Diff-aware (skip
   byte-identical); uses `sudo cp -p` for `rootfs/*` destinations.
2. **`/jetson-flash-image`** — flashes the updated `bsp_image` to the
   device.
3. (Alternate, no flash) Copy `<source.root_path>/Linux_for_Tegra/rootfs/etc/nvpmodel/<conf>`
   directly to the running target's `/etc/nvpmodel/<conf>`, then
   `sudo nvpmodel -m <id>` (or reboot to pick up the new `DEFAULT=`).

Editing `<source.root_path>/...` without committing — or editing
`<bsp_image.root_path>/...` directly — does nothing for `/jetson-promote-image`
and is silently lost on the next `/jetson-init-image` re-extract.
