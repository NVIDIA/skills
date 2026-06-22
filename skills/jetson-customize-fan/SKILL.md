---
name: jetson-customize-fan
description: >-
  Use when you need to add, remove, edit, list, or change the boot
  default of an nvfancontrol fan profile on a Jetson/Tegra (Orin,
  Thor) target. Triggers: edit fan profile, tune fan curve.
version: 0.0.1
license: "Apache-2.0"
metadata:
  data-classification: public
  author: "Jetson Team"
  tags:
    - thermal
    - fan
    - nvfancontrol
    - profile
  domain: thermal
---

# Modify nvfancontrol Fan Profile (BSP-side)

## Purpose

Edit the per-board nvfancontrol configuration so the device boots
with the desired fan curve / control mode / governor / default
profile. BSP-side only — all writes land in the overlay tracker,
the upstream `bsp_image` copy is read-only.

This skill handles BSP-side edits to the per-board nvfancontrol configuration file: adding profiles, removing profiles, editing the temp → PWM / RPM curves, changing the boot default profile / control mode / governor, and listing defined profiles. Applies on Jetson / Tegra platforms (T234 Orin, T264 Thor).

## File format (canonical, per the BSP file header)

```
POLLING_INTERVAL <seconds>

<FAN <index>>
    TMARGIN <ENABLED|DISABLED>
    FAN_GOVERNOR <type> {
        STEP_SIZE <int>
    }
    FAN_CONTROL <close_loop|open_loop> {
        RPM_TOLERANCE <rpm>
    }
    FAN_PROFILE <name> {
        # TEMP HYST PWM RPM
        <T0> <H0> <P0> <R0>
        ...
    }
    FAN_PROFILE <name> { ... }       # one or more profiles
    THERMAL_GROUP <id> {
        GROUP_MAX_TEMP <C>
        # zone-name <coeffs csv> <max-temp>
        <zone> <coeffs> <max-temp>
        ...
    }
    FAN_DEFAULT_CONTROL  <close_loop|open_loop>
    FAN_DEFAULT_PROFILE  <name>
    FAN_DEFAULT_GOVERNOR <type>
    KICKSTART_PWM <0..255>
```

Rules:

- **Profile curve tuples are 4-column**: `TEMP HYST PWM RPM`. Sort points by ascending `TEMP`; the daemon interpolates between them.
- **`PWM` is `0..255`** (8-bit duty cycle); **`RPM`** is the close-loop target speed. A trailing `0 0` row at the high end pins the fan off above `GROUP_MAX_TEMP`.
- **`HYST`** is hysteresis (°C) at that point — the controller waits `HYST` degrees of cooling before stepping the curve down.
- **`FAN_DEFAULT_PROFILE`** must reference an existing `FAN_PROFILE` block in the same `<FAN N>`. nvfancontrol fails to start if the default names a missing profile.
- **`FAN_DEFAULT_CONTROL`** = `close_loop` (drives toward target RPM, requires tach) or `open_loop` (writes PWM directly).
- **`FAN_DEFAULT_GOVERNOR`** = `cont` (continuous interpolation) and other family-specific values; copy verbatim from existing per-board files when introducing one.
- **`THERMAL_GROUP`** maps thermal zones to the controller's input. Coefficients are a 20-element CSV — copy verbatim from existing entries; values vary by chip family and zone.
- **`<FAN N>`** is one block per fan index (typical: `<FAN 1>`). Block boundaries `<...>` and `{...}` are strict; preserve indentation matching neighboring lines.
- **Curves are characterised, not invented.** Add or edit curves from real thermal-acoustic data for the platform; do not interpolate from neighbouring profiles or copy across chip families.

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
Linux_for_Tegra/rootfs/etc/nvpower/nvfancontrol/nvfancontrol_<active-sku>.conf
```

It lives in **two** roots; the skill walks both:

| Role | Location | Skill writes? |
|---|---|---|
| Detection + pristine source | `<bsp_image.root_path>/Linux_for_Tegra/rootfs/etc/nvpower/nvfancontrol/` | no — read-only |
| Overlay edit target + git commit | `<source.root_path>/Linux_for_Tegra/rootfs/etc/nvpower/nvfancontrol/` | yes |

Subsequent sections refer to **the per-board file** to mean the overlay copy
under `<source.root_path>`. Operations 1–4 all read, edit, and save against
that overlay copy. The `<bsp_image.root_path>` copy is read once during the
"Resolving `<active-sku>`" detection step and once during the
[Overlay edit recipe](#overlay-edit-recipe-apply-before-any-operation)'s
pristine-import step, then never touched again.

### Resolving `<active-sku>` — which file to edit

Filename conventions vary per product family:

- `nvfancontrol_<module.id>_<module.sku>.conf` — most Orin parts (e.g. `nvfancontrol_p3767_0000.conf`, `nvfancontrol_p3701_0008.conf`).
- `nvfancontrol_<module.id>_<module.sku>_<carrier.id>_<carrier.sku>.conf` — Thor variants where the carrier disambiguates (e.g. `nvfancontrol_p3834_0008_p4071_0000.conf`).
- `nvfancontrol_<carrier>_<sku>_<rev>.conf` — IGX revision variants (e.g. `nvfancontrol_p3740_0002_b01.conf`).

The nvfancontrol daemon resolves the right file at startup based on the booted hardware's DT compatible plus board IDs (no helper script — it's done inside the binary). To map BSP-side without a running target, against `<bsp_image.root_path>`:

1. List `<bsp_image.root_path>/Linux_for_Tegra/rootfs/etc/nvpower/nvfancontrol/`.
2. Filter to filenames that contain `<module.id>_<module.sku>` from the active profile.
3. If multiple candidates remain, refine with `<carrier.id>_<carrier.sku>` and (when relevant) the carrier revision tag.
4. If still ambiguous, run the per-board flash conf dispatch chain to read the kernel DTB's `compatible` and pick the file whose name aligns with the resolved board / carrier.
5. Verify the chosen file actually exists under `<bsp_image.root_path>/Linux_for_Tegra/rootfs/etc/nvpower/nvfancontrol/`.

Don't blindly compose a filename — the naming convention varies by product family.

### Propagation set — confs to keep in sync

The active-SKU file is rarely the only conf that should carry a customization. After editing it, **apply the same edit to every sibling in the propagation set** so the change survives regardless of which module / baseboard SKU is booted:

- **The reference platform's nvfancontrol conf** — the upstream conf the active file was forked from (resolve via `reference_devkit` in the active profile, or `jetson-derive-carrier` fork ancestry). For a BSP that contains only the reference (no derived carriers), this is the same file as the active and the rule reduces to a no-op.
- **Every carrier-derived nvfancontrol conf** — each `nvfancontrol_*.conf` produced by `jetson-derive-carrier` for a custom carrier on top of the same module SKU.

**"Apply the same edit" ≠ blanket file copy.** Port the changed `FAN_PROFILE` / control lines into each sibling; preserve every other line. Sibling confs often hold carrier-specific deltas (different `THERMAL_GROUP` coefficients for a different thermal solution, different tach `RPM` ceilings) that must stay intact — a blanket overwrite would mis-tune the fan on those carriers. Blanket-copying is safe **only** when both confs were byte-identical before the edit.

## Overlay edit recipe (apply before any Operation)

Follow the canonical
[Off-skill edits recipe](../../context/bsp-customization-workflow.md#off-skill-edits)
in the workflow doc — pristine import + customization commit pair, both
gated by the preview gate. Apply once per run, covering every per-board
file the run touches (the active conf plus every sibling in the
[Propagation set](#propagation-set--confs-to-keep-in-sync)).

Concrete substitutions for this skill:

- `<rel>/<file>` is `rootfs/etc/nvpower/nvfancontrol/<conf>`.
- Suggested pristine-import message:
  `import pristine: <comma-separated rel paths of imported confs>`,
  body `Source: <bsp_image.root_path>/Linux_for_Tegra/ (BSP <bsp_image.version>)`.
- Suggested customization-commit header:
  `jetson-customize-fan: <summary>`,
  body lines like `nvfancontrol_p3767_0000.conf: added FAN_PROFILE static (open-loop, PWM=255 flat), FAN_DEFAULT_CONTROL close_loop -> open_loop, FAN_DEFAULT_PROFILE quiet -> static`.

## Instructions

Pick the operation that matches the user's intent and follow the
matching subsection. All write-side operations (1–4) must first
apply the [Overlay edit recipe](#overlay-edit-recipe-apply-before-any-operation).

- Operation 1 — Add a new fan profile.
- Operation 2 — Remove a fan profile.
- Operation 3 — Edit an existing fan profile (curve, hysteresis).
- Operation 4 — Change the boot default (control / profile / governor).
- Operation 5 — List defined fan profiles (read-only).

After any write-side operation, run the Deploy chain (`## Deploy`)
to land the change on the device.

## Examples

Add an aggressive profile to a P3767-0000 (Orin Nano dev kit) target
and pin it as the boot default:

```
/jetson-customize-fan
> add a profile called "aggressive" with the curve {0:255, 40:150, 80:0}
> set FAN_DEFAULT_PROFILE to aggressive
```

Soften the fan ramp on the existing `quiet` profile (raise HYST,
lower mid-PWM):

```
/jetson-customize-fan
> edit FAN_PROFILE quiet — raise HYST to 5 from 30 °C up, drop PWM at 60 °C to 80
```

List which profiles the active SKU currently defines and which is
the boot default:

```
/jetson-customize-fan
> list defined fan profiles
```

## Operation 1 — Add a new fan profile

Apply the [Overlay edit recipe](#overlay-edit-recipe-apply-before-any-operation) first.

1. In the per-board file, inside the `<FAN N>` block, add a new `FAN_PROFILE <name> { ... }` between existing profiles. Daemon parse order doesn't matter, but grouping with siblings keeps the file scannable.
2. Pick `<name>` lowercase, no whitespace (e.g. `quiet`, `cool`, `aggressive`).
3. Fill the curve table — 4-column `TEMP HYST PWM RPM` tuples, ascending `TEMP`. End with at least one row above `GROUP_MAX_TEMP` so behavior at over-temp is defined.

Example skeleton:

```
FAN_PROFILE aggressive {
    #TEMP HYST PWM RPM
    0    0    255 6000
    20   2    255 6000
    40   2    150 3500
    60   2    50  1500
    80   0    0   0
    105  0    0   0
}
```

Validate (see Rules):

- Every `TEMP` lies within `0..GROUP_MAX_TEMP` of the `THERMAL_GROUP`.
- `PWM` ∈ `[0, 255]`; `RPM` ≤ the platform's tach-reported max (varies by fan part).
- Tuples are sorted ascending by `TEMP`.

If you intend the new profile to be the boot default, follow Operation 4.

## Operation 2 — Remove a fan profile

Apply the [Overlay edit recipe](#overlay-edit-recipe-apply-before-any-operation) first.

1. In the per-board file, delete the entire `FAN_PROFILE <name> { ... }` block including all curve rows and the closing `}`.
2. If the deleted `<name>` matches the trailing `FAN_DEFAULT_PROFILE`, point `FAN_DEFAULT_PROFILE` at a remaining profile — otherwise the daemon fails to start.
3. Search for hard-coded references in the rootfs before declaring the change safe:
   ```bash
   grep -rn "nvfancontrol.*profile\|FAN_PROFILE" \
     Linux_for_Tegra/rootfs/etc 2>/dev/null
   ```

The remaining profiles do not need renaming.

## Operation 3 — Edit an existing fan profile

Apply the [Overlay edit recipe](#overlay-edit-recipe-apply-before-any-operation) first.

1. In the per-board file, modify the curve rows of the target `FAN_PROFILE <name>`. Keep the 4-column `TEMP HYST PWM RPM` shape.
2. Maintain ascending `TEMP` ordering; insert or remove rows as needed.
3. Adjust `HYST` to tune anti-oscillation: too low → fan thrashes near a curve point; too high → fan lags reality.
4. Validate per Operation 1's rules.

Edits to `KICKSTART_PWM`, `RPM_TOLERANCE` (inside `FAN_CONTROL`), or `STEP_SIZE` (inside `FAN_GOVERNOR`) sit outside the profiles but tune the same fan; same per-board file.

## Operation 4 — Change the boot default

Apply the [Overlay edit recipe](#overlay-edit-recipe-apply-before-any-operation) first.

Edit the trailing default lines inside the `<FAN N>` block:

```
FAN_DEFAULT_CONTROL  <close_loop|open_loop>
FAN_DEFAULT_PROFILE  <name>
FAN_DEFAULT_GOVERNOR <type>
```

`FAN_DEFAULT_PROFILE` must reference an existing `FAN_PROFILE` in the same `<FAN N>` block. `FAN_DEFAULT_CONTROL` and `FAN_DEFAULT_GOVERNOR` must reference values the binary supports — copy from existing per-board files when changing.

## Operation 5 — List defined fan profiles

This is a read-only operation; no overlay-tracker setup is needed. Run
against whichever copy you want to inspect (`<bsp_image.root_path>/...`
for the pristine state, `<source.root_path>/...` for the post-edit state):

```bash
grep -E '^[[:space:]]*FAN_PROFILE ' <per-board file>
grep -E '^[[:space:]]*FAN_DEFAULT_'  <per-board file>
```

The first prints every profile name in the file; the second prints the boot defaults (control / profile / governor).

## Limitations

- BSP-side scope only — this skill never touches a running target's
  `/etc/nvpower/nvfancontrol/` directly. Live tuning requires
  reboot via Deploy, or the side-channel `scp + systemctl restart`
  flow described in `## Deploy`.
- Edits land in the overlay copy under `<source.root_path>` only;
  the `<bsp_image.root_path>` copy is read-only and is rewritten by
  `/jetson-promote-image`. Hand-editing `bsp_image` is silently lost
  on the next `/jetson-init-image` re-extract.
- Curve tuples must reflect characterised thermal-acoustic data for
  the platform — this skill does not interpolate or copy curves
  across chip families.
- Propagation across sibling carriers (same module SKU) is partial
  by design: only the changed `FAN_PROFILE` / control lines are
  ported, never a blanket file overwrite, since sibling confs may
  hold carrier-specific `THERMAL_GROUP` / RPM ceilings.
- Curve point limits are family-specific; copy verbatim from existing
  confs when introducing new `THERMAL_GROUP` coefficients or
  `FAN_DEFAULT_GOVERNOR` values.

## Troubleshooting

| Error | Cause | Solution |
|---|---|---|
| nvfancontrol.service fails to start after edit | `FAN_DEFAULT_PROFILE` references a profile that was removed or renamed | Set `FAN_DEFAULT_PROFILE` to an existing `FAN_PROFILE <name>` in the same `<FAN N>`. |
| Fan thrashes near a curve point | `HYST` too low | Raise `HYST` on the affected row (typical 2–5 °C). |
| Fan lags reality | `HYST` too high | Lower `HYST` on the affected row. |
| Fan never spins up at high temp | Curve missing a row above `GROUP_MAX_TEMP`, or trailing `0 0` truncates above ambient | Add a high-temp tuple with non-zero PWM/RPM; pin the over-temp row only above `GROUP_MAX_TEMP`. |
| Daemon parse error referencing column count | Curve row not 4-column `TEMP HYST PWM RPM` | Restore 4-column shape; remove trailing whitespace and stray columns. |
| `RPM` target never reached in close-loop | Target above tach-reported max, or `FAN_CONTROL` set to `open_loop` | Lower `RPM` to within the fan's mechanical max, or switch `FAN_DEFAULT_CONTROL` to `close_loop`. |
| Change vanished after `/jetson-init-image` re-extract | Edit landed in `<bsp_image.root_path>` instead of `<source.root_path>` overlay | Re-apply via the [Overlay edit recipe](#overlay-edit-recipe-apply-before-any-operation) so the change is committed in the overlay tracker. |
| Sibling carrier boots with wrong tach ceiling after propagation | Blanket file copy clobbered carrier-specific `THERMAL_GROUP` / `RPM` rows | Port only the changed lines per [Propagation set](#propagation-set--confs-to-keep-in-sync); restore the carrier's original `THERMAL_GROUP`. |

## Deploy

The customization commit in the overlay tracker does not reach the device
on its own. The Deploy chain:

1. **`/jetson-promote-image`** — copies every tracked file in the overlay
   into `<bsp_image.root_path>/Linux_for_Tegra/`. Diff-aware (skip
   byte-identical); uses `sudo cp -p` for `rootfs/*` destinations.
2. **`/jetson-flash-image`** — flashes the updated `bsp_image` to the
   device.
3. (Alternate, no flash) Copy `<source.root_path>/Linux_for_Tegra/rootfs/etc/nvpower/nvfancontrol/<conf>`
   directly to the running target's `/etc/nvpower/nvfancontrol/<conf>`,
   then `sudo systemctl restart nvfancontrol.service` (or reboot).

Editing `<source.root_path>/...` without committing — or editing
`<bsp_image.root_path>/...` directly — does nothing for `/jetson-promote-image`
and is silently lost on the next `/jetson-init-image` re-extract.
