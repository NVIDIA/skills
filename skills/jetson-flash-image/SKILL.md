---
name: jetson-flash-image
description: Use to flash a promoted BSP image to a Jetson DUT in RCM mode via flash.sh or l4t_initrd_flash.sh. Do NOT use for BSP customization, image promotion, or carrier derivation.
version: 0.0.1
license: "Apache-2.0"
metadata:
  data-classification: public
  author: "Jetson Team"
  tags:
    - bsp
    - flash
  domain: meta
---

# Flash BSP Image

## Purpose

Push a promoted `bsp_image` to a Jetson DUT by running the NVIDIA flashing toolchain (`flash.sh` or `l4t_initrd_flash.sh`) from the in-tree `Linux_for_Tegra/`. The DUT must be in RCM (recovery) mode at flash time. This is the **flash leg** of the BSP overlay deploy chain (`/jetson-promote-image` → `/jetson-flash-image`); see the BSP overlay workflow at `../../context/bsp-customization-workflow.md`.

**Design principle.** Four invariants govern this skill; each is fully explained in the Instructions step that owns it.

- Host-side flash variables (`<board>`, `<boot-dev>`, flow tool, per-board `.conf`, `boardctl`) are resolved from the active profile or the in-tree BSP.
- Artifact paths (`DTB_FILE`, `BPFDTB_FILE`, partition XML, BCTs, DRAM training) are resolved inside `flash.sh` / `l4t_initrd_flash.sh` at flash time from `board_sku` / `board_FAB` read off the DUT's EEPROM.
- The DUT's EEPROM is authoritative; the profile is an authoring-time prediction reconciled by the preflight cross-check. Empty EEPROM values are valid, not refusal triggers.
- The user explicitly confirms a printed *resolution* before flashing.

Out of scope: BSP customization (use the `/jetson-customize-*` skills), promoting the overlay tracker into `bsp_image` (use `/jetson-promote-image`), and producing a custom carrier's flash conf (use `/jetson-derive-carrier`).

## Prerequisites

- Active target-platform profile resolved per `../../context/target-platform-contract.md` with a populated `bsp_image:` block. Refuse and route to `/jetson-init-image` if missing.
- `<bsp_image.root_path>/Linux_for_Tegra/` exists on the host and has been through `apply_binaries.sh` (route to `/jetson-init-image` otherwise).
- A per-board flash `.conf` resolvable via the active-block precedence rule (`custom_carrier.flash_config` → `reference_devkit.flash_config`). Refuse and route to `/jetson-derive-carrier` or `/jetson-init-image` if absent.
- For the prior overlay → image leg: `/jetson-promote-image` has already run (or a standalone re-flash with no promotion changes since the last flash).
- A DUT cabled to the host that can be driven into RCM mode either via the in-tree `boardctl` or by manual recovery + reset buttons.

## When to invoke

- After `/jetson-promote-image` has updated `bsp_image` to carry the desired customizations.
- Standalone re-flash with no promotion changes since the last flash.
- DUT must be in RCM mode before invocation.

## Instructions

### Resolve target and `bsp_image`

Resolve the active profile per
[`../../context/target-platform-contract.md`](../../context/target-platform-contract.md).
Refuse if `bsp_image:` is missing or `<bsp_image.root_path>/Linux_for_Tegra/`
does not exist (route the user to `/jetson-init-image`).

### Resolve the flash conf path

Pick the per-board `.conf` using the active-block precedence rule
from
[`target-platform-contract.md`](../../context/target-platform-contract.md):
`custom_carrier.flash_config` when present, else
`reference_devkit.flash_config`. Verify the chosen file exists under
`<bsp_image.root_path>/Linux_for_Tegra/`. Refuse if absent — route
the user at `/jetson-derive-carrier` (custom carrier) or
`/jetson-promote-image` / `/jetson-init-image` (reference devkit).

Bind `<board>` as the basename of the resolved `.conf` minus the
`.conf` suffix (e.g. `jetson-agx-thor-devkit.conf` →
`<board>=jetson-agx-thor-devkit`). the "Invoke the flash" step's command shape consumes
this binding directly.

**No dispatch preview at this stage.** Artifact resolution (`DTB_FILE`,
`BPFDTB_FILE`, partition XML, MB1 BCTs, DRAM training tables) happens
inside `flash.sh` / `l4t_initrd_flash.sh` during image generation,
using `board_sku` and `board_FAB` **read from the DUT's EEPROM in
recovery mode** — not values typed in from the profile. The profile
is an authoring-time *prediction*; the DUT EEPROM is *authoritative*
at flash time. The "Preflight checks" step's EEPROM cross-check reads EEPROM and refuses the flash on profile
/ EEPROM mismatch, so passing preflight is what guarantees the
flash-time dispatch will pick the artifacts the profile expects.

For static analyses outside the flash flow that need predicted
artifact paths (KB generation, `customize-*` skills locating files
to edit), see the standalone snippet in
[per-board conf dispatch](../../context/bsp-customization-software-layers.md#per-board-conf-dispatch--update_flash_args_common)
— that snippet is valid for *BSP-side* resolution but **not** as a
flash-time preview.

### Select `<boot-dev>` and flash flow

`<boot-dev>` is resolved deliberately, never assumed. Source order:

1. If the active profile records the boot device, use it.
2. Otherwise, prompt the user with the choices the per-board conf
   actually supports (`internal`, `external`, `nvme0n1p1`, `mmcblk0p1`,
   etc., depending on chip family and conf variant).

Pick the flow tool from the matrix below:

| Chip family | Default boot media | Tool |
|---|---|---|
| T234 / Orin | eMMC / SD | `flash.sh` |
| T234 / Orin | NVMe / USB | `l4t_initrd_flash.sh` |
| T264 / Thor | NVMe / UFS | `l4t_initrd_flash.sh` |

Massflash and `--flash-only` re-runs use the same tool selection but
add their respective flags.

### Put the DUT into recovery mode

Drive the device into recovery via the in-tree `boardctl` (preferred)
or by manually pressing the recovery button followed by the reset button.

The full procedure — where to find `boardctl` under `Linux_for_Tegra/`,
how to enumerate `-t` targets and pick one (recommend `topo`), the
exact `recovery` verb to use, and the manual fallback — lives in
[`references/recovery-mode-boardctl.md`](references/recovery-mode-boardctl.md).

Bind the resolved binary path as `<boardctl>` for the remainder of
this skill. Never substitute a `$PATH` `boardctl`, never invent a
target name not present in `<boardctl> -h`. The "Preflight checks" step's DUT-recovery verification covers — for
both paths — that the DUT actually landed in RCM mode.

### Preflight checks

Performing the following preflight checks in the specified order,
and never skip each check. Host-side checks run first (fail early
before asking the user to flip recovery), then DUT-side checks once
the user has put the device in RCM mode. EEPROM-dependent checks come
after RCM detection, since the recovery channel is what makes the read
possible.

**5.1 `bsp_image` readiness (host-side).**

- Per-board conf exists at `<bsp_image.root_path>/Linux_for_Tegra/<flash_config>`.
- Artifact subtrees populated: `kernel/dtb/`, `bootloader/`,
  `bootloader/generic/BCT/`, `rootfs/`.
- `apply_binaries.sh` has been run — check for
  `rootfs/etc/nv_tegra_release` and the chip's `nvidia-l4t-bsp-*`
  marker package files.
- The *exact* resolved `DTB_FILE` / `BPFDTB_FILE` / BCT filenames
  are intentionally not checked here — those are picked from
  EEPROM at flash time (see the "Resolve the flash conf path" step).
- **Kernel `Image` mirror invariant.** When both
  `<LFT_DST>/kernel/Image` and `<LFT_DST>/rootfs/boot/Image` exist,
  they must be byte-identical (`cmp -s`); drift means
  `/jetson-promote-image`'s [Mirror step](../jetson-promote-image/SKILL.md#mirror-kernel-image-into-rootfs-when-kernel-changed)
  was skipped — route back. **Initramfs presence.**
  `l4t_initrd_flash.sh` requires `<LFT_DST>/bootloader/l4t_initrd.img`
  and `<LFT_DST>/rootfs/boot/initrd`; absent → route to
  `/jetson-init-image`. (Freshness vs.
  `rootfs/lib/modules/<ver>/` is not checked here — owned by
  `/jetson-promote-image`'s
  [Refresh initramfs step](../jetson-promote-image/SKILL.md#refresh-initramfs-when-kernel-or-modules-changed).)

**5.2 DUT in RCM mode.**

- Verifies the outcome of the "Put the DUT into recovery mode" step (either the user-selected
  `<boardctl> -t <target>` invocation or the manual fallback).
- `lsusb -d 0955:` must report at least one device matching the
  active chip family's recovery VID:PID pair:

  | Chip family | Recovery VID:PID |
  |---|---|
  | T23x / Orin | `0955:7X23` (X is any hex digit — module variant) |
  | T26x / Thor | `0955:7026` |

  For T23x, match on the trailing `23` (e.g. `lsusb -d 0955: | grep -E ' 0955:7.23 '`); a literal `0955:7023` check will miss valid T23x variants.

- Absent → if the "Put the DUT into recovery mode" step used `boardctl`, surface its output and
  refuse; if the "Put the DUT into recovery mode" step used the manual path, re-prompt the user to
  confirm the jumper / button and power-cycle.
- **This is the gate for everything downstream.** `flash.sh`'s
  image-generation phase reads EEPROM over the same recovery
  channel; a device that's not in RCM here will fail there too.

**5.3 EEPROM cross-check vs. active profile.**

Read `board_sku` and `board_FAB` (and any additional dispatch inputs
the active chip family uses) from the DUT's EEPROM in recovery mode
using `sudo ./nvautoflash.sh --print_boardid` from
`<bsp_image.root_path>/Linux_for_Tegra/`. The full reference —
sample output, label-to-dispatch-input mapping, empty-value
semantics, and the EEPROM-vs-profile reconciliation table — lives in
[`references/eeprom-cross-check.md`](references/eeprom-cross-check.md).

Refusal trigger is a *real non-empty disagreement*, never a missing
value. Empty EEPROM values are valid and not a refusal trigger. The
cross-check is the primary defense against the wrong-target /
wrong-SKU class of failures whenever both sides supply enough
information to disagree.

**5.4 Default user staging (host-side, interactive).**

A freshly applied `bsp_image` has no Linux user pre-staged. Detect via
`<bsp_image.root_path>/Linux_for_Tegra/rootfs/home/` and
`rootfs/etc/passwd` UID ≥ 1000. If none, issue one `AskUserQuestion`
with four click-to-select options: `ubuntu / ubuntu`, `nvidia / nvidia`,
`custom` (sub-prompt for username + password), or `skip` (OEM wizard
on first boot). Non-`skip` picks run
`l4t_create_default_user.sh --autologin --accept-license`. Full
invocation + rationale in
[`references/default-user-staging.md`](references/default-user-staging.md).

Record the resolution so the "Confirm resolution" step can display it. This step is **not**
a refusal gate — it is a user-interaction point.

### Confirm resolution

Print the resolved plan and require explicit acceptance. The format
is the resolution, not the shell command — the user is approving
*what* will flash, not the *string* that will be executed:

```
Target:         <reference_devkit.name> [+ custom_carrier.name]
Profile:        target-platform/<active>.yaml
bsp_image:      <bsp_image.root_path>/Linux_for_Tegra  (version <X>)
Flash conf:     <flash_config>   (path verified in the "Resolve the flash conf path" step)

DUT EEPROM  →   board_sku=<value-or-(empty)>  board_FAB=<value-or-(empty)>
Profile     →   module.sku=<value-or-(any)>    module.revision=<value-or-(any)>
                (reconciled per the "Preflight checks" step's EEPROM cross-check)

Boot device:    <boot-dev>
Flow tool:      flash.sh  |  l4t_initrd_flash.sh
boardctl:       <bsp_image.root_path>/Linux_for_Tegra/tools/board_automation/boardctl
                (or the path the "Put the DUT into recovery mode" step resolved)
RCM entry:      <boardctl> -t <user-selected target> recovery  |  manual recovery + reset buttons
Post-flash:     <boardctl> -t <user-selected target> reset   (T26x / Thor)
                not required — flash tool resets internally    (T23x / Orin)
Default user:   <username> (autologin)
              | already staged in rootfs (kept)
              | none — OEM config wizard on first boot
```

Artifact paths (`DTB_FILE`, `BPFDTB_FILE`, partition XML, BCTs) are
not shown — they are resolved by `flash.sh` from EEPROM at flash
time, not by this skill. The "Preflight checks" step's EEPROM cross-check is what
guarantees those flash-time picks will line up with what the profile
expects.

This is the user-acceptance gate. Refuse to fall back to a raw
"paste this command" workflow — that path is how stale doc snippets,
wrong dashes, and prompt-character paste artifacts make it into
production flashes.

### Invoke the flash

Construct the command from the resolved variables — never accept a
verbatim command from the user, the docs, or memory:

```bash
cd <bsp_image.root_path>/Linux_for_Tegra
sudo ./<flow-tool> [<resolved flags>] <board> <boot-dev>
```

Abort and surface the failed step on the first non-zero exit. Do
not auto-retry on transient USB errors.

### Post-flash reset (T26x / Thor only)

T26x platforms do **not** auto-reboot from the freshly flashed image
when `flash.sh` / `l4t_initrd_flash.sh` returns. Run, using the
`<boardctl>` resolved in the "Put the DUT into recovery mode" step and the same target the user
selected for RCM entry:

```bash
<boardctl> -t <user-selected target> reset
```

T23x / Orin issues the reset internally; skip this step on Orin. If
the "Put the DUT into recovery mode" step used the manual path, prompt the user to remove the
force-recovery jumper / button and power-cycle by hand. Gate this
step on chip family resolved from the active profile.

### Summary

Report: command line(s) used (flash + post-flash reset if it ran),
exit code, log location (if teed). Persist the resolved plan and
outcome where validation can re-read it.

## Limitations

- **DUT must be in RCM mode.** Image-generation (EEPROM read) and the flash itself both use the recovery USB channel; not-in-RCM at the gate fails image-gen too.
- **Artifact paths resolved at flash time.** `DTB_FILE`, `BPFDTB_FILE`, partition XML, BCTs, DRAM training are picked from EEPROM (`board_sku` / `board_FAB`) by `flash.sh` / `l4t_initrd_flash.sh`; this skill validates only host-side scaffolding.
- **EEPROM is authoritative over the profile.** Real non-empty disagreement refuses; *empty* EEPROM values are valid.
- **No raw-command bypass.** Refuses to fall back to user-supplied "paste this command" — that path is how stale doc snippets and prompt-character paste artifacts reach production flashes.
- **No transient-error auto-retry.** USB hiccups abort; re-enter RCM and re-invoke.
- **T26x needs an explicit post-flash reset.** T26x doesn't auto-reboot from a freshly flashed image; `<boardctl> -t <target> reset` (or manual power-cycle) required. T23x resets internally.
- **Massflash / `--flash-only`** uses the same tool selection + respective flags; massflash topology setup is out of scope here.

## Troubleshooting

| Error | Cause | Solution |
|---|---|---|
| `lsusb -d 0955:` reports nothing matching `0955:7X23` (T23x) or `0955:7026` (T26x) | DUT did not enter RCM mode — `boardctl recovery` failed or the manual recovery + reset sequence didn't take. | If `boardctl` was used, surface its output and re-prompt; manual path, re-confirm the recovery jumper / button and power-cycle. Re-run the "DUT in RCM mode" preflight. |
| Preflight EEPROM cross-check refuses with `board_sku` / `board_FAB` mismatch vs. profile | EEPROM holds a *real non-empty* value that disagrees with `module.sku` / `module.revision` in the active profile. | Fix the profile (`/jetson-set-target` or `/jetson-init-target`) to match the DUT's actual EEPROM. Do **not** override EEPROM from the profile — EEPROM is authoritative. |
| Preflight refuses with "per-board conf not found" | Active profile points at a `flash_config` that does not exist under `<bsp_image.root_path>/Linux_for_Tegra/`. | For custom carriers: run `/jetson-derive-carrier` to produce the conf. For reference devkits: re-run `/jetson-promote-image` / `/jetson-init-image` to repopulate the BSP image. |
| Preflight refuses because `rootfs/etc/nv_tegra_release` is missing | `apply_binaries.sh` has not been run against `<bsp_image.root_path>/Linux_for_Tegra/`. | Re-run `/jetson-init-image` (which invokes `apply_binaries.sh`) or run `apply_binaries.sh` manually from `Linux_for_Tegra/`. |
| `boardctl -t <target> recovery` errors out or has no effect | Wrong `boardctl` (a `$PATH` binary instead of the in-tree one) or a target name not enumerated by `<boardctl> -h`. | Use the in-tree `<bsp_image.root_path>/Linux_for_Tegra/tools/board_automation/boardctl`; pick the target from `<boardctl> -h`. Fall back to manual recovery + reset buttons if needed. |
| Flash succeeds on T26x but the DUT stays in recovery / does not boot the new image | T26x does not auto-reset out of recovery after `flash.sh` / `l4t_initrd_flash.sh` returns. | Run `<boardctl> -t <target> reset` (same target used for RCM entry), or for the manual path remove the force-recovery jumper / button and power-cycle. T23x / Orin needs no extra step. |
| First boot lands on Ubuntu's OEM configuration wizard despite expecting autologin | No default user was staged into `rootfs` before flashing. | Either re-flash after staging via `l4t_create_default_user.sh --autologin --accept-license` (see the default-user-staging step), or complete the OEM wizard on the DUT this once. |
| `flash.sh` exits non-zero on an artifact-not-found error (DTB / BPFDTB / partition XML) | EEPROM-driven dispatch resolved a filename that doesn't exist in `bsp_image` — usually `/jetson-promote-image` didn't copy the customized artifact, or the EEPROM SKU is unsupported by the BSP. | Re-run `/jetson-promote-image`. If the SKU is unsupported, the DUT needs a different BSP version. |
| `kernel/Image` ↔ `rootfs/boot/Image` drift, missing `bootloader/l4t_initrd.img` / `rootfs/boot/initrd`, or `modprobe` "disagrees about version of symbol …" on the DUT | `/jetson-promote-image`'s mirror / refresh steps were skipped, or `bsp_image` was hand-edited outside Deploy. | Re-run `/jetson-promote-image` and re-flash. If initramfs files are entirely absent, run `/jetson-init-image` first. See [promote-image's kernel-image-and-initramfs reference](../jetson-promote-image/references/kernel-image-and-initramfs.md) for the manual escape hatch. |

## References

- [`references/recovery-mode-boardctl.md`](references/recovery-mode-boardctl.md) — locate the in-tree `boardctl`, enumerate targets, invoke `recovery` / `reset`, manual fallback (the "Put the DUT into recovery mode" step / the "Post-flash reset" step details).
- [`references/eeprom-cross-check.md`](references/eeprom-cross-check.md) — `nvautoflash.sh --print_boardid` sample, label-to-dispatch-input map, EEPROM-vs-profile reconciliation table (used by the "Preflight checks" step).
- [`references/default-user-staging.md`](references/default-user-staging.md) — `l4t_create_default_user.sh` invocation + flag rationale (used by the "Preflight checks" step).
- [`../../context/target-platform-contract.md`](../../context/target-platform-contract.md) — target-platform contract.
- [`../../context/bsp-customization-workflow.md`](../../context/bsp-customization-workflow.md#workflow-invariants) — Workspace edit protocol (this skill is the flash leg of Deploy).
- [Per-board conf dispatch](../../context/bsp-customization-software-layers.md#per-board-conf-dispatch--update_flash_args_common) — `<board>`, DTB, and BCT resolution from the active profile.
- [`../jetson-promote-image/SKILL.md`](../jetson-promote-image/SKILL.md) — prior leg; copies overlay → bsp_image.
- [`../jetson-derive-carrier/SKILL.md`](../jetson-derive-carrier/SKILL.md) — produces the custom carrier's flash conf consumed by the "Resolve the flash conf path" step.
