---
name: jetson-optimize-memory
description: >-
  Reclaim DRAM by disabling unused subsystems across MB1 BCT, MB2 BCT,
  kernel reserved-memory, and SWIOTLB. Use for headless or no-camera
  Jetson deployments; not for CPU/GPU frequency tuning.

version: 0.0.1
license: "Apache-2.0"
argument-hint: "headless | no-camera | swiotlb"
metadata:
  data-classification: public
  author: "Jetson Team"
  tags:
    - memory
    - dram
    - carveout
    - swiotlb
  domain: memory
---

# jetson-optimize-memory

Memory is reserved across four layers ordered by boot chronology
(higher row = earlier in boot, closer to hardware):

| Layer | Content | Key files |
|---|---|---|
| MB1 BCT | firmware carveouts | per-module misc DTS |
| MB2 BCT | firmware loading + AST controls | per-module misc DTS |
| Kernel DTS | reserved-memory and driver binding | per-module DTS |
| SWIOTLB | DMA bounce pool size | `<module>.conf.common` (`CMDLINE_ADD`) |

**Critical rules:**

- Only the scenarios in **Scenario recipes** are validated. Refuse any
  request to disable a carveout/cluster/node not in that table.
- **Zeroing a carveout requires both: disabling the cluster's loading
  controls AND removing the AST that references it.**
- **Emit explicit overrides for every cluster in the recipe**, regardless
  of how the source `#ifdef`/`#else` looks. Verify the merged binary.
- **GPU SW stack must match the chip**: T234 -> nvgpu, T264 and later
  -> OpenRM. Follow the shared derived-platform rule in
  `target-platform-contract.md`; stop on mismatches instead of guessing.
- Do not set SWIOTLB to 0 — some peripherals can't use the IOMMU.

---

## Scenario recipes

| Keyword | MB1 BCT carveouts | MB2 BCT | Kernel DTS |
|---|---|---|---|
| `headless` | DCE-family (see chip table) | DCE `auxp_controls` + DCE AST(s) | `display@<addr>` (and `dce@<addr>` if exposed) → disabled |
| `no-camera` | RCE/VI/ISP-family | RCE `auxp_controls` (each instance) + RCE AST(s) | recommended: VI/ISP/NVCSI → disabled |

### Chip-specific carveouts

| Scenario | T234 (Orin) | T264 (Thor) |
|---|---|---|
| `headless` | `CARVEOUT_BPMP_DCE`, `CARVEOUT_DCE`, `CARVEOUT_DCE_TSEC`, `CARVEOUT_TSEC_DCE`, `CARVEOUT_DISP_EARLY_BOOT_FB` | `CARVEOUT_DCE`, `CARVEOUT_TSEC_DCE`, `CARVEOUT_HPSE_DCE`, `CARVEOUT_DISP_EARLY_BOOT_FB` |
| `no-camera` | `CARVEOUT_RCE`, `CARVEOUT_CAMERA_TASKLIST` | `CARVEOUT_RCE`, `CARVEOUT_RCE1`, `CARVEOUT_RCE_RW`, `CARVEOUT_VI_TASKLIST`, `CARVEOUT_VI1_TASKLIST`, `CARVEOUT_ISP_TASKLIST`, `CARVEOUT_ISP1_TASKLIST` |

> **Post-boot:** `headless` → `sudo systemctl set-default multi-user.target`.

---

## MB1 BCT carveout overrides

File: `Linux_for_Tegra/bootloader/generic/BCT/tegra<chip>-mb1-bct-misc-<module>.dts`
(e.g. `tegra234-mb1-bct-misc-p3767-0000.dts` for Orin Nano,
`tegra264-mb1-bct-misc-p3834-0008-p4071-0000.dts` for Thor).

For each carveout, add inside the existing `carveout` node:

```dts
aux_info@<CARVEOUT_NAME> {
    pref_base = <0x0 0x0>;
    size      = <0x0 0x0>;
    alignment = <0x0 0x0>;
};
```

---

## MB2 BCT cluster + AST overrides

File: `Linux_for_Tegra/bootloader/generic/BCT/tegra<chip>-mb2-bct-misc-<module>.dts`
(includes `tegra<chip>-mb2-bct-common.dtsi`).

For each target cluster:

1. Override `auxp_controls@<index>`:
   ```dts
   auxp_controls@<index> {
       enable_init    = <0>;
       enable_fw_load = <0>;
       enable_unhalt  = <0>;
   };
   ```
2. `/delete-node/ auxp_ast_config@<idx>;`

Look up indices in `common.dtsi`: `auxp_controls@N` carries a comment
naming its cluster; `auxp_ast_config@N` has `ast_region` children whose
`carveout = <CARVEOUT_…>;` lines identify the owner.

---

## Kernel DT reserved-memory

```sh
DTB=Linux_for_Tegra/kernel/dtb/<platform-dtb-name>.dtb
dtc -I dtb -O dts -o /tmp/platform.dts $DTB
# edit: status = "disabled" on target nodes
dtc -I dts -O dtb -o $DTB /tmp/platform.dts
```

**Display** — disable `display@<addr>`, plus `dce@<addr>` if exposed as
a separate kernel node.

**Camera** — under `host1x@<addr>`, disable whichever of `vi*` / `isp*`
/ `nvcsi` exist on the BSP (only emit present nodes):

Locate the display controller node in the decompiled DTS and disable it.
The node's unit address is chip-specific — find it by `compatible` string
(e.g. `nvidia,tegra234-display`) rather than hard-coding the address.

```dts
host1x@<addr> {
    vi0@<addr>   { status = "disabled"; };
    vi1@<addr>   { status = "disabled"; };
    isp@<addr>   { status = "disabled"; };
    isp1@<addr>  { status = "disabled"; };
    nvcsi@<addr> { status = "disabled"; };
};
```

---

## SWIOTLB DMA bounce pool

The NVIDIA IOMMU covers peripheral DMA, so SWIOTLB is rarely used.
Edit `CMDLINE_ADD` (**never** `CMDLINE`) in `Linux_for_Tegra/<module>.conf.common`:

```sh
# Total bytes = swiotlb_value × 2048; 4 MiB pool:
CMDLINE_ADD="... swiotlb=2048"
```

---

## Override verification (mandatory)

After every patched MB1/MB2 BCT `.dts`, reproduce the BSP's compile +
decompile using the same `-D…` flags from `bct_flags.append(...)` in
`bootloader/tegraflash_impl_t<chip>.py`:

```sh
gcc -E -nostdinc -x assembler-with-cpp \
    -DENABLE_<FLAG_1> -DENABLE_<FLAG_2> \
    -I bootloader -I bootloader/generic/BCT \
    -o /tmp/cpp.dts <patched-bct.dts>
dtc -q -I dts -O dtb -o /tmp/cpp.dtb /tmp/cpp.dts
dtc -q -I dtb -O dts /tmp/cpp.dtb | less
```

Confirm in the merged output:
- Each zeroed `aux_info@<NAME>` (or `aux_info@<id>U` post macro expansion)
  has `size = <0x0 0x0>` and `pref_base = <0x0 0x0>`.
- Each disabled `auxp_controls@<idx>` has all three `enable_*` fields `<0>`.
- Each `/delete-node/`'d `auxp_ast_config@<idx>` is absent.

---

## Verification (on booted target)

```sh
sudo cat /proc/iomem | grep -iE 'nv-reserved|cma|fb|carveout'
ls /proc/device-tree/reserved-memory/
dmesg | grep -iE 'firmware|carveout|bpmp|reserved|fail|error' | head -20
free -m
```

| Scenario | Sysfs | dmesg grep |
|---|---|---|
| Display off | `ls /sys/class/drm/` (empty) | `tegra-drm\|nvdisplay\|dce\|host1x\|fb0` |
| Camera off | `ls /dev/video* 2>/dev/null` (none) | `rce\|nvcsi\|tegra-camera\|vi0\|vi1\|isp` |
| SWIOTLB shrink | `cat /sys/kernel/debug/swiotlb/io_tlb_nslabs` matches cmdline | `swiotlb` |

For SWIOTLB: `/proc/cmdline` must contain `swiotlb=<value>`, and
`watch -n5 cat /sys/kernel/debug/swiotlb/io_tlb_used` must stay under
`io_tlb_nslabs` during full workload — if exceeded, restore original
`CMDLINE_ADD` and re-flash `kernel-dtb`.

## Purpose

Cut the unused DRAM carveouts that ship enabled in the reference BSP
when a Jetson deployment skips display, camera, or other peripherals,
freeing the freed bytes for the application. Always edits the four
layers in boot order so an early-stage carveout never outranks a
later-stage shrink.

## Prerequisites

- Active target profile resolved per
  `../../context/target-platform-contract.md`.
- BSP image extracted and source tree initialized
  (`/jetson-init-image`, `/jetson-init-source` complete).
- For headless / no-camera recipes: confirm the workload truly does not
  need display or camera.

## Limitations

- Only the validated recipes (`headless`, `no-camera`, `swiotlb`) are
  exposed; ad-hoc subsystem disables outside the recipe set are
  refused.
- SWIOTLB shrink is bounded by peak in-flight DMA — exceeding the new
  `io_tlb_nslabs` requires reverting the change.
- BPMP-DTB edits land in the overlay tracker only after Customize +
  Build + Deploy run; this skill does not flash on its own.

## Troubleshooting

- **Boot fails after MB1 BCT carveout disable** — restore the pristine
  misc DTS and re-flash; the missing carveout is mandatory for the
  active SoC.
- **`io_tlb_used` exceeds `io_tlb_nslabs`** — revert `swiotlb=` in
  `CMDLINE_ADD` and re-flash the kernel DTB partition.
- **Reclaimed delta smaller than expected** — verify the recipe truly
  matched the deployment (e.g. display still attached); use the
  `dmesg | grep -iE 'firmware|carveout'` check in this file to confirm.
- **Validation `dmesg` shows the disabled subsystem still probing** —
  the change probably did not promote through to `bsp_image`; re-run
  `/jetson-promote-image`.

---
