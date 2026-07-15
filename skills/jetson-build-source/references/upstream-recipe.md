# Upstream NVIDIA Build Recipe (reference)

Reference material for `../SKILL.md` — the verbatim NVIDIA build
recipe this skill drives + the three deliberate divergences.

## NVIDIA's Jetson Linux Developer Guide

→ **Kernel Customization** in the Developer Guide for the active
BSP release is the authoritative source. The current online
archive lives under
<https://docs.nvidia.com/jetson/archives/> — pick the release
that matches the workspace's `bsp_image.version` and follow
*Software Development → Kernel → Kernel Customization*. Verbatim
official steps (paths adjusted to use `<install-path>` as NVIDIA
writes it):

```bash
# Prerequisites
sudo apt install git build-essential bc flex bison libssl-dev zstd

# Build kernel
cd <install-path>/Linux_for_Tegra/source
export CROSS_COMPILE=<toolchain-path>/aarch64-none-linux-gnu/bin/aarch64-none-linux-gnu-
make -C kernel

# Install kernel
export INSTALL_MOD_PATH=<install-path>/Linux_for_Tegra/rootfs/
sudo -E make install -C kernel
cp kernel/kernel-noble/arch/arm64/boot/Image \
  <install-path>/Linux_for_Tegra/kernel/Image

# Build + install OOT modules
export KERNEL_HEADERS=$PWD/kernel/kernel-noble
export kernel_name=noble
make modules
sudo -E make modules_install
cd <install-path>/Linux_for_Tegra && sudo ./tools/l4t_update_initrd.sh

# Build + install DTBs
cd <install-path>/Linux_for_Tegra/source
make dtbs
cp kernel-devicetree/generic-dts/dtbs/* \
  <install-path>/Linux_for_Tegra/kernel/dtb/
```

## Three divergences from this skill

| NVIDIA's docs | This skill |
|---|---|
| `$PWD == Linux_for_Tegra/source` | `$PWD == $KS == <source.root_path>/bsp_sources` (workspace-side checkout, not the BSP-shipped source tree). The user's `customize-*` edits live in `$KS`. |
| `cp / sudo install` writes directly into `bsp_image` | Outputs stay in-tree under `$KS`; `jetson-build-source` writes a manifest, `jetson-promote-image` does the bsp_image copy under Deploy. |
| Always builds full chain (kernel → modules → dtbs) | Build modes (`dt`/`oot`/`kernel`/`full`) match the dirty-repo profile; skips unaffected legs. |

The build commands themselves are identical — this skill just
points them at the workspace source root and skips legs the
dirty set doesn't imply.

## Spec status

**Locked in for v0.x:**

- **Build modes** — `dt`, `oot`, `kernel`, `full`, plus
  auto-dispatch from the dirty repo set; manual selection via
  the `argument-hint` skill argument.
- **Toolchain ownership** — `jetson-init-source` resolves and
  authors `source.toolchain`; this skill reads only.
- **Env / repo-list source of truth** —
  `kernel_src_build_env.sh` sourced from the extracted BSP.
- **Build driver** — drive NVIDIA's `Linux_for_Tegra/source/Makefile`
  + `kernel/Makefile` against `$KS`. Do not replicate
  `nvidia-dtbs` / `modules` / `kernel` recipes in skill code.
- **No overlay staging.** Build outputs stay in-tree; the
  manifest at `$MANIFEST` is the contract to
  `jetson-promote-image`.
- **Dirty-repo detection** — watermark file `.build-state.yaml`
  comparing per-repo HEADs, plus working-tree diff for
  uncommitted edits.
- **Trace-to-dirty manifest policy** — manifest emits only
  artifacts whose source files map to a dirty repo's kbuild
  target. Baseline-divergence noise is filtered out.
- **Install consolidation** — opt-in for `dt`/`oot`/`kernel`
  modes, default-on for `full`. User-owned stage dir, no sudo.
