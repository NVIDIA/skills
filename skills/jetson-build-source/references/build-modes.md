# Per-mode build snippets

Detailed reference for `../SKILL.md` — the actual
`make` invocations for each of the four build modes (`dt`, `oot`,
`kernel`, `full`) plus the optional install consolidation pass. All
snippets assume the `## Prerequisites` block in the parent SKILL.md
has bound `$KS`, `$KOUT`, `$STAGE`, `$BSP_SRC`, `$KERNEL_SRC_DIR`,
`$KERNEL_DEF_CONFIG`, `$ARCH`, `$CROSS_COMPILE`, `$kernel_name`.

## Common setup (all modes)

Ensure the top-level Tegra OOT orchestrator Makefile and the
`kernel/Makefile` umbrella are present at `$KS/Makefile` and
`$KS/kernel/Makefile`. The top-level must be the **Tegra
orchestrator**, not the dGPU/OpenRM proprietary Makefile some R36.x
Branch-A extractions leave behind (see Limitations):

```bash
# Tegra orchestrator signature: `modules:` chains hwpm → nvidia-oot
# → nvgpu → nvidia-display. Force-replace on mismatch (covers
# Branch B/C absent-file AND R36.x Branch-A dGPU collision).
if [ ! -f "$KS/Makefile" ] || \
   ! grep -qE '^modules:[[:space:]]+hwpm[[:space:]]+nvidia-oot[[:space:]]+nvgpu' "$KS/Makefile"; then
  cp -f "$BSP_SRC/Makefile" "$KS/Makefile"
fi
[ -f "$KS/kernel/Makefile" ] || cp "$BSP_SRC/kernel/Makefile" "$KS/kernel/Makefile"
cd "$KS"
# ARCH and CROSS_COMPILE already exported in the "Resolve toolchain" step.
```

`/jetson-init-source` step 3a is the primary defense (extract-time);
this is the safety net for Branch B/C and regressions. If it fires,
don't commit the swap — `git checkout HEAD -- Makefile` restores.

## DT-only (`dt` argument)

Fast-path optimization over NVIDIA's official recipe (which does
the full `make -C kernel` before `make dtbs`). DTBs only need the
kbuild scripts + generated headers, so we `defconfig` + `prepare`
against an out-of-tree build dir, then invoke `nvidia-dtbs`.
Verified to produce byte-identical DTBs to the official flow.

```bash
mkdir -p "$KOUT"

# 1. Configure kernel (writes .config under KOUT).
make -C "$KS/kernel/$KERNEL_SRC_DIR" O="$KOUT" "$KERNEL_DEF_CONFIG"

# 2. Prepare kbuild generated headers.
make -C "$KS/kernel/$KERNEL_SRC_DIR" O="$KOUT" prepare -j"$(nproc)"

# 3. Build all NVIDIA DTBs.
make -C "$KS" \
  KERNEL_HEADERS="$KS/kernel/$KERNEL_SRC_DIR" \
  KERNEL_OUTPUT="$KOUT" \
  -j"$(nproc)" \
  dtbs
```

**`KERNEL_HEADERS` must point at the kernel source dir, not
`$KOUT`** — the `nvidia-dtbs` target includes
`$(KERNEL_HEADERS)/scripts/Makefile.compiler`, which only exists
in the source tree (not materialized into `$KOUT` by `make prepare`).
Setting both to `$KOUT` fails immediately with
`No rule to make target '$KOUT/scripts/Makefile.compiler'`.
`KERNEL_OUTPUT` remains `$KOUT` (objtree). They have different
semantics — srctree vs objtree — and must not be collapsed.

Output: `$KS/kernel-devicetree/generic-dts/dtbs/*.dtb` (flat list
of every variant the hardware tree carries).

## OOT modules (`oot` argument)

Per NVIDIA's Jetson Linux Developer Guide. Requires the kernel
source tree to have been previously prepared (auto-chain from a
clean state runs `kernel` mode first; a manual `oot` invocation
against a never-built tree refuses with "run `kernel` first").

```bash
export KERNEL_HEADERS="$KS/kernel/$KERNEL_SRC_DIR"   # in-tree, not KOUT
export kernel_name="$kernel_name"                    # from env file
make -j"$(nproc)" modules
```

Outputs (in-tree, under each OOT repo):
`$KS/{nvidia-oot,nvgpu,nvdisplay,nvethernetrm,hwpm,unifiedgpudisp}/<path>/*.ko`.

## Kernel only (`kernel` argument)

Per NVIDIA's Jetson Linux Developer Guide. The `kernel/Makefile`
orchestrator runs `defconfig` + `Image` + kernel-side `dtbs` +
in-tree `modules` in one chain:

```bash
make -C kernel
```

Outputs (in-tree):

- `$KS/kernel/$KERNEL_SRC_DIR/arch/arm64/boot/Image`
- `$KS/kernel/$KERNEL_SRC_DIR/**/*.ko` (full in-tree loadable module set)
- `$KS/kernel/$KERNEL_SRC_DIR/arch/arm64/boot/dts/**/*.dtb`
  (vanilla in-tree DTs — *not* the NVIDIA-platform DTBs, which
  the DT-only and Full modes' `dtbs` target produces)

## Full (`full` argument)

Sequence the above in dependency order:

```bash
make -C kernel              # kernel-only leg
make -j"$(nproc)" modules   # OOT-modules leg
make -j"$(nproc)" dtbs      # nvidia-dtbs against the in-tree-built kernel
```

Then optionally consolidate via the install legs (the "Install
consolidation" section below).

## Install consolidation (automatic in `full`)

Useful for (a) clean install layout that mirrors the shipped
`rootfs/lib/modules/<ver>/` tree, (b) manual scp deploy to a DUT
without going through Deploy.

```bash
mkdir -p "$STAGE/boot" "$STAGE/lib/modules"
INSTALL_MOD_PATH="$STAGE" make install -C kernel   # Image + in-tree modules
INSTALL_MOD_PATH="$STAGE" make modules_install     # OOT modules
```

User-owned stage dir → **no sudo needed** (diverges from NVIDIA's
recipe, which installs directly into `bsp_image/rootfs/` with
sudo; this workflow forbids that — `bsp_image` is read-only
outside Deploy).
