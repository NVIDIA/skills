# Build manifest + watermark schema

Reference for `../SKILL.md` — the on-disk schemas
`jetson-build-source` writes after each successful run.

## `.build-manifest.yaml` (consumed by `/jetson-promote-image`)

**Trace-to-dirty policy.** A build run can produce artifacts that
*change* relative to upstream even when no customization edit drove
the change (toolchain version, build timestamps, upstream-vs-shipped
divergence). The manifest lists **only** artifacts traceable to a
dirty source repo. Promoting baseline-divergence noise would
attribute it to a customization's audit trail — forbidden.

Walk the `DIRTY` set and emit a manifest entry per implied artifact.
Source-file → kbuild-target → overlay-destination mapping:

| Dirty repo | Source change | Built artifact | `dst:` (relative to `<bsp_image.root_path>/Linux_for_Tegra/`) |
|---|---|---|---|
| `hardware/nvidia/<chip>/nv-public` (composite custom overlay) | `*-custom.dts` per `bsp-customization-kernel-dtb.md` | `$KS/kernel-devicetree/generic-dts/dtbs/<name>.dtbo` | `kernel/dtb/<name>.dtbo` |
| `hardware/nvidia/<chip>/nv-public` or `kernel-devicetree` (other) | `*.dts` / `*.dtsi` | `$KS/kernel-devicetree/generic-dts/dtbs/<name>.dtb` | `kernel/dtb/<name>.dtb` |
| OOT repos | `*.c` / `*.h` in OOT subtree | `$KS/<oot-repo>/<path>/<name>.ko` | `rootfs/lib/modules/<ver>/updates/<path>/<name>.ko` |
| `kernel/$KERNEL_SRC_DIR` (`obj-m`) | `*.c` for in-tree loadable module | `$KS/kernel/$KERNEL_SRC_DIR/<path>/<name>.ko` | `rootfs/lib/modules/<ver>/kernel/<path>/<name>.ko` |
| `kernel/$KERNEL_SRC_DIR` (`obj-y`) | `*.c` baked into `vmlinux` | `$KS/kernel/$KERNEL_SRC_DIR/arch/arm64/boot/Image` | `kernel/Image` |

**Note on the kernel `Image` dst.** The manifest emits one canonical
dst per build artifact — `kernel/Image`, the bootloader-side copy
read by the flash tool. The kernel `Image` lives in **two** places
inside `bsp_image` (`kernel/Image` and `rootfs/boot/Image`), but
the rootfs-side mirror is **not** the build's responsibility:
`/jetson-promote-image`'s "Mirror kernel Image into rootfs (when
kernel changed)" step copies `kernel/Image` to `rootfs/boot/Image`
before invoking the initramfs refresh. Adding a second
`kernel/Image` entry to the manifest would conflict with that
contract; keep this row as-written.

Manifest schema (YAML):

```yaml
mode: <auto-picked or skill-argument value>
toolchain: <CROSS_COMPILE>
bsp_version: "<bsp_image.version>"
rebuilt_at: <ISO-8601 timestamp>
dirty_repos:
  - <rel>: <new HEAD short sha>
artifacts:
  - kind: dtb|kernel_image|in_tree_module|oot_module
    src: <abs path under $KS>
    dst: <rel under <bsp_image.root_path>/Linux_for_Tegra/>
    source_repo: <which dirty repo this traces to>
```

Filter rules:

- Skip artifacts whose source repo is not in `DIRTY`.
- Skip artifacts byte-identical to the corresponding upstream copy.
- Skip `.dtb` files with no matching destination under `kernel/dtb/`
  (dispatch variants not in this BSP — `nvidia-dtbs` rebuild
  leftovers).

Atomic write: stage to `${MANIFEST}.tmp`, then `mv -f`.

## `.build-state.yaml` (per-repo watermark)

On success, rewrite `$STATE`:

```yaml
repos:
  <each rel>: <git rev-parse HEAD>
toolchain: <CROSS_COMPILE>
bsp_version: <bsp_image.version>
mode: <last-run mode>
```

The next `/jetson-build-source` run reads this back during the
"Detect dirty source repos" step — HEAD ≠ watermark → dirty.
