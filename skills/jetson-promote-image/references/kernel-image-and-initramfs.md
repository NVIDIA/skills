# Kernel `Image` mirror + initramfs refresh

Detailed reference for the two procedure steps that keep
`bsp_image`'s kernel and initramfs coherent after a kernel /
module promote: ["Mirror kernel Image into rootfs"](../SKILL.md#mirror-kernel-image-into-rootfs-when-kernel-changed)
and ["Refresh initramfs"](../SKILL.md#refresh-initramfs-when-kernel-or-modules-changed).

## Why this matters

Two failure modes converge on the same fix:

1. **Module shadowing.** The Jetson bootloader loads the initramfs
   **before** the NVMe rootfs is mounted and `pivot_root` runs.
   Any module that auto-loads during early boot — `pwm-tegra`,
   `nvgpu`, modules listed in `/etc/modules-load.d/`, modules
   pulled in by udev uevents on platform-bus enumeration — comes
   from the initramfs copy. Once a module is live in the kernel,
   the rootfs copy at `/lib/modules/<ver>/` **cannot** replace it
   (you cannot overwrite a loaded module). The rebuilt initramfs
   must carry the customized modules in place from boot one.
2. **Symbol-version / vermagic skew.** A kernel `Image` rebuild
   changes the kernel's exported symbol versions (and may shift
   internal layouts) even when `UTS_RELEASE` stays the same. The
   modules in the stale initramfs were built against the
   *previous* kernel and will be rejected at load time with
   "disagrees about version of symbol X", or worse silently bind
   to a different ABI. Rebuilding the initramfs from the rootfs's
   freshly built `/lib/modules/<ver>/` keeps the on-disk modules
   in lockstep with the kernel they have to load into.

Both modes are closed by the same gate on `kernel/Image` or
`rootfs/lib/modules/*`. Below is why the mirror step is required
*alongside* the refresh.

## Mirror semantics — kernel `Image` lands in two places

The kernel `Image` lives in two paths inside `bsp_image`:

- `<LFT_DST>/kernel/Image` — read by the flash tool, written to
  the kernel-DTB / boot partition the bootloader actually loads.
- `<LFT_DST>/rootfs/boot/Image` — the copy that lives inside the
  flashed rootfs (visible as `/boot/Image` on the booted DUT, and
  visible as `/boot/Image` from inside any chroot rooted at
  `rootfs/`).

The build manifest only carries the `kernel/Image` dst (build
output goes to one canonical destination per artifact). Without
the mirror, the next step (`l4t_update_initrd.sh`) chroots into
`rootfs/` and runs `nv-update-initrd`, which resolves the kernel
via `/boot/Image` (= `rootfs/boot/Image`) — i.e. the **stale**
copy — and would build the initramfs against the previous
kernel's symbol table even though we just promoted a new one.
Flashing the resulting image leaves the DUT with a fresh
`kernel/Image` paired with stale-vermagic modules baked into the
initramfs.

### Mirror snippet

Mirror immediately after the diff-aware copy, before invoking the
refresh tool. Diff-aware: skip if already byte-identical (the
flash tool may have synced them on a prior run; rerunning is a
no-op).

```bash
KIMG_SRC="$LFT_DST/kernel/Image"
KIMG_DST="$LFT_DST/rootfs/boot/Image"
KIMG_MIRRORED="skipped (already identical or kernel/Image absent)"
if [ -f "$KIMG_SRC" ]; then
  if ! [ -f "$KIMG_DST" ] || ! cmp -s "$KIMG_SRC" "$KIMG_DST"; then
    sudo cp -p "$KIMG_SRC" "$KIMG_DST"
    KIMG_MIRRORED="copied $KIMG_SRC -> $KIMG_DST"
    INITRD_DIRTY=1   # mirror is a write under rootfs/; refresh now mandatory
  fi
fi
```

The `INITRD_DIRTY=1` set here covers the corner case where the
manifest carried only `kernel/Image` (no `rootfs/lib/modules/*`
entries — e.g. an obj-y-only kernel edit) and `copy_one` never
flipped the flag. After this step, `INITRD_DIRTY=1` iff either
the kernel binary or any module changed in the rootfs.

## Refresh tool — `l4t_update_initrd.sh`

`tools/l4t_update_initrd.sh` ships inside every Linux_for_Tegra
tree. It rebuilds the initramfs by chrooting into
`<LFT_DST>/rootfs/`, running NVIDIA's `nv-update-initrd` inside
it, then copying the result to **both**
`<LFT_DST>/bootloader/l4t_initrd.img` (the bootloader-side base
used by the flash tool) and `<LFT_DST>/rootfs/boot/initrd` (the
rootfs copy).

### Refresh snippet

```bash
if [ "$INITRD_DIRTY" = 1 ]; then
  [ -x "$LFT_DST/tools/l4t_update_initrd.sh" ] || {
    # refuse: "tool not found at
    # $LFT_DST/tools/l4t_update_initrd.sh; re-run /jetson-init-image
    # to repopulate Linux_for_Tegra/tools/."
  }
  ( cd "$LFT_DST" && sudo ./tools/l4t_update_initrd.sh ) || {
    # refuse: "l4t_update_initrd.sh exited non-zero;
    # bsp_image initrd is in indeterminate state. Re-run promote
    # after fixing the underlying cause."
  }
  INITRD_STATUS="rebuilt"
else
  INITRD_STATUS="skipped (no kernel/Image or rootfs/lib/modules/ changes)"
fi
```

### Outputs (when the tool runs)

- `<LFT_DST>/bootloader/l4t_initrd.img` — the base image used by
  the flash tool to populate the bootloader-side initramfs (lands
  in the kernel-DTB / boot partition at flash time, depending on
  the carrier).
- `<LFT_DST>/rootfs/boot/initrd` — becomes `/boot/initrd` on the
  booted DUT.

(Note: `<LFT_DST>/bootloader/initrd` is a separate, pre-staged
file used during the initrd-flash boot sequence — it is **not**
touched by this step; `l4t_initrd_flash.sh` regenerates / consumes
it independently.)

### Skip behavior

Skip the entire step when `INITRD_DIRTY=0` (typical for
overlay-only edits like `nvfancontrol.conf` / `nvpmodel.conf` /
BPMP DTB / MB1 PMIC BCT — none of which touch the kernel `Image`
or `rootfs/lib/modules/`).

The tool is idempotent: rerunning after a no-op promote produces
byte-identical initrd files. Typical runtime: ~30 s.

### Out-of-scope workarounds

DUT-side workarounds (e.g. `update-initramfs -u` + manual `cp` of
`/boot/initrd.img-<ver>` over `/boot/initrd`) are explicitly out
of scope here — fix the gap at promote time so flash ships a
coherent image. The static + on-target checks in
[`../jetson-validate-image/SKILL.md`](../../jetson-validate-image/SKILL.md)
catch drift if it ever re-appears.
