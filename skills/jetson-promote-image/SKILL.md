---
name: jetson-promote-image
description: >-
  Use to promote overlay files and built artifacts into the staged
  BSP image. Do NOT use to flash or build. Triggers: promote bsp
  image.
version: 0.0.1
license: "Apache-2.0"
metadata:
  data-classification: public
  author: "Jetson Team"
  team: pts
  tags:
    - bsp
    - promote
    - deploy
  domain: meta
---

# Promote BSP Image

## Purpose

Stage every Customize-* and Build output into `bsp_image` so it is
ready for `/jetson-flash-image`. This is the **promote leg of
Deploy** — it copies files, never flashes and never builds.

## Prerequisites

- Active target-platform profile with both `source:` and `bsp_image:`
  resolved (run `/jetson-init-source` and `/jetson-init-image` first).
- `<source.root_path>/Linux_for_Tegra/` initialized as a git repo
  (overlay tracker) with a clean working tree.
- `<bsp_image.root_path>/Linux_for_Tegra/` extracted from a BSP
  tarball + `apply_binaries.sh` already run.
- `git`, `yq`, `cmp`, and `sudo` (for `rootfs/*` destinations) on
  the host.
- `<source.root_path>/.build-manifest.yaml` + `.build-state.yaml`
  from `/jetson-build-source` (required when kernel-side repos
  have customize-* commits).

## Overview

This is the **promote leg of Deploy** — see
[`../../context/bsp-customization-workflow.md`](../../context/bsp-customization-workflow.md)
for the pipeline view. The two channels this skill walks are:

| Channel | Source | Carrier | Owner |
|---|---|---|---|
| **Overlay tracker** | `<source.root_path>/Linux_for_Tegra/` (git repo at HEAD) | Customize-* outputs that don't require a build (e.g. `nvfancontrol.conf`, `nvpmodel.conf`, BPMP DTB hand-edits) | Customize `customize-*` skills commit here |
| **Build manifest** | `<source.root_path>/.build-manifest.yaml` | Rebuilt kernel `Image`, in-tree `.ko`, OOT `.ko`, NVIDIA DTBs | Build [`jetson-build-source`](../jetson-build-source/SKILL.md) writes here |

The skill computes the union of files to copy and writes each into
`<bsp_image.root_path>/Linux_for_Tegra/` with diff-aware
skip-if-identical logic. When the copy pass touches the kernel
`Image` or anything under `rootfs/lib/modules/`, it also rebuilds
the initramfs via NVIDIA's `tools/l4t_update_initrd.sh` so the
freshly promoted kernel + modules ship in the initrd the
bootloader actually loads. After it returns, `bsp_image` carries
every Customize and Build output. The skill does **not** flash and
does **not** modify the workspace.

## When to invoke

- First leg of the typical Deploy chain
  `jetson-promote-image → jetson-flash-image → jetson-validate-image`.
- Standalone, when the user wants `bsp_image` updated but isn't
  ready to flash yet (e.g. to inspect resolved files, run an
  out-of-band build that reads bsp_image, or hand bsp_image to a
  separate flashing host).

## Procedure

### Resolve active target + paths

Resolve the active profile per the contract in
[`../../context/target-platform-contract.md`](../../context/target-platform-contract.md).

Refuse and route in these cases:

| Condition | Refuse with |
|---|---|
| No active profile, or `active: NA` | Route to `/jetson-set-target` or `/jetson-init-target`. |
| Profile lacks `bsp_image:` | Route to `/jetson-init-image`. |
| `<bsp_image.root_path>/Linux_for_Tegra/` missing | Route to `/jetson-init-image`. |
| `<source.root_path>/Linux_for_Tegra/` missing or not a git repo | Route to `/jetson-init-source`. |

Resolve paths:

- `<workspace>` = parent of the active profile's `target-platform/`
  directory (discovered at load time).
- `<bsp_image.root_path>` from `bsp_image.root_path:` if present,
  else `<workspace>/Image`.
- `<source.root_path>` from `source.root_path:` if present, else
  `<workspace>/Source`.

Bind shell variables for the rest of the procedure:

```bash
LFT_SRC="<source.root_path>/Linux_for_Tegra"   # overlay tracker
LFT_DST="<bsp_image.root_path>/Linux_for_Tegra"
MANIFEST="<source.root_path>/.build-manifest.yaml"   # build outputs
```

### Validate the two channels

The skill needs at least one channel populated. Refuse if the
overlay tracker has uncommitted changes (`status --porcelain`
non-empty), if `$MANIFEST` exists but doesn't parse as YAML, or
if both channels are empty. Records `OVERLAY_HAS_COMMITS` /
`OVERLAY_HEAD` and `MANIFEST_PRESENT` for downstream steps.

See [`references/copy-pass-snippets.md`](references/copy-pass-snippets.md#validate-the-two-channels)
for the shell snippet and refuse messages.

### Verify build-source freshness

Refuse if `.build-state.yaml` shows any kernel-side repo in
`Source/bsp_sources/` dirty since the last `/jetson-build-source`
— otherwise the copy pass would silently ship stale artifacts.
Detection rules + shell snippet in
[`references/build-source-freshness-gate.md`](references/build-source-freshness-gate.md).
Records `BUILD_FRESH=1`.

### Pre-promote collision check (overlay only)

When the overlay tracks a remote, refuse if upstream has commits
not yet pulled. Skip gracefully when no remote is configured
(the default `git init` empty tracker from `jetson-init-source`).
Manifest channel has no git remote concept — this check is
overlay-only. Records `COLLISION_CHECK` for the Summary.

See [`references/copy-pass-snippets.md`](references/copy-pass-snippets.md#pre-promote-collision-check-overlay-only)
for the shell snippet.

### Enumerate sources (both channels)

**Channel A — overlay**: `git ls-files` against `$LFT_SRC` is
the source of truth (transparent to symlink mounts when
`source.repos.Linux_for_Tegra` was overridden, excludes
untracked / `.gitignore`d files). Each entry maps
`src = $LFT_SRC/<rel>` → `dst = $LFT_DST/<rel>`.

**Channel B — manifest**: parse `artifacts[].{src,dst}` from
`$MANIFEST`. Refuse if any `src` is missing on disk (build was
interrupted, or manifest stale — re-run `/jetson-build-source`).
The manifest schema is written by
[`jetson-build-source` v0.2.0](../jetson-build-source/references/manifest-schema.md).

See [`references/copy-pass-snippets.md`](references/copy-pass-snippets.md#enumerate-sources)
for both shell snippets and the manifest YAML schema.

### Diff-aware copy into bsp_image

Iterate the union of overlay files and manifest entries. For
each `dst`: if byte-identical, skip; otherwise `cp -p` (with
`sudo` for `rootfs/*` destinations, where the sample rootfs
was extracted as root). Tag `INITRD_DIRTY=1` on any
`rootfs/lib/modules/*` or `kernel/Image` write — the
"Refresh initramfs" step gates on this flag. Counts /
`FIRST` / `LAST` are recorded for the Summary.

Fail-fast: if any `cp` fails, surface the failed path and stop.
`bsp_image` may be left partially updated — re-running after
fixing the cause resumes via the diff-aware skip. **Channel
order** is overlay first, then manifest: on a `dst` collision
the manifest wins (freshly built artifact beats the older
overlay copy).

See [`references/copy-pass-snippets.md`](references/copy-pass-snippets.md#diff-aware-copy-into-bsp_image)
for the `copy_one()` function and the two driving loops.

### Mirror kernel Image into rootfs (when kernel changed)

The kernel `Image` lives in two paths inside `bsp_image`:
`<LFT_DST>/kernel/Image` (read by the flash tool) and
`<LFT_DST>/rootfs/boot/Image` (the rootfs-side copy, visible as
`/boot/Image` from inside the rootfs chroot the refresh tool
will run in). The build manifest only carries the `kernel/Image`
dst, so this step mirrors `kernel/Image` → `rootfs/boot/Image`
(diff-aware, no-op when already in sync) so the chrooted refresh
tool resolves the kernel against the freshly promoted binary,
not the stale rootfs copy. The mirror also sets `INITRD_DIRTY=1`
so a kernel-only promote (no `rootfs/lib/modules/*` writes) still
triggers the refresh.

See [`references/kernel-image-and-initramfs.md`](references/kernel-image-and-initramfs.md#mirror-semantics--kernel-image-lands-in-two-places)
for the shell snippet, the failure mode this prevents, and the
`INITRD_DIRTY` corner case.

### Refresh initramfs (when kernel or modules changed)

Run `tools/l4t_update_initrd.sh` from `<LFT_DST>/` whenever
`INITRD_DIRTY=1` (set by the diff-aware copy or the mirror step
above). The tool chroots into `rootfs/`, runs NVIDIA's
`nv-update-initrd`, and writes both
`<LFT_DST>/bootloader/l4t_initrd.img` (used by the flash tool)
and `<LFT_DST>/rootfs/boot/initrd` (`/boot/initrd` on the DUT).
Idempotent; ~30 s. Skip when `INITRD_DIRTY=0` (overlay-only
edits). DUT-side workarounds (`update-initramfs -u` + manual
`cp`) are out of scope — fix the gap here so flash ships a
coherent image.

See [`references/kernel-image-and-initramfs.md`](references/kernel-image-and-initramfs.md#refresh-tool--l4t_update_initrdsh)
for the shell snippet, refuse paths, the "module shadowing" and
"vermagic skew" failure modes the rebuild closes, and why
`bootloader/initrd` (a different file) is left alone.

### Summary

Report:

- Overlay scope: `overlay HEAD ($OVERLAY_HEAD)` or "(empty)".
- Manifest scope: `mode=<...>, bsp_version=<...>, rebuilt_at=<...>,
  N artifacts` or "(absent)".
- Collision check: `$COLLISION_CHECK`.
- Counts:
  - overlay: `$COPIED_OVERLAY copied, $IDENTICAL_OVERLAY identical`
  - manifest: `$COPIED_MANIFEST copied, $IDENTICAL_MANIFEST identical`
- Kernel Image mirror: `$KIMG_MIRRORED` and initramfs:
  `$INITRD_STATUS` (`copied …` / `rebuilt` when triggered by
  `kernel/Image` or `rootfs/lib/modules/*` writes; `skipped …`
  otherwise).
- First / last paths copied (omit if both `COPIED` totals are 0).
- Resolved `<source.root_path>`, `<bsp_image.root_path>`.
- Next step: `/jetson-flash-image` (or `/jetson-validate-image` if
  the user only wanted bsp_image refreshed for inspection / static
  validation).

## Limitations

- **Two channels, one destination.** `bsp_image/Linux_for_Tegra/`
  is written by both passes. Overlay carries customize-* outputs
  (overlay-only edits like nvfancontrol.conf); manifest carries
  rebuilt binaries (kernel/OOT/DT). The two are intentionally
  disjoint by construction: build outputs don't go into the
  overlay, and customize-* edits to non-build files don't enter
  the manifest.
- **Build manifest is the trace-to-dirty contract.** Anything in
  the manifest came from a dirty source repo (per
  `jetson-build-source`'s "Write the build manifest" step trace policy). Promoting the
  manifest is therefore safe: every entry is a customization-bearing
  artifact, not toolchain-divergence noise. The skill does not
  re-derive the trace — it trusts the manifest.
- **Manifest entries can outlive their build outputs.** If the
  user wipes `Source/.build/` or `bsp_sources/`'s build artifacts
  between `jetson-build-source` and `jetson-promote-image`, the
  manifest will reference missing files. The "Enumerate sources (both channels)" step refuses in that
  case and points the user at `/jetson-build-source` to rebuild.
- **Manifest absence is fine when only overlay edits happened.**
  A purely overlay-side customization (e.g. `customize-fan`)
  produces no build outputs and writes no manifest — the "Enumerate sources (both channels)" step is a
  no-op, the "Diff-aware copy into bsp_image" step promotes only overlay files. The skill prints
  "manifest: (absent)" in the summary and continues.
- **Diff-aware, idempotent.** Re-running with no overlay commits
  or manifest changes since the last promote is a no-op (all
  files identical). Use this to confirm bsp_image is in sync
  without side effects.
- **Symlink-mount transparency.** When
  `source.repos.Linux_for_Tegra` was overridden in
  `jetson-init-source`, the canonical mount is a symlink into
  `<source.root_path>/.repos/Linux_for_Tegra/<subdir>`. `git -C`,
  `cp -p`, and `cmp -s` all follow it transparently — no special
  handling needed at this layer. Manifest `src` paths are
  absolute, so symlinks under `bsp_sources/` don't matter for the
  manifest channel.
- **`sudo` is scoped to `rootfs/` destinations.** Files under
  `rootfs/` were extracted with `sudo tar xpjf` by
  `jetson-init-image`, so they carry root ownership and special
  mode bits the flashing toolchain reads back. `sudo cp -p`
  preserves them. Everything else (`bootloader/`, `kernel/`,
  `kernel/dtb/`, `tools/`, etc.) is user-owned and does not need
  `sudo`. This applies to both channels.
- **Channel-overlap precedence.** If the same `dst` appears in
  both overlay and manifest, manifest wins (later in the "Diff-aware copy into bsp_image" step's
  loop). This is the desired semantic — manifest entries are
  freshly built, overlay entries may be older state. Hand-editing
  binary files into the overlay is discouraged (Build's job
  is to rebuild them); the precedence rule makes such mistakes
  recoverable.
- **`bsp_image` is read-only outside Deploy.** This skill is the
  only writer in the normal flow (matches the workflow invariant).
  Hand-edits to `<bsp_image.root_path>/Linux_for_Tegra/` outside
  Deploy will be silently overwritten on the next promote run if
  the same path exists in either channel; conversely they will
  *not* be reverted if no entry shadows them. Both behaviors are
  wrong for the diff trail — never hand-edit upstream.
- **Scope is overlay HEAD only (channel A).** Named tags /
  manifests / commit ranges are deferred (see below). To promote
  a historical state, `git -C $LFT_SRC checkout <ref>` first,
  then re-run. The manifest channel has no ranged scope — it
  reflects whatever `jetson-build-source`'s last run produced.
- **No automatic rollback on partial failure.** If `cp` fails
  partway through, `bsp_image` is left in an intermediate state.
  Fix the underlying cause (usually permissions / disk full) and
  re-run — the "Diff-aware copy into bsp_image" step will resume by skipping already-promoted files.
- **Kernel `Image` mirror + initramfs refresh.** Gated on copy-pass
  writes to `kernel/Image` or `rootfs/lib/modules/*`; the mirror
  feeds the refresh's chroot. Both are diff-aware and skipped on
  pure-overlay edits. `tools/l4t_update_initrd.sh` must exist in
  `bsp_image` (ships with `apply_binaries.sh`); a missing tool
  refuses and routes to `/jetson-init-image`. See
  [`references/kernel-image-and-initramfs.md`](references/kernel-image-and-initramfs.md)
  for the full contract and failure modes.

## Troubleshooting

| Error | Cause | Solution |
|---|---|---|
| `Overlay has uncommitted changes at <LFT_SRC>` | Customize-* edits not committed before promote | Run `git -C $LFT_SRC commit` (or stash), then re-run. |
| `origin has N unpulled commits on <upstream>` | Remote overlay diverged from local | `git -C $LFT_SRC pull`, resolve conflicts, then re-run. |
| `Both overlay and manifest are empty — nothing to promote` | No Customize-* commits and no Build manifest | Run a customize-* skill or `/jetson-build-source` first. |
| `Kernel-side source(s) changed since last /jetson-build-source` | Freshness gate detected unprocessed customize-* edits under `Source/bsp_sources/` | Commit pending edits, run `/jetson-build-source`, re-run promote. |
| `Manifest entry references missing build output: <src>` | `bsp_sources/` build outputs wiped or stale manifest | Re-run `/jetson-build-source` to regenerate. |
| `Build manifest at <MANIFEST> is not valid YAML` | Manifest hand-edited or partially written | Re-run `/jetson-build-source` to rewrite the manifest. |
| `cp: permission denied` under `rootfs/` | Missing `sudo` privilege on the host | Run on an account that can `sudo cp`; re-run resumes via diff-aware copy. |
| Profile lacks `bsp_image:` / `source:` | Workspace not bootstrapped | Run `/jetson-init-image` and/or `/jetson-init-source`. |
| `tool not found at <LFT_DST>/tools/l4t_update_initrd.sh` | `tools/` was pruned, or bsp_image extracted from a non-NVIDIA tarball | Re-run `/jetson-init-image` to repopulate. |
| `l4t_update_initrd.sh exited non-zero` | Insufficient sudo, broken rootfs (missing `lib/modules/<ver>/modules.dep`), or out-of-space `/tmp` | Run `depmod -a -b <LFT_DST>/rootfs <ver>` against the rootfs first; verify `/tmp` headroom; rerun promote. |
| DUT boots with stale kernel / modules after promote, modules fail to load with `disagrees about version of symbol …`, or initramfs ships pre-customize modules even after the refresh ran | The mirror / refresh gate didn't fire (manual hand-edit under `<LFT_DST>` outside the skill), or `rootfs/boot/Image` drifted from `kernel/Image` so the chrooted refresh built against the stale kernel | Force the gate by `sudo touch <LFT_DST>/kernel/Image` + re-run promote, or run the two steps manually: `sudo cp -p <LFT_DST>/kernel/Image <LFT_DST>/rootfs/boot/Image && cd <LFT_DST> && sudo ./tools/l4t_update_initrd.sh`. Then re-flash. See [`references/kernel-image-and-initramfs.md`](references/kernel-image-and-initramfs.md). |

## Spec status

**Locked in for v0.2.0:**

- **Two-channel scope** — overlay HEAD + build manifest, both
  diff-aware, both copying into `<bsp_image.root_path>/Linux_for_Tegra/`.
- **Channel-overlap precedence** — manifest wins on `dst` collision.
- **Source-repo collision check** — overlay only; manifest has no
  remote concept and source repos under `bsp_sources/` are not
  fetched (their state was sealed when `jetson-build-source`
  wrote the manifest).
- **Atomicity** — fail-fast, no rollback. Diff-aware copy makes
  resume natural.
- **Audit trail** — stdout-only at promote time. The overlay
  tracker's git log is the canonical record for channel A; the
  manifest itself is the canonical record for channel B.
- **Kernel Image mirror + initramfs refresh.** Locked in as a
  paired step. The mirror copies `kernel/Image` →
  `rootfs/boot/Image` whenever the copy pass touched
  `kernel/Image`; the refresh runs
  `tools/l4t_update_initrd.sh` whenever `kernel/Image` or any
  `rootfs/lib/modules/*` was promoted, rebuilding both
  `bootloader/l4t_initrd.img` and `rootfs/boot/initrd`.
  Inseparable because the refresh chroots into `rootfs/` and
  resolves the kernel through `/boot/Image` — the mirror has to
  run first. Closes both module-shadowing and vermagic-skew
  failure modes; both diff-aware, both skipped on overlay-only
  edits. Full contract in
  [`references/kernel-image-and-initramfs.md`](references/kernel-image-and-initramfs.md).

**Still deferred:**

- **Named-tag / commit-range scope** for the overlay channel.
  Revisit when a "promote release X" use case appears.
- **Manifest history.** Currently only the last build's manifest
  exists; if a user wants to roll bsp_image back to a previous
  build state, they'd need to re-run `/jetson-build-source` at
  the prior commit. A manifest archive (saved per-build-mode or
  per-commit) would enable rollback without rebuild.
- **Sidecar manifest in `bsp_image`.** Revisit when promotion
  happens on a host that does not have access to the overlay
  tracker repo (or the workspace's manifest file).

## References

- [`references/kernel-image-and-initramfs.md`](references/kernel-image-and-initramfs.md) — full contract for the kernel `Image` mirror + `l4t_update_initrd.sh` refresh: shell snippets, failure modes, tool semantics, output filenames.
- [`../../context/target-platform-contract.md`](../../context/target-platform-contract.md) — target-platform contract.
- [`../../context/bsp-customization-workflow.md`](../../context/bsp-customization-workflow.md#workflow-invariants) — workspace edit protocol (this skill is the promote leg of Deploy).
- [`../jetson-init-source/SKILL.md`](../jetson-init-source/SKILL.md) — Setup; materializes the overlay tracker this skill reads (channel A) and authors `source.toolchain`.
- [`../jetson-build-source/SKILL.md`](../jetson-build-source/SKILL.md) — Build builder; writes the `.build-manifest.yaml` this skill reads (channel B).
- [`../jetson-flash-image/SKILL.md`](../jetson-flash-image/SKILL.md) — next leg; flashes the just-promoted bsp_image to the DUT.
- [`../jetson-validate-image/SKILL.md`](../jetson-validate-image/SKILL.md) — final leg; static + on-target validation.
