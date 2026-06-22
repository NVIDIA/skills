# Branch A — Local archive extraction (`public_sources.tbz2`)

Reference for `jetson-init-source` materializing `bsp_sources/` from a
pre-downloaded `public_sources.tbz2`. This is the default branch
(selected when an `archive:` is set in the profile, or when the entry
is absent and `<workspace>/Downloads/public_sources.tbz2` exists).

## Tarball shape

NVIDIA's `public_sources.tbz2` is a two-level archive:

- **Outer**: contains `Linux_for_Tegra/source/{<inner>.tbz2 + nvbuild.sh + Makefile + …}`.
- **Inner**: ~40 per-component tarballs that **extract flat** into a
  single source directory, producing the canonical sub-paths
  (`kernel/`, `nvgpu/`, `nvidia-oot/`, `hwpm/`, `hardware/`,
  `kernel-devicetree/`, `nvethernetrm/`, …) plus a top-level
  Makefile and `nv_*_src_build.sh` scripts.

## Archive path resolution

- `archive:` in profile, absolute → use as-is.
- `archive:` in profile, relative → resolved against `<workspace>/`
  (so `archive: Downloads/public_sources.tbz2` works after
  `/quick-start`).
- `archive:` in profile, tilde / env vars → expanded.
- Entry absent → auto-discover at
  `<workspace>/Downloads/public_sources.tbz2`. Print an INFO line
  so the user sees what's happening:

  ```
  INFO: using Downloads/public_sources.tbz2 (Branch A, default).
        To force source_sync.sh instead, delete the tarball or set
        source.repos.bsp_sources.url: in the profile.
  ```

  Auto-detection does **not** prompt — Branch A is the default by
  policy. Persistence stays explicit: auto-discovery is *not*
  written back into the profile.

## Extraction procedure

```bash
ARCHIVE="<resolved absolute path to public_sources.tbz2>"
DEST="<source.root_path>/bsp_sources"

# Pre-flight: refuse if DEST has prior content (user must clean it
# manually — we never auto-delete possibly-customized work).
if [ -d "$DEST" ] && [ -n "$(ls -A "$DEST" 2>/dev/null)" ]; then
  echo "REFUSE: $DEST is non-empty; remove or move it before re-extracting."; exit 1
fi
mkdir -p "$DEST"

# Verify tarball magic before doing anything else.
file -b "$ARCHIVE" | grep -q "bzip2 compressed" || {
  echo "REFUSE: $ARCHIVE is not a bzip2 tarball."; exit 1; }

# 1. Outer extract to a temp dir.
TMP=$(mktemp -d); trap "rm -rf '$TMP'" EXIT
tar xjf "$ARCHIVE" -C "$TMP"

# 2. Extract every inner tarball IN-PLACE at $DEST/. NVIDIA designed
#    each inner so its top-level paths ARE the canonical sub-paths
#    (or sub-paths of them); no per-tarball -C mapping table needed.
for inner in "$TMP"/Linux_for_Tegra/source/*.tbz2; do
  tar xjf "$inner" -C "$DEST"
done

# 3. Carry over the NVIDIA-provided build scripts at source/ top-level.
#    Drives `make modules` / nvbuild.sh against the extracted sub-paths.
cp -a "$TMP"/Linux_for_Tegra/source/*.sh "$DEST/" 2>/dev/null || true

# 3a. Pick the canonical Tegra OOT orchestrator for $DEST/Makefile.
#     public_sources.tbz2 ships TWO inner tarballs that each contain a
#     top-level Makefile: kernel_oot_modules_src.tbz2 (Tegra
#     orchestrator) and nvidia_kernel_display_driver_source_without_
#     root_dir.tbz2 (dGPU/OpenRM proprietary). On R36.x the dGPU tarball
#     extracts last and wins $DEST/Makefile, breaking arm64 cross-
#     builds downstream. Force-replace from <bsp_image>/Linux_for_Tegra/
#     source/Makefile (the canonical Tegra orchestrator placed by
#     apply_binaries.sh) when the current $DEST/Makefile lacks the
#     `modules: hwpm nvidia-oot nvgpu nvidia-display` signature.
TEGRA_MK="<bsp_image.root_path>/Linux_for_Tegra/source/Makefile"
if [ -f "$TEGRA_MK" ] && { [ ! -f "$DEST/Makefile" ] || \
   ! grep -qE '^modules:[[:space:]]+hwpm[[:space:]]+nvidia-oot[[:space:]]+nvgpu' "$DEST/Makefile"; }; then
  cp -f "$TEGRA_MK" "$DEST/Makefile"
fi

# 4. Single mono-repo: one `git init` at $DEST/ covering the entire
#    extracted tree (all canonical sub-paths + top-level Makefile /
#    glue scripts). Customize customization skills track changes in
#    this single repo's history regardless of which sub-path they
#    edit.
( cd "$DEST" \
  && git init -q \
  && git add -A \
  && git -c user.name=quick-start -c user.email=quick-start@local \
         commit -q -m "import pristine: extracted from $(basename "$ARCHIVE")" )
```

Branch A intentionally creates one mono-repo at `bsp_sources/.git`
covering canonical sub-paths plus top-level build glue. Branches B and C
may produce per-component repos; downstream build logic still walks the
canonical sub-paths under `bsp_sources/`.
