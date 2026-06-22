# Two-channel copy-pass — shell snippets

Detailed reference for `../SKILL.md`'s procedure steps:
- ["Validate the two channels"](../SKILL.md#validate-the-two-channels)
- ["Pre-promote collision check"](../SKILL.md#pre-promote-collision-check-overlay-only)
- ["Enumerate sources (build-manifest channel)"](../SKILL.md#enumerate-sources-both-channels)
- ["Diff-aware copy into bsp_image"](../SKILL.md#diff-aware-copy-into-bsp_image)

All snippets assume the shell variables `$LFT_SRC`, `$LFT_DST`,
`$MANIFEST` are bound per
[SKILL.md's "Resolve active target + paths" step](../SKILL.md#resolve-active-target--paths).

## Validate the two channels

The skill needs at least one channel populated. Refuse if the
overlay tracker has uncommitted changes, the manifest YAML is
invalid, or both channels are empty.

```bash
# Channel A: overlay tracker — clean, on a real commit.
OVERLAY_HAS_COMMITS=0
if git -C "$LFT_SRC" rev-parse --verify HEAD >/dev/null 2>&1; then
  if [ -n "$(git -C "$LFT_SRC" status --porcelain)" ]; then
    # refuse: "Overlay has uncommitted changes at $LFT_SRC.
    # Commit or stash them, then re-run."
  fi
  OVERLAY_HAS_COMMITS=1
  OVERLAY_HEAD=$(git -C "$LFT_SRC" rev-parse --short HEAD)
fi

# Channel B: build manifest — exists and parses.
MANIFEST_PRESENT=0
if [ -f "$MANIFEST" ]; then
  yq '.' "$MANIFEST" >/dev/null || {
    # refuse: "Build manifest at $MANIFEST is not valid YAML."
  }
  MANIFEST_PRESENT=1
fi

if [ "$OVERLAY_HAS_COMMITS" = 0 ] && [ "$MANIFEST_PRESENT" = 0 ]; then
  # refuse / report: "Both overlay and manifest are empty — nothing
  # to promote. Run a customize-* or /jetson-build-source first."
fi
```

## Pre-promote collision check (overlay only)

When the overlay tracks a remote, refuse if the remote has commits
the local user hasn't pulled. Skip gracefully when no remote is
configured (the default `git init` empty tracker from
`jetson-init-source`). The manifest channel has no git remote
concept — this check applies only to the overlay.

```bash
COLLISION_CHECK="n/a (overlay has no commits)"
if [ "$OVERLAY_HAS_COMMITS" = 1 ]; then
  if UPSTREAM=$(git -C "$LFT_SRC" rev-parse --abbrev-ref \
                --symbolic-full-name '@{u}' 2>/dev/null); then
    git -C "$LFT_SRC" fetch origin
    BEHIND=$(git -C "$LFT_SRC" rev-list --count "HEAD..$UPSTREAM")
    if [ "$BEHIND" -gt 0 ]; then
      # refuse: "origin has $BEHIND unpulled commits on $UPSTREAM.
      # Run `git -C $LFT_SRC pull`, resolve any conflicts, then re-run."
    fi
    COLLISION_CHECK="passed ($UPSTREAM)"
  else
    COLLISION_CHECK="skipped (no remote — single-user overlay)"
  fi
fi
```

## Enumerate sources

### 4a — Overlay tracked files

`git ls-files` is the source of truth — it natively handles
symlink mounts (when `source.repos.Linux_for_Tegra` was
overridden in `jetson-init-source`) and excludes untracked /
`.gitignore`d files.

```bash
OVERLAY_FILES=()
if [ "$OVERLAY_HAS_COMMITS" = 1 ]; then
  mapfile -t OVERLAY_FILES < <(git -C "$LFT_SRC" ls-files)
fi
```

Each overlay entry maps `src = $LFT_SRC/<rel>` →
`dst = $LFT_DST/<rel>`.

### 4b — Build manifest entries

Schema written by `jetson-build-source` v0.2.0:

```yaml
mode: <auto-picked or skill-argument value>
toolchain: <CROSS_COMPILE>
bsp_version: <bsp_image.version>
rebuilt_at: <ISO-8601 timestamp>
dirty_repos:
  - <rel>: <new HEAD short sha>
artifacts:
  - kind: dtb|kernel_image|in_tree_module|oot_module
    src: <abs path under <source.root_path>/bsp_sources/>
    dst: <rel under <bsp_image.root_path>/Linux_for_Tegra/>
    source_repo: <rel under bsp_sources/ — which dirty repo this traces to>
```

Read each `artifacts[].src` and `artifacts[].dst`. Refuse the
manifest if any `src` does not exist on disk (build was
interrupted, or manifest stale relative to actual build state —
re-run `/jetson-build-source` to regenerate).

```bash
MANIFEST_ENTRIES=()
if [ "$MANIFEST_PRESENT" = 1 ]; then
  while IFS= read -r line; do
    MANIFEST_ENTRIES+=("$line")
  done < <(yq -r '.artifacts[] | "\(.src)\t\(.dst)"' "$MANIFEST")
  for line in "${MANIFEST_ENTRIES[@]}"; do
    src=${line%%$'\t'*}
    [ -f "$src" ] || {
      # refuse: "Manifest entry references missing build output: $src.
      # Re-run /jetson-build-source."
    }
  done
fi
```

## Diff-aware copy into bsp_image

Iterate the union of overlay files (paths relative to `$LFT_SRC`)
and manifest entries (absolute `src` + relative `dst`). For each:
if `bsp_image`'s copy is byte-identical, skip. Otherwise copy
with mode / owner / timestamp preserved. Use `sudo` only for
paths under `rootfs/`, where the sample rootfs was extracted as
root and ownership must round-trip. Tag `INITRD_DIRTY=1` on any
`rootfs/lib/modules/*` or `kernel/Image` write — the
["Refresh initramfs"](../SKILL.md#refresh-initramfs-when-kernel-or-modules-changed)
step gates on this flag.

```bash
COPIED_OVERLAY=0;   IDENTICAL_OVERLAY=0
COPIED_MANIFEST=0;  IDENTICAL_MANIFEST=0
FIRST=""; LAST=""
INITRD_DIRTY=0      # set when kernel/Image or rootfs/lib/modules/* is copied

copy_one() {
  local src="$1" dst="$2" channel="$3"

  if [ -f "$dst" ] && cmp -s "$src" "$dst"; then
    case "$channel" in
      overlay)  IDENTICAL_OVERLAY=$((IDENTICAL_OVERLAY + 1));;
      manifest) IDENTICAL_MANIFEST=$((IDENTICAL_MANIFEST + 1));;
    esac
    return
  fi

  local rel="${dst#$LFT_DST/}"
  case "$rel" in
    rootfs/*)
      sudo mkdir -p "$(dirname "$dst")"
      sudo cp -p "$src" "$dst"
      ;;
    *)
      mkdir -p "$(dirname "$dst")"
      cp -p "$src" "$dst"
      ;;
  esac

  case "$rel" in
    rootfs/lib/modules/*|kernel/Image) INITRD_DIRTY=1 ;;
  esac

  case "$channel" in
    overlay)  COPIED_OVERLAY=$((COPIED_OVERLAY + 1));;
    manifest) COPIED_MANIFEST=$((COPIED_MANIFEST + 1));;
  esac
  [ -z "$FIRST" ] && FIRST="$rel"
  LAST="$rel"
}

# Channel A: overlay
for rel in "${OVERLAY_FILES[@]}"; do
  copy_one "$LFT_SRC/$rel" "$LFT_DST/$rel" overlay
done

# Channel B: manifest
for line in "${MANIFEST_ENTRIES[@]}"; do
  src=${line%%$'\t'*}
  dst_rel=${line##*$'\t'}
  copy_one "$src" "$LFT_DST/$dst_rel" manifest
done
```

Fail-fast: if any `cp` fails, surface the failed path and stop.
`bsp_image` may be left partially updated — re-running after
fixing the underlying issue (permission / disk) resumes
naturally because the loop is diff-aware and skips files
already copied.

**Channel order.** Overlay first, then manifest. If the same
`dst` appears in both (rare — an in-overlay hand-edit to a
binary that's also rebuilt), the manifest's copy wins — that's
the desired semantic, since the manifest represents the
freshly built artifact and the overlay copy is typically older
state.
