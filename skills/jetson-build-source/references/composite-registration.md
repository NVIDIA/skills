# Composite Custom-Overlay Registration

Reference material for `../SKILL.md` the "Register composite custom overlay" step.

The composite custom overlay slot is documented in
`../../../references/bsp-customization-kernel-dtb.md`;
its **content** (the per-target `.dts` file) is owned by each
customize-* skill. **Build / Makefile / flash-conf registration**
is owned by `jetson-build-source` the "Register composite custom overlay" step — this doc carries
the shell snippets and the rationale for the non-obvious choices.

## Path resolution

```bash
SOC=$(case "$CHIP_FAMILY" in t23x) echo 234 ;; t264) echo 264 ;; esac)
# CARRIER_ID_SKU is custom_carrier.carrier.{id}-{sku} when a custom
# carrier is active, otherwise reference_devkit.carrier.{id}-{sku}.
# MODULE_ID is reference_devkit.module.id.
COMPOSITE_BASE="tegra${SOC}-${CARRIER_ID_SKU}+${MODULE_ID}-xxxx-custom"
case "$CHIP_FAMILY" in
  t23x) COMPOSITE_SUBDIR="t23x/nv-public/overlay" ;;
  t264) COMPOSITE_SUBDIR="t264/nv-public/nv-platform" ;;
esac
COMPOSITE_DTS="$KS/hardware/nvidia/$COMPOSITE_SUBDIR/$COMPOSITE_BASE.dts"
COMPOSITE_MK="$KS/hardware/nvidia/$COMPOSITE_SUBDIR/Makefile"
```

If `$COMPOSITE_DTS` does not exist, skip the entire registration
pass — no kernel-DT customizations applied to this target yet.

## Makefile patch (idempotent, position-sensitive)

NVIDIA's per-dir Makefiles run a `$(addprefix $(makefile-path)/,$(dtbo-y))`
prefix-prepending pass on the literal-named `dtbo-y +=` block,
then merge in the parent dir's existing list via
`dtbo-y += $(old-dtbo)`:

```make
old-dtbo := $(dtbo-y)
dtbo-y :=
makefile-path := t264/nv-public/nv-platform
dtbo-y += tegra264-foo.dtbo     # literal-named entries (insert here)
dtbo-y += tegra264-bar.dtbo
ifneq ($(dtbo-y),)
dtbo-y := $(addprefix $(makefile-path)/,$(dtbo-y))   # prefix block
endif
dtbo-y += $(old-dtbo)           # merge-back — DO NOT insert after
```

The new entry MUST land *before* the `ifneq` prefix block, i.e.
after the **last literal-named** `dtbo-y +=` line. Inserting after
`dtbo-y += $(old-dtbo)` skips the prefix pass — `nvidia-dtbs` then
can't find the source `.dts` and the build silently drops the
composite. The regex below filters to literal-named entries
(alphanumeric first char after `+=`), excluding any `$(…)`
continuations.

```bash
line="dtbo-y += ${COMPOSITE_BASE}.dtbo"
if ! grep -qxF "$line" "$COMPOSITE_MK"; then
  LAST=$(grep -n '^dtbo-y *+= *[a-zA-Z0-9]' "$COMPOSITE_MK" | tail -1 | cut -d: -f1)
  [ -n "$LAST" ] || refuse \
    "Per-dir Makefile $COMPOSITE_MK has no literal-named 'dtbo-y +=' line — manual insertion required."
  sed -i "${LAST}a\\
$line" "$COMPOSITE_MK"
  git -C "$KS" add "${COMPOSITE_MK#$KS/}"
  git -C "$KS" commit -m "jetson-build-source: register ${COMPOSITE_BASE}.dtbo"
fi
```

## Flash-conf patch (idempotent, with first-touch pristine import)

The carrier flash conf lives in the
`<source.root_path>/Linux_for_Tegra/` overlay tracker. On a fresh
workspace this tracker is `git init` empty — the carrier flash
conf must be imported from bsp_image before any customization
commit can land on it. The workflow contract (see
`../../../../context/bsp-customization-workflow.md#commit-batching-in-the-overlay-tracker`)
requires a **pristine + customization commit pair** with a user
acceptance gate before each commit.

```bash
# Prefer custom_carrier.flash_config; fall back to reference_devkit.flash_config.
# Many user-facing flash confs in bsp_image are symlinks (e.g.
# jetson-orin-nano-devkit.conf -> p3768-0000-p3767-0000-a0.conf). Resolve
# the symlink to the real file BEFORE pristine-import: patch the real file
# name in the overlay tracker so every symlink in bsp_image that points
# at it keeps working after promote, and so the diff trail tracks one
# canonical target.
OT="<source.root_path>/Linux_for_Tegra"
UPSTREAM_LINK="<bsp_image.root_path>/Linux_for_Tegra/${ACTIVE_FLASH_CONFIG}"
[ -e "$UPSTREAM_LINK" ] || refuse \
  "Carrier flash conf $UPSTREAM_LINK not found in bsp_image."
UPSTREAM_FLASH_CONF=$(readlink -f -- "$UPSTREAM_LINK")     # real-file target
REAL_NAME=$(basename -- "$UPSTREAM_FLASH_CONF")
FLASH_CONF="$OT/$REAL_NAME"

# 3a. Pristine import on first touch.
if [ ! -f "$FLASH_CONF" ]; then
  cp -p -- "$UPSTREAM_FLASH_CONF" "$FLASH_CONF"
  # Acceptance gate: surface the imported file + diff vs HEAD before commit.
  git -C "$OT" add "$REAL_NAME"
  git -C "$OT" commit -m "pristine: import $REAL_NAME from bsp_image"
fi

# 3b. Customization commit: append the composite registration.
line='OVERLAY_DTB_FILE+=",'"${COMPOSITE_BASE}.dtbo"'"; # custom-bsp: composite'
if ! grep -qxF "$line" "$FLASH_CONF"; then
  # Append AFTER the last existing OVERLAY_DTB_FILE+= line so the
  # composite stacks after NVIDIA's *-dynamic.dtbo entry.
  LAST=$(grep -n '^OVERLAY_DTB_FILE *+=' "$FLASH_CONF" | tail -1 | cut -d: -f1)
  if [ -n "$LAST" ]; then
    sed -i "${LAST}a\\
$line" "$FLASH_CONF"
  else
    # No `+=` line yet — append at EOF (next to the initial
    # OVERLAY_DTB_FILE= assignment, if any).
    printf '%s\n' "$line" >> "$FLASH_CONF"
  fi
  # Acceptance gate again before this customization commit.
  git -C "$OT" add "$(basename "$FLASH_CONF")"
  git -C "$OT" commit -m "$(basename "$FLASH_CONF"): register ${COMPOSITE_BASE}.dtbo in OVERLAY_DTB_FILE"
fi
```

## Self-check before invoking nvidia-dtbs

```bash
grep -qxF "dtbo-y += ${COMPOSITE_BASE}.dtbo" "$COMPOSITE_MK" \
  || refuse "Composite Makefile registration missing after patch."
grep -qxF "$line" "$FLASH_CONF" \
  || refuse "Composite flash-conf registration missing after patch."
```

## Why sed-based splicing, not awk

Avoid bare `$0` in shell snippets inside SKILL.md. When this skill
is invoked with an argument (e.g. `/jetson-build-source dt`), the
harness templates `$0` against the script's own `$0` before
handing the rendered prompt to the model. An awk body like
`{ lines[NR]=$0 }` arrives with `$0` substituted away and silently
breaks. The `grep -n` + `sed Xa\\` form above sidesteps this
entirely; if an awk approach is unavoidable, route `$0` through a
named variable: `awk -v ROW="$0"`.

## Idempotency contract

Re-running this skill against an already-registered composite is a
no-op for both registration patches (both `grep -qxF` guards
return true). Safe to chain multiple times — the only side effect
is a no-op `git status`. The dirty-source-repo detection in
SKILL.md the "Detect dirty source repos" step covers the composite's parent sub-repo, so
re-runs after a customize-* append flow naturally pick up the
change.

## Cleanup pass on composite removal

The registration gate is symmetric: when `$COMPOSITE_DTS` does
**not** exist, this skill not only skips fresh registration but
also removes any stale registration left behind by an earlier run.
A customize-\* skill that retracts every fragment it owns leaves
the composite empty, and a follow-up `rm` (or `git restore`)
removes the `.dts` entirely; the next `jetson-build-source` run
must in turn strip the dangling `dtbo-y +=` line from the per-dir
Makefile and the matching `OVERLAY_DTB_FILE+=` line from the
carrier flash conf — otherwise the build / flash chain points at a
`.dtbo` that nvidia-dtbs no longer produces and `flash.sh` aborts
with a missing-file error.

```bash
if [ ! -f "$COMPOSITE_DTS" ]; then
  # Strip stale Makefile line.
  if grep -qxF "dtbo-y += ${COMPOSITE_BASE}.dtbo" "$COMPOSITE_MK"; then
    sed -i "\\|^dtbo-y += ${COMPOSITE_BASE}\\.dtbo\$|d" "$COMPOSITE_MK"
    git -C "$KS" add "$(realpath --relative-to="$KS" "$COMPOSITE_MK")"
    git -C "$KS" commit -m "$(basename "$COMPOSITE_MK"): unregister ${COMPOSITE_BASE}.dtbo (composite removed)"
  fi

  # Strip stale flash-conf line. We own the line whose trailing
  # marker is `# custom-bsp: composite`; NVIDIA-shipped entries
  # have no such marker and are never touched. Match the exact line
  # as a fixed string (the `+` in COMPOSITE_BASE would be an ERE
  # metacharacter under `grep -qE`, so `grep -qxF` + `awk` is the
  # robust idiom here).
  stale_line='OVERLAY_DTB_FILE+=",'"${COMPOSITE_BASE}.dtbo"'"; # custom-bsp: composite'
  if [ -f "$FLASH_CONF" ] && grep -qxF "$stale_line" "$FLASH_CONF"; then
    awk -v line="$stale_line" '$0 != line' "$FLASH_CONF" > "${FLASH_CONF}.tmp" \
      && mv -- "${FLASH_CONF}.tmp" "$FLASH_CONF"
    git -C "$OT" add "$REAL_NAME"
    git -C "$OT" commit -m "$REAL_NAME: unregister ${COMPOSITE_BASE}.dtbo from OVERLAY_DTB_FILE"
  fi

  # Skip the fresh-registration patches below; nothing to add when
  # there's no composite source.
  return 0 2>/dev/null || exit 0
fi
```

Marker discipline: only lines carrying the `# custom-bsp: composite`
trailer are removed. NVIDIA-shipped entries (e.g. the per-board
`*-dynamic.dtbo` registration) and any other `OVERLAY_DTB_FILE+=`
content stays untouched. The Makefile line has no equivalent
trailer; restrict its removal to the exact composite basename.
