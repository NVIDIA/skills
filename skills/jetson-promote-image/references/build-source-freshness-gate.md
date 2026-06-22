# Build-source freshness gate

Reference material for `../SKILL.md` the "Verify build-source
freshness" step.

The gate enforces: every customize-* commit that lands in a
kernel-side source repo (composite custom overlay `.dts`, kernel
sources, OOT module sources, per-carrier overlay `.dts`) must have
been processed by `/jetson-build-source` before
`/jetson-promote-image` runs. Without the gate, a kernel-side
customize-* commit can sit in `Source/bsp_sources/<sub-repo>/`
indefinitely while `/jetson-promote-image` keeps promoting only
the **previous** build's `.build-manifest.yaml` — the DUT then
boots with stale `.dtbo` / `.ko` / `Image` that don't reflect the
current source tree, and the failure mode is silent.

## Detection contract

The check mirrors `/jetson-build-source`'s own "Detect dirty source
repos" step. A sub-repo is **dirty** if any of the following holds:

1. Uncommitted or staged edits (`git diff --quiet` or
   `git diff --cached --quiet` non-zero).
2. `.build-state.yaml` exists, lists this sub-repo, and current
   HEAD differs from the recorded watermark.
3. `.build-state.yaml` does **not** exist, the sub-repo's `.git`
   is per-sub-repo (Branch-B layout), and `git rev-list --count HEAD`
   is greater than 1 — i.e. customize-* commits sit on top of the
   pristine-init commit but `/jetson-build-source` has never run.

If any sub-repo is dirty, refuse with the list. The user runs
`/jetson-build-source`, which rewrites `.build-state.yaml` with the
new watermarks and the `.build-manifest.yaml` with the rebuilt
artifacts, and re-runs `/jetson-promote-image`.

## Path resolution

```bash
STATE="<source.root_path>/.build-state.yaml"
BSP_SRC="<source.root_path>/Source/bsp_sources"
```

## Shell snippet

```bash
DIRTY=()

# Discover kernel-side sub-repos. Per-sub-repo `.git` directories
# under bsp_sources/ are the Branch-B layout (one `.git` per
# canonical kernel-side path). Branch-A has a single top-level
# `.git` at bsp_sources/ — fall back to that when no nested
# `.git` directories are present.
REPOS=()
while IFS= read -r gitdir; do
  REPOS+=("${gitdir%/.git}")
done < <(find "$BSP_SRC" -mindepth 2 -maxdepth 5 -type d -name '.git' -prune 2>/dev/null)
BRANCH_A=0
if [ "${#REPOS[@]}" -eq 0 ] && [ -e "$BSP_SRC/.git" ]; then
  REPOS=("$BSP_SRC")
  BRANCH_A=1
fi

for repo in "${REPOS[@]}"; do
  rel="${repo#$BSP_SRC/}"
  [ "$rel" = "$repo" ] && rel="(bsp_sources)"

  # Rule 1: uncommitted or staged edits.
  if ! git -C "$repo" diff --quiet 2>/dev/null \
     || ! git -C "$repo" diff --cached --quiet 2>/dev/null; then
    DIRTY+=("$rel (uncommitted)")
    continue
  fi

  if ! cur=$(git -C "$repo" rev-parse HEAD 2>/dev/null); then
    DIRTY+=("$rel (cannot resolve HEAD)")
    continue
  fi

  if [ -f "$STATE" ]; then
    # Rule 2: HEAD must match watermark for every recorded repo.
    wm=$(yq -r ".repos[\"$rel\"] // \"\"" "$STATE")
    if [ -n "$wm" ] && [ "$cur" != "$wm" ]; then
      DIRTY+=("$rel (HEAD $cur != watermark $wm)")
    fi
  else
    # Rule 3 (Branch-B only): no .build-state.yaml + commits past
    # pristine-init = /jetson-build-source has never run for this
    # customize-* edit. Branch-A mono-repo's HEAD includes upstream
    # history, so this proxy doesn't apply there — the user is
    # expected to run /jetson-build-source at least once after
    # bootstrap to materialize .build-state.yaml.
    if [ "$BRANCH_A" = "0" ] \
       && [ "$(git -C "$repo" rev-list --count HEAD)" -gt 1 ]; then
      DIRTY+=("$rel (no /jetson-build-source run)")
    fi
  fi
done

if [ "${#DIRTY[@]}" -gt 0 ]; then
  refuse "Kernel-side source(s) changed since last /jetson-build-source — \
build outputs are stale. Run /jetson-build-source first.
Dirty: ${DIRTY[*]}"
fi
BUILD_FRESH=1
```

## Why this gate is in promote, not flash or validate

The two-channel abstraction (overlay tracker + build manifest, see
`../SKILL.md` the "Overview" section) routes every build output
through `.build-manifest.yaml`. `/jetson-promote-image` is the only
skill that *consumes* the manifest into `bsp_image`; once the copy
pass completes, `/jetson-flash-image` and
`/jetson-validate-image` operate on `bsp_image` only and have no
view into `Source/bsp_sources/`. Gating at promote breaks the chain
before any stale artifact lands in `bsp_image`, so the downstream
skills don't need their own copy of this check.

## Idempotency

Re-running `/jetson-promote-image` after a clean
`/jetson-build-source` pass is a no-op for this gate: every
sub-repo's HEAD matches its fresh watermark, no uncommitted edits
exist, `BUILD_FRESH=1` is recorded, the step returns. Safe to
chain in CI or autopilot flows.
