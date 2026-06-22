---
name: jetson-build-source
description: >-
  Use when you need to rebuild the BSP overlay — DT, OOT modules,
  or kernel — from changes under bsp_sources/. Triggers: build
  bsp, rebuild dtb, rebuild kernel.
version: 0.0.1
license: "Apache-2.0"
argument-hint: "dt | oot | kernel | full"
metadata:
  data-classification: public
  author: "Jetson Team"
  team: pts
  tags:
    - bsp
    - build
  domain: meta
---

# Build BSP Source

## Purpose

Rebuild the kernel-side artifacts (DTBs, OOT modules, in-tree
modules, kernel `Image`) implied by changes under
`<source.root_path>/bsp_sources/`, and write a manifest that
`/jetson-promote-image` reads to stage those outputs into the BSP
image. The skill never writes into `<bsp_image.root_path>` itself.

## Prerequisites

- Active target-platform profile with `bsp_image:` and
  `source.toolchain:` resolved (run `/jetson-init-image` and
  `/jetson-init-source` first).
- `<source.root_path>/bsp_sources/` populated with the kernel-side
  checkout layout `/jetson-init-source` materializes.
- `<bsp_image.root_path>/Linux_for_Tegra/source/kernel_src_build_env.sh`
  present (extracted from `public_sources.tbz2`).
- Host packages: `flex`, `bison`, `libssl-dev` (hard); `git`,
  `build-essential`, `bc`, `zstd` (warn-only).
- Cross-toolchain at `${source.toolchain}gcc` resolvable on disk.

## Overview

This skill is the **Build** stage of the workflow — see
[`../../context/bsp-customization-workflow.md`](../../context/bsp-customization-workflow.md)
for where it sits in the Setup → Customize → Build → Deploy pipeline
and what triggers it. The skill takes source-side customization
commits, rebuilds the implied artifacts, and records which were
rebuilt in a manifest. Outputs stay in-tree under
`<source.root_path>/bsp_sources/`;
[`jetson-promote-image`](../jetson-promote-image/SKILL.md) reads the
manifest at Deploy to copy each rebuilt artifact into the matching
path under `<bsp_image.root_path>/Linux_for_Tegra/`.

Overlay-only edits (`nvpmodel.conf`, `nvfancontrol.conf`, BPMP DTB)
skip Build — `customize-*` stages them directly to the overlay
tracker; BPMP DTB uses the `dtc` decompile → edit → recompile loop in
[`../../references/bsp-customization-bpmp-dtb.md`](../../references/bsp-customization-bpmp-dtb.md).

**Custom-overlay slot ownership.** Kernel-DT customizations from
every customize-* skill collect into a single composite
`tegra<soc>-<carrier-id-sku>+<module-id>-xxxx-custom.dts` per
active target — see
[`../../references/bsp-customization-kernel-dtb.md`](../../references/bsp-customization-kernel-dtb.md)
for the filename / location / append protocol. This skill is the
**sole owner** of the composite's per-dir Makefile registration
(`dtbo-y += <name>.dtbo`) and the carrier flash conf's
`OVERLAY_DTB_FILE+=` line (the "Register composite custom overlay" step).

**Four build modes** matched to the dirty-repo profile:

| Mode | What's built | Auto-picks when |
|---|---|---|
| **dt** | NVIDIA DTBs only | only `hardware/nvidia/*` or `kernel-devicetree` dirty |
| **oot** | OOT modules (six repos) | only OOT repos dirty |
| **kernel** | Kernel `Image` + full in-tree `.ko` set + kernel-side dtbs | only `kernel/$KERNEL_SRC_DIR` dirty |
| **full** | Everything above + optional install consolidation | mixed dirty set |

Mode selection: **auto** (default — invoke `/jetson-build-source`
with no argument) walks the dirty-repo set; force a specific mode
by passing it as the skill argument.

**Design principle: delegate to upstream.** Every build primitive
already exists in `<bsp_image.root_path>/Linux_for_Tegra/source/`
— the env file, top-level Makefile (`nvidia-dtbs` / `modules` /
`modules_install`), kernel Makefile (`kernel` / `install`). The
skill drives those primitives against
`<source.root_path>/bsp_sources/` — never duplicates their
logic in shell.

## When to invoke

- **Auto-chained** at the end of a Customize `customize-*` invocation
  whenever Customize committed to a kernel-side source repo.
- **Manual re-run** via `/jetson-build-source [<mode>]` when:
  - the auto-chained build was interrupted,
  - source commits arrived via `git pull` from other users,
  - the user wants to force a rebuild without a fresh edit,
  - the user wants a specific mode (e.g. install consolidation for
    a manual scp deploy to a DUT).

## Instructions

### Resolve active target + paths + upstream env

Resolve the active profile per
[`../../context/target-platform-contract.md`](../../context/target-platform-contract.md).
Refuse and route in these cases:

| Condition | Route to |
|---|---|
| No active profile, or `active: NA` | `/jetson-set-target` or `/jetson-init-target` |
| Profile lacks `bsp_image:` | `/jetson-init-image` |
| Profile lacks `source.toolchain:` | `/jetson-init-source` |
| `<source.root_path>/bsp_sources/` missing or empty | `/jetson-init-source` |
| `<bsp_image.root_path>/Linux_for_Tegra/source/kernel_src_build_env.sh` missing | `/jetson-init-image` (BSP not properly extracted) |

Bind:

```bash
WORKSPACE=<parent of target-platform/>
BSP_SRC=<bsp_image.root_path>/Linux_for_Tegra/source   # NVIDIA's build primitives
KS=<source.root_path>/bsp_sources                      # our kernel-side checkout
KOUT=<source.root_path>/.build/kernel-out              # DT-mode out-of-tree build dir
STAGE=<source.root_path>/.build/install-stage          # install consolidation (full mode / opt-in)
STATE=<source.root_path>/.build-state.yaml             # per-repo watermark
MANIFEST=<source.root_path>/.build-manifest.yaml       # rebuilt-artifact list for jetson-promote-image
```

**Source the NVIDIA build env** to inherit canonical names — never
hardcode `kernel-noble`, the OOT module list, or `KERNEL_DEF_CONFIG`:

```bash
source "$BSP_SRC/kernel_src_build_env.sh"
# Now in scope: KERNEL_SRC_DIR (e.g. kernel-noble), KERNEL_DEF_CONFIG,
# OOT_SOURCE_LIST, kernel_name (e.g. noble), KERNEL_MODULAR_BUILD
```

Refuse if `$KS/kernel/$KERNEL_SRC_DIR/` is missing or if any name
in `$OOT_SOURCE_LIST` is missing under `$KS/`. Route to
`/jetson-init-source`.

### Resolve toolchain (read-only)

Read `source.toolchain` from the active profile (authored by
`jetson-init-source`). Validate:

```bash
export ARCH=arm64
export CROSS_COMPILE=<source.toolchain>   # trailing dash mandatory
[ -f "${CROSS_COMPILE}gcc" ] || refuse \
  "source.toolchain points at ${CROSS_COMPILE}gcc which does not exist. Re-run /jetson-init-source."
```

A trailing dash on `CROSS_COMPILE` is mandatory — kbuild treats it
as a prefix (`${CROSS_COMPILE}gcc`); a missing dash breaks with
`command not found`. The `[ -f ]` check catches it before any
`make` runs.

This skill **never** prompts for the toolchain or attempts to
resolve a missing one — that's `jetson-init-source`'s exclusive
responsibility. A missing field is a Setup gap; route there.

Verify build-host prerequisites once:

```bash
for p in flex bison libssl-dev; do
  dpkg -s "$p" >/dev/null 2>&1 || refuse "host package missing: $p"
done
for p in git build-essential bc zstd; do
  dpkg -s "$p" >/dev/null 2>&1 || warn "host package missing: $p"
done
```

### Detect dirty source repos

The watermark file `$STATE` records the last successfully built
commit per kernel-side repo. The repo list is derived at runtime
from `OOT_SOURCE_LIST` + `kernel/$KERNEL_SRC_DIR`. For each repo:
HEAD ≠ watermark → dirty; uncommitted edits (`git diff --quiet`
non-zero) → also dirty.

**Branch-A note**: when `bsp_sources/` is one mono-repo with a
single `.git`, every canonical sub-path shares the same HEAD —
the watermark schema still keys per-sub-path and the dirty set
still works (any change anywhere flips every sub-path's HEAD).

If `STATE` is absent (first build), treat all repos as clean
unless the auto-chain context says "Customize just committed". If
`DIRTY` is empty and no mode argument was passed: report
"nothing to build" and return.

### Pick build mode

Map the dirty set to a mode (auto), or honor the mode argument:

| Dirty repos (auto) | Mode |
|---|---|
| Only `hardware/nvidia/*` or `kernel-devicetree` | `dt` |
| Only OOT subset of `$OOT_SOURCE_LIST` | `oot` |
| Only `kernel/$KERNEL_SRC_DIR` | `kernel` |
| Any mix spanning the above | `full` |

Modes are union-able: `full` runs `kernel` → `oot` → `dt` in that
order (kernel produces headers OOT needs; `nvidia-dtbs` uses the
same generated headers). A manually passed mode argument skips
auto-detection.

### Execute build

#### Common setup + per-mode build snippets

Common setup (validate orchestrator Makefiles, `cd $KS`) and the
exact `make` invocations for each mode (`dt`, `oot`, `kernel`,
`full`) plus the optional install consolidation pass live in
[`references/build-modes.md`](references/build-modes.md). Drive
the relevant mode's snippet against the bindings from the
"Resolve active target + paths + upstream env" step.

#### Register composite custom overlay (dt + full only)

Skip unless the selected mode is `dt` or `full`. The composite
overlay slot is documented in
[`../../references/bsp-customization-kernel-dtb.md`](../../references/bsp-customization-kernel-dtb.md);
this sub-step owns the build / Makefile / flash-conf side of it.

Resolve the composite path for the active target (`$COMPOSITE_BASE`,
`$COMPOSITE_DTS`, `$COMPOSITE_MK`) using the active profile's
chip family, carrier ID/SKU, and module ID — full snippet in
[`references/composite-registration.md`](references/composite-registration.md#path-resolution).

**Gate (symmetric).** `$COMPOSITE_DTS` drives both directions:
present → apply the two idempotent patches below; absent → run the
[cleanup pass](references/composite-registration.md#cleanup-pass-on-composite-removal)
to strip any stale `dtbo-y +=` / `OVERLAY_DTB_FILE+=` line from a
prior run. Either path keeps `OVERLAY_DTB_FILE+=` from referencing
an unbuilt `.dtbo` — the build-time enforcement of the
[no-direct-in-tree-DT-edits rule](../../references/bsp-customization-kernel-dtb.md#hard-rule-no-direct-in-tree-dts--dtsi-edits).

1. **Per-dir Makefile** — append `dtbo-y += <name>.dtbo` after the
   last *literal-named* `dtbo-y +=` entry. Inserting after the
   `$(old-dtbo)` merge-back line skips the `$(addprefix
   makefile-path/,…)` prefix pass and the build silently drops
   the composite. Commit to the `bsp_sources/` mono-repo. Full
   snippet + rationale:
   [`references/composite-registration.md`](references/composite-registration.md#makefile-patch-idempotent-position-sensitive).
2. **Carrier flash conf** — append `OVERLAY_DTB_FILE+=",<name>.dtbo"`
   with first-touch pristine import on the overlay tracker. On a
   fresh workspace the tracker is empty git-init; import the
   conf from `bsp_image` and commit as `pristine:` *before* the
   customization commit (workflow contract). Full snippet:
   [`references/composite-registration.md`](references/composite-registration.md#flash-conf-patch-idempotent-with-first-touch-pristine-import).

The composite's parent sub-repo flipping HEAD during a customize-*
append is what the "Detect dirty source repos" step's dirty detection consumes — no extra
bookkeeping needed here.

**Self-check** before invoking `nvidia-dtbs`:

```bash
grep -qxF "dtbo-y += ${COMPOSITE_BASE}.dtbo" "$COMPOSITE_MK" \
  || refuse "Composite Makefile registration missing after patch."
grep -qxF "$line" "$FLASH_CONF" \
  || refuse "Composite flash-conf registration missing after patch."
```

### Write the build manifest

Walk the `DIRTY` set and emit a manifest entry per implied artifact,
following the **trace-to-dirty policy** — only artifacts traceable
to a dirty source repo. Promoting baseline-divergence noise would
attribute it to a customization's audit trail (forbidden).

The full source → kbuild → destination mapping, YAML schema, and
filter rules live in
[`references/manifest-schema.md`](references/manifest-schema.md).

Atomic write: stage to `${MANIFEST}.tmp`, then `mv -f`.

### Update watermark + summary

On success, rewrite `$STATE` with the new per-repo HEADs, toolchain,
`bsp_image.version`, and last-run mode (schema in
[`references/manifest-schema.md`](references/manifest-schema.md)).

Report:

- Toolchain (from `source.toolchain`).
- Build mode: `<dt|oot|kernel|full>` (auto-picked or forced by skill argument).
- Dirty repos and their new HEADs.
- Artifacts built: counts per kind (`.dtb`, in-tree `.ko`, OOT `.ko`, `Image`).
- Manifest path + entry count.
- Consolidated install stage path (if the "Install consolidation" step ran).
- Next step: `/jetson-promote-image`.

If a Customize skill triggered this run, prompt the user to re-issue
their original request.

## Limitations

The top tier — failure modes that block a build or silently
produce wrong artifacts. See
[`references/long-tail-gotchas.md`](references/long-tail-gotchas.md)
for invariants, deploy patterns, and performance hints.

- **Toolchain resolution is `jetson-init-source`'s job.** This
  skill only reads `source.toolchain` and exports `CROSS_COMPILE`.
  Missing field → refuse and route, never resolve in-skill.
- **R36.x Branch-A `$KS/Makefile` collision.** R36.x's
  `public_sources.tbz2` can leave the dGPU/OpenRM proprietary
  Makefile at `$KS/Makefile` instead of the Tegra orchestrator;
  its `modules` target recurses into `kernel-open/` + `src/nvidia/`,
  pulls host `/lib/modules` headers, and breaks arm64 cross-builds
  with `'-mlittle-endian' unrecognized`. R38+ extractions are
  unaffected. `/jetson-init-source`'s step 3a is the primary defense
  (extract-time); the "Common setup" check here is the safety net.
  Don't relax the regex.
- **Kernel-DT changes: composite-overlay-only, split ownership.**
  Direct edits to in-tree `.dts` / `.dtsi` files under `bsp_sources/`
  are forbidden for customize-\* skills — every kernel-DT change
  lands as a fragment in the composite overlay slot
  ([rule + rationale](../../references/bsp-customization-kernel-dtb.md#hard-rule-no-direct-in-tree-dts--dtsi-edits)).
  The composite `.dts` *content* is owned by each customize-\*
  skill; the **build / Makefile / flash-conf registration** is
  owned by this skill (gated on the composite `.dts` existing — so
  `OVERLAY_DTB_FILE+=` can't reference an unbuilt `.dtbo`).
- **the "Register composite custom overlay" step Makefile insertion point matters.** Insert after
  the last *literal-named* `dtbo-y +=` entry; inserting after
  `$(old-dtbo)` skips the `$(addprefix makefile-path/,…)` prefix
  pass and the build silently drops your `.dtbo`. The regex
  `^dtbo-y *+= *[a-zA-Z0-9]` filters correctly; do not relax it.
- **the "Register composite custom overlay" step first-touch needs pristine import.** On a fresh
  workspace the overlay tracker is empty git-init — the carrier
  flash conf is imported from `bsp_image` and committed as
  `pristine:` *before* the customization commit. Both commits go
  through the workflow acceptance gate.
- **Avoid bare `$0` in shell snippets inside this SKILL.md.** When
  invoked with an argument, the harness expands skill-body `$0`
  against the caller's `$0` before handing the rendered prompt to
  the model. Use sed-based line splicing or
  `awk -v ROW="$0"`. See
  [`references/composite-registration.md`](references/composite-registration.md#why-sed-based-splicing-not-awk).
- **No overlay staging.** Build outputs stay where the build put
  them; the manifest is the contract to `jetson-promote-image`.
  Deliberate divergence from the original overlay→promote
  indirection — keeps the full build output set out of the
  overlay tracker's git history.
- **`KERNEL_HEADERS` vs `KERNEL_OUTPUT` in DT mode.** Different
  semantics (srctree vs objtree); do not collapse. the "DT-only" step's
  snippet is correct as written.
- **OOT mode prereq: previously-built kernel headers.** A manual
  `oot` invocation against a never-built tree refuses with "run
  `kernel` (or `full`) first" rather than producing a confusing
  build error.

## Examples

Auto-detect mode from the dirty source tree (typical invocation):

```
/jetson-build-source
```

Force a single mode (skips auto-detect):

```
/jetson-build-source dt       # rebuild NVIDIA DTBs only
/jetson-build-source oot      # rebuild OOT modules only
/jetson-build-source kernel   # rebuild kernel Image + in-tree modules
/jetson-build-source full     # rebuild everything + install consolidation
```

Typical chain after a customize-* skill commits to a kernel-side
repo (the customize-* skill calls this automatically):

```
/jetson-customize-pcie ...   # commits to hardware/nvidia/.../nv-public
   ↓
/jetson-build-source         # auto-picks `dt` from the dirty set
   ↓
/jetson-promote-image        # reads .build-manifest.yaml, stages into bsp_image
   ↓
/jetson-flash-image          # flashes
```

## Troubleshooting

| Error | Cause | Solution |
|---|---|---|
| `source.toolchain points at <...>gcc which does not exist` | Toolchain field stale (path moved, install missing) | Re-run `/jetson-init-source` to re-resolve. This skill never resolves toolchain itself. |
| `No rule to make target '$KOUT/scripts/Makefile.compiler'` | `KERNEL_HEADERS` set to `$KOUT` instead of `$KS/kernel/$KERNEL_SRC_DIR` | Use the DT-mode snippet in [`references/build-modes.md`](references/build-modes.md) verbatim — srctree vs objtree must not collapse. |
| `'-mlittle-endian' unrecognized` during `make modules` | R36.x Branch-A `$KS/Makefile` collision — dGPU/OpenRM Makefile in place of Tegra orchestrator | The Common-setup safety net normally repairs it; if not, `git checkout HEAD -- Makefile` then re-run `/jetson-init-source` step 3a. |
| `run kernel (or full) first` on manual `oot` invocation | Kernel source tree never prepared | Run `/jetson-build-source kernel` (or `full`) once, then `oot`. |
| `nothing to build` and dirty edits exist | Edits uncommitted in a sub-repo but `.build-state.yaml` watermark already matches HEAD | Commit the edits, or re-run with an explicit mode argument (`/jetson-build-source dt` etc.). |
| Composite `.dtbo` silently missing from the output set | Per-dir Makefile insertion landed after `$(old-dtbo)` merge-back line | See [`references/composite-registration.md`](references/composite-registration.md#makefile-patch-idempotent-position-sensitive) — insert after the last *literal-named* `dtbo-y +=` entry. |
| Promote step copies stale baseline artifacts | Trace-to-dirty filter skipped after a manual `cp` into `$KS` | Only edit via a customize-* skill or `git`; the dirty detector keys on git HEAD, not file mtime. |
| `host package missing: <pkg>` | `flex`/`bison`/`libssl-dev` refuse; others warn. | `sudo apt install <pkg>` per [`references/upstream-recipe.md`](references/upstream-recipe.md). |

## See also

- [`references/build-modes.md`](references/build-modes.md) — per-mode `make` snippets + install consolidation.
- [`references/manifest-schema.md`](references/manifest-schema.md) — `.build-manifest.yaml` + `.build-state.yaml` schemas.
- [`references/composite-registration.md`](references/composite-registration.md) — the "Register composite custom overlay" step full snippets + rationale.
- [`references/long-tail-gotchas.md`](references/long-tail-gotchas.md) — invariants, deploy patterns, performance hints.
- [`references/upstream-recipe.md`](references/upstream-recipe.md) — verbatim NVIDIA recipe, divergences, spec status.
- [`../../context/bsp-customization-workflow.md`](../../context/bsp-customization-workflow.md#workflow-invariants) — workspace edit protocol.
- [`../../references/bsp-customization-kernel-dtb.md`](../../references/bsp-customization-kernel-dtb.md) — composite custom-overlay contract.
- [`../../references/bsp-customization-bpmp-dtb.md`](../../references/bsp-customization-bpmp-dtb.md) — BPMP-DTB edit contract (routed around this skill).
- [`../jetson-init-source/SKILL.md`](../jetson-init-source/SKILL.md) — Setup; authors `source.toolchain`.
- [`../jetson-promote-image/SKILL.md`](../jetson-promote-image/SKILL.md) — Deploy promoter; reads this skill's manifest.
