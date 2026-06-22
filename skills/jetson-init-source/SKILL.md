---
name: jetson-init-source
description: >-
  Set up the BSP source workspace: Linux_for_Tegra overlay tracker,
  bsp_sources, Crosstool-NG toolchain. Use after jetson-init-image;
  not for fetching inputs.
version: 0.0.1
license: "Apache-2.0"
metadata:
  data-classification: public
  author: "Jetson Team"
  tags:
    - bsp
    - workspace
    - kernel
    - bootstrap
  domain: meta
---

# Initialize BSP Customization Workspace

## Overview

This skill bootstraps the source-side workspace that customize-* / build
skills depend on: the `Linux_for_Tegra` overlay tracker (git repo for
pristine + customization commits), the `bsp_sources/` mono-tree (kernel,
OOT, nvgpu, display, hwpm, hardware DTs), and a working NVIDIA
Crosstool-NG cross-compile prefix. It owns only the `source:` block in
the active profile and the on-disk source workspace under
`<source.root_path>` (default: `<workspace>/Source`).

Responsibilities:

1. Optionally record a non-default `source.root_path`.
2. Create or mount the `Linux_for_Tegra` overlay tracker.
3. Materialize `bsp_sources` using the precedence in the "Materialize the BSP-sources baseline" step.
4. Resolve and record `source.toolchain`.
5. Clone extra user-defined repos from `source.repos:`.

## When to invoke

- The user asks to bootstrap, init, or sync the BSP customization
  workspace.
- A downstream customization skill refused with "no workspace
  tracker at `<source.root_path>/Linux_for_Tegra/`".
- After `jetson-init-image` (Setup's next step on a fresh target).

## Procedure

### Quick-start prefill mapping

Follow the shared
[`quick_start_prefill` contract](../../context/bsp-customization-workflow.md#quick_start_prefill-contract).
This skill has source-specific mappings:

- `quick_start_prefill.source.public_sources_archive` maps to the
  Branch-A archive candidate.
- `quick_start_prefill.source.repos` maps to proposed `source.repos:`
  entries; validate reserved keys and `url:` / `archive:` mutual
  exclusion before writing.
- `quick_start_prefill.source.toolchain` may be a cross-compile prefix,
  a `gcc` path, a containing `bin/` directory, an `x-tools.tbz2` archive
  path, or `skip`.

This skill remains the only owner of `source.root_path`, `source.repos:`,
and `source.toolchain` profile writes.

### Resolve the active target + paths

Resolve the active profile + workspace defaults per the contract in
[`../../context/target-platform-contract.md`](../../context/target-platform-contract.md).

- **Refuse** if `<bsp_image.root_path>` does not contain
  `Linux_for_Tegra/` (BSP not extracted — route to `/jetson-init-image`).
- If the profile has `source.root_path:`, use it. Otherwise
  `<source.root_path>` defaults to `<workspace>/Source`; use that default
  silently and do not write `source.root_path:` to the profile. Ask only
  for an explicit custom path, unrelated content at the default path, or an
  unwritable parent.

Read `source.repos:` (if present) into a map keyed by entry name,
each carrying optional `url`, `ref`, `subdir`, `path`. Reserved keys:
`Linux_for_Tegra` (overlay tracker), `bsp_sources` (kernel-source
repo). Every other key is an extra user-defined repo.

#### (Optional) prompt for `source.root_path` override

Only when `source.root_path` is absent from the profile **and** one of the
override conditions above applies:

> `source.root_path`: default = `<workspace>/Source`. Press Enter to
> accept, or enter an absolute path to override.

- **On Enter** — keep the default; do not touch the profile.
- **On override path** — validate the closest existing parent is
  writable; refuse and re-prompt if not. Edit
  `target-platform/<active>.yaml` in place to add/update
  `source.root_path:`. Preserve all other blocks, comments, and
  quoting — use a round-tripping YAML loader (e.g. `ruamel.yaml`).

Fires at most once per profile. Otherwise create `<workspace>/Source` as
needed and continue without prompting.

### Materialize `Linux_for_Tegra`

Mount path is canonical: `<source.root_path>/Linux_for_Tegra/`.

**Default** (no `source.repos.Linux_for_Tegra` entry):

```bash
LFT="<source.root_path>/Linux_for_Tegra"
mkdir -p "$LFT"
[ -d "$LFT/.git" ] || git -C "$LFT" init
```

Empty tracker. **Do not commit anything here** — pristine imports
happen file-by-file when customization skills run.

**Override** (`url`, `ref`, optional `subdir`):

```bash
# Clone the user's repo to a side location, then mount the
# expected tree (subdir or repo root) at the canonical path.
CLONE="<source.root_path>/.repos/Linux_for_Tegra"
git clone <url> -b <ref> "$CLONE"
ln -s "$CLONE/<subdir or .>" "<source.root_path>/Linux_for_Tegra"
```

If the mount already exists with valid git state, skip; refuse if it
exists with unrelated content.

### Materialize the BSP-sources baseline

Three branches, dispatched **in precedence order** against the
profile entry `source.repos.bsp_sources`:

| Order | Profile state | Branch |
|---|---|---|
| 1 | `url:` set | **C. Customer git clone** (explicit override always wins) |
| 2 | `archive:` set, **OR** entry absent **AND** `<workspace>/Downloads/public_sources.tbz2` exists | **A. Local archive extraction** (default) |
| 3 | Entry absent **AND** no local archive | **B. `source_sync.sh`** (fallback) |

`url:` and `archive:` are **mutually exclusive** — refuse if both are
set in the same entry.

Branch A is the preferred default because it sidesteps NVIDIA git
egress entirely (the most common Setup failure mode). Branch B
exists for fresh workspaces with no pre-downloaded tarball.
Branch C is for customer forks of the whole BSP layout.

#### Branch A — Local archive extraction (default)

Default branch: extract a pre-downloaded `public_sources.tbz2` into
`<source.root_path>/bsp_sources/` as a single mono-repo (`git init` +
pristine commit). See
[`references/branch-a-extraction.md`](references/branch-a-extraction.md)
for the full archive shape, path-resolution rules, and the extraction
script (including the Tegra OOT Makefile force-replace workaround for
R36.x).

Branches B and C may produce per-component repos instead; downstream
build logic still walks the canonical sub-paths under
`<source.root_path>/bsp_sources/`.

#### Branch B — `source_sync.sh` (fallback)

Runs only when no local archive is found and no `url:` is set.
Create the `bsp_sources/` mount directory under `<source.root_path>`
and run `source_sync.sh` from the extracted BSP with two flags:

```bash
mkdir -p "<source.root_path>/bsp_sources"
bash "<bsp_image.root_path>/Linux_for_Tegra/source/source_sync.sh" \
     -d "<source.root_path>/bsp_sources" \
     -t "jetson_<major.minor>"
```

- `-d <source.root_path>/bsp_sources` — write clones into the
  `bsp_sources/` subdir of the workspace, so the on-disk folder
  matches the schema key. Without `-d`, the script writes under its
  own directory (the BSP itself) — wrong for the overlay model.
- `-t jetson_<major.minor>` — pin the tag to the BSP release line.
  Derive from `bsp_image.version` by truncating to the first two
  dotted components: `"38.4.0"` → `jetson_38.4`. **Tag-format
  fallback**: if rejected, try `jetson_<bsp_image.version>` (older
  L4T sometimes uses the full form). If that also fails, surface the
  error and stop — never fall back to "latest" silently.

Refuse if `source_sync.sh` does not exist: re-run `/jetson-init-image`
to repopulate `Linux_for_Tegra/source/`.

`source_sync.sh` exits 0 even when every clone failed — verify by
counting `Failed to clone` lines in its output and refuse if
non-zero. The most likely cause of universal failure is **blocked
git egress** to `gitlab.com/nvidia/nv-tegra` /
`nv-tegra.nvidia.com`; surface that explicitly and route the user
to download `public_sources.tbz2` via `/quick-start` for Branch A.

#### Branch C — Customer git clone (`url:` override)

Triggered by an explicit `url:` field. Clone the customer repo once
and expose its canonical kernel-side sub-paths under
`<source.root_path>/bsp_sources/`. The canonical sub-path list is
**read from source_sync.sh's `SOURCE_INFO` at runtime** — do not
hard-code it, so future NVIDIA additions/removals propagate
automatically:

```bash
# Parse canonical sub-paths from source_sync.sh's SOURCE_INFO
# (only the kernel-side entries marked `k:` in the second field).
SUBPATHS=$(grep -oP '^\s*k:[^:]+:' \
  "<bsp_image.root_path>/Linux_for_Tegra/source/source_sync.sh" \
  | sed 's/^\s*k://; s/:$//')

mkdir -p "<source.root_path>/bsp_sources"
CLONE="<source.root_path>/.repos/bsp_sources"
git clone <url> -b <ref> "$CLONE"
ROOT="$CLONE/<subdir or .>"
for SUB in $SUBPATHS; do
  [ -d "$ROOT/$SUB" ] && \
    ln -s "$ROOT/$SUB" "<source.root_path>/bsp_sources/$SUB"
done
```

Report any canonical sub-path expected for the active chip family
but not present inside the customer repo (warn, don't refuse —
customer may legitimately not have all repos).

### Resolve cross-compile toolchain

The downstream `jetson-build-source` reads `source.toolchain` from
this profile and exports it as `CROSS_COMPILE`. This step **must**
land a valid prefix before init-source returns, or any subsequent
kernel / OOT / DT build will refuse.

NVIDIA's official **Crosstool-NG Toolchain gcc** is the canonical
toolchain for L4T. `jetson-download-bsp` owns any network fetch of
`x-tools.tbz2`; this skill only discovers, extracts, validates, and
writes the resolved prefix. Resolution follows a three-step ladder:

#### Auto-discover

Look under `<workspace>/toolchain/x-tools/` for the Crosstool-NG
layout — typically one of:

```
<workspace>/toolchain/x-tools/aarch64-none-linux-gnu/bin/aarch64-none-linux-gnu-gcc
<workspace>/toolchain/x-tools/aarch64-buildroot-linux-gnu/bin/aarch64-buildroot-linux-gnu-gcc
```

Glob: `<workspace>/toolchain/x-tools/aarch64-*-linux-gnu/bin/aarch64-*-linux-gnu-gcc`.
If exactly one match, bind:

```bash
TC_PREFIX=<absolute path to that .../bin/<triple>->   # trailing dash mandatory
```

Skip to "Write to profile" below. If zero matches, fall through
to "Auto-extract from `Downloads/x-tools.tbz2`" below. If multiple, refuse with the list and ask the user to
remove the unwanted ones (we never pick one silently among
ambiguous installs — different Crosstool-NG flavors produce ABI-
incompatible binaries).

#### Auto-extract from `Downloads/x-tools.tbz2`

If `<workspace>/Downloads/x-tools.tbz2` exists (mirrors the
`public_sources.tbz2` Branch-A pattern in the "Materialize the BSP-sources baseline" step — air-gapped /
no-egress users drop archives there):

```bash
file -b "<workspace>/Downloads/x-tools.tbz2" | grep -q "bzip2 compressed" || \
  refuse "<workspace>/Downloads/x-tools.tbz2 is not a bzip2 tarball"
mkdir -p "<workspace>/toolchain"
tar xjf "<workspace>/Downloads/x-tools.tbz2" -C "<workspace>/toolchain"
```

Then re-run the "Auto-discover" pass above. Refuse if extraction succeeds
but no `x-tools/aarch64-*-linux-gnu/bin/` is produced (archive
content doesn't match the Crosstool-NG layout).

#### Prompt the user

If both Auto-discover and Auto-extract came up empty, ask:

> No Crosstool-NG toolchain found at `<workspace>/toolchain/` or in
> `<workspace>/Downloads/x-tools.tbz2`.
>
> Reply with one of:
>   - absolute path to your `aarch64-*-linux-gnu-gcc` binary or its
>     containing `bin/` directory,
>   - `cancel` to abort.
>
> To fetch the archive instead, cancel this run, run
> `/jetson-download-bsp`, then re-run `/jetson-init-source`.

For a path reply, validate via `[ -f "${TC_PREFIX}gcc" ]`. Refuse
and re-prompt on failure.

#### Write to profile

Once `$TC_PREFIX` resolves and `${TC_PREFIX}gcc` exists, write it
into the active profile using a round-tripping YAML loader:

```yaml
source:
  toolchain: <TC_PREFIX>   # absolute, with trailing dash
```

If `source:` is otherwise empty (no `root_path` override, no
`repos:` entries), the `source:` block is now non-empty and stays
in the profile. Future `jetson-init-source` runs **skip the "Resolve cross-compile toolchain" step**
if `source.toolchain` is already set and points at a working `gcc`.

### Clone extra user-defined repos

For each entry under `source.repos:` whose name is not
`Linux_for_Tegra` or `bsp_sources`:

```bash
MOUNT="<source.root_path>/<entry.path or entry.name>"
if [ -n "<entry.subdir>" ]; then
  CLONE="<source.root_path>/.repos/<entry.name>"
  git clone <entry.url> -b <entry.ref> "$CLONE"
  ln -s "$CLONE/<entry.subdir>" "$MOUNT"
else
  git clone <entry.url> -b <entry.ref> "$MOUNT"
fi
```

Refuse if a mount path already exists with unrelated content.

### Summary

Print:

- Resolved `<workspace>`, `<bsp_image.root_path>`, and
  `<source.root_path>`.
- For each materialized component: created, reused, skipped, or refused.
- For `bsp_sources`: branch selected plus key evidence (archive path,
  `source_sync.sh` failure count, or clone URL/ref).
- Toolchain prefix, resolution source, and `${TC_PREFIX}gcc --version`
  first line.
- Reminder that customize-* skills stage future BSP edits in
  `<source.root_path>/Linux_for_Tegra/`; promote is what later copies
  committed overlay changes into `bsp_image`.

If a downstream skill triggered this run, tell the user to re-issue
their original request.

## Gotchas

- `Linux_for_Tegra` and `bsp_sources` mount paths are canonical.
  `path:` applies only to extra user-defined repos.
- The default `Linux_for_Tegra` tracker is intentionally empty; do not
  pre-populate it.
- `bsp_sources` precedence is `url:` → `archive:` → auto-discovered
  `Downloads/public_sources.tbz2` → `source_sync.sh`. `url:` and
  `archive:` are mutually exclusive.
- Branch A auto-discovery does not prompt and is not written back to the
  profile. Persist it only with `source.repos.bsp_sources.archive:`.
- **Branch A `$DEST/Makefile` collision.** Inner tarballs in
  `public_sources.tbz2` ship two files named `Makefile`: the Tegra
  orchestrator (`kernel_oot_modules_src.tbz2`) and the dGPU/OpenRM
  proprietary Makefile (`nvidia_kernel_display_driver_source_without_
  root_dir.tbz2`). Alphabetical extraction order lets the dGPU one
  win on R36.x; downstream arm64 cross-builds then fail with
  `'-mlittle-endian' unrecognized`. Step 3a force-replaces from
  `<bsp_image>/Linux_for_Tegra/source/Makefile` when the Tegra
  `modules: hwpm nvidia-oot nvgpu nvidia-display` signature is
  missing. R38+ extractions already match; the check is a no-op there.
- Branch C customer repos must expose the canonical `source_sync.sh`
  sub-path layout, optionally shifted by `subdir:`.
- Derive the `source_sync.sh` tag from `bsp_image.version` as
  `jetson_<major.minor>` first; never fall back to an unpinned latest.
- `jetson-download-bsp` owns network downloads of `public_sources.tbz2`
  and `x-tools.tbz2`; this skill consumes local archives only.
- `source.toolchain` must be an NVIDIA Crosstool-NG prefix with trailing
  dash and a working `${prefix}gcc`. Never silently use `$PATH`.
- Use a round-tripping YAML writer for profile edits.

## Prerequisites

- Active target profile resolved per
  `../../context/target-platform-contract.md`.
- `/jetson-init-image` already run so `bsp_image.version` is recorded
  (Branch B `source_sync.sh` tag derives from it).
- For Branch A: a local `public_sources.tbz2` (and optionally
  `x-tools.tbz2`) staged under `Downloads/`.
- For Branch C: customer Git access to the override repo URL.

## Limitations

- Owns only the `source:` block; never edits `bsp_image`,
  `reference_devkit`, `custom_carrier`, or `documents`.
- Network egress only for Branch B (`source_sync.sh`) and Branch C
  (customer Git clone); Branch A is fully offline.
- Refuses to silently substitute a system toolchain — the NVIDIA
  Crosstool-NG prefix must be present or extractable.

## Troubleshooting

- **`${toolchain}gcc` not found** — re-stage `x-tools.tbz2` under
  `Downloads/` and rerun, or pass a verified absolute prefix path.
- **`source_sync.sh` cannot resolve `jetson_<major.minor>` tag** — the
  recorded `bsp_image.version` is wrong; re-run `/jetson-init-image`
  to refresh it.
- **`Linux_for_Tegra/.git` shows uncommitted hand-edits** — abort and
  ask the user to commit or stash; this skill expects a clean tracker.
- **Branch C clone missing canonical sub-paths** — repo layout doesn't
  match `source_sync.sh`; set `subdir:` to the right sub-root or fall
  back to multi-repo overrides under `source.repos:`.

## References

- [`../../references/platform_template.yaml`](../../references/platform_template.yaml) — `source:` schema, including the `repos:` map.
- [`../../context/target-platform-contract.md`](../../context/target-platform-contract.md) — target-platform contract.
- [`../../context/bsp-customization-workflow.md`](../../context/bsp-customization-workflow.md#workflow-invariants) — Workspace edit protocol.
- [`../jetson-init-target/SKILL.md`](../jetson-init-target/SKILL.md) — authors the profile this skill consumes.
- [`../jetson-init-image/SKILL.md`](../jetson-init-image/SKILL.md) — extracts the BSP and back-fills `bsp_image.version`; run before this skill.
