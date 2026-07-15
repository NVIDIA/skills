---
name: jetson-init-image
description: >-
  Extract Jetson Linux + sample-rootfs tarballs and run
  apply_binaries.sh for the active target, then record bsp_image in
  the profile. Use after jetson-init-target; not for source-tree
  setup.
version: 0.0.1
license: "Apache-2.0"
metadata:
  data-classification: public
  author: "Jetson Team"
  tags:
    - bsp
    - image
    - bootstrap
  domain: meta
---

# Initialize BSP Image

Output is only `bsp_image:` in the active profile: derived `version`
plus `root_path` only when overriding `<workspace>/Image`.

## When to invoke

- The user asks to extract, prepare, or initialize the BSP image.
- A downstream skill reports missing
  `<bsp_image.root_path>/Linux_for_Tegra/`.
- The active profile has no `bsp_image:` block yet.

## Procedure

### Resolve target and image path

Resolve the active profile per
[`../../context/target-platform-contract.md`](../../context/target-platform-contract.md).
Refuse if there is no active profile or `reference_devkit:` is missing.

`<workspace>` is the parent of the active profile's `target-platform/`
directory. `<bsp_image.root_path>` defaults to `<workspace>/Image`.

| Profile state | Action |
|---|---|
| `bsp_image.root_path` exists | Use it without prompting. |
| `bsp_image:` exists without `root_path` | Use `<workspace>/Image`. |
| No `bsp_image:` block | Ask once: Enter for default, or absolute override path. |

For an override, validate that the closest existing parent is writable.
Omit `root_path` when the default is used.

### Determine GPU stack

Use the shared GPU-driver invariant from
[`../../context/target-platform-contract.md`](../../context/target-platform-contract.md#derived-platform-facts).
Derive the expected stack from `reference_devkit.module.id` and the
catalogue:

| Chip family | Module IDs | Stack | `apply_binaries.sh` flag |
|---|---|---|---|
| T234 / Orin | `p3701`, `p3767` | nvgpu | none |
| T264 and later / Thor+ | `p3834` | OpenRM | `--openrm` |

Refuse unknown module IDs. The `--openrm` flag is only valid on BSP
releases that ship the OpenRM stack; if the active BSP doesn't expose
the flag, omit it regardless of what the target wants.

### Reuse or extract

If `<bsp_image.root_path>/Linux_for_Tegra/` already exists:

1. Do not extract over it unless the user explicitly requested
   re-extraction and accepted the overwrite risk.
2. Derive the on-disk version from `Linux_for_Tegra/nv_tegra_release`.
   Ask before replacing a different recorded `bsp_image.version`.
3. Verify the installed GPU stack against the platform-derived
   expectation when possible. Detection precedence (first probe that
   yields a definitive answer wins):

   1. `Linux_for_Tegra/rootfs/etc/nv_tegra_release` carries an
      `INSTALL_TYPE=` token on BSP releases that expose it (newer
      lines). Read and compare directly.
   2. Otherwise, `find Linux_for_Tegra -name nvgpu.ko`: present →
      nvgpu, absent → OpenRM.
   3. If the chip family has only ever shipped one stack (e.g. T234 /
      Orin is always nvgpu in current BSPs), fall back to the
      catalogue-derived expectation without disk probing.

If the installed stack conflicts with the active target, refuse and ask
the user to re-extract with the correct stack or fix the target profile.
Otherwise skip extraction and update the profile.

### Locate tarballs

When extraction is needed, search:

1. `<bsp_image.root_path>/`
2. `<workspace>/`
3. current working directory

Prompt for absolute paths for anything missing. Required filenames:

- `Jetson_Linux_R<ver>_aarch64.tbz2`
- `Tegra_Linux_Sample-Root-Filesystem_R<ver>_aarch64.tbz2`

Both filenames must contain the same `R<ver>` token. Refuse mismatches
and record `<ver>` as `bsp_image.version`. Do not download tarballs.

### Extract and apply binaries

Use absolute tarball paths; they may live outside
`<bsp_image.root_path>`.

```bash
ROOT="<bsp_image.root_path>"
BSP_TARBALL="<absolute path to Jetson_Linux_R<ver>_aarch64.tbz2>"
ROOTFS_TARBALL="<absolute path to Tegra_Linux_Sample-Root-Filesystem_R<ver>_aarch64.tbz2>"

mkdir -p "$ROOT"
tar xjf "$BSP_TARBALL" -C "$ROOT"
sudo tar xpjf "$ROOTFS_TARBALL" -C "$ROOT/Linux_for_Tegra/rootfs"

cd "$ROOT/Linux_for_Tegra"
if [ "$GPU_STACK" = "openrm" ]; then
  sudo ./apply_binaries.sh --openrm
else
  sudo ./apply_binaries.sh
fi
```

Set `GPU_STACK` from the "Determine GPU stack" step above. Abort on the first failing command and
surface the failed command.

### Update the active profile

Persist the resolved BSP image metadata in the active target profile so
later skills can find the BSP without re-prompting. Preserve existing
blocks, comments, and quoted SKU values; use a round-tripping YAML
writer such as `ruamel.yaml`.

```yaml
bsp_image:
  root_path: <absolute override path>  # omit for <workspace>/Image
  version: "<derived version>"
```

Rules:

- Same version and same root path: no rewrite.
- Different version: ask before updating.
- Different recorded `root_path`: refuse automatic rewrite.
- Always quote `version`.

## Finish

Report the image path, extracted vs reused state, GPU stack, derived
version, and profile update status. Then suggest `/jetson-init-source`.

## Purpose

Materialize `Linux_for_Tegra/` on disk by extracting the right Jetson
Linux + sample-rootfs tarballs and running `apply_binaries.sh` with
the GPU-stack flag derived from the active target (nvgpu for T234,
OpenRM for T264+). Then commit the derived BSP version into the
profile's `bsp_image:` block.

## Prerequisites

- Active target profile resolved per
  `../../context/target-platform-contract.md`.
- Jetson Linux BSP tarball and matching sample-rootfs tarball staged
  on disk (e.g. by `/jetson-download-bsp` or hand-placed).
- Write access to the workspace `Image/` root (or the override
  `bsp_image.root_path`).
- `sudo` available for `apply_binaries.sh`.

## Limitations

- Writes only the `bsp_image:` block; source tree, documents, and
  carrier profile are owned by sibling skills.
- Refuses to overwrite an existing `Linux_for_Tegra/` without explicit
  user direction.
- Does not download tarballs; rely on `/jetson-download-bsp` or
  hand-stage the inputs.

## Troubleshooting

- **`apply_binaries.sh` exits non-zero** — re-read its console output;
  most failures are missing `sudo`, missing rootfs tarball, or wrong
  GPU stack flag for the SoC generation.
- **`nv_tegra_release` absent after extract** — extraction stopped
  early; verify tarball integrity and rerun.
- **Recorded `version` disagrees with the tarball filename** — the
  tarball was renamed; trust the value parsed from
  `Linux_for_Tegra/nv_tegra_release` over filenames.
- **Different recorded `root_path` already in profile** — refuse and
  ask the user to confirm before overwriting.

## References

- [`../../context/target-platform-contract.md`](../../context/target-platform-contract.md)
- [`../../references/bsp-platforms-catalogue.md`](../../references/bsp-platforms-catalogue.md)
- [`../../references/platform_template.yaml`](../../references/platform_template.yaml)
- [`../jetson-init-target/SKILL.md`](../jetson-init-target/SKILL.md)
- [`../jetson-init-source/SKILL.md`](../jetson-init-source/SKILL.md)
- [`../jetson-flash-image/SKILL.md`](../jetson-flash-image/SKILL.md)
