---
name: jetson-print-bsp-info
description: Use when you need to print Jetson BSP info (L4T version, board configs, rootfs state) from a Linux_for_Tegra root on the host PC. This is an example skill.
version: 0.0.1
license: "Apache-2.0"
metadata:
  author: "Jetson Team"
  tags: [jetson, bsp, info]
  languages: [bash]
  data-classification: public
---

# jetson-print-bsp-info

Prints a concise summary of a Jetson Linux_for_Tegra (BSP) tree on the host PC.

This skill is intended as a reference example for the `jetson-bsp-skills` repo and the NVIDIA-wide skills CI. It performs read-only inspection — no flashing, no rootfs changes.

## Purpose

Capture a baseline snapshot of a Linux_for_Tegra BSP tree (release, board configs, rootfs state) before flashing, so issues like "wrong L4T version" or "rootfs never populated" are caught early.

## When to use

- A user has unpacked a Jetson BSP tarball and wants to confirm the L4T version, supported boards, and rootfs state before flashing.
- You need a quick sanity check that a `Linux_for_Tegra/` directory looks valid (expected scripts and config files present).

## Prerequisites

- Running on the host PC (Linux), not on the Jetson target.
- A `Linux_for_Tegra/` directory extracted from a Jetson BSP tarball.
- Standard CLIs available: `ls`, `head`, `cat`, `paste`, `sed`.

## Inputs

- `L4T_ROOT` (optional): absolute path to the `Linux_for_Tegra/` directory. If unset, use the current working directory.

## Instructions

Run each step in order and print the captured values into the report shown under [Output format](#output-format).

1. **Resolve** `L4T_ROOT` and **validate** the directory is a Linux_for_Tegra root — exit early otherwise. `flash.sh` and `nv_tegra/` are the two anchor artifacts that every BSP ships:
   ```bash
   L4T_ROOT="${L4T_ROOT:-$PWD}"
   if [ ! -f "$L4T_ROOT/flash.sh" ] || [ ! -d "$L4T_ROOT/nv_tegra" ]; then
     echo "Not a Linux_for_Tegra root: '$L4T_ROOT' (missing flash.sh or nv_tegra/)"
     exit 1
   fi
   echo "$L4T_ROOT"
   ```
2. **Extract** the L4T release header line. The canonical host-side location is `nv_tegra/nv_tegra_release`; the same file is copied into the rootfs by `apply_binaries.sh`. Only the first line is useful — the rest is a long list of library SHAs:
   ```bash
   head -1 "$L4T_ROOT/nv_tegra/nv_tegra_release" 2>/dev/null \
     || head -1 "$L4T_ROOT/rootfs/etc/nv_tegra_release" 2>/dev/null \
     || echo "L4T release info not found"
   ```
3. **List** supported board config files and **join** them onto one comma-separated line:
   ```bash
   (cd "$L4T_ROOT" && ls *.conf 2>/dev/null) | paste -sd, -
   ```
4. **Check** whether the rootfs has been populated. An empty `rootfs/` means `apply_binaries.sh` has not been run yet:
   ```bash
   if [ -f "$L4T_ROOT/rootfs/etc/passwd" ]; then
     echo "populated"
   else
     echo "empty"
   fi
   ```

## Output format

Print a short report with these sections, one line each where possible:

```text
L4T root:        <path>
L4T release:     <release header line>
Board configs:   <comma-separated list>
Rootfs:          populated | empty
```

## Examples

Example output on an Orin AGX BSP (L4T R36):

```text
L4T root:        $HOME/Linux_for_Tegra
L4T release:     # R36 (release), REVISION: 3.0
Board configs:   jetson-agx-orin-devkit.conf,jetson-orin-nano-devkit.conf
Rootfs:          populated
```

Example output on a freshly untarred BSP where `apply_binaries.sh` has not been run yet:

```text
L4T root:        /tmp/Linux_for_Tegra
L4T release:     # R39 (release), REVISION: 0.0
Board configs:   jetson-agx-thor-devkit.conf
Rootfs:          empty
```

## Error handling

Each command falls back to a clearly labeled `"... not found"` string if the underlying file is missing — the skill never errors out mid-report. If `L4T_ROOT` does not contain `flash.sh` and `nv_tegra/`, exit early with a clear "not a Linux_for_Tegra root" message rather than printing misleading info.

## Limitations

- Read-only inspection only — does not validate signatures, kernel images, or device-tree overlays.
- Only checks the presence of `rootfs/etc/passwd` as a populated-rootfs proxy; will not detect a half-populated rootfs.
- Lists all `*.conf` board configs in `L4T_ROOT/`; does not try to infer which one the user intends to flash.

## Troubleshooting

- **Error:** `Not a Linux_for_Tegra root: '...' (missing flash.sh or nv_tegra/)`
  **Cause:** `L4T_ROOT` points at a parent directory, an extracted rootfs, or an unrelated path.
  **Solution:** Point `L4T_ROOT` at the directory that contains `flash.sh` (typically `Linux_for_Tegra/`).

- **Error:** `L4T release info not found`
  **Cause:** Neither `nv_tegra/nv_tegra_release` nor `rootfs/etc/nv_tegra_release` exists — the BSP tarball may be incomplete or `apply_binaries.sh` was never run.
  **Solution:** Re-extract the BSP tarball or run `apply_binaries.sh` to populate the rootfs.

## Notes

- Do not modify any files. This skill is read-only.
- If multiple board config files exist, list all of them — do not try to guess which one the user intends to flash.
