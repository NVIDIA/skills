---
name: jetson-print-device-info
description: Use when you need to print Jetson device info (module model, L4T version, kernel, OS version, current power mode) from a running Jetson target. This is an example skill.
version: 0.0.1
license: "Apache-2.0"
metadata:
  author: "Jetson Team"
  tags: [jetson, device, info]
  languages: [bash]
  data-classification: public
---

# jetson-print-device-info

Prints a concise summary of the Jetson device this skill runs on.

This skill is intended as a reference example for the `jetson-device-skills` repo and the NVIDIA-wide skills CI. It runs on the Jetson target (not the host PC) and performs read-only inspection — useful as a baseline capture before running performance tests.

## Purpose

Capture a baseline snapshot of a Jetson target's software stack and power mode before running performance, regression, or compatibility tests.

## When to use

- Starting a performance run and you want a captured baseline of the device's software stack and power mode.
- Verifying that a freshly flashed Jetson matches an expected L4T / JetPack version.

## Prerequisites

- Running on a Jetson target (not the host PC).
- Standard CLIs available: `tr`, `cat`, `uname`, `uptime`, `lsb_release` (or `/etc/os-release`).
- `nvpmodel` available for power mode (optional — skill falls back gracefully).

## Inputs

None. The skill reads only from the local Jetson filesystem and standard CLIs.

## Instructions

Run each step in order and print the captured values into the report shown under [Output format](#output-format).

1. **Capture** the module model and **validate** it is a Jetson target — exit early otherwise:
   ```bash
   # Device-tree strings are null-terminated, so strip NULs before printing.
   model=$(tr -d '\0' < /proc/device-tree/model 2>/dev/null)
   case "$model" in
     *Jetson*) ;;
     *) echo "Not running on a Jetson target (model: '${model:-unknown}')"; exit 1 ;;
   esac
   echo "$model"
   ```
2. **Extract** the L4T release header line — skip the rest of the file (long list of library SHAs):
   ```bash
   head -1 /etc/nv_tegra_release 2>/dev/null || echo "L4T release info not found"
   # Equivalent: grep -m1 '^# R' /etc/nv_tegra_release
   ```
3. **Run** `nvpmodel -q` and **join** its two output lines (`NV Power Mode: <name>` and the mode number) onto one line. `paste -sd` only uses the first char of its delimiter, so use `awk` to insert the literal ` / ` separator:
   ```bash
   nvpmodel -q 2>/dev/null | awk 'NR==1{a=$0; next} {print a" / "$0}' \
     || echo "nvpmodel not available"
   ```
4. **Print** the kernel version and uptime:
   ```bash
   uname -r
   uptime -p
   ```
5. **Print** the OS version (prefer `lsb_release`, fall back to `/etc/os-release`):
   ```bash
   lsb_release -ds 2>/dev/null || (. /etc/os-release && echo "$PRETTY_NAME") || echo "OS version not found"
   ```

## Output format

Print a short report with these sections, one line each where possible:

```text
Model:           <device-tree model string>
L4T release:     <release header line>
Power mode:      <nvpmodel name> / <mode number>
Kernel:          <uname -r>
Uptime:          <uptime -p>
OS version:      <lsb_release / /etc/os-release output>
```

## Examples

Example output on an Orin AGX dev kit (L4T R36 / Ubuntu 22.04):

```text
Model:           NVIDIA Jetson AGX Orin Developer Kit
L4T release:     # R36 (release), REVISION: 3.0
Power mode:      NV Power Mode: MAXN / 0
Kernel:          5.15.136-tegra
Uptime:          up 2 hours, 14 minutes
OS version:      Ubuntu 22.04.4 LTS
```

Example output on an AGX Thor (L4T R39 / Ubuntu 24.04):

```text
Model:           NVIDIA Jetson AGX Thor Developer Kit
L4T release:     # R39 (release), REVISION: 0.0
Power mode:      NV Power Mode: 120W / 1
Kernel:          6.8.0-tegra
Uptime:          up 12 minutes
OS version:      Ubuntu 24.04 LTS
```

## Error handling

Each command falls back to a clearly labeled `"... not found"` / `"... not available"` string if the underlying file or binary is missing — the skill never errors out mid-report. If `/proc/device-tree/model` is missing or does not contain a Jetson string, exit early with a clear "not running on a Jetson target" message.

## Limitations

- Read-only inspection only — does not detect GPU/CPU clocks, thermal state, or per-rail power.
- `nvpmodel` output format varies between L4T versions; the skill prints it verbatim rather than parsing.

## Troubleshooting

- **Error:** `/proc/device-tree/model: No such file or directory`
  **Cause:** Running on a host PC, not a Jetson target.
  **Solution:** Run the skill on the Jetson device directly (e.g. via SSH).

- **Error:** `nvpmodel: command not found`
  **Cause:** L4T BSP not installed, or running in a minimal container without `nvpmodel`.
  **Solution:** Expected on non-Jetson or stripped environments — the skill prints `"nvpmodel not available"` and continues.

## Notes

- Read-only. Do not change power mode, install packages, or modify any files.
- If the skill is invoked on a host PC by mistake, `/proc/device-tree/model` will not contain a Jetson model string — detect that and exit with a clear message rather than printing misleading info.
