---
name: jetson-headless-mode
description: Plan and apply safe Jetson headless-mode changes to reclaim GUI and daemon memory.
version: 0.0.1
license: "Apache-2.0"
metadata:
  author: "Jetson Team"
  tags: [jetson, headless, memory]
  languages: [bash]
  data-classification: public
---

# Jetson Headless Mode

Plan-then-apply for safe, reversible user-space memory reclamation: switch the default systemd target away from `graphical.target` and disable a curated set of non-essential daemons. This is the highest-yield, lowest-risk memory win on Jetson.

## Purpose

Build a user-approved headless-mode plan from live audit data, then apply only safe, reversible user-space changes that reduce desktop and daemon memory use on Jetson.

## When to use

- "Free as much memory as possible — I don't need the GUI."
- "I'm shipping this Jetson as an inference appliance / edge node."
- After `jetson-memory-audit` shows `default_systemd_target=graphical.target` or shows `gdm3` / `lightdm` / `sddm` active on a system the user describes as headless.

## When NOT to use

- The user needs the local desktop, display output, kiosk UI, or any X/Wayland session. In that case, do not recommend disabling the graphical target or display manager; use `jetson-memory-audit` for a read-only view and suggest non-GUI memory options instead.
- You do not have current audit data. Run `jetson-memory-audit` first, or ask the user for its output, before proposing changes or estimating savings.

Use live device data as the source of truth. Jetson family, SKU/variant, memory totals, active display services, and savings estimates must come from `jetson-diagnostic/scripts/detect_jetson.sh`, `audit.json`, or a fresh `jetson-memory-audit` run. If a value is not available, say it is unknown instead of guessing. The savings numbers below are upper bounds; the real delta is whatever a before/after audit reports.

## Prerequisites

- Start from a current `jetson-memory-audit` JSON snapshot.
- Confirm the user does not need the local desktop, display output, kiosk UI, or X/Wayland session.
- Mutating changes require `sudo` and explicit user approval; dry-run first unless approval was already given in the same prompt.
- Run on the Jetson host or in a host-visible sandbox with access to systemd state.

## Available Scripts

| Script | Purpose | Arguments |
|--------|---------|-----------|
| `scripts/plan.sh` | Reads a memory audit JSON and emits a plan containing safe, reversible recommendations. | `--audit PATH` or `--audit -`, plus `--human`. |
| `scripts/apply.sh` | Prints or applies the safe commands from a plan JSON. Dry-run by default. | `--plan PATH` or `--plan -`, `--apply`, `--reboot`, `--drop-caches`. |

If your agent runtime supports `run_script`, use it to run `scripts/plan.sh` and `scripts/apply.sh` and summarize the returned output. Otherwise run the scripts with `bash` from the repository root.

## Instructions

1. Run `scripts/plan.sh` to read `audit.json` (from `jetson-memory-audit`) and emit a plan with only `safety: safe` knobs (target switch, display managers, audio, print, modem, etc.).
2. Show the plan to the user and confirm.
3. Run `scripts/apply.sh --plan plan.json` for a **dry run**. Re-run with `--apply` to execute. Add `--drop-caches` to flush the page cache afterward, or `--reboot` to take effect immediately.
4. Re-run `jetson-memory-audit/scripts/audit.sh` to verify the actual delta.

## Expected workflow

Use the scripts for estimates and application so recommendations are based on the current device state rather than the static upper-bound table alone.

- For "what would headless save", "estimate", "plan", or production planning prompts, run `scripts/plan.sh --audit <audit.json>` and report `estimated_total_savings_mb`, the top `recommendations[*].knob`, and whether any display manager or `graphical.target` is active. Do not run `apply.sh`.
- For prompts where the user explicitly says to apply headless mode now, run `scripts/apply.sh --plan <plan.json>` once as a dry run first. If the user has already approved mutation in the same prompt, re-run the same command with `--apply` and mention the reversible command(s).
- If direct execution fails in an agent runtime, invoke scripts with `bash {baseDir}/scripts/<script-name> ...`. Do not try to `chmod` installed skill files.

## Plan / apply contract

- `plan.sh` emits the same JSON shape as `jetson-inference-mem-tune/scripts/recommend.py`: an array of `recommendations` with `{layer, knob, estimated_savings_mb, safety, command, reversible_command, rationale}`.
- `apply.sh` filters entries to `safety == "safe"` with a non-empty `command`, then re-checks the filtered safety marker in the shell loop before execution. Anything else, such as kernel command-line changes, device-tree changes, or accuracy tradeoffs, is out of scope for this skill.
- Default mode is **dry-run**. `--apply` is required to mutate the system.

## Knobs covered

| Knob                            | Action                                       | Estimated savings | Reversible? |
|---------------------------------|----------------------------------------------|-------------------|-------------|
| `disable-graphical-target`      | `systemctl set-default multi-user.target`    | up to 865 MB      | yes         |
| `stop-gdm3` / `gdm` / `lightdm` / `sddm` / `display-manager` | `systemctl disable --now <svc>` | ~200 MB / svc | yes |
| `stop-pulseaudio`               | disable audio daemon                         | ~8 MB             | yes         |
| `stop-bluetooth`                | disable Bluetooth stack                      | ~6 MB             | yes         |
| `stop-ModemManager`             | disable WWAN manager                         | ~4 MB             | yes         |
| `stop-cups` / `stop-cups-browsed` | disable print stack                        | ~5 / ~3 MB        | yes         |
| `stop-snapd`                    | disable Snap daemon                          | ~30 MB            | yes         |
| `stop-whoopsie` / `kerneloops`  | disable crash reporters                      | ~4 / ~2 MB        | yes         |
| `stop-avahi-daemon`             | disable mDNS                                 | ~3 MB             | yes         |
| `stop-unattended-upgrades` / `packagekit` | disable background package work    | ~6 / ~8 MB        | yes         |

## Do NOT disable these services

- `nvargus-daemon` — required for any libargus camera pipeline.
- `nvgetty.service` — serial console; disabling can lock you out of recovery.
- `nvpmodel` — power-mode service; required for clock/power tuning.
- `containerd` / `docker` — leave on if you run containers (most inference workloads do).
- `nvfb` / `nvdisplay`-related kernel services — tied to boot-time display configuration, so this skill does not change them.

## Safety

- Does not edit `/boot/extlinux/extlinux.conf`, the device tree, or boot-time memory reservations.
- Does not disable services it does not have an explicit entry for (no blanket "disable everything not whitelisted").
- Every applied change has a documented `reversible_command`. Re-running the plan with the reverts is sufficient to restore.
- Dry-run by default. `--apply` is the only way to mutate.
- Report only device facts and savings figures that came from live detection or audit output.

## Cross-platform behavior

The same set of knobs applies to every Jetson family in the matrix above. The script reads `JETSON_GENERATION` / `JETSON_PRODUCT_LINE` / `JETSON_VARIANT` from `jetson-diagnostic/scripts/detect_jetson.sh` (and still exports legacy `JETSON_SKU`) so the agent can attribute the savings correctly in its summary, but it does not branch on product line.
