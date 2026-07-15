# Clock control model

Reference material for `../SKILL.md`.

## Layer stack

```
userspace ── nvpower.sh ──► Kernel cpufreq/devfreq ──► BPMP
  (governor, min/max,         (sysfs scaling             (clock_max_rate
   static via sysfs)           within ceiling)            ceiling, EMC bwmgr)
```

Each layer constrains the one below. The **BPMP DTB** sets the per-clock hard ceiling and gates EMC DVFS. The kernel scales dynamically within that ceiling per the active governor. `nvpower.sh` runs at boot to pick the governor and (optionally) write per-device rates to sysfs.

## Two ceilings inside the BPMP DTB

A clock node carries up to two ceiling properties:

- `max-rate-maxn` — silicon / MAXN power-profile ceiling. Read-only for customization.
- `max-rate-custom` — optional customer override. The only knob; set to lower the runtime ceiling, remove (or omit) to run at `max-rate-maxn`.

Detailed semantics + the nvpmodel ↔ BPMP clock-node mapping live in the sibling reference `bpmp-dtb-clock-edits.md`.

## Effective runtime ceiling

```
effective = min(BPMP cap, active nvpmodel cap)
```

Either layer can be the binding constraint. The active nvpmodel mode can clamp below `max-rate-custom`; use the sibling skill `/jetson-customize-nvpmodel` to inspect.
