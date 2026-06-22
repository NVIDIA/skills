# Recovery-mode entry via `boardctl`

Detailed reference for `../SKILL.md`'s "Put the DUT into recovery mode" step: locating the
in-tree `boardctl` binary, enumerating targets, invoking recovery, and
the manual fallback.

## Locate `boardctl` inside the BSP tree

`boardctl` ships with the BSP, not the host distro — always prefer the
in-tree binary over anything on `$PATH`, since a stale system-wide
`boardctl` from a previous BSP install is a common source of
"`<target>` not supported" failures.

Search under `<bsp_image.root_path>/Linux_for_Tegra/`, starting at the
conventional location:

```bash
ls <bsp_image.root_path>/Linux_for_Tegra/tools/board_automation/boardctl
# fallback if the conventional path is absent:
find <bsp_image.root_path>/Linux_for_Tegra/ -type f -name boardctl
```

Bind the resolved absolute path as `<boardctl>`. If neither lookup
finds an executable, skip to the manual fallback below — do **not**
silently fall through to a `$PATH` lookup.

## Enumerate targets

Run the resolved binary and parse the targets it advertises under the
`-t` switch:

```bash
<boardctl> -h          # read the targets listed under the `-t` switch
```

Parse that list and present it to the user. Recommend `topo` when
present — it is USB-topology-driven and does not require manual
power-cycle. Other listed targets (chip-family names like `t234`,
`t264`, or board-specific variants) remain selectable when the user
has reason to prefer them or when `topo` is absent.

## Invoke recovery

Once the user selects, invoke:

```bash
<boardctl> -t <user-selected target> recovery
```

`recovery` is the verb that drops the DUT into RCM. Do not substitute
`reset`, `reset --recovery`, `enter-recovery`, `rcm`, or anything else
copied from external docs — `boardctl`'s own help is authoritative for
the verb list, and `recovery` is what the current BSPs accept.

## Manual fallback

If `boardctl` cannot be found under `Linux_for_Tegra/`, or
`<boardctl> -h` lists no usable targets, or the user knows from
experience that none will function on their setup, prompt the user to
set the force-recovery jumper or button and power-cycle the device by
hand. The skill never invents a target name not present in
`<boardctl> -h`, and never falls back to a `$PATH` `boardctl`; either
the in-tree `boardctl` runs with a user-confirmed target or the user
does it by hand.

The skill's "Preflight checks" step verifies — for both paths — that the DUT actually
landed in RCM mode.

## Post-flash reset (T26x / Thor)

The same `<boardctl>` resolved above is reused after a successful
flash on T26x platforms, using the same target the user selected for
RCM entry:

```bash
<boardctl> -t <user-selected target> reset
```

T23x / Orin issues the reset internally from the flash tool; an extra
`boardctl reset` there is unnecessary noise. If the "Put the DUT into recovery mode" step used the manual
jumper / power-cycle path, the operator must remove the
force-recovery jumper / release the button and power-cycle the device
by hand on T26x.
