# nvpower.sh edits

Reference material for `../SKILL.md` Operation 2.

`nvpower.sh` lives at `Linux_for_Tegra/rootfs/etc/systemd/nvpower.sh` and runs at boot via `nvpower.service`. It sets cpufreq / devfreq governors and (optionally) per-device min / max / static rates.

## Functions of interest

| Function | Writes to | Default (non-safety platforms) |
|---|---|---|
| `set_cpufreq_governor()` | `/sys/devices/system/cpu/cpu*/cpufreq/scaling_governor` | `schedutil` |
| `set_devfreq_governor()` | `/sys/class/devfreq/<dev>/governor` | `tegra_wmark` — but GPU and nvjpg are skipped |

There is **no dedicated function** for per-device min / max / static rates. To pin or bound a device's rate, write `min_freq` / `max_freq` to its sysfs node from a helper added inside (or invoked alongside) the existing flow.

## Common edits

| Goal | Edit |
|---|---|
| Pin all CPUs to Fmax | In `set_cpufreq_governor`: set `desired_cpufreq_gov="performance"` unconditionally (remove the `IS_SAFETY_PLATFORM` → `schedutil` override). |
| Pin all devfreq devices to Fmax | In `set_devfreq_governor`: set `desired_devfreq_gov="performance"` and remove the `gpu` / `nvjpg` skip block. |
| Static custom rate on one device | After the governor is set, write `<rate>` to `min_freq` then `max_freq` (order matters when raising; reverse when lowering). |
| Min / max bounds without pinning | Same as static, but `min < max` — governor stays dynamic within the bounds. |

## Package-upgrade caveat

`nvpower.sh` is shipped by the `nvidia-l4t-init` deb. Package upgrades clobber hand edits to the rootfs file. For long-lived test setups, prefer a systemd drop-in or a separate helper file checked into the rootfs alongside `nvpower.sh` rather than editing it in place.

## Apply

`nvpower.service` runs the edited script on the next boot after Deploy re-flashes the rootfs — see the sibling skill `/jetson-flash-image`.
