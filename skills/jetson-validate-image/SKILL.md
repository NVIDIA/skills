---
name: jetson-validate-image
description: >-
  Use after jetson-flash-image to run static BSP checks, on-target
  smoke/regression tests on a flashed DUT, or both. Not for build
  or flash steps. Triggers: validate bsp, on-target validation.
version: 0.0.1
license: "Apache-2.0"
metadata:
  data-classification: public
  author: "Jetson Team"
  tags:
    - bsp
    - validation
    - test
    - deploy
  domain: meta
---

# Validate BSP Image

> **Status:** the DUT-access contract is stable; the rest of the
> validation procedure is a skeleton.

## Purpose

Confirm that a freshly customized BSP landed correctly — both as a
static artifact on disk and as a running system on the target —
without re-promoting or re-flashing. Forms the **validation tail of
Deploy** in the Setup → Customize → Build → Deploy pipeline (see
[`../../context/bsp-customization-workflow.md`](../../context/bsp-customization-workflow.md)
for the pipeline view) and is independently re-runnable.

## Prerequisites

- Active target-platform profile with `bsp_image:` resolved (run
  `/jetson-init-image` first).
- For static-only scope: nothing further; the skill reads
  `<bsp_image.root_path>` directly.
- For on-target scope:
  - `/jetson-flash-image` already pushed the staged BSP to the DUT.
  - `dut_access:` block authored in the active profile (or filled
    interactively at runtime — see [`## DUT access`](#dut-access)).
  - Host tooling per transport: `ssh` + `sshpass` for SSH; Python
    3.6+ with `pyserial` for UART.
  - Required env vars resolvable on the host when `auth=password`
    or `sudo.method=password` (`password_env` / `sudo.password_env`
    name the env var; never inline the secret in YAML).

## When to invoke

- After `jetson-flash-image` has put the BSP onto a target.
- The user explicitly asks to validate, test, or run smoke /
  regression checks on a flashed device.
- As a CI gate before declaring a customization batch shippable.

## Instructions

The procedure below is a skeleton.

1. **Read the active target** per the prerequisite contract.
2. **Choose validation scope** — static (against `bsp_image` on
   disk, no DUT needed) and/or on-target (DUT must be booted
   from the just-flashed image).
3. **Static checks** (if in scope):
     - Required artifacts present at expected paths in
       `<bsp_image.root_path>/Linux_for_Tegra/`.
     - DTB / module checksum or signature verification.
     - Partition-layout sanity vs. the per-board `.conf`'s XML.
     - Cross-check overlay-staged outputs against `bsp_image`
       (they should be identical post-promotion).
     - **Initramfs ↔ kernel + rootfs module coherence.** Extract
       `<bsp_image>/Linux_for_Tegra/bootloader/l4t_initrd.img` and
       `<bsp_image>/Linux_for_Tegra/rootfs/boot/initrd`; verify
       three invariants against the *promoted* `bsp_image` state:
       (a) `<bsp_image>/Linux_for_Tegra/kernel/Image` matches
       `<bsp_image>/Linux_for_Tegra/rootfs/boot/Image` byte-for-
       byte — drift here means
       [the "Mirror kernel Image into rootfs" step](../jetson-promote-image/SKILL.md#mirror-kernel-image-into-rootfs-when-kernel-changed)
       was skipped and any subsequent initramfs refresh built
       against the previous kernel.
       (b) for each module path the two initrd images ship, the
       bytes / md5 match the file under
       `<bsp_image>/Linux_for_Tegra/rootfs/lib/modules/<ver>/` —
       any drift means modules will be shadowed at early boot.
       (c) the vermagic stamped on every initramfs `*.ko` matches
       the `UTS_RELEASE` reachable from
       `<bsp_image>/Linux_for_Tegra/kernel/Image` (e.g. parse the
       `Linux version …` string with `strings`) — a vermagic skew
       means the kernel `Image` was refreshed without rerunning
       `l4t_update_initrd.sh`, and modules will fail to load with
       "disagrees about version of symbol …". All three failure
       modes are closed by `/jetson-promote-image`'s gate on
       either `kernel/Image` or `rootfs/lib/modules/` plus the
       kernel-Image mirror; surface drift here and route the user
       back to a clean promote. See
       [`../jetson-promote-image/SKILL.md`](../jetson-promote-image/SKILL.md#refresh-initramfs-when-kernel-or-modules-changed).
4. **On-target checks** (if in scope):
     - **Connect to the DUT** per the [`## DUT access`](#dut-access)
       section below — resolve transport (ssh / uart), credentials,
       and sudo method from the active profile's `dut_access:`
       block (with interactive fallback when fields are missing /
       marked `prompt`), then run the connection probe and refuse
       if it fails.
     - Confirm boot reached userspace.
     - Run the selected test suite (smoke, regression, focused
       per-customization, ad-hoc).
     - **Loaded-module srcversion drift.** For modules a
       customization is known to have rebuilt, compare
       `cat /sys/module/<name>/srcversion` on the DUT against the
       `modinfo /lib/modules/$(uname -r)/.../<name>.ko | awk
       '/srcversion/ {print $2}'` reading. A mismatch means the
       kernel is running an older copy than the rootfs ships —
       almost always a stale initramfs (the bootloader-side initrd
       shipped a pre-customize module, it loaded first, and the
       rootfs copy cannot replace a live module). Recommend
       re-running `/jetson-promote-image` and re-flashing. Note:
       some modules don't emit `srcversion`; fall back to an md5
       check against the binary the kernel loaded by extracting the
       region under `/sys/module/<name>/sections/` or by comparing
       behaviorally (printk / sysfs nodes / DT properties the new
       version is known to expose).
     - **Running kernel vs. rootfs kernel `Image`.** Compare
       `cat /proc/version` (or `uname -v`) on the DUT against the
       `Linux version …` string extracted from
       `/boot/Image` (`strings /boot/Image | grep -m1 'Linux version'`).
       A mismatch — usually the build timestamp / `LOCALVERSION` —
       means the bootloader is running an older kernel `Image` than
       the rootfs holds, almost certainly because a fresh `Image`
       was promoted but the initramfs / `extlinux.conf` / QSPI
       boot partition wasn't refreshed. Modules in the rootfs
       will then have a different vermagic and any subsequent
       `modprobe` of a built-against-the-new-kernel `.ko` will
       fail. Recommend `/jetson-promote-image` and re-flash.
     - **Userspace dmesg readability.** Ubuntu 22.04 sets
       `kernel.dmesg_restrict=1`; non-root `dmesg` reads return
       "Operation not permitted" and silently zero hits. Every
       `dmesg`-based check must run with `sudo` (or temporarily
       lower the restriction via `sudo sysctl
       kernel.dmesg_restrict=0`). Surfacing this in the validate
       layer keeps printk-based customization checks honest.
     - Collect results, logs, artifacts.
5. **Summary**: per-check pass/fail, overall verdict, where
   logs and artifacts landed.

## DUT access

The on-target leg needs a way to reach the just-flashed DUT.
Two transports are supported as full peers: `ssh` (primary) and
`uart` (fallback for DUTs with no network).

The contract is locked in but lives in
[`references/dut-access.md`](references/dut-access.md) to keep this
SKILL.md under the agent-routing budget. That reference covers:

- The `dut_access:` profile schema (ssh, uart, sudo, workdir).
- Resolution order (profile → env var → interactive prompt) with
  the full refusal-trigger table.
- The mandatory connection probe (`uname -r` +
  `cat /etc/nv_tegra_release`) and its output-validation rules.
- File transfer per transport (`scp` vs. base64-over-tty, with
  the >100 KB warning).
- Sudo invocation matrix per `sudo.method` × transport.
- UART implementation contract that `scripts/uart_session.py`
  honors (state machines for login / exec / push / pull, exit
  codes, robustness notes).
- Security notes (password handling, host-key pinning).

## Available Scripts

| Script | Purpose | Arguments |
|---|---|---|
| `scripts/uart_session.py` | UART transport black box for the on-target leg: login, command exec (with optional sudo), and base64-over-tty file transfer. Replaces ssh when the DUT has no network or ssh is broken. | `--tty <dev> [--baud <n>] --user <name> --password-env <ENVVAR> [--sudo-password-env <ENVVAR>] [--shell-prompt <regex>] [--lock-strategy refuse\|wait] <probe\|exec\|push\|pull> [...]` |

Invocation (the skill calls scripts as a subprocess — `run_script()`
in agent-runtime terminology):

```bash
# run_script: probe the DUT over UART
DUT_UART_PASSWORD_ENV=DUT_UART_PWD \
DUT_UART_PWD="$(read -rs -p 'UART login pw: '; echo "$REPLY")" \
  scripts/uart_session.py \
    --tty /dev/ttyACM0 --baud 115200 \
    --user ubuntu --password-env DUT_UART_PASSWORD_ENV \
    probe

# run_script: exec a sudo command and capture exit code
scripts/uart_session.py --tty /dev/ttyACM0 --user ubuntu \
  --password-env DUT_UART_PASSWORD_ENV \
  --sudo-password-env DUT_SUDO_PWD \
  exec --use-sudo 'dmesg | tail -200'
```

The script's exit code is the contract — see the exit-code table in
[`UART implementation contract`](references/dut-access.md#uart-implementation-contract).

## Examples

Static-only validation (no DUT needed):

```
/jetson-validate-image
> static checks only against the staged BSP
```

On-target validation over SSH after a freshly flashed DUT:

```
/jetson-flash-image
   ↓
/jetson-validate-image
> on-target checks via dut_access.ssh
```

On-target validation over UART (no network on the DUT):

```
/jetson-validate-image
> use the uart transport at /dev/ttyACM0; the dut_access.uart block
  in the profile already has the tty and login_password_env wired up
```

## Limitations

- Placeholder skill — only the DUT-access contract and the
  `uart_session.py` helper are locked in. The static-check list,
  test-suite selection, result-sink layout, and pass/fail policy
  are tracked under `## Open items` and may change.
- UART file transfer is byte-banged base64 at ~10 KB/s on 115200
  baud — emits a warning for sources > 100 KB but proceeds. For
  high-volume transfers, switch to the SSH transport.
- `uart_session.py` opens and closes the tty per subcommand
  invocation (~1–2 s login per call). Validation passes running
  > 10 commands amortize poorly on UART; prefer SSH.
- SSH uses `StrictHostKeyChecking=accept-new` with a per-profile
  `known_hosts` file. A fingerprint change refuses — typically
  means the DUT was reflashed (host keys regenerated) or the IP
  was reassigned. Remove the per-profile entry manually rather
  than auto-accepting.
- Passwords are never inlined in the profile YAML — only
  `password_env` (env-var name) is persisted. `auth=prompt` /
  `sudo.method=prompt` exposes the password in the conversation
  log, which is the user's responsibility to manage.
- `transport=uart` with `lock_strategy=steal` is **not implemented**
  (would require sending control characters that could corrupt the
  holder's state).

## Troubleshooting

| Error | Cause | Solution |
|---|---|---|
| `no DUT transport configured` (refuse) | On-target scope requested but `dut_access.transport` unresolved across profile / env / interactive prompt | Author `dut_access:` in the active profile, or rerun with static-only scope. |
| `$PASSWORD_ENV is unset` (refuse) | `auth=password` (or `sudo.method=password`) names an env var that isn't exported on the host | `export <ENVVAR>=...` before invoking the skill; do NOT inline the password in YAML. |
| `tty held by another process` (refuse, exit 3) | `lock_strategy=refuse` and `fuser` reports another holder (minicom, picocom, getty) | Close the holding process (`sudo fuser -k <tty>` only if you know what's there), or rerun with `lock_strategy=wait`. |
| `pyserial import failed` (exit 4) | UART transport selected but `pyserial` not installed in the skill's Python | `apt install python3-serial` or `pip install pyserial`. |
| SSH fingerprint refused | DUT's host keys changed since the per-profile `known_hosts` was pinned (typically a reflash) | Remove the matching line from `<workspace>/target-platform/<profile-stem>.known_hosts` and rerun — the new key will be accepted on first connect. |
| `login failed` (exit 128) | UART probe couldn't match a login or shell prompt within timeout | Confirm the DUT is powered + booted to userspace; check `--shell-prompt` override if the DUT carries an unusual `PS1`; verify the `--user` matches a real account. |
| `DUT not booted from the just-flashed BSP` (warn / refuse) | `/etc/nv_tegra_release` on the DUT doesn't match `bsp_image.version` | Re-run `/jetson-flash-image` and confirm the DUT actually power-cycled into the new BSP, not the previous one. |
| `sudo prompt detected but no sudo password configured` (exit 130) | `--use-sudo` passed to `uart_session.py exec` but `--sudo-password-env` not set, and DUT user lacks NOPASSWD | Set `sudo.method=password` + `sudo.password_env` in the profile, or grant NOPASSWD on the DUT via `/etc/sudoers.d/`. |
| `command timed out` (exit 129) | Long-running DUT command exceeded the script's timeout | Run the command directly via `ssh` transport, or split into shorter steps. |

## References

- [`references/dut-access.md`](references/dut-access.md) — full DUT-access contract: profile schema, resolution order, probe, file transfer, sudo, UART implementation, security.
- [`../../context/target-platform-contract.md`](../../context/target-platform-contract.md) — target-platform contract.
- [`../../context/bsp-customization-workflow.md`](../../context/bsp-customization-workflow.md#workflow-invariants) — Workspace edit protocol (this skill is the Deploy tail).
- [`../jetson-build-source/SKILL.md`](../jetson-build-source/SKILL.md) — Build builder; produces the artifacts this skill validates.
