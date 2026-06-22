# Default-user staging via `l4t_create_default_user.sh`

Detailed reference for `../SKILL.md`'s "Preflight checks" step (default-user-staging sub-aspect): detecting
existing staged users, the helper invocation, and the OEM-wizard
fallback path.

## Why it matters

A freshly applied `bsp_image` has no Linux user pre-staged in the
rootfs. At first boot the DUT therefore runs Ubuntu's OEM
configuration wizard, which requires a keyboard + monitor (or serial
console interaction) to complete. To skip the wizard, stage a default
user into the rootfs **before flashing** with
`l4t_create_default_user.sh`.

## Detect an existing staged user

Check `<bsp_image.root_path>/Linux_for_Tegra/rootfs/home/` and
`rootfs/etc/passwd` for a non-system UID ≥ 1000. If a staged user is
already present, skip the staging step — re-running the helper
overwrites the staged credentials, which is rarely what the operator
wants.

## Pick credentials (one `AskUserQuestion`)

If no user is staged, prompt once with four click-to-select options:

| Option | Effect |
|---|---|
| `ubuntu / ubuntu` | Stage `username=ubuntu password=ubuntu` (BSP default convention) |
| `nvidia / nvidia` | Stage `username=nvidia password=nvidia` (NVIDIA SDKM default) |
| `custom` | Sub-prompt for `<username>` + `<password>` |
| `skip` | No staging — OEM wizard runs on first boot |

For any non-`skip` choice, run from `<bsp_image.root_path>/Linux_for_Tegra/`:

```bash
sudo ./tools/l4t_create_default_user.sh \
  --username <username> --password <password> \
  --autologin --accept-license
```

`--autologin` skips GDM at first boot; `--accept-license` is mandatory
for non-interactive use (helper otherwise blocks on EULA prompt).

Record the resolution for the "Confirm resolution" step in SKILL.md
(refusal-gate semantics are owned there).
