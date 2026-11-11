# DGX Spark Field Diagnostics Procedure

Package versions, repository locations, prerequisites, paths, and firmware UI
can change. Consult the current NVIDIA sources before execution:

- <https://docs.nvidia.com/dgx/dgx-spark/support.html#field-diagnostic-software>
- <https://docs.nvidia.com/pdf/userguide-dgx-spark-fieldiag.pdf>
- <https://nvidia.custhelp.com/app/answers/detail/a_id/5767>

Treat Customer Help article 5767 as a user-browser source. Automated agent or
script requests may return HTTP 403; this is expected and does not establish
that the link is broken or the service is unavailable. Ask the user to open it
in their browser and share relevant instructions if needed. Do not repeatedly
retry or bypass the restriction. Consult the accessible support page and PDF;
do not claim to have verified the article’s current content without seeing it.

The NVIDIA sources are authoritative. Commands below summarize them and are not
permission to execute.

## Installation and prerequisites

The current Field Diagnostics User Guide requires DOCA Host/OFED and MFT kernel
modules for CX7Stress, installed before Field Diagnostics. It currently names
the DOCA 3.3.0 Ubuntu 24.04 ARM64-SBSA repository; re-read the live guide rather
than assuming that version remains current. After approval, follow its exact
repository and package instructions.

The current package name is `dgx-spark-fieldiag`; its installed directory is
`/opt/nvidia/dgx-spark-fieldiag`. The package is ARM64 and is for NVIDIA DGX
Spark. Install it only after checking the live guide and receiving permission:

```bash
sudo apt-get update
sudo apt-get install dgx-spark-fieldiag
dpkg -l | grep dgx-spark-fieldiag
```

The current NVIDIA customer-support article describes adding the NVIDIA CUDA
Ubuntu 24.04 SBSA repository when the package is not already available. Do not
add or replace a repository preemptively. Inspect existing keyring/source files,
show the user the proposed change, and obtain approval first:

The CUDA archive key is a public package-signing key used by APT to verify
packages. It is not a password, private signing key, or user credential. The
`signed-by` option scopes that repository to this key.

The sequence below adapts NVIDIA's commands to stage the public key and source
list before privileged installation, without shell pipelines. Execute it only
after the approval above. Stop if a destination already exists; inspect it and
agree on a separate update rather than overwriting it. `gpg --show-keys` checks
that the download parses as a key; it does not independently authenticate its
fingerprint. Keep HTTPS verification enabled and the exact NVIDIA source URL.
If `gpg` is unavailable, stop and resolve that prerequisite with the user.

```bash
(
  set -eu
  umask 077
  staging_dir=$(mktemp -d /tmp/spark-cuda-repo.XXXXXXXX)
  trap 'rm -rf "$staging_dir"' EXIT
  curl --fail --silent --show-error --proto '=https' \
    --output "$staging_dir/cuda-archive-keyring.gpg" \
    https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/sbsa/cuda-archive-keyring.gpg
  test -s "$staging_dir/cuda-archive-keyring.gpg"
  mkdir "$staging_dir/gnupg"
  gpg --homedir "$staging_dir/gnupg" --batch --show-keys "$staging_dir/cuda-archive-keyring.gpg"
  printf '%s\n' 'deb [signed-by=/usr/share/keyrings/cuda-archive-keyring.gpg] https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/sbsa /' > "$staging_dir/cuda-sbsa-ubuntu2404.list"
  sudo test ! -e /usr/share/keyrings/cuda-archive-keyring.gpg
  sudo test ! -L /usr/share/keyrings/cuda-archive-keyring.gpg
  sudo test ! -e /etc/apt/sources.list.d/cuda-sbsa-ubuntu2404.list
  sudo test ! -L /etc/apt/sources.list.d/cuda-sbsa-ubuntu2404.list
  sudo install -d -m 0755 /usr/share/keyrings
  sudo install -m 0644 "$staging_dir/cuda-archive-keyring.gpg" /usr/share/keyrings/cuda-archive-keyring.gpg
  sudo install -m 0644 "$staging_dir/cuda-sbsa-ubuntu2404.list" /etc/apt/sources.list.d/cuda-sbsa-ubuntu2404.list
  sudo apt-get update
  sudo apt-get install dgx-spark-fieldiag
)
```

If installation fails partway through, report which destination files were
created. Do not retry blindly or remove system files without new permission.
Run repository setup without another package/repository configuration task in
parallel; the destination checks do not constitute a filesystem lock.

Installing the diagnostic package also installs `stress-ng`, `fio`, and
`memtester`, which support CPU, SSD, and memory tests. It does not install the
ConnectX-7 test prerequisites.

The expected tools are:

```bash
which fio
which memtester
which stress-ng
which ofed_info
which opensm
which ibstat
which mlxlink
```

Each command must print a binary path. If a command is missing, stop and resolve
it only from the refreshed NVIDIA guide. Do not silently skip CX7Stress while
claiming a full default test.

If an older `dgx-spark-fieldiag` is installed, the official guide says to purge
the package, remove its install directory, and autoremove dependencies before a
new install. These are destructive actions. First confirm the exact installed
state and target path, preserve any required logs, and obtain explicit deletion
and package-removal permission. Never remove an installation merely because the
directory exists.

## Pre-run plan

Before requesting execution permission, cover all of the following:

- save work and stop workloads because the system enters text-only multi-user
  mode;
- stable power and enough time for a roughly 30-minute run, with a documented
  maximum of three hours before timeout;
- local TTY or tested SSH access after the graphical session stops;
- package and tool verification;
- original Secure Boot state from `sudo mokutil --sb-state`;
- original GDM active and enabled states from `systemctl is-active gdm3` and
  `systemctl is-enabled gdm3`, so only diagnostic changes are reversed;
- permission for the stress run and a separate choice about later log access.

Secure Boot must be disabled for the diagnostic. If enabled, guide the user to
disable it. Confirm they can interact with UEFI using a physically connected
keyboard before rebooting. On NVIDIA DGX Spark, they can enter UEFI with Esc or
Delete during boot, or the agent can request entry after permission:

```bash
sudo systemctl reboot --firmware-setup
```

This command only enters firmware setup; it does not disable Secure Boot.
Normal OS/SSH access cannot operate the firmware menu. Guide the user to select
**Security > Secure Boot > Disabled**, save, and reboot. Resume after
reconnection and verify with `sudo mokutil --sb-state`. Use the vendor's
documented procedure for OEM systems rather than assuming identical menus.
Do not replace this handoff with EFI-variable manipulation.

## Full default hardware-health test

After permission, run from a local terminal or SSH session:

```bash
sudo init 3
cd /opt/nvidia/dgx-spark-fieldiag
sudo ./partnerdiag --field
```

`--field` is the NVIDIA-recommended single full default pass. It covers:

- `GpuStress`
- `C2CStress`
- `CpuStress1`
- `CpuStress2`
- `PowerStress`
- `ThermalStress`
- `FioSSD`
- `MemStress`
- `CX7Stress`

Do not edit the Spec JSON to tune or omit tests. Test virtual IDs and contents
can change between releases. If startup reports that `nvidia_drm` is in use, the
current guide says to stop and disable GDM with
`sudo systemctl disable gdm3 --now`, then start it after the run with
`sudo systemctl start gdm3`. Treat both as separately disclosed and authorized
state changes; do not disable GDM unless the reported condition occurs. Because
`disable --now` also changes boot enablement, restore the recorded enablement
state before the required reboot. If GDM was originally enabled, use
`sudo systemctl enable gdm3 --now`; otherwise restore only its recorded active
state. This enablement restoration is a systemd consequence of the NVIDIA
command, not a separate NVIDIA diagnostic instruction.

If the run is interrupted, power cycle before any retry. Do not retry on the
user's behalf without new permission.

## Results and logs

Record the final `PASS`, `FAIL`, or `RETEST` banner verbatim. Do not reduce a
multi-test result to an unsupported diagnosis.

NVIDIA's current customer-support article places run logs under
`/opt/nvidia/dgx-spark-fieldiag/logs-<timestamp>/`; the current PDF describes a
`dgx/logs-<yyyymmdd-hhnnss>` directory. Use the path printed by the actual run.
If log discovery is approved and the output did not identify it, search only
within the installed diagnostic tree for the matching `summary.json`; do not
scan unrelated filesystems.

Important artifacts include:

- top-level `output.log` and `run.log`;
- per-test `output.log` files;
- global telemetry such as Dmesg and BMC SEL output when produced;
- `summary.json`, whose entries contain the error code, test, virtual ID, and
  notes.

The error-code format ends in a 12-digit value; a value other than
`000000000000` indicates an error. Preserve exact codes for NVIDIA Support.
Never promise that `PASS` rules out every fault or that `FAIL` guarantees RMA
approval.

## Required restoration

After completion, or as soon as safely possible after failure:

1. Restore both the active and enabled GDM states recorded before the run. If
   this workflow disabled an originally enabled GDM, re-enable it before reboot.
2. If this workflow disabled Secure Boot, confirm the user is ready to
   interact with UEFI again and obtain permission before requesting firmware
   setup with `sudo systemctl reboot --firmware-setup`.
3. Guide the user to select **Security > Secure Boot > Enabled**, save, and
   reboot. Otherwise, reboot normally with `sudo reboot`.
4. Verify `sudo mokutil --sb-state` and confirm the expected GUI or remote access
   is restored.

The guide documents optional removal of CX7 test packages. Remove only packages
installed solely for this run, only after comparing pre-run state, and only with
new permission. Do not purge pre-existing networking components or core tools.
