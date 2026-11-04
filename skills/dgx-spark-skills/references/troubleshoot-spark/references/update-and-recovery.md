# Update-state and recovery triage

Use this reference only for update, boot, package, driver, firmware, or recovery symptoms. Refresh all linked pages first.

## Establish the update path

The [OS and Component Update Guide](https://docs.nvidia.com/dgx/dgx-spark/os-and-component-update.html) identifies DGX Dashboard as the primary, recommended update path for NVIDIA DGX Spark Founders Edition. The [DGX Dashboard guide](https://docs.nvidia.com/dgx/dgx-spark/dgx-dashboard.html) says a remote user can connect with NVIDIA Sync and select **DGX Dashboard**; the Dashboard opens through the managed tunnel. Its update operation can change packages and firmware and reboot the device, so obtain explicit permission immediately before starting it.

### Manual fallback for Founders Edition

The Spark update guide uses the historical name `apt dist-upgrade`; the current [DGX OS package-upgrade procedure](https://docs.nvidia.com/dgx/dgx-os-7-user-guide/upgrading-the-os.html) uses its newer equivalent, `apt full-upgrade`.

After explicit permission to refresh package metadata, preview the transaction:

```bash
sudo apt update
sudo apt full-upgrade -s
```

Review packages proposed for installation, upgrade, holding, or removal. Stop on broken dependencies, unexpected removals, or held NVIDIA, kernel, or driver packages; do not add flags, unhold packages, or improvise a repair.

After the user reviews the preview and explicitly authorizes the update, run each step in order and stop on failure:

```bash
sudo apt full-upgrade
sudo fwupdmgr refresh
sudo fwupdmgr upgrade
sudo reboot
```

Do not substitute plain `apt upgrade`: it never removes installed packages, so it can hold back updates whose dependency changes require removal and leave a coordinated DGX OS package transition incomplete. A one-off package upgrade, direct driver or kernel installation, mixed repositories, interrupted update, omitted firmware phase, or omitted reboot is not evidence that the documented recipe completed.

Classify the observed path as one of:

- `Dashboard path` — Dashboard performed and completed the update.
- `Documented manual fallback` — every current NVIDIA-documented step completed in order.
- `Incomplete` — an intended supported path stopped or failed.
- `Mixed or off-recipe` — additional or substituted update, driver, kernel, firmware, or repository operations occurred.
- `Unknown` — evidence is insufficient.

Do not call an off-recipe path "unsupported" unless a current NVIDIA source says so. Do say that it is outside the recipe verified by the current Spark update guide.

## Minimal read-only evidence

After explicit diagnostic consent, prefer a user-provided Dashboard update-status screenshot or these narrow observations:

```bash
grep -E '^(DGX_PRETTY_NAME|DGX_SWBUILD_VERSION|DGX_OTA_VERSION|DGX_OTA_DATE)=' /etc/dgx-release
uname -r
nvidia-smi --query-gpu=driver_version --format=csv,noheader
```

The filtered release command intentionally omits the serial number. Treat a missing file or field as evidence to investigate, not proof of corruption. Use the current [DGX OS update-verification documentation](https://docs.nvidia.com/dgx/dgx-os-7-user-guide/initial_setup.html) to interpret the release markers. When comparing against the Spark Release Notes, explain that partner GB10 update availability may lag by varying amounts and verify applicability to the user's model before treating a software or firmware version difference as unexpected.

Unexpected state includes a failed Dashboard update, a component set that conflicts with the applicable current release source, an incomplete documented sequence, or mixed/off-recipe changes that plausibly precede the symptom. Report the exact mismatch; do not infer which package to purge, downgrade, or pin.

For an incomplete or off-recipe update, direct the user to connect with [NVIDIA Sync](https://docs.nvidia.com/sync/latest/) and open **DGX Dashboard** to inspect the supported update state. Do not silently continue with package-manager repairs. If Dashboard cannot reconcile the state, use current NVIDIA troubleshooting/support information.

## Recovery boundary

The NVIDIA [System Recovery](https://docs.nvidia.com/dgx/dgx-spark/system-recovery.html) procedure is destructive and explicitly Founders Edition only. It erases the internal SSD. Before helping a Founders Edition user perform it:

1. Refresh the page and confirm model applicability.
2. Explain data loss and prerequisites.
3. Confirm the user has reviewed backups and selected the exact removable-media target.
4. Obtain explicit permission before every destructive or device-changing phase.

For an OEM GB10 system, state that NVIDIA documents model-specific recovery differences and that this Founders Edition procedure does not apply. Do not provide, adapt, or link an OEM recovery or OEM support procedure in this initial skill. Never improvise a recovery location, image, command, or key sequence.
