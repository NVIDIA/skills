# jetson-customize-uphy — Gotchas

Cross-cutting gotchas. Per-step gotchas live inline in `procedure.md`.

- **Pinmap delta is BLIND to UPHY-fed controllers.** UPHY lanes are
  differential pairs absent from pinmux `.xlsm`. Drive enable/disable
  decisions off the carrier schematic — pinmap-only delta=0 silently
  lets FMON arm on a dead PHY (SError reboot loop).

- **PCIe handoff with `jetson-customize-pcie`.** If `*-pcie.dts` already
  exists in the same `overlay/` tree, scope this overlay to MGBE / UFS
  / USB3 SS / PCIe-status-only and cite the pcie overlay. Duplicate
  `/bus@0/pcie@<addr>` fragments merge in registration order — silent
  overwrites.

- **Don't touch the upstream BSP.** All edits go to
  `<source.root_path>/Linux_for_Tegra/`. Hand-editing
  `<bsp_image.root_path>/Linux_for_Tegra/` destroys the diff trail.

- **JSON sidecar is structured state, not authoritative.** The conf's
  ODMDATA line + the on-disk overlay + the two git commits are
  authoritative. Deleting the JSON is harmless; the next re-run
  rebuilds it. Hand-editing the JSON does not change device behavior.

- **`uphy0-config-6` requires clearing `UPHY_CONFIG`.** Both the
  `t<soc>.conf.common` default AND any local `UPHY_CONFIG=` line in
  the carrier conf must be neutralized; bash last-assignment wins.
  See `procedure.md` Step 6.

- **ODMDATA namespace mixing drops the whole line.** Top-level `/uphy`
  tokens are dashed end-to-end; `/pcie/pcie@N` sub-node tokens are
  `<node>_<key>=<value>` with underscore. A single rejected token
  drops the WHOLE `ODMDATA="..."` line.

- **Missing token beats wrong token.** Only emit ODMDATA tokens that
  flip a state away from stock; redundant tokens may cause line drop.
