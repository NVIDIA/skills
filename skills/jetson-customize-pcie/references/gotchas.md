# Gotchas — jetson-customize-pcie

- **Mode hard-pinned to RC.** Every customized controller gets
  `pcie-mode=1` in ODMDATA and targets the RC sub-node in the
  overlay. EP only via explicit `mode_override="ep"` in Step 5c.
- **`enable` is derived from `<uphy-state>`, not asked.** UPHY-
  allocated → mandatorily okay; non-allocated → mandatorily
  disabled. Offering "keep disabled" on a UPHY-allocated controller
  creates inconsistent ODMDATA-vs-overlay state — abort.
- **Confirm-or-customize gate fires even when every value is
  determinable.** Silent commit of a non-trivial customization is a
  known failure mode; "Proceed as-is" is one click.
- **Node addresses come from `dtc -I dtb -O dts`, not the SoC TRM.**
  TRM lists register bases; kernel DT may use a different `reg`
  layout. Address misalignment = silent overlay miss.
- **`compatible` mismatch silently skips the overlay.** Cross-check
  the composite root `compatible` against the live DUT's
  `/proc/device-tree/compatible`. UEFI plugin-manager filters by
  compatible; mismatched = skip, no error in dmesg.
- **`OVERLAY_DTB_FILE` is `jetson-build-source`'s problem.** This
  skill never touches the carrier flash conf for overlay registration.
- **PCIe handoff with jetson-customize-uphy is intra-file.** Both
  skills append fragments to the same composite. Same target-path,
  LAST fragment wins after `dtc` flatten. This skill must run AFTER
  `/jetson-customize-uphy`; never let two fragments at the same
  target-path disagree on `status`.
- **Don't disable a stock-okay BPMP DTB controller in ODMDATA.**
  `pcie@N_status=disabled` for an already-disabled BPMP entry is a
  no-op the parser may treat as ambiguous, dropping the whole line.
  Disable via the kernel-DT overlay only when BPMP-side is already off.
- **Missing token > wrong token.** Only emit ODMDATA when the BPMP
  DTB schema exposes the field AND the token flips a value you want
  changed.
- **No upstream BSP edits.** All edits land in
  `<source.root_path>/Linux_for_Tegra/` + `<source.root_path>/bsp_sources/`.
- **Never run `git reset --hard` autonomously.** If the Step 8
  cross-check finds a contradictory ODMDATA-vs-overlay pair, stop
  and ask the user how to recover the two commits (overlay tracker
  + bsp_sources). Offer `git revert` or manual amend; let the
  operator choose.
