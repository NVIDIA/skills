# Disabling EMC DVFS via the BPMP DTB

Reference material for `../SKILL.md` the "Content edit: EMC DVFS disable / enable" step.

This doc covers **which nodes / properties to edit** for the EMC
DVFS gate. The decompile / recompile / commit mechanics live in
the canonical `../../../references/bsp-customization-bpmp-dtb.md`;
apply the snippets below inside the protocol's "Edit the DTS" step.

Disabling EMC DVFS pins EMC at the BPMP-init rate (no scaling with bandwidth demand). It is **not** a single edit — bwmgr alone leaves EMC scaling on via the surviving cactmon path (and the OSP / QoS path on T26x). All required edits must land in the same BPMP DTB or the result is undefined.

## SoC detection (run first)

Decompile the resolved BPMP DTB and check for the `osp-controller` node:

```bash
dtc -I dtb -O dts <bpmp-dtb> | grep -c osp-controller
```

| Result | SoC family | Applies edits |
|---|---|---|
| `0` | T23x (Orin) | #1 + #2 only |
| `≥1` | T26x (Thor) | #1 + #2 + #3 |

## Edits

### #1 — `bwmgr.enabled = <0x00>` (all SoCs, mandatory)

Disables bandwidth-driven scaling requests.

```dts
bwmgr {
    enabled = <0x00>;   /* 0x01 = scale (default); 0x00 = pin at init rate */
    ...
};
```

### #2 — `cactmon.enabled = <0x00>` (all SoCs, mandatory)

The EMC activity monitor that drives DVFS hints. Leaving it active while bwmgr is off lets the monitor keep firing on a frozen frequency target, producing spurious actmon events / latency anomalies.

```dts
cactmon {
    enabled = <0x00>;
    ...
};
```

### #3 — Remove the `osp-controller` node (T26x mandatory, T23x skip)

The OSP (operating-state / power) controller arbitrates EMC frequency floors based on QoS clients. With bwmgr off it has nothing valid to act on; leaving the node in place can re-introduce frequency changes from the QoS path.

T23x (Orin) BPMP DTBs do not contain this node — skip on T23x.

Delete the node entirely (do NOT use `status = "disabled"` — OSP isn't a standard kernel-DT consumer; the node must not exist for BPMP to skip the path):

```dts
/delete-node/ osp-controller;
```

Equivalent inline form when editing inside the parent node: replace the `osp-controller { … };` block with `/delete-node/ osp-controller;` at the same scope.

## Re-enabling EMC DVFS

Reverse the edits that were applied:

- `bwmgr.enabled = <0x01>`
- `cactmon.enabled = <0x01>`
- (T26x only) restore the `osp-controller` node from the pristine BPMP DTS.

Capture the pristine `osp-controller` block before the first disable, or recover it later by `dtc`-decompiling the pristine `bpmp-fw-dtb` from `<bsp_image.root_path>`.

## Why all of this is gated together

| Surviving path | What it does | Effect if left active while bwmgr is off |
|---|---|---|
| `cactmon` (EMC activity monitor) | Drives DVFS hints from bandwidth observations | Keeps firing on a frozen frequency target → spurious events / latency anomalies |
| `osp-controller` (T26x only) | Arbitrates EMC frequency floors from QoS clients | Re-issues frequency changes via the QoS path → EMC drifts despite bwmgr being off |

Both surviving paths are independent enough of `bwmgr` to keep operating against a target the firmware has already pinned. Disabling them is the only way to actually pin the EMC frequency.
