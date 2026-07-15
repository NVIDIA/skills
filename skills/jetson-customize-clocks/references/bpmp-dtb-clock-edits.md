# BPMP DTB clock-node edits

Reference material for `../SKILL.md` Operation 1.

This doc covers **what to edit** (node names, properties, valid
ranges, mapping). The decompile / recompile / sanity-check /
commit mechanics live in the canonical
`../../../references/bsp-customization-bpmp-dtb.md`;
every snippet below applies inside the protocol's "Edit the DTS" step (DTS
editing).

## Two ceilings in the BPMP DTB

A clock node may carry two ceiling properties:

| Property | Role | Edit policy |
|---|---|---|
| `max-rate-maxn` | Silicon / MAXN power-profile ceiling. Sometimes absent. | **Read-only.** Never edit — it's the hardware cap, not a customer override. |
| `max-rate-custom` | Optional customer override. | **The only knob.** Set to lower the runtime ceiling below `max-rate-maxn`. Remove (or omit) to run at `max-rate-maxn`. |

BPMP rejects `max-rate-custom ≥ max-rate-maxn`. When `max-rate-maxn` is absent from a node, query the live cap on a running target of the same chip / SKU first (one-time prerequisite):

```bash
cat /sys/kernel/debug/bpmp/debug/clk/<clock-name>/max_rate
```

Then pick `max-rate-custom` strictly below that value.

The runtime ceiling combines this BPMP cap with the active nvpmodel mode's cap — see [`clock-control-model.md#effective-runtime-ceiling`](clock-control-model.md#effective-runtime-ceiling) for the formula.

## DTS edit form

Inside the named clock node — never inside `lateinit`:

```dts
<clock-node-label>: <clock-node-name> {
    /* existing properties ... */
    max-rate-maxn   = <0x83215600>;   /* 2.2 GHz, read-only — silicon / MAXN ceiling */
    max-rate-custom = <0x77359400>;   /* 2.0 GHz lowered ceiling — must be < max-rate-maxn */
};
```

`dtc -O dts` may print these as decimal (`2.2e9`, `2.0e9` Hz). Either form is valid DTS; hex is shown here for clarity.

Example — lower CPU cluster 0's runtime ceiling to 2.0 GHz on a part where MAXN is 2.2 GHz:

```dts
cluster0_clk: cluster0 {
    /* existing properties ... */
    max-rate-maxn   = <0x83215600>;   /* 2.2 GHz */
    max-rate-custom = <0x77359400>;   /* 2.0 GHz */
};
```

To return a clock to its `max-rate-maxn` ceiling, delete the `max-rate-custom` line (or never add one).

## nvpmodel knob ↔ BPMP clock-node mapping

T234 Orin mapping. T264 Thor uses different knob and clock-node naming — derive from an existing nvpmodel block paired with the corresponding BPMP DTB.

| nvpmodel knob | BPMP clock node |
|---|---|
| `CPU_A78_0..5 MAX_FREQ` | `cluster0` / `cluster1` / `cluster2` (two cores per cluster) |
| `GPU MAX_FREQ` | `nafll_gpusys` **and** every `nafll_gpcX` (X = 0, 1, …) — see GPU note below |
| `EMC MAX_FREQ` | `dram` |

Units: `MAX_FREQ` is **kHz** for `CPU_A78_*`, **Hz** for `GPU` / `EMC`. Value `-1` = no nvpmodel clamp (the BPMP cap is binding for that clock).

### T23x GPU cap — `nafll_gpusys` + all `nafll_gpcX`

On T23x (Orin) the GPU clock is split across the SYS partition (`nafll_gpusys`) and one or more GPC partitions (`nafll_gpc0`, `nafll_gpc1`, …). The effective GPU ceiling is the **max** of the per-node `max-rate-custom` values — capping only `nafll_gpusys` leaves the `nafll_gpcX` nodes free to run at their `max-rate-maxn`, so the GPU still ramps above the intended cap. Apply the same `max-rate-custom` value to `nafll_gpusys` and to every `nafll_gpcX` node present in the DTS.

Enumerate the GPC nodes once per BPMP DTB:

```bash
grep -nE '^\s*nafll_gpc[0-9]+\s*:' <decompiled.dts>
```

The recompile / sanity-check / commit cycle remains a single round of the BPMP-DTB protocol — apply all GPU clock-node edits in the same "Edit the DTS" step.

## Inspection cookbook

**BPMP-side — list every clock that already carries a customer override or has a documented MAXN ceiling:**

```bash
grep -nE 'max-rate-(maxn|custom)' <decompiled.dts>
```

Clocks with no `max-rate-custom` run at `max-rate-maxn`. Nodes defining neither require the on-target `max_rate` query above before customizing.

**nvpmodel-side — print the boot default mode's per-clock `MAX_FREQ` values:**

```bash
NVP=Linux_for_Tegra/rootfs/etc/nvpmodel/nvpmodel_<sku>.conf
DEFAULT=$(grep -E '^< PM_CONFIG DEFAULT=' "$NVP" | sed -E 's/.*=([0-9]+).*/\1/')
awk -v d="$DEFAULT" '/^< POWER_MODEL/{p=($0 ~ "ID="d" ")} p && /MAX_FREQ/' "$NVP"
```
