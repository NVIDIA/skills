# Reading `tegrastats` output

`tegrastats` is the canonical Jetson telemetry tool. **Output tokens and spelling change across Jetson Linux / L4T releases and SoCs** (for example older guides document `EMC X%@Y` while newer Orin/Thor guides document `EMC_FREQ` with several alternate shapes). Treat any single cheat-sheet as approximate.

**There is no single NVIDIA table that covers every minor Jetson Linux revision.** The **Reported Statistics** glossary can differ between nearby releases (for example **R38.2.x vs R38.4.0**). Prefer the archive whose **Jetson Linux version string** best matches `cat /etc/nv_tegra_release` on the device; if NVIDIA has not published that exact slug, use the **closest published** archive for your product line and still sanity-check against live `tegrastats` output.

## Official documentation (NVIDIA)

The authoritative field glossary and CLI options live in the **Jetson Linux Developer Guide**, topic **Tegrastats Utility** (under *Applications and Tools → Jetson Linux Development Tools*). The HTML basename is stable:

`DeveloperGuide/AT/JetsonLinuxDevelopmentTools/TegrastatsUtility.html`

### URL pattern

Substitute the archive folder name (Jetson Linux release) for `<SLUG>`:

`https://docs.nvidia.com/jetson/archives/<SLUG>/DeveloperGuide/AT/JetsonLinuxDevelopmentTools/TegrastatsUtility.html`

Browse NVIDIA’s [Jetson documentation archives](https://docs.nvidia.com/jetson/archives/) to find the `<SLUG>` that matches your image (for example `r38.4`, `r36.4.4`). Not every **minor** gets its own folder (for example **`r38.2.2` and `r38.4.0` may 404** while **`r38.2.1`** and **`r38.4`** exist); use the nearest published slug.

### Per-archive **Tegrastats Utility** pages (known-good slugs)

The links below were **spot-checked** (HTTP `200`); NVIDIA may add or retire archive folders over time — if one fails, use the [archives index](https://docs.nvidia.com/jetson/archives/) and the URL pattern above.

| Archive slug | Tegrastats Utility (Reported Statistics) |
|--------------|------------------------------------------|
| `r35.5.0` | [Tegrastats Utility — r35.5.0](https://docs.nvidia.com/jetson/archives/r35.5.0/DeveloperGuide/AT/JetsonLinuxDevelopmentTools/TegrastatsUtility.html) |
| `r35.6.4` | [Tegrastats Utility — r35.6.4](https://docs.nvidia.com/jetson/archives/r35.6.4/DeveloperGuide/AT/JetsonLinuxDevelopmentTools/TegrastatsUtility.html) |
| `r36.2` | [Tegrastats Utility — r36.2](https://docs.nvidia.com/jetson/archives/r36.2/DeveloperGuide/AT/JetsonLinuxDevelopmentTools/TegrastatsUtility.html) |
| `r36.3` | [Tegrastats Utility — r36.3](https://docs.nvidia.com/jetson/archives/r36.3/DeveloperGuide/AT/JetsonLinuxDevelopmentTools/TegrastatsUtility.html) |
| `r36.4.3` | [Tegrastats Utility — r36.4.3](https://docs.nvidia.com/jetson/archives/r36.4.3/DeveloperGuide/AT/JetsonLinuxDevelopmentTools/TegrastatsUtility.html) |
| `r36.4.4` | [Tegrastats Utility — r36.4.4](https://docs.nvidia.com/jetson/archives/r36.4.4/DeveloperGuide/AT/JetsonLinuxDevelopmentTools/TegrastatsUtility.html) |
| `r36.5` | [Tegrastats Utility — r36.5](https://docs.nvidia.com/jetson/archives/r36.5/DeveloperGuide/AT/JetsonLinuxDevelopmentTools/TegrastatsUtility.html) |
| `r38.2` | [Tegrastats Utility — r38.2](https://docs.nvidia.com/jetson/archives/r38.2/DeveloperGuide/AT/JetsonLinuxDevelopmentTools/TegrastatsUtility.html) |
| `r38.2.1` | [Tegrastats Utility — r38.2.1](https://docs.nvidia.com/jetson/archives/r38.2.1/DeveloperGuide/AT/JetsonLinuxDevelopmentTools/TegrastatsUtility.html) |
| `r38.4` | [Tegrastats Utility — r38.4](https://docs.nvidia.com/jetson/archives/r38.4/DeveloperGuide/AT/JetsonLinuxDevelopmentTools/TegrastatsUtility.html) |

**Welcome** indices (overview only, not the field table): [r38.4](https://docs.nvidia.com/jetson/archives/r38.4/DeveloperGuide/index.html), [r36.5](https://docs.nvidia.com/jetson/archives/r36.5/DeveloperGuide/index.html), [r35.6.4](https://docs.nvidia.com/jetson/archives/r35.6.4/DeveloperGuide/index.html).

Welcome-only context: Jetson Linux **38.2** documentation explicitly states that release **does not support the Jetson Orin product family** at GA time; use a **36.x** guide for Orin-focused `tegrastats` wording when that applies to your stack ([r38.2 Welcome](https://docs.nvidia.com/jetson/archives/r38.2/DeveloperGuide/index.html)).

### Man page?

Many images ship **`tegrastats` without a `man` page**; use `tegrastats --help` on the device for a short option list, and the **Tegrastats Utility** topic for full semantics. NVIDIA also documents the binary location as `/core/utils/tegrastats` in the BSP package layout.

### Very old Jetson Linux (e.g. L4T 32.x)

Archived L4T help bundles (such as [Jetson Linux 32.7.6](https://docs.nvidia.com/jetson/archives/l4t-archived/l4t-3276/index.html)) may use an older web help shell; if search fails, use the PDF/HTML **Development Guide** for that release from the [Jetson Download Center](https://developer.nvidia.com/embedded/downloads) or cross-check a nearby **r35.x** archive link above.

## Example line (typical Jetson Linux 36.x / 38.x style, Orin-class)

Not guaranteed character-for-character on every release; compare to your device and the version-matched **Tegrastats Utility** table.

```
RAM 4011/8138MB (lfb 8x4MB) SWAP 0/4068MB (cached 0MB) CPU [12%@1190,7%@1190,...] EMC_FREQ 0% GR3D_FREQ 35% AO@47C CPU@52C GPU@49C tboard@45C tdiode@46C VDD_GPU_SOC 1234mW VDD_CPU_CV 567mW
```

## Recorded samples (optional; cite Jetson Linux revision)

Real captures help humans and agents pattern-match **timestamps**, **`*_FREQ @MHz`** spellings, **Thor `GR3D_FREQ` three-tuples**, and **thermal zone names** that the generic example above does not show. Treat samples as **illustrative**: a future Jetson Linux update can change formatting even on the same SKU.

### Jetson AGX Thor Developer Kit — Jetson Linux **R38 (release), REVISION: 4.0**

Identifying lines from `/etc/nv_tegra_release`:

```text
# R38 (release), REVISION: 4.0, GCID: 43443517, BOARD: generic, EABI: aarch64, DATE: Wed Dec 31 00:15:19 UTC 2025
# KERNEL_VARIANT: oot
```

Official field glossary for this line: [Tegrastats Utility — Jetson Linux 38.4 archive](https://docs.nvidia.com/jetson/archives/r38.4/DeveloperGuide/AT/JetsonLinuxDevelopmentTools/TegrastatsUtility.html).

`tegrastats` foreground sample (~idle system; includes leading timestamp as printed by this build):

```text
04-21-2026 15:50:52 RAM 105265/125772MB (lfb 34x4MB) CPU [0%@972,0%@972,0%@972,0%@972,0%@972,0%@972,1%@972,0%@972,0%@972,0%@972,0%@972,8%@972,0%@972,0%@972] EMC_FREQ 0%@2750 GR3D_FREQ @[314,314,314] NVENC0_FREQ @314 NVENC1_FREQ @314 NVDEC0_FREQ @314 NVDEC1_FREQ @314 NVJPG0_FREQ @314 VIC off OFA_FREQ @315 PVA0_FREQ off APE 300 cpu@37.343C tj@38.875C soc012@37.468C gpu@38.875C soc345@37.406C VDD_GPU 1962mW/1962mW VDD_CPU_SOC_MSS 5887mW/5887mW VIN_SYS_5V0 5635mW/5635mW VIN 19780mW/19780mW
```

Notes for this capture: **`GR3D_FREQ` shows three frequencies** (three GPCs on Thor). **No `NVDLA*`** tokens — consistent with NVIDIA’s statement that NVDLA `tegrastats` reporting applies to AGX Orin, not Thor, on the [same guide page](https://docs.nvidia.com/jetson/archives/r38.4/DeveloperGuide/AT/JetsonLinuxDevelopmentTools/TegrastatsUtility.html).

## Field-by-field (common tokens; names drift by release)

| Field                        | Meaning                                                                 |
|------------------------------|-------------------------------------------------------------------------|
| `RAM 4011/8138MB`            | Used / total system RAM in MiB.                                         |
| `(lfb 8x4MB)`                | Largest free block: how many contiguous N-MB chunks remain. Low = fragmented. |
| `SWAP 0/4068MB`              | Used / total swap. `(cached 0MB)` is the page-cache portion.            |
| `CPU [12%@1190,...]`         | Per-core utilization% @ MHz. One entry per online core.                 |
| `EMC_FREQ` / `EMC`           | External Memory Controller (DRAM) bandwidth utilization; **token spelling varies** (see NVIDIA table for your Jetson Linux version). |
| `GR3D_FREQ`                  | GPU 3D engine utilization / frequency reporting; **Orin = 2 GPCs, Thor = 3 GPCs** in current docs — frequency tuple width can differ. |
| `AO@47C` `CPU@52C` `GPU@49C` | Thermal zones in °C; labels come from `thermal_zone*` names.          |
| `tboard@`, `tdiode@`         | Board-level thermal probes (often devkit-oriented).                     |
| `VDD_* …mW`                  | Power rails in milliwatts; **rail names and formatting differ by SoC and guide generation**. |
| Other accelerators           | Newer guides add `NVENC*`, `NVDEC*`, `NVDLA*`, `NVJPG*`, `PVA*`, `OFA` when applicable — see version-matched doc. |

## Quick triage

- **`GR3D_FREQ` near 100% sustained** → workload is GPU-bound.
- **EMC / `EMC_FREQ` near 100% with low GPU** → memory-bandwidth-bound; quantization or KV-cache reuse helps (token name depends on Jetson Linux version).
- **CPU cores pinned at 100%** but GPU low → likely Python overhead, decoder bottleneck, or pre/post-processing on CPU.
- **`AO`/`GPU` above 90 °C sustained** → thermal throttling imminent; check the cooling solution and `nvpmodel`.
- **`lfb` showing only small blocks** → memory fragmentation; large CUDA allocations may fail before total RAM is exhausted.

## Useful invocations

```bash
# One sample, one second (stdout).
tegrastats --interval 1000 | head -n 1

# 30-second log to disk.
tegrastats --interval 1000 --logfile /tmp/tegra.log &
sleep 30
tegrastats --stop

# Stop any background tegrastats (preferred):
tegrastats --stop
```

If the device policy denies access to some counters, re-run with elevated privileges only as needed (same flags).

## SKU notes

- **Orin Nano / NX / AGX**: layout above; no `nvidia-smi`.
- **Thor**: layout above is preserved; `nvidia-smi` is also available and gives a complementary GPU-centric view.
