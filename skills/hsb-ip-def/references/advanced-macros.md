# Advanced IP macros

Four `HOLOLINK_def.svh` macros are consumed by `<hsb-ip-root>/top/HOLOLINK_top.sv` and serve advanced or implementation-specific use cases — clock-domain CDC tightening, embedded-RAM resource overrides, and feature disables. The validator handles them like any other in-scope IP macro; this file documents them for engineers who encounter them in existing code or need them for a specific board.

## `SYNC_CLK_HIF_APB`

**Toggle.** When defined, declares that the host-interface clock (`i_hif_clk`) and the APB control clock (`i_apb_clk`) are synchronous — same source, fixed phase relationship. The IP uses tighter, lower-latency CDC paths. When undefined (the default), the IP uses asynchronous CDC logic that handles arbitrary phase between the two clocks.

| | |
|---|---|
| Kind | `\`define` toggle |
| Default | Undefined (asynchronous CDC) |
| RTL sites | `<hsb-ip-root>/top/HOLOLINK_top.sv:154-157` (declares the local `SYNC_CLK_HIF_APB` localparam: `1` if macro defined, `0` if not); `<hsb-ip-root>/top/HOLOLINK_top.sv:353, 417` (passed as `.SYNC_CLK` parameter to CDC modules) |
| Example-library status | Example files leave it undefined |
| Safe to use? | **Use only with explicit board-level confirmation.** Defining this macro tells the IP that the two clocks are truly synchronous; if they aren't, CDC violations will produce metastability and intermittent failures that are hard to debug. Default to leaving this undefined unless you've verified via clock-generator analysis that the two clocks share a source. |

## `SYNC_CLK_HIF_PTP`

**Toggle.** Same pattern as `SYNC_CLK_HIF_APB`, but for the relationship between the host-interface clock and the PTP clock (`i_ptp_clk`).

| | |
|---|---|
| Kind | `\`define` toggle |
| Default | Undefined |
| RTL sites | `<hsb-ip-root>/top/HOLOLINK_top.sv:160-163` (defines local `SYNC_CLK_HIF_PTP` localparam); `<hsb-ip-root>/top/HOLOLINK_top.sv:605` (passed as `.SYNC_CLK` to PTP CDC) |
| Example-library status | Example files leave it undefined |
| Safe to use? | Same caution as `SYNC_CLK_HIF_APB`. Verify the two clocks are truly synchronous on your board before defining. The default async-CDC path is correct for any clocking arrangement; defining this is only worthwhile when the synchronous-clock relationship is established. |

## `PERI_RAM_DEPTH`

**Integer override.** When defined, overrides the depth of the embedded RAM blocks inside the SPI and I²C peripheral controllers. The default is set inside the peripheral submodules; defining this macro reduces (or increases) the RAM depth on a per-instance basis.

| | |
|---|---|
| Kind | `\`define` (integer, OR undefined) |
| Default | Undefined (peripheral submodules use their internal default) |
| Example value | `32` (in the ultra-minimal archetype only) |
| RTL sites | `<hsb-ip-root>/top/HOLOLINK_top.sv:958, 1008` (`\`ifdef PERI_RAM_DEPTH .RAM_DEPTH(`PERI_RAM_DEPTH)`) — passed as `RAM_DEPTH` parameter to SPI and I²C blocks |
| When to use | Define this only on implementations where embedded RAM is at a premium and you have measured peripheral memory utilization. The default is sized for typical sensor-control register sequences (a few hundred bytes); reducing it constrains the longest single I²C/SPI transaction. |
| Safe to use? | Yes — purely a resource trade-off. Reducing it doesn't change protocol correctness; it just limits transaction length. |

## `DISABLE_COE`

**Toggle.** When defined, disables Camera-over-Ethernet (CoE — IEEE 1722) packet generation in the dataplane. CoE is used on AGX Thor for hardware accelerated networking. Defining `DISABLE_COE` removes the CoE-specific dataplane code path so the IP is RoCE/UDP-only.

| | |
|---|---|
| Kind | `\`define` toggle |
| Default | Undefined (CoE logic is included) |
| RTL sites | `<hsb-ip-root>/top/HOLOLINK_top.sv:1435` (gates CoE packet-generation logic — `\`ifdef DISABLE_COE`) |
| Safe to use? | Defining this removes the CoE-specific dataplane code path; the rest of the IP is unaffected. Define it only when the design will not use CoE. |

## When to use these

The macros documented here are advanced or board-specific — most designs will not need them. They are first-class HSB IP macros (the IP source consumes them at compile time), and the validator treats them like any other in-scope macro. Reach for them only when you have a specific implementation-resource constraint or a verified clock-source relationship that motivates the choice.

Out-of-scope macros that may appear in some workspace files (project-internal flags and IP features the team is not exposing through this skill) are silently passed through by the validator with no per-macro annotation. If you encounter a macro that's not in `macro-reference.md` or this file, treat it as project-specific and consult your design owner.
