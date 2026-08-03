# Packetizer Architecture

Use this reference when explaining how the packetizer works, why the def fields exist, or why runtime APB configuration is separate from `HOLOLINK_def.svh`.

This file documents known HSB IP rev `16'h2604` and backward-compatible rev `16'h2603`. If live HSB IP source is available, use live `HOLOLINK_top.sv` and packetizer RTL as the authority for the checked-out design. If the live source changes field mapping, bypass behavior, APB control, or packetizer internals, trust the live source and warn that this reference may need an update.

## System Blocks

The HSB packetizer documentation names four blocks (`doc/hololink/data_plane/packetizer.md:8-12`):

- `packetizer_top.sv`: top-level wrapper and controller.
- `packetizer.sv`: core data rearrangement and virtual-port routing.
- `virtual_port.sv`: per-virtual-port buffering and width conversion.
- `odd_even_gen.sv`: sorting network used for data rearrangement.

## Clock Domains

The documented packetizer domains are (`doc/hololink/data_plane/packetizer.md:18-21`):

- Sensor clock domain: `i_sclk`.
- Packet/host clock domain: `i_pclk`.
- APB clock domain: `i_aclk`.

In `HOLOLINK_top`, `packetizer_top` connects `i_sclk` to `sif_rx_clk[i]`, `i_pclk` to `i_hif_clk`, and `i_aclk` to `i_apb_clk` (`top/HOLOLINK_top.sv:1384-1408`).

## Def Fields To RTL Parameters

`HOLOLINK_top` maps `HOLOLINK_def.svh` fields into `packetizer_top` per Sensor RX interface:

- `DIN_WIDTH` receives `SIF_RX_WIDTH[i]`.
- `DOUT_WIDTH` receives `HOST_WIDTH`.
- `W_USER` receives `DATAUSER_WIDTH`.
- `SORT_RESOLUTION` receives `SIF_RX_SORT_RESOLUTION[i]` when enabled, otherwise `2`.
- `VP_COUNT` receives `SIF_RX_VP_COUNT[i]` when enabled, otherwise `1`.
- `VP_SIZE` receives `SIF_RX_VP_SIZE[i]` when enabled, otherwise `32`.
- `NUM_CYCLES` receives `SIF_RX_NUM_CYCLES[i]` when enabled, otherwise `1`.
- `PACKETIZER_ENABLE` receives `SIF_RX_PACKETIZER_EN[i]`.
- `MTU` receives `HOST_MTU`.

See `top/HOLOLINK_top.sv:1356-1383`.

## Data Flow

The documentation summarizes data flow as input processing, virtual-port management, and output handling (`doc/hololink/data_plane/packetizer.md:84-101`).

In RTL:

- Sensor data first passes through the optional Sensor RX data-generator mux, then into `packetizer_top` (`top/HOLOLINK_top.sv:1349-1399`).
- `packetizer_top` buffers control/state, pattern-RAM addressing, bypass, duplication, padding, and latency control (`packetizer/packetizer_top.sv:100-128`, `:263-384`).
- When `PACKETIZER_ENABLE` is set, `packetizer_top` instantiates `packetizer` (`packetizer/packetizer_top.sv:395-433`).
- When `PACKETIZER_ENABLE` is clear, `packetizer_top` instantiates a bypass `virtual_port` path (`packetizer/packetizer_top.sv:435-466`).

## Core Packetizer

The core packetizer implements:

- Config RAM for clear, sort, and virtual-port controls (`packetizer/packetizer.sv:69-134`).
- A multi-cycle input pipeline sized by `NUM_CYCLES` (`packetizer/packetizer.sv:140-179`).
- Optional odd-even sorting / data rearrangement controlled by `SORT_RESOLUTION` (`packetizer/packetizer.sv:186-224`).
- Virtual-port selection and output routing for `VP_COUNT` outputs (`packetizer/packetizer.sv:228-255`).

The odd-even generator computes `SORT_WIDTH = D_WIDTH/RESOLUTION` and uses resolution-sized swap groups (`packetizer/odd_even_gen.sv:17-60`).

## Virtual Port Behavior

`virtual_port.sv` treats `VP_SIZE` as a bit width:

- It declares `vp_wr_data` as `[VP_SIZE-1:0]`.
- It uses `VP_SIZE` in shift widths and counters.
- It passes `VP_SIZE` as the FIFO input data width.

See `packetizer/virtual_port.sv:18-30`, `:55-99`, `:153-170`, and `:210-212`.

In current `HOLOLINK_top`, `DYNAMIC_VP` and `MIXED_VP_SIZE` are forced to `0` (`top/HOLOLINK_top.sv:1360-1361`). Do not promise dynamic or mixed-size VP behavior from `HOLOLINK_def.svh` fields alone.

## APB Register Context

The packetizer has APB control and status registers. The register constants are:

- `pack_scratch`: `0x0000_0000`
- `pack_ram_addr`: `0x0000_0004`
- `pack_ram_data`: `0x0000_0008`
- `pack_ctrl`: `0x0000_000C`
- `pack_tvalid_cnt`: `0x0000_0080`
- `pack_psn_cnt`: `0x0000_0084`

See `reg_map/regmap_pkg.sv:246-255`.

`packetizer_top` decodes `pack_ctrl` bits for load array, padding, duplicate, max address, latency, and bypass (`packetizer/packetizer_top.sv:121-128`).

This skill does not generate APB writes or pattern RAM contents. It only chooses compile-time `HOLOLINK_def.svh` fields that size and enable the packetizer hardware.
