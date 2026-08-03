# `HOLOLINK_top` Port Map

How macros in `HOLOLINK_def.svh` gate the port set on `HOLOLINK_top`. When a gating macro is defined, the listed ports appear in the top-module's port list; when undefined, they are absent and the user's top wrapper must not connect to them. Line citations are to `<hsb-ip-root>/top/HOLOLINK_top.sv`.

This file documents known rev `16'h2604` and backward-compatible rev `16'h2603`. If live `HOLOLINK_top.sv` is available, treat that module declaration as the source of truth. For unknown or newer revisions, derive the port set from live source and warn if it differs from this reference.

The full canonical port description is in the HSB user guide at `docs/user_guide/port_description.md`.

## Always-present ports

These ports exist regardless of macro state.

### Global reset

| Port | Direction | Width | Notes |
|---|---|---|---|
| `i_sys_rst` | input | 1 | Active-high synchronous reset |

### APB (control) interface

| Port | Direction | Width | Notes |
|---|---|---|---|
| `i_apb_clk` | input | 1 | Required > 20 MHz |
| `o_apb_rst` | output | 1 | Synchronous, active-high |
| `o_apb_psel[REG_INST-1:0]` | output | `REG_INST` | Per-register-block select |
| `o_apb_penable` | output | 1 | |
| `o_apb_paddr[31:0]` | output | 32 | |
| `o_apb_pwdata[31:0]` | output | 32 | |
| `o_apb_pwrite` | output | 1 | |
| `i_apb_pready[REG_INST-1:0]` | input | `REG_INST` | |
| `i_apb_prdata[REG_INST-1:0][31:0]` | input | 32 × `REG_INST` | |
| `i_apb_pserr[REG_INST-1:0]` | input | `REG_INST` | |

### Init done

| Port | Direction | Width | Notes |
|---|---|---|---|
| `o_init_done` | output | 1 | Asserted high in `i_apb_clk` domain after `init_reg[]` sequence completes |

### Sensor event

| Port | Direction | Width | Notes |
|---|---|---|---|
| `i_sif_event[15:0]` | input | 16 | Asynchronous event flags |

### Host interface clocking

| Port | Direction | Width | Notes |
|---|---|---|---|
| `i_hif_clk` | input | 1 | 156.25 MHz typical |
| `o_hif_rst` | output | 1 | Synchronous, active-high |

### Host RX/TX AXI-Stream

`HOST_IF_INST` instances of each. All host interfaces share the scalar widths controlled by `HOST_WIDTH`, `HOSTKEEP_WIDTH`, and `HOSTUSER_WIDTH`; there is no per-host width selection.

| Port | Direction | Width | Notes |
|---|---|---|---|
| `i_hif_axis_tvalid[HOST_IF_INST-1:0]` | input | `HOST_IF_INST` | Host RX (Ethernet → IP) |
| `i_hif_axis_tlast[HOST_IF_INST-1:0]` | input | `HOST_IF_INST` | |
| `i_hif_axis_tdata[HOST_IF_INST-1:0][HOST_WIDTH-1:0]` | input | `HOST_WIDTH × HOST_IF_INST` | |
| `i_hif_axis_tkeep[HOST_IF_INST-1:0][HOSTKEEP_WIDTH-1:0]` | input | `HOSTKEEP × HOST_IF_INST` | |
| `i_hif_axis_tuser[HOST_IF_INST-1:0][HOSTUSER_WIDTH-1:0]` | input | `HOSTUSER × HOST_IF_INST` | |
| `o_hif_axis_tready[HOST_IF_INST-1:0]` | output | `HOST_IF_INST` | |
| `o_hif_axis_tvalid[HOST_IF_INST-1:0]` | output | `HOST_IF_INST` | Host TX (IP → Ethernet) |
| `o_hif_axis_tlast[HOST_IF_INST-1:0]` | output | `HOST_IF_INST` | |
| `o_hif_axis_tdata[HOST_IF_INST-1:0][HOST_WIDTH-1:0]` | output | … | |
| `o_hif_axis_tkeep[HOST_IF_INST-1:0][HOSTKEEP_WIDTH-1:0]` | output | … | |
| `o_hif_axis_tuser[HOST_IF_INST-1:0][HOSTUSER_WIDTH-1:0]` | output | … | |
| `i_hif_axis_tready[HOST_IF_INST-1:0]` | input | `HOST_IF_INST` | |

### GPIO

| Port | Direction | Width | Notes |
|---|---|---|---|
| `i_gpio[GPIO_INST-1:0]` | input | `GPIO_INST` | Sync to `i_apb_clk` |
| `o_gpio[GPIO_INST-1:0]` | output | `GPIO_INST` | Sync to `i_apb_clk` |
| `o_gpio_dir[GPIO_INST-1:0]` | output | `GPIO_INST` | 1 = output, 0 = input |

### Sensor reset control

| Port | Direction | Width | Notes |
|---|---|---|---|
| `o_sw_sen_rst[31:0]` | output | 32 | Per-sensor software reset bits |
| `o_sw_sys_rst` | output | 1 | Self-clearing system reset |

### PTP clocking

| Port | Direction | Width | Notes |
|---|---|---|---|
| `i_ptp_clk` | input | 1 | 95–105 MHz band |
| `o_ptp_rst` | output | 1 | Synchronous, active-high |

---

## Macro-gated ports

### `\`ifndef ENUM_EEPROM` — soft enumeration ports (3 ports added)

When `ENUM_EEPROM` is **undefined**, the user must drive these inputs:

| Port | Direction | Width | Notes |
|---|---|---|---|
| `i_mac_addr[HOST_IF_INST-1:0][47:0]` | input | 48 × `HOST_IF_INST` | Per-host MAC address |
| `i_board_sn[55:0]` | input | 56 | Board serial number |
| `i_enum_vld` | input | 1 | Strobe signal indicating the above are valid |

When `ENUM_EEPROM` is **defined**, these ports are absent and the IP reads MAC/SN from an external EEPROM via I²C bus 0 (7-bit address `0x50`). Site: `<hsb-ip-root>/top/HOLOLINK_top.sv:45-49`.

### `\`ifdef SENSOR_RX_IF_INST` — Sensor RX ports (7 ports added)

`SENSOR_RX_IF_INST` instances of each. Site: `<hsb-ip-root>/top/HOLOLINK_top.sv:54-65`.

| Port | Direction | Width | Notes |
|---|---|---|---|
| `i_sif_rx_clk[SENSOR_RX_IF_INST-1:0]` | input | `SENSOR_RX_IF_INST` | Per-interface clock |
| `o_sif_rx_rst[SENSOR_RX_IF_INST-1:0]` | output | `SENSOR_RX_IF_INST` | Per-interface reset |
| `i_sif_axis_tvalid[SENSOR_RX_IF_INST-1:0]` | input | `SENSOR_RX_IF_INST` | |
| `i_sif_axis_tlast[SENSOR_RX_IF_INST-1:0]` | input | `SENSOR_RX_IF_INST` | |
| `i_sif_axis_tdata[SENSOR_RX_IF_INST-1:0][DATAPATH_WIDTH-1:0]` | input | `DATAPATH × N` | tdata padded to `DATAPATH_WIDTH`; unused MSBs tied 0 if `SIF_RX_WIDTH[i] < DATAPATH_WIDTH` |
| `i_sif_axis_tkeep[SENSOR_RX_IF_INST-1:0][DATAKEEP_WIDTH-1:0]` | input | `DATAKEEP × N` | tkeep LSBs must be all 1's per HSB documentation |
| `i_sif_axis_tuser[SENSOR_RX_IF_INST-1:0][DATAUSER_WIDTH-1:0]` | input | `DATAUSER × N` | |
| `o_sif_axis_tready[SENSOR_RX_IF_INST-1:0]` | output | `SENSOR_RX_IF_INST` | |

### `\`ifdef SENSOR_TX_IF_INST` — Sensor TX ports (7 ports added)

`SENSOR_TX_IF_INST` instances of each. Site: `<hsb-ip-root>/top/HOLOLINK_top.sv:67-77`.

| Port | Direction | Width | Notes |
|---|---|---|---|
| `i_sif_tx_clk[SENSOR_TX_IF_INST-1:0]` | input | `SENSOR_TX_IF_INST` | |
| `o_sif_tx_rst[SENSOR_TX_IF_INST-1:0]` | output | `SENSOR_TX_IF_INST` | |
| `o_sif_axis_tvalid[SENSOR_TX_IF_INST-1:0]` | output | `SENSOR_TX_IF_INST` | |
| `o_sif_axis_tlast[SENSOR_TX_IF_INST-1:0]` | output | `SENSOR_TX_IF_INST` | |
| `o_sif_axis_tdata[SENSOR_TX_IF_INST-1:0][DATAPATH_WIDTH-1:0]` | output | `DATAPATH × N` | |
| `o_sif_axis_tkeep[SENSOR_TX_IF_INST-1:0][DATAKEEP_WIDTH-1:0]` | output | `DATAKEEP × N` | |
| `o_sif_axis_tuser[SENSOR_TX_IF_INST-1:0][DATAUSER_WIDTH-1:0]` | output | `DATAUSER × N` | |
| `i_sif_axis_tready[SENSOR_TX_IF_INST-1:0]` | input | `SENSOR_TX_IF_INST` | |

### `\`ifdef SPI_INST` — SPI peripheral ports (5 ports added)

`SPI_INST` instances of each. Site: `<hsb-ip-root>/top/HOLOLINK_top.sv:104-110`.

| Port | Direction | Width | Notes |
|---|---|---|---|
| `o_spi_csn[SPI_INST-1:0]` | output | `SPI_INST` | Active-low chip select |
| `o_spi_sck[SPI_INST-1:0]` | output | `SPI_INST` | |
| `o_spi_oen[SPI_INST-1:0]` | output | `SPI_INST` | Output enable for SDIO (tristate) |
| `o_spi_sdio[SPI_INST-1:0][3:0]` | output | 4 × `SPI_INST` | Quad-SPI data |
| `i_spi_sdio[SPI_INST-1:0][3:0]` | input | 4 × `SPI_INST` | |

### `\`ifdef I2C_INST` — I²C peripheral ports (4 ports added)

`I2C_INST` instances of each. Site: `<hsb-ip-root>/top/HOLOLINK_top.sv:112-117`.

| Port | Direction | Width | Notes |
|---|---|---|---|
| `i_i2c_scl[I2C_INST-1:0]` | input | `I2C_INST` | |
| `i_i2c_sda[I2C_INST-1:0]` | input | `I2C_INST` | |
| `o_i2c_scl_en[I2C_INST-1:0]` | output | `I2C_INST` | Open-drain enable for SCL |
| `o_i2c_sda_en[I2C_INST-1:0]` | output | `I2C_INST` | Open-drain enable for SDA |

### `\`ifdef UART_INST` — UART peripheral ports (5 ports added)

`UART_INST` is currently always 1 when defined. Site: `<hsb-ip-root>/top/HOLOLINK_top.sv:119`.

| Port | Direction | Width | Notes |
|---|---|---|---|
| `o_uart_tx` | output | 1 | |
| `i_uart_rx` | input | 1 | |
| `o_uart_busy` | output | 1 | |
| `i_uart_cts` | input | 1 | Optional flow control |
| `o_uart_rts` | output | 1 | Optional flow control |

### `\`ifndef EXT_PTP` — internal PTP outputs

When `EXT_PTP` is **undefined** (default), the IP generates PTP timestamps internally:

| Port | Direction | Width | Notes |
|---|---|---|---|
| `o_ptp_sec[47:0]` | output | 48 | PTP1588-2019 v2 |
| `o_ptp_nanosec[31:0]` | output | 32 | |
| `o_pps` | output | 1 | Pulse-per-second |

### `\`ifdef EXT_PTP` — external PTP inputs

When `EXT_PTP` is **defined**, the user supplies PTP timestamps:

| Port | Direction | Width | Notes |
|---|---|---|---|
| `i_ptp_sec[47:0]` | input | 48 | |
| `i_ptp_nanosec[31:0]` | input | 32 | |

(`o_pps` is *not* exposed in the external-PTP mode; the user generates PPS themselves.)

Site: `<hsb-ip-root>/top/HOLOLINK_top.sv:141, 599`.

---

## Macros that change *behavior* but not the port list

These macros affect internal IP behavior without adding or removing ports:

| Macro | Effect |
|---|---|
| `SYNC_CLK_HIF_APB` | When defined, declares HIF and APB clocks are synchronous; the IP uses tighter CDC paths. Otherwise async CDC. Site: `<hsb-ip-root>/top/HOLOLINK_top.sv:154-157, 353, 417`. |
| `SYNC_CLK_HIF_PTP` | Same pattern for HIF / PTP clock relationship. Site: `<hsb-ip-root>/top/HOLOLINK_top.sv:160-163, 605`. |
| `PERI_RAM_DEPTH` | Overrides peripheral block RAM depth in SPI and I²C blocks. Site: `<hsb-ip-root>/top/HOLOLINK_top.sv:958, 1008`. |
| `DISABLE_COE` | Disables Camera-over-Ethernet packet generation. Site: `<hsb-ip-root>/top/HOLOLINK_top.sv:1435`. |
| `SIF_RX_DATA_GEN` | When defined, instantiates a per-port `data_gen` module on each Sensor RX interface, each with its own APB registers. The data_gen's runtime `data_gen_ena` register defaults to 0 — external sensor data flows by default; the host enables test injection per port at runtime via APB. Sites: `<hsb-ip-root>/top/HOLOLINK_top.sv:1276` (instantiation), `:1350-1354` (per-port mux), `:1411` (back-pressure on enable), `data_gen.sv:272` (`o_data_gen_axis_mux = data_gen_ena`). |
| `N_INIT_REG` | When defined, instantiates the `sys_init` boot-write FSM. Site: `<hsb-ip-root>/top/HOLOLINK_top.sv:479-481`. |

## Module parameters (not macros)

These are passed at instantiation, not defined in `HOLOLINK_def.svh`:

| Parameter | Width | Notes |
|---|---|---|
| `BUILD_REV` | 48 | Build revision identifier; default `48'h0`. Pass via `HOLOLINK_top #(.BUILD_REV(48'h…)) u_…`. Site: `<hsb-ip-root>/top/HOLOLINK_top.sv:24`. |
