# `HOLOLINK_def.svh` Macro Reference

**Known-revision reference.** Where this file conflicts with the public documentation, follow this file. Conflicts are noted inline.

If live HSB IP source is available, use it as the authority for the checked-out design. This file documents known rev `16'h2604` and backward-compatible rev `16'h2603`; live `HOLOLINK_top.sv` wins for newer or locally modified IP.

## Public-doc baseline

The closest public reference is the **HSB user guide**:

- Macros: `docs/user_guide/ip_integration.md`
- Ports: `docs/user_guide/port_description.md`

Repo: `https://github.com/nvidia-holoscan/holoscan-sensor-bridge/tree/release-2.6.0-EA`

The HSB documentation includes example macro values. They are not closed allow-sets. The skill enforces hard RTL/legal constraints such as byte-alignment, derived-width equality, and documented frequency bands; it does not warn merely because a legal value differs from bundled examples.

## IP-root layout

The HSB IP RTL ships under at least two known directory roots:

| Layout | Root path |
|---|---|
| Public release | `fpga/nv_hsb_ip/` |
| NVIDIA-internal workspace | `hw/nvcpu_dgx_fpga/vrtl/hololink/` |

Both roots contain identical subdirectories (`top/`, `lib_axis/`, `packetizer/`, `sys_init/`, `ptp/`, `bootp/`, etc.). All RTL citations in this file use the form `<hsb-ip-root>/<subdir>/<file>.sv:<line>` — substitute your root.

## IP version compatibility

| Constant | Value | Site |
|---|---|---|
| `HOLOLINK_REV` (current) | `16'h2604` | `top/HOLOLINK_top.sv:151` |
| `HOLOLINK_BACKWARD_COMPAT_REV` | `16'h2603` | `top/HOLOLINK_top.sv:152` |

This file is written against `16'h2604`. It is backward-compatible with `16'h2603` (the public release rev).

For an unknown or newer live revision, read `top/HOLOLINK_top.sv` before making source-sensitive claims. If the live source changes consumed macros, conditional port gates, or macro-to-RTL behavior, trust the live source and flag that this reference may need an update.

## File wrapper requirement

`HOLOLINK_def.svh` must be wrapped in a guard + package:

```systemverilog
`ifndef HOLOLINK_def
`define HOLOLINK_def

package HOLOLINK_pkg;

  // ... `define directives and localparam arrays ...

endpackage: HOLOLINK_pkg
`endif
```

The wrapper is required: `top/HOLOLINK_top.sv:17` does `` `include "HOLOLINK_def.svh" `` and `import HOLOLINK_pkg::*;`. A file missing the wrapper will fail elaboration.

## Macro index

| # | Macro | Kind | Section |
|---|---|---|---|
| 1 | `HIF_CLK_FREQ` | `\`define` | Clocking |
| 2 | `APB_CLK_FREQ` | `\`define` | Clocking |
| 3 | `PTP_CLK_FREQ` | `\`define` | Clocking |
| 4 | `UUID` | `\`define` | Board identity |
| 5 | `ENUM_EEPROM` | `\`define` (toggle) | Enumeration |
| 6 | `EEPROM_REG_ADDR_BITS` | `\`define` | Enumeration |
| 7 | `DATAPATH_WIDTH` | `\`define` | Sensor datapath |
| 8 | `DATAKEEP_WIDTH` | `\`define` (derived) | Sensor datapath |
| 9 | `DATAUSER_WIDTH` | `\`define` | Sensor datapath |
| 10 | `SENSOR_RX_IF_INST` | `\`define` (gates ports) | Sensor RX |
| 11 | `SIF_RX_WIDTH[]` | `localparam` array | Sensor RX |
| 12 | `SIF_RX_DATA_GEN` | `\`define` (toggle) | Sensor RX |
| 13 | `SIF_RX_PACKETIZER_EN[]` | `localparam` array | Sensor RX |
| 14 | `SIF_RX_VP_COUNT[]` | `localparam` array (do not change) | Sensor RX |
| 15 | `SIF_RX_SORT_RESOLUTION[]` | `localparam` array (do not change) | Sensor RX |
| 16 | `SIF_RX_VP_SIZE[]` | `localparam` array (do not change) | Sensor RX |
| 17 | `SIF_RX_NUM_CYCLES[]` | `localparam` array (do not change) | Sensor RX |
| 18 | `SENSOR_TX_IF_INST` | `\`define` (gates ports) | Sensor TX |
| 19 | `SIF_TX_WIDTH[]` | `localparam` array | Sensor TX |
| 20 | `SIF_TX_BUF_SIZE[]` | `localparam` array | Sensor TX |
| 21 | `HOST_WIDTH` | `\`define` | Host |
| 22 | `HOSTKEEP_WIDTH` | `\`define` (derived) | Host |
| 23 | `HOSTUSER_WIDTH` | `\`define` | Host |
| 24 | `HOST_IF_INST` | `\`define` | Host |
| 25 | `HOST_MTU` | `\`define` | Host |
| 26 | `SPI_INST` | `\`define` (gates ports) | Peripherals |
| 27 | `I2C_INST` | `\`define` (gates ports) | Peripherals |
| 28 | `UART_INST` | `\`define` (gates ports) | Peripherals |
| 29 | `GPIO_INST` | `\`define` | Peripherals |
| 30 | `GPIO_RESET_VALUE` | `localparam` | Peripherals |
| 31 | `EXT_PTP` | `\`define` (gates ports) | PTP |
| 32 | `SYNC_CLK_HIF_APB` | `\`define` (toggle) | Clock sync (not in public docs) |
| 33 | `SYNC_CLK_HIF_PTP` | `\`define` (toggle) | Clock sync (not in public docs) |
| 34 | `REG_INST` | `\`define` | User registers |
| 35 | `N_INIT_REG` | `\`define` | System init |
| 36 | `init_reg[]` | `localparam` array | System init |
| 37 | `PERI_RAM_DEPTH` | `\`define` (override) | Peripherals (not in public docs) |
| 38 | `DISABLE_COE` | `\`define` (toggle) | Dataplane (not in public docs) |
| — | `BUILD_REV` | **Module parameter** (not a `\`define`) | Versioning |

---

# Clocking

## `HIF_CLK_FREQ`

Host Interface clock frequency, in Hz.

| | |
|---|---|
| Kind | `\`define` (integer) |
| Documented examples | `156250000` (with 64-bit datapath), `201416016` (with 512-bit datapath) |
| Hard rules | > 0 |
| Soft rules | Power-of-2 not required; many real configs use non-power-of-2 frequencies derived from transceiver PLLs |
| RTL sites | `top/HOLOLINK_top.sv:601` (passed to `ptp_top`); `ptp/ptp_timer.sv:19,52,55,106` (used in nanosecond increment + CDC depth calc) |
| Derives | `default_hif_inc = 10**9 * (2**W_FRAC_NS) / HIF_CLK_FREQ` (`ptp/ptp_timer.sv:55`); CDC buffer depth via `PTP_IS_LOWER_FREQ = PTP_CLK_FREQ < HIF_CLK_FREQ` (`ptp/ptp_timer.sv:106`) |
| Interacts with | `PTP_CLK_FREQ` (CDC ratio decides asymmetric latency at `ptp/ptp_timer.sv:106-119`) |

## `APB_CLK_FREQ`

APB Interface clock frequency, in Hz. Used to compute I²C clock divider when `ENUM_EEPROM` is defined.

| | |
|---|---|
| Kind | `\`define` (integer) |
| Documented examples | `19531250` (with 64-bit datapath), `100000000` (with 512-bit datapath) |
| Hard rules | > 0 |
| RTL sites | `top/HOLOLINK_top.sv:732` (passed to EEPROM info module when `ENUM_EEPROM` defined); `ptp/ptp_top.sv:927` (CDC) |
| Interacts with | `ENUM_EEPROM` (when defined, `APB_CLK_FREQ` sets the I²C baud divider) |

## `PTP_CLK_FREQ`

PTP Interface clock frequency, in Hz.

| | |
|---|---|
| Kind | `\`define` (integer) |
| HSB-documented range | "95 MHz to 105 MHz" |
| Documented examples | `100446545`, `100000000`, `100707500`, `78125000` (varies by board) |
| Hard rules | > 0 |
| Soft rules | Should be in the 95–105 MHz band per HSB documentation; out-of-band warns |
| RTL sites | `top/HOLOLINK_top.sv:602,912-913,926-927`; `ptp/ptp_timer.sv:20,47,50,106` |
| Derives | `default_inc = 10**9 * (2**W_FRAC_NS) / PTP_CLK_FREQ` (`ptp/ptp_timer.sv:50`); `PTP_IS_LOWER_FREQ` flag controls CDC latency compensation |
| Interacts with | `HIF_CLK_FREQ` (CDC ratio); `EXT_PTP` (when `EXT_PTP` is defined the user supplies PTP timestamps externally and `PTP_CLK_FREQ` only affects internal counter scaling) |

---

# Board identity

## `UUID`

Universally unique identifier used in BOOTP enumeration to identify the board for bitfile flashing.

| | |
|---|---|
| Kind | `\`define` literal |
| Width | 128 bits |
| Format | `128'h<32 hex chars, optionally with `_` separators>` |
| Hard rules | Required; must be exactly 128 bits; must be a literal (not a derived expression) |
| Soft rules | Should be unique per board. The generator inserts a random UUID by default; pin a specific value by adding `uuid:` to your profile. |
| RTL sites | `top/HOLOLINK_top.sv:416` (passed to `rx_ls_parser`); `bootp/bootp.sv:23,235-239` (byte-swapped for network transmission) |

---

# Enumeration

## `ENUM_EEPROM`

Toggle. When defined, the IP reads MAC address and serial number from an external EEPROM over I²C bus 0 (7-bit address `0x50`). When undefined, the user must drive `i_mac_addr[]`, `i_board_sn`, and `i_enum_vld` ports.

| | |
|---|---|
| Kind | `\`define` toggle (defined / undefined) |
| RTL sites | `top/HOLOLINK_top.sv:45` (gates the soft-enumeration input ports — `\`ifndef ENUM_EEPROM`); `top/HOLOLINK_top.sv:729` (instantiates `eeprom_info` when defined) |
| Effect when defined | `EEPROM_REG_ADDR_BITS` becomes meaningful; soft-enumeration ports are absent |
| Effect when undefined | Top module exposes `i_mac_addr[HOST_IF_INST-1:0][47:0]`, `i_board_sn[55:0]`, `i_enum_vld` for the user to drive |

## `EEPROM_REG_ADDR_BITS`

EEPROM register address width.

| | |
|---|---|
| Kind | `\`define` (integer) |
| Allowed values (HSB documentation) | `8` or `16` |
| Hard rules | Must be `8` or `16`; only valid when `ENUM_EEPROM` is defined |
| Footgun | Defining `EEPROM_REG_ADDR_BITS` without `ENUM_EEPROM` is silently ignored — the macro is dead unless `ENUM_EEPROM` is also defined. The validator warns. |
| RTL sites | `top/HOLOLINK_top.sv:733` (passed as `REG_ADDR_BITS` parameter to `eeprom_info`) |

---

# Sensor datapath

## `DATAPATH_WIDTH`

Width of the Sensor AXI-Stream `tdata` bus, in bits. The system-wide maximum sensor datapath width.

| | |
|---|---|
| Kind | `\`define` (integer) |
| Documented examples | `8, 16, 32, 64, 128, 256, 512, 1024` |
| Hard rules | > 0; byte-aligned (divisible by 8) |
| Soft rules | Power-of-2 (AXI bus convention; warned if violated); > 1024 warns about logic and routing resource use |
| RTL sites | `top/HOLOLINK_top.sv:61-76,169,1082,1168,1300-1302`; `lib_axis/axis_pkg.sv:42` (sensor byte-swap is fully parameterized over `DATAPATH_WIDTH`) |
| Derives | `DATAKEEP_WIDTH = DATAPATH_WIDTH/8` (mandatory equality) |
| Interacts with | `SIF_RX_WIDTH[i] ≤ DATAPATH_WIDTH` and `SIF_TX_WIDTH[i] ≤ DATAPATH_WIDTH` (per-port subsets) |

## `DATAKEEP_WIDTH`

Width of the Sensor AXI-Stream `tkeep` bus, in bits.

| | |
|---|---|
| Kind | `\`define` (derived) |
| Hard rule | Must equal `DATAPATH_WIDTH/8` |
| RTL sites | `top/HOLOLINK_top.sv:62,75,1302`; `lib_axis/axis_pkg.sv:51,59` |

## `DATAUSER_WIDTH`

Width of the Sensor AXI-Stream `tuser` sideband signal, in bits. The HSB `dataplane.md` doc describes the per-bit semantics **only for MIPI CSI-2 camera** datapaths feeding an internal ISP:

- **`i_sif_axis_tuser[0]`** — asserted during cycles containing **embedded data** (MIPI Data Type `0x12`).
- **`i_sif_axis_tuser[1]`** — asserted on the **final clock cycle of a long packet** (Line End signal).

The doc explicitly notes these signals are *"used only for high-bandwidth camera with need for internal Image Signal Processing (ISP). For further details, please contact the NVIDIA Holoscan team."* It does **not** define `tuser` semantics for non-MIPI sensor sources, and does not state a "default" or "typical" choice.

| | |
|---|---|
| Kind | `\`define` (integer) |
| HSB-documented range | `1` or `2` |
| Hard rules | > 0; ≤ 2 |
| When `DATAUSER_WIDTH = 1` | Only `tuser[0]` is present. `tuser[1]` (Line End marker) is not available. For a MIPI CSI-2 path that doesn't need the Line End marker, this carries the embedded-data marker. For non-MIPI sensors, the doc doesn't define what `tuser[0]` carries — interpretation is between the user's sensor and downstream logic. |
| When `DATAUSER_WIDTH = 2` | Both `tuser[0]` (embedded-data) and `tuser[1]` (Line End) markers present. Required when a MIPI CSI-2 path with internal ISP uses both markers. |
| Picking a value | Determined by your design: do you need the Line End marker (or both MIPI markers)? If yes → 2. If no, or if non-MIPI → 1. The doc does not declare one of these "standard." |
| RTL sites | `top/HOLOLINK_top.sv:63,76,1375` |
| Doc reference | HSB user guide — `docs/user_guide/dataplane.md` |

---

# Sensor RX

## `SENSOR_RX_IF_INST`

Number of Sensor RX (sensor → host) interfaces. Defining or undefining this macro gates an entire group of ports on the top module.

| | |
|---|---|
| Kind | `\`define` (integer, OR undefined) |
| HSB-documented range | `1`–`32`, OR undefined |
| Hard rules | When defined, `1 ≤ value ≤ 32` |
| Footgun | When undefined, RTL silently substitutes a 1-bit dummy interface — there is no compile error. The validator warns when an arrays exists (`SIF_RX_WIDTH` etc.) but `SENSOR_RX_IF_INST` is undefined. |
| RTL sites | `top/HOLOLINK_top.sv:54-65` (gates input ports `i_sif_rx_clk`, `i_sif_axis_*`, etc.); `top/HOLOLINK_top.sv:169-176,181-183` (defines `NUM_SENSOR_RX` localparam: macro value when defined, `1` otherwise) |
| Generates | All `SIF_RX_*[]` arrays are sized `[SENSOR_RX_IF_INST-1:0]` |

## `SIF_RX_WIDTH[]`

Per-RX-interface AXI-Stream tdata width, in bits.

| | |
|---|---|
| Kind | `localparam integer` array, sized `[SENSOR_RX_IF_INST-1:0]` |
| Documented examples | `8, 16, 32, 64, 128, 256, 512, 1024` |
| Hard rules | > 0; byte-aligned; `≤ DATAPATH_WIDTH` for every element |
| Soft rules | Power-of-2 (warned if violated); > 1024 warns about logic and routing resources |
| Default expression | `'{default:`DATAPATH_WIDTH}` — every interface uses the full datapath width unless overridden |
| RTL sites | `top/HOLOLINK_top.sv:1087,1099,1102,1283,1300-1302,1373,1396-1397` |
| Padding rule | When `SIF_RX_WIDTH[i] != DATAPATH_WIDTH`, the unused MSBs of tdata and the unused upper tkeep bits are tied to 0; LSBs of tkeep must still be set to all 1's per the HSB `ip_integration.md` |

## `SIF_RX_DATA_GEN`

Compile-time toggle for the **per-port test-pattern injection capability**. When defined, the IP instantiates a `data_gen` module per Sensor RX interface, each with its own APB register interface. The data_gen has an internal `data_gen_ena` register (defaults to `0` at reset) that the host software writes via APB at runtime to enable test injection per port. When enabled on a port, that port's input mux selects the generator output instead of `i_sif_axis_*`, and the IP asserts back-pressure to the external sensor (`o_sif_axis_tready[i] = 0`). When the register is `0` (the reset state), external sensor data flows. When the macro is **undefined**, the data_gen modules are not instantiated, the mux select is tied to 0, and only external sensor data ever flows.

| | |
|---|---|
| Kind | `\`define` toggle |
| Behavior when defined | data_gen modules instantiated per RX port. Real sensor data flows by default (`data_gen_ena = 0` at reset). Host enables injection per port at runtime via APB writes. |
| Behavior when undefined | No data_gen modules instantiated (saves logic + embedded RAM). Real sensor data always flows. No runtime test-injection capability. |
| RTL sites — instantiation | `top/HOLOLINK_top.sv:1276` (`\`ifdef SIF_RX_DATA_GEN`); `top/HOLOLINK_top.sv:1280-1306` (per-port `data_gen` generate block); `data_gen/data_gen.sv:272` (`o_data_gen_axis_mux = data_gen_ena` — the runtime mux select) |
| RTL sites — fallback | `top/HOLOLINK_top.sv:1308-1326` (`\`else` branch ties data_gen outputs and mux to 0) |
| RTL sites — mux | `top/HOLOLINK_top.sv:1350-1354` (per-port mux: `sen_rx_mux_axis_* = data_gen_axis_mux ? data_gen_output : i_sif_axis_*`); `top/HOLOLINK_top.sv:1411` (back-pressure: `o_sif_axis_tready[i] = data_gen_axis_mux ? 0 : sif_axis_tready[i]`) |
| When to use | Define for **boards that benefit from in-system test/diagnostics or simulation-driven bringup**. Some teams keep it on in production; others remove it to save logic and embedded RAM. Defining it does **not** disable real sensor data — it only adds the runtime-controllable injection capability. |

## `SIF_RX_PACKETIZER_EN[]`

Per-RX-interface packetizer enable flag.

| | |
|---|---|
| Kind | `localparam integer` array, sized `[SENSOR_RX_IF_INST-1:0]` |
| Allowed values | `0` or `1` per element |
| Default expression | `'{default:'1}` (all enabled) |
| Effect when `0` for interface `i` | The four "do not change" parameters (`SIF_RX_VP_COUNT[i]`, `SIF_RX_SORT_RESOLUTION[i]`, `SIF_RX_VP_SIZE[i]`, `SIF_RX_NUM_CYCLES[i]`) become don't-cares. The packetizer is bypassed; `top/HOLOLINK_top.sv:1357-1362` substitutes fallback constants `(SORT_RESOLUTION=2, VP_COUNT=1, VP_SIZE=32, NUM_CYCLES=1)`. |
| RTL sites | `top/HOLOLINK_top.sv:1243,1300,1342,1356-1362,1382,1414` |

## `SIF_RX_VP_COUNT[]`, `SIF_RX_SORT_RESOLUTION[]`, `SIF_RX_VP_SIZE[]`, `SIF_RX_NUM_CYCLES[]`

Packetizer internal parameters. **Do not change** per HSB documentation. Required when any element of `SIF_RX_PACKETIZER_EN[]` is `1`; the corresponding element is a don't-care when its `EN[i]` is `0`.

| Macro | Kind | Example values | Effect |
|---|---|---|---|
| `SIF_RX_VP_COUNT[]` | `localparam integer` array | `1`, `2`, `4` | Number of virtual ports per RX interface |
| `SIF_RX_SORT_RESOLUTION[]` | `localparam integer` array | `2`, `16`, `32`, `DATAPATH_WIDTH` | Packetizer sort resolution |
| `SIF_RX_VP_SIZE[]` | `localparam integer` array | `64`, `128`, `256`, `DATAPATH_WIDTH` | Bytes per virtual port |
| `SIF_RX_NUM_CYCLES[]` | `localparam integer` array | `1`, `3` | Cycle count |
| RTL sites (all four) | | | `top/HOLOLINK_top.sv:1356-1362` (consumed via ternary on `SIF_RX_PACKETIZER_EN[i]`) |

The validator does not enforce specific values for these; it only enforces presence (when packetizers are enabled) and array length consistency.

---

# Sensor TX

## `SENSOR_TX_IF_INST`

Number of Sensor TX (host → sensor) interfaces. Defining or undefining gates the TX port group.

| | |
|---|---|
| Kind | `\`define` (integer, OR undefined) |
| HSB-documented range | `1`–`32`, OR undefined |
| Hard rules | When defined, `1 ≤ value ≤ 32` |
| RTL sites | `top/HOLOLINK_top.sv:67-77` (gates output ports); `top/HOLOLINK_top.sv:193-200,205-207` (defines `NUM_SENSOR_TX` localparam: macro value when defined, `0` otherwise) |
| Footgun | Same silent-fallback as `SENSOR_RX_IF_INST` |

## `SIF_TX_WIDTH[]`

Per-TX-interface AXI-Stream tdata width.

| | |
|---|---|
| Kind | `localparam integer` array |
| Documented examples | `8, 64, 512` |
| Hard rules | > 0; byte-aligned; `≤ DATAPATH_WIDTH` |
| Soft rules | Power-of-2; > 1024 warns about logic and routing resources |
| Default expression | `'{default:`DATAPATH_WIDTH}` |
| RTL sites | `top/HOLOLINK_top.sv:1169,1578-1579,1586-1587,1608-1611,1735,1747-1748` |

## `SIF_TX_BUF_SIZE[]`

Per-TX-interface FIFO depth, measured as a count of `SIF_TX_WIDTH[i]`-wide elements, not bytes. Backed by FPGA embedded RAM. The buffer primarily lets the IP absorb host-to-sensor data and apply backpressure toward the host when the sensor side cannot accept data immediately; larger values allow more TX-width elements to be stored at the cost of embedded RAM.

| | |
|---|---|
| Kind | `localparam integer` array |
| Documented examples | `1024, 2048, 4096` TX-width elements |
| Default value | `2048` TX-width elements |
| Hard rules | > 0 |
| Soft rules | Power-of-2; > 8192 warns about embedded RAM resource use |
| RTL sites | `top/HOLOLINK_top.sv:1587` (passed as `FIFO_DEPTH` to `axis_buffer_tx`) |

---

# Host

## `HOST_WIDTH`

Width of the Host AXI-Stream `tdata` bus, in bits. This is one scalar width shared by every host interface selected by `HOST_IF_INST`; the IP does not support per-host `HOST_WIDTH[]` values. The value is determined by the host-side Ethernet datapath connected to HSB and the line-rate/clock pairing implemented by the user's MAC, transceiver, and FPGA clocking chain.

| | |
|---|---|
| Kind | `\`define` (integer) |
| Documented examples | `8, 16, 32, 64, 128, 256, 512` |
| Hard rules | > 0; byte-aligned |
| Soft rules | Power-of-2 (warned if violated); > 1024 warns about logic and routing resources |
| RTL sites | `top/HOLOLINK_top.sv:90,97,349,409,603,1082,1168,1254,1367,1374,1700,1741`; `lib_axis/axis_pkg.sv:24,33,68` (host byte-swap and decode are fully parameterized over `HOST_WIDTH`; the helper functions impose no per-value restriction) |
| Derives | `HOSTKEEP_WIDTH = HOST_WIDTH/8`; `HOST_BUF_DEPTH = HOST_MTU * 2 / (HOST_WIDTH / 8)` (`top/HOLOLINK_top.sv:1082`) |

**Documented Ethernet line-rate / `HOST_WIDTH` / `HIF_CLK_FREQ` pairings:**

| Line rate | `HOST_WIDTH` | `HIF_CLK_FREQ` | Matching archetype |
|---|---|---|---|
| **1 GbE** | `8` | `125_000_000` (125 MHz) | `gigabit-baseline` |
| **10 GbE** | `64` | `156_250_000` (156.25 MHz) | `mid-bandwidth-baseline`, `minimal-gateway` |
| **25 GbE / 40 GbE** | `128` / `256` | intermediate (~390 MHz / ~156 MHz) | — |
| **100 GbE (wide)** | `512` | ~`201_416_016` (~201 MHz) | `high-bandwidth-single-sensor` |
| **100 GbE+ (very high)** | `512` | ~`322_265_625` (~322 MHz) | `very-high-speed` |

Set `HOST_WIDTH` to the single AXI-Stream width shared by all HSB host interfaces. Set `HIF_CLK_FREQ` to the frequency of the clock actually driving that HSB host interface; it may be driven directly by the MAC or derived through a PLL/divider from another MAC/transceiver clock. Other byte-aligned widths are legal for the shared width; validation focuses on hard constraints and resource-risk thresholds.

## `HOSTKEEP_WIDTH`

| | |
|---|---|
| Kind | `\`define` (derived) |
| Hard rule | Must equal `HOST_WIDTH/8` |
| RTL sites | `top/HOLOLINK_top.sv:91,98`; `lib_axis/axis_pkg.sv:33,36,68` |

## `HOSTUSER_WIDTH`

Width of the Host AXI-Stream `tuser` signal.

| | |
|---|---|
| Kind | `\`define` (integer) |
| HSB-documented value | `1` (fixed) |
| Hard rules | Must be `1` |
| RTL sites | `top/HOLOLINK_top.sv:92,99` |

## `HOST_IF_INST`

Number of Host (Ethernet) interfaces.

| | |
|---|---|
| Kind | `\`define` (integer) |
| HSB-documented range | `1`–`32` |
| Hard rules | `1 ≤ value ≤ 32` |
| RTL sites | `top/HOLOLINK_top.sv:34,40,46,88-100,237-238,261,344` (sizes per-host signal arrays) |

## `HOST_MTU`

Maximum Ethernet packet payload, in bytes. **Applies to both directions** (ingress and egress).

| | |
|---|---|
| Kind | `\`define` (integer) |
| Documented examples | `1500` (standard), `4096` (jumbo) |
| Hard rules | > 0 |
| Soft rules | The IP buffer scales via `HOST_BUF_DEPTH = HOST_MTU * 2 / (HOST_WIDTH/8)`; larger MTUs cost more embedded RAM |
| RTL sites — RX (ingress) | `eth_pkt/eth_pkt.sv:240` (host RX `pkt_check`); `ptp/ptp_ingress.sv:194`; `top/HOLOLINK_top.sv:1703` (`assert_sif_input_axis`), `:1778` (`assert_hif_input_axis`) |
| RTL sites — TX (egress) | `eth_pkt/eth_pkt.sv:275` (host TX `pkt_check`); `top/HOLOLINK_top.sv:1383` (packetizer), `:1588` (TX stream buffer), `:1741` (`assert_sif_output_axis`), `:1808` (`assert_hif_output_axis`) |
| RTL sites — sizing | `top/HOLOLINK_top.sv:1082` (`HOST_BUF_DEPTH = HOST_MTU * 2 / (HOST_WIDTH/8)`); `:351, 412, 1170, 1448` (parameter passdown) |
| Practical impact | Set `HOST_MTU` to **at least the largest packet either side sends**. The RX `pkt_check` drops incoming packets larger than this value. Setting it higher than necessary wastes FPGA embedded RAM (buffer scales linearly). |

---

# Peripherals

## `SPI_INST`

Number of SPI controller instances. Defining or undefining gates the SPI port group.

| | |
|---|---|
| Kind | `\`define` (integer, OR undefined) |
| HSB-documented range | `1`–`8`, OR undefined |
| RTL sites | `top/HOLOLINK_top.sv:104-110` (gates ports `o_spi_csn[]`, `o_spi_sck[]`, `i_spi_sdio[][3:0]`, `o_spi_sdio[][3:0]`, `o_spi_oen[]`); `top/HOLOLINK_top.sv:938` (instantiates the SPI block when defined) |
| Footgun | When undefined, the SPI port group is removed entirely. Forgetting to undef when SPI is unused leaves dangling 1-bit dummy logic. |

## `I2C_INST`

Number of I²C controller instances.

| | |
|---|---|
| Kind | `\`define` (integer, OR undefined) |
| HSB-documented range | `1`–`8`, OR undefined |
| RTL sites | `top/HOLOLINK_top.sv:112-117` (gates ports); `top/HOLOLINK_top.sv:988` (instantiation) |

## `UART_INST`

Number of UART instances.

| | |
|---|---|
| Kind | `\`define` (integer, OR undefined) |
| HSB-documented range | `1`, OR undefined (only one UART is currently supported) |
| RTL sites | `top/HOLOLINK_top.sv:119` (gates ports `o_uart_tx`, `i_uart_rx`, `o_uart_busy`, `i_uart_cts`, `o_uart_rts`); `top/HOLOLINK_top.sv:1038-1040` (instantiation) |

## `GPIO_INST`

Number of bidirectional GPIO bits.

| | |
|---|---|
| Kind | `\`define` (integer) |
| HSB-documented range | `1`–`255` |
| Hard rules | `1 ≤ value ≤ 255` |
| RTL sites | `top/HOLOLINK_top.sv:128-130` (port widths `i_gpio[GPIO_INST-1:0]`, `o_gpio[GPIO_INST-1:0]`, `o_gpio_dir[GPIO_INST-1:0]`); `top/HOLOLINK_top.sv:686-687` (passed as `N_GPIO` parameter) |

## `GPIO_RESET_VALUE`

Reset value of GPIO output bits.

| | |
|---|---|
| Kind | `localparam` of width `[GPIO_INST-1:0]` |
| Default | `0` |
| Hard rules | Must be sized exactly `[GPIO_INST-1:0]` |
| RTL sites | `top/HOLOLINK_top.sv:687` (passed as `GPIO_RST_VAL`) |

---

# PTP

## `EXT_PTP`

Toggle. When defined, the user provides PTP timestamps externally via `i_ptp_sec[47:0]` and `i_ptp_nanosec[31:0]`. When undefined, the IP generates timestamps internally and exposes `o_ptp_sec[47:0]`, `o_ptp_nanosec[31:0]`, and `o_pps`.

| | |
|---|---|
| Kind | `\`define` toggle |
| RTL sites | `top/HOLOLINK_top.sv:141` (gates port direction — `\`ifndef EXT_PTP` exposes `o_ptp_*` outputs; `\`else` exposes `i_ptp_*` inputs); `top/HOLOLINK_top.sv:599` (gates internal PTP generator instantiation) |
| Effect when defined | Internal PTP generator is omitted; user must drive PTP timestamps |
| Effect when undefined | Internal PTP generator is instantiated; IP outputs PPS pulse and PTP timestamps |

---

# Clock synchronization (not in HSB public docs)

`SYNC_CLK_HIF_APB` and `SYNC_CLK_HIF_PTP` are advanced CDC-tightening toggles. See `advanced-macros.md` for canonical behavior, safety caveats, and RTL citations.

---

# User registers and system init

## `REG_INST`

Number of user-accessible APB register interfaces exposed by the IP.

| | |
|---|---|
| Kind | `\`define` (integer) |
| HSB-documented range | `1`–`8` |
| Hard rules | `1 ≤ value ≤ 8` |
| RTL sites | `top/HOLOLINK_top.sv:34,39-41,531` (sizes APB switch arrays `o_apb_psel[REG_INST-1:0]`, `i_apb_pready[REG_INST-1:0]`, `i_apb_prdata[REG_INST-1:0]`, `i_apb_pserr[REG_INST-1:0]`) |

## `N_INIT_REG`

Number of entries in the `init_reg[]` boot-time write sequence.

| | |
|---|---|
| Kind | `\`define` (integer) |
| Hard rules | > 0; must equal the actual `init_reg[]` array length |
| RTL sites | `top/HOLOLINK_top.sv:479-481` (gates `sys_init` instantiation — `\`ifdef N_INIT_REG`); `sys_init/sys_init.sv:22` (`parameter N_REG`); `sys_init/sys_init.sv:62-64` (per-entry write loop) |
| Footgun | Mismatch between `N_INIT_REG` and `init_reg` array length is an elaboration error in some tools and a silent truncation in others. Validator hard-errors. |

## `init_reg[]`

The boot-time register write sequence. The IP loops over the entries on rising edge of `i_init`, issuing APB writes; asserts `o_init_done` when complete.

| | |
|---|---|
| Kind | `localparam logic [63:0]` array, sized `[N_INIT_REG]` |
| Per-entry encoding | `[63:32]` = 32-bit APB address; `[31:0]` = 32-bit data |
| Tuple syntax | `{32'h<addr>, 32'h<data>}` per entry |
| Hard rules | Array length = `N_INIT_REG`; each entry exactly 64 bits |
| RTL sites | `top/HOLOLINK_top.sv:479-481` (passed to `sys_init`); `sys_init/sys_init.sv:30,57-58,62-64` (per-entry address/data extraction and APB write) |
| Cookbook | See `init-reg-cookbook.md` for address-space conventions |

---

# Other

## Advanced peripheral and dataplane macros

`PERI_RAM_DEPTH` and `DISABLE_COE` are advanced macros. See `advanced-macros.md` for canonical behavior, safety guidance, and RTL citations.

---

# Module parameter (not a `\`define`)

## `BUILD_REV`

48-bit Verilog parameter on the `HOLOLINK_top` module identifying the FPGA build revision. **Not** a `\`define` macro; cannot be set in `HOLOLINK_def.svh`. Pass it via parameter override at instantiation:

```systemverilog
HOLOLINK_top #(
  .BUILD_REV(48'h2604_DEAD_BEEF)
) u_hololink (
  ...
);
```

| | |
|---|---|
| Kind | Module parameter (NOT a `\`define`) |
| Width | 48 bits |
| Default | `48'h0` |
| Field encoding | `[15:0]` = patch version (packed with `HOLOLINK_REV[15:0]` to form full HSB version word); `[47:16]` = build date/timestamp (YYMMDDHH-style by team convention) |
| RTL sites | `top/HOLOLINK_top.sv:24` (parameter declaration); `top/HOLOLINK_top.sv:699-700` (used in status register: `i_hsb_ver = {BUILD_REV[15:0], HOLOLINK_REV[15:0]}`, `i_hsb_date = BUILD_REV[47:16]`) |
| Footgun | Writing `\`define BUILD_REV ...` in `HOLOLINK_def.svh` is meaningless — the macro is never read. The validator warns. |

---

# Cross-cutting rules summary

The following rules span multiple macros. Per-rule details and the canonical rule IDs (`HD-Exxx`, `HD-Wxxx`, `HD-Ixxx`) are in `validation-rules.md`.

1. **Wrapper required.** File must be wrapped in `\`ifndef HOLOLINK_def`/`\`define HOLOLINK_def`/`package HOLOLINK_pkg;` ... `endpackage; \`endif`.
2. **Byte-alignment of widths.** All width macros must be divisible by 8.
3. **Derived equalities.** `DATAKEEP_WIDTH = DATAPATH_WIDTH/8`; `HOSTKEEP_WIDTH = HOST_WIDTH/8`. Mismatch is a hard error.
4. **Per-port width caps.** `SIF_RX_WIDTH[i] ≤ DATAPATH_WIDTH`; `SIF_TX_WIDTH[i] ≤ DATAPATH_WIDTH`.
5. **`init_reg[]` cardinality.** `N_INIT_REG` must equal the literal array length of `init_reg[]`.
6. **Conditional dependencies.** `EEPROM_REG_ADDR_BITS` is only meaningful when `ENUM_EEPROM` is defined; `SIF_RX_*[]` arrays are only meaningful when `SENSOR_RX_IF_INST` is defined; `SIF_TX_*[]` arrays are only meaningful when `SENSOR_TX_IF_INST` is defined; the four packetizer "do not change" arrays are don't-cares for any element where `SIF_RX_PACKETIZER_EN[i] = 0`.
7. **PTP / HIF clock ratio.** `PTP_CLK_FREQ < HIF_CLK_FREQ` produces shorter CDC buffers (`ptp/ptp_timer.sv:106-119`); `>=` produces deeper CDC. Both work; the latter has more latency.
8. **Silent peripheral fallback.** Forgetting to define `SENSOR_RX_IF_INST`, `SENSOR_TX_IF_INST`, `SPI_INST`, `I2C_INST`, or `UART_INST` does not produce a compile error — the RTL falls back to dummy 1-bit logic. Validator emits a warning when a corresponding array (e.g. `SIF_RX_WIDTH[]`) is present but the gate macro is undefined.
9. **`BUILD_REV` is not a `\`define`.** Set it via Verilog parameter override at module instantiation.
10. **Out-of-scope macros.** A small set of macros that the IP and some workspace projects use are out of scope for this skill. The validator silently accepts them — never errors, warns, or names them. Consult your project's design owner if you encounter unfamiliar macros.
