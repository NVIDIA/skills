# Example configurations

Six example configurations distilled from real designs. **These are illustrative — not templates you must pick from.** Real designs vary continuously; use them to see legal macro combinations and as anchors for the questions in the Generate workflow. The sample defs are synthetic: edited copies of corpus material with all UUIDs replaced by placeholders, all comments stripped or rewritten as neutral guidance, and all out-of-scope macros removed.

**The UUID is board-specific.** The generator creates a random one by default; supply your own via the profile when you have one.


## Common Boilerplate

### UUID placeholder

Each sample refers back to this synthetic UUID placeholder instead of repeating it. Replace it with the board UUID through the profile, or omit `uuid:` and let the generator create one.

```systemverilog
  // Board identity placeholder.
  `define UUID 128'h0000_0000_0000_0000_0000_0000_0000_0000
```

### Init patterns

Use these notes to interpret the shortened `init_reg[]` callouts inside samples:

- **Standard two-register MAC pause init**: `N_INIT_REG = 2`, with writes to `32'h0200_0024` and `32'h0201_0024`, both using data `32'h0000_12B7`.
- **Single-register IP-rev init**: `N_INIT_REG = 1`, with write `{32'h0120_0000, 32'h0000_0001}`.
- **Single-register MAC pause init**: `N_INIT_REG = 1`, with write `{32'h0200_0024, 32'h0000_12B7}`.

For real projects, replace these with the register sequence required by the surrounding Ethernet MAC, PCS, or board-control IP. See `init-reg-cookbook.md` for address-space conventions.

## Example index

| # | Example | Use case | DATAPATH_WIDTH | HIF_CLK_FREQ | Sensor RX | Sensor TX | Host IF | MTU |
|---|---|---|---|---|---|---|---|---|
| 1 | **High-bandwidth single-sensor** | One very-fast sensor, redundant host links | 512 | ~201 MHz | 1 | 1 | 2 | 4096 |
| 2 | **Mid-bandwidth baseline** | Multiple moderate-speed sensors, dual host | 64 | 156.25 MHz | 2 | 1 | 2 | 4096 |
| 3 | **Minimal gateway** | Edge device, single sensor, minimal peripherals | 64 | 156.25 MHz | 1 | — | 1 | 1500 |
| 4 | **Ultra-minimal** | Small embedded target, TX-only | 8 | 25 MHz | — | 1 | 1 | 1500 |
| 5 | **Very-high-speed** | Single ultra-fast sensor, single host | 512 | ~322 MHz | 1 | 1 | 1 | 4096 |
| 6 | **Gigabit baseline** | 1 GbE host, single sensor, common embedded camera/sensor designs | 8 | 125 MHz | 1 | 1 | 1 | 1500 |

Most real designs don't fall cleanly onto one of these. Use them to recognize patterns, not to box your design in. The validator returns `inferred_archetype: null` for non-matching designs; the file still passes.

---

# 1. High-bandwidth single-sensor

**When to use.** One very-fast sensor (e.g. high-resolution image, JESD204B aggregate, RF-conversion stream) feeding two redundant host interfaces. Datapath is wide (512 bits) to keep the per-cycle byte rate low; HIF clock is ~201 MHz to match a typical 100 GbE PHY clock.

**Distinctive macros.**

- `HIF_CLK_FREQ = 201_416_016`
- `APB_CLK_FREQ = 100_000_000`
- `DATAPATH_WIDTH = 512`, `DATAUSER_WIDTH = 1`
- `SENSOR_RX_IF_INST = 1`, `SENSOR_TX_IF_INST = 1`
- `HOST_WIDTH = 512`, `HOST_IF_INST = 2`, `HOST_MTU = 4096`
- `ENUM_EEPROM` defined

**Sample.**

```systemverilog
`ifndef HOLOLINK_def
`define HOLOLINK_def

package HOLOLINK_pkg;

  // Clocking
  `define HIF_CLK_FREQ  201416016    // Host interface clock, Hz
  `define APB_CLK_FREQ  100000000    // APB clock, Hz
  `define PTP_CLK_FREQ  100707500    // PTP clock, Hz (95–105 MHz band)

  // Insert UUID block from Common Boilerplate.

  // Enumeration — read MAC/SN from external EEPROM on I2C bus 0, addr 0x50
  `define ENUM_EEPROM
  `ifdef ENUM_EEPROM
    `define EEPROM_REG_ADDR_BITS 8
  `endif

  // Sensor datapath
  `define DATAPATH_WIDTH  512
  `define DATAKEEP_WIDTH  `DATAPATH_WIDTH/8
  `define DATAUSER_WIDTH  1

  // Sensor RX
  `define SENSOR_RX_IF_INST  1
  `ifdef SENSOR_RX_IF_INST
    localparam integer SIF_RX_WIDTH         [`SENSOR_RX_IF_INST-1:0] = '{default:`DATAPATH_WIDTH};
    localparam integer SIF_RX_PACKETIZER_EN [`SENSOR_RX_IF_INST-1:0] = '{default:1};
    localparam integer SIF_RX_VP_COUNT      [`SENSOR_RX_IF_INST-1:0] = {4};
    localparam integer SIF_RX_SORT_RESOLUTION [`SENSOR_RX_IF_INST-1:0] = {16};
    localparam integer SIF_RX_VP_SIZE       [`SENSOR_RX_IF_INST-1:0] = {128};
    localparam integer SIF_RX_NUM_CYCLES    [`SENSOR_RX_IF_INST-1:0] = {1};
  `endif

  // Sensor TX
  `define SENSOR_TX_IF_INST  1
  `ifdef SENSOR_TX_IF_INST
    localparam integer SIF_TX_WIDTH    [`SENSOR_TX_IF_INST-1:0] = '{default:`DATAPATH_WIDTH};
    localparam integer SIF_TX_BUF_SIZE [`SENSOR_TX_IF_INST-1:0] = '{default:2048};
  `endif

  // Host interface
  `define HOST_WIDTH      512
  `define HOSTKEEP_WIDTH  `HOST_WIDTH/8
  `define HOSTUSER_WIDTH  1
  `define HOST_IF_INST    2
  `define HOST_MTU        4096

  // Peripherals
  `define SPI_INST  2
  `define I2C_INST  2
  `define GPIO_INST 16
  localparam [`GPIO_INST-1:0] GPIO_RESET_VALUE = 16'b0000000000001111;

  // User registers
  `define REG_INST 8

  // System initialization: standard two-register MAC pause init.
  // See Common Boilerplate for the exact `N_INIT_REG` and `init_reg[]` block.

endpackage: HOLOLINK_pkg
`endif
```

**Things you'll likely change.**

1. `UUID` — supply your board's UUID via `uuid:` in the profile, or omit it and let the generator create one.
2. `PTP_CLK_FREQ` — fine-tune for your PTP clock source (must be in 95–105 MHz).
3. `init_reg[]` — extend with vendor-specific MAC/PCS register writes for your Ethernet MAC IP.
4. `GPIO_RESET_VALUE` — set the boot state of board-specific control GPIOs.
5. `SIF_TX_BUF_SIZE` — increase if your TX path needs more buffering than 2048 TX-width elements.

---

# 2. Mid-bandwidth baseline

**When to use.** Two moderate-speed sensors feeding two host interfaces. Datapath is 64 bits at 156.25 MHz — the standard configuration for boards using 10 GbE PHYs. This is the most common archetype in the corpus and a good starting point for any new design that doesn't have unusual bandwidth constraints.

**Distinctive macros.**

- `HIF_CLK_FREQ = 156_250_000`
- `APB_CLK_FREQ = 19_531_250`
- `DATAPATH_WIDTH = 64`, `DATAUSER_WIDTH = 2`
- `SENSOR_RX_IF_INST = 2`, `SENSOR_TX_IF_INST = 1`
- `HOST_WIDTH = 64`, `HOST_IF_INST = 2`, `HOST_MTU = 4096`
- `ENUM_EEPROM` defined; `UART_INST = 1`

**Sample.**

```systemverilog
`ifndef HOLOLINK_def
`define HOLOLINK_def

package HOLOLINK_pkg;

  // Clocking
  `define HIF_CLK_FREQ  156250000
  `define APB_CLK_FREQ  19531250
  `define PTP_CLK_FREQ  100446545

  // Insert UUID block from Common Boilerplate.

  // Enumeration
  `define ENUM_EEPROM
  `ifdef ENUM_EEPROM
    `define EEPROM_REG_ADDR_BITS 8
  `endif

  // Sensor datapath
  `define DATAPATH_WIDTH  64
  `define DATAKEEP_WIDTH  `DATAPATH_WIDTH/8
  `define DATAUSER_WIDTH  2

  // Sensor RX (2 interfaces)
  `define SENSOR_RX_IF_INST  2
  `ifdef SENSOR_RX_IF_INST
    localparam integer SIF_RX_WIDTH         [`SENSOR_RX_IF_INST-1:0] = '{default:`DATAPATH_WIDTH};
    localparam integer SIF_RX_PACKETIZER_EN [`SENSOR_RX_IF_INST-1:0] = '{default:1};
    localparam integer SIF_RX_VP_COUNT      [`SENSOR_RX_IF_INST-1:0] = {2, 2};
    localparam integer SIF_RX_SORT_RESOLUTION [`SENSOR_RX_IF_INST-1:0] = {2, 2};
    localparam integer SIF_RX_VP_SIZE       [`SENSOR_RX_IF_INST-1:0] = {64, 64};
    localparam integer SIF_RX_NUM_CYCLES    [`SENSOR_RX_IF_INST-1:0] = {3, 3};
  `endif

  // Sensor TX
  `define SENSOR_TX_IF_INST  1
  `ifdef SENSOR_TX_IF_INST
    localparam integer SIF_TX_WIDTH    [`SENSOR_TX_IF_INST-1:0] = '{default:`DATAPATH_WIDTH};
    localparam integer SIF_TX_BUF_SIZE [`SENSOR_TX_IF_INST-1:0] = '{default:4096};
  `endif

  // Host
  `define HOST_WIDTH      64
  `define HOSTKEEP_WIDTH  `HOST_WIDTH/8
  `define HOSTUSER_WIDTH  1
  `define HOST_IF_INST    2
  `define HOST_MTU        4096

  // Peripherals
  `define SPI_INST  2
  `define I2C_INST  4
  `define UART_INST 1
  `define GPIO_INST 31
  localparam [`GPIO_INST-1:0] GPIO_RESET_VALUE = '0;

  // User registers
  `define REG_INST 8

  // System initialization: standard two-register MAC pause init.
  // See Common Boilerplate for the exact `N_INIT_REG` and `init_reg[]` block.

endpackage: HOLOLINK_pkg
`endif
```

**Things you'll likely change.**

1. `UUID` — supply your board's UUID via `uuid:` in the profile, or omit it and let the generator create one.
2. `init_reg[]` — extend with vendor-specific MAC/PCS register initialization when required by your surrounding IP. See `init-reg-cookbook.md`.
3. `SIF_RX_VP_COUNT[]` and peers — the values shown are common but should match your packetizer requirements.
4. `SIF_TX_WIDTH` — narrow per-port if your TX sensor only needs less than the full 64-bit datapath.
5. `GPIO_INST` and `GPIO_RESET_VALUE` — size to your board's actual GPIO count and boot state.
6. `UART_INST` — remove if no UART is needed.

---

# 3. Minimal gateway

**When to use.** Edge device with one sensor and one host. No TX path. Smaller MTU (1500 bytes) for compatibility with vanilla Ethernet networks. Soft enumeration (no EEPROM) — the user's top wrapper drives MAC and serial number directly.

**Distinctive macros.**

- `HIF_CLK_FREQ = 156_250_000`
- `DATAPATH_WIDTH = 64`, `DATAUSER_WIDTH = 2`
- `SENSOR_RX_IF_INST = 1`, `SENSOR_TX_IF_INST` undefined
- `HOST_WIDTH = 64`, `HOST_IF_INST = 1`, `HOST_MTU = 1500`
- `ENUM_EEPROM` undefined (soft enumeration via input ports)
- Minimal peripherals: `SPI_INST = 1`, `I2C_INST = 1`, `GPIO_INST = 3`

**Sample.**

```systemverilog
`ifndef HOLOLINK_def
`define HOLOLINK_def

package HOLOLINK_pkg;

  // Clocking
  `define HIF_CLK_FREQ  156250000
  `define APB_CLK_FREQ  19531250
  `define PTP_CLK_FREQ  100446545

  // Insert UUID block from Common Boilerplate.

  // Enumeration — soft (no EEPROM); top wrapper must drive i_mac_addr,
  // i_board_sn, and i_enum_vld input ports.
  // `define ENUM_EEPROM         <-- left undefined intentionally

  // Sensor datapath
  `define DATAPATH_WIDTH  64
  `define DATAKEEP_WIDTH  `DATAPATH_WIDTH/8
  `define DATAUSER_WIDTH  2

  // Sensor RX (single interface)
  `define SENSOR_RX_IF_INST  1
  `ifdef SENSOR_RX_IF_INST
    localparam integer SIF_RX_WIDTH         [`SENSOR_RX_IF_INST-1:0] = '{default:`DATAPATH_WIDTH};
    localparam integer SIF_RX_PACKETIZER_EN [`SENSOR_RX_IF_INST-1:0] = '{default:1};
    localparam integer SIF_RX_VP_COUNT      [`SENSOR_RX_IF_INST-1:0] = {2};
    localparam integer SIF_RX_SORT_RESOLUTION [`SENSOR_RX_IF_INST-1:0] = {2};
    localparam integer SIF_RX_VP_SIZE       [`SENSOR_RX_IF_INST-1:0] = {64};
    localparam integer SIF_RX_NUM_CYCLES    [`SENSOR_RX_IF_INST-1:0] = {3};
  `endif

  // No Sensor TX

  // Host (single interface, standard MTU)
  `define HOST_WIDTH      64
  `define HOSTKEEP_WIDTH  `HOST_WIDTH/8
  `define HOSTUSER_WIDTH  1
  `define HOST_IF_INST    1
  `define HOST_MTU        1500

  // Minimal peripherals
  `define SPI_INST  1
  `define I2C_INST  1
  `define GPIO_INST 3
  localparam [`GPIO_INST-1:0] GPIO_RESET_VALUE = '0;

  // User registers
  `define REG_INST 8

  // System initialization: standard two-register MAC pause init.
  // See Common Boilerplate for the exact `N_INIT_REG` and `init_reg[]` block.

endpackage: HOLOLINK_pkg
`endif
```

**Things you'll likely change.**

1. `UUID` — supply your board's UUID via `uuid:` in the profile, or omit it and let the generator create one.
2. Decide between `ENUM_EEPROM` (read MAC/SN from EEPROM) and soft enumeration (drive `i_mac_addr`/`i_board_sn`/`i_enum_vld` ports). The sample shows soft.
3. `HOST_MTU = 1500` is conservative; bump to 4096 if your network supports jumbo frames.
4. `SPI_INST` / `I2C_INST` / `GPIO_INST` — adjust to actual peripheral count, or undefine entirely if a peripheral isn't used.

---

# 4. Ultra-minimal

**When to use.** Very small embedded target where logic resources are at a premium. TX-only configuration — the IP forwards host commands to a sensor but does not receive sensor data. Datapath is 8 bits; clocks are slow (25 MHz). Peripheral RAM depth is overridden to keep memory tiny.

**Distinctive macros.**

- `HIF_CLK_FREQ = 25_000_000`, `APB_CLK_FREQ = 25_000_000`
- `DATAPATH_WIDTH = 8`
- `SENSOR_RX_IF_INST` **undefined** (no RX path)
- `SENSOR_TX_IF_INST = 1`, `SIF_TX_WIDTH = {8}`
- `HOST_WIDTH = 8`, `HOST_IF_INST = 1`, `HOST_MTU = 1500`
- `PERI_RAM_DEPTH = 32`

**Sample.**

```systemverilog
`ifndef HOLOLINK_def
`define HOLOLINK_def

package HOLOLINK_pkg;

  // Clocking — slow, single-clock-domain design
  `define HIF_CLK_FREQ  25000000
  `define APB_CLK_FREQ  25000000
  `define PTP_CLK_FREQ  100000000

  // Insert UUID block from Common Boilerplate.

  // Enumeration
  `define ENUM_EEPROM
  `ifdef ENUM_EEPROM
    `define EEPROM_REG_ADDR_BITS 8
  `endif

  // Sensor datapath — minimal
  `define DATAPATH_WIDTH  8
  `define DATAKEEP_WIDTH  `DATAPATH_WIDTH/8
  `define DATAUSER_WIDTH  1

  // No Sensor RX
  // `define SENSOR_RX_IF_INST  1   <-- left undefined intentionally

  // Sensor TX only
  `define SENSOR_TX_IF_INST  1
  `ifdef SENSOR_TX_IF_INST
    localparam integer SIF_TX_WIDTH    [`SENSOR_TX_IF_INST-1:0] = '{default:`DATAPATH_WIDTH};
    localparam integer SIF_TX_BUF_SIZE [`SENSOR_TX_IF_INST-1:0] = '{default:2048};
  `endif

  // Host — narrow, single interface
  `define HOST_WIDTH      8
  `define HOSTKEEP_WIDTH  `HOST_WIDTH/8
  `define HOSTUSER_WIDTH  1
  `define HOST_IF_INST    1
  `define HOST_MTU        1500

  // Peripherals — minimal, with smaller peripheral RAM
  `define SPI_INST  1
  `define I2C_INST  1
  `define GPIO_INST 54
  localparam [`GPIO_INST-1:0] GPIO_RESET_VALUE = '0;
  `define PERI_RAM_DEPTH 32

  // User registers
  `define REG_INST 8

  // System initialization: standard two-register MAC pause init.
  // See Common Boilerplate for the exact `N_INIT_REG` and `init_reg[]` block.

endpackage: HOLOLINK_pkg
`endif
```

**Things you'll likely change.**

1. `UUID` — supply your board's UUID via `uuid:` in the profile, or omit it and let the generator create one.
2. `GPIO_INST` — size to your actual GPIO count; the value shown supports a wide pinout.
3. `PERI_RAM_DEPTH = 32` is the smallest reasonable depth; raise to 64 or 128 if your I²C transactions are longer.
4. `init_reg[]` — extend for vendor-specific MAC initialization (the IP version of the MAC may need a few hundred bytes of register writes).
5. The lack of Sensor RX ports means your top wrapper does not connect any `i_sif_axis_*` signals — make sure your top instantiation reflects that.

---

# 5. Very-high-speed

**When to use.** A single ultra-fast sensor (322 MHz HIF clock, 512-bit datapath) with a single host link. Often paired with `SIF_RX_DATA_GEN` to provide a per-port test-injection capability that the host can drive at runtime — useful for in-system diagnostics or simulation-driven bringup. Defining the macro doesn't bypass real sensor data; it just adds the capability. Soft enumeration is common.

**Distinctive macros.**

- `HIF_CLK_FREQ = 322_265_625`
- `APB_CLK_FREQ = 50_000_000`
- `DATAPATH_WIDTH = 512`
- `SENSOR_RX_IF_INST = 1` with `SIF_RX_DATA_GEN` defined for bringup
- `HOST_WIDTH = 512`, `HOST_IF_INST = 1`, `HOST_MTU = 4096`
- `ENUM_EEPROM` undefined

**Sample.**

```systemverilog
`ifndef HOLOLINK_def
`define HOLOLINK_def

package HOLOLINK_pkg;

  // Clocking — high-speed transceivers
  `define HIF_CLK_FREQ  322265625
  `define APB_CLK_FREQ  50000000
  `define PTP_CLK_FREQ  100000000

  // Insert UUID block from Common Boilerplate.

  // Enumeration — soft (no EEPROM)
  // `define ENUM_EEPROM         <-- left undefined intentionally

  // Sensor datapath
  `define DATAPATH_WIDTH  512
  `define DATAKEEP_WIDTH  `DATAPATH_WIDTH/8
  `define DATAUSER_WIDTH  1

  // Sensor RX with built-in data generator (for bringup; disable in production)
  `define SENSOR_RX_IF_INST  1
  `ifdef SENSOR_RX_IF_INST
    `define SIF_RX_DATA_GEN
    localparam integer SIF_RX_WIDTH         [`SENSOR_RX_IF_INST-1:0] = '{default:`DATAPATH_WIDTH};
    localparam integer SIF_RX_PACKETIZER_EN [`SENSOR_RX_IF_INST-1:0] = '{default:1};
    localparam integer SIF_RX_VP_COUNT      [`SENSOR_RX_IF_INST-1:0] = {4};
    localparam integer SIF_RX_SORT_RESOLUTION [`SENSOR_RX_IF_INST-1:0] = {16};
    localparam integer SIF_RX_VP_SIZE       [`SENSOR_RX_IF_INST-1:0] = {128};
    localparam integer SIF_RX_NUM_CYCLES    [`SENSOR_RX_IF_INST-1:0] = {1};
  `endif

  // Sensor TX
  `define SENSOR_TX_IF_INST  1
  `ifdef SENSOR_TX_IF_INST
    localparam integer SIF_TX_WIDTH    [`SENSOR_TX_IF_INST-1:0] = '{default:`DATAPATH_WIDTH};
    localparam integer SIF_TX_BUF_SIZE [`SENSOR_TX_IF_INST-1:0] = '{default:2048};
  `endif

  // Host
  `define HOST_WIDTH      512
  `define HOSTKEEP_WIDTH  `HOST_WIDTH/8
  `define HOSTUSER_WIDTH  1
  `define HOST_IF_INST    1
  `define HOST_MTU        4096

  // Peripherals
  `define SPI_INST  1
  `define I2C_INST  1
  `define GPIO_INST 31
  localparam [`GPIO_INST-1:0] GPIO_RESET_VALUE = '0;

  // User registers
  `define REG_INST 8

  // System initialization: single-register IP-rev init.
  // See Common Boilerplate for the exact `N_INIT_REG` and `init_reg[]` block.

endpackage: HOLOLINK_pkg
`endif
```

**Things you'll likely change.**

1. `UUID` — supply your board's UUID via `uuid:` in the profile, or omit it and let the generator create one.
2. **Decide whether to keep `SIF_RX_DATA_GEN` for production.** Defining it adds per-port runtime-controlled test injection (host-driven via APB) — useful for in-system diagnostics. Removing the `\`define` removes the data_gen modules entirely, saving logic and embedded RAM. Keep it on if you want diagnostic injection capability; remove if you're tight on resources.
3. `init_reg[]` — minimal as shown; some boards need additional MAC/PCS configuration entries.
4. `SIF_TX_BUF_SIZE` — increase if your TX path needs deeper FIFO storage in TX-width elements.
5. Decide between `ENUM_EEPROM` and soft enumeration based on your board.

---

# 6. Gigabit baseline

**When to use.** A 1 GbE host link with a single sensor in each direction. Typical for embedded camera/sensor boards on standard 1 GbE infrastructure: 8-bit GMII-style host interface clocked at 125 MHz (8 b × 125 MHz = 1 Gbps). Compatible with vanilla Ethernet (1500-byte MTU). Distinguished from `ultra-minimal` (which is also 8-bit but TX-only at 25 MHz) by the 1 GbE clocking and the presence of Sensor RX.

**Distinctive macros.**

- `HIF_CLK_FREQ = 125_000_000` (8 b × 125 MHz = 1 Gbps)
- `APB_CLK_FREQ = 19_531_250`
- `DATAPATH_WIDTH = 8`, `DATAUSER_WIDTH = 1`
- `SENSOR_RX_IF_INST = 1`, `SENSOR_TX_IF_INST = 1`
- `HOST_WIDTH = 8`, `HOST_IF_INST = 1`, `HOST_MTU = 1500`
- `ENUM_EEPROM` defined; minimal peripherals (`SPI_INST = 1`, `I2C_INST = 1`, `GPIO_INST = 16`)

**Sample.**

```systemverilog
`ifndef HOLOLINK_def
`define HOLOLINK_def

package HOLOLINK_pkg;

  // Clocking — 1 GbE: 8 bits × 125 MHz
  `define HIF_CLK_FREQ  125000000
  `define APB_CLK_FREQ  19531250
  `define PTP_CLK_FREQ  100446545

  // Insert UUID block from Common Boilerplate.

  // Enumeration
  `define ENUM_EEPROM
  `ifdef ENUM_EEPROM
    `define EEPROM_REG_ADDR_BITS 8
  `endif

  // Sensor datapath
  `define DATAPATH_WIDTH  8
  `define DATAKEEP_WIDTH  `DATAPATH_WIDTH/8
  `define DATAUSER_WIDTH  1

  // Sensor RX
  `define SENSOR_RX_IF_INST  1
  `ifdef SENSOR_RX_IF_INST
    localparam integer SIF_RX_WIDTH         [`SENSOR_RX_IF_INST-1:0] = '{default:`DATAPATH_WIDTH};
    localparam integer SIF_RX_PACKETIZER_EN [`SENSOR_RX_IF_INST-1:0] = '{default:1};
    localparam integer SIF_RX_VP_COUNT      [`SENSOR_RX_IF_INST-1:0] = {1};
    localparam integer SIF_RX_SORT_RESOLUTION [`SENSOR_RX_IF_INST-1:0] = {8};
    localparam integer SIF_RX_VP_SIZE       [`SENSOR_RX_IF_INST-1:0] = {64};
    localparam integer SIF_RX_NUM_CYCLES    [`SENSOR_RX_IF_INST-1:0] = {1};
  `endif

  // Sensor TX
  `define SENSOR_TX_IF_INST  1
  `ifdef SENSOR_TX_IF_INST
    localparam integer SIF_TX_WIDTH    [`SENSOR_TX_IF_INST-1:0] = '{default:`DATAPATH_WIDTH};
    localparam integer SIF_TX_BUF_SIZE [`SENSOR_TX_IF_INST-1:0] = '{default:2048};
  `endif

  // Host — 1 GbE
  `define HOST_WIDTH      8
  `define HOSTKEEP_WIDTH  `HOST_WIDTH/8
  `define HOSTUSER_WIDTH  1
  `define HOST_IF_INST    1
  `define HOST_MTU        1500

  // Peripherals
  `define SPI_INST  1
  `define I2C_INST  1
  `define GPIO_INST 16
  localparam [`GPIO_INST-1:0] GPIO_RESET_VALUE = '0;

  // User registers — at least 1 required by the IP
  `define REG_INST 1

  // System initialization: single-register MAC pause init.
  // See Common Boilerplate for the exact `N_INIT_REG` and `init_reg[]` block.

endpackage: HOLOLINK_pkg
`endif
```

**Things you'll likely change.**

1. `UUID` — supply your board's UUID via `uuid:` in the profile, or omit it and let the generator create one.
2. `init_reg[]` — extend with your 1 GbE MAC IP's register configuration (entries needed depend on your specific MAC IP — check its datasheet).
3. Decide between `ENUM_EEPROM` (production) and soft enumeration (eval/sim).
4. `HOST_MTU = 1500` is the standard for 1 GbE; bump to a non-standard value only if your network supports jumbo and you've sized buffers accordingly.
5. `GPIO_INST` — adjust to your board's actual GPIO pin count.

---

# Common adjustments across all archetypes

These edits apply to any archetype.

| Adjustment | What to change |
|---|---|
| Board UUID | Replace `128'h0000…` with your unique 128-bit identifier. The skill never auto-generates this for you. |
| Add a SPI interface | Increment `SPI_INST` and connect the additional SPI port group in your top wrapper. Range 1..8. |
| Add an I²C interface | Increment `I2C_INST`. Range 1..8. |
| Change MTU | Set `HOST_MTU = 4096` for jumbo frames or `1500` for standard Ethernet. The IP buffer scales automatically (`HOST_BUF_DEPTH = HOST_MTU * 2 / (HOST_WIDTH/8)`). |
| Wider Sensor RX | Increase `DATAPATH_WIDTH` to a power-of-2 byte-aligned value; widen `SIF_RX_WIDTH[i]` to match. Values above 1024 trigger a resource-impact warning. |
| Add user registers | Set `REG_INST` higher (max 8); each instance gives the user 4 KB of APB-addressable register space. |
| Disable a peripheral | Comment out the corresponding `\`define` (`SPI_INST`, `I2C_INST`, `UART_INST`). The RTL will silently fall back to dummy logic; the validator warns if any per-port arrays remain. |
| External PTP timestamps | Define `EXT_PTP` and drive `i_ptp_sec[47:0]` / `i_ptp_nanosec[31:0]` in your top wrapper instead of consuming the IP's internal PTP outputs. |

For deeper per-macro guidance see `macro-reference.md`. For `init_reg[]` address-space conventions see `init-reg-cookbook.md`.
