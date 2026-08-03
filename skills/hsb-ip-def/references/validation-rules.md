# Validation Rules

Each rule has a stable ID (`HD-E…` for errors, `HD-W…` for warnings, `HD-I…` for info). The same IDs appear in `scripts/lib/rules.py` and in the JSON output of `validate_def.py`. **Drift between this file and `rules.py` is a defect.**

## Rule severity & exit behavior

| Severity | Exit code impact | Examples |
|---|---|---|
| **Error** (`HD-E…`) | Non-zero exit (1) | Missing wrapper, byte-alignment violation, derived-equality failure, init_reg cardinality mismatch |
| **Warning** (`HD-W…`) | Zero exit (info only) | Non-power-of-2 widths, resource-threshold warnings, documented frequency-band warnings, peripheral-macro footguns |
| **Info** (`HD-I…`) | Zero exit (info only) | Don't-care fields present, IP-rev backward-compat targets, archetype-classification notes |

The validator ALWAYS reports all rule findings even when the user requests "errors only" — warnings about silent peripheral fallback in particular are too important to suppress.

---

# E001–E099 · File structure (wrapper required)

| ID | Severity | Triggers when | Message template |
|---|---|---|---|
| HD-E001 | Error | First non-comment, non-blank line is not `\`ifndef HOLOLINK_def` | "File must begin with `\`ifndef HOLOLINK_def` guard. See macro-reference.md § File wrapper requirement." |
| HD-E002 | Error | `\`ifndef HOLOLINK_def` is present but `\`define HOLOLINK_def` is missing on the following line | "After `\`ifndef HOLOLINK_def`, the next directive must be `\`define HOLOLINK_def`." |
| HD-E003 | Error | `package HOLOLINK_pkg;` declaration is missing | "Missing `package HOLOLINK_pkg;` declaration. The file must be wrapped in this package." |
| HD-E004 | Error | `endpackage` is missing or does not name `HOLOLINK_pkg` | "Missing or malformed `endpackage: HOLOLINK_pkg`. Close the package before the final `\`endif`." |
| HD-E005 | Error | Closing `\`endif` is missing | "Missing closing `\`endif` for the `\`ifndef HOLOLINK_def` guard." |

# E101–E199 · Per-macro range / format

## Clocks

| ID | Severity | Triggers when | Message template |
|---|---|---|---|
| HD-E101 | Error | `HIF_CLK_FREQ` ≤ 0 or non-numeric | "HIF_CLK_FREQ must be a positive integer (Hz)." |
| HD-E102 | Error | `APB_CLK_FREQ` ≤ 0 or non-numeric | "APB_CLK_FREQ must be a positive integer (Hz)." |
| HD-E103 | Error | `PTP_CLK_FREQ` ≤ 0 or non-numeric | "PTP_CLK_FREQ must be a positive integer (Hz)." |

## Board identity

| ID | Severity | Triggers when | Message template |
|---|---|---|---|
| HD-E104 | Error | `UUID` is undefined | "UUID is required. Define a 128-bit hex literal — never derive at runtime." |
| HD-E105 | Error | `UUID` literal width != 128 | "UUID must be exactly 128 bits (e.g. `128'h…`). Got <width>-bit literal." |
| HD-E106 | Error | `UUID` literal is malformed (parse failure) | "UUID literal is malformed; expected `128'h<32 hex chars, optionally `_`-separated>`." |

## Enumeration

| ID | Severity | Triggers when | Message template |
|---|---|---|---|
| HD-E107 | Error | `ENUM_EEPROM` defined and `EEPROM_REG_ADDR_BITS` ∉ {8, 16} | "EEPROM_REG_ADDR_BITS must be 8 or 16 when ENUM_EEPROM is defined." |

## Sensor datapath

| ID | Severity | Triggers when | Message template |
|---|---|---|---|
| HD-E108 | Error | `DATAPATH_WIDTH` ≤ 0 | "DATAPATH_WIDTH must be a positive integer." |
| HD-E109 | Error | `DATAPATH_WIDTH` not divisible by 8 | "DATAPATH_WIDTH must be byte-aligned (divisible by 8). Got <value>." |
| HD-E110 | Error | `DATAUSER_WIDTH` ∉ {1, 2} | "DATAUSER_WIDTH must be 1 or 2 (per HSB documentation)." |
| HD-E111 | Error | `HOSTUSER_WIDTH` != 1 | "HOSTUSER_WIDTH is fixed at 1 in the current IP." |

## Host

| ID | Severity | Triggers when | Message template |
|---|---|---|---|
| HD-E112 | Error | `HOST_WIDTH` ≤ 0 | "HOST_WIDTH must be a positive integer." |
| HD-E113 | Error | `HOST_WIDTH` not divisible by 8 | "HOST_WIDTH must be byte-aligned (divisible by 8). Got <value>." |
| HD-E114 | Error | `HOST_IF_INST` ∉ {1..32} | "HOST_IF_INST must be in the range 1..32. Got <value>." |
| HD-E115 | Error | `HOST_MTU` ≤ 0 | "HOST_MTU must be a positive integer (bytes)." |

## Sensor RX / TX gating

| ID | Severity | Triggers when | Message template |
|---|---|---|---|
| HD-E116 | Error | `SENSOR_RX_IF_INST` defined but ∉ {1..32} | "SENSOR_RX_IF_INST must be in the range 1..32 when defined. Got <value>. (Leave it undefined to disable RX.)" |
| HD-E117 | Error | `SENSOR_TX_IF_INST` defined but ∉ {1..32} | "SENSOR_TX_IF_INST must be in the range 1..32 when defined. Got <value>." |

## Peripherals

| ID | Severity | Triggers when | Message template |
|---|---|---|---|
| HD-E118 | Error | `SPI_INST` defined but ∉ {1..8} | "SPI_INST must be in the range 1..8 when defined. Got <value>." |
| HD-E119 | Error | `I2C_INST` defined but ∉ {1..8} | "I2C_INST must be in the range 1..8 when defined. Got <value>." |
| HD-E120 | Error | `UART_INST` defined but != 1 | "UART_INST must be 1 when defined (only one UART instance is currently supported)." |
| HD-E121 | Error | `GPIO_INST` ∉ {1..255} | "GPIO_INST must be in the range 1..255. Got <value>." |
| HD-E122 | Error | `REG_INST` ∉ {1..8} | "REG_INST must be in the range 1..8. Got <value>." |

## Per-port arrays

| ID | Severity | Triggers when | Message template |
|---|---|---|---|
| HD-E123 | Error | `SIF_RX_WIDTH[i]` ≤ 0 or not byte-aligned | "SIF_RX_WIDTH[<i>] must be a positive byte-aligned integer. Got <value>." |
| HD-E124 | Error | `SIF_TX_WIDTH[i]` ≤ 0 or not byte-aligned | "SIF_TX_WIDTH[<i>] must be a positive byte-aligned integer. Got <value>." |
| HD-E125 | Error | `SIF_TX_BUF_SIZE[i]` ≤ 0 | "SIF_TX_BUF_SIZE[<i>] must be a positive integer FIFO depth in SIF_TX_WIDTH elements." |
| HD-E126 | Error | `SIF_RX_PACKETIZER_EN[i]` ∉ {0, 1} | "SIF_RX_PACKETIZER_EN[<i>] must be 0 or 1." |
| HD-E127 | Error | `GPIO_RESET_VALUE` width != `GPIO_INST` | "GPIO_RESET_VALUE must be sized exactly [GPIO_INST-1:0]. Declared width <wD>, expected <wE>." |

# E201–E299 · Cross-macro / derived equalities

| ID | Severity | Triggers when | Message template |
|---|---|---|---|
| HD-E201 | Error | `DATAKEEP_WIDTH` != `DATAPATH_WIDTH/8` | "DATAKEEP_WIDTH must equal DATAPATH_WIDTH/8. Got DATAKEEP_WIDTH=<a>, DATAPATH_WIDTH=<b>, expected <b/8>." |
| HD-E202 | Error | `HOSTKEEP_WIDTH` != `HOST_WIDTH/8` | "HOSTKEEP_WIDTH must equal HOST_WIDTH/8. Got HOSTKEEP_WIDTH=<a>, HOST_WIDTH=<b>, expected <b/8>." |
| HD-E203 | Error | Any `SIF_RX_WIDTH[i]` > `DATAPATH_WIDTH` | "SIF_RX_WIDTH[<i>] = <v> exceeds DATAPATH_WIDTH = <D>. Per-port widths must be ≤ the system-wide datapath." |
| HD-E204 | Error | Any `SIF_TX_WIDTH[i]` > `DATAPATH_WIDTH` | "SIF_TX_WIDTH[<i>] = <v> exceeds DATAPATH_WIDTH = <D>." |

# E301–E399 · Conditional dependencies / array length

| ID | Severity | Triggers when | Message template |
|---|---|---|---|
| HD-E301 | Error | `SIF_RX_WIDTH[]` length != `SENSOR_RX_IF_INST` | "SIF_RX_WIDTH array length (<L>) must equal SENSOR_RX_IF_INST (<N>)." |
| HD-E302 | Error | `SIF_RX_PACKETIZER_EN[]` length != `SENSOR_RX_IF_INST` | "SIF_RX_PACKETIZER_EN array length (<L>) must equal SENSOR_RX_IF_INST (<N>)." |
| HD-E303 | Error | `SIF_RX_VP_COUNT[]` length != `SENSOR_RX_IF_INST` (and any packetizer is enabled) | "SIF_RX_VP_COUNT array length (<L>) must equal SENSOR_RX_IF_INST (<N>) when any SIF_RX_PACKETIZER_EN element is 1." |
| HD-E304 | Error | `SIF_RX_SORT_RESOLUTION[]` length mismatch (and any packetizer enabled) | "SIF_RX_SORT_RESOLUTION array length (<L>) must equal SENSOR_RX_IF_INST (<N>)." |
| HD-E305 | Error | `SIF_RX_VP_SIZE[]` length mismatch (and any packetizer enabled) | "SIF_RX_VP_SIZE array length (<L>) must equal SENSOR_RX_IF_INST (<N>)." |
| HD-E306 | Error | `SIF_RX_NUM_CYCLES[]` length mismatch (and any packetizer enabled) | "SIF_RX_NUM_CYCLES array length (<L>) must equal SENSOR_RX_IF_INST (<N>)." |
| HD-E307 | Error | `SIF_TX_WIDTH[]` length != `SENSOR_TX_IF_INST` | "SIF_TX_WIDTH array length (<L>) must equal SENSOR_TX_IF_INST (<N>)." |
| HD-E308 | Error | `SIF_TX_BUF_SIZE[]` length != `SENSOR_TX_IF_INST` | "SIF_TX_BUF_SIZE array length (<L>) must equal SENSOR_TX_IF_INST (<N>)." |
| HD-E309 | Error | `GPIO_INST` defined but `GPIO_RESET_VALUE` missing | "GPIO_RESET_VALUE must be declared whenever GPIO_INST is defined." |

# E401–E499 · `init_reg` cardinality

| ID | Severity | Triggers when | Message template |
|---|---|---|---|
| HD-E401 | Error | `N_INIT_REG` != literal length of `init_reg[]` array | "N_INIT_REG (<N>) must equal the literal length of the init_reg[] array (<L>)." |
| HD-E402 | Error | Any `init_reg[k]` entry has width != 64 bits | "init_reg[<k>] entry must be exactly 64 bits — `{32'h<addr>, 32'h<data>}`." |
| HD-E403 | Error | `init_reg` array present but `N_INIT_REG` undefined | "init_reg[] is declared but N_INIT_REG is undefined. Define N_INIT_REG to match the array length." |
| HD-E404 | Error | `N_INIT_REG` is defined and > 0 but `init_reg[]` is missing | "N_INIT_REG is defined but init_reg[] is missing. Declare init_reg[] with exactly N_INIT_REG entries." |
| HD-E405 | Error | `N_INIT_REG` is defined as `0` | "N_INIT_REG = 0 is invalid: the IP's `\`ifdef N_INIT_REG` gate evaluates true and instantiates `sys_init` with a zero-element array, which fails elaboration. To skip the system-init sequence, leave `N_INIT_REG` undefined entirely (remove the `\`define) — the IP then omits `sys_init` cleanly." |

---

# W001–W099 · FPGA-resource warnings

| ID | Severity | Triggers when | Message template |
|---|---|---|---|
| HD-W001 | Warning | `DATAPATH_WIDTH` > 1024 | "DATAPATH_WIDTH = <v> is wider than the resource-warning threshold (1024). Confirm the target has sufficient logic and routing resources for the wider datapath." |
| HD-W002 | Warning | `HOST_WIDTH` > 1024 | "HOST_WIDTH = <v> is wider than the resource-warning threshold (1024). Confirm the target has sufficient logic and routing resources." |
| HD-W003 | Warning | Any `SIF_RX_WIDTH[i]` > 1024 | "SIF_RX_WIDTH[<i>] = <v> is wider than the resource-warning threshold (1024). Confirm logic and routing capacity." |
| HD-W004 | Warning | Any `SIF_TX_WIDTH[i]` > 1024 | "SIF_TX_WIDTH[<i>] = <v> is wider than the resource-warning threshold (1024). Confirm logic and routing capacity." |
| HD-W005 | Warning | Any `SIF_TX_BUF_SIZE[i]` > 8192 | "SIF_TX_BUF_SIZE[<i>] = <v> exceeds the resource-warning threshold (8192 TX-width elements). Confirm the target has sufficient embedded RAM resources for the deeper FIFO." |

# W101–W199 · Style (power-of-2 convention)

| ID | Severity | Triggers when | Message template |
|---|---|---|---|
| HD-W101 | Warning | `DATAPATH_WIDTH` is byte-aligned but not a power of 2 | "DATAPATH_WIDTH = <v> is byte-aligned but not a power of 2. AXI conventions favor powers of 2; some downstream IP may not handle non-power-of-2 widths cleanly." |
| HD-W102 | Warning | `HOST_WIDTH` is byte-aligned but not a power of 2 | "HOST_WIDTH = <v> is byte-aligned but not a power of 2 (AXI bus convention favors powers of 2)." |
| HD-W103 | Warning | Any `SIF_RX_WIDTH[i]` byte-aligned but not power of 2 | "SIF_RX_WIDTH[<i>] = <v> is not a power of 2 (AXI bus convention)." |
| HD-W104 | Warning | Any `SIF_TX_WIDTH[i]` byte-aligned but not power of 2 | "SIF_TX_WIDTH[<i>] = <v> is not a power of 2." |
| HD-W105 | Warning | Any `SIF_TX_BUF_SIZE[i]` not a power of 2 | "SIF_TX_BUF_SIZE[<i>] = <v> is not a power of 2. Power-of-2 FIFO depths are easier for downstream memory implementations." |

# W201–W299 · Frequency-band warnings

| ID | Severity | Triggers when | Message template |
|---|---|---|---|
| HD-W207 | Warning | `PTP_CLK_FREQ` outside the 95–105 MHz band | "PTP_CLK_FREQ = <v> Hz is outside the HSB-documented range of 95–105 MHz. PTP timing accuracy may be degraded." |

# W301–W399 · Footguns

| ID | Severity | Triggers when | Message template |
|---|---|---|---|
| HD-W301 | Warning | Any `SIF_RX_*[]` array is declared but `SENSOR_RX_IF_INST` is undefined | "SIF_RX_<name>[] array is declared but SENSOR_RX_IF_INST is undefined. The RTL silently falls back to a 1-bit dummy interface — there is no compile error. Did you mean to define SENSOR_RX_IF_INST?" |
| HD-W302 | Warning | Any `SIF_TX_*[]` array is declared but `SENSOR_TX_IF_INST` is undefined | "SIF_TX_<name>[] array is declared but SENSOR_TX_IF_INST is undefined. RTL falls back silently — define SENSOR_TX_IF_INST or remove the array." |
| HD-W303 | Warning | `EEPROM_REG_ADDR_BITS` is defined but `ENUM_EEPROM` is not | "EEPROM_REG_ADDR_BITS is defined but ENUM_EEPROM is undefined — the macro is dead. Either define ENUM_EEPROM or remove EEPROM_REG_ADDR_BITS." |
| HD-W304 | Warning | A `\`define BUILD_REV …` directive appears | "BUILD_REV is a Verilog parameter on HOLOLINK_top, not a `\`define` macro. The `\`define` here will be ignored. Pass BUILD_REV via parameter override at module instantiation: `HOLOLINK_top #(.BUILD_REV(48'h…)) u_…`." |
| HD-W305 | Warning | `SENSOR_RX_IF_INST` defined and at least one packetizer enabled but `SIF_RX_VP_COUNT[]` (or peers) is missing | "SIF_RX_<name>[] is required when any SIF_RX_PACKETIZER_EN element is 1. Provide the array (use `'{default:<value>}` for uniform configurations)." |
| HD-W306 | Warning | `GPIO_RESET_VALUE` is declared with a literal width that does not match `GPIO_INST` exactly (but happens to fit) | "GPIO_RESET_VALUE is declared as <wD> bits but GPIO_INST = <wE>. The mismatch is silently sign-extended or truncated by Verilog elaboration; declare it explicitly as [GPIO_INST-1:0] for clarity." |

---

# I001–I099 · Informational

| ID | Severity | Triggers when | Message template |
|---|---|---|---|
| HD-I001 | Info | An element of any `SIF_RX_VP_*[]`/`SORT_RESOLUTION[]`/`VP_SIZE[]`/`NUM_CYCLES[]` is provided where the corresponding `SIF_RX_PACKETIZER_EN[i] = 0` | "SIF_RX_<name>[<i>] is a don't-care because SIF_RX_PACKETIZER_EN[<i>] = 0; the value is ignored by the IP." |
| HD-I003 | Info | Configuration does not match any of the 6 archetypes | "Configuration does not cleanly match any archetype in archetypes.md. The validator confirms it is structurally valid; archetype-specific guidance is not available." |
| HD-I004 | Info | Configuration matches archetype `<name>` | "Configuration matches the `<name>` archetype. See `archetypes.md` for guidance on common adjustments." |

---

# Validator output contract

The validator emits a JSON object of the form:

```json
{
  "errors":   [{"rule": "HD-E…", "line": <int|null>, "macro": "<NAME|null>", "msg": "<rendered message>"}],
  "warnings": [{"rule": "HD-W…", "line": <int|null>, "macro": "<NAME|null>", "msg": "<rendered message>"}],
  "info":     [{"rule": "HD-I…", "line": <int|null>, "macro": "<NAME|null>", "msg": "<rendered message>"}],
  "inferred_archetype": "<name|null>",
  "ip_version_target": "16'h2604"
}
```

The validator's output **must never name out-of-scope macros** in any `msg` field. Out-of-scope macros pass silently with no annotation. This is a hard ship-blocker — see `PLAN.md` Phase 7.
