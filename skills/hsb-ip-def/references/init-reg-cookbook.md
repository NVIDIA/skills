# `init_reg[]` Cookbook

The `init_reg[]` array drives a boot-time write sequence: on rising edge of `i_init`, the IP loops through entries and issues APB writes, then asserts `o_init_done` when complete. Each entry is a 64-bit tuple `{32'h<addr>, 32'h<data>}`. The array length must equal `N_INIT_REG`.

This file documents the address-space conventions seen across in-scope archetypes and gives ready-to-copy templates per archetype.

## Address-space map

The 32-bit APB address space the IP exposes is partitioned by high-order nibbles. Most projects only touch a handful of these ranges.

| Address prefix | Block | Notes |
|---|---|---|
| `0x0100_*` | Global control / status | Read-mostly. Rarely written from `init_reg[]`. |
| `0x0120_*` | High-speed-archetype boot register(s) | Single-write configuration for the very-high-speed archetype's MAC. |
| `0x0200_0024` | Host interface 0 — destination UDP port | Set the host-side UDP destination port. Standard write: `0x0000_12B7` (hex for default port). |
| `0x0201_0024` | Host interface 1 — destination UDP port | Per-host duplicate of `0x0200_0024`. |
| `0x0202_*`, `0x0203_*` | Host interfaces 2 and 3 | Only present when `HOST_IF_INST > 2`. |
| `0x1000_*` | User register block 0 (`REG_INST_0`) | Application-specific. |
| `0x1001_*` … `0x1007_*` | User register blocks 1–7 (`REG_INST_1`..`REG_INST_7`) | Application-specific. |
| `0x2000_*`, `0x2001_*`, etc. | Vendor-supplied MAC / PCS register init | **Project-side, not HSB IP.** These addresses target the user's Ethernet MAC IP block via APB. The HSB IP's `sys_init` module merely forwards the writes — it has no semantic awareness of the MAC's register map. The exact addresses depend entirely on the MAC IP vendor and version. |

The HSB IP itself does not require *any* `init_reg[]` entries to function. The minimum useful sequence is the per-host destination-UDP-port writes (`0x0200_0024`, `0x0201_0024`, …); even those can be omitted if the user's bitstream targets a DHCP-assigned port. Larger init sequences are project-side configuration of the surrounding Ethernet MAC IP, which lives outside the HSB IP boundary. In a generator profile, use `init_reg: []` to explicitly suppress system init when using any defaults or archetype samples that might otherwise carry example writes.

## Per-archetype templates

### High-bandwidth single-sensor (2 entries)

Two host interfaces, default UDP destination port on each:

```systemverilog
`define N_INIT_REG 2
localparam logic [63:0] init_reg [`N_INIT_REG] = '{
  {32'h0200_0024, 32'h0000_12B7},
  {32'h0201_0024, 32'h0000_12B7}
};
```

### Mid-bandwidth baseline (2 entries)

Same shape as high-bandwidth — two host interfaces, two writes. **In production, expand with vendor-specific MAC/PCS register init** as required by your surrounding IP. Replace the placeholder addresses with the values supplied by your MAC IP datasheet:

```systemverilog
`define N_INIT_REG 2
localparam logic [63:0] init_reg [`N_INIT_REG] = '{
  {32'h0200_0024, 32'h0000_12B7},
  {32'h0201_0024, 32'h0000_12B7}
  // Append vendor MAC/PCS init entries here. Each entry must be 64 bits.
};
```

### Minimal gateway (2 entries)

Single host interface, single write — but you need both placeholder writes to keep the array a non-trivial example. In a real minimal-gateway design with one host you'd only need one entry:

```systemverilog
`define N_INIT_REG 1
localparam logic [63:0] init_reg [`N_INIT_REG] = '{
  {32'h0200_0024, 32'h0000_12B7}
};
```

### Ultra-minimal (1–2 entries)

The default UDP port write, plus optional MAC config:

```systemverilog
`define N_INIT_REG 1
localparam logic [63:0] init_reg [`N_INIT_REG] = '{
  {32'h0200_0024, 32'h0000_12B7}
};
```

### Very-high-speed (1 entry)

The high-speed archetype's corpus example uses a vendor MAC IP that handles its own initialization — the HSB IP only writes a single high-speed-archetype boot register:

```systemverilog
`define N_INIT_REG 1
localparam logic [63:0] init_reg [`N_INIT_REG] = '{
  {32'h0120_0000, 32'h0000_0001}
};
```

## Common mistakes

1. **`N_INIT_REG` mismatched with array length.** This is the #1 cause of silent breakage. The validator hard-errors on this (rule HD-E401). When you add or remove an entry, update both.
2. **Missing `\`define N_INIT_REG`.** Declaring `init_reg[]` without `N_INIT_REG` causes the `sys_init` module to be omitted entirely (the gating `\`ifdef N_INIT_REG` at `<hsb-ip-root>/top/HOLOLINK_top.sv:479` is false). The validator hard-errors (HD-E403).
3. **Non-32-bit address or data fields.** Each entry must be exactly `{32'h<addr>, 32'h<data>}` — 64 bits total. Mistakes like `{32'h0200_0024, 16'h12B7}` produce a 48-bit entry which Verilog elaboration may extend silently. The validator hard-errors (HD-E402).
4. **Writing to an address that doesn't exist on the user's MAC.** The HSB IP forwards the write blindly; if the address doesn't decode to anything in your MAC IP, the APB transaction may stall, producing a hung `o_init_done`. The validator can't catch this — it's project-specific.
5. **Order matters.** The IP writes entries strictly top-to-bottom, with a fixed inter-write delay (32 cycles per `<hsb-ip-root>/sys_init/m_apb_reg.sv`). If your MAC IP has dependencies (e.g., "set X before Y"), reflect them in the array order.

## Where the IP consumes `init_reg[]`

For the curious:

| Site | What it does |
|---|---|
| `<hsb-ip-root>/top/HOLOLINK_top.sv:479-481` | Conditionally instantiates the `sys_init` module when `N_INIT_REG` is defined |
| `<hsb-ip-root>/sys_init/sys_init.sv:30` | Receives the array as `i_init_reg` input |
| `<hsb-ip-root>/sys_init/sys_init.sv:62-64` | Per-entry address/data extraction (`addr = entry[63:32]`, `data = entry[31:0]`) |
| `<hsb-ip-root>/sys_init/sys_init.sv:72-83` | Loops through entries on `i_init` rising edge, issuing APB writes |
