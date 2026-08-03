# Packetizer `HOLOLINK_def.svh` Fields

Use this reference to choose or explain the packetizer-local fields in `HOLOLINK_def.svh`.

This file documents known HSB IP rev `16'h2604` and backward-compatible rev `16'h2603`. If live HSB IP source is available, verify the `HOLOLINK_top.sv` mapping into `packetizer_top` and the packetizer RTL constraints against that source before emitting enabled packetizer fields. Live source wins when it disagrees with this reference.

## Field Table

| Field | Kind | Values to discuss | Purpose |
|---|---|---|---|
| `SIF_RX_PACKETIZER_EN[]` | `localparam integer` array | `0` or `1` per Sensor RX interface | Enables packetizer hardware for each RX interface |
| `SIF_RX_VP_COUNT[]` | `localparam integer` array | documented examples include `1`, `2`, `4` | Number of virtual output streams for an enabled RX interface |
| `SIF_RX_SORT_RESOLUTION[]` | `localparam integer` array | documented examples include `2`, `16`, `32`, `DATAPATH_WIDTH` | Bit granularity used by the data-rearrangement network |
| `SIF_RX_VP_SIZE[]` | `localparam integer` array | documented examples include `64`, `128`, `256`, `DATAPATH_WIDTH` | Virtual-port size/width in bits |
| `SIF_RX_NUM_CYCLES[]` | `localparam integer` array | documented examples include `1`, `3` | Number of input cycles available to the packetizer operation |

The listed values are examples, not a closed legal set. For enabled packetizers, screen proposed values against the RTL-derived constraints below before suggesting them.

## Required Context

Know these before choosing enabled packetizer values:

- `SENSOR_RX_IF_INST`: number of Sensor RX interfaces; all packetizer arrays are sized to it.
- `SIF_RX_WIDTH[i]`: the input width for packetizer instance `i`; `HOLOLINK_top` passes this as `DIN_WIDTH`.
- The desired data manipulation per interface: pass-through, data rearrangement, split into virtual ports, replicate/duplicate, or another runtime-controlled pattern.

Do not choose packetizer values just because they appear in an archetype. Ask what the data needs to do, derive the full field set from that description, then explain the result. Do not make the user approve each packetizer field separately.

## Array Rules

- `SIF_RX_PACKETIZER_EN[]` length must equal `SENSOR_RX_IF_INST`.
- If any enable entry is `1`, all four peer arrays must be present and length-matched.
- If enable entry `i` is `0`, peer entries at index `i` are ignored by the IP.
- If every enable entry is `0`, the peer arrays are unnecessary for the packetizer decision.

Existing def skill validation mirrors these rules:

- `HD-E126`: enable element is not `0` or `1`.
- `HD-E302`: enable array length mismatch.
- `HD-E303` through `HD-E306`: peer array length mismatch when any packetizer is enabled.
- `HD-W305`: packetizer enabled but a peer array is missing.
- `HD-I001`: peer array value exists for a disabled packetizer and is a don't-care.

## Disabled Entries

For a disabled entry inside a mixed array, use the RTL bypass constants as placeholders:

| Peer field | Placeholder |
|---|---:|
| `SIF_RX_VP_COUNT[i]` | `1` |
| `SIF_RX_SORT_RESOLUTION[i]` | `2` |
| `SIF_RX_VP_SIZE[i]` | `32` |
| `SIF_RX_NUM_CYCLES[i]` | `1` |

`HOLOLINK_top` substitutes these same values when `SIF_RX_PACKETIZER_EN[i]` is `0`: `PACK_SORT_RESOLUTION=2`, `PACK_VP_COUNT=1`, `PACK_VP_SIZE=32`, and `PACK_NUM_CYCLES=1` (`top/HOLOLINK_top.sv:1356-1362`).

## RTL-Derived Constraints

Treat these as guardrails before recommending an enabled-packetizer value:

- `SIF_RX_PACKETIZER_EN[i]` must be `0` or `1`.
- `SIF_RX_VP_COUNT[i]` must be positive. `HOLOLINK_top` protects only the local `SIF_VP_INST` width when the value is zero, but still passes `PACK_VP_COUNT=SIF_RX_VP_COUNT[i]` into `packetizer_top` (`top/HOLOLINK_top.sv:1356-1382`).
- `SIF_RX_SORT_RESOLUTION[i]` must be positive and should divide `SIF_RX_WIDTH[i]`. The packetizer computes widths from `DIN_WIDTH/SORT_RESOLUTION` and indexes odd-even lookup tables from `$clog2(DIN_WIDTH)-$clog2(SORT_RESOLUTION)` (`packetizer/packetizer.sv:69-85`, `:213-224`).
- Prefer power-of-two `SIF_RX_SORT_RESOLUTION[i]` values. The odd-even network is built around powers of two (`packetizer/odd_even_gen.sv:30-60`).
- `SIF_RX_VP_SIZE[i]` is used as a bit width. It appears in vector declarations, shifts, FIFO input width, and `DIN_WIDTH/VP_SIZE` control sizing (`packetizer/virtual_port.sv:18-30`, `:55-99`, `:153-170`, `:210-212`).
- For `HOLOLINK_top`'s current packetizer instantiation, `DYNAMIC_VP` and `MIXED_VP_SIZE` are forced to `0` (`top/HOLOLINK_top.sv:1360-1361`, `:1379-1380`). Prefer a positive power-of-two `SIF_RX_VP_SIZE[i]` that is no greater than and divides `SIF_RX_WIDTH[i]`.
- `SIF_RX_NUM_CYCLES[i]` must be positive. The core declares arrays sized by `NUM_CYCLES` and uses `$clog2(NUM_CYCLES+1)` (`packetizer/packetizer.sv:77-85`, `:140-179`).
- If `SIF_RX_SORT_RESOLUTION[i] == SIF_RX_WIDTH[i]`, `packetizer_top` disables the sort network internally (`packetizer/packetizer_top.sv:393-400`).

## Output Formatting

For standalone SVH, emit only packetizer localparams. Put the block under the existing `SENSOR_RX_IF_INST` section of `HOLOLINK_def.svh`. This section is the canonical source for standalone SVH output examples; `references/handoff-contract.md` owns the `packetizer_profile_overlay` YAML examples for `hsb-ip-def`.

All disabled:

```systemverilog
localparam integer SIF_RX_PACKETIZER_EN    [`SENSOR_RX_IF_INST-1:0] = '{default:0};
```

Mixed enabled/disabled:

```systemverilog
localparam integer SIF_RX_PACKETIZER_EN    [`SENSOR_RX_IF_INST-1:0] = {1, 0};
localparam integer SIF_RX_VP_COUNT         [`SENSOR_RX_IF_INST-1:0] = {4, 1};
localparam integer SIF_RX_SORT_RESOLUTION  [`SENSOR_RX_IF_INST-1:0] = {16, 2};
localparam integer SIF_RX_VP_SIZE          [`SENSOR_RX_IF_INST-1:0] = {128, 32};
localparam integer SIF_RX_NUM_CYCLES       [`SENSOR_RX_IF_INST-1:0] = {1, 1};
```

Uniform enabled:

```systemverilog
localparam integer SIF_RX_PACKETIZER_EN    [`SENSOR_RX_IF_INST-1:0] = '{default:1};
localparam integer SIF_RX_VP_COUNT         [`SENSOR_RX_IF_INST-1:0] = '{default:4};
localparam integer SIF_RX_SORT_RESOLUTION  [`SENSOR_RX_IF_INST-1:0] = '{default:16};
localparam integer SIF_RX_VP_SIZE          [`SENSOR_RX_IF_INST-1:0] = '{default:128};
localparam integer SIF_RX_NUM_CYCLES       [`SENSOR_RX_IF_INST-1:0] = '{default:1};
```
