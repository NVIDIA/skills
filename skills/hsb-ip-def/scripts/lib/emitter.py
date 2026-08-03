"""Emit a HOLOLINK_def.svh file from a profile dict.

Public entry points:
  ARCHETYPE_DEFAULTS — per-archetype default profile dicts
  GENERIC_DEFAULTS   — defaults used when no archetype is selected
  build_profile(archetype, overrides) — merge defaults with overrides; reconciles
                                        per-port array lengths and clamps reg_inst
  emit_svh(profile) — render to SVH string in the corpus formatting style
"""

from typing import Any, Dict, List, Optional, Tuple


# -----------------------------------------------------------------------------
# Per-archetype default profiles
# -----------------------------------------------------------------------------

_HIGH_BANDWIDTH_SINGLE_SENSOR: Dict[str, Any] = {
    "archetype": "high-bandwidth-single-sensor",
    "uuid": None,  # required from user
    "hif_clk_freq": 201416016,
    "apb_clk_freq": 100000000,
    "ptp_clk_freq": 100707500,
    "enum_eeprom": True,
    "eeprom_reg_addr_bits": 8,
    "datapath_width": 512,
    "datauser_width": 1,
    "sensor_rx_count": 1,
    "sif_rx_data_gen": False,
    "sif_rx_widths": None,             # default to DATAPATH_WIDTH
    "sif_rx_packetizer_en": None,      # default to all 1s
    "sif_rx_vp_count": [4],
    "sif_rx_sort_resolution": [16],
    "sif_rx_vp_size": [128],
    "sif_rx_num_cycles": [1],
    "sensor_tx_count": 1,
    "sif_tx_widths": None,             # default to DATAPATH_WIDTH
    "sif_tx_buf_size": [2048],
    "host_width": 512,
    "host_if_inst": 2,
    "host_mtu": 4096,
    "spi_inst": 2,
    "i2c_inst": 2,
    "uart_inst": None,
    "gpio_inst": 16,
    "gpio_reset_value": "16'b0000000000001111",
    "reg_inst": 8,
    "init_reg": [
        ("32'h0200_0024", "32'h0000_12B7"),
        ("32'h0201_0024", "32'h0000_12B7"),
    ],
    "ext_ptp": False,
    "sync_clk_hif_apb": False,
    "sync_clk_hif_ptp": False,
    "peri_ram_depth": None,
    "disable_coe": False,
}

_MID_BANDWIDTH_BASELINE: Dict[str, Any] = {
    "archetype": "mid-bandwidth-baseline",
    "uuid": None,
    "hif_clk_freq": 156250000,
    "apb_clk_freq": 19531250,
    "ptp_clk_freq": 100446545,
    "enum_eeprom": True,
    "eeprom_reg_addr_bits": 8,
    "datapath_width": 64,
    "datauser_width": 2,
    "sensor_rx_count": 2,
    "sif_rx_data_gen": False,
    "sif_rx_widths": None,
    "sif_rx_packetizer_en": None,
    "sif_rx_vp_count": [2, 2],
    "sif_rx_sort_resolution": [2, 2],
    "sif_rx_vp_size": [64, 64],
    "sif_rx_num_cycles": [3, 3],
    "sensor_tx_count": 1,
    "sif_tx_widths": None,
    "sif_tx_buf_size": [4096],
    "host_width": 64,
    "host_if_inst": 2,
    "host_mtu": 4096,
    "spi_inst": 2,
    "i2c_inst": 4,
    "uart_inst": 1,
    "gpio_inst": 31,
    "gpio_reset_value": "'0",
    "reg_inst": 8,
    "init_reg": [
        ("32'h0200_0024", "32'h0000_12B7"),
        ("32'h0201_0024", "32'h0000_12B7"),
    ],
    "ext_ptp": False,
    "sync_clk_hif_apb": False,
    "sync_clk_hif_ptp": False,
    "peri_ram_depth": None,
    "disable_coe": False,
}

_MINIMAL_GATEWAY: Dict[str, Any] = {
    "archetype": "minimal-gateway",
    "uuid": None,
    "hif_clk_freq": 156250000,
    "apb_clk_freq": 19531250,
    "ptp_clk_freq": 100446545,
    "enum_eeprom": False,              # soft enumeration
    "eeprom_reg_addr_bits": 8,
    "datapath_width": 64,
    "datauser_width": 2,
    "sensor_rx_count": 1,
    "sif_rx_data_gen": False,
    "sif_rx_widths": None,
    "sif_rx_packetizer_en": None,
    "sif_rx_vp_count": [2],
    "sif_rx_sort_resolution": [2],
    "sif_rx_vp_size": [64],
    "sif_rx_num_cycles": [3],
    "sensor_tx_count": None,           # no TX
    "sif_tx_widths": None,
    "sif_tx_buf_size": None,
    "host_width": 64,
    "host_if_inst": 1,
    "host_mtu": 1500,
    "spi_inst": 1,
    "i2c_inst": 1,
    "uart_inst": None,
    "gpio_inst": 3,
    "gpio_reset_value": "'0",
    "reg_inst": 8,
    "init_reg": [
        ("32'h0200_0024", "32'h0000_12B7"),
    ],
    "ext_ptp": False,
    "sync_clk_hif_apb": False,
    "sync_clk_hif_ptp": False,
    "peri_ram_depth": None,
    "disable_coe": False,
}

_ULTRA_MINIMAL: Dict[str, Any] = {
    "archetype": "ultra-minimal",
    "uuid": None,
    "hif_clk_freq": 25000000,
    "apb_clk_freq": 25000000,
    "ptp_clk_freq": 100000000,
    "enum_eeprom": True,
    "eeprom_reg_addr_bits": 8,
    "datapath_width": 8,
    "datauser_width": 1,
    "sensor_rx_count": None,           # no RX (TX-only)
    "sif_rx_data_gen": False,
    "sif_rx_widths": None,
    "sif_rx_packetizer_en": None,
    "sif_rx_vp_count": None,
    "sif_rx_sort_resolution": None,
    "sif_rx_vp_size": None,
    "sif_rx_num_cycles": None,
    "sensor_tx_count": 1,
    "sif_tx_widths": None,
    "sif_tx_buf_size": [2048],
    "host_width": 8,
    "host_if_inst": 1,
    "host_mtu": 1500,
    "spi_inst": 1,
    "i2c_inst": 1,
    "uart_inst": None,
    "gpio_inst": 54,
    "gpio_reset_value": "'0",
    "reg_inst": 8,
    "init_reg": [
        ("32'h0200_0024", "32'h0000_12B7"),
    ],
    "ext_ptp": False,
    "sync_clk_hif_apb": False,
    "sync_clk_hif_ptp": False,
    "peri_ram_depth": 32,
    "disable_coe": False,
}

_VERY_HIGH_SPEED: Dict[str, Any] = {
    "archetype": "very-high-speed",
    "uuid": None,
    "hif_clk_freq": 322265625,
    "apb_clk_freq": 50000000,
    "ptp_clk_freq": 100000000,
    "enum_eeprom": False,              # soft enumeration
    "eeprom_reg_addr_bits": 8,
    "datapath_width": 512,
    "datauser_width": 1,
    "sensor_rx_count": 1,
    "sif_rx_data_gen": True,           # bring-up; disable in production
    "sif_rx_widths": None,
    "sif_rx_packetizer_en": None,
    "sif_rx_vp_count": [4],
    "sif_rx_sort_resolution": [16],
    "sif_rx_vp_size": [128],
    "sif_rx_num_cycles": [1],
    "sensor_tx_count": 1,
    "sif_tx_widths": None,
    "sif_tx_buf_size": [2048],
    "host_width": 512,
    "host_if_inst": 1,
    "host_mtu": 4096,
    "spi_inst": 1,
    "i2c_inst": 1,
    "uart_inst": None,
    "gpio_inst": 31,
    "gpio_reset_value": "'0",
    "reg_inst": 8,
    "init_reg": [
        ("32'h0120_0000", "32'h0000_0001"),
    ],
    "ext_ptp": False,
    "sync_clk_hif_apb": False,
    "sync_clk_hif_ptp": False,
    "peri_ram_depth": None,
    "disable_coe": False,
}


_GIGABIT_BASELINE: Dict[str, Any] = {
    "archetype": "gigabit-baseline",
    "uuid": None,
    "hif_clk_freq": 125000000,         # 1 Gbps Ethernet (8 bits × 125 MHz)
    "apb_clk_freq": 19531250,
    "ptp_clk_freq": 100446545,
    "enum_eeprom": True,
    "eeprom_reg_addr_bits": 8,
    "datapath_width": 8,
    "datauser_width": 1,
    "sensor_rx_count": 1,
    "sif_rx_data_gen": False,
    "sif_rx_widths": None,
    "sif_rx_packetizer_en": None,
    "sif_rx_vp_count": [1],
    "sif_rx_sort_resolution": [8],
    "sif_rx_vp_size": [64],
    "sif_rx_num_cycles": [1],
    "sensor_tx_count": 1,
    "sif_tx_widths": None,
    "sif_tx_buf_size": [2048],
    "host_width": 8,
    "host_if_inst": 1,
    "host_mtu": 1500,
    "spi_inst": 1,
    "i2c_inst": 1,
    "uart_inst": None,
    "gpio_inst": 16,
    "gpio_reset_value": "'0",
    "reg_inst": 1,
    "init_reg": [
        ("32'h0200_0024", "32'h0000_12B7"),
    ],
    "ext_ptp": False,
    "sync_clk_hif_apb": False,
    "sync_clk_hif_ptp": False,
    "peri_ram_depth": None,
    "disable_coe": False,
}


ARCHETYPE_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "high-bandwidth-single-sensor": _HIGH_BANDWIDTH_SINGLE_SENSOR,
    "mid-bandwidth-baseline": _MID_BANDWIDTH_BASELINE,
    "gigabit-baseline": _GIGABIT_BASELINE,
    "minimal-gateway": _MINIMAL_GATEWAY,
    "ultra-minimal": _ULTRA_MINIMAL,
    "very-high-speed": _VERY_HIGH_SPEED,
}


# Documentation-only placeholder used in the static archetype samples in
# `references/archetypes.md`. The emitter never falls back to it at generation
# time — `generate_def.py` creates a random UUID upstream when none is supplied.
PLACEHOLDER_UUID = "128'h0000_0000_0000_0000_0000_0000_0000_0000"


# Generic per-macro defaults used when no archetype is selected. Start from the
# mid-bandwidth-baseline values, but
# override opt-in hardware behavior that should never appear unless the user
# asked for it. Sensor-data manipulation (rearrange, swizzle, split, replicate)
# via the packetizer is opt-in: if the user needs it, a separate packetizer skill
# will configure SIF_RX_PACKETIZER_EN and the four "do not change" arrays. The
# boot-time init_reg[] sequence is also opt-in; the HSB IP does not require it,
# and generic generation must not inherit the mid-bandwidth archetype's example
# host UDP-port writes.
GENERIC_DEFAULTS: Dict[str, Any] = dict(_MID_BANDWIDTH_BASELINE)
GENERIC_DEFAULTS["archetype"] = None  # signal: no archetype selected
GENERIC_DEFAULTS["sif_rx_packetizer_en"] = [0]  # reconcile_arrays will extend to match sensor_rx_count
GENERIC_DEFAULTS["init_reg"] = []


# -----------------------------------------------------------------------------
# Profile assembly
# -----------------------------------------------------------------------------

_RX_ARRAY_KEYS = (
    "sif_rx_widths", "sif_rx_packetizer_en",
    "sif_rx_vp_count", "sif_rx_sort_resolution",
    "sif_rx_vp_size", "sif_rx_num_cycles",
)
_TX_ARRAY_KEYS = ("sif_tx_widths", "sif_tx_buf_size")


def _reconcile_array_length(arr: Optional[List[Any]], target: int) -> Optional[List[Any]]:
    """Truncate or extend an array to a target length. Extend by repeating last element."""
    if arr is None:
        return None
    if len(arr) == target:
        return list(arr)
    if len(arr) > target:
        return list(arr[:target])
    fill = arr[-1] if arr else 0
    return list(arr) + [fill] * (target - len(arr))


def _reconcile_arrays(profile: Dict[str, Any]) -> List[str]:
    """Make per-port arrays match interface counts. Returns a list of NOTE strings
    describing any reconciliation that happened."""
    notes: List[str] = []
    rx = profile.get("sensor_rx_count")
    if rx is None:
        # No RX interfaces — drop any RX arrays so the emitter doesn't write them.
        for k in _RX_ARRAY_KEYS:
            if profile.get(k) is not None:
                notes.append(f"{k}: dropped (no Sensor RX interfaces)")
                profile[k] = None
    else:
        for k in _RX_ARRAY_KEYS:
            arr = profile.get(k)
            if arr is None:
                continue
            if len(arr) != rx:
                profile[k] = _reconcile_array_length(arr, rx)
                notes.append(f"{k}: resized {len(arr)}→{rx} to match SENSOR_RX_IF_INST")
    tx = profile.get("sensor_tx_count")
    if tx is None:
        for k in _TX_ARRAY_KEYS:
            if profile.get(k) is not None:
                notes.append(f"{k}: dropped (no Sensor TX interfaces)")
                profile[k] = None
    else:
        for k in _TX_ARRAY_KEYS:
            arr = profile.get(k)
            if arr is None:
                continue
            if len(arr) != tx:
                profile[k] = _reconcile_array_length(arr, tx)
                notes.append(f"{k}: resized {len(arr)}→{tx} to match SENSOR_TX_IF_INST")
    return notes


def _clamp_reg_inst(profile: Dict[str, Any]) -> Optional[int]:
    """REG_INST has a hard minimum of 1 in the IP. Clamp anything lower; return
    the original value (so the caller can NOTE the clamp), or None if unchanged."""
    val = profile.get("reg_inst")
    if val is None:
        profile["reg_inst"] = 1
        return None
    if val < 1:
        profile["reg_inst"] = 1
        return val
    return None


def build_profile(archetype: Optional[str],
                  overrides: Optional[Dict[str, Any]] = None
                  ) -> Tuple[Dict[str, Any], List[str]]:
    """Return (profile, notes). The profile is a complete dict ready for `emit_svh`,
    with per-port arrays reconciled to interface counts and `reg_inst` clamped to ≥1.
    Notes are human-readable strings describing any automatic adjustments — the
    caller (`generate_def.py`) prints them to stderr.

    The baseline is `ARCHETYPE_DEFAULTS[archetype]` when `archetype` is given, or
    `GENERIC_DEFAULTS` otherwise. Profile keys are flat (e.g. `hif_clk_freq`,
    `datapath_width`) — see `generate_def.py --help` for the schema.
    """
    if archetype is None:
        profile = dict(GENERIC_DEFAULTS)
    elif archetype in ARCHETYPE_DEFAULTS:
        profile = dict(ARCHETYPE_DEFAULTS[archetype])
    else:
        raise ValueError(f"Unknown archetype: {archetype!r}. "
                         f"Valid choices: {sorted(ARCHETYPE_DEFAULTS.keys())}")
    if overrides:
        for k, v in overrides.items():
            profile[k] = v

    notes: List[str] = []
    notes.extend(_reconcile_arrays(profile))
    clamped = _clamp_reg_inst(profile)
    if clamped is not None:
        notes.append(f"REG_INST: clamped {clamped}→1 (IP minimum is 1; "
                     f"a design with no user logic still needs one register block)")
    return profile, notes


# -----------------------------------------------------------------------------
# SVH emitter — matches the formatting style of the maintained corpus
# -----------------------------------------------------------------------------

_BAR_THIN = "-" * 53   # `//-----------------------------------------------------`
_BAR_HEAVY = "-" * 78  # `//------------------------------------------------------------------------------`

_SPDX_HEADER = (
    "// SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.\n"
    "// SPDX-License-Identifier: Apache-2.0\n"
    "//\n"
    "// Licensed under the Apache License, Version 2.0 (the \"License\");\n"
    "// you may not use this file except in compliance with the License.\n"
    "// You may obtain a copy of the License at\n"
    "//\n"
    "// http://www.apache.org/licenses/LICENSE-2.0\n"
    "//\n"
    "// Unless required by applicable law or agreed to in writing, software\n"
    "// distributed under the License is distributed on an \"AS IS\" BASIS,\n"
    "// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.\n"
    "// See the License for the specific language governing permissions and\n"
    "// limitations under the License.\n"
    "\n"
)


def _section(a, title: str, subtitle: Optional[str] = None, heavy: bool = False) -> None:
    """Emit a corpus-style section banner."""
    bar = _BAR_HEAVY if heavy else _BAR_THIN
    a(f"//{bar}\n")
    a(f"// {title}\n")
    if subtitle:
        a("//\n")
        for line in subtitle.split("\n"):
            a(f"// {line}\n")
    a(f"//{bar}\n")
    a("\n")


def _array_positional(name: str, size_macro: str, values: List[Any]) -> str:
    """Render `localparam integer NAME [size-1:0] = {v1, v2, ...};`"""
    body = ", ".join(str(v) for v in values)
    return f"    localparam integer  {name:<22} [{size_macro}-1:0] = {{{body}}};\n"


def _array_default(name: str, size_macro: str, default_expr: str, comment: str = "") -> str:
    """Render `localparam integer NAME [size-1:0] = '{default: <expr>};`"""
    line = f"    localparam integer  {name:<22} [{size_macro}-1:0] = '{{default:{default_expr}}};"
    if comment:
        return f"{line} {comment}\n"
    return f"{line}\n"


def _array_uniform_or_positional(name: str, size_macro: str,
                                  values: List[Any], comment: str = "") -> str:
    """If all values are identical, emit as `'{default:V}` form (matches corpus
    style for SIF_RX_PACKETIZER_EN, SIF_RX_WIDTH defaults, etc.). Otherwise emit
    positional `{v1, v2, ...}`."""
    if values and len(set(str(v) for v in values)) == 1:
        return _array_default(name, size_macro, str(values[0]), comment)
    body = ", ".join(str(v) for v in values)
    line = f"    localparam integer  {name:<22} [{size_macro}-1:0] = {{{body}}};"
    if comment:
        return f"{line} {comment}\n"
    return f"{line}\n"


def emit_svh(p: Dict[str, Any]) -> str:
    """Render a Profile dict to a HOLOLINK_def.svh string in corpus style."""
    lines: List[str] = []
    a = lines.append

    a(_SPDX_HEADER)
    a("`ifndef HOLOLINK_def\n")
    a("`define HOLOLINK_def\n")
    a("\n")
    a("package HOLOLINK_pkg;\n")
    a("\n")

    # ---- Clocks (3 separate sections, mirroring corpus) ----
    _section(a, "Holoscan IP Host Clock Frequency",
             "Used for internal timer calculation")
    a(f"  `define HIF_CLK_FREQ  {p['hif_clk_freq']}\n")
    a("\n")

    _section(a, "Holoscan IP APB Clock Frequency",
             "Used for I2C clock divider setting")
    a(f"  `define APB_CLK_FREQ  {p['apb_clk_freq']}\n")
    a("\n")

    _section(a, "Holoscan IP PTP Clock Frequency",
             "Used for internal timer calculation")
    a(f"  `define PTP_CLK_FREQ  {p['ptp_clk_freq']}\n")
    a("\n")

    # ---- Board Info Enumeration ----
    _section(a, "Board Info Enumeration")
    a("  //UUID is used to uniquely identify the board. The UUID is sent over BOOTP.\n")
    uuid = p.get("uuid") or PLACEHOLDER_UUID
    a(f"  `define UUID                   {uuid}\n")
    a("\n")
    a("  // Define ENUM_EEPROM if board info is stored in an external EEPROM.\n")
    a("  // Otherwise, soft MAC address and Board Serial Number can be used\n")
    if p.get("enum_eeprom"):
        a("  `define ENUM_EEPROM\n")
        a("\n")
        a("  `ifdef ENUM_EEPROM\n")
        a(f"    `define EEPROM_REG_ADDR_BITS {p['eeprom_reg_addr_bits']}                //EEPROM Register Address Bits. Valid values: 8, 16\n")
        a("  `endif\n")
    else:
        a("  //`define ENUM_EEPROM\n")
    a("\n")

    # ---- Sensor Interface (datapath widths) ----
    _section(a, "Sensor Interface")
    a(f"  `define DATAPATH_WIDTH  {p['datapath_width']}                 // Sensor interface data width. This should be set to MAX width between SIF RX and TX widths\n")
    a("                                             // Valid values: 8, 16, 32, 64, 128, 256, 512, 1024\n")
    a("  `define DATAKEEP_WIDTH  `DATAPATH_WIDTH/8  // Sensor interface data keep width\n")
    a(f"  `define DATAUSER_WIDTH  {p['datauser_width']}                  // Sensor interface data user width\n")
    a("\n")

    # ---- Sensor RX IF ----
    _section(a, "Sensor RX IF")
    rx = p.get("sensor_rx_count")
    if rx is not None:
        a(f"  `define SENSOR_RX_IF_INST  {rx}               // Number of Sensor RX Interface. Valid values: undefined, 1 - 32\n")
        a("  //----------------------------------------------------------------------------------\n")
        a("  //If no Sensor RX Interfaces are used, then comment out \"`define SENSOR_RX_IF_INST\"\n")
        a("  //This will remove Sensor RX IF I/Os from HOLOLINK_top module.\n")
        a("  //The same applies for \"SENSOR_TX_IF_INST\", \"SPI_INST\", and \"I2C_INST\" definitions.\n")
        a("  //----------------------------------------------------------------------------------\n")
        a("\n")
        a("  `ifdef SENSOR_RX_IF_INST\n")
        if p.get("sif_rx_data_gen"):
            a("    `define SIF_RX_DATA_GEN             // If defined, Sensor RX Data Generator is instantiated. This can be used for bring-up.\n")
        else:
            a("    //`define SIF_RX_DATA_GEN             // If defined, Sensor RX Data Generator is instantiated. This can be used for bring-up.\n")
        a("\n")
        # SIF_RX_WIDTH
        widths = p.get("sif_rx_widths")
        if widths is None:
            a(_array_default("SIF_RX_WIDTH", "`SENSOR_RX_IF_INST",
                             "`DATAPATH_WIDTH",
                             "// Define width for each interface."))
        else:
            a(_array_positional("SIF_RX_WIDTH", "`SENSOR_RX_IF_INST", widths)
              .rstrip("\n") + " // Define width for each interface.\n")
        a("    //--------------------------------------------------------------------------------\n")
        a("    // Sensor RX Packetizer Parameters\n")
        a("    // If RX_PACKETIZER_EN is set to 0, then Packetizer is disabled for that Sensor RX interface.\n")
        a("    //--------------------------------------------------------------------------------\n")
        # SIF_RX_PACKETIZER_EN — uniform values render as `'{default:V}` to
        # match corpus style (e.g., '{default:0} when packetizer is off across
        # all RX interfaces, '{default:1} when on across all)
        pen = p.get("sif_rx_packetizer_en")
        if pen is None:
            a(_array_default("SIF_RX_PACKETIZER_EN", "`SENSOR_RX_IF_INST", "1"))
        else:
            a(_array_uniform_or_positional("SIF_RX_PACKETIZER_EN", "`SENSOR_RX_IF_INST", pen))
        # The four "do not change" arrays
        for name, key in [("SIF_RX_VP_COUNT", "sif_rx_vp_count"),
                          ("SIF_RX_SORT_RESOLUTION", "sif_rx_sort_resolution"),
                          ("SIF_RX_VP_SIZE", "sif_rx_vp_size"),
                          ("SIF_RX_NUM_CYCLES", "sif_rx_num_cycles")]:
            vals = p.get(key)
            if vals is not None:
                a(_array_positional(name, "`SENSOR_RX_IF_INST", vals))
        a("  `endif\n")
    else:
        a("  //`define SENSOR_RX_IF_INST  1               // Number of Sensor RX Interface. Valid values: undefined, 1 - 32\n")
    a("\n")

    # ---- Sensor TX IF ----
    _section(a, "Sensor TX IF")
    tx = p.get("sensor_tx_count")
    if tx is not None:
        a(f"  `define SENSOR_TX_IF_INST  {tx}               // Number of Sensor TX Interface. Valid values: undefined, 1 - 32\n")
        a("\n")
        a("  `ifdef SENSOR_TX_IF_INST\n")
        tx_widths = p.get("sif_tx_widths")
        if tx_widths is None:
            a(_array_default("SIF_TX_WIDTH", "`SENSOR_TX_IF_INST",
                             "`DATAPATH_WIDTH",
                             "// Define width for each interface."))
        elif len(set(tx_widths)) == 1:
            a(_array_default("SIF_TX_WIDTH", "`SENSOR_TX_IF_INST", str(tx_widths[0]),
                             "// Define width for each interface."))
        else:
            a(_array_positional("SIF_TX_WIDTH", "`SENSOR_TX_IF_INST", tx_widths)
              .rstrip("\n") + " // Define width for each interface.\n")
        bufs = p.get("sif_tx_buf_size")
        if bufs is None or len(set(bufs)) == 1:
            buf_val = str(bufs[0]) if bufs else "2048"
            a(_array_default("SIF_TX_BUF_SIZE", "`SENSOR_TX_IF_INST", buf_val,
                             "// Define buffer size for each interface."))
        else:
            a(_array_positional("SIF_TX_BUF_SIZE", "`SENSOR_TX_IF_INST", bufs)
              .rstrip("\n") + " // Define buffer size for each interface.\n")
        a("  `endif\n")
    else:
        a("  //`define SENSOR_TX_IF_INST  1               // Number of Sensor TX Interface. Valid values: undefined, 1 - 32\n")
    a("\n")

    # ---- Host IF ----
    _section(a, "Host IF")
    a(f"  `define HOST_WIDTH      {p['host_width']}                // Host interface data width.                     Valid values: 8, 16, 32, 64, 128, 256, 512\n")
    a("  `define HOSTKEEP_WIDTH  `HOST_WIDTH/8     // Host interface data keep width\n")
    a("  `define HOSTUSER_WIDTH  1                 // Host interface data user width\n")
    a(f"  `define HOST_IF_INST    {p['host_if_inst']}                 // Host interface instantiation number.           Valid values: 1 - 32\n")
    a(f"  `define HOST_MTU        {p['host_mtu']}              // Maximum Transmission Unit for Ethernet packet. Valid values: 1500, 4096\n")
    a("\n")

    # ---- Peripheral Control (heavy bar) ----
    _section(a, "Peripheral Control", heavy=True)
    spi = p.get("spi_inst")
    i2c = p.get("i2c_inst")
    uart = p.get("uart_inst")
    if spi is not None:
        a(f"  `define SPI_INST  {spi}   // SPI interface instantiation number. Valid values: undefined, 1 - 8\n")
    else:
        a("  //`define SPI_INST  1   // SPI interface instantiation number. Valid values: undefined, 1 - 8\n")
    if i2c is not None:
        a(f"  `define I2C_INST  {i2c}   // I2C interface instantiation number. Valid values: undefined, 1 - 8\n")
    else:
        a("  //`define I2C_INST  1   // I2C interface instantiation number. Valid values: undefined, 1 - 8\n")
    if uart is not None:
        a(f"  `define UART_INST {uart}   // UART interface instantiation number. Valid values: undefined, 1\n")
    else:
        a("  //`define UART_INST 1   // UART interface instantiation number. Valid values: undefined, 1\n")
    a(f"  `define GPIO_INST {p['gpio_inst']}  // INOUT GPIO instantiation number.    Valid values: 1 - 255\n")
    a("\n")
    a(f"  localparam [`GPIO_INST-1:0] GPIO_RESET_VALUE = {p['gpio_reset_value']};\n")
    if p.get("peri_ram_depth") is not None:
        a(f"  `define PERI_RAM_DEPTH {p['peri_ram_depth']}\n")
    a("\n")

    # ---- Optional in-scope toggles (only emit when set) ----
    advanced = []
    for flag_name, key in [("EXT_PTP", "ext_ptp"),
                           ("SYNC_CLK_HIF_APB", "sync_clk_hif_apb"),
                           ("SYNC_CLK_HIF_PTP", "sync_clk_hif_ptp"),
                           ("DISABLE_COE", "disable_coe")]:
        if p.get(key):
            advanced.append(flag_name)
    if advanced:
        for flag in advanced:
            a(f"  `define {flag}\n")
        a("\n")

    # ---- Register IF (heavy bar) ----
    _section(a, "Register IF",
             "Creates <REG_INST> number of APB register interfaces for user logic access",
             heavy=True)
    a(f"  `define REG_INST {p['reg_inst']}\n")
    a("\n")

    # ---- System Initialization (heavy bar) ----
    init_reg = p.get("init_reg") or []
    if init_reg:
        _section(a, "System Initialization",
                 "Initialization for the Host Interface registers so communication can be\n"
                 "established between the Device and the Host",
                 heavy=True)
        a(f"  `define N_INIT_REG {len(init_reg)}\n")
        a("\n")
        a("  localparam logic [63:0] init_reg [`N_INIT_REG] = '{\n")
        a("    // 32b Addr   | 32b Data\n")
        last_idx = len(init_reg) - 1
        for i, (addr, data) in enumerate(init_reg):
            comma = "," if i < last_idx else ""
            a(f"    {{{addr}, {data}}}{comma}\n")
        a("  };\n")
        a("\n")

    a("endpackage: HOLOLINK_pkg\n")
    a("`endif\n")

    return "".join(lines)
