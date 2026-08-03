"""Rules engine for HOLOLINK_def.svh validation.

Public entry point: run_all_rules(parsed) -> List[Finding]
                    archetype_classify(parsed) -> Optional[str]

Each rule is a small function that takes a ParsedDef and returns 0+ Findings.
Rule IDs (HD-Exxx, HD-Wxxx, HD-Ixxx) match references/validation-rules.md.
The skill's text content must contain ZERO references to out-of-scope macro
names; this module enforces that by treating any macro outside the
KNOWN_IP_MACROS allowlist as silent pass-through.
"""

from dataclasses import dataclass
from typing import Callable, List, Optional, Set, Tuple

from .parser import (
    ParsedDef,
    define_int,
    define_value,
    is_defined,
    parse_int_literal,
    vlog_literal_width,
)


# -----------------------------------------------------------------------------
# Known IP macros — the allowlist
# -----------------------------------------------------------------------------

KNOWN_IP_MACROS: Set[str] = {
    # Clocks
    "HIF_CLK_FREQ", "APB_CLK_FREQ", "PTP_CLK_FREQ",
    # Identity
    "UUID",
    # Enumeration
    "ENUM_EEPROM", "EEPROM_REG_ADDR_BITS",
    # Sensor datapath
    "DATAPATH_WIDTH", "DATAKEEP_WIDTH", "DATAUSER_WIDTH",
    # Sensor RX
    "SENSOR_RX_IF_INST", "SIF_RX_DATA_GEN",
    # Sensor TX
    "SENSOR_TX_IF_INST",
    # Host
    "HOST_WIDTH", "HOSTKEEP_WIDTH", "HOSTUSER_WIDTH", "HOST_IF_INST", "HOST_MTU",
    # Peripherals
    "SPI_INST", "I2C_INST", "UART_INST", "GPIO_INST",
    # PTP / clock sync
    "EXT_PTP", "SYNC_CLK_HIF_APB", "SYNC_CLK_HIF_PTP",
    # Registers / init
    "REG_INST", "N_INIT_REG",
    # Other in-scope undocumented IP macros
    "PERI_RAM_DEPTH", "DISABLE_COE",
    # Wrapper guard (always allowed, exempt from rules)
    "HOLOLINK_def",
}

KNOWN_IP_ARRAYS: Set[str] = {
    "SIF_RX_WIDTH", "SIF_RX_PACKETIZER_EN",
    "SIF_RX_VP_COUNT", "SIF_RX_SORT_RESOLUTION",
    "SIF_RX_VP_SIZE", "SIF_RX_NUM_CYCLES",
    "SIF_TX_WIDTH", "SIF_TX_BUF_SIZE",
    "GPIO_RESET_VALUE", "init_reg",
}

# BUILD_REV is treated specially: it's a Verilog parameter on HOLOLINK_top,
# not a `define. We warn if it appears as a `define.
SPECIAL_DEFINES = {"BUILD_REV"}


PTP_FREQ_BAND_HZ = (95_000_000, 105_000_000)


# -----------------------------------------------------------------------------
# Finding data class
# -----------------------------------------------------------------------------

@dataclass
class Finding:
    rule: str
    severity: str   # 'error' | 'warning' | 'info'
    line: Optional[int]
    macro: Optional[str]
    msg: str

    def to_dict(self) -> dict:
        return {
            "rule": self.rule,
            "severity": self.severity,
            "line": self.line,
            "macro": self.macro,
            "msg": self.msg,
        }


def _with_severity(severity: str) -> Callable[[str, Optional[int], Optional[str], str], Finding]:
    def make(rid: str, line: Optional[int], macro: Optional[str], msg: str) -> Finding:
        return Finding(rid, severity, line, macro, msg)
    return make


_err = _with_severity("error")
_warn = _with_severity("warning")
_info = _with_severity("info")


# -----------------------------------------------------------------------------
# Helper utilities
# -----------------------------------------------------------------------------

def is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def get_array_length(parsed: ParsedDef, name: str) -> Optional[int]:
    """Return the resolved length of an array, considering both the
    declared size and the count of literal elements (if enumerated)."""
    if name not in parsed.arrays:
        return None
    arr = parsed.arrays[name]
    if arr.is_default_init:
        return arr.size_value
    if arr.elements:
        return len(arr.elements)
    return arr.size_value


def array_int_elements(parsed: ParsedDef, name: str) -> List[Tuple[int, Optional[int]]]:
    """Return [(idx, value_or_None)] for each element of an array."""
    if name not in parsed.arrays:
        return []
    arr = parsed.arrays[name]
    if arr.is_default_init:
        size = arr.size_value or 0
        v = parse_int_literal(arr.default_value or "")
        if v is None:
            from .parser import eval_expr
            v = eval_expr(arr.default_value or "", parsed.defines)
        return [(i, v) for i in range(size)]
    out = []
    from .parser import eval_expr
    for i, e in enumerate(arr.elements):
        v = parse_int_literal(e)
        if v is None:
            v = eval_expr(e, parsed.defines)
        out.append((i, v))
    return out


# -----------------------------------------------------------------------------
# Rule checks
# -----------------------------------------------------------------------------

def check_wrapper(p: ParsedDef) -> List[Finding]:
    out: List[Finding] = []
    if not p.wrapper.has_ifndef_guard:
        out.append(_err("HD-E001", 1, None,
            "File must begin with `ifndef HOLOLINK_def guard. See macro-reference.md § File wrapper requirement."))
    if p.wrapper.has_ifndef_guard and not p.wrapper.has_define_guard:
        out.append(_err("HD-E002", 1, None,
            "After `ifndef HOLOLINK_def, the next directive must be `define HOLOLINK_def."))
    if not p.wrapper.has_package_decl:
        out.append(_err("HD-E003", None, None,
            "Missing package HOLOLINK_pkg; declaration. The file must be wrapped in this package."))
    if not p.wrapper.has_endpackage:
        out.append(_err("HD-E004", None, None,
            "Missing or malformed endpackage: HOLOLINK_pkg. Close the package before the final `endif."))
    if not p.wrapper.has_endif:
        out.append(_err("HD-E005", None, None,
            "Missing closing `endif for the `ifndef HOLOLINK_def guard."))
    return out


def check_clocks(p: ParsedDef) -> List[Finding]:
    out: List[Finding] = []
    for rid, name in [("HD-E101", "HIF_CLK_FREQ"), ("HD-E102", "APB_CLK_FREQ"), ("HD-E103", "PTP_CLK_FREQ")]:
        v = define_int(p, name)
        if v is None:
            if is_defined(p, name):
                out.append(_err(rid, p.defines[name][1], name,
                    f"{name} must be a positive integer (Hz). Got: {define_value(p, name)!r}"))
        elif v <= 0:
            out.append(_err(rid, p.defines[name][1], name,
                f"{name} must be a positive integer (Hz). Got: {v}"))
    # PTP frequency band
    ptp = define_int(p, "PTP_CLK_FREQ")
    if ptp is not None and (ptp < PTP_FREQ_BAND_HZ[0] or ptp > PTP_FREQ_BAND_HZ[1]):
        out.append(_warn("HD-W207", p.defines.get("PTP_CLK_FREQ", (None, None))[1], "PTP_CLK_FREQ",
            f"PTP_CLK_FREQ = {ptp} Hz is outside the HSB-documented range of 95–105 MHz. PTP timing accuracy may be degraded."))
    return out


def check_uuid(p: ParsedDef) -> List[Finding]:
    out: List[Finding] = []
    if not is_defined(p, "UUID"):
        out.append(_err("HD-E104", None, "UUID",
            "UUID is required. Define a 128-bit hex literal — never derive at runtime."))
        return out
    val = define_value(p, "UUID")
    line = p.defines["UUID"][1]
    width = vlog_literal_width(val or "")
    if width is None:
        out.append(_err("HD-E106", line, "UUID",
            f"UUID literal is malformed; expected `128'h<32 hex chars, optionally `_`-separated>`. Got: {val!r}"))
        return out
    if width != 128:
        out.append(_err("HD-E105", line, "UUID",
            f"UUID must be exactly 128 bits (e.g. `128'h…`). Got: {width}-bit literal."))
    return out


def check_enum_eeprom(p: ParsedDef) -> List[Finding]:
    out: List[Finding] = []
    if is_defined(p, "ENUM_EEPROM"):
        bits = define_int(p, "EEPROM_REG_ADDR_BITS")
        if bits is None or bits not in (8, 16):
            line = p.defines.get("EEPROM_REG_ADDR_BITS", (None, None))[1]
            out.append(_err("HD-E107", line, "EEPROM_REG_ADDR_BITS",
                f"EEPROM_REG_ADDR_BITS must be 8 or 16 when ENUM_EEPROM is defined. Got: {bits!r}"))
    elif is_defined(p, "EEPROM_REG_ADDR_BITS"):
        line = p.defines["EEPROM_REG_ADDR_BITS"][1]
        out.append(_warn("HD-W303", line, "EEPROM_REG_ADDR_BITS",
            "EEPROM_REG_ADDR_BITS is defined but ENUM_EEPROM is undefined — the macro is dead. Either define ENUM_EEPROM or remove EEPROM_REG_ADDR_BITS."))
    return out


def check_datapath(p: ParsedDef) -> List[Finding]:
    out: List[Finding] = []
    name = "DATAPATH_WIDTH"
    v = define_int(p, name)
    line = p.defines.get(name, (None, None))[1]
    if v is None and is_defined(p, name):
        out.append(_err("HD-E108", line, name, f"{name} must be a positive integer."))
        return out
    if v is None:
        return out
    if v <= 0:
        out.append(_err("HD-E108", line, name, f"{name} must be a positive integer. Got: {v}"))
        return out
    if v % 8 != 0:
        out.append(_err("HD-E109", line, name,
            f"{name} must be byte-aligned (divisible by 8). Got: {v}"))
    if v > 1024:
        out.append(_warn("HD-W001", line, name,
            f"{name} = {v} is wider than the resource-warning threshold (1024). Confirm the target has sufficient logic and routing resources for the wider datapath."))
    if v > 0 and v % 8 == 0 and not is_power_of_two(v):
        out.append(_warn("HD-W101", line, name,
            f"{name} = {v} is byte-aligned but not a power of 2. AXI conventions favor powers of 2; some downstream IP may not handle non-power-of-2 widths cleanly."))
    # Derived: DATAKEEP_WIDTH must equal v/8
    keep = define_int(p, "DATAKEEP_WIDTH")
    keep_line = p.defines.get("DATAKEEP_WIDTH", (None, None))[1]
    if keep is not None and keep != v // 8:
        out.append(_err("HD-E201", keep_line, "DATAKEEP_WIDTH",
            f"DATAKEEP_WIDTH must equal DATAPATH_WIDTH/8. Got DATAKEEP_WIDTH={keep}, DATAPATH_WIDTH={v}, expected {v//8}."))
    # DATAUSER_WIDTH ∈ {1, 2}
    user = define_int(p, "DATAUSER_WIDTH")
    user_line = p.defines.get("DATAUSER_WIDTH", (None, None))[1]
    if user is not None and user not in (1, 2):
        out.append(_err("HD-E110", user_line, "DATAUSER_WIDTH",
            f"DATAUSER_WIDTH must be 1 or 2 for the current IP. Got: {user}"))
    return out


def check_host(p: ParsedDef) -> List[Finding]:
    out: List[Finding] = []
    name = "HOST_WIDTH"
    v = define_int(p, name)
    line = p.defines.get(name, (None, None))[1]
    if v is None and is_defined(p, name):
        out.append(_err("HD-E112", line, name, f"{name} must be a positive integer."))
    elif v is not None:
        if v <= 0:
            out.append(_err("HD-E112", line, name, f"{name} must be a positive integer. Got: {v}"))
        elif v % 8 != 0:
            out.append(_err("HD-E113", line, name,
                f"{name} must be byte-aligned (divisible by 8). Got: {v}"))
        else:
            if v > 1024:
                out.append(_warn("HD-W002", line, name,
                    f"{name} = {v} is wider than the resource-warning threshold (1024). Confirm the target has sufficient logic and routing resources."))
            if not is_power_of_two(v):
                out.append(_warn("HD-W102", line, name,
                    f"{name} = {v} is byte-aligned but not a power of 2 (AXI bus convention favors powers of 2)."))
        # HOSTKEEP_WIDTH derived equality
        keep = define_int(p, "HOSTKEEP_WIDTH")
        keep_line = p.defines.get("HOSTKEEP_WIDTH", (None, None))[1]
        if keep is not None and v % 8 == 0 and keep != v // 8:
            out.append(_err("HD-E202", keep_line, "HOSTKEEP_WIDTH",
                f"HOSTKEEP_WIDTH must equal HOST_WIDTH/8. Got HOSTKEEP_WIDTH={keep}, HOST_WIDTH={v}, expected {v//8}."))
    # HOSTUSER_WIDTH must be 1
    user = define_int(p, "HOSTUSER_WIDTH")
    user_line = p.defines.get("HOSTUSER_WIDTH", (None, None))[1]
    if user is not None and user != 1:
        out.append(_err("HD-E111", user_line, "HOSTUSER_WIDTH",
            f"HOSTUSER_WIDTH is fixed at 1 in the current IP. Got: {user}"))
    # HOST_IF_INST in {1..32}
    inst = define_int(p, "HOST_IF_INST")
    inst_line = p.defines.get("HOST_IF_INST", (None, None))[1]
    if inst is not None and not (1 <= inst <= 32):
        out.append(_err("HD-E114", inst_line, "HOST_IF_INST",
            f"HOST_IF_INST must be in the range 1..32. Got: {inst}"))
    # HOST_MTU
    mtu = define_int(p, "HOST_MTU")
    mtu_line = p.defines.get("HOST_MTU", (None, None))[1]
    if mtu is not None:
        if mtu <= 0:
            out.append(_err("HD-E115", mtu_line, "HOST_MTU",
                f"HOST_MTU must be a positive integer (bytes). Got: {mtu}"))
    return out


def check_sensor_rx(p: ParsedDef) -> List[Finding]:
    out: List[Finding] = []
    inst = define_int(p, "SENSOR_RX_IF_INST")
    inst_line = p.defines.get("SENSOR_RX_IF_INST", (None, None))[1]
    if inst is not None and not (1 <= inst <= 32):
        out.append(_err("HD-E116", inst_line, "SENSOR_RX_IF_INST",
            f"SENSOR_RX_IF_INST must be in the range 1..32 when defined. Got: {inst}. (Leave it undefined to disable RX.)"))
    # Footgun: if any SIF_RX_*[] array is declared but SENSOR_RX_IF_INST is undefined
    if not is_defined(p, "SENSOR_RX_IF_INST"):
        rx_arrays = ["SIF_RX_WIDTH", "SIF_RX_PACKETIZER_EN", "SIF_RX_VP_COUNT",
                     "SIF_RX_SORT_RESOLUTION", "SIF_RX_VP_SIZE", "SIF_RX_NUM_CYCLES"]
        for arr_name in rx_arrays:
            if arr_name in p.arrays:
                out.append(_warn("HD-W301", p.arrays[arr_name].line, arr_name,
                    f"{arr_name}[] is declared but SENSOR_RX_IF_INST is undefined. The RTL silently falls back to a 1-bit dummy interface — there is no compile error. Did you mean to define SENSOR_RX_IF_INST?"))
    if inst is None or inst < 1:
        return out
    dpw = define_int(p, "DATAPATH_WIDTH")
    # Per-element checks for SIF_RX_WIDTH
    for arr_name, e_id, w_id, p2_id in [
        ("SIF_RX_WIDTH", "HD-E123", "HD-W003", "HD-W103"),
    ]:
        if arr_name in p.arrays:
            arr = p.arrays[arr_name]
            length = get_array_length(p, arr_name)
            if length is not None and length != inst:
                out.append(_err("HD-E301", arr.line, arr_name,
                    f"{arr_name} array length ({length}) must equal SENSOR_RX_IF_INST ({inst})."))
            for idx, val in array_int_elements(p, arr_name):
                if val is None:
                    continue
                if val <= 0 or val % 8 != 0:
                    out.append(_err(e_id, arr.line, arr_name,
                        f"{arr_name}[{idx}] must be a positive byte-aligned integer. Got: {val}"))
                    continue
                if dpw is not None and val > dpw:
                    out.append(_err("HD-E203", arr.line, arr_name,
                        f"{arr_name}[{idx}] = {val} exceeds DATAPATH_WIDTH = {dpw}. Per-port widths must be ≤ the system-wide datapath."))
                if val > 1024:
                    out.append(_warn(w_id, arr.line, arr_name,
                        f"{arr_name}[{idx}] = {val} is wider than the resource-warning threshold (1024). Confirm logic and routing capacity."))
                if not is_power_of_two(val):
                    out.append(_warn(p2_id, arr.line, arr_name,
                        f"{arr_name}[{idx}] = {val} is not a power of 2 (AXI bus convention)."))
    # Packetizer enable
    if "SIF_RX_PACKETIZER_EN" in p.arrays:
        arr = p.arrays["SIF_RX_PACKETIZER_EN"]
        length = get_array_length(p, "SIF_RX_PACKETIZER_EN")
        if length is not None and length != inst:
            out.append(_err("HD-E302", arr.line, "SIF_RX_PACKETIZER_EN",
                f"SIF_RX_PACKETIZER_EN array length ({length}) must equal SENSOR_RX_IF_INST ({inst})."))
        for idx, val in array_int_elements(p, "SIF_RX_PACKETIZER_EN"):
            if val is not None and val not in (0, 1):
                out.append(_err("HD-E126", arr.line, "SIF_RX_PACKETIZER_EN",
                    f"SIF_RX_PACKETIZER_EN[{idx}] must be 0 or 1. Got: {val}"))
    # Conditional length on the four "do not change" arrays
    pen_elements = array_int_elements(p, "SIF_RX_PACKETIZER_EN")
    any_enabled = any(v == 1 for _, v in pen_elements) if pen_elements else False
    for arr_name, e_id in [
        ("SIF_RX_VP_COUNT", "HD-E303"),
        ("SIF_RX_SORT_RESOLUTION", "HD-E304"),
        ("SIF_RX_VP_SIZE", "HD-E305"),
        ("SIF_RX_NUM_CYCLES", "HD-E306"),
    ]:
        if arr_name in p.arrays and any_enabled:
            length = get_array_length(p, arr_name)
            if length is not None and length != inst:
                out.append(_err(e_id, p.arrays[arr_name].line, arr_name,
                    f"{arr_name} array length ({length}) must equal SENSOR_RX_IF_INST ({inst}) when any SIF_RX_PACKETIZER_EN element is 1."))
        elif any_enabled and arr_name not in p.arrays:
            out.append(_warn("HD-W305", None, arr_name,
                f"{arr_name}[] is required when any SIF_RX_PACKETIZER_EN element is 1. Provide the array (use `'{{default:<value>}}` for uniform configurations)."))
    # Don't-care info for elements where packetizer is disabled.
    # Aggregate per array to avoid spamming users with N×4 messages when many
    # interfaces have packetizer disabled.
    if pen_elements:
        disabled_idxs = [i for i, v in pen_elements if v == 0]
        if disabled_idxs:
            for arr_name in ["SIF_RX_VP_COUNT", "SIF_RX_SORT_RESOLUTION",
                             "SIF_RX_VP_SIZE", "SIF_RX_NUM_CYCLES"]:
                if arr_name not in p.arrays:
                    continue
                idx_str = _format_idx_ranges(disabled_idxs)
                out.append(_info("HD-I001", p.arrays[arr_name].line, arr_name,
                    f"{arr_name}[{idx_str}] is a don't-care because the corresponding "
                    f"SIF_RX_PACKETIZER_EN entries are 0; values ignored by the IP."))
    return out


def _format_idx_ranges(idxs: List[int]) -> str:
    """Compact a list of indices into a range expression, e.g. [0,1,2,4,5] -> '0-2, 4-5'."""
    if not idxs:
        return ""
    sorted_idxs = sorted(set(idxs))
    parts: List[str] = []
    start = prev = sorted_idxs[0]
    for n in sorted_idxs[1:]:
        if n == prev + 1:
            prev = n
            continue
        parts.append(f"{start}" if start == prev else f"{start}-{prev}")
        start = prev = n
    parts.append(f"{start}" if start == prev else f"{start}-{prev}")
    return ", ".join(parts)


def check_sensor_tx(p: ParsedDef) -> List[Finding]:
    out: List[Finding] = []
    inst = define_int(p, "SENSOR_TX_IF_INST")
    inst_line = p.defines.get("SENSOR_TX_IF_INST", (None, None))[1]
    if inst is not None and not (1 <= inst <= 32):
        out.append(_err("HD-E117", inst_line, "SENSOR_TX_IF_INST",
            f"SENSOR_TX_IF_INST must be in the range 1..32 when defined. Got: {inst}"))
    if not is_defined(p, "SENSOR_TX_IF_INST"):
        for arr_name in ["SIF_TX_WIDTH", "SIF_TX_BUF_SIZE"]:
            if arr_name in p.arrays:
                out.append(_warn("HD-W302", p.arrays[arr_name].line, arr_name,
                    f"{arr_name}[] is declared but SENSOR_TX_IF_INST is undefined. RTL falls back silently — define SENSOR_TX_IF_INST or remove the array."))
        return out
    if inst is None or inst < 1:
        return out
    dpw = define_int(p, "DATAPATH_WIDTH")
    # SIF_TX_WIDTH
    if "SIF_TX_WIDTH" in p.arrays:
        arr = p.arrays["SIF_TX_WIDTH"]
        length = get_array_length(p, "SIF_TX_WIDTH")
        if length is not None and length != inst:
            out.append(_err("HD-E307", arr.line, "SIF_TX_WIDTH",
                f"SIF_TX_WIDTH array length ({length}) must equal SENSOR_TX_IF_INST ({inst})."))
        for idx, val in array_int_elements(p, "SIF_TX_WIDTH"):
            if val is None:
                continue
            if val <= 0 or val % 8 != 0:
                out.append(_err("HD-E124", arr.line, "SIF_TX_WIDTH",
                    f"SIF_TX_WIDTH[{idx}] must be a positive byte-aligned integer. Got: {val}"))
                continue
            if dpw is not None and val > dpw:
                out.append(_err("HD-E204", arr.line, "SIF_TX_WIDTH",
                    f"SIF_TX_WIDTH[{idx}] = {val} exceeds DATAPATH_WIDTH = {dpw}."))
            if val > 1024:
                out.append(_warn("HD-W004", arr.line, "SIF_TX_WIDTH",
                    f"SIF_TX_WIDTH[{idx}] = {val} is wider than the resource-warning threshold (1024). Confirm logic and routing capacity."))
            if not is_power_of_two(val):
                out.append(_warn("HD-W104", arr.line, "SIF_TX_WIDTH",
                    f"SIF_TX_WIDTH[{idx}] = {val} is not a power of 2."))
    # SIF_TX_BUF_SIZE
    if "SIF_TX_BUF_SIZE" in p.arrays:
        arr = p.arrays["SIF_TX_BUF_SIZE"]
        length = get_array_length(p, "SIF_TX_BUF_SIZE")
        if length is not None and length != inst:
            out.append(_err("HD-E308", arr.line, "SIF_TX_BUF_SIZE",
                f"SIF_TX_BUF_SIZE array length ({length}) must equal SENSOR_TX_IF_INST ({inst})."))
        for idx, val in array_int_elements(p, "SIF_TX_BUF_SIZE"):
            if val is None:
                continue
            if val <= 0:
                out.append(_err("HD-E125", arr.line, "SIF_TX_BUF_SIZE",
                    f"SIF_TX_BUF_SIZE[{idx}] must be a positive integer FIFO depth in SIF_TX_WIDTH elements. Got: {val}"))
                continue
            if val > 8192:
                out.append(_warn("HD-W005", arr.line, "SIF_TX_BUF_SIZE",
                    f"SIF_TX_BUF_SIZE[{idx}] = {val} exceeds the resource-warning threshold (8192 TX-width elements). Confirm the target has sufficient embedded RAM resources for the deeper FIFO."))
            if not is_power_of_two(val):
                out.append(_warn("HD-W105", arr.line, "SIF_TX_BUF_SIZE",
                    f"SIF_TX_BUF_SIZE[{idx}] = {val} is not a power of 2. Power-of-2 FIFO depths are easier for downstream memory implementations."))
    return out


def check_peripherals(p: ParsedDef) -> List[Finding]:
    out: List[Finding] = []
    for rid, name, lo, hi in [
        ("HD-E118", "SPI_INST", 1, 8),
        ("HD-E119", "I2C_INST", 1, 8),
    ]:
        v = define_int(p, name)
        if v is not None and not (lo <= v <= hi):
            out.append(_err(rid, p.defines[name][1], name,
                f"{name} must be in the range {lo}..{hi} when defined. Got: {v}"))
    # UART_INST: only 1 supported
    uart = define_int(p, "UART_INST")
    if uart is not None and uart != 1:
        out.append(_err("HD-E120", p.defines["UART_INST"][1], "UART_INST",
            f"UART_INST must be 1 when defined (only one UART instance is currently supported). Got: {uart}"))
    # GPIO_INST: 1..255
    gpio = define_int(p, "GPIO_INST")
    if gpio is not None and not (1 <= gpio <= 255):
        out.append(_err("HD-E121", p.defines["GPIO_INST"][1], "GPIO_INST",
            f"GPIO_INST must be in the range 1..255. Got: {gpio}"))
    # GPIO_RESET_VALUE: present when GPIO_INST defined; width = GPIO_INST
    if is_defined(p, "GPIO_INST"):
        if p.gpio_reset_value is None:
            out.append(_err("HD-E309", p.defines["GPIO_INST"][1], "GPIO_RESET_VALUE",
                "GPIO_RESET_VALUE must be declared whenever GPIO_INST is defined."))
        else:
            grv = p.gpio_reset_value
            literal_w = vlog_literal_width(grv.literal)
            if grv.width_value is not None and gpio is not None and grv.width_value != gpio:
                out.append(_err("HD-E127", grv.line, "GPIO_RESET_VALUE",
                    f"GPIO_RESET_VALUE must be sized exactly [GPIO_INST-1:0]. Declared width {grv.width_value}, expected {gpio}."))
            if literal_w is not None and gpio is not None and literal_w != gpio:
                out.append(_warn("HD-W306", grv.line, "GPIO_RESET_VALUE",
                    f"GPIO_RESET_VALUE is declared as {literal_w} bits but GPIO_INST = {gpio}. The mismatch is silently sign-extended or truncated by Verilog elaboration; declare it explicitly as [GPIO_INST-1:0] for clarity."))
    return out


def check_reg_inst(p: ParsedDef) -> List[Finding]:
    out: List[Finding] = []
    v = define_int(p, "REG_INST")
    line = p.defines.get("REG_INST", (None, None))[1]
    if v is not None and not (1 <= v <= 8):
        out.append(_err("HD-E122", line, "REG_INST",
            f"REG_INST must be in the range 1..8. Got: {v}"))
    return out


def check_init_reg(p: ParsedDef) -> List[Finding]:
    out: List[Finding] = []
    n_init = define_int(p, "N_INIT_REG")
    # N_INIT_REG = 0 causes elaboration failure: the IP's `ifdef N_INIT_REG`
    # gate is true, but sys_init's `i_init_reg [0]` zero-element array is
    # invalid SystemVerilog. The correct way to skip system init is to leave
    # the macro undefined entirely.
    if is_defined(p, "N_INIT_REG") and n_init == 0:
        line = p.defines.get("N_INIT_REG", (None, None))[1]
        out.append(_err("HD-E405", line, "N_INIT_REG",
            "N_INIT_REG = 0 is invalid: the IP's `ifdef N_INIT_REG` gate evaluates true and "
            "instantiates sys_init with a zero-element array, which fails elaboration. "
            "To skip the system-init sequence, leave N_INIT_REG undefined entirely "
            "(remove the `define) — the IP then omits sys_init cleanly."))
        return out
    if p.init_reg is None:
        if n_init is not None and n_init > 0:
            line = p.defines.get("N_INIT_REG", (None, None))[1]
            out.append(_err("HD-E404", line, "init_reg",
                f"N_INIT_REG is defined as {n_init} but init_reg[] is missing. Declare init_reg[] with exactly {n_init} entries."))
        return out
    # init_reg present
    if n_init is None:
        out.append(_err("HD-E403", p.init_reg.line, "init_reg",
            "init_reg[] is declared but N_INIT_REG is undefined. Define N_INIT_REG to match the array length."))
        return out
    actual = len(p.init_reg.entries)
    if actual != n_init:
        out.append(_err("HD-E401", p.init_reg.line, "init_reg",
            f"N_INIT_REG ({n_init}) must equal the literal length of the init_reg[] array ({actual})."))
    for k, entry in enumerate(p.init_reg.entries):
        addr_w = vlog_literal_width(entry.addr_raw)
        data_w = vlog_literal_width(entry.data_raw)
        if (addr_w is not None and addr_w != 32) or (data_w is not None and data_w != 32):
            out.append(_err("HD-E402", entry.line, "init_reg",
                f"init_reg[{k}] entry must be exactly 64 bits — `{{32'h<addr>, 32'h<data>}}`. Got addr={addr_w}b, data={data_w}b."))
    return out


def check_build_rev_define(p: ParsedDef) -> List[Finding]:
    out: List[Finding] = []
    if "BUILD_REV" in p.defines:
        line = p.defines["BUILD_REV"][1]
        out.append(_warn("HD-W304", line, "BUILD_REV",
            "BUILD_REV is a Verilog parameter on HOLOLINK_top, not a `define` macro. The `define here will be ignored. Pass BUILD_REV via parameter override at module instantiation: `HOLOLINK_top #(.BUILD_REV(48'h…)) u_…`."))
    return out


# -----------------------------------------------------------------------------
# Top-level driver
# -----------------------------------------------------------------------------

ALL_CHECKS: List[Callable[[ParsedDef], List[Finding]]] = [
    check_wrapper,
    check_clocks,
    check_uuid,
    check_enum_eeprom,
    check_datapath,
    check_host,
    check_sensor_rx,
    check_sensor_tx,
    check_peripherals,
    check_reg_inst,
    check_init_reg,
    check_build_rev_define,
]


def run_all_rules(parsed: ParsedDef) -> List[Finding]:
    findings: List[Finding] = []
    for check in ALL_CHECKS:
        try:
            findings.extend(check(parsed))
        except Exception as e:  # noqa: BLE001
            findings.append(_err("HD-INTERNAL", None, None,
                f"Internal validator error in {check.__name__}: {e}"))
    return findings


# -----------------------------------------------------------------------------
# Archetype classifier
# -----------------------------------------------------------------------------

def archetype_classify(p: ParsedDef) -> Optional[str]:
    """Best-effort classification by configuration. Returns slug or None."""
    dpw = define_int(p, "DATAPATH_WIDTH")
    hw = define_int(p, "HOST_WIDTH")
    rx = define_int(p, "SENSOR_RX_IF_INST")
    tx = define_int(p, "SENSOR_TX_IF_INST")
    host_inst = define_int(p, "HOST_IF_INST")
    hif = define_int(p, "HIF_CLK_FREQ")
    mtu = define_int(p, "HOST_MTU")

    # Ultra-minimal: 8-bit datapath, no RX (TX-only)
    if dpw == 8 and rx is None and tx is not None:
        return "ultra-minimal"
    # Very-high-speed: HIF >= 300 MHz, single sensor
    if hif is not None and hif >= 300_000_000 and (rx == 1 or rx is None):
        return "very-high-speed"
    # High-bandwidth single-sensor: HIF >= 200 MHz, dpw >= 256, single RX, dual host
    if (hif is not None and hif >= 200_000_000 and (dpw is not None and dpw >= 256)
            and (rx == 1) and (host_inst == 2)):
        return "high-bandwidth-single-sensor"
    # Gigabit baseline: 8-bit host + 1 GbE-style HIF clock (~125 MHz), with RX
    if (hw == 8 and dpw == 8 and rx is not None
            and hif is not None and 100_000_000 <= hif < 200_000_000):
        return "gigabit-baseline"
    # Minimal gateway: 1 sensor, 1 host, no TX, MTU 1500
    if (rx == 1 and host_inst == 1 and tx is None and mtu == 1500):
        return "minimal-gateway"
    # Mid-bandwidth baseline: 64-bit datapath + 64-bit host + dual host
    if (dpw == 64 and hw == 64 and host_inst == 2):
        return "mid-bandwidth-baseline"
    return None
