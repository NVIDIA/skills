"""Parser for HOLOLINK_def.svh files.

Public entry point: parse_file(path) -> ParsedDef

Handles:
  - Verilog preprocessor directives: `ifdef / `ifndef / `elsif / `else / `endif / `define
  - Block (/* ... */) and line (//) comments
  - Commented-out `define directives (these MUST be treated as undefined)
  - The package / `ifndef wrapper required by HOLOLINK_def.svh
  - localparam declarations for SIF_RX_*[], SIF_TX_*[], GPIO_RESET_VALUE, init_reg[]
  - Verilog integer literals (decimal, hex, binary) including sized forms (NN'h…)
  - Macro expansion in array sizes (`MACRO-1:0)
  - Array initializers '{default:X} and {a, b, c}

The parser is intentionally permissive: it extracts what it can, attaches
parse errors to fields it could not resolve, and lets the rules engine
decide which findings warrant errors.
"""

from dataclasses import dataclass, field
from pathlib import Path
import ast
import re
from typing import Dict, List, Optional, Tuple, Union


# -----------------------------------------------------------------------------
# Data types
# -----------------------------------------------------------------------------

@dataclass
class WrapperState:
    has_ifndef_guard: bool = False        # `ifndef HOLOLINK_def
    has_define_guard: bool = False        # `define HOLOLINK_def (after ifndef)
    has_package_decl: bool = False        # package HOLOLINK_pkg;
    has_endpackage: bool = False          # endpackage [: HOLOLINK_pkg]
    has_endif: bool = False               # final `endif
    package_line: Optional[int] = None
    endpackage_line: Optional[int] = None


@dataclass
class ArrayDecl:
    """A localparam array declaration."""
    name: str
    line: int
    size_expr: str                        # e.g. "`SENSOR_RX_IF_INST-1:0" or "3"
    size_value: Optional[int]             # resolved length, or None if unresolvable
    elements: List[str]                   # raw element strings
    is_default_init: bool                 # True if '{default:X} form
    default_value: Optional[str]          # the default expression if applicable
    raw_text: str                         # the full declaration source


@dataclass
class InitRegEntry:
    line: int
    addr_raw: str                         # raw address text, e.g. "32'h0200_0024"
    data_raw: str
    addr_value: Optional[int]
    data_value: Optional[int]


@dataclass
class InitRegArray:
    name: str = "init_reg"
    line: int = 0
    size_expr: str = ""
    declared_size_value: Optional[int] = None
    entries: List[InitRegEntry] = field(default_factory=list)


@dataclass
class GpioResetValue:
    line: int
    width_expr: str                       # e.g. "`GPIO_INST-1:0"
    width_value: Optional[int]
    literal: str                          # e.g. "16'b0000000000001111"


@dataclass
class ParseError:
    line: Optional[int]
    msg: str


@dataclass
class ParsedDef:
    path: str
    raw_lines: List[str] = field(default_factory=list)
    wrapper: WrapperState = field(default_factory=WrapperState)
    # Plain `defines (scalar): name -> (value_string_or_None, line)
    # value is None for boolean defines like `define ENUM_EEPROM`
    defines: Dict[str, Tuple[Optional[str], int]] = field(default_factory=dict)
    # Array localparams keyed by name
    arrays: Dict[str, ArrayDecl] = field(default_factory=dict)
    init_reg: Optional[InitRegArray] = None
    gpio_reset_value: Optional[GpioResetValue] = None
    # Track defines that appeared inside an inactive `ifdef branch (skipped),
    # so the validator doesn't complain about them.
    defines_skipped: List[str] = field(default_factory=list)
    parse_errors: List[ParseError] = field(default_factory=list)
    # Lines that LOOK like commented-out `define`s — useful for footgun reporting.
    commented_define_lines: List[Tuple[int, str]] = field(default_factory=list)


# -----------------------------------------------------------------------------
# Comment stripping
# -----------------------------------------------------------------------------

_BLOCK_COMMENT_RE = re.compile(r"/\*.*?\*/", re.DOTALL)


def _strip_block_comments(text: str) -> str:
    return _BLOCK_COMMENT_RE.sub(lambda m: " " * len(m.group(0)), text)


def _detect_commented_defines(raw_lines: List[str]) -> List[Tuple[int, str]]:
    """Return [(1-based line, define-name)] for lines that LOOK like
    commented-out `define directives. Used for footgun reporting.
    """
    pat = re.compile(r"^\s*//\s*`define\s+([A-Za-z_][A-Za-z0-9_]*)")
    out = []
    for i, line in enumerate(raw_lines, start=1):
        m = pat.match(line)
        if m:
            out.append((i, m.group(1)))
    return out


def _strip_line_comments(line: str) -> str:
    """Remove // comments from a single line (after block comments are gone)."""
    # Be careful with // inside string literals — but HOLOLINK_def.svh
    # contains no string literals, so a naive split is safe.
    idx = line.find("//")
    return line if idx < 0 else line[:idx]


# -----------------------------------------------------------------------------
# Verilog integer literal parsing
# -----------------------------------------------------------------------------

_VLOG_LITERAL_RE = re.compile(
    r"""^\s*
    (?P<width>\d+)?           # optional bit width
    \s*'(?P<base>[bBdDhHoO])  # base prefix
    (?P<digits>[\dA-Fa-f_xXzZ?]+)
    \s*$""",
    re.VERBOSE,
)


def parse_int_literal(s: str) -> Optional[int]:
    """Parse a Verilog or plain integer literal. Returns None on failure.

    Examples:
      "156250000"      -> 156250000
      "16'h2604"       -> 9732
      "8'b00001111"    -> 15
      "128'h7A37_7BF7…" -> integer
      "'0"             -> 0  (SystemVerilog all-zeros literal)
      "'1"             -> 1  (SystemVerilog all-ones literal, sign-context dependent)
    """
    if s is None:
        return None
    s = s.strip()
    if not s:
        return None
    # SystemVerilog 'fill literals: '0, '1, 'z, 'x
    if s in ("'0",):
        return 0
    if s in ("'1",):
        return 1  # all-ones; the actual numeric value depends on width context
    # Try plain decimal
    try:
        return int(s.replace("_", ""))
    except ValueError:
        pass
    m = _VLOG_LITERAL_RE.match(s)
    if not m:
        return None
    base_char = m.group("base").lower()
    digits = m.group("digits").replace("_", "")
    base_map = {"b": 2, "d": 10, "o": 8, "h": 16}
    base = base_map[base_char]
    # Replace x/z/? with 0 for parsing — these are unknown in Verilog
    sanitized = re.sub(r"[xXzZ?]", "0", digits)
    try:
        return int(sanitized, base)
    except ValueError:
        return None


def vlog_literal_width(s: str) -> Optional[int]:
    """Return the declared bit width of a Verilog literal, or None."""
    m = _VLOG_LITERAL_RE.match(s.strip())
    if m and m.group("width"):
        return int(m.group("width"))
    return None


# -----------------------------------------------------------------------------
# Expression evaluator (extremely limited)
# -----------------------------------------------------------------------------

_ARITH_EXPR_RE = re.compile(r"^[\d_+\-*/()% \t]+$")
_ARITH_EXPR_MAX_CHARS = 256
_ARITH_EXPR_MAX_NODES = 64
_ARITH_EXPR_MAX_PAREN_DEPTH = 32


def _safe_eval_arith(expr: str) -> Optional[int]:
    """Evaluate a tiny integer-arithmetic expression safely.

    Supported grammar is intentionally smaller than Python: integer constants,
    parentheses, unary +/-, and binary +, -, *, /, %. Division follows the
    previous behavior for positive integer expressions by using integer
    truncation toward zero.
    """
    if len(expr) > _ARITH_EXPR_MAX_CHARS or not _ARITH_EXPR_RE.fullmatch(expr):
        return None
    depth = 0
    for ch in expr:
        if ch == "(":
            depth += 1
            if depth > _ARITH_EXPR_MAX_PAREN_DEPTH:
                return None
        elif ch == ")":
            depth -= 1
            if depth < 0:
                return None
    if depth != 0:
        return None
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError:
        return None

    node_count = 0

    def walk(node: ast.AST) -> int:
        nonlocal node_count
        node_count += 1
        if node_count > _ARITH_EXPR_MAX_NODES:
            raise ValueError("expression too large")

        if isinstance(node, ast.Expression):
            return walk(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, int):
            return node.value
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = walk(node.operand)
            return value if isinstance(node.op, ast.UAdd) else -value
        if isinstance(node, ast.BinOp):
            left = walk(node.left)
            right = walk(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                if right == 0:
                    raise ZeroDivisionError
                return int(left / right)
            if isinstance(node.op, ast.Mod):
                if right == 0:
                    raise ZeroDivisionError
                return left % right
        raise ValueError("unsupported expression")

    try:
        return walk(tree)
    except (ValueError, ZeroDivisionError, OverflowError):
        return None


def eval_expr(expr: str, defines: Dict[str, Tuple[Optional[str], int]]) -> Optional[int]:
    """Try to evaluate a small arithmetic expression that may reference
    `MACRO names. Returns int or None."""
    if expr is None:
        return None
    expr = expr.strip()
    # Substitute `MACRO with the macro's value (recursively, one level).
    def sub(m):
        name = m.group(1)
        if name in defines and defines[name][0] is not None:
            return defines[name][0]
        return m.group(0)
    expanded = re.sub(r"`([A-Za-z_][A-Za-z0-9_]*)", sub, expr)
    # First try as a Verilog literal
    v = parse_int_literal(expanded)
    if v is not None:
        return v
    # Then try as a restricted integer arithmetic expression.
    return _safe_eval_arith(expanded)


# -----------------------------------------------------------------------------
# Preprocessor: walk tokens with `ifdef awareness
# -----------------------------------------------------------------------------

DEFINE_RE = re.compile(
    r"^\s*`define\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)(?:\s+(?P<value>.+?))?\s*$"
)
IFDEF_RE = re.compile(r"^\s*`ifdef\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*$")
IFNDEF_RE = re.compile(r"^\s*`ifndef\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*$")
ELSIF_RE = re.compile(r"^\s*`elsif\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*$")
ELSE_RE = re.compile(r"^\s*`else\s*$")
ENDIF_RE = re.compile(r"^\s*`endif")  # tolerate trailing comments
PACKAGE_RE = re.compile(r"^\s*package\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*;\s*$")
ENDPACKAGE_RE = re.compile(r"^\s*endpackage(?:\s*:\s*[A-Za-z_][A-Za-z0-9_]*)?\s*$")


# -----------------------------------------------------------------------------
# localparam scanner
# -----------------------------------------------------------------------------

# Scalar-array declaration: `localparam integer NAME [SIZE] = '{...};`
# Accepts both `'{...}` (default-init or assignment-pattern) and `{...}`
# (positional concatenation initializer). Body may span multiple lines.
LP_ARRAY_RE = re.compile(
    r"""localparam\s+
    (?:integer|logic\s*\[(?P<bitwidth>[^\]]+)\]|int)?\s*
    (?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*
    \[\s*(?P<size>[^\]]+)\s*\]\s*
    =\s*'?\{(?P<body>.*?)\}\s*;""",
    re.VERBOSE | re.DOTALL,
)

# GPIO_RESET_VALUE: `localparam [WIDTH-1:0] GPIO_RESET_VALUE = LITERAL;`
LP_VECTOR_RE = re.compile(
    r"""localparam\s+
    \[\s*(?P<width>[^\]]+)\s*\]\s*
    (?P<name>GPIO_RESET_VALUE)\s*
    =\s*(?P<literal>[^;]+);""",
    re.VERBOSE,
)


def _split_array_body(body: str) -> List[str]:
    """Split an array initializer body by top-level commas, handling nested {}.
    Returns the list of element strings (whitespace-trimmed)."""
    out = []
    depth = 0
    cur = []
    for ch in body:
        if ch == "{":
            depth += 1
            cur.append(ch)
        elif ch == "}":
            depth -= 1
            cur.append(ch)
        elif ch == "," and depth == 0:
            out.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    last = "".join(cur).strip()
    if last:
        out.append(last)
    return out


def _parse_array_decl(
    match: re.Match, line_num: int, defines: Dict[str, Tuple[Optional[str], int]]
) -> ArrayDecl:
    name = match.group("name")
    size_expr = match.group("size")
    size_value = eval_expr(size_expr.split(":")[0], defines)
    if ":" in size_expr:
        # Form `[N-1:0]`, so length is N
        upper = eval_expr(size_expr.split(":")[0], defines)
        lower = eval_expr(size_expr.split(":")[1], defines)
        if upper is not None and lower is not None:
            size_value = upper - lower + 1
    body = match.group("body")
    elements = _split_array_body(body)
    is_default_init = False
    default_value = None
    if len(elements) == 1 and elements[0].lower().startswith("default"):
        # '{default:X}' form
        is_default_init = True
        default_value = elements[0].split(":", 1)[1].strip()
        elements = []  # individual elements not enumerated
    return ArrayDecl(
        name=name,
        line=line_num,
        size_expr=size_expr,
        size_value=size_value,
        elements=elements,
        is_default_init=is_default_init,
        default_value=default_value,
        raw_text=match.group(0),
    )


def _parse_init_reg(
    match: re.Match, line_num: int, defines: Dict[str, Tuple[Optional[str], int]]
) -> InitRegArray:
    decl = InitRegArray(name="init_reg", line=line_num, size_expr=match.group("size"))
    decl.declared_size_value = eval_expr(match.group("size"), defines)
    elements = _split_array_body(match.group("body"))
    for elem in elements:
        # Expect form `{32'haddr, 32'hdata}`
        e = elem.strip()
        if not (e.startswith("{") and e.endswith("}")):
            decl.entries.append(InitRegEntry(line_num, e, "", None, None))
            continue
        inner = e[1:-1]
        parts = _split_array_body(inner)
        if len(parts) != 2:
            decl.entries.append(InitRegEntry(line_num, e, "", None, None))
            continue
        addr_raw, data_raw = parts[0].strip(), parts[1].strip()
        decl.entries.append(InitRegEntry(
            line=line_num,
            addr_raw=addr_raw,
            data_raw=data_raw,
            addr_value=parse_int_literal(addr_raw),
            data_value=parse_int_literal(data_raw),
        ))
    return decl


# -----------------------------------------------------------------------------
# Main parse
# -----------------------------------------------------------------------------

def parse_file(path: Union[str, Path]) -> ParsedDef:
    p = Path(path)
    text = p.read_text(encoding="utf-8", errors="replace")
    return parse_text(text, str(p))


def parse_text(text: str, path: str = "<text>") -> ParsedDef:
    raw_lines = text.split("\n")
    parsed = ParsedDef(path=path, raw_lines=raw_lines)
    parsed.commented_define_lines = _detect_commented_defines(raw_lines)

    # Strip block comments (preserve line count)
    cleaned_text = _strip_block_comments(text)
    cleaned_lines = cleaned_text.split("\n")

    # Strip line comments per-line
    code_lines = [_strip_line_comments(l) for l in cleaned_lines]

    # First pass: scan for the wrapper guard / package / endpackage / endif.
    for i, line in enumerate(code_lines, start=1):
        if not parsed.wrapper.has_ifndef_guard:
            m = IFNDEF_RE.match(line)
            if m and m.group("name") == "HOLOLINK_def":
                parsed.wrapper.has_ifndef_guard = True
                continue
        if parsed.wrapper.has_ifndef_guard and not parsed.wrapper.has_define_guard:
            m = DEFINE_RE.match(line)
            if m and m.group("name") == "HOLOLINK_def":
                parsed.wrapper.has_define_guard = True
                continue
        m = PACKAGE_RE.match(line)
        if m and m.group("name") == "HOLOLINK_pkg":
            parsed.wrapper.has_package_decl = True
            parsed.wrapper.package_line = i
        if ENDPACKAGE_RE.match(line):
            parsed.wrapper.has_endpackage = True
            parsed.wrapper.endpackage_line = i
        if ENDIF_RE.match(line):
            parsed.wrapper.has_endif = True

    # Second pass: walk with `ifdef awareness, recording defines and arrays.
    # Stack: list of (active: bool, branch_taken: bool). branch_taken tracks
    # whether any branch in the current chain has been active (for `elsif).
    cond_stack: List[Tuple[bool, bool]] = []

    def is_active() -> bool:
        return all(s[0] for s in cond_stack) if cond_stack else True

    # Accumulate multi-line localparam declarations
    accum: List[str] = []
    accum_start: Optional[int] = None

    def flush_accum_if_complete() -> None:
        nonlocal accum, accum_start
        joined = "\n".join(accum)
        # Try to match either an array localparam or a vector localparam.
        m = LP_ARRAY_RE.search(joined)
        if m:
            name = m.group("name")
            if name == "init_reg":
                parsed.init_reg = _parse_init_reg(m, accum_start or 0, parsed.defines)
            else:
                parsed.arrays[name] = _parse_array_decl(m, accum_start or 0, parsed.defines)
            accum = []
            accum_start = None
            return
        m = LP_VECTOR_RE.search(joined)
        if m:
            width_expr = m.group("width")
            literal = m.group("literal").strip()
            # Compute declared width: form `WIDTH-1:0` => width = WIDTH
            width_value = None
            if ":" in width_expr:
                upper = eval_expr(width_expr.split(":")[0], parsed.defines)
                lower = eval_expr(width_expr.split(":")[1], parsed.defines)
                if upper is not None and lower is not None:
                    width_value = upper - lower + 1
            parsed.gpio_reset_value = GpioResetValue(
                line=accum_start or 0,
                width_expr=width_expr,
                width_value=width_value,
                literal=literal,
            )
            accum = []
            accum_start = None

    for i, line in enumerate(code_lines, start=1):
        stripped = line.strip()

        # Conditional directives (must be processed regardless of active state)
        m = IFDEF_RE.match(line)
        if m:
            name = m.group("name")
            if name == "HOLOLINK_def":
                # The wrapper guard — already counted in pass 1, skip
                continue
            outer_active = is_active()
            local_active = name in parsed.defines and outer_active
            cond_stack.append((local_active, local_active))
            continue
        m = IFNDEF_RE.match(line)
        if m:
            name = m.group("name")
            if name == "HOLOLINK_def":
                continue
            outer_active = is_active()
            local_active = (name not in parsed.defines) and outer_active
            cond_stack.append((local_active, local_active))
            continue
        m = ELSIF_RE.match(line)
        if m:
            if not cond_stack:
                parsed.parse_errors.append(ParseError(i, "`elsif without matching `ifdef"))
                continue
            top_active, top_taken = cond_stack[-1]
            outer_active = all(s[0] for s in cond_stack[:-1]) if len(cond_stack) > 1 else True
            new_active = (m.group("name") in parsed.defines) and (not top_taken) and outer_active
            cond_stack[-1] = (new_active, top_taken or new_active)
            continue
        if ELSE_RE.match(line):
            if not cond_stack:
                parsed.parse_errors.append(ParseError(i, "`else without matching `ifdef"))
                continue
            top_active, top_taken = cond_stack[-1]
            outer_active = all(s[0] for s in cond_stack[:-1]) if len(cond_stack) > 1 else True
            new_active = (not top_taken) and outer_active
            cond_stack[-1] = (new_active, top_taken or new_active)
            continue
        if ENDIF_RE.match(line):
            if cond_stack:
                cond_stack.pop()
            continue

        # Past the conditional handling — only continue if we're active
        if not is_active():
            # Track defines that appear in inactive branches so we can
            # distinguish them from missing-entirely.
            m = DEFINE_RE.match(line)
            if m and m.group("name") != "HOLOLINK_def":
                parsed.defines_skipped.append(m.group("name"))
            continue

        # `define directive
        m = DEFINE_RE.match(line)
        if m:
            name = m.group("name")
            if name == "HOLOLINK_def":
                continue
            value = m.group("value")
            if value is not None:
                value = value.strip()
                # Strip trailing line comment artifacts (already removed by
                # _strip_line_comments, but trailing whitespace is ok).
            parsed.defines[name] = (value, i)
            continue

        # Accumulate localparam content
        if "localparam" in line or accum:
            if not accum:
                accum_start = i
            accum.append(line)
            # Flush on terminating semicolon at top level
            if ";" in line:
                flush_accum_if_complete()

    return parsed


# -----------------------------------------------------------------------------
# Convenience
# -----------------------------------------------------------------------------

def define_value(parsed: ParsedDef, name: str) -> Optional[str]:
    """Return the raw value string for a `define, or None if undefined."""
    if name in parsed.defines:
        return parsed.defines[name][0]
    return None


def define_int(parsed: ParsedDef, name: str) -> Optional[int]:
    """Return the integer value of a `define if it parses to one, else None."""
    v = define_value(parsed, name)
    if v is None:
        return None
    return eval_expr(v, parsed.defines)


def is_defined(parsed: ParsedDef, name: str) -> bool:
    return name in parsed.defines


def define_line(parsed: ParsedDef, name: str) -> Optional[int]:
    if name in parsed.defines:
        return parsed.defines[name][1]
    return None
