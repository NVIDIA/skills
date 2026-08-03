#!/usr/bin/env python3
"""Semantic diff between two HOLOLINK_def.svh files.

Usage:
  compare_defs.py <a.svh> <b.svh> [--json|--text]

Compares the parsed view of each file (defines, arrays, init_reg, wrapper).
Whitespace, comment churn, and out-of-scope macros are ignored. Equivalent
syntactic forms (e.g. `'{default:1}` vs `{1, 1, 1, 1}`) normalize to the
same value list before comparison.

Exit codes:
  0  no semantic differences
  1  differences found
  2  parse failure on either input
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


def _resolve_lib_path() -> None:
    here = Path(__file__).resolve().parent
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))


_resolve_lib_path()
from lib.parser import (  # noqa: E402
    ParsedDef, parse_file, define_int, define_value, eval_expr, parse_int_literal,
)
from lib.rules import KNOWN_IP_MACROS, KNOWN_IP_ARRAYS, get_array_length  # noqa: E402


def _normalize_array(p: ParsedDef, name: str) -> Optional[List[Optional[int]]]:
    """Resolve an array to a flat list of integer values, expanding default-init."""
    if name not in p.arrays:
        return None
    arr = p.arrays[name]
    if arr.is_default_init:
        size = arr.size_value or 0
        v = parse_int_literal(arr.default_value or "")
        if v is None:
            v = eval_expr(arr.default_value or "", p.defines)
        return [v] * size
    out: List[Optional[int]] = []
    for e in arr.elements:
        v = parse_int_literal(e)
        if v is None:
            v = eval_expr(e, p.defines)
        out.append(v)
    return out


def _scalar_summary(p: ParsedDef) -> Dict[str, Any]:
    """Return a dict of scalar `defines limited to KNOWN_IP_MACROS."""
    out: Dict[str, Any] = {}
    for name, (val, _line) in p.defines.items():
        if name not in KNOWN_IP_MACROS:
            continue
        # Try to resolve to int; otherwise keep raw string
        if val is None:
            out[name] = "<defined>"
            continue
        v = eval_expr(val, p.defines)
        out[name] = v if v is not None else val
    # Mark macros that appear in one file but not the other as <undefined>
    return out


def compare(a_path: Path, b_path: Path) -> Dict[str, Any]:
    a = parse_file(a_path)
    b = parse_file(b_path)
    diff: Dict[str, Any] = {
        "scalar_defines": [],
        "arrays": [],
        "init_reg": None,
        "wrapper": None,
    }

    a_scalar = _scalar_summary(a)
    b_scalar = _scalar_summary(b)
    all_keys = sorted(set(a_scalar) | set(b_scalar))
    for k in all_keys:
        av = a_scalar.get(k, "<undefined>")
        bv = b_scalar.get(k, "<undefined>")
        if av != bv:
            diff["scalar_defines"].append({"macro": k, "a": av, "b": bv})

    for name in sorted(KNOWN_IP_ARRAYS):
        if name == "init_reg":
            continue
        a_norm = _normalize_array(a, name)
        b_norm = _normalize_array(b, name)
        if a_norm != b_norm:
            diff["arrays"].append({"name": name, "a": a_norm, "b": b_norm})

    a_init = [(e.addr_value, e.data_value) for e in a.init_reg.entries] if a.init_reg else None
    b_init = [(e.addr_value, e.data_value) for e in b.init_reg.entries] if b.init_reg else None
    if a_init != b_init:
        diff["init_reg"] = {"a": a_init, "b": b_init}

    if (a.wrapper.has_ifndef_guard != b.wrapper.has_ifndef_guard
            or a.wrapper.has_package_decl != b.wrapper.has_package_decl):
        diff["wrapper"] = {
            "a": {
                "has_ifndef_guard": a.wrapper.has_ifndef_guard,
                "has_package_decl": a.wrapper.has_package_decl,
            },
            "b": {
                "has_ifndef_guard": b.wrapper.has_ifndef_guard,
                "has_package_decl": b.wrapper.has_package_decl,
            },
        }

    return diff


def has_diffs(diff: Dict[str, Any]) -> bool:
    return bool(diff["scalar_defines"]) or bool(diff["arrays"]) or diff["init_reg"] is not None or diff["wrapper"] is not None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="compare_defs.py",
        description="Semantic diff between two HOLOLINK_def.svh files.",
    )
    ap.add_argument("a", type=Path, help="First file")
    ap.add_argument("b", type=Path, help="Second file")
    fmt = ap.add_mutually_exclusive_group()
    fmt.add_argument("--json", action="store_true", default=True)
    fmt.add_argument("--text", action="store_true", default=False)
    args = ap.parse_args(argv)

    for p in (args.a, args.b):
        if not p.exists():
            print(f"ERROR: file not found: {p}", file=sys.stderr)
            return 2

    try:
        diff = compare(args.a, args.b)
    except Exception as e:  # noqa: BLE001
        print(f"ERROR: parse failure: {e}", file=sys.stderr)
        return 2

    if args.text:
        _emit_text(args.a, args.b, diff)
    else:
        json.dump(diff, sys.stdout, indent=2, default=str)
        print()

    return 1 if has_diffs(diff) else 0


def _emit_text(a_path: Path, b_path: Path, diff: Dict[str, Any]) -> None:
    print(f"=== a: {a_path}")
    print(f"=== b: {b_path}")
    if diff["wrapper"]:
        print(f"\n--- WRAPPER DIFFERENCES ---")
        print(f"  a: {diff['wrapper']['a']}")
        print(f"  b: {diff['wrapper']['b']}")
    if diff["scalar_defines"]:
        print(f"\n--- SCALAR `define DIFFERENCES ({len(diff['scalar_defines'])}) ---")
        for d in diff["scalar_defines"]:
            print(f"  {d['macro']:<22} a={d['a']!r:<22} b={d['b']!r}")
    if diff["arrays"]:
        print(f"\n--- ARRAY DIFFERENCES ({len(diff['arrays'])}) ---")
        for d in diff["arrays"]:
            print(f"  {d['name']:<22} a={d['a']}")
            print(f"  {'':22}   b={d['b']}")
    if diff["init_reg"]:
        print(f"\n--- init_reg[] DIFFERS ---")
        print(f"  a: {diff['init_reg']['a']}")
        print(f"  b: {diff['init_reg']['b']}")
    if not has_diffs(diff):
        print("\nNo semantic differences.")


if __name__ == "__main__":
    sys.exit(main())
