#!/usr/bin/env python3
"""Validate a HOLOLINK_def.svh file.

Usage:
  validate_def.py <path/to/HOLOLINK_def.svh> [--json|--text] [--ip-source <path>]

Output (default --json):
  {"errors": [...], "warnings": [...], "info": [...],
   "inferred_archetype": "<slug|null>",
   "ip_version_target": "16'h2604"}

Exit codes:
  0  no errors (warnings/info OK)
  1  errors found
  2  parse failure or filesystem error

The validator's output never names out-of-scope macros — any macro outside
the KNOWN_IP_MACROS allowlist passes silently with no annotation. See
references/validation-rules.md for the full rule catalog.
"""

import argparse
import json
import sys
from pathlib import Path


def _resolve_lib_path() -> None:
    """Make scripts/lib importable regardless of where the script is invoked."""
    here = Path(__file__).resolve().parent
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))


_resolve_lib_path()
from lib.parser import parse_file  # noqa: E402
from lib.rules import run_all_rules, archetype_classify  # noqa: E402


IP_VERSION_TARGET = "16'h2604"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="validate_def.py",
        description="Validate a HOLOLINK_def.svh file against the HSB IP rules.",
    )
    ap.add_argument("path", help="Path to HOLOLINK_def.svh")
    fmt = ap.add_mutually_exclusive_group()
    fmt.add_argument("--json", action="store_true", default=True,
                     help="Emit JSON output (default).")
    fmt.add_argument("--text", action="store_true", default=False,
                     help="Emit human-readable text output.")
    ap.add_argument("--ip-source", default=None,
                    help="Optional path to an HSB IP root directory (containing top/, lib_axis/, "
                         "etc.). When provided, the validator can cite live RTL line numbers for "
                         "richer diagnostics. Both fpga/nv_hsb_ip/ and hw/.../vrtl/hololink/ are "
                         "valid roots. Default: use only the line numbers baked into "
                         "references/macro-reference.md.")
    args = ap.parse_args(argv)

    path = Path(args.path)
    if not path.exists():
        msg = f"File not found: {path}"
        if args.text:
            print(f"ERROR: {msg}", file=sys.stderr)
        else:
            json.dump({"errors": [{"rule": "HD-FATAL", "severity": "error",
                                   "line": None, "macro": None, "msg": msg}],
                       "warnings": [], "info": [],
                       "inferred_archetype": None,
                       "ip_version_target": IP_VERSION_TARGET},
                      sys.stdout, indent=2)
            print()
        return 2

    if args.ip_source:
        ip_root = Path(args.ip_source)
        if not (ip_root / "top" / "HOLOLINK_top.sv").exists():
            print(f"WARNING: --ip-source {ip_root} does not look like an HSB IP root "
                  f"(missing top/HOLOLINK_top.sv). Diagnostics will use baked-in line numbers.",
                  file=sys.stderr)

    try:
        parsed = parse_file(path)
    except Exception as e:  # noqa: BLE001
        msg = f"Parse failure: {e}"
        if args.text:
            print(f"ERROR: {msg}", file=sys.stderr)
        else:
            json.dump({"errors": [{"rule": "HD-FATAL", "severity": "error",
                                   "line": None, "macro": None, "msg": msg}],
                       "warnings": [], "info": [],
                       "inferred_archetype": None,
                       "ip_version_target": IP_VERSION_TARGET},
                      sys.stdout, indent=2)
            print()
        return 2

    findings = run_all_rules(parsed)
    archetype = archetype_classify(parsed)

    errors = [f.to_dict() for f in findings if f.severity == "error"]
    warnings = [f.to_dict() for f in findings if f.severity == "warning"]
    info = [f.to_dict() for f in findings if f.severity == "info"]

    if archetype:
        info.append({"rule": "HD-I004", "severity": "info", "line": None,
                     "macro": None,
                     "msg": f"Configuration matches the {archetype} archetype. See archetypes.md "
                            "for guidance on common adjustments."})
    else:
        info.append({"rule": "HD-I003", "severity": "info", "line": None,
                     "macro": None,
                     "msg": "Configuration does not cleanly match any archetype in archetypes.md. "
                            "The validator confirms it is structurally valid; archetype-specific "
                            "guidance is not available."})

    result = {
        "errors": errors,
        "warnings": warnings,
        "info": info,
        "inferred_archetype": archetype,
        "ip_version_target": IP_VERSION_TARGET,
    }

    if args.text:
        _emit_text(path, result)
    else:
        json.dump(result, sys.stdout, indent=2)
        print()

    return 1 if errors else 0


def _emit_text(path: Path, result: dict) -> None:
    print(f"=== {path} ===")
    print(f"Inferred archetype: {result['inferred_archetype'] or '<none>'}")
    print(f"IP version target:  {result['ip_version_target']}")
    print()
    for severity, key in [("ERROR", "errors"), ("WARNING", "warnings"), ("INFO", "info")]:
        items = result[key]
        if not items:
            continue
        print(f"--- {severity}S ({len(items)}) ---")
        for f in items:
            line = f"line {f['line']}" if f.get("line") else "—"
            macro = f["macro"] or "—"
            print(f"  [{f['rule']}] {line} {macro}: {f['msg']}")
        print()


if __name__ == "__main__":
    sys.exit(main())
