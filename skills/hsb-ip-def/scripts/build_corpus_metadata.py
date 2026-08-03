#!/usr/bin/env python3
"""Build-time helper: capture an anonymized snapshot of the team's corpus.

Usage:
  build_corpus_metadata.py <path1> [<path2> ...]

Reads each `HOLOLINK_def.svh` path, parses it with lib/parser, strips all
identifying information (UUID values, project names, file paths, line
numbers, comments), and writes:

  assets/metadata/corpus.json        — one anonymized record per input file
  assets/metadata/corpus-stats.json  — aggregate frequency tables

This script is intended to be run once whenever the team's corpus
membership changes. The resulting JSON files are permanent skill data;
runtime validation/explanation does not call this script.

The skill never enumerates project directories by name. Pass paths via
arguments — typically from a list maintained outside the skill (CI
config, team wiki, etc.) — and this script anonymizes them on the way in.
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from statistics import median
from typing import Any, Dict, List, Optional


def _resolve_lib_path() -> None:
    here = Path(__file__).resolve().parent
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))


_resolve_lib_path()
from lib.parser import (  # noqa: E402
    ParsedDef, parse_file, define_int, eval_expr, parse_int_literal,
)
from lib.rules import (  # noqa: E402
    KNOWN_IP_MACROS, KNOWN_IP_ARRAYS, archetype_classify,
)


# Anonymization rule: UUID is the only macro that is consistently board-identity
# in the corpus. Replace its value with a placeholder; keep the fact that it
# was defined so config-shape comparisons still work.
UUID_PLACEHOLDER = "<redacted-uuid>"


# Boolean-define macros — captured by presence/absence rather than by value.
BOOLEAN_DEFINES = {
    "ENUM_EEPROM", "EXT_PTP", "SYNC_CLK_HIF_APB", "SYNC_CLK_HIF_PTP",
    "SIF_RX_DATA_GEN", "DISABLE_COE",
}


def _scalar_value(p: ParsedDef, name: str) -> Optional[Any]:
    """Return the resolved value of a scalar `define, or None."""
    if name not in p.defines:
        return None
    raw, _line = p.defines[name]
    if raw is None:
        return "<defined>"
    if name == "UUID":
        return UUID_PLACEHOLDER
    v = eval_expr(raw, p.defines)
    return v if v is not None else raw


def _normalize_array(p: ParsedDef, name: str) -> Optional[List[Optional[int]]]:
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


def _build_record(idx: int, parsed: ParsedDef) -> Dict[str, Any]:
    """Anonymize a parsed config into a metadata record."""
    scalars: Dict[str, Any] = {}
    for name in sorted(KNOWN_IP_MACROS):
        if name == "HOLOLINK_def":
            continue
        val = _scalar_value(parsed, name)
        if val is not None:
            scalars[name] = val

    arrays: Dict[str, Optional[List[Optional[int]]]] = {}
    for name in sorted(KNOWN_IP_ARRAYS):
        if name == "init_reg":
            continue
        arr = _normalize_array(parsed, name)
        if arr is not None:
            arrays[name] = arr

    init_reg_length = len(parsed.init_reg.entries) if parsed.init_reg else 0
    archetype = archetype_classify(parsed)

    return {
        "config_id": f"config-{idx:02d}",
        "scalars": scalars,
        "arrays": arrays,
        "init_reg_length": init_reg_length,
        "inferred_archetype": archetype,
    }


def _aggregate(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Build per-macro frequency tables and archetype distribution."""
    scalar_macro_names: set = set()
    for r in records:
        scalar_macro_names.update(r["scalars"].keys())

    scalar_stats: Dict[str, Any] = {}
    for name in sorted(scalar_macro_names):
        values = [r["scalars"].get(name) for r in records if name in r["scalars"]]
        defined_count = len(values)
        # Numeric values get min/max/median
        numeric_values = [v for v in values if isinstance(v, int)]
        value_counts = Counter(values)
        entry: Dict[str, Any] = {
            "defined_in_count": defined_count,
            "value_counts": {str(k): v for k, v in sorted(
                value_counts.items(), key=lambda kv: (-kv[1], str(kv[0])))},
        }
        if numeric_values:
            entry["min"] = min(numeric_values)
            entry["max"] = max(numeric_values)
            entry["median"] = int(median(numeric_values))
        scalar_stats[name] = entry

    archetype_dist = Counter(r["inferred_archetype"] for r in records)
    archetype_dist_dict = {str(k): v for k, v in sorted(
        archetype_dist.items(), key=lambda kv: (-kv[1], str(kv[0])))}

    init_reg_lengths = [r["init_reg_length"] for r in records]
    init_reg_stats = {
        "value_counts": {str(k): v for k, v in sorted(
            Counter(init_reg_lengths).items(), key=lambda kv: (-kv[1], kv[0]))},
        "min": min(init_reg_lengths) if init_reg_lengths else 0,
        "max": max(init_reg_lengths) if init_reg_lengths else 0,
        "median": int(median(init_reg_lengths)) if init_reg_lengths else 0,
    }

    return {
        "config_count": len(records),
        "scalar_macros": scalar_stats,
        "archetype_distribution": archetype_dist_dict,
        "init_reg_length": init_reg_stats,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="build_corpus_metadata.py",
        description="Anonymize and bundle a HOLOLINK_def.svh corpus into assets/metadata/.",
    )
    ap.add_argument("paths", nargs="+", type=Path,
                    help="Workspace HOLOLINK_def.svh paths to capture.")
    ap.add_argument("--output-dir", type=Path,
                    default=Path(__file__).resolve().parent.parent / "assets" / "metadata",
                    help="Output directory (default: assets/metadata/ at the skill root).")
    ap.add_argument("--ip-rev", default="16'h2604",
                    help="IP revision the corpus targets (informational).")
    args = ap.parse_args(argv)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    records: List[Dict[str, Any]] = []
    for idx, path in enumerate(args.paths, start=1):
        if not path.exists():
            print(f"ERROR: file not found: {path}", file=sys.stderr)
            return 1
        try:
            parsed = parse_file(path)
        except Exception as e:  # noqa: BLE001
            print(f"ERROR: parse failure on {path}: {e}", file=sys.stderr)
            return 1
        records.append(_build_record(idx, parsed))

    corpus_doc = {
        "schema_version": 1,
        "captured_at_ip_rev": args.ip_rev,
        "config_count": len(records),
        "configs": records,
    }
    stats_doc = {
        "schema_version": 1,
        "captured_at_ip_rev": args.ip_rev,
        **_aggregate(records),
    }

    corpus_path = args.output_dir / "corpus.json"
    stats_path = args.output_dir / "corpus-stats.json"
    corpus_path.write_text(json.dumps(corpus_doc, indent=2, sort_keys=False) + "\n")
    stats_path.write_text(json.dumps(stats_doc, indent=2, sort_keys=False) + "\n")

    print(f"Wrote {corpus_path} ({len(records)} configs)", file=sys.stderr)
    print(f"Wrote {stats_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
