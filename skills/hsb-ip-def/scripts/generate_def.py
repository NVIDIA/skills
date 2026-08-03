#!/usr/bin/env python3
"""Generate a HOLOLINK_def.svh file from a profile (and optional archetype defaults).

Usage:
  generate_def.py --profile <path> [-o <output>]
  generate_def.py --archetype <slug> [--profile <path>] [-o <output>]

The --profile argument accepts YAML or JSON (auto-detected by extension).
If no `uuid:` is supplied, the generator creates a random 128-bit UUID and
prints a NOTE to stderr — pin a specific value by adding `uuid:` to your
profile. If neither --archetype nor a profile `archetype:` key is given,
generic per-macro defaults are used and the user's profile fully specifies
the design.

The generator validates the emitted file with validate_def.py before writing.
If validation fails, the file is NOT written and the validator's errors are
printed to stderr.

Profile schema (top-level keys, all optional):

  archetype: "<slug>"          # optional; one of the 5 example archetypes
  uuid: "128'h…"               # board-specific 128-bit hex literal; auto-generated if missing
  hif_clk_freq: <int Hz>
  apb_clk_freq: <int Hz>
  ptp_clk_freq: <int Hz>
  enum_eeprom: <bool>
  eeprom_reg_addr_bits: 8 | 16
  datapath_width: <int>
  datauser_width: 1 | 2
  sensor_rx_count: <int|null>
  sif_rx_data_gen: <bool>
  sif_rx_widths: [<int>, …]    # length must match sensor_rx_count
  sif_rx_packetizer_en: [0|1, …]
  sif_rx_vp_count: [<int>, …]
  sif_rx_sort_resolution: [<int>, …]
  sif_rx_vp_size: [<int>, …]
  sif_rx_num_cycles: [<int>, …]
  sensor_tx_count: <int|null>
  sif_tx_widths: [<int>, …]
  sif_tx_buf_size: [<int>, …]  # depth in SIF_TX_WIDTH elements
  host_width: <int>
  host_if_inst: <int>
  host_mtu: <int>
  spi_inst: <int|null>
  i2c_inst: <int|null>
  uart_inst: <int|null>
  gpio_inst: <int>
  gpio_reset_value: "<SystemVerilog literal>"
  reg_inst: <int>
  init_reg: [] | [{addr: "32'h…", data: "32'h…"}, …]
  ext_ptp: <bool>
  sync_clk_hif_apb: <bool>
  sync_clk_hif_ptp: <bool>
  peri_ram_depth: <int|null>
  disable_coe: <bool>

Exit codes:
  0  generated file passes validation
  1  generated file fails validation
  2  profile / argument error
"""

import argparse
import json
import secrets
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

UUID_BITS = 128
UUID_HEX_LENGTH = 32
UUID_HEX_GROUP_SIZE = 4
EXIT_SUCCESS = 0
EXIT_VALIDATION_ERROR = 1
EXIT_USAGE_ERROR = 2


def _resolve_lib_path() -> None:
    here = Path(__file__).resolve().parent
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))


_resolve_lib_path()
from lib.emitter import ARCHETYPE_DEFAULTS, build_profile, emit_svh  # noqa: E402


def _load_profile(path: Path) -> Dict[str, Any]:
    text = path.read_text()
    if path.suffix.lower() == ".json":
        return json.loads(text)
    try:
        import yaml  # type: ignore
    except ImportError:
        sys.stderr.write("ERROR: PyYAML is required for YAML profiles. "
                         "Install with `pip install pyyaml` or use a .json profile.\n")
        sys.exit(EXIT_USAGE_ERROR)
    return yaml.safe_load(text) or {}


def _normalize_init_reg(init_reg: Any) -> Optional[List[Tuple[str, str]]]:
    """Accept init_reg as either:
      - [] to suppress system init
      - list of [addr, data] pairs
      - list of {"addr": "...", "data": "..."} dicts
    """
    if init_reg is None:
        return None
    out: List[Tuple[str, str]] = []
    for entry in init_reg:
        if isinstance(entry, dict):
            out.append((str(entry["addr"]), str(entry["data"])))
        elif isinstance(entry, (list, tuple)):
            try:
                addr, data = entry
            except ValueError as exc:
                raise ValueError(f"init_reg entry has unexpected form: {entry!r}") from exc
            out.append((str(addr), str(data)))
        else:
            raise ValueError(f"init_reg entry has unexpected form: {entry!r}")
    return out


def _generate_random_uuid() -> str:
    """Generate a random 128-bit UUID literal."""
    val = secrets.randbits(UUID_BITS)
    hex_str = f"{val:0{UUID_HEX_LENGTH}x}"
    grouped = "_".join(
        hex_str[i:i + UUID_HEX_GROUP_SIZE].upper()
        for i in range(0, UUID_HEX_LENGTH, UUID_HEX_GROUP_SIZE)
    )
    return f"128'h{grouped}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="generate_def.py",
        description="Generate a HOLOLINK_def.svh from an archetype + overrides.",
    )
    ap.add_argument("-a", "--archetype",
                    choices=sorted(ARCHETYPE_DEFAULTS.keys()),
                    help="Optional. If omitted and the profile has no `archetype:` key, "
                         "generic per-macro defaults are used.")
    ap.add_argument("-p", "--profile", type=Path,
                    help="YAML or JSON profile file with overrides.")
    ap.add_argument("-o", "--output", type=Path, default=None,
                    help="Output path. Default: print to stdout.")
    ap.add_argument("--allow-random-uuid", action="store_true",
                    help="Deprecated and ignored. Random UUID is generated by "
                         "default when none is supplied.")
    args = ap.parse_args(argv)

    overrides: Dict[str, Any] = {}
    archetype = args.archetype
    if args.profile:
        if not args.profile.exists():
            print(f"ERROR: profile file not found: {args.profile}", file=sys.stderr)
            return EXIT_USAGE_ERROR
        try:
            overrides = _load_profile(args.profile)
        except Exception as e:  # noqa: BLE001
            print(f"ERROR: failed to load profile: {e}", file=sys.stderr)
            return EXIT_USAGE_ERROR
        if not isinstance(overrides, dict):
            print(f"ERROR: profile must be a mapping, got {type(overrides).__name__}", file=sys.stderr)
            return EXIT_USAGE_ERROR
        # Profile may name its own archetype
        archetype = archetype or overrides.pop("archetype", None)
    # archetype is now optional; emitter falls back to GENERIC_DEFAULTS when None.

    # Deprecation NOTE for the legacy flag (kept as no-op for back-compat).
    if args.allow_random_uuid:
        print("NOTE: --allow-random-uuid is deprecated and ignored. "
              "Random UUID is generated by default when none is supplied.",
              file=sys.stderr)

    # Normalize init_reg if present
    if "init_reg" in overrides:
        try:
            overrides["init_reg"] = _normalize_init_reg(overrides["init_reg"])
        except Exception as e:  # noqa: BLE001
            print(f"ERROR: invalid init_reg: {e}", file=sys.stderr)
            return EXIT_USAGE_ERROR

    # UUID handling: random by default when not supplied. Pin a specific
    # value by adding `uuid:` to the profile.
    if "uuid" not in overrides or not overrides["uuid"]:
        overrides["uuid"] = _generate_random_uuid()
        print(f"NOTE: no UUID supplied; generated random UUID {overrides['uuid']}",
              file=sys.stderr)

    profile, build_notes = build_profile(archetype, overrides)
    for note in build_notes:
        print(f"NOTE: {note}", file=sys.stderr)
    svh_text = emit_svh(profile)

    # Validate via subprocess
    validator = Path(__file__).resolve().parent / "validate_def.py"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".svh", delete=False) as tf:
        tf.write(svh_text)
        tmp_path = Path(tf.name)
    try:
        proc = subprocess.run(
            [sys.executable, str(validator), str(tmp_path), "--json"],
            capture_output=True, text=True, check=False,
        )
        try:
            result = json.loads(proc.stdout)
        except json.JSONDecodeError:
            print(f"ERROR: validator did not produce valid JSON. stdout:\n{proc.stdout}", file=sys.stderr)
            return EXIT_VALIDATION_ERROR
        errors = result.get("errors", [])
        warnings = result.get("warnings", [])
        if errors:
            print("ERROR: generated file failed validation. Refusing to write output.", file=sys.stderr)
            for e in errors:
                print(f"  [{e['rule']}] line {e['line']} {e.get('macro') or '—'}: {e['msg']}", file=sys.stderr)
            return EXIT_VALIDATION_ERROR
        if warnings:
            print(f"NOTE: {len(warnings)} warning(s) from validator (not blocking):", file=sys.stderr)
            for w in warnings:
                print(f"  [{w['rule']}] line {w['line']} {w.get('macro') or '—'}: {w['msg']}", file=sys.stderr)
    finally:
        tmp_path.unlink(missing_ok=True)

    # Write output
    if args.output:
        args.output.write_text(svh_text)
        print(f"Wrote {args.output}", file=sys.stderr)
    else:
        sys.stdout.write(svh_text)
    return EXIT_SUCCESS


if __name__ == "__main__":
    sys.exit(main())
