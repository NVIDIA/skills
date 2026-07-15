#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""modify_pinmux.py - generic per-pin SFIO/direction/state configurator.

## XLSM-first flow (new default)

When invoked via /jetson-modify-pinmux the skill now takes a user-provided XLSM
as the single source of truth:

  1. ``probe`` - parse the XLSM with generate_dtsi.py, build pinmap KB for
     lookup, and persist XLSM path + carrier into session. No DTSIs written.
  2. ``lookup`` - resolve a pin query against the XLSM-derived pinmap.
  3. ``set-pin`` / ``apply`` - record per-pin edits in session.
  4. ``generate`` - bake session pin_edits into the XLSM data, then emit
     all three DTSIs (pinmux / gpio / padvoltage) to --out-dir.

``commit --confirm`` patches the cloned pinmux + gpio DTSIs under $L4T_DIR.
"""
from __future__ import annotations

import argparse
import difflib
import importlib.util
import json
import pathlib
import re
import sys
from dataclasses import dataclass

from _pinmux_common import (
    DIRECTION_VALUES,
    DRIVE_TYPE_VALUES,
    ENABLE_DISABLE_VALUES,
    INITIAL_STATE_VALUES,
    PIN_NAME_RE,
    PULL_VALUES,
    QUERY_RE,
    load_pinmap,
    load_session,
    lookup_summary,
    parse_gpio_sfio,
    print_row_details,
    resolve_pinmap_path,
    row_haystack,
    save_session,
    sfio_options,
)
from _pinmux_dt import (
    PinPatchOpts,
    PinPatchSpec,
    patch_gpio_block,
    patch_pinmux_block,
    resolve_dtsi,
)

_HAS_OPENPYXL = importlib.util.find_spec("openpyxl") is not None

_SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()

_OPENPYXL_HINT = ("error: openpyxl is required. Install with:\n"
                  "  pip install --user 'openpyxl>=3.1'")

_CARRIER_RE = re.compile(r"^[a-z0-9_\-]{2,64}$")


# ===== lookup ===========================================================

def cmd_lookup(args: argparse.Namespace) -> int:
    """Resolve a CVM ball / signal / pin query against the pinmap KB."""
    if not QUERY_RE.match(args.query or ""):
        print(f"error: --query must match {QUERY_RE.pattern}", file=sys.stderr)
        return 2

    kb = args.kb_dir.expanduser().resolve()
    session = load_session(kb)
    carrier = args.carrier or session.get("carrier_short") or session.get("carrier")
    pinmap_path = resolve_pinmap_path(kb, session, carrier)
    rows = load_pinmap(pinmap_path)

    q = args.query.lower()
    matches = [row for row in rows if any(q in h for h in row_haystack(row))]

    if not matches:
        return _print_lookup_suggestions(q, rows, args.query, pinmap_path)

    if args.json:
        print(json.dumps([lookup_summary(row) for row in matches], indent=2))
        return 0

    if len(matches) > 1:
        print(f"{len(matches)} matches for {args.query!r} - disambiguate "
              f"by DT pin name on the next call:")
        for row in matches:
            print(f"  - pin={row.get('pin')}  ball={row.get('ball')}  "
                  f"verilog={row.get('verilog_name')}  "
                  f"signal={row.get('signal_name')}  "
                  f"customer_usage={row.get('customer_usage')}")
        return 0

    print_row_details(matches[0])
    return 0


def _print_lookup_suggestions(q: str, rows: list[dict], query: str,
                              pinmap_path: pathlib.Path) -> int:
    """Print closest-match suggestions when lookup found nothing."""
    all_tokens: list[str] = []
    token_to_row: dict[str, dict] = {}
    for row in rows:
        for h in row_haystack(row):
            all_tokens.append(h)
            token_to_row.setdefault(h, row)
    suggestions = difflib.get_close_matches(q, all_tokens, n=5, cutoff=0.4)
    print(f"no pinmap row matched {query!r} in {pinmap_path.name}")
    if suggestions:
        print("did you mean:")
        seen: set[str] = set()
        for s in suggestions:
            row = token_to_row[s]
            key = row.get("pin", s)
            if key in seen:
                continue
            seen.add(key)
            print(f"  - {row.get('pin')}  ({s})")
    return 1


# ===== set-pin ==========================================================

def _validate_set_pin_args(args: argparse.Namespace) -> int:
    """Validate set-pin CLI args. Returns 0 on success, error code otherwise."""
    if not PIN_NAME_RE.match(args.pin or ""):
        print(f"error: --pin must match {PIN_NAME_RE.pattern}", file=sys.stderr)
        return 2
    if args.direction not in DIRECTION_VALUES:
        print(f"error: --direction must be one of {sorted(DIRECTION_VALUES)}",
              file=sys.stderr)
        return 2
    if args.initial_state not in INITIAL_STATE_VALUES:
        print(f"error: --initial-state must be one of "
              f"{sorted(INITIAL_STATE_VALUES)}", file=sys.stderr)
        return 2
    if args.direction != "output" and args.initial_state in ("low", "high"):
        print(f"error: initial_state={args.initial_state!r} requires "
              f"direction=output (got {args.direction!r})", file=sys.stderr)
        return 2
    checks = [
        ("--pull", args.pull, PULL_VALUES),
        ("--drive-type", args.drive_type, DRIVE_TYPE_VALUES),
        ("--open-drain", args.open_drain, ENABLE_DISABLE_VALUES),
        ("--loopback", args.loopback, ENABLE_DISABLE_VALUES),
    ]
    for flag, value, allowed in checks:
        if value and value not in allowed:
            print(f"error: {flag} must be one of {sorted(allowed)}",
                  file=sys.stderr)
            return 2
    return 0


def _build_edit_record(args: argparse.Namespace, row: dict) -> dict:
    """Build a session pin_edit record from validated args + matched row."""
    record = {
        "pin": args.pin,
        "ball": row.get("ball"),
        "verilog_name": row.get("verilog_name"),
        "sfio": args.sfio,
        "direction": args.direction,
        "initial_state": args.initial_state,
        "evidence": "user",
    }
    for key, value in (("pull", args.pull), ("drive_type", args.drive_type),
                       ("open_drain", args.open_drain),
                       ("loopback", args.loopback)):
        if value:
            record[key] = value
    return record


def cmd_set_pin(args: argparse.Namespace) -> int:
    """Record a single pin's user-chosen sfio/direction/state in session."""
    rc = _validate_set_pin_args(args)
    if rc != 0:
        return rc

    kb = args.kb_dir.expanduser().resolve()
    session = load_session(kb)
    if not session:
        print(f"error: no session.json under {kb}", file=sys.stderr)
        return 2
    carrier = session.get("carrier_short") or session.get("carrier")
    pinmap_path = resolve_pinmap_path(kb, session, carrier)
    rows = load_pinmap(pinmap_path)

    row = next((r for r in rows if r.get("pin") == args.pin), None)
    if not row:
        print(f"error: pin {args.pin!r} not found in {pinmap_path.name}",
              file=sys.stderr)
        return 1

    options = sfio_options(row)
    if args.sfio not in options:
        print(f"error: sfio={args.sfio!r} not in supported list "
              f"{options} for pin {args.pin}", file=sys.stderr)
        return 2
    if args.sfio == "gpio" and parse_gpio_sfio(row) is None:
        print(f"error: pin {args.pin!r} cannot be configured as gpio "
              f"(no gpio= entry in sfio)", file=sys.stderr)
        return 2

    record = _build_edit_record(args, row)
    pinmux = session.setdefault("pinmux", {})
    edits = list(pinmux.get("pin_edits") or [])
    pos = next((i for i, e in enumerate(edits) if e.get("pin") == args.pin), None)
    if pos is None:
        edits.append(record)
        verb = "added"
    else:
        edits[pos] = record
        verb = "updated"
    pinmux["pin_edits"] = edits
    save_session(kb, session)
    print(f"{verb} pin_edit: {json.dumps(record)}")
    return 0


# ===== apply ============================================================

def cmd_apply(args: argparse.Namespace) -> int:
    """Stage unified-diff patches under <KB>/staged/. Dry-run only."""
    kb = args.kb_dir.expanduser().resolve()
    session = load_session(kb)
    if not session:
        print(f"error: no session.json under {kb}", file=sys.stderr)
        return 2
    edits = ((session.get("pinmux") or {}).get("pin_edits") or [])
    if not edits:
        print("session.pinmux.pin_edits is empty; nothing to apply")
        return 0
    staged = kb / "staged"
    staged.mkdir(parents=True, exist_ok=True)
    plan = staged / "pinmux_plan.json"
    plan.write_text(json.dumps(edits, indent=2), encoding="utf-8")
    print(f"wrote {plan}  ({len(edits)} pin edits staged)")
    return 0


# ===== commit ===========================================================

def _augment_with_gpio(session: dict, kb: pathlib.Path,
                       edits: list[dict]) -> None:
    """Decorate edits with `_gpio` (bank/idx/kind) when sfio == 'gpio'."""
    pinmap = load_pinmap(resolve_pinmap_path(
        kb, session, session.get("carrier_short") or session.get("carrier")
    ))
    by_pin = {r.get("pin"): r for r in pinmap}
    for edit in edits:
        if edit.get("sfio") != "gpio":
            continue
        row = by_pin.get(edit.get("pin"))
        if not row:
            continue
        gp = parse_gpio_sfio(row)
        if gp:
            edit["_gpio"] = gp


@dataclass
class _FileEdit:
    """A single DTSI file's pre/post text and resolved path."""
    path: pathlib.Path | None
    before: str
    after: str | None


@dataclass
class _CommitState:
    """In-flight state during a commit run."""
    pinmux: _FileEdit
    gpio: _FileEdit
    patched: list[str]
    skipped: list[str]
    gpio_warnings: list[str]


def _apply_pinmux_edits(text: str, edits: list[dict]
                        ) -> tuple[str, list[str], list[str]]:
    """Apply all session pin_edits to a pinmux DTSI text in memory."""
    new_text = text
    patched: list[str] = []
    skipped: list[str] = []
    for edit in edits:
        spec = PinPatchSpec(
            pin=edit["pin"], sfio=edit["sfio"],
            direction=edit["direction"],
            initial_state=edit["initial_state"],
        )
        opts = PinPatchOpts(
            pull=edit.get("pull"),
            drive_type=edit.get("drive_type"),
            open_drain=edit.get("open_drain"),
            loopback=edit.get("loopback"),
        )
        new_text, found = patch_pinmux_block(new_text, spec, opts)
        (patched if found else skipped).append(edit["pin"])
    return new_text, patched, skipped


def _emit_diff(label: str, fe: _FileEdit) -> None:
    """Print a unified diff for one _FileEdit when changed."""
    if fe.after is None or fe.after == fe.before or fe.path is None:
        return
    print(f"\n--- {label} DTSI diff ---")
    for line in difflib.unified_diff(
            fe.before.splitlines(keepends=True),
            fe.after.splitlines(keepends=True),
            fromfile=str(fe.path),
            tofile=str(fe.path) + " (after)"):
        sys.stdout.write(line)


def _print_commit_plan(state: _CommitState) -> None:
    """Print the dry-run summary + unified diffs for the commit."""
    print("=== modify-pinmux commit plan ===")
    print(f"pinmux DTSI : {state.pinmux.path}")
    if state.patched:
        print(f"  will patch pin blocks: {', '.join(state.patched)}")
    if state.skipped:
        print(f"  will SKIP (no block found in this DTSI variant): "
              f"{', '.join(state.skipped)}")
    if state.gpio.path is not None:
        print(f"gpio   DTSI : {state.gpio.path}")
        for w in state.gpio_warnings:
            print(f"  warn: {w}")
    _emit_diff("pinmux", state.pinmux)
    _emit_diff("gpio", state.gpio)


def _write_commit_files(state: _CommitState, kb: pathlib.Path,
                        l4t_dir: pathlib.Path) -> int:
    """Persist patched DTSI text to disk. Returns exit code."""
    try:
        for fe in (state.pinmux, state.gpio):
            if fe.after is None or fe.after == fe.before or fe.path is None:
                continue
            fe.path.write_text(fe.after, encoding="utf-8")
            print(f"wrote {fe.path}")
    except PermissionError as e:
        sys.stderr.write(
            f"\nmodify_pinmux commit: PermissionError\n  ({e})\n"
            f"  Likely cause: $L4T_DIR/bootloader is root-owned.\n"
            f"  Re-run the commit with sudo:\n"
            f"    sudo python3 {sys.argv[0]} commit --kb-dir {kb} "
            f"--l4t-dir {l4t_dir} --confirm\n"
        )
        return 6
    return 0


def _build_commit_state(session: dict, edits: list[dict],
                        pinmux_dtsi: pathlib.Path,
                        gpio_dtsi: pathlib.Path | None) -> _CommitState:
    """Read DTSIs and compute patched text + warnings into a _CommitState."""
    pinmux_text = pinmux_dtsi.read_text(encoding="utf-8")
    new_pinmux_text, patched, skipped = _apply_pinmux_edits(pinmux_text, edits)

    gpio_warnings: list[str] = []
    new_gpio_text: str | None = None
    gpio_text = ""
    has_gpio_edits = any(e.get("sfio") == "gpio" for e in edits)
    if gpio_dtsi is not None and has_gpio_edits:
        gpio_text = gpio_dtsi.read_text(encoding="utf-8")
        platform = session.get("platform") or "thor"
        new_gpio_text, gpio_warnings = patch_gpio_block(
            gpio_text, platform, edits)

    return _CommitState(
        pinmux=_FileEdit(path=pinmux_dtsi, before=pinmux_text,
                         after=new_pinmux_text),
        gpio=_FileEdit(path=gpio_dtsi, before=gpio_text,
                       after=new_gpio_text),
        patched=patched, skipped=skipped, gpio_warnings=gpio_warnings,
    )


def _resolve_commit_targets(session: dict, edits: list[dict],
                            l4t_dir: pathlib.Path
                            ) -> tuple[pathlib.Path | None,
                                       pathlib.Path | None, int]:
    """Resolve pinmux/gpio DTSI paths under $L4T_DIR. Returns (pinmux, gpio, rc).

    rc != 0 indicates the caller should exit immediately with that code.
    """
    pinmux_dtsi = resolve_dtsi(l4t_dir, session, "pinmux")
    gpio_dtsi = resolve_dtsi(l4t_dir, session, "gpio")
    has_gpio_edits = any(e.get("sfio") == "gpio" for e in edits)

    if pinmux_dtsi is None:
        print(f"error: cannot resolve cloned pinmux DTSI for carrier "
              f"{session.get('carrier_short')!r} under "
              f"{l4t_dir/'bootloader'} - set session.cloned_pinmux_dtsi",
              file=sys.stderr)
        return None, None, 2
    if has_gpio_edits and gpio_dtsi is None:
        print(f"warn: gpio sfio edits present but cannot resolve cloned "
              f"gpio DTSI under {l4t_dir/'bootloader'} - "
              f"set session.cloned_gpio_dtsi to enable gpio-init updates",
              file=sys.stderr)
    return pinmux_dtsi, gpio_dtsi, 0


def _prep_commit(args: argparse.Namespace
                 ) -> tuple[pathlib.Path, dict, list[dict],
                            pathlib.Path, int]:
    """Validate args/session and resolve l4t_dir. Returns (kb,session,edits,l4t,rc).

    When rc != 0 caller should return that code immediately.
    """
    kb = args.kb_dir.expanduser().resolve()
    session = load_session(kb)
    if not session:
        print(f"error: no session.json under {kb}", file=sys.stderr)
        return kb, {}, [], pathlib.Path(), 2
    edits = ((session.get("pinmux") or {}).get("pin_edits") or [])
    if not edits:
        print("session.pinmux.pin_edits is empty; nothing to commit")
        return kb, session, edits, pathlib.Path(), -1
    l4t_dir = pathlib.Path(args.l4t_dir).expanduser().resolve()
    if not (l4t_dir / "flash.sh").exists():
        print("error: --l4t-dir must contain flash.sh", file=sys.stderr)
        return kb, session, edits, l4t_dir, 2
    return kb, session, edits, l4t_dir, 0


def cmd_commit(args: argparse.Namespace) -> int:
    """Patch the cloned pinmux + gpio DTSIs under $L4T_DIR."""
    kb, session, edits, l4t_dir, rc = _prep_commit(args)
    if rc == -1:
        return 0
    if rc != 0:
        return rc

    _augment_with_gpio(session, kb, edits)
    pinmux_dtsi, gpio_dtsi, rc = _resolve_commit_targets(session, edits, l4t_dir)
    if rc != 0:
        return rc

    state = _build_commit_state(session, edits, pinmux_dtsi, gpio_dtsi)
    _print_commit_plan(state)

    if not args.confirm:
        print("\nDry-run only. Re-run with --confirm to apply.")
        return 0

    rc = _write_commit_files(state, kb, l4t_dir)
    if rc != 0:
        return rc

    pinmux_session = session.setdefault("pinmux", {})
    pinmux_session["warnings"] = (
        [f"pin {p}: no per-pin block in {pinmux_dtsi.name}" for p in state.skipped]
        + state.gpio_warnings
    )
    save_session(kb, session)
    return 0


# ===== generate_dtsi loader =============================================

def _load_generate_dtsi():
    """Import generate_dtsi.py from a sibling skill tree or from this directory."""
    candidates = [
        _SCRIPT_DIR / "generate_dtsi.py",
        (_SCRIPT_DIR.parent.parent.parent / "skills" / "custom-bsp"
         / "scripts" / "generate_dtsi.py"),
        (_SCRIPT_DIR.parent.parent.parent.parent / "skills" / "custom-bsp"
         / "scripts" / "generate_dtsi.py"),
    ]
    gen_path = next((p for p in candidates if p.exists()), None)
    if gen_path is None:
        cur = _SCRIPT_DIR
        for _ in range(6):
            cands = list(cur.rglob("generate_dtsi.py"))
            if cands:
                gen_path = cands[0]
                break
            cur = cur.parent
    if gen_path is None or not gen_path.exists():
        raise ImportError(
            f"generate_dtsi.py not found (searched from {_SCRIPT_DIR}). "
            "Ensure the custom-bsp skill is installed alongside modify-pinmux."
        )
    spec = importlib.util.spec_from_file_location("generate_dtsi", gen_path)
    gen = importlib.util.module_from_spec(spec)
    sys.modules["generate_dtsi"] = gen
    spec.loader.exec_module(gen)
    return gen


# ===== probe ============================================================

def cmd_probe(args: argparse.Namespace) -> int:
    """Parse user-provided XLSM, build pinmap KB for lookup - no DTSIs written."""
    xlsm = pathlib.Path(args.xlsm).expanduser().resolve()
    if not xlsm.exists():
        print(f"error: XLSM not found: {xlsm}", file=sys.stderr)
        return 1
    if xlsm.suffix.lower() != ".xlsm":
        print(f"error: expected an .xlsm file, got: {xlsm.name}", file=sys.stderr)
        return 1

    carrier = args.carrier_name.strip()
    if not _CARRIER_RE.match(carrier):
        print(f"error: --carrier-name must be 2-64 chars [a-z0-9_-], got: {carrier!r}",
              file=sys.stderr)
        return 1

    kb = args.kb_dir.expanduser().resolve()

    if not _HAS_OPENPYXL:
        print(_OPENPYXL_HINT, file=sys.stderr)
        return 1
    try:
        gen = _load_generate_dtsi()
    except ImportError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    print(f"probing {xlsm.name} ...", file=sys.stderr)
    pins = gen.parse_xlsm(xlsm)
    gpio_count = sum(1 for p in pins if p.resolved_func == "gpio")
    rails = {p.power_rail for p in pins if p.power_rail and p.io_voltage}
    print(f"  {len(pins)} pins  ({gpio_count} gpio, {len(rails)} pad rails)",
          file=sys.stderr)

    pinmap_dir = kb / "pinmap"
    pinmap_dir.mkdir(parents=True, exist_ok=True)
    pinmap_records = gen.build_pinmap_records(pins)
    pinmap_json_path = pinmap_dir / f"{carrier}.json"
    pinmap_json_path.write_text(json.dumps(pinmap_records, indent=2), encoding="utf-8")
    print(f"wrote {pinmap_json_path}  ({len(pinmap_records)} rows)", file=sys.stderr)

    session = load_session(kb)
    session["xlsm_path"] = str(xlsm)
    session["carrier_name"] = carrier
    session["carrier_pinmap"] = carrier
    save_session(kb, session)

    print("\nprobe complete - pinmap KB ready for lookup.")
    print(f"  pinmap KB : {pinmap_json_path}")
    print("\nNext: use `lookup --query <pin>` + `set-pin` to record edits,")
    print("      then `generate` to emit the 3 DTSIs with edits baked in.")
    return 0


# ===== generate =========================================================

@dataclass
class _GenerateCtx:
    """Resolved inputs for the `generate` subcommand."""
    xlsm: pathlib.Path
    carrier: str
    out_dir: pathlib.Path
    kb: pathlib.Path
    session: dict


def _resolve_generate_ctx(args: argparse.Namespace
                          ) -> tuple[_GenerateCtx | None, int]:
    """Resolve and validate generate's inputs. Returns (ctx, exit_code)."""
    kb = args.kb_dir.expanduser().resolve()
    session = load_session(kb)

    xlsm_str = getattr(args, "xlsm", None) or session.get("xlsm_path")
    if not xlsm_str:
        print("error: no XLSM path - pass --xlsm or run `probe` first.",
              file=sys.stderr)
        return None, 1
    xlsm = pathlib.Path(xlsm_str).expanduser().resolve()
    if not xlsm.exists():
        print(f"error: XLSM not found: {xlsm}", file=sys.stderr)
        return None, 1
    if xlsm.suffix.lower() != ".xlsm":
        print(f"error: expected an .xlsm file, got: {xlsm.name}", file=sys.stderr)
        return None, 1

    carrier_arg = getattr(args, "carrier_name", None)
    carrier = (carrier_arg or session.get("carrier_name") or "").strip()
    if not _CARRIER_RE.match(carrier):
        print(f"error: --carrier-name must be 2-64 chars [a-z0-9_-], got: {carrier!r}",
              file=sys.stderr)
        return None, 1

    out_dir_arg = getattr(args, "out_dir", None)
    if not out_dir_arg:
        print("error: --out-dir is required.", file=sys.stderr)
        return None, 1
    out_dir = pathlib.Path(out_dir_arg).expanduser().resolve()

    return _GenerateCtx(xlsm=xlsm, carrier=carrier, out_dir=out_dir,
                        kb=kb, session=session), 0


def _write_generated_dtsis(gen, ctx: _GenerateCtx, pins: list) -> dict:
    """Write the 3 DTSI files. Returns dict of {label: path}."""
    ctx.out_dir.mkdir(parents=True, exist_ok=True)
    pinmux_path = ctx.out_dir / f"tegra264-mb1-bct-pinmux-{ctx.carrier}.dtsi"
    gpio_path = ctx.out_dir / f"tegra264-mb1-bct-gpio-{ctx.carrier}.dtsi"
    padv_path = ctx.out_dir / f"tegra264-mb1-bct-padvoltage-{ctx.carrier}.dtsi"
    pad_voltage_groups = gen.parse_pad_voltage_groups(ctx.xlsm)

    pinmux_path.write_text(
        gen.generate_pinmux_dtsi(pins, ctx.carrier, ctx.xlsm.name),
        encoding="utf-8")
    gpio_path.write_text(
        gen.generate_gpio_dtsi(pins, ctx.carrier, ctx.xlsm.name),
        encoding="utf-8")
    padv_path.write_text(
        gen.generate_padvoltage_dtsi(pins, ctx.carrier, ctx.xlsm.name,
                                     pad_voltage_groups=pad_voltage_groups),
        encoding="utf-8")

    print(f"wrote {pinmux_path}", file=sys.stderr)
    print(f"wrote {gpio_path}", file=sys.stderr)
    print(f"wrote {padv_path}", file=sys.stderr)
    return {"pinmux": pinmux_path, "gpio": gpio_path, "padvoltage": padv_path}


def cmd_generate(args: argparse.Namespace) -> int:
    """Bake session pin_edits into XLSM data and emit 3 DTSIs."""
    ctx, rc = _resolve_generate_ctx(args)
    if ctx is None:
        return rc

    if not _HAS_OPENPYXL:
        print(_OPENPYXL_HINT, file=sys.stderr)
        return 1
    try:
        gen = _load_generate_dtsi()
    except ImportError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    print(f"parsing {ctx.xlsm.name} ...", file=sys.stderr)
    pins = gen.parse_xlsm(ctx.xlsm)
    gpio_count = sum(1 for p in pins if p.resolved_func == "gpio")
    rails = {p.power_rail for p in pins if p.power_rail and p.io_voltage}
    print(f"  {len(pins)} pins  ({gpio_count} gpio, {len(rails)} pad rails)",
          file=sys.stderr)

    edits = (ctx.session.get("pinmux") or {}).get("pin_edits") or []
    if edits:
        print(f"  applying {len(edits)} pin edit(s) from session ...",
              file=sys.stderr)
        pins = gen.apply_pin_edits(pins, edits)

    paths = _write_generated_dtsis(gen, ctx, pins)

    pinmap_dir = ctx.kb / "pinmap"
    pinmap_dir.mkdir(parents=True, exist_ok=True)
    pinmap_records = gen.build_pinmap_records(pins)
    pinmap_json_path = pinmap_dir / f"{ctx.carrier}.json"
    pinmap_json_path.write_text(
        json.dumps(pinmap_records, indent=2), encoding="utf-8")
    print(f"wrote {pinmap_json_path}  ({len(pinmap_records)} rows)",
          file=sys.stderr)

    ctx.session["cloned_pinmux_dtsi"] = str(paths["pinmux"])
    ctx.session["cloned_gpio_dtsi"] = str(paths["gpio"])
    ctx.session["cloned_padvoltage_dtsi"] = str(paths["padvoltage"])
    ctx.session["carrier_pinmap"] = ctx.carrier
    ctx.session["xlsm_path"] = str(ctx.xlsm)
    ctx.session["carrier_name"] = ctx.carrier
    save_session(ctx.kb, ctx.session)

    print(f"\ngenerate complete - carrier={ctx.carrier!r}")
    print(f"  pinmux DTSI    : {paths['pinmux']}")
    print(f"  gpio   DTSI    : {paths['gpio']}")
    print(f"  padvoltage DTSI: {paths['padvoltage']}")
    print(f"  pinmap KB      : {pinmap_json_path}")
    if edits:
        print(f"\n{len(edits)} pin edit(s) baked in. DTSIs are ready to deploy.")
    return 0


# ===== CLI ==============================================================

def _add_probe_parser(sub) -> None:
    """Wire the `probe` subcommand."""
    sp = sub.add_parser("probe",
                        help="parse XLSM -> build pinmap KB only (no DTSIs).")
    sp.add_argument("--xlsm", required=True,
                    help="path to the carrier/devkit pinmux .xlsm file.")
    sp.add_argument("--carrier-name", required=True,
                    help="suffix used for pinmap KB filename and later DTSIs.")
    sp.add_argument("--kb-dir", required=True, type=pathlib.Path)
    sp.set_defaults(func=cmd_probe)


def _add_generate_parser(sub) -> None:
    """Wire the `generate` subcommand."""
    sg = sub.add_parser("generate",
                        help="bake session pin_edits into XLSM data and emit DTSIs.")
    sg.add_argument("--xlsm",
                    help="path to .xlsm (default: session.xlsm_path).")
    sg.add_argument("--out-dir", required=True,
                    help="directory to write the 3 DTSI files into.")
    sg.add_argument("--carrier-name",
                    help="DTSI filename suffix (default: session.carrier_name).")
    sg.add_argument("--kb-dir", required=True, type=pathlib.Path)
    sg.set_defaults(func=cmd_generate)


def _add_lookup_parser(sub) -> None:
    """Wire the `lookup` subcommand."""
    sl = sub.add_parser("lookup",
                        help="resolve a CVM pin or signal name to pinmap rows.")
    sl.add_argument("--kb-dir", required=True, type=pathlib.Path)
    sl.add_argument("--carrier", default=None,
                    help="override carrier (else session.carrier_short).")
    sl.add_argument("--query", required=True,
                    help="CVM ball, Verilog name, DT pin name, or signal name.")
    sl.add_argument("--json", action="store_true",
                    help="emit machine-readable JSON instead of text.")
    sl.set_defaults(func=cmd_lookup)


def _add_set_pin_parser(sub) -> None:
    """Wire the `set-pin` subcommand."""
    ss = sub.add_parser("set-pin",
                        help="record a single pin's sfio/direction/state.")
    ss.add_argument("--kb-dir", required=True, type=pathlib.Path)
    ss.add_argument("--pin", required=True,
                    help="DT pin name, e.g. pex_l4_rst_n_pd1.")
    ss.add_argument("--sfio", required=True,
                    help="SFIO name (must be in `lookup` supported list).")
    ss.add_argument("--direction", required=True,
                    choices=sorted(DIRECTION_VALUES))
    ss.add_argument("--initial-state", required=True,
                    choices=sorted(INITIAL_STATE_VALUES))
    ss.add_argument("--pull", default=None, choices=sorted(PULL_VALUES),
                    help="internal pull resistor (nvidia,pull).")
    ss.add_argument("--drive-type", default=None,
                    choices=sorted(DRIVE_TYPE_VALUES),
                    help="output drive strength (nvidia,drv-type).")
    ss.add_argument("--open-drain", default=None,
                    choices=sorted(ENABLE_DISABLE_VALUES),
                    help="open-drain mode (nvidia,e-io-od).")
    ss.add_argument("--loopback", default=None,
                    choices=sorted(ENABLE_DISABLE_VALUES),
                    help="loopback enable (nvidia,e-lpbk).")
    ss.set_defaults(func=cmd_set_pin)


def _add_apply_parser(sub) -> None:
    """Wire the `apply` subcommand."""
    sa = sub.add_parser("apply",
                        help="stage pin_edits to <KB>/staged/pinmux_plan.json.")
    sa.add_argument("--kb-dir", required=True, type=pathlib.Path)
    sa.set_defaults(func=cmd_apply)


def _add_commit_parser(sub) -> None:
    """Wire the `commit` subcommand."""
    sc = sub.add_parser("commit",
                        help="patch the cloned pinmux + gpio DTSIs in $L4T_DIR.")
    sc.add_argument("--kb-dir", required=True, type=pathlib.Path)
    sc.add_argument("--l4t-dir", required=True, type=str)
    sc.add_argument("--confirm", action="store_true",
                    help="actually edit the DTSIs (default: dry-run).")
    sc.set_defaults(func=cmd_commit)


def build_parser() -> argparse.ArgumentParser:
    """Construct the top-level argparse parser with all subcommands."""
    p = argparse.ArgumentParser(prog="modify-pinmux", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    _add_probe_parser(sub)
    _add_generate_parser(sub)
    _add_lookup_parser(sub)
    _add_set_pin_parser(sub)
    _add_apply_parser(sub)
    _add_commit_parser(sub)
    return p


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    p = build_parser()
    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
