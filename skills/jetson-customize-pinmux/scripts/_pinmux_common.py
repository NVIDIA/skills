#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared helpers for modify_pinmux: validation, pinmap/session IO, row helpers."""
from __future__ import annotations

import json
import pathlib
import re


# ===== validation regexes ===============================================

QUERY_RE = re.compile(r"^[A-Za-z0-9_/]{1,64}$")
PIN_NAME_RE = re.compile(r"^[a-z0-9_]{2,64}$")
DIRECTION_VALUES = {"input", "output", "bidirectional", "unused"}
INITIAL_STATE_VALUES = {"low", "high", "hi-z", "n/a"}
PULL_VALUES = {"none", "pull-up", "pull-down"}
DRIVE_TYPE_VALUES = {"normal", "high"}
ENABLE_DISABLE_VALUES = {"enable", "disable"}


# ===== platform tables ==================================================

# Recognised carriers map by glob-stem fragment (substring) -> SoC prefix.
SOC_PREFIX = {
    "thor": "tegra264",
    "orin": "tegra234",
    "orin_nano": "tegra234",
}


# ===== pinmap I/O =======================================================

def resolve_pinmap_path(kb_dir: pathlib.Path, session: dict | None,
                        carrier: str | None) -> pathlib.Path:
    """Locate the pinmap JSON file under kb_dir."""
    explicit = (session or {}).get("carrier_pinmap")
    if explicit:
        if explicit.endswith(".json"):
            p = pathlib.Path(explicit)
            if not p.is_absolute():
                p = kb_dir / "pinmap" / explicit
            if p.exists():
                return p
        else:
            p = kb_dir / "pinmap" / f"{explicit}.json"
            if p.exists():
                return p

    pinmap_dir = kb_dir / "pinmap"
    if not pinmap_dir.exists():
        raise FileNotFoundError(f"no pinmap dir at {pinmap_dir}")
    candidates = sorted(
        p for p in pinmap_dir.glob("*.json")
        if not p.stem.endswith("_by_net")
    )
    if carrier:
        narrowed = [p for p in candidates if carrier in p.stem]
        if narrowed:
            candidates = narrowed
    if not candidates:
        raise FileNotFoundError(f"no pinmap files in {pinmap_dir}")
    if len(candidates) > 1:
        names = "\n  ".join(p.name for p in candidates)
        raise FileNotFoundError(
            f"multiple pinmap candidates in {pinmap_dir}; set "
            f"session.carrier_pinmap to disambiguate. Candidates:\n  {names}"
        )
    return candidates[0]


def load_pinmap(path: pathlib.Path) -> list[dict]:
    """Load pinmap JSON, expecting a list of row dicts."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"pinmap at {path} is not a list")
    return data


def load_session(kb_dir: pathlib.Path) -> dict:
    """Load session.json from kb_dir, or return empty dict."""
    sp = kb_dir / "session.json"
    if not sp.exists():
        return {}
    return json.loads(sp.read_text(encoding="utf-8"))


def save_session(kb_dir: pathlib.Path, session: dict) -> None:
    """Persist session.json under kb_dir."""
    (kb_dir / "session.json").write_text(
        json.dumps(session, indent=2), encoding="utf-8")


# ===== pinmap row helpers ===============================================

def row_haystack(row: dict) -> list[str]:
    """Return the lower-cased strings the lookup query matches against."""
    parts: list[str] = []
    for key in ("pin", "ball", "bga_ball", "package_ball",
                "connector_pin",
                "verilog_name", "signal_name", "customer_usage",
                "devkit_usage"):
        v = row.get(key)
        if isinstance(v, str) and v:
            parts.append(v.lower())
    for ev in row.get("evidence") or []:
        net = ev.get("net")
        if isinstance(net, str) and net:
            parts.append(net.lower())
    return parts


def sfio_options(row: dict) -> list[str]:
    """Return the canonical SFIO names the silicon supports on this pin."""
    out: list[str] = []
    for entry in row.get("sfio") or []:
        if "=" not in entry:
            continue
        mode, name = entry.split("=", 1)
        mode = mode.strip().lower()
        name = name.strip()
        if mode == "gpio":
            out.append("gpio")
            continue
        out.append(name.lower())
    return out


def suggested_sfio(row: dict, options: list[str]) -> str | None:
    """Suggest an SFIO option matching the row's customer_usage."""
    cu = (row.get("customer_usage") or "").lower()
    if not cu:
        return None
    if cu.startswith("unused"):
        return None
    for o in options:
        if o and o in cu:
            return o
    return None


# ===== gpio sfio decoding ===============================================

_GPIO_DESC_RE = re.compile(
    r"GPIO(?P<group>\d+)_P(?P<bank>[A-Z]{1,2})\.(?P<idx>\d+)",
    re.IGNORECASE,
)


def parse_gpio_sfio(row: dict) -> tuple[str, int, str] | None:
    """Return (bank, idx, controller_kind) for a pin's gpio= entry, or None."""
    has_gpio = any((s or "").lower().startswith("gpio=") for s in row.get("sfio") or [])
    if not has_gpio:
        return None

    pin = row.get("pin", "")
    m = re.search(r"_p([a-z]{1,2})(\d+)$", pin, re.IGNORECASE)
    if m:
        bank = m.group(1).upper()
        idx = int(m.group(2))
    else:
        m2 = None
        for s in row.get("sfio") or []:
            m2 = _GPIO_DESC_RE.search(s)
            if m2:
                break
        if not m2:
            return None
        bank = m2.group("bank").upper()
        idx = int(m2.group("idx"))

    kind = "AON" if len(bank) >= 2 else "MAIN"
    return (bank, idx, kind)


# ===== lookup formatting ================================================

def lookup_summary(row: dict) -> dict:
    """Return a JSON-serialisable summary of a pinmap row for lookup output."""
    options = sfio_options(row)
    configurable = row.get("configurable", True)
    return {
        "pin": row.get("pin"),
        "ball": row.get("ball"),
        "verilog_name": row.get("verilog_name"),
        "signal_name": row.get("signal_name"),
        "sfio_options": options,
        "suggested_sfio": suggested_sfio(row, options),
        "customer_usage": row.get("customer_usage"),
        "default_direction": row.get("direction"),
        "pad_type": row.get("pad_type", ""),
        "configurable": configurable,
        "por_state": row.get("por_state", ""),
        "default_pull": row.get("pupd", "none") if configurable else "n/a",
        "default_drv_type": row.get("drv_type", "normal") if configurable else "n/a",
        "default_open_drain": row.get("e_io_od", "disable") if configurable else "n/a",
        "default_loopback": row.get("e_lpbk", "disable") if configurable else "n/a",
        "evidence_nets": [e.get("net") for e in (row.get("evidence") or [])],
    }


def print_row_details(row: dict) -> None:
    """Pretty-print a single pinmap row for the `lookup` subcommand."""
    options = sfio_options(row)
    suggested = suggested_sfio(row, options)
    configurable = row.get("configurable", True)
    print(f"pin             : {row.get('pin')}")
    print(f"ball (CVM)      : {row.get('connector_pin') or '-'}")
    print(f"ball (MPIO)     : {row.get('ball')}")
    print(f"verilog_name    : {row.get('verilog_name')}")
    print(f"signal_name     : {row.get('signal_name')}")
    print(f"customer_usage  : {row.get('customer_usage')}")
    print(f"pad_type        : {row.get('pad_type', '')}")
    cfg_text = "yes" if configurable else "no (fixed-function pad - skip Q4-6)"
    print(f"configurable    : {cfg_text}")
    print(f"default direction: {row.get('direction')}")
    if row.get("por_state"):
        print(f"por_state       : {row.get('por_state')}  (informational)")
    if configurable:
        print(f"default pull    : {row.get('pupd', 'none')}")
        print(f"default drv_type: {row.get('drv_type', 'normal')}")
        print(f"default open_drain: {row.get('e_io_od', 'disable')}")
    print(f"category        : {row.get('category')}")
    nets = [e.get("net") for e in (row.get("evidence") or [])]
    if nets:
        print(f"evidence nets   : {', '.join(n for n in nets if n)}")
    if not options:
        print("sfio_options    : (none - pin reserved or non-customisable)")
    else:
        print("sfio_options    :")
        for o in options:
            tag = "  (suggested)" if suggested and o == suggested else ""
            print(f"  - {o}{tag}")
