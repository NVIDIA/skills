#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Helpers shared by the renderers and runners (imported through nd_common): deterministic YAML,
strict input parsing, and the open/close steps of a render. Stdlib only."""

from __future__ import annotations

import argparse
import json
import re
import shlex
import sys
from pathlib import Path
from typing import Any, NoReturn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nd_state as ns

# --- deterministic YAML ---------------------------------------------------------------------

_PLAIN = re.compile(r"[A-Za-z_./][A-Za-z0-9_./=+-]*")
_RESERVED = {"true", "false", "yes", "no", "on", "off", "null", "y", "n", "~"}


def _scalar(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return json.dumps(value)
    text = str(value)
    if _PLAIN.fullmatch(text) and text.lower() not in _RESERVED:
        return text
    return json.dumps(text)  # a JSON string is a valid double-quoted YAML scalar


def _inline(value: Any) -> str:
    if isinstance(value, dict):
        return "{}"
    if isinstance(value, list):
        return "[]"
    return _scalar(value)


def _emit_yaml(value: Any, indent: int, lines: list[str]) -> None:
    pad = " " * indent
    if isinstance(value, dict):
        for key, item in value.items():
            if isinstance(item, (dict, list)) and item:
                lines.append(f"{pad}{_scalar(str(key))}:")
                _emit_yaml(item, indent + 2 if isinstance(item, dict) else indent, lines)
            else:
                lines.append(f"{pad}{_scalar(str(key))}: {_inline(item)}")
        return
    for item in value:
        if isinstance(item, (dict, list)) and item:
            sub: list[str] = []
            _emit_yaml(item, indent + 2, sub)
            lines.append(f"{pad}- {sub[0].lstrip()}")
            lines.extend(sub[1:])
        else:
            lines.append(f"{pad}- {_inline(item)}")


def to_yaml(obj: Any) -> str:
    lines: list[str] = []
    _emit_yaml(obj, 0, lines)
    return "\n".join(lines) + "\n"


# --- strict inputs --------------------------------------------------------------------------


class RenderError(ValueError):
    """An input value that cannot produce a valid manifest."""


def flag_input(inputs: dict[str, Any], key: str, default: bool = False) -> bool:
    """Strict boolean run input (values set from the command line are text)."""
    try:
        return ns.flag(inputs.get(key, default))
    except ValueError as exc:
        raise RenderError(f"{key}: {exc}") from None


def whole_number(inputs: dict[str, Any], key: str, default: int | None = None) -> int:
    value = inputs.get(key, default)
    try:
        number = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        raise RenderError(f"{key} must be a whole number, got {value!r}") from None
    if isinstance(value, bool) or number < 1 or str(number) != str(value).strip():
        raise RenderError(f"{key} must be a whole number of at least 1, got {value!r}")
    return number


CREDENTIAL_ARG = re.compile(
    r"nvapi-|^hf_[A-Za-z0-9]{8,}|^--?(?:[a-z0-9]+[-_])*(?:token|api[-_]key|secret|password)(?:=|$)",
    re.IGNORECASE,
)


def parse_extra_args(value: Any) -> list[str]:
    """extra_args (engine arguments) as a list: a shell string from `set`, or a list from a tuned
    result. Refuses credentials and control characters."""
    if value in (None, "", []):
        return []
    args = shlex.split(value) if isinstance(value, str) else [str(arg) for arg in value]
    for arg in args:
        if CREDENTIAL_ARG.search(arg):
            raise RenderError(
                f"extra_args {arg!r} looks like a credential; pass secrets by Secret reference"
            )
        if not arg or any(ord(char) < 32 for char in arg):
            raise RenderError("extra_args must be non-empty text without control characters")
    return args


# --- open and close a render ----------------------------------------------------------------


def render_parser(description: str, epilog: str) -> argparse.ArgumentParser:
    parser = ns.JsonArgumentParser(
        description=description,
        epilog=epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--run", required=True, help="run id")
    parser.add_argument(
        "--print",
        dest="preview",
        action="store_true",
        help="write the files and report them without advancing the run",
    )
    return parser


def _stop(
    state: str, next_: str, code: str, message: str, fix: str, exit_code: int | None = None
) -> NoReturn:
    ns.emit(False, state, next_, errors=[ns.error(code, message, fix)], code=exit_code)


def open_run(
    run_id: str, targets: tuple[str, ...], preview: bool, extra_missing: Any = None
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load a run that is ready to render and return (state, inputs); emit and exit otherwise."""
    try:
        state = ns.load_state(run_id)
    except (FileNotFoundError, ValueError) as exc:
        _stop("unknown", "ask_user", "run_not_found", str(exc), "List runs with: nimdeploy.py list")
    current = state["state"]
    if state.get("target") not in targets:
        message = f"run target is {state.get('target')!r}, expected {' or '.join(targets)}"
        _stop(
            current, "ask_user", "wrong_target", message, "Use the renderer for the run's target."
        )
    if current != "render" and not preview:
        fix = "Run the next_command from nimdeploy.py status RUN, or use --print."
        message = f"run is in state {current!r}, not 'render'"
        _stop(current, current, "wrong_state", message, fix, ns.EXIT_PRECONDITION)
    inputs = ns.run_inputs(state)
    absent = ns.missing_inputs(state) + (extra_missing(inputs) if extra_missing else [])
    if absent:
        fix = "Ask the user for each value, run nimdeploy.py set RUN key=value, then render."
        error = ns.error("missing_input", f"missing run inputs: {', '.join(absent)}", fix)
        ns.emit(False, current, "ask_user", {"missing": absent}, [error], ns.EXIT_PRECONDITION)
    return state, inputs


def reject_input(state: dict[str, Any], code: str, exc: Exception) -> NoReturn:
    fix = "Correct the input with nimdeploy.py set RUN key=value, then render again."
    _stop(state["state"], "ask_user", code, str(exc), fix)


def close_run(
    state: dict[str, Any], paths: list[Path], facts: dict[str, Any], preview: bool
) -> NoReturn:
    """Preview: report only. Otherwise hash the artifacts and move render -> validate."""
    current = state["state"]
    if preview:
        ns.emit(True, current, current, facts)
    try:
        ns.transition(state, "validate", [str(path) for path in paths])
    except ValueError as exc:
        error = ns.error("too_many_attempts", str(exc), "Ask the user before rendering again.")
        ns.emit(False, current, "ask_user", facts, [error], ns.EXIT_PRECONDITION)
    ns.emit(True, "validate", "validate", facts)


if __name__ == "__main__":
    ns.emit(True, "library", "done", {"note": "import this module; it has no command line"})
