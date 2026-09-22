#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Create or recover the explicit session identity used by TAO AutoML.

The AutoML wheel otherwise generates a new random identity on each invocation,
which makes ``resume=True`` silently start a second controller in the same
workspace. This helper is deliberately read-only: the runner persists the
identity in ``.automl/controller/<session_id>.json``.
"""

from __future__ import annotations

import argparse
import json
import re
import secrets
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any


SESSION_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class SessionResolutionError(ValueError):
    """Raised when an AutoML workspace cannot be resumed unambiguously."""


def new_session_id() -> str:
    """Return a wheel-compatible 12-character session identity."""

    return secrets.token_hex(6)


def _controller_files(workspace: Path) -> list[Path]:
    controller_dir = workspace / ".automl" / "controller"
    if not controller_dir.is_dir():
        raise SessionResolutionError(
            f"resume state is missing: {controller_dir} does not exist; "
            "refuse to start a fresh search with resume=True"
        )
    return sorted(controller_dir.glob("*.json"))


def _validate_controller(path: Path) -> None:
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SessionResolutionError(
            f"controller state is unreadable: {path}: {exc}"
        ) from exc
    if not isinstance(state, list):
        raise SessionResolutionError(
            f"controller state has unexpected shape: {path} must contain a JSON list"
        )


def resolve_session_id(workspace: Path, requested: str | None = None) -> str:
    """Resolve exactly one existing controller identity for a resumed run."""

    workspace = workspace.expanduser().resolve()
    files = _controller_files(workspace)

    if requested is not None:
        if not SESSION_ID_RE.fullmatch(requested):
            raise SessionResolutionError(f"invalid session id: {requested!r}")
        selected = workspace / ".automl" / "controller" / f"{requested}.json"
        if selected not in files:
            available = ", ".join(path.stem for path in files) or "none"
            raise SessionResolutionError(
                f"session {requested!r} has no controller state in {workspace}; "
                f"available sessions: {available}"
            )
    else:
        if not files:
            raise SessionResolutionError(
                f"resume state is missing: no controller JSON exists in "
                f"{workspace / '.automl' / 'controller'}; refuse to start a fresh "
                "search with resume=True"
            )
        if len(files) != 1:
            available = ", ".join(path.stem for path in files)
            raise SessionResolutionError(
                f"resume state is ambiguous: found {len(files)} controller sessions "
                f"in {workspace}: {available}; select the intended session explicitly "
                "with --session-id after inspecting its state"
            )
        selected = files[0]

    if not SESSION_ID_RE.fullmatch(selected.stem):
        raise SessionResolutionError(
            f"controller filename is not a valid session id: {selected.name}"
        )
    _validate_controller(selected)
    return selected.stem


def validate_session_settings(
    automl_settings: Mapping[str, Any],
    *,
    resume: bool,
    workspace: Path | None = None,
) -> str:
    """Fail before runner construction when session identity is not durable."""
    if not isinstance(automl_settings, Mapping):
        raise SessionResolutionError("automl_settings must be a mapping")
    session_id = automl_settings.get("session_id")
    if not isinstance(session_id, str) or not SESSION_ID_RE.fullmatch(session_id):
        raise SessionResolutionError(
            "automl_settings.session_id must be an explicit valid session id"
        )
    if resume:
        if workspace is None:
            raise SessionResolutionError(
                "resume=True requires the full existing workspace path"
            )
        resolved = resolve_session_id(workspace, session_id)
        if resolved != session_id:  # defensive: requested resolution must be exact
            raise SessionResolutionError(
                f"resolved session {resolved!r} does not match settings {session_id!r}"
            )
    return session_id


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("new", help="emit a new explicit session id")
    resume = subparsers.add_parser(
        "resolve", help="recover the session id from an existing run workspace"
    )
    resume.add_argument("--workspace", required=True, type=Path)
    resume.add_argument(
        "--session-id",
        help="select an existing session when a previously broken workspace is ambiguous",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "new":
            session_id = new_session_id()
        else:
            session_id = resolve_session_id(args.workspace, args.session_id)
    except SessionResolutionError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(session_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
