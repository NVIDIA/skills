# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Roll back one interrupted PAS DEFT state/log commit from its journal."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import subprocess
import sys
import tempfile


MARKER_NAME = ".deft_commit_transaction.json"


def _atomic_text(path: pathlib.Path, value: str) -> None:
    fd, temporary = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(value)
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def recover(results_dir: pathlib.Path) -> tuple[str, str]:
    results = results_dir.expanduser().resolve()
    state_path = results / "deft_state.json"
    log_path = results / "loop_log.jsonl"
    marker_path = results / MARKER_NAME
    if not marker_path.is_file():
        raise ValueError(f"no interrupted transaction journal found: {marker_path}")
    marker_text = marker_path.read_text()
    payload = json.loads(marker_text)
    if not isinstance(payload, dict):
        raise ValueError("transaction journal root must be an object")
    if (
        payload.get("schema_version") != "1"
        or payload.get("workflow") != "tao-run-deft-pas"
        or pathlib.Path(str(payload.get("results_dir", ""))).resolve() != results
    ):
        raise ValueError("transaction journal does not belong to this PAS DEFT run")
    original_state = payload.get("original_state_text")
    original_log = payload.get("original_log_text")
    if not isinstance(original_state, str):
        raise ValueError("transaction journal lacks original_state_text")
    try:
        state_payload = json.loads(original_state)
    except json.JSONDecodeError as exc:
        raise ValueError(f"journal original state is invalid JSON: {exc}") from exc
    if (
        not isinstance(state_payload, dict)
        or state_payload.get("workflow") != "tao-run-deft-pas"
        or pathlib.Path(str(state_payload.get("results_dir", ""))).resolve() != results
    ):
        raise ValueError("journal original state does not belong to this run")
    if original_log is not None and not isinstance(original_log, str):
        raise ValueError("transaction journal original_log_text must be string or null")

    _atomic_text(state_path, original_state)
    if original_log is None:
        try:
            log_path.unlink()
        except FileNotFoundError:
            pass
    else:
        _atomic_text(log_path, original_log)
    marker_path.unlink()

    audit = pathlib.Path(__file__).resolve().parent / "audit_deft_run.py"
    completed = subprocess.run(
        [sys.executable, str(audit), "--results-dir", str(results)],
        capture_output=True,
        text=True,
        check=False,
    )
    output = (completed.stdout + completed.stderr).strip()
    match = re.search(r"DEFT_RUN_STATUS=(\S+)", output)
    restored_status = match.group(1) if match else "UNKNOWN"
    if completed.returncode != 0 or restored_status not in {"IN_PROGRESS", "FAILED"}:
        # Keep recovery repeatable when the restored canonical pair is not
        # accepted. The journal remains the explicit mutation barrier.
        _atomic_text(marker_path, marker_text)
        raise ValueError(
            "rollback restored the saved pair but its audit was not a valid "
            "nonterminal state; the recovery journal was retained:\n"
            + output
        )
    target = f"{payload.get('iteration')}/{payload.get('stage')}"
    return target, output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", required=True, type=pathlib.Path)
    args = parser.parse_args(argv)
    try:
        target, output = recover(args.results_dir)
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        print(f"recover_commit: {exc}", file=sys.stderr)
        return 2
    print(f"recovered interrupted transaction for {target}")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
