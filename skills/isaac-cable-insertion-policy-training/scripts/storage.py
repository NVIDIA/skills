# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Versioned campaign records, durable journals, and artifact fingerprints."""

from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

SCHEMA_VERSION = 2


def atomic_json(path: Path, value: object) -> None:
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temp.open("w") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(path)


def read_session(session: Path) -> tuple[dict, dict]:
    path = session / "session.json"
    if path.exists():
        document = json.loads(path.read_text())
        if document["schema_version"] != SCHEMA_VERSION:
            raise ValueError("unsupported session schema_version")
        plan, state = document["plan"], document["state"]
    else:
        # Historical sessions remain readable, but are never silently migrated.
        plan = json.loads((session / "plan.json").read_text())
        state = json.loads((session / "state.json").read_text())
    if state.get("current"):
        state["current"] = next(item for item in state["attempts"]
                                if item["run_id"] == state["current"]["run_id"])
    return plan, state


def save_state(session: Path, state: dict) -> None:
    document = json.loads((session / "session.json").read_text())
    document["state"] = state
    atomic_json(session / "session.json", document)


def append_record(path: Path, record: dict) -> None:
    """A durable newline is the commit marker; repair only an incomplete tail."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        with path.open("r+b") as stream:
            stream.seek(0, os.SEEK_END)
            end = stream.tell()
            if end:
                stream.seek(end - 1)
                if stream.read(1) != b"\n":
                    # Interrupted append: retain the fragment in the event journal.
                    data = path.read_bytes()
                    boundary = data.rfind(b"\n") + 1
                    stream.truncate(boundary)
                    fragment = data[boundary:].decode(errors="replace")
                    stream.seek(0, os.SEEK_END)
                    stream.write((json.dumps({"kind": "interrupted_record", "fragment": fragment}) + "\n").encode())
    with path.open("a") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    # An interrupted final append is not a completed evaluation.
    lines = path.read_bytes().splitlines(keepends=True)
    return [json.loads(line) for line in lines if line.endswith(b"\n")]


def completed_result(path: Path, evaluation_id: str, seed: int) -> dict | None:
    result = None
    for item in records(path):
        if item.get("evaluation_id") != evaluation_id or item.get("seed") != seed:
            continue
        if item.get("kind") == "result":
            result = item
        elif item.get("kind") == "process" and (item.get("exit_code") not in (0, None)
                                                or item.get("stop_reason")):
            result = None
    return result


def slug(value: str) -> str:
    result = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")[:48].rstrip("-")
    if not result:
        raise ValueError("name must contain letters or digits")
    return result


def timestamp(value: str) -> str:
    return datetime.fromisoformat(value).astimezone(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S_UTC")


def local_path(session: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else session / path


def run_folder(session: Path, attempt: dict) -> Path:
    return local_path(session, attempt.get("record_dir", f"attempts/{attempt['run_id']}"))


def fingerprint(path: Path) -> dict:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def training_fingerprint(root: Path) -> str:
    """Identify editable task code and locked dependencies, including local edits."""
    source = root / "src/isaaclab_training"
    if not (source / "__init__.py").is_file():
        raise RuntimeError(f"Training source missing: {source}")
    files = [root / "pyproject.toml", root / "uv.lock"]
    files.extend(path for path in source.rglob("*") if path.is_file()
                 and "__pycache__" not in path.parts and path.suffix not in {".pyc", ".pyo"})
    hashes = {str(path.relative_to(root)): fingerprint(path)["sha256"] for path in sorted(files)}
    return hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()


def evaluation_provenance(session: Path, plan: dict, attempt: dict) -> dict:
    """Reject changed task defaults/config evidence before launching or reusing results."""
    expected = attempt.get("training_sha256")
    if not expected:
        raise RuntimeError("Run has no training fingerprint; historical evaluation cannot be verified")
    if training_fingerprint(Path(plan["training_root"])) != expected:
        raise RuntimeError("Training source or lockfile changed since this run; restore its environment before evaluation")
    configs = {}
    for item in attempt.get("resolved_configs", []):
        path = local_path(session, item["path"])
        if not path.is_file() or fingerprint(path)["sha256"] != item["sha256"]:
            raise RuntimeError(f"Saved training configuration changed or is missing: {path}")
        configs[path.name] = item["sha256"]
    if set(configs) != {"env.yaml", "agent.yaml"}:
        raise RuntimeError("Evaluation requires the recorded env.yaml and agent.yaml fingerprints")
    return {"training_sha256": expected, "resolved_configs": configs}


def cleanup_empty_logs(session: Path) -> None:
    for path in session.rglob("*.log"):
        if path.is_file() and path.stat().st_size == 0:
            path.unlink()
