#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Remove Cosmos Reason resumable checkpoints after an iteration is evaluated."""

from __future__ import annotations

import argparse
import os
import re
import shutil
from pathlib import Path

from workflow_common import absolute_path, atomic_write_json


TIMESTAMP_RUN = re.compile(r"^\d{14}$")
SAFETENSORS_EPOCH = re.compile(r"^epoch_\d+$")


def timestamped_run_dirs(train_dir: Path) -> list[Path]:
    """Return Cosmos Reason timestamped run directories in name order."""
    return sorted(
        child
        for child in train_dir.iterdir()
        if child.is_dir() and not child.is_symlink() and TIMESTAMP_RUN.fullmatch(child.name)
    )


def exported_safetensors(run_dirs: list[Path]) -> list[Path]:
    """Return concrete exported epoch directories that cleanup must preserve."""
    exports: list[Path] = []
    for run_dir in run_dirs:
        safetensors_dir = run_dir / "safetensors"
        if not safetensors_dir.is_dir() or safetensors_dir.is_symlink():
            continue
        exports.extend(
            child
            for child in safetensors_dir.iterdir()
            if child.is_dir() and not child.is_symlink() and SAFETENSORS_EPOCH.fullmatch(child.name)
        )
    return sorted(exports)


def directory_size(path: Path) -> int:
    """Return the size of regular files below a directory without following links."""
    total = 0
    for root, _, files in os.walk(path, followlinks=False):
        for filename in files:
            file_path = Path(root) / filename
            if not file_path.is_symlink():
                total += file_path.stat().st_size
    return total


def remove_checkpoint_path(path: Path) -> int:
    """Remove one checkpoint directory or link and return its regular-file bytes."""
    if path.is_symlink():
        path.unlink()
        return 0
    if not path.exists():
        return 0
    if not path.is_dir():
        raise ValueError(f"checkpoint path is neither a directory nor a symlink: {path}")
    size = directory_size(path)
    shutil.rmtree(path)
    return size


def cleanup_training_checkpoints(train_dir: Path) -> dict[str, object]:
    """Remove raw Cosmos-RL checkpoints while retaining exported safetensors."""
    if not train_dir.is_dir():
        raise NotADirectoryError(f"train directory does not exist: {train_dir}")

    run_dirs = timestamped_run_dirs(train_dir)
    if not run_dirs:
        raise FileNotFoundError(f"no timestamped Cosmos Reason runs found under {train_dir}")

    preserved = exported_safetensors(run_dirs)
    if not preserved:
        raise FileNotFoundError(
            f"no safetensors/epoch_<N> exports found under {train_dir}; refusing checkpoint cleanup"
        )

    targets: list[Path] = []
    best_checkpoints = train_dir / "best" / "checkpoints"
    if best_checkpoints.exists() or best_checkpoints.is_symlink():
        targets.append(best_checkpoints)

    for run_dir in run_dirs:
        checkpoints_dir = run_dir / "checkpoints"
        if checkpoints_dir.exists() or checkpoints_dir.is_symlink():
            targets.append(checkpoints_dir)
        legacy_best_checkpoints = run_dir / "best" / "checkpoints"
        if legacy_best_checkpoints.exists() or legacy_best_checkpoints.is_symlink():
            targets.append(legacy_best_checkpoints)

    removed_bytes = 0
    for target in targets:
        removed_bytes += remove_checkpoint_path(target)

    missing_exports = [str(path) for path in preserved if not path.is_dir()]
    if missing_exports:
        raise RuntimeError(f"safetensors disappeared during cleanup: {missing_exports}")

    candidates = [train_dir / "best" / "checkpoints"]
    for run_dir in run_dirs:
        candidates.extend((run_dir / "checkpoints", run_dir / "best" / "checkpoints"))
    remaining = [str(path) for path in candidates if path.exists() or path.is_symlink()]
    if remaining:
        raise RuntimeError(f"checkpoint paths remain after cleanup: {remaining}")

    return {
        "train_dir": str(train_dir),
        "removed_checkpoint_paths": [str(path) for path in targets],
        "removed_checkpoint_bytes": removed_bytes,
        "preserved_safetensors": [str(path) for path in preserved],
    }


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    """Clean one iteration train directory and write its cleanup report."""
    args = parse_args()
    train_dir = absolute_path(args.train_dir)
    report = cleanup_training_checkpoints(train_dir)
    report_path = train_dir / "checkpoint_cleanup.json"
    atomic_write_json(report_path, report)
    print(f"removed checkpoint paths: {len(report['removed_checkpoint_paths'])}")
    print(f"removed checkpoint bytes: {report['removed_checkpoint_bytes']}")
    print(f"preserved safetensors exports: {len(report['preserved_safetensors'])}")
    print(f"report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
