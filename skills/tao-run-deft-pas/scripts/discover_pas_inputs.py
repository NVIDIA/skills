#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Perform the bounded, read-only PAS archive discovery contract.

This is deliberately code rather than an agent search recipe: the same depth,
symlink, filename, and precedence rules are applied on every machine.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
from dataclasses import asdict, dataclass
from typing import Iterator


ARCHIVE_NAMES = ("images_raw.tar", "meta.tar.gz")
TEXT_AUXILIARY_SUFFIXES = frozenset({".adoc", ".md", ".prompt", ".rst", ".txt"})
TEXT_AUXILIARY_PREFIXES = ("instruction", "notice", "readme")
EXTRACTED_MARKERS = (
    "images_raw",
    "images",
    "captions",
    "rebuild.py",
    "train_pairs.json",
)
NO_CANDIDATE_PROMPT = (
    "No PAS archive pair was found within the approved search bounds. "
    "Provide the archive root or the exact paths to images_raw.tar and meta.tar.gz."
)
AMBIGUOUS_CANDIDATE_PROMPT = (
    "Multiple PAS archive candidates are within the approved search bounds. "
    "Select one reported archive root or provide the exact archive file paths."
)


@dataclass(frozen=True)
class SearchRoot:
    path: str
    reason: str
    candidate_depth: str
    status: str


@dataclass(frozen=True)
class Candidate:
    path: str
    found_under: str
    depth: int
    type: str
    images_archive: str
    images_size: int
    metadata_archive: str
    metadata_size: int
    auxiliary_text_files: tuple[dict[str, object], ...]


@dataclass(frozen=True)
class AuxiliaryTextFile:
    """Metadata-only notice for a text file that agents must not interpret."""

    path: str
    size: int
    handling: str = "not-read-untrusted-data"


def _absolute(path: pathlib.Path) -> pathlib.Path:
    return pathlib.Path(os.path.abspath(path.expanduser()))


def _has_symlink_hop(path: pathlib.Path) -> bool:
    absolute = _absolute(path)
    current = pathlib.Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        try:
            if current.is_symlink():
                return True
        except OSError:
            return True
    return False


def _classify(directory: pathlib.Path) -> str | None:
    if _has_symlink_hop(directory) or not directory.is_dir():
        return None
    archives = [directory / name for name in ARCHIVE_NAMES]
    if any(path.is_symlink() or not path.is_file() for path in archives):
        return None
    mixed = any((directory / marker).exists() for marker in EXTRACTED_MARKERS)
    return "archive-with-extracted-data" if mixed else "archive-only"


def _directories(root: pathlib.Path, maximum_depth: int) -> Iterator[tuple[pathlib.Path, int]]:
    """Yield non-symlink directories without traversing beyond the approved depth."""
    pending = [(root, 0)]
    while pending:
        directory, depth = pending.pop(0)
        yield directory, depth
        if depth >= maximum_depth:
            continue
        try:
            entries = sorted(os.scandir(directory), key=lambda item: item.name)
        except (OSError, PermissionError):
            continue
        for entry in entries:
            try:
                if entry.is_dir(follow_symlinks=False) and not entry.is_symlink():
                    pending.append((pathlib.Path(entry.path), depth + 1))
            except OSError:
                continue


def _candidate(
    directory: pathlib.Path, root: pathlib.Path, depth: int
) -> Candidate | None:
    candidate_type = _classify(directory)
    if candidate_type is None:
        return None
    images, metadata = (directory / name for name in ARCHIVE_NAMES)
    auxiliary_text_files: list[dict[str, object]] = []
    try:
        entries = sorted(os.scandir(directory), key=lambda item: item.name)
    except (OSError, PermissionError):
        entries = []
    for entry in entries:
        name = entry.name.casefold()
        try:
            is_regular_file = entry.is_file(follow_symlinks=False)
            is_symlink = entry.is_symlink()
        except OSError:
            continue
        if (
            not is_regular_file
            or is_symlink
            or entry.name in ARCHIVE_NAMES
            or (
                pathlib.Path(name).suffix not in TEXT_AUXILIARY_SUFFIXES
                and not name.startswith(TEXT_AUXILIARY_PREFIXES)
            )
        ):
            continue
        try:
            size = entry.stat(follow_symlinks=False).st_size
        except OSError:
            continue
        auxiliary_text_files.append(
            asdict(AuxiliaryTextFile(path=str(directory / entry.name), size=size))
        )
    return Candidate(
        path=str(directory),
        found_under=str(root),
        depth=depth,
        type=candidate_type,
        images_archive=str(images),
        images_size=images.stat().st_size,
        metadata_archive=str(metadata),
        metadata_size=metadata.stat().st_size,
        auxiliary_text_files=tuple(auxiliary_text_files),
    )


def _explicit_pair(images: pathlib.Path, metadata: pathlib.Path) -> dict:
    images = _absolute(images)
    metadata = _absolute(metadata)
    if images.name != ARCHIVE_NAMES[0] or metadata.name != ARCHIVE_NAMES[1]:
        raise ValueError(
            "explicit archive files must be named images_raw.tar and meta.tar.gz"
        )
    if images.parent != metadata.parent:
        raise ValueError("explicit archive files must share one parent directory")
    candidate = _candidate(images.parent, images.parent, 0)
    if candidate is None:
        raise ValueError(
            "explicit archive pair must be non-symlink regular files under a "
            "non-symlink parent"
        )
    return {
        "search_roots": [],
        "candidates": [asdict(candidate)],
        "selection": str(images.parent),
        "selection_reason": "user-supplied",
        "next_action": "use-selected-archive-pair",
        "required_user_prompt": None,
        "broad_home_repository_search": False,
    }


def discover(
    *,
    workspace: pathlib.Path,
    home: pathlib.Path,
    archive_root: pathlib.Path | None = None,
) -> dict:
    workspace = _absolute(workspace)
    home = _absolute(home)
    roots: list[tuple[pathlib.Path, str, int]]
    if archive_root is not None:
        roots = [(_absolute(archive_root), "user-supplied", 2)]
    else:
        roots = [
            (home / "pas", "conventional-~/pas", 2),
            (workspace, "workspace-root", 0),
            (workspace / "pas", "workspace-child", 2),
            (workspace / "input", "workspace-child", 2),
            (workspace / "inputs", "workspace-child", 2),
        ]

    reports: list[SearchRoot] = []
    candidates: dict[str, Candidate] = {}
    seen_roots: set[tuple[str, int]] = set()
    for root, reason, depth_limit in roots:
        key = (str(root), depth_limit)
        if key in seen_roots:
            continue
        seen_roots.add(key)
        depth_text = "0" if depth_limit == 0 else "0..2"
        if _has_symlink_hop(root):
            reports.append(SearchRoot(str(root), reason, depth_text, "unsafe-symlink"))
            continue
        if not root.exists():
            reports.append(SearchRoot(str(root), reason, depth_text, "missing"))
            continue
        if not root.is_dir():
            reports.append(SearchRoot(str(root), reason, depth_text, "inaccessible"))
            continue
        reports.append(SearchRoot(str(root), reason, depth_text, "searched"))
        for directory, depth in _directories(root, depth_limit):
            item = _candidate(directory, root, depth)
            if item is not None:
                candidates.setdefault(item.path, item)

    ordered = list(candidates.values())
    archive_only = [item for item in ordered if item.type == "archive-only"]
    if len(ordered) == 1 and len(archive_only) == 1:
        selection = archive_only[0].path
        selection_reason = "sole-archive-only"
        next_action = "use-selected-archive-pair"
        required_user_prompt = None
    elif not ordered:
        selection = None
        selection_reason = "none-found"
        next_action = "request-archive-location"
        required_user_prompt = NO_CANDIDATE_PROMPT
    else:
        selection = None
        selection_reason = "user-choice-required"
        next_action = "request-archive-choice"
        required_user_prompt = AMBIGUOUS_CANDIDATE_PROMPT
    return {
        "search_roots": [asdict(item) for item in reports],
        "candidates": [asdict(item) for item in ordered],
        "selection": selection,
        "selection_reason": selection_reason,
        "next_action": next_action,
        "required_user_prompt": required_user_prompt,
        "broad_home_repository_search": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", required=True, type=pathlib.Path)
    parser.add_argument("--home", type=pathlib.Path, default=pathlib.Path.home())
    parser.add_argument("--archive-root", type=pathlib.Path)
    parser.add_argument("--images-archive", type=pathlib.Path)
    parser.add_argument("--metadata-archive", type=pathlib.Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if (args.images_archive is None) != (args.metadata_archive is None):
            raise ValueError(
                "--images-archive and --metadata-archive must be supplied together"
            )
        if args.archive_root is not None and args.images_archive is not None:
            raise ValueError(
                "--archive-root cannot be combined with explicit archive files"
            )
        result = (
            _explicit_pair(args.images_archive, args.metadata_archive)
            if args.images_archive is not None
            else discover(
                workspace=args.workspace,
                home=args.home,
                archive_root=args.archive_root,
            )
        )
    except (OSError, ValueError) as exc:
        print(f"discover_pas_inputs: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
