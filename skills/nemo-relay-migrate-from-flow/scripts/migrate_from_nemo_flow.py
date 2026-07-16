#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Conservatively rewrite NeMo Flow identifiers to NeMo Relay identifiers."""

from __future__ import annotations

import argparse
import ctypes
import os
import secrets
import stat
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

REPLACEMENTS: tuple[tuple[str, str], ...] = (
    ("NVIDIA NeMo Flow", "NVIDIA NeMo Relay"),
    ("NeMo Flow", "NeMo Relay"),
    ("Nemo Flow", "Nemo Relay"),
    ("NeMo-Flow", "NeMo-Relay"),
    ("NemoFlow", "NemoRelay"),
    ("nemoFlow", "nemoRelay"),
    ("NEMO_FLOW", "NEMO_RELAY"),
    ("nemo-flow", "nemo-relay"),
    ("nemo_flow", "nemo_relay"),
)

LINUX_RENAME_NOREPLACE = 1
DARWIN_RENAME_EXCL = 0x00000004
DARWIN_RENAME_NOFOLLOW_ANY = 0x00000010

SKIP_DIRS = {
    ".git",
    ".hg",
    ".mypy_cache",
    ".nox",
    ".pytest_cache",
    ".ruff_cache",
    ".svn",
    ".tox",
    ".venv",
    "__pycache__",
    "_build",
    "_generated",
    "build",
    "coverage",
    "dist",
    "htmlcov",
    "node_modules",
    "target",
    "venv",
}

LOCKFILE_NAMES = {
    "Cargo.lock",
    "package-lock.json",
    "pnpm-lock.yaml",
    "poetry.lock",
    "uv.lock",
    "yarn.lock",
}

TEXT_SUFFIXES = {
    ".bash",
    ".c",
    ".cfg",
    ".cjs",
    ".cmake",
    ".cpp",
    ".css",
    ".cu",
    ".cuh",
    ".go",
    ".h",
    ".hpp",
    ".html",
    ".ini",
    ".js",
    ".json",
    ".jsx",
    ".lock",
    ".md",
    ".mjs",
    ".py",
    ".pyi",
    ".rs",
    ".rst",
    ".sh",
    ".toml",
    ".ts",
    ".tsx",
    ".txt",
    ".xml",
    ".yaml",
    ".yml",
    ".zsh",
}

TEXT_FILENAMES = {
    ".dockerignore",
    ".gitignore",
    ".gitlab-ci.yml",
    "Dockerfile",
    "Justfile",
    "Makefile",
    "justfile",
}


@dataclass(frozen=True)
class FileChange:
    path: Path
    count: int


@dataclass(frozen=True)
class PathChange:
    old: Path
    new: Path


class MutationError(RuntimeError):
    """Raised when a requested write or rename cannot be completed safely."""


def apply_replacements(text: str) -> tuple[str, int]:
    count = 0
    updated = text
    for old, new in REPLACEMENTS:
        occurrences = updated.count(old)
        if occurrences:
            updated = updated.replace(old, new)
            count += occurrences
    return updated, count


def should_skip_dir(name: str, include_generated: bool) -> bool:
    if include_generated and name == "_generated":
        return False
    return name in SKIP_DIRS


def is_safe_entry(path: Path, root: Path) -> bool:
    if path.is_symlink():
        return False
    try:
        return path.resolve(strict=True).is_relative_to(root)
    except OSError:
        return False


def should_scan_file(path: Path, include_lockfiles: bool) -> bool:
    if path.name in LOCKFILE_NAMES and not include_lockfiles:
        return False
    return path.name in TEXT_FILENAMES or path.suffix in TEXT_SUFFIXES


def supports_secure_mutation() -> bool:
    """Return whether this platform can anchor mutations to directory handles."""
    return (
        hasattr(os, "O_DIRECTORY")
        and hasattr(os, "O_NOFOLLOW")
        and hasattr(os, "fchmod")
        and os.open in os.supports_dir_fd
        and os.stat in os.supports_dir_fd
        and os.stat in os.supports_follow_symlinks
        and os.rename in os.supports_dir_fd
        and os.unlink in os.supports_dir_fd
    )


def supports_atomic_no_replace_rename() -> bool:
    libc = ctypes.CDLL(None)
    return (sys.platform.startswith("linux") and hasattr(libc, "renameat2")) or (
        sys.platform == "darwin" and hasattr(libc, "renameatx_np")
    )


def rename_no_replace_at(parent_fd: int, old_name: str, new_name: str) -> None:
    """Rename one directory entry without replacing an existing destination."""
    libc = ctypes.CDLL(None, use_errno=True)
    old_bytes = os.fsencode(old_name)
    new_bytes = os.fsencode(new_name)

    if sys.platform.startswith("linux") and hasattr(libc, "renameat2"):
        rename = libc.renameat2
        rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
        rename.restype = ctypes.c_int
        result = rename(parent_fd, old_bytes, parent_fd, new_bytes, LINUX_RENAME_NOREPLACE)
    elif sys.platform == "darwin" and hasattr(libc, "renameatx_np"):
        rename = libc.renameatx_np
        rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
        rename.restype = ctypes.c_int
        result = rename(
            parent_fd,
            old_bytes,
            parent_fd,
            new_bytes,
            DARWIN_RENAME_EXCL | DARWIN_RENAME_NOFOLLOW_ANY,
        )
    else:
        raise MutationError("atomic no-replace rename is unavailable on this platform")

    if result != 0:
        error_number = ctypes.get_errno()
        raise OSError(error_number, os.strerror(error_number), new_name)


def directory_open_flags() -> int:
    return os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW


def open_root_fd(root: Path) -> int:
    """Open an absolute root one component at a time without following links."""
    anchor = Path(root.anchor)
    directory_fd = os.open(anchor, directory_open_flags())
    try:
        for part in root.relative_to(anchor).parts:
            next_fd = os.open(part, directory_open_flags(), dir_fd=directory_fd)
            os.close(directory_fd)
            directory_fd = next_fd
    except Exception:
        os.close(directory_fd)
        raise
    return directory_fd


@contextmanager
def open_relative_directory(root_fd: int, relative_path: Path) -> Iterator[int]:
    """Open a directory below root_fd without resolving path components."""
    directory_fd = os.dup(root_fd)
    try:
        for part in relative_path.parts:
            if part in {"", "."}:
                continue
            if part == "..":
                raise ValueError("relative path escapes the confirmed root")
            next_fd = os.open(part, directory_open_flags(), dir_fd=directory_fd)
            os.close(directory_fd)
            directory_fd = next_fd
        yield directory_fd
    finally:
        os.close(directory_fd)


def same_file_version(left: os.stat_result, right: os.stat_result) -> bool:
    return (
        left.st_dev == right.st_dev
        and left.st_ino == right.st_ino
        and left.st_size == right.st_size
        and left.st_mtime_ns == right.st_mtime_ns
        and left.st_ctime_ns == right.st_ctime_ns
    )


def replace_file_at(parent_fd: int, name: str, data: bytes, source_stat: os.stat_result) -> None:
    """Atomically replace name inside parent_fd if the scanned file is unchanged."""
    current_stat = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    if not stat.S_ISREG(current_stat.st_mode) or not same_file_version(source_stat, current_stat):
        raise MutationError("path changed during scan")

    temp_name = f".{name}.nemo-relay-{secrets.token_hex(8)}.tmp"
    temp_fd: int | None = None
    try:
        temp_fd = os.open(
            temp_name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            stat.S_IMODE(source_stat.st_mode),
            dir_fd=parent_fd,
        )
        os.fchmod(temp_fd, stat.S_IMODE(source_stat.st_mode))
        with os.fdopen(temp_fd, "wb", closefd=False) as temp_file:
            temp_file.write(data)
            temp_file.flush()
            os.fsync(temp_fd)
        os.close(temp_fd)
        temp_fd = None

        current_stat = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        if not stat.S_ISREG(current_stat.st_mode) or not same_file_version(source_stat, current_stat):
            raise MutationError("path changed during scan")
        os.rename(temp_name, name, src_dir_fd=parent_fd, dst_dir_fd=parent_fd)
    finally:
        if temp_fd is not None:
            os.close(temp_fd)
        try:
            os.unlink(temp_name, dir_fd=parent_fd)
        except FileNotFoundError:
            pass


def iter_files(root: Path, include_lockfiles: bool, include_generated: bool):
    for current_root, dirs, files in os.walk(root):
        current = Path(current_root)
        dirs[:] = [
            name
            for name in dirs
            if not should_skip_dir(name, include_generated) and is_safe_entry(current / name, root)
        ]
        for filename in files:
            path = current / filename
            if is_safe_entry(path, root) and path.is_file() and should_scan_file(path, include_lockfiles):
                yield path


def rewrite_file_secure(path: Path, root: Path, root_fd: int) -> FileChange | None:
    try:
        relative_path = path.relative_to(root)
        with open_relative_directory(root_fd, relative_path.parent) as parent_fd:
            file_fd = os.open(relative_path.name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=parent_fd)
            try:
                source_stat_before = os.fstat(file_fd)
                if not stat.S_ISREG(source_stat_before.st_mode):
                    raise MutationError("path is not a regular file")
                with os.fdopen(file_fd, "rb", closefd=False) as source_file:
                    raw = source_file.read()
                source_stat = os.fstat(file_fd)
                if not same_file_version(source_stat_before, source_stat):
                    raise MutationError("file changed while it was being read")

                if b"\0" in raw:
                    return None
                try:
                    text = raw.decode("utf-8")
                except UnicodeDecodeError:
                    return None

                updated, count = apply_replacements(text)
                if count == 0:
                    return None
                replace_file_at(parent_fd, relative_path.name, updated.encode("utf-8"), source_stat)
            finally:
                os.close(file_fd)
    except (OSError, RuntimeError, ValueError) as error:
        raise MutationError(f"path changed during scan: {path}: {error}") from error

    return FileChange(path=path, count=count)


def rewrite_file(path: Path, root: Path, write: bool, root_fd: int | None = None) -> FileChange | None:
    if write:
        if root_fd is None:
            raise MutationError("write mode requires a confirmed root directory handle")
        return rewrite_file_secure(path, root, root_fd)

    if not is_safe_entry(path, root) or not path.is_file():
        print(f"skip unsafe path: {path}")
        return None

    try:
        raw = path.read_bytes()
    except OSError as error:
        print(f"skip unreadable: {path}: {error}")
        return None

    if b"\0" in raw:
        return None

    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return None

    updated, count = apply_replacements(text)
    if count == 0:
        return None

    return FileChange(path=path, count=count)


def updated_name(name: str) -> str:
    updated, _ = apply_replacements(name)
    return updated


def collect_path_changes(root: Path, include_generated: bool) -> list[PathChange]:
    changes: list[PathChange] = []
    paths: list[Path] = []
    for current_root, dirs, files in os.walk(root):
        current = Path(current_root)
        dirs[:] = [
            name
            for name in dirs
            if not should_skip_dir(name, include_generated) and is_safe_entry(current / name, root)
        ]
        for name in [*files, *dirs]:
            path = current / name
            if not is_safe_entry(path, root):
                continue
            if path.is_file() and not should_scan_file(path, include_lockfiles=False):
                continue
            paths.append(path)

    for old in sorted(paths, key=lambda path: len(path.parts), reverse=True):
        new_name = updated_name(old.name)
        if new_name != old.name:
            changes.append(PathChange(old=old, new=old.with_name(new_name)))
    return changes


def apply_path_changes(
    changes: list[PathChange],
    root: Path,
    write: bool,
    root_fd: int | None = None,
) -> list[PathChange]:
    applied: list[PathChange] = []
    for change in changes:
        if write:
            if root_fd is None:
                raise MutationError("write mode requires a confirmed root directory handle")
            try:
                old_relative = change.old.relative_to(root)
                new_relative = change.new.relative_to(root)
                if old_relative.parent != new_relative.parent:
                    raise ValueError("rename destination must use the source directory")
                with open_relative_directory(root_fd, old_relative.parent) as parent_fd:
                    source_stat = os.stat(old_relative.name, dir_fd=parent_fd, follow_symlinks=False)
                    if not (stat.S_ISREG(source_stat.st_mode) or stat.S_ISDIR(source_stat.st_mode)):
                        raise ValueError("rename source is not a regular file or directory")
                    rename_no_replace_at(parent_fd, old_relative.name, new_relative.name)
            except (OSError, RuntimeError, ValueError) as error:
                raise MutationError(f"unsafe rename: {change.old} -> {change.new}: {error}") from error
            applied.append(change)
            continue

        if not is_safe_entry(change.old, root):
            print(f"skip unsafe rename source: {change.old}")
            continue
        try:
            destination_is_safe = change.new.parent.resolve(strict=True).is_relative_to(root)
        except OSError:
            destination_is_safe = False
        if not destination_is_safe:
            print(f"skip unsafe rename destination: {change.old} -> {change.new}")
            continue
        if change.new.exists() or change.new.is_symlink():
            print(f"skip rename conflict: {change.old} -> {change.new}")
            continue
        applied.append(change)
    return applied


def print_report(
    file_changes: list[FileChange],
    path_changes: list[PathChange],
    write: bool,
    max_report: int,
) -> None:
    mode = "updated" if write else "would update"
    rename_mode = "renamed" if write else "would rename"

    for change in file_changes[:max_report]:
        print(f"{mode}: {change.path} ({change.count} replacements)")
    if len(file_changes) > max_report:
        print(f"... {len(file_changes) - max_report} more file changes omitted")

    for change in path_changes[:max_report]:
        print(f"{rename_mode}: {change.old} -> {change.new}")
    if len(path_changes) > max_report:
        print(f"... {len(path_changes) - max_report} more path changes omitted")

    print(f"summary: {len(file_changes)} files {mode}; {len(path_changes)} paths {rename_mode}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Rewrite explicit NeMo Flow names to NeMo Relay names.",
    )
    parser.add_argument(
        "root",
        nargs="?",
        default=".",
        help="Repository or project root to scan.",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Apply changes. Without this flag, only report planned changes.",
    )
    parser.add_argument(
        "--confirm-root",
        metavar="PATH",
        help="Required with --write; must resolve to the same root being changed.",
    )
    parser.add_argument(
        "--rename-paths",
        action="store_true",
        help="Also report or apply file and directory renames.",
    )
    parser.add_argument(
        "--include-lockfiles",
        action="store_true",
        help="Rewrite lockfiles directly instead of leaving them for package tools.",
    )
    parser.add_argument(
        "--include-generated",
        action="store_true",
        help="Scan generated directories named _generated.",
    )
    parser.add_argument(
        "--max-report",
        type=int,
        default=200,
        help="Maximum file changes and path changes to print.",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    if not root.exists():
        parser.error(f"root does not exist: {root}")
    if not root.is_dir():
        parser.error(f"root must be a directory: {root}")
    if args.write:
        if args.confirm_root is None:
            parser.error("--write requires --confirm-root with the reviewed root")
        try:
            confirmed_root = Path(args.confirm_root).resolve(strict=True)
        except OSError as error:
            parser.error(f"confirmed root cannot be resolved: {error}")
        if confirmed_root != root:
            parser.error(f"confirmed root does not match scan root: {confirmed_root} != {root}")
        filesystem_root = Path(root.anchor).resolve()
        if root == filesystem_root or root == Path.home().resolve():
            parser.error("refusing write mode for a filesystem root or home directory")
        if not supports_secure_mutation():
            parser.error("--write requires directory-fd and no-follow support on this platform")
        if args.rename_paths and not supports_atomic_no_replace_rename():
            parser.error("--write --rename-paths requires atomic no-replace rename support on this platform")

    root_fd: int | None = None
    try:
        if args.write:
            try:
                root_fd = open_root_fd(root)
            except OSError as error:
                parser.error(f"confirmed root cannot be opened safely: {error}")

        file_changes = [
            change
            for path in iter_files(root, args.include_lockfiles, args.include_generated)
            if (change := rewrite_file(path, root, args.write, root_fd)) is not None
        ]

        path_changes: list[PathChange] = []
        if args.rename_paths:
            path_changes = apply_path_changes(
                collect_path_changes(root, args.include_generated),
                root,
                args.write,
                root_fd,
            )
    except MutationError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    finally:
        if root_fd is not None:
            os.close(root_fd)

    print_report(file_changes, path_changes, args.write, args.max_report)
    if not args.write:
        print("dry run only; pass --write to apply changes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
