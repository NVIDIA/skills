# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Standard-library-only path validation shared by PAS host entry points."""

from __future__ import annotations

import os
import pathlib


def safe_absolute_path(
    path: pathlib.Path, name: str, *, require_exists: bool = False
) -> pathlib.Path:
    """Return one lexical absolute path after rejecting every symlink hop.

    Resolving first and validating later loses whether the caller supplied a
    symlink. Keep the lexical path, normalize only ``.``/``..``, and compare it
    with ``resolve(strict=False)`` so existing symlinks in any parent are
    rejected even when the final path has not been created yet.
    """
    expanded = path.expanduser()
    if not expanded.is_absolute():
        raise ValueError(f"{name} must be an absolute path: {path}")
    lexical = pathlib.Path(os.path.abspath(expanded))
    if lexical == pathlib.Path(lexical.anchor):
        raise ValueError(f"{name} must not be a filesystem root: {lexical}")
    if lexical.resolve(strict=False) != lexical:
        raise ValueError(f"{name} must not contain or traverse a symlink: {lexical}")
    if require_exists and not lexical.exists():
        raise ValueError(f"{name} does not exist: {lexical}")
    return lexical
