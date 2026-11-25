# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Locate skills by layer in either layout of the TAO skill bank.

The repository keeps every skill in a layer folder, ``skills/<layer>/<skill>/``,
and the layer is read off the path. The plugin package that
``scripts/build_plugin_package.py`` generates is flat, ``skills/<skill>/``, so it
carries the layer of every skill in ``catalog/skill-layers.json`` and keeps the
files that sat directly in a layer folder under ``catalog/`` as well.

Helper scripts ask this module where a skill or a layer-level file is instead of
joining a layer folder themselves, so the same script runs from a clone and from
the package. A layer folder that exists always wins; the membership file is read
only when the folder is absent.
"""

from __future__ import annotations

import json
from pathlib import Path

LAYERS = ("applications", "core", "data", "models", "platform")
CATALOG_DIR = "catalog"
MEMBERSHIP_FILE = "skill-layers.json"


def membership_path(skill_bank: Path) -> Path:
    """Path of the skill-to-layer file a flat package ships."""
    return Path(skill_bank).expanduser() / CATALOG_DIR / MEMBERSHIP_FILE


def load_membership(skill_bank: Path) -> dict[str, list[str]] | None:
    """Return ``{layer: [skill, ...]}`` from the membership file, or None without one."""
    path = membership_path(skill_bank)
    if not path.is_file():
        return None
    layers = json.loads(path.read_text(encoding="utf-8")).get("layers")
    if not isinstance(layers, dict):
        raise ValueError(f"{path} is missing a layers object")
    return {str(layer): [str(name) for name in names] for layer, names in layers.items()}


def _layer_folder(skill_bank: Path, layer: str) -> Path:
    return Path(skill_bank).expanduser() / "skills" / layer


def _flat_members(skill_bank: Path, layer: str) -> list[str] | None:
    """Skills of ``layer`` in a flat bank, or None when the bank has layer folders."""
    if _layer_folder(skill_bank, layer).is_dir():
        return None
    membership = load_membership(skill_bank)
    return None if membership is None else membership.get(layer, [])


def layer_root(skill_bank: Path, layer: str) -> Path:
    """Folder that holds the skills of ``layer``: its layer folder, or ``skills/`` when flat."""
    if _flat_members(skill_bank, layer) is None:
        return _layer_folder(skill_bank, layer)
    return Path(skill_bank).expanduser() / "skills"


def skill_dir(skill_bank: Path, layer: str, name: str) -> Path:
    """Directory of the skill ``name`` in ``layer``; the caller checks that it exists.

    In a flat bank a skill of another layer gets the layer-folder path, which is
    absent there, so asking for a model by an application's name finds nothing in
    either layout.
    """
    members = _flat_members(skill_bank, layer)
    if members is not None and name in members:
        return Path(skill_bank).expanduser() / "skills" / name
    return _layer_folder(skill_bank, layer) / name


def layer_skill_dirs(skill_bank: Path, layer: str) -> list[Path]:
    """Existing skill directories of ``layer``, sorted by name."""
    members = _flat_members(skill_bank, layer)
    if members is None:
        folder = _layer_folder(skill_bank, layer)
        if not folder.is_dir():
            return []
        return sorted(path for path in folder.iterdir() if path.is_dir())
    root = Path(skill_bank).expanduser() / "skills"
    return [root / name for name in sorted(members) if (root / name).is_dir()]


def layer_file(skill_bank: Path, layer: str, name: str) -> Path:
    """Path of a file that sits directly in a layer folder, such as a layer manifest."""
    if _flat_members(skill_bank, layer) is None:
        return _layer_folder(skill_bank, layer) / name
    return Path(skill_bank).expanduser() / CATALOG_DIR / name
