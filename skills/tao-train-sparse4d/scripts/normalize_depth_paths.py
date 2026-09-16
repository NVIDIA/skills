#!/usr/bin/env python3
"""Normalize Sparse4D OVPKL depth H5 references after dataset conversion."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Iterable


def _iter_ann_files(paths: Iterable[Path]) -> list[Path]:
    ann_files: list[Path] = []
    for path in paths:
        if path.is_dir():
            ann_files.extend(sorted(path.rglob("*.pkl")))
        else:
            ann_files.append(path)
    return ann_files


def _normalize_depth_path(value: object, data_root: Path) -> tuple[object, bool]:
    if not (
        isinstance(value, tuple)
        and len(value) == 2
        and isinstance(value[0], str)
        and isinstance(value[1], str)
    ):
        return value, False

    h5_rel, depth_key = value
    if h5_rel.endswith(".h5") or "/depth_maps/" in h5_rel:
        return value, False

    camera = Path(h5_rel).name
    scene = str(Path(h5_rel).parent)
    candidate_rel = f"{scene}/depth_maps/{camera}.h5"
    candidate_host = data_root / candidate_rel
    if not candidate_host.exists():
        return value, False

    normalized_key = Path(depth_key).name
    return (candidate_rel, normalized_key), True


def normalize_ann_file(path: Path, data_root: Path, dry_run: bool) -> int:
    with path.open("rb") as handle:
        payload = pickle.load(handle)

    infos = payload.get("infos") if isinstance(payload, dict) else payload
    if not isinstance(infos, list):
        raise ValueError(f"{path}: expected a list of infos or a dict with infos")

    changed = 0
    for info in infos:
        cams = info.get("cams", {}) if isinstance(info, dict) else {}
        for cam_info in cams.values():
            if not isinstance(cam_info, dict) or "depth_map_path" not in cam_info:
                continue
            normalized, did_change = _normalize_depth_path(cam_info["depth_map_path"], data_root)
            if did_change:
                cam_info["depth_map_path"] = normalized
                changed += 1

    if changed and not dry_run:
        with path.open("wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return changed


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Rewrite Sparse4D converted OVPKL depth_map_path tuples from "
            "scene/camera directories to scene/depth_maps/<camera>.h5 files."
        )
    )
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="Annotation .pkl files or directories containing converted .pkl files.",
    )
    parser.add_argument(
        "--data-root",
        required=True,
        type=Path,
        help="Host path that is mounted as dataset.data_root, for example aicity_root/train.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Report changes without writing files.")
    args = parser.parse_args()

    total = 0
    for ann_file in _iter_ann_files(args.paths):
        changed = normalize_ann_file(ann_file, args.data_root, args.dry_run)
        total += changed
        print(f"{ann_file}: {changed} depth path(s) normalized")
    print(f"total_depth_paths_normalized={total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
