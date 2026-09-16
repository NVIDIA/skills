#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Create and validate a TAO Data Services TMM nearest-neighbor spec.

The generated YAML is intentionally close to the upstream TAO Data Services
`nearest_neighbors.yaml` schema. It fills the required parquet paths and leaves
the mining knobs explicit for reproducibility.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover - exercised only on hosts without PyYAML.
    yaml = None


VALID_METRICS = {"euclidean", "cosine", "manhattan"}


def load_yaml(path: Path) -> dict[str, Any]:
    """Read a YAML mapping from `path`."""
    if yaml is None:
        raise RuntimeError("PyYAML is required: install with `python3 -m pip install pyyaml`.")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping.")
    return data


def write_yaml(path: Path, data: dict[str, Any]) -> None:
    """Write `data` as a stable block-style YAML mapping."""
    if yaml is None:
        raise RuntimeError("PyYAML is required: install with `python3 -m pip install pyyaml`.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)


def absolute_path(raw_path: str, *, must_exist: bool, kind: str) -> str:
    """Expand a path and return an absolute string, optionally requiring it exists."""
    path = Path(raw_path).expanduser().resolve()
    if must_exist and not path.exists():
        raise FileNotFoundError(f"{kind} does not exist: {path}")
    return str(path)


def validate_config(config: dict[str, Any]) -> None:
    """Validate the fields this skill depends on before Docker is launched."""
    required = ("source_parquet", "target_parquet", "output_parquet")
    missing = [
        key
        for key in required
        if not isinstance(config.get(key), str) or not config[key].strip()
    ]
    if missing:
        raise ValueError(f"Missing required field(s): {', '.join(missing)}")

    for key in ("source_parquet", "target_parquet"):
        path = Path(str(config[key])).expanduser()
        if not path.is_absolute():
            raise ValueError(f"{key} must be absolute: {path}")
        if not path.is_file():
            raise FileNotFoundError(f"{key} does not exist or is not a file: {path}")

    output_path = Path(str(config["output_parquet"])).expanduser()
    if not output_path.is_absolute():
        raise ValueError(f"output_parquet must be absolute: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not output_path.parent.is_dir():
        raise NotADirectoryError(f"output_parquet parent is not a directory: {output_path.parent}")
    try:
        probe = output_path.parent / ".tao_mine_nearest_neighbors_write_test"
        probe.write_text("ok\n", encoding="utf-8")
        probe.unlink()
    except OSError as exc:
        raise PermissionError(f"output_parquet parent is not writable: {output_path.parent}") from exc

    topn = config.get("topn", 5)
    if isinstance(topn, bool) or not isinstance(topn, int) or topn < 1:
        raise ValueError("topn must be an integer of at least 1.")
    config["topn"] = topn

    metric = config.get("knn_metric", "cosine")
    if not isinstance(metric, str) or metric not in VALID_METRICS:
        raise ValueError(f"knn_metric must be one of {sorted(VALID_METRICS)}.")
    config["knn_metric"] = metric

    filter_by_label = config.get("filter_by_label", "false")
    if not isinstance(filter_by_label, str):
        raise ValueError('filter_by_label must be the string "true" or "false".')
    filter_by_label = filter_by_label.lower()
    if filter_by_label not in {"true", "false"}:
        raise ValueError('filter_by_label must be the string "true" or "false".')
    config["filter_by_label"] = filter_by_label

    for key in ("source_embed_column_name", "target_embed_column_name"):
        value = config.get(key, "embedding")
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{key} must be a non-empty string.")
        config[key] = value

    distance_threshold = config.get("distance_threshold", -1.0)
    if (
        isinstance(distance_threshold, bool)
        or not isinstance(distance_threshold, (int, float))
        or not math.isfinite(distance_threshold)
    ):
        raise ValueError("distance_threshold must be a finite number.")
    config["distance_threshold"] = float(distance_threshold)


def build_config(args: argparse.Namespace) -> dict[str, Any]:
    """Merge a template with command-line values and validate the result."""
    skill_dir = Path(__file__).resolve().parents[1]
    default_template = skill_dir / "assets" / "default_nearest_neighbors.yaml"
    template = Path(args.template).expanduser().resolve() if args.template else default_template
    config = load_yaml(template)

    config.update(
        {
            "source_parquet": absolute_path(args.source_parquet, must_exist=True, kind="source_parquet"),
            "target_parquet": absolute_path(args.target_parquet, must_exist=True, kind="target_parquet"),
            "output_parquet": absolute_path(args.output_parquet, must_exist=False, kind="output_parquet"),
            "topn": args.topn,
            "knn_metric": args.knn_metric,
            "source_embed_column_name": args.source_embed_column_name,
            "target_embed_column_name": args.target_embed_column_name,
            "filter_by_label": args.filter_by_label,
            "distance_threshold": args.distance_threshold,
        }
    )
    validate_config(config)
    return config


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-parquet", required=True, help="Absolute or expandable path to source embeddings parquet.")
    parser.add_argument("--target-parquet", required=True, help="Absolute or expandable path to target embeddings parquet.")
    parser.add_argument("--output-parquet", required=True, help="Absolute or expandable path for mined output parquet.")
    parser.add_argument("--output-spec", required=True, help="Path where the generated nearest-neighbors YAML should be written.")
    parser.add_argument("--template", help="Optional YAML template. Defaults to assets/default_nearest_neighbors.yaml.")
    parser.add_argument("--topn", type=int, default=5, help="Number of neighbors to mine per target.")
    parser.add_argument("--knn-metric", default="cosine", choices=sorted(VALID_METRICS), help="cuML nearest-neighbor metric.")
    parser.add_argument("--source-embed-column-name", default="embedding", help="Embedding column in source parquet.")
    parser.add_argument("--target-embed-column-name", default="embedding", help="Embedding column in target parquet.")
    parser.add_argument("--filter-by-label", default="false", choices=["true", "false"], help="String flag consumed by TAO DS.")
    parser.add_argument("--distance-threshold", type=float, default=-1.0, help="Maximum distance; negative disables thresholding.")
    return parser.parse_args()


def main() -> int:
    """CLI entrypoint."""
    try:
        args = parse_args()
        config = build_config(args)
        output_spec = Path(args.output_spec).expanduser().resolve()
        write_yaml(output_spec, config)
        summary_path = Path(str(config["output_parquet"])).with_name("mining_summary.txt")
        print(f"Wrote nearest-neighbors spec: {output_spec}")
        print(f"Expected output parquet: {config['output_parquet']}")
        print(f"Expected summary: {summary_path}")
        return 0
    except Exception as exc:  # noqa: BLE001 - CLI should print concise failure.
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
