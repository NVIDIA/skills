#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Align mined AOI paths to source pairs and emit bare OK/NG ShareGPT JSON."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from collections import Counter
from typing import Any, Iterable

from validate_sharegpt import load_records, prompt_and_label, resolve_image


def _full_path_keys(path_text: str, media_root: pathlib.Path) -> set[str]:
    normalized = path_text.replace("\\", "/").rstrip("/")
    resolved = str(resolve_image(path_text, media_root))
    return {normalized, resolved}


def _basename(path_text: str) -> str:
    normalized = path_text.replace("\\", "/").rstrip("/")
    return pathlib.PurePosixPath(normalized).name


def _load_mined_paths(path: pathlib.Path, column: str) -> list[str]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ValueError("pyarrow is required to read the mined parquet") from exc
    table = pq.read_table(path, columns=[column])
    values = [
        str(value)
        for value in table.column(column).to_pylist()
        if value is not None and str(value).strip()
    ]
    if not values:
        raise ValueError(f"{path}: no mined paths in column {column!r}")
    return values


def _source_index(
    records: list[dict[str, Any]], media_root: pathlib.Path
) -> tuple[
    dict[str, list[tuple[int, dict[str, Any]]]],
    dict[str, list[tuple[int, dict[str, Any]]]],
]:
    by_path: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    by_name: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    for record_index, record in enumerate(records):
        images = record.get("images")
        if not isinstance(images, list) or len(images) != 2:
            raise ValueError(
                f"source record[{record_index}]: images must contain "
                "[AOI, golden_reference]"
            )
        prompt_and_label(record, context=f"source record[{record_index}]")
        target = str(images[0])
        for key in _full_path_keys(target, media_root):
            by_path.setdefault(key, []).append((record_index, record))
        by_name.setdefault(_basename(target), []).append((record_index, record))
    return by_path, by_name


def _candidates(
    keys: Iterable[str],
    index: dict[str, list[tuple[int, dict[str, Any]]]],
) -> dict[int, dict[str, Any]]:
    candidates: dict[int, dict[str, Any]] = {}
    for key in keys:
        for source_index, record in index.get(key, []):
            candidates[source_index] = record
    return candidates


def _match(
    mined_path: str,
    *,
    media_root: pathlib.Path,
    by_path: dict[str, list[tuple[int, dict[str, Any]]]],
    by_name: dict[str, list[tuple[int, dict[str, Any]]]],
) -> tuple[int, dict[str, Any], str]:
    exact = _candidates(_full_path_keys(mined_path, media_root), by_path)
    if len(exact) == 1:
        source_index, record = next(iter(exact.items()))
        return source_index, record, "exact"

    named = _candidates([_basename(mined_path)], by_name)
    if not exact and len(named) == 1:
        source_index, record = next(iter(named.items()))
        return source_index, record, "name"

    if exact or named:
        raise ValueError(
            f"ambiguous source match for mined path {mined_path!r}; "
            f"exact_candidate_indexes={sorted(exact)}; "
            f"basename_candidate_indexes={sorted(named)}"
        )
    raise ValueError(f"missing source match for mined path {mined_path!r}")


def _format_path(
    path_text: str,
    *,
    media_root: pathlib.Path,
    relative: bool,
) -> str:
    resolved = resolve_image(path_text, media_root)
    if not relative:
        return str(resolved)
    try:
        return resolved.relative_to(media_root).as_posix()
    except ValueError as exc:
        raise ValueError(
            f"cannot emit relative path outside media root: {resolved}"
        ) from exc


def emit_records(
    mined_paths: Iterable[str],
    source_records: list[dict[str, Any]],
    *,
    media_root: pathlib.Path,
    relative: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    mined_paths = list(mined_paths)
    by_path, by_name = _source_index(source_records, media_root)
    output: list[dict[str, Any]] = []
    matches: Counter[str] = Counter()
    seen_targets: set[str] = set()
    for mined_path in mined_paths:
        resolved_target = str(resolve_image(mined_path, media_root))
        if resolved_target in seen_targets:
            continue
        source_index, source, match_mode = _match(
            mined_path,
            media_root=media_root,
            by_path=by_path,
            by_name=by_name,
        )
        seen_targets.add(resolved_target)
        matches[match_mode] += 1
        prompt, label = prompt_and_label(
            source, context=f"source record[{source_index}]"
        )
        source_images = source["images"]
        record: dict[str, Any] = {
            "images": [
                _format_path(
                    mined_path, media_root=media_root, relative=relative
                ),
                _format_path(
                    str(source_images[1]),
                    media_root=media_root,
                    relative=relative,
                ),
            ],
            "conversations": [
                {"from": "human", "value": prompt},
                {"from": "gpt", "value": label},
            ],
        }
        if source.get("video_fps") is not None:
            record["video_fps"] = source["video_fps"]
        output.append(record)
    if not output:
        raise ValueError("no unique mined records were emitted")
    return output, {
        "mode": "bare_okng",
        "source_records": len(source_records),
        "output_records": len(output),
        "duplicates_skipped": len(mined_paths) - len(seen_targets),
        "match_modes": dict(sorted(matches.items())),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mined-parquet", required=True, type=pathlib.Path)
    parser.add_argument("--source-annotations", required=True, type=pathlib.Path)
    parser.add_argument("--media-root", required=True, type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    parser.add_argument("--summary", type=pathlib.Path)
    parser.add_argument("--filepath-column", default="filepath")
    parser.add_argument("--emit-relative", action="store_true")
    args = parser.parse_args(argv)
    try:
        mined_paths = _load_mined_paths(args.mined_parquet, args.filepath_column)
        records, summary = emit_records(
            mined_paths,
            load_records(args.source_annotations),
            media_root=args.media_root.expanduser().resolve(),
            relative=args.emit_relative,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(records, indent=2) + "\n")
        summary_path = args.summary or args.output.with_name(
            "emit_mined_summary.json"
        )
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"emit_mined_sharegpt: {exc}", file=sys.stderr)
        return 2
    print(
        f"emit_mined_sharegpt: wrote {len(records)} bare OK/NG records "
        f"to {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
