#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Annotation-driven selective download for TAO host-side staging.

Given an annotation file (parquet / jsonl / json / csv) that references data
files by relative path in one or more columns, resolve the *exact* set of
referenced files and download only those from an S3 source prefix into a local
staging directory.

This replaces the in-container selective-download logic of the former
``tao_sdk.script_runner`` — host-side, using ``boto3`` directly. There is no
``tao_sdk`` import and no ``fsspec``/``s3fs`` dependency.

Correctness note: a mis-parse here silently drops training samples, so key
extraction is deterministic and exhaustively unit-tested
(``tests/test_selective_download.py``). Credentials are read from the process
environment (``AWS_ACCESS_KEY_ID`` / ``AWS_SECRET_ACCESS_KEY`` /
``AWS_ENDPOINT_URL`` / ``AWS_DEFAULT_REGION``); this module never runs
``aws configure`` and never writes ``~/.aws``.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Sequence

import pandas as pd

_FORMAT_BY_SUFFIX = {
    ".parquet": "parquet",
    ".pq": "parquet",
    ".jsonl": "jsonl",
    ".ndjson": "jsonl",
    ".json": "json",
    ".csv": "csv",
}

_SUPPORTED_FORMATS = frozenset({"parquet", "jsonl", "json", "csv"})


def infer_format(annotation_path: str | os.PathLike) -> str:
    """Infer the annotation format from the file extension.

    Raises ValueError if the extension is not recognized.
    """
    suffix = Path(annotation_path).suffix.lower()
    fmt = _FORMAT_BY_SUFFIX.get(suffix)
    if fmt is None:
        raise ValueError(
            f"cannot infer annotation format from '{annotation_path}'; "
            f"pass format= explicitly (one of {sorted(_SUPPORTED_FORMATS)})"
        )
    return fmt


def _load_frame(annotation_path: str | os.PathLike, fmt: str) -> pd.DataFrame:
    if fmt not in _SUPPORTED_FORMATS:
        raise ValueError(
            f"unsupported format '{fmt}'; expected one of {sorted(_SUPPORTED_FORMATS)}"
        )
    if fmt == "parquet":
        return pd.read_parquet(annotation_path)
    if fmt == "csv":
        return pd.read_csv(annotation_path)
    if fmt == "jsonl":
        return pd.read_json(annotation_path, lines=True)
    # "json": a JSON array of records
    return pd.read_json(annotation_path)


def extract_keys(
    annotation_path: str | os.PathLike,
    keys: Sequence[str],
    fmt: str | None = None,
) -> list[str]:
    """Return the deduplicated, order-preserving list of referenced paths.

    ``keys`` names the column(s) that hold relative file paths (e.g. ["video"]
    or ["image", "mask"]). Cells may be scalars or lists; nulls are dropped.
    A key not present in the annotation is a hard error (a silent
    train-on-wrong-data risk), listing the columns that *are* present.
    """
    if isinstance(keys, str):
        keys = [keys]
    if not keys:
        raise ValueError("keys must name at least one annotation column")

    fmt = fmt or infer_format(annotation_path)
    df = _load_frame(annotation_path, fmt)

    missing = [k for k in keys if k not in df.columns]
    if missing:
        raise KeyError(
            f"annotation column(s) {missing} not found; "
            f"available columns: {list(df.columns)}"
        )

    acc: list[str] = []
    for key in keys:
        for value in df[key].tolist():
            if isinstance(value, (list, tuple)):
                for item in value:
                    if item is None:
                        continue
                    acc.append(str(item))
            else:
                # scalar: pd.isna handles None / NaN safely
                if pd.isna(value):
                    continue
                acc.append(str(value))

    # dedup, preserving first-seen order
    return list(dict.fromkeys(acc))


def download_selective(
    s3_client,
    bucket: str,
    src_prefix: str,
    rel_keys: Sequence[str],
    dest_dir: str | os.PathLike,
    *,
    skip_existing: bool = True,
) -> list[str]:
    """Download exactly ``rel_keys`` from ``s3://bucket/src_prefix/`` into dest_dir.

    Preserves each file's relative path under dest_dir, creating parent dirs.
    Idempotent: an existing non-empty local file is skipped. Failures are
    collected and raised as a single summary at the end (fail loud, never a
    silent partial dataset).
    """
    dest_root = Path(dest_dir)
    src_prefix = src_prefix.strip("/")
    results: list[str] = []
    errors: list[tuple[str, str]] = []

    for rel in rel_keys:
        rel_norm = str(rel).lstrip("/")
        s3_key = f"{src_prefix}/{rel_norm}" if src_prefix else rel_norm
        local = dest_root / rel_norm

        if skip_existing and local.exists() and local.stat().st_size > 0:
            results.append(str(local))
            continue

        local.parent.mkdir(parents=True, exist_ok=True)
        try:
            s3_client.download_file(bucket, s3_key, str(local))
            results.append(str(local))
        except Exception as exc:  # noqa: BLE001 - reported in the summary below
            errors.append((s3_key, str(exc)))

    if errors:
        preview = "; ".join(f"{k} ({e})" for k, e in errors[:5])
        more = "" if len(errors) <= 5 else f" (+{len(errors) - 5} more)"
        raise RuntimeError(
            f"{len(errors)} of {len(rel_keys)} objects failed to download: {preview}{more}"
        )
    return results


def _make_s3_client(endpoint_url: str | None = None):
    import boto3  # local import so parsing/testing needs no boto3

    return boto3.client(
        "s3",
        endpoint_url=endpoint_url or os.environ.get("AWS_ENDPOINT_URL") or None,
        region_name=os.environ.get("AWS_DEFAULT_REGION") or None,
    )


def _parse_keys(raw: list[str]) -> list[str]:
    out: list[str] = []
    for item in raw:
        out.extend(part for part in item.split(",") if part)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--annotation", required=True, help="path to the annotation file")
    parser.add_argument("--key", action="append", default=[], required=True,
                        help="annotation column holding file paths (repeatable or comma-separated)")
    parser.add_argument("--format", dest="fmt", default=None, choices=sorted(_SUPPORTED_FORMATS),
                        help="override format (inferred from extension by default)")
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--src-prefix", default="", help="source key prefix within the bucket")
    parser.add_argument("--dest", required=True, help="local staging directory")
    parser.add_argument("--endpoint-url", default=None, help="S3 endpoint (else $AWS_ENDPOINT_URL)")
    parser.add_argument("--no-skip-existing", action="store_true", help="re-download files already present")
    args = parser.parse_args(argv)

    rel_keys = extract_keys(args.annotation, _parse_keys(args.key), fmt=args.fmt)
    print(f"resolved {len(rel_keys)} referenced files from {args.annotation}", file=sys.stderr)

    client = _make_s3_client(args.endpoint_url)
    downloaded = download_selective(
        client, args.bucket, args.src_prefix, rel_keys, args.dest,
        skip_existing=not args.no_skip_existing,
    )
    print(f"staged {len(downloaded)} files into {args.dest}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
