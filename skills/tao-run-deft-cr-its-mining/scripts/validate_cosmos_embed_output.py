#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate one completed Cosmos Embed inference job."""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from workflow_common import absolute_path, atomic_write_json, load_yaml


ACCEPTED_EXIT_CODES = {0, 130}
SUCCESS_STATUSES = {"ok", "ok_with_teardown_warning"}
COMPLETION_FILENAME = "completion_validation.json"


def utc_now() -> str:
    """Return the current UTC time in ISO-8601 format."""
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def file_sha256(path: Path) -> str:
    """Return the SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_string(value: Any, label: str) -> str:
    """Require a non-empty string value."""
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")
    return value


def spec_context(spec_path: Path) -> dict[str, Any]:
    """Load one inference spec and derive its expected output contract."""
    spec = load_yaml(spec_path)
    inference = spec.get("inference")
    if not isinstance(inference, dict):
        raise ValueError(f"{spec_path}: missing object field 'inference'")
    mode = require_string(inference.get("mode"), f"{spec_path}: inference.mode")
    if mode not in {"text", "video"}:
        raise ValueError(f"{spec_path}: inference.mode must be 'text' or 'video', got {mode!r}")
    query = inference.get("query")
    if not isinstance(query, dict):
        raise ValueError(f"{spec_path}: missing object field 'inference.query'")
    input_key = "input_texts" if mode == "text" else "input_videos"
    expected = query.get(input_key)
    if not isinstance(expected, list) or not expected:
        raise ValueError(f"{spec_path}: inference.query.{input_key} must be a non-empty list")
    for index, value in enumerate(expected, start=1):
        require_string(value, f"{spec_path}: inference.query.{input_key}[{index}]")

    results_dir = Path(require_string(spec.get("results_dir"), f"{spec_path}: results_dir")).expanduser()
    if not results_dir.is_absolute():
        raise ValueError(f"{spec_path}: results_dir must be absolute: {results_dir}")
    inference_dir = results_dir / "inference"
    return {
        "spec_path": spec_path,
        "spec_sha256": file_sha256(spec_path),
        "mode": mode,
        "expected": expected,
        "inference_dir": inference_dir,
        "metadata_path": inference_dir / f"{mode}_embeddings.json",
        "completion_path": inference_dir / COMPLETION_FILENAME,
    }


def load_json_object(path: Path, label: str) -> dict[str, Any]:
    """Read a JSON object from a file."""
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object: {path}")
    return payload


def referenced_npy_path(inference_dir: Path, metadata: dict[str, Any]) -> Path:
    """Resolve the NPY array path declared by Cosmos Embed metadata."""
    value = require_string(metadata.get("npy_file"), "Cosmos Embed metadata.npy_file")
    path = Path(value).expanduser()
    return path if path.is_absolute() else inference_dir / path


def validate_outputs(spec_path: Path) -> dict[str, Any]:
    """Validate JSON/NPY outputs against every query in an inference spec."""
    context = spec_context(spec_path)
    metadata_path = context["metadata_path"]
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Cosmos Embed metadata is missing: {metadata_path}")
    metadata = load_json_object(metadata_path, "Cosmos Embed metadata")
    npy_path = referenced_npy_path(context["inference_dir"], metadata)
    if not npy_path.is_file():
        raise FileNotFoundError(f"Cosmos Embed NPY output is missing: {npy_path}")

    embeddings = np.load(npy_path, allow_pickle=False)
    expected = context["expected"]
    if embeddings.ndim != 2:
        raise ValueError(f"Cosmos Embed array must be two-dimensional, got shape {embeddings.shape}")
    if embeddings.shape[0] != len(expected) or embeddings.shape[1] < 1:
        raise ValueError(
            f"Cosmos Embed array shape {embeddings.shape} does not match "
            f"{len(expected)} expected {context['mode']} queries"
        )
    try:
        finite = bool(np.isfinite(embeddings).all())
    except TypeError as exc:
        raise ValueError(f"Cosmos Embed array must contain numeric values: {npy_path}") from exc
    if not finite:
        raise ValueError(f"Cosmos Embed array contains non-finite values: {npy_path}")

    results = metadata.get("results")
    if not isinstance(results, list) or len(results) != len(expected):
        observed = len(results) if isinstance(results, list) else "non-list"
        raise ValueError(f"Cosmos Embed metadata has {observed} results; expected {len(expected)}")
    identifier_key = "text" if context["mode"] == "text" else "video_path"
    identifiers: list[str] = []
    npy_rows: list[int] = []
    for index, result in enumerate(results, start=1):
        if not isinstance(result, dict):
            raise ValueError(f"Cosmos Embed result {index} is not an object")
        identifiers.append(require_string(result.get(identifier_key), f"result {index}.{identifier_key}"))
        npy_row = result.get("npy_row")
        if not isinstance(npy_row, int) or isinstance(npy_row, bool):
            raise ValueError(f"Cosmos Embed result {index}.npy_row must be an integer")
        npy_rows.append(npy_row)
    if sorted(npy_rows) != list(range(len(expected))):
        raise ValueError(
            "Cosmos Embed npy_row values must reference every embedding row exactly once; "
            f"observed {npy_rows}"
        )
    if Counter(identifiers) != Counter(expected):
        missing = list((Counter(expected) - Counter(identifiers)).elements())[:3]
        unexpected = list((Counter(identifiers) - Counter(expected)).elements())[:3]
        raise ValueError(
            f"Cosmos Embed {context['mode']} identifiers do not match the spec; "
            f"missing={missing!r}, unexpected={unexpected!r}"
        )

    return {
        "mode": context["mode"],
        "expected_count": len(expected),
        "observed_count": len(results),
        "embedding_shape": list(embeddings.shape),
        "metadata_path": str(metadata_path),
        "npy_path": str(npy_path),
    }


def validate_completion(spec_path: Path, exit_code: int) -> Path:
    """Validate a terminal job and write its reusable completion artifact."""
    if exit_code not in ACCEPTED_EXIT_CODES:
        raise RuntimeError(
            f"Cosmos Embed exited {exit_code}; only exit 0 or the validated teardown code 130 is accepted"
        )
    context = spec_context(spec_path)
    output = validate_outputs(spec_path)
    warning = exit_code == 130
    completion = {
        "schema_version": 1,
        "status": "ok_with_teardown_warning" if warning else "ok",
        "validated_at": utc_now(),
        "exit_code": exit_code,
        "accepted_exit_code_130": warning,
        "spec_path": str(spec_path),
        "spec_sha256": context["spec_sha256"],
        **output,
    }
    atomic_write_json(context["completion_path"], completion)
    if warning:
        print(
            "WARNING: accepted Cosmos Embed exit 130 only because all outputs matched "
            "the inference spec"
        )
    print(f"validated Cosmos Embed completion: {context['completion_path']}")
    return context["completion_path"]


def check_completion(spec_path: Path) -> dict[str, Any]:
    """Require a current completion artifact and revalidate its outputs."""
    context = spec_context(spec_path)
    completion_path = context["completion_path"]
    if not completion_path.is_file():
        raise FileNotFoundError(f"Cosmos Embed completion validation is missing: {completion_path}")
    completion = load_json_object(completion_path, "Cosmos Embed completion validation")
    if completion.get("status") not in SUCCESS_STATUSES:
        raise ValueError(f"Cosmos Embed completion is not successful: {completion_path}")
    if completion.get("exit_code") not in ACCEPTED_EXIT_CODES:
        raise ValueError(f"Cosmos Embed completion has an invalid exit code: {completion_path}")
    if completion.get("spec_sha256") != context["spec_sha256"]:
        raise ValueError(f"Cosmos Embed completion does not match the current spec: {spec_path}")
    current = validate_outputs(spec_path)
    for key in ("mode", "expected_count", "observed_count", "embedding_shape", "metadata_path", "npy_path"):
        if completion.get(key) != current[key]:
            raise ValueError(f"Cosmos Embed completion field {key!r} is stale: {completion_path}")
    return completion


def incomplete_specs(run_dir: Path) -> list[str]:
    """Return generated inference specs without a current validated completion."""
    incomplete: list[str] = []
    for spec_path in sorted((run_dir / "cosmos_embed_output").glob("*/specs/inference_*.yaml")):
        try:
            check_completion(spec_path)
        except (FileNotFoundError, OSError, ValueError, RuntimeError) as exc:
            incomplete.append(f"{spec_path}: {exc}")
    return incomplete


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inference-spec", required=True, type=Path)
    parser.add_argument("--exit-code", required=True, type=int)
    return parser.parse_args()


def main() -> int:
    """Validate one completed Cosmos Embed inference job."""
    args = parse_args()
    spec_path = absolute_path(args.inference_spec)
    if not spec_path.is_file():
        raise FileNotFoundError(f"inference spec does not exist: {spec_path}")
    validate_completion(spec_path, args.exit_code)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
