# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Drop samples selected by earlier mining iterations and commit an audit ledger.

The TAO nearest-neighbor task deduplicates candidates inside one invocation but
does not know about earlier DEFT iterations.  This host-side post-processing
step preserves the candidate order, keeps only the first occurrence of each
normalized filepath, drops paths already committed by a previous iteration,
and records enough metadata to verify or safely resume the selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import posixpath
import sys
import tempfile
from typing import Any


HISTORY_VERSION = 1
HISTORY_IDENTITY = "filepath"


def _normalize_filepath(value: Any) -> str:
    text = str(value or "").strip().replace("\\", "/")
    if not text:
        raise ValueError("every mined row requires a non-empty filepath")
    normalized = posixpath.normpath(text)
    if normalized in {"", "."}:
        raise ValueError(f"invalid mined filepath: {value!r}")
    return normalized


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: pathlib.Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def _atomic_parquet(table: Any, path: pathlib.Path) -> None:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ValueError("pyarrow is required for history-aware mining") from exc

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    os.close(fd)
    try:
        pq.write_table(table, temporary)
        with open(temporary, "rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def _validated_history(path: pathlib.Path, iteration: int) -> dict[str, Any]:
    if not path.is_file():
        if iteration != 1:
            raise FileNotFoundError(
                f"mining history is required before iteration {iteration}: {path}"
            )
        return {
            "version": HISTORY_VERSION,
            "identity": HISTORY_IDENTITY,
            "iterations": [],
        }

    payload = json.loads(path.read_text())
    if not isinstance(payload, dict) or payload.get("version") != HISTORY_VERSION:
        raise ValueError(f"unsupported or malformed mining history: {path}")
    if payload.get("identity") != HISTORY_IDENTITY:
        raise ValueError("mining history identity must be filepath")
    entries = payload.get("iterations")
    if not isinstance(entries, list):
        raise ValueError("mining history iterations must be a list")

    expected_numbers = list(range(1, len(entries) + 1))
    actual_numbers: list[int] = []
    all_selected: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("every mining history iteration must be an object")
        number = int(entry.get("iteration", 0))
        actual_numbers.append(number)
        selected = entry.get("selected_filepaths")
        if not isinstance(selected, list):
            raise ValueError(
                f"mining history iteration {number} has no selected_filepaths list"
            )
        normalized = [_normalize_filepath(value) for value in selected]
        if len(normalized) != len(set(normalized)):
            raise ValueError(
                f"mining history iteration {number} contains duplicate filepaths"
            )
        overlap = all_selected.intersection(normalized)
        if overlap:
            raise ValueError(
                f"mining history iteration {number} reselects prior filepaths: "
                f"{sorted(overlap)[:3]}"
            )
        if int(entry.get("selected_count", -1)) != len(normalized):
            raise ValueError(
                f"mining history iteration {number} selected_count disagrees with "
                "selected_filepaths"
            )
        for path_field, hash_field in (
            ("candidate_parquet", "candidate_sha256"),
            ("output_parquet", "output_sha256"),
            ("summary_file", "summary_sha256"),
        ):
            artifact = pathlib.Path(str(entry.get(path_field) or ""))
            expected_hash = str(entry.get(hash_field) or "")
            if not artifact.is_absolute() or not artifact.is_file() or not expected_hash:
                raise ValueError(
                    f"mining history iteration {number} has invalid {path_field} proof"
                )
            actual_hash = _sha256(artifact)
            if actual_hash != expected_hash:
                raise ValueError(
                    f"mining history iteration {number} {path_field} hash mismatch"
                )
        entry["selected_filepaths"] = normalized
        all_selected.update(normalized)

    if actual_numbers != expected_numbers:
        raise ValueError(
            "mining history iterations must be contiguous from 1; "
            f"found {actual_numbers}"
        )
    if payload.get("cumulative_unique_count") != len(all_selected):
        raise ValueError(
            "mining history cumulative_unique_count disagrees with "
            "selected_filepaths"
        )
    return payload


def select_novel_samples(
    *,
    candidate_parquet: pathlib.Path,
    output_parquet: pathlib.Path,
    history_file: pathlib.Path,
    summary_file: pathlib.Path,
    iteration: int,
    topn: int,
    filepath_column: str = "filepath",
    resume: bool = False,
) -> dict[str, Any]:
    """Write the current iteration's novel candidates and append the ledger."""
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ValueError("pyarrow is required for history-aware mining") from exc

    candidate_parquet = candidate_parquet.expanduser().resolve()
    output_parquet = output_parquet.expanduser().resolve()
    history_file = history_file.expanduser().resolve()
    summary_file = summary_file.expanduser().resolve()
    if iteration < 1:
        raise ValueError("iteration must be >= 1")
    if topn < 1:
        raise ValueError("topn must be >= 1")
    if candidate_parquet == output_parquet:
        raise ValueError("candidate and output parquet paths must differ")
    if not candidate_parquet.is_file():
        raise FileNotFoundError(f"candidate parquet is missing: {candidate_parquet}")
    if len({output_parquet, history_file, summary_file}) != 3:
        raise ValueError("output parquet, history, and summary paths must differ")

    state = _validated_history(history_file, iteration)
    entries: list[dict[str, Any]] = state["iterations"]
    committed = next(
        (entry for entry in entries if int(entry["iteration"]) == iteration), None
    )
    if committed is not None:
        if not resume:
            raise ValueError(
                f"iteration {iteration} is already committed in {history_file}; "
                "pass --resume to verify and reuse it"
            )
        if pathlib.Path(committed["output_parquet"]).resolve() != output_parquet:
            raise ValueError("committed output parquet does not match --output-parquet")
        if pathlib.Path(committed["summary_file"]).resolve() != summary_file:
            raise ValueError("committed summary does not match --summary")
        if pathlib.Path(committed["candidate_parquet"]).resolve() != candidate_parquet:
            raise ValueError(
                "committed candidate parquet does not match --candidate-parquet"
            )
        if _sha256(candidate_parquet) != committed.get("candidate_sha256"):
            raise ValueError("committed candidate parquet hash mismatch")
        if int(committed.get("topn", 0)) != topn:
            raise ValueError("committed topn does not match the requested topn")
        return json.loads(summary_file.read_text())
    if len(entries) != iteration - 1:
        raise ValueError(
            f"cannot select iteration {iteration}; mining history ends at "
            f"iteration {len(entries)}"
        )

    table = pq.read_table(candidate_parquet)
    if filepath_column not in table.column_names:
        raise ValueError(
            f"candidate parquet is missing {filepath_column!r}; "
            f"columns={table.column_names}"
        )

    historical = {
        value
        for entry in entries
        for value in entry["selected_filepaths"]
    }
    candidate_seen: set[str] = set()
    selected_names: list[str] = []
    selected_indices: list[int] = []
    candidate_duplicate_count = 0
    already_mined_count = 0
    for index, value in enumerate(table[filepath_column].to_pylist()):
        identity = _normalize_filepath(value)
        if identity in candidate_seen:
            candidate_duplicate_count += 1
            continue
        candidate_seen.add(identity)
        if identity in historical:
            already_mined_count += 1
            continue
        selected_names.append(identity)
        selected_indices.append(index)

    filtered = table.take(pa.array(selected_indices, type=pa.int64()))
    _atomic_parquet(filtered, output_parquet)

    historical_candidate_rate = (
        already_mined_count / len(candidate_seen) if candidate_seen else 0.0
    )
    recommendation = None
    if not selected_names and candidate_seen:
        recommendation = (
            "all candidates were selected in earlier iterations; increase topn "
            "or expand the source pool"
        )
    elif historical_candidate_rate >= 0.5:
        recommendation = (
            "at least half of the unique candidates were already mined; consider "
            "increasing topn to cast a wider net"
        )
    summary: dict[str, Any] = {
        "version": HISTORY_VERSION,
        "iteration": iteration,
        "identity": HISTORY_IDENTITY,
        "topn": topn,
        "candidate_parquet": str(candidate_parquet),
        "candidate_sha256": _sha256(candidate_parquet),
        "candidate_row_count": table.num_rows,
        "candidate_unique_count": len(candidate_seen),
        "candidate_duplicate_count": candidate_duplicate_count,
        "history_count_before": len(historical),
        "already_mined_count": already_mined_count,
        "historical_candidate_rate": historical_candidate_rate,
        "selected_count": len(selected_names),
        "cumulative_unique_count": len(historical) + len(selected_names),
        "output_parquet": str(output_parquet),
        "output_sha256": _sha256(output_parquet),
        "recommendation": recommendation,
    }
    _atomic_json(summary_file, summary)

    entry = {
        "iteration": iteration,
        "topn": topn,
        "candidate_parquet": str(candidate_parquet),
        "candidate_sha256": summary["candidate_sha256"],
        "candidate_unique_count": len(candidate_seen),
        "already_mined_count": already_mined_count,
        "selected_count": len(selected_names),
        "selected_filepaths": selected_names,
        "output_parquet": str(output_parquet),
        "output_sha256": summary["output_sha256"],
        "summary_file": str(summary_file),
        "summary_sha256": _sha256(summary_file),
    }
    entries.append(entry)
    state["iterations"] = entries
    state["cumulative_unique_count"] = len(historical) + len(selected_names)
    _atomic_json(history_file, state)
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-parquet", required=True, type=pathlib.Path)
    parser.add_argument("--output-parquet", required=True, type=pathlib.Path)
    parser.add_argument("--history-file", required=True, type=pathlib.Path)
    parser.add_argument("--summary", required=True, type=pathlib.Path)
    parser.add_argument("--iteration", required=True, type=int)
    parser.add_argument("--topn", required=True, type=int)
    parser.add_argument("--filepath-column", default="filepath")
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        summary = select_novel_samples(
            candidate_parquet=args.candidate_parquet,
            output_parquet=args.output_parquet,
            history_file=args.history_file,
            summary_file=args.summary,
            iteration=args.iteration,
            topn=args.topn,
            filepath_column=args.filepath_column,
            resume=args.resume,
        )
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        print(f"filter_mined_history: {exc}", file=sys.stderr)
        return 2
    print(
        "filter_mined_history: "
        f"candidates={summary['candidate_unique_count']} "
        f"already_mined={summary['already_mined_count']} "
        f"selected={summary['selected_count']} topn={summary['topn']}"
    )
    if summary.get("recommendation"):
        print(f"filter_mined_history: warning: {summary['recommendation']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
