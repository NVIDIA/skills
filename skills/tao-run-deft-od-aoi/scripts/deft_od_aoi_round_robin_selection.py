#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Deterministic per-query round-robin similarity selection."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from deft_od_aoi_round_robin_admission import RoundRobinAdmission


def _vectors(values: pd.Series) -> np.ndarray:
    rows = [np.asarray(value, dtype=np.float32).reshape(-1) for value in values]
    if not rows or len({row.size for row in rows}) != 1:
        raise ValueError("embeddings are empty or width-mismatched")
    matrix = np.stack(rows)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if not np.isfinite(matrix).all() or np.any(norms <= 0):
        raise ValueError("embeddings contain non-finite or zero-norm rows")
    return matrix / norms


def round_robin_rank(candidates: pd.DataFrame, queries: pd.DataFrame, *,
                     excluded: set[str], minimum: float, quota: int,
                     overfetch: int, audit_top_k: int,
                     excluded_candidates: set[str] | None = None
                     ) -> list[dict[str, Any]]:
    """Build a bounded deterministic ranking independently for each query."""
    if quota <= 0 or queries.empty or candidates.empty:
        return []
    required = {"candidate_id", "source_filepath", "embedding"}
    if not required.issubset(candidates) or not {"query_id", "embedding"}.issubset(queries):
        raise ValueError("round-robin selection inputs lack identity or embedding columns")
    if candidates.candidate_id.astype(str).duplicated().any():
        raise ValueError("candidate_id values must be unique")
    if queries.query_id.astype(str).duplicated().any():
        raise ValueError("query_id values must be unique")
    eligible = candidates[
        ~candidates.source_filepath.astype(str).isin(excluded)
    ].reset_index(drop=True)
    if excluded_candidates:
        if "filepath" not in eligible:
            raise ValueError("candidate exclusions require candidate filepath values")
        eligible = eligible[
            ~eligible.filepath.astype(str).map(
                lambda value: str(Path(value).expanduser().resolve())
            ).isin(excluded_candidates)
        ].reset_index(drop=True)
    if eligible.empty:
        return []
    candidate_vectors = _vectors(eligible.embedding)
    query_vectors = _vectors(queries.embedding)
    if candidate_vectors.shape[1] != query_vectors.shape[1]:
        raise ValueError("candidate/query embedding widths differ")
    scores = query_vectors @ candidate_vectors.T
    per_query_limit = min(
        scores.shape[1],
        max(audit_top_k, math.ceil(quota / len(queries)) * overfetch + 10),
    )
    orders: list[list[tuple[int, float]]] = []
    for row_scores in scores:
        positions = np.lexsort((np.arange(len(row_scores)), -row_scores))[:per_query_limit]
        ranked, seen = [], set()
        for position in positions:
            score = float(row_scores[position])
            parent = str(eligible.iloc[position].source_filepath)
            if score < minimum or parent in seen:
                continue
            seen.add(parent)
            ranked.append((int(position), score))
        orders.append(ranked)
    target = max(quota * overfetch, quota + 25)
    output, selected, depth = [], set(), 0
    while len(output) < target and any(depth < len(order) for order in orders):
        for query_position, order in enumerate(orders):
            if depth >= len(order):
                continue
            candidate_position, score = order[depth]
            row = eligible.iloc[candidate_position].to_dict()
            parent = str(row["source_filepath"])
            if parent in selected:
                continue
            selected.add(parent)
            row.update(
                similarity=score,
                query_id=str(queries.iloc[query_position].query_id),
            )
            output.append(row)
            if len(output) >= target:
                break
        depth += 1
    return output


def _canonical_queries(queries: pd.DataFrame) -> pd.DataFrame:
    """Use the frozen gap ordinal rather than transient hashed query IDs."""
    if queries.empty:
        return queries
    ordered = queries.copy()
    def key(position: int, value: Any) -> tuple[str, int, str, int]:
        parts = str(value).split(":", 2)
        if len(parts) >= 2:
            try:
                return parts[0], int(parts[1]), parts[2] if len(parts) > 2 else "", position
            except ValueError:
                pass
        return "", position, str(value), position

    values = (ordered.routing_order_key.tolist() if "routing_order_key" in ordered
              else list(range(len(ordered))))
    positions = sorted(range(len(ordered)), key=lambda position: key(position, values[position]))
    return ordered.iloc[positions].reset_index(drop=True)


def _refill(candidates: pd.DataFrame, queries: pd.DataFrame, *, quota: int,
            used: set[str], minimum: float, overfetches: list[int], audit_top_k: int,
            excluded_candidates: set[str], admission: RoundRobinAdmission | None,
            clean: bool, record: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    attempted: set[str] = set()
    attempts = []
    if admission is not None:
        admission.begin_branch(quota, clean)
    for overfetch in overfetches:
        ranked = round_robin_rank(
            candidates, queries, excluded=used, minimum=minimum, quota=quota,
            overfetch=overfetch, audit_top_k=audit_top_k,
            excluded_candidates=excluded_candidates,
        )
        fresh = [
            row for row in ranked if str(row["candidate_id"]) not in attempted
        ]
        attempted.update(str(row["candidate_id"]) for row in fresh)
        remaining = quota - len(selected)
        admitted = (
            admission.admit(fresh, remaining, clean, record)
            if admission is not None else fresh[:remaining]
        )
        selected.extend(admitted)
        used.update(str(row["source_filepath"]) for row in admitted)
        attempts.append({
            "overfetch": overfetch, "ranked": len(ranked), "fresh": len(fresh),
            "newly_admitted": len(admitted), "cumulative_admitted": len(selected),
            "exhaustive": overfetch == overfetches[-1],
        })
        if len(selected) == quota:
            break
    return selected, {
        "requested": quota, "ranked": sum(row["fresh"] for row in attempts),
        "admitted": len(selected), "shortfall": quota - len(selected),
        "attempts": attempts,
    }


def select(candidates: dict[str, pd.DataFrame], queries: dict[str, pd.DataFrame],
           policy: dict[str, Any], excluded: dict[str, set[str]],
           prior_real: int = 0, prior_clean: int = 0,
           excluded_candidates: dict[str, set[str]] | None = None,
           admission: RoundRobinAdmission | None = None, record: Any = None,
           ) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    retrieval, routing = policy["retrieval"], policy["routing"]
    output: dict[str, list[dict[str, Any]]] = {"real": [], "clean": []}
    audit: dict[str, Any] = {"branches": []}
    used_real = set(excluded["real"])

    def pocket(row: pd.Series) -> tuple[str, str, str]:
        values = tuple(
            str(row.get(key, "")).strip()
            for key in ("dataset_id", "texture_id", "defect_class")
        )
        if any(not value for value in values):
            raise ValueError(
                f"round-robin real query {row.get('query_id')!r} lacks pocket metadata"
            )
        return values  # type: ignore[return-value]

    overfetches = [int(value) for value in retrieval.get(
        "round_robin_refill_overfetch", (5, 15, 50, 200, 100000)
    )]
    if not overfetches or any(value < 1 for value in overfetches):
        raise ValueError("round_robin_refill_overfetch must contain positive integers")
    real = queries.get("real", pd.DataFrame())
    for reason in ("fn", "near_miss_fp"):
        branch = real[real.reason.astype(str).eq(reason)] if not real.empty else real
        branch = _canonical_queries(branch)
        groups = branch.groupby(branch.apply(pocket, axis=1), sort=True) if not branch.empty else []
        for pocket_key, rows in groups:
            if reason == "fn":
                factors = {int(value) for value in rows.get(
                    "real_factor",
                    pd.Series([routing["real_mine_factor_min"]] * len(rows)),
                ).tolist()}
                if len(factors) != 1:
                    raise ValueError("strict queries in one pocket must share real_factor")
                quota = factors.pop() * len(rows)
            else:
                quota = min(
                    int(routing["near_miss_real_factor"]) * len(rows),
                    int(routing["near_miss_real_cap_per_pocket"]),
                )
            chosen, branch_audit = _refill(
                candidates["real"], rows, quota=quota, used=used_real,
                minimum=float(retrieval["minimum_similarity"]),
                overfetches=overfetches,
                audit_top_k=int(retrieval["audit_top_k_per_query"]),
                excluded_candidates=(excluded_candidates or {}).get("real", set()),
                admission=admission, clean=False,
                record=(lambda path: record("real", path)) if record else None,
            )
            output["real"].extend(chosen)
            audit["branches"].append({
                "role": "real", "reason": reason, "pocket": list(pocket_key),
                "queries": len(rows), **branch_audit,
            })
    clean = _canonical_queries(queries.get("clean", pd.DataFrame()))
    if not clean.empty:
        clean_room = max(
            0,
            int((prior_real + len(output["real"]))
                * float(routing["clean_cumulative_cap_per_real"])) - prior_clean,
        )
        quota = min(int(routing["clean_factor"]) * len(clean), clean_room)
        used_clean = set(excluded["clean"])
        output["clean"], branch_audit = _refill(
            candidates["clean"], clean, quota=quota, used=used_clean,
            minimum=float(retrieval["minimum_similarity"]),
            overfetches=overfetches,
            audit_top_k=int(retrieval["audit_top_k_per_query"]),
            excluded_candidates=(excluded_candidates or {}).get("clean", set()),
            admission=admission, clean=True,
            record=(lambda path: record("clean", path)) if record else None,
        )
        audit["branches"].append({
            "role": "clean", "reason": "background_fp", "pocket": None,
            "queries": len(clean), **branch_audit,
        })
    return output, audit


def _previous(document: dict[str, Any], root: Path | None
              ) -> tuple[dict[str, set[str]], dict[str, int]]:
    excluded = {"real": set(), "clean": set()}
    counts = {"real": 0, "clean": 0}
    roles = {"real_defect": "real", "clean_negative": "clean"}
    for image in document.get("images", []):
        role = roles.get(str(image.get("deft_kind")))
        if not role:
            continue
        raw = image.get("source_path")
        path = Path(str(raw or image["file_name"]))
        if not path.is_absolute() and root:
            path = root / "images" / path.name
        excluded[role].add(str(path.resolve()))
        counts[role] += 1
    return excluded, counts


def _candidate_exclusions(role: str, retrieval_root: Path,
                          candidates: pd.DataFrame, manifest: dict[str, Any]) -> set[str]:
    path = retrieval_root / f"exclude_{role}_candidates.parquet"
    evidence = (manifest.get("role_status") or {}).get(role)
    if evidence:
        declared = evidence.get("exclusion_manifest")
        if declared and Path(str(declared)).resolve() != path.resolve():
            raise ValueError(f"{role} exclusion manifest path is not canonical")
        if not path.is_file():
            raise FileNotFoundError(f"{role} exclusion manifest is missing: {path}")
    elif not path.is_file():
        return set()
    frame = pd.read_parquet(path)
    if "filepath" not in frame:
        raise ValueError(f"{role} exclusion manifest lacks filepath")
    values = [str(Path(value).expanduser().resolve()) for value in frame.filepath.astype(str)]
    if len(values) != len(set(values)):
        raise ValueError(f"{role} exclusion manifest contains duplicate filepaths")
    if "filepath" not in candidates:
        raise ValueError(f"{role} candidates lack filepath")
    known = {
        str(Path(value).expanduser().resolve())
        for value in candidates.filepath.astype(str)
    }
    unmatched = set(values) - known
    if unmatched:
        raise ValueError(f"{role} exclusion manifest contains unknown candidates")
    if evidence and int(evidence.get("excluded_count", -1)) != len(values):
        raise ValueError(f"{role} exclusion count disagrees with its manifest")
    return set(values)


def _source_records(policy: dict[str, Any], role: str) -> dict[str, dict[str, Any]]:
    source = policy["sources"][role]
    images = Path(source["images"])
    document = json.loads(Path(source["coco"]).read_text())
    annotations: dict[int, list[dict[str, Any]]] = {}
    for row in document.get("annotations", []):
        annotations.setdefault(int(row["image_id"]), []).append(row)
    records = {}
    for image in document["images"]:
        path = (
            Path(str(image["source_path"])) if image.get("source_path")
            else images / str(image["file_name"])
        ).resolve()
        records[str(path)] = {
            "boxes": [row["bbox"] for row in annotations.get(int(image["id"]), [])]
        }
    return records


def materialize(policy_path: Path, candidate_root: Path, retrieval_root: Path,
                previous_path: Path | None = None) -> dict[str, Any]:
    """Write round-robin selections to the standard retrieval-stage artifacts."""
    policy = yaml.safe_load(policy_path.read_text())
    strategy = ((policy.get("retrieval") or {}).get("selection") or {}).get("strategy")
    if strategy != "round_robin_similarity":
        raise ValueError("round-robin materialization requires round_robin_similarity")
    manifest_path = retrieval_root / "query_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "COMPLETE":
        raise ValueError("query manifest is incomplete")
    enabled = list(manifest.get("enabled_roles") or [])
    if any(role not in {"real", "clean"} for role in enabled):
        raise ValueError("query manifest contains an unsupported role")
    previous = {} if previous_path is None else json.loads(previous_path.read_text())
    excluded, counts = _previous(previous, previous_path.parent if previous_path else None)
    candidates = {
        role: pd.read_parquet(candidate_root / f"{role}_candidate_embeddings.parquet")
        for role in enabled
    }
    queries = {
        role: pd.read_parquet(retrieval_root / f"{role}_query_embeddings.parquet")
        for role in enabled
    }
    candidate_exclusions = {
        role: _candidate_exclusions(role, retrieval_root, candidates[role], manifest)
        for role in enabled
    }
    records = {role: _source_records(policy, role) for role in enabled}
    previous_index = (
        previous_path.parent / "admission_index.npy" if previous_path else None
    )
    admission = RoundRobinAdmission(policy, previous_index)

    def record(role: str, path: str) -> dict[str, Any]:
        key = str(Path(path).resolve())
        if key not in records[role]:
            raise ValueError(f"{role} candidate source is absent from its frozen COCO: {key}")
        return records[role][key]

    selected, audit = select(
        candidates, queries, policy, excluded, counts["real"], counts["clean"],
        candidate_exclusions, admission, record,
    )
    outputs = {
        role: retrieval_root / f"mine_{role}" / "final_unique_files.parquet"
        for role in enabled
    }
    report_path = retrieval_root / "round_robin_selection_report.json"
    index_path = retrieval_root / "round_robin_admission_index.npy"
    existing = [
        path for path in (*outputs.values(), report_path, index_path) if path.exists()
    ]
    if existing:
        raise FileExistsError(existing[0])
    for role, path in outputs.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        columns = list(candidates[role].columns)
        columns.extend(name for name in ("similarity", "query_id") if name not in columns)
        pd.DataFrame(selected.get(role, []), columns=columns).to_parquet(path, index=False)
    admission.save(index_path)
    selected_counts = {role: len(selected.get(role, [])) for role in enabled}
    report = {
        "status": "COMPLETE", "iteration": int(manifest["iteration"]),
        "selection_strategy": strategy,
        "selected_counts": selected_counts,
        "role_status": {
            role: {"status": "SELECTED" if count else "NO_MATCHES",
                   "selected_count": count}
            for role, count in selected_counts.items()
        },
        "outputs": {role: str(path) for role, path in outputs.items()},
        "admission_index": str(index_path),
        "admission_counters": dict(sorted(admission.report.items())),
        "audit": audit,
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--retrieval-root", type=Path, required=True)
    parser.add_argument("--previous-coco", type=Path)
    args = parser.parse_args()
    result = materialize(
        args.policy.resolve(), args.candidate_root.resolve(), args.retrieval_root.resolve(),
        args.previous_coco.resolve() if args.previous_coco else None,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
