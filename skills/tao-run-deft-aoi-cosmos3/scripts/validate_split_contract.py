#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# SPDX-License-Identifier: Apache-2.0
"""Validate CR3 split isolation and monotonic generated Train lineage."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
from itertools import combinations

from validate_sharegpt import load_records, prompt_and_label


ROLE_PATHS = {
    "proxy": ("annotations", "proxy_kpi.json"),
    "benchmark": ("annotations", "benchmark_kpi.json"),
    "mining": ("annotations", "mining_pool.json"),
}


def _target(record: dict, path: pathlib.Path, index: int, media_root: pathlib.Path) -> str:
    images = record.get("images")
    if not isinstance(images, list) or len(images) != 2:
        raise ValueError(f"{path}[{index}]: images must contain [target, golden_reference]")
    if not all(isinstance(image, str) and image for image in images):
        raise ValueError(f"{path}[{index}]: image paths must be non-empty strings")
    target = pathlib.Path(images[0])
    if not target.is_absolute():
        target = media_root / target
    return str(target.resolve())


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_benchmark_hash(path: pathlib.Path) -> str | None:
    payload = json.loads(path.read_text())
    try:
        value = payload["evaluation_contract"]["benchmark"]["annotations_sha256"]
    except (KeyError, TypeError):
        return None
    return value if isinstance(value, str) and value else None


def validate(
    role_paths: dict[str, pathlib.Path],
    *,
    media_root: pathlib.Path,
    expected_benchmark_sha256: str | None = None,
) -> dict:
    missing_roles = {"proxy", "benchmark", "mining"} - set(role_paths)
    if missing_roles:
        raise ValueError(f"missing required roles: {sorted(missing_roles)}")
    role_targets: dict[str, set[str]] = {}
    role_records: dict[str, set[str]] = {}
    counts: dict[str, int] = {}
    for role, path in role_paths.items():
        records = load_records(path)
        if role == "synthetic":
            for index, record in enumerate(records):
                context = f"{path}[{index}]"
                _, label = prompt_and_label(record, context=context)
                if label != "NG":
                    raise ValueError(
                        f"{context}: synthetic label must be exactly NG, "
                        f"got {label!r}"
                    )
        targets = {
            _target(record, path, index, media_root)
            for index, record in enumerate(records)
        }
        if len(targets) != len(records):
            raise ValueError(
                f"{role}: target images must be unique ({len(records) - len(targets)} duplicate(s))"
            )
        role_targets[role] = targets
        role_records[role] = {
            json.dumps(record, sort_keys=True, separators=(",", ":"))
            for record in records
        }
        counts[role] = len(records)

    overlaps: dict[str, int] = {}
    for left, right in combinations(("proxy", "benchmark", "mining"), 2):
        shared = role_targets[left] & role_targets[right]
        overlaps[f"{left}:{right}"] = len(shared)
        if shared:
            raise ValueError(
                f"target leakage between {left} and {right}: {sorted(shared)[:5]}"
            )

    # Synthetic AnomalyGen output is a training source alongside Mining, but it
    # is still evaluation-isolated: generated boards must never appear in an
    # evaluation split.
    if "synthetic" in role_targets:
        for evaluation_role in ("proxy", "benchmark"):
            shared = role_targets["synthetic"] & role_targets[evaluation_role]
            overlaps[f"synthetic:{evaluation_role}"] = len(shared)
            if shared:
                raise ValueError(
                    f"target leakage between synthetic and {evaluation_role}: "
                    f"{sorted(shared)[:5]}"
                )

    # Iteration N>1 is seeded from the preceding committed Train artifact.
    # Treat that exact artifact as an eligible source and keep it isolated from
    # both evaluation roles just like the current iteration's synthetic data.
    if "previous_train" in role_targets:
        for evaluation_role in ("proxy", "benchmark"):
            shared = (
                role_targets["previous_train"]
                & role_targets[evaluation_role]
            )
            overlaps[f"previous_train:{evaluation_role}"] = len(shared)
            if shared:
                raise ValueError(
                    f"target leakage between previous_train and {evaluation_role}: "
                    f"{sorted(shared)[:5]}"
                )

    if "train" in role_targets:
        for evaluation_role in ("proxy", "benchmark"):
            shared = role_targets["train"] & role_targets[evaluation_role]
            overlaps[f"train:{evaluation_role}"] = len(shared)
            if shared:
                raise ValueError(
                    f"target leakage between train and {evaluation_role}: "
                    f"{sorted(shared)[:5]}"
                )
        eligible = (
            role_targets["mining"]
            | role_targets.get("previous_train", set())
            | role_targets.get("synthetic", set())
        )
        outside_eligible = role_targets["train"] - eligible
        overlaps["train:mining"] = len(
            role_targets["train"] & role_targets["mining"]
        )
        if "previous_train" in role_targets:
            overlaps["train:previous_train"] = len(
                role_targets["train"] & role_targets["previous_train"]
            )
            missing_previous = (
                role_records["previous_train"] - role_records["train"]
            )
            if missing_previous:
                raise ValueError(
                    "generated Train must retain every record from "
                    f"--previous-train ({len(missing_previous)} missing)"
                )
        if "synthetic" in role_targets:
            overlaps["train:synthetic"] = len(
                role_targets["train"] & role_targets["synthetic"]
            )
        if outside_eligible:
            sources = ["the Mining pool"]
            if "previous_train" in role_targets:
                sources.append("--previous-train")
            if "synthetic" in role_targets:
                sources.append("the current iteration's --synthetic output")
            raise ValueError(
                f"generated Train targets must come from {' or '.join(sources)}: "
                f"{sorted(outside_eligible)[:5]}"
            )

    benchmark_hash = _sha256(role_paths["benchmark"])
    benchmark_hash_verified = (
        expected_benchmark_sha256 is not None
        and benchmark_hash == expected_benchmark_sha256
    )
    if expected_benchmark_sha256 is not None and not benchmark_hash_verified:
        raise ValueError(
            "frozen Benchmark annotation hash mismatch: "
            f"expected {expected_benchmark_sha256}, got {benchmark_hash}"
        )

    return {
        "roles": {
            "proxy": "proxy_kpi_rcca_input",
            "benchmark": "benchmark_kpi_report_only",
            "mining": "mining_pool",
            **(
                {"synthetic": "anomalygen_sdg"}
                if "synthetic" in role_targets
                else {}
            ),
            **(
                {"previous_train": "previous_iteration_train"}
                if "previous_train" in role_targets
                else {}
            ),
            **(
                {
                    "train": (
                        "generated_from_mining_and_anomalygen"
                        if "synthetic" in role_targets
                        else "generated_from_mining"
                    )
                }
                if "train" in role_targets
                else {}
            ),
        },
        "records": counts,
        "target_overlap": overlaps,
        "benchmark_sha256": benchmark_hash,
        "benchmark_hash_verified": benchmark_hash_verified,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", required=True, type=pathlib.Path)
    parser.add_argument(
        "--train",
        default=None,
        type=pathlib.Path,
        help=(
            "Optional generated training JSON; its targets must come from "
            "Mining, --previous-train, or --synthetic."
        ),
    )
    parser.add_argument(
        "--previous-train",
        default=None,
        type=pathlib.Path,
        help=(
            "Immediate preceding iteration's generated Train JSON. Required "
            "for iteration N>1 so historical records remain eligible and "
            "their monotonic retention is verified."
        ),
    )
    parser.add_argument(
        "--synthetic",
        default=None,
        type=pathlib.Path,
        help=(
            "Optional AnomalyGen ShareGPT JSON for this iteration. Its targets "
            "become an eligible Train source and must not appear in Proxy or "
            "Benchmark."
        ),
    )
    for role in ROLE_PATHS:
        parser.add_argument(f"--{role}", default=None, type=pathlib.Path)
    parser.add_argument(
        "--manifest",
        default=None,
        type=pathlib.Path,
        help="Optional workspace manifest containing the frozen Benchmark sha256.",
    )
    parser.add_argument("--summary", default=None, type=pathlib.Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    workspace = args.workspace.expanduser().resolve()
    role_paths = {
        role: (getattr(args, role) or workspace.joinpath(*parts)).expanduser().resolve()
        for role, parts in ROLE_PATHS.items()
    }
    if args.synthetic is not None:
        role_paths["synthetic"] = args.synthetic.expanduser().resolve()
    if args.previous_train is not None:
        role_paths["previous_train"] = args.previous_train.expanduser().resolve()
    if args.train is not None:
        role_paths["train"] = args.train.expanduser().resolve()
    expected_hash = None
    try:
        if args.manifest is not None:
            expected_hash = _manifest_benchmark_hash(args.manifest.expanduser().resolve())
            if expected_hash is None:
                raise ValueError(
                    f"{args.manifest}: missing evaluation_contract.benchmark.annotations_sha256"
                )
        summary = validate(
            role_paths,
            media_root=workspace,
            expected_benchmark_sha256=expected_hash,
        )
        if args.summary is not None:
            args.summary.parent.mkdir(parents=True, exist_ok=True)
            args.summary.write_text(json.dumps(summary, indent=2) + "\n")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"validate_split_contract: {exc}", file=sys.stderr)
        return 2

    print(
        "validate_split_contract: OK "
        + " ".join(f"{role}={count}" for role, count in summary["records"].items())
        + f" benchmark_sha256={summary['benchmark_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
