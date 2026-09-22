#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Check every workspace annotation file against its role's field contract.

One preflight command for all three inputs. ROLE_CONTRACT below is the single
source of truth for which fields each role needs — read it (or run
``--print-contract``) instead of inferring the requirement from an error
message on the first GPU job.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

from validate_sharegpt import load_records, validate_records


# Required/optional per role. `id` is required only where cosmos-rl-evaluate
# reads it: it hard-indexes item["id"] and reuses it as the per-sample output
# filename. The training loader reads media with .get(), so Mining and the
# generated Train file do not need one — though carrying it is harmless and
# makes a mined record traceable back to its source row.
ROLE_CONTRACT = {
    "proxy": {
        "filename": "proxy_kpi.json",
        "required": ("images", "conversations", "id"),
        "optional": ("video_fps",),
        "consumer": "cosmos-rl-evaluate (RCCA source)",
    },
    "benchmark": {
        "filename": "benchmark_kpi.json",
        "required": ("images", "conversations", "id"),
        "optional": ("video_fps",),
        "consumer": "cosmos-rl-evaluate (frozen stop gate)",
    },
    "mining": {
        "filename": "mining_pool.json",
        "required": ("images", "conversations"),
        "optional": ("id", "video_fps"),
        "consumer": "emit_mined_sharegpt.py -> tao_sft_example.py",
    },
}
ALWAYS = "images is [AOI, golden_reference]; final assistant value is exactly OK or NG"


def _print_contract() -> None:
    print(f"bare_okng field contract  ({ALWAYS})\n")
    print(f"{'role':<10} {'file':<22} {'required':<34} optional")
    print("-" * 92)
    for role, spec in ROLE_CONTRACT.items():
        print(
            f"{role:<10} {spec['filename']:<22} "
            f"{', '.join(spec['required']):<34} {', '.join(spec['optional'])}"
        )
    print()
    for role, spec in ROLE_CONTRACT.items():
        print(f"  {role:<10} consumed by {spec['consumer']}")


def _media_root_hint(message: str, media_root: pathlib.Path) -> str | None:
    """Spot a media root that repeats a segment the annotations already carry.

    Annotation paths are relative to the workspace root and usually start with
    ``images/``, so pointing --media-root at ``<workspace>/images`` produces
    ``.../images/images/...``. That is the near-universal first mistake, and
    the raw "missing image file(s)" error does not suggest it.
    """
    leaf = media_root.name
    if leaf and f"/{leaf}/{leaf}/" in message:
        return (
            f"'{leaf}/{leaf}' appears in the resolved path — --media-root looks "
            f"one level too deep. Annotation paths resolve from the workspace "
            f"root, so try --media-root {media_root.parent}"
        )
    return None


def check(
    paths: dict[str, pathlib.Path], *, media_root: pathlib.Path, require_files: bool
) -> tuple[dict[str, dict], list[str]]:
    report: dict[str, dict] = {}
    failures: list[str] = []
    for role, spec in ROLE_CONTRACT.items():
        path = paths[role]
        needs_id = "id" in spec["required"]
        try:
            summary = validate_records(
                load_records(path),
                media_root=media_root,
                require_files=require_files,
                require_id=needs_id,
            )
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            message = str(exc)
            hint = _media_root_hint(message, media_root)
            if hint:
                message = f"{message}\n    hint: {hint}"
            failures.append(f"{role} ({path}): {message}")
            report[role] = {"path": str(path), "ok": False, "error": message}
            continue
        summary.update(
            {
                "path": str(path),
                "ok": True,
                "id_required": needs_id,
                "id_coverage": (
                    "n/a"
                    if not needs_id and summary["unique_ids"] == 0
                    else f"{summary['unique_ids']}/{summary['records']}"
                ),
            }
        )
        report[role] = summary
    return report, failures


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=pathlib.Path)
    parser.add_argument("--media-root", type=pathlib.Path)
    for role in ROLE_CONTRACT:
        parser.add_argument(f"--{role}", type=pathlib.Path)
    parser.add_argument("--require-files", action="store_true")
    parser.add_argument("--summary", type=pathlib.Path)
    parser.add_argument(
        "--print-contract",
        action="store_true",
        help="Print the role/field contract and exit.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.print_contract:
        _print_contract()
        return 0
    if args.workspace is None:
        print("check_annotations: --workspace is required", file=sys.stderr)
        return 2
    workspace = args.workspace.expanduser().resolve()
    media_root = (args.media_root or workspace).expanduser().resolve()
    paths = {
        role: (
            getattr(args, role)
            or workspace / "annotations" / spec["filename"]
        ).expanduser().resolve()
        for role, spec in ROLE_CONTRACT.items()
    }

    report, failures = check(
        paths, media_root=media_root, require_files=args.require_files
    )
    for role, spec in ROLE_CONTRACT.items():
        entry = report[role]
        if not entry.get("ok"):
            print(f"  {role:<10} FAIL  {entry['error']}")
            continue
        print(
            f"  {role:<10} OK    records={entry['records']:<5} "
            f"labels={entry['labels']} id={entry['id_coverage']}"
            + ("" if "id" in spec["required"] else "  (id optional)")
        )
    if args.summary is not None:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(json.dumps(report, indent=2) + "\n")

    if failures:
        for failure in failures:
            print(f"check_annotations: {failure}", file=sys.stderr)
        print(
            "check_annotations: run --print-contract to see the required fields",
            file=sys.stderr,
        )
        return 2
    print("check_annotations: OK all roles satisfy the bare_okng field contract")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
