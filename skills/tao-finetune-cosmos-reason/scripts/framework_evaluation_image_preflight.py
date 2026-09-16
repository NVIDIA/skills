#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check that a Framework SQSH contains the baked evaluation implementation.

This preflight is read-only and intentionally does not hash, extract, patch, or
mount the image.  It reads only the three installed Python modules needed by
the Framework evaluator and attests their required source capabilities before
a GPU allocation is submitted. The model-owned spec-bundle lifecycle repeats
the check inside the container as defense in depth.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REQUIRED_SUFFIXES = (
    "/cosmos_rl/evaluation/base.py",
    "/cosmos_rl/framework/runtime.py",
    "/cosmos_rl/utils/framework_torchcodec_video.py",
)

REQUIRED_SOURCE_TOKENS = {
    "/cosmos_rl/evaluation/base.py": (
        "torchcodec-cuda-on-demand",
        "FrameworkTorchCodecVideoPreprocessor",
        "iter_prepared_batches",
    ),
    "/cosmos_rl/framework/runtime.py": ("_framework_decoded_media",),
    "/cosmos_rl/utils/framework_torchcodec_video.py": (
        "class FrameworkTorchCodecVideoPreprocessor",
        "iter_prepared_batches",
        "persistent_workers=self.dataloader_persistent_workers",
        "multiprocessing_context=self.dataloader_multiprocessing_context",
    ),
}


def _installed_paths(text: str) -> dict[str, str]:
    paths = {line.split()[-1] for line in text.splitlines() if line.strip()}
    return {
        suffix: next(
            (
                path
                for path in paths
                if path.endswith(suffix)
                and "/workspace/.venv/lib/python" in f"/{path}"
                and "/site-packages/" in f"/{path}"
            ),
            "",
        )
        for suffix in REQUIRED_SUFFIXES
    }


def check_listing(
    text: str,
    image: str,
    *,
    sources: dict[str, str] | None = None,
) -> dict[str, object]:
    installed = _installed_paths(text)
    missing = [suffix for suffix, path in installed.items() if not path]
    source_mismatches: dict[str, list[str]] = {}
    if sources is not None:
        for suffix in REQUIRED_SUFFIXES:
            if suffix in missing:
                continue
            source = sources.get(suffix, "")
            absent = [token for token in REQUIRED_SOURCE_TOKENS[suffix] if token not in source]
            if absent:
                source_mismatches[suffix] = absent
    return {
        "schema_version": 1,
        "backend": "cosmos-framework",
        "image": image,
        "required_profile": "torchcodec-cuda-on-demand",
        "compatible": not missing and not source_mismatches,
        "installed_paths": installed,
        "missing_baked_paths": missing,
        "missing_source_capabilities": source_mismatches,
        "source_attestation_performed": sources is not None,
        "runtime_attestation_required": True,
    }


def _read_installed_sources(sqsh: Path, listing: str) -> dict[str, str]:
    help_result = subprocess.run(
        ["unsquashfs", "-help"],
        text=True,
        capture_output=True,
        check=False,
    )
    help_text = f"{help_result.stdout}\n{help_result.stderr}"
    if not any(line.strip().startswith("-cat") for line in help_text.splitlines()):
        # Squashfs-tools before 4.5 cannot stream a member.  Do not extract or
        # mount merely to inspect source: the spec-bundle pre-command repeats
        # this attestation inside the immutable container before startup.
        return {}
    sources: dict[str, str] = {}
    for suffix, listed_path in _installed_paths(listing).items():
        if not listed_path:
            continue
        _, marker, relative_member = listed_path.partition("squashfs-root/")
        member = relative_member if marker else listed_path
        member = member.lstrip("/")
        completed = subprocess.run(
            ["unsquashfs", "-cat", str(sqsh), member],
            text=True,
            capture_output=True,
            check=False,
        )
        if completed.returncode:
            raise RuntimeError(
                f"unable to read baked evaluator module {suffix}: {completed.stderr.strip()}"
            )
        sources[suffix] = completed.stdout
    return sources


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--sqsh", type=Path)
    source.add_argument("--listing-file", type=Path)
    args = parser.parse_args()
    try:
        if args.listing_file:
            image = "listing-file"
            listing = args.listing_file.read_text(encoding="utf-8")
            sources = None
        else:
            image = str(args.sqsh)
            completed = subprocess.run(
                ["unsquashfs", "-lc", str(args.sqsh)],
                text=True,
                capture_output=True,
                check=False,
            )
            if completed.returncode:
                raise RuntimeError(
                    f"unable to inspect SQSH file list: {completed.stderr.strip()}"
                )
            listing = completed.stdout
            read_sources = _read_installed_sources(args.sqsh, listing)
            sources = read_sources or None
        result = check_listing(listing, image, sources=sources)
        if not args.listing_file and sources is None:
            result["source_attestation_deferred_reason"] = (
                "installed unsquashfs cannot stream members; the model-owned "
                "spec-bundle pre-command performs the same source attestation "
                "inside the immutable container before evaluator startup"
            )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["compatible"] else 4
    except (OSError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
