#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""List a pool image directory into the parquet ``embedding image_embeddings`` reads.

The prep stage embeds the whole source pool once, producing the corpus every
iteration's mining searches against. That action takes a parquet with a
``filepath`` column, and until now the skill only *described* building one — an
instruction to improvise, which is how a run ends up with an artifact nobody can
reproduce.

**Paths are absolute, and that is load-bearing.** Mining selects rows by
``filepath`` and staging then looks the annotation up by basename. If the two
disagree, mining succeeds and staging reports every image as missing an
annotation — a failure that looks like bad data rather than bad paths.

**Symlinks are resolved.** A pool of symlinks into another tree is common, and a
container that mounts only the workspace cannot see the targets: the run then
fails with zero resolvable images and an error that blames the image list. This
resolves each path and reports anything that does not exist, so a broken link is
caught here rather than inside a container.

Order is sorted, so the same directory always yields the same parquet.

Requires pandas + pyarrow, via ``deft_python.sh``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")


def collect(images_dir: Path) -> tuple[list[str], list[str]]:
    """Return (resolved absolute paths, unresolvable paths)."""
    found = sorted(
        p for p in images_dir.iterdir()
        if p.suffix.lower() in IMAGE_SUFFIXES
    )
    resolved: list[str] = []
    broken: list[str] = []
    for path in found:
        target = path.resolve()
        # exists() follows the link, so a symlink into an unmounted tree lands here
        # rather than inside the container as an unexplained zero-image run.
        (resolved if target.exists() else broken).append(str(target))
    return resolved, broken


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images-dir", required=True,
                        help="Pool image directory. Not searched recursively — the pool is flat.")
    parser.add_argument("--out", required=True,
                        help="Where to write the parquet (one `filepath` column).")
    parser.add_argument("--allow-broken-links", action="store_true",
                        help="Proceed when some paths do not resolve, dropping them. By default "
                             "any unresolvable path is a hard error, because the usual cause is "
                             "a mount that does not cover the symlink targets.")
    return parser.parse_args()


def main() -> int:
    try:
        args = parse_args()

        images_dir = Path(args.images_dir).expanduser().resolve()
        if not images_dir.is_dir():
            raise NotADirectoryError(f"--images-dir does not exist: {images_dir}")

        resolved, broken = collect(images_dir)

        if broken and not args.allow_broken_links:
            sample = "\n  ".join(broken[:5])
            raise FileNotFoundError(
                f"{len(broken)} of {len(resolved) + len(broken)} image path(s) do not resolve. "
                "The usual cause is a pool of symlinks whose targets are outside the mounted "
                f"tree — the container would see none of them.\n  {sample}"
            )
        if not resolved:
            raise FileNotFoundError(
                f"no images with suffix {IMAGE_SUFFIXES} under {images_dir}"
            )

        out = Path(args.out).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"filepath": resolved}).to_parquet(out, index=False)

        print(f"pool input -> {out}")
        print(f"  {len(resolved)} image(s)")
        if broken:
            print(f"  WARNING: dropped {len(broken)} unresolvable path(s)", file=sys.stderr)
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
