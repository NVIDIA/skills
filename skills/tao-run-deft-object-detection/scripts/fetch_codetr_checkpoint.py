#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve the Co-DETR pseudo-labelling checkpoint, downloading it if needed.

Prep labels the whole source pool with Co-DETR, and the skill has already chosen
which checkpoint it wants: ``assets/overlays/codetr_inference.yaml`` pins the ViT-L/16
geometry that only this one loads. There is exactly one right answer, so a run should
not depend on a file somebody staged by hand:

    zongzhuofan/co-detr-vit-large-coco  ->  pytorch_model.pth  (Apache-2.0, 2.8 GiB)

Stdlib only, by design. The documented alternative was ``huggingface-cli``, which is
not installable on a clean box without a virtual environment -- ``pip install --user``
is refused by ``externally-managed-environment`` -- and whose entry point has since
been renamed to ``hf``. Requiring it would also put a network-fetching package in
``deft_python.sh``'s interpreter probe, making one stage's dependency a precondition
for every run, including those that never label a pool. ``urllib`` needs nothing.

Integrity is checked rather than assumed. The architecture is pinned in the overlay,
so a checkpoint whose weights do not match loads nothing, exits 0, prints
``Execution status: PASS`` and writes one empty label file per image -- the failure
``verify_pseudo_labels.py`` exists to catch. Size and SHA-256 are verified before the
file is put in place, so that failure cannot start here.

Idempotent: an existing checkpoint at the destination is reused and nothing is
fetched, so re-running Pre-Flight on a resumed run costs nothing. Reuse checks the
size, which catches the truncated file an interrupted transfer leaves; ``--verify``
checks the digest as well, for a file that is the right size but may not be the
right checkpoint.

Inputs:  --dest, optionally --plan, --verify, --expect-sha256, --url
Output:  the checkpoint path on stdout, for the caller to capture

Exits 1 on a failed download, a size or digest mismatch, or an unwritable --dest.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import urllib.error
import urllib.request
from pathlib import Path

REPO = "zongzhuofan/co-detr-vit-large-coco"
FILENAME = "pytorch_model.pth"
URL = f"https://huggingface.co/{REPO}/resolve/main/{FILENAME}"

# Verified 2026-09-16 against the copy this workflow has labelled pools with since
# August: 2,934,763,233 bytes, SHA-256 below. HuggingFace's ETag is the xet object id,
# not a content hash, so it cannot stand in for this.
EXPECT_BYTES = 2934763233
EXPECT_SHA256 = "733d2ccde180a55151a68a6cab7c9f42b117d24d38d6197b37caf3189243256c"

CHUNK = 8 * 1024 * 1024


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dest", required=True,
                        help="Directory to hold the checkpoint. Created if absent.")
    parser.add_argument("--plan", action="store_true",
                        help="Report what would be fetched and exit, without downloading. "
                             "A checkpoint already present gets the same checks as reuse "
                             "(size; the digest too with --verify) and fails here if it "
                             "would be refused later. Prints a sentence, not a path -- do "
                             "not capture it into a variable the later stages use.")
    parser.add_argument("--url", default=URL, help="Override the source URL.")
    parser.add_argument("--expect-sha256", default=EXPECT_SHA256,
                        help="Expected digest. Pass an empty string to skip the check, "
                             "which is only right when --url points somewhere else.")
    parser.add_argument("--verify", action="store_true",
                        help="When the checkpoint is already present, check its SHA-256 "
                             "too, not just its size. Off by default because it reads "
                             "2.8 GiB; use it when a reused file is suspect -- a copy "
                             "from another host, or a run whose pseudo-labels came out "
                             "empty.")
    return parser.parse_args()


def _sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def _human(n: int) -> str:
    return f"{n / (1024 ** 3):.2f} GiB"


def download(url: str, target: Path, expect_sha: str) -> None:
    """Fetch to a sibling .part, verify, then move into place.

    Downloading straight onto the destination leaves a truncated file that looks like
    a checkpoint when a transfer is interrupted, and the next run reuses it.
    """
    part = target.with_suffix(target.suffix + ".part")
    digest = hashlib.sha256()
    written = 0
    # Anything that is not a completed, verified transfer leaves nothing behind. A
    # partial file is 2.8 GiB of disk that the next run's free-space check has to
    # account for, and it is the state this whole .part dance exists to avoid --
    # so the cleanup belongs in `finally`, not on the one error class that happens
    # to be named here. A read timeout, IncompleteRead, a reset connection and
    # Ctrl-C all reach it.
    moved = False
    try:
        with urllib.request.urlopen(url, timeout=120) as response:  # nosec B310 - https endpoint from skill configuration
            declared = response.headers.get("Content-Length")
            declared = int(declared) if declared and declared.isdigit() else None
            if declared is not None and declared != EXPECT_BYTES and url == URL:
                raise ValueError(
                    f"the source reports {declared} bytes, expected {EXPECT_BYTES}. The "
                    f"published checkpoint may have changed; verify it pairs with the "
                    f"architecture pinned in assets/overlays/codetr_inference.yaml before "
                    f"updating EXPECT_BYTES/EXPECT_SHA256")
            with part.open("wb") as handle:
                while True:
                    chunk = response.read(CHUNK)
                    if not chunk:
                        break
                    handle.write(chunk)
                    digest.update(chunk)
                    written += len(chunk)
                    if declared:
                        print(f"\r  {_human(written)} / {_human(declared)}",
                              end="", file=sys.stderr, flush=True)
        if declared:
            print(file=sys.stderr)

        if expect_sha and digest.hexdigest() != expect_sha:
            raise ValueError(
                f"digest mismatch: got {digest.hexdigest()}, expected {expect_sha}. The "
                f"file was not kept. A checkpoint whose weights do not match the pinned "
                f"architecture loads nothing and still exits 0, so this is refused rather "
                f"than warned about")
        part.replace(target)
        moved = True
    except urllib.error.URLError as exc:
        raise RuntimeError(f"download failed: {exc}") from exc
    finally:
        if not moved:
            part.unlink(missing_ok=True)


def _check_existing(target: Path, args: argparse.Namespace) -> None:
    """Raise unless the checkpoint already at `target` is fit to reuse.

    Size catches truncation, which is what an interrupted transfer leaves, and a
    truncated checkpoint loads no weights while the stage still reports success. It
    cannot catch a file of the right length with the wrong bytes, which loads no
    weights just as quietly -- so the digest is checked on request (--verify) rather
    than on every resume, where it would cost a 2.8 GiB read.
    """
    size = target.stat().st_size
    if args.url == URL and size != EXPECT_BYTES:
        raise ValueError(
            f"{target} is {size} bytes, expected {EXPECT_BYTES}. Remove it and re-run; "
            f"a truncated checkpoint loads no weights and the stage still reports success")
    if args.verify:
        if not args.expect_sha256:
            raise ValueError(
                "--verify needs a digest to compare against, but --expect-sha256 is empty")
        print(f"verifying {target} ({_human(size)})", file=sys.stderr)
        actual = _sha256_of(target)
        if actual != args.expect_sha256:
            raise ValueError(
                f"{target} has digest {actual}, expected {args.expect_sha256}. It is the "
                f"right size but not the right checkpoint. It was left in place; remove "
                f"it and re-run to fetch a verified copy")


def main() -> int:
    try:
        args = parse_args()
        dest = Path(args.dest).expanduser().resolve()
        target = dest / FILENAME

        # --plan runs the same checks as reuse. A plan that only asked "is a file
        # there" would report a truncated checkpoint as ALREADY PRESENT in the
        # Pre-Flight Summary, and the run would be approved against it.
        if args.plan:
            if target.is_file():
                _check_existing(target, args)
                print(f"ALREADY PRESENT: {target} ({_human(target.stat().st_size)})",
                      file=sys.stderr)
            else:
                print(f"WILL DOWNLOAD after approval: {REPO}/{FILENAME} "
                      f"({_human(EXPECT_BYTES)}) -> {target}", file=sys.stderr)
            return 0

        # Idempotent, and checked rather than trusted.
        if target.is_file():
            _check_existing(target, args)
            print(target)
            return 0

        dest.mkdir(parents=True, exist_ok=True)
        # EXPECT_BYTES describes the published checkpoint, so it only bounds the
        # default download. A --url pointing somewhere else has no known size, and
        # demanding 3 GiB of free space for it would refuse transfers that fit.
        free = shutil.disk_usage(dest).free
        if args.url == URL and free < EXPECT_BYTES * 1.1:
            raise ValueError(
                f"{dest} has {_human(free)} free; the checkpoint needs "
                f"{_human(EXPECT_BYTES)} plus room for the partial file")

        print(f"fetching {REPO}/{FILENAME} -> {target}", file=sys.stderr)
        download(args.url, target, args.expect_sha256)
        print(target)
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"fetch_codetr_checkpoint: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
