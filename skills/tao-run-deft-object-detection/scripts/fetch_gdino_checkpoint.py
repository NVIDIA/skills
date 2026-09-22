#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve the Grounding DINO zero-shot checkpoint, pulling it from NGC if needed.

The loop scores a pretrained Grounding DINO at baseline and fine-tunes from it on
every iteration. That checkpoint is published, so a run should not depend on a file
somebody staged by hand:

    nvidia/tao/grounding_dino:grounding_dino_swin_tiny_commercial_trainable_v1.1

Verified 2026-08-03 against a hand-staged copy that had been in use under the name
``fixed_ckpt.pth``: 951 tensors each, no key differing in either direction, and all
951 bit-identical. The renamed file was the NGC download, not a patched variant.

``trainable`` is the variant to use. The sibling ``deployable`` release is for
TensorRT export and will not fine-tune.

The download also ships NVIDIA's own ``experiment.yaml``, whose model block is the
authority for the two settings that fail silently — ``log_scale: auto`` and
``class_embed_bias: True``. This script keeps that file next to the checkpoint
rather than discarding it; see ``references/grounding-dino.md``.

Idempotent: an existing checkpoint at the destination is reused and nothing is
fetched, so re-running Pre-Flight on a resumed run costs nothing.

Stdlib only. Requires the `ngc` CLI on PATH and a configured NGC account.
"""

from __future__ import annotations

import argparse
import shutil
import zipfile
import subprocess
import sys
from pathlib import Path

MODEL = "nvidia/tao/grounding_dino"
DEFAULT_VERSION = "grounding_dino_swin_tiny_commercial_trainable_v1.1"


def find_checkpoint(root: Path) -> Path | None:
    """Return the largest .pth under root, or None.

    Largest rather than first: the download lands in a version-named subdirectory
    alongside experiment.yaml, and picking by size is stable even if NGC changes
    the directory layout or adds a second, smaller artifact.
    """
    candidates = sorted(root.rglob("*.pth"), key=lambda p: p.stat().st_size, reverse=True)
    return candidates[0] if candidates else None


def verify(checkpoint: Path) -> None:
    """Raise unless the file is a readable torch archive.

    A .pth is a zip, and a truncated or interrupted download stays a plausible
    file of plausible size -- the corruption only shows up when torch reads the
    central directory at the end. Checking here turns that into an error naming
    the download, instead of `PytorchStreamReader failed reading zip archive`
    surfacing from whichever stage first loads it.
    """
    if not zipfile.is_zipfile(checkpoint):
        raise RuntimeError(
            f"{checkpoint} is not a readable torch archive — the download is corrupt. "
            "Re-run with --force."
        )
    with zipfile.ZipFile(checkpoint) as archive:
        broken = archive.testzip()
    if broken is not None:
        raise RuntimeError(
            f"{checkpoint} is a corrupt archive (first bad member: {broken}). "
            "Re-run with --force."
        )


def fetch(dest: Path, version: str) -> Path:
    if shutil.which("ngc") is None:
        raise RuntimeError(
            "the `ngc` CLI is not on PATH. Install it, or pass an explicit "
            "--zero-shot-checkpoint to skip the NGC path entirely (air-gapped runs)."
        )

    # `ngc download-version` errors out when --dest does not already exist; it will
    # not create it. Failing here with a clear message beats an opaque CLI error.
    dest.mkdir(parents=True, exist_ok=True)

    cmd = ["ngc", "registry", "model", "download-version", f"{MODEL}:{version}",
           "--dest", str(dest)]
    print(f"$ {' '.join(cmd)}", file=sys.stderr)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        tail = (result.stderr or result.stdout or "").strip().splitlines()
        raise RuntimeError(
            "ngc download-version failed:\n  " + "\n  ".join(tail[-6:])
            + "\n  Supply --zero-shot-checkpoint directly to skip the NGC path."
        )

    checkpoint = find_checkpoint(dest)
    if checkpoint is None:
        raise RuntimeError(
            f"the download reported success but no .pth landed under {dest}. "
            "Inspect that directory before retrying."
        )
    return checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dest", required=True,
                        help="Directory to download into. Created if absent.")
    parser.add_argument("--version", default=DEFAULT_VERSION,
                        help=f"NGC model version. Default {DEFAULT_VERSION}. Use a "
                             "`trainable` release — `deployable` is TensorRT-only and "
                             "cannot be fine-tuned.")
    parser.add_argument("--plan", action="store_true",
                        help="Print the path the checkpoint would occupy and exit, "
                             "downloading nothing. For Pre-Flight, which reports what a run "
                             "will do before the user has approved it.")
    parser.add_argument("--force", action="store_true",
                        help="Re-download even when a checkpoint is already present.")
    return parser.parse_args()


def main() -> int:
    try:
        args = parse_args()
        dest = Path(args.dest).expanduser().resolve()

        existing = find_checkpoint(dest) if dest.is_dir() else None

        if args.plan:
            # Pre-Flight runs before the user has approved anything, so it reports
            # what the run will do rather than doing it.
            if existing is not None:
                print(f"already present: {existing} "
                      f"({existing.stat().st_size / 1e9:.2f} GB); no download needed",
                      file=sys.stderr)
                print(existing)
            elif shutil.which("ngc") is None:
                print(f"CANNOT DOWNLOAD: {args.version} is not present at {dest} and the "
                      f"`ngc` CLI is not on PATH. Add it, or have the user supply "
                      f"--zero-shot-checkpoint. Reporting this before the gate is the point "
                      f"of --plan.", file=sys.stderr)
                return 1
            else:
                print(f"WILL DOWNLOAD after approval: {args.version} (~1.93 GB) -> {dest}",
                      file=sys.stderr)
                print(dest)
            return 0

        if not args.force:
            if existing is not None:
                verify(existing)
                print(f"reusing {existing} ({existing.stat().st_size / 1e9:.2f} GB)",
                      file=sys.stderr)
                print(existing)
                return 0

        checkpoint = fetch(dest, args.version)
        verify(checkpoint)
        spec = next(iter(sorted(dest.rglob("experiment.yaml"))), None)
        print(f"checkpoint: {checkpoint} ({checkpoint.stat().st_size / 1e9:.2f} GB)",
              file=sys.stderr)
        if spec is not None:
            print(f"NVIDIA's reference spec shipped alongside it: {spec}", file=sys.stderr)
        # stdout carries the path alone, so a caller can capture it directly.
        print(checkpoint)
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
