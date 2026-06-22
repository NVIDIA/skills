# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stage the ChangeNet pretrained backbone locally for the DEFT AOI loop.

Why this exists: TAO's `ptm_utils.load_pretrained_weights()` passes
`model.backbone.pretrained_backbone_path` straight to `torch.load(path)` (or
`safetensors.torch.load_file` for `.safetensors`). It does NOT dereference a
URL or a HuggingFace repo id, so the weights file must physically exist on the
host and be bind-mounted into the training container. Pre-Flight must stage it
before launch; an unstaged backbone fails the run (URL -> FileNotFoundError,
null -> silently degrades FAR@R=100%).

This script downloads the backbone from HuggingFace and copies it to the
workspace staging path. Idempotent: if a staged file already exists it is
reused and no download happens. Hard-fails (non-zero exit) when it cannot
produce a staged file, so Pre-Flight can hard-stop on the same signal.

The default repo `nvidia/C-RADIOv2-B` ships only `model.safetensors` (no
`.pth`). HF_TOKEN is read from the environment when present (required for gated
repos / rate limits).

CLI:

    python scripts/stage_backbone.py --workspace ~/workspace

    # or an explicit destination / different repo:
    python scripts/stage_backbone.py \
        --dest ~/workspace/augmentation/backbone/c_radio_v2_b.safetensors \
        --repo-id nvidia/C-RADIOv2-B --filename model.safetensors

On success the absolute staged path is printed to stdout as the last line, so a
caller can capture it: STAGED=$(python scripts/stage_backbone.py --workspace ...)
"""

import argparse
import os
import shutil
import sys


DEFAULT_REPO_ID = "nvidia/C-RADIOv2-B"
DEFAULT_FILENAME = "model.safetensors"
DEFAULT_STAGE_NAME = "c_radio_v2_b.safetensors"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage the ChangeNet backbone locally.")
    p.add_argument(
        "--workspace",
        help="Workspace root. The backbone is staged to "
        "<workspace>/augmentation/backbone/<stage-name>. Ignored if --dest is set.",
    )
    p.add_argument(
        "--dest",
        help="Explicit destination file path. Overrides --workspace.",
    )
    p.add_argument("--repo-id", default=DEFAULT_REPO_ID, help="HuggingFace repo id.")
    p.add_argument("--filename", default=DEFAULT_FILENAME, help="File to download from the repo.")
    p.add_argument(
        "--stage-name",
        default=DEFAULT_STAGE_NAME,
        help="Filename to use under <workspace>/augmentation/backbone/.",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if a staged file already exists.",
    )
    return p.parse_args()


def resolve_dest(args: argparse.Namespace) -> str:
    if args.dest:
        return os.path.abspath(os.path.expanduser(args.dest))
    if not args.workspace:
        sys.exit("stage_backbone: one of --dest or --workspace is required.")
    ws = os.path.abspath(os.path.expanduser(args.workspace))
    return os.path.join(ws, "augmentation", "backbone", args.stage_name)


def main() -> int:
    args = parse_args()
    dest = resolve_dest(args)

    # Idempotent: reuse an existing non-empty staged file unless --force.
    if not args.force and os.path.isfile(dest) and os.path.getsize(dest) > 0:
        print(f"stage_backbone: reusing already-staged file ({os.path.getsize(dest)} bytes).", file=sys.stderr)
        print(dest)
        return 0

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        sys.exit(
            "stage_backbone: huggingface_hub is not installed. Install it into the "
            "DEFT venv (pip install huggingface_hub) and retry."
        )

    token = os.environ.get("HF_TOKEN") or None
    try:
        src = hf_hub_download(repo_id=args.repo_id, filename=args.filename, token=token)
    except Exception as exc:  # network, auth, missing file — all are hard stops
        sys.exit(
            f"stage_backbone: failed to download {args.filename} from {args.repo_id}: {exc}\n"
            "Staging is mandatory — there is no working URL fallback. Set HF_TOKEN if the "
            "repo is gated, or pre-stage the file at the destination path manually."
        )

    os.makedirs(os.path.dirname(dest), exist_ok=True)
    shutil.copy(src, dest)

    if not (os.path.isfile(dest) and os.path.getsize(dest) > 0):
        sys.exit(f"stage_backbone: copy produced no file at {dest}.")

    print(f"stage_backbone: staged {args.repo_id}/{args.filename} -> {dest} "
          f"({os.path.getsize(dest)} bytes).", file=sys.stderr)
    print(dest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
