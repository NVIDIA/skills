# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stage the Visual ChangeNet classify backbone locally before launch.

Why this exists: TAO's `ptm_utils.load_pretrained_weights()` passes
`model.backbone.pretrained_backbone_path` straight to `torch.load(path)` (or
`safetensors.torch.load_file` for `.safetensors`). It does NOT dereference a
URL or a HuggingFace repo id, so the weights file must physically exist on the
host and be bind-mounted into the container. An unstaged backbone fails the run
(URL -> FileNotFoundError within seconds; null -> silently degrades).

The backbone is the PUBLIC `nvidia/C-RADIOv2-B` repo on HuggingFace, which ships
only `model.safetensors` (no `.pth`). It needs no NGC CLI, no NGC org, and no
credentials for the public download — `HF_TOKEN` is read from the environment
only when present (for gated mirrors / rate limits). This is the reason the VCN
workflow does not depend on an `ngc://` transfer-learning checkpoint.

Run it in the CPU shell (where host network and HF_TOKEN live) so the GPU
container never needs the token. Idempotent: an already-staged file is reused.
Hard-fails (non-zero exit) when it cannot produce a staged file.

CLI:

    python scripts/stage_backbone.py --workspace ~/tao-workspace
    # -> stages to <workspace>/pretrained_models/C-RADIOv2_B.safetensors

    # or an explicit destination:
    python scripts/stage_backbone.py \
        --dest ~/tao-workspace/pretrained_models/C-RADIOv2_B.safetensors

On success the absolute staged path is printed to stdout as the last line, so a
caller can capture it: STAGED=$(python scripts/stage_backbone.py --workspace ...)
Mount that file to the container's `/data/pretrained_models/C-RADIOv2_B.safetensors`
and set `model.backbone.pretrained_backbone_path` to the container path.
"""

import argparse
import os
import shutil
import sys


DEFAULT_REPO_ID = "nvidia/C-RADIOv2-B"
DEFAULT_FILENAME = "model.safetensors"
DEFAULT_STAGE_NAME = "C-RADIOv2_B.safetensors"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage the VCN classify backbone locally.")
    p.add_argument(
        "--workspace",
        help="Workspace root. The backbone is staged to "
        "<workspace>/pretrained_models/<stage-name>. Ignored if --dest is set.",
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
        help="Filename to use under <workspace>/pretrained_models/.",
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
    return os.path.join(ws, "pretrained_models", args.stage_name)


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
            "stage_backbone: huggingface_hub is not installed. Install it "
            "(pip install huggingface_hub) and retry."
        )

    token = os.environ.get("HF_TOKEN") or None
    try:
        src = hf_hub_download(repo_id=args.repo_id, filename=args.filename, token=token)
    except Exception as exc:  # network, auth, missing file — all are hard stops
        sys.exit(
            f"stage_backbone: failed to download {args.filename} from {args.repo_id}: {exc}\n"
            "Staging is mandatory — there is no working URL fallback. The repo is "
            "public, so no token is normally needed; set HF_TOKEN only if you hit a "
            "rate limit or a gated mirror, or pre-stage the file at the destination."
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
