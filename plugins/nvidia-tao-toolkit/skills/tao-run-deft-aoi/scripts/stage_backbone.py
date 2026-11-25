# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stage the ChangeNet pretrained backbone locally for the DEFT AOI loop.

Why this exists: TAO's `ptm_utils.load_pretrained_weights()` passes
`model.backbone.pretrained_backbone_path` straight to `torch.load(path)` (or
`safetensors.torch.load_file` for `.safetensors`). It does NOT dereference a
URL or a HuggingFace repo id, so the weights file must physically exist on the
host and be bind-mounted into the training container. Pre-Flight must stage it
before launch; an unstaged backbone fails the run (URL -> FileNotFoundError,
null -> silently degrades held-out evaluation quality).

This script downloads the backbone from HuggingFace and copies it to the
workspace staging path. Idempotent: if a staged file already exists it is
reused and no download happens. Hard-fails (non-zero exit) when it cannot
produce a staged file, so Pre-Flight can hard-stop on the same signal.

The default remains `nvidia/C-RADIOv2-B`. Pass `--backbone-type` for one of the
supported DINOv3 variants; those profiles fetch the matching timm-format
`model.safetensors` from Hugging Face. HF_TOKEN is read from the environment
when present (required for gated repos / rate limits).

CLI:

    python scripts/stage_backbone.py --workspace ~/workspace

    # Frozen DINOv3 backbone:
    python scripts/stage_backbone.py --workspace ~/workspace \
        --backbone-type vit_large_dinov3

    # or an explicit destination / different repo:
    python scripts/stage_backbone.py \
        --dest ~/workspace/augmentation/backbone/c_radio_v2_b.safetensors \
        --repo-id nvidia/C-RADIOv2-B --filename model.safetensors

On success the absolute staged path is printed to stdout as the last line, so a
caller can capture it: STAGED=$(python scripts/stage_backbone.py --workspace ...)
"""

import argparse
import os
import pathlib
import shutil
import sys

import yaml


DEFAULT_BACKBONE_TYPE = "c_radio_v2_vit_base_patch16_224"
# Values are (Hugging Face repo id, repo filename, workspace stage filename).
BACKBONE_PROFILES = {
    DEFAULT_BACKBONE_TYPE: (
        "nvidia/C-RADIOv2-B",
        "model.safetensors",
        "c_radio_v2_b.safetensors",
    ),
    "vit_small_dinov3": (
        "timm/vit_small_patch16_dinov3.lvd1689m",
        "model.safetensors",
        "vit_small_dinov3.safetensors",
    ),
    "vit_small_plus_dinov3": (
        "timm/vit_small_plus_patch16_dinov3.lvd1689m",
        "model.safetensors",
        "vit_small_plus_dinov3.safetensors",
    ),
    "vit_base_dinov3": (
        "timm/vit_base_patch16_dinov3.lvd1689m",
        "model.safetensors",
        "vit_base_dinov3.safetensors",
    ),
    "vit_large_dinov3": (
        "timm/vit_large_patch16_dinov3.lvd1689m",
        "model.safetensors",
        "vit_large_dinov3.safetensors",
    ),
    "vit_huge_plus_dinov3": (
        "timm/vit_huge_plus_patch16_dinov3.lvd1689m",
        "model.safetensors",
        "vit_huge_plus_dinov3.safetensors",
    ),
    "vit_7b_dinov3": (
        "timm/vit_7b_patch16_dinov3.lvd1689m",
        "model.safetensors",
        "vit_7b_dinov3.safetensors",
    ),
}
DEFAULT_REPO_ID, DEFAULT_FILENAME, DEFAULT_STAGE_NAME = (
    BACKBONE_PROFILES[DEFAULT_BACKBONE_TYPE]
)


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
    p.add_argument(
        "--backbone-type",
        choices=sorted(BACKBONE_PROFILES),
        help="Visual ChangeNet backbone profile. When omitted, read "
        "<workspace>/specs/baseline_spec.yaml, then default to the existing "
        "C-RADIOv2-B profile.",
    )
    p.add_argument("--repo-id", help="Override the profile's Hugging Face repo id.")
    p.add_argument("--filename", help="Override the profile's repo filename.")
    p.add_argument(
        "--stage-name",
        help="Override the profile's filename under <workspace>/augmentation/backbone/.",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if a staged file already exists.",
    )
    return p.parse_args()


def _baseline_backbone(args: argparse.Namespace) -> dict:
    """Read the workspace baseline backbone configuration when present."""
    workspace = getattr(args, "workspace", None)
    if not workspace:
        return {}
    spec_path = (
        pathlib.Path(workspace).expanduser()
        / "specs"
        / "baseline_spec.yaml"
    )
    if not spec_path.is_file():
        return {}
    spec = yaml.safe_load(spec_path.read_text()) or {}
    return spec.get("model", {}).get("backbone", {})


def resolve_backbone_type(args: argparse.Namespace) -> str:
    """Resolve an explicit profile or infer it from the baseline spec."""
    backbone = _baseline_backbone(args)
    baseline_type = str(backbone.get("type", "")).strip()
    backbone_type = (
        getattr(args, "backbone_type", None)
        or baseline_type
        or DEFAULT_BACKBONE_TYPE
    )

    if backbone_type not in BACKBONE_PROFILES:
        sys.exit(f"stage_backbone: unsupported backbone type: {backbone_type}")

    if baseline_type and baseline_type != backbone_type:
        sys.exit(
            "stage_backbone: --backbone-type does not match baseline spec: "
            f"{backbone_type} != {baseline_type}"
        )

    if (
        "dinov3" in backbone_type
        and baseline_type == backbone_type
        and not bool(backbone.get("freeze_backbone", False))
    ):
        sys.exit(
            "stage_backbone: DINOv3 must set "
            "model.backbone.freeze_backbone: true for DEFT."
        )

    return backbone_type


def _has_source_override(args: argparse.Namespace) -> bool:
    """Return whether standalone source/destination overrides were supplied."""
    return any(
        getattr(args, name, None)
        for name in ("dest", "repo_id", "filename")
    )


def resolve_configured_checkpoint(args: argparse.Namespace) -> str | None:
    """Resolve a converted DINOv3 checkpoint configured in the baseline spec."""
    workspace = getattr(args, "workspace", None)
    if _has_source_override(args) or not workspace:
        return None

    backbone = _baseline_backbone(args)
    backbone_type = resolve_backbone_type(args)
    configured_path = backbone.get("pretrained_backbone_path")
    if (
        "dinov3" not in backbone_type
        or str(backbone.get("type", "")).strip() != backbone_type
        or not isinstance(configured_path, str)
        or not configured_path.strip()
    ):
        return None

    configured = pathlib.Path(configured_path).expanduser()
    candidates = (
        configured,
        pathlib.Path(workspace).expanduser()
        / "augmentation"
        / "backbone"
        / configured.name,
    )
    for candidate in candidates:
        if candidate.is_file() and candidate.stat().st_size > 0:
            return str(candidate.resolve())
    sys.exit(
        "stage_backbone: configured DINOv3 checkpoint is not staged: "
        f"{candidates[-1]}"
    )


def resolve_source(args: argparse.Namespace) -> tuple[str, str, str]:
    """Resolve profile defaults with optional explicit source overrides."""
    if not getattr(args, "backbone_type", None) and _has_source_override(args):
        # Preserve the standalone custom-repo/destination escape hatch. The
        # workspace baseline is authoritative only for DEFT-managed staging.
        backbone_type = DEFAULT_BACKBONE_TYPE
    else:
        backbone_type = resolve_backbone_type(args)
    repo_id, filename, stage_name = BACKBONE_PROFILES[backbone_type]
    return (
        getattr(args, "repo_id", None) or repo_id,
        getattr(args, "filename", None) or filename,
        getattr(args, "stage_name", None) or stage_name,
    )


def resolve_dest(args: argparse.Namespace) -> str:
    configured = resolve_configured_checkpoint(args)
    if configured:
        return configured
    if args.dest:
        return os.path.abspath(os.path.expanduser(args.dest))
    if not args.workspace:
        sys.exit("stage_backbone: one of --dest or --workspace is required.")
    ws = os.path.abspath(os.path.expanduser(args.workspace))
    _, _, stage_name = resolve_source(args)
    return os.path.join(ws, "augmentation", "backbone", stage_name)


def main() -> int:
    args = parse_args()
    configured = resolve_configured_checkpoint(args)
    if configured:
        print(
            "stage_backbone: reusing configured converted checkpoint.",
            file=sys.stderr,
        )
        print(configured)
        return 0
    repo_id, filename, _ = resolve_source(args)
    dest = resolve_dest(args)

    # Idempotent: reuse an existing non-empty staged file unless --force.
    if not args.force and os.path.isfile(dest) and os.path.getsize(dest) > 0:
        print(f"stage_backbone: reusing already-staged file ({os.path.getsize(dest)} bytes).", file=sys.stderr)
        print(dest)
        return 0

    if (
        os.environ.get("HF_HUB_OFFLINE") == "1"
        or os.environ.get("AIR_GAPPED") == "1"
    ):
        sys.exit(
            f"stage_backbone: offline mode and staged file is missing: {dest}; "
            "no install or download was attempted"
        )

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        sys.exit(
            "stage_backbone: huggingface_hub is not installed; provision it outside "
            "this workflow or pre-stage the backbone file"
        )

    token = os.environ.get("HF_TOKEN") or None
    try:
        src = hf_hub_download(repo_id=repo_id, filename=filename, token=token)  # nosec B615 - pinned repo_id and filename from skill config
    except Exception as exc:  # network, auth, missing file — all are hard stops
        sys.exit(
            f"stage_backbone: failed to download {filename} from {repo_id}: {exc}\n"
            "Staging is mandatory — there is no working URL fallback. Set HF_TOKEN if the "
            "repo is gated, or pre-stage the file at the destination path manually."
        )

    os.makedirs(os.path.dirname(dest), exist_ok=True)
    shutil.copy(src, dest)

    if not (os.path.isfile(dest) and os.path.getsize(dest) > 0):
        sys.exit(f"stage_backbone: copy produced no file at {dest}.")

    print(f"stage_backbone: staged {repo_id}/{filename} -> {dest} "
          f"({os.path.getsize(dest)} bytes).", file=sys.stderr)
    print(dest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
