# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
"""Gate the architecture and write labels.txt, from a model's config.json.

Single implementation shared by both acquisition routes. The HuggingFace path (Step 2b)
and the NGC path (Step 2d) reach config.json at different points but need identical
behaviour, and keeping two copies in the runbook meant the NGC route silently drifted —
it extracted labels without ever rejecting non-detection architectures, so a
classification model could reach engine build. See references/model-acquire.md.

Usage:
    python3 config-to-labels.py --config <config.json> --labels <labels.txt>
    python3 config-to-labels.py --config <config.json> --labels <labels.txt> --skip-arch-check

Exits non-zero on a non-detection architecture or a missing label map. Callers must treat
a non-zero exit as fatal: never fall back to COCO, ImageNet, or any other default list.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Detection heads this skill supports. DETR-family variants use the middle two.
DETECTION_SUFFIXES = ("ForObjectDetection", "ForConditionalDetection", "ForZeroShotObjectDetection")
NON_DETECTION_SUFFIXES = (
    "ForImageClassification", "ForSemanticSegmentation", "ForInstanceSegmentation",
    "ForPanopticSegmentation", "ForDepthEstimation", "ForMaskedLM",
    "ForTokenClassification", "ForCausalLM",
)


def assert_detection(cfg: dict) -> None:
    """Abort unless config.json declares an object-detection architecture."""
    arch_list = cfg.get("architectures") or []
    if not arch_list:
        print("[labels] config.json has no 'architectures' field; skipping architecture gate",
              file=sys.stderr)
        return
    arch = arch_list[0]
    if arch.endswith(NON_DETECTION_SUFFIXES) or not arch.endswith(DETECTION_SUFFIXES):
        sys.exit(
            f"ERROR: deepstream-import-vision-model currently supports object detection models "
            f"only. Detected architecture: {arch}. Classification, segmentation, and other vision "
            f"tasks are not yet supported."
        )
    print(f"[labels] architecture OK: {arch}")


def extract_labels(cfg: dict) -> list[str]:
    """Pull the class list, trying each known config layout in order."""
    if "id2label" in cfg:                                    # standard HF layout
        return [cfg["id2label"][str(i)] for i in range(len(cfg["id2label"]))]
    if "label2id" in cfg:                                    # reversed map
        return [k for k, _ in sorted(cfg["label2id"].items(), key=lambda kv: kv[1])]
    if "names" in cfg:                                       # some YOLO repos
        names = cfg["names"]
        return [names[str(i)] for i in range(len(names))] if isinstance(names, dict) else list(names)
    sys.exit("ERROR: No label map found in config.json — cannot create labels.txt")


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate architecture and emit labels.txt.")
    ap.add_argument("--config", required=True, help="path to config.json")
    ap.add_argument("--labels", required=True, help="path to write labels.txt")
    ap.add_argument("--skip-arch-check", action="store_true",
                    help="emit labels without gating the architecture (diagnostics only)")
    args = ap.parse_args()

    try:
        cfg = json.loads(Path(args.config).read_text())
    except (OSError, json.JSONDecodeError) as exc:
        sys.exit(f"ERROR: could not read {args.config}: {exc}")

    if not args.skip_arch_check:
        assert_detection(cfg)

    labels = extract_labels(cfg)
    out = Path(args.labels)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(labels) + "\n")
    print(f"labels.txt: {len(labels)} classes -> {out}")
    print("  " + ", ".join(labels[:5]) + (" ..." if len(labels) > 5 else ""))


if __name__ == "__main__":
    main()
