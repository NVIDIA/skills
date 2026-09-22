#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Carve a validation COCO out of the prepared source pool.

``grounding_dino train`` cannot run without a validation set — the data module
subscripts ``val_data_sources["json_file"]`` unconditionally — so every iteration
needs one. Deriving it from the pool means it always exists, always matches the
target classes, and needs nothing from the user.

**The category ids are rewritten to 0-based, and that is the whole point.**
Grounding DINO's validation loader does::

    classes = [obj["category_id"] for obj in target]      # dataloader/coco.py

and uses the value *verbatim* as the index into the model's class dimension. COCO
is conventionally 1-based, so N classes yield indices 1..N against a size-N tensor
and the last class always overflows::

    IndexError: index 3 is out of bounds for dimension 0 with size 3

raised inside the loss criterion during Lightning's sanity-validation, before the
first training step — so it reads as a training bug. ``annotations convert``
emits 1-based ids, so the pool COCO cannot be used as a val source unmodified.

Deterministic given ``--seed``, so a resumed run validates against the same images.

Stdlib only.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path


def build_val_split(coco: dict, fraction: float, seed: int) -> tuple[dict, dict]:
    images = coco.get("images", [])
    if not images:
        raise ValueError("the source COCO has no images")

    count = max(1, round(len(images) * fraction))
    # Sample from a sorted list so the choice depends only on the seed, never on
    # the order the converter happened to emit.
    ordered = sorted(images, key=lambda im: str(im.get("id")))
    picked = random.Random(seed).sample(ordered, count)
    keep_ids = {im["id"] for im in picked}

    # category_id is used verbatim as a dense label index by GDINO's val loader,
    # so shift the whole space to 0-based. Preserve declaration order: those
    # indices must line up with the caption/label order used at inference.
    categories = coco.get("categories", [])
    # Dense 0..N-1 in declaration order. Subtracting the minimum id only works when
    # the ids are already contiguous: {1,3,7} would become {0,2,6}, and 6 indexes
    # past a size-3 class dimension.
    remap = {c["id"]: index for index, c in enumerate(categories)}

    val = {
        "images": picked,
        "annotations": [
            {**a, "category_id": remap[a["category_id"]]}
            for a in coco.get("annotations", [])
            if a.get("image_id") in keep_ids and a.get("category_id") in remap
        ],
        "categories": [{**c, "id": remap[c["id"]]} for c in categories],
    }
    report = {
        "source_images": len(images),
        "val_images": len(picked),
        "val_annotations": len(val["annotations"]),
        "fraction": fraction,
        "seed": seed,
        "category_ids_remapped": {c["id"]: remap[c["id"]] for c in categories},
        "categories": {c["name"]: c["id"] for c in val["categories"]},
    }
    return val, report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--coco", required=True, help="Prepared pool COCO (source_pool/coco.json).")
    parser.add_argument("--out", required=True, help="Where to write the validation COCO.")
    parser.add_argument("--fraction", type=float, default=0.10,
                        help="Share of pool images to use for validation. Default 0.10.")
    parser.add_argument("--seed", type=int, default=1337,
                        help="Selection seed; a resumed run validates on the same images.")
    parser.add_argument("--report-json", default=None)
    return parser.parse_args()


def main() -> int:
    try:
        args = parse_args()
        if not 0 < args.fraction < 1:
            raise ValueError(f"--fraction must be within (0, 1), got {args.fraction}")

        src = Path(args.coco).expanduser().resolve()
        if not src.is_file():
            raise FileNotFoundError(f"--coco does not exist: {src}")
        coco = json.loads(src.read_text(encoding="utf-8"))

        val, report = build_val_split(coco, args.fraction, args.seed)
        if not val["annotations"]:
            raise ValueError(
                "the validation split holds no annotations — raise --fraction, or check "
                "that the pool COCO is populated (validate_pool_coco.py)"
            )

        out = Path(args.out).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(val), encoding="utf-8")

        if args.report_json:
            rp = Path(args.report_json).expanduser().resolve()
            rp.parent.mkdir(parents=True, exist_ok=True)
            rp.write_text(json.dumps(report, indent=2), encoding="utf-8")

        print(f"val split -> {out}")
        print(f"  {report['val_images']}/{report['source_images']} images, "
              f"{report['val_annotations']} annotations (seed {report['seed']})")
        print(f"  category ids densified {report['category_ids_remapped']} "
              f"-> {report['categories']}")
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
